import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy_financial as npf
import io
import requests
from geopy.distance import geodesic

# ============================================================
# 1. SETUP & DATA REPOSITORY
# ============================================================
st.set_page_config(page_title="eV Charging Executive Suite", layout="wide")

BEV_RAW = """Anno\tProvincia\tElettrico
2015\tAGRIGENTO\t23
2021\tAGRIGENTO\t166
2022\tAGRIGENTO\t255
2023\tAGRIGENTO\t370
2024\tAGRIGENTO\t521
2015\tCALTANISSETTA\t11
2021\tCALTANISSETTA\t91
2022\tCALTANISSETTA\t147
2023\tCALTANISSETTA\t207
2024\tCALTANISSETTA\t313
2015\tCATANIA\t87
2021\tCATANIA\t993
2022\tCATANIA\t1578
2023\tCATANIA\t2293
2024\tCATANIA\t3254
2015\tENNA\t2
2021\tENNA\t58
2022\tENNA\t92
2023\tENNA\t129
2024\tENNA\t196
2015\tMESSINA\t179
2021\tMESSINA\t602
2022\tMESSINA\t814
2023\tMESSINA\t1075
2024\tMESSINA\t1412
2015\tPALERMO\t143
2021\tPALERMO\t753
2022\tPALERMO\t1066
2023\tPALERMO\t1530
2024\tPALERMO\t2144
2015\tRAGUSA\t16
2021\tRAGUSA\t337
2022\tRAGUSA\t586
2023\tRAGUSA\t814
2024\tRAGUSA\t1071
2015\tSIRACUSA\t54
2021\tSIRACUSA\t379
2022\tSIRACUSA\t560
2023\tSIRACUSA\t808
2024\tSIRACUSA\t1138
2015\tTRAPANI\t37
2022\tTRAPANI\t395
2023\tTRAPANI\t560
2024\tTRAPANI\t795
"""

# ============================================================
# 2. FUNZIONI TECNICHE & FORMULE
# ============================================================
def fetch_osm_competitors(lat, lon, radius_km=5):
    query = f"""[out:json];node["amenity"="charging_station"](around:{radius_km*1000},{lat},{lon});out;"""
    try:
        response = requests.post("https://overpass-api.de/api/interpreter", data=query, timeout=10)
        data = response.json()
        points = []
        for e in data['elements']:
            dist = geodesic((lat, lon), (e['lat'], e['lon'])).km
            points.append({"Lat": e['lat'], "Lon": e['lon'], "Distanza (km)": round(dist, 2)})
        return pd.DataFrame(points)
    except:
        return pd.DataFrame()

def calc_metrics(capex, opex, rev_per_kwh, cost_per_kwh, kwh_total, tax, wacc):
    ebitda = (kwh_total * (rev_per_kwh - cost_per_kwh)) - opex
    taxes = np.maximum(0, ebitda * tax)
    fcf = ebitda - taxes
    return fcf

# ============================================================
# 3. SIDEBAR: INPUT COMPLETI
# ============================================================
df_bev = pd.read_csv(io.StringIO(BEV_RAW), sep="\t")
st.sidebar.title("🎮 Pannello di Controllo")

with st.sidebar.expander("📍 Localizzazione & Competitor", expanded=True):
    provincia = st.selectbox("Seleziona Provincia", sorted(df_bev["Provincia"].unique()), index=5)
    site_lat = st.number_input("Latitudine Sito", value=38.1157, format="%.6f")
    site_lon = st.number_input("Longitudine Sito", value=13.3615, format="%.6f")
    radius = st.slider("Raggio Analisi Competitor (km)", 1, 10, 5)

with st.sidebar.expander("📈 Modello di Mercato", expanded=True):
    quota_cattura_base = st.slider("Quota Cattura Target (%)", 0.5, 20.0, 5.0) / 100
    public_share = st.slider("Affidamento a Pubblico (%)", 10, 80, 40) / 100
    kwh_annui_auto = st.number_input("Consumo Annuo/Auto (kWh)", value=3000)

with st.sidebar.expander("💸 Parametri Economici", expanded=True):
    prezzo_vendita = st.number_input("Prezzo (€/kWh)", value=0.65)
    costo_energia = st.number_input("Costo (€/kWh)", value=0.28)
    capex_stazione = st.number_input("CAPEX per singola colonnina (€)", value=45000)
    opex_annuo = st.number_input("OPEX annuo (Manutenzione/Suolo) (€)", value=3500)
    wacc = st.slider("WACC (Sconto Finanziario) %", 4, 12, 8) / 100

# ============================================================
# 4. ANALISI COMPETITOR & FATTORE DI EROSIONE
# ============================================================
st.title("⚡ Executive EV Charging Business Planner")

df_comp = fetch_osm_competitors(site_lat, site_lon, radius)
comp_factor = 1.0
if not df_comp.empty:
    n_comp = len(df_comp)
    min_dist = df_comp["Distanza (km)"].min()
    # Formula empirica: ogni competitor riduce la quota cattura del 5%, la vicinanza estrema la dimezza
    comp_factor = max(0.2, 1.0 - (n_comp * 0.05) - (0.1 / max(min_dist, 0.1)))
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Competitor Rilevati", n_comp)
    c2.metric("Distanza Minima", f"{min_dist} km")
    c3.metric("Fattore Erosione Quota", f"{comp_factor:.2f}x")
    st.map(df_comp.rename(columns={"Lat":"lat", "Lon":"lon"}))
else:
    st.info("Nessun competitor rilevato nel raggio selezionato. Fattore competizione: 1.0x")

# ============================================================
# 5. CALCOLO EVOLUTIVO (2025-2030)
# ============================================================
years = np.array([2025, 2026, 2027, 2028, 2029, 2030])
# Proiezione parco auto locale (Semplificata basata su dati storici)
bev_last = df_bev[df_bev["Provincia"] == provincia]["Elettrico"].iloc[-1]
bev_growth = bev_last * (1.25 ** (years - 2024)) # Ipotizziamo crescita 25% annua

# Funnel: Parco -> Domanda Pubblica -> Quota Cattura (Erosa da Competitor)
auto_target = bev_growth * public_share * (quota_cattura_base * comp_factor)
kwh_totali = auto_target * kwh_annui_auto

# Dimensionamento colonnine (1 ogni 150.000 kWh/anno circa)
n_colonnine = np.ceil(kwh_totali / 150000).astype(int)
capex_total = n_colonnine * capex_stazione
# Per semplicità, ipotizziamo investimento tutto all'anno 0 (2025)
opex_total = n_colonnine * opex_annuo
ebitda = (kwh_totali * (prezzo_vendita - costo_energia)) - opex_total

# ============================================================
# 6. GRAFICI DECISIONALI E SPIEGAZIONI
# ============================================================
st.divider()
st.subheader("📊 Analisi delle Performance Economiche")

col1, col2 = st.columns(2)

with col1:
    st.write("**Cash Flow Cumulato & Break Even**")
    cash_flow = ebitda.copy()
    cash_flow[0] -= capex_total[0] # Investimento iniziale
    cum_cf = np.cumsum(cash_flow)
    
    fig, ax = plt.subplots()
    ax.bar(years, cash_flow, color='skyblue', label='Flusso Annuo')
    ax.plot(years, cum_cf, color='red', marker='o', label='Cumulato')
    ax.axhline(0, color='black', linewidth=1)
    ax.set_title("Rientro dell'Investimento (Payback)")
    ax.legend()
    st.pyplot(fig)
    
    st.markdown("""
    > **Cosa osservare:** Il punto in cui la linea rossa incrocia lo zero è il tuo **Break-Even Point**. 
    > Se avviene dopo il 2030, il progetto richiede una revisione dei prezzi o del CAPEX.
    """)

with col2:
    st.write("**Analisi di Sensibilità (Tornado Chart)**")
    # Simulazione impatto variazione +/- 10% sui parametri chiave al 2030
    base_ebitda = ebitda[-1]
    sens_price = ((prezzo_vendita * 1.1 - costo_energia) * kwh_totali[-1]) - opex_total[-1]
    sens_cost = ((prezzo_vendita - costo_energia * 1.1) * kwh_totali[-1]) - opex_total[-1]
    sens_vol = (kwh_totali[-1] * 1.1 * (prezzo_vendita - costo_energia)) - opex_total[-1]
    
    labels = ['Prezzo (+10%)', 'Costo Energia (+10%)', 'Volumi (+10%)']
    impacts = [sens_price - base_ebitda, sens_cost - base_ebitda, sens_vol - base_ebitda]
    
    fig2, ax2 = plt.subplots()
    ax2.barh(labels, impacts, color=['green' if i > 0 else 'red' for i in impacts])
    ax2.set_title("Impatto sull'EBITDA 2030")
    st.pyplot(fig2)

# ============================================================
# 7. GLOSSARIO & FORMULE TECNICHE
# ============================================================
with st.expander("📚 Glossario, Formule e Logica del Modello"):
    st.markdown(r"""
    ### 1. Il Funnel di Vendita
    Il numero di auto che caricheranno da te è calcolato come:
    $$Auto_{Target} = Parco_{Elettrico} \times \%Pubblico \times (Quota_{Cattura} \times Fattore_{Competitor})$$
    
    ### 2. Sostenibilità Energetica
    Ogni colonnina DC ha una capacità limitata. Il numero di unità $N$ cresce quando:
    $$N = \lceil \frac{kWh_{totali}}{Capacità_{max}} \rceil$$
    Dove la $Capacità_{max}$ è stimata in circa 150.000 kWh/anno (pari a circa 12 sessioni/giorno).
    
    ### 3. Fattore Competitor (OSM)
    Il modello interroga il database OpenStreetMap. Il fattore di erosione riduce la tua quota di cattura basandosi sulla densità di stazioni esistenti:
    - Più stazioni ci sono nel raggio di 5km, più si riduce la tua probabilità di attrarre l'utente casuale.
    
    ### 4. Indicatori Finanziari
    - **EBITDA**: Ricavi lordi meno costi energetici e operativi.
    - **NPV (VAN)**: Valore Attuale Netto. Se $> 0$, l'investimento rende più del WACC (costo del capitale).
    """)

# Report Finale
st.subheader("📋 Tabella Riassuntiva per il Board")
df_res = pd.DataFrame({
    "Anno": years,
    "Parco BEV Stimato": bev_growth.astype(int),
    "Tua Quota (Auto)": auto_target.astype(int),
    "Erogazione (MWh)": (kwh_totali / 1000).round(1),
    "Colonnine Attive": n_colonnine,
    "EBITDA (€)": ebitda.astype(int),
    "Cash Flow Cum. (€)": cum_cf.astype(int)
}).set_index("Anno")

st.table(df_res)
