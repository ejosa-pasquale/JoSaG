import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy_financial as npf
import io
import requests
from geopy.distance import geodesic

# ============================================================
# 1. DATI STORICI E CONFIGURAZIONE
# ============================================================
st.set_page_config(page_title="Executive Charging Suite — Sicilia", layout="wide")

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
# 2. LOGICA COMPETITOR (OSM)
# ============================================================
def fetch_osm_competitors(lat, lon, r=5):
    query = f"""[out:json];node["amenity"="charging_station"](around:{r*1000},{lat},{lon});out;"""
    try:
        res = requests.post("https://overpass-api.de/api/interpreter", data=query, timeout=10).json()
        points = []
        for e in res['elements']:
            d = geodesic((lat, lon), (e['lat'], e['lon'])).km
            points.append({"Lat": e['lat'], "Lon": e['lon'], "Distanza (km)": round(d, 2)})
        return pd.DataFrame(points)
    except: return pd.DataFrame()

# ============================================================
# 3. SIDEBAR (CONTROLLO TOTALE)
# ============================================================
df_bev = pd.read_csv(io.StringIO(BEV_RAW), sep="\t")
st.sidebar.title("🎮 Parametri di Simulazione")

with st.sidebar.expander("📍 Località e Competitor", expanded=True):
    provincia = st.selectbox("Provincia", sorted(df_bev["Provincia"].unique()), index=5)
    site_lat = st.number_input("Latitudine", value=38.1157, format="%.6f")
    site_lon = st.number_input("Longitudine", value=13.3615, format="%.6f")
    radius = st.slider("Raggio Analisi (km)", 1, 10, 5)

with st.sidebar.expander("💰 Economia e Prezzi", expanded=True):
    prezzo_kwh = st.number_input("Prezzo vendita (€/kWh)", value=0.69)
    costo_kwh = st.number_input("Costo energia (€/kWh)", value=0.32)
    capex_u = st.number_input("CAPEX Unità (€)", value=35000)
    opex_u = st.number_input("OPEX Annuo (€)", value=2800)
    wacc = st.slider("WACC %", 4, 12, 8) / 100

with st.sidebar.expander("🎯 Quota di Mercato", expanded=True):
    capture_rate = st.slider("Quota Cattura Target (%)", 0.1, 20.0, 5.0) / 100
    public_share = st.slider("Domanda su Pubblico (%)", 10, 80, 40) / 100

# ============================================================
# 4. ELABORAZIONE DATI
# ============================================================
# Analisi Competitor
df_comp = fetch_osm_competitors(site_lat, site_lon, radius)
comp_factor = 1.0
if not df_comp.empty:
    n_comp = len(df_comp)
    min_dist = df_comp["Distanza (km)"].min()
    comp_factor = max(0.1, 1.0 - (n_comp * 0.05) - (0.1 if min_dist < 1 else 0))

# Proiezione Parco Auto (2025-2030)
years = np.array([2025, 2026, 2027, 2028, 2029, 2030])
bev_2024 = df_bev[df_bev["Provincia"] == provincia]["Elettrico"].iloc[-1]
bev_growth = bev_2024 * (1.28 ** (years - 2024)) # +28% CAGR

# Calcolo Volumi
auto_target = bev_growth * public_share * (capture_rate * comp_factor)
kwh_annui = auto_target * 3000 # Proxy: 3000 kWh per auto

# Dimensionamento
n_unita = np.ceil(kwh_annui / 140000).astype(int) # Soglia 140k kWh/anno per macchina
ebitda = (kwh_annui * (prezzo_kwh - costo_kwh)) - (n_unita * opex_u)
capex_flow = np.zeros(len(years))
capex_flow[0] = n_unita[0] * capex_u # Investimento iniziale

# ============================================================
# 5. DASHBOARD GRAFICA DECISIONALE
# ============================================================
st.title("🛡️ Executive Decision Board: EV Mobility")

# KPI Principali
c1, c2, c3, c4 = st.columns(4)
c1.metric("Auto Target 2030", f"{auto_target[-1]:,.0f}")
c2.metric("EBITDA 2030", f"€ {ebitda[-1]:,.0f}")
c3.metric("Unità Necessarie", n_unita[-1])
c4.metric("Fattore Competitor", f"{comp_factor:.2f}x")

st.divider()

# --- GRAFICI DECISIONALI ---
col_l, col_r = st.columns(2)

with col_l:
    st.subheader("🎯 Soglia di Sopravvivenza Operativa")
    st.write("Quante ricariche medie/giorno servono per coprire i costi?")
    
    # Calcolo soglia di ricariche per sessione media di 35kWh
    kwh_sess = 35
    margine_sess = kwh_sess * (prezzo_kwh - costo_kwh)
    costo_fisso_day = (opex_u + (capex_u / 8)) / 365 # Ammortamento 8 anni
    
    x_ricariche = np.linspace(0.5, 12, 100)
    y_profit = (x_ricariche * margine_sess) - costo_fisso_day
    
    fig1, ax1 = plt.subplots()
    ax1.plot(x_ricariche, y_profit, color='navy', lw=3)
    ax1.axhline(0, color='red', ls='--')
    be_point = costo_fisso_day / margine_sess
    ax1.scatter([be_point], [0], color='red', s=100, label=f"Break-even: {be_point:.1f} ric/gg")
    ax1.fill_between(x_ricariche, y_profit, 0, where=(y_profit > 0), color='green', alpha=0.1)
    ax1.set_xlabel("Ricariche Giornaliere (per singola unità)")
    ax1.set_ylabel("Margine Netto Giornaliero (€)")
    ax1.legend()
    st.pyplot(fig1)

with col_r:
    st.subheader("🌪️ Analisi di Sensibilità (Rischio)")
    st.write("Quale variabile impatta di più sul tuo EBITDA 2030?")
    
    # Calcolo delta EBITDA per +/- 10%
    base_ebitda = ebitda[-1]
    ebitda_prezzo_up = (kwh_annui[-1] * (prezzo_kwh*1.1 - costo_kwh)) - (n_unita[-1]*opex_u)
    ebitda_costo_up = (kwh_annui[-1] * (prezzo_kwh - costo_kwh*1.1)) - (n_unita[-1]*opex_u)
    ebitda_vol_up = (kwh_annui[-1]*1.1 * (prezzo_kwh - costo_kwh)) - (n_unita[-1]*opex_u)

    sens = {
        'Prezzo (+10%)': ebitda_prezzo_up - base_ebitda,
        'Costo Energia (+10%)': ebitda_costo_up - base_ebitda,
        'Volumi Auto (+10%)': ebitda_vol_up - base_ebitda
    }
    
    fig2, ax2 = plt.subplots()
    ax2.barh(list(sens.keys()), list(sens.values()), color=['green' if v > 0 else 'red' for v in sens.values()])
    ax2.axvline(0, color='black', lw=1)
    st.pyplot(fig2)

st.divider()

# --- TABELLA E MAPPA ---
st.subheader("🗺️ Analisi Territoriale Competitor")
if not df_comp.empty:
    st.map(df_comp.rename(columns={"Lat":"lat", "Lon":"lon"}))
    st.dataframe(df_comp, use_container_width=True)

st.subheader("📊 Business Plan Sintetico")
df_rep = pd.DataFrame({
    "Anno": years,
    "Parco BEV": bev_growth.astype(int),
    "Clienti Target": auto_target.astype(int),
    "Erogazione (MWh)": (kwh_annui/1000).astype(int),
    "Unità": n_unita,
    "EBITDA (€)": ebitda.astype(int)
}).set_index("Anno")
st.table(df_rep)

# ============================================================
# 6. GLOSSARIO TECNICO E FORMULE
# ============================================================
with st.expander("📚 Formule e Manuale Decisionale"):
    st.markdown(r"""
    ### 1. Calcolo del Funnel
    La domanda reale è filtrata tramite:
    $$Auto_{Target} = Parco_{Provinciale} \times \%Domanda_{Pubblica} \times (Quota_{Cattura} \times Fattore_{Competitor})$$

    ### 2. Sostenibilità Economica
    Il Break-even giornaliero è calcolato come:
    $$Ric_{BE} = \frac{Costo_{Op} + Amm.}{Sessione \times (Prezzo - Costo_{En})}$$
    
    **Guida alla decisione:**
    - Se il Break-even è **< 3 ricariche/gg**, il progetto è a basso rischio.
    - Se il Tornado Chart mostra che il **Costo Energia** ha barre rosse molto lunghe, valuta un contratto a prezzo fisso (PPA).
    - Se il **Fattore Competitor** è < 0.50, la zona è troppo satura per una nuova installazione.
    """)
