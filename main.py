import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy_financial as npf
import io
import requests

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

def eur(x):
    if x is None or not np.isfinite(x): return "n/a"
    return f"€ {x:,.0f}".replace(",", ".")

def forecast_bev_2030(df_hist, province, method):
    s = df_hist[df_hist["Provincia"] == province].sort_values("Anno")
    bev_2024 = int(s[s["Anno"] == 2024]["Elettrico"].iloc[0])
    if method == "CAGR 2021–2024":
        v0, v1 = float(s[s["Anno"] == 2021]["Elettrico"].iloc[0]), float(s[s["Anno"] == 2024]["Elettrico"].iloc[0])
        years_diff = 3
    else:
        v0, v1 = float(s[s["Anno"] == 2015]["Elettrico"].iloc[0]), float(s[s["Anno"] == 2024]["Elettrico"].iloc[0])
        years_diff = 9
    cagr = (v1 / max(v0, 1)) ** (1 / years_diff) - 1
    return bev_2024, int(round(bev_2024 * ((1 + cagr) ** 6))), cagr

# ============================================================
# 2. SIDEBAR - TUTTI I PARAMETRI RIPRISTINATI
# ============================================================
df_bev = pd.read_csv(io.StringIO(BEV_RAW), sep="\t")
st.sidebar.header("🗺️ Territorio & Funnel")
province_list = sorted(df_bev["Provincia"].unique())
provincia = st.sidebar.selectbox("Provincia", province_list, index=province_list.index("PALERMO"))
bev_method = st.sidebar.selectbox("Metodo Forecast", ["CAGR 2021–2024", "CAGR 2015–2024"])
bev_2024, bev_2030_auto, cagr_used = forecast_bev_2030(df_bev, provincia, bev_method)

with st.sidebar.expander("🎯 Market Funnel", expanded=True):
    target_bev_2030 = st.number_input(f"Target BEV {provincia} 2030", value=bev_2030_auto)
    public_share = st.slider("Quota Ricarica Pubblica (%)", 10, 80, 30) / 100
    target_cattura = st.slider("Quota Cattura Target (%)", 0.5, 20.0, 5.0) / 100

with st.sidebar.expander("⚙️ Tecnica & Operatività", expanded=True):
    tecnologia = st.selectbox("Tecnologia", ["DC 30 kW", "DC 60 kW"])
    ore_max = st.slider("Ore operative/giorno", 4, 24, 10)
    kwh_sess = st.number_input("kWh medi/sessione", value=35)
    uptime = st.slider("Uptime (%)", 85, 100, 97) / 100
    util_medio = st.slider("Utilizzo medio annuo (%)", 10, 80, 30) / 100

with st.sidebar.expander("💰 Finanza Avanzata (CFO)", expanded=False):
    cfo_mode = st.checkbox("Attiva Modalità CFO", value=True)
    wacc = st.slider("WACC (%)", 4, 12, 8) / 100
    tax_rate = st.slider("Tax (%)", 0, 40, 28) / 100
    prezzo_kwh = st.number_input("Prezzo vendita (€/kWh)", value=0.69)
    costo_kwh = st.number_input("Costo energia (€/kWh)", value=0.30)
    fee_roaming = st.slider("Fee Roaming (%)", 0.0, 15.0, 0.0) / 100
    ammortamento_anni = st.slider("Ammortamento (anni)", 3, 12, 5)

# ============================================================
# 3. LOGICA DI CALCOLO (VERSIONE COMPLETA)
# ============================================================
years = np.array([2026, 2027, 2028, 2029, 2030])
bev_citta = np.linspace(bev_2024, target_bev_2030, len(years))
quota_staz = np.linspace(0.02, target_cattura, len(years))
auto_clienti = bev_citta * public_share * quota_staz
energia_kwh = auto_clienti * 3000 # Proxy PDF
potenza_kw = 30 if tecnologia == "DC 30 kW" else 60

# Dimensionamento a scalini
cap_kwh_unit = potenza_kw * 8760 * uptime * util_medio
n_totale = np.ceil(energia_kwh / max(cap_kwh_unit, 1e-9)).astype(int)

# Flussi Economici
capex_unit = 25000 if tecnologia == "DC 30 kW" else 45000
opex_unit = 2000 if tecnologia == "DC 30 kW" else 3500
ricavi = energia_kwh * prezzo_kwh
costi_en = energia_kwh * costo_kwh
ebitda = (ricavi - costi_en - (ricavi * fee_roaming)) - (n_totale * opex_unit)

capex_flow = np.zeros(len(years))
prev_n = 0
for i, n in enumerate(n_totale):
    if n > prev_n:
        capex_flow[i] = (n - prev_n) * capex_unit
    prev_n = n

# ============================================================
# 4. MODULO COMPETITOR (OSM / OVERPASS)
# ============================================================
def fetch_osm(lat, lon, r=5):
    query = f"""[out:json];(node["amenity"="charging_station"](around:{r*1000},{lat},{lon}););out center;"""
    try:
        res = requests.post("https://overpass-api.de/api/interpreter", data=query, timeout=10).json()
        return pd.DataFrame([{"lat": e["lat"], "lon": e["lon"], "name": e.get("tags",{}).get("name","N/D")} for e in res["elements"]])
    except: return pd.DataFrame()

st.title("🛡️ Executive Charging Suite — Sicilia")
col_lat, col_lon, col_btn = st.columns([2,2,1])
lat = col_lat.number_input("Latitudine", value=38.1157, format="%.6f")
lon = col_lon.number_input("Longitudine", value=13.3615, format="%.6f")
if col_btn.button("🔍 Analisi Prossimità"):
    df_poi = fetch_osm(lat, lon)
    if not df_poi.empty:
        st.write(f"Trovate {len(df_poi)} stazioni nel raggio di 5km.")
        st.map(df_poi)
    else: st.warning("Nessun competitor trovato.")

# ============================================================
# 5. GRAFICI DECISIONALI & KPI
# ============================================================
st.divider()
m1, m2, m3, m4 = st.columns(4)
m1.metric("Payback Anno", years[np.where(np.cumsum(ebitda - capex_flow) > 0)[0][0]] if any(np.cumsum(ebitda - capex_flow) > 0) else "Oltre 2030")
m2.metric("EBITDA 2030", eur(ebitda[-1]))
m3.metric("Unità Necessarie", n_totale[-1])
m4.metric("ROI Semplice", f"{(ebitda.sum()/capex_flow.sum()*100):.0f}%")

c1, c2 = st.columns(2)
with c1:
    st.write("**Soglia di Sopravvivenza (Ricariche/Giorno)**")
    margine_sess = kwh_sess * (prezzo_kwh - costo_kwh)
    costo_f_gg = (opex_unit + (capex_unit/ammortamento_anni)) / 365
    x = np.linspace(0, 15, 100)
    fig, ax = plt.subplots()
    ax.plot(x, (x * margine_sess) - costo_f_gg, lw=3)
    ax.axhline(0, color='red', ls='--')
    ax.fill_between(x, (x * margine_sess) - costo_f_gg, 0, where=((x * margine_sess) - costo_f_gg > 0), color='green', alpha=0.1)
    ax.set_xlabel("Ricariche/Giorno per Unità")
    st.pyplot(fig)

with c2:
    st.write("**Cash Flow Cumulato (EBITDA - CAPEX)**")
    fig2, ax2 = plt.subplots()
    ax2.plot(years, np.cumsum(ebitda - capex_flow), marker='o', lw=3)
    ax2.axhline(0, color='black', lw=1)
    st.pyplot(fig2)

# ============================================================
# 6. REPORT ANALITICO COMPLETO
# ============================================================
st.divider()
st.subheader("📊 Report Analitico Dettagliato")
df_master = pd.DataFrame({
    "Anno": years,
    "BEV Provincia": bev_citta.astype(int),
    "Auto Target": auto_clienti.astype(int),
    "Energia (kWh)": energia_kwh.astype(int),
    "Unità Totali": n_totale,
    "Ricavi (€)": ricavi.astype(int),
    "EBITDA (€)": ebitda.astype(int),
    "CAPEX (€)": capex_flow.astype(int)
}).set_index("Anno")
st.dataframe(df_master, use_container_width=True)

if cfo_mode:
    st.subheader("🏦 Analisi Investment-Grade (NPV/IRR)")
    fcf = ebitda * (1 - tax_rate) - capex_flow
    npv = npf.npv(wacc, fcf)
    irr = npf.irr(fcf)
    st.write(f"**VAN (NPV):** {eur(npv)} | **TIR (IRR):** {irr*100:.1f}%")
