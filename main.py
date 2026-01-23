import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy_financial as npf
import io
import requests
from geopy.distance import geodesic

st.set_page_config(page_title="Executive Charging Suite — Sicilia", layout="wide")

# ============================================================
# 1. DATI STORICI E FUNZIONI ORIGINARIE
# ============================================================
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

def get_cagr(df, prov):
    s = df[df["Provincia"] == prov].sort_values("Anno")
    v_start = s[s["Anno"] == 2015]["Elettrico"].values[0]
    v_end = s[s["Anno"] == 2024]["Elettrico"].values[0]
    return (v_end / v_start)**(1/9) - 1

# ============================================================
# 2. SIDEBAR - PARAMETRI ORIGINARI + MODULI DECISIONALI
# ============================================================
df_bev = pd.read_csv(io.StringIO(BEV_RAW), sep="\t")
st.sidebar.header("🎯 Executive Summary & Inputs")

provincia = st.sidebar.selectbox("Provincia", sorted(df_bev["Provincia"].unique()), index=5)
cagr_hist = get_cagr(df_bev, provincia)
bev_2024 = df_bev[(df_bev["Provincia"] == provincia) & (df_bev["Anno"] == 2024)]["Elettrico"].values[0]

with st.sidebar.expander("📈 Funnel di Conversione", expanded=True):
    # Risposta alla tua domanda: Quante caricano per strada? 
    # Media Nazionale: 35-45% non ha garage (Fonte: Motus-E)
    quota_no_garage = st.slider("Percentuale 'No Garage' (Target strada) %", 10, 80, 40) / 100
    
    # Risposta: Quale percentuale intercettiamo?
    quota_cattura = st.slider("Quota Cattura della stazione %", 0.5, 20.0, 5.0) / 100
    
    # Risposta: Come varia con il prezzo? (Elasticità)
    prezzo_vendita = st.slider("Prezzo vendita (€/kWh)", 0.40, 0.95, 0.65)
    costo_energia = st.number_input("Costo energia (€/kWh)", value=0.30)
    
    # Calcolo elasticità: se prezzo > 0.70, la cattura cala
    elasticita = max(0.1, 1.0 - (prezzo_vendita - 0.55) * 1.5)
    cattura_reale = quota_cattura * elasticita

# ============================================================
# 3. LOGICA DI CALCOLO (ORIGINARIA E POTENZIATA)
# ============================================================
years = np.arange(2025, 2031)
parco_stimato = [int(bev_2024 * (1 + cagr_hist)**(i-2024)) for i in years]

# FUNNEL DECISIONALE: Parco -> Domanda Strada -> Cattura Reale
auto_intercettate = [int(p * quota_no_garage * cattura_reale) for p in parco_stimato]
kwh_annui_auto = 2500 # Parametro standard
energia_totale = np.array(auto_intercettate) * kwh_annui_auto

# RISPOSTA: Quante colonnine servono? 
# Una 60kW DC eroga max ~130MWh/anno al 25% di saturazione
capacita_max_unita = 60 * 8760 * 0.25 
n_colonnine = np.ceil(energia_totale / capacita_max_unita).astype(int)

# ============================================================
# 4. EXECUTIVE SUMMARY (RISPOSTE DIRETTE)
# ============================================================
st.title(f"📊 Analisi Decisionale Ricarica Elettrica: {provincia}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Auto Attese (2030)", f"{parco_stimato[-1]:,}", f"+{cagr_hist:.1%} CAGR")
col2.metric("Tua Quota (Auto)", f"{auto_intercettate[-1]}", f"{cattura_reale:.1%} netta")
col3.metric("Energia Richiesta", f"{energia_totale[-1]/1000:.0f} MWh")
col4.metric("Colonnine Necessarie", f"{n_colonnine[-1]} unità")

st.divider()

# ============================================================
# 5. GRAFICI DECISIONALI (NUOVA SEZIONE)
# ============================================================
st.subheader("📉 Strumenti per la Decisione")
g1, g2 = st.columns(2)

with g1:
    st.write("**A. Analisi del Funnel (Market Share)**")
    # Grafico che spiega quante auto intercettiamo rispetto al totale
    fig, ax = plt.subplots()
    labels = ['Parco Totale', 'Domanda Strada', 'Tuo Target']
    vals = [parco_stimato[-1], parco_stimato[-1]*quota_no_garage, auto_intercettate[-1]]
    ax.bar(labels, vals, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    st.pyplot(fig)
    st.caption("Questo grafico chiarisce quante auto 'esistono' e quante 'entrano' effettivamente nella tua stazione.")

with g2:
    st.write("**B. Stress Test Prezzo (Elasticità)**")
    # Mostra come cambiano i ricavi variando il prezzo
    range_p = np.linspace(0.40, 1.0, 20)
    ricavi_sim = [p * (kwh_annui_auto * (parco_stimato[-1] * quota_no_garage * (quota_cattura * max(0.1, 1.0 - (p - 0.55)*1.5)))) for p in range_p]
    
    fig2, ax2 = plt.subplots()
    ax2.plot(range_p, ricavi_sim, color='green', lw=3)
    ax2.axvline(prezzo_vendita, color='red', ls='--', label='Tuo Prezzo')
    ax2.set_xlabel("Prezzo €/kWh")
    ax2.set_ylabel("Ricavi Potenziali €")
    ax2.legend()
    st.pyplot(fig2)
    st.caption("Il punto di massimo della curva indica il prezzo ideale per non 'scacciare' i clienti.")

# ============================================================
# 6. MODULO COMPETITOR (ORIGINARIO)
# ============================================================
st.divider()
st.subheader("🔍 Localizzazione e Competitor (OSM)")
lat = st.number_input("Latitudine", value=38.1157, format="%.6f")
lon = st.number_input("Longitudine", value=13.3615, format="%.6f")

def fetch_osm(lat, lon):
    try:
        query = f"""[out:json];node["amenity"="charging_station"](around:5000,{lat},{lon});out;"""
        res = requests.post("https://overpass-api.de/api/interpreter", data=query).json()
        return len(res['elements'])
    except: return 0

comp_count = fetch_osm(lat, lon)
st.warning(f"Trovate {comp_count} stazioni nel raggio di 5km. Se questo numero è alto, considera di ridurre la 'Quota Cattura' nella sidebar.")

# ============================================================
# 7. TABELLA BP ORIGINARIA
# ============================================================
st.subheader("📋 Piano Finanziario 2025-2030")
df_bp = pd.DataFrame({
    "Anno": years,
    "Parco Circolante": parco_stimato,
    "Auto Clienti": auto_intercettate,
    "MWh Erogati": (energia_totale / 1000).astype(int),
    "Colonnine": n_colonnine,
    "EBITDA (€)": (energia_totale * (prezzo_vendita - costo_energia)).astype(int)
}).set_index("Anno")
st.table(df_bp)
