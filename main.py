import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy_financial as npf

st.set_page_config(page_title="Investment Readiness Tool", layout="wide")

st.title("🛡️ Investment Readiness Tool: Stazione di Servizio Palermo")
st.info("Modello integrato: Mercato + Finanza + Capacità Tecnica. Basato su Report 22/01/2026.")

# --- SIDEBAR: LEVE DECISIONALI E SCENARI ---
st.sidebar.header("🕹️ Variabili Decisionali (Il tuo controllo)")
with st.sidebar.expander("💼 Scelte Strategiche", expanded=True):
    tecnologia = st.selectbox("Tecnologia Asset", ["DC 30 kW", "DC 60 kW"])
    prezzo_kwh = st.number_input("Prezzo vendita (€/kWh)", value=0.69)
    costo_kwh = st.number_input("Costo energia (€/kWh)", value=0.30)
    wacc = st.slider("Costo del Capitale (WACC %)", 4, 12, 8) / 100

with st.sidebar.expander("⚙️ Parametri Tecnici"):
    # Logica dinamica basata sulla scelta tecnologica
    if tecnologia == "DC 30 kW":
        potenza_kw = 30
        capex_unit = st.number_input("CAPEX per unità (€)", value=25000)
        opex_unit = 2000
    else:
        potenza_kw = 60
        capex_unit = st.number_input("CAPEX per unità (€)", value=45000)
        opex_unit = 3500
    ore_max_giorno = st.slider("Disponibilità operativa (ore/giorno)", 4, 12, 10)

st.sidebar.header("📊 Ipotesi di Scenario (Rischi Esterni)")
with st.sidebar.expander("🌍 Driver di Mercato"):
    stress_bev = st.slider("Stress Test Parco BEV (%)", 50, 150, 100) / 100
    public_share = st.slider("Quota Ricarica Pubblica (%)", 10, 50, 30) / 100
    carica_media_anno = st.number_input("Carica Media/Auto (kWh/y)", value=3000)
    stress_cattura = st.slider("Efficacia Cattura (%)", 50, 150, 100) / 100

# --- LOGICA DI CALCOLO UNIFICATA ---
years = np.array([2026, 2027, 2028, 2029, 2030])
bev_palermo = np.array([2600, 3000, 3500, 4200, 5000]) * stress_bev
quota_stazione = np.array([0.02, 0.03, 0.04, 0.045, 0.05]) * stress_cattura

# Calcolo Volumi e Capacità
auto_clienti = bev_palermo * public_share * quota_stazione
energia_kwh = auto_clienti * carica_media_anno
ore_carica_richieste = energia_kwh / potenza_kw
ore_disponibili_per_asset = ore_max_giorno * 365

# Dimensionamento scalabile: aggiunge colonnine se le ore superano la capacità
n_colonnine = np.ceil(ore_carica_richieste / ore_disponibili_per_asset).astype(int)

# Conto Economico
ricavi = energia_kwh * prezzo_kwh
ebitda = (energia_kwh * (prezzo_kwh - costo_kwh)) - (n_colonnine * opex_unit)

capex_flow = np.zeros(len(years))
prev_n = 0
for i, n in enumerate(n_colonnine):
    capex_flow[i] = max(0, n - prev_n) * capex_unit
    prev_n = n

cf_netto = ebitda - capex_flow
cf_cum = np.cumsum(cf_netto)

# --- DASHBOARD KPI ---
st.subheader("🏁 Indicatori Finanziari & Operativi")
k1, k2, k3, k4 = st.columns(4)
van = npf.npv(wacc, cf_netto)
k1.metric("VAN (NPV)", f"€ {van:,.0f}", help="Se > 0 l'investimento batte il mercato")
k2.metric("TIR (IRR)", f"{(npf.irr(cf_netto)*100):.1f}%")
k3.metric("Auto/Giorno (Media)", f"{((energia_kwh/35)/365).mean():.1f}")
k4.metric("Saturazione 2030", f"{(ore_carica_richieste[-1]/(n_colonnine[-1]*ore_disponibili_per_asset)*100):.1f}%")

# --- TUTTI I GRAFICI RICHIESTI ---
st.divider()
st.subheader("📊 Intelligence Visuale")
c1, c2 = st.columns(2)

with c1:
    st.write("**1. Break-even: Auto necessarie al giorno**")
    auto_range = np.linspace(1, 15, 20)
    margine_carica = 35 * (prezzo_kwh - costo_kwh)
    costo_fisso_day = (opex_unit + (capex_unit/5)) / 365
    profitto_day = (auto_range * margine_carica) - costo_fisso_day
    fig1, ax1 = plt.subplots()
    ax1.plot(auto_range, profitto_day, color='green', linewidth=3)
    ax1.axhline(0, color='red', linestyle='--')
    ax1.set_xlabel("N. Ricariche giornaliere per unità")
    st.pyplot(fig1)

with c2:
    st.write("**2. Cash Flow Cumulato (Recupero)**")
    fig2, ax2 = plt.subplots()
    ax2.plot(years, cf_cum, marker='o', color='navy', linewidth=3)
    ax2.fill_between(years, cf_cum, 0, color='blue', alpha=0.1)
    ax2.axhline(0, color='black', linewidth=1)
    st.pyplot(fig2)

c3, c4 = st.columns(2)
with c3:
    st.write("**3. Capacità Fisica: Domanda vs Asset**")
    fig3, ax3 = plt.subplots()
    ax3.bar(years, ore_carica_richieste, color='skyblue', label="Ore ricarica richieste")
    ax3_tw = ax3.twinx()
    ax3_tw.step(years, n_colonnine * ore_disponibili_per_asset, where='post', color='red', label="Capacità ore")
    st.pyplot(fig3)

with c4:
    st.write("**4. Struttura dei Margini (Totale 5 Anni)**")
    fig4, ax4 = plt.subplots()
    labels = ['Ricavi', 'Costi Energia', 'CAPEX', 'EBITDA']
    vals = [ricavi.sum(), (energia_kwh * costo_kwh).sum(), capex_flow.sum(), ebitda.sum()]
    ax4.bar(labels, vals, color=['#2ecc71', '#e67e22', '#e74c3c', '#3498db'])
    st.pyplot(fig4)

# --- TABELLA ANALITICA COMPLETA ---
st.divider()
st.subheader("📊 Analisi Dettagliata (Dati Report + Conversione)")
df_table = pd.DataFrame({
    "Anno": years,
    "BEV Palermo": bev_palermo.astype(int),
    "Clienti Catturati": auto_clienti.astype(int),
    "Energia (kWh)": energia_kwh.astype(int),
    "Ore Richieste": ore_carica_richieste.astype(int),
    "N. Colonnine": n_colonnine,
    "EBITDA (€)": ebitda.astype(int),
    "CF Cumulato (€)": cf_cum.astype(int)
}).set_index("Anno")
st.table(df_table)
