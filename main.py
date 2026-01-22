import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy_financial as npf

st.set_page_config(page_title="Executive Charging Suite", layout="wide")

st.title("🛡️ Executive Support System: Ricarica DC Palermo")
st.markdown("Analisi completa 2026-2030: Strategia Multisito, Capacità Tecnica e Redditività.")

# --- SIDEBAR: TUTTE LE VARIABILI DECISIONALI (GM & CFO) ---
st.sidebar.header("🕹️ Leve Decisionali (Il tuo controllo)")
with st.sidebar.expander("📍 Configurazione Rete", expanded=True):
    tecnologia = st.selectbox("Tecnologia Asset", ["DC 30 kW", "DC 60 kW"], help="La 60 kW dimezza i tempi di ricarica ma costa di più.")
    allocazione = st.radio("Strategia di espansione", ["Monosito (Tutto in A)", "Multisito (Espansione in B)"])
    ore_max_giorno = st.slider("Disponibilità operativa (ore/giorno)", 4, 12, 10, help="Tempo massimo di occupazione fisica della piazzola.")

with st.sidebar.expander("💰 Parametri Finanziari"):
    prezzo_kwh = st.number_input("Prezzo vendita (€/kWh)", value=0.69)
    costo_kwh = st.number_input("Costo energia (€/kWh)", value=0.30)
    capex_30 = st.number_input("CAPEX unità 30kW (€)", value=25000)
    capex_60 = st.number_input("CAPEX unità 60kW (€)", value=45000)
    wacc = st.slider("Costo del Capitale (WACC %)", 4, 12, 8) / 100

st.sidebar.header("📊 Ipotesi di Mercato (Rischi)")
with st.sidebar.expander("🌍 Scenario Palermo"):
    stress_bev = st.slider("Stress Test Parco BEV (%)", 50, 150, 100) / 100
    public_share = st.slider("Quota Ricarica Pubblica (%)", 10, 50, 30) / 100
    carica_media_anno = st.number_input("Carica Media/Auto (kWh/y)", value=3000)
    stress_cattura = st.slider("Efficacia Cattura (%)", 50, 150, 100) / 100

# --- LOGICA DI CALCOLO UNIFICATA ---
years = np.array([2026, 2027, 2028, 2029, 2030])
potenza_kw = 30 if tecnologia == "DC 30 kW" else 60
capex_unit = capex_30 if tecnologia == "DC 30 kW" else capex_60
opex_unit = 2000 if tecnologia == "DC 30 kW" else 3500

# 1. Mercato e Domanda
bev_citta = np.array([2600, 3000, 3500, 4200, 5000]) * stress_bev
quota_mercato = np.array([0.02, 0.03, 0.04, 0.045, 0.05]) * stress_cattura
auto_clienti = bev_citta * public_share * quota_mercato
energia_kwh = auto_clienti * carica_media_anno

# 2. Capacità e Dimensionamento
ore_richieste = energia_kwh / potenza_kw
ore_disp_asset = ore_max_giorno * 365
n_totale = np.ceil(ore_richieste / ore_disp_asset).astype(int)

# 3. Allocazione Location
stazione_A = np.ones(len(years))
stazione_B = np.zeros(len(years))
for i, n in enumerate(n_totale):
    if allocazione == "Multisito (Espansione in B)" and n > 1:
        stazione_B[i] = n - 1
        stazione_A[i] = 1
    else:
        stazione_A[i] = n

# 4. Finanza
ricavi = energia_kwh * prezzo_kwh
ebitda = (energia_kwh * (prezzo_kwh - costo_kwh)) - (n_totale * opex_unit)
capex_flow = np.zeros(len(years)); prev_n = 0
for i, n in enumerate(n_totale):
    capex_flow[i] = max(0, n - prev_n) * capex_unit
    prev_n = n
cf_netto = ebitda - capex_flow
cf_cum = np.cumsum(cf_netto)

# --- VISUALIZZAZIONE KPI ---
st.subheader(f"💼 Analisi Strategica: {tecnologia}")
k1, k2, k3, k4 = st.columns(4)
k1.metric("VAN (NPV)", f"€ {npf.npv(wacc, cf_netto):,.0f}")
k2.metric("TIR (IRR)", f"{(npf.irr(cf_netto)*100):.1f}%")
k3.metric("Asset Totali 2030", f"{n_totale[-1]}")
k4.metric("Saturazione 2030", f"{(ore_richieste[-1]/(n_totale[-1]*ore_disp_asset)*100):.1f}%")

# --- TUTTI I GRAFICI RICHIESTI ---
st.divider()
st.subheader("📊 Intelligence Visuale")
c1, c2 = st.columns(2)

with c1:
    st.write("**1. Break-even: Auto/Giorno per coprire i costi**")
    auto_range = np.linspace(1, 15, 20)
    margine_carica = 35 * (prezzo_kwh - costo_kwh)
    costo_fisso_day = (opex_unit + (capex_unit/5)) / 365
    fig1, ax1 = plt.subplots()
    ax1.plot(auto_range, (auto_range * margine_carica) - costo_fisso_day, color='green', linewidth=3)
    ax1.axhline(0, color='red', linestyle='--')
    ax1.set_xlabel("Ricariche giornaliere per unità")
    st.pyplot(fig1)

with c2:
    st.write("**2. Cash Flow Cumulato (Recupero)**")
    fig2, ax2 = plt.subplots()
    ax2.plot(years, cf_cum, marker='o', color='navy', linewidth=3)
    ax2.fill_between(years, cf_cum, 0, where=(cf_cum >= 0), color='green', alpha=0.1)
    ax2.axhline(0, color='black', linewidth=1)
    st.pyplot(fig2)

c3, c4 = st.columns(2)
with c3:
    st.write("**3. Allocazione Asset per Location**")
    fig3, ax3 = plt.subplots()
    ax3.bar(years, stazione_A, label='Stazione A', color='#1f77b4')
    ax3.bar(years, stazione_B, bottom=stazione_A, label='Stazione B', color='#ff7f0e')
    ax3.set_ylabel("Numero Colonnine")
    ax3.legend()
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
st.subheader("📊 Report Analitico Dettagliato")
df_master = pd.DataFrame({
    "Anno": years,
    "BEV Palermo": bev_citta.astype(int),
    "Auto in Carica": auto_clienti.astype(int),
    "Energia (kWh)": energia_kwh.astype(int),
    "Ore Richieste": ore_richieste.astype(int),
    "Asset A": stazione_A.astype(int),
    "Asset B": stazione_B.astype(int),
    "EBITDA (€)": ebitda.astype(int),
    "CF Cumulato (€)": cf_cum.astype(int)
}).set_index("Anno")
st.table(df_master)

with st.expander("🔍 Intelligence Report - Guida alla Decisione"):
    st.markdown(f"""
    - **Capacità Fisica**: Con la scelta **{tecnologia}**, ogni auto richiede **{35/potenza_kw*60:.0f} minuti** per una ricarica media. 
    - **Soglia Multisito**: Se hai scelto 'Multisito', il sistema sposta l'investimento sulla Stazione B non appena la Stazione A supera le **{ore_max_giorno} ore** di occupazione.
    - **Stress Test**: Il VAN di **€ {npf.npv(wacc, cf_netto):,.0f}** è la tua garanzia di sicurezza. Se resta positivo anche abbassando il 'Parco BEV', l'investimento è solido.
    """)
