import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy_financial as npf

st.set_page_config(page_title="CFO Suite - Investimento DC30", layout="wide")

st.title("🏛️ Decision Support System: Infrastruttura DC30 kW")
st.markdown("Analisi strategica e finanziaria per l'installazione di colonnine in stazione di servizio.")

# --- SIDEBAR: LEVE DECISIONALI VS IPOTESI DI MERCATO ---
st.sidebar.header("🕹️ Variabili Decisionali (Le tue leve)")
with st.sidebar.expander("💼 Strategia e Pricing", expanded=True):
    prezzo_kwh = st.number_input("Prezzo vendita (€/kWh)", value=0.69, step=0.01, help="Lezione commerciale: un prezzo troppo alto riduce la domanda, uno troppo basso erode il margine.")
    costo_kwh = st.number_input("Costo energia (€/kWh)", value=0.30, step=0.01, help="Prezzo di acquisto dell'energia (materia prima).")
    capex_unit = st.number_input("CAPEX per unità (€)", value=25000, help="Costo totale di acquisto e installazione (Macchina + Scavi + Allacci).")
    wacc = st.slider("Costo del Capitale (WACC %)", 4, 12, 8, help="Tasso di rendimento minimo richiesto per giustificare l'investimento.") / 100

st.sidebar.header("📊 Ipotesi di Mercato (I Rischi)")
with st.sidebar.expander("🌍 Scenario Esterno"):
    crescita_bev = st.slider("Crescita Parco Auto (Stress Test %)", 50, 150, 100, help="Variazione rispetto alle stime ufficiali di adozione elettrica a Palermo.") / 100
    quota_cattura = st.slider("Capacità di Cattura (%)", 50, 150, 100, help="Efficacia della tua stazione nel catturare clienti rispetto alla concorrenza locale.") / 100
    utilizzazione_target = st.slider("Soglia Utilizzo Target (%)", 10, 50, 30, help="Livello di saturazione oltre il quale decidi di installare una nuova colonnina.") / 100

# --- LOGICA DI CALCOLO (Basata su Report Palermo) ---
years = np.array([2026, 2027, 2028, 2029, 2030])
bev_palermo = np.array([2600, 3000, 3500, 4200, 5000]) * crescita_bev
quota_mercato = np.array([0.02, 0.03, 0.04, 0.045, 0.05]) * quota_cattura

# Calcolo Volumi e Ricavi
energia_kwh = bev_palermo * 3000 * 0.30 * quota_mercato
ricavi = energia_kwh * prezzo_kwh
margine_lordo = energia_kwh * (prezzo_kwh - costo_kwh)

# Dimensionamento
cap_annua_colonnina = 30 * 8760 * 0.97 * utilizzazione_target
n_colonnine = np.ceil(energia_kwh / cap_annua_colonnina).astype(int)
opex = n_colonnine * 2000

# Cash Flow
ebitda = margine_lordo - opex
capex_flow = np.zeros(len(years))
prev_n = 0
for i, n in enumerate(n_colonnine):
    capex_flow[i] = max(0, n - prev_n) * capex_unit
    prev_n = n

cf_netto = ebitda - capex_flow
cf_cum = np.cumsum(cf_netto)

# --- DASHBOARD KPI ---
st.subheader("🏁 Executive Summary")
k1, k2, k3, k4 = st.columns(4)
van = npf.npv(wacc, cf_netto)
tir = npf.irr(cf_netto)

k1.metric("VAN (Net Present Value)", f"€ {van:,.0f}", help="Ricchezza netta creata dal progetto al netto del costo del denaro.")
k2.metric("TIR (IRR)", f"{tir*100:.1f}%" if not np.isnan(tir) else "N/D", help="Rendimento annuo del capitale investito.")
k3.metric("Ricariche/Giorno (Media)", f"{((energia_kwh/35)/365).mean():.1f}")
k4.metric("Payback Period", years[np.where(cf_cum >= 0)[0][0]] if any(cf_cum >= 0) else "Oltre 2030")

# --- ANALISI GRAFICA ---
st.divider()
col_g1, col_g2 = st.columns(2)

with col_g1:
    st.write("**Grafico 1: Analisi del Punto di Pareggio (Break-even)**")
    # Quante auto al giorno servono per coprire OPEX + Ammortamento (5 anni)?
    auto_range = np.linspace(1, 15, 20)
    margine_per_auto = 35 * (prezzo_kwh - costo_kwh)
    break_even_fissi = (2000 + (capex_unit/5)) / 365
    profitto_giornaliero = (auto_range * margine_per_auto) - break_even_fissi
    
    fig1, ax1 = plt.subplots()
    ax1.plot(auto_range, profitto_giornaliero, color='green', linewidth=2)
    ax1.axhline(0, color='red', linestyle='--')
    ax1.set_xlabel("Ricariche (Auto) al giorno per singola colonnina")
    ax1.set_ylabel("Utile giornaliero (€)")
    st.pyplot(fig1)
    st.caption("Il punto di incrocio indica il traffico minimo necessario per non essere in perdita.")

with col_g2:
    st.write("**Grafico 2: Cash Flow Cumulato (Rientro Investimento)**")
    fig2, ax2 = plt.subplots()
    ax2.bar(years, cf_netto, color='skyblue', alpha=0.4, label="Cash Flow Netto")
    ax2.plot(years, cf_cum, marker='o', color='navy', linewidth=3, label="Cumulato")
    ax2.axhline(0, color='black', linewidth=1)
    ax2.legend()
    st.pyplot(fig2)

st.divider()
col_g3, col_g4 = st.columns(2)

with col_g3:
    st.write("**Grafico 3: Scalabilità Asset (Domanda vs Offerta)**")
    fig3, ax3 = plt.subplots()
    ax3.bar(years, energia_kwh/1000, color='gray', alpha=0.5, label="Energia Venduta (MWh)")
    ax3_tw = ax3.twinx()
    ax3_tw.step(years, n_colonnine, where='post', color='red', linewidth=2, label="N. Colonnine")
    ax3.set_title("Evoluzione Dimensionamento")
    st.pyplot(fig3)

with col_g4:
    st.write("**Grafico 4: Struttura Costi vs Ricavi (5 anni)**")
    fig4, ax4 = plt.subplots()
    labels = ['Ricavi', 'Costi Energia', 'CAPEX', 'EBITDA']
    values = [ricavi.sum(), (energia_kwh * costo_kwh).sum(), capex_flow.sum(), ebitda.sum()]
    ax4.bar(labels, values, color=['#27ae60', '#e67e22', '#e74c3c', '#2980b9'])
    st.pyplot(fig4)

# --- GLOSSARIO E TABELLA ---
st.divider()
st.subheader("📖 Intelligence Report: Guida alle Variabili")
col_msg, col_tab = st.columns([1, 1])

with col_msg:
    st.info("""
    **Legenda per il Management:**
    - **Utilizzo Target**: La tua soglia di efficienza operativa. Se lo imposti basso, compri colonnine prima (più servizio, meno rendimento).
    - **Quota Cattura**: Il tuo potere competitivo. Se scende, significa che i clienti preferiscono altre stazioni vicine.
    - **WACC**: Rappresenta il 'costo opportunità'. Se il TIR è inferiore al WACC, conviene investire in altro.
    - **Stress Test BEV**: Simula un rallentamento delle vendite auto a Palermo.
    """)

with col_tab:
    st.write("**Dettagli Analitici**")
    st.table(pd.DataFrame({
        "Ricariche/Giorno": ((energia_kwh/35)/365).round(1),
        "EBITDA (€)": ebitda.astype(int),
        "Cash Flow Cum. (€)": cf_cum.astype(int)
    }, index=years))
