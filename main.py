import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy_financial as npf

st.set_page_config(page_title="Investment Tool DC30", layout="wide")

# --- TITOLO E INTRODUZIONE ---
st.title("⚡ DC30 Investment Strategy: Stazione di Servizio Palermo")
st.markdown("""
Questo strumento permette al Management di testare la tenuta del Business Case sotto diversi scenari di mercato e scelte di pricing.
""")

# --- SIDEBAR: TUTTI I PARAMETRI SPIEGATI ---
st.sidebar.header("🎯 Variabili Decisionali (Le tue leve)")
with st.sidebar.expander("💼 Strategia Commerciale", expanded=True):
    prezzo_kwh = st.number_input("Prezzo vendita (€/kWh)", value=0.69, step=0.01, help="Prezzo al pubblico. Influenza direttamente il margine e il TIR.")
    costo_kwh = st.number_input("Costo energia (€/kWh)", value=0.30, step=0.01, help="Costo di acquisto energia. Un aumento dello 0.10€ può distruggere la redditività.")
    capex_unit = st.number_input("CAPEX per unità (€)", value=25000, help="Costo totale di acquisto e installazione per singola colonnina.")
    wacc = st.slider("Costo del Capitale (WACC %)", 4, 12, 8, help="Il rendimento minimo atteso per giustificare il rischio di questo progetto.") / 100

st.sidebar.header("📊 Ipotesi di Scenario (Il Mercato)")
with st.sidebar.expander("🌍 Evoluzione Domanda"):
    target_util = st.slider("Utilizzo Target (%)", 10, 50, 30, help="Fattore di saturazione desiderato. Determina quando 'scatta' l'acquisto di una nuova colonnina.") / 100
    public_share = st.slider("Quota Ricarica Pubblica (%)", 10, 50, 30, help="Percentuale di utenti che ricaricano fuori casa. Più è alta, più il mercato è vasto.") / 100
    bev_crescita = st.slider("Stress Test Parco BEV (%)", 50, 150, 100, help="Varia la velocità di adozione delle auto elettriche a Palermo rispetto ai dati 2026-2030.") / 100
    cattura_share = st.slider("Quota Cattura Stazione (%)", 50, 150, 100, help="Capacità della stazione di attrarre clienti rispetto ai competitor nell'area.") / 100

# --- LOGICA DI CALCOLO ---
years = np.array([2026, 2027, 2028, 2029, 2030])
bev_est = np.array([2600, 3000, 3500, 4200, 5000]) * (bev_crescita)
quota_stazione = np.array([0.02, 0.03, 0.04, 0.045, 0.05]) * (cattura_share)

# Domanda e Dimensionamento
energia_tot = bev_est * 3000 * public_share * quota_stazione
cap_max_annua = 30 * 8760 * 0.97 * target_util
n_colonnine = np.ceil(energia_tot / cap_max_annua).astype(int)
auto_day = (energia_tot / 35) / 365 # 35 kWh ricarica media

# Flussi di Cassa
ebitda = (energia_tot * (prezzo_kwh - costo_kwh)) - (n_colonnine * 2000)
capex_flow = np.zeros(len(years)); prev_n = 0
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

k1.metric("VAN (NPV)", f"€ {van:,.0f}", help="Se positivo, il progetto crea valore sopra il costo del capitale.")
k2.metric("TIR (IRR)", f"{tir*100:.1f}%" if not np.isnan(tir) else "N/D", help="Rendimento intrinseco del progetto.")
k3.metric("Auto/Giorno (Media)", f"{auto_day.mean():.1f}")
k4.metric("Payback Period", years[np.where(cf_cum >= 0)[0][0]] if any(cf_cum >= 0) else "Oltre 2030")

# --- GRAFICI DECISIONALI ---
st.divider()
st.subheader("📊 Analisi della Redditività")
c1, c2 = st.columns(2)

with c1:
    st.write("**Grafico 1: Quante auto servono per essere in utile?**")
    auto_range = np.linspace(1, 15, 15)
    margine_per_auto = 35 * (prezzo_kwh - costo_kwh)
    # Costo fisso giornaliero = (OPEX + Ammortamento CAPEX in 5 anni) / 365
    break_even_point = (2000 + (capex_unit/5)) / 365
    profitto_day = (auto_range * margine_per_auto) - break_even_point
    
    fig1, ax1 = plt.subplots()
    ax1.plot(auto_range, profitto_day, color='green', linewidth=3, label="Profitto Giornaliero")
    ax1.axhline(0, color='red', linestyle='--', linewidth=1)
    ax1.set_xlabel("Ricariche Giornaliere (n. auto)")
    ax1.set_ylabel("Profitto Netto (€/giorno)")
    ax1.grid(alpha=0.2)
    st.pyplot(fig1)

with c2:
    st.write("**Grafico 2: Cash Flow Cumulato (Recupero Investimento)**")
    fig2, ax2 = plt.subplots()
    ax2.bar(years, cf_netto, color='skyblue', alpha=0.3, label="Cash Flow Annuo")
    ax2.plot(years, cf_cum, marker='o', color='navy', linewidth=3, label="CF Cumulato")
    ax2.axhline(0, color='black', linewidth=1)
    ax2.legend()
    st.pyplot(fig2)

st.divider()
c3, c4 = st.columns(2)

with c3:
    st.write("**Grafico 3: Scalabilità Asset (Domanda vs Colonnine)**")
    fig3, ax3 = plt.subplots()
    ax3.bar(years, energia_tot/1000, color='gray', alpha=0.5, label="Energia (MWh)")
    ax3_tw = ax3.twinx()
    ax3_tw.step(years, n_colonnine, where='post', color='red', linewidth=2, label="N. Colonnine")
    ax3_tw.set_ylim(0, n_colonnine.max() + 1)
    ax3.set_title("Evoluzione Dimensionamento")
    st.pyplot(fig3)

with g4 if 'g4' in locals() else c4:
    st.write("**Grafico 4: Struttura Costi e Ricavi (5 Anni)**")
    fig4, ax4 = plt.subplots()
    labels = ['Ricavi Tot', 'Costi Energia', 'CAPEX Tot', 'EBITDA Tot']
    vals = [ricavi.sum(), (energia_tot * costo_kwh).sum(), capex_flow.sum(), ebitda.sum()]
    ax4.bar(labels, vals, color=['#27ae60', '#e67e22', '#e74c3c', '#2980b9'])
    st.pyplot(fig4)

# --- GLOSSARIO E TABELLA ---
st.divider()
st.subheader("🔍 Intelligence Report & Glossario")
col_msg, col_tab = st.columns([1, 1])

with col_msg:
    st.info(f"""
    **Spiegazione Variabili Chiave:**
    - **Utilizzo Target**: La soglia di efficienza. Se impostata al 30%, significa che dimensioniamo l'impianto affinché ogni colonnina sia occupata ~7 ore al giorno.
    - **Quota Cattura**: Rappresenta il tuo vantaggio competitivo. Un valore del 100% significa che segui il report Palermo; 50% significa che i competitor ti stanno rubando metà dei clienti.
    - **Stress Test BEV**: Modifica l'adozione delle auto. Se il mercato elettrico rallenta, questo parametro ti dice se il progetto regge comunque.
    - **Margine/Auto**: Con le tue impostazioni, guadagni **€ {margine_per_auto:.2f}** per ogni ricarica media da 35 kWh.
    """)

with col_tab:
    st.write("**Dati Analitici Proiettati**")
    st.table(pd.DataFrame({
        "Auto/Giorno": auto_day.round(1),
        "Energia (kWh)": energia_tot.astype(int),
        "EBITDA (€)": ebitda.astype(int),
        "CF Cumulato (€)": cf_cum.astype(int)
    }, index=years))
