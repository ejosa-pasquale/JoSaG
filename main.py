import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy_financial as npf

st.set_page_config(page_title="Investment Decision Tool", layout="wide")

st.title("⛽ Business Case: Ricarica DC 30 kW - Palermo")
st.markdown("Analisi della redditività e soglie di pareggio per stazione di servizio.")

# ---------------------------------------------------------
# 1) SIDEBAR - INPUT E VARIABILI DECISIONALI
# ---------------------------------------------------------
st.sidebar.header("🎯 Variabili Decisionali")
prezzo_kwh = st.sidebar.slider("Prezzo alla Colonnina (€/kWh)", 0.40, 0.90, 0.69)
capex_unit = st.sidebar.number_input("Investimento (CAPEX) per Colonnina (€)", value=25000)
costo_energia = st.sidebar.slider("Costo Energia all'ingrosso (€/kWh)", 0.15, 0.45, 0.30)

st.sidebar.header("📊 Ipotesi di Mercato")
moltiplicatore_bev = st.sidebar.slider("Crescita Parco Auto (Moltiplicatore)", 0.5, 1.5, 1.0)
quota_cattura = st.sidebar.slider("Efficacia Cattura (%)", 50, 150, 100) / 100

# ---------------------------------------------------------
# 2) LOGICA DI CALCOLO
# ---------------------------------------------------------
years = np.array([2026, 2027, 2028, 2029, 2030])
# Dati report originali corretti per scenario
bev_est = np.array([2600, 3000, 3500, 4200, 5000]) * moltiplicatore_bev
quota_mercato = np.array([0.02, 0.03, 0.04, 0.045, 0.05]) * quota_cattura

# Energia e Operatività
energia_tot = bev_est * 3000 * 0.30 * quota_mercato
n_colonnine = np.ceil(energia_tot / (30 * 8760 * 0.97 * 0.30)).astype(int)
sessioni_giorno = (energia_tot / 35) / 365 # Media 35kWh a sessione

# Economia
margine_unitario = prezzo_kwh - costo_energia
ebitda = (energia_tot * margine_unitario) - (n_colonnine * 2000)
capex_flow = np.zeros(len(years))
prev_n = 0
for i, n in enumerate(n_colonnine):
    capex_flow[i] = max(0, n - prev_n) * capex_unit
    prev_n = n

cf_netto = ebitda - capex_flow
cf_cum = np.cumsum(cf_netto)

# ---------------------------------------------------------
# 3) VISUALIZZAZIONE KPI DECISIONALI
# ---------------------------------------------------------
st.subheader("🏁 Indicatori di Successo")
k1, k2, k3, k4 = st.columns(4)

van = npf.npv(0.08, cf_netto)
k1.metric("Guadagno Netto (VAN)", f"€ {van:,.0f}", delta="Scenario Positivo" if van > 0 else "Rischio Perdita")
k2.metric("Ricariche/Giorno Media", f"{sessioni_giorno.mean():.1f} auto")
k3.metric("Pareggio (Payback)", years[np.where(cf_cum >= 0)[0][0]] if any(cf_cum >= 0) else "Oltre 2030")
k4.metric("ROI Finale", f"{(cf_cum[-1]/np.sum(capex_flow)):.2f}x")

# ---------------------------------------------------------
# 4) GRAFICI RICHIESTI
# ---------------------------------------------------------
st.divider()
st.subheader("📊 Analisi Grafica Avanzata")

g1, g2 = st.columns(2)

with g1:
    st.write("**Grafico 1: Quante auto servono per essere in utile?**")
    # Calcoliamo il break-even operativo (quante auto al giorno servono per coprire OPEX + Ammortamento annuo)
    auto_range = np.linspace(1, 20, 20)
    ebitda_simulato = (auto_range * 365 * 35 * margine_unitario) - 2000 # EBITDA per 1 colonnina
    profitto_netto_sim = ebitda_simulato - (capex_unit / 5) # Profitto annuo con ammortamento 5 anni
    
    fig1, ax1 = plt.subplots()
    ax1.plot(auto_range, profitto_netto_sim, color='green', linewidth=2)
    ax1.axhline(0, color='red', linestyle='--')
    ax1.fill_between(auto_range, profitto_netto_sim, 0, where=(profitto_netto_sim > 0), color='green', alpha=0.1)
    ax1.set_xlabel("Auto che ricaricano al giorno (per singola colonnina)")
    ax1.set_ylabel("Utile Annuo Stimato (€)")
    ax1.set_title("Soglia di Redditività (Break-even)")
    st.pyplot(fig1)
    st.caption("Il punto in cui la linea verde incrocia la rossa indica il numero minimo di auto/giorno per non perdere soldi.")

with g2:
    st.write("**Grafico 2: Cash Flow Cumulato (Rientro Investimento)**")
    fig2, ax2 = plt.subplots()
    ax2.plot(years, cf_cum, marker='o', color='navy', linewidth=3)
    ax2.axhline(0, color='black', linewidth=1)
    ax2.fill_between(years, cf_cum, 0, color='blue', alpha=0.1)
    ax2.set_title("Evoluzione del Portafoglio nel Tempo")
    st.pyplot(fig2)

st.divider()
g3, g4 = st.columns(2)

with g3:
    st.write("**Grafico 3: Scalabilità (Energia vs Colonnine)**")
    fig3, ax3 = plt.subplots()
    ax3.bar(years, energia_tot/1000, color='gray', alpha=0.4, label="Domanda (MWh)")
    ax3_tw = ax3.twinx()
    ax3_tw.step(years, n_colonnine, where='post', color='red', label="N. Colonnine", linewidth=2)
    ax3.set_title("Crescita della Domanda a Palermo")
    st.pyplot(fig3)

with g4:
    st.write("**Grafico 4: Guadagno vs Investimento (Margini)**")
    fig4, ax4 = plt.subplots()
    labels = ['Investimento Totale', 'EBITDA Totale (5 anni)']
    values = [np.sum(capex_flow), np.sum(ebitda)]
    ax4.bar(labels, values, color=['#e74c3c', '#2ecc71'])
    ax4.set_title("Confronto CAPEX vs Ritorno Lordo")
    st.pyplot(fig4)

# Tabella Finale
st.subheader("📑 Tabella Riepilogativa")
st.dataframe(pd.DataFrame({
    "Anno": years,
    "Auto/Giorno": sessioni_giorno.round(1),
    "Energia (kWh)": energia_tot.astype(int),
    "Colonnine": n_colonnine,
    "Investimento (€)": capex_flow.astype(int),
    "Margine Lordo (€)": ebitda.astype(int),
    "Cash Flow Cum. (€)": cf_cum.astype(int)
}).set_index("Anno"), use_container_width=True)
