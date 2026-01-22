import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy_financial as npf

st.set_page_config(page_title="CFO Decision Suite - DC30", layout="wide")

# --- HEADER EXECUTIVE ---
st.title("🏛️ CFO Decision Suite: Infrastruttura Ricarica DC30")
st.markdown("""
**Analisi Strategica per Stazione di Servizio (Palermo).**
Questo modello valuta la bancabilità del progetto basandosi sui flussi di cassa netti e sulla saturazione degli asset.
""")

# --- SIDEBAR: LEVE DECISIONALI E SCENARI ---
st.sidebar.header("🕹️ Leve Decisionali (GM)")
prezzo_vendita = st.sidebar.slider("Prezzo di Vendita (€/kWh)", 0.40, 0.95, 0.69)
capex_unitario = st.sidebar.number_input("CAPEX per Colonnina (€)", value=25000)
costo_wacc = st.sidebar.slider("Costo del Capitale (WACC %)", 4, 12, 8) / 100

st.sidebar.header("⚠️ Ipotesi di Rischio (Scenario)")
stress_domanda = st.sidebar.slider("Stress Test Domanda (Auto %)", 50, 150, 100) / 100
costo_energia = st.sidebar.slider("Costo Acquisto Energia (€/kWh)", 0.15, 0.50, 0.30)

# --- LOGICA DI CALCOLO ---
years = np.array([2026, 2027, 2028, 2029, 2030])
# Dati Report Palermo applicando lo stress test sulla domanda
bev_stimati = np.array([2600, 3000, 3500, 4200, 5000]) * stress_domanda
quota_mercato = np.array([0.02, 0.03, 0.04, 0.045, 0.05])
energia_kwh = bev_stimati * 3000 * 0.30 * quota_mercato

# Dimensionamento asset (Capacità max teorica 30kW * 8760h * 0.97 uptime)
cap_max_colonnina = 30 * 8760 * 0.97
# Utilizziamo il 30% come target operativo per il dimensionamento
n_colonnine = np.ceil(energia_kwh / (cap_max_colonnina * 0.30)).astype(int)

# Conto Economico
ricavi = energia_kwh * prezzo_vendita
opex = n_colonnine * 2000
margine_lordo = energia_kwh * (prezzo_vendita - costo_energia)
ebitda = margine_lordo - opex

# Flussi di Cassa
capex_flow = np.zeros(len(years))
prev_n = 0
for i, n in enumerate(n_colonnine):
    capex_flow[i] = max(0, n - prev_n) * capex_unitario
    prev_n = n

cf_netto = ebitda - capex_flow
cf_cumulato = np.cumsum(cf_netto)

# --- OUTPUT 1: KPI FINANZIARI (VISIONE CFO) ---
st.subheader("📊 Indicatori Finanziari di Performance")
c1, c2, c3, c4 = st.columns(4)

van = npf.npv(costo_wacc, cf_netto)
tir = npf.irr(cf_netto)
roi_cum = cf_cumulato[-1] / np.sum(capex_flow) if np.sum(capex_flow) > 0 else 0

c1.metric("VAN (NPV)", f"€ {van:,.0f}", help="Valore Attuale Netto. Se > 0, l'investimento crea valore.")
c2.metric("TIR (IRR)", f"{tir*100:.1f}%" if not np.isnan(tir) else "N/D", help="Rendimento annuo del progetto.")
c3.metric("ROI Cumulato", f"{roi_cum:.2f}x", help="Quante volte rientra il capitale investito al 2030.")
c4.metric("Payback Period", years[np.where(cf_cumulato >= 0)[0][0]] if any(cf_cumulato >= 0) else "Oltre 2030")

# --- OUTPUT 2: GRAFICI DECISIONALI (VISIONE GM) ---
st.divider()
st.subheader("📈 Analisi della Redditività e Saturazione")

g1, g2 = st.columns(2)

with g1:
    st.write("**Soglia di Pareggio Operativo (Break-even)**")
    # Quante auto servono al giorno per coprire costi e ammortamento?
    auto_day_range = np.linspace(1, 20, 20)
    # 35 kWh media ricarica. 5 anni ammortamento.
    margine_per_auto = 35 * (prezzo_vendita - costo_energia)
    costi_fissi_giorno = (2000 + (capex_unitario/5)) / 365
    profitto_day = (auto_day_range * margine_per_auto) - costi_fissi_giorno
    
    fig1, ax1 = plt.subplots()
    ax1.plot(auto_day_range, profitto_day, color='green', linewidth=2, label="Utile Giornaliero")
    ax1.axhline(0, color='red', linestyle='--')
    ax1.set_xlabel("Ricariche (Auto) al Giorno per Colonnina")
    ax1.set_ylabel("Profitto Netto Giornaliero (€)")
    ax1.grid(alpha=0.3)
    st.pyplot(fig1)
    st.caption("Il punto di incontro con la linea rossa indica quante auto devi servire OGNI GIORNO per non essere in perdita.")

with g2:
    st.write("**Evoluzione Cash Flow Cumulato**")
    fig2, ax2 = plt.subplots()
    ax2.fill_between(years, cf_cumulato, color="skyblue", alpha=0.4)
    ax2.plot(years, cf_cumulato, marker='o', color="navy", linewidth=3)
    ax2.axhline(0, color='black', linewidth=1)
    ax2.set_title("Rientro dell'Investimento Totale (€)")
    st.pyplot(fig2)

g3, g4 = st.columns(2)

with g3:
    st.write("**Scalabilità Asset (Domanda vs Offerta)**")
    # Calcoliamo la saturazione: energia erogata / capacità max delle colonnine installate
    saturazione = (energia_kwh / (n_colonnine * cap_max_colonnina)) * 100
    fig3, ax3 = plt.subplots()
    ax3.bar(years, energia_kwh/1000, color='lightgray', label="Energia Venduta (MWh)")
    ax3_tw = ax3.twinx()
    ax3_tw.plot(years, n_colonnine, color='red', marker='s', label="N. Colonnine")
    ax3.set_title("Crescita del Business")
    st.pyplot(fig3)

with g4:
    st.write("**Efficienza dell'Asset (Saturazione %)**")
    fig4, ax4 = plt.subplots()
    ax4.plot(years, saturazione, marker='D', color='orange', linewidth=2)
    ax4.axhline(30, color='red', linestyle='--', label="Target Ottimale (30%)")
    ax4.set_ylim(0, 50)
    ax4.set_ylabel("% Tempo Utilizzo")
    ax4.set_title("Saturazione delle Colonnine")
    ax4.legend()
    st.pyplot(fig4)

# --- SPIEGAZIONE PER IL MANAGEMENT ---
st.divider()
st.subheader("📑 Nota Metodologica per la Decisione")
col_msg, col_tab = st.columns([1, 1])

with col_msg:
    st.markdown(f"""
    **Sintesi Manageriale:**
    1.  **Soglia di Sopravvivenza:** Con i costi attuali, la tua stazione deve servire circa **{abs(costi_fissi_giorno/margine_per_auto):.1f} auto al giorno** per ogni colonnina per coprire l'investimento.
    2.  **Rischio Energia:** Un aumento del costo dell'energia di 0.10€ riduce l'EBITDA totale del **{(0.10/prezzo_vendita)*100:.0f}%**.
    3.  **Scalabilità:** Il modello suggerisce di passare da 1 a {n_colonnine[-1]} colonnine entro il 2030. Ogni salto richiede un nuovo esborso di € {capex_unitario}.
    4.  **Verdetto:** Se il TIR ({tir*100:.1f}%) è superiore al tuo costo del capitale ({costo_wacc*100}%), l'investimento è **economicamente profittevole**.
    """)

with col_tab:
    st.write("**Tabella di Marcia Finanziaria**")
    st.table(pd.DataFrame({
        "Auto/Giorno": ((energia_kwh/35)/365).round(1),
        "EBITDA (€)": ebitda.astype(int),
        "CAPEX (€)": capex_flow.astype(int),
        "CF Cumulato (€)": cf_cumulato.astype(int)
    }, index=years))
