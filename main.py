import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy_financial as npf

st.set_page_config(page_title="CFO Suite - DC30 Palermo", layout="wide")

st.title("🏛️ Decision Support System: Infrastruttura DC30 kW")
st.markdown("Analisi strategica e finanziaria completa per il Management.")

# --- SIDEBAR: LEVE DECISIONALI E SCENARI ---
st.sidebar.header("🕹️ Variabili Decisionali (GM)")
with st.sidebar.expander("💼 Strategia e Pricing", expanded=True):
    prezzo_kwh = st.number_input("Prezzo vendita (€/kWh)", value=0.69, step=0.01)
    costo_kwh = st.number_input("Costo energia (€/kWh)", value=0.30, step=0.01)
    capex_unit = st.number_input("CAPEX per unità (€)", value=25000)
    opex_fisso = st.number_input("OPEX annuo/colonnina (€)", value=2000)

st.sidebar.header("📊 Ipotesi di Scenario (Mercato)")
with st.sidebar.expander("🌍 Driver di Domanda", expanded=True):
    stress_bev = st.slider("Stress Test Parco BEV (%)", 50, 150, 100) / 100
    public_share = st.slider("Quota Ricarica Pubblica (%)", 10, 50, 30) / 100
    carica_media_anno = st.number_input("Carica Media/Anno (kWh)", value=3000)
    stress_cattura = st.slider("Efficacia Cattura Stazione (%)", 50, 150, 100) / 100

# --- LOGICA DI CALCOLO (Fedele al Report) ---
years = np.array([2026, 2027, 2028, 2029, 2030])
bev_citta = np.array([2600, 3000, 3500, 4200, 5000]) * stress_bev
quota_stazione = np.array([0.02, 0.03, 0.04, 0.045, 0.05]) * stress_cattura

# Conversione in clienti reali
auto_in_carica = bev_citta * public_share * quota_stazione
energia_kwh = auto_in_carica * carica_media_anno

# Dimensionamento e Finanza
n_colonnine = np.ceil(energia_kwh / (30 * 8760 * 0.97 * 0.30)).astype(int)
ricavi = energia_kwh * prezzo_kwh
ebitda = (energia_kwh * (prezzo_kwh - costo_kwh)) - (n_colonnine * opex_fisso)

capex_flow = np.zeros(len(years))
prev_n = 0
for i, n in enumerate(n_colonnine):
    capex_flow[i] = max(0, n - prev_n) * capex_unit
    prev_n = n

cf_netto = ebitda - capex_flow
cf_cum = np.cumsum(cf_netto)

# --- KPI EXECUTIVE ---
st.subheader("🏁 Executive Summary")
k1, k2, k3, k4 = st.columns(4)
van = npf.npv(0.08, cf_netto)
k1.metric("VAN (NPV)", f"€ {van:,.0f}", help="Valore Attuale Netto (WACC 8%)")
k2.metric("TIR (IRR)", f"{(npf.irr(cf_netto)*100):.1f}%", help="Rendimento interno")
k3.metric("Auto/Giorno (Media)", f"{((energia_kwh/35)/365).mean():.1f}")
k4.metric("Payback", years[np.where(cf_cum >= 0)[0][0]] if any(cf_cum >= 0) else "Oltre 2030")

# --- TUTTI I GRAFICI RICHIESTI ---
st.divider()
st.subheader("📊 Analisi Decisionale Completa")

c1, c2 = st.columns(2)
with c1:
    st.write("**1. Soglia di Redditività (Break-even)**")
    auto_range = np.linspace(1, 15, 20)
    margine_carica = 35 * (prezzo_kwh - costo_kwh)
    break_even_point = (opex_fisso + (capex_unit/5)) / 365
    profitto_day = (auto_range * margine_carica) - break_even_point
    fig1, ax1 = plt.subplots()
    ax1.plot(auto_range, profitto_day, color='green', linewidth=3)
    ax1.axhline(0, color='red', linestyle='--')
    ax1.set_xlabel("Ricariche Giornaliere (n. auto)")
    ax1.set_title("Utile Giornaliero per Colonnina")
    st.pyplot(fig1)

with c2:
    st.write("**2. Cash Flow Cumulato (Recupero Investimento)**")
    fig2, ax2 = plt.subplots()
    ax2.plot(years, cf_cum, marker='o', color='navy', linewidth=3)
    ax2.fill_between(years, cf_cum, 0, where=(cf_cum >= 0), color='green', alpha=0.1)
    ax2.axhline(0, color='black', linewidth=1)
    st.pyplot(fig2)

c3, c4 = st.columns(2)
with c3:
    st.write("**3. Scalabilità Asset (Domanda vs Capacità)**")
    fig3, ax3 = plt.subplots()
    ax3.bar(years, energia_kwh/1000, color='gray', alpha=0.5, label="Energia (MWh)")
    ax3_tw = ax3.twinx()
    ax3_tw.step(years, n_colonnine, where='post', color='red', linewidth=2, label="N. Colonnine")
    ax3.set_title("MWh Venduti vs Colonnine Installate")
    st.pyplot(fig3)

with c4:
    st.write("**4. Struttura dei Margini (5 Anni)**")
    fig4, ax4 = plt.subplots()
    labels = ['Ricavi Tot', 'Costi Energia', 'CAPEX Tot', 'EBITDA Tot']
    vals = [ricavi.sum(), (energia_kwh * costo_kwh).sum(), capex_flow.sum(), ebitda.sum()]
    ax4.bar(labels, vals, color=['#2ecc71', '#e67e22', '#e74c3c', '#3498db'])
    st.pyplot(fig4)

# --- TABELLA DETTAGLIATA (Richiesta) ---
st.divider()
st.subheader("📊 Tabella Analitica: Dal Mercato alla Cassa")
df_table = pd.DataFrame({
    "Anno": years,
    "BEV in Città": bev_citta.astype(int),
    "Clienti in Carica (Previsti)": auto_in_carica.astype(int),
    "Carica Media (kWh/anno)": [carica_media_anno] * 5,
    "Energia Totale (kWh)": energia_kwh.astype(int),
    "N. Colonnine": n_colonnine,
    "EBITDA (€)": ebitda.astype(int),
    "Cash Flow Cumulato (€)": cf_cum.astype(int)
}).set_index("Anno")
st.table(df_table)

# --- SPIEGAZIONE VARIABILI ---
st.divider()
with st.expander("📚 Glossario e Spiegazione Variabili per la Decisione"):
    st.markdown(f"""
    - **BEV in Città**: Numero totale di auto elettriche stimate a Palermo. È la base del tuo mercato.
    - **Clienti in Carica**: Numero di auto che scelgono la tua stazione. Si ottiene applicando la *Quota Pubblica* (chi non carica a casa) e la *Efficacia Cattura* (tua forza rispetto ai competitor).
    - **Carica Media**: Quanta energia ogni cliente "catturato" acquista da te in un anno.
    - **Break-even Operativo**: Il Grafico 1 mostra che con i costi attuali ti servono **{break_even_point/margine_carica:.1f} auto al giorno** per coprire i costi fissi e l'ammortamento.
    - **Scalabilità Asset**: Il Grafico 3 indica quando il volume di vendita giustifica l'acquisto di una nuova macchina per non superare il target di utilizzo del 30%.
    """)
