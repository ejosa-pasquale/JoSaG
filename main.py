import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy_financial as npf

st.set_page_config(page_title="CFO Suite - DC30 Palermo", layout="wide")

st.title("🏛️ Decision Support System: DC30 kW Palermo")
st.markdown("Analisi strategica basata sul report del 22/01/2026.")

# --- SIDEBAR: LEVE DECISIONALI E SCENARI ---
st.sidebar.header("🕹️ Variabili Decisionali (GM)")
prezzo_kwh = st.sidebar.number_input("Prezzo vendita (€/kWh)", value=0.69, step=0.01)
capex_unit = st.sidebar.number_input("CAPEX per unità (€)", value=25000)
costo_kwh = st.sidebar.number_input("Costo energia (€/kWh)", value=0.30)
opex_fisso = st.sidebar.number_input("OPEX annuo/colonnina (€)", value=2000)

st.sidebar.header("📊 Ipotesi di Scenario (Mercato)")
with st.sidebar.expander("🌍 Driver di Domanda", expanded=True):
    stress_bev = st.slider("Stress Test Parco BEV (%)", 50, 150, 100) / 100
    public_share = st.slider("Quota Ricarica Pubblica (%)", 10, 50, 30, help="Percentuale di chi carica fuori casa") / 100
    carica_media_anno = st.number_input("Carica Media/Anno (kWh)", value=3000, help="Consumo annuo totale di un BEV")
    stress_cattura = st.slider("Efficacia Cattura Stazione (%)", 50, 150, 100) / 100

# --- LOGICA DI CALCOLO (FEDELE AL PDF) ---
years = np.array([2026, 2027, 2028, 2029, 2030])

# 1. Parco BEV Palermo (da report)
bev_citta = np.array([2600, 3000, 3500, 4200, 5000]) * stress_bev

# 2. Domanda Pubblica Totale (kWh)
domanda_pubblica_totale = bev_citta * carica_media_anno * public_share

# 3. Quota catturata dalla tua stazione (da report)
quota_stazione_base = np.array([0.02, 0.03, 0.04, 0.045, 0.05]) * stress_cattura
energia_kwh = domanda_pubblica_totale * quota_stazione_base

# 4. Clienti previsti in carica (Auto/anno)
auto_in_carica = energia_kwh / (carica_media_anno * public_share)

# Dimensionamento Asset
cap_max_30pct = 30 * 8760 * 0.97 * 0.30 
n_colonnine = np.ceil(energia_kwh / cap_max_30pct).astype(int)

# Finanza
ricavi = energia_kwh * prezzo_kwh
ebitda = (energia_kwh * (prezzo_kwh - costo_kwh)) - (n_colonnine * opex_fisso)

capex_flow = np.zeros(len(years))
prev_n = 0
for i, n in enumerate(n_colonnine):
    capex_flow[i] = max(0, n - prev_n) * capex_unit
    prev_n = n

cf_netto = ebitda - capex_flow
cf_cum = np.cumsum(cf_netto)

# --- VISUALIZZAZIONE KPI ---
st.subheader("🏁 Executive Summary")
k1, k2, k3, k4 = st.columns(4)
van = npf.npv(0.08, cf_netto)
k1.metric("VAN (NPV)", f"€ {van:,.0f}")
k2.metric("ROI Finale", f"{(cf_cum[-1]/np.sum(capex_flow)):.2f}x")
k3.metric("Auto/Giorno (2030)", f"{((energia_kwh[-1]/35)/365):.1f}")
k4.metric("Payback", years[np.where(cf_cum >= 0)[0][0]] if any(cf_cum >= 0) else "Oltre 2030")

# --- GRAFICI (RIPRISTINATI) ---
st.divider()
st.subheader("📈 Analisi Visuale")
c1, c2 = st.columns(2)

with c1:
    st.write("**Soglia di Redditività (Break-even)**")
    auto_range = np.linspace(1, 15, 20)
    margine_carica = 35 * (prezzo_kwh - costo_kwh) # 35kWh è la sessione media
    costo_fisso_giorno = (opex_fisso + (capex_unit/5)) / 365
    profitto_day = (auto_range * margine_carica) - costo_fisso_giorno
    fig1, ax1 = plt.subplots()
    ax1.plot(auto_range, profitto_day, color='green', linewidth=2)
    ax1.axhline(0, color='red', linestyle='--')
    ax1.set_xlabel("Ricariche/Giorno")
    ax1.set_ylabel("Utile (€/giorno)")
    st.pyplot(fig1)

with c2:
    st.write("**Rientro dell'Investimento (Cash Flow Cumulato)**")
    fig2, ax2 = plt.subplots()
    ax2.plot(years, cf_cum, marker='o', color='navy', linewidth=3)
    ax2.axhline(0, color='black', linewidth=1)
    ax2.fill_between(years, cf_cum, 0, where=(cf_cum >= 0), color='green', alpha=0.1)
    ax2.fill_between(years, cf_cum, 0, where=(cf_cum < 0), color='red', alpha=0.1)
    st.pyplot(fig2)

# --- TABELLA DETTAGLIATA (CON AGGIUNTE RICHIESTE) ---
st.divider()
st.subheader("📊 Tabella Analitica di Supporto")
df_board = pd.DataFrame({
    "Anno": years,
    "BEV in Città": bev_citta.astype(int),
    "Auto in Carica (Previste)": auto_in_carica.astype(int),
    "Carica Media (kWh/y)": [carica_media_anno] * 5,
    "Energia Totale (kWh)": energia_kwh.astype(int),
    "N. Colonnine": n_colonnine,
    "EBITDA (€)": ebitda.astype(int),
    "CF Cumulato (€)": cf_cum.astype(int)
}).set_index("Anno")

st.table(df_board)

# --- SPIEGAZIONI VARIABILI ---
st.divider()
with st.expander("🔍 Spiegazione Variabili e Intelligence Report"):
    st.markdown(f"""
    - **BEV in Città**: Il parco circolante totale di Palermo (Scenario base: crescita fino a 5.000 auto).
    - **Auto in Carica**: Quante auto del parco circolante intercetti effettivamente nella tua stazione.
    - **Carica Media**: L'energia totale che ogni cliente consuma presso di te in un intero anno.
    - **EBITDA**: Il guadagno operativo. Nel 2030, con queste impostazioni, è di **€ {ebitda[-1]:,.0f}**.
    - **VAN (NPV)**: Indica se stai creando valore. Un VAN di € {van:,.0f} significa che il progetto è ampiamente profittevole rispetto a un investimento finanziario standard.
    """)
