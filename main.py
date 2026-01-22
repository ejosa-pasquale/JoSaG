import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy_financial as npf

st.set_page_config(page_title="Executive Charging Suite", layout="wide")

st.title("🛡️ Executive Support System: eVFs for EOS")
st.markdown("### *Investment Readiness Tool: Trasformare i dati del parco auto in decisioni infrastrutturate bancabili.*")

# --- SIDEBAR: TUTTE LE VARIABILI (MARKET FUNNEL + FINANZA + TECNICA) ---
st.sidebar.header("🕹️ Variabili di Mercato (Market Funnel)")
with st.sidebar.expander("🌍 Scenario Parco Auto", expanded=True):
    bev_base_2030 = st.number_input("Target BEV Palermo 2030", value=5000)
    stress_bev = st.slider("Stress Test Mercato (%)", 50, 150, 100) / 100
    public_share = st.slider("Quota Ricarica Pubblica (%)", 10, 80, 30) / 100

with st.sidebar.expander("🎯 Strategia di Cattura"):
    target_cattura_2030 = st.slider("Quota Cattura Target 2030 (%)", 1.0, 15.0, 5.0) / 100
    stress_cattura = st.slider("Efficacia Competitiva (%)", 50, 150, 100) / 100

st.sidebar.header("⚙️ Operatività e Finanza")
with st.sidebar.expander("🔧 Scelte Tecniche"):
    tecnologia = st.selectbox("Tecnologia Asset", ["DC 30 kW", "DC 60 kW"])
    allocazione = st.radio("Strategia Location", ["Monosito (Tutto in A)", "Multisito (Espansione in B)"])
    ore_max_giorno = st.slider("Disponibilità (ore/giorno)", 4, 12, 10)
    kwh_per_sessione = st.number_input("kWh medi per sessione", value=35)

with st.sidebar.expander("💰 Financials"):
    prezzo_kwh = st.number_input("Prezzo vendita (€/kWh)", value=0.69)
    costo_kwh = st.number_input("Costo energia (€/kWh)", value=0.30)
    capex_unit = 25000 if tecnologia == "DC 30 kW" else 45000
    opex_unit = 2000 if tecnologia == "DC 30 kW" else 3500
    wacc = st.slider("WACC (Costo Capitale %)", 4, 12, 8) / 100

# --- LOGICA DI CALCOLO UNIFICATA ---
years = np.array([2026, 2027, 2028, 2029, 2030])
potenza_kw = 30 if tecnologia == "DC 30 kW" else 60

# 1. Calcolo Funnel Mercato
bev_citta = np.linspace(bev_base_2030 * 0.5, bev_base_2030, 5) * stress_bev
quota_stazione = np.linspace(0.02, target_cattura_2030, 5) * stress_cattura
auto_clienti_anno = bev_citta * public_share * quota_stazione
energia_kwh = auto_clienti_anno * 3000 # Basato su percorrenza media annua

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

# 4. Cash Flow e Margini
ricavi = energia_kwh * prezzo_kwh
ebitda = (energia_kwh * (prezzo_kwh - costo_kwh)) - (n_totale * opex_unit)
capex_flow = np.zeros(len(years)); prev_n = 0
for i, n in enumerate(n_totale):
    capex_flow[i] = max(0, n - prev_n) * capex_unit
    prev_n = n
cf_netto = ebitda - capex_flow
cf_cum = np.cumsum(cf_netto)

# --- DASHBOARD KPI ---
st.subheader(f"📊 Business Overview: {tecnologia}")
k1, k2, k3, k4 = st.columns(4)
k1.metric("VAN (NPV)", f"€ {npf.npv(wacc, cf_netto):,.0f}")
k2.metric("TIR (IRR)", f"{(npf.irr(cf_netto)*100):.1f}%")
k3.metric("Auto/Anno (2030)", f"{auto_clienti_anno[-1]:,.0f}")
k4.metric("Saturazione Asset (%)", f"{(ore_richieste[-1]/(n_totale[-1]*ore_disp_asset)*100):.1f}%")

# --- TUTTI I GRAFICI RICHIESTI ---
st.divider()
c1, c2 = st.columns(2)

with c1:
    st.write("**1. Break-even: Soglia di Profitto Giornaliera**")
    auto_range = np.linspace(1, 15, 20)
    margine_sessione = kwh_per_sessione * (prezzo_kwh - costo_kwh)
    costo_fisso_day = (opex_unit + (capex_unit/5)) / 365
    fig1, ax1 = plt.subplots()
    ax1.plot(auto_range, (auto_range * margine_sessione) - costo_fisso_day, color='green', linewidth=3)
    ax1.axhline(0, color='red', linestyle='--')
    ax1.set_xlabel("Ricariche giornaliere per unità")
    st.pyplot(fig1)
    st.markdown(r"**Formula:** $N_{auto} = \frac{OPEX_{day} + Ammortamento_{day}}{kWh_{sess} \cdot (P_{vendita} - C_{energia})}$")

with c2:
    st.write("**2. Cash Flow Cumulato: Recupero Investimento**")
    fig2, ax2 = plt.subplots()
    ax2.plot(years, cf_cum, marker='o', color='navy', linewidth=3)
    ax2.fill_between(years, cf_cum, 0, where=(cf_cum >= 0), color='green', alpha=0.1)
    ax2.axhline(0, color='black', linewidth=1)
    st.pyplot(fig2)
    st.markdown(r"**Formula:** $CF_{cum, t} = \sum_{i=2026}^{t} (EBITDA_i - CAPEX_i)$")

c3, c4 = st.columns(2)
with c3:
    st.write("**3. Allocazione Asset per Location**")
    fig3, ax3 = plt.subplots()
    ax3.bar(years, stazione_A, label='Stazione A', color='#1f77b4')
    ax3.bar(years, stazione_B, bottom=stazione_A, label='Stazione B', color='#ff7f0e')
    ax3.set_ylabel("Numero Colonnine")
    ax3.legend()
    st.pyplot(fig3)
    st.markdown(r"**Formula:** $n = \lceil \frac{E_{tot} / Potenza}{Ore_{max} \cdot 365} \rceil$")

with c4:
    st.write("**4. Struttura dei Margini (Totale 5 Anni)**")
    fig4, ax4 = plt.subplots()
    labels = ['Ricavi', 'Costi Energia', 'CAPEX', 'EBITDA']
    vals = [ricavi.sum(), (energia_kwh * costo_kwh).sum(), capex_flow.sum(), ebitda.sum()]
    ax4.bar(labels, vals, color=['#2ecc71', '#e67e22', '#e74c3c', '#3498db'])
    st.pyplot(fig4)
    st.markdown(r"**Formula:** $EBITDA = Ricavi - (Costi_{Var} + Costi_{Fissi})$")

# --- TABELLA ANALITICA COMPLETA ---
st.divider()
st.subheader("📊 Report Analitico: Funnel di Conversione e Saturazione")
df_master = pd.DataFrame({
    "Anno": years,
    "BEV Palermo": bev_citta.astype(int),
    "Domanda Pubblica (%)": [(public_share * 100)] * 5,
    "Quota Cattura (%)": (quota_stazione * 100).round(2),
    "Clienti Catturati": auto_clienti_anno.astype(int),
    "Energia Tot (kWh)": energia_kwh.astype(int),
    "Ore Occupazione": ore_richieste.astype(int),
    "Asset A": stazione_A.astype(int),
    "Asset B": stazione_B.astype(int),
    "EBITDA (€)": ebitda.astype(int),
    "CF Cumulato (€)": cf_cum.astype(int)
}).set_index("Anno")
st.table(df_master)

with st.expander("📚 Intelligence Report - Nota per il Management"):
    st.markdown(f"""
    - **Capacità Fisica**: Una {tecnologia} richiede **{kwh_per_sessione/potenza_kw*60:.0f} minuti** per ogni ricarica.
    - **Saturazione**: Al 2030 gestirai un flusso di **{auto_clienti_anno[-1]:,.0f} auto all'anno**.
    - **Strategia**: L'allocazione {allocazione} garantisce la copertura della domanda senza superare le **{ore_max_giorno} ore/giorno** di utilizzo per singola piazzola.
    """)
