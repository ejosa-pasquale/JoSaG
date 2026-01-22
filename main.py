import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy_financial as npf

st.set_page_config(page_title="Investment Tool Pro", layout="wide")

# ---------------------------------------------------------
# CSS PER UI PROFESSIONALE
# ---------------------------------------------------------
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ Decision Support System: DC 30 kW Palermo")
st.info("Modello basato su report ufficiale del 22/01/2026. Include Stress Test, Analisi Fiscale e VAN.")

# ---------------------------------------------------------
# SIDEBAR - PARAMETRI AVANZATI
# ---------------------------------------------------------
st.sidebar.header("⚙️ Configurazione Avanzata")

with st.sidebar.expander("🛡️ Scenario Stress Test", expanded=True):
    stress_auto = st.slider("Moltiplicatore Parco BEV", 0.5, 1.5, 1.0)
    stress_competizione = st.slider("Efficacia Cattura (%)", 50, 150, 100) / 100

with st.sidebar.expander("💸 Parametri Finanziari & Fiscali"):
    wacc = st.slider("Costo del Capitale (WACC %)", 4, 12, 8) / 100
    tax_rate = st.slider("Aliquota Fiscale (IRES/IRAP %)", 20, 30, 24) / 100
    prezzo_kwh = st.sidebar.number_input("Prezzo vendita (€/kWh)", value=0.69)
    costo_kwh = st.sidebar.number_input("Costo energia (€/kWh)", value=0.30)
    capex_unit = st.sidebar.number_input("CAPEX per Unità (€)", value=25000)

# ---------------------------------------------------------
# LOGICA DI CALCOLO COMPLETA
# ---------------------------------------------------------
years = np.array([2026, 2027, 2028, 2029, 2030])
bev_est = np.array([2600, 3000, 3500, 4200, 5000]) * stress_auto
quota_stazione = np.array([0.02, 0.03, 0.04, 0.045, 0.05]) * stress_competizione

# Energia e Operatività
energia_kwh = bev_est * 3000 * 0.30 * quota_stazione
n_colonnine = np.ceil(energia_kwh / (30 * 8760 * 0.97 * 0.30)).astype(int)

# Conto Economico
ricavi = energia_kwh * prezzo_kwh
costi_energia = energia_kwh * costo_kwh
opex = n_colonnine * 2000
ebitda = ricavi - costi_energia - opex

# Fisco e Ammortamenti (Ammortamento in 5 anni)
ammortamento = np.zeros(len(years))
capex_flow = np.zeros(len(years))
prev_n = 0
accum_capex = 0

for i, n in enumerate(n_colonnine):
    nuove = max(0, n - prev_n)
    capex_flow[i] = nuove * capex_unit
    accum_capex += capex_flow[i]
    ammortamento[i] = accum_capex / 5 # Semplificato lineare
    prev_n = n

ebt = ebitda - ammortamento
tasse = np.maximum(0, ebt * tax_rate)
utile_netto = ebt - tasse
cf_operativo = utile_netto + ammortamento - capex_flow # Flusso di cassa reale
cf_cum = np.cumsum(cf_operativo)

# Indicatori Decisionali
van = npf.npv(wacc, cf_operativo)
tir = npf.irr(cf_operativo)
break_even_prezzo = (costi_energia + opex) / energia_kwh # Prezzo minimo per EBITDA=0

# ---------------------------------------------------------
# UI: DASHBOARD DECISIONALE
# ---------------------------------------------------------
st.subheader("🏁 Indicatori di Redditività Netta")
c1, c2, c3, c4 = st.columns(4)

tir_val = f"{tir*100:.2f}%" if not np.isnan(tir) else "N/D"
c1.metric("VAN (Valore Attuale)", f"€ {van:,.0f}", delta="INVESTIRE" if van > 0 else "EVITARE")
c2.metric("TIR (Rendimento)", tir_val)
c3.metric("Break-even Prezzo", f"€ {break_even_prezzo.mean():.2f}/kWh", help="Sotto questo prezzo medio vai in perdita operativa.")
c4.metric("Utile Netto Totale (5y)", f"€ {np.sum(utile_netto):,.0f}")

# Analisi Grafica
st.divider()
g1, g2 = st.columns(2)

with g1:
    st.write("**Cash Flow Netto vs Tasse**")
    fig1, ax1 = plt.subplots()
    ax1.bar(years, ebitda, label="EBITDA (Lordo)", color="#bdc3c7")
    ax1.bar(years, utile_netto, label="Utile Netto (Post-Tasse)", color="#2ecc71")
    ax1.set_title("Evoluzione Profittabilità")
    ax1.legend()
    st.pyplot(fig1)

with g2:
    st.write("**Saturazione Infrastruttura**")
    saturazione = (energia_kwh / (n_colonnine * 30 * 8760 * 0.97)) * 100
    fig2, ax2 = plt.subplots()
    ax2.plot(years, saturazione, marker='s', color="#e67e22", linewidth=2)
    ax2.axhline(30, color="red", linestyle="--", label="Target 30%")
    ax2.set_ylim(0, 100)
    ax2.set_title("% Utilizzo Reale Colonnine")
    ax2.legend()
    st.pyplot(fig2)

st.subheader("📋 Prospetto Finanziario Completo")
st.dataframe(pd.DataFrame({
    "Energia (kWh)": energia_kwh.astype(int),
    "Colonnine": n_colonnine,
    "EBITDA (€)": ebitda.astype(int),
    "Ammortamento (€)": ammortamento.astype(int),
    "Tasse (€)": tasse.astype(int),
    "Utile Netto (€)": utile_netto.astype(int),
    "CF Cumulato (€)": cf_cum.astype(int)
}, index=years).T, use_container_width=True)
