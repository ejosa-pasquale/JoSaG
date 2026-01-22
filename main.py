import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy_financial as npf

st.set_page_config(page_title="Investment Decision Tool DC30", layout="wide")

st.title("🛡️ Sistema di Supporto alle Decisioni - DC30 Palermo")
st.markdown("Questo tool valuta la robustezza del Business Case al variare delle condizioni esterne e delle tue scelte commerciali.")

# ---------------------------------------------------------
# SIDEBAR - INPUT DIVISI PER TIPOLOGIA
# ---------------------------------------------------------
st.sidebar.header("🎯 Variabili Decisionali (Le tue scelte)")
with st.sidebar:
    prezzo_kwh = st.number_input("Prezzo di Vendita (€/kWh)", value=0.69, help="Il prezzo che imponi ai tuoi clienti.")
    opex_target = st.number_input("OPEX annuo/colonnina (€)", value=2000, help="Costi di manutenzione e gestione.")
    capex_unit = st.slider("Investimento per Colonnina (€)", 15000, 35000, 25000)
    wacc = st.slider("Costo del Capitale (WACC %)", 4, 12, 8) / 100

st.sidebar.header("📊 Ipotesi di Mercato (Scenario)")
with st.sidebar:
    crescita_bev = st.slider("Moltiplicatore Adozione Auto Elettriche", 0.5, 2.0, 1.0, help="1.0 = Scenario Report. 0.5 = Rallentamento mercato. 2.0 = Boom elettrico.")
    efficacia_stazione = st.slider("Capacità di Cattura (%)", 50, 150, 100, help="Capacità della stazione di attrarre clienti rispetto alla concorrenza.") / 100
    carica_media = st.number_input("Energia media caricata/anno (kWh)", value=3000, help="Quanti kWh consuma mediamente un'auto del parco circolante in un anno.")

# ---------------------------------------------------------
# LOGICA DI CALCOLO
# ---------------------------------------------------------
years = np.array([2026, 2027, 2028, 2029, 2030])
bev_base = np.array([2600, 3000, 3500, 4200, 5000]) * crescita_bev
quota_mercato = np.array([0.02, 0.03, 0.04, 0.045, 0.05]) * efficacia_stazione

# Domanda e Dimensionamento
energia_tot = bev_base * carica_media * 0.30 * quota_mercato
capacita_colonnina = 30 * 8760 * 0.97 * 0.30 
n_colonnine = np.ceil(energia_tot / capacita_colonnina).astype(int)

# Conto Economico e Fisco
ricavi = energia_tot * prezzo_kwh
ebitda = (energia_tot * (prezzo_kwh - 0.30)) - (n_colonnine * opex_target)
tasse = np.maximum(0, ebitda * 0.24) # IRES/IRAP stimata
utile_netto = ebitda - tasse

# Flussi di Cassa
capex_flow = np.zeros(len(years))
prev_n = 0
for i, n in enumerate(n_colonnine):
    capex_flow[i] = max(0, n - prev_n) * capex_unit
    prev_n = n

cash_flow = utile_netto - capex_flow
cf_cum = np.cumsum(cash_flow)

# Indicatori Finanziari
van = npf.npv(wacc, cash_flow)
tir = npf.irr(cash_flow)

# ---------------------------------------------------------
# OUTPUT CHIARO E SPIEGATO
# ---------------------------------------------------------

# 1. Riquadro Verdetto
st.subheader("✅ Analisi Sintetica per l'Investitore")
v1, v2, v3 = st.columns([1, 1, 2])

with v1:
    st.metric("Valore Creato (VAN)", f"€ {van:,.0f}")
    if van > 0: st.success("PROGETTO BANCABILE")
    else: st.error("PROGETTO NON REDDITIZIO")

with v2:
    tir_val = f"{tir*100:.1f}%" if not np.isnan(tir) else "N/D"
    st.metric("Rendimento Interno (TIR)", tir_val)
    st.caption(f"WACC obiettivo: {wacc*100}%")

with v3:
    st.markdown(f"""
    **Perché questo risultato?**
    - Hai un margine di **{(prezzo_kwh-0.30)/prezzo_kwh:.1%}%** su ogni kWh venduto.
    - La stazione raggiunge il pareggio (Payback) nel **{years[np.where(cf_cum >= 0)[0][0]] if any(cf_cum >= 0) else "Oltre 2030"}**.
    - Con lo scenario impostato, venderai **{energia_tot.sum():,.0f} kWh** in 5 anni.
    """)

# 2. Analisi di Sensibilità
st.divider()
st.subheader("📈 Stress Test: Cosa succede se lo scenario cambia?")
col_g1, col_g2 = st.columns(2)

with col_g1:
    st.write("**Resilienza del Cash Flow**")
    fig1, ax1 = plt.subplots()
    ax1.plot(years, cf_cum, marker='o', linewidth=3, color='#1f77b4')
    ax1.axhline(0, color='red', linestyle='--')
    ax1.set_title("Punto di Recupero Capitale")
    ax1.set_ylabel("Euro (€)")
    st.pyplot(fig1)

with col_g2:
    st.write("**Evoluzione Infrastruttura**")
    fig2, ax2 = plt.subplots()
    ax2.bar(years, energia_tot, color='#7f7f7f', alpha=0.5, label="Energia Venduta (kWh)")
    ax2_tw = ax2.twinx()
    ax2_tw.step(years, n_colonnine, where='post', color='red', label="N. Colonnine")
    ax2.set_title("Domanda vs Numero Colonnine")
    st.pyplot(fig2)

# 3. Tabella Dettagliata per Analisi
with st.expander("🔍 Vedi Dettagli Numerici (Conto Economico e Cash Flow)"):
    df = pd.DataFrame({
        "Anno": years,
        "Auto Elettriche (Area)": bev_base.astype(int),
        "Energia Erogata (kWh)": energia_tot.astype(int),
        "Colonnine Attive": n_colonnine,
        "EBITDA (€)": ebitda.astype(int),
        "Cash Flow Netto (€)": cash_flow.astype(int),
        "Cash Flow Cumulato (€)": cf_cum.astype(int)
    }).set_index("Anno")
    st.table(df)
