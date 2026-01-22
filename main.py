import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configurazione Pagina
st.set_page_config(page_title="Analisi DC30 - Stazione di Servizio", layout="wide")

st.title("⚡ Business Case Interattivo: Ricarica DC 30 kW")
st.markdown("Questa applicazione permette di simulare la redditività dell'installazione di punti di ricarica in una stazione di servizio.")

# ---------------------------------------------------------
# 1) SIDEBAR: PARAMETRI DI INPUT (SENSITIVITÀ)
# ---------------------------------------------------------
st.sidebar.header("🕹️ Parametri di Configurazione")

with st.sidebar.expander("📈 Mercato e Domanda", expanded=True):
    utilizzazione_target = st.slider("Utilizzo medio annuo (%)", 5, 60, 30, help="Percentuale di tempo in cui la colonnina eroga energia alla massima potenza.") / 100
    quota_pubblico = st.slider("Quota ricarica pubblica (%)", 10, 80, 30, help="Percentuale di possessori di BEV che non caricano a casa e usano la rete pubblica.") / 100

with st.sidebar.expander("💰 Economia e CAPEX"):
    prezzo_vendita = st.number_input("Prezzo vendita (€/kWh)", value=0.69, step=0.01)
    costo_energia = st.number_input("Costo energia (€/kWh)", value=0.30, step=0.01)
    capex_per_colonnina = st.slider("CAPEX per unità (€)", 15000, 35000, 25000)
    opex_annuo = st.number_input("OPEX annuo per unità (€)", value=2000, help="Costi di manutenzione, software e occupazione suolo.")

# ---------------------------------------------------------
# 2) LOGICA DI CALCOLO (Basata su Report [cite: 39, 53, 109])
# ---------------------------------------------------------
years = np.array([2026, 2027, 2028, 2029, 2030])
bev_palermo = np.array([2600, 3000, 3500, 4200, 5000]) [cite: 28]
quota_stazione = np.array([0.02, 0.03, 0.04, 0.045, 0.05]) [cite: 31]

# Calcoli Tecnici
capacita_unitaria = 30 * 8760 * 0.97 * utilizzazione_target [cite: 39]
energia_intercettata = bev_palermo * 3000 * quota_pubblico * quota_stazione [cite: 53]
colonnine_necessarie = np.ceil(energia_intercettata / capacita_unitaria).astype(int)

# Calcoli Economici
ricavi = energia_intercettata * prezzo_vendita
margine_unitario = prezzo_vendita - costo_energia
ebitda = (energia_intercettata * margine_unitario) - (colonnine_necessarie * opex_annuo)

# Cash Flow e ROI
capex_flusso = np.zeros(len(years))
prev_n = 0
for i, n in enumerate(colonnine_necessarie):
    nuove = max(0, n - prev_n)
    capex_flusso[i] = nuove * capex_per_colonnina
    prev_n = n

cf_netto = ebitda - capex_flusso
cf_cumulato = np.cumsum(cf_netto)
roi_cumulato = cf_cumulato[-1] / np.sum(capex_flusso) if np.sum(capex_flusso) > 0 else 0

# ---------------------------------------------------------
# 3) VISUALIZZAZIONE OUTPUT
# ---------------------------------------------------------

# Sezione KPI
st.subheader("📌 Indicatori Chiave di Performance (KPI)")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Payback Period", years[np.where(cf_cumulato >= 0)[0][0]] if any(cf_cumulato >= 0) else "Oltre 2030")
k2.metric("ROI Cumulato (2030)", f"{roi_cumulato:.2f}x")
k3.metric("Margine su Energia", f"{((margine_unitario/prezzo_vendita)*100):.1f}%")
k4.metric("EBITDA Totale (5y)", f"€ {np.sum(ebitda):,.0f}")

# Tabelle Dettagliate
st.subheader("📊 Report Dettagliato")
col_tab1, col_tab2 = st.columns([2, 1])

with col_tab1:
    st.write("**Tabella Economica Completa**")
    df_out = pd.DataFrame({
        "Anno": years,
        "Domanda (kWh)": energia_intercettata.astype(int),
        "N. Colonnine": colonnine_necessarie,
        "EBITDA (€)": ebitda.astype(int),
        "CAPEX (€)": capex_flusso.astype(int),
        "Cash Flow Netto (€)": cf_netto.astype(int),
        "CF Cumulato (€)": cf_cumulato.astype(int)
    })
    st.dataframe(df_out, use_container_width=True)

with col_tab2:
    st.write("**Dizionario Parametri**")
    st.info("""
    - **Utilizzo Target**: La saturazione della stazione. Il 30% è lo scenario base del report[cite: 35].
    - **Quota Pubblico**: Quanta energia viene caricata fuori casa (scenario base 30%).
    - **Quota Stazione**: La capacità della TUA stazione di attrarre clienti a Palermo[cite: 31].
    - **CAPEX**: Costo acquisto e installazione (Scenario Base: 25.000€)[cite: 12].
    """)

# Grafici
st.subheader("📈 Analisi Grafica")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

# Grafico 1: Scalabilità
ax1.bar(years, ebitda, color='skyblue', label='EBITDA (€)')
ax1_tw = ax1.twinx()
ax1_tw.step(years, colonnine_necessarie, where='post', color='red', label='N. Colonnine', linewidth=2)
ax1.set_title("Crescita EBITDA vs Numero Colonnine")
ax1.legend(loc='upper left')

# Grafico 2: Cash Flow Cumulato
ax2.plot(years, cf_cumulato, marker='o', color='green', linewidth=2)
ax2.axhline(0, color='black', linestyle='--')
ax2.set_title("Rientro dell'Investimento (Cash Flow Cumulato)")
ax2.grid(True, alpha=0.3)

st.pyplot(fig)
