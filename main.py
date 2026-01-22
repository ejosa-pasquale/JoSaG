import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configurazione Pagina
st.set_page_config(page_title="Simulatore Business Case DC30", layout="wide")

st.title("⚡ Business Case Interattivo: Ricarica DC 30 kW")
st.markdown("Analisi della redditività per stazione di servizio a Palermo (2026-2030)")

# ---------------------------------------------------------
# 1) SIDEBAR: PARAMETRI MODIFICABILI (SENSITIVITÀ)
# ---------------------------------------------------------
st.sidebar.header("🕹️ Parametri di Simulazione")

# Parametri Mercato
st.sidebar.subheader("Mercato e Domanda")
utilizzazione_target = st.sidebar.slider("Utilizzo medio annuo (%)", 10, 50, 30) / 100 # [cite: 186]
quota_pubblico = st.sidebar.slider("Quota ricarica pubblica (%)", 10, 50, 30) / 100 # [cite: 181]

# Parametri Economici
st.sidebar.subheader("Economia (Unitario)")
prezzo_vendita = st.sidebar.number_input("Prezzo vendita (€/kWh)", value=0.69, step=0.01) # [cite: 187]
costo_energia = st.sidebar.number_input("Costo energia (€/kWh)", value=0.30, step=0.01) # [cite: 188]
capex_per_colonnina = st.sidebar.slider("CAPEX per unità (€)", 15000, 35000, 25000, step=1000) # [cite: 161, 163]
opex_annuo = st.sidebar.number_input("OPEX annuo per unità (€)", value=2000) # [cite: 162, 189]

# ---------------------------------------------------------
# 2) LOGICA DI CALCOLO (Basata su Report)
# ---------------------------------------------------------
years = np.array([2026, 2027, 2028, 2029, 2030])
bev_palermo = np.array([2600, 3000, 3500, 4200, 5000]) # [cite: 179]
quota_stazione = np.array([0.02, 0.03, 0.04, 0.045, 0.05]) # 

# Capacità tecnica [cite: 190]
capacita_unitaria = 30 * 8760 * 0.97 * utilizzazione_target

# Calcolo Domanda e Dimensionamento [cite: 204, 205, 206]
energia_intercettata = bev_palermo * 3000 * quota_pubblico * quota_stazione
colonnine_necessarie = np.ceil(energia_intercettata / capacita_unitaria).astype(int)

# Calcolo Economico [cite: 233, 234]
ricavi = energia_intercettata * prezzo_vendita
margine_energia = energia_intercettata * (prezzo_vendita - costo_energia)
ebitda = margine_energia - (colonnine_necessarie * opex_annuo)

# Cash Flow [cite: 260, 262]
capex_anno = np.zeros(len(years))
prev_n = 0
for i, n in enumerate(colonnine_necessarie):
    nuove = max(0, n - prev_n)
    capex_anno[i] = nuove * capex_per_colonnina
    prev_n = n

cf_netto = ebitda - capex_anno
cf_cumulato = np.cumsum(cf_netto)

# ---------------------------------------------------------
# 3) VISUALIZZAZIONE RISULTATI
# ---------------------------------------------------------

# KPI Principali
k1, k2, k3, k4 = st.columns(4)
k1.metric("EBITDA 2030", f"€ {ebitda[-1]:,.0f}")
k2.metric("CAPEX Totale", f"€ {np.sum(capex_anno):,.0f}")
k3.metric("CF Cumulato 2030", f"€ {cf_cumulato[-1]:,.0f}")
payback_year = years[np.where(cf_cumulato >= 0)[0][0]] if any(cf_cumulato >= 0) else "Oltre 2030"
k4.metric("Anno di Payback", payback_year)

# Tabelle
st.subheader("📋 Dettaglio Annuale")
df_output = pd.DataFrame({
    "Anno": years,
    "Energia (kWh)": energia_intercettata.astype(int),
    "N. Colonnine": colonnine_necessarie,
    "Ricavi (€)": ricavi.astype(int),
    "EBITDA (€)": ebitda.astype(int),
    "CF Cumulato (€)": cf_cumulato.astype(int)
})
st.dataframe(df_output, use_container_width=True)

# Grafici
st.subheader("📈 Analisi dei Trend")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Grafico Energia e EBITDA
ax1.plot(years, energia_intercettata/1000, marker='o', label="MWh/anno")
ax1.bar(years, ebitda/1000, alpha=0.3, color='green', label="EBITDA (k€)")
ax1.set_title("Energia Intercettata vs EBITDA")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Grafico Cash Flow
ax2.plot(years, cf_cumulato/1000, marker='s', color='orange', label="CF Cumulato (k€)")
ax2.axhline(0, color='black', linewidth=1)
ax2.set_title("Evoluzione Ritorno Economico")
ax2.legend()
ax2.grid(True, alpha=0.3)

st.pyplot(fig)
