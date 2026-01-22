import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configurazione della pagina
st.set_page_config(page_title="Business Case DC30 Palermo", layout="wide")

st.title("⚡ Business Case: Ricarica DC 30 kW in Stazione di Servizio")
st.markdown("""
Questa applicazione simula la redditività di una stazione di ricarica nel centro di Palermo. 
Il modello è scalabile: il numero di colonnine aumenta automaticamente per soddisfare la domanda crescente.
""")

# ---------------------------------------------------------
# 1) SIDEBAR: CONFIGURAZIONE PARAMETRI (INPUT)
# ---------------------------------------------------------
st.sidebar.header("🕹️ Parametri di Scelta")

with st.sidebar.expander("📈 Mercato e Domanda", expanded=True):
    utilizzazione_target = st.slider(
        "Utilizzo Target (%)", 5, 60, 30, 
        help="La saturazione desiderata della colonnina. Al 30%, la colonnina lavora circa 7 ore al giorno. Più è alto, più l'impianto è efficiente ma rischia code."
    ) / 100
    
    quota_pubblico = st.slider(
        "Quota Ricarica Pubblica (%)", 10, 80, 30, 
        help="Percentuale di proprietari di auto elettriche che non ricaricano a casa e devono usare stazioni pubbliche."
    ) / 100

with st.sidebar.expander("💰 Economia e Investimento", expanded=True):
    prezzo_vendita = st.number_input(
        "Prezzo vendita (€/kWh)", value=0.69, step=0.01,
        help="Prezzo finale pagato dall'utente alla colonnina."
    )
    costo_energia = st.number_input(
        "Costo energia (€/kWh)", value=0.30, step=0.01,
        help="Costo di acquisto della materia prima energia (prezzo all'ingrosso + oneri)."
    )
    capex_per_colonnina = st.slider(
        "CAPEX per unità (€)", 15000, 35000, 25000,
        help="Costo di acquisto, installazione e allaccio di una singola colonnina DC 30 kW."
    )
    opex_annuo = st.number_input(
        "OPEX annuo per unità (€)", value=2000,
        help="Spese correnti: manutenzione, canoni software, occupazione suolo e assicurazione."
    )

# ---------------------------------------------------------
# 2) LOGICA DI CALCOLO (Dati da Report 22/01/2026)
# ---------------------------------------------------------
years = np.array([2026, 2027, 2028, 2029, 2030])

# Dati stimati per l'area di Palermo 
bev_palermo = np.array([2600, 3000, 3500, 4200, 5000]) 
quota_stazione = np.array([0.02, 0.03, 0.04, 0.045, 0.05]) 

# Capacità tecnica annua di una colonnina (kWh) 
# Calcolo: Potenza (30kW) * Ore anno (8760) * Uptime (97%) * Utilizzo Target
capacita_unitaria = 30 * 8760 * 0.97 * utilizzazione_target

# Calcolo Energia Intercettata (kWh) [cite: 53]
# Formula: Parco BEV * Consumo annuo (3000 kWh) * Quota Pubblica * Quota Stazione
energia_intercettata = bev_palermo * 3000 * quota_pubblico * quota_stazione

# Numero colonnine necessarie (scalabilità) 
colonnine_necessarie = np.ceil(energia_intercettata / capacita_unitaria).astype(int)

# Calcoli Economici [cite: 82, 83]
ricavi = energia_intercettata * prezzo_vendita
margine_energia = energia_intercettata * (prezzo_vendita - costo_energia)
ebitda = margine_energia - (colonnine_necessarie * opex_annuo)

# Cash Flow e Investimento [cite: 109]
capex_annuale = np.zeros(len(years))
prev_n = 0
for i, n in enumerate(colonnine_necessarie):
    nuove_installazioni = max(0, n - prev_n)
    capex_annuale[i] = nuove_installazioni * capex_per_colonnina
    prev_n = n

cf_netto = ebitda - capex_annuale
cf_cumulato = np.cumsum(cf_netto)

# ---------------------------------------------------------
# 3) OUTPUT VISIVI
# ---------------------------------------------------------

# KPI in alto
c1, c2, c3, c4 = st.columns(4)
c1.metric("Payback (Rientro)", years[np.where(cf_cumulato >= 0)[0][0]] if any(cf_cumulato >= 0) else "Oltre 2030")
c2.metric("ROI Cumulato 2030", f"{(cf_cumulato[-1] / np.sum(capex_annuale)):.2f}x")
c3.metric("EBITDA 2030", f"€ {ebitda[-1]:,.0f}")
c4.metric("Colonnine Finali", f"{colonnine_necessarie[-1]}")

# Tabelle
st.subheader("📊 Proiezioni Finanziarie 2026-2030")
df_display = pd.DataFrame({
    "Anno": years,
    "BEV Palermo": bev_palermo,
    "Energia Venduta (kWh)": energia_intercettata.astype(int),
    "Colonnine Attive": colonnine_necessarie,
    "Ricavi (€)": ricavi.astype(int),
    "EBITDA (€)": ebitda.astype(int),
    "CAPEX Anno (€)": capex_annuale.astype(int),
    "Cash Flow Cumulato (€)": cf_cumulato.astype(int)
})
st.dataframe(df_display.set_index("Anno"), use_container_width=True)

# Grafici
st.subheader("📈 Analisi di Crescita e Payback")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

# Grafico 1: Scalabilità Domanda vs Colonnine
ax1.bar(years, energia_intercettata/1000, color='skyblue', alpha=0.6, label="MWh Venduti")
ax1.set_ylabel("MWh / anno")
ax1_tw = ax1.twinx()
ax1_tw.step(years, colonnine_necessarie, where='post', color='red', linewidth=2, label="N. Colonnine")
ax1_tw.set_ylabel("Numero Colonnine")
ax1.set_title("Evoluzione Domanda e Infrastruttura")
ax1.legend(loc='upper left')

# Grafico 2: Cash Flow Cumulato
ax2.plot(years, cf_cumulato/1000, marker='o', color='green', linewidth=2, label="CF Cumulato (k€)")
ax2.axhline(0, color='black', linestyle='--', linewidth=1)
ax2.set_title("Punto di Pareggio (Break-even)")
ax2.set_ylabel("k€ Cumulati")
ax2.grid(True, alpha=0.3)
ax2.legend()

st.pyplot(fig)

# Spiegazione Parametri
with st.expander("📚 Glossario e Spiegazione Parametri"):
    st.write("""
    - **Utilizzo Target**: È il parametro critico. Se la stazione è in una zona di alto passaggio, l'utilizzo sarà alto (es. 35%). Se l'utilizzo è basso, l'investimento impiega più anni a rientrare.
    - **Quota Ricarica Pubblica**: Indica quanto il mercato locale dipende dalle colonnine. Più è alta (es. 50%), più energia venderai.
    - **EBITDA**: Rappresenta il guadagno operativo prima delle tasse e degli ammortamenti. Si calcola come: `(Margine su kWh * kWh venduti) - Spese operative (OPEX)`.
    - **Payback**: L'anno in cui la linea verde del grafico tocca lo zero, indicando che hai recuperato tutto il capitale investito (CAPEX).
    """)
