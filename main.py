import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy_financial as npf
import io
import requests

# ============================================================
# CONFIGURAZIONE E DATI STORICI SICILIA
# ============================================================
st.set_page_config(page_title="Executive Charging Suite — Sicilia", layout="wide")

BEV_RAW = """Anno\tProvincia\tElettrico
2015\tAGRIGENTO\t23
2021\tAGRIGENTO\t166
2022\tAGRIGENTO\t255
2023\tAGRIGENTO\t370
2024\tAGRIGENTO\t521
2015\tCALTANISSETTA\t11
2021\tCALTANISSETTA\t91
2022\tCALTANISSETTA\t147
2023\tCALTANISSETTA\t207
2024\tCALTANISSETTA\t313
2015\tCATANIA\t87
2021\tCATANIA\t993
2022\tCATANIA\t1578
2023\tCATANIA\t2293
2024\tCATANIA\t3254
2015\tENNA\t2
2021\tENNA\t58
2022\tENNA\t92
2023\tENNA\t129
2024\tENNA\t196
2015\tMESSINA\t179
2021\tMESSINA\t602
2022\tMESSINA\t814
2023\tMESSINA\t1075
2024\tMESSINA\t1412
2015\tPALERMO\t143
2021\tPALERMO\t753
2022\tPALERMO\t1066
2023\tPALERMO\t1530
2024\tPALERMO\t2144
2015\tRAGUSA\t16
2021\tRAGUSA\t337
2022\tRAGUSA\t586
2023\tRAGUSA\t814
2024\tRAGUSA\t1071
2015\tSIRACUSA\t54
2021\tSIRACUSA\t379
2022\tSIRACUSA\t560
2023\tSIRACUSA\t808
2024\tSIRACUSA\t1138
2015\tTRAPANI\t37
2022\tTRAPANI\t395
2023\tTRAPANI\t560
2024\tTRAPANI\t795
"""

# ============================================================
# FUNZIONI UTILITY
# ============================================================
def eur(x):
    if x is None or not np.isfinite(x): return "n/a"
    return f"€ {x:,.0f}".replace(",", ".")

def forecast_bev_2030(df_hist, province):
    s = df_hist[df_hist["Provincia"] == province].sort_values("Anno")
    v0, v1 = float(s[s["Anno"] == 2021]["Elettrico"].iloc[0]), float(s[s["Anno"] == 2024]["Elettrico"].iloc[0])
    cagr = (v1 / max(v0, 1)) ** (1/3) - 1
    bev_2024 = int(s[s["Anno"] == 2024]["Elettrico"].iloc[0])
    bev_2030 = int(round(bev_2024 * ((1 + cagr) ** 6)))
    return bev_2024, bev_2030, cagr

# ============================================================
# SIDEBAR - INPUT PARAMETERS
# ============================================================
df_bev = pd.read_csv(io.StringIO(BEV_RAW), sep="\t")
st.sidebar.header("📍 Localizzazione e Mercato")
provincia = st.sidebar.selectbox("Provincia", sorted(df_bev["Provincia"].unique()), index=5) # Default Palermo
bev_2024, bev_2030_est, cagr_val = forecast_bev_2030(df_bev, provincia)

with st.sidebar.expander("🎯 Funnel di Cattura", expanded=True):
    target_bev_2030 = st.number_input(f"Target BEV {provincia} (2030)", value=bev_2030_est)
    quota_pubblica = st.slider("Domanda su colonnine pubbliche (%)", 10, 80, 30) / 100
    quota_cattura = st.slider("Quota di mercato della stazione (%)", 0.5, 20.0, 5.0) / 100

with st.sidebar.expander("💰 Finanza e Asset", expanded=True):
    tecnologia = st.selectbox("Hardware", ["DC 30 kW", "DC 60 kW"])
    capex_unit = 25000 if tecnologia == "DC 30 kW" else 45000
    opex_unit = 2000 if tecnologia == "DC 30 kW" else 3500
    prezzo_kwh = st.number_input("Prezzo vendita (€/kWh)", value=0.69)
    costo_kwh = st.number_input("Costo energia (€/kWh)", value=0.30)
    ammortamento = st.slider("Anni ammortamento", 3, 10, 5)

# ============================================================
# CALCOLI CORE
# ============================================================
years = np.array([2026, 2027, 2028, 2029, 2030])
bev_growth = np.linspace(bev_2024, target_bev_2030, len(years))
auto_target = bev_growth * quota_pubblica * quota_cattura
energia_tot = auto_target * 3000 # 3000 kWh/anno media proxy
potenza_kw = 30 if tecnologia == "DC 30 kW" else 60

# Dimensionamento
cap_max_kwh_unit = potenza_kw * 8760 * 0.97 * 0.30 # kW * ore * uptime * utilizzo
n_unita = np.ceil(energia_tot / cap_max_kwh_unit).astype(int)

# Economia
ricavi = energia_tot * prezzo_kwh
ebitda = (energia_tot * (prezzo_kwh - costo_kwh)) - (n_unita * opex_unit)
capex_flow = np.zeros(len(years))
capex_flow[0] = n_unita[0] * capex_unit # Semplificato: investimento iniziale
cf_cum = np.cumsum(ebitda) - capex_flow.sum()

# ============================================================
# DASHBOARD DECISIONALE
# ============================================================
st.title("🛡️ Executive Decision Tool: Charging Station")
st.subheader(f"Analisi di fattibilità: {provincia} | {tecnologia}")

# KPI Strategici
m1, m2, m3, m4 = st.columns(4)
m1.metric("Auto Target (2030)", f"{auto_target[-1]:,.0f}")
m2.metric("Break-even (Ricariche/gg)", f"{(opex_unit + (capex_unit/ammortamento))/(365*35*(prezzo_kwh-costo_kwh)):.1f}")
m3.metric("Fatturato 2030", eur(ricavi[-1]))
m4.metric("ROI Periodo", f"{(cf_cum[-1]/capex_flow.sum()*100):.0f}%")

# --- SEZIONE GRAFICI DECISIONALI ---
st.divider()
st.markdown("### 📊 Analisi di Sostenibilità e Rischio")
c1, c2 = st.columns(2)

with c1:
    st.write("**A) Punto di Pareggio Operativo (Go/No-Go)**")
    # Calcolo margine e costi fissi giornalieri
    margine_sessione = 35 * (prezzo_kwh - costo_kwh) # 35 kWh media sessione
    costo_fisso_day = (opex_unit + (capex_unit / ammortamento)) / 365
    
    x_ricariche = np.linspace(0.5, 12, 50)
    y_utile = (x_ricariche * margine_sessione) - costo_fisso_day
    
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(x_ricariche, y_utile, color='navy', lw=3, label="Utile Giornaliero")
    ax1.axhline(0, color='red', ls='--', alpha=0.5)
    ax1.fill_between(x_ricariche, y_utile, 0, where=(y_utile > 0), color='green', alpha=0.1)
    ax1.fill_between(x_ricariche, y_utile, 0, where=(y_utile < 0), color='red', alpha=0.1)
    
    be_val = costo_fisso_day / margine_sessione
    ax1.scatter([be_val], [0], color='red', s=100, zorder=5, label=f"BE: {be_val:.1f} ricariche/gg")
    
    ax1.set_xlabel("Ricariche Giornaliere Medie per Unità")
    ax1.set_ylabel("Margine Netto Giornaliero (€)")
    ax1.legend()
    st.pyplot(fig1)
    st.caption("Questo grafico mostra quante ricariche servono ogni giorno per coprire i costi dell'asset.")

with c2:
    st.write("**B) Stress Test: Capacità vs Domanda Reale**")
    cap_max_auto = cap_max_kwh_unit / 3000
    
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.bar([str(y) for y in years], auto_target, color='skyblue', label="Auto nel tuo Target")
    ax2.axhline(cap_max_auto, color='darkorange', lw=2, label="Capacità Max 1 Unità")
    
    ax2.set_ylabel("Numero di Clienti (Auto/Anno)")
    ax2.set_title("La tua quota di mercato è servibile?")
    ax2.legend()
    st.pyplot(fig2)
    
    # Messaggio dinamico
    if auto_target[-1] > cap_max_auto:
        st.warning(f"⚠️ Al 2030, una singola {tecnologia} non basterà per servire {auto_target[-1]:.0f} auto.")
    else:
        st.success(f"✅ Una singola unità gestisce comodamente il target di {provincia}.")

# ============================================================
# REPORT E INTELLIGENCE
# ============================================================
st.divider()
st.subheader("📋 Piano Finanziario")
df_rep = pd.DataFrame({
    "Anno": years,
    "BEV Totali": bev_growth.astype(int),
    "Clienti Target": auto_target.astype(int),
    "Fatturato (€)": ricavi.astype(int),
    "EBITDA (€)": ebitda.astype(int),
    "Cassa Cumulata (€)": cf_cum.astype(int)
}).set_index("Anno")
st.table(df_rep)

st.info("""
**Manuale per la decisione:**
1. **Verifica il Break-even:** Se il punto rosso è > 8 ricariche/gg, il rischio operativo è alto.
2. **Verifica la Capacità:** Se le barre azzurre superano la linea arancione, devi pianificare l'acquisto di una seconda colonnina già nel 2029/2030.
3. **Contesto Locale:** Se a Palermo ci sono 2000 auto, catturarne il 5% (100 auto) è realistico. Se ne servono 500 per pareggiare, rivedi il canone o il prezzo.
""")
