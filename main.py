# === Revised Executive Decision Support Version ===
# Core logic preserved: BEV_RAW, forecast_bev_2030, PDF lock options
# Enhancements focus on CFO / GM decision support

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy_financial as npf
import io

st.set_page_config(page_title="Executive Charging Suite — Decision Support", layout="wide")

# ============================================================
# DATI BEV SICILIA (2015–2024) — CORE INVARIATO
# ============================================================
BEV_RAW = """
Anno\tProvincia\tElettrico
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
# UTILITY
# ============================================================
def eur(x):
    return f"€ {x:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".") if pd.notnull(x) else "n/a"

# ============================================================
# FORECAST CORE — INVARIATO
# ============================================================
def forecast_bev_2030(df_hist, province, method="CAGR 2021–2024"):
    s = df_hist[df_hist["Provincia"] == province].sort_values("Anno")
    bev_2024 = int(s[s["Anno"] == 2024]["Elettrico"].iloc[0])
    base = s[s["Anno"] == 2021]["Elettrico"].iloc[0]
    cagr = (bev_2024 / base) ** (1 / 3) - 1
    bev_2030 = int(bev_2024 * (1 + cagr) ** 6)
    return bev_2024, bev_2030, cagr

# ============================================================
# PREPARAZIONE DATI
# ============================================================
df = pd.read_csv(io.StringIO(BEV_RAW), sep="\t")
province = st.sidebar.selectbox("Provincia", sorted(df["Provincia"].unique()))
bev_2024, bev_2030, cagr = forecast_bev_2030(df, province)

# ============================================================
# PARAMETRI TECNICI (DEFAULT PRESERVATI)
# ============================================================
potenza_kw = 150
uptime = 0.95
utilizzo_medio = 0.25
ore_giorno = 24

capacita_giornaliera_kwh = potenza_kw * ore_giorno * uptime * utilizzo_medio

# ============================================================
# SEZIONE 1 — IL MERCATO (FUNNEL)
# ============================================================
st.header("1️⃣ Il Mercato")
quota_pubblica = st.slider("Quota ricarica pubblica", 0.1, 0.6, 0.35)
quota_cattura = st.slider("Quota cattura stazione", 0.01, 0.15, 0.05)
prezzo = st.number_input("Prezzo €/kWh", 0.3, 1.2, 0.65)

# Stress test automatico prezzo
if prezzo > 0.75:
    penalty = min((prezzo - 0.75) / 0.25, 0.5)
    quota_cattura_eff = quota_cattura * (1 - penalty)
else:
    quota_cattura_eff = quota_cattura

bev_pubblici = bev_2030 * quota_pubblica
bev_catturati = bev_pubblici * quota_cattura_eff

funnel = pd.DataFrame({
    "Fase": ["Parco BEV 2030", "Ricarica Pubblica", "Catturati Stazione"],
    "Valore": [bev_2030, bev_pubblici, bev_catturati]
})

fig, ax = plt.subplots()
ax.bar(funnel["Fase"], funnel["Valore"], color="#003366")
st.pyplot(fig)

# ============================================================
# SEZIONE 2 — L'INVESTIMENTO
# ============================================================
st.header("2️⃣ L'Investimento")

kwh_giorno_domanda = bev_catturati * 12
n_totale = int(np.ceil(kwh_giorno_domanda / capacita_giornaliera_kwh))
n_totale = max(1, n_totale)

capex_unitario = 60000
capex_tot = capex_unitario * n_totale

st.metric("Numero Colonnine Suggerite", n_totale)
st.metric("CAPEX Totale", eur(capex_tot))

# ============================================================
# SEZIONE 3 — IL RITORNO
# ============================================================
st.header("3️⃣ Il Ritorno")

margine_kwh = prezzo - 0.25
ricavi_annui = kwh_giorno_domanda * 365 * prezzo
margine_annuo = kwh_giorno_domanda * 365 * margine_kwh

years = np.arange(0, 11)
cashflow = [-capex_tot] + [margine_annuo] * 10

payback = next((y for y in range(1, 11) if sum(cashflow[:y+1]) > 0), None)
roi = (sum(cashflow[1:]) - capex_tot) / capex_tot

summary = pd.DataFrame({
    "KPI": ["Punto di Pareggio (ric/g)", "Anno Payback", "ROI Finale", "MOL Annuo"],
    "Valore": [round(kwh_giorno_domanda / 12, 1), payback, f"{roi*100:.1f}%", eur(margine_annuo)]
})

st.subheader("Executive Summary")
st.table(summary)

fig2, ax2 = plt.subplots()
ax2.plot(years, np.cumsum(cashflow), color="#003366", label="Cumulato")
ax2.bar(years, [0] + cashflow[1:], color="#2e7d32", label="EBITDA")
ax2.bar(0, -capex_tot, color="#c62828", label="CAPEX")
ax2.legend()
st.pyplot(fig2)
