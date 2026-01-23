import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy_financial as npf
import io

st.set_page_config(page_title="Executive Charging Suite — Sicilia - eVFSs", layout="wide")

# ============================================================
# DATI BEV SICILIA (2015–2024)
# ============================================================
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

def eur(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    s = f"{x:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"€ {s}"

def pct(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x*100:.1f}%"

def forecast_bev_2030(df_hist, province, method="CAGR 2021–2024"):
    s = df_hist[df_hist["Provincia"] == province].sort_values("Anno")
    bev_2024 = int(s[s["Anno"] == 2024]["Elettrico"].iloc[0])
    bev_2021 = int(s[s["Anno"] == 2021]["Elettrico"].iloc[0])
    cagr = (bev_2024 / bev_2021) ** (1 / 3) - 1
    bev_2030 = int(bev_2024 * ((1 + cagr) ** 6))
    return bev_2024, bev_2030, cagr

# ============================================================
# INPUT SIDEBAR (INVARIATI)
# ============================================================
df_bev = pd.read_csv(io.StringIO(BEV_RAW), sep="\t")

st.sidebar.header("Input Strategici")
provincia = st.sidebar.selectbox("Provincia", sorted(df_bev["Provincia"].unique()))
public_share = st.sidebar.slider("Quota ricarica pubblica", 0.1, 0.6, 0.3)
quota_stazione = st.sidebar.slider("Quota cattura stazione", 0.01, 0.20, 0.06)
cfo_mode = st.sidebar.checkbox("CFO Mode")

# ============================================================
# FORECAST BEV
# ============================================================
bev_2024, bev_2030, cagr = forecast_bev_2030(df_bev, provincia)
years = np.arange(2024, 2031)
bev_citta = np.linspace(bev_2024, bev_2030, len(years)).astype(int)

# ============================================================
# LOGICA OPERATIVA (INVARIATA)
# ============================================================
auto_clienti_anno = bev_citta * public_share * quota_stazione
energia_kwh = auto_clienti_anno * 45
sessioni_anno = auto_clienti_anno
sessioni_giorno_tot = sessioni_anno / 365

stazione_A = np.ceil(sessioni_giorno_tot / 12)
stazione_B = np.ceil(sessioni_giorno_tot / 24)
n_totale = stazione_A + stazione_B

saturazione_sessioni = sessioni_giorno_tot / (n_totale * 24)

ricavi = energia_kwh * 0.65
opex = ricavi * 0.35
ebitda = ricavi - opex

capex_flow = np.zeros_like(ricavi)
capex_flow[0] = n_totale[0] * 45000

cf_netto = ebitda - capex_flow
cf_cum = np.cumsum(cf_netto)

# ============================================================
# KPI PRINCIPALI
# ============================================================
st.title("⚡ Executive Charging Investment Dashboard")

c1, c2, c3, c4 = st.columns(4)
c1.metric("BEV 2030", f"{bev_2030:,}")
c2.metric("Quota cattura", pct(quota_stazione))
c3.metric("EBITDA anno 1", eur(ebitda[0]))
c4.metric("Payback stimato", f"{np.argmax(cf_cum>0)} anni")

# ============================================================
# EXECUTIVE INVESTMENT SUMMARY (GM VIEW)
# ============================================================
st.subheader("📋 Executive Investment Summary (GM View)")

df_exec = pd.DataFrame({
    "Anno": years,
    "Clienti medi / giorno": auto_clienti_anno / 365,
    "Saturazione Asset %": saturazione_sessioni,
    "EBITDA": ebitda,
    "CAPEX annuo": capex_flow,
    "Cash Flow cumulato": cf_cum,
}).set_index("Anno")

st.dataframe(df_exec.style.format({
    "Clienti medi / giorno": "{:.1f}",
    "Saturazione Asset %": pct,
    "EBITDA": eur,
    "CAPEX annuo": eur,
    "Cash Flow cumulato": eur,
}))

# ============================================================
# MARKET FUNNEL ANALYSIS (STRATEGIC VIEW)
# ============================================================
st.subheader("📊 Market Funnel Analysis (Strategic View)")

df_funnel = pd.DataFrame({
    "Anno": years,
    "Parco BEV Provincia": bev_citta,
    "Bacino Pubblico": bev_citta * public_share,
    "Quota Cattura %": quota_stazione,
    "Energia totale venduta (kWh)": energia_kwh,
}).set_index("Anno")

st.dataframe(df_funnel.style.format({
    "Parco BEV Provincia": "{:,.0f}",
    "Bacino Pubblico": "{:,.0f}",
    "Quota Cattura %": pct,
    "Energia totale venduta (kWh)": "{:,.0f}",
}))

# ============================================================
# STRESS TEST & SENSITIVITY (CFO VIEW)
# ============================================================
if cfo_mode:
    st.subheader("⚠️ Stress Test & Sensitivity (CFO View)")

    scenarios = {
        "Bear": 0.8,
        "Base": 1.0,
        "Bull": 1.2,
    }

    rows = []
    for k, m in scenarios.items():
        cf = cf_netto * m
        rows.append({
            "Scenario": k,
            "NPV": npf.npv(0.08, cf),
            "IRR": npf.irr(cf),
            "Payback (anni)": np.argmax(np.cumsum(cf) > 0)
        })

    df_stress = pd.DataFrame(rows).set_index("Scenario")

    st.dataframe(df_stress.style.format({
        "NPV": eur,
        "IRR": "{:.2%}",
        "Payback (anni)": "{:.1f}",
    }))

# ============================================================
# REPORT ANALITICO — DF_MASTER RIORDINATO
# ============================================================
st.subheader("📈 Report Operativo & Finanziario")

df_master = pd.DataFrame({
    "Anno": years,

    # Driver operativi
    "BEV": bev_citta,
    "Quota pubblica": public_share,
    "Quota cattura": quota_stazione,
    "Auto target": auto_clienti_anno.astype(int),
    "Energia kWh": energia_kwh.astype(int),
    "Sessioni/giorno": sessioni_giorno_tot.round(1),
    "Unità totali": n_totale.astype(int),
    "Saturazione": saturazione_sessioni,

    # Finanziari (a destra)
    "Fatturato": ricavi,
    "EBITDA": ebitda,
    "CF netto": cf_netto,
    "CF cumulato": cf_cum,
}).set_index("Anno")

st.dataframe(df_master.style.format({
    "Quota pubblica": pct,
    "Quota cattura": pct,
    "Saturazione": pct,
    "Fatturato": eur,
    "EBITDA": eur,
    "CF netto": eur,
    "CF cumulato": eur,
}))
