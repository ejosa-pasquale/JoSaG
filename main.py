import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import ceil

# =========================
# CONFIG STREAMLIT
# =========================
st.set_page_config(
    page_title="BEV Charging – Executive Investment Tool",
    layout="wide"
)

st.title("🔌 BEV Charging Investment – GM & CFO Dashboard")

# =========================
# INPUT STRATEGICI
# =========================
st.sidebar.header("Assunzioni Strategiche")

province = st.sidebar.selectbox("Provincia", ["Catania", "Palermo"])

pct_no_garage = st.sidebar.slider(
    "Quota BEV senza garage",
    0.30, 0.40, 0.35, 0.01
)

capture_rate_base = st.sidebar.slider(
    "Quota di cattura base",
    0.05, 0.30, 0.15, 0.01
)

price_kwh = st.sidebar.slider(
    "Prezzo vendita €/kWh",
    0.55, 0.90, 0.70, 0.01
)

cost_energy = st.sidebar.slider(
    "Costo energia €/kWh",
    0.15, 0.40, 0.25, 0.01
)

sessions_per_car = st.sidebar.number_input(
    "Sessioni / auto / anno",
    50, 200, 120
)

kwh_per_session = st.sidebar.number_input(
    "kWh per sessione",
    20, 80, 40
)

power_kw = st.sidebar.selectbox("Potenza colonnina (kW)", [50, 100, 150])
uptime = 0.95
utilization = 0.25

capex_unit = st.sidebar.number_input(
    "CAPEX per colonnina (€)",
    30000, 120000, 60000
)

opex_annual = st.sidebar.number_input(
    "OPEX annuo per colonnina (€)",
    3000, 15000, 7000
)

# =========================
# DATI BEV (mock → sostituibile con main-44.py)
# =========================
def forecast_bev_2030_mock():
    return {
        2025: 6000,
        2026: 8000,
        2027: 10500,
        2028: 13500,
        2029: 17000,
        2030: 21000
    }

bev_forecast = forecast_bev_2030_mock()

# =========================
# ELASTICITÀ PREZZO
# =========================
def adjusted_capture_rate(price, base_rate):
    if price <= 0.70:
        return base_rate
    delta_cents = (price - 0.70) / 0.01
    return max(0, base_rate * (1 - 0.015 * delta_cents))

capture_rate = adjusted_capture_rate(price_kwh, capture_rate_base)

# =========================
# CAPACITÀ COLONNINA
# =========================
capacity_per_unit = power_kw * 8760 * uptime * utilization

# =========================
# SIMULAZIONE FINANZIARIA
# =========================
results = []

units_installed = 1
cashflow_cum = 0
total_capex = capex_unit

for year, bev_total in bev_forecast.items():

    bev_target = bev_total * pct_no_garage
    bev_captured = bev_target * capture_rate

    sessions_year = bev_captured * sessions_per_car
    energy_demand = sessions_year * kwh_per_session

    units_required = ceil(energy_demand / capacity_per_unit)

    capex = 0
    if units_required > units_installed:
        added = units_required - units_installed
        capex = added * capex_unit
        units_installed = units_required
        total_capex += capex

    revenue = energy_demand * price_kwh
    energy_cost = energy_demand * cost_energy
    margin = revenue - energy_cost
    opex = units_installed * opex_annual
    ebitda = margin - opex

    cashflow = ebitda - capex
    cashflow_cum += cashflow

    results.append([
        year, bev_total, bev_target, bev_captured,
        energy_demand, units_installed,
        revenue, margin, opex, ebitda,
        capex, cashflow, cashflow_cum
    ])

df = pd.DataFrame(results, columns=[
    "Anno", "BEV Totali", "BEV Target", "BEV Catturati",
    "Domanda kWh", "Colonnine",
    "Ricavi", "Margine", "OPEX", "EBITDA",
    "CAPEX", "Cash Flow", "Cash Flow Cumulato"
])

# =========================
# KPI EXECUTIVE
# =========================
payback_year = df[df["Cash Flow Cumulato"] > 0]["Anno"].min()
roi_5y = df.iloc[:5]["Cash Flow"].sum() / total_capex

st.subheader("📊 Snapshot Finanziario – CFO View")
st.dataframe(df.style.format("{:,.0f}"))

col1, col2, col3 = st.columns(3)
col1.metric("Payback Period", f"{payback_year}")
col2.metric("ROI 5 anni", f"{roi_5y:.1%}")
col3.metric("Auto servite / giorno",
             f"{(df.iloc[0]['BEV Catturati']*sessions_per_car)/365:.1f}")

# =========================
# WATERFALL FUNNEL
# =========================
st.subheader("🚗 Funnel di Mercato")

funnel_values = [
    df.iloc[0]["BEV Totali"],
    df.iloc[0]["BEV Target"],
    df.iloc[0]["BEV Catturati"]
]

labels = ["Parco BEV", "No Garage", "Catturati"]

fig, ax = plt.subplots()
ax.bar(labels, funnel_values)
ax.set_ylabel("Numero Auto")
st.pyplot(fig)

# =========================
# SATURAZIONE
# =========================
st.subheader("⚡ Saturazione – Domanda vs Capacità")

fig, ax = plt.subplots()
ax.plot(df["Anno"], df["Domanda kWh"], label="Domanda kWh")
ax.plot(df["Anno"], df["Colonnine"] * capacity_per_unit,
        label="Capacità Installata")
ax.legend()
st.pyplot(fig)

# =========================
# SENSITIVITÀ PREZZO
# =========================
st.subheader("💶 Sensibilità Prezzo – EBITDA")

prices = np.arange(0.55, 0.90, 0.01)
ebitdas = []

for p in prices:
    cr = adjusted_capture_rate(p, capture_rate_base)
    bev_cap = df.iloc[0]["BEV Target"] * cr
    energy = bev_cap * sessions_per_car * kwh_per_session
    rev = energy * p
    cost = energy * cost_energy
    ebitdas.append(rev - cost - opex_annual)

fig, ax = plt.subplots()
ax.plot(prices, ebitdas)
ax.set_xlabel("Prezzo €/kWh")
ax.set_ylabel("EBITDA")
st.pyplot(fig)

opt_price = prices[np.argmax(ebitdas)]
st.success(f"🎯 Prezzo ottimo per EBITDA: {opt_price:.2f} €/kWh")

# =========================
# TORNADO CHART
# =========================
st.subheader("🌪️ Tornado Chart – Rischi Principali")

base_ebitda = df.iloc[0]["EBITDA"]

risks = {
    "Adozione BEV": (-0.2, 0.2),
    "Costo Energia": (-0.3, 0.3),
    "Quota di Cattura": (-0.25, 0.25)
}

impact = []

for r, (low, high) in risks.items():
    impact.append([
        r,
        base_ebitda * (1 + low),
        base_ebitda * (1 + high)
    ])

df_tornado = pd.DataFrame(impact, columns=["Rischio", "Worst", "Best"])

fig, ax = plt.subplots()
ax.barh(df_tornado["Rischio"], df_tornado["Best"] - base_ebitda,
        left=base_ebitda)
ax.barh(df_tornado["Rischio"], df_tornado["Worst"] - base_ebitda,
        left=base_ebitda)
ax.axvline(base_ebitda)
st.pyplot(fig)
