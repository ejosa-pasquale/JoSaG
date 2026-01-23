# ============================================================
# EV CHARGING – BUSINESS INTELLIGENCE EXECUTIVE TOOL
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="EV Charging – Executive BI",
    layout="wide"
)

st.title("⚡ EV Charging – Executive Business Intelligence")

# ============================================================
# SIDEBAR – INPUT STRATEGICI
# ============================================================

st.sidebar.header("📊 Input Strategici")

# --- Scenario BEV
bev_2024 = st.sidebar.number_input(
    "Parco BEV 2024 (provincia)",
    min_value=1_000,
    value=25_000,
    step=1_000
)

cagr_used = st.sidebar.slider(
    "CAGR storico BEV (%)",
    5.0, 35.0, 22.0
) / 100

# --- Funnel
street_share = st.sidebar.slider(
    "Utenti senza ricarica privata (%)",
    20, 60, 35
) / 100

capture_base = st.sidebar.slider(
    "Capture Rate stazione (%)",
    1, 20, 5
) / 100

# --- Parametri energetici
kwh_annui_per_auto = st.sidebar.number_input(
    "Consumo annuo medio per auto (kWh)",
    value=2_200
)

# --- Parametri tecnici colonnina
potenza_kw = st.sidebar.selectbox(
    "Potenza colonnina (kW)",
    [50, 100, 150, 300],
    index=1
)

uptime = st.sidebar.slider(
    "Uptime tecnico (%)",
    90, 99, 97
) / 100

utilizzo_medio = st.sidebar.slider(
    "Utilizzo medio (%)",
    5, 40, 18
) / 100

# --- Prezzi e costi
prezzo_kwh = st.sidebar.slider(
    "Prezzo vendita (€ / kWh)",
    0.40, 1.00, 0.65
)

costo_kwh = st.sidebar.slider(
    "Costo energia (€ / kWh)",
    0.15, 0.45, 0.28
)

# ============================================================
# FUNZIONI CORE BI
# ============================================================

def capacity_unit_kwh_year(p_kw, uptime, utilization):
    return p_kw * 8760 * uptime * utilization


def price_elasticity_capture(capture, price, threshold=0.70, elasticity=1.5):
    if price <= threshold:
        return capture
    delta = price - threshold
    return max(0, capture * (1 - elasticity * delta))


# ============================================================
# ORIZZONTE TEMPORALE
# ============================================================

years = np.arange(2024, 2031)

# ============================================================
# STADIO 1 – PARCO BEV
# ============================================================

bev_forecast = bev_2024 * (1 + cagr_used) ** (years - 2024)

# ============================================================
# STADIO 2 – TARGET STRADA
# ============================================================

bev_street = bev_forecast * street_share

# ============================================================
# STADIO 3 – CATTURA (CON ELASTICITÀ PREZZO)
# ============================================================

capture_adj = price_elasticity_capture(
    capture_base,
    prezzo_kwh
)

auto_clienti = bev_street * capture_adj
energia_kwh = auto_clienti * kwh_annui_per_auto

# ============================================================
# DIMENSIONAMENTO DINAMICO UNITÀ
# ============================================================

cap_kwh_unit = capacity_unit_kwh_year(
    potenza_kw,
    uptime,
    utilizzo_medio
)

n_units = np.ceil(energia_kwh / cap_kwh_unit).astype(int)

# ============================================================
# ECONOMICS
# ============================================================

ricavi = energia_kwh * prezzo_kwh
costi = energia_kwh * costo_kwh
ebitda = ricavi - costi

# semplice proxy cash flow
capex_unit = 45_000
opex_unit = 2_000

capex = np.diff(
    np.insert(n_units * capex_unit, 0, 0)
)
opex = n_units * opex_unit

cash_flow = ebitda - capex - opex

# ============================================================
# LAYOUT PRINCIPALE
# ============================================================

col1, col2 = st.columns(2)

# ============================================================
# FUNNEL DI CONVERSIONE
# ============================================================

with col1:
    st.subheader("🔻 Funnel Street-Capture")

    funnel_vals = [
        bev_forecast[-1],
        bev_street[-1],
        auto_clienti[-1]
    ]

    fig, ax = plt.subplots()
    ax.bar(
        ["Parco BEV", "Domanda Strada", "Target Stazione"],
        funnel_vals
    )
    ax.set_ylabel("Veicoli / anno")
    st.pyplot(fig)

# ============================================================
# SATURAZIONE CAPACITÀ
# ============================================================

with col2:
    st.subheader("📈 Saturazione Capacità")

    units_equivalent = energia_kwh / cap_kwh_unit

    fig, ax = plt.subplots()
    ax.plot(years, units_equivalent, marker="o")
    ax.axhline(1, linestyle="--", label="Capacità 1 unità")
    ax.set_ylabel("Unità equivalenti richieste")
    ax.set_xlabel("Anno")
    ax.legend()
    st.pyplot(fig)

# ============================================================
# PROFIT vs PRICE (SWEET SPOT)
# ============================================================

st.subheader("💰 Elasticità Prezzo – Profit vs Price")

price_range = np.linspace(0.40, 1.00, 30)
profits = []

for p in price_range:
    cap_adj = price_elasticity_capture(capture_base, p)
    energia_adj = bev_street[-1] * cap_adj * kwh_annui_per_auto
    profits.append(
        energia_adj * (p - costo_kwh)
    )

fig, ax = plt.subplots()
ax.plot(price_range, profits, linewidth=3)
ax.axvline(
    price_range[np.argmax(profits)],
    linestyle="--",
    label="Sweet Spot"
)
ax.set_xlabel("Prezzo €/kWh")
ax.set_ylabel("Margine Operativo (€)")
ax.legend()
st.pyplot(fig)

st.success(
    f"🎯 Sweet Spot Prezzo ≈ {price_range[np.argmax(profits)]:.2f} €/kWh"
)

# ============================================================
# TABELLA EXECUTIVE
# ============================================================

st.subheader("📋 Executive Financial Table")

df = pd.DataFrame({
    "BEV totali": bev_forecast.astype(int),
    "Target strada": bev_street.astype(int),
    "Auto catturate": auto_clienti.astype(int),
    "Energia (kWh)": energia_kwh.astype(int),
    "Unità richieste": n_units,
    "Ricavi (€)": ricavi.astype(int),
    "EBITDA (€)": ebitda.astype(int),
    "Cash Flow (€)": cash_flow.astype(int)
}, index=years)

st.dataframe(df, use_container_width=True)

# ============================================================
# KPI FINALI
# ============================================================

st.markdown("---")
st.metric("Unità installate a regime", int(n_units.max()))
st.metric("EBITDA medio annuo (€)", int(ebitda.mean()))
st.metric("Cash Flow cumulato (€)", int(cash_flow.sum()))
