import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Configurazione Pagina
st.set_page_config(page_title="Business Case DC30 kW Palermo", layout="wide")

st.title("⚡ Business Case: Stazione di Ricarica DC 30 kW")
st.sidebar.header("Parametri di Input")

# -----------------------------
# 1) INPUT NELLA SIDEBAR
# -----------------------------
target_utilization = st.sidebar.slider("Utilizzo Target (%)", 5, 50, 30) / 100
price_kwh = st.sidebar.number_input("Prezzo vendita (€/kWh)", value=0.69)
cost_kwh = st.sidebar.number_input("Costo energia (€/kWh)", value=0.30)
opex_unit = st.sidebar.number_input("OPEX annuo per colonnina (€)", value=2000)

@dataclass
class Inputs:
    years = np.array([2026, 2027, 2028, 2029, 2030])
    bev = np.array([2600, 3000, 3500, 4200, 5000])
    station_share = np.array([0.02, 0.03, 0.04, 0.045, 0.05])
    consumption_kwh_per_bev_year: float = 3000
    public_share: float = 0.30
    avg_session_kwh: float = 35
    charger_power_kw: float = 30
    uptime: float = 0.97
    target_utilization: float = target_utilization
    price_eur_per_kwh: float = price_kwh
    cogs_eur_per_kwh: float = cost_kwh
    opex_eur_per_charger_year: float = opex_unit
    capex_sensitivity: tuple = (20000, 25000, 30000)

inputs = Inputs()

# -----------------------------
# 2) CALCOLI
# -----------------------------
cap_per_charger = inputs.charger_power_kw * 8760 * inputs.uptime * inputs.target_utilization
city_public_energy = inputs.bev * inputs.consumption_kwh_per_bev_year * inputs.public_share
captured_kwh = city_public_energy * inputs.station_share
sessions_day = (captured_kwh / inputs.avg_session_kwh) / 365
chargers_needed = np.ceil(captured_kwh / cap_per_charger).astype(int)

revenue = captured_kwh * inputs.price_eur_per_kwh
ebitda = (captured_kwh * (inputs.price_eur_per_kwh - inputs.cogs_eur_per_kwh)) - (chargers_needed * inputs.opex_eur_per_charger_year)

# -----------------------------
# 3) UI STREAMLIT
# -----------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Capacità/Colonnina", f"{int(cap_per_charger)} kWh/y")
col2.metric("Energia Totale 2030", f"{int(captured_kwh[-1])} kWh")
col3.metric("Max Colonnine", f"{chargers_needed[-1]}")

st.subheader("📊 Tabella di Riepilogo")
df = pd.DataFrame({
    "Anno": inputs.years,
    "BEV Palermo": inputs.bev,
    "Energia (kWh)": captured_kwh.astype(int),
    "Sessioni/Giorno": sessions_day.round(1),
    "N. Colonnine": chargers_needed,
    "EBITDA (€)": ebitda.astype(int)
})
st.dataframe(df, use_container_width=True)

st.subheader("📈 Analisi Grafica")
fig, ax1 = plt.subplots(figsize=(10, 4))
ax1.bar(inputs.years, ebitda, color='skyblue', label='EBITDA')
ax1.set_ylabel("EBITDA (€)")
ax2 = ax1.twinx()
ax2.plot(inputs.years, chargers_needed, color='red', marker='o', label='N. Colonnine')
ax2.set_ylabel("Numero Colonnine")
st.pyplot(fig)

# Sensitività CAPEX
st.subheader("💰 Sensitività CAPEX e Payback")
sens_rows = []
for cap in inputs.capex_sensitivity:
    capex_ann = np.zeros_like(inputs.years, dtype=float)
    prev = 0
    for t, n in enumerate(chargers_needed):
        capex_ann[t] = max(0, n - prev) * cap
        prev = n
    cf_cum = np.cumsum(ebitda - capex_ann)
    idx = np.where(cf_cum >= 0)[0]
    pb = str(inputs.years[idx[0]]) if len(idx) > 0 else "> 2030"
    sens_rows.append({"CAPEX Unitario (€)": cap, "Cash Flow 2030 (€)": int(cf_cum[-1]), "Payback": pb})

st.table(pd.DataFrame(sens_rows))
