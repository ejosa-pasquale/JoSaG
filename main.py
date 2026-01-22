import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass

# -----------------------------
# 1) CONFIGURAZIONE INPUT
# -----------------------------

@dataclass
class Inputs:
    years: np.ndarray
    bev: np.ndarray
    consumption_kwh_per_bev_year: float = 3000
    public_share: float = 0.30
    station_share: np.ndarray = None  # Quota mercato della stazione specifica

    avg_session_kwh: float = 35
    charger_power_kw: float = 30
    uptime: float = 0.97
    target_utilization: float = 0.30

    price_eur_per_kwh: float = 0.69
    cogs_eur_per_kwh: float = 0.30
    opex_eur_per_charger_year: float = 2000

    capex_sensitivity: tuple = (20000, 25000, 30000)

# Dati Palermo Scenario Base
inputs = Inputs(
    years=np.array([2026, 2027, 2028, 2029, 2030]),
    bev=np.array([2600, 3000, 3500, 4200, 5000]),
    station_share=np.array([0.02, 0.03, 0.04, 0.045, 0.05]),
)

# -----------------------------
# 2) LOGICA DI CALCOLO
# -----------------------------

def compute_capacity_kwh_per_charger(i: Inputs) -> float:
    # Ore totali anno * Uptime * % di tempo in cui è occupata al 100% della potenza
    return i.charger_power_kw * 8760 * i.uptime * i.target_utilization

def compute_demand_and_sizing(i: Inputs):
    city_public_energy = i.bev * i.consumption_kwh_per_bev_year * i.public_share
    captured_kwh = city_public_energy * i.station_share

    sessions_year = captured_kwh / i.avg_session_kwh
    sessions_day = sessions_year / 365

    cap_per_charger = compute_capacity_kwh_per_charger(i)
    # Calcolo colonnine necessarie per soddisfare la domanda target
    chargers_needed = np.ceil(captured_kwh / cap_per_charger).astype(int)

    return captured_kwh, sessions_year, sessions_day, chargers_needed, cap_per_charger

def cashflows(i: Inputs, chargers_needed: np.ndarray, ebitda: np.ndarray, capex_per_unit: float):
    capex_annual = np.zeros_like(i.years, dtype=float)
    prev_n = 0
    for t, n in enumerate(chargers_needed):
        new_units = max(0, n - prev_n)
        capex_annual[t] = new_units * capex_per_unit
        prev_n = n

    cf_net = ebitda - capex_annual
    cf_cum = np.cumsum(cf_net)
    
    # Calcolo Payback dinamico
    idx = np.where(cf_cum >= 0)[0]
    payback = str(i.years[idx[0]]) if len(idx) > 0 else "> 2030"
    
    return capex_annual, cf_net, cf_cum, payback

# -----------------------------
# 3) ESECUZIONE E OUTPUT
# -----------------------------

captured_kwh, sessions_year, sessions_day, chargers_needed, cap_per_charger = compute_demand_and_sizing(inputs)

# Economics
revenue = captured_kwh * inputs.price_eur_per_kwh
opex_total = chargers_needed * inputs.opex_eur_per_charger_year
ebitda = (captured_kwh * (inputs.price_eur_per_kwh - inputs.cogs_eur_per_kwh)) - opex_total

# Scenario Base (CAPEX 25k)
BASE_CAPEX = 25000
capex_ann, cf_net, cf_cum, pb = cashflows(inputs, chargers_needed, ebitda, BASE_CAPEX)

# DataFrame Risultati
df = pd.DataFrame({
    "Anno": inputs.years,
    "BEV Palermo": inputs.bev,
    "Energia (kWh)": captured_kwh.astype(int),
    "Sessions/Day": sessions_day.round(1),
    "N. Colonnine": chargers_needed,
    "EBITDA (€)": ebitda.astype(int),
    "CF Cumulato (€)": cf_cum.astype(int)
})

print("\n--- REPORT BUSINESS CASE DC30 PALERMO ---")
print(df.to_string(index=False))

# Sensitività
sens_data = []
for c in inputs.capex_sensitivity:
    _, _, c_cum, p_back = cashflows(inputs, chargers_needed, ebitda, c)
    sens_data.append({"CAPEX Unitario": c, "CF 2030": int(c_cum[-1]), "Payback": p_back})

print("\n--- SENSITIVITÀ CAPEX ---")
print(pd.DataFrame(sens_data).to_string(index=False))

# -----------------------------
# 4) GRAFICI
# -----------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Grafico 1: EBITDA e Colonnine
ax1.bar(inputs.years, ebitda, color='skyblue', label='EBITDA Annuo')
ax1.set_ylabel("Euro (€)")
ax1_tw = ax1.twinx()
ax1_tw.step(inputs.years, chargers_needed, where='post', color='red', label='N. Colonnine', linewidth=2)
ax1_tw.set_ylabel("Numero Colonnine")
ax1.set_title("Evoluzione Economica vs Dimensionamento")
ax1.legend(loc='upper left')

# Grafico 2: Cash Flow Cumulato (Sensitività)
for c in inputs.capex_sensitivity:
    _, _, c_cum, _ = cashflows(inputs, chargers_needed, ebitda, c)
    ax2.plot(inputs.years, c_cum, marker='o', label=f"CAPEX {c/1000}k")

ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
ax2.set_title("Analisi del Payback Period")
ax2.set_ylabel("Cash Flow Cumulato (€)")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
