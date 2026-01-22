"""
Business Case - DC30 kW Palermo (replica logica report)

INPUT:
- Anni, parco BEV, consumi, quota pubblico, quota stazione
- Parametri sessione (kWh/sessione), potenza (kW), uptime, utilizzo target
- Prezzo energia, costo energia, OPEX annuo per colonnina
- CAPEX per colonnina (per sensitività)

OUTPUT:
- Tabella domanda (kWh, sessioni, colonnine)
- Tabella economica (ricavi, margini, EBITDA)
- Tabella cashflow (CAPEX, CF netto, CF cumulato, payback)
- Sensitività CAPEX 20k/25k/30k
- Grafici principali
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# 1) INPUT (modifica qui)
# -----------------------------

@dataclass
class Inputs:
    years: np.ndarray
    bev: np.ndarray
    consumption_kwh_per_bev_year: float = 3000
    public_share: float = 0.30
    station_share: np.ndarray = None  # per anno

    avg_session_kwh: float = 35
    charger_power_kw: float = 30
    uptime: float = 0.97
    target_utilization: float = 0.30

    price_eur_per_kwh: float = 0.69
    cogs_eur_per_kwh: float = 0.30
    opex_eur_per_charger_year: float = 2000

    capex_sensitivity: tuple = (20000, 25000, 30000)


inputs = Inputs(
    years=np.array([2026, 2027, 2028, 2029, 2030]),
    bev=np.array([2600, 3000, 3500, 4200, 5000]),
    station_share=np.array([0.02, 0.03, 0.04, 0.045, 0.05]),  # scenario base report
)


# -----------------------------
# 2) CORE FUNCTIONS
# -----------------------------

def compute_capacity_kwh_per_charger(i: Inputs) -> float:
    """Capacità annua utile per colonnina (kWh/anno)"""
    return i.charger_power_kw * 8760 * i.uptime * i.target_utilization


def compute_demand_and_sizing(i: Inputs):
    """Calcola energia intercettata, sessioni e colonnine necessarie"""
    city_public_energy = i.bev * i.consumption_kwh_per_bev_year * i.public_share
    captured_kwh = city_public_energy * i.station_share

    sessions_year = captured_kwh / i.avg_session_kwh
    sessions_day = sessions_year / 365

    cap_per_charger = compute_capacity_kwh_per_charger(i)
    chargers_needed = np.ceil(captured_kwh / cap_per_charger).astype(int)

    return captured_kwh, sessions_year, sessions_day, chargers_needed, cap_per_charger


def compute_economics(i: Inputs, captured_kwh: np.ndarray, chargers_needed: np.ndarray):
    """Ricavi, margine energia, EBITDA"""
    revenue = captured_kwh * i.price_eur_per_kwh
    spread = i.price_eur_per_kwh - i.cogs_eur_per_kwh
    gross_margin = captured_kwh * spread
    ebitda = gross_margin - chargers_needed * i.opex_eur_per_charger_year
    return revenue, gross_margin, ebitda


def cashflows(i: Inputs, chargers_needed: np.ndarray, ebitda: np.ndarray, capex_per_charger: float):
    """CAPEX per anno, cash flow netto e cumulato, payback (anno)"""
    capex_add = np.zeros_like(i.years, dtype=float)
    prev = 0
    for t, n in enumerate(chargers_needed):
        add = max(0, n - prev)
        capex_add[t] = add * capex_per_charger
        prev = n

    capex_cum = np.cumsum(capex_add)
    cf_net = ebitda - capex_add
    cf_cum = np.cumsum(cf_net)

    payback = None
    idx = np.where(cf_cum >= 0)[0]
    if len(idx) > 0:
        payback = int(i.years[idx[0]])

    return capex_add, capex_cum, cf_net, cf_cum, payback


# -----------------------------
# 3) RUN + OUTPUT TABLES
# -----------------------------

captured_kwh, sessions_year, sessions_day, chargers_needed, cap_per_charger = compute_demand_and_sizing(inputs)
revenue, gross_margin, ebitda = compute_economics(inputs, captured_kwh, chargers_needed)

# scenario base cashflow con CAPEX medio (25k)
BASE_CAPEX = 25000
capex_add, capex_cum, cf_net, cf_cum, payback = cashflows(inputs, chargers_needed, ebitda, BASE_CAPEX)

# tabella principale
df = pd.DataFrame({
    "Anno": inputs.years,
    "BEV": inputs.bev,
    "Quota stazione": inputs.station_share,
    "Energia intercettata (kWh)": captured_kwh.round(0).astype(int),
    "Ricariche/anno": sessions_year.round(0).astype(int),
    "Ricariche/giorno": np.round(sessions_day, 1),
    "Colonnine DC30": chargers_needed,
    "Ricavi (€)": revenue.round(0).astype(int),
    "Margine energia (€)": gross_margin.round(0).astype(int),
    "OPEX (€)": (chargers_needed * inputs.opex_eur_per_charger_year).astype(int),
    "EBITDA (€)": ebitda.round(0).astype(int),
    f"CAPEX anno (€) [{BASE_CAPEX}]": capex_add.astype(int),
    f"CF netto (€) [{BASE_CAPEX}]": cf_net.round(0).astype(int),
    f"CF cumulato (€) [{BASE_CAPEX}]": cf_cum.round(0).astype(int),
})

print("\n=== Capacità annua utile per colonnina ===")
print(f"{cap_per_charger:,.1f} kWh/anno".replace(",", "."))

print("\n=== Tabella risultati (scenario base) ===")
print(df.to_string(index=False))

roi_year1 = ebitda[0] / capex_add[0] if capex_add[0] > 0 else np.nan
roi_cum_2030 = (cf_cum[-1] / capex_cum[-1]) if capex_cum[-1] > 0 else np.nan

print("\n=== KPI scenario base (CAPEX 25k) ===")
print(f"ROI anno 1: {roi_year1*100:.1f}%")
print(f"Payback: {payback}")
print(f"Cash flow cumulato 2030: {cf_cum[-1]:,.0f} €".replace(",", "."))
print(f"CAPEX cumulato 2030: {capex_cum[-1]:,.0f} €".replace(",", "."))
print(f"ROI cumulato 2030: {roi_cum_2030:.2f}x")


# -----------------------------
# 4) SENSITIVITY CAPEX
# -----------------------------

sens_rows = []
for cap in inputs.capex_sensitivity:
    cap_add_s, cap_cum_s, cf_net_s, cf_cum_s, pb_s = cashflows(inputs, chargers_needed, ebitda, cap)
    roi_cum = (cf_cum_s[-1] / cap_cum_s[-1]) if cap_cum_s[-1] > 0 else np.nan
    sens_rows.append({
        "CAPEX per colonnina (€)": cap,
        "CAPEX cumulato 2030 (€)": int(cap_cum_s[-1]),
        "CF cumulato 2030 (€)": int(cf_cum_s[-1]),
        "ROI cumulato 2030 (x)": round(roi_cum, 2),
        "Payback (anno)": pb_s
    })

sens_df = pd.DataFrame(sens_rows)
print("\n=== Sensitività CAPEX ===")
print(sens_df.to_string(index=False))


# -----------------------------
# 5) PLOTS
# -----------------------------

def plot_line(x, y, title, ylabel):
    plt.figure(figsize=(8, 4.5))
    plt.plot(x, y, marker="o")
    plt.title(title)
    plt.xlabel("Anno")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

plot_line(inputs.years, inputs.bev, "Evoluzione parco BEV (Palermo)", "Numero BEV")
plot_line(inputs.years, captured_kwh/1000, "Energia intercettata dalla stazione", "Energia (MWh/anno)")
plot_line(inputs.years, chargers_needed, "Numero colonnine DC30 necessarie", "Colonnine (n.)")
plot_line(inputs.years, ebitda/1000, "EBITDA incrementale annuo", "EBITDA (k€)")

# cash flow cumulato per CAPEX sensitivity
plt.figure(figsize=(8, 4.5))
for cap in inputs.capex_sensitivity:
    _, _, _, cf_cum_s, _ = cashflows(inputs, chargers_needed, ebitda, cap)
    plt.plot(inputs.years, cf_cum_s/1000, marker="o", label=f"CAPEX {cap/1000:.0f}k€")
plt.title("Cash flow cumulato (EBITDA - CAPEX) per sensitività CAPEX")
plt.xlabel("Anno")
plt.ylabel("Cash flow cumulato (k€)")
plt.axhline(0)
plt.legend()
plt.tight_layout()
plt.show()


# -----------------------------
# 6) OPTIONAL: EXPORT CSV
# -----------------------------
# df.to_csv("business_case_dc30_output.csv", index=False)
# sens_df.to_csv("business_case_dc30_sensitivity.csv", index=False)
