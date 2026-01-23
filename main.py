import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy_financial as npf

st.set_page_config(page_title="Charging Decision Tool (chiaro per GM + CFO)", layout="wide")

# ============================================================
# Helpers
# ============================================================
def eur(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    s = f"{x:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"€ {s}"

def pct(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x*100:.1f}%"

def payback_years_from_cum(cum, years_axis):
    """Return interpolated payback year where cum crosses zero."""
    for i in range(1, len(cum)):
        if cum[i] >= 0 and cum[i-1] < 0:
            frac = (0 - cum[i-1]) / (cum[i] - cum[i-1]) if cum[i] != cum[i-1] else 0
            return years_axis[i-1] + frac * (years_axis[i] - years_axis[i-1])
    return np.nan

def piecewise_interp(x, ax, ay):
    return np.interp(x, ax, ay)

def clamp01(x):
    return max(0.0, min(1.0, x))

# ============================================================
# Model
# ============================================================
def run_model(p):
    years = p["years"]
    n = len(years)
    t = np.arange(n)

    potenza_kw = 30 if p["tecnologia"] == "DC 30 kW" else 60
    effective_kw = potenza_kw * p["taper_factor"]

    # session time
    session_minutes = (p["kwh_per_session"] / max(effective_kw, 1e-6)) * 60 + p["overhead_min"]

    # capacity per unit / year
    sessions_per_unit_day = (p["ore_max_giorno"] * 60) / max(session_minutes, 1e-6)
    cap_sessions_unit_year = sessions_per_unit_day * 365 * p["uptime"] * p["util_target"]

    # Ramp-up
    if p["ramp_years_to_full"] <= 0:
        ramp = np.ones(n)
    else:
        ramp = np.minimum(1.0, p["ramp_start_pct"] + (1 - p["ramp_start_pct"]) * (t / p["ramp_years_to_full"]))

    # Price & cost
    price_kwh = p["prezzo_kwh"] * ((1 + p["price_escal"]) ** t)
    cost_kwh = p["costo_kwh"] * ((1 + p["cost_escal"]) ** t)
    infl = (1 + p["inflation"]) ** t

    # Demand
    if p["demand_model"] == "Bottom-up (Traffico sito)":
        traffic_day = p["traffic_veicoli_giorno"] * ((1 + p["traffic_growth"]) ** t)
        ev_share = piecewise_interp(years, [years[0], 2030, years[-1]], [p["ev_share_2026"], p["ev_share_2030"], p["ev_share_2035"]])
        ev_share = np.clip(ev_share, 0, 1)

        conv_adj = p["conversione_ricarica"] * p["competitor_factor"] * ((p["prezzo_mercato_ref"] / np.maximum(price_kwh, 1e-6)) ** p["price_elasticity"])
        conv_adj = np.clip(conv_adj, 0, 1)

        sessions_demand = traffic_day * 365 * p["ingress_rate"] * ev_share * conv_adj * ramp
        energy_demand = sessions_demand * p["kwh_per_session"]
    else:
        bev_2030 = p["bev_target_2030"] * p["stress_bev"]
        bev_2026 = bev_2030 * p["bev_start_ratio_2026"]
        bev_2035 = bev_2030 * p["bev_growth_2035_vs_2030"]
        bev_city = piecewise_interp(years, [years[0], 2030, years[-1]], [bev_2026, bev_2030, bev_2035])

        bev_public = bev_city * p["public_share"]
        capture = piecewise_interp(years, [years[0], 2030, years[-1]], [p["capture_2026"], p["capture_2030"], p["capture_2035"]])
        capture = np.clip(capture * p["stress_cattura"], 0, 1)
        bev_captured = bev_public * capture

        energy_per_ev_year = p["km_per_ev_year"] * (p["kwh_per_100km"] / 100.0)
        energy_demand = bev_captured * energy_per_ev_year * p["share_kwh_at_station"] * ramp
        sessions_demand = energy_demand / max(p["kwh_per_session"], 1e-6)

    # Sizing
    req_units = np.ceil(sessions_demand / max(cap_sessions_unit_year, 1e-6)).astype(int)

    # Constraints
    max_units_grid = p["max_units_installabili"]
    if p["potenza_disponibile_kw"] > 0:
        max_units_grid = min(max_units_grid, int(np.floor(p["potenza_disponibile_kw"] / potenza_kw)))

    units = np.clip(req_units, p["min_units"], max_units_grid)

    # Allocation A/B
    units_A = np.zeros(n, dtype=int)
    units_B = np.zeros(n, dtype=int)
    for i in range(n):
        if p["allocazione"] == "Multisito (Espansione in B)" and units[i] > 1:
            units_A[i] = 1
            units_B[i] = units[i] - 1
        else:
            units_A[i] = units[i]
            units_B[i] = 0

    cap_sessions = units * cap_sessions_unit_year
    served_sessions = np.minimum(sessions_demand, cap_sessions)
    lost_sessions = np.maximum(0, sessions_demand - served_sessions)

    served_energy = served_sessions * p["kwh_per_session"]
    lost_energy = lost_sessions * p["kwh_per_session"]
    lost_pct = np.where(sessions_demand > 0, lost_sessions / sessions_demand, 0.0)

    utilization = np.where(cap_sessions > 0, served_sessions / cap_sessions, 0.0)

    # Revenues & costs
    rev_charging = served_energy * price_kwh
    anc_session = p["ancillary_margin_per_session"] * infl
    rev_ancillary = served_sessions * anc_session
    revenue_total = rev_charging + rev_ancillary

    energy_cost = served_energy * cost_kwh
    payment_fees = rev_charging * p["payment_fee_pct"]

    opex_unit_year = (p["maint_per_unit"] + p["backend_per_unit"] + p["lease_per_unit"]) * infl
    fixed_opex_site = p["fixed_opex_site"] * infl

    contracted_kw = units * potenza_kw * p["contracted_power_factor"]
    demand_charges = contracted_kw * p["demand_charge_eur_per_kw_year"] * infl

    opex_total = units * opex_unit_year + (units > 0).astype(int) * fixed_opex_site + demand_charges

    ebitda = revenue_total - energy_cost - payment_fees - opex_total

    # CAPEX schedule (inizio anno)
    capex_units = np.zeros(n)
    prev_u = 0
    for i in range(n):
        add = max(0, units[i] - prev_u)
        capex_units[i] = add * p["capex_unit"]
        prev_u = units[i]

    capex_fixed_A = np.zeros(n)
    if np.any(units_A > 0):
        first_A = int(np.argmax(units_A > 0))
        capex_fixed_A[first_A] = p["capex_fixed_site_A"]

    capex_fixed_B = np.zeros(n)
    if np.any(units_B > 0):
        first_B = int(np.argmax(units_B > 0))
        capex_fixed_B[first_B] = p["capex_fixed_site_B"]

    capex = capex_units + capex_fixed_A + capex_fixed_B

    # Depreciation + tax + WC (CFO-ready ma “dietro le quinte”)
    dep_life = max(int(p["depr_years"]), 1)
    depreciation = np.zeros(n)
    for i in range(n):
        if capex[i] > 0:
            annual = capex[i] / dep_life
            for j in range(i, min(n, i + dep_life)):
                depreciation[j] += annual

    ebit = ebitda - depreciation
    taxes = np.maximum(0, ebit) * p["tax_rate"]
    nopat = ebit - taxes
    op_cf = nopat + depreciation

    nwc = revenue_total * p["nwc_pct"]
    delta_nwc = np.diff(np.r_[0.0, nwc])
    op_cf_after_wc = op_cf - delta_nwc

    # Residual value (approssimazione NBV residuo)
    nbv_end = 0.0
    for i in range(n):
        years_elapsed = n - i
        remaining = max(0, dep_life - years_elapsed)
        nbv_end += capex[i] * (remaining / dep_life)

    residual_value = nbv_end * p["residual_recovery_pct"] - p["decommissioning_cost"]

    # Cashflow nodes for NPV/IRR (CFO timeline)
    cf_nodes = np.zeros(n + 1)
    cf_nodes[0] = -capex[0]
    for i in range(1, n):
        cf_nodes[i] = op_cf_after_wc[i - 1] - capex[i]
    cf_nodes[n] = op_cf_after_wc[n - 1] + residual_value

    npv = npf.npv(p["wacc"], cf_nodes)
    irr = npf.irr(cf_nodes)

    # Simple ROI (manager-friendly): (Cassa operativa cumulata - CAPEX cumulato) / CAPEX cumulato
    total_capex = capex.sum()
    total_opcf = op_cf_after_wc.sum()
    simple_roi = (total_opcf - total_capex) / total_capex if total_capex > 0 else np.nan

    # Payback on simple annual view (undiscounted) using annual net cash approx
    annual_net = op_cf_after_wc - capex
    cum_annual = np.cumsum(annual_net)
    payback_year = payback_years_from_cum(cum_annual, years)

    df = pd.DataFrame({
        "Anno": years,
        "Unità Tot": units,
        "Unità A": units_A,
        "Unità B": units_B,
        "Sessioni Domanda": sessions_demand,
        "Sessioni Servite": served_sessions,
        "Sessioni Perse": lost_sessions,
        "Vendite Perse %": lost_pct * 100,
        "Energia Domanda (kWh)": energy_demand,
        "Energia Servita (kWh)": served_energy,
        "Energia Persa (kWh)": lost_energy,
        "Utilizzo %": utilization * 100,
        "Prezzo (€/kWh)": price_kwh,
        "Costo energia (€/kWh)": cost_kwh,
        "Fatturato ricarica (€)": rev_charging,
        "Fatturato extra (€)": rev_ancillary,
        "Fatturato Tot (€)": revenue_total,
        "EBITDA (€)": ebitda,
        "CAPEX (€)": capex,
        "OPCF after WC (€)": op_cf_after_wc,
        "Net Cash (€/anno)": annual_net
    }).set_index("Anno")

    return {
        "df": df,
        "potenza_kw": potenza_kw,
        "session_minutes": float(session_minutes),
        "npv": float(npv) if np.isfinite(npv) else np.nan,
        "irr": float(irr) if (irr is not None and np.isfinite(irr)) else np.nan,
        "simple_roi": float(simple_roi) if np.isfinite(simple_roi) else np.nan,
        "total_capex": float(total_capex),
        "total_revenue": float(revenue_total.sum()),
        "total_ebitda": float(ebitda.sum()),
        "payback_year": float(payback_year) if np.isfinite(payback_year) else np.nan,
        "cf_nodes": cf_nodes,
        "cum_annual": cum_annual,
        "annual_net": annual_net,
        "residual_value": float(residual_value),
        "lost_energy_total": float(lost_energy.sum()),
    }

def npv_from_overrides(base_params, overrides):
    p2 = dict(base_params)
    p2.update(overrides)
    return run_model(p2)["npv"]

# ============================================================
# UI
# ============================================================
st.title("⚡ Charging Decision Tool")
st.markdown("**Obiettivo:** capire in modo semplice **investimento**, **fatturato**, **ROI** e **colli di bottiglia** (vendite perse).")

years = np.arange(2026, 2036)
focus_year = 2030

# ---------------- Sidebar ----------------
st.sidebar.header("1) Domanda: scegli modello")
demand_model = st.sidebar.radio("Approccio", ["Bottom-up (Traffico sito)", "Top-down (Parco BEV città)"], index=0)

st.sidebar.header("2) Asset & vincoli")
with st.sidebar.expander("Tecnica & capacità", expanded=True):
    tecnologia = st.selectbox("Tecnologia", ["DC 30 kW", "DC 60 kW"], index=1)
    allocazione = st.radio("Strategia location", ["Monosito (Tutto in A)", "Multisito (Espansione in B)"], index=0)

    ore_max_giorno = st.slider("Ore operative al giorno", 4, 24, 12)
    kwh_per_session = st.number_input("kWh medi per sessione", min_value=5.0, max_value=200.0, value=35.0, step=1.0)

    taper_factor = st.slider("Potenza effettiva % (tapering)", 50, 100, 75) / 100
    overhead_min = st.slider("Overhead per sessione (min)", 0, 15, 4)

    uptime = st.slider("Uptime tecnico %", 85, 100, 97) / 100
    util_target = st.slider("Target utilizzo % (evita code)", 50, 95, 80) / 100

    min_units = st.slider("Min unità (0 = anche 'non installare')", 0, 6, 1)
    max_units_installabili = st.slider("Max unità installabili (vincolo area)", 1, 20, 6)
    potenza_disponibile_kw = st.number_input("Potenza disponibile (kW) (0 = ignora)", min_value=0.0, value=0.0, step=10.0)

st.sidebar.header("3) Pricing & costi")
with st.sidebar.expander("Prezzo e costo energia", expanded=True):
    prezzo_kwh = st.number_input("Prezzo vendita (€/kWh)", min_value=0.10, value=0.69, step=0.01)
    costo_kwh = st.number_input("Costo energia (€/kWh)", min_value=0.01, value=0.30, step=0.01)
    price_escal = st.slider("Variazione prezzo annua %", -10.0, 10.0, 0.0) / 100
    cost_escal = st.slider("Variazione costo energia annua %", -10.0, 15.0, 0.0) / 100
    payment_fee_pct = st.slider("Fee pagamenti/roaming % (su ricavi ricarica)", 0.0, 15.0, 3.0) / 100

with st.sidebar.expander("Extra ricavi (bar/shop)", expanded=True):
    ancillary_margin_per_session = st.number_input("Margine extra per sessione (€)", min_value=0.0, value=0.60, step=0.10)

with st.sidebar.expander("CAPEX (investimento)", expanded=True):
    capex_unit_default = 25000 if tecnologia == "DC 30 kW" else 45000
    capex_unit = st.number_input("CAPEX per unità (€)", min_value=0.0, value=float(capex_unit_default), step=1000.0)
    capex_fixed_site_A = st.number_input("CAPEX fisso sito A (€) (rete/opere/permessi)", min_value=0.0, value=30000.0, step=5000.0)
    capex_fixed_site_B = st.number_input("CAPEX fisso sito B (€) (se multisito)", min_value=0.0, value=25000.0, step=5000.0)
    decommissioning_cost = st.number_input("Costo decommissioning (€)", min_value=0.0, value=0.0, step=5000.0)
    residual_recovery_pct = st.slider("Recovery residuo %", 0, 50, 10) / 100

with st.sidebar.expander("OPEX (costi fissi)", expanded=True):
    maint_per_unit = st.number_input("Maintenance per unità/anno (€)", min_value=0.0, value=2200.0, step=100.0)
    backe
