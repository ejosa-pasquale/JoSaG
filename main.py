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
    backend_per_unit = st.number_input("Backend+SIM+customer care per unità/anno (€)", min_value=0.0, value=900.0, step=100.0)
    lease_per_unit = st.number_input("Canone/royalty sito per unità/anno (€)", min_value=0.0, value=1200.0, step=100.0)
    fixed_opex_site = st.number_input("Costi fissi sito/anno (€)", min_value=0.0, value=2500.0, step=250.0)
    demand_charge_eur_per_kw_year = st.number_input("Demand charges (€/kW-anno)", min_value=0.0, value=0.0, step=5.0)
    contracted_power_factor = st.slider("Fattore potenza contrattuale", 0.3, 1.0, 0.8)
    inflation = st.slider("Inflazione OPEX annua %", 0.0, 8.0, 2.0) / 100

st.sidebar.header("4) Parametri CFO (opzionali)")
with st.sidebar.expander("WACC / Tasse / Ammortamenti / Working capital", expanded=False):
    wacc = st.slider("WACC %", 4.0, 15.0, 8.0) / 100
    tax_rate = st.slider("Tax rate effettivo %", 0.0, 40.0, 28.0) / 100
    depr_years = st.slider("Ammortamento (anni)", 3, 15, 8)
    nwc_pct = st.slider("Working capital (% fatturato)", 0.0, 20.0, 2.0) / 100

st.sidebar.header("5) Soglie semaforo (decisione)")
with st.sidebar.expander("Soglie", expanded=True):
    max_payback_years = st.slider("Payback massimo accettabile (anni)", 2, 12, 6)
    max_lost_sales_pct = st.slider("Vendite perse massime (%) nel 2030", 0, 40, 10)
    min_roi_pct = st.slider("ROI minimo (%) sull'orizzonte", -20, 200, 20)

st.sidebar.header("6) Demand inputs")
if demand_model == "Bottom-up (Traffico sito)":
    with st.sidebar.expander("Traffico → EV → Conversione", expanded=True):
        traffic_veicoli_giorno = st.number_input("Traffico passante (veicoli/giorno)", min_value=0, value=12000, step=500)
        traffic_growth = st.slider("Crescita traffico annua %", -5.0, 10.0, 0.0) / 100
        ingress_rate = st.slider("Ingress rate % (entrano nel sito)", 0.0, 20.0, 3.0) / 100
        ev_share_2026 = st.slider("Quota EV 2026 %", 0.0, 20.0, 2.0) / 100
        ev_share_2030 = st.slider("Quota EV 2030 %", 0.0, 40.0, 10.0) / 100
        ev_share_2035 = st.slider("Quota EV 2035 %", 0.0, 60.0, 18.0) / 100
        conversione_ricarica = st.slider("Conversione a ricarica %", 0.0, 50.0, 8.0) / 100
        competitor_factor = st.slider("Fattore concorrenza (0.5 forte, 1 neutro, 1.3 vantaggio)", 0.50, 1.30, 1.00)
        prezzo_mercato_ref = st.number_input("Prezzo mercato riferimento (€/kWh)", min_value=0.1, value=0.69, step=0.01)
        price_elasticity = st.slider("Elasticità prezzo", 0.0, 2.0, 0.6)
        ramp_start_pct = st.slider("Ramp-up 1° anno %", 10, 100, 60) / 100
        ramp_years_to_full = st.slider("Anni per arrivare al 100%", 0, 3, 1)
else:
    with st.sidebar.expander("Parco BEV → cattura → quota kWh a te", expanded=True):
        bev_target_2030 = st.number_input("BEV città target 2030", min_value=0, value=5000, step=500)
        stress_bev = st.slider("Stress adozione BEV %", 50, 150, 100) / 100
        bev_start_ratio_2026 = st.slider("BEV 2026 come % del 2030", 10, 90, 50) / 100
        bev_growth_2035_vs_2030 = st.slider("BEV 2035 come % del 2030", 100, 200, 130) / 100
        public_share = st.slider("Quota dipendenza ricarica pubblica %", 5, 80, 30) / 100
        capture_2026 = st.slider("Quota cattura 2026 %", 0.1, 10.0, 1.0) / 100
        capture_2030 = st.slider("Quota cattura 2030 %", 0.1, 20.0, 5.0) / 100
        capture_2035 = st.slider("Quota cattura 2035 %", 0.1, 25.0, 6.0) / 100
        stress_cattura = st.slider("Stress efficacia competitiva %", 50, 150, 100) / 100
        km_per_ev_year = st.number_input("Km/anno per EV", min_value=1000, value=12000, step=500)
        kwh_per_100km = st.number_input("Consumo EV (kWh/100km)", min_value=8.0, value=18.0, step=0.5)
        share_kwh_at_station = st.slider("Quota kWh EV erogata da te %", 1, 80, 25) / 100
        ramp_start_pct = st.slider("Ramp-up 1° anno %", 10, 100, 60) / 100
        ramp_years_to_full = st.slider("Anni per arrivare al 100%", 0, 3, 1)

# ============================================================
# Params
# ============================================================
base_params = {
    "years": years,
    "demand_model": demand_model,
    "tecnologia": tecnologia,
    "allocazione": allocazione,

    "ore_max_giorno": float(ore_max_giorno),
    "kwh_per_session": float(kwh_per_session),
    "taper_factor": float(taper_factor),
    "overhead_min": float(overhead_min),
    "uptime": float(uptime),
    "util_target": float(util_target),

    "min_units": int(min_units),
    "max_units_installabili": int(max_units_installabili),
    "potenza_disponibile_kw": float(potenza_disponibile_kw),

    "prezzo_kwh": float(prezzo_kwh),
    "costo_kwh": float(costo_kwh),
    "price_escal": float(price_escal),
    "cost_escal": float(cost_escal),
    "payment_fee_pct": float(payment_fee_pct),
    "ancillary_margin_per_session": float(ancillary_margin_per_session),

    "capex_unit": float(capex_unit),
    "capex_fixed_site_A": float(capex_fixed_site_A),
    "capex_fixed_site_B": float(capex_fixed_site_B),
    "decommissioning_cost": float(decommissioning_cost),
    "residual_recovery_pct": float(residual_recovery_pct),

    "maint_per_unit": float(maint_per_unit),
    "backend_per_unit": float(backend_per_unit),
    "lease_per_unit": float(lease_per_unit),
    "fixed_opex_site": float(fixed_opex_site),
    "demand_charge_eur_per_kw_year": float(demand_charge_eur_per_kw_year),
    "contracted_power_factor": float(contracted_power_factor),
    "inflation": float(inflation),

    "wacc": float(wacc),
    "tax_rate": float(tax_rate),
    "depr_years": int(depr_years),
    "nwc_pct": float(nwc_pct),

    "ramp_start_pct": float(ramp_start_pct),
    "ramp_years_to_full": int(ramp_years_to_full),
}

if demand_model == "Bottom-up (Traffico sito)":
    base_params.update({
        "traffic_veicoli_giorno": int(traffic_veicoli_giorno),
        "traffic_growth": float(traffic_growth),
        "ingress_rate": float(ingress_rate),
        "ev_share_2026": float(ev_share_2026),
        "ev_share_2030": float(ev_share_2030),
        "ev_share_2035": float(ev_share_2035),
        "conversione_ricarica": float(conversione_ricarica),
        "competitor_factor": float(competitor_factor),
        "prezzo_mercato_ref": float(prezzo_mercato_ref),
        "price_elasticity": float(price_elasticity),

        # placeholders top-down
        "bev_target_2030": 0, "stress_bev": 1.0, "bev_start_ratio_2026": 0.5, "bev_growth_2035_vs_2030": 1.3,
        "public_share": 0.3, "capture_2026": 0.01, "capture_2030": 0.05, "capture_2035": 0.06, "stress_cattura": 1.0,
        "km_per_ev_year": 12000, "kwh_per_100km": 18.0, "share_kwh_at_station": 0.25,
    })
else:
    base_params.update({
        "bev_target_2030": int(bev_target_2030),
        "stress_bev": float(stress_bev),
        "bev_start_ratio_2026": float(bev_start_ratio_2026),
        "bev_growth_2035_vs_2030": float(bev_growth_2035_vs_2030),
        "public_share": float(public_share),
        "capture_2026": float(capture_2026),
        "capture_2030": float(capture_2030),
        "capture_2035": float(capture_2035),
        "stress_cattura": float(stress_cattura),
        "km_per_ev_year": float(km_per_ev_year),
        "kwh_per_100km": float(kwh_per_100km),
        "share_kwh_at_station": float(share_kwh_at_station),

        # placeholders bottom-up
        "traffic_veicoli_giorno": 0, "traffic_growth": 0.0, "ingress_rate": 0.03,
        "ev_share_2026": 0.02, "ev_share_2030": 0.10, "ev_share_2035": 0.18,
        "conversione_ricarica": 0.08,
        "competitor_factor": 1.0,
        "prezzo_mercato_ref": float(prezzo_kwh),
        "price_elasticity": 0.0,
    })

# ============================================================
# Run
# ============================================================
out = run_model(base_params)
df = out["df"]

fy = focus_year if focus_year in df.index else int(df.index.max())
r = df.loc[fy]

# ============================================================
# Semaforo decisionale (semplice)
# ============================================================
payback_ok = np.isfinite(out["payback_year"]) and (out["payback_year"] <= fy + max_payback_years - (fy - years[0]))
# per semplicità confronto payback (anno) contro "inizio + max"
payback_ok = np.isfinite(out["payback_year"]) and (out["payback_year"] <= years[0] + max_payback_years)

lost_ok = (r["Vendite Perse %"] <= max_lost_sales_pct) if np.isfinite(r["Vendite Perse %"]) else False
roi_ok = np.isfinite(out["simple_roi"]) and (out["simple_roi"] * 100 >= min_roi_pct)

score = sum([payback_ok, lost_ok, roi_ok])

st.subheader("🚦 Decisione (semaforo)")
if score == 3:
    st.success("VERDE — Caso robusto: rientro accettabile, ROI ok, colli di bottiglia sotto controllo.")
elif score == 2:
    st.warning("GIALLO — Caso borderline: funziona, ma serve ottimizzare (prezzo/costi/capacità) o validare domanda.")
else:
    st.error("ROSSO — Caso debole: o non rientra, o ROI troppo basso, o perdi troppi clienti per vincoli.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Investimento totale (CAPEX)", eur(out["total_capex"]))
c2.metric("Fatturato totale (10 anni)", eur(out["total_revenue"]))
c3.metric("ROI semplice (10 anni)", f"{out['simple_roi']*100:.0f}%" if np.isfinite(out["simple_roi"]) else "n/a")
c4.metric("Payback (anno)", f"{out['payback_year']:.1f}" if np.isfinite(out["payback_year"]) else "n/a")

st.caption("ROI semplice = (cassa operativa cumulata − CAPEX) / CAPEX. È un indicatore 'manager-friendly'. I dettagli CFO sono sotto.")

st.divider()

# ============================================================
# Cruscotto decisionale: 4 grafici chiarissimi
# ============================================================
st.subheader("📊 Cruscotto (capire subito cosa succede)")

# 1) ricariche/giorno + break-even
st.write("### 1) Ricariche/giorno: previste vs soglia (break-even)")
sessions_day_series = (df["Sessioni Servite"] / 365.0).values

# break-even per unità (anno focus)
units_year = max(int(r["Unità Tot"]), 1)
energy_cost_y = r["Energia Servita (kWh)"] * r["Costo energia (€/kWh)"]
payment_fees_y = r["Fatturato ricarica (€)"] * payment_fee_pct
opex_total_y = r["Fatturato Tot (€)"] - energy_cost_y - payment_fees_y - r["EBITDA (€)"]

fixed_per_unit_day = (opex_total_y / units_year) / 365.0
amort_per_unit_day = (capex_unit / max(depr_years, 1)) / 365.0

t_idx = int(fy - years[0])
anc_session = ancillary_margin_per_session * ((1 + inflation) ** t_idx)
margin_per_kwh_net = r["Prezzo (€/kWh)"] - r["Costo energia (€/kWh)"] - r["Prezzo (€/kWh)"] * payment_fee_pct
margin_session = kwh_per_session * margin_per_kwh_net + anc_session

be_sessions_day = (fixed_per_unit_day + amort_per_unit_day) / margin_session if margin_session > 0 else np.nan

fig, ax = plt.subplots()
ax.plot(df.index, sessions_day_series, marker="o", linewidth=3, label="Ricariche/giorno (servite)")
if np.isfinite(be_sessions_day):
    ax.axhline(be_sessions_day, linestyle="--", linewidth=2, label=f"Soglia break-even ≈ {be_sessions_day:.1f}/g")
ax.set_xlabel("Anno")
ax.set_ylabel("Ricariche/giorno")
ax.legend()
st.pyplot(fig)

if np.isfinite(be_sessions_day):
    if (r["Sessioni Servite"]/365.0) >= be_sessions_day:
        st.success(f"Nel {fy} sei SOPRA soglia: {r['Sessioni Servite']/365.0:.1f} vs {be_sessions_day:.1f} ricariche/giorno.")
    else:
        st.warning(f"Nel {fy} sei SOTTO soglia: {r['Sessioni Servite']/365.0:.1f} vs {be_sessions_day:.1f}.")
else:
    st.error("Break-even non calcolabile: margine per sessione ≤ 0 (prezzo troppo basso o costi troppo alti).")

st.divider()

# 2) colli di bottiglia: domanda vs servito + vendite perse %
st.write("### 2) Colli di bottiglia: domanda vs servito (e vendite perse)")
fig, ax = plt.subplots()
ax.plot(df.index, df["Sessioni Domanda"], marker="o", linewidth=3, label="Domanda (sessioni/anno)")
ax.plot(df.index, df["Sessioni Servite"], marker="o", linewidth=3, label="Servite (sessioni/anno)")
ax.fill_between(df.index, df["Sessioni Servite"], df["Sessioni Domanda"],
                where=(df["Sessioni Domanda"] > df["Sessioni Servite"]), alpha=0.2)
ax.set_xlabel("Anno")
ax.set_ylabel("Sessioni/anno")
ax.legend()
st.pyplot(fig)

fig2, ax2 = plt.subplots()
ax2.plot(df.index, df["Vendite Perse %"], marker="o", linewidth=3)
ax2.axhline(max_lost_sales_pct, linestyle="--", linewidth=1)
ax2.set_xlabel("Anno")
ax2.set_ylabel("Vendite perse (%)")
st.pyplot(fig2)

st.info("Se le vendite perse sono alte, il collo di bottiglia è capacità/potenza/ore/uptime. "
        "Riduci code aumentando unità, potenza disponibile, ore operative o migliorando uptime/turnover.")

st.divider()

# 3) Piano unità (quante e quando)
st.write("### 3) Piano installazioni: quante unità servono e quando")
fig, ax = plt.subplots()
ax.bar(df.index, df["Unità A"], label="Sito A")
ax.bar(df.index, df["Unità B"], bottom=df["Unità A"], label="Sito B")
ax.set_xlabel("Anno")
ax.set_ylabel("Unità installate")
ax.legend()
st.pyplot(fig)

st.caption("Se il grafico 'si ferma' ma la domanda cresce, hai un vincolo (max unità o potenza disponibile).")

st.divider()

# 4) Investimento e ritorno (cassa cumulata + fatturato)
st.write("### 4) Investimento e ritorno: quando recupero e quanto fatturo")
cum_annual = out["cum_annual"]
fig, ax = plt.subplots()
ax.plot(df.index, cum_annual, marker="o", linewidth=3)
ax.axhline(0, linewidth=1)
ax.set_xlabel("Anno")
ax.set_ylabel("Cassa cumulata (OPCF - CAPEX) €")
st.pyplot(fig)

if np.isfinite(out["payback_year"]):
    st.success(f"Payback stimato intorno al: **{out['payback_year']:.1f}**.")
else:
    st.warning("Payback non raggiunto nell’orizzonte: serve più domanda/margine o meno CAPEX/OPEX.")

fig, ax = plt.subplots()
ax.plot(df.index, df["Fatturato Tot (€)"], marker="o", linewidth=3)
ax.set_xlabel("Anno")
ax.set_ylabel("Fatturato annuo (€)")
st.pyplot(fig)

st.divider()

# ============================================================
# “Spiegazione in una riga” dei colli di bottiglia
# ============================================================
st.subheader("🔎 Diagnosi colli di bottiglia (spiegata)")
bottlenecks = []
if r["Vendite Perse %"] > max_lost_sales_pct:
    bottlenecks.append("❗ Capacità insufficiente → stai perdendo clienti (aumenta unità/potenza/ore o riduci durata sessione/overhead).")
if potenza_disponibile_kw > 0:
    potenza_kw = 30 if tecnologia == "DC 30 kW" else 60
    max_by_grid = int(np.floor(potenza_disponibile_kw / potenza_kw))
    if max_by_grid < max_units_installabili:
        bottlenecks.append("⚠️ Vincolo rete/potenza disponibile → limita il numero massimo di unità installabili.")
if max_units_installabili <= 2 and r["Sessioni Domanda"] > r["Sessioni Servite"]:
    bottlenecks.append("⚠️ Vincolo spazio/permessi → limita la crescita, valuta multisito o layout diverso.")
if margin_session <= 0:
    bottlenecks.append("❗ Margine per sessione ≤ 0 → il modello non sta in piedi (rivedi prezzo, costo energia, fee, ancillary).")

if not bottlenecks:
    st.success("Nessun collo di bottiglia evidente nel focus: domanda servita e margini coerenti con le soglie impostate.")
else:
    for b in bottlenecks:
        st.write(b)

# ============================================================
# CFO details (facoltativi, in expander)
# ============================================================
with st.expander("🏦 Dettagli CFO (NPV/IRR) — opzionale"):
    colA, colB, colC = st.columns(3)
    colA.metric("NPV (VAN)", eur(out["npv"]))
    colB.metric("IRR (TIR)", pct(out["irr"]))
    colC.metric("Valore residuo (fine)", eur(out["residual_value"]))

# ============================================================
# Tabella (dettaglio)
# ============================================================
with st.expander("📋 Tabella completa (dettaglio)", expanded=False):
    st.dataframe(df.style.format({
        "Sessioni Domanda": "{:,.0f}",
        "Sessioni Servite": "{:,.0f}",
        "Sessioni Perse": "{:,.0f}",
        "Energia Domanda (kWh)": "{:,.0f}",
        "Energia Servita (kWh)": "{:,.0f}",
        "Energia Persa (kWh)": "{:,.0f}",
        "Fatturato ricarica (€)": "{:,.0f}",
        "Fatturato extra (€)": "{:,.0f}",
        "Fatturato Tot (€)": "{:,.0f}",
        "EBITDA (€)": "{:,.0f}",
        "CAPEX (€)": "{:,.0f}",
        "OPCF after WC (€)": "{:,.0f}",
        "Net Cash (€/anno)": "{:,.0f}",
        "Vendite Perse %": "{:.1f}",
        "Utilizzo %": "{:.1f}",
        "Prezzo (€/kWh)": "{:.3f}",
        "Costo energia (€/kWh)": "{:.3f}",
    }))
