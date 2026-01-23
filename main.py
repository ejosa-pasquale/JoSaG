import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy_financial as npf

st.set_page_config(page_title="Executive Charging Suite - DC Charging for Gas Stations - eVFs", layout="wide")

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def safe_pct(x):
    return f"{x*100:.1f}%" if (x is not None and np.isfinite(x)) else "n/a"

def safe_eur(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    s = f"{x:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"€ {s}"

def compute_payback_period(cf_nodes):
    """Simple payback on undiscounted node cashflows."""
    cum = np.cumsum(cf_nodes)
    for i in range(1, len(cum)):
        if cum[i] >= 0 and cum[i - 1] < 0:
            frac = (0 - cum[i - 1]) / (cum[i] - cum[i - 1]) if cum[i] != cum[i - 1] else 0
            return (i - 1) + frac
    return np.nan

def compute_discounted_payback_period(cf_nodes, wacc):
    disc = np.array([cf_nodes[t] / ((1 + wacc) ** t) for t in range(len(cf_nodes))])
    cum = np.cumsum(disc)
    for i in range(1, len(cum)):
        if cum[i] >= 0 and cum[i - 1] < 0:
            frac = (0 - cum[i - 1]) / (cum[i] - cum[i - 1]) if cum[i] != cum[i - 1] else 0
            return (i - 1) + frac
    return np.nan

def piecewise_interp(x, anchors_x, anchors_y):
    return np.interp(x, anchors_x, anchors_y)

def run_model(p):
    """
    Timeline CFO a nodi:
      cf_nodes[0] = -CAPEX inizio 2026
      cf_nodes[1] = OPCF 2026 - CAPEX inizio 2027
      ...
      cf_nodes[n] = OPCF ultimo anno + residual (inizio anno successivo)
    """
    years = p["years"]
    n = len(years)
    t = np.arange(n)

    potenza_kw = 30 if p["tecnologia"] == "DC 30 kW" else 60
    effective_kw = potenza_kw * p["taper_factor"]

    # Tempo sessione (min): energia/potenza effettiva + overhead
    session_minutes = (p["kwh_per_session"] / max(effective_kw, 1e-6)) * 60 + p["overhead_min"]

    # Capacità per unità: sessioni/anno con uptime e target utilizzo
    sessions_per_unit_day = (p["ore_max_giorno"] * 60) / max(session_minutes, 1e-6)
    cap_sessions_unit_year = sessions_per_unit_day * 365 * p["uptime"] * p["util_target"]

    # Ramp-up (marketing/apertura)
    if p["ramp_years_to_full"] <= 0:
        ramp = np.ones(n)
    else:
        ramp = np.minimum(1.0, p["ramp_start_pct"] + (1 - p["ramp_start_pct"]) * (t / p["ramp_years_to_full"]))

    # Prezzi/costi indicizzati
    price_kwh = p["prezzo_kwh"] * ((1 + p["price_escal"]) ** t)
    cost_kwh = p["costo_kwh"] * ((1 + p["cost_escal"]) ** t)
    inflation = (1 + p["inflation"]) ** t

    # -------------------------
    # Demand model
    # -------------------------
    if p["demand_model"] == "Bottom-up (Traffico sito)":
        traffic_day = p["traffic_veicoli_giorno"] * ((1 + p["traffic_growth"]) ** t)

        ev_share = piecewise_interp(
            years,
            [years[0], 2030, years[-1]],
            [p["ev_share_2026"], p["ev_share_2030"], p["ev_share_2035"]],
        )
        ev_share = np.clip(ev_share, 0, 1)

        # Conversione aggiustata da concorrenza e prezzo relativo
        conv_adj = p["conversione_ricarica"] * p["competitor_factor"] * (
            (p["prezzo_mercato_ref"] / np.maximum(price_kwh, 1e-6)) ** p["price_elasticity"]
        )
        conv_adj = np.clip(conv_adj, 0, 1)

        sessions_demand = traffic_day * 365 * p["ingress_rate"] * ev_share * conv_adj * ramp
        energy_demand = sessions_demand * p["kwh_per_session"]

    else:
        # Top-down: separo consumo EV e quota kWh catturata dalla stazione
        bev_2030 = p["bev_target_2030"] * p["stress_bev"]
        bev_2026 = bev_2030 * p["bev_start_ratio_2026"]
        bev_2035 = bev_2030 * p["bev_growth_2035_vs_2030"]

        bev_city = piecewise_interp(years, [years[0], 2030, years[-1]], [bev_2026, bev_2030, bev_2035])
        bev_public = bev_city * p["public_share"]

        capture = piecewise_interp(
            years,
            [years[0], 2030, years[-1]],
            [p["capture_2026"], p["capture_2030"], p["capture_2035"]],
        )
        capture = np.clip(capture * p["stress_cattura"], 0, 1)
        bev_captured = bev_public * capture

        energy_per_ev_year = p["km_per_ev_year"] * (p["kwh_per_100km"] / 100.0)
        energy_demand = bev_captured * energy_per_ev_year * p["share_kwh_at_station"] * ramp
        sessions_demand = energy_demand / max(p["kwh_per_session"], 1e-6)

    # -------------------------
    # Sizing (unità)
    # -------------------------
    req_units = np.ceil(sessions_demand / max(cap_sessions_unit_year, 1e-6)).astype(int)

    max_units_grid = p["max_units_installabili"]
    if p["potenza_disponibile_kw"] > 0:
        max_units_grid = min(max_units_grid, int(np.floor(p["potenza_disponibile_kw"] / potenza_kw)))

    units = np.clip(req_units, p["min_units"], max_units_grid)

    # allocazione A/B (regola semplice)
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

    utilization = np.where(cap_sessions > 0, served_sessions / cap_sessions, 0.0)

    # -------------------------
    # Economics
    # -------------------------
    rev_charging = served_energy * price_kwh
    anc_per_session = p["ancillary_margin_per_session"] * inflation
    rev_ancillary = served_sessions * anc_per_session
    revenue_total = rev_charging + rev_ancillary

    energy_cost = served_energy * cost_kwh
    payment_fees = rev_charging * p["payment_fee_pct"]

    opex_unit_year = (p["maint_per_unit"] + p["backend_per_unit"] + p["lease_per_unit"]) * inflation
    fixed_opex_site = p["fixed_opex_site"] * inflation

    contracted_kw = units * potenza_kw * p["contracted_power_factor"]
    demand_charges = contracted_kw * p["demand_charge_eur_per_kw_year"] * inflation

    opex_total = units * opex_unit_year + (units > 0).astype(int) * fixed_opex_site + demand_charges

    ebitda = revenue_total - energy_cost - payment_fees - opex_total

    # -------------------------
    # CAPEX (inizio anno)
    # -------------------------
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

    # -------------------------
    # Depreciation + tax + WC
    # -------------------------
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

    # Residual value (approssimato su NBV residuo)
    nbv_end = 0.0
    for i in range(n):
        years_elapsed = n - i
        remaining = max(0, dep_life - years_elapsed)
        nbv_end += capex[i] * (remaining / dep_life)

    residual_value = nbv_end * p["residual_recovery_pct"] - p["decommissioning_cost"]

    # Cashflow nodes CFO
    cf_nodes = np.zeros(n + 1)
    cf_nodes[0] = -capex[0]
    for i in range(1, n):
        cf_nodes[i] = op_cf_after_wc[i - 1] - capex[i]
    cf_nodes[n] = op_cf_after_wc[n - 1] + residual_value

    npv = npf.npv(p["wacc"], cf_nodes)
    irr = npf.irr(cf_nodes)
    mirr = npf.mirr(cf_nodes, p["wacc"], p["wacc"])
    payback = compute_payback_period(cf_nodes)
    dpayback = compute_discounted_payback_period(cf_nodes, p["wacc"])

    df = pd.DataFrame({
        "Anno": years,
        "Unità Tot": units,
        "Unità A": units_A,
        "Unità B": units_B,
        "Sessioni Domanda": np.round(sessions_demand, 0),
        "Sessioni Servite": np.round(served_sessions, 0),
        "Sessioni Perse": np.round(lost_sessions, 0),
        "Energia Domanda (kWh)": np.round(energy_demand, 0),
        "Energia Servita (kWh)": np.round(served_energy, 0),
        "Energia Persa (kWh)": np.round(lost_energy, 0),
        "Utilizzo (%)": np.round(utilization * 100, 1),
        "Prezzo (€/kWh)": np.round(price_kwh, 3),
        "Costo energia (€/kWh)": np.round(cost_kwh, 3),
        "Ricavi ricarica (€)": np.round(rev_charging, 0),
        "Ricavi ancillary (€)": np.round(rev_ancillary, 0),
        "Ricavi tot (€)": np.round(revenue_total, 0),
        "EBITDA (€)": np.round(ebitda, 0),
        "Ammortamenti (€)": np.round(depreciation, 0),
        "EBIT (€)": np.round(ebit, 0),
        "Tasse (€)": np.round(taxes, 0),
        "OPCF after WC (€)": np.round(op_cf_after_wc, 0),
        "CAPEX (inizio anno) (€)": np.round(capex, 0),
    }).set_index("Anno")

    return {
        "years": years,
        "potenza_kw": potenza_kw,
        "effective_kw": effective_kw,
        "session_minutes": float(session_minutes),
        "df": df,
        "cf_nodes": cf_nodes,
        "npv": float(npv) if np.isfinite(npv) else np.nan,
        "irr": float(irr) if (irr is not None and np.isfinite(irr)) else np.nan,
        "mirr": float(mirr) if (mirr is not None and np.isfinite(mirr)) else np.nan,
        "payback": float(payback) if np.isfinite(payback) else np.nan,
        "dpayback": float(dpayback) if np.isfinite(dpayback) else np.nan,
        "residual_value": float(residual_value),
    }

def npv_from_overrides(base_params, overrides):
    p2 = dict(base_params)
    p2.update(overrides)
    return run_model(p2)["npv"]

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.title("🛡️ Executive Support System — Charging Investment eVFs")
st.markdown("### domanda → capacità → margini → ritorno investimento *")

years = np.arange(2026, 2036)  # 10 anni: 2026-2035
focus_year = 2030

st.sidebar.header("🧭 Scelta Modello Domanda")
demand_model = st.sidebar.radio(
    "Approccio",
    ["Bottom-up (Traffico sito)", "Top-down (Parco BEV città)"],
    index=0
)

st.sidebar.header("⚙️ Tecnica & Operatività")
with st.sidebar.expander("Asset & qualità servizio", expanded=True):
    tecnologia = st.selectbox("Tecnologia", ["DC 30 kW", "DC 60 kW"], index=1)
    allocazione = st.radio("Strategia location", ["Monosito (Tutto in A)", "Multisito (Espansione in B)"], index=0)

    ore_max_giorno = st.slider("Ore operative al giorno", 4, 24, 12)
    kwh_per_session = st.number_input("kWh medi per sessione", min_value=5.0, max_value=200.0, value=35.0, step=1.0)

    taper_factor = st.slider("Potenza effettiva (tapering) %", 50, 100, 75) / 100
    overhead_min = st.slider("Overhead per sessione (min) (plug/unplug/parcheggio)", 0, 15, 4)

    uptime = st.slider("Uptime tecnico %", 85, 100, 97) / 100
    util_target = st.slider("Target utilizzo % (per evitare code)", 50, 95, 80) / 100

    min_units = st.slider("Min unità installate (0 = possibile non installare)", 0, 6, 1)
    max_units_installabili = st.slider("Max unità installabili (vincolo piazzale)", 1, 20, 6)
    potenza_disponibile_kw = st.number_input("Potenza disponibile (kW) (0 = ignora vincolo)", min_value=0.0, value=0.0, step=10.0)

st.sidebar.header("📈 Domanda (driver)")
if demand_model == "Bottom-up (Traffico sito)":
    with st.sidebar.expander("Traffico → Ingressi → EV → Conversione", expanded=True):
        traffic_veicoli_giorno = st.number_input("Traffico passante (veicoli/giorno)", min_value=0, value=12000, step=500)
        traffic_growth = st.slider("Crescita traffico annua %", -5.0, 10.0, 0.0) / 100

        ingress_rate = st.slider("Ingress rate % (quanti entrano nel sito)", 0.0, 20.0, 3.0) / 100

        ev_share_2026 = st.slider("Quota EV nel traffico 2026 %", 0.0, 20.0, 2.0) / 100
        ev_share_2030 = st.slider("Quota EV nel traffico 2030 %", 0.0, 40.0, 10.0) / 100
        ev_share_2035 = st.slider("Quota EV nel traffico 2035 %", 0.0, 60.0, 18.0) / 100

        conversione_ricarica = st.slider("Conversione a ricarica % (EV entrati che ricaricano)", 0.0, 50.0, 8.0) / 100

        competitor_factor = st.slider("Fattore concorrenza (0.5 forte, 1 neutro, 1.3 vantaggio)", 0.50, 1.30, 1.00)
        prezzo_mercato_ref = st.number_input("Prezzo mercato riferimento (€/kWh)", min_value=0.1, value=0.69, step=0.01)
        price_elasticity = st.slider("Elasticità prezzo (0=nessun effetto, 1=moderato)", 0.0, 2.0, 0.6)

        ramp_start_pct = st.slider("Ramp-up domanda 1° anno %", 10, 100, 60) / 100
        ramp_years_to_full = st.slider("Anni per arrivare al 100% (ramp-up)", 0, 3, 1)

else:
    with st.sidebar.expander("Parco BEV → Pubblica → Cattura → Quota kWh a te", expanded=True):
        bev_target_2030 = st.number_input("BEV città target 2030", min_value=0, value=5000, step=500)
        stress_bev = st.slider("Stress adozione BEV %", 50, 150, 100) / 100
        bev_start_ratio_2026 = st.slider("BEV 2026 come % del 2030", 10, 90, 50) / 100
        bev_growth_2035_vs_2030 = st.slider("BEV 2035 come % del 2030", 100, 200, 130) / 100

        public_share = st.slider("Quota dipendenza ricarica pubblica %", 5, 80, 30) / 100

        capture_2026 = st.slider("Quota cattura 2026 %", 0.1, 10.0, 1.0) / 100
        capture_2030 = st.slider("Quota cattura 2030 %", 0.1, 20.0, 5.0) / 100
        capture_2035 = st.slider("Quota cattura 2035 %", 0.1, 25.0, 6.0) / 100
        stress_cattura = st.slider("Stress efficacia competitiva %", 50, 150, 100) / 100

        km_per_ev_year = st.number_input("Km/anno per EV (media)", min_value=1000, value=12000, step=500)
        kwh_per_100km = st.number_input("Consumo EV (kWh/100km)", min_value=8.0, value=18.0, step=0.5)
        share_kwh_at_station = st.slider("Quota dei kWh EV catturati erogata da te %", 1, 80, 25) / 100

        ramp_start_pct = st.slider("Ramp-up domanda 1° anno %", 10, 100, 60) / 100
        ramp_years_to_full = st.slider("Anni per arrivare al 100% (ramp-up)", 0, 3, 1)

st.sidebar.header("💰 Pricing, Costi, CAPEX/OPEX")
with st.sidebar.expander("Ricavi & costi variabili", expanded=True):
    prezzo_kwh = st.number_input("Prezzo vendita (€/kWh)", min_value=0.10, value=0.69, step=0.01)
    costo_kwh = st.number_input("Costo energia (€/kWh)", min_value=0.01, value=0.30, step=0.01)

    price_escal = st.slider("Crescita prezzo annua %", -10.0, 10.0, 0.0) / 100
    cost_escal = st.slider("Crescita costo energia annua %", -10.0, 15.0, 0.0) / 100

    payment_fee_pct = st.slider("Fee pagamenti/roaming (% su ricavi ricarica)", 0.0, 15.0, 3.0) / 100
    ancillary_margin_per_session = st.number_input("Margine ancillary per sessione (€) (bar/shop)", min_value=0.0, value=0.60, step=0.10)

with st.sidebar.expander("CAPEX all-in", expanded=True):
    capex_unit_default = 25000 if tecnologia == "DC 30 kW" else 45000
    capex_unit = st.number_input("CAPEX per unità (€/colonnina) (hardware+install)", min_value=0.0, value=float(capex_unit_default), step=1000.0)

    capex_fixed_site_A = st.number_input("CAPEX fisso sito A (rete/opere/permessi)", min_value=0.0, value=30000.0, step=5000.0)
    capex_fixed_site_B = st.number_input("CAPEX fisso sito B (se multisito)", min_value=0.0, value=25000.0, step=5000.0)

    decommissioning_cost = st.number_input("Costo decommissioning fine orizzonte (€)", min_value=0.0, value=0.0, step=5000.0)
    residual_recovery_pct = st.slider("Recovery su NBV residuo %", 0, 50, 10) / 100

with st.sidebar.expander("OPEX completo", expanded=True):
    maint_per_unit = st.number_input("Maintenance per unità/anno (€)", min_value=0.0, value=2200.0, step=100.0)
    backend_per_unit = st.number_input("Backend+SIM+customer care per unità/anno (€)", min_value=0.0, value=900.0, step=100.0)
    lease_per_unit = st.number_input("Canone/royalty sito per unità/anno (€)", min_value=0.0, value=1200.0, step=100.0)
    fixed_opex_site = st.number_input("Costi fissi sito/anno (€) (assicurazioni/pulizia)", min_value=0.0, value=2500.0, step=250.0)

    demand_charge_eur_per_kw_year = st.number_input("Demand charges (€/kW-anno) (0 se non applicabile)", min_value=0.0, value=0.0, step=5.0)
    contracted_power_factor = st.slider("Fattore potenza contrattuale (vs nominale)", 0.3, 1.0, 0.8)

    inflation = st.slider("Inflazione OPEX/ancillary annua %", 0.0, 8.0, 2.0) / 100

st.sidebar.header("🏦 CFO Settings")
with st.sidebar.expander("WACC, Tasse, Ammortamenti, Working Capital", expanded=True):
    wacc = st.slider("WACC %", 4.0, 15.0, 8.0) / 100
    tax_rate = st.slider("Tax rate effettivo %", 0.0, 40.0, 28.0) / 100
    depr_years = st.slider("Vita utile ammortamento (anni)", 3, 15, 8)
    nwc_pct = st.slider("Working capital (% ricavi totali)", 0.0, 20.0, 2.0) / 100

# ------------------------------------------------------------
# Params dict
# ------------------------------------------------------------
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
        "traffic_veicoli_giorno": 0,
        "traffic_growth": 0.0,
        "ingress_rate": 0.03,
        "ev_share_2026": 0.02,
        "ev_share_2030": 0.10,
        "ev_share_2035": 0.18,
        "conversione_ricarica": 0.08,
        "competitor_factor": 1.0,
        "prezzo_mercato_ref": base_params["prezzo_kwh"],
        "price_elasticity": 0.0,
    })

# ------------------------------------------------------------
# Compute
# ------------------------------------------------------------
out = run_model(base_params)
df = out["df"]
cf_nodes = out["cf_nodes"]

# ------------------------------------------------------------
# Executive summary (super chiaro)
# ------------------------------------------------------------
st.subheader("📌 Executive Summary (chiaro per tutti)")
k1, k2, k3, k4, k5 = st.columns(5)

k1.metric("NPV (VAN)", safe_eur(out["npv"]))
k2.metric("Payback (anni)", f"{out['payback']:.1f}" if np.isfinite(out["payback"]) else "n/a")
k3.metric("Unità nel 2030", f"{int(df.loc[focus_year, 'Unità Tot'])}" if focus_year in df.index else "n/a")
k4.metric("Utilizzo nel 2030", f"{df.loc[focus_year, 'Utilizzo (%)']:.1f}%" if focus_year in df.index else "n/a")
k5.metric("Energia persa 2030", f"{int(df.loc[focus_year, 'Energia Persa (kWh)']):,} kWh".replace(",", ".") if focus_year in df.index else "n/a")

st.caption(
    "Interpretazione rapida: **NPV > 0** e **payback ragionevole** → caso più sano. "
    "**Energia persa** alta → stai perdendo vendite (vincoli di capacità o potenza)."
)

st.divider()

# ------------------------------------------------------------
# Vista Rapida 2x2 (come il tuo layout originale)
# ------------------------------------------------------------
st.subheader("📈 Vista Rapida (4 grafici — layout originale)")

# Pre-calcoli coerenti per margini/opex
energy_cost_series = df["Energia Servita (kWh)"] * df["Costo energia (€/kWh)"]
payment_fees_series = df["Ricavi ricarica (€)"] * payment_fee_pct
opex_series = df["Ricavi tot (€)"] - energy_cost_series - payment_fees_series - df["EBITDA (€)"]

year_sel = st.selectbox(
    "Anno focus (per Break-even e margini)",
    df.index.tolist(),
    index=df.index.tolist().index(focus_year) if focus_year in df.index else len(df.index.tolist()) - 1
)
row = df.loc[year_sel]
units_year = max(int(row["Unità Tot"]), 1)

price = float(row["Prezzo (€/kWh)"])
cost = float(row["Costo energia (€/kWh)"])
t_idx = int(year_sel - years[0])

anc_session = ancillary_margin_per_session * ((1 + inflation) ** t_idx)
contrib_session = (
    kwh_per_session * (price - cost)
    - (kwh_per_session * price * payment_fee_pct)
    + anc_session
)

energy_cost_y = float(row["Energia Servita (kWh)"]) * cost
payment_fees_y = float(row["Ricavi ricarica (€)"]) * payment_fee_pct
opex_total_y = float(row["Ricavi tot (€)"]) - energy_cost_y - payment_fees_y - float(row["EBITDA (€)"])

fixed_per_unit_day = (opex_total_y / units_year) / 365.0
amort_per_unit_day = (capex_unit / max(depr_years, 1)) / 365.0
be_sessions_day = (fixed_per_unit_day + amort_per_unit_day) / contrib_session if contrib_session > 0 else np.nan

c1, c2 = st.columns(2)

with c1:
    st.write("**1) Break-even: ricariche/giorno per unità**")
    st.caption("Soglia pratica: sopra questa linea l’unità “sta in piedi” (vista semplice).")

    max_sessions_day = max(20, int(np.ceil((ore_max_giorno * 60) / max(out["session_minutes"], 1e-6))))
    sessions_range = np.arange(0, max_sessions_day + 1)
    profit_day = sessions_range * contrib_session - (fixed_per_unit_day + amort_per_unit_day)

    fig, ax = plt.subplots()
    ax.plot(sessions_range, profit_day, linewidth=3)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Ricariche/giorno per unità")
    ax.set_ylabel("Margine giornaliero per unità (€)")
    st.pyplot(fig)

    if np.isfinite(be_sessions_day):
        st.markdown(
            f"**Break-even stimato ({year_sel})**: **{be_sessions_day:.1f}** ricariche/giorno per unità."
        )
    else:
        st.warning("Break-even non calcolabile: margine per sessione ≤ 0 (controlla prezzo/costo/fee).")

with c2:
    st.write("**2) Cash Flow cumulato: recupero investimento**")
    st.caption("Sopra lo zero = investimento recuperato (convenzione CFO su nodi temporali).")

    fig, ax = plt.subplots()
    cum = np.cumsum(cf_nodes)
    ax.plot(np.arange(len(cf_nodes)), cum, marker="o", linewidth=3)
    ax.axhline(0, linewidth=1)
    ax.set_xlabel("Tempo (t0=in.2026 ... t10=in.2036)")
    ax.set_ylabel("€ cumulati")
    st.pyplot(fig)

c3, c4 = st.columns(2)

with c3:
    st.write("**3) Allocazione asset per location (A/B)**")
    st.caption("Serve per capire crescita e quando ha senso un secondo sito (B).")

    fig, ax = plt.subplots()
    ax.bar(df.index, df["Unità A"], label="Sito A")
    ax.bar(df.index, df["Unità B"], bottom=df["Unità A"], label="Sito B")
    ax.set_ylabel("Numero unità")
    ax.legend()
    st.pyplot(fig)

with c4:
    st.write("**4) Struttura margini (totale orizzonte)**")
    st.caption("Dove vanno i soldi: ricavi, energia, fee, opex, capex, margine operativo (EBITDA).")

    vals = {
        "Ricavi Tot": df["Ricavi tot (€)"].sum(),
        "Costi Energia": energy_cost_series.sum(),
        "Fee Pagamenti": payment_fees_series.sum(),
        "OPEX": opex_series.sum(),
        "CAPEX": df["CAPEX (inizio anno) (€)"].sum(),
        "EBITDA": df["EBITDA (€)"].sum(),
    }

    fig, ax = plt.subplots()
    ax.bar(list(vals.keys()), list(vals.values()))
    ax.set_ylabel("€ (somma orizzonte)")
    ax.tick_params(axis="x", rotation=20)
    st.pyplot(fig)

# ------------------------------------------------------------
# Operatività: domanda vs servito (in expander, chiaro)
# ------------------------------------------------------------
with st.expander("➕ Operatività: Domanda vs Servito (per capire code / vendite perse)"):
    fig, ax = plt.subplots()
    ax.plot(df.index, df["Energia Domanda (kWh)"], marker="o", label="Domanda (kWh)")
    ax.plot(df.index, df["Energia Servita (kWh)"], marker="o", label="Servita (kWh)")
    ax.set_xlabel("Anno")
    ax.set_ylabel("kWh")
    ax.legend()
    st.pyplot(fig)

    st.caption(
        "Se ‘Servita’ resta sotto ‘Domanda’, significa che stai perdendo sessioni per vincoli: "
        "max unità, potenza disponibile, ore/giorno, uptime o target utilizzo."
    )

# ------------------------------------------------------------
# CFO details + Tornado (nascosti)
# ------------------------------------------------------------
with st.expander("🏦 Dettagli CFO (FCF after-tax, Discounted Payback, Sensitivity Tornado)"):
    colA, colB, colC, colD = st.columns(4)
    colA.metric("NPV (VAN)", safe_eur(out["npv"]))
    colB.metric("IRR (TIR)", safe_pct(out["irr"]))
    colC.metric("MIRR", safe_pct(out["mirr"]))
    colD.metric("Disc. Payback", f"{out['dpayback']:.1f} anni" if np.isfinite(out["dpayback"]) else "n/a")

    st.write("**Tornado Sensitivity — quali driver muovono di più il VAN**")
    base_npv = out["npv"]
    scenarios = [
        ("Traffico veicoli/giorno", {"traffic_veicoli_giorno": int(base_params.get("traffic_veicoli_giorno", 0) * 0.8)},
                                   {"traffic_veicoli_giorno": int(base_params.get("traffic_veicoli_giorno", 0) * 1.2)}),
        ("Conversione a ricarica", {"conversione_ricarica": base_params.get("conversione_ricarica", 0.08) * 0.8},
                                   {"conversione_ricarica": base_params.get("conversione_ricarica", 0.08) * 1.2}),
        ("Prezzo vendita €/kWh", {"prezzo_kwh": base_params["prezzo_kwh"] * 0.9}, {"prezzo_kwh": base_params["prezzo_kwh"] * 1.1}),
        ("Costo energia €/kWh", {"costo_kwh": base_params["costo_kwh"] * 0.9}, {"costo_kwh": base_params["costo_kwh"] * 1.1}),
        ("CAPEX unità", {"capex_unit": base_params["capex_unit"] * 0.8}, {"capex_unit": base_params["capex_unit"] * 1.2}),
        ("Uptime", {"uptime": max(0.0, base_params["uptime"] - 0.02)}, {"uptime": min(1.0, base_params["uptime"] + 0.02)}),
    ]

    rows = []
    for name, low_override, high_override in scenarios:
        if name == "Traffico veicoli/giorno" and base_params["demand_model"] != "Bottom-up (Traffico sito)":
            continue
        npv_low = npv_from_overrides(base_params, low_override)
        npv_high = npv_from_overrides(base_params, high_override)
        rows.append({"Driver": name, "NPV Low": npv_low, "NPV Base": base_npv, "NPV High": npv_high, "Delta": npv_high - npv_low})

    sdf = pd.DataFrame(rows).sort_values("Delta", ascending=True)

    fig, ax = plt.subplots()
    y = np.arange(len(sdf))
    ax.hlines(y, sdf["NPV Low"].values, sdf["NPV High"].values, linewidth=6)
    ax.plot(sdf["NPV Base"].values, y, "o")
    ax.set_yticks(y)
    ax.set_yticklabels(sdf["Driver"].values)
    ax.set_xlabel("NPV (VAN) €")
    ax.axvline(base_npv, linewidth=1)
    st.pyplot(fig)

    st.dataframe(sdf.style.format({
        "NPV Low": "{:,.0f}",
        "NPV Base": "{:,.0f}",
        "NPV High": "{:,.0f}",
        "Delta": "{:,.0f}",
    }))

# ------------------------------------------------------------
# Report table (dettaglio)
# ------------------------------------------------------------
st.divider()
st.subheader("📊 Report Analitico (10 anni)")

st.dataframe(df.style.format({
    "Sessioni Domanda": "{:,.0f}",
    "Sessioni Servite": "{:,.0f}",
    "Sessioni Perse": "{:,.0f}",
    "Energia Domanda (kWh)": "{:,.0f}",
    "Energia Servita (kWh)": "{:,.0f}",
    "Energia Persa (kWh)": "{:,.0f}",
    "Ricavi ricarica (€)": "{:,.0f}",
    "Ricavi ancillary (€)": "{:,.0f}",
    "Ricavi tot (€)": "{:,.0f}",
    "EBITDA (€)": "{:,.0f}",
    "Ammortamenti (€)": "{:,.0f}",
    "EBIT (€)": "{:,.0f}",
    "Tasse (€)": "{:,.0f}",
    "OPCF after WC (€)": "{:,.0f}",
    "CAPEX (inizio anno) (€)": "{:,.0f}",
    "Prezzo (€/kWh)": "{:.3f}",
    "Costo energia (€/kWh)": "{:.3f}",
    "Utilizzo (%)": "{:.1f}",
}))

with st.expander("📌 Note rapide (per GM)"):
    st.markdown(
        f"""
- **Durata sessione stimata**: ~ **{out["session_minutes"]:.0f} min** (include tapering + overhead).
- **Utilizzo target**: {util_target*100:.0f}% → serve a evitare code (più alto = più ricavi ma più rischio congestione).
- **Energia persa**: è domanda che non riesci a servire (vincoli: max unità, potenza disponibile, uptime, ore/giorno).
- Se vuoi un “semaforo decisionale” (verde/giallo/rosso) posso aggiungerlo con 3 regole semplici.
        """
    )
