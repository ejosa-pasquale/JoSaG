import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy_financial as npf
import io
import requests

st.set_page_config(page_title="Executive Charging Suite — Sicilia - eVFSs", layout="wide")

# ============================================================
# DATI BEV SICILIA (2015–2024) — SORGENTE: input utente
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

def forecast_bev_2030(df_hist: pd.DataFrame, province: str, method: str = "CAGR 2021–2024"):
    """
    Restituisce (bev_2024, bev_2030_forecast, cagr_used).
    Metodo esplicito per trasparenza.
    """
    s = df_hist[df_hist["Provincia"] == province].sort_values("Anno")
    bev_2024 = int(s[s["Anno"] == 2024]["Elettrico"].iloc[0]) if (s["Anno"] == 2024).any() else int(s["Elettrico"].iloc[-1])

    if method == "CAGR 2021–2024" and (s["Anno"] == 2021).any() and (s["Anno"] == 2024).any():
        v0 = float(s[s["Anno"] == 2021]["Elettrico"].iloc[0])
        v1 = float(s[s["Anno"] == 2024]["Elettrico"].iloc[0])
        years = 3
        cagr = (v1 / max(v0, 1e-9)) ** (1 / years) - 1
    elif method == "CAGR 2015–2024" and (s["Anno"] == 2015).any() and (s["Anno"] == 2024).any():
        v0 = float(s[s["Anno"] == 2015]["Elettrico"].iloc[0])
        v1 = float(s[s["Anno"] == 2024]["Elettrico"].iloc[0])
        years = 9
        cagr = (v1 / max(v0, 1e-9)) ** (1 / years) - 1
    else:
        tail = s.tail(3)
        v0 = float(tail["Elettrico"].iloc[0])
        v1 = float(tail["Elettrico"].iloc[-1])
        years = int(tail["Anno"].iloc[-1] - tail["Anno"].iloc[0])
        cagr = (v1 / max(v0, 1e-9)) ** (1 / max(years, 1)) - 1

    bev_2030 = int(round(bev_2024 * ((1 + cagr) ** 6)))  # 2024 -> 2030
    return bev_2024, max(bev_2030, bev_2024), cagr

# ============================================================
# HEADER
# ============================================================
st.title("🛡️ Executive Planning Tool — Fast DC - Service Stations")
st.markdown("### eV Field Service")

# ============================================================
# SIDEBAR — DATI TERRITORIO + FUNNEL (ORA ENERGY-DRIVEN)
# ============================================================
df_bev = pd.read_csv(io.StringIO(BEV_RAW), sep="\t")

st.sidebar.header("🗺️ Territorio (Sicilia) — dati BEV 2015–2024")
prov_list = sorted(df_bev["Provincia"].unique())
province = st.sidebar.selectbox("Provincia (default Palermo)", prov_list, index=prov_list.index("PALERMO") if "PALERMO" in prov_list else 0)
bev_forecast_method = st.sidebar.selectbox("Metodo forecast BEV 2030 (trasparente)", ["CAGR 2021–2024", "CAGR 2015–2024"], index=0)

bev_2024, bev_2030_auto, cagr_used = forecast_bev_2030(df_bev, province, bev_forecast_method)

with st.sidebar.expander("📈 Anteprima dato + forecast", expanded=False):
    st.write(f"**{province}** — BEV 2024: **{bev_2024:,}**".replace(",", "."))
    st.write(f"Forecast 2030 ({bev_forecast_method}): **{bev_2030_auto:,}** (CAGR ~ {cagr_used*100:.1f}%)".replace(",", "."))

st.sidebar.header("🕹️ Domanda (Energy-led)")

with st.sidebar.expander("🌍 Scenario Parco BEV", expanded=True):
    use_auto_bev_2030 = st.checkbox("Usa forecast BEV 2030 dal dataset Sicilia", value=True)
    bev_base_2030 = st.number_input(
        f"Target BEV {province} 2030 (Scenario Base)",
        value=int(bev_2030_auto) if use_auto_bev_2030 else 5000,
        min_value=0
    )
    stress_bev = st.slider("Stress Test Adozione BEV (%)", 50, 150, 100) / 100

with st.sidebar.expander("⚡ Energia da caricare (dal parco circolante)", expanded=True):
    km_annui_per_bev = st.number_input("Km/anno per BEV (media)", value=12000, step=500, min_value=1000)
    kwh_100km = st.number_input("Consumo medio (kWh/100km)", value=18.0, step=0.5, min_value=8.0)
    public_share = st.slider(
        "Quota energia ricaricata su pubblico (%)",
        5, 80, 30,
        help="Quota di energia che passa da infrastruttura pubblica (chi non ha wallbox / ricarica condominiale, ecc.)."
    ) / 100

with st.sidebar.expander("🎯 Strategia di cattura (quota su energia pubblica)", expanded=True):
    target_cattura_2030 = st.slider("Quota cattura target 2030 (%)", 0.5, 25.0, 5.0) / 100
    stress_cattura = st.slider("Efficacia competitiva (%)", 50, 150, 100) / 100

# ============================================================
# SIDEBAR — MODULI 30 kW + OPERATIVITÀ + FINANZA
# ============================================================
st.sidebar.header("⚙️ Moduli 30 kW, Operatività e Finanza")

with st.sidebar.expander("🔧 Moduli 30 kW (ottimizzazione saturazione)", expanded=True):
    module_kw = 30
    efficienza = st.slider("Efficienza complessiva (perdite, conversione) (%)", 85, 100, 95) / 100
    uptime = st.slider("Uptime tecnico (%)", 85, 100, 97) / 100
    target_saturazione = st.slider(
        "Target utilizzo massimo (anti-coda) (%)",
        30, 95, 80,
        help="Se il carico previsto supera questo target, aggiungiamo moduli per evitare code e vendite perse."
    ) / 100

    kwh_per_sessione = st.number_input("kWh medi per sessione", value=35.0, step=1.0, min_value=5.0)
    ore_max_giorno = st.slider("Disponibilità operativa (ore/giorno)", 4, 24, 10)

with st.sidebar.expander("💰 Pricing & Costi", expanded=True):
    prezzo_kwh = st.number_input("Prezzo vendita (€/kWh)", value=0.69, step=0.01)
    costo_kwh = st.number_input("Costo energia (€/kWh)", value=0.30, step=0.01)

    fee_roaming = st.slider("Fee roaming/acquiring (% su ricavi ricarica) [CFO]", 0.0, 15.0, 0.0) / 100
    canone_potenza = st.number_input("Canone potenza / demand charges (€/kW-anno) [CFO]", value=0.0, step=5.0)

    capex_modulo = st.number_input("CAPEX per modulo 30 kW (€)", value=25000.0, step=1000.0)
    opex_modulo = st.number_input("OPEX per modulo/anno (€)", value=2000.0, step=100.0)
    opex_sito_fisso = st.number_input("OPEX fisso sito/anno (€)", value=0.0, step=500.0)
    ammortamento_anni = st.slider("Ammortamento gestionale (anni) (break-even)", 3, 12, 5)

with st.sidebar.expander("🏦 Modalità CFO (tasse, WC, scenari, tornado)", expanded=False):
    cfo_mode = st.checkbox("Attiva modalità CFO", value=False)
    wacc = st.slider("WACC (%)", 4, 12, 8) / 100
    tax_rate = st.slider("Tax rate effettivo (%)", 0, 40, 28) / 100
    wc_pct = st.slider("Working capital (% ricavi totali)", 0.0, 20.0, 2.0) / 100
    scenario_view = st.selectbox("Scenario da visualizzare", ["Base", "Bear", "Bull"], index=0)

with st.sidebar.expander("🧭 Orizzonte e allocazione moduli", expanded=False):
    start_year = st.number_input("Anno inizio piano", value=2026, min_value=2024, max_value=2035)
    end_year = st.number_input("Anno fine piano", value=2030, min_value=int(start_year), max_value=2040)
    allocazione = st.radio("Strategia Location", ["Monosito (tutto in A)", "Multisito (espansione in B)"], index=0)

# ============================================================
# COMPETITOR 5 KM (OSM / OVERPASS - NO API KEY)
# ============================================================
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

@st.cache_data(ttl=60*60)
def fetch_osm_chargers_overpass(lat, lon, radius_km=5):
    radius_m = int(radius_km * 1000)
    query = f"""
    [out:json][timeout:25];
    (
      node["amenity"="charging_station"](around:{radius_m},{lat},{lon});
      way["amenity"="charging_station"](around:{radius_m},{lat},{lon});
      relation["amenity"="charging_station"](around:{radius_m},{lat},{lon});
    );
    out center tags;
    """
    url = "https://overpass-api.de/api/interpreter"
    headers = {"User-Agent": "evfs-executive-suite/1.0 (contact: you@example.com)"}

    try:
        r = requests.post(url, data=query.encode("utf-8"), headers=headers, timeout=35)
    except requests.exceptions.RequestException as e:
        return False, pd.DataFrame(), f"Errore rete/timeout Overpass: {e}"

    if r.status_code != 200:
        txt = (r.text or "").strip()
        if len(txt) > 800:
            txt = txt[:800] + "…"
        return False, pd.DataFrame(), f"HTTP {r.status_code} da Overpass: {txt}"

    try:
        data = r.json()
    except ValueError:
        return False, pd.DataFrame(), "Risposta Overpass non-JSON (endpoint instabile)."

    rows = []
    for el in data.get("elements", []):
        tags = el.get("tags", {}) or {}

        if el.get("type") == "node":
            lat2 = el.get("lat")
            lon2 = el.get("lon")
        else:
            center = el.get("center", {}) or {}
            lat2 = center.get("lat")
            lon2 = center.get("lon")

        if lat2 is None or lon2 is None:
            continue

        dist = float(haversine_km(lat, lon, float(lat2), float(lon2)))
        name = tags.get("name", "charging_station")
        operator = tags.get("operator", "")
        capacity = tags.get("capacity", "")
        access = tags.get("access", "")
        sockets = [k for k in tags.keys() if k.startswith("socket:")]

        rows.append({
            "Nome": name,
            "Operatore": operator,
            "Capacity_tag": capacity,
            "Access": access,
            "Socket_tags_presenti": ", ".join(sockets[:6]) if sockets else "",
            "Distanza_km": dist,
            "Lat": float(lat2),
            "Lon": float(lon2),
            "OSM_type": el.get("type"),
            "OSM_id": el.get("id"),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Distanza_km")
    return True, df, None

def competition_factor_osm(df_poi):
    """
    Moltiplicatore 'soft' (0.60–1.00) basato su:
    - n. stazioni nel raggio
    - distanza del competitor più vicino
    """
    if df_poi is None or df_poi.empty:
        return 1.00, {"n_sites": 0, "nearest_km": None}

    n_sites = int(len(df_poi))
    nearest_km = float(df_poi["Distanza_km"].min())

    pen = 0.00
    pen += min(0.30, 0.02 * n_sites)                      # fino a -30% per densità
    pen += min(0.15, 0.10 * max(0, (2.0 - nearest_km)))    # vicino <2 km fino a -15%

    factor = max(0.60, 1.00 - pen)
    meta = {"n_sites": n_sites, "nearest_km": nearest_km}
    return factor, meta

# ============================================================
# MODEL CORE — ENERGY -> MODULI -> ECONOMICS
# ============================================================
def build_bev_trajectory(bev_2030: float, years_arr: np.ndarray):
    """Traiettoria lineare (50% -> 100% al 2030/ultimo anno) per trasparenza."""
    return np.linspace(bev_2030 * 0.5, bev_2030, len(years_arr))

def compute_plan(bev_series, capture_series, competition_factor=1.0):
    """
    Calcola energia pubblica, energia catturata, moduli necessari (30kW),
    e flussi economici.
    """
    # Energia annua per BEV
    kwh_per_km = (kwh_100km / 100.0)
    kwh_per_bev_year = km_annui_per_bev * kwh_per_km

    # Domanda energia pubblica (totale territorio)
    energy_public_kwh = bev_series * kwh_per_bev_year * public_share

    # Quota cattura effettiva (su energia pubblica) con fattore competizione
    capture_eff = np.clip(capture_series * competition_factor, 0.0, 1.0)
    energy_station_kwh = energy_public_kwh * capture_eff

    # Sessioni annue stimate
    sessions_year = energy_station_kwh / max(kwh_per_sessione, 1e-9)
    sessions_day = sessions_year / 365.0

    # Capacità modulo: energia annua a 100% (poi dimensioniamo per stare sotto target_saturazione)
    hours_year = 8760.0
    cap_module_kwh_year_100 = module_kw * hours_year * uptime * efficienza
    modules_needed = np.ceil(energy_station_kwh / max(cap_module_kwh_year_100 * target_saturazione, 1e-9)).astype(int)
    modules_needed = np.maximum(modules_needed, 1)

    # Saturazione attesa (utilizzo reale vs max teorico)
    saturation = np.where(modules_needed > 0, energy_station_kwh / (modules_needed * cap_module_kwh_year_100), 0.0)

    # CAPEX “a scalini”
    capex_flow = np.zeros_like(bev_series, dtype=float)
    prev = 0
    for i, n in enumerate(modules_needed):
        add = max(0, int(n) - int(prev))
        capex_flow[i] = add * capex_modulo
        prev = int(n)

    # OPEX: per modulo + fisso sito
    opex = modules_needed * opex_modulo + opex_sito_fisso

    # Ricavi e costi
    ricavi = energy_station_kwh * prezzo_kwh
    costi_energia = energy_station_kwh * costo_kwh
    fee_roaming_eur = ricavi * fee_roaming
    canone_potenza_eur = (modules_needed * module_kw) * canone_potenza

    ebitda = (ricavi - costi_energia - fee_roaming_eur) - opex - canone_potenza_eur
    cf_netto = ebitda - capex_flow
    cf_cum = np.cumsum(cf_netto)

    return {
        "energy_public_kwh": energy_public_kwh,
        "energy_station_kwh": energy_station_kwh,
        "capture_eff": capture_eff,
        "sessions_year": sessions_year,
        "sessions_day": sessions_day,
        "modules": modules_needed,
        "saturation": saturation,
        "capex_flow": capex_flow,
        "opex": opex,
        "ricavi": ricavi,
        "costi_energia": costi_energia,
        "fee_roaming": fee_roaming_eur,
        "canone_potenza": canone_potenza_eur,
        "ebitda": ebitda,
        "cf_netto": cf_netto,
        "cf_cum": cf_cum,
        "cap_module_kwh_year_100": cap_module_kwh_year_100,
        "kwh_per_bev_year": kwh_per_bev_year,
    }

def compute_cfo_kpis(years_arr, plan):
    """
    FCF investment-grade: NOPAT + Dep - dWC - CAPEX.
    """
    rev = plan["ricavi"].astype(float)
    ebitda_s = plan["ebitda"].astype(float)
    capex_s = plan["capex_flow"].astype(float)

    # Depreciation proxy: lineare sul CAPEX dell'anno (semplificazione manageriale)
    dep = capex_s / max(ammortamento_anni, 1)
    ebit = ebitda_s - dep

    taxes = np.maximum(0, ebit) * tax_rate
    nopat = ebit - taxes
    opcf = nopat + dep

    wc = rev * wc_pct
    delta_wc = np.diff(np.r_[0.0, wc])
    fcf = opcf - delta_wc - capex_s

    npv = npf.npv(wacc, fcf)
    irr = npf.irr(fcf) if np.any(fcf != 0) else np.nan

    return {"rev": rev, "ebitda": ebitda_s, "capex": capex_s, "fcf": fcf, "npv": npv, "irr": irr}

# ============================================================
# COSTRUZIONE SERIE (ORIZZONTE)
# ============================================================
years = np.arange(int(start_year), int(end_year) + 1)
bev_series = build_bev_trajectory(bev_base_2030, years) * stress_bev
capture_series = np.linspace(0.02, target_cattura_2030, len(years)) * stress_cattura

competition_factor = float(st.session_state.get("competition_factor", 1.0))
apply_competition = st.sidebar.checkbox("Applica fattore competizione OSM alla quota cattura (se disponibile)", value=True)
competition_applied = competition_factor if apply_competition else 1.0

plan = compute_plan(bev_series, capture_series, competition_factor=competition_applied)

# Allocazione moduli A/B (solo per comunicazione di rollout)
mods_A = np.ones(len(years), dtype=int)
mods_B = np.zeros(len(years), dtype=int)
for i, n in enumerate(plan["modules"]):
    if allocazione.startswith("Multisito") and n > 1:
        mods_A[i] = 1
        mods_B[i] = int(n - 1)
    else:
        mods_A[i] = int(n)
        mods_B[i] = 0

# KPI sintetici
tot_capex = float(plan["capex_flow"].sum())
roi_semplice = float(plan["cf_netto"].sum() / tot_capex) if tot_capex > 0 else np.nan

payback = np.nan
for i in range(len(years)):
    if plan["cf_cum"][i] >= 0:
        payback = years[i]
        break

# ============================================================
# SEZIONE DATI SICILIA (contestualizzazione)
# ============================================================
st.subheader("🗺️ Dati BEV Sicilia (2015–2024) — trasparenza territorio")
st.caption("Questi dati sono il fondamento territoriale del tool: seleziona una provincia e vedi crescita storica + forecast.")

col_d1, col_d2 = st.columns([1, 1])
with col_d1:
    sub = df_bev[df_bev["Provincia"] == province].sort_values("Anno")
    st.write(f"**Provincia selezionata:** {province}")
    st.dataframe(sub.set_index("Anno"))

with col_d2:
    fig, ax = plt.subplots()
    ax.plot(sub["Anno"], sub["Elettrico"], marker="o", linewidth=3)
    ax.set_xlabel("Anno")
    ax.set_ylabel("BEV (stock)")
    st.pyplot(fig)
    st.markdown(
        f"""
**Lettura rapida**
- BEV 2024: **{bev_2024:,}**
- Forecast 2030 ({bev_forecast_method}): **{bev_2030_auto:,}** (CAGR ~ {cagr_used*100:.1f}%)

**Nuova logica (energy-led)**
- kWh/BEV/anno: **{plan['kwh_per_bev_year']:,.0f}**
- Energia pubblica = BEV × kWh/BEV × quota pubblico
- Energia catturata = energia pubblica × quota cattura (corretta per competizione se attivo)
        """.replace(",", ".")
    )

# ============================================================
# EXECUTIVE SUMMARY
# ============================================================
st.subheader(f"📌 Executive Summary — {province} | Moduli {int(module_kw)} kW")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(f"Energia pubblica {int(years[-1])}", f"{plan['energy_public_kwh'][-1]/1e6:.2f} GWh")
c2.metric(f"Energia catturata {int(years[-1])}", f"{plan['energy_station_kwh'][-1]/1e6:.2f} GWh")
c3.metric(f"Moduli richiesti {int(years[-1])}", int(plan["modules"][-1]))
c4.metric("CAPEX Tot (orizzonte)", eur(tot_capex))
c5.metric("Payback (anno)", f"{int(payback)}" if np.isfinite(payback) else "n/a")

k1, k2, k3, k4 = st.columns(4)
k1.metric(f"Fatturato {int(years[-1])}", eur(float(plan["ricavi"][-1])))
k2.metric(f"EBITDA {int(years[-1])}", eur(float(plan["ebitda"][-1])))
k3.metric("ROI semplice (periodo)", f"{roi_semplice*100:.0f}%" if np.isfinite(roi_semplice) else "n/a")
k4.metric(f"CF cumulato {int(years[-1])}", eur(float(plan["cf_cum"][-1])))

if cfo_mode:
    # Scenari: agiscono su domanda (BEV) e cattura, prezzi/costi/capex/opex
    SCENARIOS = {
        "Base": dict(mult_bev=1.0, mult_capture=1.0, delta_price=0.0, delta_cost=0.0, mult_capex=1.0, mult_opex=1.0, mult_fee=1.0, mult_canone=1.0),
        "Bear": dict(mult_bev=0.85, mult_capture=0.80, delta_price=-0.05, delta_cost=0.10, mult_capex=1.10, mult_opex=1.10, mult_fee=1.0, mult_canone=1.0),
        "Bull": dict(mult_bev=1.10, mult_capture=1.15, delta_price=0.05, delta_cost=-0.05, mult_capex=0.95, mult_opex=0.95, mult_fee=1.0, mult_canone=1.0),
    }

    def compute_scenario(name):
        p = SCENARIOS[name]
        # applico variazioni ai driver
        bev_s = bev_series * p["mult_bev"]
        cap_s = capture_series * p["mult_capture"]
        # temporaneamente override global inputs (pulito e trasparente)
        global prezzo_kwh, costo_kwh, capex_modulo, opex_modulo, fee_roaming, canone_potenza
        P0, C0 = prezzo_kwh, costo_kwh
        CAP0, OPEX0 = capex_modulo, opex_modulo
        F0, D0 = fee_roaming, canone_potenza

        prezzo_kwh = P0 * (1 + p["delta_price"])
        costo_kwh = C0 * (1 + p["delta_cost"])
        capex_modulo = CAP0 * p["mult_capex"]
        opex_modulo = OPEX0 * p["mult_opex"]
        fee_roaming = F0 * p["mult_fee"]
        canone_potenza = D0 * p["mult_canone"]

        pl = compute_plan(bev_s, cap_s, competition_factor=competition_applied)
        kpis = compute_cfo_kpis(years, pl)

        # restore
        prezzo_kwh, costo_kwh = P0, C0
        capex_modulo, opex_modulo = CAP0, OPEX0
        fee_roaming, canone_potenza = F0, D0
        return pl, kpis

    _, kpis_view = compute_scenario(scenario_view)
    k6, k7 = st.columns(2)
    k6.metric(f"NPV (CFO) — {scenario_view}", eur(float(kpis_view["npv"])))
    k7.metric(f"IRR (CFO) — {scenario_view}", f"{float(kpis_view['irr'])*100:.1f}%" if np.isfinite(kpis_view["irr"]) else "n/a")

# ============================================================
# GRAFICI
# ============================================================
st.divider()
g1, g2 = st.columns(2)

with g1:
    st.write("**1) Domanda energia: pubblico vs catturata**")
    fig, ax = plt.subplots()
    ax.plot(years, plan["energy_public_kwh"] / 1e6, marker="o", linewidth=3, label="Energia pubblica (GWh)")
    ax.plot(years, plan["energy_station_kwh"] / 1e6, marker="o", linewidth=3, label="Energia catturata (GWh)")
    ax.set_xlabel("Anno")
    ax.set_ylabel("GWh/anno")
    ax.legend()
    st.pyplot(fig)

    st.caption("La domanda nasce dal parco BEV e dai km/anno: questo rende la logica solida e scalabile tra province.")

with g2:
    st.write("**2) Moduli 30 kW richiesti e saturazione attesa**")
    fig, ax = plt.subplots()
    ax.bar(years, plan["modules"], linewidth=0)
    ax.set_xlabel("Anno")
    ax.set_ylabel("Moduli installati (tot)")
    st.pyplot(fig)

    st.markdown(
        f"""
**Capacità modulo (100%)**
- kWh/modulo/anno = {int(module_kw)} × 8760 × uptime × efficienza
- = **{plan['cap_module_kwh_year_100']:,.0f} kWh/anno** per modulo

**Regola di ottimizzazione**
- moduli = ceil( energia_catturata / (capacità×target_saturazione) )
- target_saturazione attuale: **{target_saturazione:.0%}**
        """.replace(",", ".")
    )

g3, g4 = st.columns(2)
with g3:
    st.write("**3) Cash Flow cumulato**")
    fig, ax = plt.subplots()
    ax.plot(years, plan["cf_cum"], marker='o', linewidth=3)
    ax.fill_between(years, plan["cf_cum"], 0, where=(plan["cf_cum"] >= 0), alpha=0.15)
    ax.axhline(0, linewidth=1)
    ax.set_xlabel("Anno")
    ax.set_ylabel("€ cumulati")
    st.pyplot(fig)

with g4:
    st.write("**4) Struttura margini (somma periodo)**")
    labels = ['Ricavi', 'Costi Energia', 'Fee Roaming', 'Canone Potenza', 'OPEX', 'EBITDA']
    vals = [
        plan["ricavi"].sum(),
        plan["costi_energia"].sum(),
        plan["fee_roaming"].sum(),
        plan["canone_potenza"].sum(),
        plan["opex"].sum(),
        plan["ebitda"].sum()
    ]
    fig, ax = plt.subplots()
    ax.bar(labels, vals)
    ax.tick_params(axis='x', rotation=20)
    ax.set_ylabel("€ (somma periodo)")
    st.pyplot(fig)

# ============================================================
# REPORT TABELLARE + EXPORT
# ============================================================
st.divider()
st.subheader("📊 Report Analitico: energia, moduli, saturazione, ricavi e ritorno")

df_master = pd.DataFrame({
    "Anno": years,
    "BEV (scenario)": bev_series.round(0).astype(int),
    "kWh/BEV/anno": int(round(plan["kwh_per_bev_year"])),
    "Quota energia pubblico (%)": (public_share * 100),
    "Quota cattura effettiva (%)": (plan["capture_eff"] * 100).round(2),
    "Energia pubblica (kWh)": plan["energy_public_kwh"].round(0).astype(int),
    "Energia catturata (kWh)": plan["energy_station_kwh"].round(0).astype(int),
    "Sessioni (anno)": plan["sessions_year"].round(0).astype(int),
    "Sessioni/giorno": plan["sessions_day"].round(1),
    "Moduli tot": plan["modules"].astype(int),
    "Moduli A": mods_A.astype(int),
    "Moduli B": mods_B.astype(int),
    "Saturazione attesa (%)": (plan["saturation"] * 100).round(1),
    "Fatturato (€)": plan["ricavi"].round(0).astype(int),
    "EBITDA (€)": plan["ebitda"].round(0).astype(int),
    "CAPEX (€)": plan["capex_flow"].round(0).astype(int),
    "CF netto (€)": plan["cf_netto"].round(0).astype(int),
    "CF cumulato (€)": plan["cf_cum"].round(0).astype(int),
}).set_index("Anno")

st.dataframe(df_master, use_container_width=True)

csv = df_master.reset_index().to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Scarica report CSV", data=csv, file_name=f"report_{province.lower()}_{int(start_year)}_{int(end_year)}.csv", mime="text/csv")

with st.expander("📚 Note modello (cosa è cambiato e perché)", expanded=False):
    st.markdown(
        f"""
### Cosa è cambiato (richiesta: energy-led + moduli 30 kW)
- La domanda non è più “auto→kWh con un proxy fisso”: ora parte da **parco BEV** × **km/anno** × **consumo**.
- La quota pubblico è applicata come **quota di energia** (non solo “quota utenti”).
- La capacità è dimensionata in **moduli da 30 kW** e ottimizzata con un **target di saturazione** (anti-coda).
- CAPEX è “a scalini” in base ai moduli aggiunti quando serve.

### Come leggere la saturazione
- Saturazione attesa = energia catturata / (moduli × capacità teorica)
- Se supera il target: il modello aggiunge moduli → CAPEX sale, ma riduce code/vendite perse.

### Limiti (trasparenti)
- km/anno e consumo sono parametri: se hai dato locale/telematico, inseriscilo qui per rendere l’output investment-grade.
- L’analisi OSM misura “densità stazioni” (non kW installati): è un correttivo soft, utile per screening.
        """
    )

# ============================================================
# SEZIONE PROSSIMITÀ (UI)
# ============================================================
st.divider()
st.subheader("🗺️ Sezione Prossimità — Analisi colonnine entro 5 km (OSM/Overpass)")

with st.expander("Impostazioni prossimità", expanded=True):
    site_lat = st.number_input("Latitudine sito", value=38.1157, format="%.6f")
    site_lon = st.number_input("Longitudine sito", value=13.3615, format="%.6f")
    run_prox = st.button("Esegui analisi prossimità (5 km)")

if run_prox:
    ok, df_poi, err = fetch_osm_chargers_overpass(site_lat, site_lon, radius_km=5)

    if not ok:
        st.error("Impossibile interrogare Overpass (OSM).")
        st.code(err, language="text")
        st.info("Overpass può essere lento o rate-limited: riprova tra poco.")
    else:
        if df_poi.empty:
            st.warning("Nessuna charging_station OSM trovata entro 5 km (o dati OSM incompleti).")
            st.session_state["competition_factor"] = 1.0
            st.session_state["competitors_5km"] = df_poi
        else:
            factor, meta = competition_factor_osm(df_poi)

            c1, c2, c3 = st.columns(3)
            c1.metric("Stazioni nel raggio (5 km)", f"{meta['n_sites']}")
            c2.metric("Competitor più vicino", f"{meta['nearest_km']:.2f} km")
            c3.metric("Fattore competizione (OSM)", f"{factor:.2f}x")

            map_site = pd.DataFrame([{"lat": site_lat, "lon": site_lon}])
            map_comp = df_poi.rename(columns={"Lat": "lat", "Lon": "lon"})[["lat", "lon"]]
            st.map(pd.concat([map_site, map_comp], ignore_index=True))

            st.caption("Dettaglio punti (OSM amenity=charging_station)")
            st.dataframe(df_poi, use_container_width=True)

            st.session_state["competition_factor"] = float(factor)
            st.session_state["competitors_5km"] = df_poi

            st.success("Fattore competizione salvato: puoi applicarlo dalla sidebar alla quota cattura.")

