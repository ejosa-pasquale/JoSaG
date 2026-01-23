import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy_financial as npf

st.set_page_config(page_title="Executive Charging Suite", layout="wide")

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

def first_year_exceeds(series, years, threshold=1.0):
    for y, v in zip(years, series):
        if np.isfinite(v) and v > threshold:
            return int(y)
    return None

def depreciation_schedule(capex_flow, life_years):
    """Straight-line depreciation by vintage. capex_flow is yearly CAPEX at start of year."""
    n = len(capex_flow)
    dep = np.zeros(n)
    life = max(int(life_years), 1)
    for i in range(n):
        if capex_flow[i] <= 0:
            continue
        annual = capex_flow[i] / life
        for j in range(i, min(n, i + life)):
            dep[j] += annual
    return dep

def compute_case(params):
    """
    Returns a dict with:
      - df (annual detail)
      - manager KPIs
      - CFO KPIs (if enabled)
    """
    years = params["years"]
    n = len(years)
    potenza_kw = params["potenza_kw"]

    # -------------------------
    # 1) Market funnel (trasparente)
    # -------------------------
    bev_citta = np.linspace(params["bev_base_2030"] * 0.5, params["bev_base_2030"], n) * params["stress_bev"]
    quota_stazione = np.linspace(0.02, params["target_cattura_2030"], n) * params["stress_cattura"]
    auto_target = bev_citta * params["public_share"] * quota_stazione

    energia_kwh = auto_target * params["kwh_annui_per_auto"]  # domanda catturata in kWh/anno

    # -------------------------
    # 2) Capacity sizing (ore e target saturazione)
    # -------------------------
    ore_disp_asset = params["ore_max_giorno"] * 365
    ore_richieste = energia_kwh / max(potenza_kw, 1e-9)
    n_totale = np.ceil((ore_richieste / max(params["saturazione_target"], 1e-9)) / max(ore_disp_asset, 1e-9)).astype(int)
    n_totale = np.maximum(n_totale, 0)

    # allocazione A/B (semplice)
    stazione_A = np.ones(n)
    stazione_B = np.zeros(n)
    for i, units in enumerate(n_totale):
        if params["allocazione"].startswith("Multisito") and units > 1:
            stazione_A[i] = 1
            stazione_B[i] = units - 1
        else:
            stazione_A[i] = units
            stazione_B[i] = 0

    # -------------------------
    # 3) P&L manageriale
    # -------------------------
    ricavi = energia_kwh * params["prezzo_kwh"]
    costo_energia = energia_kwh * params["costo_kwh"]

    fee_roaming = ricavi * params["fee_roaming_pct"]  # CFO-mode (0 in manager mode by default)

    # canoni potenza (demand charges) su kW contrattuali per unità
    contracted_kw = n_totale * potenza_kw * params["contracted_power_factor"]
    canoni_potenza = contracted_kw * params["demand_charge_eur_per_kw_year"]

    opex = n_totale * params["opex_unit"]

    margine_lordo = ricavi - costo_energia - fee_roaming
    ebitda = margine_lordo - opex - canoni_potenza

    # CAPEX (incrementale)
    capex_flow = np.zeros(n)
    prev_n = 0
    for i, units in enumerate(n_totale):
        capex_flow[i] = max(0, units - prev_n) * params["capex_unit"]
        prev_n = units

    # manager cashflow (semplice)
    cf_manager = ebitda - capex_flow
    cf_cum_manager = np.cumsum(cf_manager)

    capex_tot = capex_flow.sum()
    ricavi_tot = ricavi.sum()
    ebitda_tot = ebitda.sum()
    roi_semplice = (cf_manager.sum() / capex_tot) if capex_tot > 0 else np.nan

    payback_year = None
    for y, v in zip(years, cf_cum_manager):
        if v >= 0:
            payback_year = int(y)
            break

    # colli di bottiglia (proxy sessioni)
    sessioni_anno = energia_kwh / max(params["kwh_per_sessione"], 1e-9)
    sessioni_giorno_tot = sessioni_anno / 365
    sessioni_giorno_per_unita = np.where(n_totale > 0, sessioni_giorno_tot / n_totale, 0.0)

    capacita_sessioni_giorno_unit = (params["ore_max_giorno"] * potenza_kw) / max(params["kwh_per_sessione"], 1e-9)
    saturazione_sessioni = np.where(n_totale > 0, sessioni_giorno_per_unita / max(capacita_sessioni_giorno_unit, 1e-9), 0.0)
    lost_sales_pct = np.maximum(0, saturazione_sessioni - 1.0) * 100
    lost_sales_kwh = np.maximum(0, energia_kwh * (saturazione_sessioni - 1.0))

    # -------------------------
    # CFO mode (tax, WC, depreciation)
    # -------------------------
    dep = depreciation_schedule(capex_flow, params["depr_years"])
    ebit = ebitda - dep
    taxes = np.maximum(0, ebit) * params["tax_rate"]
    nopat = ebit - taxes
    op_cf = nopat + dep

    wc = ricavi * params["nwc_pct"]
    delta_wc = np.diff(np.r_[0.0, wc])
    fcf = op_cf - delta_wc - capex_flow
    cum_fcf = np.cumsum(fcf)

    npv = npf.npv(params["wacc"], fcf)
    irr = npf.irr(fcf)

    df = pd.DataFrame({
        "Anno": years,
        "BEV città": bev_citta,
        "BEV pubbliche (proxy)": bev_citta * params["public_share"],
        "Quota cattura %": quota_stazione * 100,
        "Auto target/anno": auto_target,
        "Energia (kWh)": energia_kwh,
        "Unità tot": n_totale,
        "Unità A": stazione_A,
        "Unità B": stazione_B,
        "Ricavi (€)": ricavi,
        "Costo energia (€)": costo_energia,
        "Fee roaming (€)": fee_roaming,
        "Canoni potenza (€)": canoni_potenza,
        "OPEX (€)": opex,
        "EBITDA (€)": ebitda,
        "CAPEX (€)": capex_flow,
        "CF manager (€)": cf_manager,
        "CF cumulato manager (€)": cf_cum_manager,
        "Dep (€)": dep,
        "EBIT (€)": ebit,
        "Tasse (€)": taxes,
        "WC (€)": wc,
        "ΔWC (€)": delta_wc,
        "FCF CFO (€)": fcf,
        "FCF cumulato (€)": cum_fcf,
        "Saturazione sessioni (x)": saturazione_sessioni,
        "Lost sales % (proxy)": lost_sales_pct,
        "Lost sales kWh (proxy)": lost_sales_kwh,
    }).set_index("Anno")

    return {
        "df": df,
        "years": years,
        "potenza_kw": potenza_kw,
        "bev_citta": bev_citta,
        "auto_target": auto_target,

        "n_totale": n_totale,
        "capex_flow": capex_flow,
        "capex_unit": params["capex_unit"],
        "capex_tot": capex_tot,
        "ricavi_tot": ricavi_tot,
        "ebitda_tot": ebitda_tot,
        "roi_semplice": roi_semplice,
        "payback_year": payback_year,
        "npv": npv,
        "irr": irr,
    }

# ============================================================
# Header
# ============================================================
st.title("🛡️ Executive Support System: eVFs - Ricarica DC Sicilia")
st.markdown("### *Tool decisionale chiaro per GM + modalità CFO completa.*")

# ============================================================
# Sidebar
# ============================================================
st.sidebar.header("🧭 Modalità")
mode = st.sidebar.radio(
    "Vista",
    ["Manager (chiara)", "CFO (dettagli + scenari)"],
    index=0,
    help="Manager: grafici e KPI semplici. CFO: aggiunge tasse, WC, canoni potenza, fee roaming, scenari e tornado."
)

# Inputs base (sempre)
st.sidebar.header("🕹️ Mercato (città) e funnel")
with st.sidebar.expander("🌍 Scenario Parco Auto", expanded=True):
    bev_base_2030 = st.number_input("Target BEV città 2030 (scenario base)", value=5000, min_value=0)
    stress_bev = st.slider("Stress test adozione BEV (%)", 50, 150, 100) / 100
    public_share = st.slider("Quota ricarica pubblica (%)", 10, 80, 30) / 100

with st.sidebar.expander("🎯 Strategia di cattura", expanded=True):
    target_cattura_2030 = st.slider("Quota cattura target 2030 (%)", 1.0, 15.0, 5.0) / 100
    stress_cattura = st.slider("Efficacia competitiva (%)", 50, 150, 100) / 100

st.sidebar.header("⚙️ Operatività")
with st.sidebar.expander("🔧 Tecnica e servizio", expanded=True):
    tecnologia = st.selectbox("Tecnologia asset", ["DC 30 kW", "DC 60 kW"])
    allocazione = st.radio("Strategia location", ["Monosito (Tutto in A)", "Multisito (Espansione in B)"])
    ore_max_giorno = st.slider("Disponibilità operativa (ore/giorno)", 4, 24, 12)
    kwh_per_sessione = st.number_input("kWh medi per ricarica (sessione)", value=35, min_value=5)
    saturazione_target = st.slider("Target saturazione (anti-coda) (%)", 50, 95, 85) / 100

with st.sidebar.expander("📌 Assunzioni base (leggibilità)", expanded=False):
    kwh_annui_per_auto = st.number_input("kWh/anno per auto (per convertire in energia)", value=3000, min_value=500)
    ammortamento_anni = st.number_input("Ammortamento 'manageriale' (anni) per break-even", value=5, min_value=1)

st.sidebar.header("💰 Prezzi e costi")
with st.sidebar.expander("Vendita e energia", expanded=True):
    prezzo_kwh = st.number_input("Prezzo vendita (€/kWh)", value=0.69, min_value=0.05, step=0.01)
    costo_kwh = st.number_input("Costo energia (€/kWh)", value=0.30, min_value=0.01, step=0.01)

with st.sidebar.expander("CAPEX / OPEX base", expanded=True):
    potenza_kw = 30 if tecnologia == "DC 30 kW" else 60
    capex_unit_default = 25000 if tecnologia == "DC 30 kW" else 45000
    opex_unit_default = 2000 if tecnologia == "DC 30 kW" else 3500
    capex_unit = st.number_input("CAPEX per unità (€)", value=float(capex_unit_default), min_value=0.0, step=1000.0)
    opex_unit = st.number_input("OPEX annuo per unità (€)", value=float(opex_unit_default), min_value=0.0, step=100.0)

# CFO inputs (solo CFO mode, ma con default innocui)
st.sidebar.header("🏦 CFO inputs")
with st.sidebar.expander("WACC, tasse, WC, canoni potenza, fee roaming", expanded=(mode.startswith("CFO"))):
    wacc = st.slider("WACC (%)", 4.0, 15.0, 8.0) / 100
    tax_rate = st.slider("Tax rate effettivo (%)", 0.0, 40.0, 28.0) / 100
    nwc_pct = st.slider("Working capital (% ricavi)", 0.0, 20.0, 2.0) / 100
    depr_years = st.slider("Ammortamento fiscale (anni)", 3, 15, 8)

    fee_roaming_pct = st.slider("Fee roaming / acquiring (% su ricavi)", 0.0, 20.0, 3.0) / 100

    demand_charge_eur_per_kw_year = st.number_input(
        "Canone potenza (€/kW-anno)", value=0.0, min_value=0.0, step=5.0,
        help="Metti 0 se non applicabile. Se applicabile, impatta molto i siti ad alta potenza."
    )
    contracted_power_factor = st.slider(
        "Fattore potenza contrattuale", 0.3, 1.0, 0.8,
        help="kW contrattuali ≈ unità × potenza nominale × fattore"
    )

# ============================================================
# Compute base case
# ============================================================
years = np.array([2026, 2027, 2028, 2029, 2030])

params_base = {
    "years": years,
    "potenza_kw": potenza_kw,
    "bev_base_2030": bev_base_2030,
    "stress_bev": stress_bev,
    "public_share": public_share,
    "target_cattura_2030": target_cattura_2030,
    "stress_cattura": stress_cattura,
    "allocazione": allocazione,
    "ore_max_giorno": ore_max_giorno,
    "kwh_per_sessione": kwh_per_sessione,
    "saturazione_target": saturazione_target,
    "kwh_annui_per_auto": kwh_annui_per_auto,
    "prezzo_kwh": prezzo_kwh,
    "costo_kwh": costo_kwh,
    "capex_unit": capex_unit,
    "opex_unit": opex_unit,
    "ammortamento_anni": ammortamento_anni,

    # CFO params
    "wacc": wacc,
    "tax_rate": tax_rate,
    "nwc_pct": nwc_pct,
    "depr_years": depr_years,
    "fee_roaming_pct": fee_roaming_pct,
    "demand_charge_eur_per_kw_year": demand_charge_eur_per_kw_year,
    "contracted_power_factor": contracted_power_factor,
}

out = compute_case(params_base)
df = out["df"]

# ============================================================
# KPI summary
# ============================================================
st.subheader(f"📊 Executive Summary — {tecnologia}")
# CAPEX anno per anno (per budgeting)
capex_years = out["years"]
capex_flow = out.get("capex_flow", np.zeros(len(capex_years)))
n_totale = out.get("n_totale", np.zeros(len(capex_years)))
capex_unit_val = out.get("capex_unit", np.nan)

# Mostriamo i primi 5 anni (o meno) per chiarezza
show_n = min(5, len(capex_years))
cap_cols = st.columns(show_n + 1)
for i in range(show_n):
    cap_cols[i].metric(f"CAPEX {int(capex_years[i])}", eur(float(capex_flow[i])))
cap_cols[show_n].metric("CAPEX Tot (orizzonte)", eur(float(out["capex_tot"])))

with st.expander("🔎 Dettaglio CAPEX (per capire da dove arriva)", expanded=False):
    added_units = np.r_[int(n_totale[0]) if len(n_totale)>0 else 0, np.maximum(0, np.diff(n_totale)).astype(int)]
    df_capex = pd.DataFrame({
        "Anno": capex_years,
        "Unità totali (fine anno)": n_totale.astype(int),
        "Unità aggiunte nell'anno": added_units[:len(capex_years)],
        "CAPEX unitario (€)": [capex_unit_val]*len(capex_years),
        "CAPEX anno (€)": capex_flow.astype(int),
    }).set_index("Anno")
    st.dataframe(df_capex)
    st.caption(
        "Esempio: se **CAPEX 2026 = 50.000 €** e CAPEX unitario = 25.000 €, "
        "significa che nel 2026 stai installando **2 unità** (2 × 25.000). "
        "Se questo non è voluto, riduci il target/cattura iniziale o attiva ramp-up."
    )


c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("CAPEX (totale)", eur(out["capex_tot"]))
c2.metric("Fatturato (5 anni)", eur(out["ricavi_tot"]))
c3.metric("EBITDA (5 anni)", eur(out["ebitda_tot"]))
c4.metric("ROI semplice (5 anni)", f"{out['roi_semplice']*100:.0f}%" if np.isfinite(out["roi_semplice"]) else "n/a")
c5.metric("Payback (anno)", f"{out['payback_year']}" if out["payback_year"] else "n/a")

if mode.startswith("CFO"):
    c6, c7, c8 = st.columns(3)
    c6.metric("NPV (FCF)", eur(out["npv"]))
    c7.metric("IRR (FCF)", pct(out["irr"]) if np.isfinite(out["irr"]) else "n/a")
    c8.metric("Fee roaming", f"{fee_roaming_pct*100:.1f}%")

st.caption(
    "Nota: in modalità **Manager** il cash flow è EBITDA − CAPEX (semplificato). "
    "In modalità **CFO** il cash flow è FCF (include tasse, WC, ammortamenti, fee roaming, canoni potenza)."
)

st.divider()

# ============================================================
# Grafici "stile originale" con spiegazioni sotto
# ============================================================
g1, g2 = st.columns(2)

with g1:
    st.write("## 1) Break-even operativo (ricariche/giorno per unità)")
    st.caption("Domanda: **quante ricariche al giorno servono perché 1 colonnina stia in piedi?**")

    auto_range = np.linspace(1, 40, 40)
    margine_sessione = kwh_per_sessione * (prezzo_kwh - costo_kwh)

    costo_fisso_day = (opex_unit + (capex_unit / max(ammortamento_anni, 1))) / 365

    prof_day = (auto_range * margine_sessione) - costo_fisso_day

    fig, ax = plt.subplots()
    ax.plot(auto_range, prof_day, linewidth=3)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Ricariche giornaliere per unità")
    ax.set_ylabel("Margine giornaliero per unità (€)")
    st.pyplot(fig)

    be = (costo_fisso_day / max(margine_sessione, 1e-9))
    st.markdown(
        rf"""
**Formula**
- Margine per sessione: $M_{{sess}} = kWh_{{sess}}\cdot (P - C)$
- Break-even: $N_{{BE}} = \frac{{OPEX_{{day}} + Amm_{{day}}}}{{M_{{sess}}}}$

**Con gli input attuali**
- Margine/sessione ≈ **{margine_sessione:.2f} €**
- Costi fissi+capitale ≈ **{costo_fisso_day:.2f} €/giorno**
- Break-even ≈ **{be:.1f} ricariche/giorno per unità**

**Interpretazione**
- Sopra {be:.1f} ricariche/giorno/unità → l’unità è “in pari”.
- Sotto → serve più domanda (cattura), più margine (prezzo/costo) o meno costi (CAPEX/OPEX).
        """
    )

with g2:
    st.write("## 2) Cash Flow cumulato: quando recupero l’investimento?")
    st.caption("Domanda: **in che anno la somma dei flussi diventa positiva (payback)?**")

    if mode.startswith("CFO"):
        series = df["FCF cumulato (€)"].values
        ylab = "€ cumulati (FCF)"
        expl = "FCF = flusso di cassa dopo tasse, WC, CAPEX."
    else:
        series = df["CF cumulato manager (€)"].values
        ylab = "€ cumulati (EBITDA - CAPEX)"
        expl = "Vista semplice: EBITDA − CAPEX."

    fig, ax = plt.subplots()
    ax.plot(years, series, marker="o", linewidth=3)
    ax.axhline(0, linewidth=1)
    ax.set_xlabel("Anno")
    ax.set_ylabel(ylab)
    st.pyplot(fig)

    st.markdown(
        rf"""
**Formula**
- $CF_{{cum,t}} = \sum_{{i=2026}}^t CF_i$

**Come leggerlo**
- Linea sotto 0 → capitale non ancora recuperato.
- Primo punto sopra 0 → payback.

**Nota**
- {expl}
        """
    )

g3, g4 = st.columns(2)

with g3:
    st.write("## 3) Quante unità servono (e dove): piano installazioni")
    st.caption("Domanda: **quante colonnine devo avere ogni anno per non andare in overload?**")

    fig, ax = plt.subplots()
    ax.bar(years, df["Unità A"], label="Sito A")
    ax.bar(years, df["Unità B"], bottom=df["Unità A"], label="Sito B")
    ax.set_xlabel("Anno")
    ax.set_ylabel("Numero unità")
    ax.legend()
    st.pyplot(fig)

    st.markdown(
        rf"""
**Formula (dimensionamento)**
- Ore richieste: $Ore = \frac{{kWh}}{{kW}}$
- Unità: $n = \lceil \frac{{Ore}}{{Ore_{{disp}}\cdot Sat_{{target}}}} \rceil$

**Interpretazione**
- Se le unità crescono → la domanda catturata cresce e per mantenere qualità (anti-coda) servono più punti.
- In multisito: appena n>1 si apre B (regola semplice, modificabile).
        """
    )

with g4:
    st.write("## 4) Struttura margini: dove vanno i soldi")
    st.caption("Domanda: **il business sta in piedi perché il margine copre costi e investimenti?**")

    vals = [
        df["Ricavi (€)"].sum(),
        df["Costo energia (€)"].sum(),
        df["Fee roaming (€)"].sum(),
        df["Canoni potenza (€)"].sum(),
        df["OPEX (€)"].sum(),
        df["CAPEX (€)"].sum(),
        df["EBITDA (€)"].sum(),
    ]
    labels = ["Ricavi", "Costo energia", "Fee roaming", "Canoni potenza", "OPEX", "CAPEX", "EBITDA"]

    fig, ax = plt.subplots()
    ax.bar(labels, vals)
    ax.tick_params(axis="x", rotation=20)
    ax.set_ylabel("€ (somma 2026–2030)")
    st.pyplot(fig)

    st.markdown(
        rf"""
**Formula**
- $EBITDA = Ricavi - Costi\_energia - Fee - OPEX - Canoni\_potenza$

**Come leggerlo**
- Se **EBITDA** è “grande” rispetto a CAPEX → caso più solido.
- Se **fee roaming** o **canoni potenza** crescono → comprimono margini (tipico in siti ad alta potenza).
        """
    )

# ============================================================
# 5) Domanda città + target cattura + "1 unità basta?"
# ============================================================
st.divider()
st.subheader("✅ Sezione 5 — Domanda in crescita e target cattura: 1 unità basta?")

with st.expander("⚙️ Assunzione per 'auto equivalenti' (solo per leggere capacità 1 unità)", expanded=True):
    kwh_annui_per_auto_sez5 = st.number_input(
        "kWh/anno per auto (Sezione 5)", min_value=500, max_value=8000, value=int(kwh_annui_per_auto), step=100
    )
    sat_unit = st.slider("Target saturazione 1 unità (%)", 30, 95, int(saturazione_target*100)) / 100

# capacità 1 unità -> auto eq / anno
session_min = (kwh_per_sessione / max(potenza_kw, 1e-9)) * 60.0
session_day_capacity = (ore_max_giorno * 60.0 / max(session_min, 1e-9)) * sat_unit
kwh_year_capacity = session_day_capacity * 365.0 * kwh_per_sessione
auto_eq_year_capacity = kwh_year_capacity / max(kwh_annui_per_auto_sez5, 1e-9)

bev_pubbliche = df["BEV pubbliche (proxy)"].values
auto_target = df["Auto target/anno"].values
units_needed_for_target = np.where(auto_eq_year_capacity > 0, auto_target / auto_eq_year_capacity, np.nan)

year_break = first_year_exceeds(units_needed_for_target, years, threshold=1.0)

m1, m2, m3 = st.columns(3)
m1.metric("Capacità 1 unità (auto eq./anno)", f"{auto_eq_year_capacity:,.0f}".replace(",", "."))
m2.metric("Capacità 1 unità (ricariche/giorno)", f"{session_day_capacity:.1f}")
m3.metric("Anno in cui 1 unità NON basta", f"{year_break}" if year_break else "Mai (entro 2030)")

c5a, c5b = st.columns(2)
with c5a:
    st.write("### 5A) Bacino pubblico città vs target cattura")
    fig, ax = plt.subplots()
    ax.plot(years, bev_pubbliche, marker="o", linewidth=3, label="BEV domanda pubblica (proxy)")
    ax.plot(years, auto_target, marker="o", linewidth=3, label="Target cattura (auto/anno)")
    ax.set_xlabel("Anno")
    ax.set_ylabel("Auto/anno")
    ax.legend()
    st.pyplot(fig)

    st.markdown(
        rf"""
**Formule**
- $BEV_{{pub}} = BEV_{{città}}\cdot Quota_{{pubblica}}$
- $Auto_{{target}} = BEV_{{pub}}\cdot Quota_{{cattura}}$

**Interpretazione**
- Se il bacino cresce ma il target resta basso → è una scelta strategica (cattura prudente).
- Se il target cresce molto → serve pianificare capacità (Sez. 5B).
        """
    )

with c5b:
    st.write(f"### 5B) Quante unità servono per servire il target? (1 = basta)")
    fig, ax = plt.subplots()
    ax.plot(years, units_needed_for_target, marker="o", linewidth=3, label="Unità necessarie (equivalenti)")
    ax.axhline(1, linestyle="--", linewidth=1)
    ax.set_xlabel("Anno")
    ax.set_ylabel("Unità richieste")
    ax.legend()
    st.pyplot(fig)

    st.markdown(
        rf"""
**Capacità 1 unità (con input attuali)**
- Tempo sessione ≈ **{session_min:.0f} min**
- Ricariche/giorno (anti-coda) ≈ **{session_day_capacity:.1f}**
- Auto eq./anno ≈ **{auto_eq_year_capacity:,.0f}**

**Come leggerlo**
- Linea < 1 → 1 unità basta.
- Linea > 1 → serve scalare (più unità / più ore / meno kWh sessione / più potenza).
        """.replace(",", ".")
    )

# ============================================================
# Modalità CFO: scenari + tornado
# ============================================================
if mode.startswith("CFO"):
    st.divider()
    st.subheader("🏦 Modalità CFO — Scenari Base / Bear / Bull + Sensitivity (Tornado)")

    st.caption(
        "Questa sezione serve per il CFO: include **tasse, working capital, canoni potenza, fee roaming** "
        "e permette di confrontare scenari e sensitività sul NPV."
    )

    # ---- Scenari (regole semplici e trasparenti) ----
    scenarios = {
        "Base": dict(),
        "Bear": {
            "stress_bev": params_base["stress_bev"] * 0.85,
            "target_cattura_2030": params_base["target_cattura_2030"] * 0.80,
            "prezzo_kwh": params_base["prezzo_kwh"] * 0.95,
            "costo_kwh": params_base["costo_kwh"] * 1.10,
            "capex_unit": params_base["capex_unit"] * 1.10,
            "opex_unit": params_base["opex_unit"] * 1.05,
        },
        "Bull": {
            "stress_bev": params_base["stress_bev"] * 1.15,
            "target_cattura_2030": params_base["target_cattura_2030"] * 1.20,
            "prezzo_kwh": params_base["prezzo_kwh"] * 1.05,
            "costo_kwh": params_base["costo_kwh"] * 0.92,
            "capex_unit": params_base["capex_unit"] * 0.95,
            "opex_unit": params_base["opex_unit"] * 0.98,
        }
    }

    rows = []
    for name, overrides in scenarios.items():
        p = dict(params_base)
        p.update(overrides)
        o = compute_case(p)
        rows.append({
            "Scenario": name,
            "CAPEX": o["capex_tot"],
            "Ricavi": o["ricavi_tot"],
            "EBITDA": o["ebitda_tot"],
            "ROI semplice": o["roi_semplice"],
            "NPV (FCF)": o["npv"],
            "IRR (FCF)": o["irr"],
            "Payback": o["payback_year"] if o["payback_year"] else np.nan
        })

    sdf = pd.DataFrame(rows).set_index("Scenario")
    st.dataframe(sdf.style.format({
        "CAPEX": "{:,.0f}",
        "Ricavi": "{:,.0f}",
        "EBITDA": "{:,.0f}",
        "ROI semplice": "{:.2f}",
        "NPV (FCF)": "{:,.0f}",
        "IRR (FCF)": "{:.2%}",
        "Payback": "{:.0f}",
    }))

    st.markdown(
        """
**Come usare gli scenari**
- **Bear**: adozione e cattura più basse + margine peggiore (prezzo giù / costo su) + CAPEX/OPEX più alti.
- **Bull**: adozione e cattura più alte + margine migliore + CAPEX/OPEX leggermente più favorevoli.
"""
    )

    # ---- Tornado sensitivity su NPV ----
    st.write("### Tornado: quali driver muovono di più il NPV?")
    base_npv = out["npv"]

    drivers = [
        ("Prezzo €/kWh", {"prezzo_kwh": prezzo_kwh * 0.90}, {"prezzo_kwh": prezzo_kwh * 1.10}),
        ("Costo energia €/kWh", {"costo_kwh": costo_kwh * 0.90}, {"costo_kwh": costo_kwh * 1.10}),
        ("Quota cattura 2030", {"target_cattura_2030": target_cattura_2030 * 0.80}, {"target_cattura_2030": target_cattura_2030 * 1.20}),
        ("BEV 2030 (stress)", {"stress_bev": stress_bev * 0.85}, {"stress_bev": stress_bev * 1.15}),
        ("CAPEX unità", {"capex_unit": capex_unit * 0.85}, {"capex_unit": capex_unit * 1.15}),
        ("OPEX unità", {"opex_unit": opex_unit * 0.85}, {"opex_unit": opex_unit * 1.15}),
        ("Fee roaming %", {"fee_roaming_pct": max(0.0, fee_roaming_pct - 0.02)}, {"fee_roaming_pct": fee_roaming_pct + 0.02}),
    ]

    trows = []
    for name, low, high in drivers:
        npv_low = compute_case({**params_base, **low})["npv"]
        npv_high = compute_case({**params_base, **high})["npv"]
        trows.append({"Driver": name, "NPV Low": npv_low, "NPV Base": base_npv, "NPV High": npv_high, "Delta": npv_high - npv_low})

    tdf = pd.DataFrame(trows).sort_values("Delta", ascending=True)

    fig, ax = plt.subplots()
    y = np.arange(len(tdf))
    ax.hlines(y, tdf["NPV Low"].values, tdf["NPV High"].values, linewidth=6)
    ax.plot(tdf["NPV Base"].values, y, "o")
    ax.set_yticks(y)
    ax.set_yticklabels(tdf["Driver"].values)
    ax.set_xlabel("NPV (FCF) €")
    ax.axvline(base_npv, linewidth=1)
    st.pyplot(fig)

    st.dataframe(tdf.style.format({
        "NPV Low": "{:,.0f}",
        "NPV Base": "{:,.0f}",
        "NPV High": "{:,.0f}",
        "Delta": "{:,.0f}",
    }))

# ============================================================
# Report table
# ============================================================
st.divider()
st.subheader("📊 Report Analitico (dettaglio anno per anno)")
st.dataframe(df.style.format({
    "BEV città": "{:,.0f}",
    "BEV pubbliche (proxy)": "{:,.0f}",
    "Quota cattura %": "{:.2f}",
    "Auto target/anno": "{:,.0f}",
    "Energia (kWh)": "{:,.0f}",
    "Unità tot": "{:,.0f}",
    "Ricavi (€)": "{:,.0f}",
    "Costo energia (€)": "{:,.0f}",
    "Fee roaming (€)": "{:,.0f}",
    "Canoni potenza (€)": "{:,.0f}",
    "OPEX (€)": "{:,.0f}",
    "EBITDA (€)": "{:,.0f}",
    "CAPEX (€)": "{:,.0f}",
    "CF manager (€)": "{:,.0f}",
    "CF cumulato manager (€)": "{:,.0f}",
    "Dep (€)": "{:,.0f}",
    "EBIT (€)": "{:,.0f}",
    "Tasse (€)": "{:,.0f}",
    "WC (€)": "{:,.0f}",
    "ΔWC (€)": "{:,.0f}",
    "FCF CFO (€)": "{:,.0f}",
    "FCF cumulato (€)": "{:,.0f}",
    "Saturazione sessioni (x)": "{:.2f}",
    "Lost sales % (proxy)": "{:.1f}",
    "Lost sales kWh (proxy)": "{:,.0f}",
}))
