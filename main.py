import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy_financial as npf
import io

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
2025\tPALERMO\t2600
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

def forecast_bev_2030(df_hist, province, method="CAGR 2021–2024"):
    """
    Restituisce (bev_2024, bev_2030_forecast, cagr_used).
    Metodo esplicito per trasparenza in riunione.
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
        # fallback: CAGR sugli ultimi 3 punti disponibili
        tail = s.tail(3)
        v0 = float(tail["Elettrico"].iloc[0])
        v1 = float(tail["Elettrico"].iloc[-1])
        years = int(tail["Anno"].iloc[-1] - tail["Anno"].iloc[0])
        cagr = (v1 / max(v0, 1e-9)) ** (1 / max(years, 1)) - 1

    bev_2030 = int(round(bev_2024 * ((1 + cagr) ** 6)))  # 2024 -> 2030
    return bev_2024, max(bev_2030, bev_2024), cagr

def forecast_bev_path_2024_2030(bev_2024: int, bev_2030: int, cap_exp: float = 0.18):
    """
    Costruisce una traiettoria BEV 2024–2030 coerente con la richiesta:
    - 2025 e 2026: crescita lineare (incremento costante dal valore 2024)
    - 2027–2030: crescita esponenziale "non aggressiva" (tasso annuo limitato da cap_exp, se possibile)

    Ritorna un dict {anno: valore}.
    Nota: se per raggiungere il target 2030 serve un tasso > cap_exp, la funzione aumenta
    la quota "lineare" al 2026 (quindi alza il livello 2026) finché il tasso rientra nel cap
    o finché la quota raggiunge un limite ragionevole.
    """
    bev_2024 = float(bev_2024)
    bev_2030 = float(max(bev_2030, bev_2024))

    if bev_2030 <= bev_2024 + 1e-9:
        return {y: int(round(bev_2024)) for y in range(2024, 2031)}

    delta = bev_2030 - bev_2024

    # Quota del gap già "raggiunta" nel 2026 con crescita lineare 2025–2026.
    share_2026 = 0.30
    for _ in range(20):
        bev_2026 = bev_2024 + delta * share_2026
        r_req = (bev_2030 / max(bev_2026, 1e-9)) ** (1 / 4) - 1  # 2026 -> 2030 (4 anni)
        if r_req <= cap_exp or share_2026 >= 0.90:
            break
        share_2026 += 0.05

    share_2025 = share_2026 / 2
    bev_2025 = bev_2024 + delta * share_2025
    bev_2026 = bev_2024 + delta * share_2026

    r = (bev_2030 / max(bev_2026, 1e-9)) ** (1 / 4) - 1

    path = {2024: bev_2024, 2025: bev_2025, 2026: bev_2026}
    v = bev_2026
    for y in (2027, 2028, 2029, 2030):
        v = v * (1 + r)
        path[y] = v

    # Forza il target esatto al 2030
    path[2030] = bev_2030

    return {y: int(round(path[y])) for y in range(2024, 2031)}


# ============================================================
# HEADER
# ============================================================
st.title("🛡️ Executive Planning Tool — Fast DC - Service Stations")
st.markdown("### eV Field Service ")

# ============================================================
# SIDEBAR — DATI TERRITORIO + FUNNEL
# ============================================================
df_bev = pd.read_csv(io.StringIO(BEV_RAW), sep="\t")

st.sidebar.header("🗺️ Territorio (Sicilia) — dati BEV 2015–2024")
province = st.sidebar.selectbox("Provincia (default Palermo)", sorted(df_bev["Provincia"].unique()), index=sorted(df_bev["Provincia"].unique()).index("PALERMO"))
bev_forecast_method = st.sidebar.selectbox("Metodo forecast BEV 2030 (trasparente)", ["CAGR 2021–2024", "CAGR 2015–2024"], index=0)

bev_2024, bev_2030_auto, cagr_used = forecast_bev_2030(df_bev, province, bev_forecast_method)

with st.sidebar.expander("📈 Anteprima dato + forecast", expanded=False):
    st.write(f"**{province}** — BEV 2024: **{bev_2024:,}**".replace(",", "."))
    st.write(f"Forecast 2030 ({bev_forecast_method}): **{bev_2030_auto:,}** (CAGR ~ {cagr_used*100:.1f}%)".replace(",", "."))

st.sidebar.header("🕹️ Market Funnel (domanda)")
with st.sidebar.expander("🌍 Bacino energetico BEV (kWh)", expanded=True):
    use_auto_bev_2030 = st.checkbox("Usa forecast BEV 2030 dal dataset Sicilia", value=True)
    bev_base_2030 = st.number_input(
        f"Target BEV {province} 2030 (Scenario Base)",
        value=int(bev_2030_auto) if use_auto_bev_2030 else 5000,
        min_value=0
    )
    stress_bev = st.slider("Stress Test Adozione BEV (%)", 50, 150, 100) / 100
    public_share = st.slider(
        "Quota Ricarica Pubblica (%)", 10, 80, 30,
        help="Percentuale di proprietari BEV senza ricarica privata (domanda che va sul pubblico)."
    ) / 100

with st.sidebar.expander("🎯 Strategia di Cattura", expanded=True):
    target_cattura_2030 = st.slider("Quota Cattura Target 2030 (%)", 0.5, 20.0, 5.0) / 100
    stress_cattura = st.slider("Efficacia Competitiva Stazione (%)", 50, 150, 100) / 100

# ============================================================
# SIDEBAR — TECNICA + FINANZA (con modalità CFO)
# ============================================================
st.sidebar.header("⚙️ Asset, Operatività e Finanza")
with st.sidebar.expander("🔧 Scelte Tecniche", expanded=True):
    tecnologia = st.selectbox("Tecnologia Asset", ["DC 30 kW", "DC 60 kW"])
    allocazione = st.radio("Strategia Location", ["Monosito (Tutto in A)", "Multisito (Espansione in B)"])
    ore_max_giorno = st.slider("Disponibilità Operativa (ore/giorno)", 4, 24, 10)
    kwh_per_sessione = st.number_input("kWh medi richiesti per ricarica", min_value=5, value=35)
    kwh_annui_per_auto = st.number_input("Consumo annuo medio per BEV (kWh/auto/anno)", value=3000, min_value=500, help="Proxy per convertire il parco BEV in domanda energetica. Se hai dato locale (km/anno × kWh/km), sostituisci qui.")
    uptime = st.slider("Uptime tecnico (%)", 85, 100, 97) / 100
    # Importantissimo: il PDF usa utilizzo medio annuo (30%) per dimensionare
    utilizzo_medio_annuo = st.slider("Utilizzo medio annuo per dimensionamento (PDF) (%)", 10, 80, 30) / 100
    saturazione_target = st.slider("Target saturazione operativa (anti-coda) (%)", 30, 95, 80) / 100

with st.sidebar.expander("💰 Pricing & Costi", expanded=True):
    prezzo_kwh = st.number_input("Prezzo vendita (€/kWh)", value=0.69, step=0.01)
    costo_kwh = st.number_input("Costo energia (€/kWh)", value=0.30, step=0.01)

    fee_roaming = st.slider("Fee roaming/acquiring (% su ricavi ricarica) [CFO]", 0.0, 15.0, 0.0) / 100
    canone_potenza = st.number_input("Canone potenza / demand charges (€/kW-anno) [CFO]", value=0.0, step=5.0)

    capex_unit = 25000 if tecnologia == "DC 30 kW" else 45000
    opex_unit = 2000 if tecnologia == "DC 30 kW" else 3500

    capex_unit = st.number_input("CAPEX per unità (override) (€)", value=float(capex_unit), step=1000.0)
    opex_unit = st.number_input("OPEX per unità/anno (override) (€)", value=float(opex_unit), step=100.0)
    ammortamento_anni = st.slider("Ammortamento gestionale (anni) (break-even)", 3, 12, 5)

with st.sidebar.expander("🏦 Modalità CFO (tasse, WC, scenari, tornado)", expanded=False):
    cfo_mode = st.checkbox("Attiva modalità CFO", value=False)
    wacc = st.slider("WACC (%)", 4, 12, 8) / 100
    tax_rate = st.slider("Tax rate effettivo (%)", 0, 40, 28) / 100
    wc_pct = st.slider("Working capital (% ricavi totali)", 0.0, 20.0, 2.0) / 100
    # scenario pack
    scenario_view = st.selectbox("Scenario da visualizzare", ["Base", "Bear", "Bull"], index=0)

# ============================================================
# LOGICA (pdf-like + manager-friendly)
# ============================================================
years = np.array([2026, 2027, 2028, 2029, 2030])
potenza_kw = 30 if tecnologia == "DC 30 kW" else 60

# (A) Strength PDF: percorso annuale "a scalini" (opzionale lock)
with st.sidebar.expander("🔒 PDF Lock (per replica 1:1 del report PDF)", expanded=False):
    pdf_lock = st.checkbox("Allinea a ipotesi PDF (BEV e cattura a scalini + capacità PDF)", value=False)
    st.caption("Quando ON: BEV e quota cattura usano scalini del PDF e capacità per unità usa kW×8760×uptime×utilizzo medio.")

if pdf_lock:
    # Valori presi dal PDF (DC30 Palermo) — se cambi provincia, qui rimane 'pdf reference'
    bev_citta = np.array([2600, 3000, 3500, 4200, 5000], dtype=float) * stress_bev
    quota_stazione = np.array([0.02, 0.03, 0.04, 0.045, 0.05], dtype=float) * stress_cattura
else:
    # traiettoria trasparente: base 2024 (dataset) → 2025–2026 lineare → 2027–2030 esponenziale soft
    bev_path = forecast_bev_path_2024_2030(bev_2024, bev_base_2030)
    bev_citta = np.array([bev_path[int(y)] for y in years], dtype=float) * stress_bev
    quota_stazione = np.linspace(0.02, target_cattura_2030, len(years)) * stress_cattura

# Funnel (ENERGIA) — focus kWh (bacino pubblico → target stazione)
# 1) Domanda energia totale BEV (kWh/anno)
energia_tot_bev_kwh = bev_citta * kwh_annui_per_auto

# 2) Bacino *pubblico* (kWh/anno): quota di utenti/consumi che passano dal pubblico
energia_pubblica_kwh = energia_tot_bev_kwh * public_share

# 3) Target catturato dalla stazione (kWh/anno)
energia_kwh = energia_pubblica_kwh * quota_stazione

# Per reporting (opzionale): “auto equivalenti” catturate
auto_clienti_anno = energia_kwh / max(kwh_annui_per_auto, 1e-9)

# (B) Capacità/Dimensionamento — due metodi: PDF (kWh/anno) o Session-based (anti-coda)
metodo_capacita = "PDF kWh/anno (kW×8760×uptime×utilizzo)" if pdf_lock else st.sidebar.selectbox(
    "Metodo capacità (dimensionamento)",
    ["Session-based (ore/giorno + kWh/sessione + anti-coda)", "PDF kWh/anno (kW×8760×uptime×utilizzo)"],
    index=0
)

if metodo_capacita.startswith("PDF"):
    # Formula PDF: kW * 8760 * uptime * utilizzo_medio_annuo
    cap_kwh_unit_anno = potenza_kw * 8760 * uptime * utilizzo_medio_annuo
    n_totale = np.ceil(energia_kwh / max(cap_kwh_unit_anno, 1e-9)).astype(int)

    # per reporting: sessioni stimate (solo informativo)
    sessioni_anno = energia_kwh / max(kwh_per_sessione, 1e-9)
    sessioni_giorno_tot = sessioni_anno / 365
    saturazione_sessioni = np.nan * np.ones_like(energia_kwh)  # non usato in questo metodo
else:
    # Session-based: molto leggibile per GM (ricariche/giorno, colli di bottiglia, ecc.)
    minuti_sessione = (kwh_per_sessione / max(potenza_kw, 1e-9)) * 60
    sessioni_giorno_per_unita = (ore_max_giorno * 60 / max(minuti_sessione, 1e-9)) * uptime * saturazione_target
    sessioni_anno_per_unita = sessioni_giorno_per_unita * 365

    sessioni_anno = energia_kwh / max(kwh_per_sessione, 1e-9)
    n_totale = np.ceil(sessioni_anno / max(sessioni_anno_per_unita, 1e-9)).astype(int)

    sessioni_giorno_tot = sessioni_anno / 365
    saturazione_sessioni = np.where(n_totale > 0, sessioni_anno / (n_totale * sessioni_anno_per_unita), 0)

# Allocazione location
stazione_A = np.ones(len(years))
stazione_B = np.zeros(len(years))
for i, n in enumerate(n_totale):
    if allocazione == "Multisito (Espansione in B)" and n > 1:
        stazione_B[i] = n - 1
        stazione_A[i] = 1
    else:
        stazione_A[i] = n

# Ricavi e margini "Base"
ricavi = energia_kwh * prezzo_kwh
costi_energia = energia_kwh * costo_kwh
fee_roaming_eur = ricavi * fee_roaming
# canone potenza: € / (kW-anno) * kW contrattuali (proxy: potenza_kw per unità)
canone_potenza_eur = (n_totale * potenza_kw) * canone_potenza
ebitda = (ricavi - costi_energia - fee_roaming_eur) - (n_totale * opex_unit) - canone_potenza_eur

# CAPEX flow
capex_flow = np.zeros(len(years)); prev_n = 0
for i, n in enumerate(n_totale):
    capex_flow[i] = max(0, n - prev_n) * capex_unit
    prev_n = n

# Cash flow manager-friendly (come PDF): CF netto = EBITDA - CAPEX
cf_netto = ebitda - capex_flow
cf_cum = np.cumsum(cf_netto)

# ROI semplice sull'orizzonte (come PDF)
tot_capex = capex_flow.sum()
roi_semplice = ((cf_netto.sum()) / tot_capex) if tot_capex > 0 else np.nan

# Payback
payback = np.nan
for i in range(len(years)):
    if cf_cum[i] >= 0:
        payback = years[i]
        break

# ============================================================
# Modalità CFO — calcolo FCF after-tax + WC + NPV/IRR + scenari + tornado
# ============================================================
def compute_cfo_kpis(mult_bev=1.0, mult_capture=1.0, delta_price=0.0, delta_cost=0.0, mult_capex=1.0, mult_opex=1.0, mult_fee=1.0, mult_canone=1.0):
    # ricostruisco serie coerente con input (non ricalcolo dimensionamento per semplicità)
    # -> per scenari "puliti" in riunione, in alternativa si può ricalcolare anche n_totale.
    E = energia_kwh * mult_bev * mult_capture
    P = prezzo_kwh * (1 + delta_price)
    C = costo_kwh * (1 + delta_cost)

    rev = E * P
    energy_cost = E * C
    fee = rev * (fee_roaming * mult_fee)
    canone = (n_totale * potenza_kw) * (canone_potenza * mult_canone)

    ebitda_s = (rev - energy_cost - fee) - (n_totale * (opex_unit * mult_opex)) - canone
    capex_s = capex_flow * mult_capex

    # ammortamento lineare su capex (proxy)
    dep = np.zeros_like(ebitda_s)
    for i in range(len(dep)):
        dep[i] = capex_s[i] / max(ammortamento_anni, 1)

    ebit_s = ebitda_s - dep
    taxes = np.maximum(0, ebit_s) * tax_rate
    nopat = ebit_s - taxes
    opcf = nopat + dep

    wc = (rev) * wc_pct
    delta_wc = np.diff(np.r_[0.0, wc])
    fcf = opcf - delta_wc - capex_s

    # NPV/IRR su flussi annuali (scontati da 2026)
    npv = npf.npv(wacc, fcf)
    irr = npf.irr(fcf) if np.any(fcf != 0) else np.nan

    return {
        "rev": rev, "ebitda": ebitda_s, "capex": capex_s, "fcf": fcf,
        "npv": npv, "irr": irr
    }

# scenari (con logica semplice e dichiarata)
SCENARIOS = {
    "Base": dict(mult_bev=1.0, mult_capture=1.0, delta_price=0.0, delta_cost=0.0, mult_capex=1.0, mult_opex=1.0, mult_fee=1.0, mult_canone=1.0),
    "Bear": dict(mult_bev=0.85, mult_capture=0.80, delta_price=-0.05, delta_cost=0.10, mult_capex=1.10, mult_opex=1.10, mult_fee=1.0, mult_canone=1.0),
    "Bull": dict(mult_bev=1.10, mult_capture=1.15, delta_price=0.05, delta_cost=-0.05, mult_capex=0.95, mult_opex=0.95, mult_fee=1.0, mult_canone=1.0),
}

# ============================================================
# SEZIONE DATI SICILIA (chiara, per contestualizzare Palermo)
# ============================================================
st.subheader("🗺️ Dati BEV Sicilia (2015–2024) — trasparenza territorio")
st.caption("Questi dati sono il fondamento territoriale del tool: puoi selezionare una provincia e vedere la crescita storica.")

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
        """.replace(",", ".")
    )

# ============================================================
# DASHBOARD KPI (Manager + CFO)
# ============================================================
st.subheader(f"📌 Executive Summary — {province} | {tecnologia}")

# --- DECISIONE (kWh-based): conviene? quante unità? piano per anno ---
P_netto = prezzo_kwh * (1 - fee_roaming)  # €/kWh incassati (netto fee)
margine_kwh_exec = P_netto - costo_kwh   # €/kWh
opex_day_unit = opex_unit / 365.0
amm_day_unit = (capex_unit / max(ammortamento_anni, 1e-9)) / 365.0
canone_pot_day_unit = (canone_potenza * potenza_kw) / 365.0
costo_fisso_day_unit = opex_day_unit + amm_day_unit + canone_pot_day_unit

kwh_be_day_unit = costo_fisso_day_unit / max(margine_kwh_exec, 1e-9)  # kWh/giorno per unità
kwh_day_tot = energia_kwh / 365.0                                     # kWh/giorno totali stazione (domanda catturata)
kwh_day_pub = energia_pubblica_kwh / 365.0                             # kWh/giorno bacino pubblico
kw_avg_pub = energia_pubblica_kwh / 8760.0                               # kW medi annui bacino pubblico

# capacità giornaliera (anti-coda) per unità — usata solo come check operativo
cap_kwh_day_unit = potenza_kw * ore_max_giorno * uptime * saturazione_target

# quante unità "convengono" (massimo numero che resta sopra BE se la domanda si divide in modo uniforme)
n_profit_max = np.floor(kwh_day_tot / max(kwh_be_day_unit, 1e-9)).astype(int)
n_profit_max = np.maximum(n_profit_max, 0)

# quante unità "servirebbero" per servire tutto il kWh/giorno senza superare capacità target (anti-coda)
n_cap_min = np.ceil(kwh_day_tot / max(cap_kwh_day_unit, 1e-9)).astype(int)
n_cap_min = np.maximum(n_cap_min, 0)

# Raccomandazione per anno: installa fino a n_profit_max (se 0 => non conviene)
n_rec = n_profit_max.copy()

# Sintesi: primo anno e ultimo anno
conviene_2026 = (kwh_day_tot[0] >= kwh_be_day_unit) and (margine_kwh_exec > 0)
conviene_2030 = (kwh_day_tot[-1] >= kwh_be_day_unit) and (margine_kwh_exec > 0)

# Tabella anno per anno (decisione)
df_decision = pd.DataFrame({
    "Anno": years.astype(int),
    "Bacino pubblico (kWh/g)": np.round(kwh_day_pub, 0),
    "Bacino pubblico (kW medi)": np.round(kw_avg_pub, 2),
    "Target cattura (kWh/g)": np.round(kwh_day_tot, 0),
    "BE per unità (kWh/g)": np.round(np.repeat(kwh_be_day_unit, len(years)), 0),
    "Max unità che convengono (profit)": n_profit_max,
    "Min unità per capacità (anti-coda)": n_cap_min,
})


# ---- Executive, subito: parametri finanziari + raccomandazione ----
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Prezzo netto (€/kWh)", f"{P_netto:.2f}")
m2.metric("Costo energia (€/kWh)", f"{costo_kwh:.2f}")
m3.metric("Margine (€/kWh)", f"{margine_kwh_exec:.2f}")
m4.metric("BE per unità", f"{kwh_be_day_unit:.0f} kWh/g")
m5.metric("Conviene installare?", "SÌ (2026)" if conviene_2026 else "NO (2026)")

st.caption(
    "Decisione basata su **kWh/giorno per unità**: se il target cattura (kWh/g) per unità supera il BE, l’unità 'sta in piedi'. "
    "La colonna 'Max unità che convengono' ti dice quante installarne (economicamente) in ciascun anno."
)

# Piano per anno (solo se serve più di 1 unità)
if int(n_rec.max()) > 1:
    # segmentazione per anno: periodi con stesso numero di unità consigliate
    segments = []
    cur_n = int(n_rec[0])
    start_y = int(years[0])
    for i in range(1, len(years)):
        if int(n_rec[i]) != cur_n:
            segments.append((start_y, int(years[i-1]), cur_n))
            cur_n = int(n_rec[i])
            start_y = int(years[i])
    segments.append((start_y, int(years[-1]), cur_n))

    seg_txt = []
    for a, b, n in segments:
        if n <= 0:
            continue
        if a == b:
            seg_txt.append(f"• {a}: **{n}** unità")
        else:
            seg_txt.append(f"• {a}–{b}: **{n}** unità")
    if seg_txt:
        st.info("📅 Piano consigliato (segmentazione per anno):\n" + "\n".join(seg_txt))

st.dataframe(df_decision, use_container_width=True)

# --- Fasi di rollout (più leggibile: cosa succede quando crescono le unità) ---
# Costruisco una tabella "a fasi" raggruppando gli anni con lo stesso numero di unità consigliate.
segments_all = []
cur_n = int(n_rec[0]) if len(n_rec) else 0
start_y = int(years[0]) if len(years) else 0
for i in range(1, len(years)):
    if int(n_rec[i]) != cur_n:
        segments_all.append((start_y, int(years[i-1]), cur_n))
        cur_n = int(n_rec[i])
        start_y = int(years[i])
if len(years):
    segments_all.append((start_y, int(years[-1]), cur_n))

fase_rows = []
prev_units = 0
fase_idx = 0
for a, b, units in segments_all:
    # salto fasi a 0 unità
    if units <= 0:
        prev_units = max(prev_units, units)
        continue
    fase_idx += 1
    nuove = max(0, units - prev_units)
    capex_fase = nuove * capex_unit

    if allocazione == "Multisito (Espansione in B)":
        unita_A = 1 if units >= 1 else 0
        unita_B = max(0, units - 1)
        nota_loc = f"A={unita_A}, B={unita_B}"
    else:
        unita_A = units
        unita_B = 0
        nota_loc = f"A={unita_A}"

    fase_rows.append({
        "Fase": f"Fase {fase_idx}",
        "Anni": f"{a}" if a == b else f"{a}–{b}",
        "Unità consigliate": int(units),
        "Nuove unità (inizio fase)": int(nuove),
        "CAPEX (inizio fase)": int(capex_fase),
        "Distribuzione location": nota_loc,
    })
    prev_units = units

df_fasi = pd.DataFrame(fase_rows)

with st.expander("🧩 Fasi (rollout) ", expanded=False):
    if df_fasi.empty:
        st.info("In nessun anno il modello consiglia installazioni (unità consigliate = 0).")
    else:
        st.dataframe(df_fasi, use_container_width=True)
        st.caption(
            "Questa tabella raggruppa gli anni in **fasi operative**: quando il numero di unità consigliate cambia, "
            "parte una nuova fase. Il CAPEX è conteggiato solo sulle **nuove unità** installate all'inizio della fase."
        )

# --- CAPEX anno per anno (stile budgeting) ---
capex_years = years[:3] if len(years) >= 3 else years
capex_vals = capex_flow[:len(capex_years)]
cA, cB, cC, cD = st.columns(4)
for i, (yy, vv) in enumerate(zip(capex_years, capex_vals)):
    [cA, cB, cC][i].metric(f"CAPEX {int(yy)}", eur(float(vv)))
cD.metric("CAPEX Tot (orizzonte)", eur(float(tot_capex)))

st.caption(
    "Il CAPEX è mostrato **anno per anno**: indica quanto investi in ciascun anno (nuove unità + eventuali upgrade). "
    "Se vedi un valore alto nel 2026, significa che il dimensionamento sta installando più unità già dal primo anno."
)

# --- KPI principali (GM + CFO) ---
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Fatturato 2030", eur(float(ricavi[-1])))
k2.metric("EBITDA 2030", eur(float(ebitda[-1])))
k3.metric("ROI semplice (periodo)", f"{roi_semplice*100:.0f}%" if np.isfinite(roi_semplice) else "n/a")
k4.metric("Payback (anno)", f"{payback}" if np.isfinite(payback) else "n/a")
k5.metric("CF cumulato 2030", eur(float(cf_cum[-1])))

if cfo_mode:
    cfo = compute_cfo_kpis(**SCENARIOS[scenario_view])
    k6, k7 = st.columns(2)
    k6.metric(f"NPV (CFO) — {scenario_view}", eur(float(cfo["npv"])))
    k7.metric(f"IRR (CFO) — {scenario_view}", f"{float(cfo['irr'])*100:.1f}%" if np.isfinite(cfo["irr"]) else "n/a")

# ============================================================
# GRAFICI “stile PDF” con spiegazioni ricche
# ============================================================
st.divider()
c1, c2 = st.columns(2)

with c1:
    st.write("**1) Break-even: soglia di energia (kWh/giorno) per unità**")

    # Margine netto per kWh (considero fee roaming % sui ricavi, se attiva)
    prezzo_netto_kwh = prezzo_kwh * (1 - fee_roaming)
    margine_kwh = max(0.0, prezzo_netto_kwh - costo_kwh)

    # Costi fissi/giorno per unità (OPEX + ammortamento CAPEX + eventuale canone potenza)
    costo_fisso_day = (opex_unit + (capex_unit / ammortamento_anni) + (canone_potenza * potenza_kw)) / 365

    # Range kWh/giorno per grafico (dinamico rispetto a potenza e ore disponibili)
    cap_kwh_day_theoretical = potenza_kw * ore_max_giorno * uptime * saturazione_target
    kwh_range = np.linspace(10, max(500, cap_kwh_day_theoretical * 1.6), 60)

    fig1, ax1 = plt.subplots()
    ax1.plot(kwh_range, (kwh_range * margine_kwh) - costo_fisso_day, linewidth=3)
    ax1.axhline(0, linestyle='--')
    ax1.set_xlabel("kWh/giorno per unità")
    ax1.set_ylabel("Margine giornaliero per unità (€)")
    st.pyplot(fig1)

    with st.expander("📝 Note sotto il grafico (formula + lettura)", expanded=False):
        st.markdown(r"""
**Formula (operativa)**
- $kWh_{BE/day} = \frac{OPEX_{day} + Amm_{day} + CanonePot_{day}}{P_{netto} - C_{energia}}$
- (opz.) $Sessioni_{BE/day} = \frac{kWh_{BE/day}}{kWh_{sess}}$
""")

        st.markdown(
            f"""
**Con gli input attuali**
- Margine per kWh (netto fee): **{margine_kwh:.2f} € / kWh**
- Costi fissi (per unità/giorno): **{costo_fisso_day:.2f} € / giorno**
- Break-even: **{(costo_fisso_day / max(margine_kwh, 1e-9)):.0f} kWh/giorno per unità** (≈ **{(costo_fisso_day / max(margine_kwh, 1e-9) / max(kwh_per_sessione, 1e-9)):.1f} ricariche/giorno**)

**Come leggerlo**
- Sopra la linea 0 → la singola colonnina “sta in piedi” (in senso operativo).
- Sotto 0 → anche se installi, senza traffico non rientri.

**Azioni tipiche**
- Aumentare margine (prezzo, costo energia, ancillary), ridurre OPEX o migliorare cattura.
            """.replace(",", ".")
        )


with c2:
    st.write("**2) Cash Flow cumulato: quando recuperi l’investimento**")
    fig2, ax2 = plt.subplots()
    ax2.plot(years, cf_cum, marker='o', linewidth=3)
    ax2.fill_between(years, cf_cum, 0, where=(cf_cum >= 0), alpha=0.15)
    ax2.axhline(0, linewidth=1)
    ax2.set_xlabel("Anno")
    ax2.set_ylabel("€ cumulati")
    st.pyplot(fig2)

    with st.expander("📝 Note sotto il grafico (formula + interpretazione)", expanded=False):
        st.markdown(rf"""
**Formula (come PDF)**
- $CF_{{netto,t}} = EBITDA_t - CAPEX_t$
- $CF_{{cum,t}} = \sum CF_{{netto}}$

**Interpretazione**
- Il punto in cui la curva supera 0 è il **Payback**.

**Perché può cambiare**
- Se cambia il numero di unità (dimensionamento), cambia il CAPEX e quindi il payback.
- Se attivi fee roaming o canoni potenza (CFO), il payback può peggiorare.
    """)

c3, c4 = st.columns(2)

with c3:
    st.write("**3) Colli di bottiglia: domanda vs capacità (unità richieste)**")
    fig3, ax3 = plt.subplots()
    ax3.bar(years, n_totale, linewidth=0)
    ax3.set_xlabel("Anno")
    ax3.set_ylabel("Unità richieste")
    st.pyplot(fig3)

    with st.expander("📝 Note sotto il grafico (capacità e lettura)", expanded=False):


        if metodo_capacita.startswith("PDF"):
            cap_kwh_unit_anno = potenza_kw * 8760 * uptime * utilizzo_medio_annuo
            st.markdown(rf"""
    **Formula capacità (PDF)**
    - $kWh_{{unit,anno}} = kW \cdot 8760 \cdot Uptime \cdot Utilizzo$

    **Con input attuali**
    - Capacità 1 unità: **{cap_kwh_unit_anno:,.0f} kWh/anno**  
    - Se la domanda (kWh) cresce, le unità richieste aumentano “a scalini”.
            """.replace(",", "."))
        else:
            st.markdown(rf"""
    **Capacità session-based (più leggibile per GM)**
    - Sessioni/giorno per unità dipendono da ore/giorno, kWh/sessione, uptime e target anti-coda.
    - Se la saturazione stimata si avvicina a 100%, iniziano code e vendite perse.
            """)


with c4:
    st.write("**4) Struttura margini (totale periodo)**")
    labels = ['Ricavi', 'Costi Energia', 'Fee Roaming', 'Canone Potenza', 'OPEX', 'EBITDA']
    vals = [
        ricavi.sum(),
        costi_energia.sum(),
        fee_roaming_eur.sum(),
        canone_potenza_eur.sum(),
        (n_totale * opex_unit).sum(),
        ebitda.sum()
    ]
    fig4, ax4 = plt.subplots()
    ax4.bar(labels, vals)
    ax4.tick_params(axis='x', rotation=20)
    ax4.set_ylabel("€ (somma periodo)")
    st.pyplot(fig4)

    with st.expander("📝 Note sotto il grafico (cosa mostra)", expanded=False):
        st.markdown(r"""
**Cosa mostra**
- Quanto “mangiano” energia, OPEX e (in modalità CFO) fee e canoni.
- Se l’EBITDA è piccolo rispetto ai costi, serve lavorare su margine o cattura.
    """)

# ============================================================
# 5) Capacità 1 unità vs domanda città + target cattura (KWH-based)
# ============================================================
st.divider()


# ============================================================
# SEZIONE 7 — Decision Making & CAPEX (Moduli 30 kW)
# ============================================================
st.divider()
st.subheader("🧭 Wizard — Unità nel tempo (domanda vs budget vs piano)")

st.caption(
    "Questa sezione mette **in un'unica tabella** la stessa cosa vista da tre angolazioni:\n"
    "1) **Unità richieste (domanda)**: quante unità servirebbero per servire la domanda stimata.\n"
    "2) **Unità installabili (budget)**: quante unità puoi permetterti, anno per anno.\n"
    "3) **Unità del piano (scelta)**: il rollout effettivo (min tra domanda e budget cumulato)."
)

# --- Parametri base (allineati al resto del report, con fallback sicuri)
UNIT_KW = float(globals().get("potenza_kw", 30.0))
UNIT_COST = float(globals().get("capex_unit", 25_000.0))
UPTIME = float(globals().get("uptime", 0.97))
SAT_TARGET = float(globals().get("saturazione_target", 0.80))
EFFICIENCY = 0.95
HOURS_PER_DAY = float(globals().get("ore_max_giorno", 24.0))

st.write("**Assunzioni economiche (Wizard)**")
cE1, cE2, cE3, cE4 = st.columns(4)
avg_kwh_session = cE1.number_input("kWh medi per sessione", value=28.0, min_value=5.0, step=1.0)
price_kwh = cE2.number_input("Prezzo vendita €/kWh", value=0.65, min_value=0.10, step=0.01)
energy_cost = cE3.number_input("Costo energia €/kWh", value=0.25, min_value=0.01, step=0.01)
opex_fixed_annual = cE4.number_input("OPEX annuo fisso sito", value=18_000.0, min_value=0.0, step=500.0)

scale_opex = st.checkbox("OPEX proporzionale alle unità (oltre al fisso)", value=False)
opex_per_unit = 0.0
if scale_opex:
    opex_per_unit = st.number_input("OPEX annuo per unità", value=float(globals().get("opex_unit", 4_000.0)), min_value=0.0, step=250.0)

st.write("**Assunzioni domanda (Wizard)**")
public_share_local = st.slider(
    "Quota ricarica pubblica locale (%)",
    10, 80,
    int(public_share*100) if 'public_share' in globals() else 30
) / 100

capture_rate = st.slider("Capture rate (quota cattura sul pubblico) (%)", 0.5, 20.0, 3.0) / 100

# --- Capacità per unità (kWh/giorno), con saturazione target per evitare code
kwh_day_per_unit = UNIT_KW * HOURS_PER_DAY * UPTIME * EFFICIENCY * SAT_TARGET

with st.expander("ℹ️ Cosa significa “unità” qui?", expanded=False):
    st.markdown(
        f"- In tutto il report, **1 unità = 1 modulo/colonnina** da circa **{UNIT_KW:.0f} kW**.\n"
        f"- CAPEX per unità: **{eur(UNIT_COST)}**.\n"
        f"- Capacità “comfort” per unità: **{kwh_day_per_unit:,.0f} kWh/giorno** (include uptime, efficienza e saturazione target)."
    )

# --- Orizzonte wizard (3 anni)
years_3y = np.array([2026, 2027, 2028])

# Stima BEV per anno usando il CAGR selezionato nel sidebar (dataset Sicilia)
bev_years_3y = np.array([bev_2024 * ((1 + cagr_used) ** (y - 2024)) for y in years_3y])

# Domanda energia target (kWh/giorno)
kwh_target_day_3y = (bev_years_3y * kwh_annui_per_auto * public_share_local * capture_rate) / 365

# Unità richieste dalla domanda (senza vincoli di budget)
units_demand_3y = np.ceil(kwh_target_day_3y / max(kwh_day_per_unit, 1e-9)).astype(int)
units_demand_3y = np.maximum(1, units_demand_3y)

st.write("**Budget (Wizard) — quanto puoi investire ogni anno**")
b1, b2, b3 = st.columns(3)
budget_2026 = b1.number_input("Budget 2026 (€)", value=25_000.0, min_value=0.0, step=1_000.0)
budget_2027 = b2.number_input("Budget 2027 (€)", value=25_000.0, min_value=0.0, step=1_000.0)
budget_2028 = b3.number_input("Budget 2028 (€)", value=25_000.0, min_value=0.0, step=1_000.0)
budget_3y = np.array([budget_2026, budget_2027, budget_2028], dtype=float)

# Unità installabili (budget) e piano effettivo
units_affordable_new = np.floor(budget_3y / max(UNIT_COST, 1e-9)).astype(int)
units_affordable_cum = np.cumsum(units_affordable_new)

units_plan_3y = np.minimum(units_demand_3y, units_affordable_cum)  # rollout reale, vincolato dal budget
units_new_plan_3y = np.diff(np.insert(units_plan_3y, 0, 0))
capex_year_plan_3y = units_new_plan_3y * UNIT_COST
capex_cum_plan_3y = np.cumsum(capex_year_plan_3y)

# Energia vendibile: se sei sotto-dimensionato, non puoi vendere oltre la capacità
kwh_sell_day_3y = np.minimum(kwh_target_day_3y, units_plan_3y * kwh_day_per_unit)
kwh_sell_year_3y = kwh_sell_day_3y * 365

margin_kwh_simple = price_kwh - energy_cost
opex_year_3y = opex_fixed_annual + (units_plan_3y * opex_per_unit if scale_opex else 0.0)

ebitda_year_3y = (kwh_sell_year_3y * margin_kwh_simple) - opex_year_3y
fcf_year_3y = ebitda_year_3y - capex_year_plan_3y

# --- NPV a 3 anni (criterio decisionale del Wizard)
discount_rate = st.slider("Tasso di sconto per NPV (%)", 6.0, 12.0, 8.0, 0.5) / 100
t = np.arange(1, len(years_3y) + 1)
npv_3y = float(np.sum(fcf_year_3y / ((1 + discount_rate) ** t)))

cum_fcf = np.cumsum(fcf_year_3y)
payback_year = None
if np.any(cum_fcf >= 0):
    payback_year = int(years_3y[int(np.argmax(cum_fcf >= 0))])

decision_txt = "✅ INVESTIRE" if npv_3y > 0 else "❌ NON INVESTIRE"

# --- KPI in testata: separo bene domanda vs piano
kA, kB, kC, kD, kE = st.columns(5)
kA.metric("Unità richieste 2026 (domanda)", int(units_demand_3y[0]))
kB.metric("Unità piano 2026 (budget)", int(units_plan_3y[0]))
kC.metric("CAPEX 2026 (piano)", eur(float(capex_year_plan_3y[0])))
kD.metric("NPV (3 anni)", eur(npv_3y))
kE.metric("Decisione Wizard", decision_txt)

# --- Tabella unica (source of truth)
df_wizard = pd.DataFrame({
    "Anno": years_3y,
    "BEV stimati": np.round(bev_years_3y, 0).astype(int),
    "Domanda (kWh/g)": np.round(kwh_target_day_3y, 0).astype(int),
    "Unità richieste (domanda)": units_demand_3y,
    "Unità installabili (budget, cum)": units_affordable_cum,
    "Unità piano (installate)": units_plan_3y,
    "Nuove unità (piano)": units_new_plan_3y,
    "CAPEX anno (piano)": capex_year_plan_3y,
    "CAPEX cumulato (piano)": capex_cum_plan_3y,
    "kWh vendibili (kWh/g)": np.round(kwh_sell_day_3y, 0).astype(int),
    "EBITDA anno": ebitda_year_3y,
    "FCF anno": fcf_year_3y,
})

df_wizard_fmt = df_wizard.copy()
df_wizard_fmt["CAPEX anno (piano)"] = df_wizard_fmt["CAPEX anno (piano)"].apply(eur)
df_wizard_fmt["CAPEX cumulato (piano)"] = df_wizard_fmt["CAPEX cumulato (piano)"].apply(eur)
df_wizard_fmt["EBITDA anno"] = df_wizard_fmt["EBITDA anno"].apply(eur)
df_wizard_fmt["FCF anno"] = df_wizard_fmt["FCF anno"].apply(eur)

st.dataframe(df_wizard_fmt, use_container_width=True)

# --- Grafico chiaro: domanda vs budget vs piano
figU, axU = plt.subplots()
axU.plot(years_3y, units_demand_3y, linewidth=3, label="Unità richieste (domanda)")
axU.plot(years_3y, units_affordable_cum, linewidth=3, label="Unità installabili (budget cum)")
axU.plot(years_3y, units_plan_3y, linewidth=4, label="Unità piano (installate)")
axU.set_xlabel("Anno")
axU.set_ylabel("Numero unità")
axU.set_xticks(years_3y)
axU.grid(True, alpha=0.25)
axU.legend()
st.pyplot(figU, use_container_width=True)

with st.expander("📌 Lettura rapida (perché a volte “domanda=3” ma “piano=1”)", expanded=False):
    st.markdown(
        "- **Unità richieste (domanda)** risponde a: *quante unità servirebbero per servire il mercato?*\n"
        "- **Unità piano (installate)** risponde a: *quante unità posso mettere davvero con il budget disponibile?*\n"
        "- Se il piano è più basso della domanda, la colonna **kWh vendibili** ti mostra che la vendita è **limitata dalla capacità**."
    )

# Salvo in session_state per riuso in altre sezioni (coerenza del report)
st.session_state["wizard_units_table"] = df_wizard
st.session_state["wizard_units_plan_3y"] = units_plan_3y
st.session_state["wizard_units_demand_3y"] = units_demand_3y
st.session_state["wizard_npv_3y"] = npv_3y
# ============================================================



# ============================================================


st.subheader("Stima Numero Colonnine: **1 unità basta?** (kWh)")

with st.expander("📝 Nota (cosa confrontano i grafici 5A/5B)", expanded=False):
    st.write(
        f"Confronto **bacino pubblico (kWh/anno)**, **target cattura (kWh/anno)** e **capacità di 1 unità {tecnologia} (kWh/anno)**."
    )


# Capacità 1 unità (kWh/anno)
cap_unit_kwh = potenza_kw * 8760 * uptime * utilizzo_medio_annuo  # stile PDF: kW×8760×uptime×utilizzo
units_needed_for_target = np.where(cap_unit_kwh > 0, energia_kwh / cap_unit_kwh, np.nan)

col5a, col5b = st.columns(2)
with col5a:
    st.write("**5A) Bacino pubblico vs target (kWh/anno)**")
    fig, ax = plt.subplots()
    ax.plot(years, energia_pubblica_kwh, marker="o", linewidth=3, label="Bacino pubblico (kWh/anno)")
    ax.plot(years, energia_kwh, marker="o", linewidth=3, label="Target cattura stazione (kWh/anno)")
    ax.set_xlabel("Anno")
    ax.set_ylabel("kWh/anno")
    ax.legend()
    st.pyplot(fig)

    with st.expander("📝 Note sotto il grafico 5A", expanded=False):
        st.markdown(rf"""
    **Interpretazione (kWh)**
    - Il **bacino pubblico** è la quota di energia BEV che passa dal pubblico: $E_{{pub}} = E_{{BEV}}\cdot quota_{{pub}}$.
    - Il **target** è l’energia che ti aspetti di catturare: $E_{{target}} = E_{{pub}}\cdot quota_{{stazione}}$.

    **Con input attuali (2030)**
    - Bacino pubblico 2030 ≈ **{energia_pubblica_kwh[-1]:,.0f} kWh/anno**
    - Target 2030 ≈ **{energia_kwh[-1]:,.0f} kWh/anno** (≈ **{(energia_kwh[-1]/max(energia_pubblica_kwh[-1],1e-9))*100:.1f}%** del bacino)
    - (Solo reporting) “auto equivalenti” 2030 ≈ **{auto_clienti_anno[-1]:,.0f}**
        """.replace(",", "."))

with col5b:
    st.write("**5B) Quante unità servono per servire il target? (soglia=1)**")
    fig, ax = plt.subplots()
    ax.plot(years, units_needed_for_target, marker="o", linewidth=3, label="Unità necessarie (kWh-based)")
    ax.axhline(1, linestyle="--", linewidth=1)
    ax.set_xlabel("Anno")
    ax.set_ylabel("Unità richieste")
    ax.legend()
    st.pyplot(fig)

    with st.expander("📝 Note sotto il grafico 5B", expanded=False):
        st.markdown(rf"""
    **Capacità 1 unità (kWh/anno)**
    - $kWh_{{unit,anno}} = {potenza_kw}\cdot 8760 \cdot {uptime:.2f} \cdot {utilizzo_medio_annuo:.2f} \approx {cap_unit_kwh:,.0f}$

    **Come leggere**
    - Se la linea supera 1 → **1 unità non basta** per il target energetico.
    - L’anno in cui supera 1 è un ottimo **indicatore decisionale** (quando serve installare la 2ª unità).
        """.replace(",", "."))

# Indicatore: primo anno in cui 1 unità non basta
anno_non_basta = None
for i, y in enumerate(years):
    if np.isfinite(units_needed_for_target[i]) and units_needed_for_target[i] > 1:
        anno_non_basta = int(y)
        break

if anno_non_basta:
    st.warning(f"📌 Indicatore: con gli input attuali, **1 unità non basta a partire dal {anno_non_basta}** (servono ≥ 2 unità).")
else:
    st.success("📌 Indicatore: con gli input attuali, **1 unità basta** per il target energetico su tutto l’orizzonte.")
# ============================================================
# 6) Modalità CFO: scenari Base/Bear/Bull + Tornado
# ============================================================
if cfo_mode:
    st.divider()
    st.subheader("🏦 Modalità CFO — scenari e sensitività")

    # Tabella scenari
    rows = []
    for name, params in SCENARIOS.items():
        res = compute_cfo_kpis(**params)
        rows.append({
            "Scenario": name,
            "NPV": res["npv"],
            "IRR": res["irr"],
            "Fatturato 2030": float(res["rev"][-1]),
            "EBITDA 2030": float(res["ebitda"][-1]),
        })
    df_scen = pd.DataFrame(rows).set_index("Scenario")
    st.write("### Scenari (Base / Bear / Bull)")
    st.dataframe(df_scen.style.format({
        "NPV": "{:,.0f}",
        "IRR": "{:.2%}",
        "Fatturato 2030": "{:,.0f}",
        "EBITDA 2030": "{:,.0f}",
    }))

    # Tornado (NPV)
    st.write("### Tornado Sensitivity (NPV)")
    base_npv = compute_cfo_kpis(**SCENARIOS["Base"])["npv"]
    drivers = [
        ("Prezzo €/kWh", {"delta_price": -0.10}, {"delta_price": 0.10}),
        ("Costo energia €/kWh", {"delta_cost": -0.10}, {"delta_cost": 0.10}),
        ("BEV (domanda città)", {"mult_bev": 0.90}, {"mult_bev": 1.10}),
        ("Quota cattura", {"mult_capture": 0.85}, {"mult_capture": 1.15}),
        ("CAPEX", {"mult_capex": 0.90}, {"mult_capex": 1.10}),
        ("OPEX", {"mult_opex": 0.90}, {"mult_opex": 1.10}),
        ("Fee roaming", {"mult_fee": 0.50}, {"mult_fee": 1.50}),
        ("Canone potenza", {"mult_canone": 0.50}, {"mult_canone": 1.50}),
    ]

    tor_rows = []
    for dname, low, high in drivers:
        low_params = dict(SCENARIOS["Base"]); low_params.update(low)
        high_params = dict(SCENARIOS["Base"]); high_params.update(high)
        npv_low = compute_cfo_kpis(**low_params)["npv"]
        npv_high = compute_cfo_kpis(**high_params)["npv"]
        tor_rows.append({"Driver": dname, "NPV Low": npv_low, "NPV Base": base_npv, "NPV High": npv_high, "Delta": npv_high - npv_low})

    tdf = pd.DataFrame(tor_rows).sort_values("Delta", ascending=True)

    fig, ax = plt.subplots()
    y = np.arange(len(tdf))
    ax.hlines(y, tdf["NPV Low"].values, tdf["NPV High"].values, linewidth=6)
    ax.plot(tdf["NPV Base"].values, y, "o")
    ax.set_yticks(y)
    ax.set_yticklabels(tdf["Driver"].values)
    ax.set_xlabel("NPV (VAN) €")
    ax.axvline(base_npv, linewidth=1)
    st.pyplot(fig)

    with st.expander("📝 Note sotto il grafico Tornado", expanded=False):
        st.write("Il Tornado mostra quali variabili muovono di più il VAN: ottimo per decidere su cosa fare due diligence.")

# ============================================================
# REPORT TABELLARE
# ============================================================
st.divider()
st.subheader("📊 Report Analitico: funnel, saturazione, ricavi e ritorno")

df_master = pd.DataFrame({
    "Anno": years,
    "BEV (scenario)": bev_citta.astype(int),
    "Quota pubblica (%)": (public_share * 100),
    "Quota cattura (%)": (quota_stazione * 100),
    "Auto target (anno)": auto_clienti_anno.astype(int),
    "Energia (kWh)": energia_kwh.astype(int),
    "Sessioni (anno)": sessioni_anno.astype(int),
    "Sessioni/giorno tot": sessioni_giorno_tot.round(1),
    "Unità tot": n_totale.astype(int),
    "Unità A": stazione_A.astype(int),
    "Unità B": stazione_B.astype(int),
    "Saturazione stimata (%)": np.where(np.isfinite(saturazione_sessioni), (saturazione_sessioni * 100).round(1), np.nan),
    "Fatturato (€)": ricavi.astype(int),
    "EBITDA (€)": ebitda.astype(int),
    "CAPEX (€)": capex_flow.astype(int),
    "CF netto (€)": cf_netto.astype(int),
    "CF cumulato (€)": cf_cum.astype(int),
}).set_index("Anno")

st.dataframe(df_master)

with st.expander("📚 Intelligence Report (spiegazioni + limiti + allineamento PDF)", expanded=False):
    st.markdown(
        f"""
### Punti di forza “stile PDF” che questo tool include
- Ipotesi **esplicite** (BEV, quota pubblica, quota cattura, prezzo/costo, uptime, utilizzo medio annuo).
- Dimensionamento “**a scalini**” e spiegabile.
- Opzione **PDF lock** per replicare BEV e quota cattura a scalini e la capacità per unità (kW×8760×uptime×utilizzo).

### Limiti e trasparenza
- Il consumo annuo medio per BEV usa **{kwh_annui_per_auto} kWh/auto/anno** (proxy): se hai un dato locale (km/anno × kWh/km), sostituiscilo.
- In modalità **Base** il cash flow è **EBITDA − CAPEX** (come nel PDF).
- In modalità **CFO** aggiungiamo: tasse, WC, fee roaming, canoni potenza, scenari e tornado (investment-grade).
        """
    )
# =============================
# SEZIONE 6 — COMPETITOR 5 KM (OSM / OVERPASS - NO API KEY)
# =============================
import requests
import pandas as pd
import numpy as np
import streamlit as st

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

@st.cache_data(ttl=60*60)
def fetch_osm_chargers_overpass(lat, lon, radius_km=5):
    """
    Cerca stazioni di ricarica (amenity=charging_station) entro radius_km.
    Robusto contro 504/timeout: fallback su più endpoint + retry/backoff.
    Ritorna (ok: bool, df: DataFrame, err: str|None)
    """
    import time

    # Endpoint pubblici (fallback). L'ordine conta: prova prima i più "capienti".
    overpass_endpoints = [
        "https://lz4.overpass-api.de/api/interpreter",
        "https://overpass-api.de/api/interpreter",
        "https://overpass.private.coffee/api/interpreter",
    ]

    headers = {
        # User-Agent "gentile" (riduce chance di blocco e facilita contatto in caso di problemi)
        "User-Agent": "palermo-charging-suite/1.0 (contact: you@example.com)"
    }

    def _build_query(_radius_m: int) -> str:
        # Query Overpass: nodi + ways + relations
        # NOTA: out center serve per ways/relations
        return f"""
        [out:json][timeout:50];
        (
          node[\"amenity\"=\"charging_station\"](around:{_radius_m},{lat},{lon});
          way[\"amenity\"=\"charging_station\"](around:{_radius_m},{lat},{lon});
          relation[\"amenity\"=\"charging_station\"](around:{_radius_m},{lat},{lon});
        );
        out center tags;
        """

    def _parse_elements(data: dict) -> pd.DataFrame:
        elements = data.get("elements", [])
        rows = []
        for el in elements:
            tags = el.get("tags", {}) or {}

            # coordinate: node => lat/lon; ways/relations => center
            if el.get("type") == "node":
                lat2 = el.get("lat")
                lon2 = el.get("lon")
            else:
                center = el.get("center", {}) or {}
                lat2 = center.get("lat")
                lon2 = center.get("lon")

            if lat2 is None or lon2 is None:
                continue

            dist_km = haversine_km(lat, lon, lat2, lon2)
            rows.append({
                "type": el.get("type"),
                "id": el.get("id"),
                "name": tags.get("name"),
                "operator": tags.get("operator"),
                "network": tags.get("network"),
                "socket:type2": tags.get("socket:type2"),
                "socket:chademo": tags.get("socket:chademo"),
                "socket:ccs": tags.get("socket:ccs"),
                "capacity": tags.get("capacity"),
                "lat": lat2,
                "lon": lon2,
                "dist_km": float(dist_km),
            })
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).sort_values("dist_km").reset_index(drop=True)
        return df

    # Strategia:
    # 1) prova radius richiesto
    # 2) se fallisce per 504/timeout, riprova con radius ridotto (x0.6) e fallback endpoint
    radius_m_main = max(200, int(radius_km * 1000))
    radius_m_fallback = max(200, int(radius_m_main * 0.6))

    attempts_plan = [
        ("main", radius_m_main, 2),        # 2 retry per endpoint
        ("fallback", radius_m_fallback, 1) # 1 retry per endpoint (query più leggera)
    ]

    last_err = None
    for label, radius_m, retries_per_endpoint in attempts_plan:
        query = _build_query(radius_m)

        for url in overpass_endpoints:
            for attempt in range(retries_per_endpoint + 1):
                try:
                    # POST consigliato per Overpass; timeout client > timeout query
                    r = requests.post(url, data=query.encode("utf-8"), headers=headers, timeout=75)
                except requests.exceptions.RequestException as e:
                    last_err = f"Errore rete/timeout Overpass ({label}) su {url}: {e}"
                else:
                    if r.status_code == 200:
                        try:
                            data = r.json()
                        except ValueError:
                            last_err = f"Risposta Overpass non-JSON da {url} (endpoint instabile)."
                        else:
                            df = _parse_elements(data)
                            return True, df, None
                    else:
                        # 504 spesso ritorna HTML/XML; tronchiamo per UI
                        txt = (r.text or "").strip()
                        if len(txt) > 800:
                            txt = txt[:800] + "…"
                        last_err = f"HTTP {r.status_code} da Overpass ({label}) su {url}: {txt}"

                # backoff solo se dobbiamo ritentare
                if attempt < retries_per_endpoint:
                    time.sleep(1.0 * (2 ** attempt))

    return False, pd.DataFrame(), (last_err or "Errore Overpass sconosciuto.")


def competition_factor_osm(df_poi: pd.DataFrame):
    '''Calcola un fattore di competizione (<= 1.0) a partire dai POI OSM.

    Idea: più competitor e più vicini => quota cattura effettiva più bassa.

    Ritorna (factor: float, meta: dict)
    - factor è clampato tra 0.60 e 1.00 (non amplifica mai, solo riduce)
    - usa la colonna dist_km prodotta da fetch_osm_chargers_overpass
    '''
    if df_poi is None or df_poi.empty or 'dist_km' not in df_poi.columns:
        return 1.0, {'n_sites': 0, 'nearest_km': float('inf')}

    n_sites = int(len(df_poi))
    nearest_km = float(df_poi['dist_km'].min())

    # Penalità per numerosità (log per evitare eccessi)
    p_n = 0.12 * np.log1p(n_sites)

    # Penalità per prossimità: alta se <1km, decresce rapidamente oltre ~3km
    if nearest_km <= 0.0:
        p_d = 0.30
    else:
        p_d = 0.30 * np.exp(-nearest_km / 1.5)

    factor = 1.0 - float(p_n + p_d)
    factor = float(max(0.60, min(1.00, factor)))

    meta = {
        'n_sites': n_sites,
        'nearest_km': nearest_km,
        'penalty_n': float(p_n),
        'penalty_d': float(p_d),
    }
    return factor, meta

st.subheader("🗺️ Analisi prossimità colonnine -  Competitor")

with st.expander("Impostazioni prossimità", expanded=True):
    site_lat = st.number_input("Latitudine sito", value=38.1157, format="%.6f")
    site_lon = st.number_input("Longitudine sito", value=13.3615, format="%.6f")
    radius_km = st.slider("Raggio analisi (km)", min_value=1, max_value=10, value=5, step=1)
    apply_to_capture = st.checkbox("Usa fattore competizione (OSM) per correggere quota cattura", value=True)
    run_prox = st.button(f"Esegui analisi prossimità ({radius_km} km)")

if run_prox:
    ok, df_poi, err = fetch_osm_chargers_overpass(site_lat, site_lon, radius_km=radius_km)

    if not ok:
        st.error("Impossibile interrogare Overpass (OSM).")
        st.code(err, language="text")
        st.info("Suggerimento: Overpass può essere lento o rate-limited. Riprova tra 1-2 minuti.")
        st.session_state["competition_factor"] = 1.0
        st.session_state["competitors_5km"] = pd.DataFrame()
    else:
        if df_poi.empty:
            st.warning(f"Nessuna charging_station OSM trovata entro {radius_km} km (oppure dati OSM incompleti).")
            st.session_state["competition_factor"] = 1.0
            st.session_state["competitors_5km"] = df_poi
        else:
            factor, meta = competition_factor_osm(df_poi)

            c1, c2, c3 = st.columns(3)
            c1.metric(f"Stazioni nel raggio ({radius_km} km)", f"{meta['n_sites']}")
            c2.metric("Competitor più vicino", f"{meta['nearest_km']:.2f} km")
            c3.metric("Fattore competizione (OSM)", f"{factor:.2f}x")

            # Mappa semplice (no pydeck)
            map_site = pd.DataFrame([{"lat": site_lat, "lon": site_lon}])
            map_comp = df_poi[['lat', 'lon']].copy()
            st.map(pd.concat([map_site, map_comp], ignore_index=True))

            st.caption(f"Dettaglio punti (OSM amenity=charging_station) entro {radius_km} km")
            st.dataframe(df_poi, use_container_width=True)

            # Output per il modello
            st.session_state["competition_factor"] = float(factor) if apply_to_capture else 1.0
            st.session_state["competitors_5km"] = df_poi

            if apply_to_capture:
                st.success("Fattore competizione pronto. Applicalo alla quota cattura PRIMA del funnel (vedi nota).")
            else:
                st.info("Fattore non applicato: sezione solo informativa.")

            st.markdown(
                """
**Nota integrazione nel modello (1 riga):**
Applica questo moltiplicatore **prima** del calcolo energia/funnel:
`quota_stazione_eff = quota_stazione * st.session_state.get("competition_factor", 1.0)`
                """
            )


# SEZIONE 8 — Executive Investment Summary
# ============================================================
st.subheader("📈 Executive Investment Summary")

years = list(range(2024, 2031))
cum_cf = 0
rows = []

for y in years:
    growth_factor = (1 + cagr_used) ** (y - 2024)
    kwh_day_y = kwh_target_day * growth_factor
    kwh_year_y = kwh_day_y * 365
    ebitda_y = kwh_year_y * margin_kwh_simple - opex_annual
    cum_cf += ebitda_y - (capex if y == 2024 else 0)

    rows.append({
        "Anno": y,
        "Target kWh/giorno": round(kwh_day_y, 0),
        "Target kWh/anno": round(kwh_year_y, 0),
        "Moduli 30kW": modules_needed,
        "Saturazione Asset %": asset_saturation,
        "EBITDA": ebitda_y,
        "CAPEX": capex if y == 2024 else 0,
        "Cash Flow Cumulato": cum_cf
    })

df_exec = pd.DataFrame(rows)
df_exec["Saturazione Asset %"] = df_exec["Saturazione Asset %"].apply(pct)
df_exec["EBITDA"] = df_exec["EBITDA"].apply(eur)
df_exec["CAPEX"] = df_exec["CAPEX"].apply(eur)
df_exec["Cash Flow Cumulato"] = df_exec["Cash Flow Cumulato"].apply(eur)

st.dataframe(df_exec, use_container_width=True)

# ============================================================
# SEZIONE 9 — Business Driver
# ============================================================
st.subheader("🧮 Business Driver")

breakeven_kwh_day = opex_annual / (max(margin_kwh_simple, 1e-9) * 365)
breakeven_sessions_day = breakeven_kwh_day / max(avg_kwh_session, 1e-9)

bd = pd.DataFrame({
    "Indicatore": ["Margine per kWh", "Punto di Pareggio (kWh/giorno)", "Punto di Pareggio (ricariche/giorno)"],
    "Valore": [f"{margin_kwh_simple:.2f} €/kWh", round(breakeven_kwh_day, 0), round(breakeven_sessions_day, 1)]
})

st.table(bd)
