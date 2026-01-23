mport streamlit as st
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
with st.sidebar.expander("🌍 Scenario Parco Auto BEV", expanded=True):
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
    tecnologia = st.selectbox("Tecnologia Asset", ["DC 30 kW", "DC 60 kW"], index=0)
    allocazione = st.radio("Strategia Location", ["Monosito (Tutto in A)", "Multisito (Espansione in B)"], index=0)
    ore_max_giorno = st.slider("Disponibilità Operativa (ore/giorno)", 4, 24, 10)
    kwh_per_sessione = st.number_input("kWh medi richiesti per ricarica", value=35, min_value=5)
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
    # traiettoria trasparente e modificabile (lineare verso target)
    bev_citta = np.linspace(bev_base_2030 * 0.5, bev_base_2030, len(years)) * stress_bev
    quota_stazione = np.linspace(0.02, target_cattura_2030, len(years)) * stress_cattura

# Funnel
auto_clienti_anno = bev_citta * public_share * quota_stazione

# Strength PDF: conversione in energia annua per auto (proxy dichiarato)
kwh_annui_per_auto = 3000  # come PDF (proxy)
energia_kwh = auto_clienti_anno * kwh_annui_per_auto

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
    st.write("**1) Break-even: soglia di ricariche/giorno per unità**")
    auto_range = np.linspace(1, 40, 40)

    margine_sessione = kwh_per_sessione * (prezzo_kwh - costo_kwh)  # versione base, leggibile
    costo_fisso_day = (opex_unit + (capex_unit / ammortamento_anni)) / 365

    fig1, ax1 = plt.subplots()
    ax1.plot(auto_range, (auto_range * margine_sessione) - costo_fisso_day, linewidth=3)
    ax1.axhline(0, linestyle='--')
    ax1.set_xlabel("Ricariche giornaliere per unità")
    ax1.set_ylabel("Margine giornaliero per unità (€)")
    st.pyplot(fig1)

    st.markdown(r"""
**Formula (operativa)**
- $N_{BE} = \frac{OPEX_{day} + Amm_{day}}{kWh_{sess}\cdot(P_{vend} - C_{energia})}$
""")

st.markdown(
    f"""
**Con gli input attuali**
- Margine/sessione (solo energia): **{margine_sessione:.2f} €**
- OPEX + ammortamento (per unità/giorno): **{costo_fisso_day:.2f} € / giorno**
- Quindi la soglia è dell’ordine di poche ricariche/giorno (dipende da prezzo/costo e OPEX).

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

    st.markdown(r"""
**Cosa mostra**
- Quanto “mangiano” energia, OPEX e (in modalità CFO) fee e canoni.
- Se l’EBITDA è piccolo rispetto ai costi, serve lavorare su margine o cattura.
    """)

# ============================================================
# 5) Capacità 1 unità vs domanda città + target cattura (e indicatore "quando non basta")
# ============================================================
st.divider()
st.subheader("5) Domanda in crescita e target cattura: **1 unità basta?**")

st.caption(
    f"Questa sezione confronta: **bacino pubblico** (BEV×quota pubblica), **target cattura** e la **capacità di 1 unità {tecnologia}**."
)

# capacità 1 unità in auto equivalenti (stile PDF: kWh/anno -> auto/anno)
cap_unit_kwh = potenza_kw * 8760 * uptime * utilizzo_medio_annuo
cap_unit_auto_eq = cap_unit_kwh / kwh_annui_per_auto
units_needed_for_target = np.where(cap_unit_auto_eq > 0, auto_clienti_anno / cap_unit_auto_eq, np.nan)

bev_pubbliche = bev_citta * public_share

col5a, col5b = st.columns(2)
with col5a:
    st.write("**5A) Bacino pubblico vs target**")
    fig, ax = plt.subplots()
    ax.plot(years, bev_pubbliche, marker="o", linewidth=3, label="BEV domanda pubblica (proxy)")
    ax.plot(years, auto_clienti_anno, marker="o", linewidth=3, label="Target cattura (auto/anno)")
    ax.set_xlabel("Anno")
    ax.set_ylabel("Auto/anno")
    ax.legend()
    st.pyplot(fig)

    st.markdown(rf"""
**Interpretazione**
- Se il bacino pubblico cresce ma il target resta basso, non è un limite di mercato ma di strategia (cattura).
- Il target dipende da location, competizione, servizi, prezzo percepito.

**Con input attuali**
- Bacino pubblico 2030 ≈ **{bev_pubbliche[-1]:,.0f} auto**
- Target 2030 ≈ **{auto_clienti_anno[-1]:,.0f} auto** (≈ **{(auto_clienti_anno[-1]/max(bev_pubbliche[-1],1e-9))*100:.1f}%** del bacino)
    """.replace(",", "."))

with col5b:
    st.write("**5B) Quante unità servono per servire il target? (soglia=1)**")
    fig, ax = plt.subplots()
    ax.plot(years, units_needed_for_target, marker="o", linewidth=3, label="Unità necessarie (equivalenti)")
    ax.axhline(1, linestyle="--", linewidth=1)
    ax.set_xlabel("Anno")
    ax.set_ylabel("Unità richieste")
    ax.legend()
    st.pyplot(fig)

    st.markdown(rf"""
**Capacità 1 unità (stile PDF)**
- $kWh_{{unit,anno}} = {potenza_kw}\cdot 8760 \cdot {uptime:.2f} \cdot {utilizzo_medio_annuo:.2f} \approx {cap_unit_kwh:,.0f}$
- Auto equivalenti/anno: **{cap_unit_auto_eq:,.0f} auto**

**Come leggere**
- Se la linea supera 1 → **1 unità non basta** per il target.
- L’anno in cui supera 1 è un ottimo **indicatore decisionale** (quando servono 2 unità).
    """.replace(",", "."))

# Indicatore: primo anno in cui 1 unità non basta
anno_non_basta = None
for i, y in enumerate(years):
    if units_needed_for_target[i] > 1:
        anno_non_basta = int(y)
        break

if anno_non_basta:
    st.warning(f"📌 Indicatore: con gli input attuali, **1 unità non basta a partire dal {anno_non_basta}** (servono ≥ 2 unità).")
else:
    st.success("📌 Indicatore: con gli input attuali, **1 unità basta** per tutto l’orizzonte 2026–2030.")

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

    st.caption("Il Tornado mostra quali variabili muovono di più il VAN: ottimo per decidere su cosa fare due diligence.")

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
- La conversione Auto → kWh usa **{kwh_annui_per_auto} kWh/auto/anno** (proxy): se hai dato locale, sostituiscilo.
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
    Ritorna (ok: bool, df: DataFrame, err: str|None)
    """
    radius_m = int(radius_km * 1000)

    # Query Overpass: nodi + ways + relations
    # NOTA: out center serve per ways/relations
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
    headers = {
        # User-Agent "gentile" (riduce chance di blocco)
        "User-Agent": "palermo-charging-suite/1.0 (contact: you@example.com)"
    }

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

        dist = float(haversine_km(lat, lon, lat2, lon2))

        # Metadati utili quando disponibili
        name = tags.get("name", "charging_station")
        operator = tags.get("operator", "")
        capacity = tags.get("capacity", "")
        access = tags.get("access", "")
        # a volte compaiono socket tags: socket:type2, socket:chademo, socket:ccs...
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
    (OSM spesso non ha potenza: qui non usiamo kW)
    """
    if df_poi is None or df_poi.empty:
        return 1.00, {"n_sites": 0, "nearest_km": None}

    n_sites = int(len(df_poi))
    nearest_km = float(df_poi["Distanza_km"].min())

    # penalità “morbida” (tarabile)
    pen = 0.00
    pen += min(0.30, 0.02 * n_sites)                 # fino a -30% per densità
    pen += min(0.15, 0.10 * max(0, (2.0 - nearest_km)))  # vicino <2 km fino a -15%

    factor = max(0.60, 1.00 - pen)
    meta = {"n_sites": n_sites, "nearest_km": nearest_km}
    return factor, meta

# -----------------------------
# UI
# -----------------------------
st.divider()
st.subheader("🗺️ Sezione 6 — Analisi prossimità colonnine (OSM/Overpass, 5 km periurbano)")

with st.expander("Impostazioni prossimità", expanded=True):
    site_lat = st.number_input("Latitudine sito", value=38.1157, format="%.6f")
    site_lon = st.number_input("Longitudine sito", value=13.3615, format="%.6f")
    apply_to_capture = st.checkbox("Usa fattore competizione (OSM) per correggere quota cattura", value=True)
    run_prox = st.button("Esegui analisi prossimità (5 km)")

if run_prox:
    ok, df_poi, err = fetch_osm_chargers_overpass(site_lat, site_lon, radius_km=5)

    if not ok:
        st.error("Impossibile interrogare Overpass (OSM).")
        st.code(err, language="text")
        st.info("Suggerimento: Overpass può essere lento o rate-limited. Riprova tra 1-2 minuti.")
    else:
        if df_poi.empty:
            st.warning("Nessuna charging_station OSM trovata entro 5 km (oppure dati OSM incompleti).")
            st.session_state["competition_factor"] = 1.0
            st.session_state["competitors_5km"] = df_poi
        else:
            factor, meta = competition_factor_osm(df_poi)

            c1, c2, c3 = st.columns(3)
            c1.metric("Stazioni nel raggio (5 km)", f"{meta['n_sites']}")
            c2.metric("Competitor più vicino", f"{meta['nearest_km']:.2f} km")
            c3.metric("Fattore competizione (OSM)", f"{factor:.2f}x")

            # Mappa semplice (no pydeck)
            map_site = pd.DataFrame([{"lat": site_lat, "lon": site_lon}])
            map_comp = df_poi.rename(columns={"Lat": "lat", "Lon": "lon"})[["lat", "lon"]]
            st.map(pd.concat([map_site, map_comp], ignore_index=True))

            st.caption("Dettaglio punti (OSM amenity=charging_station)")
            st.dataframe(df_poi, use_container_width=True)

            # Output per il modello
            st.session_state["competition_factor"] = float(factor)
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

