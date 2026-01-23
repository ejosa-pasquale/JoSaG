import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy_financial as npf

st.set_page_config(page_title="Executive Charging Suite", layout="wide")

# ============================================================
# INTESTAZIONE
# ============================================================
st.title("🛡️ Executive Support System: eVFs - Ricarica DC Sicilia")
st.markdown(
    "### *Investment Readiness Tool: trasformare i dati del parco auto in decisioni infrastrutturali comprensibili e bancabili.*"
)

# ============================================================
# SIDEBAR: INPUT (Market funnel + Tecnica + Finanza)
# ============================================================
st.sidebar.header("🕹️ Variabili di Mercato (Market Funnel)")
with st.sidebar.expander("🌍 Scenario Parco Auto", expanded=True):
    bev_base_2030 = st.number_input("Target BEV Palermo 2030 (Scenario Base)", value=5000, min_value=0)
    stress_bev = st.slider("Stress Test Adozione BEV (%)", 50, 150, 100) / 100
    public_share = st.slider(
        "Quota Ricarica Pubblica (%)", 10, 80, 30,
        help="Percentuale di proprietari BEV senza ricarica privata (domanda che va sul pubblico)."
    ) / 100

with st.sidebar.expander("🎯 Strategia di Cattura", expanded=True):
    target_cattura_2030 = st.slider("Quota Cattura Target 2030 (%)", 1.0, 15.0, 5.0) / 100
    stress_cattura = st.slider(
        "Efficacia Competitiva Stazione (%)", 50, 150, 100,
        help=">100% = posizione/servizio migliori della media; <100% = concorrenza forte"
    ) / 100

st.sidebar.header("⚙️ Operatività e Finanza")
with st.sidebar.expander("🔧 Scelte Tecniche", expanded=True):
    tecnologia = st.selectbox("Tecnologia Asset", ["DC 30 kW", "DC 60 kW"])
    allocazione = st.radio("Strategia Location", ["Monosito (Tutto in A)", "Multisito (Espansione in B)"])
    ore_max_giorno = st.slider("Disponibilità Operativa (ore/giorno)", 4, 24, 12)
    kwh_per_sessione = st.number_input("kWh medi richiesti per ricarica", value=35, min_value=5)
    saturazione_target = st.slider(
        "Target saturazione (anti-coda) (%)", 50, 95, 85,
        help="Regola pratica: oltre ~85–90% aumentano code e clienti persi."
    ) / 100

with st.sidebar.expander("📌 Assunzioni Base (modificabili)", expanded=False):
    kwh_annui_per_auto = st.number_input(
        "Consumo annuo medio per auto (kWh/anno)", value=3000, min_value=500,
        help="Serve a convertire 'auto catturate' in energia annua. Sostituiscilo con un dato locale se lo hai."
    )
    ammortamento_anni = st.number_input(
        "Ammortamento usato nel break-even (anni)", value=5, min_value=1,
        help="Usato SOLO per la soglia break-even 'manageriale'. Non è la durata fiscale."
    )

with st.sidebar.expander("💰 Financials", expanded=True):
    prezzo_kwh = st.number_input("Prezzo vendita (€/kWh)", value=0.69, min_value=0.05, step=0.01)
    costo_kwh = st.number_input("Costo energia (€/kWh)", value=0.30, min_value=0.01, step=0.01)

    capex_unit = 25000 if tecnologia == "DC 30 kW" else 45000
    opex_unit = 2000 if tecnologia == "DC 30 kW" else 3500

    wacc = st.slider("Costo del Capitale (WACC %)", 4, 15, 8) / 100

# ============================================================
# LOGICA DI CALCOLO
# ============================================================
years = np.array([2026, 2027, 2028, 2029, 2030])
potenza_kw = 30 if tecnologia == "DC 30 kW" else 60

# 1) Funnel mercato (Palermo - semplificato e trasparente)
bev_citta = np.linspace(bev_base_2030 * 0.5, bev_base_2030, len(years)) * stress_bev
quota_stazione = np.linspace(0.02, target_cattura_2030, len(years)) * stress_cattura
auto_clienti_anno = bev_citta * public_share * quota_stazione

energia_kwh = auto_clienti_anno * kwh_annui_per_auto  # energia annua servibile (domanda catturata)

# 2) Capacità e dimensionamento
ore_disp_asset = ore_max_giorno * 365
ore_richieste = energia_kwh / potenza_kw  # (kWh)/(kW) = ore

# dimensionamento con target saturazione (anti-coda): se target 85%, aumentiamo unità per lasciare margine
n_totale = np.ceil((ore_richieste / max(saturazione_target, 1e-9)) / ore_disp_asset).astype(int)
n_totale = np.maximum(n_totale, 0)

# 3) Allocazione location
stazione_A = np.ones(len(years))
stazione_B = np.zeros(len(years))
for i, n in enumerate(n_totale):
    if allocazione == "Multisito (Espansione in B)" and n > 1:
        stazione_B[i] = n - 1
        stazione_A[i] = 1
    else:
        stazione_A[i] = n

# 4) Ricavi, costi e cashflow (approccio semplice e leggibile)
ricavi = energia_kwh * prezzo_kwh
costo_energia = energia_kwh * costo_kwh
margine_lordo = ricavi - costo_energia

ebitda = margine_lordo - (n_totale * opex_unit)

capex_flow = np.zeros(len(years))
prev_n = 0
for i, n in enumerate(n_totale):
    capex_flow[i] = max(0, n - prev_n) * capex_unit
    prev_n = n

cf_netto = ebitda - capex_flow
cf_cum = np.cumsum(cf_netto)

# KPI extra richiesti: investimento, fatturato, ROI, colli di bottiglia
capex_tot = capex_flow.sum()
ricavi_tot = ricavi.sum()
ebitda_tot = ebitda.sum()

roi_semplice = (cf_netto.sum() / capex_tot) if capex_tot > 0 else np.nan  # manager-friendly

# Payback (anno) sul cumulato non scontato
payback_anno = None
for i, v in enumerate(cf_cum):
    if v >= 0:
        payback_anno = int(years[i])
        break

# sessioni e colli di bottiglia (leggibili)
sessioni_anno = energia_kwh / max(kwh_per_sessione, 1e-9)
sessioni_giorno_tot = sessioni_anno / 365
sessioni_giorno_per_unita = np.where(n_totale > 0, sessioni_giorno_tot / n_totale, 0)

# capacità teorica in sessioni/giorno per unità (approssimazione semplice)
capacita_sessioni_giorno_unit = (ore_max_giorno * potenza_kw) / max(kwh_per_sessione, 1e-9)
saturazione_sessioni = np.where(n_totale > 0, sessioni_giorno_per_unita / capacita_sessioni_giorno_unit, 0)

# "vendite perse" stimata se saturazione > 100% (proxy semplice)
lost_sales_pct = np.maximum(0, (saturazione_sessioni - 1.0)) * 100
lost_sales_kwh = np.maximum(0, energia_kwh * (saturazione_sessioni - 1.0))

# ============================================================
# DASHBOARD KPI (super chiara)
# ============================================================
st.subheader(f"📊 Executive Summary — {tecnologia}")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Investimento (CAPEX Tot)", f"€ {capex_tot:,.0f}")
k2.metric("Fatturato Tot (5 anni)", f"€ {ricavi_tot:,.0f}")
k3.metric("EBITDA Tot (5 anni)", f"€ {ebitda_tot:,.0f}")
k4.metric("ROI semplice (5 anni)", f"{roi_semplice*100:.0f}%" if np.isfinite(roi_semplice) else "n/a")
k5.metric("Payback (anno)", f"{payback_anno}" if payback_anno else "n/a")

st.caption(
    "📌 **ROI semplice** = (somma Cash Flow netti) / CAPEX totale. "
    "È un indicatore 'manageriale'. NPV/IRR sono disponibili sotto come metriche finanziarie."
)

k6, k7, k8, k9 = st.columns(4)
k6.metric("VAN (NPV)", f"€ {npf.npv(wacc, cf_netto):,.0f}")
irr_val = npf.irr(cf_netto)
k7.metric("TIR (IRR)", f"{(irr_val*100):.1f}%" if np.isfinite(irr_val) else "n/a")
k8.metric("Auto/Anno (2030)", f"{auto_clienti_anno[-1]:,.0f}")
k9.metric("Saturazione (2030)", f"{(ore_richieste[-1]/(max(n_totale[-1],1)*ore_disp_asset)*100):.1f}%")

st.divider()

# ============================================================
# GRAFICI + SPIEGAZIONI (stile originale, ma più parlante)
# ============================================================
c1, c2 = st.columns(2)

# ------------------ 1) Break-even ------------------
with c1:
    st.write("## 1) Break-even operativo (ricariche/giorno per unità)")
    st.caption("Domanda chiave: **quante ricariche al giorno servono perché 1 colonnina stia in piedi?**")

    auto_range = np.linspace(1, 40, 40)
    margine_sessione = kwh_per_sessione * (prezzo_kwh - costo_kwh)

    # costo fisso giornaliero "per unità" = opex annuo/365 + ammortamento manageriale capex
    costo_fisso_day = (opex_unit + (capex_unit / ammortamento_anni)) / 365

    profitto_day = (auto_range * margine_sessione) - costo_fisso_day

    fig1, ax1 = plt.subplots()
    ax1.plot(auto_range, profitto_day, linewidth=3)
    ax1.axhline(0, linestyle="--", linewidth=2)
    ax1.set_xlabel("Ricariche giornaliere per unità")
    ax1.set_ylabel("Margine giornaliero per unità (€)")
    st.pyplot(fig1)

    be = (costo_fisso_day / max(margine_sessione, 1e-9))
    st.markdown(
        rf"""
**Formula (manageriale):**  
$N_{{BE}} = \frac{{OPEX_{{day}} + Amm_{{day}}}}{{kWh_{{sess}}\cdot(P_{{vend}}-C_{{ene}})}}$

**Valori con gli input attuali**
- **Soglia break-even**: ~ **{be:.1f} ricariche/giorno/unità**
- **Margine per sessione**: ~ **{margine_sessione:.2f} €** (solo energia: {kwh_per_sessione} kWh × (prezzo−costo))
- **Costo fisso/day per unità**: ~ **{costo_fisso_day:.2f} €** (OPEX + “ammortamento” su {ammortamento_anni} anni)

**Come leggerlo (1 minuto)**
- Se la linea è **sopra 0** → l’unità genera margine.
- Se è **sotto 0** → serve più domanda o più margine (prezzo/costo energia/OPEX).
- Break-even è una soglia “di pancia” utile in riunione, non sostituisce il modello fiscale.
        """
    )

# ------------------ 2) Cashflow cumulato ------------------
with c2:
    st.write("## 2) Recupero investimento (Cash Flow cumulato)")
    st.caption("Domanda chiave: **in che anno rientro dell'investimento?**")

    fig2, ax2 = plt.subplots()
    ax2.plot(years, cf_cum, marker="o", linewidth=3)
    ax2.fill_between(years, cf_cum, 0, where=(cf_cum >= 0), alpha=0.15)
    ax2.axhline(0, linewidth=1)
    ax2.set_xlabel("Anno")
    ax2.set_ylabel("Cash flow cumulato (€)")
    st.pyplot(fig2)

    if payback_anno:
        msg = f"✅ **Payback stimato**: il cumulato diventa positivo nel **{payback_anno}**."
    else:
        msg = "⚠️ **Payback non raggiunto** entro il 2030 con gli input attuali."

    st.markdown(
        rf"""
**Formula:**  
$CF_{{cum,t}} = \sum (EBITDA_t - CAPEX_t)$

**Come leggerlo**
- Quando attraversa lo **zero** → hai recuperato l’investimento.
- I “gradini” verso il basso sono anni in cui fai **nuovo CAPEX** (aggiunta colonnine).
- Se la pendenza è bassa → margine insufficiente (prezzo/costo energia/OPEX) o domanda bassa.

{msg}
        """
    )

st.divider()
c3, c4 = st.columns(2)

# ------------------ 3) Allocazione e capacità ------------------
with c3:
    st.write("## 3) Piano colonnine (A/B) + capacità operativa")
    st.caption("Domanda chiave: **quante colonnine servono e dove le metto?**")

    fig3, ax3 = plt.subplots()
    ax3.bar(years, stazione_A, label="Stazione A")
    ax3.bar(years, stazione_B, bottom=stazione_A, label="Stazione B")
    ax3.set_ylabel("Numero colonnine")
    ax3.set_xlabel("Anno")
    ax3.legend()
    st.pyplot(fig3)

    sat_2030 = saturazione_sessioni[-1] * 100 if n_totale[-1] > 0 else 0

    st.markdown(
        rf"""
**Formula dimensionamento (con margine anti-coda):**  
$n = \left\lceil \frac{{(E_{{tot}}/Potenza) / TargetSat}}{{Ore_{{max}}\cdot 365}} \right\rceil$

**Cosa ti dice questo grafico**
- Trasforma la domanda (kWh/anno) in **numero colonnine** necessarie.
- Con **Multisito**, quando n>1 assegna 1 unità ad A e sposta l’eccesso su B.

**Capacità (semplificata ma utile)**
- Capacità teorica per unità: **{capacita_sessioni_giorno_unit:.1f} ricariche/giorno/unità**
- Ricariche/giorno previste nel 2030 (per unità): **{sessioni_giorno_per_unita[-1]:.1f}**
- Saturazione stimata 2030: **{sat_2030:.0f}%** (target: {saturazione_target*100:.0f}%)
        """
    )

# ------------------ 4) Struttura margini + bottleneck ------------------
with c4:
    st.write("## 4) Fatturato, margini e colli di bottiglia (5 anni)")
    st.caption("Domanda chiave: **quanto fatturo, quanto mi resta e dove si blocca?**")

    fig4, ax4 = plt.subplots()
    labels = ["Fatturato", "Costo energia", "OPEX", "CAPEX", "EBITDA"]
    vals = [
        ricavi.sum(),
        costo_energia.sum(),
        (n_totale * opex_unit).sum(),
        capex_flow.sum(),
        ebitda.sum(),
    ]
    ax4.bar(labels, vals)
    ax4.set_ylabel("€ (somma 2026–2030)")
    st.pyplot(fig4)

    gross_margin_pct = (margine_lordo.sum() / ricavi.sum() * 100) if ricavi.sum() > 0 else np.nan
    ebitda_margin_pct = (ebitda.sum() / ricavi.sum() * 100) if ricavi.sum() > 0 else np.nan

    sat_flag = "✅" if (saturazione_sessioni[-1] <= saturazione_target) else "⚠️"
    lost_kwh_2030 = float(lost_sales_kwh[-1])
    lost_pct_2030 = float(lost_sales_pct[-1])

    st.markdown(
        rf"""
**Lettura economica (semplice)**
- **Margine lordo** (Fatturato − Costo energia): **{gross_margin_pct:.1f}%**
- **Margine EBITDA** (dopo OPEX): **{ebitda_margin_pct:.1f}%**
- **CAPEX** è ciò che “pesa” sul rientro (anni in cui installi nuove unità).

**Colli di bottiglia (proxy operativa)**
- Target saturazione: **{saturazione_target*100:.0f}%** — Saturazione 2030: **{saturazione_sessioni[-1]*100:.0f}%** → {sat_flag}
- Vendite perse stimate (solo se oltre 100% capacità): **{lost_pct_2030:.1f}%**, ~ **{lost_kwh_2030:,.0f} kWh** nel 2030

**Azioni tipiche**
- Se saturazione alta → aggiungi unità, estendi ore, aumenta potenza/turnover (riduci kWh medi), oppure passa a 60 kW.
- Se margini bassi → lavora su prezzo, costo energia (contratto), OPEX, o revenue add-on (bar/shop).
        """
    )

# ============================================================
# REPORT ANALITICO
# ============================================================
st.divider()
st.subheader("📊 Report Analitico: funnel, saturazione, ricavi e ritorno")

df_master = pd.DataFrame({
    "Anno": years,
    "BEV Palermo": bev_citta.astype(int),
    "Quota pubblica (%)": (public_share * 100),
    "Quota cattura (%)": (quota_stazione * 100).round(2),
    "Auto catturate": auto_clienti_anno.astype(int),
    "Energia (kWh)": energia_kwh.astype(int),
    "Sessioni/anno": sessioni_anno.astype(int),
    "Sessioni/giorno tot": sessioni_giorno_tot.round(1),
    "Unità tot": n_totale.astype(int),
    "Unità A": stazione_A.astype(int),
    "Unità B": stazione_B.astype(int),
    "Saturazione stimata (%)": (saturazione_sessioni * 100).round(1),
    "Fatturato (€)": ricavi.astype(int),
    "EBITDA (€)": ebitda.astype(int),
    "CAPEX (€)": capex_flow.astype(int),
    "CF netto (€)": cf_netto.astype(int),
    "CF cumulato (€)": cf_cum.astype(int),
}).set_index("Anno")

st.dataframe(df_master)

with st.expander("📚 Intelligence Report (spiegazioni + limiti del modello)", expanded=False):
    st.markdown(
        f"""
### Come usare questo tool in riunione (GM/CFO)
1. **Validare domanda**: BEV 2030, quota pubblica, quota cattura.
2. **Verificare colli di bottiglia**: target saturazione e unità richieste.
3. **Leggere il business**: fatturato, EBITDA, CAPEX, payback e ROI.

### Limiti e trasparenza (cosa NON include)
- Modello **top-down** e semplificato: non include stagionalità, picchi orari, downtime dettagliato, code reali, pricing dinamico.
- Conversione **Auto → kWh** usa **{kwh_annui_per_auto} kWh/auto/anno** (proxy): sostituiscilo con dato locale se disponibile.
- Cash flow usato per payback/ROI è **EBITDA − CAPEX**: non include tasse, working capital, canoni rete complessi, revenue sharing, fee roaming, ecc.
- **Break-even** usa ammortamento gestionale su **{ammortamento_anni} anni**: serve a capire la soglia “operativa”.

Se vuoi, si può aggiungere una modalità “CFO” con: tasse, WC, canoni potenza, fee roaming, scenari Base/Bear/Bull e sensitività (tornado).
        """
    )
