import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy_financial as npf

st.set_page_config(page_title="Executive Charging Suite", layout="wide")

st.title("🛡️ Executive Support System: eVFs for EOS")
st.markdown("### *Investment Readiness Tool: Trasformare i dati del parco auto in decisioni infrastrutturate bancabili.*")

# --- SIDEBAR: TUTTI I PARAMETRI RICHIESTI ---
st.sidebar.header("🕹️ Variabili di Mercato (Il tuo Scenario)")
with st.sidebar.expander("🌍 Dimensionamento Parco Auto", expanded=True):
    # Possibilità di cambiare il numero base di auto (BEV) previste
    bev_base_2030 = st.number_input("Target BEV Palermo al 2030", value=5000, step=100)
    stress_bev = st.slider("Moltiplicatore Stress Test (%)", 50, 150, 100, help="Simula scenari di adozione più lenti o più veloci.") / 100
    
    # Percentuale di ricarica pubblica (chi non carica a casa)
    public_share = st.slider("Quota Ricarica Pubblica (%)", 10, 80, 30, help="Percentuale di ricariche effettuate fuori casa/ufficio.") / 100

with st.sidebar.expander("🎯 Strategia di Cattura"):
    # Percentuale che si vuole catturare del mercato pubblico
    target_cattura_final = st.slider("Quota Cattura Target al 2030 (%)", 1.0, 15.0, 5.0, step=0.5) / 100
    stress_cattura = st.slider("Efficacia Operativa Stazione (%)", 50, 150, 100, help="Tua capacità di attrarre clienti rispetto ai competitor.") / 100

st.sidebar.header("⚙️ Configurazione Tecnica & Prezzi")
with st.sidebar.expander("🔧 Asset & Operatività"):
    tecnologia = st.selectbox("Tecnologia Asset", ["DC 30 kW", "DC 60 kW"])
    allocazione = st.radio("Strategia di espansione", ["Monosito (Tutto in A)", "Multisito (Espansione in B)"])
    ore_max_giorno = st.slider("Disponibilità operativa (ore/giorno)", 4, 12, 10)
    carica_media_sessione = st.number_input("Ricarica media singola (kWh)", value=35)

with st.sidebar.expander("💰 Financials"):
    prezzo_kwh = st.number_input("Prezzo vendita (€/kWh)", value=0.69)
    costo_kwh = st.number_input("Costo energia (€/kWh)", value=0.30)
    capex_30 = 25000
    capex_60 = 45000
    wacc = 0.08

# --- LOGICA DI CALCOLO DINAMICA ---
years = np.array([2026, 2027, 2028, 2029, 2030])
potenza_kw = 30 if tecnologia == "DC 30 kW" else 60
capex_unit = capex_30 if tecnologia == "DC 30 kW" else capex_60
opex_unit = 2000 if tecnologia == "DC 30 kW" else 3500

# Evoluzione Parco Auto (lineare verso il target impostato)
bev_citta = np.linspace(bev_base_2030 * 0.5, bev_base_2030, 5) * stress_bev

# Evoluzione Quota Cattura (dal 2% al target impostato)
quota_stazione = np.linspace(0.02, target_cattura_final, 5) * stress_cattura

# Calcolo Volumi
auto_clienti_anno = bev_citta * public_share * quota_stazione
energia_kwh = auto_clienti_anno * 3000 # Consumo annuo stimato per auto

# Capacità e Asset
ore_richieste = energia_kwh / potenza_kw
ore_disp_asset = ore_max_giorno * 365
n_totale = np.ceil(ore_richieste / ore_disp_asset).astype(int)

# Allocazione Stazioni
stazione_A = np.ones(len(years))
stazione_B = np.zeros(len(years))
for i, n in enumerate(n_totale):
    if allocazione == "Multisito (Espansione in B)" and n > 1:
        stazione_B[i] = n - 1
        stazione_A[i] = 1
    else:
        stazione_A[i] = n

# Finanza
ebitda = (energia_kwh * (prezzo_kwh - costo_kwh)) - (n_totale * opex_unit)
capex_flow = np.zeros(len(years)); prev_n = 0
for i, n in enumerate(n_totale):
    capex_flow[i] = max(0, n - prev_n) * capex_unit
    prev_n = n
cf_netto = ebitda - capex_flow
cf_cum = np.cumsum(cf_netto)

# --- VISUALIZZAZIONE KPI ---
st.subheader(f"💼 Analisi Strategica: {tecnologia} ({allocazione})")
k1, k2, k3, k4 = st.columns(4)
k1.metric("VAN (NPV)", f"€ {npf.npv(wacc, cf_netto):,.0f}")
k2.metric("TIR (IRR)", f"{(npf.irr(cf_netto)*100):.1f}%" if not np.isnan(npf.irr(cf_netto)) else "N/D")
k3.metric("Auto in Carica (2030)", f"{auto_clienti_anno[-1]:,.0f}")
k4.metric("Quota Mercato Totale (%)", f"{(target_cattura_final*100):.1f}%")

# --- GRAFICI E SPIEGAZIONI ---
st.divider()
c1, c2 = st.columns(2)

with c1:
    st.write("**1. Break-even: Auto/Giorno per coprire i costi**")
    auto_range = np.linspace(1, 20, 20)
    margine_carica = carica_media_sessione * (prezzo_kwh - costo_kwh)
    costo_fisso_day = (opex_unit + (capex_unit/5)) / 365
    fig1, ax1 = plt.subplots()
    ax1.plot(auto_range, (auto_range * margine_carica) - costo_fisso_day, color='green', linewidth=3)
    ax1.axhline(0, color='red', linestyle='--')
    ax1.set_xlabel("Ricariche giornaliere per singola unità")
    st.pyplot(fig1)
    st.markdown(r"""
    **Formula:** $N_{auto} = \frac{OPEX_{day} + Ammortamento_{day}}{kWh_{sessione} \cdot (P_{vendita} - C_{energia})}$  
    *Spiegazione:* Il punto di pareggio dipende da quante auto "catturi" ogni giorno. Se la quota pubblica cala, dovrai aumentare l'efficacia di cattura per restare sopra la linea rossa.
    """)

with c2:
    st.write("**2. Cash Flow Cumulato (Recupero Investimento)**")
    fig2, ax2 = plt.subplots()
    ax2.plot(years, cf_cum, marker='o', color='navy', linewidth=3)
    ax2.fill_between(years, cf_cum, 0, where=(cf_cum >= 0), color='green', alpha=0.1)
    ax2.axhline(0, color='black', linewidth=1)
    st.pyplot(fig2)
    st.markdown(r"""
    **Formula:** $CF_{cum} = \sum (EBITDA - CAPEX)$  
    *Spiegazione:* Se riduci il numero di auto totali o la quota di ricarica pubblica, osserverai la curva appiattirsi o il punto di recupero (Payback) spostarsi verso destra.
    """)

# --- TABELLA ANALITICA CON TUTTE LE VARIABILI ---
st.divider()
st.subheader("📊 Report Analitico: Funnel di Conversione e Saturazione")
df_master = pd.DataFrame({
    "Anno": years,
    "BEV Palermo (Totali)": bev_citta.astype(int),
    "Domanda Pubblica (%)": [(public_share * 100)] * 5,
    "Tua Quota Cattura (%)": (quota_stazione * 100).round(2),
    "Clienti Catturati (Auto/y)": auto_clienti_anno.astype(int),
    "Energia Tot (kWh)": energia_kwh.astype(int),
    "Ore Occupazione/y": ore_richieste.astype(int),
    "Asset Stazione A": stazione_A.astype(int),
    "Asset Stazione B": stazione_B.astype(int),
    "EBITDA (€)": ebitda.astype(int),
    "CF Cumulato (€)": cf_cum.astype(int)
}).set_index("Anno")
st.table(df_master)

with st.expander("🔍 Intelligence Report per il Board"):
    st.markdown(f"""
    - **Analisi del Funnel**: Su un parco auto di **{bev_citta[-1]:,.0f}** BEV a Palermo, solo il **{public_share*100:.0f}%** cerca ricarica pubblica. Di questi, la tua strategia mira a catturarne il **{target_cattura_final*100:.1f}%**.
    - **Efficienza Tecnica**: Ogni ricarica da {carica_media_sessione} kWh occupa la colonnina per **{carica_media_sessione/potenza_kw*60:.0f} minuti**.
    - **Risultato**: Al 2030 gestirai un flusso di **{auto_clienti_anno[-1]:,.0f} auto all'anno**, distribuite su {n_totale[-1]} asset.
    """)
