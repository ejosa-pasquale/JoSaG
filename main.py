import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy_financial as npf

st.set_page_config(page_title="Executive Charging Suite (Full)", layout="wide")

# ------------------------------------------------------------
# LOGICA DI CALCOLO AVANZATA (DAL TUO MODELLO)
# ------------------------------------------------------------
def run_model(p):
    years = np.arange(2026, 2036)
    n = len(years)
    t = np.arange(n)
    
    # Prezzi & Costi indicizzati (Inflazione/Escalation)
    price_kwh = p["prezzo_kwh"] * ((1 + p["price_escal"]) ** t)
    cost_kwh = p["costo_kwh"] * ((1 + p["cost_escal"]) ** t)
    
    # 1. MODELLO DOMANDA TOP-DOWN (Basato su tuo PDF Palermo)
    bev_2030 = p["bev_target_2030"] * p["stress_bev"]
    bev_city = np.interp(years, [2026, 2030, 2035], 
                         [bev_2030 * 0.5, bev_2030, bev_2030 * 1.3])
    
    # Funnel di Conversione
    bev_captured = bev_city * p["public_share"] * p["capture_rate"] * p["stress_cattura"]
    energy_demand = bev_captured * 3000 * p["share_kwh_station"] # 3000 kWh/anno media
    sessions_demand = energy_demand / p["kwh_per_session"]
    
    # 2. CAPACITÀ FISICA E VINCOLI
    potenza_kw = 30 if p["tecnologia"] == "DC 30 kW" else 60
    session_min = (p["kwh_per_session"] / (potenza_kw * 0.75)) * 60 + 4
    cap_sessions_unit = ((p["ore_max"] * 60) / session_min) * 365 * p["uptime"] * p["util_target"]
    
    req_units = np.ceil(sessions_demand / cap_sessions_unit).astype(int)
    # Vincolo potenza di rete
    if p["power_limit"] > 0:
        max_by_power = int(p["power_limit"] // potenza_kw)
        units = np.clip(req_units, 1, max_by_power)
    else:
        units = np.clip(req_units, 1, 10)
        
    served_sessions = np.minimum(sessions_demand, units * cap_sessions_unit)
    lost_sessions = np.maximum(0, sessions_demand - served_sessions)
    served_energy = served_sessions * p["kwh_per_session"]
    
    # 3. FINANCIALS (NPV/IRR/CASH FLOW)
    ebitda = (served_energy * (price_kwh - cost_kwh)) - (units * 4500) # 4500€ OPEX unitario medio
    
    capex = np.zeros(n)
    prev_u = 0
    for i in range(n):
        if units[i] > prev_u:
            capex[i] = (units[i] - prev_u) * p["capex_unit"]
            if i == 0: capex[i] += 25000 # Costi fissi allaccio primo anno
        prev_u = units[i]
        
    cf_netto = ebitda - (np.maximum(0, ebitda) * 0.28) - capex
    
    return {
        "df": pd.DataFrame({"Units": units, "Energy": served_energy, "Lost": lost_sessions * p["kwh_per_session"], "EBITDA": ebitda, "CF": cf_netto, "CUM": np.cumsum(cf_netto)}, index=years),
        "npv": npf.npv(0.08, cf_netto),
        "irr": npf.irr(cf_netto),
        "total_lost_energy": np.sum(lost_sessions * p["kwh_per_session"])
    }

# ------------------------------------------------------------
# INTERFACCIA STREAMLIT
# ------------------------------------------------------------
st.sidebar.header("🎯 Driver Strategici")
with st.sidebar.expander("Mercato Palermo", expanded=True):
    bev_target = st.number_input("Target BEV 2030", value=5000)
    pub_share = st.slider("Quota ricarica pubblica %", 10, 80, 30) / 100
    cap_rate = st.slider("Tua quota cattura %", 1.0, 15.0, 5.0) / 100
    stress_cattura = st.slider("Stress test competizione %", 50, 150, 100) / 100

with st.sidebar.expander("Tecnica & Vincoli"):
    tec = st.selectbox("Tecnologia", ["DC 30 kW", "DC 60 kW"])
    p_limit = st.number_input("Limite Potenza Rete (kW) [0=No limit]", value=100)
    util_target = st.slider("Target Utilizzo (evita code) %", 50, 95, 80) / 100

with st.sidebar.expander("Finanza"):
    p_kwh = st.number_input("Prezzo vendita €/kWh", value=0.69)
    c_kwh = st.number_input("Costo energia €/kWh", value=0.30)
    capex_u = 25000 if tec == "DC 30 kW" else 45000

# Esecuzione
params = {
    "bev_target_2030": bev_target, "public_share": pub_share, "capture_rate": cap_rate,
    "stress_bev": 1.0, "stress_cattura": stress_cattura/100, "share_kwh_station": 0.25,
    "tecnologia": tec, "power_limit": p_limit, "util_target": util_target,
    "prezzo_kwh": p_kwh, "costo_kwh": c_kwh, "capex_unit": capex_u,
    "price_escal": 0.0, "cost_escal": 0.0, "uptime": 0.97, "ore_max": 12, "kwh_per_session": 35
}
out = run_model(params)
df = out["df"]

# --- DASHBOARD ---
st.subheader("📊 Executive Summary")
m1, m2, m3, m4 = st.columns(4)
m1.metric("VAN (NPV)", f"€ {out['npv']:,.0f}".replace(",", "."))
m2.metric("TIR (IRR)", f"{out['irr']*100:.1f}%")
m3.metric("Payback", "2028" if out['npv'] > 0 else "N/A")
m4.metric("Energia Persa (Saturazione)", f"{out['total_lost_energy']:,.0f} kWh")

st.divider()

# GRAFICI GESTIONALI
c1, c2 = st.columns(2)

with c1:
    st.write("**Domanda Servita vs Vendite Perse**")
    fig, ax = plt.subplots(figsize=(8,4))
    ax.stackplot(df.index, df["Energy"], df["Lost"], labels=['Energia Venduta', 'Vendite Perse (Coda)'], colors=['#2ecc71', '#e74c3c'], alpha=0.7)
    ax.legend(loc='upper left')
    ax.set_ylabel("kWh")
    st.pyplot(fig)
    st.caption("Se l'area rossa cresce, devi aumentare la potenza di rete o il numero di colonnine.")

with c2:
    st.write("**Rientro dell'Investimento (Cash Flow Cumulato)**")
    fig2, ax2 = plt.subplots(figsize=(8,4))
    colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in df["CUM"]]
    ax2.bar(df.index, df["CUM"], color=colors, alpha=0.6)
    ax2.axhline(0, color='black', linewidth=1)
    st.pyplot(fig2)
    st.caption("Il punto in cui le barre diventano verdi indica l'anno di reale profitto netto.")

st.divider()

# ANALISI DI SENSITIVITÀ (TORNADO)
st.write("**Analisi di Sensitività (Tornado Chart)**")
# Calcolo rapido per il grafico
sens_data = {
    "Prezzo Vendita (+10%)": run_model({**params, "prezzo_kwh": p_kwh*1.1})["npv"],
    "Costo Energia (+10%)": run_model({**params, "costo_kwh": c_kwh*1.1})["npv"],
    "Cattura Mercato (-20%)": run_model({**params, "stress_cattura": (stress_cattura*0.8)/100})["npv"],
    "CAPEX (+20%)": run_model({**params, "capex_unit": capex_u*1.2})["npv"]
}
fig3, ax3 = plt.subplots(figsize=(8, 4))
keys = list(sens_data.keys())
values = list(sens_data.values())
ax3.barh(keys, [v - out["npv"] for v in values], color='#3498db')
ax3.axvline(0, color='black', linewidth=1)
st.pyplot(fig3)
st.info("Questo grafico mostra quale variabile 'sposta' di più il risultato economico. Utile per decidere dove focalizzare il marketing o le trattative.")

# TABELLA DETTAGLIATA
st.write("**Dettaglio Analitico 10 Anni**")
st.dataframe(df.style.format("{:,.0f}"))
