import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy_financial as npf # Necessario per TIR e VAN

st.set_page_config(page_title="Investment Tool DC30", layout="wide")

st.title("🛡️ Investment Decision Tool: DC 30 kW Palermo")
st.markdown("""
Questo tool valuta la bancabilità del progetto. 
Oltre ai dati operativi, include indicatori finanziari (VAN, TIR) e stress test sulla domanda.
""")

# ---------------------------------------------------------
# 1) SIDEBAR - PARAMETRI E STRESS TEST
# ---------------------------------------------------------
st.sidebar.header("🕹️ Pannello di Controllo")

with st.sidebar.expander("📊 Stress Test: Parco Auto & Domanda", expanded=True):
    # Stress test sul parco auto (Moltiplicatore)
    stress_bev = st.slider("Moltiplicatore Parco BEV", 0.5, 1.5, 1.0, 
                           help="Simula uno scenario con più o meno auto elettriche rispetto alle stime ufficiali.")
    
    # Stress test sulla capacità di cattura
    stress_quota = st.slider("Efficacia Cattura (%)", 50, 150, 100, 
                             help="Varia la quota di mercato intercettata dalla stazione (es. causa concorrenza).") / 100

with st.sidebar.expander("💸 Parametri Finanziari"):
    costo_capitale = st.slider("Tasso di Attualizzazione (WACC %)", 3, 15, 8) / 100
    prezzo_vendita = st.number_input("Prezzo vendita (€/kWh)", value=0.69)
    costo_energia = st.number_input("Costo energia (€/kWh)", value=0.30)
    capex_unitario = st.slider("CAPEX per Colonnina (€)", 20000, 30000, 25000)

# ---------------------------------------------------------
# 2) LOGICA DI CALCOLO AVANZATA
# ---------------------------------------------------------
years = np.array([2026, 2027, 2028, 2029, 2030])

# Dati base con applicazione STRESS TEST
bev_base = np.array([2600, 3000, 3500, 4200, 5000]) * stress_bev [cite: 28]
quota_stazione_base = np.array([0.02, 0.03, 0.04, 0.045, 0.05]) * stress_quota [cite: 31]

# Calcolo operativo
capacita_unitaria = 30 * 8760 * 0.97 * 0.30 # [cite: 39]
energia_kwh = bev_base * 3000 * 0.30 * quota_stazione_base # [cite: 53]
n_colonnine = np.ceil(energia_kwh / capacita_unitaria).astype(int) [cite: 54]

# Flussi di Cassa
ebitda = (energia_kwh * (prezzo_vendita - costo_energia)) - (n_colonnine * 2000) [cite: 83]

capex_flow = np.zeros(len(years))
prev_n = 0
for i, n in enumerate(n_colonnine):
    capex_flow[i] = max(0, n - prev_n) * capex_unitario
    prev_n = n

cf_netto = ebitda - capex_flow
cf_cumulato = np.cumsum(cf_netto)

# INDICATORI FINANZIARI (Decision Making)
van = npf.npv(costo_capitale, cf_netto)
tir = npf.irr(cf_netto)
ebitda_margin = (np.sum(ebitda) / np.sum(energia_kwh * prezzo_vendita)) * 100

# ---------------------------------------------------------
# 3) DASHBOARD DECISIONALE
# ---------------------------------------------------------

# Riquadro semaforico per decisione
st.subheader("🏁 Verdetto sull'Investimento")
dec_col1, dec_col2, dec_col3 = st.columns(3)

with dec_col1:
    if van > 0:
        st.success(f"**VAN POSITIVO: € {van:,.0f}**")
        st.caption("Il progetto crea valore sopra il costo del capitale scelto.")
    else:
        st.error(f"**VAN NEGATIVO: € {van:,.0f}**")
        st.caption("Il progetto non copre il costo del capitale. Rischio alto.")

with dec_col2:
    tir_display = f"{tir*100:.2f}%" if not np.isnan(tir) else "N/D"
    st.metric("TIR (Tasso di Rendimento)", tir_display)
    st.caption("Confrontalo con il rendimento di altri investimenti (es. BTP o borsa).")

with dec_col3:
    payback_idx = np.where(cf_cumulato >= 0)[0]
    payback = years[payback_idx[0]] if len(payback_idx) > 0 else "Non rientra"
    st.metric("Anno di Payback", payback)

# Tabelle e Stress Test
st.divider()
col_left, col_right = st.columns([2, 1])

with col_left:
    st.write("**Tabella di Marcia Finanziaria**")
    df = pd.DataFrame({
        "Anno": years,
        "BEV (Stressed)": bev_base.astype(int),
        "Energia (kWh)": energia_kwh.astype(int),
        "EBITDA (€)": ebitda.astype(int),
        "Cash Flow Netto (€)": cf_netto.astype(int),
        "Cash Flow Cumulato (€)": cf_cumulato.astype(int)
    }).set_index("Anno")
    st.dataframe(df, use_container_width=True)

with col_right:
    st.write("**Efficienza Operativa**")
    st.info(f"""
    - **Margine Operativo Medio**: {ebitda_margin:.1f}% [cite: 112]
    - **Saturazione Media**: 30.0% [cite: 35]
    - **CAPEX Totale**: € {np.sum(capex_flow):,.0f} [cite: 116]
    - **ROI Cumulato**: {(cf_cumulato[-1]/np.sum(capex_flow)):.2f}x [cite: 114]
    """)

# Grafico Stress Test
st.subheader("📉 Sensibilità del Cash Flow Cumulato")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(years, cf_cumulato, marker='o', color='navy', linewidth=3)
ax.fill_between(years, cf_cumulato, color='blue', alpha=0.1)
ax.axhline(0, color='red', linestyle='--')
ax.set_title("Andamento del Rientro Economico (Scenario Stressato)")
ax.grid(True, alpha=0.2)
st.pyplot(fig)
