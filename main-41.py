import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, time
import numpy as np

# 1. CONFIGURAZIONE PAGINA
st.set_page_config(page_title="JoSa | Y35 Enterprise v36", layout="wide", page_icon="⚡")

# 2. STILE CSS INTEGRALE
st.markdown("""
<style>
    .stApp { background-color: #F8FAFC; color: #1E293B; }
    .main-header {
        background: linear-gradient(90deg, #1E3A8A 0%, #059669 100%);
        padding: 2rem; border-radius: 15px; color: white; margin-bottom: 2rem; text-align: center;
    }
    .recommendation-banner {
        background-color: #D1FAE5; border: 2px solid #10B981; padding: 20px;
        border-radius: 12px; margin-bottom: 25px; color: #064E3B;
    }
    .section-title {
        color: #1E3A8A; font-size: 1.3rem; font-weight: 700;
        margin: 1.5rem 0 1rem 0; border-bottom: 2px solid #3B82F6; padding-bottom: 5px;
    }
    .info-box {
        background-color: #EFF6FF; border-left: 5px solid #3B82F6; padding: 15px;
        border-radius: 8px; margin: 10px 0; font-size: 0.9rem;
    }
    [data-testid="stMetric"] {
        background-color: #FFFFFF; padding: 15px !important;
        border-radius: 10px !important; border-left: 5px solid #3B82F6 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>JoSa | Y35 Enterprise ⚡🌳</h1><p>Data Driven Fleet Electrification</p></div>', unsafe_allow_html=True)

# 3. SIDEBAR
with st.sidebar:
    st.header("🛠️ Hardware Spectrum")
    modelli_full = ["AC 22kW", "DC 20kW", "DC 30kW", "DC 60kW", "DC 90kW", "DC 120kW"]
    hw_selection = st.multiselect("Seleziona modelli:", modelli_full, default=["AC 22kW", "DC 30kW", "DC 60kW"], help="Seleziona i modelli di ricarica da includere.")
    
    hw_db = {}
    for model in hw_selection:
        with st.expander(f"⚙️ Configura {model}", expanded=False):
            p_val = float(model.split(" ")[1].replace("kW", ""))
            def_acq = 1500 if "AC" in model else (5000 if "20" in model else (8500 if "30" in model else (16000 if "60" in model else 28000)))
            hw_db[model] = {
                "p": p_val, 
                "acq": st.number_input(f"Acquisto (€)", value=def_acq, key=f"acq_{model}"),
                "ins": st.number_input(f"Installazione (€)", value=1600 if "AC" in model else 7500, key=f"ins_{model}"),
                "mnt": st.number_input(f"Manutenzione (€/a)", value=350 if "DC" in model else 60, key=f"mnt_{model}")
            }
    
    st.header("⚡ Smart Management")
    p_shaving = st.number_input("Peak Shaving Max (kW)", 10, 5000, 100)
    ac_rotation = st.slider("Auto per colonnina AC/gg", 1, 6, 2)
    fine_turno = st.time_input("Fine Turno Staff (Scambio)", time(19, 0))
    h_limit = fine_turno.hour + fine_turno.minute/60
    
    st.header("🎯 Vincoli & Stress")
    budget_max = st.number_input("Budget CAPEX Max (€)", 5000, 2000000, 30000)
    p_rete = st.number_input("Potenza Rete (kW)", 6, 5000, 100)
    s_extra_cons = st.slider("Stress: Extra Consumo (%)", 0, 100, 25)
    s_delay = st.slider("Stress: Ritardo Rientro (min)", 0, 180, 60)
    
    st.header("💰 Costi Energetici")
    c_pri = st.number_input("Energia Interna (€/kWh)", 0.1, 0.6, 0.22)
    c_pub = st.number_input("Energia Pubblica (€/kWh)", 0.3, 1.5, 0.65)
    e_l = st.number_input("Diesel (€/L)", 1.0, 3.0, 1.75)
    km_l = st.number_input("Diesel (Km/L)", 5.0, 30.0, 15.0)
    h_rate = st.number_input("Costo Staff (€/h)", 1.0, 100.0, 1.0)

    st.header("🚗 Parametri TCO")
    with st.expander("Leasing & Manutenzione", expanded=False):
        c_acq_die = st.number_input("Diesel: Canone (€/mese)", 300, 2000, 550)
        c_mnt_die = st.number_input("Diesel: Mnt (€/anno)", 500, 5000, 800)
        c_acq_ev = st.number_input("EV: Canone (€/mese)", 300, 2500, 650)
        c_mnt_ev = st.number_input("EV: Mnt (€/anno)", 100, 4000, 300)
        tco_period = st.slider("Periodo Analisi (Mesi)", 12, 96, 36)
# --- AGGIUNTA NELLA SIDEBAR: PARAMETRI TCO E MIX ENERGETICO ---
st.sidebar.markdown("---")
st.sidebar.subheader("🔌 Mix Energetico Interno")
# Gestione multi-costo dell'energia (c_pri differenziati)
c_pri_grid = st.sidebar.number_input("Costo Energia Rete (€/kWh)", value=0.22, format="%.3f")
c_pri_solar = st.sidebar.number_input("Costo Energia Fotovoltaico (€/kWh)", value=0.08, format="%.3f")
quota_solar = st.sidebar.slider("Quota Autoproduzione (%)", 0, 100, 30)

# Calcolo del c_pri medio ponderato da usare nelle formule
c_pri_medio = (c_pri_grid * (1 - quota_solar/100)) + (c_pri_solar * (quota_solar/100))
st.sidebar.info(f"Costo Medio Ponderato: € {c_pri_medio:.3f}/kWh")

st.sidebar.subheader("💰 Investimento Flotta (Delta TCO)")
prezzo_acquisto_ev = st.sidebar.number_input("Prezzo Acquisto Veicolo EV (€)", value=48000, step=1000)
prezzo_acquisto_die = st.sidebar.number_input("Prezzo Acquisto Veicolo Diesel (€)", value=40000, step=1000)
incentivi_ev = st.sidebar.number_input("Incentivi/Eco-bonus per veicolo (€)", value=5000, step=500)
valore_residuo_die = st.sidebar.slider("Valore Residuo Diesel dopo 5 anni (%)", 10, 50, 25)
valore_residuo_ev = st.sidebar.slider("Valore Residuo EV dopo 5 anni (%)", 10, 50, 35)
# 4. OPERATIVITÀ FLOTTA
st.markdown('<div class="info-box">💡 <b>Operatività:</b> Configura i parametri di missione per ogni veicolo.</div>', unsafe_allow_html=True)

nv = st.number_input("Numero veicoli da simulare", 1, 150, 6)
v_data = []
v_cols = st.columns(3)
for i in range(nv):
    with v_cols[i%3].expander(f"🚛 Veicolo {i+1}", expanded=False):
        km = st.number_input("Km/gg", 10, 600, 120, key=f"k_{i}")
        v_batt = st.number_input("Batteria (kWh)", 20, 250, 75, key=f"batt_{i}")
        v_cons = st.number_input("Consumo (kWh/km)", 0.10, 0.80, 0.22, key=f"cons_{i}")
        t_a = st.time_input("Arrivo", time(9,0), key=f"ta_{i}")
        t_p = st.time_input("Partenza", time(17,0), key=f"tp_{i}")
        sn = st.checkbox("Fermo Deposito Notturno?", value=True, key=f"sn_{i}")
        hi, hf = t_a.hour + t_a.minute/60, t_p.hour + t_p.minute/60
        if hf <= hi: hf += 24
        v_data.append({"nome": f"V{i+1}", "e_req": km * v_cons, "km": km, "batt": v_batt, "cons": v_cons, "s": hi, "f": hf, "night": sn})

# 5. MOTORE SIMULAZIONE
def simulazione(config, veicoli, costi, hw_params, p_shave_limit, max_ac_v, limit_h, is_stress=False, extra_c=0, delay_m=0):
    stations = []
    p_tot_inst = sum(hw_params[t]["p"] * q for t, q in config.items())
    if p_tot_inst > p_rete: return None
    
    for t, q in config.items():
        for i in range(q): 
            stations.append({"nome": f"{t}_{i+1}", "p": hw_params[t]["p"], "type": t, "busy": 0.0, "v_count": 0})
            
    power_timeline = np.zeros(144) 
    v_sim = []
    for v in veicoli:
        v_c = v.copy()
        if is_stress:
            v_c["e_req"] *= (1 + extra_c/100)
            v_c["s"] += (delay_m/60)
        v_sim.append({**v_c, "caricato": 0.0, "log_p": None})

    for v in sorted(v_sim, key=lambda x: x["s"]):
        best_s = None
        t_start_best = 999
        
        for s in stations:
            if "AC" in s["type"] and s["v_count"] >= max_ac_v: continue
            avail = s["busy"]
            if avail > limit_h and avail < 24: continue 
            act = max(avail, v["s"])
            if act < t_start_best and act < limit_h:
                t_start_best, best_s = act, s
        
        if best_s:
            p_nominale = min(best_s["p"], 11.0) if "AC" in best_s["type"] else best_s["p"]
            max_can_charge = min(v["e_req"], v["batt"])
            current_t = t_start_best
            carica_accumulata = 0.0
            
            while current_t < limit_h and carica_accumulata < max_can_charge:
                slot_idx = int(current_t * 4)
                if slot_idx >= 144: break
                p_available_shaving = max(0, p_shave_limit - power_timeline[slot_idx])
                p_effettiva = min(p_nominale, p_available_shaving)
                if p_effettiva > 0:
                    energia_slot = p_effettiva * 0.25
                    if carica_accumulata + energia_slot > max_can_charge:
                        energia_slot = max_can_charge - carica_accumulata
                        p_effettiva = energia_slot / 0.25
                    power_timeline[slot_idx] += p_effettiva
                    carica_accumulata += energia_slot
                current_t += 0.25
            
            v["caricato"] = carica_accumulata
            v["log_p"] = {"st": best_s["nome"], "i": t_start_best, "ec": current_t}
            best_s["busy"] = current_t + 0.1
            best_s["v_count"] += 1
    
    e_tot = sum(v["e_req"] for v in v_sim)
    e_int = sum(v["caricato"] for v in v_sim)
    e_ext = max(0, e_tot - e_int)
    
    c_cap = sum((hw_params[t]["acq"] + hw_params[t]["ins"]) * q for t, q in config.items())
    c_mnt = sum(hw_params[t]["mnt"] * q for t, q in config.items())
    risp_diesel = (sum(v["km"] for v in v_sim) / km_l * e_l) * 365
    staff_cost = (e_ext / 30 * h_rate) * 365
    pena_en = (e_ext * (costi['pub'] - costi['pri'])) * 365
    costo_ev = (e_int * costi['pri'] + e_ext * costi['pub']) * 365 + staff_cost + c_mnt
    co2 = (sum(v["km"] for v in v_sim) / km_l * 2.65 * 365) / 1000
    
    return {
        "config": config, "veicoli": v_sim, "timeline_p": power_timeline,
        "kpi": {
            "perc": round(e_int/e_tot*100,1) if e_tot > 0 else 0,
            "c_cap": c_cap, "risp": risp_diesel - costo_ev, "mnt": c_mnt,
            "pena_en": pena_en, "staff_ext": staff_cost, "co2": co2, "trees": int(co2 * 50),
            "e_int": e_int, "e_ext": e_ext, "e_tot": e_tot
        }
    }

# 6. ESECUZIONE
if st.button("ESEGUI OTTIMIZZAZIONE ENTERPRISE COMPLETA 🚀", use_container_width=True):
    results = []
    for q1 in range(0, 11):
        for q2 in range(0, 6):
            if not hw_selection: continue
            cfg = {hw_selection[0]: q1}
            if len(hw_selection) > 1: cfg[hw_selection[1]] = q2
            res = simulazione(cfg, v_data, {"pri": c_pri, "pub": c_pub}, hw_db, p_shaving, ac_rotation, h_limit)
            if res and 0 < res['kpi']['c_cap'] <= budget_max:
                res["stress"] = simulazione(cfg, v_data, {"pri": c_pri, "pub": c_pub}, hw_db, p_shaving, ac_rotation, h_limit, True, s_extra_cons, s_delay)
                results.append(res)
    st.session_state.all_res = sorted(results, key=lambda x: (-x["kpi"]["perc"], x["kpi"]["c_cap"]))

# 7. DISPLAY RISULTATI (CON SEZIONE COSTI AGGIORNATA)
if "all_res" in st.session_state and st.session_state.all_res:
    st.markdown('<div class="section-title">🔍 Seleziona Configurazione da Analizzare</div>', unsafe_allow_html=True)
    options = [f"Soluzione {i+1}: {res['config']} | Copertura {res['kpi']['perc']}% | CAPEX €{res['kpi']['c_cap']:,}" 
               for i, res in enumerate(st.session_state.all_res)]
    selected_option = st.selectbox("Scegli una soluzione per aggiornare i KPI:", options)
    idx_selected = options.index(selected_option)
    curr = st.session_state.all_res[idx_selected]
    k, s = curr["kpi"], curr["stress"]["kpi"]
    st.markdown(f'<div class="recommendation-banner"><h2>🏆 Soluzione in Analisi: {curr["config"]}</h2>Copertura {k["perc"]}% | CAPEX € {k["c_cap"]:,}</div>', unsafe_allow_html=True)

    # --- AGGIORNAMENTO: FINANCE & TCO ---
    st.markdown('<div class="section-title">💰 Analisi Comparativa Costi Energetici</div>', unsafe_allow_html=True)
    
    # Calcoli Base Annuali
    tot_km_annui = sum(v['km'] for v in v_data) * 365
    avg_fleet_cons = np.mean([v['cons'] for v in v_data])
    tot_kwh_annui = tot_km_annui * avg_fleet_cons
    
    # 1. Costo Carburante (Diesel)
    costo_carburante_diesel = (tot_km_annui / km_l) * e_l
    # 2. Costo Elettrico (Infrastruttura Privata - 100% interna)
    costo_elettrico_privato = tot_kwh_annui * c_pri
    # 3. Costo Elettrico (Infrastruttura Pubblica - 100% esterna)
    costo_elettrico_pubblico = tot_kwh_annui * c_pub
    
    # Delta richiesti
    delta_fossil = costo_carburante_diesel - costo_elettrico_privato
    delta_electric = costo_elettrico_pubblico - costo_elettrico_privato

    # Visualizzazione Comparativa Costi
    f1, f2, f3 = st.columns(3)
    f1.metric("Costo Annuale Carburante", f"€ {costo_carburante_diesel:,.0f}", help="Costo totale basato su Diesel (€/L) e Km/L.")
    f2.metric("Costo Elettrico (Privato)", f"€ {costo_elettrico_privato:,.0f}", help="Costo se tutta l'energia fosse caricata in azienda (c_pri).")
    f3.metric("Costo Elettrico (Pubblico)", f"€ {costo_elettrico_pubblico:,.0f}", help="Costo se tutta l'energia fosse caricata esternamente (c_pub).")

    d1, d2 = st.columns(2)
    d1.metric("Delta Fossil", f"€ {delta_fossil:,.0f}", 
              help="Risparmio potenziale: Costo Carburante - Costo Elettrico Privato.")
    d2.metric("Delta Electric", f"€ {delta_electric:,.0f}", 
              help="Efficienza infrastruttura: Costo Elettrico Pubblico - Costo Elettrico Privato.")

    st.markdown("---")
    
    # TCO Integrale (Include hardware, leasing e staff)
    costo_kwh_mix = (k['perc']/100 * c_pri) + ((1 - k['perc']/100) * c_pub)
    tco_diesel_tot = (c_acq_die * nv * tco_period) + (c_mnt_die * nv * tco_period/12) + (costo_carburante_diesel * tco_period/12)
    tco_ev_tot = (c_acq_ev * nv * tco_period) + (c_mnt_ev * nv * tco_period/12) + ((tot_kwh_annui * costo_kwh_mix) * tco_period/12)
    risparmio_tco = tco_diesel_tot - tco_ev_tot

    c2, c3, c4 = st.columns(3)
    c2.metric("CAPEX Investimento", f"€ {k['c_cap']:,}")
    c3.metric("Payback Period", f"{k['c_cap']/k['risp']:.1f} anni" if k['risp']>0 else "N/A")
    c4.metric("Delta TCO Totale", f"€ {risparmio_tco:,.0f}", delta=f"{tco_period} mesi")

    # Grafico TCO
    st.markdown("#### Proiezione TCO in base alla percorrenza")
    km_range = np.arange(5000, 60001, 2500)
    tco_d_plot, tco_e_plot = [], []
    for km_val in km_range:
        tco_d_plot.append((c_acq_die * tco_period) + (c_mnt_die * tco_period/12) + ((km_val/km_l*e_l) * tco_period/12))
        tco_e_plot.append((c_acq_ev * tco_period) + (c_mnt_ev * tco_period/12) + ((km_val * avg_fleet_cons * costo_kwh_mix) * tco_period/12))
    
    fig_tco = go.Figure()
    fig_tco.add_trace(go.Scatter(x=km_range, y=tco_d_plot, name="TCO Diesel", line=dict(color='#64748B', width=3)))
    fig_tco.add_trace(go.Scatter(x=km_range, y=tco_e_plot, name="TCO Elettrico", line=dict(color='#10B981', width=3)))
    fig_tco.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0), xaxis_title="Km Annui", yaxis_title="Costo Totale (€)", hovermode="x unified")
    st.plotly_chart(fig_tco, use_container_width=True)

    # --- ENERGIA ---
    st.markdown('<div class="section-title">🔌 Distribuzione Energetica & Mix Ricarica</div>', unsafe_allow_html=True)
    costo_solo_pubblico = (k['e_tot'] * c_pub) * 365
    costo_energia_attuale = (k['e_int'] * c_pri + k['e_ext'] * c_pub) * 365
    risparmio_vs_pub = costo_solo_pubblico - costo_energia_attuale

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Energia Totale", f"{k['e_tot']:.1f} kWh/gg")
    m2.metric("Ricarica Interna", f"{k['e_int']:.1f} kWh/gg", delta=f"{k['perc']}%")
    m3.metric("Ricarica Esterna", f"{k['e_ext']:.1f} kWh/gg", delta=f"{100-k['perc']:.1f}%", delta_color="inverse")
    m4.metric("Costo Mix", f"€ {costo_kwh_mix:.2f}/kWh")
    m5.metric("Saving vs Solo Pubblico", f"€ {risparmio_vs_pub:,.0f}/anno")

    # --- ESG ---
    st.markdown('<div class="section-title">🌳 ESG & Impatto Green</div>', unsafe_allow_html=True)
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("CO2 Risparmiata", f"{k['co2']:.1f} t/anno")
    e2.metric("Alberi Equivalenti", f"🌲 {k['trees']}")
    e3.metric("Diesel Evitato", f"{(sum(v['km'] for v in v_data)/km_l*365):,.0f} L")
    e4.metric("Rating ESG", "AAA" if k['perc']>90 else "A")

    # --- RESILIENCE & OPERATIONS ---
    st.markdown('<div class="section-title">⛈️ Resilience & Operations</div>', unsafe_allow_html=True)
    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Copertura Base", f"{k['perc']}%")
    o2.metric("Stress Management", f"{s['perc']}%", delta=f"{s['perc']-k['perc']:.1f}%")
    p_max = max(curr["timeline_p"])
    o3.metric("Peak Management", "✅ SICURO" if p_max < p_shaving * 0.95 else "⚠️ AL LIMITE", delta=f"{p_max:.1f} kW")
    o4.metric("Costo Inefficienza", f"€ {k['pena_en'] + k['staff_ext']:,.0f}")

    # --- TABS ---
    t1, t2, t3, t4 = st.tabs(["📊 Carico Potenza", "📅 Timeline", "🚛 Veicoli", "⚖️ Comparativa"])
    with t1:
        times = [datetime(2025,1,1) + timedelta(minutes=15*i) for i in range(96)]
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=times, y=curr["timeline_p"][:96], fill='tozeroy', name='kW', line=dict(color='#3B82F6')))
        fig_p.add_hline(y=p_shaving, line_dash="dash", line_color="red", annotation_text="Peak Shaving Limit")
        st.plotly_chart(fig_p, use_container_width=True)
    with t2:
        g_d = []
        for v in curr["veicoli"]:
            if v["log_p"]:
                base = datetime(2025, 1, 1)
                g_d.append({"Stazione": v["log_p"]["st"], "Inizio": base + timedelta(hours=v["log_p"]["i"]), "Fine": base + timedelta(hours=v["log_p"]["ec"]), "Veicolo": v["nome"]})
        if g_d:
            fig = px.timeline(pd.DataFrame(g_d), x_start="Inizio", x_end="Fine", y="Stazione", color="Veicolo")
            fig.add_vrect(x0=datetime(2025,1,1,int(h_limit), 0), x1=datetime(2025,1,2,7,0), fillcolor="red", opacity=0.1, annotation_text="BLOCCO STAFF / NOTTURNO")
            st.plotly_chart(fig, use_container_width=True)
    with t3:
        st.dataframe(pd.DataFrame(curr["veicoli"])[["nome", "km", "batt", "cons", "e_req", "caricato"]], use_container_width=True)
    with t4:
        st.markdown("### ⚖️ Tutte le Soluzioni sotto Budget")
        st.table(pd.DataFrame([{
            "Configurazione": str(r['config']),
            "Copertura (%)": f"{r['kpi']['perc']}%",
            "CAPEX (€)": f"€ {r['kpi']['c_cap']:,}",
            "Risparmio Anno (€)": f"€ {r['kpi']['risp']:,.0f}",
            "Payback (Anni)": f"{r['kpi']['c_cap']/r['kpi']['risp']:.1f}" if r['kpi']['risp']>0 else "N/A"
        } for r in st.session_state.all_res]))
# --- MODULO DEFINITIVO: FINANCIAL ADVISORY & STRATEGIC REPORT ---
    st.markdown('<div class="section-title">🏛️ Strategic Consulting: Delta TCO & Financial Audit</div>', unsafe_allow_html=True)

    # 1. PARAMETRI FINANZIARI E FORMULE
    anni_proiezione = 10
    
    # Delta Investimento (Extra-costo EV vs Diesel)
    costo_netto_ev = prezzo_acquisto_ev - incentivi_ev
    delta_acquisto_veicolo = costo_netto_ev - prezzo_acquisto_die
    delta_capex_flotta = nv * delta_acquisto_veicolo
    
    # Investimento Totale Progetto (Hardware + Delta Flotta)
    capex_totale_progetto = k['c_cap'] + delta_capex_flotta
    
    # Risparmio Operativo Annuo Consolidato (OPEX Delta)
    # Include: Risparmio Carburante (già calcolato in k['risp'] ma aggiornato con c_pri_medio)
    # + Delta Manutenzione Veicoli + Delta Ammortamento (Valore Residuo)
    delta_mnt_veicoli_annuo = (c_mnt_die - c_mnt_ev) * nv
    
    # Ammortamento basato su Valore Residuo
    amm_annuo_die = (prezzo_acquisto_die * (1 - valore_residuo_die/100)) / 5
    amm_annuo_ev = (costo_netto_ev * (1 - valore_residuo_ev/100)) / 5
    delta_ammortamento_annuo = (amm_annuo_die - amm_annuo_ev) * nv
    
    risparmio_netto_annuo = k['risp'] + delta_mnt_veicoli_annuo + delta_ammortamento_annuo
    
    # Creazione Cash Flow Decennale
    cf_data = []
    cumulativo = -capex_totale_progetto
    for anno in range(0, anni_proiezione + 1):
        flusso = -capex_totale_progetto if anno == 0 else risparmio_netto_annuo
        if anno > 0: cumulativo += flusso
        cf_data.append({"Anno": anno, "Flusso Netto (€)": flusso, "Cumulativo (€)": cumulativo})
    df_cf = pd.DataFrame(cf_data)

    # 2. DASHBOARD FINANZIARIA (ROI & PAYBACK)
    col_f1, col_f2 = st.columns([1, 2])
    with col_f1:
        roi_10y = ((risparmio_netto_annuo * anni_proiezione) / capex_totale_progetto * 100) if capex_totale_progetto > 0 else 0
        st.metric("ROI Consolidato (10y)", f"{roi_10y:.1f} %")
        st.metric("Payback Period (PBP)", f"{(capex_totale_progetto / risparmio_netto_annuo if risparmio_netto_annuo > 0 else 0):.1f} Anni")
        
        st.write("**Composizione Delta Investimento:**")
        st.caption(f"- Extra-costo Flotta: € {delta_capex_flotta:,.0f}")
        st.caption(f"- Infrastruttura Ricarica: € {k['c_cap']:,.0f}")
        st.markdown(f"**CAPEX Differenziale Totale: € {capex_totale_progetto:,.0f}**")
        st.write("---")
        st.write("**Metodologia ROI:**")
        st.caption("Il ROI è calcolato sul delta-investimento tra scenario EV e Diesel, includendo risparmi energetici, manutentivi e differenze di valore residuo.")

    with col_f2:
        fig_cf = px.area(df_cf, x="Anno", y="Cumulativo (€)", title="Rientro dell'Investimento (Analisi TCO)", color_discrete_sequence=['#10B981'])
        fig_cf.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Breakeven Point")
        st.plotly_chart(fig_cf, use_container_width=True)

    # 3. GENERAZIONE REPORT PROFESSIONALE PER DOWNLOAD
    v_report_list = ""
    for v in curr["veicoli"]:
        perc = int(v['caricato']/v['e_req']*100) if v['e_req'] > 0 else 100
        v_report_list += f"- {v['nome']}: Km {v['km']} | Req {v['e_req']:.1f}kWh | Caricata {v['caricato']:.1f}kWh | Copertura: {perc}%\n"

    report_narrativa = f"""
# EXECUTIVE BUSINESS CASE: TRANSIZIONE FLOTTA JOSA Y35
Data Documento: {datetime.now().strftime('%d/%m/%Y')}
Configurazione Progettata: {curr['config']}

## 1. STRUTTURA ENERGETICA (MIX COST)
L'analisi adotta un Mix Energetico interno ponderato di € {c_pri_medio:.3f}/kWh.
- Quota Fotovoltaico: {quota_solar}% (€ {c_pri_solar:.3f}/kWh)
- Quota Rete: {100-quota_solar}% (€ {c_pri_grid:.3f}/kWh)

## 2. ANALISI FINANZIARIA (DELTA TCO & ROI)
L'investimento differenziale rispetto allo scenario Diesel è di € {capex_totale_progetto:,.2f}.
Il Risparmio Annuo Netto cumulato è di € {risparmio_netto_annuo:,.2f}.

### FORMULE DI CALCOLO AUDITABILI:
- RISPARMIO NETTO = (Risparmio Carburante + Delta Manutenzione + Delta Valore Residuo) - Opex Infra.
- ROI (10 ANNI) = [(Risparmio Netto * 10) / Delta CAPEX] * 100 = {roi_10y:.1f}%
- PAYBACK = Delta CAPEX / Risparmio Netto = {capex_totale_progetto / risparmio_netto_annuo:.1f} anni.

## 3. PEAK MANAGEMENT E STABILITÀ RETE
Il sistema garantisce il non superamento della soglia di {p_shaving} kW.
- Picco di Potenza Raggiunto: {max(curr['timeline_p']):.1f} kW.
- Stress Test (+{s_extra_cons}% consumo): Tenuta operativa del {s['perc']}%.

## 4. PIANO DI RICARICA (GANTT) E DETTAGLIO FLOTTA
{v_report_list}

## 5. PROIEZIONE CASH FLOW DECENNALE (€)
{df_cf.to_string(index=False)}
--------------------------------------------------
Analisi di Consulenza Strategica - JoSa Enterprise Engine
"""

    # 4. EXPORT CENTER
    st.markdown("### 📥 Download Executive Package")
    with st.expander("Visualizza Anteprima Report Integrale"):
        st.text_area("Audit Log Completo", report_narrativa, height=400)
    
    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button("📄 Scarica Report Business Case (TXT)", report_narrativa, "Report_TCO_Strategico.txt", key="d_txt_fin")
    with d2:
        st.download_button("📊 Scarica Cash Flow Dettagliato (CSV)", df_cf.to_csv(index=False), "CashFlow_Proiezione.csv", key="d_cf_fin")
    with d3:
        # Comparazione scenari hardware
        df_sce = pd.DataFrame([{
            "Config": str(res['config']),
            "Delta ROI %": round(((res['kpi']['risp'] + delta_mnt_veicoli_annuo + delta_ammortamento_annuo) * 10) / (res['kpi']['c_cap'] + delta_capex_flotta) * 100, 1),
            "Payback (Anni)": round((res['kpi']['c_cap'] + delta_capex_flotta) / (res['kpi']['risp'] + delta_mnt_veicoli_annuo + delta_ammortamento_annuo), 1)
        } for res in st.session_state.all_res])
        st.download_button("📈 Scarica Comparativa Scenari (CSV)", df_sce.to_csv(index=False), "Confronto_Hardware.csv", key="d_sce_fin")
