# Run: streamlit run dashboard/app.py
# Requires: pip install plotly

import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(
    page_title='EV Battery Early Warning System',
    page_icon='⚡',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ── Models & Data (cached) ────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    return (
        joblib.load('pipeline/classifier.pkl'),
        joblib.load('pipeline/regressor.pkl'),
        joblib.load('pipeline/label_encoder.pkl'),
        joblib.load('pipeline/scaler.pkl'),
    )

@st.cache_data
def load_data():
    df = pd.read_csv('nasa_processed.csv')
    try:
        dcir = pd.read_csv('data/dcir_all_cells.csv')
    except FileNotFoundError:
        dcir = pd.DataFrame(columns=['cell', 'cycle', 'dcir'])
    return df, dcir

clf, reg, le, scaler = load_models()
df, dcir_df = load_data()

FEATURES = ['Re', 'Rct', 'total_impedance', 'capacity_fade',
            'cumulative_fade', 'ambient_temperature', 'gas_ppm', 'smoke_density']

TIER_CFG = {
    'long':  dict(color='#059669', cls='teal',  icon='✅', label='HEALTHY',
                  msg='Battery health is nominal. All parameters within safe operating range.'),
    'mid':   dict(color='#d97706', cls='amber', icon='⚠️', label='MODERATE RISK',
                  msg='Elevated degradation signals detected. Schedule preventive maintenance.'),
    'short': dict(color='#dc2626', cls='red',   icon='🚨', label='CRITICAL',
                  msg='Battery failure imminent — remove from service immediately.'),
}

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root {
    --teal:  #0f766e;
    --amber: #d97706;
    --red:   #dc2626;
    --ink:   #0f172a;
    --muted: #64748b;
    --bdr:   #e2e8f0;
    --bg:    #f1f5f9;
    --white: #ffffff;
}

html, body, .stApp { background: var(--bg) !important; }
.block-container { padding: 1.25rem 2rem 2.5rem !important; max-width: 1380px; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #0f766e 0%, #115e59 45%, #0c3333 100%);
    border-radius: 20px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.75rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(15,118,110,.22);
}
.hero::before {
    content: ''; position: absolute;
    top: -60px; right: -60px;
    width: 240px; height: 240px;
    border-radius: 50%; background: rgba(255,255,255,.05);
}
.hero::after {
    content: ''; position: absolute;
    bottom: -80px; left: 35%;
    width: 300px; height: 300px;
    border-radius: 50%; background: rgba(255,255,255,.03);
}
.hero-badge {
    display: inline-flex; align-items: center; gap: .35rem;
    background: rgba(255,255,255,.15); border: 1px solid rgba(255,255,255,.22);
    border-radius: 999px; padding: .28rem .9rem;
    font-size: .78rem; font-weight: 600; color: rgba(255,255,255,.95);
    margin-bottom: .7rem; backdrop-filter: blur(8px);
}
.hero-title {
    font-size: 1.9rem; font-weight: 800; color: #fff;
    letter-spacing: -.025em; line-height: 1.2; margin: 0 0 .45rem;
}
.hero-sub {
    font-size: .95rem; color: rgba(255,255,255,.82);
    line-height: 1.65; margin: 0; max-width: 620px;
}

/* ── Section headers ── */
.sec-hdr { display: flex; align-items: center; gap: .65rem; margin: 0 0 1rem; }
.sec-icon {
    width: 38px; height: 38px; border-radius: 11px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem; flex-shrink: 0;
}
.sec-title { font-size: 1.05rem; font-weight: 700; color: var(--ink); margin: 0; }
.sec-sub   { font-size: .82rem; color: var(--muted); margin: .1rem 0 0; }

/* ── Metric cards ── */
.mcard {
    background: var(--white);
    border: 1px solid var(--bdr);
    border-radius: 16px;
    padding: 1.1rem 1.2rem;
    box-shadow: 0 4px 20px rgba(15,23,42,.05);
}
.mcard-teal  { border-left: 4px solid var(--teal); }
.mcard-amber { border-left: 4px solid var(--amber); }
.mcard-red   { border-left: 4px solid var(--red); }
.mcard-lbl   { font-size: .73rem; font-weight: 700; text-transform: uppercase; letter-spacing: .1em; color: var(--muted); margin-bottom: .4rem; }
.mcard-val   { font-size: 2.1rem; font-weight: 800; line-height: 1; }
.mcard-sub   { font-size: .82rem; color: var(--muted); margin-top: .35rem; }

/* ── Status banner ── */
.sbanner {
    display: flex; align-items: flex-start; gap: .75rem;
    border-radius: 12px; padding: .9rem 1.1rem;
    margin: .65rem 0 .25rem; font-size: .9rem;
}
.sbanner.ok   { background: #f0fdf4; border: 1.5px solid #86efac; color: #166534; }
.sbanner.warn { background: #fffbeb; border: 1.5px solid #fcd34d; color: #7c2d12; }
.sbanner.crit { background: #fef2f2; border: 1.5px solid #fca5a5; color: #991b1b;
                animation: pulse-red 2s infinite; }
@keyframes pulse-red {
    0%,100% { box-shadow: 0 0 0 0 rgba(220,38,38,.25); }
    50%      { box-shadow: 0 0 0 6px rgba(220,38,38,0); }
}
.sbanner .sbi { font-size: 1.15rem; margin-top: .05rem; flex-shrink: 0; }

/* ── Sensor chips ── */
.sgrid {
    display: grid; grid-template-columns: repeat(4,1fr); gap: .6rem; margin: .5rem 0 .25rem;
}
.schip {
    background: var(--bg); border: 1px solid var(--bdr);
    border-radius: 10px; padding: .6rem .75rem;
}
.schip-lbl {
    font-size: .71rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: .07em; color: var(--muted);
}
.schip-val { font-size: 1.05rem; font-weight: 700; color: var(--ink); margin-top: .15rem; }

/* ── Tabs ── */
div[data-baseweb="tab-list"] {
    background: var(--white) !important;
    border-radius: 12px 12px 0 0;
    border: 1px solid var(--bdr);
    border-bottom: none;
    padding: .35rem .5rem 0;
    gap: .25rem;
}
button[data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px 8px 0 0 !important;
    color: var(--muted) !important;
    font-weight: 600 !important;
    font-size: .9rem !important;
    padding: .55rem 1.1rem !important;
    border: none !important;
    transition: color .15s, background .15s;
}
button[data-baseweb="tab"]:hover {
    color: var(--ink) !important;
    background: var(--bg) !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--teal) !important;
    background: var(--bg) !important;
    border-bottom: 3px solid var(--teal) !important;
}
div[data-baseweb="tab-panel"] {
    background: var(--white);
    border: 1px solid var(--bdr);
    border-top: none;
    border-radius: 0 0 12px 12px;
    padding: 1.25rem 1rem 1rem;
}

/* ── Divider ── */
.sdivider { border: none; border-top: 1px solid var(--bdr); margin: 1.5rem 0; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #1e293b !important;
    border-right: 1px solid #334155 !important;
}
section[data-testid="stSidebar"] .block-container {
    padding: 1.25rem 1rem !important;
}
/* Force all text inside sidebar to be legible */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] .stRadio label p,
section[data-testid="stSidebar"] .stSelectbox label p,
section[data-testid="stSidebar"] .stSlider label p,
section[data-testid="stSidebar"] .stTextInput label p,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: #e2e8f0 !important;
}
/* Expander header */
section[data-testid="stSidebar"] details summary p,
section[data-testid="stSidebar"] details summary span,
section[data-testid="stSidebar"] [data-testid="stExpander"] summary span {
    color: #f1f5f9 !important;
    font-weight: 600 !important;
}
/* Expander background */
section[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: rgba(255,255,255,.05) !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
}
/* Slider value labels */
section[data-testid="stSidebar"] [data-testid="stTickBarMin"],
section[data-testid="stSidebar"] [data-testid="stTickBarMax"] {
    color: #94a3b8 !important;
}
/* Selectbox & text input background */
section[data-testid="stSidebar"] [data-baseweb="select"] > div,
section[data-testid="stSidebar"] [data-baseweb="input"] {
    background: #0f172a !important;
    border-color: #475569 !important;
    color: #f1f5f9 !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] [data-baseweb="tag"],
section[data-testid="stSidebar"] [data-baseweb="select"] span {
    color: #f1f5f9 !important;
}
/* Radio buttons */
section[data-testid="stSidebar"] [data-baseweb="radio"] label {
    color: #e2e8f0 !important;
}
/* HR divider inside sidebar */
section[data-testid="stSidebar"] hr {
    border-color: #334155 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def pick_value(ui, label, min_v, max_v, default_v, step, key, mode):
    if mode == 'Slider':
        return float(ui.slider(label, min_v, max_v, default_v, step=step, key=f'{key}_s'))
    raw = ui.text_input(label, value=str(default_v), key=f'{key}_m')
    try:
        v = float(raw)
    except ValueError:
        ui.warning(f'Invalid — using {default_v}')
        return float(default_v)
    if not (min_v <= v <= max_v):
        v = min(max(v, min_v), max_v)
        ui.warning(f'Clamped to [{min_v}, {max_v}]')
    return float(v)


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style='text-align:center;padding:.5rem 0 1rem;'>
  <div style='font-size:2rem;'>⚡</div>
  <div style='font-weight:800;font-size:.95rem;color:#0f172a;margin-top:.2rem;'>Simulation Controls</div>
  <div style='font-size:.78rem;color:#64748b;margin-top:.15rem;'>Adjust inputs to update diagnostics</div>
</div>
""", unsafe_allow_html=True)

input_mode = st.sidebar.radio('Input method', ['Slider', 'Manual Entry'], horizontal=True)
cell = st.sidebar.selectbox('Battery cell (trend charts)', df['battery_id'].unique())
st.sidebar.markdown("<hr style='border-color:#e2e8f0;margin:.75rem 0;'>", unsafe_allow_html=True)

elec_exp = st.sidebar.expander('⚗️  Electrochemical', expanded=True)
Re  = pick_value(elec_exp, 'Re — electrolyte resistance (Ω)', 0.04, 0.20,  0.07,  0.001, 're',   input_mode)
Rct = pick_value(elec_exp, 'Rct — charge transfer (Ω)',       0.05, 0.40,  0.15,  0.001, 'rct',  input_mode)

cap_exp = st.sidebar.expander('🔋  Capacity', expanded=True)
c_f   = pick_value(cap_exp, 'Capacity fade (Ah/cycle)', -0.05, 0.0,  -0.005, 0.001, 'cf',   input_mode)
c_cum = pick_value(cap_exp, 'Cumulative fade (Ah)',       0.0,  0.5,   0.1,   0.001, 'ccum', input_mode)

env_exp = st.sidebar.expander('🌡️  Environmental & Safety', expanded=True)
temp  = int(pick_value(env_exp, 'Temperature (°C)', 0,   60,  24,  1,     'temp',  input_mode))
gas   =     pick_value(env_exp, 'Gas level (ppm)',  150, 800, 220, 1,     'gas',   input_mode)
smoke =     pick_value(env_exp, 'Smoke density',    0.0, 0.3, 0.05, 0.001,'smoke', input_mode)


# ── Prediction ────────────────────────────────────────────────────────────────
row   = pd.DataFrame([[Re, Rct, Re+Rct, c_f, c_cum, temp, gas, smoke]], columns=FEATURES)
row_s = pd.DataFrame(scaler.transform(row), columns=FEATURES)
tier  = le.inverse_transform(clf.predict(row_s))[0]
rul   = max(0, round(reg.predict(row_s)[0]))
tc    = TIER_CFG[tier]


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <div class="hero-badge">⚡ EV Battery Analytics Platform</div>
  <h1 class="hero-title">Early Warning System</h1>
  <p class="hero-sub">
    Simulate sensor readings to estimate fault tier and Remaining Useful Life (RUL),
    then inspect SoH and DCIR degradation trends to monitor long-term ageing behaviour.
  </p>
</div>
""", unsafe_allow_html=True)


# ── Section 1 · Live Diagnostics ─────────────────────────────────────────────
st.markdown("""
<div class="sec-hdr">
  <div class="sec-icon" style="background:#ecfdf5;">🔍</div>
  <div>
    <p class="sec-title">Live Diagnostics</p>
    <p class="sec-sub">Real-time fault classification and RUL estimation from current sensor inputs</p>
  </div>
</div>
""", unsafe_allow_html=True)

col_cards, col_gauge = st.columns([3, 2])

with col_cards:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="mcard mcard-{tc['cls']}">
          <div class="mcard-lbl">Fault Tier</div>
          <div class="mcard-val" style="color:{tc['color']};">{tier.upper()}</div>
          <div class="mcard-sub">{tc['icon']} {tc['label']}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        rul_color = '#059669' if rul > 100 else '#d97706' if rul > 30 else '#dc2626'
        st.markdown(f"""
        <div class="mcard">
          <div class="mcard-lbl">Cycles Remaining (RUL)</div>
          <div class="mcard-val" style="color:{rul_color};">{rul:,}</div>
          <div class="mcard-sub">📊 Estimated charge cycles left</div>
        </div>
        """, unsafe_allow_html=True)

    status_cls = {'long': 'ok', 'mid': 'warn', 'short': 'crit'}[tier]
    st.markdown(f"""
    <div class="sbanner {status_cls}">
      <span class="sbi">{tc['icon']}</span>
      <span><strong>{tc['label']}:</strong>&nbsp;{tc['msg']}</span>
    </div>
    """, unsafe_allow_html=True)

with col_gauge:
    fig_g = go.Figure(go.Indicator(
        mode='gauge+number',
        value=rul,
        title=dict(text='Remaining Useful Life', font=dict(size=13, color='#1e293b', family='Inter')),
        number=dict(font=dict(size=42, color='#0f172a', family='Inter'), suffix=' cycles'),
        gauge=dict(
            axis=dict(range=[0, 250], tickcolor='#475569', tickfont=dict(size=10, family='Inter', color='#1e293b')),
            bar=dict(color=tc['color'], thickness=0.22),
            bgcolor='#f8fafc',
            borderwidth=0,
            steps=[
                dict(range=[0,   30], color='rgba(220,38,38,.12)'),
                dict(range=[30, 100], color='rgba(245,158,11,.10)'),
                dict(range=[100,250], color='rgba(16,185,129,.10)'),
            ],
        )
    ))
    fig_g.update_layout(
        height=215,
        margin=dict(l=20, r=20, t=55, b=5),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, sans-serif'),
    )
    st.plotly_chart(fig_g, use_container_width=True)


# ── Sensor reading chips ──────────────────────────────────────────────────────
st.markdown("""
<div class="sec-hdr" style="margin-top:.75rem;">
  <div class="sec-icon" style="background:#eff6ff;">📡</div>
  <div>
    <p class="sec-title">Active Sensor Readings</p>
    <p class="sec-sub">Values currently fed into the prediction model</p>
  </div>
</div>
""", unsafe_allow_html=True)
st.markdown(f"""
<div class="sgrid">
  <div class="schip"><div class="schip-lbl">Re (Ω)</div><div class="schip-val">{Re:.4f}</div></div>
  <div class="schip"><div class="schip-lbl">Rct (Ω)</div><div class="schip-val">{Rct:.4f}</div></div>
  <div class="schip"><div class="schip-lbl">Total Impedance</div><div class="schip-val">{Re+Rct:.4f}</div></div>
  <div class="schip"><div class="schip-lbl">Cap. Fade (Ah/cyc)</div><div class="schip-val">{c_f:.4f}</div></div>
  <div class="schip"><div class="schip-lbl">Cum. Fade (Ah)</div><div class="schip-val">{c_cum:.4f}</div></div>
  <div class="schip"><div class="schip-lbl">Temperature (°C)</div><div class="schip-val">{temp}</div></div>
  <div class="schip"><div class="schip-lbl">Gas (ppm)</div><div class="schip-val">{gas:.0f}</div></div>
  <div class="schip"><div class="schip-lbl">Smoke Density</div><div class="schip-val">{smoke:.3f}</div></div>
</div>
""", unsafe_allow_html=True)


# ── Section 2 · Historical Trend Analysis ─────────────────────────────────────
st.markdown("<hr class='sdivider'>", unsafe_allow_html=True)
st.markdown(f"""
<div class="sec-hdr">
  <div class="sec-icon" style="background:#fdf4ff;">📈</div>
  <div>
    <p class="sec-title">Historical Trend Analysis</p>
    <p class="sec-sub">Long-term degradation trends for cell <strong>{cell}</strong></p>
  </div>
</div>
""", unsafe_allow_html=True)

tab_soh, tab_dcir = st.tabs(['🔋  Battery SoH', '⚡  DCIR Resistance'])

with tab_soh:
    cell_df = df[df['battery_id'] == cell].sort_values('test_id')
    soh_min = max(0,   cell_df['SoH'].min() - 3)
    soh_max = min(105, cell_df['SoH'].max() + 2)

    fig_soh = go.Figure()

    # Coloured background zones
    fig_soh.add_hrect(y0=90,      y1=soh_max, fillcolor='rgba(16,185,129,.07)',  line_width=0)
    fig_soh.add_hrect(y0=80,      y1=90,      fillcolor='rgba(245,158,11,.09)',  line_width=0)
    fig_soh.add_hrect(y0=soh_min, y1=80,      fillcolor='rgba(220,38,38,.07)',   line_width=0)

    fig_soh.add_trace(go.Scatter(
        x=cell_df['test_id'], y=cell_df['SoH'],
        mode='lines+markers',
        line=dict(color='#0f766e', width=2.5),
        marker=dict(size=4, color='#0f766e', opacity=0.65),
        name='SoH',
        hovertemplate='<b>Cycle %{x}</b><br>SoH: %{y:.2f}%<extra></extra>',
    ))

    fig_soh.add_hline(y=90, line_dash='dot', line_color='#f59e0b', line_width=1.8,
                      annotation_text=' 90% early warning', annotation_position='bottom right',
                      annotation_font=dict(size=11, color='#b45309'))
    fig_soh.add_hline(y=80, line_dash='dot', line_color='#dc2626', line_width=1.8,
                      annotation_text=' 80% critical',      annotation_position='bottom right',
                      annotation_font=dict(size=11, color='#991b1b'))

    fig_soh.update_layout(
        height=395,
        margin=dict(l=5, r=5, t=20, b=5),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248,250,252,1)',
        xaxis=dict(title='Cycle', showgrid=True, gridcolor='#e2e8f0', zeroline=False,
                   title_font=dict(color='#374151'), tickfont=dict(color='#374151')),
        yaxis=dict(title='State of Health (%)', showgrid=True, gridcolor='#e2e8f0',
                   range=[soh_min, soh_max],
                   title_font=dict(color='#374151'), tickfont=dict(color='#374151')),
        font=dict(family='Inter, sans-serif', color='#1e293b', size=12),
        hovermode='x unified',
        showlegend=False,
    )
    st.plotly_chart(fig_soh, use_container_width=True)
    st.markdown(
        f'<p style="color:#1e293b;font-size:.88rem;margin-top:-.25rem;">'
        f'<strong>Cell {cell}</strong> · {len(cell_df)} cycles recorded. &nbsp;'
        f'🟢 Healthy &gt;90% &nbsp;·&nbsp; 🟡 Warning 80–90% &nbsp;·&nbsp; 🔴 Critical &lt;80%</p>',
        unsafe_allow_html=True,
    )

with tab_dcir:
    if dcir_df.empty:
        st.info('DCIR data not found. Run the ECE notebook export step to generate `data/dcir_all_cells.csv`.')
    else:
        dcir_cells  = sorted(dcir_df['cell'].astype(str).unique())
        default_sel = str(cell) if str(cell) in dcir_cells else dcir_cells[0]
        dcir_cell   = st.selectbox('Cell', dcir_cells,
                                   index=dcir_cells.index(default_sel), key='dcir_sel')

        dcir_cd = (dcir_df[dcir_df['cell'].astype(str) == str(dcir_cell)]
                   .sort_values('cycle').copy())
        dcir_cd['dcir_ma'] = dcir_cd['dcir'].rolling(5, min_periods=1).mean()

        fig_d = go.Figure()
        fig_d.add_trace(go.Scatter(
            x=dcir_cd['cycle'], y=dcir_cd['dcir'],
            mode='lines', line=dict(color='rgba(211,84,0,.30)', width=1.5),
            name='Raw DCIR',
            hovertemplate='<b>Cycle %{x}</b><br>DCIR: %{y:.5f} Ω<extra></extra>',
        ))
        fig_d.add_trace(go.Scatter(
            x=dcir_cd['cycle'], y=dcir_cd['dcir_ma'],
            mode='lines', line=dict(color='#d35400', width=2.5),
            name='5-cycle avg',
            hovertemplate='<b>Cycle %{x}</b><br>Avg: %{y:.5f} Ω<extra></extra>',
        ))
        fig_d.update_layout(
            height=395,
            margin=dict(l=5, r=5, t=20, b=5),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(248,250,252,1)',
            xaxis=dict(title='Cycle', showgrid=True, gridcolor='#e2e8f0', zeroline=False,
                       title_font=dict(color='#374151'), tickfont=dict(color='#374151')),
            yaxis=dict(title='DCIR (Ω)', showgrid=True, gridcolor='#e2e8f0',
                       title_font=dict(color='#374151'), tickfont=dict(color='#374151')),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0,
                        font=dict(color='#374151')),
            font=dict(family='Inter, sans-serif', color='#1e293b', size=12),
            hovermode='x unified',
        )
        st.plotly_chart(fig_d, use_container_width=True)
        st.caption(
            f'**Cell {dcir_cell}** · DC internal resistance over cycles.  '
            f'Rising DCIR is an early indicator of electrolyte degradation.  '
            f'Faded line: raw values · Solid line: 5-cycle rolling average.'
        )
