"""
AquaGuard AI — Smart Water Quality & Treatment Monitoring System.

Wastewater-treatment operating-condition classification built on the existing
UCI Water Treatment Plant dataset and the project's saved models.
Labels GOOD/MODERATE/POOR and LOW/MEDIUM/HIGH describe TREATMENT CONDITION,
not drinking-water safety certification.
"""
import os
import sys
import io
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "plotly_white"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# Load .env (AQUAGUARD_LIVE_*) if present — without this, a .env file alone
# does nothing and LIVE mode silently stays unconfigured.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, '.env'))
except ImportError:
    pass
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')

from src.aquaguard_config import (  # noqa: E402
    INPUT_SPEC, OPERATIONAL_LIMITS, PARAMETER_GLOSSARY, NEEDS_CONFIGURATION,
)
from src import aquaguard_utils as U  # noqa: E402
from src import alerts as AL  # noqa: E402
from src import history as HI  # noqa: E402
from src import monitoring as MO  # noqa: E402
from src import government_water_data as GV  # noqa: E402

# ------------------------------------------------- theme: water operations ---
TEAL = "#0F766E"
NAVY = "#123D3A"
ACCENT = "#19B7A5"
WARN = "#D97706"
CRIT = "#B91C1C"
OKC = "#16A34A"
BORDER = "#BBF7D0"
QCOLOR = {'GOOD': OKC, 'MODERATE': WARN, 'POOR': CRIT}
RCOLOR = {'LOW': OKC, 'MEDIUM': WARN, 'HIGH': CRIT}

st.set_page_config(page_title="AquaGuard AI — Water Treatment Monitoring",
                   page_icon="🛡️", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown(f"""
<style>
.stApp {{ background: linear-gradient(145deg, #F4F8F5 0%, #EAF3F1 54%, #F7F8F3 100%); color: {NAVY}; }}
section[data-testid="stSidebar"] {{ background: #102F31; border-right: 1px solid #285153; }}
section[data-testid="stSidebar"] * {{ color: #E8F3EF; }}
section[data-testid="stSidebar"] [data-testid="stRadio"] label {{ color: #D2E4DE; }}
section[data-testid="stSidebar"] button {{ border-color: #416668; }}
.main-title {{ font-size: 1.9rem; font-weight: 800; color: {NAVY}; margin-bottom: 0; }}
.sub-title {{ color: #52706C; margin-top: 0.15rem; margin-bottom: 1rem; }}
.card {{ background: #FFFFFF; border: 1px solid #D7E4DF; border-radius: 8px;
     padding: 1rem 1.2rem; box-shadow: 0 8px 24px rgba(18,61,58,.06); margin-bottom: 0.8rem; }}
.kpi {{ background: #FFFFFF; border: 1px solid #D7E4DF; border-top: 3px solid {TEAL};
    border-radius: 8px; padding: 0.9rem 1rem; text-align: left; box-shadow: 0 6px 18px rgba(18,61,58,.05); }}
.kpi .v {{ font-size: 1.45rem; font-weight: 800; color: {NAVY}; word-break: break-word; line-height: 1.2; }}
.kpi .l {{ font-size: 0.78rem; color: #627A75; text-transform: uppercase; letter-spacing: .04em; }}
.badge-ok {{ background: #DCFCE7; color: #166534; border-radius: 999px; padding: 0.15rem 0.7rem; font-weight: 700; }}
.badge-warn {{ background: #FEF3C7; color: #92400E; border-radius: 999px; padding: 0.15rem 0.7rem; font-weight: 700; }}
.badge-crit {{ background: #FEE2E2; color: #991B1B; border-radius: 999px; padding: 0.15rem 0.7rem; font-weight: 700; }}
.result-good {{ background: #F0FAF5; border: 1px solid {OKC}; border-radius: 8px; padding: 1rem; text-align: center; }}
.result-mod {{ background: #FFFAEB; border: 1px solid {WARN}; border-radius: 8px; padding: 1rem; text-align: center; }}
.result-poor {{ background: #FFF3F1; border: 1px solid {CRIT}; border-radius: 8px; padding: 1rem; text-align: center; }}
.stButton > button {{ background: {TEAL}; color: white; font-weight: 700; border-radius: 6px;
                                            border: 1px solid {TEAL}; padding: 0.55rem 1.2rem; transition: background .18s ease, transform .18s ease; }}
.stButton > button:hover {{ background: {ACCENT}; color: white; }}
h1, h2, h3 {{ color: {NAVY} !important; letter-spacing: 0; }}
[data-testid="stTabs"] [role="tablist"] {{ gap: .4rem; border-bottom: 1px solid #D2E0DB; }}
[data-testid="stTabs"] button[role="tab"] {{ border-radius: 5px 5px 0 0; color: #607873; }}
[data-testid="stTabs"] button[aria-selected="true"] {{ color: {TEAL}; border-bottom-color: {TEAL}; }}
@media (max-width: 760px) {{
    .main-title {{ font-size: 1.55rem; }}
    [data-testid="stHorizontalBlock"] {{ gap: .65rem; }}
    .kpi {{ padding: .7rem; }}
    .kpi .v {{ font-size: 1.2rem; }}
}}
</style>
""", unsafe_allow_html=True)

DISCLAIMER = ("Model predictions describe wastewater-treatment operating condition from a 1990 "
              "single-plant dataset. They do NOT certify drinking-water safety and do not replace "
              "laboratory testing or regulatory assessment.")

# ------------------------------------------------------------- helpers ----
@st.cache_data
def _load_df():
    return U.load_data()


@st.cache_data
def _load_saved_history():
    return HI.load_samples()


@st.cache_resource
def _load_models():
    return U.load_model('quality'), U.load_model('risk')

def mode_badge(mode):
    color = {'HISTORICAL': TEAL, 'SIMULATED': WARN, 'LIVE': OKC}.get(mode, TEAL)
    st.markdown(f"Data source: <span style='background:{color}22;color:{color};border:1px solid {color};"
                f"border-radius:999px;padding:.15rem .8rem;font-weight:700'>{mode}</span>",
                unsafe_allow_html=True)

def explain_result(q, r):
    qm = {'GOOD': 'Treatment indicators look typical of well-treated output in the training data.',
          'MODERATE': 'Some indicators are outside the typical well-treated range; check the contributing parameters.',
          'POOR': 'Indicators resemble poorly treated output in the training data; review treatment stages.'}
    rm = {'LOW': 'Few operational-limit exceedances.', 'MEDIUM': 'Several operational limits exceeded.',
          'HIGH': 'Multiple operational limits substantially exceeded.'}
    return qm.get(q, ''), rm.get(r, '')

# ----------------------------------------------------------------- app ----
def main():
    st.markdown('<p class="main-title">🛡️ AquaGuard AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Smart Water Quality &amp; Treatment Monitoring System — '
                'wastewater-treatment condition classification (UCI plant dataset, 1990). Not a drinking-water safety certificate.</p>',
                unsafe_allow_html=True)

    df = _load_df()
    if df is None:
        st.error("No processed data found. Run: `py src/pipeline.py` from the project folder.")
        return
    qm, rm = _load_models()

    nav_options = ["Home / Dashboard", "Analyse a Water Sample", "Prediction Results",
                   "Analysis History", "Trends & Analytics",
                   "How the ML Model Works", "Data Sources & Limitations"]

    with st.sidebar:
        st.header("AquaGuard AI")
        current_page = st.session_state.get("nav_page", "Home / Dashboard")
        page = st.radio("Navigation", nav_options, index=nav_options.index(current_page) if current_page in nav_options else 0)
        st.session_state["nav_page"] = page
        st.markdown("---")
        st.subheader("Data mode")
        mode_options = ["HISTORICAL", "SIMULATED"]
        live_available = MO.live_source_available()
        if live_available:
            mode_options.append("LIVE")
        mode = st.radio("Source", mode_options, index=0, horizontal=True,
                        help="HISTORICAL = saved 1990 dataset. SIMULATED = reproducible demo values, labelled simulated. "
                             "LIVE = a configured real sensor/API only.")
        if not live_available:
            st.caption("LIVE mode is unavailable until a genuine live source is configured.")
        lu = MO.get_last_update()
        st.caption(f"Last update: {lu['at']} ({lu['mode']})")
        st.markdown("---")
        st.caption("Treatment-condition classes: GOOD / MODERATE / POOR. "
                   "Contamination-risk indicators: LOW / MEDIUM / HIGH.")
        with st.expander("Parameter glossary"):
            for k, v in PARAMETER_GLOSSARY.items():
                st.markdown(f"**{k}** — {v}")
        st.markdown("---")
        st.subheader("Official Government Water Data")
        st.link_button("Visit National Water Data Portal (NWDP)", "https://nwdp.nwic.gov.in/")
        st.link_button("Central Pollution Control Board (CPCB)", "https://cpcb.nic.in/")

    if page == "Home / Dashboard":
        show_dashboard(df, qm, mode)
    elif page == "Analyse a Water Sample":
        show_predict(df, qm, rm)
    elif page == "Prediction Results":
        show_prediction_results()
    elif page == "Analysis History":
        show_reports(df, qm, rm)
    elif page == "Trends & Analytics":
        show_analytics(df)
    elif page == "How the ML Model Works":
        show_how_it_works(qm, rm)
    elif page == "Data Sources & Limitations":
        show_sources_limitations()

# ------------------------------------------------------------ dashboard ---
# Dashboard accents are shared by the score gauge and parameter cards.
HERO_BLUE = "#0F766E"


def _dq_css():
    st.markdown("""
<style>
.ws-title { font-size: 1.8rem; font-weight: 800; color: #F4FBF8; margin-bottom: 0; }
.ws-sub { color: #B5D2CA; margin-top: 0.25rem; margin-bottom: 0; font-size: .92rem; }
.ws-hero { display: flex; flex-direction: column; align-items: flex-start; gap: .85rem; padding: 1.15rem 1.4rem;
    margin: .25rem 0 1rem; border-radius: 8px; color: #F4FBF8; background: linear-gradient(115deg, #102F31 0%, #155A57 70%, #1B746D 100%);
    box-shadow: 0 14px 32px rgba(16,47,49,.16); position: relative; overflow: hidden; }
.ws-hero:after { content: ""; position: absolute; width: 220px; height: 220px; right: 11%; top: -135px;
    border: 1px solid rgba(201,245,227,.13); border-radius: 50%; box-shadow: 0 0 0 24px rgba(201,245,227,.035), 0 0 0 48px rgba(201,245,227,.025); }
.ws-hero-copy { display: flex; align-items: center; gap: .85rem; position: relative; z-index: 1; }
.ws-brand-mark { display: inline-flex; width: 40px; height: 40px; align-items: center; justify-content: center;
    background: #A7F3D0; color: #123D3A; border-radius: 8px; font-size: 1.25rem; flex: 0 0 40px; }
.ws-hero-meta { display: flex; flex-wrap: wrap; justify-content: flex-start; gap: .5rem; position: relative; z-index: 1; }
.ws-pill { display: inline-block; background: rgba(233,249,242,.1); border: 1px solid rgba(218,244,232,.28); color: #E8F7F0;
    border-radius: 4px; padding: 0.38rem 0.68rem; font-weight: 650; font-size: 0.76rem; white-space: nowrap; }
.ws-card { background: #FFFFFF; border: 1px solid #D9E5E0; border-radius: 8px; padding: 1rem 1.1rem; margin-bottom: 0.8rem;
    box-shadow: 0 6px 20px rgba(18,61,58,.045); }
.ws-tile { background: #FFFFFF; border: 1px solid #D9E5E0; border-radius: 8px; padding: 0.9rem 1rem; margin-bottom: 0.8rem;
    box-shadow: 0 6px 20px rgba(18,61,58,.045); transition: transform .18s ease, box-shadow .18s ease; }
.ws-tile:hover { transform: translateY(-2px); box-shadow: 0 10px 24px rgba(18,61,58,.09); }
.ws-tile .ic { display: inline-flex; align-items: center; justify-content: center; width: 32px; height: 32px;
    border-radius: 7px; background: #E5F4EF; font-size: 1rem; margin-bottom: 0.45rem; }
.ws-tile .t { font-size: 0.78rem; color: #627A75; }
.ws-tile .big { font-size: 1.4rem; font-weight: 800; color: #123D3A; margin-top: .2rem; }
.ws-tile .nobr { white-space: nowrap; }
.ws-tile .u { font-size: 0.75rem; color: #718681; font-weight: 400; }
.ws-bar { background: #E7EFEB; border-radius: 999px; height: 5px; margin: 0.5rem 0 0.3rem 0; }
.ws-bar > div { height: 5px; border-radius: 999px; }
.ws-pill2 { border-radius: 4px; padding: 0.16rem 0.5rem; font-size: 0.7rem; font-weight: 700; }
.ws-task { display: flex; align-items: center; gap: 0.6rem; padding: 0.5rem 0; border-bottom: 1px solid #E6EEEA; }
.ws-task:last-child { border-bottom: none; }
.ws-task .dot { width: 30px; height: 30px; border-radius: 10px; display: inline-flex; align-items: center;
  justify-content: center; font-size: 1rem; flex-shrink: 0; }
.ws-task .lbl { font-size: 0.85rem; color: #123D3A; font-weight: 650; }
.ws-task .sub { font-size: 0.75rem; color: #627A75; }
.ws-note { background: #E8F4F0; border-left: 3px solid #19B7A5; border-radius: 4px; padding: 0.7rem 0.9rem; font-size: 0.82rem;
    color: #234F49; margin-top: 0.6rem; }
@media (max-width: 760px) { .ws-hero { padding: 1rem; } .ws-title { font-size: 1.5rem; } .ws-hero-meta { gap: .4rem; } }
</style>
""", unsafe_allow_html=True)


def _sparkline(values, color="#38BDF8"):
    return None  # replaced by inline SVG sparklines (see _spark_svg)


def _spark_svg(values, color="#38BDF8", w=260, h=46):
    """Inline SVG sparkline for the last ~60 readings; embeds inside dark cards."""
    v = [float(x) for x in list(values)[-60:]]
    if not v:
        return ""
    lo, hi = min(v), max(v)
    span = (hi - lo) or 1.0
    n = len(v)
    pts = []
    for i, x in enumerate(v):
        px = i / max(n - 1, 1) * w
        py = h - 4 - (x - lo) / span * (h - 8)
        pts.append(f"{px:.1f},{py:.1f}")
    return (f'<svg width="100%" height="{h}" viewBox="0 0 {w} {h}" preserveAspectRatio="none">'
            f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="2"/></svg>')


def _level_status(param, value):
    """Return (pill_label, pill_bg, pill_fg, bar_color, bar_pct)."""
    cfg = OPERATIONAL_LIMITS[param]
    if param == 'PH_S':
        if value < cfg['critical_low'] or value > cfg['critical_high']:
            return ('Critical', '#FEE2E2', '#991B1B', '#EF4444', 100)
        if value < cfg['warn_low'] or value > cfg['warn_high']:
            return ('Warning', '#FEF3C7', '#92400E', '#F59E0B', 65)
        return ('Normal', '#DCFCE7', '#166534', '#22C55E', 30)
    span = cfg['critical'] * 1.5
    pct = min(100, max(4, value / span * 100))
    if value > cfg['critical']:
        return ('Critical', '#FEE2E2', '#991B1B', '#EF4444', pct)
    if value > cfg['warn']:
        return ('Warning', '#FEF3C7', '#92400E', '#F59E0B', pct)
    return ('Normal', '#DCFCE7', '#166534', '#22C55E', pct)


def _dq_param_card(dft, param, title, subtitle, decimals=1, icon="💧"):
    vals = dft[param].dropna()
    v = float(vals.iloc[-1]) if len(vals) else float('nan')
    import math
    # Input-stage params (e.g. PH_E) share units/bands with their treated-output
    # counterpart for display status; thresholds remain the documented ones.
    status_key = {'PH_E': 'PH_S'}.get(param, param)
    unit = OPERATIONAL_LIMITS[status_key]['unit']
    if not math.isfinite(v):
        st.markdown(f"""
<div class="ws-tile">
  <div class="ic">{icon}</div>
  <div class="t">{title} · {subtitle}
    <span class="ws-pill2" style="background:#E2E8F0;color:#4B6B54;float:right">No data</span></div>
  <div class="big"><span class="nobr">— <span class="u">{unit}</span></span></div>
  <div class="ws-note">No valid reading in the latest record.</div>
</div>
""", unsafe_allow_html=True)
        return
    pill, bg, fg, bar, pct = _level_status(status_key, v)
    # Plausibility guard: flag physically impossible readings instead of
    # presenting them as normal measurements (known raw-column quirks).
    implausible = (param in ('PH_S', 'PH_E') and (v < 0 or v > 14))
    note = ''
    if implausible:
        pill, bg, fg, bar = ('Check data', '#FEE2E2', '#991B1B', '#EF4444')
        note = ('<div class="ws-note">Outside the plausible range for this parameter — '
                'likely a dataset recording issue. Treat with caution.</div>')
    st.markdown(f"""
<div class="ws-tile">
  <div class="ic">{icon}</div>
  <div class="t">{title} · {subtitle}
    <span class="ws-pill2" style="background:{bg};color:{fg};float:right">{pill}</span></div>
  <div class="big"><span class="nobr">{v:.{decimals}f} <span class="u">{unit}</span></span></div>
  <div class="ws-bar"><div style="width:{pct:.0f}%;background:{bar}"></div></div>
  {_spark_svg(vals, bar)}
  {note}
</div>
""", unsafe_allow_html=True)


def show_prediction_results():
    st.header("Prediction Results")
    result = st.session_state.get('latest_prediction')
    if not result:
        st.info("No analysis has been run yet. Go to Analyse a Water Sample to submit a reading.")
        return

    q = result['quality_class']
    r = result['risk_class']
    qexp, rexp = explain_result(q, r)
    qcls = {'GOOD': 'result-good', 'MODERATE': 'result-mod'}.get(q, 'result-poor')
    rcls = {'LOW': 'result-good', 'MEDIUM': 'result-mod'}.get(r, 'result-poor')

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"<div class='{qcls}'><div style='font-size:1.5rem;font-weight:800'>{q}</div><div>treatment condition</div></div>", unsafe_allow_html=True)
        st.caption(qexp)
        if result.get('quality_probabilities'):
            st.write("Model confidence for treatment condition:")
            for k, v in sorted(result['quality_probabilities'].items(), key=lambda x: -x[1]):
                st.progress(min(max(v, 0.0), 1.0), text=f"{k}: {v * 100:.1f}%")
    with c2:
        st.markdown(f"<div class='{rcls}'><div style='font-size:1.5rem;font-weight:800'>{r}</div><div>contamination-risk indicator</div></div>", unsafe_allow_html=True)
        st.caption(rexp)
        if result.get('risk_probabilities'):
            st.write("Model confidence for risk indicator:")
            for k, v in sorted(result['risk_probabilities'].items(), key=lambda x: -x[1]):
                st.progress(min(max(v, 0.0), 1.0), text=f"{k}: {v * 100:.1f}%")

    st.markdown("---")
    st.subheader("Submitted readings")
    input_df = pd.DataFrame([
        {'Parameter': k, 'Value': round(float(v), 4), 'Unit': next((s['unit'] for s in INPUT_SPEC if s['key'] == k), '')}
        for k, v in result['inputs'].items() if k in {s['key'] for s in INPUT_SPEC}
    ])
    st.dataframe(input_df, use_container_width=True)

    if result.get('warnings'):
        st.warning("Unusual readings detected based on the project’s documented training range checks:")
        for feat, val, lo, hi in result['warnings'][:10]:
            st.write(f"- {feat}: {val:.2f} (training range {lo:.2f} to {hi:.2f})")

    st.caption(f"Analysis timestamp: {result['timestamp']} | Source: {result['source_mode']} | Sample: {result['sample_name']}")
    st.caption("This model predicts wastewater-treatment operating condition from a historical UCI plant dataset. It does not certify drinking-water safety.")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Analyse a new sample"):
            st.session_state['nav_page'] = 'Analyse a Water Sample'
            st.rerun()
    with c2:
        if st.button("View full analysis history"):
            st.session_state['nav_page'] = 'Analysis History'
            st.rerun()


def show_how_it_works(qm, rm):
    st.header("How the ML Model Works")
    st.markdown("User Input → Validation → Training-compatible preprocessing → Saved ML Model → Predicted Class → Explanation → Saved Result")

    st.subheader("1) User input")
    st.write("The sample form collects measured process values such as pH, BOD, COD, suspended solids, conductivity, and calendar metadata. Each field has a unit and a valid range.")

    st.subheader("2) Validation and preprocessing")
    st.write("The app validates numeric values and checks that the inputs fit the training constraints. The feature vector is then created with the same 35-feature order used during training, using the preprocessing logic in the project utilities.")

    st.code("build_feature_vector(cleaned) -> safe_predict(model_dict, feature_vector)", language="python")

    st.subheader("3) Actual models in use")
    if qm is not None and qm.get('metadata'):
        meta = qm['metadata']
        best_quality = meta.get('best_model_name', 'unknown')
        best_risk = rm['metadata'].get('best_model_name', 'unknown') if rm and rm.get('metadata') else 'unknown'
        st.write(f"Quality model: {best_quality}")
        st.write(f"Risk model: {best_risk}")
        st.write(f"Target variable: {meta.get('target', 'Quality')} / {rm['metadata'].get('target', 'Risk') if rm and rm.get('metadata') else 'Risk'}")
        st.write(f"Feature count: {len(meta.get('feature_names', []))}")
        st.write(f"Classes: {meta.get('classes', [])}")
    else:
        st.warning("No trained model metadata was found in the saved artifacts.")

    st.subheader("4) Output and explanation")
    st.write("The app returns a predicted treatment-condition class and a risk indicator. It also explains the results using rule-based operational thresholds and valid model probabilities when available.")

    st.subheader("5) Saved result")
    st.write("Each saved analysis is stored in the project history file so the dashboard, history, and analytics pages reflect the same result from one source of truth.")


def show_sources_limitations():
    st.header("Data Sources & Limitations")
    st.subheader("Actual project data sources")
    st.markdown("- Historical UCI Water Treatment Plant dataset in the raw and processed data folders.")
    st.markdown("- Saved trained model and metadata in the models folder.")
    st.markdown("- Government data from the National Water Data Portal, cached locally.")
    st.markdown("- Optional live source via environment variables if a real sensor/API is configured.")

    st.subheader("Important limitations")
    st.warning(DISCLAIMER)
    st.markdown("- The target labels describe wastewater-treatment operating condition, not drinking-water safety.")
    st.markdown("- The official government data is displayed separately and is not used as model input.")
    st.markdown("- LIVE mode only appears when a genuine source is configured; no fake live data is created.")
    st.markdown("- Model confidence is shown only when the trained model exposes a valid probability estimate.")


def show_dashboard(df, qm, mode):
    _dq_css()
    saved = HI.history_summary(_load_saved_history())
    dft = df.sort_values('Date').reset_index(drop=True)
    latest = dft.iloc[-1]
    prev = dft.iloc[-2] if len(dft) > 1 else latest
    n = len(dft)
    counts_q = df['Quality_Class'].value_counts()
    counts_r = df['Risk_Class'].value_counts()
    if not saved.empty:
        saved_counts_q = saved['quality_class'].value_counts().to_dict()
        saved_counts_r = saved['risk_class'].value_counts().to_dict()
        sample_recent = saved.head(5)
    else:
        saved_counts_q = {}
        saved_counts_r = {}
        sample_recent = pd.DataFrame(columns=['record_id', 'quality_class', 'risk_class', 'created_at'])
    latest_sample = {p: float(latest[p]) for p in OPERATIONAL_LIMITS if p in latest and pd.notna(latest[p])}
    latest_alerts = AL.evaluate_output_sample(latest_sample, sample_id=f"hist-{latest['Date'].date()}",
                                              source_mode='HISTORICAL',
                                              timestamp=str(latest['Date']))
    n_warn = sum(1 for a in latest_alerts if a['severity'] == 'Warning')
    n_crit = sum(1 for a in latest_alerts if a['severity'] == 'Critical')
    open_stored = AL.load_alerts()
    n_open = len(open_stored[open_stored['status'] == 'open']) if len(open_stored) else 0

    st.markdown(f"""
<section class="ws-hero">
    <div class="ws-hero-copy"><span class="ws-brand-mark">◉</span>
        <div><div class="ws-title">Operations overview</div>
        <div class="ws-sub">AquaGuard · historical plant monitoring</div></div>
    </div>
    <div class="ws-hero-meta"><span class="ws-pill">{mode} SOURCE</span>
        <span class="ws-pill">Latest · {latest['Date'].date()}</span>
        <span class="ws-pill">{n} plant records</span></div>
</section>
""", unsafe_allow_html=True)

    tab_read, tab_index, tab_alerts, tab_trends = st.tabs(
        ["Process water", "Quality index", "Alerts", "Trends & GIS"])

    with tab_read:
        st.caption(f"Latest {mode} sample: {latest['Date'].date()} · {n} plant records total")
        if not saved.empty:
            st.subheader("Saved predictions")
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.markdown(f"<div class='kpi'><div class='v'>{len(saved)}</div><div class='l'>Saved readings</div></div>", unsafe_allow_html=True)
            with k2:
                st.markdown(f"<div class='kpi'><div class='v'>{saved_counts_q.get('GOOD', 0)}</div><div class='l'>GOOD</div></div>", unsafe_allow_html=True)
            with k3:
                st.markdown(f"<div class='kpi'><div class='v'>{saved_counts_q.get('MODERATE', 0)}</div><div class='l'>MODERATE</div></div>", unsafe_allow_html=True)
            with k4:
                st.markdown(f"<div class='kpi'><div class='v'>{saved_counts_q.get('POOR', 0)}</div><div class='l'>POOR</div></div>", unsafe_allow_html=True)
            st.dataframe(sample_recent[['record_id', 'quality_class', 'risk_class', 'created_at']].rename(columns={'created_at': 'timestamp'}), width='stretch')
        else:
            st.info("No saved sample predictions yet. Run a prediction in Test a Water Sample to populate the dashboard summary.")
        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            _dq_param_card(dft, 'PH_E', 'pH Level', 'Acidity / alkalinity', icon="🧪")
        with r1c2:
            _dq_param_card(dft, 'DBO_S', 'BOD', 'Organic load', decimals=0, icon="🦠")
        with r1c3:
            _dq_param_card(dft, 'DQO_S', 'COD', 'Chemical load', decimals=0, icon="⚗️")
        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            _dq_param_card(dft, 'COND_S', 'Conductivity', 'Dissolved salts', decimals=0, icon="🌊")
        with r2c2:
            _dq_param_card(dft, 'SS_S', 'Suspended Solids', 'Particles', decimals=0, icon="🔹")
        with r2c3:
            vals = dft['ZN_E'].dropna()
            v = float(vals.iloc[-1]) if len(vals) else float('nan')
            import math as _m
            if not _m.isfinite(v):
                st.markdown("""<div class="ws-tile"><div class="ic">🪙</div>
<div class="t">Zinc · Metal indicator</div>
<div class="big">—</div><div class="ws-note">No valid reading.</div></div>""",
                            unsafe_allow_html=True)
            else:
                pill = 'Normal' if v <= 3 else ('Warning' if v <= 5 else 'Critical')
                bg = {'Normal': '#DCFCE7', 'Warning': '#FEF3C7', 'Critical': '#FEE2E2'}[pill]
                fg = {'Normal': '#166534', 'Warning': '#92400E', 'Critical': '#991B1B'}[pill]
                bar = {'Normal': '#22C55E', 'Warning': '#F59E0B', 'Critical': '#EF4444'}[pill]
                pct = min(100, max(4, v / 7.5 * 100))
                st.markdown(f"""
<div class="ws-tile">
  <div class="ic">🪙</div>
  <div class="t">Zinc · Metal indicator
    <span class="ws-pill2" style="background:{bg};color:{fg};float:right">{pill}</span></div>
  <div class="big"><span class="nobr">{v:.2f} <span class="u">mg/L</span></span></div>
  <div class="ws-bar"><div style="width:{pct:.0f}%;background:{bar}"></div></div>
  {_spark_svg(vals, bar)}
</div>
""", unsafe_allow_html=True)

    with tab_index:
        score = float(latest['Quality_Score'])
        qcls = str(latest['Quality_Class'])
        rcls = str(latest['Risk_Class'])
        delta = score - float(prev['Quality_Score'])
        gcol = {'GOOD': '#22C55E', 'MODERATE': '#F59E0B'}.get(qcls, '#EF4444')
        c1, c2 = st.columns([3, 2])
        with c1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=score,
                number={'suffix': ' / 100', 'font': {'color': '#FFFFFF', 'size': 30}},
                title={'text': 'Water Quality Index<br><span style="font-size:12px;color:#D6EAFB">Overall treatment assessment</span>'},
                gauge={'axis': {'range': [0, 100], 'tickcolor': '#FFFFFF'},
                       'bar': {'color': '#FFFFFF'},
                       'bgcolor': 'rgba(255,255,255,.25)',
                       'steps': [{'range': [0, 40], 'color': 'rgba(255,255,255,.15)'},
                                 {'range': [40, 70], 'color': 'rgba(255,255,255,.15)'},
                                 {'range': [70, 100], 'color': 'rgba(255,255,255,.15)'}]}))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=10),
                              paper_bgcolor=HERO_BLUE, font={'color': '#FFFFFF'})
            st.plotly_chart(fig, width="stretch")
            st.caption("Gauge = latest composite quality score (0–100) from treated-output readings.")
        with c2:
            st.markdown(f"""
<div class="ws-card">
  <div class="t" style="font-size:.8rem;color:#4B6B54">Status</div>
  <div style="font-size:1.3rem;font-weight:800;color:#14532D">{qcls}
    <span class="ws-pill2" style="background:{gcol}22;color:{gcol}">{rcls} risk</span></div>
  <div style="font-size:.85rem;color:#4B6B54;margin-top:.3rem">Trend: {"▲" if delta >= 0 else "▼"} {abs(delta):.1f} pts vs previous sample</div>
  <div class="ws-note">{"Treatment indicators look typical of well-treated output." if qcls == 'GOOD'
    else "Some indicators are outside the well-treated range — check contributing parameters." if qcls == 'MODERATE'
    else "Indicators resemble poorly treated output — review treatment stages."}</div>
</div>
""", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.pie(df, names='Quality_Class', title='Treatment-condition distribution',
                         color='Quality_Class', color_discrete_map=QCOLOR, hole=0.55)
            fig.update_layout(height=300)
            st.plotly_chart(fig, width="stretch")
            st.caption("Share of all 527 records per treatment-condition class.")
        with c2:
            fig = px.pie(df, names='Risk_Class', title='Risk-indicator distribution',
                         color='Risk_Class', color_discrete_map=RCOLOR, hole=0.55)
            fig.update_layout(height=300)
            st.plotly_chart(fig, width="stretch")
            st.caption("Share of all 527 records per contamination-risk indicator.")

    with tab_alerts:
        st.caption(f"Latest-sample alerts: ⚠ {n_warn} Warning · 🔴 {n_crit} Critical · {n_open} open stored")
        if latest_alerts:
            for a in latest_alerts:
                bg = '#FEF3C7' if a['severity'] == 'Warning' else '#FEE2E2'
                fg = '#92400E' if a['severity'] == 'Warning' else '#991B1B'
                st.markdown(f"""
<div class="ws-card"><span class="ws-pill2" style="background:{bg};color:{fg}">{a['severity']}</span>
<b style="color:#14532D"> {a['label']}</b><br><span style="font-size:.85rem;color:#4B6B54">{a['message']}</span></div>
""", unsafe_allow_html=True)
        else:
            st.markdown('<div class="ws-card"><b style="color:#14532D">✓ Normal</b> '
                        '<span style="color:#4B6B54">— all checked parameters within operational limits.</span></div>',
                        unsafe_allow_html=True)
        st.caption("Full alert history, filters and review workflow live under Alerts & Monitoring.")

        # Tasks panel — every number below is computed from real data
        st.subheader("Tasks")
        _checked = [p for p in latest_sample]
        _flagged = {(a['parameter']) for a in latest_alerts}
        _within = sum(1 for p in _checked if p not in _flagged)
        _within_pct = round(_within / max(len(_checked), 1) * 100)
        _stored_total = len(open_stored)
        _stored_done = len(open_stored[open_stored['status'].isin(['reviewed', 'resolved'])]) if _stored_total else 0
        _months = int(df['Date'].dt.month.nunique())

        def _task_row(icon, bgc, label, sub, pct=None):
            bar = (f"<div class='ws-bar'><div style='width:{pct}%;background:{HERO_BLUE}'></div></div>"
                   f"<div class='sub'>{pct}%</div>" if pct is not None else f"<div class='sub'>{sub}</div>")
            st.markdown(f"""<div class="ws-task"><span class="dot" style="background:{bgc}">{icon}</span>
<div style="flex:1"><div class="lbl">{label}</div>{bar}</div></div>""", unsafe_allow_html=True)

        t1, t2 = st.columns(2)
        with t1:
            st.markdown("<div class='ws-card'>", unsafe_allow_html=True)
            _task_row("✅", "#DCFCE7", "Parameters within limits",
                      f"{_within} of {len(_checked)} latest readings OK", _within_pct)
            if _stored_total:
                _task_row("🔔", "#FEF3C7", "Alerts reviewed",
                          f"{_stored_done} of {_stored_total} stored", round(_stored_done / _stored_total * 100))
            else:
                _task_row("🔔", "#E8F4FD", "Alerts reviewed", "No stored alerts yet")
            _task_row("📅", "#E0F2FE", "Months covered",
                      f"{_months} of 12 months (1990)", round(_months / 12 * 100))
            st.markdown("</div>", unsafe_allow_html=True)
        with t2:
            st.markdown(f"""<div class="ws-card"><div class="lbl" style="font-size:.9rem;color:#14532D;font-weight:700">
Complete your review</div><div class="sub" style="font-size:.8rem;color:#4B6B54">
{n_open} open stored alert(s) · {n_warn + n_crit} flag(s) on the latest sample.</div>
<div class="ws-note">Work through Alerts &amp; Monitoring, then save samples from Test a Water Sample.</div></div>""",
                        unsafe_allow_html=True)

    with tab_trends:
        fig = px.scatter(dft, x='Date', y='Quality_Score', color='Quality_Class',
                         color_discrete_map=QCOLOR, labels={'Quality_Score': 'Quality score'},
                         title='Historical quality trend')
        fig.update_layout(height=300, margin=dict(t=40, b=20))
        st.plotly_chart(fig, width="stretch")
        st.caption("Each point is one daily plant record; color shows its treatment-condition class.")
        mo = df.groupby([df['Date'].dt.month, 'Quality_Class'], observed=True).size().unstack(fill_value=0)
        fig = px.bar(mo, title='Records per month × condition', color_discrete_map=QCOLOR)
        fig.update_layout(height=280)
        st.plotly_chart(fig, width="stretch")
        st.caption("Monthly record counts — the plant logged readings every month of 1990.")
        with st.expander("🗺 GIS Map (location tracking status)"):
            st.info("No GPS or plant-map coordinates exist in the 1990 UCI dataset, so no live map can be drawn. "
                    "Use Reports & History to record optional free-text locations per sample; a map layer can be "
                    "added once geocoded monitoring points are available.")
        st.subheader("Recent parameter changes (last two historical samples)")
        chg = []
        for p in ['PH_S', 'DBO_S', 'DQO_S', 'SS_S', 'COND_S']:
            unit = OPERATIONAL_LIMITS[p]['unit']
            d = float(latest[p]) - float(prev[p])
            chg.append({'Parameter': OPERATIONAL_LIMITS[p]['label'], 'Previous': round(float(prev[p]), 2),
                        'Latest': round(float(latest[p]), 2), 'Change': round(d, 2), 'Unit': unit})
        st.dataframe(pd.DataFrame(chg), width="stretch")
        avg = df[['PH_S', 'DBO_S', 'DQO_S', 'SS_S', 'COND_S']].mean().round(2).reset_index()
        avg.columns = ['Parameter', 'Average']
        st.dataframe(avg, width="stretch")

    st.caption(f"Latest data timestamp: {df['Date'].max()} · Source: UCI Water Treatment Plant (single plant, 1990). "
               f"Viewing mode: {mode}. {DISCLAIMER}")

# ------------------------------------------------------------ predict -----
def show_predict(df, qm, rm):
    st.header("Test a Water Sample")
    st.info("Enter treatment-stage readings below. Units are shown beside every field. "
            "Use **Load sample values** to try the app with a real historical record.")
    if st.button("📥 Load sample values (historical row)"):
        row = df.sort_values('Date').iloc[-1]
        clamped = []
        for s in INPUT_SPEC:
            if s['key'] in row and pd.notna(row[s['key']]):
                v = float(row[s['key']])
                lo, hi = float(s['min']), float(s['max'])
                if v < lo or v > hi:
                    clamped.append(f"{s['label']} ({v} → clipped to {min(max(v, lo), hi)})")
                    v = min(max(v, lo), hi)
                st.session_state[f"in_{s['key']}"] = v
        st.session_state['in_Month'] = int(row['Month']) if 'Month' in row and pd.notna(row['Month']) else 6
        st.session_state['in_DayOfYear'] = int(row['DayOfYear']) if 'DayOfYear' in row and pd.notna(row['DayOfYear']) else 180
        st.success("Loaded the latest historical record into the form.")
        if clamped:
            st.warning("Some historical values exceeded the form's valid ranges and were clipped: "
                       + "; ".join(clamped) + ". Predictions with clipped values are approximate.")

    # Seed + clamp session values BEFORE widget creation, then use key-only
    # widgets (passing value= alongside key= triggers Streamlit warnings and
    # can resurrect out-of-range values that crash number_input).
    for s in INPUT_SPEC:
        _k = f"in_{s['key']}"
        try:
            _v = float(st.session_state.get(_k, s['default']))
        except (TypeError, ValueError):
            _v = float(s['default'])
        if not np.isfinite(_v):
            _v = float(s['default'])
        st.session_state[_k] = min(max(_v, float(s['min'])), float(s['max']))
    try:
        st.session_state['in_Month'] = min(max(int(st.session_state.get('in_Month', 6)), 1), 12)
    except (TypeError, ValueError):
        st.session_state['in_Month'] = 6
    try:
        st.session_state['in_DayOfYear'] = min(max(int(st.session_state.get('in_DayOfYear', 180)), 1), 365)
    except (TypeError, ValueError):
        st.session_state['in_DayOfYear'] = 180

    values = {}
    for group in ['Raw water intake', 'Primary settler', 'Secondary settler']:
        with st.expander(group, expanded=(group == 'Raw water intake')):
            cols = st.columns(3)
            for i, s in enumerate([x for x in INPUT_SPEC if x['group'] == group]):
                with cols[i % 3]:
                    values[s['key']] = st.number_input(
                        f"{s['label']} ({s['unit']})", min_value=float(s['min']),
                        max_value=float(s['max']),
                        key=f"in_{s['key']}", help=s['help'])
    c1, c2 = st.columns(2)
    with c1:
        values['Month'] = st.number_input("Month (1–12)", 1, 12, key='in_Month')
    with c2:
        values['DayOfYear'] = st.number_input("Day of year (1–365)", 1, 365, key='in_DayOfYear')
    c1, c2, c3 = st.columns(3)
    with c1:
        sample_name = st.text_input("Sample name / ID", value="field-sample-01")
    with c2:
        location = st.text_input("Location (optional)", value="", placeholder="e.g. Plant inlet, Ward 4")
    with c3:
        collected = st.text_input("Collection date/time (optional)", value=datetime.now(timezone.utc).date().isoformat())

    if st.button("🔮 Predict treatment condition & risk", type="primary"):
        errors, cleaned = U.validate_inputs(values)
        if errors:
            for e in errors:
                st.error(e)
            return
        vec = U.build_feature_vector(cleaned)
        oor = U.check_out_of_range(vec)
        with st.spinner("Running models..."):
            qres = U.safe_predict(qm, vec)
            rres = U.safe_predict(rm, vec)
        if not qres['ok']:
            st.error(qres['error'])
            return
        if not rres['ok']:
            st.error(rres['error'])
            return
        q, r = qres['class'], rres['class']
        qexp, rexp = explain_result(q, r)
        _qm_name = (qm or {}).get('metadata', {}).get('best_model_name', 'trained model')
        _rm_name = (rm or {}).get('metadata', {}).get('best_model_name', 'trained model')
        st.caption(f"Model used: **{_qm_name}** for treatment condition, **{_rm_name}** for risk — both trained "
                   f"on the 1990 UCI treatment-plant dataset (see ML Model Insights for measured metrics). "
                   f"A prediction is the model's best guess from your inputs, not a laboratory measurement.")
        cls_box = {'GOOD': 'result-good', 'MODERATE': 'result-mod'}.get(q, 'result-poor')
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"<div class='{cls_box}'><div style='font-size:1.5rem;font-weight:800'>{q}</div>"
                        f"<div>treatment condition</div></div>", unsafe_allow_html=True)
            if qres['probabilities']:
                st.write("Model confidence (predicted probabilities):")
                for k, v in sorted(qres['probabilities'].items(), key=lambda x: -x[1]):
                    st.progress(min(max(v, 0.0), 1.0), text=f"{k}: {v*100:.1f}%")
            st.caption(qexp)
        with c2:
            rcls = {'LOW': 'result-good', 'MEDIUM': 'result-mod'}.get(r, 'result-poor')
            st.markdown(f"<div class='{rcls}'><div style='font-size:1.5rem;font-weight:800'>{r} risk</div>"
                        f"<div>contamination-risk indicator</div></div>", unsafe_allow_html=True)
            if rres['probabilities']:
                st.write("Model confidence (predicted probabilities):")
                for k, v in sorted(rres['probabilities'].items(), key=lambda x: -x[1]):
                    st.progress(min(max(v, 0.0), 1.0), text=f"{k}: {v*100:.1f}%")
            st.caption(rexp)
        if oor:
            with st.warning("Some inputs fall outside the training data's known range — treat the prediction with extra caution:"):
                for f, v, lo, hi in oor[:10]:
                    st.write(f"- {f}: {v:.2f} (training range {lo:.2f}–{hi:.2f})")
        st.subheader("Parameters contributing most (tree-model importance)")
        top = U.top_contributing_features(qm, vec)
        if top:
            for i, (f, imp, val) in enumerate(top, 1):
                st.write(f"{i}. **{f}** — importance {imp:.3f}, your value {val:.2f}")
        else:
            st.info("The deployed model does not expose reliable per-feature importance, so no contribution breakdown is shown.")
        st.warning(DISCLAIMER)
        qc = max(qres['probabilities'].values()) if qres['probabilities'] else None
        rc = max(rres['probabilities'].values()) if rres['probabilities'] else None
        st.session_state['latest_prediction'] = {
            'quality_class': q,
            'risk_class': r,
            'quality_probabilities': qres.get('probabilities', {}),
            'risk_probabilities': rres.get('probabilities', {}),
            'inputs': cleaned,
            'warnings': oor,
            'source_mode': 'MANUAL',
            'sample_name': sample_name or 'field-sample-01',
            'timestamp': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        }
        if st.button("💾 Save this sample to history"):
            rid = HI.save_sample(sample_name, location, collected, 'MANUAL', cleaned, q,
                                 round(qc, 4) if qc else None, r, round(rc, 4) if rc else None)
            st.success(f"Saved as {rid}. Find it under Reports & History.")
            st.session_state['latest_saved_record'] = rid
            st.rerun()

# ------------------------------------------------------------- alerts -----
def show_alerts(df, mode):
    st.header("Alerts & Monitoring")
    mode_badge(mode)
    st.caption("Severity: Normal = within operational limits · Warning = crosses warn limit · "
               "Critical = crosses critical limit. Limits below are configurable OPERATIONAL treatment targets, "
               "not drinking-water safety standards.")
    with st.expander("Configured operational limits (source documented)", expanded=True):
        rows = []
        for p, c in OPERATIONAL_LIMITS.items():
            if p == 'PH_S':
                rows.append({'Parameter': c['label'], 'Warning when': f"outside {c['warn_low']}–{c['warn_high']} {c['unit']}",
                             'Critical when': f"< {c['critical_low']} or > {c['critical_high']} {c['unit']}", 'Source': c['source']})
            else:
                rows.append({'Parameter': c['label'], 'Warning when': f"> {c['warn']} {c['unit']}",
                             'Critical when': f"> {c['critical']} {c['unit']}", 'Source': c['source']})
        st.dataframe(pd.DataFrame(rows), width="stretch")
        st.caption("Raw-water input limits (Q_E, ZN_E, PH_E, DBO_E, DQO_E, SS_E, COND_E): no validated threshold in the "
                   "dataset — marked as requiring operator configuration. They are intentionally NOT auto-flagged.")

    current, sid, ts = None, None, None
    if mode == 'HISTORICAL':
        row = df.sort_values('Date').iloc[-1]
        current = {p: float(row[p]) for p in OPERATIONAL_LIMITS if pd.notna(row[p])}
        sid, ts = f"hist-{row['Date'].date()}", str(row['Date'])
        MO.set_last_update('HISTORICAL', sid)
    elif mode == 'SIMULATED':
        seed = st.number_input("Simulation seed (reproducible)", 1, 9999, 42)
        if st.button("🎲 Generate simulated reading"):
            st.session_state['sim'] = MO.simulate_reading(df, seed=int(seed))
            MO.set_last_update('SIMULATED', f"seed={seed}")
        current = st.session_state.get('sim')
        if current:
            sid, ts = f"sim-seed-{current.get('_seed')}", current.get('_at')
            st.warning("These values are SIMULATED for demo purposes — not real measurements.")
            current = {k: v for k, v in current.items() if not k.startswith('_')}
        else:
            st.info("Press “Generate simulated reading” to create a labelled demo sample.")
            current = None
    else:  # LIVE
        status, payload, msg = MO.read_live_source()
        if status == 'not_configured':
            st.warning(msg)
            with st.expander("How to connect a genuine live source", expanded=True):
                st.markdown("""
1. Point `AQUAGUARD_LIVE_CSV` to a CSV file whose **last row** holds `PH_S,DBO_S,DQO_S,SS_S,COND_S`
   (optional `timestamp`, `sample_id` columns), **or**
2. Point `AQUAGUARD_LIVE_JSON` to a JSON file with those keys, **or**
3. Point `AQUAGUARD_LIVE_URL` to an HTTPS endpoint returning that JSON.
4. Copy `.env.example` to `.env` and set the variable (never commit secrets).
5. Return here and select LIVE mode again.

On failure the app shows the error and keeps HISTORICAL/SIMULATED working — it never invents live readings.
""")
            st.code("AQUAGUARD_LIVE_CSV=C:/sensors/plant_live.csv", language="bash")
        elif status == 'error':
            st.error(msg + " No fake live data generated.")
        else:
            st.success(msg)
            current = {k: float(payload[k]) for k in OPERATIONAL_LIMITS if k in payload}
            sid, ts = str(payload.get('sample_id', 'live')), str(payload.get('timestamp', datetime.now(timezone.utc).isoformat(timespec='seconds')))
            MO.set_last_update('LIVE', sid)

        if current is None:
            st.markdown("---")
            st.subheader("Upload a LIVE CSV instead (no .env needed)")
            st.download_button("⬇ Download CSV template", MO.LIVE_CSV_TEMPLATE,
                               "live_template.csv", "text/csv")
            up = st.file_uploader("Drop your sensor CSV here", type=["csv"],
                                  help="Columns PH_S,DBO_S,DQO_S,SS_S,COND_S required; last row is used.")
            if up is not None:
                ustatus, upayload, umsg = MO.read_live_upload(up)
                if ustatus == 'ok':
                    st.success(umsg)
                    current = {k: float(upayload[k]) for k in OPERATIONAL_LIMITS if k in upayload}
                    sid, ts = 'upload', datetime.now(timezone.utc).isoformat(timespec='seconds')
                    MO.set_last_update('LIVE', sid)
                else:
                    st.error(umsg + " Fix the file (or grab the template above) and re-upload.")

    if current:
        st.subheader(f"Current reading — {sid} ({ts})")
        st.dataframe(pd.DataFrame([{'Parameter': OPERATIONAL_LIMITS[k]['label'], 'Reading': v,
                                    'Unit': OPERATIONAL_LIMITS[k]['unit']} for k, v in current.items()]),
                     width="stretch")
        new_alerts = AL.evaluate_output_sample(current, sample_id=sid, source_mode=mode, timestamp=ts)
        if new_alerts:
            for a in new_alerts:
                badge = 'badge-warn' if a['severity'] == 'Warning' else 'badge-crit'
                st.markdown(f"<span class='{badge}'>{a['severity']}</span> **{a['label']}** — {a['message']} "
                            f"(sample `{a['sample_id']}`, {a['timestamp']})", unsafe_allow_html=True)
            added = AL.record_alerts(new_alerts)
            st.caption(f"{added} new alert(s) stored ({len(new_alerts) - added} duplicate(s) skipped).")
        else:
            st.success("Normal — all checked parameters are within operational limits.")

    st.subheader("Alert history")
    adf = AL.load_alerts()
    if len(adf) == 0:
        st.info("No alerts stored yet. Evaluate a reading above to generate alerts.")
        return
    f1, f2, f3 = st.columns(3)
    with f1:
        sev = st.selectbox("Severity", ["All", "Warning", "Critical"])
    with f2:
        stat = st.selectbox("Status", ["All", "open", "reviewed", "resolved"])
    with f3:
        q = st.text_input("Search (parameter / sample / message)", "")
    view = adf.copy()
    if sev != 'All':
        view = view[view['severity'] == sev]
    if stat != 'All':
        view = view[view['status'] == stat]
    if q:
        ql = q.lower()
        view = view[view.apply(lambda r: ql in str(r.values).lower(), axis=1)]
    if len(view) == 0:
        st.info("No alerts match the current filters.")
        return
    st.dataframe(view.sort_values('timestamp', ascending=False), width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        aid = st.selectbox("Select alert to update", view['alert_id'].tolist())
    with c2:
        ns = st.selectbox("New status", ["reviewed", "resolved", "open"])
    if st.button("Update alert status"):
        st.success("Updated." if AL.set_alert_status(aid, ns) else "Alert not found.")
        st.rerun()
    buf = io.StringIO()
    view.to_csv(buf, index=False)
    st.download_button("⬇ Export filtered alerts (CSV)", buf.getvalue(), "aquaguard_alerts.csv", "text/csv")

# ---------------------------------------------------------- analytics -----
def show_analytics(df):
    st.header("Analytics & Trends")
    saved = HI.history_summary(HI.load_samples())
    if not saved.empty:
        st.subheader("Saved prediction history")
        saved_chart = saved[['created_at', 'quality_class', 'quality_conf', 'risk_class', 'risk_conf']].copy()
        saved_chart['created_at'] = pd.to_datetime(saved_chart['created_at'], errors='coerce')
        fig = px.line(saved_chart, x='created_at', y='quality_conf', color='quality_class',
                      title='Saved prediction confidence over time', markers=True)
        fig.update_layout(height=320)
        st.plotly_chart(fig, width='stretch')
        st.dataframe(saved[['record_id', 'sample_name', 'quality_class', 'risk_class', 'quality_conf', 'risk_conf', 'created_at']].rename(columns={'created_at': 'timestamp'}), width='stretch')

    qp = st.multiselect("Quality filter", ['GOOD', 'MODERATE', 'POOR'], default=['GOOD', 'MODERATE', 'POOR'])
    rp = st.multiselect("Risk filter", ['LOW', 'MEDIUM', 'HIGH'], default=['LOW', 'MEDIUM', 'HIGH'])
    dff = df[df['Quality_Class'].isin(qp) & df['Risk_Class'].isin(rp)]
    if len(dff) == 0:
        st.warning("No data matches the selected filters. Widen the selection.")
        return
    c1, c2 = st.columns(2)
    with c1:
        fig = px.pie(dff, names='Quality_Class', title='Treatment-condition distribution',
                     color='Quality_Class', color_discrete_map=QCOLOR)
        fig.update_layout(height=320)
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = px.pie(dff, names='Risk_Class', title='Risk-indicator distribution',
                     color='Risk_Class', color_discrete_map=RCOLOR)
        fig.update_layout(height=320)
        st.plotly_chart(fig, width="stretch")

    st.subheader("Parameter distributions & box plots")
    param = st.selectbox("Parameter", ['PH_S', 'DBO_S', 'DQO_S', 'SS_S', 'COND_S', 'PH_E', 'DBO_E', 'DQO_E', 'SS_E', 'COND_E', 'ZN_E'])
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(dff, x=param, nbins=30, title=f"{param} distribution")
        fig.update_layout(height=300)
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = px.box(dff, y=param, title=f"{param} box plot")
        fig.update_layout(height=300)
        st.plotly_chart(fig, width="stretch")

    st.subheader("Correlation heatmap (key parameters)")
    keys = [k for k in ['PH_E', 'DBO_E', 'DQO_E', 'SS_E', 'COND_E', 'ZN_E', 'PH_S', 'DBO_S', 'DQO_S', 'SS_S', 'COND_S'] if k in dff.columns]
    corr = dff[keys].corr(numeric_only=True)
    fig = px.imshow(corr, text_auto='.2f', aspect='auto', zmin=-1, zmax=1, color_continuous_scale='Tealrose',
                    title='Correlation heatmap')
    fig.update_layout(height=480)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Historical parameter trend")
    tp = st.selectbox("Trend parameter", ['PH_S', 'DBO_S', 'DQO_S', 'SS_S', 'COND_S'], key='trend')
    unit = OPERATIONAL_LIMITS[tp]['unit']
    dft = dff.sort_values('Date')
    fig = px.line(dft, x='Date', y=tp, title=f"{tp} over time ({unit})")
    fig.update_layout(height=300)
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Monthly pattern (valid 1990 timestamps)")
        mo = dff.groupby([dff['Date'].dt.month, 'Quality_Class'], observed=True).size().unstack(fill_value=0)
        fig = px.bar(mo, title='Records per month × condition', color_discrete_map=QCOLOR)
        fig.update_layout(height=300)
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.subheader("Seasonal pattern")
        so = dff.groupby(['Season', 'Quality_Class'], observed=True).size().unstack(fill_value=0)
        fig = px.bar(so, title='Records per season × condition', color_discrete_map=QCOLOR)
        fig.update_layout(height=300)
        st.plotly_chart(fig, width="stretch")

    st.subheader("Risk vs condition cross-analysis")
    cross = pd.crosstab(dff['Quality_Class'], dff['Risk_Class'])
    fig = px.imshow(cross, text_auto=True, aspect='auto', title='Quality × Risk counts', color_continuous_scale='teal')
    fig.update_layout(height=300)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Operational-limit exceedance (treated output)")
    rows = []
    checks = {'DBO_S': 30.0, 'DQO_S': 100.0, 'SS_S': 100.0, 'COND_S': 1500.0}
    for p, t in checks.items():
        c = int((dff[p] > t).sum())
        rows.append({'Parameter': p, 'Critical limit': t, 'Exceedances': c, 'Exceedance %': round(c / len(dff) * 100, 1)})
    edf = pd.DataFrame(rows)
    fig = px.bar(edf, x='Parameter', y='Exceedance %', title='Share of records above critical operational limit')
    fig.update_layout(height=300)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(edf, width="stretch")

# ------------------------------------------------------------- models -----
def show_models(qm, rm):
    st.header("ML Model Insights")
    if qm is None or qm.get('metadata') is None:
        st.warning("No trained models found. Run `py src/pipeline.py` first.")
        return
    for title, md in [("Treatment-condition classification", qm), ("Contamination-risk classification", rm)]:
        meta = md.get('metadata') if md else None
        if not meta:
            st.info(f"{title}: model artifact missing.")
            continue
        st.subheader(title)
        st.caption(f"Best model: **{meta['best_model_name']}** · Classes: {meta['classes']} · "
                   f"Features: {len(meta['feature_names'])} · Trained: {meta.get('created_at', 'unknown')} · "
                   "Evaluation: stratified 80/20 split + 5-fold CV (F1 macro), scaler fit on train only.")
        comp = pd.DataFrame([{'Model': k, 'Accuracy': v['Accuracy'], 'F1 macro': v['F1 Macro'],
                              'F1 weighted': v['F1 Weighted'], 'ROC-AUC': v['ROC-AUC'],
                              'CV F1 mean': v.get('CV F1 Macro (mean)'), 'CV F1 std': v.get('CV F1 Macro (std)')}
                             for k, v in meta['results'].items()])
        st.dataframe(comp.round(4), width="stretch")
        fig = px.bar(comp, x='Model', y=['Accuracy', 'F1 macro', 'F1 weighted'],
                     barmode='group', title=f"{title} — metric comparison")
        fig.update_layout(height=320)
        st.plotly_chart(fig, width="stretch")
        if hasattr(md['model'], 'feature_importances_'):
            imp = pd.DataFrame({'Feature': meta['feature_names'],
                                'Importance': [float(x) for x in md['model'].feature_importances_]})
            imp = imp.sort_values('Importance', ascending=False).head(15)
            fig = px.bar(imp, x='Importance', y='Feature', orientation='h',
                         title='Top 15 feature importances (tree model)', color='Importance',
                         color_continuous_scale='teal')
            fig.update_layout(height=420)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Feature importance is only shown for tree-based models that expose it reliably.")
    st.subheader("Confusion matrices (from training pipeline)")
    for f in ['confusion_matrix_quality.png', 'confusion_matrix_risk.png']:
        p = os.path.join(FIGURES_DIR, f)
        if os.path.exists(p):
            st.image(p, caption=f)
    st.info("SHAP explainability is not shown because the `shap` package is not installed in this environment "
            "(see requirements.txt). Tree-model feature importance above is the reliable fallback; "
            "no per-prediction SHAP values are fabricated.")
    st.caption("Known limitation carried over from the original pipeline: some engineered training features "
               "(removal efficiencies, averages including treated output) use treated-output values, which leaks "
               "output information into training. Saved artifacts are preserved as-is; new-sample predictions set "
               "removal efficiencies to 0 and average only E/P/D stages, and out-of-range warnings flag extrapolation.")

# ------------------------------------------------------------ reports -----
def show_reports(df, qm, rm):
    st.header("Reports & History")
    st.subheader("Sample history (persistent)")
    sdf = HI.history_summary(HI.load_samples())
    if len(sdf) == 0:
        st.info("No saved samples yet. Test a sample and press “Save this sample to history”.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            sq = st.text_input("Search ID / name / location", "")
        with c2:
            fq = st.selectbox("Quality", ["All", "GOOD", "MODERATE", "POOR"])
        with c3:
            fr = st.selectbox("Risk", ["All", "LOW", "MEDIUM", "HIGH"])
        with c4:
            fs = st.selectbox("Source", ["All", "MANUAL", "HISTORICAL", "SIMULATED", "LIVE"])
        d1, d2 = st.columns(2)
        with d1:
            date_from = st.date_input("From date", value=None)
        with d2:
            date_to = st.date_input("To date", value=None)
        view = HI.filter_samples(sdf, search=sq, quality=fq, risk=fr, source=fs,
                                date_from=date_from.isoformat() if date_from else None,
                                date_to=date_to.isoformat() if date_to else None)
        if len(view) == 0:
            st.info("No saved samples match the filters.")
        else:
            st.dataframe(view.drop(columns=['inputs_json']), width="stretch")
            locs = sorted(view['location'].dropna().astype(str).unique())
            if len(locs) > 1 or (len(locs) == 1 and locs[0]):
                loc = st.selectbox("Compare readings for location", locs)
                lv = view[view['location'] == loc].sort_values('created_at')
                if len(lv) >= 2:
                    st.line_chart(pd.DataFrame({'quality_conf': lv['quality_conf'].astype(float).values},
                                               index=lv['created_at'].values))
                    st.caption(f"{len(lv)} saved reading(s) at “{loc}” over time (prediction confidence).")
            buf = io.StringIO()
            view.to_csv(buf, index=False)
            st.download_button("⬇ Export sample history (CSV)", buf.getvalue(), "aquaguard_samples.csv", "text/csv")

            st.subheader("Individual sample report")
            rid = st.selectbox("Choose record", view['record_id'].tolist())
            rec = view[view['record_id'] == rid].iloc[0].to_dict()
            try:
                inputs = json.loads(rec.get('inputs_json') or '{}')
            except Exception:
                inputs = {}
            lines = [f"# AquaGuard AI sample report — {rec['record_id']}",
                     f"- Sample: {rec['sample_name']} | Location: {rec['location'] or 'not given'} | Collected: {rec['collected_at'] or 'not given'}",
                     f"- Created: {rec['created_at']} | Source mode: {rec['source_mode']}",
                     f"- Treatment condition: **{rec['quality_class']}** (confidence {rec['quality_conf']})",
                     f"- Risk indicator: **{rec['risk_class']}** (confidence {rec['risk_conf']})",
                     "- Readings:"] + [f"  - {k}: {v}" for k, v in inputs.items()]
            lines += [f"- Model: {(qm or {}).get('metadata', {}).get('best_model_name', 'unknown')} "
                      f"(features: {len((qm or {}).get('metadata', {}).get('feature_names', []))})",
                      f"- Limitation: {DISCLAIMER}"]
            report = "\n".join(lines)
            st.text_area("Report preview", report, height=220)
            st.download_button("⬇ Download report (.md)", report, f"{rec['record_id']}.md", "text/markdown")
            try:
                import importlib
                if importlib.util.find_spec("fpdf") or importlib.util.find_spec("reportlab"):
                    st.caption("A PDF library is available — extend export here if needed.")
                else:
                    st.caption("PDF export: no PDF library installed (fpdf/reportlab absent), so Markdown/CSV export is provided instead. "
                               "Install `fpdf2` to enable one-click PDF.")
            except Exception:
                pass

    st.subheader("Live connection guide")
    st.code("copy .env.example .env   # then set AQUAGUARD_LIVE_CSV / _JSON / _URL", language="bash")


@st.cache_data(ttl=86400)
def _govt_states(kind):
    try:
        return GV.list_states(kind)
    except Exception:
        return {}


def show_government():
    """Government Water Quality Data page: genuine CPCB/NWDP observations only."""
    st.header("Government Water Quality Data")
    st.caption("Official CPCB surface-water observations via the National Water Data Portal. "
               "Displayed independently — never fed into the UCI-trained treatment models "
               "(incompatible features, units and target meaning).")

    c1, c2, c3 = st.columns(3)
    with c1:
        kind = st.selectbox("Dataset type", ["physical", "chemical", "biological"],
                            help="Physical: monthly. Chemical: annual. Biological: monthly.")
    states_map = _govt_states(kind) or {}
    state_opts = sorted(states_map) or ["Tamil Nadu"]
    with c2:
        state = st.selectbox("State", state_opts,
                             index=state_opts.index("Tamil Nadu") if "Tamil Nadu" in state_opts else 0)
    period_opts = states_map.get(state, ["2021 - 2025"])
    # Default to the latest period known to hold published observations ("2021 - 2025").
    # Later buckets (e.g. 2026 - 2030) exist on the portal but are header-only files
    # with 0 data rows, so defaulting to the last entry breaks first-time loads.
    # NOTE: no widget key here on purpose — the fallback button below drives the
    # selection through 'govt_period_pref' (setting a widget key post-creation
    # raises StreamlitAPIException).
    _def_period = "2021 - 2025" if "2021 - 2025" in period_opts else period_opts[-1]
    _want = st.session_state.get("govt_period_pref", _def_period)
    with c3:
        period = st.selectbox("Period", period_opts,
                              index=period_opts.index(_want) if _want in period_opts
                              else period_opts.index(_def_period))

    if st.button("🔄 Refresh Government Data", type="primary"):
        with st.spinner("Contacting nwdp.nwic.gov.in ..."):
            st.session_state['govt'] = GV.refresh(kind, state, period)
        st.rerun()

    if 'govt' not in st.session_state:
        # First visit: one genuine fetch attempt (falls back to last good cache).
        with st.spinner("Contacting nwdp.nwic.gov.in ..."):
            st.session_state['govt'] = GV.refresh(kind, state, period)

    res = st.session_state['govt']
    if res['status'] == 'error' or res['df'] is None:
        st.error(res['message'])
        st.caption(f"Dataset catalogue: {GV.CATALOGUE_URL}")
        if "2021 - 2025" in period_opts and period != "2021 - 2025":
            if st.button("⬇ Load 2021–2025 instead (has published observations)"):
                st.session_state["govt_period_pref"] = "2021 - 2025"
                with st.spinner("Contacting nwdp.nwic.gov.in ..."):
                    st.session_state['govt'] = GV.refresh(kind, state, "2021 - 2025")
                st.rerun()
        st.info("Manual fallback: download the state CSV from the catalogue page above, then load it "
                "via Alerts & Monitoring > LIVE mode (AQUAGUARD_LIVE_CSV) for threshold checks.")
        return

    df, meta = res['df'], res['meta']
    if res['fresh']:
        st.success(f"🏛 OFFICIAL GOVERNMENT DATA — {res['message']}")
    else:
        st.warning(f"🏛 OFFICIAL GOVERNMENT DATA (cached) — {res['message']}")
    for n in res.get('notes', []):
        st.caption(n)
    st.markdown(f"**Dataset:** {meta['dataset']}  ")
    st.markdown(f"**Source file:** [{meta['source_url']}]({meta['source_url']}) · "
                f"[dataset page]({meta['portal_page']})  ")
    st.caption(f"Collection frequency: {meta.get('frequency', 'see portal')} · "
               f"Latest observation: {meta.get('latest_observation', 'unknown')} · "
               f"Last retrieval: {meta.get('retrieved_at', 'unknown')} · Rows: {meta.get('rows', len(df))}")

    # Water Quality Explorer — metric cards (actual loaded data, never hardcoded)
    st.subheader("Water Quality Explorer")
    st.caption("Each card below is computed from the retrieved records. "
               "Use the filters to narrow by station and observation date.")
    _n_stations = df['Station'].dropna().astype(str).nunique() if 'Station' in df.columns else 0
    _valid_dt = df["_observation_dt"].dropna() if "_observation_dt" in df.columns else pd.Series(dtype=float)
    _latest = _valid_dt.max() if len(_valid_dt) else None
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"<div class='kpi'><div class='v'>{len(df)}</div><div class='l'>Available records</div></div>", unsafe_allow_html=True)
    with m2:
        st.markdown(f"<div class='kpi'><div class='v'>{_n_stations}</div><div class='l'>Monitoring stations</div></div>", unsafe_allow_html=True)
    with m3:
        st.markdown(f"<div class='kpi'><div class='v'>{_latest.date() if _latest is not None else '–'}</div><div class='l'>Latest observation</div></div>", unsafe_allow_html=True)
    with m4:
        st.markdown(f"<div class='kpi'><div class='v'>{len(GV.numeric_params(df, kind))}</div><div class='l'>Measured parameters</div></div>", unsafe_allow_html=True)

    # Filters
    stations = sorted(df['Station'].dropna().astype(str).unique()) if 'Station' in df.columns else []
    all_params = GV.numeric_params(df, kind)
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        sel_station = st.selectbox("Station", ["All stations"] + stations)
    valid_dt = df["_observation_dt"].dropna()
    with f2:
        dmin = st.date_input("From", valid_dt.min().date() if len(valid_dt) else None)
    with f3:
        dmax = st.date_input("To", valid_dt.max().date() if len(valid_dt) else None)
    with f4:
        sel_params = st.multiselect("Parameters shown", all_params, default=all_params,
                                    help="Choose which measurement columns appear in the table below.")
    view = df.copy()
    if sel_station != "All stations":
        view = view[view['Station'].astype(str) == sel_station]
    if dmin:
        view = view[view["_observation_dt"] >= pd.Timestamp(dmin)]
    if dmax:
        view = view[view["_observation_dt"] <= pd.Timestamp(dmax) + pd.Timedelta(days=1)]
    if len(view) == 0:
        st.warning("No government records match the current filters.")
        return

    st.subheader(f"Records ({len(view)})")
    _id_cols = [c for c in ["Station", "District", "Data Acquisition Time", "Latitude", "Longitude"] if c in view.columns]
    show_cols = _id_cols + [p for p in sel_params if p in view.columns]
    st.dataframe(view[show_cols] if show_cols else view.head(0), width="stretch")
    st.caption("Each row is one published government observation. Missing values are shown as blank/NaN "
               "and are excluded from charts — never filled with invented numbers.")

    params = GV.numeric_params(view, kind)
    if params and valid_dt is not None and view["_observation_dt"].notna().sum() >= 2:
        st.subheader("Trend")
        t1, t2 = st.columns(2)
        with t1:
            param = st.selectbox("Parameter", params)
        with t2:
            tstations = sorted(view['Station'].dropna().astype(str).unique()) if 'Station' in view.columns else []
            tstation = st.selectbox("Trend station", tstations if tstations else ["All stations"])
        tv = view.copy()
        if tstation != "All stations":
            tv = tv[tv['Station'].astype(str) == tstation]
        tv = tv.dropna(subset=["_observation_dt", param]).sort_values("_observation_dt")
        if len(tv) >= 2:
            fig = px.line(tv, x="_observation_dt", y=param, color="Station" if tstation == "All stations" else None,
                          title=f"{param} over time ({tstation})")
            fig.update_layout(height=320)
            st.plotly_chart(fig, width="stretch")
            st.caption("How to read: each point is a published observation (monthly for physical/biological, "
                       "annual for chemical). Rising lines mean increasing values at that station; gaps mean "
                       "no observation was published for that period.")
        else:
            st.info("Not enough valid dated observations for a trend with this selection.")
    else:
        st.info("Trend unavailable: need at least 2 valid dated numeric observations.")

    # Station map (real coordinates only — never invented)
    st.subheader("Station map")
    _pts, _missing = GV.map_points(view)
    if len(_pts):
        fig = px.scatter_map(_pts, lat="Latitude", lon="Longitude", hover_name="Station",
                             hover_data={"District": True, "records": True, "Latitude": False, "Longitude": False},
                             zoom=6, height=420, title="Monitoring stations (official coordinates)")
        fig.update_layout(map_style="open-street-map", margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, width="stretch")
        st.caption("How to read: each dot is a monitoring station at its official published coordinates; "
                   "hover for station name, district and record count.")
        if _missing:
            st.caption(f"{_missing} station(s) have no valid coordinates and are listed in the table only.")
    else:
        st.info("No valid coordinates in the retrieved records, so no map can be drawn. "
                "Station names and districts remain available in the records table above.")

    # Station comparison (two stations, one parameter)
    st.subheader("Station comparison")
    st.caption("Compare the same parameter at two stations. Statistics use only valid (non-missing) values.")
    if len(stations) >= 2 and params:
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            st_a = st.selectbox("Station A", stations, index=0)
        with cc2:
            st_b = st.selectbox("Station B", stations, index=min(1, len(stations) - 1))
        with cc3:
            cmp_param = st.selectbox("Comparison parameter", params)
        if st_a == st_b:
            st.info("Pick two different stations to compare.")
        else:
            cmp = GV.compare_station_stats(view, st_a, st_b, cmp_param)
            if not cmp["comparable"]:
                st.warning("Not enough valid data to compare these stations on this parameter.")
            else:
                r1, r2 = st.columns(2)
                for col, key in [(r1, "a"), (r2, "b")]:
                    s = cmp[key]
                    with col:
                        st.markdown(f"<div class='kpi'><div class='v'>{s['mean']:.2f}</div>"
                                    f"<div class='l'>{s['station']} — mean of {s['n']} readings "
                                    f"(min {s['min']:.2f}, max {s['max']:.2f})</div></div>", unsafe_allow_html=True)
                both = view[view['Station'].astype(str).isin([st_a, st_b])].dropna(
                    subset=["_observation_dt", cmp_param]).sort_values("_observation_dt")
                if len(both) >= 2:
                    fig = px.line(both, x="_observation_dt", y=cmp_param, color="Station",
                                  title=f"{cmp_param}: {st_a} vs {st_b}")
                    fig.update_layout(height=320)
                    st.plotly_chart(fig, width="stretch")
    else:
        st.info("Comparison needs at least 2 stations and 1 numeric parameter in the current filter.")

    # Explainable monitoring: user-configured thresholds (NOT CPCB standards)
    st.subheader("Watchlist alerts (your thresholds)")
    st.caption("Set your own watch limits for the parameters above. These are YOUR operational limits — "
               "they are not CPCB standards and say nothing about drinking-water safety.")
    _thr = st.session_state.get("govt_thresholds", {})
    with st.expander("Configure thresholds", expanded=False):
        _new_thr = {}
        for p in params:
            _wk, _ck = f"thr_w_{kind}_{p}", f"thr_c_{kind}_{p}"
            st.session_state.setdefault(_wk, float(_thr.get(p, {}).get("warn", 0.0)))
            st.session_state.setdefault(_ck, float(_thr.get(p, {}).get("critical", 0.0)))
            c1, c2 = st.columns(2)
            with c1:
                w = st.number_input(f"{p} — Warning above", min_value=0.0, key=_wk)
            with c2:
                c = st.number_input(f"{p} — Critical above", min_value=0.0, key=_ck)
            if w > 0 and c > 0:
                _new_thr[p] = {"warn": w, "critical": c}
        if st.button("Apply watchlist"):
            st.session_state["govt_thresholds"] = _new_thr
            st.success(f"Watching {len(_new_thr)} parameter(s).")
            st.rerun()
    _active = st.session_state.get("govt_thresholds", {})
    if not _active:
        st.info("No watch limits configured yet — open “Configure thresholds” above.")
    else:
        _alerts, _terr = GV.evaluate_user_thresholds(view, _active)
        for e in _terr:
            st.error(e)
        if len(_alerts) == 0:
            st.success("Normal — no filtered reading exceeds your configured limits.")
        else:
            sev_f = st.selectbox("Severity filter", ["All", "Warning", "Critical"])
            _show = _alerts if sev_f == "All" else _alerts[_alerts["severity"] == sev_f]
            st.dataframe(_show, width="stretch")
            st.caption("Each alert shows station, date, measured value, your limit, and the reason it fired.")

    buf = io.StringIO()
    view.to_csv(buf, index=False)
    st.download_button("⬇ Download retrieved records (CSV)", buf.getvalue(),
                       f"govt_{kind}_{state.replace(' ', '_')}.csv", "text/csv")
    # Browser downloads go to the browser's own Downloads folder, which users
    # often miss — so also keep a copy inside the project and show its path.
    try:
        export_dir = os.path.join(BASE_DIR, "data", "app_data", "govt_exports")
        os.makedirs(export_dir, exist_ok=True)
        export_path = os.path.join(export_dir, f"govt_{kind}_{state.replace(' ', '_')}.csv")
        view.to_csv(export_path, index=False)
        st.caption(f"Browser download: check your browser's Downloads folder. "
                   f"Project copy saved ({len(view)} rows): `{export_path}`")
    except Exception as e:
        st.caption(f"Browser download: check your browser's Downloads folder. "
                   f"(Project copy failed: {e})")
    st.warning("These are periodically published surface-water observations (monthly/annual), not real-time "
               "sensor data. ML predictions on this page are intentionally disabled: the deployed models were "
               "trained on UCI wastewater-treatment-plant features and are incompatible with this schema. "
               "A separately trained and validated model would be needed.")


if __name__ == '__main__':
    main()
