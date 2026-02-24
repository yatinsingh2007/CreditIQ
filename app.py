import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.graph_objects as go
import altair as alt

st.set_page_config(
    page_title="CreditIQ — Credit Risk Intelligence",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="auto"
)

# ─── GLOBAL STYLES ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700;800&family=Source+Serif+4:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif !important;
    font-size: 14px;
    -webkit-font-smoothing: antialiased;
}

/* ── App Background: Pure White ── */
.stApp {
    background: #FFFFFF;
    color: #000000;
}

/* ── Hide Streamlit branding ── */
#MainMenu, footer { visibility: hidden; }
[data-testid="stToolbar"] { visibility: hidden !important; }

[data-testid="stHeader"] {
    background: #FFFFFF !important;
    border-bottom: 1px solid #CCCCCC;
}

/* ── Sidebar Toggle Buttons ── */
button[kind="header"] {
    color: #000000 !important;
    background-color: transparent !important;
}
button[aria-label="Expand sidebar"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stExpandSidebarButton"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    align-items: center !important;
    justify-content: center !important;
    color: #000000 !important;
    background: #FFFFFF !important;
    border: 1px solid #CCCCCC !important;
    border-radius: 4px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
    margin-top: 5px !important;
    margin-left: 5px !important;
    transition: all 0.2s ease !important;
    z-index: 999999 !important;
}
div:has(> [data-testid="stExpandSidebarButton"]) {
    visibility: visible !important;
    z-index: 999999 !important;
}
button[aria-label="Expand sidebar"]:hover,
[data-testid="collapsedControl"]:hover,
[data-testid="stSidebarCollapsedControl"]:hover,
[data-testid="stExpandSidebarButton"]:hover {
    background: #F5F5F5 !important;
    border-color: #000000 !important;
}
button[aria-label="Expand sidebar"] svg,
[data-testid="collapsedControl"] svg,
[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="stExpandSidebarButton"] svg {
    fill: currentColor !important;
}

/* ── Sidebar Close ── */
button[aria-label="Collapse sidebar"],
[data-testid="stSidebarCollapseButton"] {
    visibility: visible !important;
    color: #222222 !important;
    transition: color 0.2s ease !important;
}
button[aria-label="Collapse sidebar"]:hover,
[data-testid="stSidebarCollapseButton"]:hover {
    color: #000000 !important;
}
button[aria-label="Collapse sidebar"] svg,
[data-testid="stSidebarCollapseButton"] svg {
    fill: currentColor !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #F5F5F5 !important;
    border-right: 1px solid #CCCCCC !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 2rem;
}

/* ── Sidebar Brand ── */
.sidebar-brand {
    text-align: center;
    padding: 2rem 1rem 1.5rem;
    border-bottom: 2px solid #000000;
    margin-bottom: 2rem;
}
.sidebar-brand .logo {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #000000;
    letter-spacing: -0.5px;
    display: block;
}
.sidebar-brand .tagline {
    font-size: 0.72rem;
    color: #222222;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-top: 0.5rem;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
}

/* ── Model Badge ── */
.model-badge {
    background: #FFFFFF;
    border: 1px solid #CCCCCC;
    border-left: 3px solid #000000;
    border-radius: 4px;
    padding: 0.85rem 1rem;
    margin: 0.8rem 0;
    font-size: 0.82rem;
    transition: box-shadow 0.2s ease;
}
.model-badge:hover {
    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
}
.model-badge .badge-label {
    color: #222222;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 0.62rem;
    margin-bottom: 0.3rem;
    font-weight: 700;
}
.model-badge .badge-value {
    color: #000000;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    font-size: 0.85rem;
}
.model-badge .badge-sub {
    color: #222222;
    font-size: 0.72rem;
    margin-top: 0.2rem;
}

/* ── Page Header ── */
.page-header {
    padding: 2rem 0 1.5rem;
    border-bottom: 2px solid #000000;
    margin-bottom: 3rem;
}
.page-header h1 {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: #000000;
    margin: 0;
    letter-spacing: -0.5px;
    line-height: 1.2;
}
.page-header p {
    color: #222222;
    margin: 0.6rem 0 0;
    font-size: 1rem;
    font-style: italic;
}

/* ── Metric Cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1.5rem;
    margin-bottom: 3rem;
}
.metric-card {
    background: #FFFFFF;
    border: 1px solid #000000;
    border-radius: 4px;
    padding: 1.5rem 1.8rem;
    transition: box-shadow 0.2s ease;
}
.metric-card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}
.metric-card .m-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #222222;
    margin-bottom: 0.7rem;
    font-weight: 700;
}
.metric-card .m-value {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #000000;
    line-height: 1;
}
.metric-card .m-sub {
    font-size: 0.82rem;
    color: #222222;
    margin-top: 0.6rem;
    font-style: italic;
}

/* ── Section Title ── */
.section-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: #000000;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 3rem 0 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #CCCCCC;
}

/* ── Feature Tags ── */
.feature-tags { display: flex; flex-wrap: wrap; gap: 0.6rem; margin-top: 0.8rem; }
.feature-tag {
    background: #FFFFFF;
    border: 1px solid #000000;
    color: #000000;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    padding: 0.45rem 0.9rem;
    border-radius: 3px;
    transition: all 0.2s ease;
    font-weight: 500;
}
.feature-tag:hover {
    background: #000000;
    color: #FFFFFF;
}

/* ── Tab Overrides ── */
[data-testid="stTab"] button {
    font-family: 'Playfair Display', Georgia, serif !important;
    font-weight: 600;
    color: #222222 !important;
    font-size: 1rem;
}
[data-testid="stTab"] button[aria-selected="true"] {
    color: #000000 !important;
    border-bottom-color: #000000 !important;
    font-weight: 700;
}

/* ── Perf Cards (4-up) ── */
.perf-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1.2rem;
    margin-bottom: 2rem;
}
.perf-card {
    background: #FFFFFF;
    border: 1px solid #000000;
    border-radius: 4px;
    padding: 1.3rem 1.5rem;
    text-align: center;
    transition: box-shadow 0.2s ease;
}
.perf-card:hover {
    box-shadow: 0 3px 10px rgba(0,0,0,0.08);
}
.perf-card .pc-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #222222;
    margin-bottom: 0.5rem;
    font-weight: 700;
}
.perf-card .pc-value {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #000000;
}

/* ── Report Table ── */
.report-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
    border: 1px solid #000000;
}
.report-table th {
    background: #F5F5F5;
    color: #000000;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    padding: 0.8rem 1rem;
    text-align: right;
    border-bottom: 2px solid #000000;
    font-weight: 700;
}
.report-table th:first-child { text-align: left; }
.report-table td {
    padding: 0.85rem 1rem;
    border-bottom: 1px solid #CCCCCC;
    color: #000000;
    text-align: right;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
}
.report-table td:first-child {
    text-align: left;
    font-family: 'Inter', sans-serif;
    color: #000000;
    font-weight: 600;
    font-size: 0.9rem;
}
.report-table tr.divider td { border-top: 2px solid #000000; }
.report-table tr:hover td { background: #F5F5F5; }

/* ── Predict Form ── */
.form-section-label {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: #000000;
    font-weight: 700;
    margin: 1.5rem 0 0.8rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #000000;
}

/* ── Result Panel ── */
.result-card {
    border-radius: 4px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin-bottom: 1.5rem;
    background: #FFFFFF;
}
.result-card.good {
    border: 2px solid #000000;
}
.result-card.bad {
    border: 2px solid #000000;
}
.result-card .result-icon {
    font-size: 1rem;
    margin-bottom: 0.8rem;
    display: block;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #000000;
}
.result-card .result-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1.8rem;
    font-weight: 700;
    margin-bottom: 0.4rem;
    color: #000000;
}
.result-card .result-sub {
    color: #222222;
    font-size: 0.9rem;
    font-style: italic;
}

.risk-badge {
    display: inline-block;
    padding: 0.4rem 1.2rem;
    border-radius: 3px;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-top: 1rem;
    font-family: 'IBM Plex Mono', monospace;
    background: #FFFFFF;
    color: #000000;
    border: 1px solid #000000;
}
.risk-low { background: #FFFFFF; color: #000000; border: 2px solid #000000; }
.risk-med { background: #F5F5F5; color: #000000; border: 2px solid #000000; }
.risk-high { background: #000000; color: #FFFFFF; border: 2px solid #000000; }

.prob-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    color: #000000;
    margin-bottom: 0.4rem;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
}
.prob-track {
    background: #F5F5F5;
    border: 1px solid #CCCCCC;
    border-radius: 2px;
    height: 10px;
    overflow: hidden;
    margin-bottom: 1.5rem;
}
.prob-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.6s ease;
    background: #000000;
}

.summary-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem 0;
    border-bottom: 1px solid #CCCCCC;
    font-size: 0.9rem;
}
.summary-row .sr-label {
    color: #222222;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 700;
}
.summary-row .sr-value {
    color: #000000;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
}

.fi-row {
    display: flex;
    align-items: center;
    gap: 0.9rem;
    padding: 0.65rem 0;
    border-bottom: 1px solid #CCCCCC;
    transition: background 0.2s ease;
}
.fi-row:hover {
    background: #F5F5F5;
}
.fi-rank {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #000000;
    font-weight: 700;
    width: 1.5rem;
    flex-shrink: 0;
}
.fi-name { color: #000000; font-size: 0.88rem; flex: 1; font-weight: 500; }
.fi-bar-wrap { width: 80px; background: #F5F5F5; border: 1px solid #CCCCCC; border-radius: 2px; height: 6px; }
.fi-bar { height: 100%; background: #000000; border-radius: 2px; }
.fi-score { font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; color: #000000; width: 3.5rem; text-align: right; }

/* ── Callout Box ── */
.callout-box {
    background: #F5F5F5;
    border: 1px solid #000000;
    border-left: 4px solid #000000;
    border-radius: 4px;
    padding: 1rem 1.2rem;
    margin-top: 1.2rem;
    font-size: 0.88rem;
    color: #000000;
}

/* ── Fade-in Animation ── */
.fade-in { animation: fadeUp 0.4s ease both; }
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Override Streamlit slider ── */
[data-testid="stSlider"] { padding-top: 0.2rem; }

/* ── Submit Button: Black BG, White Text, Hover Inverts ── */
[data-testid="stFormSubmitButton"] button {
    background: #000000 !important;
    color: #FFFFFF !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    border: 2px solid #000000 !important;
    border-radius: 4px !important;
    padding: 0.75rem 1.5rem !important;
    font-size: 0.95rem !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.05em !important;
}
[data-testid="stFormSubmitButton"] button:hover {
    background: #FFFFFF !important;
    color: #000000 !important;
}

/* ── Override Streamlit radio ── */
[data-testid="stRadio"] label {
    font-size: 0.92rem !important;
    padding: 0.3rem 0 !important;
    font-family: 'Inter', sans-serif !important;
    color: #000000 !important;
    font-weight: 500 !important;
}

/* ── Override Streamlit form inputs ── */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
    background: #FFFFFF !important;
    color: #000000 !important;
    border: 1px solid #CCCCCC !important;
    border-radius: 4px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.92rem !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus {
    border-color: #000000 !important;
    box-shadow: none !important;
}

/* ── Override Streamlit selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: #FFFFFF !important;
    color: #000000 !important;
    border: 1px solid #CCCCCC !important;
    border-radius: 4px !important;
}
[data-testid="stSelectbox"] label {
    color: #000000 !important;
    font-weight: 600 !important;
}

/* ── Override all Streamlit labels ── */
.stSlider label, .stSelectbox label, .stNumberInput label, .stTextInput label {
    color: #000000 !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
}

/* ── Subheader ── */
h2, h3 {
    color: #000000 !important;
    font-family: 'Playfair Display', Georgia, serif !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #FFFFFF; }
::-webkit-scrollbar-thumb { background: #CCCCCC; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #999999; }

/* ── Sidebar radio labels ── */
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: #000000 !important;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: #FFFFFF;
}
</style>
""", unsafe_allow_html=True)


# ─── DATA LOADING ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    for path in ["dt_model.pkl", "model/dt_model.pkl"]:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    return None

pkg = load_model()

if pkg is None:
    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:center;height:60vh;
                flex-direction:column;gap:1rem;background:#FFFFFF;">
        <div style="font-family:'Playfair Display',Georgia,serif;font-size:1.8rem;
                    color:#000000;font-weight:700;">Model Not Found</div>
        <div style="color:#222222;font-size:0.95rem;">
            Place dt_model.pkl in the same directory as app.py</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

model          = pkg["model"]
lr_model       = pkg["lr_model"]
scaler         = pkg["scaler"]
encoders       = pkg.get("encoders", {})
feature_cols   = pkg["feature_columns"]
dt_threshold   = pkg.get("dt_threshold", 0.35)
lr_threshold   = pkg.get("lr_threshold", 0.35)
dinfo          = pkg["dataset_info"]
dtm            = pkg["dt_metrics"]
lrm            = pkg.get("lr_metrics", {})

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <span class="logo">CreditIQ</span>
        <span class="tagline">Risk Intelligence Platform</span>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("", ["Overview", "Performance", "Predict"], label_visibility="collapsed")

    st.markdown(f"""
    <div class="model-badge">
        <div class="badge-label">Primary Model</div>
        <div class="badge-value">Decision Tree Classifier</div>
        <div class="badge-sub">Threshold: {dt_threshold} &middot; Accuracy: {dtm.get('test_accuracy',0)*100:.2f}%</div>
    </div>
    <div class="model-badge">
        <div class="badge-label">Secondary Model</div>
        <div class="badge-value">Logistic Regression</div>
        <div class="badge-sub">Threshold: {lr_threshold} &middot; Accuracy: {lrm.get('test_accuracy',0)*100:.2f}%</div>
    </div>
    <div class="model-badge">
        <div class="badge-label">ROC-AUC Scores</div>
        <div class="badge-value">DT: {dtm.get('roc_auc',0):.4f} &middot; LR: {lrm.get('roc_auc',0):.4f}</div>
        <div class="badge-sub">Area under curve</div>
    </div>
    <div class="model-badge">
        <div class="badge-label">Dataset</div>
        <div class="badge-value">{dinfo.get('total_samples',0):,} samples</div>
        <div class="badge-sub">{dinfo.get('n_features',0)} features &middot; Binary classification</div>
    </div>
    """, unsafe_allow_html=True)


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def _get_default_class(class_metrics):
    return next((v for k, v in class_metrics.items() if "1" in str(k)), {})

def _get_good_class(class_metrics):
    return next((v for k, v in class_metrics.items() if "0" in str(k)), {})

def fmt(v, pct=False):
    if v is None: return "—"
    return f"{v*100:.2f}%" if pct else f"{v:.4f}"

def _matplotlib_light():
    plt.rcParams.update({
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor":   "#FFFFFF",
        "axes.edgecolor":   "#CCCCCC",
        "axes.labelcolor":  "#000000",
        "xtick.color":      "#000000",
        "ytick.color":      "#000000",
        "text.color":       "#000000",
        "grid.color":       "#F5F5F5",
        "grid.linestyle":   "-",
        "grid.alpha":       1.0,
        "font.family":      "serif",
        "font.size":        11,
    })

_matplotlib_light()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown("""
    <div class="page-header">
        <h1>Credit Risk Intelligence</h1>
        <p>Machine learning pipeline to classify loan applicants as Good Loan or Default</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Metric Cards ──────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card fade-in">
            <div class="m-label">Total Samples</div>
            <div class="m-value">{dinfo.get('total_samples',0):,}</div>
            <div class="m-sub">Full dataset size</div>
        </div>
        <div class="metric-card fade-in" style="animation-delay:0.05s">
            <div class="m-label">Training Samples</div>
            <div class="m-value">{dinfo.get('train_samples',0):,}</div>
            <div class="m-sub">80% split</div>
        </div>
        <div class="metric-card fade-in" style="animation-delay:0.1s">
            <div class="m-label">Test Samples</div>
            <div class="m-value">{dinfo.get('test_samples',0):,}</div>
            <div class="m-sub">20% split</div>
        </div>
        <div class="metric-card fade-in" style="animation-delay:0.15s">
            <div class="m-label">Features</div>
            <div class="m-value">{dinfo.get('n_features',0)}</div>
            <div class="m-sub">Input dimensions</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Model Comparison Chart ────────────────────────────────────────────────
    st.markdown('<div class="section-title">Model Comparison</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#222222; font-size:0.95rem; margin-top:-0.5rem; margin-bottom:1.5rem; font-style:italic;">Interactive benchmark across key classification metrics.</p>', unsafe_allow_html=True)

    dt_def = _get_default_class(dtm.get("class_metrics", {}))
    lr_def = _get_default_class(lrm.get("class_metrics", {}))

    metrics = ["Accuracy", "ROC-AUC", "Precision (Default)", "Recall (Default)", "F1-Score (Default)"]
    dt_vals = [
        dtm.get("test_accuracy",0),
        dtm.get("roc_auc",0),
        dt_def.get("precision",0),
        dt_def.get("recall",0),
        dt_def.get("f1_score", dt_def.get("f1-score",0))
    ]
    lr_vals = [
        lrm.get("test_accuracy",0),
        lrm.get("roc_auc",0),
        lr_def.get("precision",0),
        lr_def.get("recall",0),
        lr_def.get("f1_score", lr_def.get("f1-score",0))
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=metrics, y=dt_vals,
        name='Decision Tree',
        marker_color='#000000',
        text=[f"{v:.3f}" for v in dt_vals],
        textposition='outside',
        textfont=dict(color='#000000', size=12),
    ))
    fig.add_trace(go.Bar(
        x=metrics, y=lr_vals,
        name='Logistic Regression',
        marker_color='#CCCCCC',
        text=[f"{v:.3f}" for v in lr_vals],
        textposition='outside',
        textfont=dict(color='#000000', size=12),
    ))

    fig.update_layout(
        barmode='group',
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
        font=dict(family="Inter, sans-serif", color="#000000", size=13),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.12,
            xanchor="center", x=0.5,
            font=dict(size=13, color="#000000"),
            bgcolor="#FFFFFF",
            bordercolor="#CCCCCC",
            borderwidth=1,
        ),
        margin=dict(l=10, r=10, t=80, b=10),
        yaxis=dict(gridcolor='#F5F5F5', range=[0, 1.15], linecolor='#CCCCCC'),
        xaxis=dict(gridcolor='#FFFFFF', linecolor='#CCCCCC'),
        bargap=0.25,
        bargroupgap=0.08,
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # ── Feature List ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Input Features</div>', unsafe_allow_html=True)
    tags = "".join(f'<span class="feature-tag">{c}</span>' for c in feature_cols)
    st.markdown(f'<div class="feature-tags fade-in">{tags}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Performance":
    st.markdown("""
    <div class="page-header">
        <h1>Model Performance</h1>
        <p>Evaluation metrics, confusion matrices, and feature diagnostics</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["  Decision Tree  ", "  Logistic Regression  "])

    def render_model_tab(metrics, model_name):
        if not metrics:
            st.warning("No metrics available.")
            return

        acc   = metrics.get("test_accuracy", 0)
        roc   = metrics.get("roc_auc", 0)
        wf1   = metrics.get("weighted_avg", {}).get("f1_score", metrics.get("weighted_avg", {}).get("f1-score", 0))
        def_m = _get_default_class(metrics.get("class_metrics", {}))
        prec  = def_m.get("precision", 0)
        rec   = def_m.get("recall", 0)

        # ── 4 Perf Cards ──────────────────────────────────────────────────────
        st.markdown(f"""
        <div class="perf-grid">
            <div class="perf-card">
                <div class="pc-label">Accuracy</div>
                <div class="pc-value">{acc*100:.2f}%</div>
            </div>
            <div class="perf-card">
                <div class="pc-label">ROC-AUC</div>
                <div class="pc-value">{roc:.4f}</div>
            </div>
            <div class="perf-card">
                <div class="pc-label">F1 Weighted</div>
                <div class="pc-value">{wf1:.4f}</div>
            </div>
            <div class="perf-card">
                <div class="pc-label">Default Precision</div>
                <div class="pc-value">{prec:.4f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Confusion Matrix + Classification Report ─────────────────────────
        cm_col, rep_col = st.columns([1.1, 1], gap="large")

        with cm_col:
            st.markdown('<div class="section-title" style="margin-top:0">Confusion Matrix</div>', unsafe_allow_html=True)
            cm = metrics.get("confusion_matrix")
            if cm:
                fig, ax = plt.subplots(figsize=(5, 4))
                cm_arr = np.array(cm)
                total = cm_arr.sum()

                # Normalize for color mapping (0 to 1)
                cm_norm = cm_arr / cm_arr.max()

                # Auto-contrast: white text on dark cells, black text on light
                annot_colors = []
                for row_idx in range(cm_arr.shape[0]):
                    row_colors = []
                    for col_idx in range(cm_arr.shape[1]):
                        intensity = cm_norm[row_idx, col_idx]
                        row_colors.append("#FFFFFF" if intensity > 0.55 else "#000000")
                    annot_colors.append(row_colors)

                labels = np.array([[f"{v}\n({v/total*100:.1f}%)" for v in row] for row in cm_arr])

                # Light grey scale: #F5F5F5 (light) to #555555 (dark)
                from matplotlib.colors import LinearSegmentedColormap
                cmap = LinearSegmentedColormap.from_list('bw', ['#F5F5F5', '#AAAAAA', '#555555'])

                sns.heatmap(
                    cm_arr, annot=False, fmt="", cmap=cmap,
                    xticklabels=["Good Loan", "Default"],
                    yticklabels=["Good Loan", "Default"],
                    linewidths=3, linecolor="#FFFFFF",
                    cbar=False, ax=ax,
                )

                # Manually place text with auto-contrast colors
                for row_idx in range(cm_arr.shape[0]):
                    for col_idx in range(cm_arr.shape[1]):
                        ax.text(col_idx + 0.5, row_idx + 0.5,
                                labels[row_idx, col_idx],
                                ha='center', va='center',
                                fontsize=13, fontweight='bold',
                                color=annot_colors[row_idx][col_idx])

                ax.set_xlabel("Predicted", labelpad=12, fontsize=12, fontweight='bold')
                ax.set_ylabel("Actual", labelpad=12, fontsize=12, fontweight='bold')
                ax.set_title(f"{model_name} — Confusion Matrix",
                            fontsize=13, pad=14, color="#000000",
                            fontweight='bold', fontfamily='serif')
                ax.tick_params(colors='#000000', labelsize=11)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        with rep_col:
            st.markdown('<div class="section-title" style="margin-top:0">Classification Report</div>', unsafe_allow_html=True)
            cm_d = metrics.get("class_metrics", {})
            good = _get_good_class(cm_d)
            deflt = _get_default_class(cm_d)
            macro = metrics.get("macro_avg", {})
            weight = metrics.get("weighted_avg", {})

            def f1v(d): return d.get("f1_score", d.get("f1-score", 0))

            def pfmt(v): return fmt(v, False)

            table = f"""
            <table class="report-table">
                <thead><tr>
                    <th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th>
                </tr></thead>
                <tbody>
                <tr>
                    <td>Good Loan</td>
                    <td>{pfmt(good.get('precision'))}</td>
                    <td>{pfmt(good.get('recall'))}</td>
                    <td>{pfmt(f1v(good))}</td>
                    <td>{good.get('support','—')}</td>
                </tr>
                <tr>
                    <td>Default</td>
                    <td>{pfmt(deflt.get('precision'))}</td>
                    <td>{pfmt(deflt.get('recall'))}</td>
                    <td>{pfmt(f1v(deflt))}</td>
                    <td>{deflt.get('support','—')}</td>
                </tr>
                <tr class="divider">
                    <td>Macro Avg</td>
                    <td>{pfmt(macro.get('precision'))}</td>
                    <td>{pfmt(macro.get('recall'))}</td>
                    <td>{pfmt(f1v(macro))}</td>
                    <td>—</td>
                </tr>
                <tr class="divider">
                    <td>Weighted Avg</td>
                    <td>{pfmt(weight.get('precision'))}</td>
                    <td>{pfmt(weight.get('recall'))}</td>
                    <td>{pfmt(f1v(weight))}</td>
                    <td>—</td>
                </tr>
                </tbody>
            </table>
            """
            st.markdown(table, unsafe_allow_html=True)

            # Recall callout
            st.markdown(f"""
            <div class="callout-box">
                <span style="font-weight:700;">Default Class Recall</span><br>
                <span>Model catches
                <span style="font-family:'IBM Plex Mono',monospace;font-weight:700;">
                {rec*100:.1f}%</span> of actual defaults.
                Low recall means undetected risk.</span>
            </div>
            """, unsafe_allow_html=True)

        # ── Feature Importance ────────────────────────────────────────────────
        fi = metrics.get("feature_importance")
        if fi:
            st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)
            sorted_fi = sorted(fi.items(), key=lambda x: x[1])
            names  = [k for k, _ in sorted_fi]
            scores = [v for _, v in sorted_fi]
            max_score = max(scores) if scores else 1

            # Monochrome grey scale for bars
            grey_shades = [f'#{int(60 + (i/len(names))*120):02x}{int(60 + (i/len(names))*120):02x}{int(60 + (i/len(names))*120):02x}' for i in range(len(names))]

            fig, ax = plt.subplots(figsize=(9, max(3.5, len(names)*0.42)))
            bars = ax.barh(names, scores, color=grey_shades, height=0.6, edgecolor="#CCCCCC", linewidth=0.5)
            for bar, score in zip(bars, scores):
                ax.text(score + max_score*0.008, bar.get_y() + bar.get_height()/2,
                        f"{score:.4f}", va="center", fontsize=9,
                        color="#000000", fontfamily="monospace", fontweight="bold")
            ax.set_xlabel("Importance Score", fontsize=11, fontweight="bold")
            ax.set_title(f"{model_name} — Feature Importance",
                        fontsize=13, pad=14, color="#000000",
                        fontweight="bold", fontfamily="serif")
            ax.set_xlim(0, max_score * 1.18)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_color("#CCCCCC")
            ax.spines["left"].set_color("#CCCCCC")
            ax.grid(axis="x", color="#F5F5F5")
            ax.tick_params(colors="#000000")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with tab1: render_model_tab(dtm, "Decision Tree")
    with tab2: render_model_tab(lrm, "Logistic Regression")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Predict":
    st.markdown("""
    <div class="page-header">
        <h1>Risk Prediction</h1>
        <p>Enter applicant details to receive an instant credit risk assessment</p>
    </div>
    """, unsafe_allow_html=True)

    form_col, result_col = st.columns([1, 1], gap="large")

    with form_col:
        with st.form("predict_form"):

            st.markdown('<div class="form-section-label">Select Model</div>', unsafe_allow_html=True)
            selected_model_name = st.radio(
                "Model",
                ["Decision Tree", "Logistic Regression"],
                horizontal=True,
                label_visibility="collapsed"
            )

            st.markdown('<div class="form-section-label">Applicant Profile</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1: person_age = st.slider("Age", 18, 100, 30)
            with c2: person_emp_length = st.slider("Employment Length (yrs)", 0.0, 60.0, 5.0, 0.5)
            person_income = st.number_input("Annual Income ($)", min_value=0, step=500, value=50000)
            person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])

            st.markdown('<div class="form-section-label">Loan Details</div>', unsafe_allow_html=True)
            c3, c4 = st.columns(2)
            with c3:
                loan_intent = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
                loan_amnt   = st.number_input("Loan Amount ($)", min_value=500, step=500, value=10000)
            with c4:
                loan_int_rate = st.slider("Interest Rate (%)", 5.0, 25.0, 11.0, 0.01)

            st.markdown('<div class="form-section-label">Credit History</div>', unsafe_allow_html=True)
            c5, c6 = st.columns(2)
            with c5: cb_default = st.selectbox("Prior Default on File", ["N", "Y"])
            with c6: cred_hist  = st.slider("Credit History Length (yrs)", 2, 30, 5)

            submitted = st.form_submit_button("Run Prediction", use_container_width=True)

    with result_col:
        if not submitted:
            st.markdown("""
            <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                        height:500px;gap:1.2rem;border:1px solid #000000;border-radius:4px;background:#FFFFFF;">
                <div style="font-family:'Playfair Display',Georgia,serif;font-size:1.6rem;
                            color:#000000;font-weight:700;">
                    Awaiting Input
                </div>
                <div style="color:#222222;font-size:0.92rem;text-align:center;font-style:italic;line-height:1.8;">
                    Complete the applicant form<br>and submit to generate a prediction.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            with st.spinner("Analyzing risk profile..."):
                active_model     = model        if selected_model_name == "Decision Tree" else lr_model
                active_threshold = dt_threshold if selected_model_name == "Decision Tree" else lr_threshold

                loan_percent_income = round(loan_amnt / person_income, 4) if person_income > 0 else 0.0

                risk_score = (loan_int_rate * 2) + (loan_percent_income * 100) - cred_hist
                if risk_score <= 25:   derived_grade = "A"
                elif risk_score <= 35: derived_grade = "B"
                elif risk_score <= 45: derived_grade = "C"
                elif risk_score <= 55: derived_grade = "D"
                elif risk_score <= 65: derived_grade = "E"
                elif risk_score <= 75: derived_grade = "F"
                else:                  derived_grade = "G"

                grade_map = {"A":0,"B":1,"C":2,"D":3,"E":4,"F":5,"G":6}
                raw = {
                    "person_age":               person_age,
                    "person_income($)":          person_income,
                    "person_home_ownership":     person_home_ownership,
                    "person_emp_length":         person_emp_length,
                    "loan_intent":               loan_intent,
                    "loan_grade":                grade_map[derived_grade],
                    "loan_amnt($)":              loan_amnt,
                    "loan_int_rate":             loan_int_rate,
                    "loan_percent_income":       loan_percent_income,
                    "cb_person_default_on_file": cb_default,
                    "cb_person_cred_hist_length":cred_hist,
                    "person_income":             person_income,
                    "loan_amnt":                 loan_amnt,
                }

                enc = raw.copy()
                for col in feature_cols:
                    if col in encoders:
                        le  = encoders[col]
                        val = enc.get(col)
                        enc[col] = int(le.transform([val])[0]) if val in le.classes_ else 0

                try:
                    X    = pd.DataFrame([[enc.get(c, 0) for c in feature_cols]], columns=feature_cols)
                    X_sc = scaler.transform(X)

                    proba        = active_model.predict_proba(X_sc)[0]
                    default_prob = float(proba[1])
                    pred         = 1 if default_prob >= active_threshold else 0
                    conf         = max(default_prob, 1 - default_prob) * 100

                    if default_prob < 0.30:   risk, risk_cls = "LOW RISK",    "risk-low"
                    elif default_prob < 0.60: risk, risk_cls = "MEDIUM RISK", "risk-med"
                    else:                     risk, risk_cls = "HIGH RISK",   "risk-high"

                    # ── Result Card ────────────────────────────────────────────
                    if pred == 0:
                        st.markdown(f"""
                        <div class="result-card good fade-in">
                            <span class="result-icon">APPROVED</span>
                            <div class="result-title">Good Loan</div>
                            <div class="result-sub">Low probability of default &middot; via {selected_model_name}</div>
                            <span class="risk-badge {risk_cls}">{risk}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-card bad fade-in">
                            <span class="result-icon">FLAGGED</span>
                            <div class="result-title">High Default Risk</div>
                            <div class="result-sub">Applicant likely to default &middot; via {selected_model_name}</div>
                            <span class="risk-badge {risk_cls}">{risk}</span>
                        </div>
                        """, unsafe_allow_html=True)

                    # ── Probability Gauge ──────────────────────────────────────
                    pct = int(default_prob * 100)
                    st.markdown(f"""
                    <div style="margin: 0.5rem 0 1.5rem;">
                        <div class="prob-label">
                            <span>Default Probability</span>
                            <span>{pct}%</span>
                        </div>
                        <div class="prob-track">
                            <div class="prob-fill" style="width:{pct}%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Summary Rows ──────────────────────────────────────────
                    st.markdown(f"""
                    <div style="margin-bottom:1.5rem;">
                        <div class="summary-row">
                            <span class="sr-label">Model Used</span>
                            <span class="sr-value">{selected_model_name}</span>
                        </div>
                        <div class="summary-row">
                            <span class="sr-label">Decision Threshold</span>
                            <span class="sr-value">{active_threshold}</span>
                        </div>
                        <div class="summary-row">
                            <span class="sr-label">Estimated Loan Grade</span>
                            <span class="sr-value">Grade {derived_grade}</span>
                        </div>
                        <div class="summary-row">
                            <span class="sr-label">Loan % of Income</span>
                            <span class="sr-value">{loan_percent_income*100:.1f}%</span>
                        </div>
                        <div class="summary-row">
                            <span class="sr-label">Predicted Class</span>
                            <span class="sr-value">{"Good Loan" if pred == 0 else "Default"}</span>
                        </div>
                        <div class="summary-row">
                            <span class="sr-label">Confidence</span>
                            <span class="sr-value">{conf:.2f}%</span>
                        </div>
                        <div class="summary-row">
                            <span class="sr-label">Default Probability</span>
                            <span class="sr-value">{default_prob*100:.2f}%</span>
                        </div>
                        <div class="summary-row">
                            <span class="sr-label">Risk Level</span>
                            <span class="risk-badge {risk_cls}" style="font-size:0.62rem;">{risk}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Feature Insights ──────────────────────────────────────
                    if selected_model_name == "Decision Tree":
                        fi = dtm.get("feature_importance", {})
                        if fi:
                            st.markdown('<div class="section-title" style="margin-top:0;font-size:0.78rem;">Top Features (Importance)</div>', unsafe_allow_html=True)
                            top3 = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:3]
                            max_fi = top3[0][1] if top3 else 1
                            rows_html = ""
                            for i, (fname, fscore) in enumerate(top3, 1):
                                bar_w = int(fscore / max_fi * 100)
                                rows_html += f"""
                                <div class="fi-row">
                                    <span class="fi-rank">#{i}</span>
                                    <span class="fi-name">{fname}</span>
                                    <div class="fi-bar-wrap"><div class="fi-bar" style="width:{bar_w}%;"></div></div>
                                    <span class="fi-score">{fscore:.4f}</span>
                                </div>"""
                            st.markdown(rows_html, unsafe_allow_html=True)
                    else:
                        coef = lrm.get("feature_coefficients", {})
                        if coef:
                            st.markdown('<div class="section-title" style="margin-top:0;font-size:0.78rem;">Top Risk Drivers (Coefficients)</div>', unsafe_allow_html=True)
                            top3 = sorted(coef.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                            max_abs = max(abs(v) for _, v in top3) if top3 else 1
                            rows_html = ""
                            for i, (fname, fcoef) in enumerate(top3, 1):
                                bar_w = int(abs(fcoef) / max_abs * 100)
                                sign  = "Default" if fcoef > 0 else "Good Loan"
                                rows_html += f"""
                                <div class="fi-row">
                                    <span class="fi-rank">#{i}</span>
                                    <span class="fi-name">{fname}</span>
                                    <div class="fi-bar-wrap"><div class="fi-bar" style="width:{bar_w}%;"></div></div>
                                    <span class="fi-score" style="width:5.5rem;">{fcoef:+.4f} {sign}</span>
                                </div>"""
                            st.markdown(rows_html, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Prediction failed: {e}")