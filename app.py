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
    page_title="CreditIQ — Risk Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="auto"
)

# ─── GLOBAL STYLES ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

/* Reset & Base */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
}

/* App Background */
.stApp {
    background: #080c14;
    color: #e2e8f0;
}

/* Hide Streamlit branding but keep sidebar toggle */
#MainMenu, footer { visibility: hidden; }
[data-testid="stHeader"] {
    background: transparent !important;
}
[data-testid="stToolbar"] {
    visibility: hidden !important;
}

/* ─── Sidebar ─── */
[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1e2d40 !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 2rem;
}

.sidebar-brand {
    text-align: center;
    padding: 1.5rem 1rem 1rem;
    border-bottom: 1px solid #1e2d40;
    margin-bottom: 1.5rem;
}
.sidebar-brand .logo {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #38bdf8;
    letter-spacing: -1px;
    display: block;
}
.sidebar-brand .tagline {
    font-size: 0.72rem;
    color: #4a5568;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 0.2rem;
}
.model-badge {
    background: linear-gradient(135deg, #0f2027, #203a43);
    border: 1px solid #1e3a4a;
    border-left: 3px solid #38bdf8;
    border-radius: 6px;
    padding: 0.85rem 1rem;
    margin: 1.2rem 0;
    font-size: 0.78rem;
}
.model-badge .badge-label {
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-size: 0.68rem;
    margin-bottom: 0.3rem;
}
.model-badge .badge-value {
    color: #38bdf8;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    font-size: 0.82rem;
}
.model-badge .badge-sub {
    color: #10b981;
    font-size: 0.7rem;
    margin-top: 0.25rem;
}

/* ─── Page Header ─── */
.page-header {
    padding: 1.5rem 0 1rem;
    border-bottom: 1px solid #1e2d40;
    margin-bottom: 2rem;
}
.page-header h1 {
    font-size: 2rem;
    font-weight: 700;
    color: #f1f5f9;
    margin: 0;
    letter-spacing: -0.5px;
}
.page-header p {
    color: #64748b;
    margin: 0.3rem 0 0;
    font-size: 0.9rem;
}

/* ─── Metric Cards ─── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 1rem;
    margin-bottom: 2rem;
}
.metric-card {
    background: #0d1117;
    border: 1px solid #1e2d40;
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
}
.metric-card:hover { border-color: #2d4a6a; }
.metric-card .m-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #4a5568;
    margin-bottom: 0.5rem;
    font-weight: 600;
}
.metric-card .m-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1;
}
.metric-card .m-sub {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 0.4rem;
}
.metric-card .m-icon {
    position: absolute;
    right: 1.2rem; top: 1.2rem;
    font-size: 1.4rem;
    opacity: 0.15;
}

/* ─── Comparison Table ─── */
.comp-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.875rem;
    margin-top: 1rem;
}
.comp-table th {
    background: #0d1117;
    color: #64748b;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 0.75rem 1rem;
    text-align: left;
    border-bottom: 2px solid #1e2d40;
    font-weight: 600;
}
.comp-table th:last-child { text-align: right; }
.comp-table th:nth-child(2) { text-align: right; }
.comp-table td {
    padding: 0.85rem 1rem;
    border-bottom: 1px solid #1a2535;
    color: #94a3b8;
    vertical-align: middle;
}
.comp-table td:nth-child(2), .comp-table td:last-child { text-align: right; }
.comp-table tr:hover td { background: rgba(56, 189, 248, 0.03); }
.comp-table .winner {
    color: #10b981;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 0.95rem;
}
.comp-table .loser {
    color: #475569;
    font-family: 'JetBrains Mono', monospace;
}
.comp-table .metric-name { color: #cbd5e1; font-weight: 500; }
.comp-badge {
    display: inline-block;
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.25);
    color: #10b981;
    font-size: 0.6rem;
    padding: 0.1rem 0.4rem;
    border-radius: 3px;
    margin-left: 0.5rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    vertical-align: middle;
    font-weight: 700;
}

/* ─── Section Title ─── */
.section-title {
    font-size: 1rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 2rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1e2d40;
}

/* ─── Feature Tags ─── */
.feature-tags { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.5rem; }
.feature-tag {
    background: #0f1923;
    border: 1px solid #1e2d40;
    color: #38bdf8;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    padding: 0.35rem 0.75rem;
    border-radius: 4px;
}

/* ─── Tab Overrides ─── */
[data-testid="stTab"] button {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500;
    color: #64748b !important;
    font-size: 0.88rem;
}
[data-testid="stTab"] button[aria-selected="true"] {
    color: #38bdf8 !important;
    border-bottom-color: #38bdf8 !important;
}

/* ─── Perf Cards (4-up) ─── */
.perf-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 0.85rem;
    margin-bottom: 1.5rem;
}
.perf-card {
    background: #0d1117;
    border: 1px solid #1e2d40;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    text-align: center;
}
.perf-card .pc-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #4a5568;
    margin-bottom: 0.4rem;
    font-weight: 600;
}
.perf-card .pc-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: #38bdf8;
}

/* ─── Report Table ─── */
.report-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
}
.report-table th {
    background: #111827;
    color: #4a5568;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 0.6rem 0.8rem;
    text-align: right;
    border-bottom: 1px solid #1e2d40;
}
.report-table th:first-child { text-align: left; }
.report-table td {
    padding: 0.65rem 0.8rem;
    border-bottom: 1px solid #111827;
    color: #94a3b8;
    text-align: right;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
}
.report-table td:first-child {
    text-align: left;
    font-family: 'Space Grotesk', sans-serif;
    color: #cbd5e1;
    font-weight: 500;
    font-size: 0.82rem;
}
.report-table tr.divider td { border-top: 1px solid #1e2d40; color: #64748b; }

/* ─── Predict Form ─── */
.form-section-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: #4a5568;
    font-weight: 700;
    margin: 1.2rem 0 0.6rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #1a2535;
}

/* ─── Result Panel ─── */
.result-card {
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1.25rem;
    position: relative;
    overflow: hidden;
}
.result-card.good {
    background: linear-gradient(135deg, #052e16, #064e3b);
    border: 1px solid #10b981;
    box-shadow: 0 0 30px rgba(16, 185, 129, 0.15);
}
.result-card.bad {
    background: linear-gradient(135deg, #1c0a0a, #2d1515);
    border: 1px solid #ef4444;
    box-shadow: 0 0 30px rgba(239, 68, 68, 0.15);
}
.result-card .result-icon { font-size: 3rem; margin-bottom: 0.5rem; display: block; }
.result-card .result-title {
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
}
.result-card.good .result-title { color: #34d399; }
.result-card.bad .result-title { color: #f87171; }
.result-card .result-sub { color: #64748b; font-size: 0.85rem; }

.risk-badge {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.75rem;
}
.risk-low { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
.risk-med { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
.risk-high { background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }

.prob-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.75rem;
    color: #4a5568;
    margin-bottom: 0.3rem;
    font-family: 'JetBrains Mono', monospace;
}
.prob-track {
    background: #1a2535;
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
    margin-bottom: 1.25rem;
}
.prob-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.6s ease;
}

.summary-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.6rem 0;
    border-bottom: 1px solid #1a2535;
    font-size: 0.83rem;
}
.summary-row .sr-label { color: #4a5568; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; }
.summary-row .sr-value { color: #cbd5e1; font-family: 'JetBrains Mono', monospace; font-weight: 600; }

.fi-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 0.5rem 0;
    border-bottom: 1px solid #111827;
}
.fi-rank {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #38bdf8;
    font-weight: 700;
    width: 1.5rem;
    flex-shrink: 0;
}
.fi-name { color: #94a3b8; font-size: 0.82rem; flex: 1; }
.fi-bar-wrap { width: 80px; background: #1a2535; border-radius: 3px; height: 5px; }
.fi-bar { height: 100%; background: #38bdf8; border-radius: 3px; }
.fi-score { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #475569; width: 3.5rem; text-align: right; }

/* ─── Stagger animation ─── */
.fade-in { animation: fadeUp 0.4s ease both; }
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* Override Streamlit slider */
[data-testid="stSlider"] { padding-top: 0.2rem; }

/* Subheader color */
h2, h3 { color: #e2e8f0 !important; }
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
    <div style="display:flex;align-items:center;justify-content:center;height:60vh;flex-direction:column;gap:1rem;">
        <div style="font-size:4rem;">⚠️</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:1.1rem;color:#ef4444;">dt_model.pkl not found</div>
        <div style="color:#4a5568;font-size:0.85rem;">Place the pickle file in the same directory as app.py</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

model          = pkg["model"]                 # Decision Tree (primary)
lr_model       = pkg["lr_model"]             # Logistic Regression
scaler         = pkg["scaler"]
encoders       = pkg["encoders"]
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

    page = st.radio("", ["🏠 Overview", "📊 Performance", "🔮 Predict"], label_visibility="collapsed")

    st.markdown(f"""
    <div class="model-badge">
        <div class="badge-label">Primary Model</div>
        <div class="badge-value">Decision Tree Classifier</div>
        <div class="badge-sub">✓ Threshold: {dt_threshold} · Acc: {dtm.get('test_accuracy',0)*100:.2f}%</div>
    </div>
    <div class="model-badge" style="border-left-color:#818cf8; margin-top:0.6rem;">
        <div class="badge-label">Secondary Model</div>
        <div class="badge-value" style="color:#818cf8;">Logistic Regression</div>
        <div class="badge-sub" style="color:#a78bfa;">✓ Threshold: {lr_threshold} · Acc: {lrm.get('test_accuracy',0)*100:.2f}%</div>
    </div>
    <div class="model-badge" style="border-left-color:#10b981; margin-top:0.6rem;">
        <div class="badge-label">ROC-AUC</div>
        <div class="badge-value" style="color:#10b981;">DT: {dtm.get('roc_auc',0):.4f} · LR: {lrm.get('roc_auc',0):.4f}</div>
        <div class="badge-sub" style="color:#38bdf8;">Area under curve — both models</div>
    </div>
    <div class="model-badge" style="border-left-color:#818cf8; margin-top:0.6rem;">
        <div class="badge-label">Dataset</div>
        <div class="badge-value" style="color:#818cf8; font-size:0.85rem;">{dinfo.get('total_samples',0):,} samples</div>
        <div class="badge-sub" style="color:#64748b;">{dinfo.get('n_features',0)} features · Binary classification</div>
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

def _matplotlib_dark():
    plt.rcParams.update({
        "figure.facecolor": "#0d1117",
        "axes.facecolor":   "#0d1117",
        "axes.edgecolor":   "#1e2d40",
        "axes.labelcolor":  "#64748b",
        "xtick.color":      "#4a5568",
        "ytick.color":      "#4a5568",
        "text.color":       "#94a3b8",
        "grid.color":       "#1a2535",
        "grid.linestyle":   "--",
        "grid.alpha":       0.6,
    })

_matplotlib_dark()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("""
    <div class="page-header">
        <h1>Credit Risk Intelligence</h1>
        <p>Machine learning pipeline to classify loan applicants as <strong>Good Loan</strong> or <strong>Default</strong></p>
    </div>
    """, unsafe_allow_html=True)

    # ── Metric Cards ──────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card fade-in">
            <span class="m-icon">📦</span>
            <div class="m-label">Total Samples</div>
            <div class="m-value">{dinfo.get('total_samples',0):,}</div>
            <div class="m-sub">Full dataset size</div>
        </div>
        <div class="metric-card fade-in" style="animation-delay:0.05s">
            <span class="m-icon">🎯</span>
            <div class="m-label">Training Samples</div>
            <div class="m-value">{dinfo.get('train_samples',0):,}</div>
            <div class="m-sub">80% split</div>
        </div>
        <div class="metric-card fade-in" style="animation-delay:0.1s">
            <span class="m-icon">🧪</span>
            <div class="m-label">Test Samples</div>
            <div class="m-value">{dinfo.get('test_samples',0):,}</div>
            <div class="m-sub">20% split</div>
        </div>
        <div class="metric-card fade-in" style="animation-delay:0.15s">
            <span class="m-icon">⚙️</span>
            <div class="m-label">Features</div>
            <div class="m-value">{dinfo.get('n_features',0)}</div>
            <div class="m-sub">Input dimensions</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Model Comparison ──────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Model Comparison</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#64748b; font-size:0.85rem; margin-top:-0.5rem; margin-bottom:1rem;">Interactive benchmark across key classification metrics.</p>', unsafe_allow_html=True)

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
        marker_color='#38bdf8',
        text=[f"{v:.3f}" for v in dt_vals],
        textposition='auto',
    ))
    fig.add_trace(go.Bar(
        x=metrics, y=lr_vals,
        name='Logistic Regression',
        marker_color='#475569',
        text=[f"{v:.3f}" for v in lr_vals],
        textposition='auto',
    ))

    fig.update_layout(
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Space Grotesk", color="#94a3b8"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=30, b=0),
        yaxis=dict(gridcolor='#1e2d40', range=[0, 1.1]),
        xaxis=dict(gridcolor='rgba(0,0,0,0)')
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # ── Feature List ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Input Features</div>', unsafe_allow_html=True)
    tags = "".join(f'<span class="feature-tag">{c}</span>' for c in feature_cols)
    st.markdown(f'<div class="feature-tags fade-in">{tags}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Performance":
    st.markdown("""
    <div class="page-header">
        <h1>Model Performance</h1>
        <p>Evaluation metrics, confusion matrices, and feature diagnostics</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["  🌲 Decision Tree  ", "  📈 Logistic Regression  "])

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

        # ── Confusion Matrix + Classification Report ───────────────────────────
        cm_col, rep_col = st.columns([1.1, 1], gap="large")

        with cm_col:
            st.markdown('<div class="section-title" style="margin-top:0">Confusion Matrix</div>', unsafe_allow_html=True)
            cm = metrics.get("confusion_matrix")
            if cm:
                fig, ax = plt.subplots(figsize=(5, 4))
                cm_arr = np.array(cm)
                total = cm_arr.sum()
                labels = np.array([[f"{v}\n({v/total*100:.1f}%)" for v in row] for row in cm_arr])
                cmap = sns.color_palette("Blues", as_cmap=True)
                sns.heatmap(
                    cm_arr, annot=labels, fmt="", cmap=cmap,
                    xticklabels=["Good Loan", "Default"],
                    yticklabels=["Good Loan", "Default"],
                    linewidths=2, linecolor="#0d1117",
                    cbar=False, ax=ax,
                    annot_kws={"size": 12, "weight": "bold", "color": "white"}
                )
                ax.set_xlabel("Predicted", labelpad=10, fontsize=11)
                ax.set_ylabel("Actual", labelpad=10, fontsize=11)
                ax.set_title(f"{model_name} — Confusion Matrix", fontsize=12, pad=12, color="#94a3b8")
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

            # Recall highlight callout
            st.markdown(f"""
            <div style="background:#0f1923;border:1px solid #1e2d40;border-left:3px solid #f59e0b;
                        border-radius:6px;padding:0.75rem 1rem;margin-top:1rem;font-size:0.8rem;">
                <span style="color:#f59e0b;font-weight:600;">⚡ Default Class Recall</span><br>
                <span style="color:#64748b;">Model catches 
                <span style="color:#f59e0b;font-family:'JetBrains Mono',monospace;font-weight:700;">
                {rec*100:.1f}%</span> of actual defaults. 
                Low recall = undetected risk.</span>
            </div>
            """, unsafe_allow_html=True)

        # ── Feature Importance (DT only) ──────────────────────────────────────
        fi = metrics.get("feature_importance")
        if fi:
            st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)
            sorted_fi = sorted(fi.items(), key=lambda x: x[1])
            names  = [k for k, _ in sorted_fi]
            scores = [v for _, v in sorted_fi]
            max_score = max(scores) if scores else 1

            colors = plt.cm.plasma(np.linspace(0.2, 0.85, len(names)))
            fig, ax = plt.subplots(figsize=(9, max(3.5, len(names)*0.42)))
            bars = ax.barh(names, scores, color=colors, height=0.6, edgecolor="none")
            for bar, score in zip(bars, scores):
                ax.text(score + max_score*0.005, bar.get_y() + bar.get_height()/2,
                        f"{score:.4f}", va="center", fontsize=8.5,
                        color="#94a3b8", fontfamily="monospace")
            ax.set_xlabel("Importance Score", fontsize=10)
            ax.set_title(f"{model_name} — Feature Importance", fontsize=12, pad=12, color="#94a3b8")
            ax.set_xlim(0, max_score * 1.18)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="x")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with tab1: render_model_tab(dtm, "Decision Tree")
    with tab2: render_model_tab(lrm, "Logistic Regression")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict":
    st.markdown("""
    <div class="page-header">
        <h1>Risk Prediction</h1>
        <p>Enter applicant details to get an instant credit risk assessment</p>
    </div>
    """, unsafe_allow_html=True)

    form_col, result_col = st.columns([1, 1], gap="large")

    with form_col:
        with st.form("predict_form"):

            # ── Model Selector ────────────────────────────────────────────────
            st.markdown('<div class="form-section-label">🤖 Select Model</div>', unsafe_allow_html=True)
            selected_model_name = st.radio(
                "Model",
                ["Decision Tree", "Logistic Regression"],
                horizontal=True,
                label_visibility="collapsed"
            )

            st.markdown('<div class="form-section-label">👤 Applicant Profile</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1: person_age = st.slider("Age", 18, 100, 30)
            with c2: person_emp_length = st.slider("Employment Length (yrs)", 0.0, 60.0, 5.0, 0.5)
            person_income = st.number_input("Annual Income ($)", min_value=0, step=500, value=50000)
            person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])

            st.markdown('<div class="form-section-label">💳 Loan Details</div>', unsafe_allow_html=True)
            c3, c4 = st.columns(2)
            with c3:
                loan_intent = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
                loan_amnt   = st.number_input("Loan Amount ($)", min_value=500, step=500, value=10000)
            with c4:
                loan_int_rate = st.slider("Interest Rate (%)", 5.0, 25.0, 11.0, 0.01)

            st.markdown('<div class="form-section-label">📋 Credit History</div>', unsafe_allow_html=True)
            c5, c6 = st.columns(2)
            with c5: cb_default = st.selectbox("Prior Default on File", ["N", "Y"])
            with c6: cred_hist  = st.slider("Credit History Length (yrs)", 2, 30, 5)

            submitted = st.form_submit_button("🔮  Run Prediction", use_container_width=True)

    with result_col:
        if not submitted:
            st.markdown("""
            <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                        height:500px;gap:1rem;border:1px dashed #1e2d40;border-radius:12px;">
                <div style="font-size:3rem;opacity:0.3;">📊</div>
                <div style="color:#2d3748;font-size:0.85rem;text-align:center;">
                    Fill in applicant details<br>and click <strong>Run Prediction</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            with st.spinner("Analyzing risk profile…"):
                # ── Route to the model the user selected ──────────────────────
                active_model     = model        if selected_model_name == "Decision Tree" else lr_model
                active_threshold = dt_threshold if selected_model_name == "Decision Tree" else lr_threshold

                # Auto-calculate loan_percent_income (loan amount ÷ annual income)
                loan_percent_income = round(loan_amnt / person_income, 4) if person_income > 0 else 0.0

                # Derive loan grade from a risk proxy score
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

                    # Use stored threshold — NOT the sklearn default 0.50
                    proba        = active_model.predict_proba(X_sc)[0]
                    default_prob = float(proba[1])
                    pred         = 1 if default_prob >= active_threshold else 0
                    conf         = max(default_prob, 1 - default_prob) * 100

                    if default_prob < 0.30:   risk, risk_cls = "LOW RISK",    "risk-low"
                    elif default_prob < 0.60: risk, risk_cls = "MEDIUM RISK", "risk-med"
                    else:                     risk, risk_cls = "HIGH RISK",   "risk-high"

                    bar_color = "#10b981" if pred == 0 else "#ef4444"

                    # ── Result Card ────────────────────────────────────────────
                    if pred == 0:
                        st.markdown(f"""
                        <div class="result-card good fade-in">
                            <span class="result-icon">✅</span>
                            <div class="result-title">Good Loan</div>
                            <div class="result-sub">Low probability of default · via {selected_model_name}</div>
                            <span class="risk-badge {risk_cls}">{risk}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-card bad fade-in">
                            <span class="result-icon">⚠️</span>
                            <div class="result-title">High Default Risk</div>
                            <div class="result-sub">Applicant likely to default · via {selected_model_name}</div>
                            <span class="risk-badge {risk_cls}">{risk}</span>
                        </div>
                        """, unsafe_allow_html=True)

                    # ── Probability Gauge ───────────────────────────────────────
                    pct = int(default_prob * 100)
                    st.markdown(f"""
                    <div style="margin: 0.5rem 0 1.25rem;">
                        <div class="prob-label">
                            <span>Default Probability</span>
                            <span style="color:{bar_color};font-weight:700;">{pct}%</span>
                        </div>
                        <div class="prob-track">
                            <div class="prob-fill" style="width:{pct}%;background:{bar_color};"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Summary Rows ────────────────────────────────────────────
                    st.markdown(f"""
                    <div style="margin-bottom:1.25rem;">
                        <div class="summary-row">
                            <span class="sr-label">Model Used</span>
                            <span class="sr-value" style="color:#818cf8;">{selected_model_name}</span>
                        </div>
                        <div class="summary-row">
                            <span class="sr-label">Decision Threshold</span>
                            <span class="sr-value">{active_threshold}</span>
                        </div>
                        <div class="summary-row">
                            <span class="sr-label">Estimated Loan Grade</span>
                            <span class="sr-value">Grade <span style="color:#38bdf8;">{derived_grade}</span></span>
                        </div>
                        <div class="summary-row">
                            <span class="sr-label">Loan % of Income</span>
                            <span class="sr-value" style="color:#f59e0b;">{loan_percent_income*100:.1f}% <span style="color:#4a5568;font-size:0.68rem;">(auto)</span></span>
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
                            <span class="sr-value" style="color:{bar_color};">{default_prob*100:.2f}%</span>
                        </div>
                        <div class="summary-row">
                            <span class="sr-label">Risk Level</span>
                            <span class="risk-badge {risk_cls}" style="font-size:0.65rem;">{risk}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Model-specific feature insights ─────────────────────────
                    if selected_model_name == "Decision Tree":
                        fi = dtm.get("feature_importance", {})
                        if fi:
                            st.markdown('<div class="section-title" style="margin-top:0;font-size:0.72rem;">Top Features (Importance)</div>', unsafe_allow_html=True)
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
                            st.markdown('<div class="section-title" style="margin-top:0;font-size:0.72rem;">Top Risk Drivers (Coefficients)</div>', unsafe_allow_html=True)
                            top3 = sorted(coef.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                            max_abs = max(abs(v) for _, v in top3) if top3 else 1
                            rows_html = ""
                            for i, (fname, fcoef) in enumerate(top3, 1):
                                bar_w = int(abs(fcoef) / max_abs * 100)
                                c_col = "#ef4444" if fcoef > 0 else "#10b981"
                                sign  = "↑ Default" if fcoef > 0 else "↓ Good Loan"
                                rows_html += f"""
                                <div class="fi-row">
                                    <span class="fi-rank">#{i}</span>
                                    <span class="fi-name">{fname}</span>
                                    <div class="fi-bar-wrap"><div class="fi-bar" style="width:{bar_w}%;background:{c_col};"></div></div>
                                    <span class="fi-score" style="color:{c_col};width:5.5rem;">{fcoef:+.4f} {sign}</span>
                                </div>"""
                            st.markdown(rows_html, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Prediction failed: {e}")
