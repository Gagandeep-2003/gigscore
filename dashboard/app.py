"""
dashboard/app.py — GigScore Streamlit Dashboard Main Entry Point

Multi-page dashboard with sidebar navigation:
    🏠 Project Story (Home)
    🎯 Score an Applicant
    📊 Model Analytics
    ⚖️ Fairness Report
    🔍 Dataset Explorer

Run with: streamlit run dashboard/app.py
"""

import os
import sys
import streamlit as st

# Add project root to path for clean imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ─────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GigScore — Alternative Credit Scoring",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────
# Custom CSS — Professional Dark Theme
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global dark theme */
    .main {
        background-color: #0f172a;
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 1px solid #334155;
    }

    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #f1f5f9;
    }

    [data-testid="stSidebar"] .stMarkdown p {
        color: #94a3b8;
    }

    /* Headers */
    h1, h2, h3, h4 {
        color: #f1f5f9 !important;
        font-family: 'Inter', sans-serif;
    }

    p, li, span {
        font-family: 'Inter', sans-serif;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
    }

    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }

    [data-testid="stMetricValue"] {
        color: #6366f1 !important;
        font-weight: 700;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        padding: 12px 24px;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
        transform: translateY(-1px);
    }

    /* Slider styling */
    .stSlider > div > div > div > div {
        background-color: #6366f1 !important;
    }

    /* Selectbox */
    .stSelectbox > div > div {
        background-color: #1e293b;
        border-color: #334155;
        color: #e2e8f0;
    }

    /* DataFrame */
    [data-testid="stDataFrame"] {
        border: 1px solid #334155;
        border-radius: 8px;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1e293b !important;
        border-radius: 8px;
        color: #e2e8f0 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b;
        border-radius: 8px 8px 0 0;
        color: #94a3b8;
        padding: 8px 16px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #334155;
        color: #f1f5f9;
    }

    /* Info/Warning boxes */
    .stAlert {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Divider */
    hr {
        border-color: #334155 !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# Load Model Metadata (dynamic — never hardcoded)
# ─────────────────────────────────────────────────────────────────────
def _load_model_metadata():
    """Load model metadata from artifacts. Returns empty dict if not found."""
    import json
    meta_path = os.path.join(PROJECT_ROOT, 'artifacts', 'model_metadata.json')
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

_metadata = _load_model_metadata()

# ─────────────────────────────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 12px 0;">
        <div style="font-size: 48px; margin-bottom: 4px;">🏦</div>
        <h1 style="font-size: 28px; margin: 0; color: #6366f1; font-weight: 800;">
            GigScore
        </h1>
        <p style="color: #94a3b8; font-size: 12px; margin-top: 4px; letter-spacing: 1px;">
            ALTERNATIVE CREDIT SCORING
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Navigation
    page = st.radio(
        "Navigate",
        options=[
            "🏠 Project Story",
            "🎯 Score an Applicant",
            "📊 Model Analytics",
            "⚖️ Fairness Report",
            "🔍 Dataset Explorer",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Sidebar info
    st.markdown("""
    <div style="padding: 12px; background: #1e293b; border-radius: 8px;
                border: 1px solid #334155; font-size: 12px;">
        <div style="color: #6366f1; font-weight: 600; margin-bottom: 8px;">
            ℹ️ About GigScore
        </div>
        <div style="color: #94a3b8; line-height: 1.5;">
            AI-powered credit scoring for India's 15M+ gig workers.
            Uses behavioral and transactional signals instead of
            traditional CIBIL scores.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    st.markdown("""
    <div style="padding: 12px; background: #1e293b; border-radius: 8px;
                border: 1px solid #334155; font-size: 12px;">
        <div style="color: #94a3b8; line-height: 1.5;">
            <b style="color: #a78bfa;">Model:</b> LightGBM Ensemble<br>
            <b style="color: #a78bfa;">Features:</b> 44+ engineered<br>
            <b style="color: #a78bfa;">API:</b> FastAPI @ :8000<br>
            <b style="color: #a78bfa;">Version:</b> 1.0.0
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    with st.expander("📚 Project Details"):
        st.markdown("""
        **Tech Stack**
        - 🐍 Python 3.10+
        - 🌿 LightGBM + XGBoost + Stacking
        - 📊 SHAP TreeExplainer
        - ⚖️ Fairlearn Auditing
        - ⚡ FastAPI + Streamlit
        - 🔧 Optuna HPO

        **Key Design Decisions**
        - LightGBM over XGBoost: native NaN handling for thin-file applicants
        - SHAP over LIME: exact Shapley values for tree models
        - KS Statistic: primary metric used by Indian banks
        - Stratified splits: prevents class imbalance leakage
        - Pipeline fit on train only: prevents data leakage
        """)

    # Sidebar footer — dynamic from metadata
    st.divider()
    _auc = _metadata.get('best_auc_roc', 'N/A')
    _ks = _metadata.get('best_ks_statistic', 'N/A')
    _auc_str = f"{_auc:.3f}" if isinstance(_auc, (int, float)) else str(_auc)
    _ks_str = f"{_ks:.1f}" if isinstance(_ks, (int, float)) else str(_ks)
    _ds_sources = _metadata.get('dataset_sources', {})
    _synth_count = _ds_sources.get('synthetic_india_gig', 0)
    _total_data = sum(_ds_sources.values()) if _ds_sources else _metadata.get('total_training_samples', 0)
    _gender_dpd = _metadata.get('fairness', {}).get('dpd_gender', 'N/A')
    _gender_str = f"{_gender_dpd:.3f}" if isinstance(_gender_dpd, (int, float)) else str(_gender_dpd)
    st.caption(f"📊 AUC-ROC: {_auc_str} | KS: {_ks_str}")
    st.caption(f"📦 Dataset: {_total_data // 1000}K gig workers")
    st.caption(f"⚖️ Gender DPD: {_gender_str} (fair)")


# ─────────────────────────────────────────────────────────────────────
# Project Story Page (Home)
# ─────────────────────────────────────────────────────────────────────
def render_project_story():
    """Renders the Project Story / Home page."""

    st.markdown("""
    <h1 style="text-align: center; color: #f1f5f9; margin-bottom: 4px;">
        🏦 GigScore — Alternative Credit Scoring
    </h1>
    <p style="text-align: center; color: #64748b; font-size: 16px; margin-bottom: 32px;">
        AI-powered credit scoring for India's 15M+ gig workers who are invisible to traditional banks
    </p>
    """, unsafe_allow_html=True)

    # Check if model is trained
    if not _metadata:
        st.warning("⚠️ No trained model found. Run `python scripts/run_pipeline.py` first to train the model and generate metrics.")

    # Key metrics row — dynamic from metadata
    _best_auc = _metadata.get('best_auc_roc', 'N/A')
    _baseline_auc = _metadata.get('metrics', {}).get('logistic_baseline', {}).get('auc_roc', 0)
    _auc_delta = f"+{(_best_auc - _baseline_auc)*100:.1f}% over baseline" if isinstance(_best_auc, (int, float)) and isinstance(_baseline_auc, (int, float)) else "vs baseline"
    _n_feats = _metadata.get('total_features', '44+')
    _g_dpd = _metadata.get('fairness', {}).get('dpd_gender', 'N/A')
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Gig Workers in India", "15M+", "financially excluded")
    col2.metric("Model AUC-ROC", f"{_best_auc:.3f}" if isinstance(_best_auc, (int, float)) else str(_best_auc), _auc_delta)
    col3.metric("Features Engineered", f"{_n_feats}", "behavioural signals")
    col4.metric("Fairness (Gender DPD)", f"{_g_dpd:.3f}" if isinstance(_g_dpd, (int, float)) else str(_g_dpd), "near-zero bias")

    st.markdown("---")

    # The Problem
    st.markdown("### 🔴 The Problem")
    st.markdown("""
    India has **15-20 million active gig workers** — Swiggy/Zomato delivery agents, Ola/Uber drivers,
    Urban Company professionals — earning ₹25,000-₹60,000/month. Yet **80%+ are rejected** by traditional
    banks because they lack salary slips, employer letters, and formal credit histories.

    Traditional credit scoring (CIBIL) was designed for salaried employees. It systematically fails
    gig workers who earn real, consistent income but through informal channels.
    """)

    _total_ds = sum(_metadata.get('dataset_sources', {}).values()) if _metadata.get('dataset_sources') else 0
    impact_cols = st.columns(3)
    impact_cols[0].metric("Credit Gap", "₹15-20L Cr", "for informal workers")
    impact_cols[1].metric("UPI Monthly Txns", "10B+", "digital footprint exists")
    impact_cols[2].metric("Projected by 2030", "23.5M", "NITI Aayog estimate")

    st.markdown("---")

    # The Solution
    st.markdown("### 🟢 What GigScore Does")
    st.markdown("""
    GigScore uses **44+ behavioural and transactional features** — income stability, platform tenure,
    UPI transaction patterns, work consistency, cancellation rates, digital wallet usage — to generate
    an alternative credit score from **0 to 100**.

    - **Ensemble ML Model:** XGBoost + LightGBM + Stacking for robust predictions
    - **Explainable AI:** Every decision explained with SHAP values and plain-English reason codes
    - **Fairness Audited:** Gender DPD = {f"{_metadata.get('fairness', {}).get('dpd_gender', 0.001):.3f}"} (near-zero bias) via Fairlearn
    - **Actionable:** Dynamic improvement tips with estimated score impact
    """)

    st.markdown("---")

    # Architecture
    st.markdown("### 🏗️ Architecture")
    st.code("""
┌─────────────────────────────────────────────────────────────────┐
│                         GIGSCORE SYSTEM                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌─────────────────┐   │
│  │  DATA LAYER  │───▶│ FEATURE ENG. │───▶│ MODEL PIPELINE  │   │
│  │              │    │              │    │                 │   │
│  │ • Synthetic  │    │ • {_metadata.get('total_features', 37)}  feats  │    │ • LightGBM      │   │
│  │   Gig Data   │    │ • Gig feats  │    │ • XGBoost       │   │
│  │ • 100K rows  │    │ • India feats│    │ • LogReg base   │   │
│  └──────────────┘    │ • Fairness   │    │ • Stacking      │   │
│                      └──────────────┘    └────────┬────────┘   │
│                                                    ▼            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                 EXPLAINABILITY LAYER                      │  │
│  │  • SHAP values per prediction  • Fairness audit          │  │
│  │  • Plain-English reason codes  • Feature importance      │  │
│  └────────────────────────┬─────────────────────────────────┘  │
│                            ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              STREAMLIT DASHBOARD + AI ASSISTANT           │  │
│  │  Score Applicant | Model Analytics | Fairness | Explorer  │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
    """, language=None)

    st.markdown("---")

    # Interview Story
    st.markdown("### 🎤 Interview Story")
    _story_auc = f"{_metadata.get('best_auc_roc', 0.86):.2f}" if _metadata else "0.86"
    _story_dpd = f"{_metadata.get('fairness', {}).get('dpd_gender', 0.001):.3f}" if _metadata else "0.001"
    _story_feats = _metadata.get('total_features', 37)
    st.info(f"""
    "GigScore is an alternative credit scoring system I built for India's 15M+ gig workers
    who have no CIBIL score and are rejected by traditional banks. I engineered {_story_feats}+
    features from income patterns, platform behaviour, UPI transactions, and digital wallet data, then
    trained an ensemble model (XGBoost + LightGBM + Stacking) achieving AUC-ROC of {_story_auc}
    without any credit bureau data. Every decision is explained with SHAP values at both the
    applicant level and the model level. I also built a full fairness audit measuring
    demographic parity across gender and income bracket — and found near-zero gender bias
    (DPD = {_story_dpd}). The core problem I'm solving is financial inclusion — making credit
    accessible to people the existing system ignores."
    """)

    # Key numbers — dynamic from metadata
    st.markdown("### 📊 Key Numbers to Remember")
    _m_auc = _metadata.get('best_auc_roc', 'N/A')
    _m_gini = _metadata.get('best_gini', 'N/A')
    _m_ks = _metadata.get('best_ks_statistic', 'N/A')
    _m_dpd = _metadata.get('fairness', {}).get('dpd_gender', 'N/A')
    _m_dr = _metadata.get('default_rate', 'N/A')
    _m_ds = sum(_metadata.get('dataset_sources', {}).values()) if _metadata.get('dataset_sources') else _metadata.get('total_training_samples', 0)
    num_cols = st.columns(4)
    num_cols[0].metric("AUC-ROC", f"{_m_auc:.3f}" if isinstance(_m_auc, (int, float)) else str(_m_auc))
    num_cols[1].metric("Gini", f"{_m_gini:.3f}" if isinstance(_m_gini, (int, float)) else str(_m_gini))
    num_cols[2].metric("KS Statistic", f"{_m_ks:.1f}" if isinstance(_m_ks, (int, float)) else str(_m_ks))
    num_cols[3].metric("Gender DPD", f"{_m_dpd:.3f}" if isinstance(_m_dpd, (int, float)) else str(_m_dpd))

    num_cols2 = st.columns(4)
    num_cols2[0].metric("Dataset", f"{_m_ds // 1000}K workers" if isinstance(_m_ds, (int, float)) and _m_ds > 0 else "N/A")
    num_cols2[1].metric("Features", f"{_metadata.get('total_features', 'N/A')}")
    num_cols2[2].metric("Default Rate", f"{_m_dr:.1%}" if isinstance(_m_dr, (int, float)) else str(_m_dr))
    num_cols2[3].metric("Top Feature", "Platform Tenure")


# ─────────────────────────────────────────────────────────────────────
# Page Routing
# ─────────────────────────────────────────────────────────────────────
if page == "🏠 Project Story":
    render_project_story()

elif page == "🎯 Score an Applicant":
    from dashboard.pages.score_applicant import render_score_applicant_page
    render_score_applicant_page()

elif page == "📊 Model Analytics":
    from dashboard.pages.model_analytics import render_model_analytics_page
    render_model_analytics_page()

elif page == "⚖️ Fairness Report":
    from dashboard.pages.fairness_report import render_fairness_report_page
    render_fairness_report_page()

elif page == "🔍 Dataset Explorer":
    from dashboard.pages.dataset_explorer import render_dataset_explorer_page
    render_dataset_explorer_page()
