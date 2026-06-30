# =============================================================================
# app.py
# -----------------------------------------------------------------------------
# Application: FinApprove Pro - Intelligent Banking Decision Platform
# -----------------------------------------------------------------------------
# Purpose:
#   Admin-dashboard style UI (AdminLTE / Tabler / Metronic inspired) with a
#   permanent left sidebar built using streamlit-option-menu. This file ONLY
#   builds the presentation layer — all data/ML logic is delegated to the
#   existing, untouched modules:
#       preprocessing.py   -> dataset loading + cleaning
#       visualization.py   -> Plotly chart functions
#       model_training.py  -> training + evaluation pipeline
#       prediction.py      -> single-applicant prediction
#
# DEPENDENCY NOTE:
#   This file requires the "streamlit-option-menu" package.
#   Install with: pip install streamlit-option-menu
# =============================================================================

import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu

from preprocessing import run_preprocessing_pipeline
from visualization import (
    plot_loan_approval_distribution,
    plot_annual_income_distribution,
    plot_credit_score_distribution,
    plot_loan_amount_distribution,
    plot_correlation_heatmap,
    plot_employment_type_distribution,
    plot_education_level_distribution,
    plot_loan_purpose_distribution,
    plot_property_ownership_distribution,
)
from model_training import run_training_pipeline
from prediction import predict_loan


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="FinApprove Pro",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# DESIGN TOKENS — Professional banking palette (charcoal + orange)
# -----------------------------------------------------------------------------
#   App background   : #F2F3F5  (light neutral, not pure white)
#   Surface (cards)   : #FFFFFF
#   Border            : #E1E3E8
#   Sidebar           : #20262E  (charcoal)
#   Sidebar border    : #2D3540
#   Primary (orange)  : #C7611A
#   Primary hover     : #A8500F
#   Text (primary)    : #1C2128
#   Text (secondary)  : #5B6472
#   Success           : #16693F
#   Danger            : #B23A2E
# =============================================================================

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Source+Serif+4:wght@600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;
        font-size: 17px;
    }

    .stApp { background: #F2F3F5; color: #1C2128; }

    #MainMenu, header[data-testid="stHeader"], footer {visibility: hidden;}
    .main .block-container {
        max-width: 98%;
        padding-top: 1rem;
        padding-bottom: 3rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* ---------------- Sidebar shell (structure unchanged, sizing only) ---------------- */
    section[data-testid="stSidebar"] {
        background: #20262E;
        border-right: 1px solid #2D3540;
        min-width: 280px !important;
        font-size: 16px;
    }
    section[data-testid="stSidebar"] .block-container {padding-top: 1.3rem; padding-left: 1rem; padding-right: 1rem;}
    section[data-testid="stSidebar"] * {font-size: 16px;}

    .brand-row {
        display: flex; align-items: center; gap: 0.7rem;
        padding: 0.2rem 0.3rem 1.1rem 0.3rem;
        border-bottom: 1px solid #323B46;
        margin-bottom: 1rem;
    }
    .brand-mark {
        width: 38px; height: 38px; flex-shrink: 0; border-radius: 7px;
        background: #C7611A; color: #FFFFFF; font-weight: 700; font-size: 1rem;
        display: flex; align-items: center; justify-content: center;
        font-family: 'Source Serif 4', serif;
    }
    .brand-title {font-weight: 700; font-size: 1.1rem; color: #F4F5F7; letter-spacing: -0.01em; line-height: 1.2;}
    .brand-sub {font-size: 0.72rem; color: #8A93A1; letter-spacing: 0.05em; margin-top: 0.15rem;}

    .sidebar-session {
        background: #262D36; border: 1px solid #323B46; border-radius: 8px;
        padding: 0.75rem 0.85rem; margin-bottom: 1.1rem;
    }
    .sidebar-session-label {font-size: 0.66rem; color: #7C8696; text-transform: uppercase; letter-spacing: 0.05em;}
    .sidebar-session-value {font-size: 0.88rem; color: #E7E9ED; font-weight: 600; margin-top: 0.15rem;}

    .sidebar-footer {
        margin-top: 1.4rem;
        padding: 0.9rem 0.3rem 0.2rem 0.3rem;
        border-top: 1px solid #323B46;
        font-size: 0.74rem;
        color: #6B7280;
        line-height: 1.65;
    }

    /* Sidebar nav items: smooth hover animation, same look/structure otherwise */
    section[data-testid="stSidebar"] [data-testid="stOptionMenu"] li > a,
    section[data-testid="stSidebar"] .nav-link {
        transition: background-color 0.18s ease, color 0.18s ease, transform 0.18s ease !important;
    }
    section[data-testid="stSidebar"] [data-testid="stOptionMenu"] li > a:hover,
    section[data-testid="stSidebar"] .nav-link:hover {
        transform: translateX(2px);
    }

    /* ---------------- Card surface ---------------- */
    .card {
        background: #FFFFFF;
        border: 1px solid #E1E3E8;
        border-radius: 16px;
        padding: 30px 32px;
        margin-bottom: 1.6rem;
        box-shadow: 0 1px 3px rgba(20,22,26,0.05);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 22px rgba(20,22,26,0.08);
    }

    .card-title {
        font-size: 0.9rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #5B6472;
        margin-bottom: 1.1rem;
        border-bottom: 1px solid #ECEDF0;
        padding-bottom: 0.8rem;
    }
    .card-section-title {
        font-size: 24px;
        font-weight: 700;
        color: #1C2128;
        margin: 0.2rem 0 1rem 0;
        padding-left: 0.6rem;
        border-left: 4px solid #F97316;
    }

    /* ---------------- Page header ---------------- */
    .page-head {margin-bottom: 1.8rem; padding-bottom: 1.1rem; border-bottom: 1px solid #E1E3E8;}
    .page-head h1 {font-family: 'Source Serif 4', serif; font-size: 40px; font-weight: 700; margin: 0 0 0.3rem 0; color: #1C2128;}
    .page-head p {color: #5B6472; margin: 0; font-size: 18px;}

    /* ---------------- KPI cards ---------------- */
    .kpi {
        background: #FFFFFF;
        border: 1px solid #E1E3E8;
        border-left: 4px solid #F97316;
        border-radius: 16px;
        padding: 30px 28px;
        height: 100%;
        min-height: 138px;
        box-shadow: 0 1px 3px rgba(20,22,26,0.05);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }
    .kpi:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 24px rgba(249,115,22,0.14);
        border-left-color: #EA580C;
    }
    .kpi-label {font-size: 0.78rem; color: #5B6472; letter-spacing: 0.04em; text-transform: uppercase; margin-bottom: 0.6rem;}
    .kpi-value {font-size: 2.1rem; font-weight: 800; color: #1C2128; font-variant-numeric: tabular-nums; line-height: 1.15;}
    .kpi-note {font-size: 0.85rem; color: #B86A1F; margin-top: 0.45rem;}

    /* ---------------- Workflow list ---------------- */
    .step-row {
        display: flex; gap: 0.9rem; padding: 0.8rem 0;
        border-bottom: 1px solid #ECEDF0;
    }
    .step-row:last-child {border-bottom: none;}
    .step-no {
        color: #F97316; font-size: 0.95rem; font-weight: 700;
        font-variant-numeric: tabular-nums; width: 1.8rem; flex-shrink: 0;
    }
    .step-label {font-size: 17px; color: #353C46;}

    /* ---------------- Tags / badges ---------------- */
    .tag {
        display: inline-block; padding: 0.35rem 0.85rem; border-radius: 6px;
        font-size: 0.82rem; font-weight: 700; letter-spacing: 0.02em;
        background: #FDBA74; color: #9A3F0C; border: 1px solid #EA580C;
    }

    /* ---------------- Decision result ---------------- */
    .result-card {
        border-radius: 16px;
        padding: 2.1rem 2.2rem;
        margin-bottom: 1.2rem;
        border: 1px solid #E1E3E8;
    }
    .result-approved {background: #F2F8F4; border-left: 5px solid #16693F;}
    .result-rejected {background: #FBF3F1; border-left: 5px solid #B23A2E;}
    .result-title {font-family: 'Source Serif 4', serif; font-size: 1.7rem; font-weight: 700; margin-bottom: 0.3rem;}
    .result-title.approved {color: #16693F;}
    .result-title.rejected {color: #B23A2E;}
    .result-message {color: #5B6472; font-size: 1rem; margin-bottom: 1.4rem;}
    .result-row {display: flex; gap: 3rem; margin-bottom: 1.1rem; flex-wrap: wrap;}
    .result-item-label {font-size: 0.78rem; color: #5B6472; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0.3rem;}
    .result-item-value {font-size: 1.35rem; font-weight: 700; color: #1C2128; font-variant-numeric: tabular-nums;}
    .recommendation-box {
        background: #FFFFFF; border: 1px solid #E1E3E8; border-radius: 10px;
        padding: 1.1rem 1.3rem; font-size: 0.96rem; color: #353C46; margin-top: 1.1rem;
    }
    .recommendation-label {font-size: 0.78rem; color: #5B6472; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0.45rem; font-weight: 600;}

    /* ---------------- Info cards (About page) ---------------- */
    .info-card {
        background: #FFFFFF;
        border: 1px solid #E1E3E8;
        border-top: 4px solid #F97316;
        border-radius: 16px;
        padding: 26px 28px;
        margin-bottom: 1.6rem;
        height: 100%;
        box-shadow: 0 1px 3px rgba(20,22,26,0.05);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }
    .info-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 24px rgba(20,22,26,0.09);
    }
    .info-card-title {font-size: 0.78rem; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; color: #B86A1F; margin-bottom: 0.9rem;}
    .info-row {display: flex; justify-content: space-between; gap: 1rem; padding: 0.5rem 0; border-bottom: 1px solid #F0F1F3; font-size: 16px;}
    .info-row:last-child {border-bottom: none;}
    .info-row-label {color: #5B6472;}
    .info-row-value {color: #1C2128; font-weight: 600; text-align: right;}
    .info-text {font-size: 16px; color: #353C46; line-height: 1.7; margin: 0;}

    /* ---------------- Streamlit widget overrides ---------------- */
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E1E3E8;
        border-left: 4px solid #F97316;
        border-radius: 12px;
        padding: 1rem 1.2rem;
    }
    div[data-testid="stMetricLabel"] {color: #5B6472 !important; font-size: 0.85rem !important;}
    div[data-testid="stMetricValue"] {color: #1C2128 !important; font-size: 1.6rem !important;}

    div.element-container .stButton > button,
    div.element-container .stFormSubmitButton > button {
        background: #F97316 !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 17px !important;
        border: 1px solid #F97316 !important;
        border-radius: 10px !important;
        padding: 0.85rem 1.4rem !important;
        box-shadow: none !important;
        transition: background-color 0.18s ease, box-shadow 0.18s ease !important;
    }
    div.element-container .stButton > button:hover,
    div.element-container .stFormSubmitButton > button:hover {
        background: #FB923C !important;
        border-color: #FB923C !important;
        box-shadow: 0 6px 16px rgba(249,115,22,0.35) !important;
    }

    input, .stNumberInput input, .stTextInput input {
        background: #FFFFFF !important;
        color: #1C2128 !important;
        border: 1px solid #D7DAE0 !important;
        border-radius: 8px !important;
        font-size: 16px !important;
        padding: 0.55rem 0.7rem !important;
    }
    div[data-baseweb="select"] > div {
        background: #FFFFFF !important;
        border: 1px solid #D7DAE0 !important;
        border-radius: 8px !important;
        color: #1C2128 !important;
        font-size: 16px !important;
    }
    label {color: #5B6472 !important; font-size: 0.95rem !important; font-weight: 500 !important;}

    .stProgress > div > div > div > div {
        background: #F97316 !important;
    }

    div[data-testid="stExpander"] {
        background: #FFFFFF;
        border: 1px solid #E1E3E8;
        border-radius: 10px;
    }

    /* ---------------- Dataframe / table styling ---------------- */
    .stDataFrame {border-radius: 10px; overflow: hidden;}
    [data-testid="stDataFrame"] thead tr th {
        background-color: #F97316 !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        padding: 14px 12px !important;
    }
    [data-testid="stDataFrame"] tbody tr td {
        font-size: 16px !important;
        padding: 12px 12px !important;
    }
    [data-testid="stDataFrame"] tbody tr:hover {
        background-color: #FDE9D8 !important;
    }

    hr {border-color: #E1E3E8 !important;}

    .app-footer {
        text-align: left; color: #8A93A1; font-size: 0.9rem;
        padding-top: 1.3rem; border-top: 1px solid #E1E3E8; margin-top: 0.8rem;
    }
</style>
"""


# =============================================================================
# CACHED DATA / MODEL LOADERS
# =============================================================================
@st.cache_data(show_spinner=False)
def get_preprocessed_data():
    """Runs the existing preprocessing pipeline once and caches the result."""
    return run_preprocessing_pipeline(apply_scaling=False)


@st.cache_resource(show_spinner=False)
def get_training_results():
    """Runs the existing model training pipeline once and caches the result."""
    return run_training_pipeline()


# =============================================================================
# PERMANENT LEFT SIDEBAR — built with streamlit-option-menu
# -----------------------------------------------------------------------------
# option_menu renders inside `with st.sidebar:` on every script rerun, so it
# persists across all page navigations (it is not conditionally mounted).
# =============================================================================
with st.sidebar:
    st.markdown(
        """
        <div class="brand-row">
            <div class="brand-mark">FP</div>
            <div>
                <div class="brand-title">FinApprove Pro</div>
                <div class="brand-sub">BANKING DECISION PLATFORM</div>
            </div>
        </div>
        <div class="sidebar-session">
            <div class="sidebar-session-label">Logged in as</div>
            <div class="sidebar-session-value">Credit Underwriting Desk</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    page = option_menu(
        menu_title=None,
        options=["Dashboard", "Dataset", "Analytics", "Model Performance", "Loan Prediction", "About"],
        icons=["speedometer2", "table", "bar-chart-line", "cpu", "credit-card", "info-circle"],
        default_index=0,
        styles={
            "container": {"padding": "0", "background-color": "#20262E"},
            "icon": {"color": "#8A93A1", "font-size": "0.9rem"},
            "nav-link": {
                "font-size": "0.87rem",
                "font-weight": "500",
                "color": "#C7CCD4",
                "text-align": "left",
                "margin": "0.16rem 0",
                "padding": "0.6rem 0.8rem",
                "border-radius": "6px",
                "--hover-color": "#2A313B",
            },
            "nav-link-selected": {
                "background-color": "#C7611A",
                "color": "#FFFFFF",
                "font-weight": "600",
                "icon-color": "#FFFFFF",
            },
        }
    )

    st.markdown(
        """
        <div class="sidebar-footer">
            FinApprove Pro — Internal Underwriting System<br>Release 1.0
        </div>
        """,
        unsafe_allow_html=True
    )


def page_header(title: str, subtitle: str):
    """Renders a consistent page header used on every page."""
    st.markdown(
        f"""
        <div class="page-head">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# =============================================================================
# PAGE: DASHBOARD
# =============================================================================
def render_dashboard():
    page_header("Dashboard", "Overview of the underwriting model and applicant portfolio")

    data = get_preprocessed_data()
    raw_df = data["raw_df"]

    results = get_training_results()
    comparison_df = results["comparison_df"]
    best_model_name = results["best_model_name"]
    best_accuracy = comparison_df.loc[comparison_df["Model"] == best_model_name, "Accuracy"].iloc[0]

    target_col = "loan_approved"
    approval_rate = raw_df[target_col].mean() * 100 if target_col in raw_df.columns else 0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"""<div class="kpi"><div class="kpi-label">Dataset Size</div>
            <div class="kpi-value">{raw_df.shape[0]:,}</div>
            <div class="kpi-note">{raw_df.shape[1]} fields per record</div></div>""",
            unsafe_allow_html=True
        )
    with k2:
        st.markdown(
            f"""<div class="kpi"><div class="kpi-label">Model Accuracy</div>
            <div class="kpi-value">{best_accuracy * 100:.1f}%</div>
            <div class="kpi-note">on held-out test set</div></div>""",
            unsafe_allow_html=True
        )
    with k3:
        st.markdown(
            f"""<div class="kpi"><div class="kpi-label">Best Model</div>
            <div class="kpi-value" style="font-size:1.1rem;">{best_model_name}</div>
            <div class="kpi-note">selected on test accuracy</div></div>""",
            unsafe_allow_html=True
        )
    with k4:
        st.markdown(
            f"""<div class="kpi"><div class="kpi-label">Approval Rate</div>
            <div class="kpi-value">{approval_rate:.1f}%</div>
            <div class="kpi-note">historical applicants</div></div>""",
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns([1.1, 1])

    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Recent Performance</div>', unsafe_allow_html=True)
        display_df = comparison_df[["Model", "Accuracy", "F1 Score"]].copy()
        display_df["Accuracy"] = (display_df["Accuracy"] * 100).round(2)
        display_df["F1 Score"] = (display_df["F1 Score"] * 100).round(2)
        st.dataframe(
            display_df.style.format({"Accuracy": "{:.2f}%", "F1 Score": "{:.2f}%"}),
            use_container_width=True, hide_index=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Workflow</div>', unsafe_allow_html=True)
        steps = [
            "Applicant data ingested and validated",
            "Missing values handled, categories encoded",
            "Stratified 80/20 train-test split applied",
            "Four models trained and evaluated",
            "Best model selected and persisted",
            "Decision served through this platform",
        ]
        for i, label in enumerate(steps, start=1):
            st.markdown(
                f"""<div class="step-row">
                <div class="step-no">{i:02d}</div>
                <div class="step-label">{label}</div></div>""",
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Project Summary</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <p style="color:#353C46; font-size:0.87rem; line-height:1.6; margin:0;">
        FinApprove Pro evaluates loan applications using income, employment,
        credit history and existing liabilities. The platform benchmarks four
        classification models and deploys the most accurate one for live
        underwriting decisions.
        </p>
        """,
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# PAGE: DATASET
# =============================================================================
def render_dataset():
    page_header("Dataset", "Applicant records used to train the underwriting model")

    data = get_preprocessed_data()
    raw_df = data["raw_df"]

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""<div class="kpi"><div class="kpi-label">Total Rows</div>
        <div class="kpi-value">{raw_df.shape[0]:,}</div></div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class="kpi"><div class="kpi-label">Total Columns</div>
        <div class="kpi-value">{raw_df.shape[1]}</div></div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class="kpi"><div class="kpi-label">Missing Values</div>
        <div class="kpi-value">{int(raw_df.isnull().sum().sum())}</div></div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class="kpi"><div class="kpi-label">Duplicate Rows</div>
        <div class="kpi-value">{int(raw_df.duplicated().sum())}</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    head_col1, head_col2 = st.columns([4, 1])
    with head_col1:
        st.markdown('<div class="card-title">Applicant Records (first 10 rows)</div>', unsafe_allow_html=True)
    with head_col2:
        st.download_button(
            "Download CSV",
            data=raw_df.to_csv(index=False).encode("utf-8"),
            file_name="loan_data.csv",
            mime="text/csv",
            use_container_width=True
        )
    st.dataframe(raw_df.head(10), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Column Data Types</div>', unsafe_allow_html=True)
        st.dataframe(raw_df.dtypes.astype(str).rename("dtype"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Missing Values per Column</div>', unsafe_allow_html=True)
        st.dataframe(raw_df.isnull().sum().rename("missing"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Statistical Summary</div>', unsafe_allow_html=True)
    st.dataframe(raw_df.describe(), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# PAGE: ANALYTICS
# =============================================================================
def render_analytics():
    page_header("Analytics", "Distribution and correlation insights across the applicant base")

    data = get_preprocessed_data()
    raw_df = data["raw_df"]

    chart_pairs = [
        (plot_loan_approval_distribution, plot_annual_income_distribution),
        (plot_credit_score_distribution, plot_loan_amount_distribution),
        (plot_employment_type_distribution, plot_education_level_distribution),
        (plot_loan_purpose_distribution, plot_property_ownership_distribution),
    ]

    for left_fn, right_fn in chart_pairs:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.plotly_chart(left_fn(raw_df), use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.plotly_chart(right_fn(raw_df), use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Feature Correlation Heatmap</div>', unsafe_allow_html=True)
    st.plotly_chart(plot_correlation_heatmap(raw_df), use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# PAGE: MODEL PERFORMANCE
# =============================================================================
def render_model_performance():
    page_header("Model Performance", "Benchmark of every trained algorithm and the deployed model")

    with st.spinner("Loading model performance data..."):
        results = get_training_results()

    comparison_df = results["comparison_df"]
    best_model_name = results["best_model_name"]
    best_row = comparison_df.loc[comparison_df["Model"] == best_model_name].iloc[0]

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Deployed Model</div>', unsafe_allow_html=True)
    st.markdown(f'<span class="tag">{best_model_name}</span>', unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{best_row['Accuracy'] * 100:.2f}%")
    m2.metric("Precision", f"{best_row['Precision'] * 100:.2f}%")
    m3.metric("Recall", f"{best_row['Recall'] * 100:.2f}%")
    m4.metric("F1 Score", f"{best_row['F1 Score'] * 100:.2f}%")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Model Comparison</div>', unsafe_allow_html=True)
    display_df = comparison_df.drop(columns=["Confusion Matrix"]).copy()
    for metric_col in ["Accuracy", "Precision", "Recall", "F1 Score"]:
        display_df[metric_col] = (display_df[metric_col] * 100).round(2)
    st.dataframe(
        display_df.style.format({
            "Accuracy": "{:.2f}%", "Precision": "{:.2f}%",
            "Recall": "{:.2f}%", "F1 Score": "{:.2f}%"
        }),
        use_container_width=True, hide_index=True
    )

    for _, row in comparison_df.iterrows():
        st.markdown(
            f'<span style="font-size:0.82rem; color:#9CA3AF;">{row["Model"]}</span>',
            unsafe_allow_html=True
        )
        st.progress(min(float(row["Accuracy"]), 1.0))
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Confusion Matrices</div>', unsafe_allow_html=True)
    for _, row in comparison_df.iterrows():
        with st.expander(f"{row['Model']}"):
            cm_df = pd.DataFrame(
                row["Confusion Matrix"],
                index=["Actual: Rejected", "Actual: Approved"],
                columns=["Predicted: Rejected", "Predicted: Approved"]
            )
            st.dataframe(cm_df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# PAGE: LOAN PREDICTION
# =============================================================================
def render_loan_prediction():
    page_header("Loan Prediction", "Enter applicant details to generate an underwriting decision")

    st.markdown(
        """
        <div class="card" style="display:flex; justify-content:space-between; align-items:center; padding:0.9rem 1.4rem;">
            <div>
                <div class="card-title" style="margin:0; border:none; padding:0;">Application Form</div>
                <div style="font-size:0.78rem; color:#5B6472; margin-top:0.2rem;">Form Ref. ULA-100 &middot; Unsecured / Secured Loan Underwriting</div>
            </div>
            <span class="tag">In Progress</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    with st.form("loan_prediction_form"):

        st.markdown('<div class="card-section-title">01 &nbsp;Financial Information</div>', unsafe_allow_html=True)
        f1, f2 = st.columns(2)
        with f1:
            annual_income = st.number_input("Annual Income (₹)", min_value=0, value=600000, step=10000)
            existing_loans = st.number_input("Existing Loans", min_value=0, value=1, step=1)
            loan_tenure_months = st.number_input("Loan Tenure (months)", min_value=1, value=60, step=1)
        with f2:
            monthly_expenses = st.number_input("Monthly Expenses (₹)", min_value=0, value=18000, step=1000)
            loan_amount = st.number_input("Loan Amount (₹)", min_value=0, value=200000, step=10000)

        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="card-section-title">02 &nbsp;Applicant Information</div>', unsafe_allow_html=True)
        a1, a2 = st.columns(2)
        with a1:
            employment_years = st.number_input("Employment Years", min_value=0, value=5, step=1)
            employment_type = st.selectbox(
                "Employment Type", ["Salaried", "Self-Employed", "Business", "Government"]
            )
            education_level = st.selectbox(
                "Education Level", ["High School", "Bachelor", "Master", "PhD"]
            )
        with a2:
            loan_purpose = st.selectbox(
                "Loan Purpose", ["Home", "Education", "Vehicle", "Business", "Personal"]
            )
            owns_property = st.selectbox(
                "Property Ownership", ["Owned", "Not Owned"]
            )

        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="card-section-title">03 &nbsp;Credit Assessment</div>', unsafe_allow_html=True)
        sc1, sc2 = st.columns([4, 1])
        with sc1:
            credit_score = st.slider("Credit Score", min_value=300, max_value=900, value=720, label_visibility="collapsed")
        with sc2:
            st.markdown(
                f"""<div style="text-align:center; padding-top:0.1rem;">
                <div class="kpi-label" style="margin-bottom:0.15rem;">Bureau Score</div>
                <div class="kpi-value" style="font-size:1.25rem;">{credit_score}</div>
                </div>""",
                unsafe_allow_html=True
            )

        st.markdown(
            """
            <div class="recommendation-box" style="margin-top:1rem;">
            By submitting this application, the underwriting desk confirms that the
            details above have been collected from the applicant and are accurate
            to the best of its knowledge.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<div style='height:0.7rem'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Submit for Decision", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if not submitted:
        return

    # NOTE: the trained model's feature set uses "loan_tenure_months", not
    # "loan_tenure" — kept consistent with the fix applied in the prior pass.
    # property_ownership is a binary flag (1 = Owned, 0 = Not Owned) in the
    # training data, not a free-text category — converted here accordingly.
    applicant_data = {
        "annual_income": annual_income,
        "employment_years": employment_years,
        "credit_score": credit_score,
        "existing_loans": existing_loans,
        "loan_amount": loan_amount,
        "loan_tenure_months": loan_tenure_months,
        "monthly_expenses": monthly_expenses,
        "employment_type": employment_type,
        "education_level": education_level,
        "loan_purpose": loan_purpose,
        "property_ownership": 1 if owns_property == "Owned" else 0,
    }

    with st.spinner("Scoring applicant..."):
        try:
            result = predict_loan(applicant_data)
        except Exception as error:
            st.error(f"Prediction failed: {error}")
            return

    is_approved = result["prediction"] == "Approved"
    confidence = result["probability"]

    # The model returns confidence in whichever class it predicted, not the
    # probability of approval specifically. Convert to an approval-likelihood
    # score before deriving a risk band, so a confidently REJECTED applicant
    # is correctly shown as High Risk rather than Low Risk.
    approval_likelihood = confidence if is_approved else (100 - confidence)

    if approval_likelihood >= 75:
        risk_level = "Low Risk"
        recommendation = "Applicant meets standard approval criteria. Proceed with standard terms."
    elif approval_likelihood >= 40:
        risk_level = "Moderate Risk"
        recommendation = "Applicant is borderline. Consider manual review of supporting documents."
    else:
        risk_level = "High Risk"
        recommendation = "Applicant does not meet minimum criteria. Decline or request additional collateral."

    card_class = "result-approved" if is_approved else "result-rejected"
    title_class = "approved" if is_approved else "rejected"
    title_text = "Loan Approved" if is_approved else "Loan Rejected"

    st.markdown(f'<div class="result-card {card_class}">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="result-title {title_class}">{title_text}</div>
        <div class="result-message">{result['message']}</div>
        <div class="result-row">
            <div>
                <div class="result-item-label">Approval Likelihood</div>
                <div class="result-item-value">{approval_likelihood:.1f}%</div>
            </div>
            <div>
                <div class="result-item-label">Model Confidence</div>
                <div class="result-item-value">{confidence}%</div>
            </div>
            <div>
                <div class="result-item-label">Risk Level</div>
                <div class="result-item-value">{risk_level}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.progress(min(approval_likelihood / 100, 1.0))
    st.markdown(
        f"""
        <div class="recommendation-box">
            <div class="recommendation-label">Recommendation</div>
            {recommendation}
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# PAGE: ABOUT
# =============================================================================
def render_about():
    page_header("About", "Project background and technical details")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-card-title">Project</div>
                <div class="info-row"><div class="info-row-label">Name</div><div class="info-row-value">FinApprove Pro</div></div>
                <div class="info-row"><div class="info-row-label">Domain</div><div class="info-row-value">Banking &amp; Financial Risk Analytics</div></div>
                <div class="info-row"><div class="info-row-label">Target Variable</div><div class="info-row-value">loan_approved</div></div>
                <div class="info-row"><div class="info-row-label">Type</div><div class="info-row-value">Binary Classification</div></div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="info-card">
                <div class="info-card-title">Architecture</div>
                <div class="info-row"><div class="info-row-label">preprocessing.py</div><div class="info-row-value">Cleans data, encodes categories</div></div>
                <div class="info-row"><div class="info-row-label">visualization.py</div><div class="info-row-value">Reusable chart functions</div></div>
                <div class="info-row"><div class="info-row-label">model_training.py</div><div class="info-row-value">Trains, evaluates, selects best model</div></div>
                <div class="info-row"><div class="info-row-label">prediction.py</div><div class="info-row-value">Serves single-applicant predictions</div></div>
                <div class="info-row"><div class="info-row-label">app.py</div><div class="info-row-value">This dashboard</div></div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-card-title">Technology</div>
                <div class="info-row"><div class="info-row-label">Language</div><div class="info-row-value">Python</div></div>
                <div class="info-row"><div class="info-row-label">Interface</div><div class="info-row-value">Streamlit</div></div>
                <div class="info-row"><div class="info-row-label">Machine Learning</div><div class="info-row-value">Scikit-Learn</div></div>
                <div class="info-row"><div class="info-row-label">Data Handling</div><div class="info-row-value">Pandas, NumPy</div></div>
                <div class="info-row"><div class="info-row-label">Visualization</div><div class="info-row-value">Plotly</div></div>
                <div class="info-row"><div class="info-row-label">Persistence</div><div class="info-row-value">Joblib</div></div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="info-card">
                <div class="info-card-title">Models</div>
                <div class="info-row"><div class="info-row-label">Logistic Regression</div><div class="info-row-value">Linear Classifier</div></div>
                <div class="info-row"><div class="info-row-label">Decision Tree</div><div class="info-row-value">Tree-Based</div></div>
                <div class="info-row"><div class="info-row-label">Random Forest</div><div class="info-row-value">Ensemble (Bagging)</div></div>
                <div class="info-row"><div class="info-row-label">K-Nearest Neighbors</div><div class="info-row-value">Instance-Based</div></div>
                <p class="info-text" style="margin-top:0.9rem;">
                    The best performer on the held-out test set is automatically
                    persisted and used for every live prediction.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="info-card">
                <div class="info-card-title">Developer</div>
                <div class="info-row"><div class="info-row-label">Team</div><div class="info-row-value">FinApprove Pro Development Team</div></div>
                <div class="info-row"><div class="info-row-label">Platform</div><div class="info-row-value">Internal Underwriting System</div></div>
                <div class="info-row"><div class="info-row-label">Release</div><div class="info-row-value">1.0</div></div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="app-footer">FinApprove Pro — Intelligent Banking Decision Platform. '
        'Built with Streamlit, Scikit-Learn and Plotly.</div>',
        unsafe_allow_html=True
    )


# =============================================================================
# PAGE ROUTER
# =============================================================================
if page == "Dashboard":
    render_dashboard()
elif page == "Dataset":
    render_dataset()
elif page == "Analytics":
    render_analytics()
elif page == "Model Performance":
    render_model_performance()
elif page == "Loan Prediction":
    render_loan_prediction()
elif page == "About":
    render_about()