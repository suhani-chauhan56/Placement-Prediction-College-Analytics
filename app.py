import json
import os
import pickle
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

try:
    import shap
except Exception:
    shap = None


st.set_page_config(
    page_title="Placement Analytics & Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #f5f1ea;
        --bg-2: #ece4d8;
        --surface: rgba(255, 255, 255, 0.72);
        --surface-strong: rgba(255, 255, 255, 0.88);
        --text: #1f2937;
        --muted: #6b7280;
        --line: rgba(31, 41, 55, 0.08);
        --blue: #174ea6;
        --blue-soft: rgba(23, 78, 166, 0.12);
        --green: #0f8b5f;
        --green-soft: rgba(15, 139, 95, 0.12);
        --gold: #c58b2a;
        --gold-soft: rgba(197, 139, 42, 0.14);
        --shadow: 0 20px 55px rgba(31, 41, 55, 0.12);
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(23, 78, 166, 0.12), transparent 28%),
            radial-gradient(circle at top right, rgba(197, 139, 42, 0.10), transparent 24%),
            linear-gradient(180deg, var(--bg) 0%, #f8f4ee 50%, var(--bg-2) 100%);
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }

    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        background-image:
            linear-gradient(rgba(31,41,55,0.04) 1px, transparent 1px),
            linear-gradient(90deg, rgba(31,41,55,0.04) 1px, transparent 1px);
        background-size: 56px 56px;
        mask-image: linear-gradient(180deg, rgba(0,0,0,0.35), transparent 90%);
        pointer-events: none;
        z-index: 0;
    }

    [data-testid="stAppViewContainer"] > .main {
        position: relative;
        z-index: 1;
    }

    .hero-wrap {
        position: relative;
        overflow: hidden;
        background: linear-gradient(135deg, rgba(255,255,255,0.84), rgba(255,255,255,0.66));
        border: 1px solid var(--line);
        border-radius: 28px;
        padding: 2rem 2rem 1.4rem 2rem;
        box-shadow: var(--shadow);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        animation: fadeUp 700ms ease-out both;
    }

    .hero-wrap::before,
    .hero-wrap::after {
        content: "";
        position: absolute;
        border-radius: 999px;
        filter: blur(2px);
        opacity: 0.7;
        animation: floaty 7s ease-in-out infinite;
    }

    .hero-wrap::before {
        width: 180px;
        height: 180px;
        background: radial-gradient(circle, rgba(23, 78, 166, 0.20), transparent 70%);
        top: -48px;
        right: 14%;
    }

    .hero-wrap::after {
        width: 240px;
        height: 240px;
        background: radial-gradient(circle, rgba(197, 139, 42, 0.16), transparent 72%);
        bottom: -90px;
        left: -36px;
        animation-delay: -2.8s;
    }

    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 0.55rem;
        padding: 0.45rem 0.8rem;
        border: 1px solid rgba(23, 78, 166, 0.16);
        border-radius: 999px;
        background: rgba(255,255,255,0.72);
        color: var(--blue);
        font-size: 0.82rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(2.2rem, 4vw, 4.4rem);
        line-height: 0.98;
        margin: 0;
        font-weight: 800;
        letter-spacing: -0.04em;
        background: linear-gradient(135deg, #111827 15%, #174ea6 52%, #c58b2a 88%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        position: relative;
        z-index: 1;
    }

    .hero-subtitle {
        color: var(--muted);
        font-size: 1.02rem;
        max-width: 68ch;
        margin-top: 0.9rem;
        line-height: 1.65;
        position: relative;
        z-index: 1;
    }

    .feature-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        border-radius: 999px;
        padding: 0.45rem 0.8rem;
        background: rgba(255,255,255,0.82);
        border: 1px solid var(--line);
        color: var(--text);
        font-size: 0.84rem;
        margin: 0.25rem 0.45rem 0 0;
        animation: fadeUp 800ms ease-out both;
    }

    .glass-card {
        background: linear-gradient(180deg, var(--surface-strong), var(--surface));
        border: 1px solid var(--line);
        border-radius: 22px;
        padding: 1.2rem 1.15rem;
        box-shadow: var(--shadow);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        animation: fadeUp 700ms ease-out both;
    }

    .metric-card {
        background: linear-gradient(180deg, var(--surface-strong), var(--surface));
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 1.1rem 1rem;
        height: 100%;
        box-shadow: 0 14px 38px rgba(31,41,55,0.08);
        transition: transform 180ms ease, border-color 180ms ease, box-shadow 180ms ease;
    }

    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(23, 78, 166, 0.22);
        box-shadow: 0 18px 50px rgba(31,41,55,0.12);
    }

    .metric-label {
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.11em;
        font-size: 0.75rem;
        margin-bottom: 0.5rem;
    }

    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2rem;
        line-height: 1;
        font-weight: 700;
        color: var(--text);
    }

    .metric-note {
        margin-top: 0.55rem;
        color: var(--muted);
        font-size: 0.86rem;
    }

    .placed-card, .unplaced-card {
        border-radius: 20px;
        padding: 1.4rem 1.2rem;
        border: 1px solid var(--line);
        background: rgba(255,255,255,0.82);
        box-shadow: var(--shadow);
        animation: fadeUp 700ms ease-out both;
    }

    .placed-card {
        background: linear-gradient(135deg, rgba(15, 139, 95, 0.14), rgba(255,255,255,0.86));
        border-color: rgba(15, 139, 95, 0.22);
    }

    .unplaced-card {
        background: linear-gradient(135deg, rgba(185, 28, 28, 0.12), rgba(255,255,255,0.86));
        border-color: rgba(185, 28, 28, 0.18);
    }

    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.25rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0 0 0.6rem 0;
        color: var(--text);
    }

    .subtle-copy {
        color: var(--muted);
        line-height: 1.55;
        font-size: 0.95rem;
    }

    .insight-box {
        border-radius: 16px;
        padding: 0.9rem 1rem;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        color: var(--text);
        margin-bottom: 0.7rem;
    }

    .quick-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.75rem;
    }

    .quick-item {
        padding: 0.8rem 0.9rem;
        border-radius: 15px;
        background: rgba(255,255,255,0.80);
        border: 1px solid var(--line);
    }

    .quick-item .k {
        color: var(--muted);
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .quick-item .v {
        color: var(--text);
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: 0.25rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        height: 48px;
        padding: 0 16px;
        border-radius: 14px;
        background: rgba(255,255,255,0.82);
        border: 1px solid var(--line);
        color: var(--text);
        font-weight: 600;
        transition: all 180ms ease;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(23, 78, 166, 0.96), rgba(15, 139, 95, 0.92)) !important;
        color: #ffffff !important;
        box-shadow: 0 12px 30px rgba(23, 78, 166, 0.22);
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(245, 241, 234, 0.96));
        border-right: 1px solid var(--line);
    }

    .stNumberInput input,
    div[data-baseweb="select"] > div {
        background-color: rgba(255,255,255,0.92) !important;
        color: var(--text) !important;
        border-color: var(--line) !important;
    }

    .stSlider [data-baseweb="slider"] {
        color: var(--blue);
    }

    @keyframes fadeUp {
        from {
            opacity: 0;
            transform: translateY(16px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes floaty {
        0%, 100% { transform: translateY(0px) scale(1); }
        50% { transform: translateY(16px) scale(1.03); }
    }

    .pulse-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 999px;
        background: var(--green);
        box-shadow: 0 0 0 rgba(15,139,95,0.45);
        animation: pulse 1.8s infinite;
    }

    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(15,139,95,0.4); }
        70% { box-shadow: 0 0 0 18px rgba(15,139,95,0); }
        100% { box-shadow: 0 0 0 0 rgba(15,139,95,0); }
    }
</style>
""",
    unsafe_allow_html=True,
)


def render_metric(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {"<div class='metric-note'>" + note + "</div>" if note else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def score_readiness(cgpa, backlogs, internships, certifications, coding_skills, communication_skills, project_count):
    score = (
        cgpa * 8
        + coding_skills * 5
        + communication_skills * 4
        + internships * 8
        + certifications * 3
        + project_count * 4
        - backlogs * 8
    )
    return int(np.clip(score, 0, 100))


def style_chart_ax(ax, title, xlabel="", ylabel="", grid_alpha=0.08):
    ax.set_title(title, color="#1f2937", fontsize=11, pad=8, fontweight="semibold")
    ax.set_xlabel(xlabel, color="#334155", fontsize=10, labelpad=6)
    ax.set_ylabel(ylabel, color="#334155", fontsize=10, labelpad=6)
    ax.tick_params(colors="#475569", labelsize=9)
    ax.grid(color=(31 / 255, 41 / 255, 55 / 255, grid_alpha), linestyle="--", linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color((31 / 255, 41 / 255, 55 / 255, 0.12))


def make_plotly_theme(fig, height=340):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=18, r=18, t=26, b=18),
        font=dict(color="#1f2937", family="Inter"),
    )
    return fig


def ensure_runtime_columns(df: pd.DataFrame) -> pd.DataFrame:
    runtime = df.copy()
    if "has_internship" not in runtime.columns and "internship_status" in runtime.columns:
        runtime["has_internship"] = runtime["internship_status"].map({"Yes": 1, "No": 0})
    if "has_certification" not in runtime.columns and "certification_status" in runtime.columns:
        runtime["has_certification"] = runtime["certification_status"].map({"Yes": 1, "No": 0})
    if "total_skill_score" not in runtime.columns and {"coding_skills", "communication_skills"}.issubset(runtime.columns):
        runtime["total_skill_score"] = runtime["coding_skills"] + runtime["communication_skills"]
    if "academic_performance_index" not in runtime.columns and {"cgpa", "backlogs"}.issubset(runtime.columns):
        runtime["academic_performance_index"] = (runtime["cgpa"] * 10) - (runtime["backlogs"] * 5)
    if "employability_score" not in runtime.columns and {"total_skill_score", "cgpa", "internships"}.issubset(runtime.columns):
        runtime["employability_score"] = (runtime["total_skill_score"] * 0.4) + (runtime["cgpa"] * 0.4) + (runtime["internships"] * 2.0)
    if "placement_year" not in runtime.columns and "student_id" in runtime.columns:
        runtime["placement_year"] = 2019 + ((runtime["student_id"] - 1) % 5)
    if "placement_risk_score" not in runtime.columns and {
        "cgpa",
        "backlogs",
        "internships",
        "certifications",
        "coding_skills",
        "communication_skills",
        "project_count",
    }.issubset(runtime.columns):
        runtime["placement_risk_score"] = (
            100
            - (runtime["cgpa"] * 6.0)
            - (runtime["coding_skills"] * 4.0)
            - (runtime["communication_skills"] * 3.0)
            - (runtime["internships"] * 8.0)
            - (runtime["project_count"] * 4.0)
            - (runtime["certifications"] * 2.0)
            + (runtime["backlogs"] * 12.0)
        ).clip(0, 100).round(1)
    if "risk_band" not in runtime.columns and "placement_risk_score" in runtime.columns:
        runtime["risk_band"] = pd.cut(
            runtime["placement_risk_score"],
            bins=[-0.1, 35, 65, 100],
            labels=["Low", "Medium", "High"],
        ).astype(str)
    if "eligible_for_shortlist" not in runtime.columns and {"cgpa", "backlogs"}.issubset(runtime.columns):
        runtime["eligible_for_shortlist"] = ((runtime["cgpa"] >= 6.5) & (runtime["backlogs"] == 0)).astype(int)
    if "student_intervention" not in runtime.columns and "risk_band" in runtime.columns:
        runtime["student_intervention"] = np.where(
            runtime["risk_band"] == "High",
            "Improve aptitude; complete one internship; increase DSA score",
            np.where(
                runtime["risk_band"] == "Medium",
                "Strengthen projects and communication",
                "Maintain momentum and continue interview prep",
            ),
        )
    if "intervention_priority" not in runtime.columns and "risk_band" in runtime.columns:
        runtime["intervention_priority"] = np.where(
            runtime["risk_band"] == "High",
            "Immediate",
            np.where(runtime["risk_band"] == "Medium", "Monitor", "Track"),
        )
    return runtime


def placement_risk_score(cgpa, backlogs, internships, certifications, coding_skills, communication_skills, project_count):
    score = (
        100
        - (cgpa * 6.0)
        - (coding_skills * 4.0)
        - (communication_skills * 3.0)
        - (internships * 8.0)
        - (project_count * 4.0)
        - (certifications * 2.0)
        + (backlogs * 12.0)
    )
    return float(np.clip(score, 0, 100))


def risk_band(score):
    if score <= 35:
        return "Low"
    if score <= 65:
        return "Medium"
    return "High"


def recommendation_actions(cgpa, backlogs, internships, certifications, coding_skills, communication_skills, project_count):
    actions = []
    if backlogs > 0:
        actions.append("Clear backlogs to reduce screening risk")
    if cgpa < 7.0:
        actions.append("Raise CGPA above 7.0")
    if internships == 0:
        actions.append("Complete at least one internship")
    if coding_skills < 6:
        actions.append("Practice DSA and coding interviews")
    if communication_skills < 6:
        actions.append("Work on communication and mock interviews")
    if project_count < 2:
        actions.append("Add more portfolio projects")
    if certifications == 0:
        actions.append("Earn one industry certification")
    return actions[:4] if actions else ["Profile is competitive; keep polishing interview readiness"]


def explain_profile(model, input_data, reference_df):
    if shap is not None and hasattr(model, "named_steps"):
        preprocessor = model.named_steps.get("preprocessor")
        estimator = model.named_steps.get("classifier") or model.named_steps.get("regressor")
        if preprocessor is not None and estimator is not None:
            try:
                sample = reference_df.sample(min(200, len(reference_df)), random_state=42)
                background = preprocessor.transform(sample)
                transformed = preprocessor.transform(input_data)
                if hasattr(background, "toarray"):
                    background = background.toarray()
                if hasattr(transformed, "toarray"):
                    transformed = transformed.toarray()

                if estimator.__class__.__name__ == "LogisticRegression":
                    explainer = shap.LinearExplainer(estimator, background)
                    values = explainer(transformed).values[0]
                elif hasattr(estimator, "feature_importances_"):
                    explainer = shap.TreeExplainer(estimator)
                    values = explainer(transformed).values[0]
                else:
                    explainer = shap.Explainer(estimator, background)
                    values = explainer(transformed).values[0]

                feature_names = list(preprocessor.get_feature_names_out())
                if len(values) == len(feature_names):
                    return pd.Series(values, index=feature_names).sort_values(ascending=False)
            except Exception:
                pass

    row = input_data.iloc[0]
    drivers = {
        "CGPA": (row["cgpa"] - 7.0) * 12,
        "Backlogs": -row["backlogs"] * 15,
        "Internships": row["internships"] * 14,
        "Certifications": row["certifications"] * 4,
        "Coding": (row["coding_skills"] - 5) * 8,
        "Communication": (row["communication_skills"] - 5) * 6,
        "Projects": (row["project_count"] - 1) * 7,
    }
    return pd.Series(drivers).sort_values(ascending=False)


def load_executive_marts():
    payload = {}
    for key, file_name in [
        ("branch_summary", "data/marts/branch_kpi_summary.csv"),
        ("company_summary", "data/marts/company_kpi_summary.csv"),
        ("yearly_summary", "data/marts/yearly_placement_trend.csv"),
        ("gender_summary", "data/marts/gender_placement_summary.csv"),
    ]:
        if os.path.exists(file_name):
            payload[key] = pd.read_csv(file_name)
    return payload


SQL_QUERY_LIBRARY = {
    "Top 10 companies hiring": """
SELECT company_name,
       COUNT(*) AS hires,
       ROUND(AVG(package_lpa), 2) AS avg_package
FROM placement_clean_raw
WHERE placement_status = 'Placed'
GROUP BY company_name
ORDER BY hires DESC, avg_package DESC
LIMIT 10;
""",
    "Branch with highest package": """
SELECT branch,
       ROUND(MAX(package_lpa), 2) AS highest_package
FROM placement_clean_raw
WHERE placement_status = 'Placed'
GROUP BY branch
ORDER BY highest_package DESC
LIMIT 1;
""",
    "Average CGPA by company": """
SELECT company_name,
       ROUND(AVG(cgpa), 2) AS avg_cgpa,
       COUNT(*) AS hires
FROM placement_clean_raw
WHERE placement_status = 'Placed'
GROUP BY company_name
ORDER BY avg_cgpa DESC;
""",
    "Students with internship but no placement": """
SELECT student_id,
       branch,
       cgpa,
       internships,
       placement_risk_score,
       student_intervention
FROM placement_clean_raw
WHERE internship_status = 'Yes'
  AND placement_status = 'Not Placed'
ORDER BY placement_risk_score DESC;
""",
    "Placement rate by gender": """
SELECT gender,
       COUNT(*) AS total_students,
       SUM(CASE WHEN placement_status = 'Placed' THEN 1 ELSE 0 END) AS placed_students,
       ROUND(SUM(CASE WHEN placement_status = 'Placed' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS placement_rate_pct
FROM placement_clean_raw
GROUP BY gender;
""",
    "Package distribution": """
SELECT CASE
           WHEN package_lpa = 0 THEN 'Not Placed'
           WHEN package_lpa < 6 THEN '3-6 LPA'
           WHEN package_lpa < 9 THEN '6-9 LPA'
           WHEN package_lpa < 12 THEN '9-12 LPA'
           ELSE '12+ LPA'
       END AS package_band,
       COUNT(*) AS students
FROM placement_clean_raw
GROUP BY package_band
ORDER BY students DESC;
""",
    "Students requiring intervention": """
SELECT student_id,
       branch,
       cgpa,
       backlogs,
       placement_risk_score,
       risk_band,
       student_intervention
FROM placement_clean_raw
WHERE risk_band = 'High'
ORDER BY placement_risk_score DESC;
""",
}


def build_sql_connection(df: pd.DataFrame) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    df.to_sql("placement_clean_raw", conn, index=False, if_exists="replace")
    return conn


def run_sql_query(df: pd.DataFrame, query: str) -> pd.DataFrame:
    conn = build_sql_connection(df)
    try:
        return pd.read_sql_query(query, conn)
    finally:
        conn.close()


st.markdown(
    """
    <div class="hero-wrap">
        <div class="eyebrow"><span class="pulse-dot"></span> Placement intelligence for students, colleges, and recruiters</div>
        <h1 class="hero-title">Student Placement Prediction & Analytics</h1>
        <p class="hero-subtitle">
            Explore placement trends, estimate salary outcomes, and test a student profile against the trained model with a polished,
            interactive dashboard designed for quick decision-making.
        </p>
        <div style="margin-top: 1rem;">
            <span class="feature-chip">⚡ Live placement prediction</span>
            <span class="feature-chip">📈 College analytics dashboard</span>
            <span class="feature-chip">🎯 Salary estimation</span>
            <span class="feature-chip">✨ Animated glass UI</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


models_exist = os.path.exists("models/placement_model.pkl") and os.path.exists("models/salary_model.pkl")
data_exists = os.path.exists("data/cleaned/placement_data_cleaned.csv")

if not (models_exist and data_exists):
    st.warning("Project data and models have not been generated yet.")
    st.info("Click the button below to enrich the raw student dataset and train the predictive models.")

    if st.button("Initialize Project & Train Models", type="primary"):
        with st.spinner("Setting up workspace, cleaning data, and training ML models..."):
            import prepare_project

            prepare_project.setup_project()
            enriched_df = prepare_project.enrich_data()
            cleaned_df = prepare_project.clean_and_process_data(enriched_df)
            prepare_project.train_models(cleaned_df)
            st.success("Project successfully initialized! Reloading app...")
            st.rerun()
    st.stop()


@st.cache_resource
def load_models():
    with open("models/placement_model.pkl", "rb") as f:
        placement_model = pickle.load(f)
    with open("models/salary_model.pkl", "rb") as f:
        salary_model = pickle.load(f)
    return placement_model, salary_model


@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned/placement_data_cleaned.csv")


try:
    placement_clf, salary_reg = load_models()
    df_clean = ensure_runtime_columns(load_data())
except Exception as e:
    st.error(f"Error loading models or dataset: {e}")
    st.stop()


gender_palette = {"Male": "#38bdf8", "Female": "#f472b6"}
degree_order = sorted(df_clean["degree"].dropna().unique().tolist())
branch_order = sorted(df_clean["branch"].dropna().unique().tolist())
executive_marts = load_executive_marts()


st.markdown(
    f"""
    <div class="quick-grid" style="margin: 1rem 0 1.2rem;">
        <div class="quick-item"><div class="k">Records</div><div class="v">{len(df_clean):,}</div></div>
        <div class="quick-item"><div class="k">Placed</div><div class="v">{int((df_clean['placement_status'] == 'Placed').sum()):,}</div></div>
        <div class="quick-item"><div class="k">Branches</div><div class="v">{df_clean['branch'].nunique()}</div></div>
        <div class="quick-item"><div class="k">Top package</div><div class="v">{df_clean['package_lpa'].max():.2f}</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

top_branch_row = (
    df_clean.groupby("branch", as_index=False)
    .agg(placement_rate=("placement_status", lambda s: (s == "Placed").mean() * 100))
    .sort_values("placement_rate", ascending=False)
    .iloc[0]
)
top_recruiter_row = (
    df_clean[df_clean["placement_status"] == "Placed"]
    .groupby("company_name", as_index=False)
    .agg(hires=("student_id", "count"))
    .sort_values("hires", ascending=False)
    .iloc[0]
)

st.markdown(
    f"""
    <div class="glass-card" style="margin-bottom: 1rem;">
        <div style="display:flex; justify-content:space-between; gap:1rem; flex-wrap:wrap; align-items:center;">
            <div>
                <div class="section-title" style="margin-bottom:0.2rem;">Executive Snapshot</div>
                <div class="subtle-copy">A quick view of the strongest placement signals in the current dataset.</div>
            </div>
            <div style="display:flex; gap:0.75rem; flex-wrap:wrap;">
                <div class="quick-item"><div class="k">Top branch</div><div class="v" style="font-size:1rem;">{top_branch_row['branch']}</div></div>
                <div class="quick-item"><div class="k">Best placement rate</div><div class="v" style="font-size:1rem;">{top_branch_row['placement_rate']:.1f}%</div></div>
                <div class="quick-item"><div class="k">Top recruiter</div><div class="v" style="font-size:1rem;">{top_recruiter_row['company_name']}</div></div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


tab_predict, tab_analytics, tab_sql, tab_department = st.tabs(
    ["🔮 Predict Placement & Salary", "📊 Executive Dashboard", "🗃️ SQL Analysis", "🏢 Department Dashboard"]
)


with tab_predict:
    st.markdown("<div class='section-title'>Predict a Student Profile</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtle-copy'>Use this workspace to assess placement likelihood, expected salary, and the strongest improvement levers in a clean decision flow.</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.15, 0.95], gap="large")

    with left:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title">Student Profile</div>
                <div class="subtle-copy">Enter the academic and skill signals used by the model.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        form_col1, form_col2, form_col3 = st.columns(3)
        with form_col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.number_input("Age", min_value=17, max_value=30, value=21)
            degree = st.selectbox("Degree", degree_order)
        with form_col2:
            branch = st.selectbox("Branch / Department", branch_order)
            cgpa = st.slider("CGPA", min_value=1.0, max_value=10.0, value=7.5, step=0.1)
            backlogs = st.number_input("Backlogs", min_value=0, max_value=10, value=0)
        with form_col3:
            internships = st.number_input("Internships", min_value=0, max_value=5, value=0)
            certifications = st.number_input("Certifications", min_value=0, max_value=10, value=0)
            project_count = st.number_input("Projects Completed", min_value=0, max_value=5, value=1)

        skill_col1, skill_col2 = st.columns(2)
        with skill_col1:
            coding_skills = st.slider("Coding Skills Score", min_value=1, max_value=10, value=5)
        with skill_col2:
            communication_skills = st.slider("Communication Skills Score", min_value=1, max_value=10, value=6)

        readiness = score_readiness(
            cgpa, backlogs, internships, certifications, coding_skills, communication_skills, project_count
        )

        st.markdown(
            f"""
            <div class="glass-card" style="margin: 0.3rem 0 0.8rem; padding: 0.9rem 1rem;">
                <div style="display:flex; justify-content:space-between; gap:1rem; align-items:center; flex-wrap:wrap;">
                    <div>
                        <div class="section-title" style="margin-bottom:0.2rem;">Career Readiness</div>
                        <div class="subtle-copy">A compact pre-check based on the current inputs.</div>
                    </div>
                    <div style="font-family:'Space Grotesk', sans-serif; font-size:1.8rem; font-weight:700; color:#174ea6;">
                        {readiness}/100
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        submitted = st.button("Run Prediction Model", type="primary", use_container_width=True)

        if submitted:
            has_internship = 1 if internships > 0 else 0
            has_certification = 1 if certifications > 0 else 0
            total_skill_score = coding_skills + communication_skills
            academic_performance_index = (cgpa * 10) - (backlogs * 5)
            employability_score = (total_skill_score * 0.4) + (cgpa * 0.4) + (internships * 2.0)

            input_data = pd.DataFrame(
                [
                    {
                        "gender": gender,
                        "degree": degree,
                        "branch": branch,
                        "age": age,
                        "cgpa": cgpa,
                        "backlogs": backlogs,
                        "internships": internships,
                        "certifications": certifications,
                        "coding_skills": coding_skills,
                        "communication_skills": communication_skills,
                        "project_count": project_count,
                        "has_internship": has_internship,
                        "has_certification": has_certification,
                        "total_skill_score": total_skill_score,
                        "academic_performance_index": academic_performance_index,
                        "employability_score": employability_score,
                    }
                ]
            )

            placement_prob = float(placement_clf.predict_proba(input_data)[0][1])
            is_placed = int(placement_clf.predict(input_data)[0])
            expected_salary = max(3.0, float(salary_reg.predict(input_data)[0])) if is_placed == 1 else 0.0

            summary_cards = st.columns(2, gap="medium")
            summary_cards[0].markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Placement probability</div>
                    <div class="metric-value">{placement_prob * 100:.1f}%</div>
                    <div class="metric-note">{'Placed' if is_placed == 1 else 'Not placed'}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            summary_cards[1].markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Expected salary</div>
                    <div class="metric-value">{expected_salary:.2f} LPA</div>
                    <div class="metric-note">Forecast only</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            status_text = "Likely to be placed" if is_placed == 1 else "Needs improvement"
            status_color = "#22c55e" if is_placed == 1 else "#ef4444"
            st.markdown(
                f"""
                <div class="glass-card" style="margin-top: 0.8rem; padding: 1rem 1.05rem;">
                    <div style="display:flex; justify-content:space-between; gap:1rem; flex-wrap:wrap; align-items:center;">
                        <div>
                            <div class="section-title" style="margin-bottom:0.2rem;">Outcome</div>
                            <div class="subtle-copy">{status_text}</div>
                        </div>
                        <div style="font-family:'Space Grotesk', sans-serif; font-size:1.9rem; font-weight:700; color:{status_color};">
                            {placement_prob * 100:.2f}%
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            salary_text = f"{expected_salary:.2f} LPA" if is_placed == 1 else "0.00 LPA"
            salary_subtitle = "Forecasted package" if is_placed == 1 else "Shown only for likely placements"
            salary_color = "#1f2937" if is_placed == 1 else "#6b7280"
            st.markdown(
                f"""
                <div class="placed-card" style="margin-top: 0.8rem; padding: 1rem 1.05rem; background: linear-gradient(135deg, rgba(59, 130, 246, 0.20), rgba(15, 23, 42, 0.72)); border-color: rgba(59, 130, 246, 0.32);">
                    <div style="display:flex; justify-content:space-between; gap:1rem; flex-wrap:wrap; align-items:center;">
                        <div>
                            <div class="section-title" style="margin-bottom:0.2rem;">Salary Offer</div>
                            <div class="subtle-copy">{salary_subtitle}</div>
                        </div>
                        <div style="font-family:'Space Grotesk', sans-serif; font-size:1.9rem; font-weight:700; color:{salary_color};">
                            {salary_text}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            risk_score = placement_risk_score(
                cgpa, backlogs, internships, certifications, coding_skills, communication_skills, project_count
            )
            risk_label = risk_band(risk_score)
            actions = recommendation_actions(
                cgpa, backlogs, internships, certifications, coding_skills, communication_skills, project_count
            )
            explanation = explain_profile(placement_clf, input_data, df_clean)

            st.markdown(
                f"""
                <div class="glass-card" style="margin-top: 0.8rem; padding: 1rem 1.05rem;">
                    <div style="display:flex; justify-content:space-between; gap:1rem; flex-wrap:wrap; align-items:center;">
                        <div>
                            <div class="section-title" style="margin-bottom:0.2rem;">Student Risk Score</div>
                            <div class="subtle-copy">Decision support mode for placement intervention.</div>
                        </div>
                        <div style="font-family:'Space Grotesk', sans-serif; font-size:1.9rem; font-weight:700; color:#b45309;">
                            {risk_score:.0f}/100 <span style="font-size:1rem; color:#6b7280;">({risk_label})</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            ex_col1, ex_col2 = st.columns([1.2, 0.8], gap="large")
            with ex_col1:
                st.markdown("#### Explain Why")
                if isinstance(explanation, pd.Series) and not explanation.empty:
                    explanation_df = explanation.reset_index()
                    explanation_df.columns = ["Feature", "Contribution"]
                    explanation_df = explanation_df.head(6)
                    expl_fig = px.bar(
                        explanation_df,
                        x="Contribution",
                        y="Feature",
                        orientation="h",
                        color="Contribution",
                        color_continuous_scale="RdYlGn",
                    )
                    expl_fig.update_layout(coloraxis_showscale=False)
                    st.plotly_chart(make_plotly_theme(expl_fig, height=300), use_container_width=True)
                else:
                    st.info("Explainability output is not available for the current model.")

            with ex_col2:
                st.markdown("#### Recommend How to Improve")
                for tip in actions:
                    st.write(f"- {tip}")

            tips = []
            if backlogs > 0:
                tips.append("Reduce backlogs to improve academic standing.")
            if cgpa < 7.0:
                tips.append("Lift CGPA above 7.0 to strengthen the profile.")
            if internships == 0:
                tips.append("Add at least one internship to improve employability.")
            if project_count < 2:
                tips.append("Build more projects to show practical depth.")
            if certifications == 0:
                tips.append("Earn one certification to improve shortlisting chances.")

            if tips:
                with st.expander("Improvement notes", expanded=False):
                    for tip in tips[:3]:
                        st.write(f"- {tip}")

            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=placement_prob * 100,
                    number={"suffix": "%", "font": {"size": 24, "color": "#1f2937"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#6b7280"},
                        "bar": {"color": "#22c55e" if is_placed == 1 else "#ef4444"},
                        "steps": [
                            {"range": [0, 40], "color": "rgba(239,68,68,0.18)"},
                            {"range": [40, 70], "color": "rgba(250,204,21,0.18)"},
                            {"range": [70, 100], "color": "rgba(34,197,94,0.18)"},
                        ],
                    },
                    title={"text": "Placement Confidence", "font": {"size": 18, "color": "#1f2937"}},
                )
            )
            st.plotly_chart(make_plotly_theme(gauge, height=220), use_container_width=True)

    with right:
        st.markdown("<div class='section-title'>Profile Preview</div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="glass-card">
                <div class="subtle-copy">This panel gives a quick visual read on the student profile before and after prediction.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        preview = pd.DataFrame(
            {
                "Metric": [
                    "Age",
                    "CGPA",
                    "Backlogs",
                    "Internships",
                    "Certifications",
                    "Coding",
                    "Communication",
                    "Projects",
                ],
                "Value": [age, cgpa, backlogs, internships, certifications, coding_skills, communication_skills, project_count],
            }
        )

        radar = px.line_polar(
            preview,
            r="Value",
            theta="Metric",
            line_close=True,
            markers=True,
            range_r=[0, max(10, preview["Value"].max() + 1)],
            color_discrete_sequence=["#174ea6"],
        )
        radar.update_traces(fill="toself", line=dict(color="#174ea6", width=3), marker=dict(size=7, color="#0f8b5f"))
        radar.update_layout(
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=220,
            margin=dict(l=8, r=8, t=8, b=8),
            polar=dict(radialaxis=dict(gridcolor="rgba(31,41,55,0.08)", tickfont=dict(color="#1f2937"))),
            showlegend=False,
        )
        st.plotly_chart(radar, use_container_width=True)

        st.markdown(
            f"""
            <div class="glass-card" style="padding: 0.9rem 1rem;">
                <div style="display:flex; gap:0.6rem; flex-wrap:wrap; font-size:0.9rem; color:#4b5563;">
                    <span><strong>Readiness:</strong> {readiness}/100</span>
                    <span><strong>CGPA:</strong> {cgpa:.1f}</span>
                    <span><strong>Internships:</strong> {internships}</span>
                    <span><strong>Projects:</strong> {project_count}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


with tab_analytics:
    st.markdown("<div class='section-title'>College Analytics Dashboard</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtle-copy'>Filter the dataset and explore placement patterns across branches, internships, CGPA, and recruiters.</div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Dashboard Filters")
        branch_filter = st.multiselect("Branch", branch_order, default=branch_order)
        degree_filter = st.multiselect("Degree", degree_order, default=degree_order)
        min_pkg, max_pkg = float(df_clean["package_lpa"].min()), float(df_clean["package_lpa"].max())
        pkg_range = st.slider("Package range", min_value=min_pkg, max_value=max_pkg, value=(min_pkg, max_pkg))
        status_filter = st.multiselect("Placement status", sorted(df_clean["placement_status"].unique()), default=sorted(df_clean["placement_status"].unique()))
        st.caption("These filters only affect the analytics tab.")

    filtered = df_clean[
        df_clean["branch"].isin(branch_filter)
        & df_clean["degree"].isin(degree_filter)
        & df_clean["placement_status"].isin(status_filter)
        & df_clean["package_lpa"].between(pkg_range[0], pkg_range[1])
    ].copy()

    if filtered.empty:
        st.warning("No rows match the current filters. Expand one of the filters to continue.")
        st.stop()

    placed_only = filtered[filtered["placement_status"] == "Placed"]
    total_studs = len(filtered)
    placed_studs = len(placed_only)
    placement_rate = (placed_studs / total_studs) * 100 if total_studs else 0
    avg_pkg = placed_only["package_lpa"].mean() if not placed_only.empty else 0.0
    highest_pkg = filtered["package_lpa"].max() if not filtered.empty else 0.0
    avg_cgpa = filtered["cgpa"].mean()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric("Total students", f"{total_studs:,}", "Filtered cohort size")
    with col2:
        render_metric("Placement rate", f"{placement_rate:.1f}%", "Placed out of filtered records")
    with col3:
        render_metric("Average package", f"{avg_pkg:.2f} LPA", "Only placed students")
    with col4:
        render_metric("Highest package", f"{highest_pkg:.2f} LPA", "Top offer in filtered data")

    st.markdown("### Placement Snapshot")
    p1, p2 = st.columns([1, 1], gap="large")

    with p1:
        donut = px.pie(
            names=["Placed", "Not Placed"],
            values=[placed_studs, total_studs - placed_studs],
            hole=0.62,
            color_discrete_sequence=["#0f8b5f", "#b91c1c"],
        )
        donut.update_layout(
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=300,
            margin=dict(l=20, r=20, t=30, b=20),
            showlegend=True,
        )
        st.plotly_chart(donut, use_container_width=True)

    with p2:
        summary_box = st.container()
        with summary_box:
            st.markdown(
                f"""
                <div class="glass-card">
                    <div class="section-title">Insight Summary</div>
                    <div class="insight-box">Average CGPA in the filtered set: <strong>{avg_cgpa:.2f}</strong></div>
                    <div class="insight-box">Placed count: <strong>{placed_studs}</strong></div>
                    <div class="insight-box">Not placed count: <strong>{total_studs - placed_studs}</strong></div>
                    <div class="insight-box">Placement rate: <strong>{placement_rate:.1f}%</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    g1_col1, g1_col2 = st.columns(2)

    with g1_col1:
        st.markdown("#### Placement Rate by Department")
        dept_placements = (
            filtered.groupby("branch")["placement_status"]
            .value_counts(normalize=True)
            .unstack(fill_value=0)
            .reindex(columns=["Placed", "Not Placed"], fill_value=0)
            * 100
        )
        dept_placements = dept_placements.sort_values(by="Placed", ascending=True)

        fig, ax = plt.subplots(figsize=(5.4, 3.4), dpi=120)
        fig.patch.set_facecolor("#f8f4ee")
        ax.set_facecolor("#ffffff")
        dept_placements["Placed"].plot(kind="barh", color="#174ea6", ax=ax)
        style_chart_ax(ax, "Hiring Percentage per Department", "Placement Rate (%)", "")
        plt.tight_layout()
        st.pyplot(fig)

    with g1_col2:
        st.markdown("#### Salary Package Distribution")
        fig, ax = plt.subplots(figsize=(5.4, 3.4), dpi=120)
        fig.patch.set_facecolor("#f8f4ee")
        ax.set_facecolor("#ffffff")
        if not placed_only.empty:
            sns.histplot(placed_only["package_lpa"], kde=True, color="#0f8b5f", bins=15, ax=ax)
            style_chart_ax(ax, "Salary Offers Histogram (LPA)", "Package (LPA)", "Count")
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No placed students are available in the current filter set.")

    g2_col1, g2_col2 = st.columns(2)

    with g2_col1:
        st.markdown("#### Internship Impact")
        internship_rates = (
            filtered.assign(is_placed=(filtered["placement_status"] == "Placed").astype(int))
            .groupby("internship_status", as_index=False)["is_placed"]
            .mean()
        )
        internship_rates["is_placed"] *= 100

        fig, ax = plt.subplots(figsize=(5.4, 3.4), dpi=120)
        fig.patch.set_facecolor("#f8f4ee")
        ax.set_facecolor("#ffffff")
        sns.barplot(
            x="internship_status",
            y="is_placed",
            data=internship_rates,
            errorbar=None,
            palette="coolwarm",
            ax=ax,
        )
        style_chart_ax(ax, "Placement Rate: With vs Without Internships", "Has Completed Internship", "Placement Percentage (%)")
        ax.set_ylim(0, 100)
        plt.tight_layout()
        st.pyplot(fig)

    with g2_col2:
        st.markdown("#### CGPA by Placement Status")
        fig, ax = plt.subplots(figsize=(5.4, 3.4), dpi=120)
        fig.patch.set_facecolor("#f8f4ee")
        ax.set_facecolor("#ffffff")
        sns.boxplot(x="placement_status", y="cgpa", data=filtered, palette="Set2", ax=ax)
        style_chart_ax(ax, "CGPA Distributions", "Placement Status", "CGPA")
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("#### Top Recruiting Partners")
    rec_sum = (
        placed_only.groupby("company_name")
        .size()
        .reset_index(name="Hires")
        .sort_values("Hires", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(11.0, 3.6), dpi=120)
    fig.patch.set_facecolor("#f8f4ee")
    ax.set_facecolor("#ffffff")
    if not rec_sum.empty:
        sns.barplot(x="Hires", y="company_name", data=rec_sum, palette="Blues_r", ax=ax)
        style_chart_ax(ax, "Hires per Company Partner", "Count of Placements", "")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("No company hiring data is available because there are no placed students.")

    st.markdown("#### Branch and Salary Summary")
    branch_summary = (
        placed_only.groupby("branch", as_index=False)
        .agg(avg_package=("package_lpa", "mean"), avg_cgpa=("cgpa", "mean"), hires=("student_id", "count"))
        .sort_values("avg_package", ascending=False)
    )
    if not branch_summary.empty:
        branch_fig = px.bar(
            branch_summary,
            x="branch",
            y="avg_package",
            color="avg_package",
            text_auto=".2f",
            color_continuous_scale="Turbo",
        )
        st.plotly_chart(make_plotly_theme(branch_fig, height=280), use_container_width=True)


with tab_sql:
    st.markdown("<div class='section-title'>SQL Analysis</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtle-copy'>Select a query, run it against the cleaned dataset, and inspect the result like a real analytics workspace.</div>",
        unsafe_allow_html=True,
    )

    sql_col1, sql_col2 = st.columns([1.05, 0.95], gap="large")
    with sql_col1:
        st.markdown("#### SQL Query Runner")
        selected_query = st.selectbox("Choose a query", list(SQL_QUERY_LIBRARY.keys()))
        editable_query = st.text_area("Edit query", value=SQL_QUERY_LIBRARY[selected_query], height=220)
        run_query = st.button("Run SQL Query", type="primary", use_container_width=True)

    with sql_col2:
        st.markdown("#### What this runner shows")
        st.markdown(
            """
            <div class="glass-card">
                <div class="insight-box">Top 10 companies hiring</div>
                <div class="insight-box">Branch with highest package</div>
                <div class="insight-box">Average CGPA by company</div>
                <div class="insight-box">Students with internship but no placement</div>
                <div class="insight-box">Placement rate by gender</div>
                <div class="insight-box">Package distribution</div>
                <div class="insight-box">Students requiring intervention</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if run_query:
        try:
            result_df = run_sql_query(df_clean, editable_query)
            st.markdown("#### Query Result")
            st.dataframe(result_df, use_container_width=True, hide_index=True)

            if not result_df.empty:
                numeric_cols = result_df.select_dtypes(include="number").columns.tolist()
                if numeric_cols:
                    chart_col = numeric_cols[0]
                    chart_fig = px.bar(result_df.head(12), x=result_df.columns[0], y=chart_col, color=chart_col)
                    st.markdown("#### Quick Visual")
                    st.plotly_chart(make_plotly_theme(chart_fig, height=320), use_container_width=True)
        except Exception as exc:
            st.error(f"SQL query failed: {exc}")
    else:
        st.markdown("#### Preview")
        st.code(editable_query, language="sql")

        preview_df = run_sql_query(df_clean, SQL_QUERY_LIBRARY[selected_query])
        st.dataframe(preview_df.head(10), use_container_width=True, hide_index=True)


with tab_department:
    st.markdown("<div class='section-title'>Department Dashboard</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtle-copy'>A placement officer view for branch-level performance, hiring trends, and at-risk students.</div>",
        unsafe_allow_html=True,
    )

    dept_branch = st.selectbox("Select Department / Branch", branch_order, index=0)
    dept_df = df_clean[df_clean["branch"] == dept_branch].copy()
    dept_placed = dept_df[dept_df["placement_status"] == "Placed"]

    d1, d2, d3, d4 = st.columns(4)
    with d1:
        render_metric("Placement %", f"{(len(dept_placed) / len(dept_df) * 100 if len(dept_df) else 0):.1f}%", "Department conversion")
    with d2:
        render_metric("Average Package", f"{dept_placed['package_lpa'].mean():.2f} LPA" if not dept_placed.empty else "0.00 LPA", "Placed students only")
    with d3:
        render_metric("Recruiter Count", f"{dept_placed['company_name'].nunique()}", "Unique companies")
    with d4:
        render_metric("Unplaced Students", f"{len(dept_df) - len(dept_placed):,}", "Need follow-up")

    dept_left, dept_right = st.columns(2, gap="large")

    with dept_left:
        st.markdown("#### Branch Risk Mix")
        risk_mix = (
            dept_df.groupby("risk_band")["student_id"].count().reindex(["Low", "Medium", "High"], fill_value=0).reset_index()
        )
        risk_mix.columns = ["Risk Band", "Students"]
        risk_fig = px.pie(risk_mix, names="Risk Band", values="Students", hole=0.6, color_discrete_sequence=["#0f8b5f", "#f59e0b", "#ef4444"])
        st.plotly_chart(make_plotly_theme(risk_fig, height=320), use_container_width=True)

    with dept_right:
        st.markdown("#### Hiring Trend")
        dept_trend = (
            dept_df.groupby("placement_year", as_index=False)
            .agg(placed=("placement_status", lambda s: (s == "Placed").sum()), total=("student_id", "count"))
            .sort_values("placement_year")
        )
        trend_fig = px.line(dept_trend, x="placement_year", y="placed", markers=True, title="Placed Students by Year")
        trend_fig.add_bar(x=dept_trend["placement_year"], y=dept_trend["total"], name="Total Students", opacity=0.35)
        st.plotly_chart(make_plotly_theme(trend_fig, height=320), use_container_width=True)

    st.markdown("#### Department Insights")
    dept_metrics = pd.DataFrame(
        {
            "Metric": ["Average CGPA", "Average Employability", "Average Risk Score", "Internship Rate"],
            "Value": [
                dept_df["cgpa"].mean(),
                dept_df["employability_score"].mean(),
                dept_df["placement_risk_score"].mean(),
                dept_df["has_internship"].mean() * 100,
            ],
        }
    )
    dept_bar = px.bar(dept_metrics, x="Metric", y="Value", color="Value", text="Value", color_continuous_scale="Turbo")
    st.plotly_chart(make_plotly_theme(dept_bar, height=300), use_container_width=True)

    st.markdown("### Executive Dashboard")
    exec1, exec2 = st.columns(2, gap="large")

    with exec1:
        st.markdown("#### Branch Risk Heatmap")
        risk_heat = (
            filtered.pivot_table(
                index="branch",
                columns="risk_band",
                values="student_id",
                aggfunc="count",
                fill_value=0,
            )
            .reindex(columns=["Low", "Medium", "High"], fill_value=0)
        )
        if not risk_heat.empty:
            fig, ax = plt.subplots(figsize=(6.2, 3.5), dpi=120)
            fig.patch.set_facecolor("#f8f4ee")
            ax.set_facecolor("#ffffff")
            sns.heatmap(risk_heat, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
            style_chart_ax(ax, "Risk Distribution by Branch", "", "Branch")
            plt.tight_layout()
            st.pyplot(fig)

    with exec2:
        st.markdown("#### Placement Funnel")
        funnel_df = pd.DataFrame(
            {
                "Stage": ["All Students", "Eligible", "Internship Holders", "Placed"],
                "Count": [
                    len(filtered),
                    int(filtered["eligible_for_shortlist"].sum()),
                    int(filtered["has_internship"].sum()),
                    int((filtered["placement_status"] == "Placed").sum()),
                ],
            }
        )
        funnel_fig = px.funnel(funnel_df, x="Count", y="Stage", color="Stage")
        st.plotly_chart(make_plotly_theme(funnel_fig, height=310), use_container_width=True)

    exec3, exec4 = st.columns(2, gap="large")

    with exec3:
        st.markdown("#### Company Analytics")
        comp_summary = (
            placed_only.groupby("company_name", as_index=False)
            .agg(
                hires=("student_id", "count"),
                avg_package=("package_lpa", "mean"),
                avg_cgpa=("cgpa", "mean"),
                avg_employability=("employability_score", "mean"),
            )
            .sort_values("hires", ascending=False)
        )
        if not comp_summary.empty:
            comp_fig = px.bar(
                comp_summary.head(10),
                x="company_name",
                y="hires",
                color="avg_package",
                text="hires",
                color_continuous_scale="Viridis",
            )
            st.plotly_chart(make_plotly_theme(comp_fig, height=310), use_container_width=True)

    with exec4:
        st.markdown("#### Time Series Trend")
        trend_df = (
            filtered.groupby("placement_year", as_index=False)
            .agg(
                placed=("placement_status", lambda s: (s == "Placed").sum()),
                total=("student_id", "count"),
                avg_package=("package_lpa", lambda s: s[s > 0].mean() if (s > 0).any() else 0),
            )
            .sort_values("placement_year")
        )
        if not trend_df.empty:
            trend_fig = px.line(
                trend_df,
                x="placement_year",
                y="placed",
                markers=True,
                title="Placed Students by Year",
            )
            trend_fig.add_bar(x=trend_df["placement_year"], y=trend_df["total"], name="Total Students", opacity=0.35)
            st.plotly_chart(make_plotly_theme(trend_fig, height=310), use_container_width=True)

    st.markdown("#### Executive KPIs")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_metric("Internship conversion", f"{filtered['has_internship'].mean() * 100:.1f}%", "Students with internships")
    with k2:
        render_metric("Eligible students", f"{int(filtered['eligible_for_shortlist'].sum()):,}", "CGPA >= 6.5 and no backlogs")
    with k3:
        render_metric("At-risk students", f"{int((filtered['risk_band'] == 'High').sum()):,}", "High intervention priority")
    with k4:
        render_metric("Avg. risk score", f"{filtered['placement_risk_score'].mean():.1f}", "Lower is better")
