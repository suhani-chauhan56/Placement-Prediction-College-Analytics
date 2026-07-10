import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Placement Analytics & Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium glassmorphism and modern UI design
st.markdown("""
<style>
    /* Main container styling */
    .reportview-container {
        background: #0f172a;
    }
    
    /* Title styling */
    .main-title {
        font-family: 'Inter', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 5px 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Gradient prediction cards */
    .placed-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(5, 150, 105, 0.1));
        border: 2px solid #10b981;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-top: 1rem;
        box-shadow: 0 10px 15px -3px rgba(16, 185, 129, 0.2);
    }
    
    .unplaced-card {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(220, 38, 38, 0.1));
        border: 2px solid #ef4444;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-top: 1rem;
        box-shadow: 0 10px 15px -3px rgba(239, 68, 68, 0.2);
    }
    
    /* Inputs */
    div[data-baseweb="select"] > div {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
    }
    
    .stNumberInput input {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1e293b;
        border-radius: 8px 8px 0px 0px;
        color: #94a3b8;
        font-weight: 600;
        font-size: 1rem;
        padding: 10px 20px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown("<h1 class='main-title'>🎓 Student Placement Prediction & Analytics</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>A machine-learning-powered college decision tool for freshers, placement cells, and recruiters.</p>", unsafe_allow_html=True)

# Check if data and models are initialized
models_exist = os.path.exists('models/placement_model.pkl') and os.path.exists('models/salary_model.pkl')
data_exists = os.path.exists('data/cleaned/placement_data_cleaned.csv')

if not (models_exist and data_exists):
    st.warning("⚠️ Project data and models have not been generated yet.")
    st.info("Click the button below to enrich the raw student dataset and train the predictive models.")
    
    if st.button("🚀 Initialize Project & Train Models", type="primary"):
        with st.spinner("Setting up workspace, cleaning data, and training ML models..."):
            import prepare_project
            prepare_project.setup_project()
            enriched_df = prepare_project.enrich_data()
            cleaned_df = prepare_project.clean_and_process_data(enriched_df)
            prepare_project.train_models(cleaned_df)
            st.success("🎉 Project successfully initialized! Reloading app...")
            st.rerun()
    st.stop()

# Helper function to load models
@st.cache_resource
def load_models():
    with open('models/placement_model.pkl', 'rb') as f:
        placement_model = pickle.load(f)
    with open('models/salary_model.pkl', 'rb') as f:
        salary_model = pickle.load(f)
    return placement_model, salary_model

# Helper function to load dataset
@st.cache_data
def load_data():
    return pd.read_csv('data/cleaned/placement_data_cleaned.csv')

try:
    placement_clf, salary_reg = load_models()
    df_clean = load_data()
except Exception as e:
    st.error(f"Error loading models or dataset: {e}")
    st.stop()

# Define tabs
tab_predict, tab_analytics = st.tabs(["🔮 Predict Placement & Salary", "📊 College Analytics Dashboard"])

# -----------------------------------
# TAB 1: PREDICTION
# -----------------------------------
with tab_predict:
    st.markdown("### Enter Student Details for Prediction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=17, max_value=30, value=21)
        degree = st.selectbox("Degree", sorted(df_clean['degree'].unique()))
        branch = st.selectbox("Branch / Department", sorted(df_clean['branch'].unique()))
        
    with col2:
        cgpa = st.slider("CGPA", min_value=1.0, max_value=10.0, value=7.5, step=0.1)
        backlogs = st.number_input("Number of Backlogs", min_value=0, max_value=10, value=0)
        internships = st.number_input("Number of Internships", min_value=0, max_value=5, value=0)
        certifications = st.number_input("Number of Certifications", min_value=0, max_value=10, value=0)
        
    with col3:
        coding_skills = st.slider("Coding Skills Score (1-10)", min_value=1, max_value=10, value=5)
        communication_skills = st.slider("Communication Skills Score (1-10)", min_value=1, max_value=10, value=6)
        project_count = st.number_input("Number of Projects Completed", min_value=0, max_value=5, value=1)
        
    if st.button("🔮 Run Prediction Model", type="primary", use_container_width=True):
        # Calculate derived features
        has_internship = 1 if internships > 0 else 0
        has_certification = 1 if certifications > 0 else 0
        total_skill_score = coding_skills + communication_skills
        academic_performance_index = (cgpa * 10) - (backlogs * 5)
        employability_score = (total_skill_score * 0.4) + (cgpa * 0.4) + (internships * 2.0)
        
        # Build features DataFrame in correct column order
        input_data = pd.DataFrame([{
            'gender': gender,
            'degree': degree,
            'branch': branch,
            'age': age,
            'cgpa': cgpa,
            'backlogs': backlogs,
            'internships': internships,
            'certifications': certifications,
            'coding_skills': coding_skills,
            'communication_skills': communication_skills,
            'project_count': project_count,
            'has_internship': has_internship,
            'has_certification': has_certification,
            'total_skill_score': total_skill_score,
            'academic_performance_index': academic_performance_index,
            'employability_score': employability_score
        }])
        
        # Run classification
        placement_prob = placement_clf.predict_proba(input_data)[0][1]
        is_placed = placement_clf.predict(input_data)[0]
        
        st.markdown("<hr style='border-color: rgba(255,255,255,0.05);'>", unsafe_allow_html=True)
        
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            if is_placed == 1:
                st.markdown(f"""
                <div class="placed-card">
                    <h2 style="color: #10b981; margin-top:0;">🎉 Likely to be Placed</h2>
                    <p style="font-size: 1.1rem; color: #d1fae5;">Placement Probability: <b>{placement_prob * 100:.2f}%</b></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="unplaced-card">
                    <h2 style="color: #ef4444; margin-top:0;">❌ Unlikely to be Placed</h2>
                    <p style="font-size: 1.1rem; color: #fee2e2;">Placement Probability: <b>{placement_prob * 100:.2f}%</b></p>
                    <p style="font-size: 0.9rem; color: #fca5a5; margin-top: 10px;">Tips: Try lowering backlogs, improving CGPA, or completing internships.</p>
                </div>
                """, unsafe_allow_html=True)
                
        with res_col2:
            if is_placed == 1:
                # Predict package
                expected_salary = salary_reg.predict(input_data)[0]
                # Ensure salary scales nicely
                expected_salary = max(3.0, expected_salary)
                
                st.markdown(f"""
                <div class="placed-card" style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(37, 99, 235, 0.1)); border-color: #3b82f6; box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.2);">
                    <h2 style="color: #3b82f6; margin-top:0;">💼 Expected Salary Offer</h2>
                    <p style="font-size: 2.2rem; font-weight: bold; color: #ffffff; margin: 10px 0;">{expected_salary:.2f} LPA</p>
                    <p style="font-size: 0.9rem; color: #dbeafe;">Forecasted based on academic details and skill scores.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="unplaced-card" style="background: linear-gradient(135deg, rgba(148, 163, 184, 0.2), rgba(100, 116, 139, 0.1)); border-color: #94a3b8; box-shadow: none;">
                    <h2 style="color: #94a3b8; margin-top:0;">💼 Expected Salary Offer</h2>
                    <p style="font-size: 2.2rem; font-weight: bold; color: #cbd5e1; margin: 10px 0;">0.00 LPA</p>
                    <p style="font-size: 0.9rem; color: #cbd5e1;">Salary forecasts are only calculated for placed profiles.</p>
                </div>
                """, unsafe_allow_html=True)

# -----------------------------------
# TAB 2: ANALYTICS DASHBOARD
# -----------------------------------
with tab_analytics:
    # Top metrics row
    total_studs = len(df_clean)
    placed_studs = len(df_clean[df_clean['placement_status'] == 'Placed'])
    placement_rate = (placed_studs / total_studs) * 100
    avg_pkg = df_clean[df_clean['placement_status'] == 'Placed']['package_lpa'].mean()
    highest_pkg = df_clean['package_lpa'].max()
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Students</div>
            <div class="metric-value">{total_studs}</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Overall Placement Rate</div>
            <div class="metric-value">{placement_rate:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Average Package</div>
            <div class="metric-value">{avg_pkg:.2f} LPA</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Highest Package</div>
            <div class="metric-value">{highest_pkg:.2f} LPA</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<hr style='border-color: rgba(255,255,255,0.05);'>", unsafe_allow_html=True)
    
    # Graphs Row 1
    g1_col1, g1_col2 = st.columns(2)
    
    with g1_col1:
        st.markdown("#### 🏢 Placement Rate by Department")
        dept_placements = df_clean.groupby('branch')['placement_status'].value_counts(normalize=True).unstack() * 100
        dept_placements = dept_placements.sort_values(by='Placed', ascending=True)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#1e293b')
        
        dept_placements['Placed'].plot(kind='barh', color='#3b82f6', ax=ax)
        ax.set_title("Hiring Percentage per Department", color='white', fontsize=12)
        ax.set_xlabel("Placement Rate (%)", color='white')
        ax.set_ylabel("", color='white')
        ax.tick_params(colors='white')
        ax.grid(color=(1, 1, 1, 0.1), linestyle='--')
        plt.tight_layout()
        st.pyplot(fig)
        
    with g1_col2:
        st.markdown("#### 📈 Salary Package Distribution (Placed Students)")
        placed_only = df_clean[df_clean['placement_status'] == 'Placed']
        
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#1e293b')
        
        sns.histplot(placed_only['package_lpa'], kde=True, color='#10b981', bins=15, ax=ax)
        ax.set_title("Salary Offers Histogram (LPA)", color='white', fontsize=12)
        ax.set_xlabel("Package (LPA)", color='white')
        ax.set_ylabel("Count", color='white')
        ax.tick_params(colors='white')
        ax.grid(color=(1, 1, 1, 0.1), linestyle='--')
        plt.tight_layout()
        st.pyplot(fig)
        
    # Graphs Row 2
    g2_col1, g2_col2 = st.columns(2)
    
    with g2_col1:
        st.markdown("#### 💼 Internship Impact on Placements")
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#1e293b')
        
        sns.barplot(x='internship_status', y=(df_clean['placement_status'] == 'Placed').astype(int) * 100, data=df_clean, errorbar=None, palette='coolwarm', ax=ax)
        ax.set_title("Placement Rate: With vs Without Internships", color='white', fontsize=12)
        ax.set_ylabel("Placement Percentage (%)", color='white')
        ax.set_xlabel("Has Completed Internship", color='white')
        ax.tick_params(colors='white')
        ax.set_ylim(0, 100)
        ax.grid(color=(1, 1, 1, 0.1), linestyle='--')
        plt.tight_layout()
        st.pyplot(fig)
        
    with g2_col2:
        st.markdown("#### 🎓 CGPA Boxplot by Placement Status")
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#1e293b')
        
        sns.boxplot(x='placement_status', y='cgpa', data=df_clean, palette='Set2', ax=ax)
        ax.set_title("CGPA Distributions", color='white', fontsize=12)
        ax.set_xlabel("Placement Status", color='white')
        ax.set_ylabel("CGPA", color='white')
        ax.tick_params(colors='white')
        ax.grid(color=(1, 1, 1, 0.1), linestyle='--')
        plt.tight_layout()
        st.pyplot(fig)
        
    # Company recruitments row
    st.markdown("#### 🏢 Top Recruiting Partners & Hires")
    rec_sum = placed_only.groupby('company_name').size().reset_index(name='Hires').sort_values('Hires', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 3))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')
    
    sns.barplot(x='Hires', y='company_name', data=rec_sum, palette='Blues_r', ax=ax)
    ax.set_title("Hires per Company Partner", color='white', fontsize=12)
    ax.set_xlabel("Count of Placements", color='white')
    ax.set_ylabel("", color='white')
    ax.tick_params(colors='white')
    ax.grid(color=(1, 1, 1, 0.1), linestyle='--')
    plt.tight_layout()
    st.pyplot(fig)
