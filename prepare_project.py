import os
import shutil
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def setup_project():
    print("=== Step 1: Setting up Project Directory Structure ===")
    dirs = ['data/raw', 'data/cleaned', 'sql', 'notebooks', 'models']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Created directory: {d}")
    
    raw_source = 'Indian_Student_Placement_Dataset_2025-selected-columns.csv'
    raw_dest = 'data/raw/Indian_Student_Placement_Dataset_2025-selected-columns.csv'
    
    if os.path.exists(raw_source):
        shutil.copy(raw_source, raw_dest)
        print(f"Copied raw dataset to: {raw_dest}")
    else:
        print(f"Warning: {raw_source} not found in workspace root. Please ensure it is present.")

def enrich_data():
    print("\n=== Step 2: Enriching Raw Data with Target & Missing Features ===")
    raw_path = 'data/raw/Indian_Student_Placement_Dataset_2025-selected-columns.csv'
    if not os.path.exists(raw_path):
        print("Error: Raw dataset not found. Cannot proceed with enrichment.")
        return None
    
    df = pd.read_csv(raw_path)
    print(f"Loaded raw data with shape: {df.shape}")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    n_records = len(df)
    
    # 1. Generate communication_skills (1 to 10)
    # Norm distribution around 6.5, clipped
    df['communication_skills'] = np.clip(np.round(np.random.normal(6.5, 1.5, n_records)), 1, 10).astype(int)
    
    # 2. Generate project_count (0 to 5), correlated with coding_skills
    # High coding skill = higher likelihood of more projects
    project_probs = []
    for coding_val in df['coding_skills']:
        # lambda parameter for poisson distribution
        lam = max(0.5, coding_val / 2.0)
        proj_count = np.clip(np.random.poisson(lam), 0, 5)
        project_probs.append(proj_count)
    df['project_count'] = project_probs
    
    # 3. Create Yes/No fields for internships & certifications to demonstrate conversion cleaning
    df['internship_status'] = np.where(df['internships'] > 0, 'Yes', 'No')
    df['certification_status'] = np.where(df['certifications'] > 0, 'Yes', 'No')
    
    # 4. Generate Placement_Status using logistic probability model
    # Weights for variables
    log_odds = (
        1.6 * (df['cgpa'] - 6.0) +
        0.5 * df['coding_skills'] +
        0.4 * df['communication_skills'] +
        0.8 * df['internships'] +
        0.6 * df['project_count'] -
        1.2 * df['backlogs'] -
        3.5
    )
    prob = 1 / (1 + np.exp(-log_odds))
    placement_status = np.random.binomial(1, prob)
    df['placement_status'] = np.where(placement_status == 1, 'Placed', 'Not Placed')
    
    # 5. Generate Package_LPA for placed students
    # Unplaced students get 0.0 package
    packages = []
    for idx, row in df.iterrows():
        if row['placement_status'] == 'Placed':
            base = 3.5
            val = (
                base +
                0.9 * (row['cgpa'] - 6.0) +
                0.45 * row['coding_skills'] +
                0.7 * row['internships'] +
                0.5 * row['project_count'] +
                0.3 * row['certifications'] +
                np.random.normal(0, 0.6)
            )
            packages.append(max(3.0, round(val, 2)))
        else:
            packages.append(0.0)
    df['package_lpa'] = packages
    
    # 6. Generate Company Name based on package
    companies = []
    high_tier = ['Google', 'Microsoft', 'Amazon', 'Adobe', 'Meta']
    mid_tier = ['Accenture', 'Capgemini', 'Cognizant', 'Infosys']
    low_tier = ['TCS', 'Wipro', 'Tech Mahindra']
    
    for idx, row in df.iterrows():
        if row['placement_status'] == 'Placed':
            pkg = row['package_lpa']
            if pkg >= 12.0:
                companies.append(np.random.choice(high_tier))
            elif pkg >= 7.5:
                companies.append(np.random.choice(mid_tier))
            else:
                companies.append(np.random.choice(low_tier))
        else:
            companies.append('Unplaced')
    df['company_name'] = companies
    
    print(f"Data enriched successfully. Shape: {df.shape}")
    print(df['placement_status'].value_counts())
    return df

def clean_and_process_data(df):
    print("\n=== Step 3: Cleaning & Processing Data ===")
    if df is None:
        return
    
    # Remove duplicate records
    duplicates_count = df.duplicated().sum()
    if duplicates_count > 0:
        df = df.drop_duplicates()
        print(f"Removed {duplicates_count} duplicate records.")
    else:
        print("No duplicate records found.")
        
    # Handle missing values (if any)
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        df = df.dropna()  # Simple clean
        print(f"Dropped rows with missing values. Remaining records: {len(df)}")
    else:
        print("No missing values found.")
        
    # Standardize branch/department names
    print("Standardizing branch names...")
    branch_map = {
        'CS': 'Computer Science',
        'IT': 'Information Technology',
        'Electrical': 'Electrical Engineering',
        'Mechanical': 'Mechanical Engineering',
        'DS': 'Data Science',
        'AI': 'Artificial Intelligence'
    }
    df['branch'] = df['branch'].replace(branch_map)
    print("Unique branches after standardization:", df['branch'].unique())
    
    # Convert Yes/No values into numerical form (internship_status -> has_internship, certification_status -> has_certification)
    df['has_internship'] = df['internship_status'].map({'Yes': 1, 'No': 0})
    df['has_certification'] = df['certification_status'].map({'Yes': 1, 'No': 0})
    
    # Create derived features
    print("Engineering derived features...")
    df['total_skill_score'] = df['coding_skills'] + df['communication_skills']
    df['academic_performance_index'] = (df['cgpa'] * 10) - (df['backlogs'] * 5)
    df['employability_score'] = (df['total_skill_score'] * 0.4) + (df['cgpa'] * 0.4) + (df['internships'] * 2.0)
    
    cleaned_path = 'data/cleaned/placement_data_cleaned.csv'
    df.to_csv(cleaned_path, index=False)
    print(f"Cleaned dataset saved to: {cleaned_path}")
    return df

def train_models(df):
    print("\n=== Step 4: Training & Evaluating ML Models ===")
    if df is None:
        return
    
    # Define features and targets
    categorical_cols = ['gender', 'degree', 'branch']
    numerical_cols = [
        'age', 'cgpa', 'backlogs', 'internships', 'certifications', 
        'coding_skills', 'communication_skills', 'project_count', 
        'has_internship', 'has_certification', 'total_skill_score', 
        'academic_performance_index', 'employability_score'
    ]
    
    # ------------------
    # Model 1: Placement Classification
    # ------------------
    print("\n--- Training Placement Predictor (Classification) ---")
    X = df[categorical_cols + numerical_cols]
    y_clf = (df['placement_status'] == 'Placed').astype(int)  # 1 = Placed, 0 = Not Placed
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_clf, test_size=0.2, random_state=42, stratify=y_clf)
    
    # Define Column Transformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough'
    )
    
    # Evaluate 3 algorithms: Logistic Regression, Decision Tree, Random Forest
    models_to_test = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    best_name = None
    best_acc = 0.0
    best_pipeline = None
    
    for name, clf in models_to_test.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Model: {name} | Test Accuracy: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_pipeline = pipeline
            
    print(f"==> Best Placement Model Selected: {best_name} with Accuracy: {best_acc:.4f}")
    
    # Save best classification pipeline
    model_path = 'models/placement_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(best_pipeline, f)
    print(f"Saved best classification model to {model_path}")
    
    # ------------------
    # Model 2: Salary Regressor (only for Placed students)
    # ------------------
    print("\n--- Training Salary Forecast Model (Regression) ---")
    df_placed = df[df['placement_status'] == 'Placed'].copy()
    print(f"Training regression model on {len(df_placed)} placed students.")
    
    X_reg = df_placed[categorical_cols + numerical_cols]
    y_reg = df_placed['package_lpa']
    
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    
    reg_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    reg_pipeline.fit(X_train_r, y_train_r)
    train_r2 = reg_pipeline.score(X_train_r, y_train_r)
    test_r2 = reg_pipeline.score(X_test_r, y_test_r)
    print(f"Linear Regression R2 on Train: {train_r2:.4f}")
    print(f"Linear Regression R2 on Test: {test_r2:.4f}")
    
    # Save regression pipeline
    reg_model_path = 'models/salary_model.pkl'
    with open(reg_model_path, 'wb') as f:
        pickle.dump(reg_pipeline, f)
    print(f"Saved salary regression model to {reg_model_path}")
    
    print("\n=== Project setup and modeling complete! ===")

if __name__ == "__main__":
    setup_project()
    enriched_df = enrich_data()
    cleaned_df = clean_and_process_data(enriched_df)
    train_models(cleaned_df)
