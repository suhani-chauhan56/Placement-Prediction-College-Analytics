<div align="center">

# 🎓 Placement-Prediction-College-Analytics

### 🚀 Placement analytics, SQL insights, dashboards, and ML prediction in one project

**Python** -> **MySQL** -> **Power BI** -> **Machine Learning**

![Python](https://img.shields.io/badge/Python-Pandas%20%7C%20Sklearn-3776AB?style=for-the-badge&logo=python&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-MySQL-4479A1?style=for-the-badge&logo=mysql&logoColor=white)
![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-F2C811?style=for-the-badge&logo=powerbi&logoColor=black)
![Status](https://img.shields.io/badge/Status-Partially%20Complete-yellow?style=for-the-badge)

### 🌟 [Live Demo - Try the Prediction App](https://placement-prediction-college-analytics-j7evnb2vmmmjxmnc6g4fdq.streamlit.app/)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://placement-prediction-college-analytics-j7evnb2vmmmjxmnc6g4fdq.streamlit.app/)

</div>

---

Every placement season, colleges sit on a goldmine of data: CGPA, internships, certifications, coding scores, communication scores, and placement outcomes. This project turns that data into a portfolio-ready analytics product instead of only a prediction script.

---

## 📌 Business Problem

The placement cell needs a faster way to answer:

- Which branches are performing best?
- Which recruiters hire the most students?
- Which students need early intervention?
- How do internships and skills affect placement outcomes?
- What is the expected salary for a student profile?

---

## 🧭 Architecture

```text
Raw Student Dataset
    |
    v
Python ETL
    |
    v
Cleaned Dataset
    |
    v
SQL Analysis Layer
    |
    v
EDA + Power BI Dashboard
    |
    v
Machine Learning Models
    |
    v
Placement Prediction + Salary Forecast
```

---

## 🛠️ Tech Stack

| Layer | Tool | Purpose |
|---|---|---|
| Data Cleaning | Python (Pandas) | Deduplication, missing values, feature engineering |
| Data Modeling & Analysis | MySQL (SQL) | Normalized schema, joins, views, analysis queries |
| Visualization | Power BI | KPI cards, dashboard reporting, slicers |
| Machine Learning | Scikit-learn | Placement classification + salary regression |
| Deployment | Streamlit | Live interactive prediction app |

---

## 📊 Dataset

- **Size:** 12,000 student records
- **Base columns:** `student_id`, `gender`, `age`, `degree`, `branch`, `cgpa`, `backlogs`, `internships`, `certifications`, `coding_skills`
- **Engineered columns:** `communication_skills`, `project_count`, `internship_status`, `certification_status`, `placement_status`, `package_lpa`, `company_name`, `total_skill_score`, `academic_performance_index`, `employability_score`
- **Outcome split:** 10,937 placed / 1,063 not placed

---

## 📸 Screenshots

Use the existing assets already in the repo:

- `Screen Shots (sql,visualizations)/`
- `PowerBI/`

Suggested screenshots to feature:

- SQL query result
- Power BI executive dashboard
- Placement distribution chart
- Streamlit prediction screen

---

## 📈 Dashboard

The Power BI dashboard focuses on executive placement reporting:

- KPI cards for total students, placement rate, average salary, and highest salary
- branch-wise placement breakdown
- gender-wise placement distribution
- company-wise hiring summary
- salary insights by branch and recruiter
- slicers for branch, gender, placement status, and company

---

## 🔍 What This Project Analyzes

- **Branch-wise placement** - placement rate and salary across AI, CS, DS, Electrical, IT, and Mechanical
- **CGPA and backlogs** - how academic performance affects placement outcomes
- **Skill impact** - internships, certifications, coding, and communication scores
- **Company analysis** - top recruiters by hire volume and average package
- **Salary distribution** - average, top-decile, and highest packages

---

## 🗃️ SQL Analysis

### Key Results

**Branch Placement Rate**

| Branch | Total | Placed | Rate |
|---|---:|---:|---:|
| Data Science | 1,991 | 1,829 | 91.86% |
| Information Technology | 1,955 | 1,793 | 91.71% |
| Artificial Intelligence | 2,016 | 1,838 | 91.17% |
| Mechanical Engineering | 2,107 | 1,919 | 91.08% |
| Computer Science | 1,950 | 1,772 | 90.87% |
| Electrical Engineering | 1,981 | 1,786 | 90.16% |

**Top Recruiters**

| Company | Hires | Avg Package (LPA) |
|---|---:|---:|
| Infosys | 1,629 | 9.93 |
| Accenture | 1,601 | 9.97 |
| Cognizant | 1,599 | 9.94 |
| Capgemini | 1,558 | 9.92 |
| Adobe | 740 | 13.58 |
| Meta | 751 | 13.56 |
| Amazon | 679 | 13.48 |

**High-impact findings**

- Communication score rises from about 75% placement rate at score 2 to about 97% at score 10.
- Average employability score is 11.12 for placed students vs. 8.14 for unplaced students.
- Product-based companies pay more than mass recruiters but hire fewer students.

---

## 🤖 Models

**Placement Prediction**

- Target: `placement_status`
- Algorithms: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting

**Salary Prediction**

- Target: `package_lpa`
- Algorithms: Linear Regression, Random Forest Regressor, Gradient Boosting Regressor

### Workflow

- preprocess student inputs
- predict placement probability
- estimate expected salary
- show results in the Streamlit app

---

## ✅ Results

- Placement prediction and salary prediction are both available in the app.
- Internships, communication, CGPA, and backlogs are the biggest practical signals.
- Branch and company analytics give the project a real placement-office feel.

---

## 💡 Business Recommendations

- Increase internship participation early in the academic year.
- Run communication and mock interview workshops for at-risk students.
- Prioritize DSA and coding practice for weaker branches.
- Track recruiter-wise hiring patterns to improve company targeting.
- Identify students with low CGPA and backlogs before final-year placement season.

---

## ⭐ Key Insights

- Students with internships show higher placement rates and stronger salary outcomes.
- Communication scores above 7 correlate strongly with better placement outcomes.
- Product-based companies pay more than mass recruiters, but hire fewer students.
- Certifications add measurable value for students with limited experience.
- Employability score is one of the strongest separators between placed and unplaced students.

---

## 🧾 Project Progress Checklist

| Stage | Status |
|---|---|
| Step 1: Data Cleaning (Python) | Done |
| Step 2: Exploratory Data Analysis (EDA) | Done |
| Step 3: SQL Analysis & Schema | Done |
| Step 4: Power BI Dashboard | Done |
| Step 5: ML - Placement Prediction | Done |
| Step 6: ML - Salary Prediction | Done |
| Business Insights Report | Done |

---

## 🗂️ Project Structure

```text
Placement-Prediction-College-Analytics
|-- data/
|   |-- raw/
|   |-- cleaned/
|-- sql_analysis/
|   |-- schema.sql
|   |-- import.sql
|   |-- cleaning.sql
|   |-- queries.sql
|   |-- views.sql
|-- notebooks/
|-- powerbi/
|-- models/
|-- app.py
|-- README.md
|-- requirements.txt
```

---

## 🚀 Future Scope

- Finish the Power BI dashboard and connect it to `Branch_Performance` and `Company_Recruitment_Summary`
- Add a student recommendation engine
- Integrate with college ERP or placement cell data
- Add forecasting for year-wise placement trends
- Extend explainability with SHAP-based feature analysis

---

## 👩‍💻 Author

**Suhani Chauhan**

*Aspiring Data Analyst | Python | SQL | Power BI | Machine Learning*

---

> Disclaimer: Predictions from this project are for educational and analytical purposes only. They do not guarantee actual placement outcomes.

