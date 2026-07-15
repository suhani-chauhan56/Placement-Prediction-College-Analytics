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

## 🧭 Final Architecture

```text
Student Data
    │
    ▼
ETL Pipeline (Python + SQL)
    │
    ▼
Data Warehouse
    │
    ▼
Exploratory Data Analysis
    │
    ▼
Power BI Executive Dashboard
    │
    ▼
ML Placement Prediction
    │
    ▼
SHAP Explainability
    │
    ▼
Student Risk Score
    │
    ▼
Personalized Recommendations
    │
    ▼
Placement Officer Dashboard
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

## 🧱 ETL Pipeline

```text
Raw CSV
   ↓
Cleaning
   ↓
Transformation
   ↓
Feature Engineering
   ↓
SQL Database
   ↓
Python Analytics
   ↓
Power BI
```

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

### Executive Dashboard

Instead of only a student dashboard, the project is framed as a **Placement Officer Dashboard** with:

- Placement %
- Department Performance
- Recruiter Count
- Average Package
- Unplaced Students
- Placement Funnel

### Department Dashboard

The app also includes a department-level view with:

- branch-wise placement rate
- risk mix by branch
- hiring trend
- average CGPA
- average employability
- intervention lists

---

## 🔍 What This Project Analyzes

- **Branch-wise placement** - placement rate and salary across AI, CS, DS, Electrical, IT, and Mechanical
- **CGPA and backlogs** - how academic performance affects placement outcomes
- **Skill impact** - internships, certifications, coding, and communication scores
- **Company analysis** - top recruiters by hire volume and average package
- **Salary distribution** - average, top-decile, and highest packages

---

## 🗃️ SQL Analysis

Recruiters love SQL because it shows practical analytics thinking.

### Example analyses

- Top 10 companies hiring
- Branch with highest package
- Average CGPA by company
- Students with internship but no placement
- Placement rate by gender
- Package distribution
- Students requiring intervention

### Why SQL matters here

- It demonstrates data modeling, joins, grouping, and reporting.
- It helps create repeatable analysis for the placement cell.
- It supports dashboard refreshes and executive reporting.

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

## 🎯 Student Risk Score

Instead of only showing:

- Placed
- Not Placed

The project can be presented as decision support:

- Employability Score: `92/100`
- Placement Risk: `Low`
- Recommendation:
  - Improve aptitude
  - Complete one internship
  - Increase DSA score

This is more useful than a binary result because it explains what to do next.

---

## 🧠 SHAP Explainability

Instead of only saying:

- Prediction = Not Placed

Show the main drivers behind the result:

- CGPA contributed `+18%`
- Internship `+15%`
- Communication `+12%`
- Coding `-8%`
- Projects `-10%`

This makes the model easier to trust and easier to explain in interviews.

---

## 📊 Branch Analytics

Branch-focused dashboards can answer:

- Highest package?
- Lowest placement?
- Average CGPA?
- Hiring trend?

Useful branches to track:

- CSE
- IT
- ECE
- ME
- Civil
- MBA

---

## 🏢 Company Analytics

Instead of student analytics only, the project also fits recruiter-style reporting:

- Top Recruiters
- Hiring Trend
- Average Package
- Selection Rate
- CTC Distribution
- Offer Acceptance
- Repeat Recruiters

That makes the project feel closer to real HR analytics.

---

## 📈 Time Series Analysis

Placement data becomes stronger when it is viewed over time.

Example questions:

- Placement trend in 2019
- Placement trend in 2020
- Placement trend in 2021
- Placement trend in 2022
- Placement trend in 2023
- Forecast for 2024

Recruiters and placement officers both value trend forecasting.

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
- show the output inside the Streamlit app

### Recommendation Engine

Instead of stopping at a prediction, the project can recommend how to improve:

- SQL
- Python
- Communication
- Projects

Example:

- Placement Chance: `61%`
- After recommendations: `82%`

That turns prediction into guidance.

---

## ✅ Results

- Placement prediction and salary prediction are both available in the app.
- The dataset shows a strong relationship between internships, communication, CGPA, and placement outcome.
- Branch and company analytics give the project a real placement-office feel.
- The README now frames the project as analytics plus decision support, not only prediction.

---

## 💡 Business Recommendations

- Increase internship participation early in the academic year.
- Run communication and mock interview workshops for at-risk students.
- Prioritize DSA and coding practice for weaker branches.
- Track recruiter-wise hiring patterns to improve company targeting.
- Identify students with low CGPA and backlogs before final-year placement season.
- Prioritize departments with declining placement trends.
- Conduct company-specific mock interviews.
- Focus on communication workshops for students with low soft-skill scores.

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
|   |-- marts/
|-- sql_analysis/
|   |-- schema.sql
|   |-- import.sql
|   |-- cleaning.sql
|   |-- queries.sql
|   |-- views.sql
|-- sql/
|   |-- placement_flat_table.csv
|-- notebooks/
|-- powerbi/
|-- models/
|-- app.py
|-- README.md
|-- requirements.txt
```

---

## 🚀 Deployment

You can deploy or test the app with:

- Streamlit
- Render
- Azure
- Hugging Face Spaces

That makes it easy for recruiters to open and test the project instantly.

---

## 🚀 Future Scope

- Finish the Power BI dashboard and connect it to `Branch_Performance` and `Company_Recruitment_Summary`
- Add a student recommendation engine
- Integrate with college ERP or placement cell data
- Add forecasting for year-wise placement trends
- Extend explainability with SHAP-based feature analysis
- Add a richer department dashboard for placement officers
- Deploy the dashboard with Streamlit, Render, Azure, or Hugging Face Spaces

---

## 👩‍💻 Author

**Suhani Chauhan**

*Aspiring Data Analyst | Python | SQL | Power BI | Machine Learning*

---

> Disclaimer: Predictions from this project are for educational and analytical purposes only. They do not guarantee actual placement outcomes.
