# 💳 CreditIQ: Intelligent Credit Risk Scoring System

> **An AI-powered risk intelligence platform for evaluating borrower profiles and predicting loan default probabilities.**

This project uses a robust Machine Learning pipeline and a premium Streamlit web application to provide real-time credit risk assessments. Built with **Scikit-learn**, it supports evaluating applicants instantly through an interactive interface, relying on models trained on thousands of historical loan applications.

---

## ✨ Key Features

- **Real-Time Risk Prediction**: An interactive frontend form accepts applicant details (Age, Income, Home Ownership, Loan Intent, etc.) and returns an instant credit risk assessment.
- **Risk Banding & Grading**: Automatically assigns applicants into **LOW**, **MEDIUM**, or **HIGH** risk tiers. It also generates a simulated Loan Grade (A-G) based on a composite risk score.
- **Multi-Model Support**: Compare predictions between the primary **Decision Tree Classifier** and a secondary benchmark **Logistic Regression** model.
- **Performance Dashboards**: View comprehensive evaluation metrics within the app, including Accuracy, ROC-AUC, F1-Scores, Confusion Matrices, Classification Reports, and Feature Importance.
- **Premium Dark-Themed UI**: A highly customized Streamlit interface built with extensive custom CSS, featuring clean typography (Google Fonts), hover animations, probability gauges, and stylized metric cards.
- **Data-Driven Insights**: Top risk drivers and feature importance scores explain _why_ a particular decision was made for an applicant.

---

## 🛠️ Tech Stack

- **Frontend Application**: [Streamlit](https://streamlit.io/)
- **Machine Learning**: [Scikit-learn](https://scikit-learn.org/) (Decision Tree Classifier, Logistic Regression)
- **Data Processing**: Pandas, NumPy
- **Visualizations**: Matplotlib, Seaborn, Plotly, Altair
- **Serialization**: Pickle

---

## 📁 Project Structure

```text
Credit_Score_Capstone_Project_GenAI/
├── app.py                                   # Main Streamlit web application
├── dt_model.pkl                             # Serialized model pipeline (pickle)
├── requirements.txt                         # Python dependency list
├── README.md                                # This file
├── cleaned.md                               # Detailed file and dataset documentation
├── data/
│   ├── raw/
│   │   └── credit_risk_dataset_raw.csv      # Original Kaggle dataset (32.5k rows)
│   └── cleaned/
│       └── cleaned_credit_risk.csv          # Fully cleaned & processed dataset
└── notebook/
    ├── data_cleaning.ipynb                  # Data cleaning & EDA Jupyter notebook
    └── model_training.ipynb                 # ML model training & evaluation notebook
```

---

## 🔬 How It Works

### 1. Data Cleaning (`data_cleaning.ipynb`)

The pipeline begins with raw data from Kaggle. Outliers (like impossible ages or 60+ year employment histories) are clipped using IQR bounds rather than deleted, and missing values in interest rates and employment lengths are imputed using medians. A custom `loan_grade` was also simulated for better portfolio analysis.

### 2. Model Training (`model_training.ipynb`)

We trained two separate models to balance baseline linear insights against non-linear pattern recognition:

- **Decision Tree (Primary)**: Tuned with `max_depth=10`, acting as the main predictor. Achieves ~91.0% Test Accuracy.
- **Logistic Regression (Secondary)**: Used as a benchmark.
  The pipeline packages StandardScaler, LabelEncoders, Evaluation Metrics, and both Models into a single `dt_model.pkl` file for seamless production deployment.

### 3. Application (`app.py`)

The Streamlit app loads the `dt_model.pkl` artifact. When an applicant's data is entered, the app encodes the categorical features, scales the numerics, and calculates the Default Probability using `.predict_proba()`. The result defines the risk class, progress bars, and feature importance explanations on the UI.

---

## 📝 Input Features

The model relies on up to 10 crucial features:

- `person_age`: Applicant's Age (18-100)
- `person_income`: Annual Income ($)
- `person_home_ownership`: RENT, OWN, MORTGAGE, or OTHER
- `person_emp_length`: Employment length in years
- `loan_intent`: PERSONAL, EDUCATION, MEDICAL, VENTURE, HOME IMPROVEMENT, or DEBT CONSOLIDATION
- `loan_amnt`: Total Loan Amount Requested ($)
- `loan_int_rate`: Interest Rate (%)
- `cb_person_default_on_file`: Historical default on file (Y/N)
- `cb_person_cred_hist_length`: Credit history length in years
- `loan_percent_income`: (Auto-calculated) Loan Amount ÷ Income

---

## 🚀 Setup & Installation

Follow these steps to run the application locally:

### 1. Clone the repository

```bash
git clone <repository-url>
cd Credit_Score_Capstone_Project_GenAI
```

### 2. Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

Ensure that `dt_model.pkl` is located in the root directory. Then execute:

```bash
streamlit run app.py
```

The application will launch in your default web browser (typically on `http://localhost:8501`).

---

## 📊 Evaluation & Performance

The Decision Tree algorithm was selected based on its strong predictive capability and explainability.

- **Training Accuracy**: ~92.9%
- **Testing Accuracy**: ~91.0%
- **ROC-AUC**: Evaluates the model's ability to distinguish between Good Loans and Defaults.

_Comprehensive metric reports (Precision, Recall, F1) and interactive Confusion Matrices can be explored directly within the **Performance** tab of the Streamlit application._
