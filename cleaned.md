# 📂 Credit Score Capstone Project — Detailed File Descriptions

> **Project**: Intelligent Credit Risk Scoring System  
> **Stack**: Python · Streamlit · Scikit-learn · Pandas · Matplotlib · Seaborn · Plotly  
> **Goal**: Evaluate borrower profiles and predict loan default probability using machine learning.

---

## 📁 Project Directory Tree

```
Credit_Score_Capstone_Project_GenAI/
├── app.py                                   # Main Streamlit web application
├── dt_model.pkl                             # Serialized model pipeline (pickle)
├── requirements.txt                         # Python dependency list
├── README.md                                # Project overview & instructions
├── cleaned.md                               # This file — detailed file descriptions
├── .gitignore                               # Git ignore rules
├── data/
│   ├── raw/
│   │   └── credit_risk_dataset_raw.csv      # Original untouched dataset
│   └── cleaned/
│       └── cleaned_credit_risk.csv          # Fully cleaned & processed dataset
└── notebook/
    ├── data_cleaning.ipynb                  # Data cleaning Jupyter notebook
    └── model_training.ipynb                 # Model training & evaluation notebook
```

---

## 1. `app.py` — Main Streamlit Application

| Property | Value |
|----------|-------|
| **Lines of Code** | 1,047 |
| **Size** | ~43 KB |
| **Language** | Python (Streamlit) |
| **Entry Point** | `streamlit run app.py` |

### Overview

`app.py` is the main web application file that powers the **CreditIQ — Risk Intelligence Platform**. It is a single-file Streamlit app with a polished dark-themed UI, built entirely with custom CSS and HTML rendered via `unsafe_allow_html`.

### Key Sections

#### 1. Custom CSS & Page Config (Lines 1–454)
- Sets page title to **"CreditIQ — Credit Risk Scoring"** with a wide layout and a credit-card favicon.
- Injects ~450 lines of custom CSS defining a premium dark UI (background `#0a0e14`, text in `#94a3b8`, accent `#38bdf8`).
- Styled components include: sidebar branding, metric cards, comparison tables, performance cards, classification report tables, prediction forms, result panels, risk badges, probability gauges, feature importance rows, and fade-in animations.
- Uses Google Fonts: **Space Grotesk** and **JetBrains Mono**.

#### 2. Model Loading (Lines 457–488)
- `load_model()` — A `@st.cache_resource` function that loads `dt_model.pkl` from either the current directory or a `model/` subdirectory.
- Extracts from the pickle package:
  - **`model`** — Primary Decision Tree Classifier
  - **`lr_model`** — Secondary Logistic Regression model
  - **`scaler`** — StandardScaler for feature scaling
  - **`encoders`** — Dictionary of LabelEncoders for categorical columns
  - **`feature_columns`** — Ordered list of feature names
  - **`dt_threshold`** / **`lr_threshold`** — Custom decision thresholds (default: 0.35)
  - **`dataset_info`** — Metadata (sample counts, feature count)
  - **`dt_metrics`** / **`lr_metrics`** — Evaluation metrics for both models

#### 3. Sidebar (Lines 489–521)
- Displays the **CreditIQ** brand logo and tagline.
- Navigation radio: **🏠 Overview**, **📊 Performance**, **🔮 Predict**.
- Model info badges showing primary/secondary model names, thresholds, accuracy, ROC-AUC, and dataset size.

#### 4. Page 1 — Overview (Lines 552–648)
- **Metric Cards**: Total Samples, Training Samples, Test Samples, Feature Count.
- **Model Comparison Chart**: Interactive Plotly grouped bar chart comparing Decision Tree vs Logistic Regression across Accuracy, ROC-AUC, Precision, Recall, and F1-Score.
- **Input Features**: Displays all feature column names as styled tags.

#### 5. Page 2 — Performance (Lines 651–815)
- Two tabs: **🌲 Decision Tree** and **📈 Logistic Regression**.
- Each tab renders via `render_model_tab()`:
  - **4 Performance Cards**: Accuracy, ROC-AUC, F1 Weighted, Default Precision.
  - **Confusion Matrix**: Matplotlib heatmap (seaborn) with percentages.
  - **Classification Report Table**: Good Loan, Default, Macro Avg, Weighted Avg — showing Precision, Recall, F1-Score, Support.
  - **Default Recall Callout**: Highlighted warning about recall for detecting defaults.
  - **Feature Importance**: Horizontal bar chart (DT only; plasma colormap).

#### 6. Page 3 — Predict (Lines 818–1047)
- Two-column layout: **Input Form** (left) and **Result Panel** (right).
- **Input Form** collects: Model selector (DT/LR), Age, Employment Length, Income, Home Ownership, Loan Intent, Loan Amount, Interest Rate, Prior Default, Credit History Length.
- **Processing Logic**:
  - Auto-calculates `loan_percent_income` (loan ÷ income).
  - Derives `loan_grade` (A–G) from a composite risk score.
  - Encodes categorical features using stored LabelEncoders.
  - Scales features with stored StandardScaler.
  - Runs prediction using the selected model with its custom threshold.
- **Result Panel** displays:
  - ✅ Good Loan or ⚠️ High Default Risk result card.
  - Risk Badge (LOW / MEDIUM / HIGH).
  - Default Probability gauge bar.
  - Summary rows: Model Used, Threshold, Estimated Grade, Loan % of Income, Predicted Class, Confidence, Default Probability, Risk Level.
  - Top 3 Feature Importance (DT) or Top 3 Risk Drivers with coefficients (LR).

### Dependencies Used
`streamlit`, `pandas`, `numpy`, `pickle`, `os`, `matplotlib`, `seaborn`, `plotly`, `altair`

---

## 2. `dt_model.pkl` — Serialized Model Package

| Property | Value |
|----------|-------|
| **Size** | ~47 KB |
| **Format** | Python Pickle |

### Contents

A single Python dictionary serialized with `pickle.dump()`, containing:

| Key | Type | Description |
|-----|------|-------------|
| `model` | `DecisionTreeClassifier` | Primary trained Decision Tree model |
| `lr_model` | `LogisticRegression` | Secondary trained Logistic Regression model |
| `scaler` | `StandardScaler` | Fitted scaler for numerical feature normalization |
| `encoders` | `dict[str, LabelEncoder]` | Label encoders for categorical columns |
| `feature_columns` | `list[str]` | Ordered feature names matching training data |
| `dt_threshold` | `float` | Optimized decision threshold for DT (~0.35) |
| `lr_threshold` | `float` | Optimized decision threshold for LR (~0.35) |
| `dataset_info` | `dict` | Total/train/test sample counts, feature count |
| `dt_metrics` | `dict` | DT evaluation: accuracy, ROC-AUC, confusion matrix, class metrics, feature importance |
| `lr_metrics` | `dict` | LR evaluation: accuracy, ROC-AUC, confusion matrix, class metrics, feature coefficients |

### Performance Summary (from stored metrics)
- **Decision Tree**: ~92.9% training accuracy, ~91.0% test accuracy
- **Logistic Regression**: Stored as secondary benchmark

---

## 3. `requirements.txt` — Python Dependencies

| Property | Value |
|----------|-------|
| **Lines** | 6 |

### Contents

```
streamlit>=1.32.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

> **Note**: `plotly` and `altair` are also imported in `app.py` but are **not listed** in `requirements.txt`. These should be added for production deployments.

---

## 4. `README.md` — Project Documentation

| Property | Value |
|----------|-------|
| **Lines** | 80 |
| **Size** | ~3.2 KB |

### Contents

A well-structured project README covering:
- **Project title**: 💳 Intelligent Credit Risk Scoring System
- **Features**: Real-time prediction, risk banding (Low/Medium/High), batch CSV processing, automated Loan-to-Income ratio, visual feedback
- **Tech Stack**: Streamlit, Scikit-learn (Decision Tree), Pandas, NumPy, Matplotlib, Seaborn
- **Project Structure**: Directory layout overview
- **Installation**: Clone + `pip install -r requirements.txt`
- **Usage**: `streamlit run app.py`, form-based prediction and batch mode instructions
- **Model Performance**: Training ~92.9%, Testing ~91.0%
- **Input Features**: Lists all 10 input features with descriptions (`person_age`, `person_income`, `person_home_ownership`, `person_emp_length`, `loan_intent`, `loan_amnt`, `loan_int_rate`, `loan_percent_income`, `cb_person_default_on_file`, `cb_person_cred_hist_length`)

---

## 5. `.gitignore` — Git Ignore Rules

| Property | Value |
|----------|-------|
| **Lines** | 1 |

### Contents

```
.DS_Store
```

Only ignores macOS `.DS_Store` metadata files.

---

## 6. `data/raw/credit_risk_dataset_raw.csv` — Raw Dataset

| Property | Value |
|----------|-------|
| **Rows** | 32,581 (+ 1 header) |
| **Size** | ~1.68 MB |
| **Source** | Kaggle (`busrapehlivan/credit-risk-dataset`) |

### Columns (11)

| # | Column | Description |
|---|--------|-------------|
| 1 | `person_age` | Age of the applicant |
| 2 | `person_income` | Annual income |
| 3 | `person_home_ownership` | Home ownership status (RENT, OWN, MORTGAGE, OTHER) |
| 4 | `person_emp_length` | Employment length in years |
| 5 | `loan_intent` | Purpose of loan (PERSONAL, EDUCATION, MEDICAL, VENTURE, etc.) |
| 6 | `loan_amnt` | Loan amount requested |
| 7 | `loan_int_rate` | Interest rate |
| 8 | `loan_status` | Target variable (0 = Good Loan, 1 = Default) |
| 9 | `loan_percent_income` | Loan amount as a percentage of income |
| 10 | `cb_person_default_on_file` | Historical default on file (Y/N) |
| 11 | `cb_person_cred_hist_length` | Credit history length in years |

### Known Issues (Pre-Cleaning)
- Missing values in `loan_int_rate` and `person_emp_length`
- Outlier ages (< 18 and > 100)
- Extreme loan amounts skewing distributions
- Employment lengths > 60 years (data entry errors)

---

## 7. `data/cleaned/cleaned_credit_risk.csv` — Cleaned Dataset

| Property | Value |
|----------|-------|
| **Rows** | 32,576 (+ 1 header) |
| **Size** | ~1.79 MB |
| **Source** | Output of `data_cleaning.ipynb` |

### Columns (12) — includes `loan_grade` added during cleaning

| # | Column | Description |
|---|--------|-------------|
| 1 | `person_age` | Age (filtered to 18–100 range) |
| 2 | `person_income($)` | Annual income (renamed with $ unit) |
| 3 | `person_home_ownership` | Home ownership status |
| 4 | `person_emp_length` | Employment length (capped at 60, missing imputed with median) |
| 5 | `loan_intent` | Purpose of loan |
| 6 | `loan_grade` | Loan grade (A–G, derived from risk proxy) |
| 7 | `loan_amnt($)` | Loan amount (renamed with $ unit, IQR-clipped) |
| 8 | `loan_int_rate` | Interest rate (missing imputed with median) |
| 9 | `loan_status` | Target variable (0/1) |
| 10 | `loan_percent_income` | Loan % of income |
| 11 | `cb_person_default_on_file` | Historical default (Y/N) |
| 12 | `cb_person_cred_hist_length` | Credit history length in years |

### Key Differences from Raw
- **5 fewer rows** (removed unrealistic ages outside 18–100)
- **1 new column**: `loan_grade`
- **Renamed columns**: `person_income` → `person_income($)`, `loan_amnt` → `loan_amnt($)`
- **Zero missing values**
- **Outliers handled** via clipping (not deletion)

---

## 8. `notebook/data_cleaning.ipynb` — Data Cleaning Notebook

| Property | Value |
|----------|-------|
| **Total Cells** | ~80 |
| **Size** | ~1.2 MB (includes embedded output images) |
| **Platform** | Google Colab |

### Overview

A comprehensive step-by-step data cleaning pipeline with **before/after visualizations** for every transformation. Downloads the raw dataset from Kaggle and outputs `cleaned_credit_risk.csv`.

### Cleaning Steps

| Step | Operation | Details |
|------|-----------|---------|
| **Step 1** | Load Dataset | Downloads from Kaggle via `kagglehub`, loads into DataFrame (32,581 rows × 11 columns) |
| **Step 2** | Explore Data | `df.head()`, `df.info()`, `df.describe()` — initial exploration |
| **Step 3** | Missing Values Analysis | Identifies missing values in `loan_int_rate` and `person_emp_length`; visualizes with bar charts |
| **Step 4** | Impute Loan Interest Rate | **Median imputation** — chosen for robustness against skewed financial data |
| **Step 5** | Analyze Age Distribution | Examines age range, identifies outliers (< 18 or > 100) |
| **Step 6** | Clean Age Outliers | Filters dataset to keep only 18 ≤ age ≤ 100 (rows dropped: 5) |
| **Step 7** | Clean Loan Amount | **IQR-based clipping** — clips extreme loan amounts to upper/lower bounds |
| **Step 8** | Rename Columns for Clarity | Adds `($)` unit suffix: `person_income` → `person_income($)`, `loan_amnt` → `loan_amnt($)` |
| **Step 9** | Analyze Employment Length | Identifies outliers (> 60 years) and missing values |
| **Step 10** | Clean Employment Length | **Clips** at 60 years + **median imputation** for missing values |
| **Step 11** | Final Data Quality Assessment | Verifies zero missing values; creates completeness bar chart and distribution summary plots |
| **Step 12** | Save Cleaned Dataset | Exports to `cleaned_credit_risk.csv` |

### Visualization Inventory (20 Plots)

| Plot # | Description |
|--------|-------------|
| 1 | Missing values bar chart |
| 2 | Missing values percentage heatmap |
| 3 | Histogram of loan interest rate (before) |
| 4 | Histogram of loan interest rate (after imputation) |
| 5 | Before vs after comparison — interest rate |
| 6 | Box plot of age (before) |
| 7 | Histogram of age (before) |
| 8 | Box plot of age (after filtering) |
| 9 | Histogram of age (after filtering) |
| 10 | Box plot of loan amount (before) |
| 11 | Histogram of loan amount (before) |
| 12 | Histogram of loan amount (after clipping) |
| 13 | Loan amount — before vs after comparison |
| 14 | Box plot of employment length (before) |
| 15 | Histogram of employment length (before) |
| 16 | Box plot of employment length (after) |
| 17 | Histogram of employment length (after) |
| 18 | Employment length — before vs after comparison |
| 19 | Data completeness summary (bar chart) |
| 20 | Final distribution summary (6 subplots) |

### Key Takeaways
- ✅ **Preservation over Deletion** — clipped outliers instead of removing rows
- ✅ **Median Imputation** — robust to outliers for skewed distributions
- ✅ **Domain Knowledge** — age 18–100, employment ≤ 60 years
- ✅ **High Data Retention** — only 5 rows removed out of 32,581

---

## 9. `notebook/model_training.ipynb` — Model Training & Evaluation Notebook

| Property | Value |
|----------|-------|
| **Total Cells** | ~98 |
| **Size** | ~1.17 MB (includes embedded output images) |
| **Platform** | Google Colab |

### Overview

Complete machine learning pipeline from data loading through model training, evaluation, comparison, overfitting checks, and model serialization. Trains and compares two models: **Decision Tree** and **Logistic Regression**.

### Pipeline Steps

| Step | Operation | Details |
|------|-----------|---------|
| **Step 1** | Install Libraries | `scikit-learn`, `kagglehub` |
| **Step 2** | Import Libraries | pandas, numpy, sklearn (preprocessing, models, metrics), matplotlib, seaborn |
| **Step 3** | Load Cleaned Dataset | Loads `cleaned_credit_risk.csv` (32,576 rows × 12 columns) |
| **Step 4** | Exploratory Data Analysis | Dataset shape, dtypes, describe, class distribution of `loan_status` |
| **Step 5** | Visualize Target Distribution | Bar chart and pie chart of Good Loan (0) vs Default (1) |
| **Step 6** | Feature Engineering & Encoding | Label encoding for categorical features (`person_home_ownership`, `loan_intent`, `cb_person_default_on_file`) |
| **Step 7** | Feature Selection & Split | Drops `loan_status` (target); 80/20 train-test split with `random_state=42` |
| **Step 8** | Feature Scaling | StandardScaler fitted on training data, applied to both train and test |
| **Step 9** | Train Models | **Logistic Regression** (`max_iter=1000`, `random_state=42`) and **Decision Tree** (`max_depth=10`, `random_state=42`) |
| **Step 10** | Basic Evaluation Metrics | Accuracy, Precision, Recall, F1-Score for both models |
| **Step 11** | ROC-AUC Score | Computes ROC-AUC using predicted probabilities for both models |
| **Step 12** | Confusion Matrices | Matplotlib visualizations with `ConfusionMatrixDisplay` |
| **Step 13** | ROC Curves | Individual curves for LR and DT, plus side-by-side comparison |
| **Step 14** | Overfitting Check | Compares training vs test accuracy; Decision Tree gap analysis |
| **Step 15** | Final Comparison | Comprehensive table + grouped bar chart dashboard |
| **Step 16** | Save Model Package | Serializes everything into `dt_model.pkl` using pickle |

### Visualization Inventory (16 Plots)

| Plot # | Description |
|--------|-------------|
| 1 | Target variable distribution (bar chart) |
| 2 | Target variable distribution (pie chart) |
| 3 | Feature correlation heatmap |
| 4 | Numerical feature distributions (histograms) |
| 5 | Categorical feature distributions (count plots) |
| 6 | LR coefficients bar chart |
| 7 | DT feature importance bar chart |
| 8–9 | DT feature importance (top 5 + full) |
| 10 | LR confusion matrix |
| 11 | DT confusion matrix |
| 12 | LR ROC curve |
| 13 | DT ROC curve |
| 14 | Both ROC curves overlaid |
| 15 | Training vs Test accuracy comparison |
| 16 | Model comparison dashboard (Accuracy + ROC-AUC) |

### Model Hyperparameters

| Parameter | Decision Tree | Logistic Regression |
|-----------|---------------|---------------------|
| `max_depth` | 10 | — |
| `max_iter` | — | 1000 |
| `random_state` | 42 | 42 |
| Custom Threshold | 0.35 | 0.35 |

### Saved Pipeline Components
The final step packages all trained artifacts (models, scaler, encoders, metrics, thresholds, dataset info) into a single pickle file for deployment.

---

## Summary Table

| File | Type | Purpose | Size |
|------|------|---------|------|
| `app.py` | Python | Streamlit web app — 3-page credit risk platform | 43 KB |
| `dt_model.pkl` | Pickle | Complete model package (DT + LR + scaler + encoders + metrics) | 47 KB |
| `requirements.txt` | Text | Python package dependencies | 100 B |
| `README.md` | Markdown | Project documentation and usage instructions | 3.2 KB |
| `.gitignore` | Config | Git ignore rules (.DS_Store) | 9 B |
| `data/raw/credit_risk_dataset_raw.csv` | CSV | Original Kaggle dataset (32,581 rows × 11 cols) | 1.68 MB |
| `data/cleaned/cleaned_credit_risk.csv` | CSV | Cleaned dataset (32,576 rows × 12 cols) | 1.79 MB |
| `notebook/data_cleaning.ipynb` | Notebook | 12-step cleaning pipeline with 20 visualizations | 1.2 MB |
| `notebook/model_training.ipynb` | Notebook | 16-step ML pipeline with 16 visualizations | 1.17 MB |
