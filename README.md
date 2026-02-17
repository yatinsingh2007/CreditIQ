# 💳 Intelligent Credit Risk Scoring System

This project is an **AI-powered Credit Risk Scoring System** designed to evaluate borrower profiles and predict the probability of loan default. Built with **Streamlit** and **Scikit-learn**, it provides an interactive interface for real-time risk assessment and supports batch processing for analyzing multiple applicants at once.

## 🚀 Features

- **Real-Time Prediction**: Interactive form to input applicant details (Age, Income, Loan Amount, etc.) and get instant risk assessments.
- **Risk Banding**: Automatically categorizes applicants into **Low**, **Medium**, or **High** risk based on predicted default probability.
- **Batch Processing**: Upload a CSV file to process multiple loan applications simultaneously and download the results.
- **Automated Insights**: Auto-calculates key metrics like the **Loan-to-Income Ratio**.
- **Visual Feedback**: visual indicators and progress bars to display risk levels and model confidence.

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Machine Learning**: [Scikit-learn](https://scikit-learn.org/) (Decision Tree Classifier)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn

## 📂 Project Structure

```
├── app.py                # Main Streamlit application file
├── dt_model.pkl          # Pre-trained Decision Tree model pipeline (includes scaler/encoders)
├── requirements.txt      # Project dependencies
├── data/                 # Directory for datasets (raw/processed)
└── notebook/             # Jupyter notebooks for EDA and model training
```

## ⚙️ Installation

1.  **Clone the repository**:

    ```bash
    git clone <repository-url>
    cd Credit_Score_Capstone_Project_GenAI
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🏃‍♂️ Usage

1.  **Run the Streamlit app**:

    ```bash
    streamlit run app.py
    ```

2.  **Using the Interface**:
    - **Applicant Information**: Fill in the details such as Age, Income, Home Ownership, and Loan Intent.
    - **Predict**: Click "Predict Credit Risk" to see the Default Probability and Risk Category.
    - **Batch Mode**: Upload a CSV file containing applicant data to generate predictions for a whole dataset.

## 📊 Model Performance

The system uses a tuned **Decision Tree Classifier** with the following performance metrics:

- **Training Accuracy**: ~92.9%
- **Testing Accuracy**: ~91.0%

The model ensures balanced generalization to avoid overfitting while maintaining high predictive accuracy.

## 📝 Input Features

The model relies on the following key features:

- `person_age`: Age of the applicant
- `person_income`: Annual Income
- `person_home_ownership`: Home ownership status (RENT, OWNER, MORTGAGE, etc.)
- `person_emp_length`: Employment length in years
- `loan_intent`: Purpose of the loan (Education, Medical, Venture, etc.)
- `loan_amnt`: Loan amount requested
- `loan_int_rate`: Interest rate
- `loan_percent_income`: Loan amount as a percentage of income
- `cb_person_default_on_file`: Historical default status (Y/N)
- `cb_person_cred_hist_length`: Length of credit history in years
