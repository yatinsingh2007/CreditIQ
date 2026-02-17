import streamlit as st
import pickle
import pandas as pd
import numpy as np

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Intelligent Credit Risk Scoring",
    page_icon="💳",
    layout="wide"
)

# -------------------------
# LOAD MODEL
# -------------------------
@st.cache_resource
def load_model():
    return pickle.load(open("dt_model.pkl", "rb"))

package = load_model()
model = package["model"]
scaler = package["scaler"]
encoders = package["encoders"]

# -------------------------
# FEATURES (MUST MATCH TRAINING)
# -------------------------
FEATURES = [
    'person_age',
    'person_income($)',
    'person_home_ownership',
    'person_emp_length',
    'loan_intent',
    'loan_amnt($)',
    'loan_int_rate',
    'loan_percent_income',
    'cb_person_default_on_file',
    'cb_person_cred_hist_length'
]

# -------------------------
# HEADER
# -------------------------
st.title("💳 Intelligent Credit Risk Scoring System")
st.markdown("""
This ML-powered system evaluates borrower profiles and predicts 
**loan default probability** using a trained Decision Tree model.
""")

st.divider()

# -------------------------
# INPUT SECTION
# -------------------------
st.subheader("📌 Applicant Information")

col1, col2 = st.columns(2)

with col1:
    person_age = st.number_input("Age", 18, 100, 30)
    person_income = st.number_input("Annual Income ($)", 1, 1_000_000, 50000)
    person_home_ownership = st.selectbox(
        "Home Ownership",
        ["RENT", "MORTGAGE", "OWN", "OTHER"]
    )
    person_emp_length = st.number_input("Employment Length (Years)", 0.0, 40.0, 5.0)

with col2:
    loan_intent = st.selectbox(
        "Loan Purpose",
        ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL",
         "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"]
    )
    loan_amnt = st.number_input("Loan Amount ($)", 1, 1_000_000, 10000)
    loan_int_rate = st.number_input("Interest Rate (%)", 0.0, 100.0, 10.0)
    cb_person_default_on_file = st.selectbox("Historical Default", ["Y", "N"])
    cb_person_cred_hist_length = st.number_input("Credit History (Years)", 0, 50, 3)

# Auto-calculate Loan-to-Income Ratio
loan_percent_income = loan_amnt / person_income

st.info(f"📊 Auto Calculated Loan-to-Income Ratio: {loan_percent_income:.2f}")

st.divider()

# -------------------------
# RISK BAND FUNCTION
# -------------------------
def get_risk_band(prob):
    if prob < 0.30:
        return "Low Risk 🟢"
    elif prob < 0.70:
        return "Medium Risk 🟡"
    else:
        return "High Risk 🔴"

# -------------------------
# PREDICTION
# -------------------------
if st.button("🚀 Predict Credit Risk"):

    input_data = pd.DataFrame([[
        person_age,
        person_income,
        person_home_ownership,
        person_emp_length,
        loan_intent,
        loan_amnt,
        loan_int_rate,
        loan_percent_income,
        cb_person_default_on_file,
        cb_person_cred_hist_length
    ]], columns=FEATURES)

    try:
        # Encode categorical features
        for col, encoder in encoders.items():
            if col in input_data.columns:
                input_data[col] = encoder.transform(input_data[col])

        # Scale numerical features
        input_scaled = scaler.transform(input_data)

        # Predict probability
        probability = model.predict_proba(input_scaled)[0][1]

        # Avoid pure 1.0 display issue
        probability = min(probability, 0.9999)

        risk_band = get_risk_band(probability)

        st.subheader("📊 Risk Assessment Result")

        colA, colB, colC = st.columns(3)

        colA.metric("Probability of Default", f"{probability*100:.2f}%")
        colB.metric("Risk Category", risk_band)
        colC.metric("Model Confidence", f"{max(probability, 1-probability)*100:.2f}%")

        st.progress(int(probability * 100))

    except Exception as e:
        st.error(f"Prediction Error: {e}")

# -------------------------
# BATCH PROCESSING
# -------------------------
st.divider()
st.subheader("📂 Batch Prediction (CSV Upload)")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    batch_df = pd.read_csv(uploaded_file)

    if st.button("Process Batch File"):
        try:
            processed_df = batch_df.copy()

            # Auto compute Loan-to-Income
            processed_df["loan_percent_income"] = (
                processed_df["loan_amnt($)"] /
                processed_df["person_income($)"]
            )

            # Encode categorical columns
            for col, encoder in encoders.items():
                if col in processed_df.columns:
                    processed_df[col] = encoder.transform(processed_df[col])

            # Scale
            processed_scaled = scaler.transform(processed_df[FEATURES])

            probabilities = model.predict_proba(processed_scaled)[:, 1]
            probabilities = np.minimum(probabilities, 0.9999)

            processed_df["Default_Probability"] = probabilities
            processed_df["Risk_Category"] = [
                get_risk_band(p) for p in probabilities
            ]

            st.dataframe(processed_df)

            st.download_button(
                "Download Results",
                processed_df.to_csv(index=False),
                "credit_risk_predictions.csv",
                "text/csv"
            )

        except Exception as e:
            st.error(f"Batch Processing Error: {e}")

# -------------------------
# MODEL PERFORMANCE
# -------------------------
st.divider()
st.subheader("📈 Model Performance Summary")

st.write("""
- Decision Tree Classifier (max_depth tuned)
- Training Accuracy ≈ 92.9%
- Testing Accuracy ≈ 91.0%
- Balanced generalization performance
""")

st.caption("Milestone 1 – ML-Based Credit Risk Scoring System")