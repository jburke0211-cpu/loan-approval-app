# -*- coding: utf-8 -*-
import streamlit as st
import pickle
import pandas as pd
import sklearn  # Needed so sklearn objects inside the pickle can be loaded

# Load the trained (logistic regression) model
# Make sure my_logmodel.pkl is in the same folder as this app file.
with open("my_logmodel.pkl", "rb") as file:
    model = pickle.load(file)

# App title
st.markdown(
    "<h1 style='text-align: center; background-color: #ffcccc; "
    "padding: 10px; color: #cc0000;'><b>Personal Loan Approval</b></h1>",
    unsafe_allow_html=True
)

st.header("Enter Applicant Details")

# Numeric inputs
requested_loan_amount = st.number_input(
    "Requested Loan Amount", min_value=0.0, step=500.0
)

granted_loan_amount = st.number_input(
    "Granted Loan Amount (if already offered; else leave at 0)",
    min_value=0.0, step=500.0
)

fico_score = st.slider(
    "FICO Score", min_value=300, max_value=850, step=5
)

monthly_gross_income = st.number_input(
    "Monthly Gross Income", min_value=0.0, step=100.0
)

monthly_housing_payment = st.number_input(
    "Monthly Housing Payment", min_value=0.0, step=50.0
)

ever_bankrupt = st.selectbox(
    "Ever Bankrupt or Foreclosed?",
    ["No", "Yes"]
)


# Categorical inputs

reason = st.selectbox(
    "Reason for Loan",
    [
        "cover_an_unexpected_cost",
        "credit_card_refinancing",
        "home_improvement",
        "major_purchase",
        "other",
        "debt_conslidation"
    ]
)

employment_status = st.selectbox(
    "Employment Status",
    [
        "full_time",
        "part_time",
        "unemployed"
    ]
)

employment_sector = st.selectbox(
    "Employment Sector",
    [
        "consumer_discretionary",
        "information_technology",
        "energy",
        "consumer_staples",
        "communication_services",
        "materials",
        "utilities",
        "real_estate",
        "health_care",
        "industrials",
        "financials"
    ]
)

lender = st.selectbox(
    "Original Lender (if known)",
    ["A", "B", "C"]
)


# Build single-row DataFrame
input_data = pd.DataFrame(
    {
        "Requested_Loan_Amount": [requested_loan_amount],
        "Granted_Loan_Amount": [granted_loan_amount],
        "FICO_score": [fico_score],
        "Monthly_Gross_Income": [monthly_gross_income],
        "Monthly_Housing_Payment": [monthly_housing_payment],
        "Ever_Bankrupt_or_Foreclose": [1 if ever_bankrupt == "Yes" else 0],
        "Reason": [reason],
        "Employment_Status": [employment_status],
        "Employment_Sector": [employment_sector],
        "Lender": [lender],
    }
)


# Prepare data for prediction

# One-hot encode categorical variables
categorical_cols = [
    "Reason",
    "Employment_Status",
    "Employment_Sector",
    "Lender",
]
input_data_encoded = pd.get_dummies(input_data, columns=categorical_cols)

# Ensure we have exactly the columns the model expects
model_columns = model.feature_names_in_

# Add any missing columns (set to 0)
for col in model_columns:
    if col not in input_data_encoded.columns:
        input_data_encoded[col] = 0

# Reorder and drop unknown columns
input_data_encoded = input_data_encoded[model_columns]

# Prediction
if st.button("Evaluate Loan"):
    # Class prediction (0 = denied, 1 = approved)
    pred_class = model.predict(input_data_encoded)[0]

    # Probability of approval from logistic regression
    pred_proba_str = "N/A" # Default string if probability is not available
    try:
        # probability that Approved = 1
        pred_proba = float(model.predict_proba(input_data_encoded)[0, 1])
        pred_proba_str = f"{pred_proba * 100:.2f}%" # Format to 2 decimal places
    except Exception:
        # Handle cases where predict_proba might fail or not be available
        pass # pred_proba_str remains "N/A"

    # Show class prediction with probability
    if pred_class == 1:
        st.success(f"Prediction: **Loan Approved** (Probability: **{pred_proba_str}**)")
    else:
        st.error(f"Prediction: **Loan Denied** (Probability: **{pred_proba_str}**)")
