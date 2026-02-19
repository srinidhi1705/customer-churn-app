import streamlit as st
import pandas as pd
import joblib

# Load model, scaler and feature names
feature_names = joblib.load("model/feature_names.pkl")
model = joblib.load("model/churn_model.pkl")
scaler = joblib.load("model/scaler.pkl")

st.title("Telecom Customer Churn Prediction")
st.write("Enter customer details below:")
st.sidebar.header("Customer Information")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0, value=70)
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No"])
payment_method = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

tenure = st.number_input("Tenure (months)", 0, 72)
monthly_charges = st.number_input("Monthly Charges", 0.0)

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

if st.button("Predict"):

    # Create input dataframe
    input_data = pd.DataFrame([{
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "Contract": contract,
        "InternetService": internet
    }])

    # Apply one-hot encoding (same as training)
    input_data = pd.get_dummies(input_data)

    # Add missing columns with 0
    for col in feature_names:
        if col not in input_data.columns:
            input_data[col] = 0

    # Ensure same column order as training
    input_data = input_data[feature_names]

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("Customer is likely to churn ❌")
    else:
       st.success("Customer is likely to stay ✅")
    prob = model.predict_proba(input_scaled)[0][1]
    st.write(f"Churn Probability: {prob:.2f}")
 

