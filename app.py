import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ---------------- PAGE SETTINGS ---------------- #
st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# ---------------- MODERN CSS ---------------- #
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

.block-container {
    padding-top: 2rem;
}

.result-card {
    padding: 25px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    background: rgba(255, 255, 255, 0.08);
    box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
    text-align: center;
}

.stButton>button {
    background: linear-gradient(to right, #ff416c, #ff4b2b);
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}

.stDownloadButton>button {
    background: linear-gradient(to right, #00b09b, #96c93d);
    color: white;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ---------------- #
feature_names = joblib.load("model/feature_names.pkl")
model = joblib.load("model/churn_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# ---------------- HEADER ---------------- #
st.markdown("<h1 style='text-align:center;'>📊 Telecom Customer Churn Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>AI Powered Customer Risk Analysis System</h4>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- SINGLE CUSTOMER PREDICTION ---------------- #
st.subheader("🧍 Single Customer Analysis")

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

with col2:
    monthly_charges = st.slider("Monthly Charges", 0.0, 150.0, 50.0)
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

if st.button("🚀 Predict Churn Risk"):

    input_data = pd.DataFrame([{
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "Contract": contract,
        "InternetService": internet
    }])

    input_data = pd.get_dummies(input_data)

    for col in feature_names:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[feature_names]
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]

    st.markdown("---")

    colA, colB = st.columns(2)

    # -------- RESULT CARD -------- #
    with colA:
        if prediction[0] == 1:
            st.markdown(f"""
            <div class="result-card">
                <h2 style='color:#ff4b4b;'>⚠️ High Risk of Churn</h2>
                <h3>Probability: {probability*100:.2f}%</h3>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card">
                <h2 style='color:#00ff99;'>✅ Customer Likely to Stay</h2>
                <h3>Probability: {(1-probability)*100:.2f}%</h3>
            </div>
            """, unsafe_allow_html=True)

    


# ---------------- BATCH PREDICTION ---------------- #
st.markdown("---")
st.subheader("📂 Batch Prediction (Upload CSV File)")

uploaded_file = st.file_uploader("Upload customer CSV file", type=["csv"])

if uploaded_file is not None:

    df_upload = pd.read_csv(uploaded_file)

    st.markdown("### 🔍 Uploaded Data Preview")
    st.dataframe(df_upload.head())

    df_encoded = pd.get_dummies(df_upload)

    for col in feature_names:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    df_encoded = df_encoded[feature_names]
    df_scaled = scaler.transform(df_encoded)

    predictions = model.predict(df_scaled)
    probabilities = model.predict_proba(df_scaled)[:, 1]

    df_upload["Churn Prediction"] = predictions
    df_upload["Churn Probability"] = probabilities

    st.markdown("### 📊 Prediction Results")
    st.dataframe(df_upload)

    csv = df_upload.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇ Download Results",
        data=csv,
        file_name="churn_predictions.csv",
        mime="text/csv"
    )




