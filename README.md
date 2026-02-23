# 📊 Telecom Customer Churn Prediction Dashboard

An **interactive Machine Learning web application** that predicts customer churn risk using a trained classification model and provides real-time analytics through a modern Streamlit dashboard.

---

## 🚀 Live Demo

🔗 **[Open the deployed app](https://churn-intelligence-sys.streamlit.app/)**

---

## 📌 Project Overview

Customer churn — when customers stop using a service — is a major issue for telecom companies.  
This project builds an **ML-powered dashboard** to predict churn risk and help businesses understand and act before losing customers.

The application includes:

- ✅ Single customer churn prediction
- ✅ Batch CSV upload for multiple predictions
- ✅ Visual probability chart
- ✅ Downloadable prediction results

---

## 🧠 Machine Learning Model

The churn prediction model is trained using:

- **Logistic Regression** (selected as the best performing model)
- Data preprocessing includes:
  - Handling missing values
  - One-hot encoding of categorical variables
  - Feature scaling

### Model Evaluation (Example Scores)

| Metric        | Score      |
|---------------|------------|
| Accuracy      | 81.97%     |
| Precision     | 68.3%      |
| Recall        | 59.5%      |
| F1 Score      | 63.6%      |

*(Replace with your final actual scores if different)*

---

## 🛠️ Tech Stack

- 🐍 **Python**
- 🎨 **Streamlit**
- 📊 **Scikit-Learn**
- 📈 **Matplotlib**
- 📁 **Joblib**
- 📍 **GitHub**
- ☁️ **Streamlit Community Cloud (Deployment)**

---

## 📂 Project Structure
customer-churn-app/ │ ├── app.py                  
# Streamlit UI + deployment ├── requirements.txt        
# Dependencies list ├── model/ │   ├── churn_model.pkl     
# Saved trained model │   ├── scaler.pkl          
# Saved scaler │   └── feature_names.pkl   
# Saved feature names └── README.md               
# Project documentation
---

## 🧩 Features

### 🔹 Single Customer Prediction
Enter customer details (tenure, contract type, charges, etc.)  
➡ Predict churn likelihood  
➡ See probability and chart visualization  

### 🔹 Batch Prediction
Upload a CSV file of customers  
➡ Generate churn predictions for all  
➡ Download results as a new CSV

---

## 🚀 Deployment

To **run locally**:

```bash
pip install -r requirements.txt
streamlit run app.py
```
💡 Future Enhancements
Add feature importance charts
Add customer segmentation analytics
Convert model to a REST API (FastAPI)
Connect to a true frontend (React / Tailwind)
Deploy entire system with backend + frontend

👩‍💻 Author
Nidhi
Aspiring Data Scientist | Machine Learning Enthusiast
