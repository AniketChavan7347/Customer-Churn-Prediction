import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Churn Prediction", layout="wide")

# Title
st.title("📊 Customer Churn Prediction System")
st.markdown("Predict whether a customer will leave the service or stay.")

# Load & train model (only once)
@st.cache_resource
def load_model():
    data = pd.read_csv("Telco-Customer-Churn.csv")

    data.drop("customerID", axis=1, errors="ignore", inplace=True)

    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data["TotalCharges"].fillna(data["TotalCharges"].mean(), inplace=True)

    data["Churn"] = data["Churn"].replace({"Yes": 1, "No": 0})
    data = data[data["Churn"].isin([0, 1])]

    selected_cols = [
        "tenure","MonthlyCharges","TotalCharges","Contract",
        "InternetService","PaymentMethod","OnlineSecurity",
        "TechSupport","SeniorCitizen","Churn"
    ]

    data = data[selected_cols]
    data = pd.get_dummies(data, drop_first=True)

    X = data.drop("Churn", axis=1)
    y = data["Churn"]

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    model.fit(X, y)

    return model, X.columns

model, columns = load_model()

# Sidebar Inputs
st.sidebar.header("📝 Customer Details")

tenure = st.sidebar.slider("Tenure", 0, 72, 12)
monthly = st.sidebar.slider("Monthly Charges", 0, 150, 70)
total = st.sidebar.slider("Total Charges", 0, 10000, 2000)

contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
security = st.sidebar.selectbox("Online Security", ["Yes", "No"])
support = st.sidebar.selectbox("Tech Support", ["Yes", "No"])
senior = st.sidebar.selectbox("Senior Citizen", [0, 1])

# Input DataFrame
input_df = pd.DataFrame({
    "tenure":[tenure],
    "MonthlyCharges":[monthly],
    "TotalCharges":[total],
    "SeniorCitizen":[senior],
    "Contract":[contract],
    "InternetService":[internet],
    "PaymentMethod":[payment],
    "OnlineSecurity":[security],
    "TechSupport":[support]
})

input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=columns, fill_value=0)

# Prediction
if st.button("🔍 Predict"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.subheader("📈 Prediction Result")

    st.write(f"Churn Probability: {prob*100:.1f}%")  

    if prob > 0.5:
        st.error("⚠️ High Risk: Customer likely to Churn")
    else:
        st.success("✅ Low Risk: Customer will Stay")


 
# Insights Section
st.markdown("---")
st.subheader("📊 Key Insights")

st.write("""
- Customers with **month-to-month contracts** are more likely to churn  
- High **monthly charges** increase churn risk  
- Lack of **tech support & security** leads to higher churn  
""")