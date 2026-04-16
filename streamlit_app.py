import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("data/telco_churn.csv")

# -------------------------------
# LOAD MODEL
# -------------------------------
model = pickle.load(open("model/churn_model.pkl", "rb"))

# -------------------------------
# TITLE
# -------------------------------
st.title("📊 Customer Churn Prediction Dashboard")
st.write("Analyze customer behavior and predict churn risk")

# -------------------------------
# KPI SECTION
# -------------------------------
st.subheader("📈 Key Metrics")

col1, col2, col3 = st.columns(3)

total_customers = len(df)
churn_rate = df['Churn'].value_counts(normalize=True)[1] * 100 if 1 in df['Churn'].values else 0

col1.metric("Total Customers", total_customers)
col2.metric("Churn Rate (%)", f"{churn_rate:.2f}")
col3.metric("Avg Monthly Charges", f"{df['MonthlyCharges'].mean():.2f}")

# -------------------------------
# CHARTS
# -------------------------------
st.subheader("📊 Data Insights")

col1, col2 = st.columns(2)

with col1:
    st.write("Churn Distribution")
    st.bar_chart(df['Churn'].value_counts())

with col2:
    st.write("Contract Type Distribution")
    st.bar_chart(df['Contract'].value_counts())

st.write("Tenure Distribution")
st.line_chart(df['tenure'])

# -------------------------------
# PREDICTION SECTION
# -------------------------------
st.subheader("🤖 Predict Customer Churn")

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (months)", 0, 72)
    monthly_charges = st.number_input("Monthly Charges", value=50.0)

with col2:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

# -------------------------------
# SIMPLE ENCODING (adjust if needed)
# -------------------------------
contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}

contract_val = contract_map[contract]
internet_val = internet_map[internet]

# -------------------------------
# PREDICT BUTTON
# -------------------------------
if st.button("Predict Churn"):

    # ⚠️ Adjust number of features based on your model
    input_data = np.array([[tenure, monthly_charges, contract_val, internet_val]])

    try:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.subheader("🔍 Prediction Result")

        if prediction == 1:
            st.error(f"⚠️ High Risk of Churn ({probability*100:.2f}%)")
        else:
            st.success(f"✅ Low Risk ({probability*100:.2f}%)")

    except Exception as e:
        st.error("⚠️ Prediction failed. Model input mismatch.")
        st.write(e)