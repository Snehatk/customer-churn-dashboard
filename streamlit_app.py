import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("model/churn_model.pkl", "rb"))

st.set_page_config(page_title="Churn Dashboard", layout="centered")

st.title("📊 Customer Churn Prediction")
st.write("Predict whether a customer will churn")

# Inputs
tenure = st.slider("Tenure (months)", 0, 72)
monthly_charges = st.number_input("Monthly Charges")

if st.button("Predict"):
    data = np.array([[tenure, monthly_charges]])
    prediction = model.predict(data)[0]

    if prediction == 1:
        st.error("⚠️ Customer likely to churn")
    else:
        st.success("✅ Customer likely to stay")