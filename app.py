#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import joblib
import os

# File paths
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "encoder.pkl"

# Check if required files exist
if not os.path.exists(MODEL_PATH):
    st.error("⚠️ Model file `model.pkl` not found! Please upload it to the project directory.")
    st.stop()

if not os.path.exists(SCALER_PATH) or not os.path.exists(ENCODER_PATH):
    st.warning("⚠️ Some preprocessing files (`scaler.pkl`, `encoder.pkl`) are missing. Predictions may be inaccurate.")

# Load model & preprocessors
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
label_encoders = joblib.load(ENCODER_PATH) if os.path.exists(ENCODER_PATH) else {}

def preprocess_input(df):
    """Preprocess input data using pre-trained encoder and scaler."""
    df = df.drop(columns=["nameDest"], errors="ignore")  # Drop unnecessary columns

    categorical_cols = ["type"]
    numerical_cols = ["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "isFlaggedFraud"]

    # Encode categorical variables
    for col in categorical_cols:
        if col in label_encoders:
            df[col] = df[col].map(lambda x: label_encoders[col].transform([x])[0] if x in label_encoders[col].classes_ else -1)

    # Scale numerical variables
    if scaler:
        df[numerical_cols] = scaler.transform(df[numerical_cols])

    return df

# Streamlit UI
st.set_page_config(page_title="Loan Fraud Detection", page_icon="💰")
st.title("🔍 Loan Fraud Detection System")

st.markdown("""
This application predicts whether a loan transaction is fraudulent or not.  
Please enter the transaction details below:
""")

with st.form("loan_form"):
    step = st.number_input("📌 Step (Time Step)", min_value=0, value=1)
    transaction_type = st.selectbox("💳 Transaction Type", ["PAYMENT", "TRANSFER", "CASH OUT", "DEBIT", "CASH IN"])
    amount = st.number_input("💰 Amount", min_value=0.0, value=100.0)
    oldbalanceOrg = st.number_input("🏦 Old Balance (Origin)", min_value=0.0, value=0.0)
    newbalanceOrig = st.number_input("🏦 New Balance (Origin)", min_value=0.0, value=0.0)
    oldbalanceDest = st.number_input("🏦 Old Balance (Destination)", min_value=0.0, value=0.0)
    newbalanceDest = st.number_input("🏦 New Balance (Destination)", min_value=0.0, value=0.0)
    isFlaggedFraud = st.selectbox("🚩 Is Flagged Fraud?", [0, 1])

    submit_button = st.form_submit_button("🔍 Predict")

if submit_button:
    input_data = {
        "step": step,
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        "isFlaggedFraud": isFlaggedFraud,
    }

    df = pd.DataFrame([input_data])

    try:
        df = preprocess_input(df)
        prediction = model.predict(df)
        result = "🚨 Fraud Detected!" if prediction[0] == 1 else "✅ No Fraud Detected."
        st.success(f"Prediction: {result}")

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")

