#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import joblib
import os

# Load the trained model and preprocessors
model_path = "model.pkl"
scaler_path = "scaler.pkl"
encoder_path = "encoder.pkl"

if not os.path.exists(model_path):
    st.error("Model file not found! Please upload `model.pkl` to the project directory.")
else:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    encoder = joblib.load(encoder_path) if os.path.exists(encoder_path) else None

def preprocess_input(df):
    """Preprocess input data using pre-trained encoder and scaler."""
    # Drop unnecessary columns
    df = df.drop(columns=["nameDest"], errors="ignore")

    categorical_cols = ["type"]
    numerical_cols = ["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "isFlaggedFraud"]

    # Encode categorical variables
    if encoder:
        df[categorical_cols] = encoder.transform(df[categorical_cols])

    # Scale numerical variables
    if scaler:
        df[numerical_cols] = scaler.transform(df[numerical_cols])

    return df

st.set_page_config(page_title="Loan Fraud Detection", page_icon="💰")

st.title("Loan Fraud Detection System")

st.markdown("""
This application predicts whether a loan transaction is fraudulent or not. Please enter the transaction details below:
""")

with st.form("loan_form"):
    step = st.number_input("Step (Time Step)", min_value=0, value=1)
    transaction_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH OUT", "DEBIT", "CASH IN"])
    amount = st.number_input("Amount", min_value=0.0, value=100.0)
    oldbalanceOrg = st.number_input("Old Balance (Origin)", min_value=0.0, value=0.0)
    newbalanceOrig = st.number_input("New Balance (Origin)", min_value=0.0, value=0.0)
    oldbalanceDest = st.number_input("Old Balance (Destination)", min_value=0.0, value=0.0)
    newbalanceDest = st.number_input("New Balance (Destination)", min_value=0.0, value=0.0)
    isFlaggedFraud = st.selectbox("Is Flagged Fraud?", [0, 1])

    submit_button = st.form_submit_button("Predict")

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
        st.error(f"An error occurred: {e}")

