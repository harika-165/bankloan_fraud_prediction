#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the trained model
model = joblib.load('model.pkl')

def preprocess_input(df):
    """Preprocesses input data for prediction."""
    categorical_cols = df.select_dtypes(include=['object']).columns
    numerical_cols = df.select_dtypes(include=['number']).columns

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

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
    nameDest = st.text_input("Destination Account ID", value="M1234567890")
    oldbalanceDest = st.number_input("Old Balance (Destination)", min_value=0.0, value=0.0)
    newbalanceDest = st.number_input("New Balance (Destination)", min_value=0.0, value=0.0)
    isFlaggedFraud = st.number_input("Is Flagged Fraud (0 or 1)", min_value=0, max_value=1, value=0)

    submit_button = st.form_submit_button("Predict")

if submit_button:
    input_data = {
        "step": step,
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "nameDest": nameDest,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        "isFlaggedFraud": isFlaggedFraud,
    }

    df = pd.DataFrame([input_data])

    try:
        df = preprocess_input(df)
        prediction = model.predict(df)
        result = "Fraud Detected" if prediction[0] == 1 else "No Fraud Detected"
        st.success(f"Prediction: {result}")

    except Exception as e:
        st.error(f"An error occurred: {e}")

