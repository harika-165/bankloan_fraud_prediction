#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import joblib
import os

# Load model & preprocessors
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "encoder.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("⚠️ Model file `model.pkl` not found!")
    st.stop()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
label_encoders = joblib.load(ENCODER_PATH) if os.path.exists(ENCODER_PATH) else {}

# Define feature order (must match training data)
FEATURE_COLUMNS = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig", 
                   "oldbalanceDest", "newbalanceDest", "isFlaggedFraud"]

def preprocess_input(df):
    """Preprocess input data to match the trained model format."""
    df = df[FEATURE_COLUMNS]  # Ensure correct column order

    # Encode categorical variables
    for col in ["type"]:
        if col in label_encoders:
            df[col] = df[col].map(lambda x: label_encoders[col].transform([x])[0] if x in label_encoders[col].classes_ else -1)

    # Scale numerical variables
    if scaler:
        df[df.select_dtypes(include=['number']).columns] = scaler.transform(df[df.select_dtypes(include=['number']).columns])

    return df

# Streamlit UI
st.set_page_config(page_title="Loan Fraud Detection", page_icon="💰")
st.title("🔍 Loan Fraud Detection System")

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
        df_processed = preprocess_input(df)
        prediction = model.predict(df_processed)
        isFraud = int(prediction[0])  # Convert prediction result to integer

        # Display results
        st.subheader("🔎 Prediction Result")
        st.write(f"**Transaction Type:** {transaction_type}")
        st.write(f"**Amount:** ${amount:,.2f}")
        st.write(f"**Old Balance (Origin):** ${oldbalanceOrg:,.2f}")
        st.write(f"**New Balance (Origin):** ${newbalanceOrig:,.2f}")
        st.write(f"**Old Balance (Destination):** ${oldbalanceDest:,.2f}")
        st.write(f"**New Balance (Destination):** ${newbalanceDest:,.2f}")
        st.write(f"**Flagged as Fraud?** {'✅ Yes' if isFlaggedFraud else '❌ No'}")

        # Show fraud detection result
        fraud_status = "🚨 Fraud Detected!" if isFraud == 1 else "✅ No Fraud Detected."
        st.success(f"**Prediction:** {fraud_status}")
        st.write(f"**isFraud (Output):** {isFraud}")

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")

