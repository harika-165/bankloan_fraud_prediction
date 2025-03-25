#!/usr/bin/env python
# coding: utf-8

# In[8]:


import streamlit as st
import pandas as pd
import pickle

st.title("Bank Loan Fraud Detection")

uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Loaded Successfully!")
else:
    st.warning("Please upload a dataset to proceed.")

# Load trained model
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

# Predict function
def predict_fraud(model, input_data):
    prediction = model.predict(input_data)
    return prediction

# Streamlit UI
st.set_page_config(page_title="Bank Loan Fraud Detection", layout="wide")
st.title("🚀 Bank Loan Fraud Detection System")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview:")
    st.dataframe(data.head())
    
    model = load_model()
    
    if st.button("Predict Fraud Cases"):
        predictions = predict_fraud(model, data.drop(columns=['isFraud']))
        data['Fraud Prediction'] = predictions
        st.write("### Prediction Results:")
        st.dataframe(data)
        
        fraud_count = data['Fraud Prediction'].sum()
        st.write(f"**Total Fraudulent Transactions Detected: {fraud_count}**")


# In[ ]:




