#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# app.py
import flask
from flask import Flask, request, render_template
import pandas as pd
import joblib  # For loading the model
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

app = Flask(__name__)

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

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        df = pd.DataFrame([data])
        df = preprocess_input(df)
        prediction = model.predict(df)
        result = "Fraud Detected" if prediction[0] == 1 else "No Fraud Detected"
        return render_template('result.html', result=result)

    except Exception as e:
        return render_template('result.html', result=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=False) #debug=False for Render

