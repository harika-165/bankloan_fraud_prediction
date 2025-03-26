#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Load dataset
df = pd.read_csv("Fraud.csv")

# Drop unnecessary columns
df = df.drop(["nameOrig", "nameDest"], axis=1, errors="ignore")

# Encode categorical variables
categorical_cols = ["type"]
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save encoders

# Convert incorrect data types
numerical_cols = ["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "isFlaggedFraud"]

for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")  # Convert invalid strings to NaN

# Drop rows with NaN values in numerical columns
df = df.dropna(subset=numerical_cols)

# Scale numerical variables
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Ensure target variable does not contain NaN values
df = df.dropna(subset=["isFraud"])

# Prepare training data
X = df.drop(columns=["isFraud"])  # Remove target from input features
y = df["isFraud"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the model, scaler, and encoder
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "encoder.pkl")

print("✅ Model, scaler, and encoder saved successfully!")


# In[ ]:




