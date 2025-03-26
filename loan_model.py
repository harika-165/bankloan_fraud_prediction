#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Load the dataset
df = pd.read_csv('Fraud.csv')

# Drop unnecessary columns
df = df.drop(columns=['nameOrig', 'nameDest'], errors='ignore')

# Identify categorical and numerical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

# Encode categorical columns
label_encoders = {}  # Dictionary to store encoders
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le  # Save the encoder for later use

# Scale numerical columns
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Drop NaN values in the target column
df = df.dropna(subset=['isFraud'])

# Convert target to integer
df['isFraud'] = df['isFraud'].astype(int)

# Split data
X = df.drop(columns=['isFraud'])
y = df['isFraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model and preprocessors
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'encoder.pkl')

print("✅ Model and preprocessors saved successfully!")


# In[ ]:




