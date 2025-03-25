#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("Fraud.csv")

# Reduce dataset size for faster training
data = data.sample(100000, random_state=42)

# Encode categorical features
label_encoder = LabelEncoder()
data['type'] = label_encoder.fit_transform(data['type'])

# Feature selection
X = data.drop(columns=['isFraud', 'nameOrig', 'nameDest'])
y = data['isFraud']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train optimized model
model = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as model.pkl")


# In[ ]:




