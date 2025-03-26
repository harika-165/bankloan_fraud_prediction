#!/usr/bin/env python
# coding: utf-8

# In[13]:


import joblib
import os

def load_model():
    """Loads the trained model and preprocessing objects."""
    model_path = "model.pkl"
    scaler_path = "scaler.pkl"
    encoder_path = "encoder.pkl"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file 'model.pkl' not found!")
    
    model = joblib.load(model_path)
    
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    label_encoders = joblib.load(encoder_path) if os.path.exists(encoder_path) else {}
    
    return model, scaler, label_encoders

if __name__ == "__main__":
    try:
        model, scaler, label_encoders = load_model()
        print("✅ Model and preprocessing objects loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")


# In[ ]:




