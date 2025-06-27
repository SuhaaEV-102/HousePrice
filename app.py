# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved model, scaler, and features
model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

st.title("🏠 House Price Prediction App")

# Feature selection
selected_features = st.multiselect("Select Features to Input", features, default=features)

# Dynamic input fields
user_input = {}
for feat in selected_features:
    user_input[feat] = st.number_input(f"Enter {feat}", step=1.0, format="%.2f")

# Predict button
if st.button("Predict Price"):
    # Convert to DataFrame
    input_df = pd.DataFrame([user_input])
    
    # Fill missing features with zero
    for col in features:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Reorder columns
    input_df = input_df[features]

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    st.success(f"🏷️ Predicted House Price: ${prediction:,.2f}")
