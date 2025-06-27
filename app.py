import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and data tools
model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

# Load dataset to get valid zipcode values
df = pd.read_csv("kc_house_data.csv")
df = df.drop(columns=['id', 'date'])

# UI Title
st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🏠 House Price Prediction App")
st.markdown("Enter the property details below to estimate the house price.")

# --- Layout: Columns for inputs ---
col1, col2 = st.columns(2)

with col1:
    bedrooms = st.slider("🛏 Bedrooms", 1, 10, 3)
    bathrooms = st.slider("🛁 Bathrooms", 1, 5, 2, step=1)
    sqft_living = st.number_input("📐 Living Area (sqft)", 300, 10000, 1500)
    floors = st.selectbox("🏢 Number of Floors", [1, 2, 3, 4])
    yr_built = st.slider("📅 Year Built", 1900, 2022, 2000)

with col2:
    grade = st.slider("🏗️ Construction Grade (1 = Low, 13 = High)", 1, 13, 7)
    view = st.selectbox("👀 View Quality", [0, 1, 2, 3, 4], format_func=lambda x: ["None", "Poor", "Average", "Good", "Excellent"][x])
    waterfront = st.radio("🌊 Waterfront Property", ["No", "Yes"])
    zipcode = st.selectbox("📍 Zipcode", sorted(df["zipcode"].unique()))

# Map categorical input
waterfront_val = 1 if waterfront == "Yes" else 0

# Build input dictionary
input_data = {
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'sqft_living': sqft_living,
    'floors': floors,
    'grade': grade,
    'view': view,
    'waterfront': waterfront_val,
    'zipcode': zipcode,
    'yr_built': yr_built
}

# Fill in other unused features with 0
for feat in features:
    if feat not in input_data:
        input_data[feat] = 0

# Arrange inputs into model format
input_df = pd.DataFrame([input_data])
input_df = input_df[features]
input_scaled = scaler.transform(input_df)

# Predict
if st.button("💰 Predict House Price"):
    with st.spinner("Predicting..."):
        prediction = model.predict(input_scaled)[0]
        st.success(f"🏷️ Estimated House Price: **${prediction:,.2f}**")
