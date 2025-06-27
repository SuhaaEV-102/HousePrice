House Price Prediction Web App - README
Overview
This project is a machine learning-powered web application that estimates the price of a house
based on user-provided property details. It uses a trained Gradient Boosting Regressor model and is
built with Streamlit for an interactive frontend.
Dataset
* Source: King County House Sales dataset (kc_house_data.csv)
* Target Variable: price (house sale price)
* Key Features: sqft_living, bedrooms, bathrooms, floors, grade, yr_built, zipcode, view, waterfront,
etc.
Model
* Algorithm: GradientBoostingRegressor (scikit-learn)
* Preprocessing: StandardScaler
* Training: Model trained on selected features, saved as house_price_model.pkl. Scaler and feature
list also saved for consistent input formatting.
Web App (Frontend)
* Framework: Streamlit
* UI: Sliders, dropdowns, radio buttons
* Features: Real-time prediction, friendly labels, formatted price output
* Deployment: Local or Streamlit Cloud
Files Included
* kc_house_data.csv - Dataset
* train_model.py - Model training script
* app.py - Streamlit web app
* house_price_model.pkl, scaler.pkl, features.pkl - Model artifacts
* requirements.txt - Dependency file
Outcome
Users can input property details and receive an instant estimate of the house price, making this a
practical tool for real estate analysis, education, or demos.
