import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = "kc_house_data.csv"  # Ensure the file is in the same directory
df = pd.read_csv(file_path)

# Drop unnecessary columns
df_cleaned = df.drop(columns=['id', 'date'])

# Fill missing values with median
df_cleaned.fillna(df_cleaned.median(), inplace=True)

# Define features and target variable
X = df_cleaned.drop(columns=['price'])
y = df_cleaned['price']

# Normalize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Gradient Boosting model
gb_model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.1, max_depth=5, subsample=0.8, random_state=42)
gb_model.fit(X_train, y_train)
joblib.dump(gb_model, "model/house_price_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
# Make predictions
y_pred_gb = gb_model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred_gb)
rmse = mean_squared_error(y_test, y_pred_gb) ** 0.5
r2 = r2_score(y_test, y_pred_gb)

# Print results
print(f"MAE: ${mae:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"R² Score: {r2:.2%}")
