# train_model.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("kc_house_data.csv")

# Drop unnecessary columns
df = df.drop(columns=['id', 'date'])

# Define features and target
X = df.drop(columns=['price'])
y = df['price']

# Save feature names
features = X.columns.tolist()
joblib.dump(features, "features.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = GradientBoostingRegressor()
model.fit(X_train_scaled, y_train)

# Save model and scaler
joblib.dump(model, "house_price_model.pkl")
joblib.dump(scaler, "scaler.pkl")
