import pandas as pd
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime
import os
import numpy as np

# === Load dữ liệu ===
data = pd.read_csv("data/GlobalTemperatures.csv")

# Xử lý missing
data = data.dropna()

# Feature engineering
data['dt'] = pd.to_datetime(data['dt'])
data['year'] = data['dt'].dt.year
data['month'] = data['dt'].dt.month
data['day_of_year'] = data['dt'].dt.dayofyear

features = ["year", "month", "day_of_year",
            "LandMaxTemperature", "LandMinTemperature", "LandAndOceanAverageTemperatureUncertainty"]
target = "LandAverageTemperature"

X = data[features]
y = data[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Scaling ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Train model Ridge Regression ===
best_params = {"alpha": 1.0}
model = Ridge(**best_params)
model.fit(X_train_scaled, y_train)

# === Evaluation ===
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# === Save model, scaler, metadata ===
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/final_model.pkl")
joblib.dump(scaler, "models/final_scaler.pkl")

metadata = {
    "best_model": "Ridge Regression",
    "best_params": best_params,
    "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "features": features,
    "target": target,
    "performance": {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }
}
with open("models/final_model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("✅ Model trained and saved successfully!")
print(f"MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
