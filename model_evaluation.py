import pandas as pd
import joblib
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# === Load model, scaler, metadata ===
model = joblib.load("models/final_model.pkl")
scaler = joblib.load("models/final_scaler.pkl")

with open("models/final_model_metadata.json", "r") as f:
    metadata = json.load(f)

features = metadata["features"]
target = metadata["target"]

# === Load dữ liệu test lại ===
data = pd.read_csv("data/GlobalTemperatures.csv")
data = data.dropna()

data['dt'] = pd.to_datetime(data['dt'])
data['year'] = data['dt'].dt.year
data['month'] = data['dt'].dt.month
data['day_of_year'] = data['dt'].dt.dayofyear

X = data[features]
y = data[target]

X_scaled = scaler.transform(X)
y_pred = model.predict(X_scaled)

# === Evaluation ===
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

print("🔎 Model Evaluation Results")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2:   {r2:.4f}")
