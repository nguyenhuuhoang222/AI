import joblib
import json
import numpy as np

# === Load model, scaler, metadata ===
model = joblib.load("models/final_model.pkl")
scaler = joblib.load("models/final_scaler.pkl")

with open("models/final_model_metadata.json", "r") as f:
    metadata = json.load(f)

features = metadata["features"]

def predict(input_dict):
    """
    input_dict: dict có keys giống 'features'
    Ví dụ:
    {
        "year": 2025,
        "month": 9,
        "day_of_year": 251,
        "LandMaxTemperature": 25.0,
        "LandMinTemperature": 15.0,
        "LandAndOceanAverageTemperatureUncertainty": 0.1
    }
    """
    x = np.array([[input_dict[f] for f in features]])
    x_scaled = scaler.transform(x)
    prediction = model.predict(x_scaled)[0]
    return prediction

# Demo
if __name__ == "__main__":
    sample_input = {
        "year": 2025,
        "month": 9,
        "day_of_year": 251,
        "LandMaxTemperature": 25.0,
        "LandMinTemperature": 15.0,
        "LandAndOceanAverageTemperatureUncertainty": 0.1
    }
    print("🌡️ Predicted LandAverageTemperature:", predict(sample_input))
