import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("models/final_model.pkl")
scaler = joblib.load("models/final_scaler.pkl")

# Load metadata để lấy đúng features
import json
with open("models/final_model_metadata.json") as f:
    metadata = json.load(f)
feature_names = metadata["features"]

st.title("🌡️ Land Average Temperature Prediction")

# Tạo input form
inputs = {}
for feat in feature_names:
    inputs[feat] = st.number_input(f"Nhập {feat}:", value=0.0)

# Khi bấm nút Predict
if st.button("Dự đoán"):
    sample = pd.DataFrame([list(inputs.values())], columns=feature_names)
    sample_scaled = scaler.transform(sample)
    prediction = model.predict(sample_scaled)[0]
    st.success(f"🌍 Dự đoán nhiệt độ trung bình đất: {prediction:.2f} °C")
