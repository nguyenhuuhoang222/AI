import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 1. Load và khám phá dữ liệu
print("🔍 Đang load và phân tích dữ liệu...")
df = pd.read_csv('data/GlobalTemperatures.csv')

print("📊 Thông tin dataset:")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Date range: {df['dt'].min()} to {df['dt'].max()}")
print("\n" + "="*50)

# 2. Tiền xử lý dữ liệu nâng cao
def preprocess_data(df):
    # Tạo bản copy
    df_clean = df.copy()
    
    # Chuyển đổi ngày tháng
    df_clean['dt'] = pd.to_datetime(df_clean['dt'])
    df_clean['year'] = df_clean['dt'].dt.year
    df_clean['month'] = df_clean['dt'].dt.month
    df_clean['day_of_year'] = df_clean['dt'].dt.dayofyear
    
    # Chọn features quan trọng
    features = ['year', 'month', 'day_of_year', 
                'LandAverageTemperatureUncertainty',
                'LandMaxTemperature', 'LandMinTemperature']
    
    # Target variable
    target = 'LandAverageTemperature'
    
    # Lọc dữ liệu
    df_clean = df_clean[features + [target]].copy()
    
    # Xử lý missing values
    df_clean.dropna(inplace=True)
    
    # Loại bỏ outliers (ngoại lai)
    Q1 = df_clean[target].quantile(0.25)
    Q3 = df_clean[target].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df_clean = df_clean[(df_clean[target] >= lower_bound) & 
                       (df_clean[target] <= upper_bound)]
    
    return df_clean, features, target

df_clean, features, target = preprocess_data(df)

print("✅ Đã xử lý xong dữ liệu")
print(f"Số lượng mẫu sau cleaning: {len(df_clean)}")
print(f"Features sử dụng: {features}")
print(f"Target: {target}")

# 3. Chuẩn bị dữ liệu cho training
X = df_clean[features]
y = df_clean[target]

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, shuffle=False
)

print(f"\n📈 Kích thước dữ liệu:")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# 4. Định nghĩa các models để thử nghiệm
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
    'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1),
    'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), 
                                  max_iter=1000, random_state=42)
}

# 5. Train và đánh giá các models
results = {}
best_model = None
best_score = float('inf')

print("\n🚀 Đang train các models...")
print("-" * 60)

for name, model in models.items():
    print(f"Training {name}...")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Dự đoán
    y_pred = model.predict(X_test)
    
    # Tính metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores.mean())
    
    # Lưu kết quả
    results[name] = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'CV_RMSE': cv_rmse,
        'model': model
    }
    
    print(f"  MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    
    # Chọn model tốt nhất dựa trên RMSE
    if rmse < best_score:
        best_score = rmse
        best_model = name

print("\n" + "="*60)
print(f"🏆 BEST MODEL: {best_model} (RMSE: {best_score:.4f})")
print("="*60)

# 6. So sánh kết quả các models
print("\n📊 BẢNG SO SÁNH HIỆU SUẤT:")
print("-" * 80)
print(f"{'Model':25} {'MAE':8} {'RMSE':8} {'R²':8} {'CV RMSE':10}")
print("-" * 80)

for name, metrics in results.items():
    print(f"{name:25} {metrics['MAE']:8.4f} {metrics['RMSE']:8.4f} "
          f"{metrics['R2']:8.4f} {metrics['CV_RMSE']:10.4f}")

# 7. Lưu model tốt nhất và scaler
best_model_obj = results[best_model]['model']

with open('models/best_model.pkl', 'wb') as f:
    pickle.dump(best_model_obj, f)

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Lưu metadata
metadata = {
    'best_model': best_model,
    'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'features': features,
    'target': target,
    'performance': results[best_model]
}

with open('models/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)

print(f"\n💾 Đã lưu model tốt nhất: {best_model}")
print(f"📁 Model saved: models/best_model.pkl")
print(f"📁 Scaler saved: models/scaler.pkl")
print(f"📁 Metadata saved: models/model_metadata.json")

# 8. Visualization kết quả
print("\n📈 Đang tạo biểu đồ đánh giá...")

# Biểu đồ so sánh hiệu suất
plt.figure(figsize=(12, 8))
models_list = list(results.keys())
rmse_scores = [results[m]['RMSE'] for m in models_list]

plt.barh(models_list, rmse_scores, color='skyblue')
plt.xlabel('RMSE Score')
plt.title('So sánh hiệu suất các models (RMSE)')
plt.tight_layout()
plt.savefig('models/model_comparison.png', dpi=300, bbox_inches='tight')

# Dự đoán vs Thực tế
y_pred_best = best_model_obj.predict(X_test)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_best, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Temperature (°C)')
plt.ylabel('Predicted Temperature (°C)')
plt.title(f'Actual vs Predicted - {best_model}')
plt.tight_layout()
plt.savefig('models/prediction_vs_actual.png', dpi=300, bbox_inches='tight')

print("✅ Đã tạo xong biểu đồ đánh giá")
print("🎉 Quá trình training hoàn tất!")

# 9. Demo dự đoán
print("\n🔮 DEMO DỰ ĐOÁN:")
print("-" * 30)

# Lấy dữ liệu mới nhất để demo
latest_data = X_test[-1:].copy()
prediction = best_model_obj.predict(latest_data)[0]
actual = y_test.iloc[-1]

print(f"Dự đoán nhiệt độ: {prediction:.2f}°C")
print(f"Thực tế: {actual:.2f}°C")
print(f"Sai số: {abs(prediction - actual):.2f}°C")