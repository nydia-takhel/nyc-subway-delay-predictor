import pandas as pd
import numpy as np
import json
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv("features_data.csv")
df = df.sample(300000, random_state=42)

print("Dataset shape:", df.shape)

# ==============================
# SORT (TIME ORDER)
# ==============================
df = df.sort_values('actual_sec')

# ==============================
# REMOVE LEAKAGE COLUMNS (CRITICAL FIX)
# ==============================
leakage_cols = [
    'delay',              # target
    'trip_id_x',
    'trip_id_y',
    'arrival_time_x',
    'arrival_time_y',
    'datetime'
]

# Keep only existing columns
leakage_cols = [col for col in leakage_cols if col in df.columns]

# ==============================
# TARGET
# ==============================
y = df['delay']

# ==============================
# FEATURES (SAFE)
# ==============================
drop_cols = [
    'delay',
    'actual_sec',
    'scheduled_sec',
    'arrival_time_x',
    'arrival_time_y',
    'datetime',
    'trip_id_x',
    'trip_id_y'
]

drop_cols = [col for col in drop_cols if col in df.columns]

X = df.drop(columns=drop_cols)

# keep only numeric
X = X.select_dtypes(include=[np.number])

# ==============================
# SPLIT (TIME BASED)
# ==============================
split = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

# ==============================
# BASELINE
# ==============================
print("\n📉 Baseline: Mean")

baseline = np.full(len(y_test), y_train.mean())

print("MAE:", mean_absolute_error(y_test, baseline))
print("RMSE:", np.sqrt(mean_squared_error(y_test, baseline)))
print("R2:", r2_score(y_test, baseline))

# ==============================
# MODELS
# ==============================
models = {
    "XGBoost": XGBRegressor(n_estimators=20, max_depth=5),
}

results = []

# ==============================
# TRAIN + EVALUATE
# ==============================
for name, model in models.items():
    print(f"\n🚀 Training {name}...")

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    results.append([name, mae, rmse, r2])

    print(f"{name}:")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2:", r2)

# ==============================
# RESULTS TABLE
# ==============================
results_df = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "R2"])

print("\n📊 FINAL MODEL COMPARISON")
print(results_df.sort_values(by="RMSE"))

# ==============================
# SAVE MODEL
# ==============================
joblib.dump(model, "xgboost_model.pkl")
print("✅ Model saved: xgboost_model.pkl")

# ==============================
# EXTRACT FEATURE IMPORTANCE
# ==============================
# Get feature names (in order)
feature_names = X.columns.tolist()
print(f"\nFeatures used: {feature_names}")

# Get importance from XGBoost
importance_scores = model.get_booster().get_score(importance_type='weight')

# Create a dict with feature names
feature_importance = {}
for feature in feature_names:
    feature_importance[feature] = importance_scores.get(feature, 0)

# Normalize to percentages
total_importance = sum(feature_importance.values())
if total_importance > 0:
    feature_importance = {
        k: round((v / total_importance) * 100, 1) 
        for k, v in feature_importance.items()
    }
else:
    # Fallback if no importance scores (shouldn't happen)
    feature_importance = {feat: round(100/len(feature_names), 1) for feat in feature_names}

# Sort by importance (descending)
feature_importance = dict(sorted(
    feature_importance.items(), 
    key=lambda x: x[1], 
    reverse=True
))

print("\n🧠 Feature Importance:")
for feat, imp in feature_importance.items():
    print(f"  {feat}: {imp}%")

# Save to JSON
with open('feature_importance.json', 'w') as f:
    json.dump(feature_importance, f, indent=2)

print("✅ Feature importance saved: feature_importance.json")

# ==============================
# SAVE RESULTS
# ==============================
results_df.to_csv("model_results.csv", index=False)
print("✅ Results saved: model_results.csv")

# ==============================
# PRINT FINAL SUMMARY
# ==============================
print("\n" + "="*60)
print("🎯 TRAINING COMPLETE")
print("="*60)
print(f"\n📊 Best Model: {results_df.iloc[0]['Model']}")
print(f"   R² Score: {results_df.iloc[0]['R2']:.4f}")
print(f"   MAE: {results_df.iloc[0]['MAE']:.2f} seconds")
print(f"   RMSE: {results_df.iloc[0]['RMSE']:.2f} seconds")
print(f"\n📁 Files generated:")
print(f"   • xgboost_model.pkl (trained model)")
print(f"   • feature_importance.json (feature weights)")
print(f"   • model_results.csv (metrics)")
print("="*60 + "\n")
