import pandas as pd
import numpy as np

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
# X = df.drop(columns=leakage_cols)
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
    # "Linear": LinearRegression(),
    # "Ridge": Ridge(alpha=1.0),
    # "Lasso": Lasso(alpha=0.1),
    # "RandomForest": RandomForestRegressor(n_estimators=20, max_depth=8),
    # "ExtraTrees": ExtraTreesRegressor(n_estimators=20),
    # "GradientBoosting": GradientBoostingRegressor(n_estimators=20),
    "XGBoost": XGBRegressor(n_estimators=20, max_depth=5),
    # "LightGBM": LGBMRegressor(n_estimators=20)
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

# SAVE RESULTS
results_df.to_csv("model_results.csv", index=False)

print("\n🏆 Best Model:")
print(results_df.sort_values(by="RMSE").iloc[0])

import joblib
joblib.dump(model, "xgboost_model.pkl")
