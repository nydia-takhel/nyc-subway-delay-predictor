import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv("features_data.csv")

print("Dataset shape:", df.shape)

# ==============================
# SORT (CRITICAL - NO LEAKAGE)
# ==============================
df = df.sort_values('actual_sec')

# ==============================
# TARGET + FEATURES
# ==============================
y = df['delay']

X = df.drop(columns=['delay'])

# Keep only numeric columns for ML
X = X.select_dtypes(include=[np.number])

# ==============================
# SPLIT (TIME BASED)
# ==============================
split = int(len(df) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

# ==============================
# BASELINE 1: MEAN
# ==============================
print("\n📉 Baseline: Mean")

baseline = np.full(len(y_test), y_train.mean())

print("MAE:", mean_absolute_error(y_test, baseline))
print("RMSE:", np.sqrt(mean_squared_error(y_test, baseline)))
print("R2:", r2_score(y_test, baseline))


# ==============================
# MODELS TO COMPARE
# ==============================
# models = {
#     "Linear Regression": LinearRegression(),
#     "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=10),
#     "XGBoost": XGBRegressor(n_estimators=50, max_depth=6),
#     "LightGBM": LGBMRegressor(n_estimators=50)
# }


models = {
    "Linear": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),
    "RandomForest": RandomForestRegressor(n_estimators=50, max_depth=10),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=50),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=50),
    "XGBoost": XGBRegressor(n_estimators=50, max_depth=6),
    "LightGBM": LGBMRegressor(n_estimators=50)
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
# FINAL COMPARISON TABLE
# ==============================
results_df = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "R2"])

print("\n📊 FINAL MODEL COMPARISON")
print(results_df.sort_values(by="RMSE"))

# OPTIONAL: SAVE RESULTS
results_df.to_csv("model_results.csv", index=False)
print("\nBest Model based on RMSE:")
print(results_df.sort_values(by="RMSE").iloc[0])