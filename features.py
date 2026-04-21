import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv("features_data.csv")

# use smaller sample for speed
df = df.sample(300000, random_state=42)

print("📥 Dataset shape:", df.shape)

# ==============================
# SORT (TIME ORDER)
# ==============================
df = df.sort_values('actual_sec')

# ==============================
# TARGET
# ==============================
y = df['delay']

# ==============================
# REMOVE LEAKAGE FEATURES
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

# keep only numeric features
X = X.select_dtypes(include=[np.number])

print("📊 Features used:", X.columns.tolist())

# ==============================
# TRAIN-TEST SPLIT (TIME BASED)
# ==============================
split = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

# ==============================
# BASELINE MODEL
# ==============================
print("\n📉 Baseline (Mean Prediction)")

baseline = np.full(len(y_test), y_train.mean())

print("MAE:", mean_absolute_error(y_test, baseline))
print("RMSE:", np.sqrt(mean_squared_error(y_test, baseline)))
print("R2:", r2_score(y_test, baseline))

# ==============================
# XGBOOST MODEL
# ==============================
print("\n🚀 Training XGBoost...")

model = XGBRegressor(
    n_estimators=20,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train, y_train)

preds = model.predict(X_test)

# ==============================
# EVALUATION
# ==============================
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print("\n📊 XGBoost Results:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)

# ==============================
# SAVE RESULTS
# ==============================
results_df = pd.DataFrame([["XGBoost", mae, rmse, r2]],
                          columns=["Model", "MAE", "RMSE", "R2"])

results_df.to_csv("model_results.csv", index=False)

print("\n🏆 Final Model: XGBoost")
print(results_df)
