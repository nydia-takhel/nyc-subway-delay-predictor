import pandas as pd
import os

BASE_PATH = "/home/bel/Desktop/nydia/NYC subway delay"

df = pd.read_csv(os.path.join(BASE_PATH, "cleaned_data.csv"))

# ==============================
# FEATURE ENGINEERING
# ==============================

# time-based
df['hour'] = df['scheduled_sec'] // 3600
df['is_peak'] = df['hour'].isin([7,8,9,16,17,18]).astype(int)

# sequential features
df = df.sort_values(['trip_id_x', 'stop_sequence'])

df['prev_delay'] = df.groupby('trip_id_x')['delay'].shift(1)
df['rolling_delay'] = df.groupby('trip_id_x')['delay'].rolling(3).mean().reset_index(0, drop=True)
df['cumulative_delay'] = df.groupby('trip_id_x')['delay'].cumsum()

df = df.dropna()

# ==============================
# SAVE
# ==============================
output_path = os.path.join(BASE_PATH, "features_data.csv")
df.to_csv(output_path, index=False)

print("✅ Features ready:", df.shape)
