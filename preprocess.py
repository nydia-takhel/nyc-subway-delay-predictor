import pandas as pd
import os

# ==============================
# BASE PATH
# ==============================
BASE_PATH = "/home/bel/Desktop/nydia/NYC subway delay"

# ==============================
# LOAD DATA
# ==============================
print("📥 Loading data...")

# df = pd.read_csv(os.path.join(BASE_PATH, "realtime_data.csv"))
df = pd.read_csv(
    os.path.join(BASE_PATH, "realtime_data.csv"),
    nrows=500000   # 🔥 LIMIT DATA
)
stop_times = pd.read_csv(os.path.join(BASE_PATH, "gtfs_subway/stop_times.txt"))

print("Realtime shape:", df.shape)
print("Stop_times shape:", stop_times.shape)

# ==============================
# CREATE ACTUAL TIME (REALTIME)
# ==============================
df['arrival_time'] = pd.to_datetime(df['arrival_time'], unit='s')

df['actual_sec'] = (
    df['arrival_time'].dt.hour * 3600 +
    df['arrival_time'].dt.minute * 60 +
    df['arrival_time'].dt.second
)

# ==============================
# CREATE SCHEDULED TIME (GTFS)
# ==============================
def time_to_sec(t):
    try:
        h, m, s = map(int, t.split(":"))
        return h*3600 + m*60 + s
    except:
        return None

stop_times['scheduled_sec'] = stop_times['arrival_time'].apply(time_to_sec)

# ==============================
# CLEAN stop_id (CRITICAL FIX)
# ==============================
print("\n🧹 Cleaning stop_id...")

# realtime → keep only numbers
df['stop_id'] = df['stop_id'].astype(str)
df['stop_id'] = df['stop_id'].str.extract(r'(\d+)')

# gtfs → keep only numbers
stop_times['stop_id'] = stop_times['stop_id'].astype(str)
stop_times['stop_id'] = stop_times['stop_id'].str.extract(r'(\d+)')

# drop bad rows
df = df.dropna(subset=['stop_id'])
stop_times = stop_times.dropna(subset=['stop_id'])

# ==============================
# CLEAN DATA
# ==============================
df = df.drop_duplicates(['trip_id', 'stop_id', 'arrival_time'])

# ==============================
# DEBUG CHECK
# ==============================
print("\n🔍 Checking matches...")

print("Realtime stop_id sample:", df['stop_id'].head())
print("GTFS stop_id sample:", stop_times['stop_id'].head())

common_ids = len(set(df['stop_id']) & set(stop_times['stop_id']))
print("Common stop_ids:", common_ids)

# ==============================
# MERGE DATA (ONLY ON stop_id)
# ==============================
print("\n🔗 Merging datasets...")

merged = df.merge(
    stop_times,
    on='stop_id',
    how='inner'
)

print("Merged shape:", merged.shape)

# ==============================
# STOP IF FAILED
# ==============================
if merged.empty:
    print("❌ ERROR: Merge failed — still no matching stop_ids")
    exit()

# ==============================
# CREATE TARGET (DELAY)
# ==============================
print("\n🎯 Creating delay...")

merged['delay'] = merged['actual_sec'] - merged['scheduled_sec']

# remove extreme noise
merged = merged[(merged['delay'] > -600) & (merged['delay'] < 3600)]

print("After filtering:", merged.shape)

# ==============================
# FEATURE ENGINEERING
# ==============================
print("\n⚙️ Creating features...")

merged['hour'] = merged['scheduled_sec'] // 3600

# peak hours (NYC typical)
merged['is_peak'] = merged['hour'].isin([7,8,9,16,17,18]).astype(int)



# sort for sequential features
# merged = merged.sort_values(['trip_id', 'stop_sequence'])

merged = merged.sort_values(['trip_id_x', 'stop_sequence'])

# previous delay
# merged['prev_delay'] = merged.groupby('trip_id')['delay'].shift(1)
merged['prev_delay'] = merged.groupby('trip_id_x')['delay'].shift(1)

# drop nulls
merged = merged.dropna()

print("Final dataset shape:", merged.shape)

# ==============================
# SAVE CLEAN DATA
# ==============================
output_path = os.path.join(BASE_PATH, "cleaned_data.csv")
merged.to_csv(output_path, index=False)

print(f"\n✅ Preprocessing complete: {merged.shape}")
print(f"💾 Saved to: {output_path}")

# ==============================
# FINAL SAMPLING (CRITICAL)
# ==============================
merged = merged.sample(500000, random_state=42)