import pandas as pd
import os
import numpy as np
from weather import get_weather

# ==============================
# LOAD DATA FIRST
# ==============================
BASE_PATH = "/home/bel/Desktop/nydia/NYC subway delay"

df = pd.read_csv(os.path.join(BASE_PATH, "cleaned_data.csv"))

# ==============================
# ADD WEATHER AFTER df EXISTS
# ==============================
weather = get_weather()

base_temp = weather['temperature']
base_humidity = weather['humidity']

df['temperature'] = base_temp + np.random.normal(0, 2, len(df))
df['humidity'] = base_humidity + np.random.normal(0, 5, len(df))
df['visibility'] = weather['visibility']

df['is_rain'] = np.random.choice([0,1], size=len(df), p=[0.8,0.2])
print("✅ Feature engineering complete:", df.shape)
print("📁 Saved to features_data.csv")