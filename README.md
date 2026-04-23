# 🚇 NYC Subway Delay Predictor

> A Machine Learning System for Real-Time Transit Logistics

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![Model](https://img.shields.io/badge/Model-XGBoost-orange)](https://xgboost.readthedocs.io)
[![R²](https://img.shields.io/badge/R%C2%B2%20Score-0.885-brightgreen)]()
[![MAE](https://img.shields.io/badge/MAE-~3.7%20min-yellow)]()

**System Architects:** Nydia Takhellambam · Abhirohan Kashyap · Kuntal Maity

---

## 📖 Overview

The NYC Subway system is the world's 9th busiest rapid transit network — 5.5M+ daily riders, 472 stations, 245 miles of track, running 24/7. A single delayed train mathematically propagates disruption down every downstream stop. Predicting those cascades before they compound is an open, high-impact problem.

This project builds an end-to-end machine learning pipeline that:
1. **Collects** real-time MTA GTFS-RT feeds continuously over multiple days
2. **Merges and cleans** live vehicle positions against static schedule data
3. **Engineers temporal and sequential features** that mirror the physical mechanics of delay propagation
4. **Trains an XGBoost model** that achieves **R² = 0.885** and a mean error of **~3.7 minutes**
5. **Deploys predictions** inside an interactive real-time monitoring dashboard

---

## 📊 Model Performance

| Metric | Value | Interpretation |
|---|---|---|
| **R² Score** | 0.885 | Explains 88.5% of delay variance |
| **MAE** | 224 seconds | ~3.7 min average prediction error |
| **RMSE** | 467 seconds | Penalised error for large misses |
| **Train/Test Split** | 80/20 chronological | No data leakage |
| **Training Samples** | 240,000 | Drawn from 37.6M row dataset |

> **Why XGBoost?** Chosen over Linear Regression and Decision Trees to capture complex, non-linear system relationships (bimodal rush-hour peaks, feature interactions between time-of-day and delay propagation) without overfitting.

---

## 🗂 Project Structure

```
nyc-subway-delay-predictor/
│
├── collector.py          # Real-time GTFS-RT data collection daemon
├── preprocess.py         # Data merging, cleaning & delay computation
├── features.py           # Feature engineering pipeline
├── train.py              # XGBoost model training & evaluation
├── dashboard/            # Interactive monitoring dashboard
│   └── app.py
│
├── data/
│   ├── raw/              # Raw GTFS-RT snapshots (collected over 4 days)
│   ├── static/           # Static GTFS schedule (stops, stop_times, trips)
│   └── processed/        # ML-ready dataset (300K sample)
│
├── models/
│   └── xgboost_delay.pkl # Trained model artifact
│
└── requirements.txt
```

---

## ⚙️ Pipeline Walkthrough

### 1. `collector.py` — Real-Time Data Collection

This script is a continuous polling daemon that hits the MTA's GTFS-Realtime API every **30 seconds** and saves vehicle position snapshots to disk.

**What it does:**
- Connects to the MTA GTFS-RT Protocol Buffer feed
- Polls train position data at 30-second intervals
- Captures `trip_id`, `stop_id`, `arrival_time`, `vehicle_id`, and `route_id` for each vehicle observation
- Appends records to a rolling CSV, accumulating data over time

**Key output:** After running for approximately **4 days**, the collector produced a raw dataset of **~500,000 rows** representing live vehicle observations across the NYC subway network.

**Why 4 days?** Capturing multiple weekdays and a weekend cycle ensures the dataset contains the full temporal rhythm of the system — morning peaks, evening peaks, overnight lulls, and weekend schedule variations.

```bash
python collector.py --interval 30 --output data/raw/gtfs_live.csv
```

---

### 2. `preprocess.py` — Merging, Cleaning & Delay Computation

This is the most complex stage of the pipeline. Raw transit data is inherently messy — ID mismatches, many-to-many joins, missing timestamps, and ghost train artefacts. This script resolves all of it.

**What it does:**

**Step 1 — Stop ID Normalisation**
The real-time feed uses numeric stop IDs (e.g., `06`, `07`) while the static schedule uses a different namespace (e.g., `101`, `103`). These are not the same stops with different formats — they are entirely different encoding schemes. The script identifies the **58 stop IDs that genuinely appear in both datasets** and joins exclusively on those, preventing false matches that would corrupt the delay computation.

**Step 2 — Solving the Merge Explosion**
A naive join on `stop_id` alone created **639.9 million rows** due to massive many-to-many relationships (multiple trips pass through the same stop). The fix: constrain the join to also require a match on `trip_id`, then apply a temporal proximity filter. This reduced the dataset from **639M → 37.6M rows** (94% reduction) while preserving only genuine trip-stop alignments.

**Step 3 — Computing the Target Variable**
The GTFS feed does not contain a "delay" field. Delay is computed mathematically:

```
delay (seconds) = actual_arrival_unix − scheduled_arrival_unix
```

A positive value = train arrived late. Negative = early arrival (also valid operational data).

**Step 4 — Outlier & Noise Removal**
Ghost trains, data feed glitches, and re-logged cancelled trips produce extreme delay values (e.g., a "6-hour delay") that are almost certainly artefacts. Valid delay values are filtered to the range **[-600, +3600] seconds**. Duplicate records from repeated GTFS-RT broadcasts and rows with missing arrival timestamps are also dropped.

**Step 5 — Timestamp Unification**
Static GTFS times are in `HH:MM:SS` format including times beyond 24:00 (e.g., `25:30:00` = 1:30 AM next day). These are converted to Unix timestamps relative to the service date to enable arithmetic subtraction.

**Final output shape:** `37,616,635 rows × 15 columns`

```bash
python preprocess.py \
  --realtime data/raw/gtfs_live.csv \
  --static data/static/ \
  --output data/processed/merged_clean.parquet
```

| Stage | Records | Note |
|---|---|---|
| Real-Time GTFS (raw) | 500,000 | 5 columns |
| Static GTFS Schedule | 502,755 | 5 columns |
| After naive merge | 639,900,928 | Many-to-many explosion |
| After delay filtering | 37,616,876 | 94.1% reduction |
| After feature engineering | 37,616,635 | 15 columns |
| Training subset (80%) | 240,000 | Chronological split |
| Test subset (20%) | 60,000 | Genuinely unseen future data |

---

### 3. `features.py` — Feature Engineering

Raw fields are not enough. This script constructs the three feature families that allow the model to understand the physics of the subway system.

**Family 1: Time-Based Features** *(captures rush-hour surges and weekly structure)*

| Feature | Type | Description |
|---|---|---|
| `hour` | Integer (0–23) | Hour of day at observation |
| `day_of_week` | Integer (0–6) | Monday=0 through Sunday=6 |
| `is_peak` | Binary | 1 if 7–9 AM or 5–7 PM |
| `is_weekend` | Binary | 1 if Saturday or Sunday |

**Family 2: Sequential Features** *(the core engine — models physical delay propagation)*

These are computed per trip, in chronological order. The dataset must be sorted by `[trip_id, actual_arrival_time]` before computation, or the shift operations will read from the wrong stops.

| Feature | Computation | Description |
|---|---|---|
| `prev_delay` | `delay.shift(1)` per trip | Delay at the immediately preceding stop |
| `rolling_delay` | `delay.rolling(3).mean()` per trip | Average delay over the last 3 stops |
| `cumulative_delay` | `delay.cumsum()` per trip | Total delay accumulated on this trip so far |

> **Why these matter:** EDA found that delay at stop *n* correlates with delay at stop *n+1* at **r > 0.75** across all lines. Delay is not random — it is a physics problem. `prev_delay` alone accounts for **31.8%** of model feature importance, and `rolling_delay` accounts for **38.9%**.

**Family 3: Behavioral / Structural**

| Feature | Description |
|---|---|
| `stop_sequence` | Position of this stop within the trip's route |
| `stop_id` | Encoded stop identifier |

NaN values at the start of each trip's sequence (where no previous stop exists) are filled with `0`, assuming trips begin without accumulated delay.

```bash
python features.py \
  --input data/processed/merged_clean.parquet \
  --output data/processed/features_300k.parquet \
  --sample 300000
```

---

### 4. `train.py` — Model Training & Evaluation

Trains the XGBoost regressor on the engineered feature set and evaluates it against a chronologically held-out test set.

**Why chronological split?** A random train/test split would allow the model to see future data during training, inflating test scores artificially. Using the **first 80% of records by time as training** and the **last 20% as test** mirrors real deployment: train on the past, evaluate on genuinely unseen future.

**Baseline comparison:**

| Model | MAE (s) | RMSE (s) | R² |
|---|---|---|---|
| Mean baseline | 1,047.8 | 1,210.1 | ~0.00 |
| Persistence baseline | ~250 | ~520 | ~0.82 |
| **XGBoost (ours)** | **224** | **467** | **0.885** |

**Feature importance (XGBoost):**

```
rolling_delay      ████████████████████████ 38.9%
prev_delay         ████████████████████     31.8%
stop_sequence      ███████                  12.2%
stop_id            ██████                   11.0%
hour               ███                       5.5%
timestamp          ██                        3.3%
cumulative_delay   █                         1.0%
is_peak            ▏                        <0.5%
```

> **Key insight:** The system's *immediate past state* (rolling and previous delay) dominates predictions far more than static schedule data or time-of-day. Delays are a physics problem: they propagate deterministically unless actively interrupted.

```bash
python train.py \
  --input data/processed/features_300k.parquet \
  --model-out models/xgboost_delay.pkl \
  --test-split 0.2
```

---

### 5. `dashboard/app.py` — Interactive Monitoring Dashboard

The trained model is deployed inside a real-time command center with four views:

- **Top bar:** Live system-wide average delay and on-time percentage
- **Map view:** Spatial visualization of severe, delayed, and on-time routes across all 5 boroughs
- **Left panel:** Interactive trip planner — enter origin, destination, and departure time to receive ML-predicted delay forecasts for your route
- **Right panel:** Live line-by-line health status showing average delay per line

```bash
cd dashboard
python app.py
# Open http://localhost:8050
```

---

## 📈 System Diagnostics

**Delay follows a bimodal daily rhythm:**
- Delays spike sharply during **morning rush (7–9 AM)** and **evening rush (4–6 PM)**
- The probability of a severe delay (>5 min) peaks at **0.73–0.74** during rush hours
- Even at 3 AM, severe delay probability never drops below **0.60**, indicating persistent infrastructural fragility beyond just volume stress

**Delay is not uniform across lines:**
- Specific lines absorb a disproportionate share of the network's failures
- Transfer-heavy stations (Times Square, Grand Central) show the highest dwell-time extensions
- Peak hours light up the entire network simultaneously, creating system-wide saturation

---

## 🔧 Setup & Installation

```bash
git clone https://github.com/nydia-takhel/nyc-subway-delay-predictor.git
cd nyc-subway-delay-predictor
pip install -r requirements.txt
```

**Requirements include:** `gtfs-realtime-bindings`, `pandas`, `numpy`, `xgboost`, `scikit-learn`, `plotly`, `dash`, `pyarrow`

**To run the full pipeline from scratch:**

```bash
# 1. Collect data (runs for N hours/days)
python collector.py --duration 96h --output data/raw/

# 2. Preprocess and merge
python preprocess.py

# 3. Engineer features and sample
python features.py

# 4. Train model
python train.py

# 5. Launch dashboard
python dashboard/app.py
```

---

## ⚠️ Limitations

- **Data scope:** Currently trained on a static 4-day snapshot (~37M rows, 300K sampled for training) rather than a continuously updating live feed.
- **Routing approximation:** Trip routing uses sequential stop approximation rather than full GTFS `stop_times.txt` rigid routing.
- **Weather exclusion:** Weather features were intentionally excluded. Historical hourly weather data was not available for alignment, and using simulated weather values would have introduced information leakage. Real weather integration remains a meaningful future addition.
- **Stop coverage:** Only 58 stop IDs could be reliably matched across the real-time and static feeds. Full network coverage requires a more complete ID normalisation layer.

---

## 🚀 Future Roadmap

- **Live MTA integration:** Connect directly to the real-time MTA API pipe for continuously updating dashboard states
- **LSTM upgrade:** Implement Long Short-Term Memory (LSTM) time-series models to capture extended temporal sequences beyond the 3-stop rolling window
- **Full stop coverage:** Expand the stop ID normalisation to match all 472 stations
- **Weather signal:** Integrate genuine historical hourly weather (rain, temperature, visibility) once archival data is accessible
- **Mobile UI:** A commuter-facing mobile interface with push alerts for high-delay predictions on saved routes

---

## 📚 References

- MTA GTFS-Realtime Developer Documentation: [api.mta.info](https://api.mta.info)
- Google GTFS Static Specification: [gtfs.org](https://gtfs.org)
- Chen & Guestrin (2016), *XGBoost: A Scalable Tree Boosting System*, KDD '16
- NYC MTA Open Data Portal: [new.mta.info/open-data](https://new.mta.info/open-data)
