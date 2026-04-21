# 🚇 NYC Subway Delay Prediction System

## 📌 Overview

This project builds an end-to-end machine learning system to predict NYC subway delays using real-time transit data and external weather conditions.

The system combines data engineering, feature engineering, and regression modeling to capture delay patterns and improve prediction accuracy.

---

## 🎯 Problem Formulation

We model subway delay prediction as a regression problem:

y = f(X)

Where:

* X = input features (time, sequence, weather)
* y = predicted delay (in seconds)

---

## 📊 Data Sources

* MTA GTFS Realtime API (train updates)
* Static GTFS dataset (schedule data)
* Weather API (temperature, humidity, visibility)

---

## ⚙️ Project Pipeline

### 1. Data Collection

* Real-time subway data collected via API
* 30M+ records generated
* Stored trip-level and stop-level information

---

### 2. Data Preprocessing

* Removed duplicate records
* Fixed inconsistencies in `stop_id`
* Merged real-time data with scheduled GTFS data
* Created delay variable:

delay = actual_time − scheduled_time

* Removed extreme outliers:

  * delay < -600 sec
  * delay > 3600 sec

---

## 🧠 Feature Engineering

### ⏱ Temporal Features

* Hour of day
* Day of week
* Peak hour indicator
* Weekend indicator

---

### 🔁 Sequential Features

* Previous delay
* Rolling average delay
* Cumulative delay

Mathematically:

y_t ≈ f(y_{t-1}, X_t)

This captures delay propagation across stations.

---

### 🌦️ Weather Features

* Temperature
* Humidity
* Visibility
* Rain indicator

---

## 📐 Mathematical Modeling

### Objective

Learn a function:

f: X → y

that minimizes prediction error.

---

### Loss Functions

MAE = (1/n) Σ |yᵢ - ŷᵢ|

RMSE = √[(1/n) Σ (yᵢ - ŷᵢ)²]

---

### Evaluation Metric

R² Score:

R² = 1 - (Σ (yᵢ - ŷᵢ)² / Σ (yᵢ - ȳ)²)

---

## 🚨 Data Leakage Prevention

Instead of random splitting, we use:

Time-based split:

Training set = past data
Test set = future data

This ensures realistic evaluation.

---

## 🤖 Models Used

* Linear Regression
* Ridge / Lasso
* Random Forest
* Extra Trees
* Gradient Boosting
* XGBoost
* LightGBM

---

## 📊 Evaluation Strategy

Models are compared using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

Baseline models:

* Mean delay
* Previous delay (persistence model)

---

## 📁 Repository Structure

```
collector.py     → data collection  
preprocess.py    → cleaning & merging  
features.py      → feature engineering  
train.py         → model training  
weather.py       → weather integration  
data/            → sample dataset  
```

---

## 🔍 Key Insights

* Delay propagation is a strong predictor
* Temporal patterns significantly impact delays
* Tree-based models perform best on tabular data
* Weather adds real-world context

---

## 🚀 Future Work

* Integrate historical weather data
* Add spatial route features
* Deploy real-time prediction system
* Explore ensemble learning

---

## 🧑‍💻 Author

Nydia
