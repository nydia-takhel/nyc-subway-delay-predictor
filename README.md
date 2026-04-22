# 🚇 NYC Subway Delay Predictor

## 📌 Overview

Urban transit systems often break down under real-world conditions—delays cascade, schedules become unreliable, and official estimates lose accuracy.

This project builds a machine learning system to **predict subway delays under irregular service conditions**, focusing on how delays propagate across time rather than relying solely on static schedules.

---

## 🎯 Problem Statement

Existing platforms (e.g., Metropolitan Transportation Authority and Google) primarily provide:

* real-time tracking
* schedule-based estimates

However, during disruptions:

* trains bunch together
* delays propagate unpredictably
* arrival estimates become unreliable

👉 This project addresses that gap by modeling **delay dynamics and sequential patterns**.

---

## 💡 Key Idea

Instead of asking:

> “What is the scheduled arrival time?”

This model asks:

> “Given what just happened, what will happen next?”

It leverages:

* temporal patterns (time of day, weekday/weekend)
* sequential dependencies (previous train delays, headways)

---

## ⚙️ Features

### ⏱ Temporal Features

* Hour of day (rush hour vs off-peak)
* Day of week
* Weekend vs weekday patterns

### 🔁 Sequential Features (Core Strength)

* Previous train delay
* Time since last train (headway)
* Rolling delay trends (lag features)

👉 These features allow the model to capture **delay propagation**, which is often ignored in basic approaches.

---

## 🧠 Model Approach

The project follows a progressive modeling strategy:

1. Baseline (historical averages)
2. XGbBoost

The goal is to **quantify the predictive power of sequential signals**.

---

## 📊 Evaluation

Model performance is evaluated using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)

Additionally, predictions can be compared against:

* baseline averages
* real-world observed delays

---

## 📈 Dashboard

An interactive dashboard allows users to:

* input station and time
* simulate delay scenarios
* view predicted wait times

This demonstrates how predictions change under different conditions.

---

## 🚀 What Makes This Project Different

Unlike standard transit apps, this system:

* focuses on **prediction during disruptions**, not normal conditions
* models **sequential dependencies between trains**
* aims to estimate **future delay propagation**, not just current status

👉 In short:
**From “Where is the train?” → to “What will happen next?”**

---

## 🔮 Future Work

* Integrate real-time transit APIs
* Add delay explainability (feature importance)
* Compare predictions directly with official estimates
* Expand to multi-line or network-level prediction

---

## 🛠 Tech Stack

* Python
* Pandas / NumPy
* Scikit-learn
* (Optional) Streamlit / Flask for dashboard

---

## 📎 Conclusion

This project explores how machine learning can move beyond static schedules and toward **dynamic, context-aware transit predictions**, especially in scenarios where traditional systems struggle the most.
