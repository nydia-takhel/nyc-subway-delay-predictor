🚇 NYC Subway Delay Prediction System
📌 Project Overview

This project develops a machine learning-based system to predict subway delays in the New York City transit network. By leveraging historical operational data and GTFS datasets, the system estimates delay propagation across stations and provides a user-friendly dashboard for real-time interaction.

The system integrates:

Data preprocessing pipeline
Feature engineering
XGBoost regression model
Flask-based web application
Interactive delay prediction dashboard
🎯 Problem Statement

Subway delays significantly impact urban mobility and commuter planning. Delays often propagate across multiple stations due to cascading effects in the transit network.

This project models delay prediction as a supervised regression problem:

y
^
	​

=f(X)

Where:

X = feature vector (time, previous delay, weather, etc.)
y
^
	​

 = predicted delay (in seconds)
📊 Data Sources

Two primary datasets were used:

1. Real-Time Data
Actual arrival times
Trip-level operational data
2. GTFS Data (General Transit Feed Specification)
stops.txt → station metadata
stop_times.txt → scheduled timings
trips.txt, routes.txt → route structure
📐 Delay Definition
Delay=Actual Arrival Time−Scheduled Arrival Time
🧹 Data Preprocessing

Key preprocessing steps:

Removal of duplicate records
Timestamp normalization
Conversion of categorical identifiers
Merging real-time and scheduled datasets
Outlier filtering:
Valid delay range: [-600, 3600] seconds
Memory optimization:
Sampling (~300,000 rows) for efficient training
⚙️ Feature Engineering

Feature engineering was critical to capturing temporal and sequential dependencies.

🕒 Time-Based Features
hour
is_peak (rush hours: 7–9 AM, 5–7 PM)
🔁 Sequential Features
prev_delay
rolling_delay
cumulative_delay
🌦 Weather Features
temperature
humidity
visibility
is_rain

(Weather data is currently simulated for modeling purposes.)

⚠️ Challenges & Solutions
1. Data Leakage
Issue: Model used future information (e.g., actual arrival)
Fix: Removed leakage columns and enforced time-based split
2. Memory Constraints
Issue: Dataset too large (~3M+ rows)
Fix: Sampling + efficient model selection
3. File Path Errors
Issue: Incorrect dataset paths (stops.txt not found)
Fix: Absolute path handling using os.path.join()
4. ID Mismatch (String vs Integer)
Issue: GTFS stop_id stored as string
Fix:
stops_df["stop_id"] = stops_df["stop_id"].astype(str)
5. Flask Template Errors
Issue: TemplateNotFound
Fix: Correct folder structure (templates/index.html)
🤖 Model Selection: XGBoost

XGBoost was selected due to:

Ability to model nonlinear relationships
Robustness to noisy data
High performance on structured/tabular datasets
Built-in regularization
📐 Model Representation
y
^
	​

=
k=1
∑
K
	​

f
k
	​

(X)

Where each f
k
	​

 is a decision tree.

📈 Model Evaluation

Evaluation metrics:

MAE (Mean Absolute Error)
MAE=
n
1
	​

∑∣y−
y
^
	​

∣
RMSE (Root Mean Squared Error)
RMSE=
n
1
	​

∑(y−
y
^
	​

)
2
	​

R² Score
R
2
=1−
SS
tot
	​

SS
res
	​

	​

✅ Final Results
Metric	Value
MAE	~224 sec
RMSE	~467 sec
R²	~0.85

👉 Indicates strong predictive performance.

🌐 System Architecture
Raw Data → Preprocessing → Feature Engineering → XGBoost Model → Flask App → Dashboard
🖥️ Web Application

A Flask-based web interface allows users to:

Select subway station (mapped from GTFS)
Input:
Time
Previous delay
Weather condition
View predicted delays across upcoming stations
🎨 Dashboard Features
Station-wise delay predictions
Color-coded delay levels:
🟢 Low (<120 sec)
🟡 Medium (120–300 sec)
🔴 High (>300 sec)
Route simulation visualization
⚠️ Limitations
Uses simulated weather data
Current route prediction uses sequential approximation (+i)
No real-time subway tracking
GTFS not fully integrated for route sequencing
🚀 Future Work
Integration with real-time APIs (MTA feeds)
Use of LSTM / Time Series models
Real route prediction using stop_times.txt
Map-based visualization (Leaflet / Mapbox)
Mobile-friendly UI
📌 Conclusion

This project demonstrates how machine learning can be applied to urban transit systems to predict delays and improve commuter decision-making.

By combining data engineering, predictive modeling, and web deployment, the system provides a scalable foundation for real-world transit intelligence applications.

📂 Project Structure
NYC-subway-delay-predictor/
│
├── data/
├── gtfs_subway/
├── features.py
├── train.py
├── app.py
├── templates/
│   └── index.html
├── xgboost_model.pkl
└── README.md
👩‍💻 Author

Nydia Takhel
