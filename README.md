🚇 NYC Subway Delay Prediction System
📌 Project Overview

This project presents a machine learning-based system for predicting subway delays in the New York City transit network. Using historical operational data and GTFS (General Transit Feed Specification) datasets, the system models delay propagation across stations and provides an interactive web-based dashboard for users.

The system includes:

Data preprocessing pipeline
Feature engineering for temporal and sequential patterns
XGBoost regression model
Flask-based deployment
Interactive delay prediction dashboard
🎯 Problem Statement

Subway delays significantly affect urban mobility and commuter experience. Delays are not isolated events; they propagate across stations due to operational dependencies.

This project formulates delay prediction as a supervised regression problem:

y^=f(X)

Where:

X: input features (time, previous delay, sequence position)
y^: predicted delay (in seconds)
📊 Data Sources

The model is built using transit data from:

1. Realtime Operational Data
Actual train arrival times
Trip and stop-level information
2. GTFS Dataset
stops.txt → station names and metadata
stop_times.txt → scheduled arrival times
trips.txt, routes.txt → route structure
📐 Delay Calculation
Delay=Actual Arrival Time−Scheduled Arrival Time
Example:
Scheduled: 08:00
Actual: 08:02
Delay=120 seconds
🧹 Data Preprocessing

Key preprocessing steps:

Removal of duplicate entries
Timestamp conversion to numeric format
Normalization of station identifiers (stop_id)
Merging real-time and scheduled datasets
Outlier filtering:
Valid delay range: [-600, 3600] seconds
⚠️ Memory Optimization

Due to large dataset size, sampling (~300,000 rows) was applied for efficient training.

⚙️ Feature Engineering

Feature engineering focuses on capturing temporal patterns and delay propagation behavior.

🕒 Time-Based Features
hour → hour of day
is_peak → rush hour indicator

is_peak={1; if hour ∈ [7–9, 16–18] and 0 otherwise
	​

🔁 Sequential Features

These capture how delays evolve over time:

Previous Delay
dt−1
	​

Rolling Delay
3
d
t−2
	​

+d
t−1
	​

+d
t
	​

	​

Cumulative Delay
∑d
i
	​

📌 Example
Station	Delay (sec)
A	100
B	150
C	200
Rolling delay at C = 150
Cumulative delay at C = 450
⚠️ Challenges & Solutions
❌ Data Leakage

Issue: Model used future information (e.g., actual timestamps)

Fix: Removed:

actual_sec
scheduled_sec
arrival time columns
❌ Memory Constraints

Issue: Large dataset caused memory errors

Fix: Sampling + optimized processing

❌ Stop ID Mismatch

Issue: stop_id stored inconsistently

Fix:

stops_df["stop_id"] = stops_df["stop_id"].astype(str)
❌ File Path Errors

Issue: Incorrect dataset paths

Fix: Used absolute paths with os.path.join()

🤖 Model Selection: XGBoost

XGBoost was selected due to:

strong performance on tabular data
ability to model nonlinear relationships
robustness to noise
efficient computation
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

Each f
k
	​

 represents a decision tree.

📈 Model Evaluation

Metrics used:

Mean Absolute Error (MAE)
MAE=
n
1
	​

∑∣y−
y
^
	​

∣
Root Mean Squared Error (RMSE)
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

✅ Results
Metric	Value
MAE	~224 sec
RMSE	~467 sec
R²	~0.85
🌐 System Architecture
Raw Data → Preprocessing → Feature Engineering → XGBoost Model → Flask App → Dashboard
🖥️ Web Application

A Flask-based web interface allows users to:

Input:
Station (mapped from GTFS)
Hour of day
Previous delay
Output:
Predicted delay across upcoming stations
Delay propagation visualization
🎨 Dashboard Features
Station-wise delay predictions
Color-coded delay levels:
🟢 Low (<120 sec)
🟡 Medium (120–300 sec)
🔴 High (>300 sec)
Route simulation
⚠️ Limitations
Route progression currently uses sequential approximation (+i)
No real-time integration
No geospatial visualization
Simplified delay propagation logic
🚀 Future Work
Integrate real-time transit APIs
Use LSTM / time-series models
Implement real route mapping using stop_times.txt
Add map-based visualization
Deploy as a scalable web service
📌 Conclusion

This project demonstrates how machine learning can be applied to model delay propagation in urban transit systems.

By combining:

data engineering
predictive modeling
web deployment

the system provides a practical framework for intelligent transit prediction.
