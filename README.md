# 5G RAN Failure Prediction using AI/ML

This project predicts potential 5G Radio Access Network (RAN) failures using telecom KPIs, alarm behavior, and basic machine learning models.

## What this project does
- Generates a realistic synthetic 5G RAN KPI dataset
- Builds features from radio, mobility, load, and service KPIs
- Trains multiple classification models
- Selects the best model using F1-score and ROC-AUC
- Saves the trained model for reuse
- Provides single-record prediction support
- Includes a simple Streamlit dashboard

## Problem Statement
In 5G RAN, service degradation and failures can happen because of poor signal quality, congestion, handover issues, and abnormal alarms. This project predicts whether a cell is likely to fail using historical KPI patterns.

## Input Features
- rsrp
- rsrq
- sinr
- cqi
- prb_util
- throughput_mbps
- active_users
- handover_sr
- rlf_count
- latency_ms
- packet_loss_pct
- alarm_count
- critical_alarm
- peak_hour
- signal_quality_index
- load_pressure_index

## Target
- `failure = 0` -> healthy cell
- `failure = 1` -> high failure risk / likely failure

## Project Structure
```
5g_ran_failure_prediction/
│
├── data/
│   └── 5g_ran_kpi_data.csv
├── models/
│   ├── best_model.pkl
│   └── feature_columns.pkl
├── src/
│   ├── generate_data.py
│   ├── train.py
│   └── predict.py
├── app/
│   └── streamlit_app.py
├── results/
│   ├── feature_importance.png
│   └── model_metrics.csv
├── requirements.txt
└── README.md
```

## Installation
```bash
pip install -r requirements.txt
```

## How to Run
### 1) Generate dataset
```bash
python src/generate_data.py
```

### 2) Train models
```bash
python src/train.py
```

### 3) Predict for a sample cell
```bash
python src/predict.py
```

### 4) Launch dashboard
```bash
streamlit run app/streamlit_app.py
```

## Models Compared
- Logistic Regression
- Random Forest
- Gradient Boosting

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

## Example Use Case
A network operations team can use this system to detect cells at high failure risk and take preventive action before an outage impacts users.

## Resume Description
Developed an AI/ML-based predictive analytics system for 5G RAN failure prediction using telecom KPIs such as RSRP, RSRQ, SINR, PRB utilization, handover success rate, latency, packet loss, and alarm counts. Built an end-to-end pipeline including synthetic data generation, preprocessing, feature engineering, model training, evaluation, and a Streamlit dashboard for risk scoring.
