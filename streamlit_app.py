from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
FEATURE_PATH = BASE_DIR / "models" / "feature_columns.pkl"
METRICS_PATH = BASE_DIR / "results" / "model_metrics.csv"
IMPORTANCE_PATH = BASE_DIR / "results" / "feature_importance.png"

st.set_page_config(page_title="5G RAN Failure Prediction", layout="wide")
st.title("5G RAN Failure Prediction Dashboard")
st.write("Enter telecom KPI values to estimate cell failure risk.")

if not MODEL_PATH.exists():
    st.warning("Model not found. Run generate_data.py and train.py first.")
    st.stop()

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_PATH)

with st.sidebar:
    st.header("Input KPIs")
    rsrp = st.slider("RSRP (dBm)", -125, -70, -105)
    rsrq = st.slider("RSRQ (dB)", -20, -3, -13)
    sinr = st.slider("SINR (dB)", -5, 35, 8)
    cqi = st.slider("CQI", 1, 15, 7)
    prb_util = st.slider("PRB Utilization (%)", 0, 100, 85)
    throughput_mbps = st.slider("Throughput (Mbps)", 0, 300, 90)
    active_users = st.slider("Active Users", 0, 250, 110)
    handover_sr = st.slider("Handover Success Rate (%)", 0, 100, 88)
    rlf_count = st.slider("Radio Link Failure Count", 0, 20, 4)
    latency_ms = st.slider("Latency (ms)", 0, 120, 38)
    packet_loss_pct = st.slider("Packet Loss (%)", 0.0, 15.0, 1.8)
    alarm_count = st.slider("Alarm Count", 0, 10, 2)
    critical_alarm = 1 if alarm_count >= 3 else 0
    peak_hour = st.selectbox("Peak Hour", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

signal_quality_index = round((sinr * 0.5) + (rsrq * 0.3) + (rsrp * 0.2), 2)
load_pressure_index = round((prb_util * 0.6) + (active_users * 0.25) + (latency_ms * 0.15), 2)

input_df = pd.DataFrame(
    [[
        rsrp, rsrq, sinr, cqi, prb_util, throughput_mbps, active_users,
        handover_sr, rlf_count, latency_ms, packet_loss_pct, alarm_count,
        critical_alarm, peak_hour, signal_quality_index, load_pressure_index
    ]],
    columns=feature_columns,
)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Current Input")
    st.dataframe(input_df, use_container_width=True)

    if st.button("Predict Failure Risk"):
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])
        if prediction == 1:
            st.error(f"High Failure Risk detected. Probability: {probability:.2%}")
        else:
            st.success(f"Cell appears healthy. Failure probability: {probability:.2%}")

with col2:
    st.subheader("Derived Indicators")
    st.metric("Signal Quality Index", signal_quality_index)
    st.metric("Load Pressure Index", load_pressure_index)
    st.metric("Critical Alarm Flag", critical_alarm)

    if METRICS_PATH.exists():
        st.subheader("Model Comparison")
        st.dataframe(pd.read_csv(METRICS_PATH), use_container_width=True)

    if IMPORTANCE_PATH.exists():
        st.subheader("Feature Importance")
        st.image(str(IMPORTANCE_PATH), use_container_width=True)
