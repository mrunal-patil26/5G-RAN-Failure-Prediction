from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
FEATURE_PATH = BASE_DIR / "models" / "feature_columns.pkl"


def sample_input() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rsrp": -112,
                "rsrq": -15,
                "sinr": 6,
                "cqi": 5,
                "prb_util": 91,
                "throughput_mbps": 72,
                "active_users": 142,
                "handover_sr": 81,
                "rlf_count": 7,
                "latency_ms": 49,
                "packet_loss_pct": 3.8,
                "alarm_count": 4,
                "critical_alarm": 1,
                "peak_hour": 1,
                "signal_quality_index": -24.3,
                "load_pressure_index": 92.15,
            }
        ]
    )


def main():
    if not MODEL_PATH.exists() or not FEATURE_PATH.exists():
        raise FileNotFoundError("Trained model not found. Run src/train.py first.")

    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_PATH)
    input_df = sample_input()[feature_columns]

    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1])

    print("Sample prediction for one 5G cell:\n")
    print(input_df.to_string(index=False))
    print("\nPrediction:", "Failure Risk" if prediction == 1 else "Healthy")
    print(f"Failure Probability: {probability:.2%}")


if __name__ == "__main__":
    main()
