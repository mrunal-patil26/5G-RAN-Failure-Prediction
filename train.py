from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "5g_ran_kpi_data.csv"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

FEATURE_COLUMNS = [
    "rsrp",
    "rsrq",
    "sinr",
    "cqi",
    "prb_util",
    "throughput_mbps",
    "active_users",
    "handover_sr",
    "rlf_count",
    "latency_ms",
    "packet_loss_pct",
    "alarm_count",
    "critical_alarm",
    "peak_hour",
    "signal_quality_index",
    "load_pressure_index",
]
TARGET_COLUMN = "failure"


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Run src/generate_data.py first."
        )
    df = pd.read_csv(DATA_PATH)
    df = df.dropna().drop_duplicates()
    return df


def evaluate_model(name, model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    probs = model.predict_proba(x_test)[:, 1]

    return {
        "model": name,
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds), 4),
        "recall": round(recall_score(y_test, preds), 4),
        "f1_score": round(f1_score(y_test, preds), 4),
        "roc_auc": round(roc_auc_score(y_test, probs), 4),
    }


def save_feature_importance(model, feature_names):
    if hasattr(model.named_steps["classifier"], "feature_importances_"):
        importances = model.named_steps["classifier"].feature_importances_
        series = pd.Series(importances, index=feature_names).sort_values()
        plt.figure(figsize=(10, 7))
        series.plot(kind="barh")
        plt.title("Feature Importance - Best Model")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "feature_importance.png")
        plt.close()


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    x = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", GradientBoostingClassifier(random_state=42)),
            ]
        ),
    }

    metrics = []
    fitted_models = {}

    for name, model in models.items():
        result = evaluate_model(name, model, x_train, x_test, y_train, y_test)
        metrics.append(result)
        fitted_models[name] = model.fit(x_train, y_train)

    metrics_df = pd.DataFrame(metrics).sort_values(by=["f1_score", "roc_auc"], ascending=False)
    metrics_df.to_csv(RESULTS_DIR / "model_metrics.csv", index=False)

    best_model_name = metrics_df.iloc[0]["model"]
    best_model = fitted_models[best_model_name]

    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")
    joblib.dump(FEATURE_COLUMNS, MODELS_DIR / "feature_columns.pkl")
    save_feature_importance(best_model, FEATURE_COLUMNS)

    print("Training complete.\n")
    print(metrics_df.to_string(index=False))
    print(f"\nBest model saved: {best_model_name}")
    print(f"Model path: {MODELS_DIR / 'best_model.pkl'}")


if __name__ == "__main__":
    main()
