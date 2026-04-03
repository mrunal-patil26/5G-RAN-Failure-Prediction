import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def clip_round(values, low=None, high=None, decimals=2):
    arr = np.array(values, dtype=float)
    if low is not None:
        arr = np.maximum(arr, low)
    if high is not None:
        arr = np.minimum(arr, high)
    return np.round(arr, decimals)


def build_dataset(n_rows: int = 5000) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=n_rows, freq="5min")
    cell_ids = [f"CELL_{i:03d}" for i in np.random.randint(1, 101, size=n_rows)]
    hours = timestamps.hour
    peak_hour = ((hours >= 9) & (hours <= 12)) | ((hours >= 18) & (hours <= 22))

    rsrp = clip_round(np.random.normal(-95, 8, n_rows), -125, -70)
    rsrq = clip_round(np.random.normal(-11, 2.5, n_rows), -20, -3)
    sinr = clip_round(np.random.normal(16, 6, n_rows), -5, 35)
    cqi = clip_round(np.random.normal(10, 3, n_rows), 1, 15, 0)
    prb_util = clip_round(np.random.normal(68, 18, n_rows) + peak_hour.astype(int) * 8, 10, 100)
    throughput_mbps = clip_round(np.random.normal(120, 35, n_rows), 5, 300)
    active_users = clip_round(np.random.normal(85, 30, n_rows) + peak_hour.astype(int) * 20, 5, 250, 0)
    handover_sr = clip_round(np.random.normal(95, 4, n_rows), 60, 100)
    rlf_count = clip_round(np.random.poisson(2, n_rows), 0, 20, 0)
    latency_ms = clip_round(np.random.normal(22, 8, n_rows) + (prb_util > 85).astype(int) * 12, 5, 120)
    packet_loss_pct = clip_round(np.random.normal(0.8, 0.9, n_rows), 0, 15)
    alarm_count = clip_round(np.random.poisson(1.2, n_rows), 0, 10, 0)

    # Inject degraded situations to create more meaningful failure cases.
    degraded_idx = np.random.choice(n_rows, size=int(0.18 * n_rows), replace=False)
    rsrp[degraded_idx] = clip_round(rsrp[degraded_idx] - np.random.uniform(8, 18, len(degraded_idx)), -125, -70)
    rsrq[degraded_idx] = clip_round(rsrq[degraded_idx] - np.random.uniform(1, 5, len(degraded_idx)), -20, -3)
    sinr[degraded_idx] = clip_round(sinr[degraded_idx] - np.random.uniform(5, 14, len(degraded_idx)), -5, 35)
    handover_sr[degraded_idx] = clip_round(handover_sr[degraded_idx] - np.random.uniform(8, 22, len(degraded_idx)), 60, 100)
    latency_ms[degraded_idx] = clip_round(latency_ms[degraded_idx] + np.random.uniform(10, 30, len(degraded_idx)), 5, 120)
    packet_loss_pct[degraded_idx] = clip_round(packet_loss_pct[degraded_idx] + np.random.uniform(1, 5, len(degraded_idx)), 0, 15)
    alarm_count[degraded_idx] = clip_round(alarm_count[degraded_idx] + np.random.randint(1, 4, len(degraded_idx)), 0, 10, 0)
    rlf_count[degraded_idx] = clip_round(rlf_count[degraded_idx] + np.random.randint(1, 6, len(degraded_idx)), 0, 20, 0)

    signal_quality_index = clip_round((sinr * 0.5) + (rsrq * 0.3) + (rsrp * 0.2), -50, 30)
    load_pressure_index = clip_round((prb_util * 0.6) + (active_users * 0.25) + (latency_ms * 0.15), 0, 200)
    critical_alarm = (alarm_count >= 3).astype(int)

    # Failure risk score from domain-inspired KPI thresholds.
    risk_score = (
        (rsrp < -108).astype(int) * 2
        + (rsrq < -14).astype(int) * 2
        + (sinr < 8).astype(int) * 2
        + (prb_util > 88).astype(int) * 2
        + (handover_sr < 85).astype(int) * 2
        + (rlf_count >= 5).astype(int) * 2
        + (latency_ms > 40).astype(int) * 1
        + (packet_loss_pct > 2.5).astype(int) * 1
        + (alarm_count >= 3).astype(int) * 2
        + peak_hour.astype(int) * 0.5
    )
    noise = np.random.normal(0, 1.1, n_rows)
    failure = ((risk_score + noise) >= 6).astype(int)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "cell_id": cell_ids,
            "rsrp": rsrp,
            "rsrq": rsrq,
            "sinr": sinr,
            "cqi": cqi.astype(int),
            "prb_util": prb_util,
            "throughput_mbps": throughput_mbps,
            "active_users": active_users.astype(int),
            "handover_sr": handover_sr,
            "rlf_count": rlf_count.astype(int),
            "latency_ms": latency_ms,
            "packet_loss_pct": packet_loss_pct,
            "alarm_count": alarm_count.astype(int),
            "critical_alarm": critical_alarm,
            "peak_hour": peak_hour.astype(int),
            "signal_quality_index": signal_quality_index,
            "load_pressure_index": load_pressure_index,
            "failure": failure,
        }
    )
    return df


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[1]
    out_path = base_dir / "data" / "5g_ran_kpi_data.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_dataset()
    df.to_csv(out_path, index=False)

    print(f"Dataset saved to: {out_path}")
    print(df.head())
    print("\nFailure distribution:")
    print(df["failure"].value_counts(normalize=True).round(3))
