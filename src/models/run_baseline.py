
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "power_price_features.csv"
)

REPORT_FILE = (
    PROJECT_ROOT / "reports" / "baseline_evaluation.csv"
)


def main():

    df = pd.read_csv(INPUT_FILE)

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"], utc=True
    )

    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    df["target_price_24h"] = df["price_eur_mwh"].shift(-24)

    feature_columns = [
        "price_eur_mwh",
        "price_lag_1h",
        "price_lag_24h",
        "price_lag_48h",
        "price_lag_168h",
        "rolling_mean_24h",
        "rolling_mean_168h",
        "rolling_std_168h",
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
    ]

    df = df.dropna(
        subset=feature_columns + ["target_price_24h"]
    ).copy()

    current_time = pd.Timestamp.now(tz="UTC")

    df = df[
        df["timestamp_utc"] + pd.Timedelta(hours=24) <= current_time
    ].copy()

    if len(df) < 100:
        raise ValueError("Not enough historical observations for evaluation.")

    split_index = int(len(df) * 0.8)

    train = df.iloc[:split_index]
    test = df.iloc[split_index:]

    actual = test["target_price_24h"].to_numpy(dtype=float)
    predictions = test["price_eur_mwh"].to_numpy(dtype=float)

    mae = np.mean(np.abs(actual - predictions))

    denominator = (np.abs(actual) + np.abs(predictions)) / 2

    valid = denominator != 0

    smape = np.mean(
        100 * np.abs(actual[valid] - predictions[valid])
        / denominator[valid]
    )

    results = pd.DataFrame([{
        "model": "24-hour persistence baseline",
        "training_observations": len(train),
        "testing_observations": len(test),
        "MAE_EUR_MWh": mae,
        "sMAPE_percent": smape,
    }])

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORT_FILE, index=False)

    print("\nBASELINE EVALUATION COMPLETE")
    print("----------------------------")
    print(results.to_string(index=False))
    print(f"\nReport saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
