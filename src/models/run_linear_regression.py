
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "power_price_features.csv"
REPORT_FILE = PROJECT_ROOT / "reports" / "linear_regression_evaluation.csv"


def main():

    df = pd.read_csv(INPUT_FILE)

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"], utc=True
    )

    df = df.sort_values("timestamp_utc").reset_index(drop=True)

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

    df["target_price_24h"] = df["price_eur_mwh"].shift(-24)

    df = df.dropna(
        subset=feature_columns + ["target_price_24h"]
    ).copy()

    current_time = pd.Timestamp.now(tz="UTC")

    df = df[
        df["timestamp_utc"] + pd.Timedelta(hours=24) <= current_time
    ].copy()

    X = df[feature_columns].to_numpy(dtype=float)
    y = df["target_price_24h"].to_numpy(dtype=float)

    split_index = int(len(df) * 0.8)

    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_std[X_std == 0] = 1

    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std

    y_mean = y_train.mean()
    y_std = y_train.std()

    y_train_scaled = (y_train - y_mean) / y_std

    weights = np.zeros(X_train.shape[1])
    bias = 0.0

    learning_rate = 0.05
    epochs = 300
    sample_count = len(X_train)

    for epoch in range(epochs):

        predictions = np.sum(
            X_train * weights, axis=1
        ) + bias

        errors = predictions - y_train_scaled

        weight_gradient = (
            2 / sample_count
        ) * np.sum(
            X_train * errors[:, None], axis=0
        )

        bias_gradient = 2 * errors.mean()

        weights -= learning_rate * weight_gradient
        bias -= learning_rate * bias_gradient

    test_predictions_scaled = (
        np.sum(X_test * weights, axis=1) + bias
    )

    predictions = (
        test_predictions_scaled * y_std + y_mean
    )

    mae = np.mean(np.abs(y_test - predictions))

    denominator = (np.abs(y_test) + np.abs(predictions)) / 2
    valid = denominator != 0

    smape = np.mean(
        100 * np.abs(y_test[valid] - predictions[valid])
        / denominator[valid]
    )

    results = pd.DataFrame([{
        "model": "NumPy Linear Regression",
        "training_observations": len(X_train),
        "testing_observations": len(X_test),
        "MAE_EUR_MWh": mae,
        "sMAPE_percent": smape,
        "epochs": epochs,
    }])

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORT_FILE, index=False)

    print("\nLINEAR REGRESSION COMPLETE")
    print("---------------------------")
    print(results.to_string(index=False))
    print(f"\nReport saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
