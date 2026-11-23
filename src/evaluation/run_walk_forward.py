
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "power_price_features.csv"
)

REPORT_FILE = (
    PROJECT_ROOT / "reports" / "walk_forward_evaluation.csv"
)

FEATURE_COLUMNS = [
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


def calculate_metrics(actual, predictions):

    mae = np.mean(np.abs(actual - predictions))

    denominator = (np.abs(actual) + np.abs(predictions)) / 2
    valid = denominator != 0

    smape = np.mean(
        100 * np.abs(actual[valid] - predictions[valid])
        / denominator[valid]
    )

    return mae, smape


def train_linear_regression(X_train, y_train, X_test):

    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_std[X_std == 0] = 1

    X_train_scaled = (X_train - X_mean) / X_std
    X_test_scaled = (X_test - X_mean) / X_std

    y_mean = y_train.mean()
    y_std = y_train.std()

    y_train_scaled = (y_train - y_mean) / y_std

    weights = np.zeros(X_train.shape[1])
    bias = 0.0

    learning_rate = 0.05
    epochs = 300
    sample_count = len(X_train)

    for _ in range(epochs):

        predictions = (
            np.sum(X_train_scaled * weights, axis=1) + bias
        )

        errors = predictions - y_train_scaled

        weight_gradient = (
            2 / sample_count
        ) * np.sum(
            X_train_scaled * errors[:, None], axis=0
        )

        bias_gradient = 2 * errors.mean()

        weights -= learning_rate * weight_gradient
        bias -= learning_rate * bias_gradient

    test_predictions_scaled = (
        np.sum(X_test_scaled * weights, axis=1) + bias
    )

    return test_predictions_scaled * y_std + y_mean


def main():

    df = pd.read_csv(INPUT_FILE)

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"], utc=True
    )

    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    df["target_price_24h"] = df["price_eur_mwh"].shift(-24)

    df = df.dropna(
        subset=FEATURE_COLUMNS + ["target_price_24h"]
    ).copy()

    current_time = pd.Timestamp.now(tz="UTC")

    df = df[
        df["timestamp_utc"] + pd.Timedelta(hours=24) <= current_time
    ].reset_index(drop=True)

    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df["target_price_24h"].to_numpy(dtype=float)

    initial_train_size = int(len(df) * 0.60)
    test_size = int(len(df) * 0.10)

    results = []

    for fold in range(4):

        train_end = initial_train_size + fold * test_size
        test_end = train_end + test_size

        X_train = X[:train_end]
        y_train = y[:train_end]

        X_test = X[train_end:test_end]
        y_test = y[train_end:test_end]

        # Persistence benchmark: use the current price.
        baseline_predictions = X_test[:, 0]

        baseline_mae, baseline_smape = calculate_metrics(
            y_test, baseline_predictions
        )

        # Train linear regression using only past observations.
        linear_predictions = train_linear_regression(
            X_train, y_train, X_test
        )

        linear_mae, linear_smape = calculate_metrics(
            y_test, linear_predictions
        )

        results.append({
            "fold": fold + 1,
            "training_observations": len(X_train),
            "testing_observations": len(X_test),
            "test_start": df.iloc[train_end]["timestamp_utc"],
            "test_end": df.iloc[test_end - 1]["timestamp_utc"],
            "baseline_MAE": baseline_mae,
            "baseline_sMAPE": baseline_smape,
            "linear_MAE": linear_mae,
            "linear_sMAPE": linear_smape,
        })

        print(
            f"Fold {fold + 1}/4 | "
            f"Baseline MAE: {baseline_mae:.2f} | "
            f"Linear MAE: {linear_mae:.2f}"
        )

    results_df = pd.DataFrame(results)

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(REPORT_FILE, index=False)

    print("\nWALK-FORWARD VALIDATION COMPLETE")
    print("--------------------------------")
    print(results_df.to_string(index=False))

    print("\nAVERAGE PERFORMANCE")
    print("-------------------")
    print(
        f"Baseline average MAE: "
        f"{results_df['baseline_MAE'].mean():.2f} €/MWh"
    )
    print(
        f"Linear average MAE: "
        f"{results_df['linear_MAE'].mean():.2f} €/MWh"
    )

    print(f"\nReport saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
