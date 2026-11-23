
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "data/processed/power_price_features.csv"
REPORT = ROOT / "reports/day_ahead_walk_forward_results.csv"

HORIZON = 24
FOLDS = 4
INITIAL_TRAIN_FRACTION = 0.60

FEATURES = [
    "price_eur_mwh",
    "price_lag_1h",
    "price_lag_24h",
    "price_lag_48h",
    "price_lag_168h",
    "rolling_mean_24h",
    "rolling_mean_168h",
    "rolling_std_168h",
    "target_hour",
    "target_day_of_week",
    "target_month",
    "target_is_weekend",
]


def calculate_mae(actual, predicted):
    return np.mean(np.abs(actual - predicted))


def calculate_smape(actual, predicted):
    denominator = np.abs(actual) + np.abs(predicted)
    valid = denominator > 0

    return np.mean(
        200 * np.abs(actual[valid] - predicted[valid])
        / denominator[valid]
    )


def prepare_dataset():
    df = pd.read_csv(INPUT)

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"], utc=True
    )

    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    # The target is the electricity price 24 hours after forecast origin.
    df["target_price"] = df["price_eur_mwh"].shift(-HORIZON)
    df["target_timestamp"] = df["timestamp_utc"].shift(-HORIZON)

    target_local = df["target_timestamp"].dt.tz_convert(
        "Europe/Berlin"
    )

    # Calendar information for tomorrow is known in advance.
    df["target_hour"] = target_local.dt.hour
    df["target_day_of_week"] = target_local.dt.day_of_week
    df["target_month"] = target_local.dt.month
    df["target_is_weekend"] = (
        target_local.dt.day_of_week >= 5
    ).astype(int)

    # Keep only features available at forecast origin.
    model_data = df[
        ["timestamp_utc", "target_timestamp", "target_price"]
        + FEATURES
    ].copy()

    model_data[FEATURES] = model_data[FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )

    model_data["target_price"] = pd.to_numeric(
        model_data["target_price"], errors="coerce"
    )

    model_data = model_data.dropna(
        subset=FEATURES + ["target_price"]
    ).reset_index(drop=True)

    return model_data


def fit_and_predict(train, test):
    X_train = train[FEATURES].to_numpy(dtype=float)
    X_test = test[FEATURES].to_numpy(dtype=float)

    y_train = train["target_price"].to_numpy(dtype=float)

    # Standardisation parameters come exclusively from training data.
    means = X_train.mean(axis=0)
    stds = X_train.std(axis=0)
    stds[stds == 0] = 1.0

    X_train = (X_train - means) / stds
    X_test = (X_test - means) / stds

    X_train = np.column_stack([
        np.ones(len(X_train)), X_train
    ])

    X_test = np.column_stack([
        np.ones(len(X_test)), X_test
    ])

    coefficients = np.linalg.lstsq(
        X_train, y_train, rcond=None
    )[0]

    return X_test @ coefficients


def main():
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    df = prepare_dataset()

    total = len(df)
    initial_train = int(total * INITIAL_TRAIN_FRACTION)
    test_size = (total - initial_train) // FOLDS

    results = []

    print("\nDAY-AHEAD FORECASTING")
    print(f"Usable observations: {total:,}")
    print(f"Forecast horizon: {HORIZON} hours")
    print(f"Features: {len(FEATURES)}")

    for fold in range(FOLDS):
        train_end = initial_train + fold * test_size
        test_end = (
            initial_train + (fold + 1) * test_size
        )

        train = df.iloc[:train_end].copy()
        test = df.iloc[train_end:test_end].copy()

        if train.empty or test.empty:
            continue

        # Purge training rows whose target timestamp reaches
        # or crosses the first test forecast origin.
        first_test_origin = test["timestamp_utc"].min()

        train = train[
            train["target_timestamp"] < first_test_origin
        ]

        predictions = fit_and_predict(train, test)

        actual = test["target_price"].to_numpy(dtype=float)

        # Persistence predicts tomorrow's price using today's price.
        baseline = test["price_eur_mwh"].to_numpy(dtype=float)

        results.append({
            "fold": fold + 1,
            "training_rows": len(train),
            "testing_rows": len(test),
            "test_start": test["target_timestamp"].min(),
            "test_end": test["target_timestamp"].max(),
            "baseline_mae": calculate_mae(actual, baseline),
            "model_mae": calculate_mae(actual, predictions),
            "baseline_smape": calculate_smape(actual, baseline),
            "model_smape": calculate_smape(actual, predictions),
        })

        print(
            f"\nFold {fold + 1}: "
            f"Baseline MAE = {results[-1]['baseline_mae']:.2f} | "
            f"Model MAE = {results[-1]['model_mae']:.2f}"
        )

    report = pd.DataFrame(results)
    report.to_csv(REPORT, index=False)

    print("\nWALK-FORWARD RESULTS")
    print(report.to_string(index=False))

    print("\nAVERAGE PERFORMANCE")
    print(
        f"Persistence MAE: "
        f"{report['baseline_mae'].mean():.2f} €/MWh"
    )
    print(
        f"Day-ahead model MAE: "
        f"{report['model_mae'].mean():.2f} €/MWh"
    )
    print(
        f"Persistence sMAPE: "
        f"{report['baseline_smape'].mean():.2f}%"
    )
    print(
        f"Day-ahead model sMAPE: "
        f"{report['model_smape'].mean():.2f}%"
    )

    print(f"\nReport saved: {REPORT}")


if __name__ == "__main__":
    main()
