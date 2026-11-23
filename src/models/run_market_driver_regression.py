
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "data/processed/power_market_dataset.csv"
OUTPUT = ROOT / "reports/market_driver_regression_evaluation.csv"

FEATURES = [
    "grid_load",
    "wind_offshore",
    "wind_onshore",
    "solar_generation",
]


def calculate_mae(actual, predicted):
    return np.mean(np.abs(actual - predicted))


def calculate_smape(actual, predicted):
    denominator = np.abs(actual) + np.abs(predicted)
    valid = denominator != 0

    if not valid.any():
        return 0.0

    return np.mean(
        200 * np.abs(actual[valid] - predicted[valid])
        / denominator[valid]
    )


def main():
    df = pd.read_csv(INPUT)

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"], utc=True
    )

    df = df.sort_values("timestamp_utc")

    columns = ["price_eur_mwh"] + FEATURES

    df[columns] = df[columns].apply(
        pd.to_numeric, errors="coerce"
    )

    df = df.dropna(subset=columns).reset_index(drop=True)

    split = int(len(df) * 0.8)

    train = df.iloc[:split]
    test = df.iloc[split:]

    X_train = train[FEATURES].to_numpy(dtype=float)
    X_test = test[FEATURES].to_numpy(dtype=float)

    y_train = train["price_eur_mwh"].to_numpy(dtype=float)
    y_test = test["price_eur_mwh"].to_numpy(dtype=float)

    # Standardise using training data only.
    means = X_train.mean(axis=0)
    stds = X_train.std(axis=0)
    stds[stds == 0] = 1

    X_train_scaled = (X_train - means) / stds
    X_test_scaled = (X_test - means) / stds

    # Add intercept.
    X_train_design = np.column_stack([
        np.ones(len(X_train_scaled)),
        X_train_scaled
    ])

    X_test_design = np.column_stack([
        np.ones(len(X_test_scaled)),
        X_test_scaled
    ])

    # Least-squares regression using NumPy.
    coefficients = np.linalg.lstsq(
        X_train_design, y_train, rcond=None
    )[0]

    predictions = X_test_design @ coefficients

    results = pd.DataFrame([{
        "model": "Market Driver Linear Regression",
        "training_rows": len(train),
        "testing_rows": len(test),
        "features": len(FEATURES),
        "mae_eur_mwh": calculate_mae(y_test, predictions),
        "smape_percent": calculate_smape(y_test, predictions),
        "evaluation_type": "Exploratory contemporaneous benchmark"
    }])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT, index=False)

    print("\nMARKET DRIVER REGRESSION COMPLETE")
    print(results.to_string(index=False))

    print("\nStandardised-feature coefficients:")
    for name, coefficient in zip(FEATURES, coefficients[1:]):
        print(f"{name}: {coefficient:.4f}")


if __name__ == "__main__":
    main()
