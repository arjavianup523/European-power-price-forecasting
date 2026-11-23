
from pathlib import Path

import pandas as pd

from src.features.build_features import (
    add_price_lags,
    add_rolling_features,
    add_calendar_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "processed_hourly_power_prices.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "power_price_features.csv"
)


def main():

    df = pd.read_csv(INPUT_FILE)

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"], utc=True
    )

    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    df = add_price_lags(df)
    df = add_rolling_features(df)
    df = add_calendar_features(df)

    feature_columns = [
        "price_lag_1h",
        "price_lag_24h",
        "price_lag_48h",
        "price_lag_168h",
        "rolling_mean_24h",
        "rolling_mean_168h",
        "rolling_std_168h",
        "hour",
        "day_of_week",
        "day_of_month",
        "month",
        "day_of_year",
        "is_weekend",
    ]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nFEATURE ENGINEERING COMPLETE")
    print("----------------------------")
    print(f"Total observations: {len(df):,}")
    print(f"Total engineered features: {len(feature_columns)}")
    print(f"Output file: {OUTPUT_FILE}")
    print("\nMissing values in engineered features:")

    print(df[feature_columns].isna().sum())

    print("\nFeature dataset saved successfully.")


if __name__ == "__main__":
    main()
