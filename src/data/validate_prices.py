
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "reports"

DATASETS = {
    "hour": {
        "file": "smard_day_ahead_prices_hour.csv",
        "expected_interval": pd.Timedelta(hours=1),
    },
    "quarterhour": {
        "file": "smard_day_ahead_prices_quarterhour.csv",
        "expected_interval": pd.Timedelta(minutes=15),
    },
}


def validate_dataset(name, config):
    file_path = RAW_DIR / config["file"]
    expected_interval = config["expected_interval"]

    print(f"\n{'=' * 65}")
    print(f"VALIDATING: {name.upper()} DATA")
    print(f"{'=' * 65}")

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    df = pd.read_csv(file_path)

    required_columns = [
        "timestamp_ms",
        "price_eur_mwh",
        "timestamp_utc",
        "timestamp_local",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    df["timestamp_ms"] = pd.to_numeric(
        df["timestamp_ms"], errors="coerce"
    )

    df["price_eur_mwh"] = pd.to_numeric(
        df["price_eur_mwh"], errors="coerce"
    )

    # Normalise all timestamps to UTC.
    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"],
        utc=True,
        errors="coerce",
    )

    # Mixed CET/CEST offsets are normalised to UTC.
    df["timestamp_local_utc"] = pd.to_datetime(
        df["timestamp_local"],
        utc=True,
        errors="coerce",
    )

    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    total_rows = len(df)

    invalid_timestamps = int(
        df["timestamp_utc"].isna().sum()
    )

    invalid_prices = int(
        df["price_eur_mwh"].isna().sum()
    )

    duplicate_timestamps = int(
        df["timestamp_ms"].duplicated().sum()
    )

    valid = df.dropna(
        subset=["timestamp_utc", "timestamp_ms"]
    ).copy()

    valid = valid.sort_values("timestamp_utc")

    differences = valid["timestamp_utc"].diff().dropna()

    gap_count = int(
        (differences > expected_interval).sum()
    )

    missing_intervals = int(
        np.maximum(
            (differences / expected_interval).round() - 1,
            0,
        ).sum()
    )

    irregular_intervals = int(
        (differences % expected_interval != pd.Timedelta(0)).sum()
    )

    # Detect future observations.
    now_utc = pd.Timestamp.now(tz="UTC")

    future_rows = valid[
        valid["timestamp_utc"] > now_utc
    ]

    # Check consistency between local and UTC timestamps.
    expected_local_utc = valid["timestamp_utc"]

    local_timestamp_mismatches = int(
        (
            valid["timestamp_local_utc"]
            != expected_local_utc
        ).sum()
    )

    prices = valid["price_eur_mwh"].dropna()

    negative_prices = int((prices < 0).sum())

    if len(prices):
        q1 = prices.quantile(0.25)
        q3 = prices.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr

        extreme_prices = int(
            (
                (prices < lower_bound)
                | (prices > upper_bound)
            ).sum()
        )

        price_min = prices.min()
        price_max = prices.max()
        price_mean = prices.mean()
        price_median = prices.median()
        price_std = prices.std()

    else:
        extreme_prices = 0
        price_min = price_max = np.nan
        price_mean = price_median = price_std = np.nan

    results = {
        "Dataset": name,
        "Total observations": total_rows,
        "Invalid timestamps": invalid_timestamps,
        "Invalid prices": invalid_prices,
        "Duplicate timestamps": duplicate_timestamps,
        "Timestamp gaps": gap_count,
        "Missing intervals": missing_intervals,
        "Irregular intervals": irregular_intervals,
        "Future observations": len(future_rows),
        "Negative prices": negative_prices,
        "Extreme prices": extreme_prices,
        "Local timestamp mismatches": local_timestamp_mismatches,
        "First timestamp UTC": valid["timestamp_utc"].min(),
        "Last timestamp UTC": valid["timestamp_utc"].max(),
        "Minimum price": price_min,
        "Maximum price": price_max,
        "Mean price": price_mean,
        "Median price": price_median,
        "Price standard deviation": price_std,
    }

    for key, value in results.items():
        print(f"{key}: {value}")

    if len(future_rows):
        print("\nFUTURE-DATED OBSERVATIONS:")
        print(
            future_rows[
                ["timestamp_utc", "price_eur_mwh"]
            ].tail(10).to_string(index=False)
        )

    return results


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for name, config in DATASETS.items():
        result = validate_dataset(name, config)
        all_results.append(result)

    report = pd.DataFrame(all_results)

    output_file = REPORT_DIR / "data_quality_report.csv"

    report.to_csv(output_file, index=False)

    print(f"\n{'=' * 65}")
    print("VALIDATION FINISHED")
    print(f"Report saved to: {output_file}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
