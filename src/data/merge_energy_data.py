
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

PRICE_FILE = ROOT / "data/processed/processed_hourly_power_prices.csv"
DRIVER_FILE = ROOT / "data/raw/smard_energy_drivers_hourly.csv"
OUTPUT_FILE = ROOT / "data/processed/power_market_dataset.csv"
REPORT_FILE = ROOT / "reports/energy_data_merge_report.csv"


def main():
    prices = pd.read_csv(PRICE_FILE)
    drivers = pd.read_csv(DRIVER_FILE)

    prices["timestamp_utc"] = pd.to_datetime(
        prices["timestamp_utc"], utc=True
    )
    drivers["timestamp_utc"] = pd.to_datetime(
        drivers["timestamp_utc"], utc=True
    )

    prices = prices.drop_duplicates("timestamp_utc")
    drivers = drivers.drop_duplicates("timestamp_utc")

    dataset = prices.merge(
        drivers.drop(columns=["timestamp_local"], errors="ignore"),
        on="timestamp_utc",
        how="inner",
        validate="one_to_one"
    )

    dataset = dataset.sort_values("timestamp_utc").reset_index(drop=True)

    driver_columns = [
        "grid_load",
        "wind_offshore",
        "wind_onshore",
        "solar_generation"
    ]

    report = pd.DataFrame({
        "metric": [
            "price_rows",
            "driver_rows",
            "matched_rows",
            "first_timestamp",
            "last_timestamp",
            "missing_price",
            "missing_grid_load",
            "missing_wind_offshore",
            "missing_wind_onshore",
            "missing_solar_generation"
        ],
        "value": [
            len(prices),
            len(drivers),
            len(dataset),
            dataset["timestamp_utc"].min(),
            dataset["timestamp_utc"].max(),
            dataset["price_eur_mwh"].isna().sum(),
            dataset["grid_load"].isna().sum(),
            dataset["wind_offshore"].isna().sum(),
            dataset["wind_onshore"].isna().sum(),
            dataset["solar_generation"].isna().sum()
        ]
    })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    dataset.to_csv(OUTPUT_FILE, index=False)
    report.to_csv(REPORT_FILE, index=False)

    print("\nENERGY MARKET DATASET CREATED")
    print(f"Rows: {len(dataset):,}")
    print(f"Columns: {len(dataset.columns)}")
    print(f"Output: {OUTPUT_FILE}")
    print("\nMissing values:")
    print(dataset[["price_eur_mwh"] + driver_columns].isna().sum())
    print("\nMerge report:")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
