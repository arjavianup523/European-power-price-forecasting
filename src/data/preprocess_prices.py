
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"


QUARTER_HOUR_FILE = (
    RAW_DATA_DIR / "smard_day_ahead_prices_quarterhour.csv"
)

HOURLY_FILE = (
    RAW_DATA_DIR / "smard_day_ahead_prices_hour.csv"
)

OUTPUT_FILE = (
    PROCESSED_DATA_DIR / "processed_hourly_power_prices.csv"
)

QUALITY_REPORT_FILE = (
    REPORTS_DIR / "preprocessing_quality_report.csv"
)


# --------------------------------------------------
# LOAD RAW DATA
# --------------------------------------------------

def load_price_data(file_path):
    """Load raw SMARD price data without filling missing values."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required dataset not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    required_columns = [
        "timestamp_ms",
        "price_eur_mwh",
        "timestamp_utc",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns in {file_path.name}: "
            f"{missing_columns}"
        )

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"],
        utc=True,
        errors="coerce"
    )

    df["price_eur_mwh"] = pd.to_numeric(
        df["price_eur_mwh"],
        errors="coerce"
    )

    df = df.dropna(subset=["timestamp_utc"]).copy()

    if df["timestamp_utc"].duplicated().any():
        raise ValueError(
            f"Duplicate timestamps found in {file_path.name}"
        )

    return df.sort_values("timestamp_utc").reset_index(drop=True)


# --------------------------------------------------
# AGGREGATE QUARTER-HOURLY PRICES
# --------------------------------------------------

def aggregate_quarter_hourly(df):
    """
    Aggregate quarter-hourly prices into UTC hours.

    An hourly price is calculated only when exactly
    four quarter-hour observations exist and all four
    have valid prices.
    """

    df = df.copy()

    # Assign each observation to its UTC delivery hour.
    df["hour_utc"] = df["timestamp_utc"].dt.floor("h")

    grouped = df.groupby("hour_utc")

    hourly = grouped.agg(
        quarterhour_records=("price_eur_mwh", "size"),
        valid_quarterhour_prices=("price_eur_mwh", "count"),
        quarterhour_price_mean=("price_eur_mwh", "mean"),
        quarterhour_price_min=("price_eur_mwh", "min"),
        quarterhour_price_max=("price_eur_mwh", "max"),
    ).reset_index()

    # Require all four quarter-hour prices.
    complete_hours = (
        (hourly["quarterhour_records"] == 4)
        & (hourly["valid_quarterhour_prices"] == 4)
    )

    hourly["price_eur_mwh"] = np.where(
        complete_hours,
        hourly["quarterhour_price_mean"],
        np.nan
    )

    hourly["quarterhour_coverage"] = (
        hourly["valid_quarterhour_prices"] / 4
    )

    hourly["is_complete_hour"] = complete_hours

    hourly = hourly.rename(
        columns={"hour_utc": "timestamp_utc"}
    )

    return hourly


# --------------------------------------------------
# COMPARE WITH ORIGINAL HOURLY SOURCE
# --------------------------------------------------

def compare_hourly_sources(processed, original_hourly):
    """Compare aggregated prices with SMARD hourly prices."""

    reference = original_hourly[
        ["timestamp_utc", "price_eur_mwh"]
    ].rename(
        columns={
            "price_eur_mwh": "smard_hourly_price_eur_mwh"
        }
    )

    merged = processed.merge(
        reference,
        on="timestamp_utc",
        how="outer",
        validate="one_to_one"
    )

    merged["source_price_difference_eur_mwh"] = (
        merged["price_eur_mwh"]
        - merged["smard_hourly_price_eur_mwh"]
    )

    merged["has_aggregated_price"] = (
        merged["price_eur_mwh"].notna()
    )

    merged["has_smard_hourly_price"] = (
        merged["smard_hourly_price_eur_mwh"].notna()
    )

    merged["source_prices_match"] = np.isclose(
        merged["price_eur_mwh"],
        merged["smard_hourly_price_eur_mwh"],
        rtol=0,
        atol=0.01,
        equal_nan=False
    )

    return merged


# --------------------------------------------------
# GENERATE QUALITY REPORT
# --------------------------------------------------

def generate_quality_report(df, now_utc):
    """Create a summary of the processed dataset."""

    valid_prices = df["price_eur_mwh"].dropna()

    future_records = df[
        df["timestamp_utc"] > now_utc
    ]

    comparable = df[
        df["price_eur_mwh"].notna()
        & df["smard_hourly_price_eur_mwh"].notna()
    ]

    if not comparable.empty:
        differences = (
            comparable["source_price_difference_eur_mwh"]
        )

        mean_absolute_difference = differences.abs().mean()
        maximum_absolute_difference = differences.abs().max()
        matching_prices = comparable["source_prices_match"].sum()
    else:
        mean_absolute_difference = np.nan
        maximum_absolute_difference = np.nan
        matching_prices = 0

    report = [
        ("total_observations", len(df)),
        ("valid_aggregated_prices", len(valid_prices)),
        ("missing_aggregated_prices", df["price_eur_mwh"].isna().sum()),
        ("complete_hours", df["is_complete_hour"].sum()),
        ("incomplete_hours", (~df["is_complete_hour"]).sum()),
        ("future_delivery_records", len(future_records)),
        ("future_records_with_prices", future_records["price_eur_mwh"].notna().sum()),
        ("comparable_source_prices", len(comparable)),
        ("source_prices_matching_within_0_01", matching_prices),
        ("mean_absolute_source_difference_eur_mwh", mean_absolute_difference),
        ("maximum_absolute_source_difference_eur_mwh", maximum_absolute_difference),
        ("minimum_price_eur_mwh", valid_prices.min() if not valid_prices.empty else np.nan),
        ("maximum_price_eur_mwh", valid_prices.max() if not valid_prices.empty else np.nan),
        ("mean_price_eur_mwh", valid_prices.mean()),
        ("median_price_eur_mwh", valid_prices.median()),
    ]

    return pd.DataFrame(
        report,
        columns=["metric", "value"]
    )


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def main():

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 65)
    print("EUROPEAN POWER PRICE PREPROCESSING PIPELINE")
    print("=" * 65)

    print("\n1. Loading quarter-hourly raw data...")

    quarterhourly = load_price_data(
        QUARTER_HOUR_FILE
    )

    print(
        f"Quarter-hourly observations: "
        f"{len(quarterhourly):,}"
    )

    print("\n2. Loading original hourly raw data...")

    original_hourly = load_price_data(
        HOURLY_FILE
    )

    print(
        f"Original hourly observations: "
        f"{len(original_hourly):,}"
    )

    print("\n3. Aggregating quarter-hourly prices...")

    processed = aggregate_quarter_hourly(
        quarterhourly
    )

    print(
        f"Aggregated hourly observations: "
        f"{len(processed):,}"
    )

    print("\n4. Comparing with original SMARD hourly prices...")

    processed = compare_hourly_sources(
        processed,
        original_hourly
    )

    # Convert UTC timestamps into German local time.
    processed["timestamp_local"] = (
        processed["timestamp_utc"]
        .dt.tz_convert("Europe/Berlin")
    )

    # Informational flag only.
    # Future delivery records are NOT automatically deleted.
    now_utc = pd.Timestamp.now(tz="UTC")

    processed["is_future_delivery"] = (
        processed["timestamp_utc"] > now_utc
    )

    # Arrange columns for easier inspection.
    column_order = [
        "timestamp_utc",
        "timestamp_local",
        "price_eur_mwh",
        "smard_hourly_price_eur_mwh",
        "source_price_difference_eur_mwh",
        "quarterhour_records",
        "valid_quarterhour_prices",
        "quarterhour_coverage",
        "quarterhour_price_min",
        "quarterhour_price_max",
        "is_complete_hour",
        "has_aggregated_price",
        "has_smard_hourly_price",
        "source_prices_match",
        "is_future_delivery",
    ]

    processed = processed[column_order]

    processed = (
        processed
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )

    print("\n5. Generating preprocessing quality report...")

    report = generate_quality_report(
        processed,
        now_utc
    )

    # Save processed dataset.
    processed.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # Save quality report.
    report.to_csv(
        QUALITY_REPORT_FILE,
        index=False
    )

    print("\n" + "=" * 65)
    print("PREPROCESSING COMPLETED")
    print("=" * 65)

    print("\nPROCESSED DATASET SUMMARY")

    print(
        f"Total observations: "
        f"{len(processed):,}"
    )

    print(
        f"Valid aggregated prices: "
        f"{processed['price_eur_mwh'].notna().sum():,}"
    )

    print(
        f"Missing aggregated prices: "
        f"{processed['price_eur_mwh'].isna().sum():,}"
    )

    print(
        f"Complete hours: "
        f"{processed['is_complete_hour'].sum():,}"
    )

    print(
        f"Incomplete hours: "
        f"{(~processed['is_complete_hour']).sum():,}"
    )

    print("\nSOURCE COMPARISON")

    print(
        f"Comparable prices: "
        f"{report.loc[report['metric'] == 'comparable_source_prices', 'value'].iloc[0]}"
    )

    print(
        f"Matching prices within €0.01/MWh: "
        f"{report.loc[report['metric'] == 'source_prices_matching_within_0_01', 'value'].iloc[0]}"
    )

    print("\nOUTPUT FILES")

    print(f"Processed data: {OUTPUT_FILE}")
    print(f"Quality report: {QUALITY_REPORT_FILE}")

    print("\nRaw source files were not modified.")


if __name__ == "__main__":
    main()
