
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "reports"

DATASETS = {
    "hour": {
        "file": "smard_day_ahead_prices_hour.csv",
        "interval": pd.Timedelta(hours=1),
    },
    "quarterhour": {
        "file": "smard_day_ahead_prices_quarterhour.csv",
        "interval": pd.Timedelta(minutes=15),
    },
}


def diagnose_dataset(name, config):
    file_path = RAW_DIR / config["file"]
    expected_interval = config["interval"]

    print("\n" + "=" * 70)
    print(f"TIMESTAMP DIAGNOSTICS: {name.upper()}")
    print("=" * 70)

    df = pd.read_csv(file_path)

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"],
        utc=True,
        errors="coerce",
    )

    df["price_eur_mwh"] = pd.to_numeric(
        df["price_eur_mwh"],
        errors="coerce",
    )

    df = (
        df.dropna(subset=["timestamp_utc"])
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )

    df["timestamp_local"] = (
        df["timestamp_utc"].dt.tz_convert("Europe/Berlin")
    )

    differences = df["timestamp_utc"].diff()

    gap_indices = differences[
        differences > expected_interval
    ].index

    gap_records = []

    if len(gap_indices) == 0:
        print("No timestamp gaps detected.")

    for index in gap_indices:
        previous_time = df.loc[index - 1, "timestamp_utc"]
        next_time = df.loc[index, "timestamp_utc"]
        gap_duration = next_time - previous_time

        missing_count = int(
            gap_duration / expected_interval
        ) - 1

        record = {
            "dataset": name,
            "previous_timestamp_utc": previous_time,
            "next_timestamp_utc": next_time,
            "previous_timestamp_berlin": previous_time.tz_convert(
                "Europe/Berlin"
            ),
            "next_timestamp_berlin": next_time.tz_convert(
                "Europe/Berlin"
            ),
            "gap_duration": str(gap_duration),
            "missing_intervals": missing_count,
        }

        gap_records.append(record)

        print(f"\nGap detected:")
        print(f"  Last observation before gap: {record['previous_timestamp_berlin']}")
        print(f"  First observation after gap:  {record['next_timestamp_berlin']}")
        print(f"  Gap duration:                 {gap_duration}")
        print(f"  Missing intervals:            {missing_count}")

    # Inspect delivery timestamps later than the current
    # time in the German market timezone.
    now_berlin = pd.Timestamp.now(tz="Europe/Berlin")
    future_rows = df[
        df["timestamp_local"] > now_berlin
    ]

    print("\nDataset coverage:")
    print(f"  Observations: {len(df):,}")
    print(f"  First delivery timestamp: {df['timestamp_local'].min()}")
    print(f"  Last delivery timestamp:  {df['timestamp_local'].max()}")

    print("\nFuture-dated delivery observations:")
    print(f"  Count: {len(future_rows):,}")

    if not future_rows.empty:
        print(f"  First: {future_rows['timestamp_local'].min()}")
        print(f"  Last:  {future_rows['timestamp_local'].max()}")

    return gap_records


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    all_gaps = []

    for name, config in DATASETS.items():
        gaps = diagnose_dataset(name, config)
        all_gaps.extend(gaps)

    output_file = REPORT_DIR / "timestamp_gap_diagnostics.csv"

    if all_gaps:
        pd.DataFrame(all_gaps).to_csv(output_file, index=False)
        print(f"\nGap report saved to: {output_file}")
    else:
        print("\nNo gaps found. No gap report was necessary.")

    print("\n" + "=" * 70)
    print("TIMESTAMP DIAGNOSTICS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
