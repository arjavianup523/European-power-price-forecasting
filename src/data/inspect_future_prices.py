
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def inspect_dataset(filename, resolution, now_utc):
    file_path = RAW_DATA_DIR / filename

    df = pd.read_csv(file_path)

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"],
        utc=True,
        errors="coerce"
    )

    df["price_eur_mwh"] = pd.to_numeric(
        df["price_eur_mwh"],
        errors="coerce"
    )

    future = df[df["timestamp_utc"] > now_utc]
    historical = df[df["timestamp_utc"] <= now_utc]

    future_with_prices = future[
        future["price_eur_mwh"].notna()
    ]

    future_without_prices = future[
        future["price_eur_mwh"].isna()
    ]

    historical_missing = historical[
        historical["price_eur_mwh"].isna()
    ]

    print("\n" + "=" * 65)
    print(f"FUTURE-DATA DIAGNOSTIC: {resolution.upper()}")
    print("=" * 65)

    print(f"Total observations: {len(df):,}")
    print(f"Future observations: {len(future):,}")
    print(f"Future observations with prices: {len(future_with_prices):,}")
    print(f"Future observations without prices: {len(future_without_prices):,}")
    print(f"Historical observations: {len(historical):,}")
    print(f"Historical missing prices: {len(historical_missing):,}")

    if not future.empty:
        print("\nFuture timestamp range:")
        print(f"First: {future['timestamp_utc'].min()}")
        print(f"Last:  {future['timestamp_utc'].max()}")

    if not future_with_prices.empty:
        print("\nWARNING: Future records containing actual prices:")
        print(
            future_with_prices[
                ["timestamp_utc", "price_eur_mwh"]
            ].head(20).to_string(index=False)
        )
    else:
        print("\nNo future records with actual prices found.")

    if not historical_missing.empty:
        print("\nHistorical records with missing prices:")
        print(
            historical_missing[
                ["timestamp_utc", "price_eur_mwh"]
            ].head(10).to_string(index=False)
        )


def main():
    now_utc = pd.Timestamp.now(tz="UTC")

    print(f"Diagnostic reference time (UTC): {now_utc}")

    inspect_dataset(
        "smard_day_ahead_prices_hour.csv",
        "hour",
        now_utc
    )

    inspect_dataset(
        "smard_day_ahead_prices_quarterhour.csv",
        "quarterhour",
        now_utc
    )


if __name__ == "__main__":
    main()
