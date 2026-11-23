
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "processed_hourly_power_prices.csv"
)

REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"


def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"],
        utc=True
    )

    df["timestamp_local"] = (
        df["timestamp_utc"].dt.tz_convert("Europe/Berlin")
    )

    df["price_eur_mwh"] = pd.to_numeric(
        df["price_eur_mwh"],
        errors="coerce"
    )

    return df.sort_values("timestamp_utc").reset_index(drop=True)


def create_descriptive_statistics(df):
    prices = df["price_eur_mwh"].dropna()

    statistics = {
        "total_observations": len(df),
        "valid_prices": len(prices),
        "missing_prices": df["price_eur_mwh"].isna().sum(),
        "negative_prices": (prices < 0).sum(),
        "zero_prices": (prices == 0).sum(),
        "positive_prices": (prices > 0).sum(),
        "negative_price_percentage": (prices < 0).mean() * 100,
        "mean_eur_mwh": prices.mean(),
        "median_eur_mwh": prices.median(),
        "standard_deviation": prices.std(),
        "minimum_eur_mwh": prices.min(),
        "maximum_eur_mwh": prices.max(),
        "first_valid_price_timestamp": df.loc[
            df["price_eur_mwh"].notna(),
            "timestamp_local"
        ].min(),
        "last_valid_price_timestamp": df.loc[
            df["price_eur_mwh"].notna(),
            "timestamp_local"
        ].max(),
    }

    quantiles = prices.quantile(
        [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    )

    for quantile, value in quantiles.items():
        statistics[f"price_quantile_{quantile:.2f}"] = value

    report = pd.DataFrame(
        statistics.items(),
        columns=["metric", "value"]
    )

    report.to_csv(
        REPORT_DIR / "price_descriptive_statistics.csv",
        index=False
    )

    return report


def prepare_analysis_columns(df):
    df = df.copy()

    df["year"] = df["timestamp_local"].dt.year
    df["month"] = df["timestamp_local"].dt.month
    df["year_month"] = (
        df["timestamp_local"].dt.strftime("%Y-%m")
    )

    df["hour"] = df["timestamp_local"].dt.hour
    df["day_of_week"] = (
        df["timestamp_local"].dt.day_name()
    )

    df["date"] = df["timestamp_local"].dt.date

    return df


def plot_price_history(df):
    prices = df.dropna(subset=["price_eur_mwh"]).copy()

    plt.figure(figsize=(16, 6))

    plt.plot(
        prices["timestamp_local"],
        prices["price_eur_mwh"],
        linewidth=0.35
    )

    plt.title("German Day-Ahead Electricity Prices")
    plt.xlabel("Delivery date")
    plt.ylabel("Price (€/MWh)")
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "01_price_history.png",
        dpi=200
    )

    plt.close()


def plot_price_distribution(df):
    prices = df["price_eur_mwh"].dropna()

    plt.figure(figsize=(12, 6))

    sns.histplot(
        prices,
        bins=100,
        kde=True
    )

    plt.title("Distribution of German Electricity Prices")
    plt.xlabel("Price (€/MWh)")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.2)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "02_price_distribution.png",
        dpi=200
    )

    plt.close()


def plot_monthly_average(df):
    monthly = (
        df.dropna(subset=["price_eur_mwh"])
        .groupby("year_month")["price_eur_mwh"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(16, 6))

    plt.plot(
        monthly["year_month"],
        monthly["price_eur_mwh"],
        marker=".",
        linewidth=1
    )

    plt.title("Monthly Average Day-Ahead Electricity Price")
    plt.xlabel("Month")
    plt.ylabel("Average price (€/MWh)")
    plt.xticks(rotation=60)
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "03_monthly_average_price.png",
        dpi=200
    )

    plt.close()


def plot_hourly_price_profile(df):
    hourly = (
        df.dropna(subset=["price_eur_mwh"])
        .groupby("hour")["price_eur_mwh"]
        .mean()
    )

    plt.figure(figsize=(12, 6))

    plt.plot(
        hourly.index,
        hourly.values,
        marker="o"
    )

    plt.title("Average Electricity Price by Hour of Day")
    plt.xlabel("Hour of day (Europe/Berlin)")
    plt.ylabel("Average price (€/MWh)")
    plt.xticks(range(24))
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "04_hourly_price_profile.png",
        dpi=200
    )

    plt.close()


def plot_negative_price_frequency(df):
    valid = df.dropna(subset=["price_eur_mwh"]).copy()

    valid["is_negative"] = (
        valid["price_eur_mwh"] < 0
    )

    annual = (
        valid.groupby("year")["is_negative"]
        .mean()
        .mul(100)
        .reset_index(name="negative_price_percentage")
    )

    plt.figure(figsize=(12, 6))

    plt.bar(
        annual["year"].astype(str),
        annual["negative_price_percentage"]
    )

    plt.title("Annual Frequency of Negative Electricity Prices")
    plt.xlabel("Year")
    plt.ylabel("Negative-price observations (%)")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "05_negative_price_frequency.png",
        dpi=200
    )

    plt.close()


def plot_missing_prices(df):
    missing = (
        df.assign(
            missing_price=df["price_eur_mwh"].isna()
        )
        .groupby("year")["missing_price"]
        .sum()
    )

    plt.figure(figsize=(12, 6))

    plt.bar(
        missing.index.astype(str),
        missing.values
    )

    plt.title("Missing Electricity Price Observations by Year")
    plt.xlabel("Year")
    plt.ylabel("Missing hourly observations")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "06_missing_prices.png",
        dpi=200
    )

    plt.close()


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("GERMAN ELECTRICITY MARKET: EXPLORATORY DATA ANALYSIS")
    print("=" * 65)

    print("\n1. Loading processed dataset...")

    df = load_data()
    df = prepare_analysis_columns(df)

    print(f"Observations: {len(df):,}")

    print("\n2. Calculating descriptive statistics...")

    statistics = create_descriptive_statistics(df)

    print(
        statistics.to_string(index=False)
    )

    print("\n3. Generating visual analysis...")

    plot_price_history(df)
    print("Created: 01_price_history.png")

    plot_price_distribution(df)
    print("Created: 02_price_distribution.png")

    plot_monthly_average(df)
    print("Created: 03_monthly_average_price.png")

    plot_hourly_price_profile(df)
    print("Created: 04_hourly_price_profile.png")

    plot_negative_price_frequency(df)
    print("Created: 05_negative_price_frequency.png")

    plot_missing_prices(df)
    print("Created: 06_missing_prices.png")

    print("\n" + "=" * 65)
    print("EXPLORATORY DATA ANALYSIS COMPLETED")
    print("=" * 65)

    print(f"Statistics: {REPORT_DIR / 'price_descriptive_statistics.csv'}")
    print(f"Charts: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
