
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "data/processed/power_market_dataset.csv"
REPORTS = ROOT / "reports"

DRIVERS = [
    "grid_load",
    "wind_offshore",
    "wind_onshore",
    "solar_generation",
]


def create_quartiles(series):
    # Rank first so repeated values do not create duplicate bin edges.
    ranked = series.rank(method="first")

    quartiles = pd.qcut(
        ranked,
        q=4,
        labels=["Q1", "Q2", "Q3", "Q4"]
    )

    return quartiles


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT)

    columns = ["price_eur_mwh"] + DRIVERS

    for column in columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    clean = df[columns].dropna()

    if len(clean) < 30:
        raise ValueError(
            "Insufficient complete observations for analysis."
        )

    pearson = clean.corr(method="pearson")["price_eur_mwh"]
    spearman = clean.corr(method="spearman")["price_eur_mwh"]

    correlations = pd.DataFrame({
        "driver": DRIVERS,
        "pearson_correlation": [
            pearson[driver] for driver in DRIVERS
        ],
        "spearman_correlation": [
            spearman[driver] for driver in DRIVERS
        ],
        "complete_observations": len(clean),
    })

    correlations["absolute_pearson"] = (
        correlations["pearson_correlation"].abs()
    )

    correlations = correlations.sort_values(
        "absolute_pearson",
        ascending=False
    )

    correlations.to_csv(
        REPORTS / "energy_driver_correlations.csv",
        index=False
    )

    clean.describe().T.to_csv(
        REPORTS / "energy_driver_summary.csv"
    )

    relationship_tables = []

    for driver in DRIVERS:
        temp = clean[[driver, "price_eur_mwh"]].copy()

        temp["driver_quartile"] = create_quartiles(
            temp[driver]
        )

        grouped = temp.groupby(
            "driver_quartile",
            observed=True
        )["price_eur_mwh"].agg(
            observations="count",
            average_price="mean",
            median_price="median",
            price_std="std"
        ).reset_index()

        grouped.insert(0, "driver", driver)
        relationship_tables.append(grouped)

    relationships = pd.concat(
        relationship_tables,
        ignore_index=True
    )

    relationships.to_csv(
        REPORTS / "energy_driver_price_relationships.csv",
        index=False
    )

    print("\nENERGY MARKET DRIVER ANALYSIS COMPLETE")
    print(f"Complete observations: {len(clean):,}")

    print("\nPRICE CORRELATIONS")
    print(correlations.to_string(index=False))

    print("\nPRICE BY DRIVER QUARTILE")
    print(relationships.to_string(index=False))

    print("\nReports saved in:", REPORTS)


if __name__ == "__main__":
    main()
