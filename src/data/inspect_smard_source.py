
from pathlib import Path

import pandas as pd
import requests


BASE_URL = "https://www.smard.de/app/chart_data"
FILTER_ID = 4169
REGION = "DE-LU"

RESOLUTIONS = ["hour", "quarterhour"]

START = pd.Timestamp("2026-11-19 00:00", tz="Europe/Berlin")
END = pd.Timestamp("2026-11-22 12:00", tz="Europe/Berlin")


def inspect_resolution(session, resolution):
    print("\n" + "=" * 70)
    print(f"SMARD SOURCE INSPECTION: {resolution.upper()}")
    print("=" * 70)

    index_url = (
        f"{BASE_URL}/{FILTER_ID}/{REGION}/"
        f"index_{resolution}.json"
    )

    response = session.get(index_url, timeout=30)
    response.raise_for_status()

    timestamps = sorted(response.json()["timestamps"])

    target_ms = int(START.tz_convert("UTC").timestamp() * 1000)

    eligible = [ts for ts in timestamps if ts <= target_ms]

    if not eligible:
        raise RuntimeError("No weekly source chunk found for target date.")

    chunk_timestamp = eligible[-1]

    filename = (
        f"{FILTER_ID}_{REGION}_{resolution}_{chunk_timestamp}.json"
    )

    chunk_url = (
        f"{BASE_URL}/{FILTER_ID}/{REGION}/{filename}"
    )

    response = session.get(chunk_url, timeout=30)
    response.raise_for_status()

    payload = response.json()
    records = payload.get("series", [])

    df = pd.DataFrame(
        records,
        columns=["timestamp_ms", "price_eur_mwh"],
    )

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_ms"],
        unit="ms",
        utc=True,
    )

    df["timestamp_local"] = (
        df["timestamp_utc"].dt.tz_convert("Europe/Berlin")
    )

    df["price_eur_mwh"] = pd.to_numeric(
        df["price_eur_mwh"],
        errors="coerce",
    )

    window = df[
        (df["timestamp_local"] >= START)
        & (df["timestamp_local"] < END)
    ].copy()

    print(f"Source chunk: {filename}")
    print(f"Total source records: {len(df):,}")
    print(f"Records in inspection window: {len(window):,}")
    print(f"Non-null prices in window: {window['price_eur_mwh'].notna().sum():,}")
    print(f"Null prices in window: {window['price_eur_mwh'].isna().sum():,}")

    print("\nSource observations around the missing period:")

    print(
        window[
            ["timestamp_local", "price_eur_mwh"]
        ].to_string(index=False)
    )

    return window


def main():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "EuropeanPowerPriceForecasting/1.0"
    })

    for resolution in RESOLUTIONS:
        inspect_resolution(session, resolution)

    print("\n" + "=" * 70)
    print("SOURCE INSPECTION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
