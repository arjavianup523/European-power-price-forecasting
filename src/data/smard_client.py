
from pathlib import Path
import argparse
import time

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE_URL = "https://www.smard.de/app/chart_data"
FILTER_ID = 4169
REGION = "DE-LU"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def create_session():
    """Create an HTTP session with retry handling."""

    retry = Retry(
        total=4,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    adapter = HTTPAdapter(max_retries=retry)

    session = requests.Session()
    session.mount("https://", adapter)

    session.headers.update({
        "User-Agent": "EuropeanPowerPriceForecasting/1.0"
    })

    return session


def get_timestamps(session, resolution):
    """Retrieve available SMARD data chunk timestamps."""

    url = f"{BASE_URL}/{FILTER_ID}/{REGION}/index_{resolution}.json"

    response = session.get(url, timeout=30)
    response.raise_for_status()

    timestamps = response.json()["timestamps"]

    return sorted(timestamps)


def fetch_chunk(session, timestamp, resolution):
    """Download one historical data chunk."""

    filename = f"{FILTER_ID}_{REGION}_{resolution}_{timestamp}.json"
    url = f"{BASE_URL}/{FILTER_ID}/{REGION}/{filename}"

    response = session.get(url, timeout=30)
    response.raise_for_status()

    return response.json().get("series", [])


def download_prices(resolution="hour", limit=None):
    """Download SMARD day-ahead electricity prices."""

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    session = create_session()

    print(f"Fetching {resolution} day-ahead prices for {REGION}...")

    timestamps = get_timestamps(session, resolution)

    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be a positive integer.")

        timestamps = timestamps[-limit:]

    print(f"Chunks selected: {len(timestamps)}")

    records = []

    for index, timestamp in enumerate(timestamps, start=1):

        series = fetch_chunk(session, timestamp, resolution)
        records.extend(series)

        if index % 25 == 0 or index == len(timestamps):
            print(f"Downloaded {index}/{len(timestamps)} chunks")

        time.sleep(0.1)

    if not records:
        raise RuntimeError("No price observations were returned.")

    # Convert the original source records into a DataFrame.
    # Missing source prices are preserved as NaN.
    df = pd.DataFrame(
        records,
        columns=["timestamp_ms", "price_eur_mwh"]
    )

    df["timestamp_ms"] = pd.to_numeric(
        df["timestamp_ms"],
        errors="coerce"
    )

    df["price_eur_mwh"] = pd.to_numeric(
        df["price_eur_mwh"],
        errors="coerce"
    )

    # Convert Unix timestamps into timezone-aware datetime columns.
    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_ms"],
        unit="ms",
        utc=True,
        errors="coerce"
    )

    df["timestamp_local"] = (
        df["timestamp_utc"].dt.tz_convert("Europe/Berlin")
    )

    # Preserve rows with missing prices.
    # Only remove invalid timestamps and duplicate timestamps.
    df = (
        df.dropna(subset=["timestamp_ms"])
        .drop_duplicates(subset=["timestamp_ms"])
        .sort_values("timestamp_ms")
        .reset_index(drop=True)
    )

    output_file = (
        RAW_DATA_DIR /
        f"smard_day_ahead_prices_{resolution}.csv"
    )

    df.to_csv(output_file, index=False)

    missing_prices = df["price_eur_mwh"].isna().sum()

    print("\nDownload completed.")
    print("-" * 45)
    print(f"Resolution: {resolution}")
    print(f"Observations: {len(df):,}")
    print(f"Missing prices preserved: {missing_prices:,}")
    print(f"Valid prices: {df['price_eur_mwh'].notna().sum():,}")
    print(f"First timestamp: {df['timestamp_local'].iloc[0]}")
    print(f"Last timestamp: {df['timestamp_local'].iloc[-1]}")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Download German day-ahead electricity prices from SMARD."
    )

    parser.add_argument(
        "--resolution",
        choices=["hour", "quarterhour"],
        default="hour",
        help="Data resolution: hour or quarterhour."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of most recent chunks to download. Omit for full history."
    )

    args = parser.parse_args()

    download_prices(
        resolution=args.resolution,
        limit=args.limit
    )
