
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data" / "raw" / "smard_energy_drivers_hourly.csv"

BASE = "https://www.smard.de/app/chart_data"
REGION = "DE"
RESOLUTION = "hour"

DRIVERS = {
    "grid_load": 410,
    "wind_offshore": 1225,
    "wind_onshore": 4067,
    "solar_generation": 4068,
}

END = pd.Timestamp.now(tz="UTC")
START = END - pd.Timedelta(days=30)

HEADERS = {"User-Agent": "PowerForecasting/1.0"}


def fetch_json(url):
    response = requests.get(url, headers=HEADERS, timeout=25)
    response.raise_for_status()
    return response.json()


def download_chunk(args):
    filter_id, timestamp = args
    url = (
        f"{BASE}/{filter_id}/{REGION}/"
        f"{filter_id}_{REGION}_{RESOLUTION}_{timestamp}.json"
    )
    return fetch_json(url).get("series", [])


def download_driver(name, filter_id):
    print(f"Fetching {name}...", flush=True)

    index_url = f"{BASE}/{filter_id}/{REGION}/index_{RESOLUTION}.json"
    timestamps = fetch_json(index_url).get("timestamps", [])

    start_ms = int(START.timestamp() * 1000)
    end_ms = int(END.timestamp() * 1000)

    selected = [
        ts for ts in timestamps
        if start_ms <= ts <= end_ms
    ]

    previous = [ts for ts in timestamps if ts < start_ms]
    if previous:
        selected.insert(0, previous[-1])

    if not selected:
        raise ValueError(f"No chunks available for {name}")

    records = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(download_chunk, (filter_id, ts))
            for ts in selected
        ]

        for future in as_completed(futures):
            for item in future.result():
                if len(item) == 2:
                    records.append(item)

    df = pd.DataFrame(records, columns=["timestamp_ms", name])
    df["timestamp_ms"] = pd.to_numeric(
        df["timestamp_ms"], errors="coerce"
    )
    df[name] = pd.to_numeric(df[name], errors="coerce")
    df = df.dropna(subset=["timestamp_ms"])
    df = df.drop_duplicates("timestamp_ms")

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_ms"], unit="ms", utc=True
    )

    df = df[
        (df["timestamp_utc"] >= START)
        & (df["timestamp_utc"] <= END)
    ]

    print(f"  {len(df):,} rows", flush=True)

    return df[["timestamp_utc", name]]


def main():
    combined = None

    for name, filter_id in DRIVERS.items():
        df = download_driver(name, filter_id)

        if combined is None:
            combined = df
        else:
            combined = combined.merge(
                df, on="timestamp_utc", how="outer"
            )

    combined["timestamp_local"] = (
        combined["timestamp_utc"].dt.tz_convert("Europe/Berlin")
    )

    combined = combined.sort_values("timestamp_utc")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT, index=False)

    print("\nCOMPLETE")
    print(f"Rows: {len(combined):,}")
    print(f"File: {OUTPUT}")
    print(f"Size: {OUTPUT.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
