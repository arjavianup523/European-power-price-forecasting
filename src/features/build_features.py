
import pandas as pd


def add_price_lags(df):
    """
    Create historical electricity price lag features.
    """

    features = df.copy()

    features["price_lag_1h"] = features["price_eur_mwh"].shift(1)
    features["price_lag_24h"] = features["price_eur_mwh"].shift(24)
    features["price_lag_48h"] = features["price_eur_mwh"].shift(48)
    features["price_lag_168h"] = features["price_eur_mwh"].shift(168)

    return features


def add_rolling_features(df):
    """
    Create historical rolling statistical features.
    """

    features = df.copy()

    historical_prices = features["price_eur_mwh"].shift(48)

    features["rolling_mean_24h"] = (
        historical_prices.rolling(window=24, min_periods=24).mean()
    )

    features["rolling_mean_168h"] = (
        historical_prices.rolling(window=168, min_periods=168).mean()
    )

    features["rolling_std_168h"] = (
        historical_prices.rolling(window=168, min_periods=168).std()
    )

    return features


def add_calendar_features(df):
    """
    Create calendar-based electricity price features.
    """

    features = df.copy()

    features["timestamp_local"] = pd.to_datetime(
        features["timestamp_utc"], utc=True
    ).dt.tz_convert("Europe/Berlin")

    features["hour"] = features["timestamp_local"].dt.hour
    features["day_of_week"] = features["timestamp_local"].dt.day_of_week
    features["day_of_month"] = features["timestamp_local"].dt.day
    features["month"] = features["timestamp_local"].dt.month
    features["day_of_year"] = features["timestamp_local"].dt.day_of_year

    features["is_weekend"] = (
        features["day_of_week"] >= 5
    ).astype(int)

    return features
