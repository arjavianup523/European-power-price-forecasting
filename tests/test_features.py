
import pandas as pd

from src.features.build_features import (
    add_price_lags,
    add_rolling_features,
    add_calendar_features
)


def test_add_price_lags():

    df = pd.DataFrame({
        "price_eur_mwh": range(1, 201)
    })

    result = add_price_lags(df)

    assert result.loc[1, "price_lag_1h"] == 1
    assert result.loc[24, "price_lag_24h"] == 1
    assert result.loc[48, "price_lag_48h"] == 1
    assert result.loc[168, "price_lag_168h"] == 1

    assert pd.isna(result.loc[0, "price_lag_1h"])


def test_add_rolling_features():

    df = pd.DataFrame({
        "price_eur_mwh": range(1, 301)
    })

    result = add_rolling_features(df)

    assert result.loc[71, "rolling_mean_24h"] == 12.5
    assert result.loc[215, "rolling_mean_168h"] == 84.5

    assert pd.isna(result.loc[70, "rolling_mean_24h"])


def test_add_calendar_features():

    df = pd.DataFrame({
        "timestamp_utc": [
            "2026-11-23 09:00:00+00:00",
            "2026-11-24 09:00:00+00:00",
            "2026-11-28 09:00:00+00:00"
        ]
    })

    result = add_calendar_features(df)

    assert result.loc[0, "hour"] == 10
    assert result.loc[0, "day_of_week"] == 0
    assert result.loc[0, "month"] == 11

    assert result.loc[0, "is_weekend"] == 0
    assert result.loc[2, "is_weekend"] == 1
