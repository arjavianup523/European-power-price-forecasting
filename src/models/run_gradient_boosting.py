
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "power_price_features.csv"
REPORT_FILE = PROJECT_ROOT / "reports" / "gradient_boosting_evaluation.csv"


class DecisionStump:
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left_value = 0.0
        self.right_value = 0.0

    def fit(self, X, residuals, bins=32):

        best_loss = float("inf")

        for feature in range(X.shape[1]):

            values = X[:, feature]
            thresholds = np.unique(
                np.quantile(values, np.linspace(0.05, 0.95, bins))
            )

            for threshold in thresholds:

                left = values <= threshold
                right = ~left

                if not left.any() or not right.any():
                    continue

                left_value = residuals[left].mean()
                right_value = residuals[right].mean()

                loss = (
                    np.sum((residuals[left] - left_value) ** 2)
                    + np.sum((residuals[right] - right_value) ** 2)
                )

                if loss < best_loss:
                    best_loss = loss
                    self.feature = feature
                    self.threshold = threshold
                    self.left_value = left_value
                    self.right_value = right_value

        return self

    def predict(self, X):

        values = X[:, self.feature]

        return np.where(
            values <= self.threshold,
            self.left_value,
            self.right_value
        )


class GradientBoostingRegressor:

    def __init__(self, estimators=50, learning_rate=0.05):
        self.estimators = estimators
        self.learning_rate = learning_rate
        self.initial_prediction = 0.0
        self.models = []

    def fit(self, X, y):

        self.initial_prediction = np.mean(y)

        predictions = np.full(
            len(y), self.initial_prediction, dtype=float
        )

        self.models = []

        for _ in range(self.estimators):

            residuals = y - predictions

            model = DecisionStump()
            model.fit(X, residuals)

            update = model.predict(X)

            predictions += self.learning_rate * update

            self.models.append(model)

        return self

    def predict(self, X):

        predictions = np.full(
            len(X), self.initial_prediction, dtype=float
        )

        for model in self.models:
            predictions += self.learning_rate * model.predict(X)

        return predictions


def main():

    df = pd.read_csv(INPUT_FILE)

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"], utc=True
    )

    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    feature_columns = [
        "price_eur_mwh",
        "price_lag_1h",
        "price_lag_24h",
        "price_lag_48h",
        "price_lag_168h",
        "rolling_mean_24h",
        "rolling_mean_168h",
        "rolling_std_168h",
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
    ]

    df["target_price_24h"] = df["price_eur_mwh"].shift(-24)

    df = df.dropna(
        subset=feature_columns + ["target_price_24h"]
    ).copy()

    current_time = pd.Timestamp.now(tz="UTC")

    df = df[
        df["timestamp_utc"] + pd.Timedelta(hours=24) <= current_time
    ].copy()

    X = df[feature_columns].to_numpy(dtype=float)
    y = df["target_price_24h"].to_numpy(dtype=float)

    split_index = int(len(df) * 0.8)

    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    model = GradientBoostingRegressor(
        estimators=50,
        learning_rate=0.05
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = np.mean(np.abs(y_test - predictions))

    denominator = (np.abs(y_test) + np.abs(predictions)) / 2
    valid = denominator != 0

    smape = np.mean(
        100 * np.abs(y_test[valid] - predictions[valid])
        / denominator[valid]
    )

    results = pd.DataFrame([{
        "model": "NumPy Gradient Boosting",
        "training_observations": len(X_train),
        "testing_observations": len(X_test),
        "MAE_EUR_MWh": mae,
        "sMAPE_percent": smape,
        "estimators": model.estimators,
    }])

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORT_FILE, index=False)

    print("\nGRADIENT BOOSTING COMPLETE")
    print("--------------------------")
    print(results.to_string(index=False))
    print(f"\nReport saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
