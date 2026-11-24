
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]

PRICE_FILE = ROOT / "data/processed/processed_hourly_power_prices.csv"
MODEL_FILE = ROOT / "reports/day_ahead_walk_forward_results.csv"
DRIVER_FILE = ROOT / "reports/energy_driver_correlations.csv"


st.set_page_config(
    page_title="European Power Market Intelligence",
    page_icon="⚡",
    layout="wide",
)


@st.cache_data
def load_data():
    prices = pd.read_csv(PRICE_FILE)

    prices["timestamp_utc"] = pd.to_datetime(
        prices["timestamp_utc"],
        utc=True,
        errors="coerce",
    )

    # Remove invalid timestamps and future delivery observations.
    now_utc = pd.Timestamp.now(tz="UTC")

    prices = prices.dropna(
        subset=["timestamp_utc"]
    ).copy()

    prices = prices.loc[
        prices["timestamp_utc"] <= now_utc
    ].copy()

    prices = prices.sort_values(
        "timestamp_utc"
    ).reset_index(drop=True)

    model_results = pd.read_csv(MODEL_FILE)

    if DRIVER_FILE.exists():
        drivers = pd.read_csv(DRIVER_FILE)
    else:
        drivers = pd.DataFrame()

    return prices, model_results, drivers


def calculate_css(
    electricity,
    gas,
    carbon,
    efficiency,
    emissions_factor,
):
    gas_cost = gas / efficiency
    carbon_cost = carbon * emissions_factor / efficiency

    return (
        electricity - gas_cost - carbon_cost,
        gas_cost,
        carbon_cost,
    )


st.title("⚡ European Power Price Forecasting")

st.caption(
    "Germany (DE-LU) | Day-ahead electricity prices | "
    "Energy market intelligence"
)

try:
    prices, model_results, drivers = load_data()

except FileNotFoundError as error:
    st.error(f"Required project data file not found: {error}")
    st.stop()


# MODEL PERFORMANCE

st.header("Forecasting Model Performance")

baseline_mae = model_results["baseline_mae"].mean()
model_mae = model_results["model_mae"].mean()
baseline_smape = model_results["baseline_smape"].mean()
model_smape = model_results["model_smape"].mean()

improvement = (
    (baseline_mae - model_mae) / baseline_mae * 100
    if baseline_mae != 0
    else 0
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Day-ahead Model MAE",
    f"€{model_mae:.2f}/MWh",
)

col2.metric(
    "Persistence MAE",
    f"€{baseline_mae:.2f}/MWh",
)

col3.metric(
    "MAE Improvement",
    f"{improvement:.1f}%",
)

col4.metric(
    "Model sMAPE",
    f"{model_smape:.2f}%",
)


# PRICE MARKET OVERVIEW

st.header("Electricity Price Market Overview")

valid_prices = prices.dropna(
    subset=["price_eur_mwh"]
).copy()

if valid_prices.empty:
    st.error(
        "No valid historical electricity price observations "
        "are available up to the current time."
    )
    st.stop()

valid_prices = valid_prices.sort_values(
    "timestamp_utc"
).reset_index(drop=True)

latest_row = valid_prices.iloc[-1]
latest_price = latest_row["price_eur_mwh"]
latest_timestamp = latest_row["timestamp_utc"]

average_price = valid_prices["price_eur_mwh"].mean()
median_price = valid_prices["price_eur_mwh"].median()

negative_hours = (
    valid_prices["price_eur_mwh"] < 0
).sum()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Last Recorded Price",
    f"€{latest_price:.2f}/MWh",
)

col2.metric(
    "Historical Average",
    f"€{average_price:.2f}/MWh",
)

col3.metric(
    "Historical Median",
    f"€{median_price:.2f}/MWh",
)

col4.metric(
    "Negative Price Hours",
    f"{negative_hours:,}",
)

st.caption(
    "Latest eligible delivery timestamp: "
    f"{latest_timestamp.tz_convert('Europe/Berlin'):%d %b %Y, %H:%M} "
    "(Europe/Berlin). Future-dated observations are excluded."
)


# PRICE HISTORY

st.header("Electricity Price History")

# Use the original price data, not valid_prices.
# This preserves NaN observations so missing periods appear as gaps.
history = prices.tail(24 * 30).copy()

history["timestamp_local"] = (
    history["timestamp_utc"].dt.tz_convert("Europe/Berlin")
)

history = history.sort_values(
    "timestamp_utc"
)

history = history.set_index("timestamp_local")

st.line_chart(
    history["price_eur_mwh"],
    x_label="Time",
    y_label="Electricity price (€/MWh)",
)


# WALK-FORWARD MODEL EVALUATION

st.header("Walk-Forward Validation")

st.caption(
    "Four chronological evaluation folds. "
    "Lower MAE and sMAPE indicate better performance."
)

display_results = model_results[
    [
        "fold",
        "training_rows",
        "testing_rows",
        "baseline_mae",
        "model_mae",
        "baseline_smape",
        "model_smape",
    ]
].copy()

st.dataframe(
    display_results.round(2),
    use_container_width=True,
    hide_index=True,
)

chart_results = model_results.set_index("fold")[
    ["baseline_mae", "model_mae"]
]

st.bar_chart(
    chart_results,
    y_label="MAE (€/MWh)",
    stack=False,
)


# ENERGY MARKET DRIVERS

st.header("Energy Market Driver Analysis")

if not drivers.empty:
    st.caption(
        "Exploratory correlations from the available 30-day "
        "contemporaneous market-driver sample. "
        "Correlation does not establish causation."
    )

    st.dataframe(
        drivers,
        use_container_width=True,
        hide_index=True,
    )

else:
    st.info(
        "Driver correlation report is not available yet."
    )


# CLEAN SPARK SPREAD CALCULATOR

st.header("Clean Spark Spread Calculator")

st.caption(
    "Illustrative gas-fired power generation economics. "
    "This is not a live market feed or a net-profit calculation."
)

col1, col2, col3 = st.columns(3)

with col1:
    electricity = st.number_input(
        "Electricity price (€/MWh)",
        value=120.0,
        step=1.0,
    )

with col2:
    gas = st.number_input(
        "Natural gas price (€/MWh thermal)",
        value=35.0,
        min_value=0.0,
        step=1.0,
    )

with col3:
    carbon = st.number_input(
        "Carbon allowance price (€/tCO₂)",
        value=80.0,
        min_value=0.0,
        step=1.0,
    )

efficiency = st.slider(
    "Power plant electrical efficiency",
    min_value=0.30,
    max_value=0.70,
    value=0.55,
    step=0.01,
)

emissions_factor = st.number_input(
    "Emissions factor (tCO₂/MWh thermal)",
    min_value=0.0,
    value=0.202,
    step=0.001,
    format="%.3f",
)

if efficiency > 0:
    spread, gas_cost, carbon_cost = calculate_css(
        electricity,
        gas,
        carbon,
        efficiency,
        emissions_factor,
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Gas Cost",
        f"€{gas_cost:.2f}/MWh",
    )

    col2.metric(
        "Carbon Cost",
        f"€{carbon_cost:.2f}/MWh",
    )

    col3.metric(
        "Clean Spark Spread",
        f"€{spread:.2f}/MWh",
    )

    if spread > 0:
        st.success("Positive clean spark spread")
    else:
        st.warning("Negative or zero clean spark spread")

    st.caption(
        "Excludes operating and maintenance costs, start-up costs, "
        "and other plant-specific expenses."
    )


# PROJECT METHODOLOGY

with st.expander("Methodology and limitations"):
    st.markdown(
        """
        **Forecasting**
        - 24-hour-ahead electricity price target.
        - Historical price and calendar features.
        - Chronological expanding-window validation.
        - Persistence benchmark comparison.

        **Market analysis**
        - Historical wind, solar and grid-load observations.
        - Exploratory correlation analysis.
        - Actual market-driver observations are not treated as
          verified day-ahead forecast inputs.

        **Data quality**
        - Future-dated delivery observations are excluded
          from dashboard historical price metrics.
        - Historical source publication timestamps are not
          available to verify all feature vintages.
        - Missing hourly price observations are preserved
          as gaps in the historical price chart.

        **Energy economics**
        - Simplified clean spark spread.
        - Configurable plant efficiency and emissions factor.
        - Not a net-profit or trading recommendation.
        """
    )

st.divider()

st.caption(
    "European Power Price Forecasting & Energy Market Intelligence System | "
    "Portfolio project"
)
