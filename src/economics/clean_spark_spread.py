
import argparse


def calculate_clean_spark_spread(
    electricity_price,
    gas_price,
    carbon_price,
    efficiency=0.55,
    emissions_factor=0.202,
):
    """
    Calculate clean spark spread in EUR/MWh of electricity.

    electricity_price: EUR/MWh electricity
    gas_price: EUR/MWh thermal
    carbon_price: EUR/tCO2
    efficiency: power plant electrical efficiency
    emissions_factor: tCO2/MWh thermal
    """

    values = [
        electricity_price,
        gas_price,
        carbon_price,
        efficiency,
        emissions_factor,
    ]

    if not all(map(lambda value: isinstance(value, (int, float)), values)):
        raise TypeError("All inputs must be numeric.")

    if not all(map(lambda value: abs(value) != float("inf"), values)):
        raise ValueError("Inputs must be finite.")

    if not 0 < efficiency <= 1:
        raise ValueError("Efficiency must be greater than 0 and at most 1.")

    if gas_price < 0 or carbon_price < 0 or emissions_factor < 0:
        raise ValueError("Gas, carbon and emissions factor cannot be negative.")

    gas_cost = gas_price / efficiency

    carbon_cost = (
        carbon_price * emissions_factor / efficiency
    )

    clean_spark_spread = (
        electricity_price - gas_cost - carbon_cost
    )

    return {
        "electricity_price_eur_mwh": electricity_price,
        "gas_cost_eur_mwh": gas_cost,
        "carbon_cost_eur_mwh": carbon_cost,
        "clean_spark_spread_eur_mwh": clean_spark_spread,
        "spread_status": (
            "Positive spread"
            if clean_spark_spread > 0
            else "Negative or zero spread"
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Clean Spark Spread Calculator"
    )

    parser.add_argument(
        "--electricity",
        type=float,
        required=True,
        help="Electricity price in EUR/MWh",
    )

    parser.add_argument(
        "--gas",
        type=float,
        required=True,
        help="Natural gas price in EUR/MWh thermal",
    )

    parser.add_argument(
        "--carbon",
        type=float,
        required=True,
        help="Carbon allowance price in EUR/tCO2",
    )

    parser.add_argument(
        "--efficiency",
        type=float,
        default=0.55,
        help="Plant efficiency (default: 0.55)",
    )

    parser.add_argument(
        "--emissions-factor",
        type=float,
        default=0.202,
        help="Emissions factor in tCO2/MWh thermal",
    )

    args = parser.parse_args()

    result = calculate_clean_spark_spread(
        electricity_price=args.electricity,
        gas_price=args.gas,
        carbon_price=args.carbon,
        efficiency=args.efficiency,
        emissions_factor=args.emissions_factor,
    )

    print("\nCLEAN SPARK SPREAD ANALYSIS")
    print("-" * 40)

    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

    print("\nNote: This is a simplified operating spread,")
    print("not net profit. It excludes other operating costs.")


if __name__ == "__main__":
    main()
