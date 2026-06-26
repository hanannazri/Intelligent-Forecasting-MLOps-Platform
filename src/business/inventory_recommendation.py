import pandas as pd
import numpy as np
from pathlib import Path


FORECAST_PATH = Path("reports/future_28_day_forecast.csv")
OUTPUT_PATH = Path("reports/inventory_recommendations.csv")

LEAD_TIME_DAYS = 7
SAFETY_STOCK_MULTIPLIER = 1.5


def load_forecast():
    df = pd.read_csv(FORECAST_PATH)
    df["date"] = pd.to_datetime(df["date"])
    print(f"Forecast loaded: {df.shape}")
    return df


def create_inventory_recommendations(df):
    summary = (
        df.groupby(["item_id", "dept_id", "cat_id", "store_id"])
        .agg(
            total_28_day_forecast=("forecast_sales", "sum"),
            avg_daily_forecast=("forecast_sales", "mean"),
            max_daily_forecast=("forecast_sales", "max"),
            forecast_std=("forecast_sales", "std"),
        )
        .reset_index()
    )

    summary["forecast_std"] = summary["forecast_std"].fillna(0)

    summary["lead_time_demand"] = summary["avg_daily_forecast"] * LEAD_TIME_DAYS

    summary["safety_stock"] = summary["forecast_std"] * SAFETY_STOCK_MULTIPLIER

    summary["target_stock_level"] = (
        summary["total_28_day_forecast"] + summary["safety_stock"]
    )

    # Since we do not have real inventory data, simulate current stock.
    # In a real company, this would come from ERP/inventory database.
    np.random.seed(42)
    summary["current_inventory"] = np.random.randint(
        low=0,
        high=30,
        size=len(summary),
        )

    summary["recommended_order_qty"] = np.ceil(
        summary["target_stock_level"] - summary["current_inventory"]
    ).astype(int)

    summary["recommended_order_qty"] = summary["recommended_order_qty"].clip(lower=0)

    summary["reorder_point"] = (
        summary["lead_time_demand"] + summary["safety_stock"]
    )

    summary["needs_reorder"] = (
        summary["current_inventory"] <= summary["reorder_point"]
    ).astype(int)

    summary["stock_status"] = np.where(
        summary["current_inventory"] <= summary["reorder_point"],
        "Reorder Needed",
        "Sufficient Stock",
    )

    summary["risk_level"] = pd.cut(
        summary["target_stock_level"],
        bins=[-1, 7, 28, float("inf")],
        labels=["Low Demand", "Medium Demand", "High Demand"],
    )

    return summary


def save_recommendations(summary):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_PATH, index=False)
    print(f"Inventory recommendations saved to: {OUTPUT_PATH}")


def main():
    df = load_forecast()
    summary = create_inventory_recommendations(df)
    save_recommendations(summary)

    print("\nSample inventory recommendations:")
    print(summary.head(10))


if __name__ == "__main__":
    main()