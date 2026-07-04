import pandas as pd
import numpy as np
from pathlib import Path
import joblib


FEATURE_DATA_PATH = Path("data/features/sales_ca_1_foods_features.parquet")
OUTPUT_PATH = Path("reports/future_28_day_forecast.csv")

MODEL_PATH = Path("models/lightgbm_model.pkl")

FORECAST_DAYS = 28

CATEGORICAL_COLS = [
    "item_id", "dept_id", "cat_id", "store_id", "state_id"
]

FEATURES = [
    "item_id", "dept_id", "cat_id", "store_id", "state_id", "wm_yr_wk",
    "wday", "month", "year", "snap_CA", "snap_TX", "snap_WI", "sell_price",
    "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_28", "lag_56",
    "rolling_mean_3", "rolling_mean_7", "rolling_mean_14", "rolling_mean_28",
    "rolling_mean_56", "rolling_std_3", "rolling_std_7", "rolling_std_14",
    "rolling_std_28", "rolling_std_56", "item_avg_sales", "item_median_sales",
    "item_max_sales", "item_zero_sales_ratio", "dept_daily_sales",
    "dept_avg_sales", "dept_rolling_mean_7", "dept_rolling_mean_28",
    "cat_avg_sales", "cat_rolling_mean_7", "cat_rolling_mean_28",
    "days_since_last_sale", "sales_momentum_7", "sales_momentum_28",
    "sales_momentum_pct_7", "sales_momentum_pct_28", "item_avg_price",
    "price_vs_avg", "cv_28", "cv_7", "is_weekend", "day_of_month",
    "week_of_year", "quarter", "previous_price", "price_change",
    "price_change_pct", "price_lag_7", "price_lag_28", "price_change_7d",
    "price_change_28d", "price_change_pct_7d", "price_change_pct_28d",
    "dept_momentum_7_28", "dept_momentum_pct_7_28", "item_share_of_dept",
    "item_share_of_cat", "week_ratio", "month_ratio", "price_elasticity_proxy",
    "snap_lag7_interaction", "snap_rolling_mean_7_interaction",
    "snap_sales_momentum_7_interaction", "dept_rolling_std_7",
    "dept_rolling_std_28", "dept_cv_7", "dept_cv_28", "cat_rolling_std_7",
    "cat_rolling_std_28", "cat_cv_7", "cat_cv_28", "sales_acceleration",
    "sales_acceleration_pct", "days_since_price_change", "has_event"
]


def load_data():
    df = pd.read_parquet(FEATURE_DATA_PATH)
    df = df.sort_values(["item_id", "date"])
    print(f"Feature data loaded: {df.shape}")
    return df


def load_model():
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded from: {MODEL_PATH}")
    return model

def create_future_forecast(df, model):
    last_date = df["date"].max()
    print(f"Last available date: {last_date.date()}")

    latest_rows = df.groupby("item_id").tail(1).copy()
    sales_history = {
        item_id: group.sort_values("date")["sales"].tolist()
        for item_id, group in df.groupby("item_id")
    }

    all_forecasts = []

    for day in range(1, FORECAST_DAYS + 1):
        future_date = last_date + pd.Timedelta(days=day)

        future_rows = latest_rows.copy()
        future_rows["date"] = future_date

        future_rows["wday"] = future_date.dayofweek + 1
        future_rows["month"] = future_date.month
        future_rows["year"] = future_date.year
        future_rows["day_of_month"] = future_date.day
        future_rows["week_of_year"] = future_date.isocalendar().week
        future_rows["quarter"] = future_date.quarter
        future_rows["is_weekend"] = int(future_date.dayofweek >= 5)

        for idx, row in future_rows.iterrows():
            item = row["item_id"]
            history = sales_history[item]

            def lag(n):
                return history[-n] if len(history) >= n else 0

            future_rows.at[idx, "lag_1"] = lag(1)
            future_rows.at[idx, "lag_2"] = lag(2)
            future_rows.at[idx, "lag_3"] = lag(3)
            future_rows.at[idx, "lag_7"] = lag(7)
            future_rows.at[idx, "lag_14"] = lag(14)
            future_rows.at[idx, "lag_28"] = lag(28)
            future_rows.at[idx, "lag_56"] = lag(56)

            for window in [3, 7, 14, 28, 56]:
                values = history[-window:] if len(history) >= window else history
                future_rows.at[idx, f"rolling_mean_{window}"] = np.mean(values)
                future_rows.at[idx, f"rolling_std_{window}"] = np.std(values)

            future_rows.at[idx, "sales_momentum_7"] = lag(1) - lag(7)
            future_rows.at[idx, "sales_momentum_28"] = lag(1) - lag(28)

            future_rows.at[idx, "sales_momentum_pct_7"] = (
                (lag(1) - lag(7)) / (lag(7) + 1e-6)
            )
            future_rows.at[idx, "sales_momentum_pct_28"] = (
                (lag(1) - lag(28)) / (lag(28) + 1e-6)
            )

            future_rows.at[idx, "sales_acceleration"] = lag(1) - lag(2)
            future_rows.at[idx, "sales_acceleration_pct"] = (
                (lag(1) - lag(2)) / (lag(2) + 1e-6)
            )

        X_future = future_rows[FEATURES].copy()

        for col in CATEGORICAL_COLS:
            X_future[col] = X_future[col].astype("category")

        predictions = model.predict(X_future)
        predictions = np.clip(predictions, 0, None)

        future_rows["forecast_sales"] = predictions
        future_rows["recommended_stock"] = np.ceil(predictions).astype(int)

        output = future_rows[
            ["date", "item_id", "dept_id", "cat_id", "store_id", "forecast_sales", "recommended_stock"]
        ].copy()

        all_forecasts.append(output)

        for item_id, pred in zip(future_rows["item_id"], predictions):
            sales_history[item_id].append(pred)

        print(f"Forecast completed for: {future_date.date()}")

    forecast_df = pd.concat(all_forecasts, ignore_index=True)
    return forecast_df


def save_forecast(forecast_df):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nFuture 28-day forecast saved to: {OUTPUT_PATH}")


def main():
    df = load_data()
    model = load_model()

    forecast_df = create_future_forecast(df, model)
    save_forecast(forecast_df)

    print("\nSample future forecast:")
    print(forecast_df.head())


if __name__ == "__main__":
    main()
    