import pandas as pd
from pathlib import Path
import mlflow.pyfunc


FEATURE_DATA_PATH = Path("data/features/sales_ca_1_foods_features.parquet")
MODEL_URI = "runs:/90e6d2c0ae0b452e85769b857fd27929/lightgbm_model"
OUTPUT_PATH = Path("reports/forecast_predictions.csv")


DROP_COLS = [
    "sales",
    "date",
    "id",
    "d",
    "weekday",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
]


CATEGORICAL_COLS = [
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
]


def load_data():
    df = pd.read_parquet(FEATURE_DATA_PATH)
    print(f"Feature data loaded: {df.shape}")
    return df


def load_model():
    model = mlflow.pyfunc.load_model(MODEL_URI)
    print("LightGBM model loaded successfully from MLflow")
    return model


def prepare_prediction_data(df, model):
    df = df.sort_values(["item_id", "date"])

    prediction_df = df.groupby("item_id").tail(28).copy()

    feature_cols = [col for col in prediction_df.columns if col not in DROP_COLS]

    X_pred = prediction_df[feature_cols].copy()

    for col in CATEGORICAL_COLS:
        if col in X_pred.columns:
            X_pred[col] = X_pred[col].astype("category")

    trained_features = [
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

    X_pred = X_pred[trained_features]

    print(f"Prediction feature shape: {X_pred.shape}")
    print(f"Number of prediction features: {len(X_pred.columns)}")

    return prediction_df, X_pred


def generate_predictions(model, prediction_df, X_pred):
    predictions = model.predict(X_pred)

    output_df = prediction_df[["date", "item_id", "sales"]].copy()
    output_df.rename(columns={"sales": "actual_sales"}, inplace=True)

    output_df["predicted_sales"] = predictions
    output_df["predicted_sales"] = output_df["predicted_sales"].clip(lower=0)

    return output_df


def save_predictions(output_df):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Forecast predictions saved to: {OUTPUT_PATH}")


def main():
    df = load_data()
    model = load_model()

    prediction_df, X_pred = prepare_prediction_data(df, model)

    output_df = generate_predictions(model, prediction_df, X_pred)
    save_predictions(output_df)

    print("\nSample predictions:")
    print(output_df.head())


if __name__ == "__main__":
    main()