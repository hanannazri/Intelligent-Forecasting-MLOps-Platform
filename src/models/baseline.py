from pathlib import Path

import mlflow
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


FEATURE_DATA_PATH = Path("data/features/sales_ca_1_foods_features.parquet")


def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    non_zero_mask = y_true != 0

    if non_zero_mask.sum() == 0:
        return 0

    return np.mean(
        np.abs(
            (y_true[non_zero_mask] - y_pred[non_zero_mask])
            / y_true[non_zero_mask]
        )
    ) * 100


def main():
    print("Loading feature dataset...")

    df = pd.read_parquet(FEATURE_DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])

    print("Dataset loaded.")
    print(df.shape)

    print("Creating time-based train/test split...")

    cutoff_date = df["date"].max() - pd.Timedelta(days=28)

    train_df = df[df["date"] <= cutoff_date]
    test_df = df[df["date"] > cutoff_date]

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Cutoff date: {cutoff_date.date()}")

    print("Running baseline forecast...")

    # Baseline prediction:
    # Use rolling_mean_28 as the forecast.
    # Business meaning:
    # Predict future demand using recent average demand.
    y_test = test_df["sales"]
    y_pred = test_df["rolling_mean_28"]

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print("\nBaseline Model Results")
    print("----------------------")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")

    print("\nLogging baseline results to MLflow...")

    mlflow.set_experiment("retail-demand-forecasting")

    with mlflow.start_run(run_name="baseline_rolling_mean_28"):
        mlflow.log_param("model_type", "baseline")
        mlflow.log_param("forecast_strategy", "rolling_mean_28")
        mlflow.log_param("test_days", 28)
        mlflow.log_param("train_rows", train_df.shape[0])
        mlflow.log_param("test_rows", test_df.shape[0])

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mape", mape)

    print("Baseline run logged to MLflow.")


if __name__ == "__main__":
    main()