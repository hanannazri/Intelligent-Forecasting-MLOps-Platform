import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.lightgbm
import lightgbm as lgb
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error


DATA_PATH = "data/features/sales_ca_1_foods_features.parquet"
EXPERIMENT_NAME = "retail-demand-forecasting"


def calculate_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan

    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calculate_wape(y_true, y_pred):
    denominator = np.sum(np.abs(y_true))
    if denominator == 0:
        return np.nan
    return (np.sum(np.abs(y_true - y_pred)) / denominator) * 100


def calculate_smape(y_true, y_pred):
    denominator = np.abs(y_true) + np.abs(y_pred)
    return np.mean(
        2 * np.abs(y_pred - y_true) / (denominator + 1e-8)
    ) * 100


def calculate_rmsse(y_true, y_pred, y_train):
    numerator = np.mean((y_true - y_pred) ** 2)
    denominator = np.mean(np.diff(y_train) ** 2)

    if denominator == 0:
        return np.nan

    return np.sqrt(numerator / denominator)


def main():
    print("Loading feature dataset...")
    df = pd.read_parquet(DATA_PATH)

    print("Dataset shape:", df.shape)

    df["date"] = pd.to_datetime(df["date"])
    df = df.reset_index(drop=True)

    print("Shape after reset index:", df.shape)

    # Time-based split: last 28 days as test
    max_date = df["date"].max()
    test_start_date = max_date - pd.Timedelta(days=27)

    train_df = df[df["date"] < test_start_date].copy()
    test_df = df[df["date"] >= test_start_date].copy()

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    print("Test start date:", test_start_date.date())
    print("Test end date:", max_date.date())

    target = "sales"

    drop_cols = [
        "sales",
        "date",
        "id",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
        "d",
        "weekday",
    ]

    feature_cols = [col for col in df.columns if col not in drop_cols]

    categorical_cols = [
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
    ]

    for col in categorical_cols:
        train_df[col] = train_df[col].astype("category")
        test_df[col] = test_df[col].astype("category")

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target]

    X_test = test_df[feature_cols].copy()
    y_test = test_df[target]

    # Safer than df.dropna(); avoids memory crash and handles missing feature values
    X_train = X_train.replace([np.inf, -np.inf], 0).fillna(0)
    X_test = X_test.replace([np.inf, -np.inf], 0).fillna(0)

    print("Number of features:", len(feature_cols))
    print("Features used:")
    print(feature_cols)

    # Persistence baseline: predicts current sales using lag_1
    # This is the recruiter-facing forecasting baseline.
    if "lag_1" not in X_test.columns:
        raise ValueError("lag_1 column is missing. Cannot calculate persistence baseline.")

    persistence_baseline_predictions = X_test["lag_1"].fillna(0).clip(lower=0)
    persistence_baseline_mae = mean_absolute_error(
        y_test,
        persistence_baseline_predictions
    )

    # Mean-sales baseline: weaker sanity-check baseline
    mean_baseline_predictions = np.full(len(y_test), y_train.mean())
    mean_baseline_mae = mean_absolute_error(y_test, mean_baseline_predictions)

    print("\nBaseline Metrics")
    print("Mean-sales Baseline MAE:", round(mean_baseline_mae, 4))
    print("Persistence Baseline MAE:", round(persistence_baseline_mae, 4))

    model = lgb.LGBMRegressor(
        objective="regression_l1",
        metric="mae",
        boosting_type="gbdt",
        n_estimators=1932,
        learning_rate=0.03330732614147012,
        num_leaves=200,
        max_depth=9,
        min_child_samples=98,
        subsample=0.537264052769952,
        colsample_bytree=0.5061555847824688,
        reg_alpha=1.475649304728376,
        reg_lambda=6.379136939832477e-05,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="lightgbm_regressor_final"):
        mlflow.log_param("model_type", "LightGBMRegressor")
        mlflow.log_param("train_rows", len(train_df))
        mlflow.log_param("test_rows", len(test_df))
        mlflow.log_param("mean_baseline_mae", mean_baseline_mae)
        mlflow.log_param("persistence_baseline_mae", persistence_baseline_mae)
        mlflow.log_param("baseline_type", "persistence_lag_1")
        mlflow.log_param("test_period_days", 28)

        for param, value in model.get_params().items():
            mlflow.log_param(param, value)

        print("\nTraining LightGBM model...")

        model.fit(
            X_train,
            y_train,
            categorical_feature=categorical_cols,
        )

        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importances_,
        })

        importance_df = importance_df.sort_values(
            by="importance",
            ascending=False,
        )

        print("\nTop 25 Important Features:")
        print(importance_df.head(25))

        importance_output_path = "reports/lightgbm_feature_importance.csv"
        os.makedirs("reports", exist_ok=True)

        importance_df.to_csv(
            importance_output_path,
            index=False,
        )

        MODEL_DIR = Path("models")
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, MODEL_DIR / "lightgbm_model.pkl")

        print("\nLightGBM model saved to models/lightgbm_model.pkl")

        mlflow.log_artifact(importance_output_path)

        print("\nFeature importance saved to:")
        print(importance_output_path)

        print("\nGenerating predictions...")
        predictions = model.predict(X_test)
        predictions = np.maximum(predictions, 0)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mape = calculate_mape(y_test, predictions)
        wape = calculate_wape(y_test, predictions)
        smape = calculate_smape(y_test, predictions)
        rmsse = calculate_rmsse(y_test.values, predictions, y_train.values)

        improvement_over_persistence = (
            (persistence_baseline_mae - mae) / persistence_baseline_mae
        ) * 100

        improvement_over_mean = (
            (mean_baseline_mae - mae) / mean_baseline_mae
        ) * 100

        print("\nLightGBM Metrics")
        print("MAE:", round(mae, 4))
        print("RMSE:", round(rmse, 4))
        print("MAPE:", round(mape, 2))
        print("WAPE:", round(wape, 2))
        print("sMAPE:", round(smape, 2))
        print("RMSSE:", round(rmsse, 4))

        print("\nBaseline Comparison")
        print("Mean-sales Baseline MAE:", round(mean_baseline_mae, 4))
        print("Persistence Baseline MAE:", round(persistence_baseline_mae, 4))
        print("LightGBM MAE:", round(mae, 4))
        print(
            "Improvement over persistence baseline:",
            round(improvement_over_persistence, 2),
            "%",
        )
        print(
            "Improvement over mean-sales baseline:",
            round(improvement_over_mean, 2),
            "%",
        )

        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAPE", mape)
        mlflow.log_metric("WAPE", wape)
        mlflow.log_metric("sMAPE", smape)
        mlflow.log_metric("RMSSE", rmsse)
        mlflow.log_metric("mean_baseline_mae", mean_baseline_mae)
        mlflow.log_metric("persistence_baseline_mae", persistence_baseline_mae)
        mlflow.log_metric(
            "improvement_over_persistence_baseline_percent",
            improvement_over_persistence,
        )
        mlflow.log_metric(
            "improvement_over_mean_sales_baseline_percent",
            improvement_over_mean,
        )

        mlflow.lightgbm.log_model(
            model,
            name="lightgbm_model",
        )

        print("\nModel logged to MLflow successfully.")


if __name__ == "__main__":
    main()