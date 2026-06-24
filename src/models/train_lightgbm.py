import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.lightgbm
import lightgbm as lgb

from sklearn.metrics import mean_absolute_error, mean_squared_error


DATA_PATH = "data/features/sales_ca_1_foods_features.parquet"
EXPERIMENT_NAME = "retail-demand-forecasting"
BASELINE_MAE = 1.3836


def calculate_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan

    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def calculate_wape(y_true, y_pred):
    return (np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))) * 100


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

    # Remove rows where lag/rolling features are not available
    df = df.dropna().reset_index(drop=True)

    print("Shape after dropping missing feature rows:", df.shape)

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

    X_train = train_df[feature_cols]
    y_train = train_df[target]

    X_test = test_df[feature_cols]
    y_test = test_df[target]

    print("Number of features:", len(feature_cols))
    print("Features used:")
    print(feature_cols)

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=710,
        learning_rate=0.061276667676087915,
        num_leaves=30,
        max_depth=12,
        min_child_samples=62,
        subsample=0.9902294402093923,
        colsample_bytree=0.9425387261963855,
        reg_alpha=0.6143714201894848,
        reg_lambda=0.27607575777962956,
        random_state=42,
        n_jobs=-1,
        verbose=-1)

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="lightgbm_regressor"):
        mlflow.log_param("model_type", "LightGBMRegressor")
        mlflow.log_param("train_rows", len(train_df))
        mlflow.log_param("test_rows", len(test_df))
        mlflow.log_param("baseline_mae", BASELINE_MAE)
        mlflow.log_param("test_period_days", 28)

        for param, value in model.get_params().items():
            mlflow.log_param(param, value)

        print("Training LightGBM model...")
        categorical_cols = [
                "item_id",
                "dept_id",
                "cat_id",
                "store_id",
                "state_id"]

        for col in categorical_cols:
            train_df[col] = train_df[col].astype("category")
            test_df[col] = test_df[col].astype("category")

        X_train = train_df[feature_cols]
        X_test = test_df[feature_cols]

        model.fit(
            X_train,
            y_train,
            categorical_feature=categorical_cols)
        
        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importances_
            })

        importance_df = importance_df.sort_values(
            by="importance",
            ascending=False
            )

        print("\nTop 25 Important Features:")
        print(importance_df.head(25))

        importance_output_path = "reports/lightgbm_feature_importance.csv"

        os.makedirs("reports", exist_ok=True)

        importance_df.to_csv(
            importance_output_path,
            index=False
            )

        mlflow.log_artifact(importance_output_path)

        print("\nFeature importance saved to:")
        print(importance_output_path)

        print("Generating predictions...")
        predictions = model.predict(X_test)

        predictions = np.maximum(predictions, 0)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mape = calculate_mape(y_test, predictions)
        wape = calculate_wape(y_test, predictions)
        smape = calculate_smape(y_test, predictions)
        rmsse = calculate_rmsse(y_test.values, predictions, y_train.values)

        print("\nLightGBM Metrics")
        print("MAE:", round(mae, 4))
        print("RMSE:", round(rmse, 4))
        print("MAPE:", round(mape, 2))
        print("WAPE:", round(wape, 2))
        print("sMAPE:", round(smape, 2))
        print("RMSSE:", round(rmsse, 4))

        improvement = ((BASELINE_MAE - mae) / BASELINE_MAE) * 100

        print("\nBaseline MAE:", BASELINE_MAE)
        print("LightGBM MAE:", round(mae, 4))
        print("Improvement over baseline:", round(improvement, 2), "%")

        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAPE", mape)
        mlflow.log_metric("WAPE", wape)
        mlflow.log_metric("sMAPE", smape)
        mlflow.log_metric("RMSSE", rmsse)
        mlflow.log_metric("improvement_over_baseline_percent", improvement)

        mlflow.lightgbm.log_model(model, "lightgbm_model")

        print("\nModel logged to MLflow successfully.")


if __name__ == "__main__":
    main()