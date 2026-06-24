import os
import pandas as pd
import numpy as np
import optuna
import mlflow
import mlflow.lightgbm
import lightgbm as lgb

from sklearn.metrics import mean_absolute_error, mean_squared_error


DATA_PATH = "data/features/sales_ca_1_foods_features.parquet"
EXPERIMENT_NAME = "retail-demand-forecasting"
BASELINE_MAE = 1.3836
N_TRIALS = 15


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


def load_data():
    print("Loading feature dataset...")

    df = pd.read_parquet(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna().reset_index(drop=True)

    max_date = df["date"].max()
    test_start_date = max_date - pd.Timedelta(days=27)

    train_df = df[df["date"] < test_start_date].copy()
    test_df = df[df["date"] >= test_start_date].copy()

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

    categorical_cols = [
        col for col in categorical_cols
        if col in feature_cols
    ]

    for col in categorical_cols:
        train_df[col] = train_df[col].astype("category")
        test_df[col] = test_df[col].astype("category")

    X_train = train_df[feature_cols]
    y_train = train_df[target]

    X_test = test_df[feature_cols]
    y_test = test_df[target]

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("Number of features:", len(feature_cols))

    return X_train, y_train, X_test, y_test, feature_cols, categorical_cols


def main():
    X_train, y_train, X_test, y_test, feature_cols, categorical_cols = load_data()

    mlflow.set_experiment(EXPERIMENT_NAME)

    def objective(trial):
        params = {
            "objective": "regression",
            "n_estimators": trial.suggest_int("n_estimators", 300, 900),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.08),
            "num_leaves": trial.suggest_int("num_leaves", 20, 80),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 120),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }

        model = lgb.LGBMRegressor(**params)

        model.fit(
            X_train,
            y_train,
            categorical_feature=categorical_cols
        )

        preds = model.predict(X_test)
        preds = np.maximum(preds, 0)

        mae = mean_absolute_error(y_test, preds)

        return mae

    print("\nStarting Optuna tuning...")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS)

    print("\nBest Trial:")
    print("Best MAE:", study.best_value)
    print("Best Params:")
    print(study.best_params)

    best_params = study.best_params

    final_params = {
        **best_params,
        "objective": "regression",
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    final_model = lgb.LGBMRegressor(**final_params)

    with mlflow.start_run(run_name="lightgbm_optuna_tuned"):
        mlflow.log_param("model_type", "LightGBM_Optuna_Tuned")
        mlflow.log_param("baseline_mae", BASELINE_MAE)
        mlflow.log_param("n_trials", N_TRIALS)
        mlflow.log_param("test_period_days", 28)
        mlflow.log_param("num_features", len(feature_cols))

        for param, value in final_params.items():
            mlflow.log_param(param, value)

        print("\nTraining final tuned LightGBM model...")

        final_model.fit(
            X_train,
            y_train,
            categorical_feature=categorical_cols
        )

        predictions = final_model.predict(X_test)
        predictions = np.maximum(predictions, 0)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mape = calculate_mape(y_test, predictions)
        wape = calculate_wape(y_test, predictions)
        smape = calculate_smape(y_test, predictions)
        rmsse = calculate_rmsse(y_test.values, predictions, y_train.values)

        improvement = ((BASELINE_MAE - mae) / BASELINE_MAE) * 100

        print("\nOptuna Tuned LightGBM Metrics")
        print("MAE:", round(mae, 4))
        print("RMSE:", round(rmse, 4))
        print("MAPE:", round(mape, 2))
        print("WAPE:", round(wape, 2))
        print("sMAPE:", round(smape, 2))
        print("RMSSE:", round(rmsse, 4))

        print("\nBaseline MAE:", BASELINE_MAE)
        print("Tuned LightGBM MAE:", round(mae, 4))
        print("Improvement over baseline:", round(improvement, 2), "%")

        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAPE", mape)
        mlflow.log_metric("WAPE", wape)
        mlflow.log_metric("sMAPE", smape)
        mlflow.log_metric("RMSSE", rmsse)
        mlflow.log_metric("improvement_over_baseline_percent", improvement)

        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": final_model.feature_importances_
        }).sort_values(by="importance", ascending=False)

        os.makedirs("reports", exist_ok=True)

        importance_output_path = "reports/lightgbm_optuna_feature_importance.csv"

        importance_df.to_csv(importance_output_path, index=False)

        mlflow.log_artifact(importance_output_path)

        mlflow.lightgbm.log_model(
            final_model,
            "lightgbm_optuna_model"
        )

        print("\nTuned model logged to MLflow successfully.")


if __name__ == "__main__":
    main()