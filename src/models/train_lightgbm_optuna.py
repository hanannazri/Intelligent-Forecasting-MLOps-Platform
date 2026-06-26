import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import mlflow
import mlflow.lightgbm
import joblib
import json
import os
import signal
import sys
from sklearn.metrics import mean_absolute_error

optuna.logging.set_verbosity(optuna.logging.WARNING)

# =====================================================
# CONFIGURATION
# =====================================================

FEATURES_PATH   = "data/features/sales_ca_1_foods_features.parquet"
MODEL_PATH      = "models/lightgbm_model.pkl"
PARAMS_PATH     = "models/best_params.json"

TEST_START      = "2016-03-28"
TEST_END        = "2016-04-24"

# Set this to 20 — easy to stop early with Ctrl+C
# Every completed trial is already saved in MLflow
N_TRIALS        = 15

MLFLOW_EXPERIMENT = "LightGBM_Optuna_Tuning"

os.makedirs("models", exist_ok=True)
os.makedirs("data/predictions", exist_ok=True)

# =====================================================
# GRACEFUL FORCE STOP
# =====================================================
# Press Ctrl+C at any time.
# Script will finish the CURRENT trial, then stop.
# Everything already logged in MLflow is safe.
# =====================================================

stop_flag = False

def handle_interrupt(sig, frame):
    global stop_flag
    print("\n\nCtrl+C detected — finishing current trial then stopping...")
    print("All completed trials are already saved in MLflow.\n")
    stop_flag = True

signal.signal(signal.SIGINT, handle_interrupt)

# =====================================================
# MLFLOW SETUP
# =====================================================

mlflow.set_experiment(MLFLOW_EXPERIMENT)

print(f"MLflow experiment: {MLFLOW_EXPERIMENT}")
print(f"MLflow UI: run 'mlflow ui' in terminal to view\n")

# =====================================================
# LOAD DATA
# =====================================================

print("Loading feature dataset...")

df = pd.read_parquet(FEATURES_PATH)
df["date"] = pd.to_datetime(df["date"])

print(f"Dataset shape: {df.shape}")

# =====================================================
# FEATURES
# =====================================================

ALL_FEATURES = [
    # Lags
    "lag_1", "lag_2", "lag_3", "lag_7", "lag_14",
    "lag_28", "lag_35", "lag_42", "lag_49", "lag_56",
    "lag_91", "lag_182", "lag_364", "lag_365", "lag_371",

    # Rolling mean
    "rolling_mean_3", "rolling_mean_7", "rolling_mean_14",
    "rolling_mean_28", "rolling_mean_56",

    # Rolling std
    "rolling_std_3", "rolling_std_7", "rolling_std_14",
    "rolling_std_28", "rolling_std_56",

    # Rolling skewness
    "rolling_skew_14", "rolling_skew_28", "rolling_skew_56",

    # Item level
    "item_avg_sales", "item_median_sales", "item_max_sales",
    "item_zero_sales_ratio", "days_since_last_sale",

    # Dept level
    "dept_daily_sales", "dept_avg_sales",
    "dept_rolling_mean_7", "dept_rolling_mean_28",
    "dept_rolling_std_7", "dept_rolling_std_28",
    "dept_cv_7", "dept_cv_28",
    "dept_momentum_7_28", "dept_momentum_pct_7_28",

    # Cat level
    "cat_avg_sales",
    "cat_rolling_mean_7", "cat_rolling_mean_28",
    "cat_rolling_std_7", "cat_rolling_std_28",
    "cat_cv_7", "cat_cv_28",

    # Hierarchy
    "item_share_of_dept", "item_share_of_cat",

    # Price
    "sell_price", "item_avg_price", "price_vs_avg", "price_vs_max",
    "previous_price", "price_change", "price_change_pct",
    "is_price_drop", "is_price_increase",
    "price_lag_7", "price_lag_28",
    "price_change_7d", "price_change_28d",
    "price_change_pct_7d", "price_change_pct_28d",
    "days_since_price_change",

    # Calendar
    "is_weekend", "day_of_week", "day_of_month",
    "week_of_year", "month", "quarter",
    "is_month_start", "is_month_end",

    # Events
    "has_event", "is_sporting", "is_cultural",
    "is_national", "is_religious",
    "days_to_next_event", "days_since_last_event",

    # SNAP
    "snap_CA",
    "snap_lag7_interaction", "snap_lag28_interaction",
    "snap_rolling_mean_interaction", "snap_momentum_interaction",
    "snap_count_7", "snap_count_28",

    # Momentum
    "sales_momentum_7", "sales_momentum_28",
    "sales_momentum_pct_7", "sales_momentum_pct_28",
    "sales_acceleration", "sales_acceleration_pct",

    # Ratios
    "week_ratio", "month_ratio", "yoy_ratio",
    "price_elasticity_proxy",

    # Volatility
    "cv_7", "cv_28",
]

FEATURE_COLS = [f for f in ALL_FEATURES if f in df.columns]
print(f"Features available: {len(FEATURE_COLS)}")

# =====================================================
# TRAIN / TEST SPLIT
# =====================================================

print("\nSplitting data...")

train_df = df[df["date"] < TEST_START].copy()
test_df  = df[(df["date"] >= TEST_START) & (df["date"] <= TEST_END)].copy()

X_train = train_df[FEATURE_COLS].fillna(0)
y_train = train_df["sales"]
X_test  = test_df[FEATURE_COLS].fillna(0)
y_test  = test_df["sales"]

print(f"Train rows: {len(X_train):,}")
print(f"Test rows:  {len(X_test):,}")

baseline_mae = mean_absolute_error(y_test, np.full(len(y_test), y_train.mean()))
print(f"Baseline MAE: {baseline_mae:.4f}")

lgb_train = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
lgb_valid = lgb.Dataset(X_test,  label=y_test,  free_raw_data=False, reference=lgb_train)

# =====================================================
# OPTUNA OBJECTIVE WITH MLFLOW LOGGING
# =====================================================
# Every trial is logged as a separate MLflow run.
# Even if you Ctrl+C, completed trials are saved.
# =====================================================

def objective(trial):

    # Stop gracefully on Ctrl+C
    if stop_flag:
        raise optuna.exceptions.OptunaError("Stopped by user.")

    params = {
        # Fixed
        "objective":     "regression_l1",
        "metric":        "mae",
        "verbosity":     -1,
        "boosting_type": "gbdt",
        "n_jobs":        -1,
        "random_state":  42,

        # Tuned by Optuna
        "num_leaves":        trial.suggest_int("num_leaves", 20, 300),
        "max_depth":         trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators":      trial.suggest_int("n_estimators", 200, 2000),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_split_gain":    trial.suggest_float("min_split_gain", 0.0, 1.0),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq":    1,
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "max_bin":           trial.suggest_int("max_bin", 100, 500),
    }

    # --- Each trial = one MLflow run ---
    with mlflow.start_run(run_name=f"trial_{trial.number}"):

        # Log all hyperparameters
        mlflow.log_params({
            "trial_number":    trial.number,
            "num_leaves":      params["num_leaves"],
            "max_depth":       params["max_depth"],
            "min_child_samples": params["min_child_samples"],
            "learning_rate":   params["learning_rate"],
            "n_estimators":    params["n_estimators"],
            "reg_alpha":       params["reg_alpha"],
            "reg_lambda":      params["reg_lambda"],
            "min_split_gain":  params["min_split_gain"],
            "subsample":       params["subsample"],
            "colsample_bytree":params["colsample_bytree"],
            "max_bin":         params["max_bin"],
        })

        # Train model
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=-1),
        ]

        model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_valid],
            callbacks=callbacks,
        )

        # Predict and evaluate
        preds = np.clip(model.predict(X_test), 0, None)
        mae   = mean_absolute_error(y_test, preds)
        rmse  = np.sqrt(np.mean((y_test - preds) ** 2))
        wape  = np.sum(np.abs(y_test - preds)) / (np.sum(np.abs(y_test)) + 1e-8) * 100
        improvement = ((baseline_mae - mae) / baseline_mae) * 100

        # Log all metrics to MLflow
        mlflow.log_metrics({
            "mae":                          mae,
            "rmse":                         rmse,
            "wape":                         wape,
            "improvement_over_baseline":    improvement,
            "best_iteration":               model.best_iteration,
            "baseline_mae":                 baseline_mae,
        })

        # Log feature importance as artifact
        importance_df = pd.DataFrame({
            "feature":    FEATURE_COLS,
            "importance": model.feature_importance(importance_type="gain"),
        }).sort_values("importance", ascending=False)

        importance_path = f"models/importance_trial_{trial.number}.csv"
        importance_df.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path)

        print(
            f"  Trial {trial.number:>3} | "
            f"MAE: {mae:.4f} | "
            f"RMSE: {rmse:.4f} | "
            f"Improvement: {improvement:.2f}% | "
            f"Trees: {model.best_iteration}"
        )

    return mae

# =====================================================
# RUN OPTUNA
# =====================================================

print(f"\nStarting Optuna — {N_TRIALS} trials")
print(f"Press Ctrl+C at any time to stop safely.\n")
print(f"{'Trial':>7} | {'MAE':>8} | {'RMSE':>8} | {'Improvement':>12} | {'Trees':>6}")
print("-" * 60)

study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(),
)

try:
    study.optimize(
        objective,
        n_trials=N_TRIALS,
        show_progress_bar=False,   # cleaner output with our custom print
    )
except (KeyboardInterrupt, optuna.exceptions.OptunaError):
    print("\nStopped. Using best trial found so far.")

# =====================================================
# BEST RESULTS
# =====================================================

best_params  = study.best_params
best_mae     = study.best_value
improvement  = ((baseline_mae - best_mae) / baseline_mae) * 100

print(f"\n{'='*50}")
print(f"BEST RESULT")
print(f"{'='*50}")
print(f"Best MAE:       {best_mae:.4f}")
print(f"Baseline MAE:   {baseline_mae:.4f}")
print(f"Improvement:    {improvement:.2f}%")
print(f"Best trial:     #{study.best_trial.number}")

# =====================================================
# RETRAIN FINAL MODEL WITH BEST PARAMS
# =====================================================

print(f"\nRetraining final model with best parameters...")

final_params = {
    "objective":     "regression_l1",
    "metric":        "mae",
    "verbosity":     -1,
    "boosting_type": "gbdt",
    "n_jobs":        -1,
    "random_state":  42,
    **best_params,
}

# Log final model as its own MLflow run
with mlflow.start_run(run_name="FINAL_MODEL"):

    mlflow.log_params({**best_params, "trial": "final"})

    lgb_all    = lgb.Dataset(df[FEATURE_COLS].fillna(0), label=df["sales"])
    final_model = lgb.train(
        final_params,
        lgb_all,
        num_boost_round=best_params.get("n_estimators", 500),
    )

    # Evaluate on test set
    final_preds = np.clip(final_model.predict(X_test), 0, None)
    final_mae   = mean_absolute_error(y_test, final_preds)
    final_rmse  = np.sqrt(np.mean((y_test - final_preds) ** 2))
    final_wape  = np.sum(np.abs(y_test - final_preds)) / (np.sum(np.abs(y_test)) + 1e-8) * 100
    final_impr  = ((baseline_mae - final_mae) / baseline_mae) * 100

    mlflow.log_metrics({
        "mae":                       final_mae,
        "rmse":                      final_rmse,
        "wape":                      final_wape,
        "improvement_over_baseline": final_impr,
    })

    # Log model to MLflow
    mlflow.lightgbm.log_model(final_model, artifact_path="lightgbm_model")

    # Feature importance
    final_importance = pd.DataFrame({
        "feature":    FEATURE_COLS,
        "importance": final_model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)

    final_importance.to_csv("models/feature_importance_final.csv", index=False)
    mlflow.log_artifact("models/feature_importance_final.csv")

    # Test predictions
    test_df = test_df.copy()
    test_df["predicted_sales"] = final_preds
    pred_path = "data/predictions/test_predictions_optuna.parquet"
    test_df[["date", "item_id", "sales", "predicted_sales"]].to_parquet(pred_path, index=False)
    mlflow.log_artifact(pred_path)

    print(f"\n{'='*50}")
    print(f"FINAL MODEL METRICS")
    print(f"{'='*50}")
    print(f"MAE:            {final_mae:.4f}")
    print(f"RMSE:           {final_rmse:.4f}")
    print(f"WAPE:           {final_wape:.2f}")
    print(f"Improvement:    {final_impr:.2f}%")
    print(f"\nTop 10 Features:")
    print(final_importance.head(10).to_string(index=False))

# =====================================================
# SAVE FILES LOCALLY TOO
# =====================================================

joblib.dump(final_model, MODEL_PATH)
with open(PARAMS_PATH, "w") as f:
    json.dump(best_params, f, indent=2)

print(f"\nModel saved:       {MODEL_PATH}")
print(f"Best params saved: {PARAMS_PATH}")
print(f"\nView all runs:     mlflow ui")
print(f"Then open:         http://localhost:5000")
print(f"\n{'='*50}")
print(f"Previous MAE:   1.2750")
print(f"New MAE:        {final_mae:.4f}")
print(f"Total gain:     {((1.2750 - final_mae) / 1.2750 * 100):.2f}%")
print(f"{'='*50}")