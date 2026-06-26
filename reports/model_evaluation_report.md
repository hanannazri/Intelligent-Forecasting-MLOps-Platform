# Model Evaluation Report

## Project Name

Intelligent Forecasting & MLOps Platform

## Objective

The objective of this stage is to evaluate multiple machine learning models for retail demand forecasting and select the best-performing model for future deployment.

The project uses the M5 Forecasting dataset and focuses on demand forecasting for the CA_1 store and FOODS category.

## Dataset Summary

* Dataset: M5 Forecasting Dataset
* Scope: CA_1 store, FOODS category
* Processed records: 2.7M+ rows
* Unique products: 1,400+
* Target variable: sales
* Problem type: Time-series demand forecasting

## Models Evaluated

The following models were trained and evaluated:

1. Persistence Baseline
2. LightGBM
3. XGBoost

## Feature Engineering

The models were trained using time-series features such as:

* Lag features: 7-day, 14-day, and 28-day lags
* Rolling mean features: 7-day, 14-day, and 28-day rolling averages
* Rolling standard deviation features
* Calendar-based features such as weekday, month, year, and SNAP indicators
* Price-based features using sell price

## Evaluation Metrics

The models were evaluated using the following forecasting metrics:

* MAE: Mean Absolute Error
* RMSE: Root Mean Squared Error
* MAPE: Mean Absolute Percentage Error
* WAPE: Weighted Absolute Percentage Error
* sMAPE: Symmetric Mean Absolute Percentage Error
* RMSSE: Root Mean Squared Scaled Error

## Model Results

| Model    |    MAE |   RMSE |  MAPE |  WAPE |  sMAPE |  RMSSE | Improvement over Baseline |
| -------- | -----: | -----: | ----: | ----: | -----: | -----: | ------------------------: |
| Baseline | 1.3836 |      - |     - |     - |      - |      - |                         - |
| LightGBM | 1.2735 | 2.2769 | 52.55 | 60.52 | 111.38 | 0.6481 |                     7.95% |
| XGBoost  | 1.2797 | 2.3020 | 52.42 | 60.81 | 113.99 | 0.6553 |                     7.51% |

## Best Model Selection

LightGBM was selected as the final model because it achieved the best overall performance.

LightGBM had:

* Lowest MAE: 1.2735
* Lowest RMSE: 2.2769
* Lowest WAPE: 60.52
* Lowest RMSSE: 0.6481
* Highest improvement over baseline: 7.95%

Although XGBoost performed closely, LightGBM produced slightly better results across the most important forecasting metrics.

## Conclusion

The model evaluation stage is complete. LightGBM is selected as the final forecasting model for the next stage of the project.

The next stage will focus on generating forecast outputs using the trained LightGBM model and preparing the model for deployment through an API.
