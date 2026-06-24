import pandas as pd
import numpy as np

# =====================================================
# LOAD DATA
# =====================================================

print("\nLoading transformed dataset...")

df = pd.read_parquet("data/processed/sales_ca_1_foods.parquet")

df["date"] = pd.to_datetime(df["date"])

print("Dataset loaded.")
print(df.shape)

# =====================================================
# SORT DATA
# =====================================================

print("\nSorting data...")

df = df.sort_values(["item_id", "date"]).reset_index(drop=True)

# =====================================================
# LAG FEATURES
# =====================================================

print("\nCreating lag features...")

lag_days = [1, 2, 3, 7, 14, 28, 56]

for lag in lag_days:
    df[f"lag_{lag}"] = df.groupby("item_id")["sales"].shift(lag)

# =====================================================
# ROLLING MEAN FEATURES
# =====================================================

print("\nCreating rolling mean features...")

rolling_windows = [3, 7, 14, 28, 56]

for window in rolling_windows:
    df[f"rolling_mean_{window}"] = (
        df.groupby("item_id")["sales"]
        .transform(lambda x: x.shift(1).rolling(window).mean())
    )

# =====================================================
# ROLLING STD FEATURES
# =====================================================

print("\nCreating rolling std features...")

for window in rolling_windows:
    df[f"rolling_std_{window}"] = (
        df.groupby("item_id")["sales"]
        .transform(lambda x: x.shift(1).rolling(window).std())
    )

# =====================================================
# PRODUCT-LEVEL DEMAND FEATURES
# =====================================================

print("\nCreating product-level demand features...")

df["item_avg_sales"] = (
    df.groupby("item_id")["sales"]
    .transform(lambda x: x.shift(1).expanding().mean())
)

df["item_median_sales"] = (
    df.groupby("item_id")["sales"]
    .transform(lambda x: x.shift(1).expanding().median())
)

df["item_max_sales"] = (
    df.groupby("item_id")["sales"]
    .transform(lambda x: x.shift(1).expanding().max())
)

df["item_zero_sales_ratio"] = (
    df.groupby("item_id")["sales"]
    .transform(lambda x: x.shift(1).eq(0).expanding().mean())
)

# =====================================================
# HIERARCHICAL DEMAND FEATURES
# =====================================================

print("\nCreating hierarchical demand features...")

# Department-level daily demand
df["dept_daily_sales"] = (
    df.groupby(["dept_id", "date"])["sales"]
    .transform("sum")
)

# Department-level historical average demand
df["dept_avg_sales"] = (
    df.groupby("dept_id")["sales"]
    .transform(lambda x: x.shift(1).expanding().mean())
)

# Department-level rolling demand
df["dept_rolling_mean_7"] = (
    df.groupby("dept_id")["sales"]
    .transform(lambda x: x.shift(1).rolling(7).mean())
)

df["dept_rolling_mean_28"] = (
    df.groupby("dept_id")["sales"]
    .transform(lambda x: x.shift(1).rolling(28).mean())
)

# Category-level historical average demand
df["cat_avg_sales"] = (
    df.groupby("cat_id")["sales"]
    .transform(lambda x: x.shift(1).expanding().mean())
)

# Category-level rolling demand
df["cat_rolling_mean_7"] = (
    df.groupby("cat_id")["sales"]
    .transform(lambda x: x.shift(1).rolling(7).mean())
)

df["cat_rolling_mean_28"] = (
    df.groupby("cat_id")["sales"]
    .transform(lambda x: x.shift(1).rolling(28).mean())
)

# =====================================================
# DAYS SINCE LAST SALE FEATURE
# =====================================================

print("\nCreating days since last sale feature...")

def days_since_last_sale(sales_series):
    days = []
    last_sale_index = None

    for i, value in enumerate(sales_series):
        if last_sale_index is None:
            days.append(999)
        else:
            days.append(i - last_sale_index)

        if value > 0:
            last_sale_index = i

    return pd.Series(days, index=sales_series.index)

df["days_since_last_sale"] = (
    df.groupby("item_id")["sales"]
    .transform(days_since_last_sale)
)
# =====================================================
# DEMAND MOMENTUM FEATURES
# =====================================================

print("\nCreating demand momentum features...")

df["sales_momentum_7"] = df["lag_1"] - df["lag_7"]
df["sales_momentum_28"] = df["lag_7"] - df["lag_28"]

df["sales_momentum_pct_7"] = (
    df["sales_momentum_7"] / (df["lag_7"] + 1e-8)
)

df["sales_momentum_pct_28"] = (
    df["sales_momentum_28"] / (df["lag_28"] + 1e-8)
)

# =====================================================
# PRICE RELATIVE FEATURES
# =====================================================

print("\nCreating relative price features...")

df["item_avg_price"] = (
    df.groupby("item_id")["sell_price"]
    .transform(lambda x: x.shift(1).expanding().mean())
)

df["price_vs_avg"] = (
    df["sell_price"] / (df["item_avg_price"] + 1e-8)
)

# =====================================================
# DEMAND VOLATILITY FEATURES
# =====================================================

print("\nCreating demand volatility features...")

df["cv_28"] = (
    df["rolling_std_28"] / (df["rolling_mean_28"] + 1e-8)
)

df["cv_7"] = (
    df["rolling_std_7"] / (df["rolling_mean_7"] + 1e-8)
)
# =====================================================
# CALENDAR FEATURES
# =====================================================

print("\nCreating calendar features...")

df["is_weekend"] = (
    df["weekday"]
    .isin(["Saturday", "Sunday"])
    .astype(int)
)

df["day_of_month"] = df["date"].dt.day
df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
df["quarter"] = df["date"].dt.quarter

# =====================================================
# PRICE FEATURES
# =====================================================

print("\nCreating price features...")

df["previous_price"] = df.groupby("item_id")["sell_price"].shift(1)

df["price_change"] = df["sell_price"] - df["previous_price"]

df["price_change_pct"] = df["price_change"] / df["previous_price"]

df["price_lag_7"] = df.groupby("item_id")["sell_price"].shift(7)
df["price_lag_28"] = df.groupby("item_id")["sell_price"].shift(28)

df["price_change_7d"] = df["sell_price"] - df["price_lag_7"]
df["price_change_28d"] = df["sell_price"] - df["price_lag_28"]

df["price_change_pct_7d"] = df["price_change_7d"] / df["price_lag_7"]
df["price_change_pct_28d"] = df["price_change_28d"] / df["price_lag_28"]

# =====================================================
# ADVANCED RETAIL FORECASTING FEATURES
# =====================================================

print("\nCreating advanced retail forecasting features...")

# Department demand momentum
df["dept_momentum_7_28"] = (
    df["dept_rolling_mean_7"]
    -
    df["dept_rolling_mean_28"]
)

df["dept_momentum_pct_7_28"] = (
    df["dept_momentum_7_28"]
    /
    (df["dept_rolling_mean_28"] + 1e-8)
)

# Item share within department/category
df["item_share_of_dept"] = (
    df["item_avg_sales"]
    /
    (df["dept_avg_sales"] + 1e-8)
)

df["item_share_of_cat"] = (
    df["item_avg_sales"]
    /
    (df["cat_avg_sales"] + 1e-8)
)

# Weekly/monthly demand ratios
df["week_ratio"] = (
    df["lag_1"]
    /
    (df["lag_7"] + 1e-8)
)

df["month_ratio"] = (
    df["lag_7"]
    /
    (df["lag_28"] + 1e-8)
)

# Price elasticity proxy
df["price_elasticity_proxy"] = (
    df["sales_momentum_pct_7"]
    /
    (df["price_change_pct_7d"].abs() + 1e-8)
)

# SNAP interaction features
df["snap_lag7_interaction"] = (
    df["snap_CA"]
    *
    df["lag_7"]
)

df["snap_rolling_mean_7_interaction"] = (
    df["snap_CA"]
    *
    df["rolling_mean_7"]
)

df["snap_sales_momentum_7_interaction"] = (
    df["snap_CA"]
    *
    df["sales_momentum_7"]
)

# =====================================================
# FINAL ADVANCED HIERARCHY + RETAIL FEATURES
# =====================================================

print("\nCreating final advanced hierarchy and retail features...")

# Department-level volatility
df["dept_rolling_std_7"] = (
    df.groupby("dept_id")["sales"]
    .transform(lambda x: x.shift(1).rolling(7).std())
)

df["dept_rolling_std_28"] = (
    df.groupby("dept_id")["sales"]
    .transform(lambda x: x.shift(1).rolling(28).std())
)

df["dept_cv_7"] = (
    df["dept_rolling_std_7"] / (df["dept_rolling_mean_7"] + 1e-8)
)

df["dept_cv_28"] = (
    df["dept_rolling_std_28"] / (df["dept_rolling_mean_28"] + 1e-8)
)

# Category-level volatility
df["cat_rolling_std_7"] = (
    df.groupby("cat_id")["sales"]
    .transform(lambda x: x.shift(1).rolling(7).std())
)

df["cat_rolling_std_28"] = (
    df.groupby("cat_id")["sales"]
    .transform(lambda x: x.shift(1).rolling(28).std())
)

df["cat_cv_7"] = (
    df["cat_rolling_std_7"] / (df["cat_rolling_mean_7"] + 1e-8)
)

df["cat_cv_28"] = (
    df["cat_rolling_std_28"] / (df["cat_rolling_mean_28"] + 1e-8)
)

# Demand acceleration
df["sales_acceleration"] = (
    df["sales_momentum_7"] - df["sales_momentum_28"]
)

df["sales_acceleration_pct"] = (
    df["sales_acceleration"] / (df["sales_momentum_28"].abs() + 1e-8)
)

# Days since price change
print("\nCreating days since price change feature...")

def days_since_price_change(price_series):
    days = []
    last_change_index = None
    previous_price = None

    for i, value in enumerate(price_series):
        if previous_price is None:
            days.append(999)
        else:
            if value != previous_price:
                last_change_index = i

            if last_change_index is None:
                days.append(999)
            else:
                days.append(i - last_change_index)

        previous_price = value

    return pd.Series(days, index=price_series.index)

df["days_since_price_change"] = (
    df.groupby("item_id")["sell_price"]
    .transform(days_since_price_change)
)

# =====================================================
# EVENT FEATURE
# =====================================================

print("\nCreating event feature...")

df["has_event"] = (
    df["event_name_1"]
    .notnull()
    .astype(int)
)

# =====================================================
# MISSING VALUES
# =====================================================

print("\nHandling missing values...")

event_columns = [
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
]

for col in event_columns:
    df[col] = df[col].fillna("No_Event").astype(str)

numeric_columns = df.select_dtypes(
    include=["int64", "float64", "int32", "float32", "UInt32"]
).columns

df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], 0)
df[numeric_columns] = df[numeric_columns].fillna(0)

# =====================================================
# SAVE FEATURES
# =====================================================

output_path = "data/features/sales_ca_1_foods_features.parquet"

df.to_parquet(output_path, index=False)

print("\nFeature engineering completed.")

print("\nFinal Shape:")
print(df.shape)

print("\nFeature Dataset Saved:")
print(output_path)

print("\nNew Features Created:")

new_features = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_7",
    "lag_14",
    "lag_28",
    "lag_56",
    "rolling_mean_3",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_mean_56",
    "rolling_std_3",
    "rolling_std_7",
    "rolling_std_14",
    "rolling_std_28",
    "rolling_std_56",
    "item_avg_sales",
    "item_median_sales",
    "item_max_sales",
    "item_zero_sales_ratio",
    "days_since_last_sale",
    "is_weekend",
    "day_of_month",
    "week_of_year",
    "quarter",
    "previous_price",
    "price_change",
    "price_change_pct",
    "price_lag_7",
    "price_lag_28",
    "price_change_7d",
    "price_change_28d",
    "price_change_pct_7d",
    "price_change_pct_28d",
    "dept_daily_sales",
    "dept_avg_sales",
    "dept_rolling_mean_7",
    "dept_rolling_mean_28",
    "cat_avg_sales",
    "cat_rolling_mean_7",
    "cat_rolling_mean_28",
    "dept_momentum_7_28",
    "dept_momentum_pct_7_28",
    "item_share_of_dept",
    "item_share_of_cat",
    "week_ratio",
    "month_ratio",
    "price_elasticity_proxy",
    "snap_lag7_interaction",
    "snap_rolling_mean_7_interaction",
    "snap_sales_momentum_7_interaction",
    "dept_rolling_std_7",
    "dept_rolling_std_28",
    "dept_cv_7",
    "dept_cv_28",
    "cat_rolling_std_7",
    "cat_rolling_std_28",
    "cat_cv_7",
    "cat_cv_28",
    "sales_acceleration",
    "sales_acceleration_pct",
    "days_since_price_change",
    "has_event",
]

for feature in new_features:
    print(feature)