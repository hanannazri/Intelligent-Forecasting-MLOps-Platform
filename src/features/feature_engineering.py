import pandas as pd
import numpy as np

# =====================================================
# LOAD DATA
# =====================================================

print("\nLoading dataset...")

df = pd.read_parquet("data/processed/sales_ca_1_foods.parquet")
df["date"] = pd.to_datetime(df["date"])

print(f"Dataset loaded: {df.shape}")

# =====================================================
# SORT — always sort before creating lag features
# =====================================================

df = df.sort_values(["item_id", "date"]).reset_index(drop=True)

# =====================================================
# LAG FEATURES
# =====================================================
# lag_1 to lag_28 are safe during backtesting because
# actual sales data exists for every day in test set.
#
# During real future forecasting, predicted values are
# fed back as lags (handled in forecasting script).
# =====================================================

print("\nCreating lag features...")

lag_days = [
    1, 2, 3, 7, 14,       # short-term: captures recent daily patterns
    28,                    # 4 weeks ago: captures monthly cycle
    35, 42, 49, 56,        # 5–8 weeks ago: medium-term trends
    91,                    # 3 months ago
    182,                   # 6 months ago
    364, 365, 371,         # same period last year: yearly seasonality
]

for lag in lag_days:
    df[f"lag_{lag}"] = df.groupby("item_id")["sales"].shift(lag)

# =====================================================
# ROLLING MEAN FEATURES
# =====================================================
# shift(1) is safe during backtesting.
# Captures recent average demand trend.
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
# Captures how volatile / unpredictable an item is.
# =====================================================

print("\nCreating rolling std features...")

for window in rolling_windows:
    df[f"rolling_std_{window}"] = (
        df.groupby("item_id")["sales"]
        .transform(lambda x: x.shift(1).rolling(window).std())
    )

# =====================================================
# ROLLING SKEWNESS
# =====================================================
# Skewness tells the model: "this item tends to spike"
# High skew = item occasionally sells a LOT in one day
# → model will predict higher instead of flat average.
# =====================================================

print("\nCreating rolling skewness features...")

for window in [14, 28, 56]:
    df[f"rolling_skew_{window}"] = (
        df.groupby("item_id")["sales"]
        .transform(lambda x: x.shift(1).rolling(window).skew())
    )

# =====================================================
# ITEM-LEVEL DEMAND PROFILE
# =====================================================
# Gives model a "baseline expectation" per item.
# Example: FOODS_1_001 avg = 1.5 sales/day
#          FOODS_1_002 avg = 8.0 sales/day
# Model learns each item's natural demand level.
# =====================================================

print("\nCreating item-level demand features...")

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
# DAYS SINCE LAST SALE
# =====================================================
# If item hasn't sold in 5 days → likely slow-moving.
# Helps model distinguish active vs inactive items.
# =====================================================

print("\nCreating days since last sale...")

def days_since_last_sale(series):
    result = []
    last_idx = None
    for i, val in enumerate(series):
        result.append(999 if last_idx is None else i - last_idx)
        if val > 0:
            last_idx = i
    return pd.Series(result, index=series.index)

df["days_since_last_sale"] = (
    df.groupby("item_id")["sales"]
    .transform(days_since_last_sale)
)

# =====================================================
# HIERARCHICAL FEATURES — DEPT AND CATEGORY
# =====================================================
# If the whole FOODS_1 department is trending up,
# individual items likely are too.
# Gives model a "bigger picture" beyond one item.
# =====================================================

print("\nCreating hierarchical demand features...")

df["dept_daily_sales"] = (
    df.groupby(["dept_id", "date"])["sales"].transform("sum")
)

df["dept_avg_sales"] = (
    df.groupby("dept_id")["sales"]
    .transform(lambda x: x.shift(1).expanding().mean())
)

df["dept_rolling_mean_7"] = (
    df.groupby("dept_id")["sales"]
    .transform(lambda x: x.shift(1).rolling(7).mean())
)

df["dept_rolling_mean_28"] = (
    df.groupby("dept_id")["sales"]
    .transform(lambda x: x.shift(1).rolling(28).mean())
)

df["dept_rolling_std_7"] = (
    df.groupby("dept_id")["sales"]
    .transform(lambda x: x.shift(1).rolling(7).std())
)

df["dept_rolling_std_28"] = (
    df.groupby("dept_id")["sales"]
    .transform(lambda x: x.shift(1).rolling(28).std())
)

df["dept_cv_7"]  = df["dept_rolling_std_7"]  / (df["dept_rolling_mean_7"]  + 1e-8)
df["dept_cv_28"] = df["dept_rolling_std_28"] / (df["dept_rolling_mean_28"] + 1e-8)

df["cat_avg_sales"] = (
    df.groupby("cat_id")["sales"]
    .transform(lambda x: x.shift(1).expanding().mean())
)

df["cat_rolling_mean_7"] = (
    df.groupby("cat_id")["sales"]
    .transform(lambda x: x.shift(1).rolling(7).mean())
)

df["cat_rolling_mean_28"] = (
    df.groupby("cat_id")["sales"]
    .transform(lambda x: x.shift(1).rolling(28).mean())
)

df["cat_rolling_std_7"] = (
    df.groupby("cat_id")["sales"]
    .transform(lambda x: x.shift(1).rolling(7).std())
)

df["cat_rolling_std_28"] = (
    df.groupby("cat_id")["sales"]
    .transform(lambda x: x.shift(1).rolling(28).std())
)

df["cat_cv_7"]  = df["cat_rolling_std_7"]  / (df["cat_rolling_mean_7"]  + 1e-8)
df["cat_cv_28"] = df["cat_rolling_std_28"] / (df["cat_rolling_mean_28"] + 1e-8)

# Item share within department and category
df["item_share_of_dept"] = df["item_avg_sales"] / (df["dept_avg_sales"] + 1e-8)
df["item_share_of_cat"]  = df["item_avg_sales"] / (df["cat_avg_sales"]  + 1e-8)

df["dept_momentum_7_28"]     = df["dept_rolling_mean_7"] - df["dept_rolling_mean_28"]
df["dept_momentum_pct_7_28"] = df["dept_momentum_7_28"]  / (df["dept_rolling_mean_28"] + 1e-8)

# =====================================================
# PRICE FEATURES
# =====================================================
# Price drops → demand spikes.
# Price vs average → is item cheap or expensive now?
# =====================================================

print("\nCreating price features...")

df["item_avg_price"] = (
    df.groupby("item_id")["sell_price"]
    .transform(lambda x: x.shift(1).expanding().mean())
)

df["price_vs_avg"] = df["sell_price"] / (df["item_avg_price"] + 1e-8)

df["price_vs_max"] = (
    df["sell_price"] /
    (df.groupby("item_id")["sell_price"].transform("max") + 1e-8)
)

df["previous_price"]    = df.groupby("item_id")["sell_price"].shift(1)
df["price_change"]      = df["sell_price"] - df["previous_price"]
df["price_change_pct"]  = df["price_change"] / (df["previous_price"] + 1e-8)
df["is_price_drop"]     = (df["price_change"] < 0).astype(int)
df["is_price_increase"] = (df["price_change"] > 0).astype(int)

df["price_lag_7"]          = df.groupby("item_id")["sell_price"].shift(7)
df["price_lag_28"]         = df.groupby("item_id")["sell_price"].shift(28)
df["price_change_7d"]      = df["sell_price"] - df["price_lag_7"]
df["price_change_28d"]     = df["sell_price"] - df["price_lag_28"]
df["price_change_pct_7d"]  = df["price_change_7d"]  / (df["price_lag_7"]  + 1e-8)
df["price_change_pct_28d"] = df["price_change_28d"] / (df["price_lag_28"] + 1e-8)

# =====================================================
# DAYS SINCE PRICE CHANGE
# =====================================================

print("\nCreating days since price change...")

def days_since_price_change(series):
    result = []
    last_change_idx = None
    prev_price = None
    for i, val in enumerate(series):
        if prev_price is None:
            result.append(999)
        else:
            if val != prev_price:
                last_change_idx = i
            result.append(999 if last_change_idx is None else i - last_change_idx)
        prev_price = val
    return pd.Series(result, index=series.index)

df["days_since_price_change"] = (
    df.groupby("item_id")["sell_price"]
    .transform(days_since_price_change)
)

# =====================================================
# DEMAND VOLATILITY
# =====================================================

print("\nCreating demand volatility features...")

df["cv_7"]  = df["rolling_std_7"]  / (df["rolling_mean_7"]  + 1e-8)
df["cv_28"] = df["rolling_std_28"] / (df["rolling_mean_28"] + 1e-8)

# =====================================================
# DEMAND MOMENTUM
# =====================================================
# Is demand going up or down recently?
# momentum_7  = yesterday vs 7 days ago
# momentum_28 = yesterday vs 28 days ago
# =====================================================

print("\nCreating demand momentum features...")

df["sales_momentum_7"]      = df["lag_1"] - df["lag_7"]
df["sales_momentum_28"]     = df["lag_7"] - df["lag_28"]
df["sales_momentum_pct_7"]  = df["sales_momentum_7"]  / (df["lag_7"]  + 1e-8)
df["sales_momentum_pct_28"] = df["sales_momentum_28"] / (df["lag_28"] + 1e-8)

df["sales_acceleration"]     = df["sales_momentum_7"] - df["sales_momentum_28"]
df["sales_acceleration_pct"] = (
    df["sales_acceleration"] / (df["sales_momentum_28"].abs() + 1e-8)
)

# =====================================================
# DEMAND RATIOS
# =====================================================

df["week_ratio"]  = df["lag_1"]  / (df["lag_7"]   + 1e-8)  # today vs same day last week
df["month_ratio"] = df["lag_7"]  / (df["lag_28"]  + 1e-8)  # this week vs this month
df["yoy_ratio"]   = df["lag_28"] / (df["lag_365"] + 1e-8)  # this month vs last year

# =====================================================
# PRICE ELASTICITY PROXY
# =====================================================
# How much does demand change when price changes?
# =====================================================

df["price_elasticity_proxy"] = (
    df["sales_momentum_pct_7"] / (df["price_change_pct_7d"].abs() + 1e-8)
)

# =====================================================
# CALENDAR FEATURES
# =====================================================

print("\nCreating calendar features...")

df["is_weekend"]     = df["weekday"].isin(["Saturday", "Sunday"]).astype(int)
df["day_of_week"]    = df["date"].dt.dayofweek       # 0=Mon, 6=Sun
df["day_of_month"]   = df["date"].dt.day
df["week_of_year"]   = df["date"].dt.isocalendar().week.astype(int)
df["month"]          = df["date"].dt.month
df["quarter"]        = df["date"].dt.quarter

# Payday effects — people buy more at start/end of month
df["is_month_start"] = (df["day_of_month"] <= 5).astype(int)
df["is_month_end"]   = (df["day_of_month"] >= 25).astype(int)

# =====================================================
# EVENT FEATURES
# =====================================================
# has_event = 1 only on event day — too weak.
# Demand rises BEFORE events (people stock up).
# days_to_next_event captures that pre-event spike.
# =====================================================

print("\nCreating event features...")

df["has_event"]    = df["event_name_1"].notnull().astype(int)
df["is_sporting"]  = (df["event_type_1"] == "Sporting").astype(int)
df["is_cultural"]  = (df["event_type_1"] == "Cultural").astype(int)
df["is_national"]  = (df["event_type_1"] == "National").astype(int)
df["is_religious"] = (df["event_type_1"] == "Religious").astype(int)

event_dates = sorted(df.loc[df["has_event"] == 1, "date"].unique())

def days_to_next(d):
    future = [e for e in event_dates if e >= d]
    return (future[0] - d).days if future else 999

def days_since_last(d):
    past = [e for e in event_dates if e <= d]
    return (d - past[-1]).days if past else 999

unique_dates   = df["date"].unique()
next_event_map = {d: days_to_next(d)    for d in unique_dates}
last_event_map = {d: days_since_last(d) for d in unique_dates}

df["days_to_next_event"]    = df["date"].map(next_event_map)
df["days_since_last_event"] = df["date"].map(last_event_map)

# =====================================================
# SNAP FEATURES
# =====================================================
# SNAP = government food stamps.
# On SNAP days, FOODS item sales spike significantly.
# =====================================================

print("\nCreating SNAP features...")

df["snap_lag7_interaction"]         = df["snap_CA"] * df["lag_7"]
df["snap_lag28_interaction"]        = df["snap_CA"] * df["lag_28"]
df["snap_rolling_mean_interaction"] = df["snap_CA"] * df["rolling_mean_7"]
df["snap_momentum_interaction"]     = df["snap_CA"] * df["sales_momentum_7"]

df["snap_count_7"] = (
    df.groupby("item_id")["snap_CA"]
    .transform(lambda x: x.shift(1).rolling(7).sum())
)

df["snap_count_28"] = (
    df.groupby("item_id")["snap_CA"]
    .transform(lambda x: x.shift(1).rolling(28).sum())
)

# =====================================================
# HANDLE MISSING VALUES
# =====================================================

print("\nHandling missing values...")

event_cols = ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]
for col in event_cols:
    if col in df.columns:
        df[col] = df[col].fillna("No_Event").astype(str)

num_cols = df.select_dtypes(
    include=["int64", "float64", "int32", "float32", "UInt32"]
).columns
df[num_cols] = df[num_cols].replace([np.inf, -np.inf], 0)
df[num_cols] = df[num_cols].fillna(0)

# =====================================================
# SAVE
# =====================================================

output_path = "data/features/sales_ca_1_foods_features.parquet"
df.to_parquet(output_path, index=False)

print(f"\nSaved to: {output_path}")
print(f"Final shape: {df.shape}")

# =====================================================
# FEATURE SUMMARY
# =====================================================

feature_groups = {
    "Lag Features":            [c for c in df.columns if c.startswith("lag_")],
    "Rolling Mean":            [c for c in df.columns if c.startswith("rolling_mean_")],
    "Rolling Std":             [c for c in df.columns if c.startswith("rolling_std_")],
    "Rolling Skewness":        [c for c in df.columns if c.startswith("rolling_skew_")],
    "Item-Level Stats":        ["item_avg_sales","item_median_sales","item_max_sales",
                                "item_zero_sales_ratio","days_since_last_sale"],
    "Dept-Level":              [c for c in df.columns if c.startswith("dept_")],
    "Cat-Level":               [c for c in df.columns if c.startswith("cat_")],
    "Hierarchy Ratios":        ["item_share_of_dept","item_share_of_cat"],
    "Price Features":          [c for c in df.columns if "price" in c or "is_price" in c],
    "Calendar Features":       ["is_weekend","day_of_week","day_of_month","week_of_year",
                                "month","quarter","is_month_start","is_month_end"],
    "Event Features":          ["has_event","is_sporting","is_cultural","is_national",
                                "is_religious","days_to_next_event","days_since_last_event"],
    "SNAP Features":           [c for c in df.columns if "snap" in c],
    "Momentum":                ["sales_momentum_7","sales_momentum_28",
                                "sales_momentum_pct_7","sales_momentum_pct_28",
                                "sales_acceleration","sales_acceleration_pct"],
    "Demand Ratios":           ["week_ratio","month_ratio","yoy_ratio",
                                "price_elasticity_proxy"],
    "Volatility":              ["cv_7","cv_28","dept_cv_7","dept_cv_28",
                                "cat_cv_7","cat_cv_28","days_since_price_change"],
}

print("\n========== FEATURE SUMMARY ==========")
total = 0
for group, features in feature_groups.items():
    existing = [f for f in features if f in df.columns]
    print(f"\n{group} ({len(existing)}):")
    for f in existing:
        print(f"  ✓ {f}")
    total += len(existing)

print(f"\n=====================================")
print(f"TOTAL FEATURES: {total}")
print(f"=====================================")