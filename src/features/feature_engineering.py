import pandas as pd

# =====================================================
# LOAD DATA
# =====================================================

print("\nLoading transformed dataset...")

df = pd.read_parquet(
    "data/processed/sales_ca_1_foods.parquet"
)

df["date"] = pd.to_datetime(df["date"])

print("Dataset loaded.")
print(df.shape)


# =====================================================
# SORT DATA
# =====================================================

print("\nSorting data...")

df = df.sort_values(
    ["item_id", "date"]
)

# =====================================================
# LAG FEATURES
# =====================================================

print("\nCreating lag features...")

df["lag_7"] = (
    df.groupby("item_id")["sales"]
      .shift(7)
)

df["lag_14"] = (
    df.groupby("item_id")["sales"]
      .shift(14)
)

df["lag_28"] = (
    df.groupby("item_id")["sales"]
      .shift(28)
)

# =====================================================
# ROLLING MEAN FEATURES
# =====================================================

print("\nCreating rolling mean features...")

df["rolling_mean_7"] = (
    df.groupby("item_id")["sales"]
      .transform(
          lambda x:
          x.shift(1)
           .rolling(7)
           .mean()
      )
)

df["rolling_mean_28"] = (
    df.groupby("item_id")["sales"]
      .transform(
          lambda x:
          x.shift(1)
           .rolling(28)
           .mean()
      )
)

# =====================================================
# ROLLING STD FEATURES
# =====================================================

print("\nCreating rolling std features...")

df["rolling_std_7"] = (
    df.groupby("item_id")["sales"]
      .transform(
          lambda x:
          x.shift(1)
           .rolling(7)
           .std()
      )
)

df["rolling_std_28"] = (
    df.groupby("item_id")["sales"]
      .transform(
          lambda x:
          x.shift(1)
           .rolling(28)
           .std()
      )
)

# =====================================================
# WEEKEND FEATURE
# =====================================================

print("\nCreating weekend feature...")

df["is_weekend"] = (
    df["weekday"]
    .isin(["Saturday", "Sunday"])
    .astype(int)
)

# =====================================================
# PRICE FEATURES
# =====================================================

print("\nCreating price features...")

df["previous_price"] = (
    df.groupby("item_id")["sell_price"]
      .shift(1)
)

df["price_change"] = (
    df["sell_price"]
    - df["previous_price"]
)

df["price_change_pct"] = (
    df["price_change"]
    /
    df["previous_price"]
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

# Fill missing event values as text
event_columns = [
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
]

for col in event_columns:
    df[col] = df[col].fillna("No_Event").astype(str)


# Fill missing numeric values with 0
numeric_columns = [
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_std_7",
    "rolling_std_28",
    "sell_price",
    "previous_price",
    "price_change",
    "price_change_pct",
]

for col in numeric_columns:
    df[col] = df[col].fillna(0)

# =====================================================
# SAVE FEATURES
# =====================================================

output_path = (
    "data/features/"
    "sales_ca_1_foods_features.parquet"
)

df.to_parquet(
    output_path,
    index=False
)

print("\nFeature engineering completed.")

print("\nFinal Shape:")
print(df.shape)

print("\nFeature Dataset Saved:")
print(output_path)

print("\nNew Features Created:")

new_features = [
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_std_7",
    "rolling_std_28",
    "is_weekend",
    "price_change",
    "price_change_pct",
    "has_event"
]

for feature in new_features:
    print(feature)