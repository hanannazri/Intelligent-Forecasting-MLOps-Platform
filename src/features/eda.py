import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# LOAD DATA
# =====================================================

print("\n" + "="*60)
print("LOADING DATA")
print("="*60)

df = pd.read_parquet(
    "data/processed/sales_ca_1_foods.parquet"
)

df["date"] = pd.to_datetime(df["date"])

print("Dataset Loaded Successfully")
print()


# =====================================================
# 1. DATASET OVERVIEW
# =====================================================

print("\n" + "="*60)
print("1. DATASET OVERVIEW")
print("="*60)

print("Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nSales Summary:")
print(df["sales"].describe())


# =====================================================
# 2. MISSING VALUE ANALYSIS
# =====================================================

print("\n" + "="*60)
print("2. MISSING VALUE ANALYSIS")
print("="*60)

missing_values = df.isnull().sum()

print(missing_values)

missing_percentage = (
    missing_values / len(df)
) * 100

print("\nMissing Percentage:")
print(missing_percentage.sort_values(ascending=False))


# =====================================================
# 3. DEMAND SPARSITY ANALYSIS
# =====================================================

print("\n" + "="*60)
print("3. DEMAND SPARSITY ANALYSIS")
print("="*60)

zero_sales_pct = (
    (df["sales"] == 0).mean()
) * 100

print(f"Percentage of zero sales rows: {zero_sales_pct:.2f}%")

positive_sales_pct = (
    (df["sales"] > 0).mean()
) * 100

print(f"Percentage of positive sales rows: {positive_sales_pct:.2f}%")


# =====================================================
# 4. PRODUCT ANALYSIS
# =====================================================

print("\n" + "="*60)
print("4. PRODUCT ANALYSIS")
print("="*60)

print(
    f"Unique Products: "
    f"{df['item_id'].nunique()}"
)

top_products = (
    df.groupby("item_id")["sales"]
      .sum()
      .sort_values(ascending=False)
      .head(10)
)

print("\nTop 10 Products By Total Sales:")
print(top_products)


# =====================================================
# 5. SALES TREND OVER TIME
# =====================================================

print("\n" + "="*60)
print("5. TIME SERIES ANALYSIS")
print("="*60)

daily_sales = (
    df.groupby("date")["sales"]
      .sum()
)

print(
    f"Number of Dates: "
    f"{df['date'].nunique()}"
)

plt.figure(figsize=(14,6))
daily_sales.plot()

plt.title(
    "Total Daily Sales Over Time"
)

plt.xlabel("Date")
plt.ylabel("Sales")

plt.tight_layout()

plt.show()


# =====================================================
# 6. MONTHLY SEASONALITY
# =====================================================

print("\n" + "="*60)
print("6. MONTH ANALYSIS")
print("="*60)

monthly_sales = (
    df.groupby("month")["sales"]
      .mean()
)

print(monthly_sales)

plt.figure(figsize=(10,5))
monthly_sales.plot(kind="bar")

plt.title(
    "Average Sales By Month"
)

plt.xlabel("Month")
plt.ylabel("Average Sales")

plt.tight_layout()

plt.show()


# =====================================================
# 7. WEEKDAY ANALYSIS
# =====================================================

print("\n" + "="*60)
print("7. WEEKDAY ANALYSIS")
print("="*60)

weekday_sales = (
    df.groupby("weekday")["sales"]
      .mean()
)

print(weekday_sales)

plt.figure(figsize=(10,5))
weekday_sales.plot(kind="bar")

plt.title(
    "Average Sales By Weekday"
)

plt.xlabel("Weekday")
plt.ylabel("Average Sales")

plt.tight_layout()

plt.show()


# =====================================================
# 8. EVENT IMPACT ANALYSIS
# =====================================================

print("\n" + "="*60)
print("8. EVENT IMPACT ANALYSIS")
print("="*60)

df["has_event"] = (
    df["event_name_1"]
      .notnull()
)

event_sales = (
    df.groupby("has_event")["sales"]
      .mean()
)

print(event_sales)

print(
    "\nFalse = No Event Day"
)

print(
    "True = Event Day"
)


# =====================================================
# 9. SNAP IMPACT ANALYSIS
# =====================================================

print("\n" + "="*60)
print("9. SNAP ANALYSIS")
print("="*60)

snap_sales = (
    df.groupby("snap_CA")["sales"]
      .mean()
)

print(snap_sales)

print(
    "\n0 = No SNAP Day"
)

print(
    "1 = SNAP Day"
)


# =====================================================
# 10. PRICE ANALYSIS
# =====================================================

print("\n" + "="*60)
print("10. PRICE ANALYSIS")
print("="*60)

price_variation = (
    df.groupby("item_id")["sell_price"]
      .nunique()
)

print(
    "\nProducts With Most Price Changes:"
)

print(
    price_variation
    .sort_values(ascending=False)
    .head(10)
)

price_sales = (
    df.groupby("sell_price")["sales"]
      .mean()
)

print(
    "\nAverage Sales By Price:"
)

print(
    price_sales.head(10)
)


# =====================================================
# 11. MISSING PRICE INVESTIGATION
# =====================================================

print("\n" + "="*60)
print("11. MISSING PRICE INVESTIGATION")
print("="*60)

missing_price_count = (
    df["sell_price"]
      .isnull()
      .sum()
)

print(
    f"Missing Price Rows: "
    f"{missing_price_count}"
)

missing_price_products = (
    df[df["sell_price"].isnull()]
      ["item_id"]
      .value_counts()
      .head(20)
)

print(
    "\nTop Products With Missing Prices:"
)

print(missing_price_products)


# =====================================================
# FINAL SUMMARY
# =====================================================

print("\n" + "="*60)
print("EDA COMPLETED")
print("="*60)
