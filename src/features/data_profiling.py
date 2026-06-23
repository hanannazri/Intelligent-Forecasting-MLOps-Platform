import pandas as pd

df = pd.read_parquet(
    "data/processed/sales_ca_1_foods.parquet"
)

print("Sample Data:")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSales Summary:")
print(df["sales"].describe())

