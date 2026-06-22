from ingest import (
    load_calendar,
    load_sell_prices,
    load_sales_validation
)


calendar = load_calendar()
prices = load_sell_prices()
sales = load_sales_validation()

print("\nNumber of stores:")
print(sales["store_id"].nunique())

print("\nStores:")
print(sales["store_id"].unique())

print("\nNumber of categories:")
print(sales["cat_id"].nunique())

print("\nCategories:")
print(sales["cat_id"].unique())

print("\nNumber of departments:")
print(sales["dept_id"].nunique())

print("\nDepartments:")
print(sales["dept_id"].unique())