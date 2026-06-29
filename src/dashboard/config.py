from pathlib import Path

INVENTORY_PATH = Path("reports/inventory_recommendations.csv")

DISPLAY_COLUMNS = [
    "item_id",
    "store_id",
    "dept_id",
    "cat_id",
    "expected_28_day_demand",
    "current_inventory",
    "recommended_order_qty",
    "stock_status",
    "risk_level",
]