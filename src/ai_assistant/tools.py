import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "reports" / "inventory_recommendations.csv"


def load_inventory_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Inventory file not found at: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def get_stockout_products(top_n=10):
    df = load_inventory_data()

    result_df = df[
        (df["needs_reorder"] == True) &
        (df["recommended_order_qty"] > 0)
    ].copy()

    result_df = result_df.sort_values(
        by=["risk_level", "recommended_order_qty"],
        ascending=[True, False]
    )

    return result_df.head(top_n).to_dict(orient="records")


def get_reorder_recommendation(item_id):
    df = load_inventory_data()

    item_data = df[df["item_id"].astype(str).str.upper() == str(item_id).upper()]

    if item_data.empty:
        return {"error": f"No recommendation found for item_id: {item_id}"}

    return item_data.iloc[0].to_dict()


def get_inventory_summary():
    df = load_inventory_data()

    return {
        "total_products": int(df["item_id"].nunique()),
        "total_rows": int(len(df)),
        "high_risk_products": int(
            df["risk_level"].astype(str).str.lower().str.contains("high").sum()
        ),
        "total_reorder_quantity": int(df["recommended_order_qty"].sum()),
        "average_reorder_quantity": round(float(df["recommended_order_qty"].mean()), 2),
        "products_needing_reorder": int(df["needs_reorder"].sum()),
    }