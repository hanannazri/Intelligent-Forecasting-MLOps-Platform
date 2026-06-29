import pandas as pd
import streamlit as st
from datetime import datetime

from src.dashboard.config import INVENTORY_PATH


def get_last_updated():
    return datetime.fromtimestamp(
        INVENTORY_PATH.stat().st_mtime
    ).strftime("%d %B %Y")


@st.cache_data
def load_inventory_data():
    df = pd.read_csv(INVENTORY_PATH)

    df = df.rename(
        columns={"total_28_day_forecast": "expected_28_day_demand"}
    )

    df["expected_28_day_demand"] = df["expected_28_day_demand"].round().astype(int)
    df["current_inventory"] = df["current_inventory"].round().astype(int)
    df["recommended_order_qty"] = df["recommended_order_qty"].round().astype(int)

    return df