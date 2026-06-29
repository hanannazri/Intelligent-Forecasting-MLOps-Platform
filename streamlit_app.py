import streamlit as st

from src.dashboard.styles import load_css
from src.dashboard.data_loader import load_inventory_data, get_last_updated
from src.dashboard.sections import (
    show_header,
    show_inventory_summary,
    show_urgent_items,
    show_product_recommendation,
    show_inventory_planner,
    show_business_insights,
    show_footer,
)

st.set_page_config(
    page_title="Retail Demand Forecasting Dashboard",
    layout="wide",
)

load_css()

inventory_df = load_inventory_data()
last_updated = get_last_updated()

show_header()
show_inventory_summary(inventory_df)
show_urgent_items(inventory_df)
show_product_recommendation(inventory_df)
show_inventory_planner(inventory_df)
show_business_insights(inventory_df)
show_footer(last_updated)