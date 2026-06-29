import streamlit as st
import plotly.express as px

from src.dashboard.config import DISPLAY_COLUMNS
from src.dashboard.components import (
    section_title,
    subsection_title,
    kpi_card,
    info_card,
    recommend_card,
)


def show_header():
    st.markdown(
        """
        <div class="main-title">
            Demand Forecasting & Inventory Intelligence Dashboard
        </div>

        <div class="subtitle">
            Operational dashboard for demand forecasting, inventory planning,
            and intelligent replenishment decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_inventory_summary(inventory_df):
    section_title("📦 Inventory Summary")

    total_items = len(inventory_df)
    reorder_items = (inventory_df["stock_status"] == "Reorder Needed").sum()
    total_forecast_demand = inventory_df["expected_28_day_demand"].sum()
    total_order_qty = inventory_df["recommended_order_qty"].sum()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        kpi_card("📦 Total Products", f"{total_items:,}", "kpi-blue")

    with col2:
        kpi_card("🔴 Products Needing Reorder", f"{reorder_items:,}", "kpi-red")

    with col3:
        kpi_card("📈 Total 28-Day Demand", f"{int(total_forecast_demand):,}", "kpi-purple")

    with col4:
        kpi_card("🛒 Total Units to Order", f"{int(total_order_qty):,}", "kpi-green")

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()


def show_urgent_items(inventory_df):
    st.markdown(
        """
        <div class="section-title">🔥 Top Urgent Restock Items</div>
        <div class="subtitle">
            Top 10 products that require immediate replenishment based on forecasted demand and current inventory.
        </div>
        """,
        unsafe_allow_html=True,
    )

    urgent_df = inventory_df[
        inventory_df["stock_status"] == "Reorder Needed"
    ][DISPLAY_COLUMNS].sort_values(
        by="recommended_order_qty",
        ascending=False,
    )

    st.dataframe(
        urgent_df.head(10),
        use_container_width=True,
        hide_index=True,
        column_config={
            "item_id": st.column_config.TextColumn("Item ID", width="medium"),
            "store_id": st.column_config.TextColumn("Store", width="small"),
            "dept_id": st.column_config.TextColumn("Dept", width="small"),
            "cat_id": st.column_config.TextColumn("Category", width="small"),
            "expected_28_day_demand": st.column_config.NumberColumn("28-Day Demand", width="small"),
            "current_inventory": st.column_config.NumberColumn("Inventory", width="small"),
            "recommended_order_qty": st.column_config.NumberColumn("Order Qty", width="small"),
            "stock_status": st.column_config.TextColumn("Status", width="medium"),
            "risk_level": st.column_config.TextColumn("Risk", width="small"),
        },
    )

    st.divider()


def show_product_recommendation(inventory_df):
    section_title("🔍 Product Recommendation")

    item_ids = sorted(inventory_df["item_id"].unique())
    selected_item = st.selectbox("Select Product", item_ids)

    item_row = inventory_df[inventory_df["item_id"] == selected_item].iloc[0]

    st.markdown("### Product Information")

    c1, c2, c3 = st.columns(3)

    product_cards = [
        ("📦 Item ID", item_row["item_id"]),
        ("🏬 Department", item_row["dept_id"]),
        ("🏪 Store", item_row["store_id"]),
    ]

    for col, (title, value) in zip([c1, c2, c3], product_cards):
        with col:
            info_card(title, value)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### Forecast Summary")

    c1, c2, c3, c4 = st.columns(4)

    risk_display = (
        "🔴 High Demand"
        if item_row["risk_level"] == "High Demand"
        else "🟠 Medium Demand"
        if item_row["risk_level"] == "Medium Demand"
        else "🟢 Low Demand"
    )

    forecast_cards = [
        ("📈 Expected Demand", f"{int(item_row['expected_28_day_demand'])} Units"),
        ("📦 Inventory Available", f"{int(item_row['current_inventory'])} Units"),
        ("🛒 Order Recommendation", f"{int(item_row['recommended_order_qty'])} Units"),
        ("⚠️ Risk Level", risk_display),
    ]

    for col, (title, value) in zip([c1, c2, c3, c4], forecast_cards):
        with col:
            recommend_card(title, value)

    st.markdown("<br>", unsafe_allow_html=True)

    if item_row["recommended_order_qty"] > 0:
        st.error(
            f"""
#### 🔴 **Recommended Action**: Order **{int(item_row['recommended_order_qty'])} units**.
#### **Reason**: Current inventory is below the target stock level.
"""
        )
    else:
        st.success(
            """
#### 🟢 **Inventory Sufficient**
"""
        )

    st.divider()


def show_inventory_planner(inventory_df):
    section_title("📋 Inventory Planner")

    col1, col2, col3 = st.columns(3)

    selected_department = col1.multiselect(
        "Department",
        options=sorted(inventory_df["dept_id"].unique()),
        default=[],
    )

    selected_status = col2.multiselect(
        "Stock Status",
        options=sorted(inventory_df["stock_status"].unique()),
        default=[],
    )

    selected_risk = col3.multiselect(
        "Risk Level",
        options=sorted(inventory_df["risk_level"].unique()),
        default=[],
    )

    filtered_df = inventory_df.copy()

    if selected_department:
        filtered_df = filtered_df[
            filtered_df["dept_id"].isin(selected_department)
        ]

    if selected_status:
        filtered_df = filtered_df[
            filtered_df["stock_status"].isin(selected_status)
        ]

    if selected_risk:
        filtered_df = filtered_df[
            filtered_df["risk_level"].isin(selected_risk)
        ]

    filtered_df = filtered_df[DISPLAY_COLUMNS].sort_values(
        by="recommended_order_qty",
        ascending=False,
    )

    filtered_df["stock_status"] = filtered_df["stock_status"].replace(
        {
            "Reorder Needed": "🔴 Reorder Needed",
            "Sufficient Stock": "🟢 Sufficient Stock",
        }
    )

    filtered_df["risk_level"] = filtered_df["risk_level"].replace(
        {
            "High Demand": "🔴 High Demand",
            "Medium Demand": "🟠 Medium Demand",
            "Low Demand": "🟢 Low Demand",
        }
    )

    filtered_df = filtered_df.rename(
        columns={
            "item_id": "Item",
            "store_id": "Store",
            "dept_id": "Department",
            "cat_id": "Category",
            "expected_28_day_demand": "28-Day Demand",
            "current_inventory": "Current Inventory",
            "recommended_order_qty": "Recommended Order",
            "stock_status": "Status",
            "risk_level": "Risk",
        }
    )

    csv = filtered_df.to_csv(index=False).encode("utf-8")

    table_col, download_col = st.columns([3, 1])

    with table_col:
        st.write(f"Showing **{len(filtered_df)}** items")

    with download_col:
        st.download_button(
            label="⬇ Download CSV",
            data=csv,
            file_name="filtered_inventory_recommendations.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if filtered_df.empty:
        st.success("✅ No products match the selected filters.")
    else:
        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True,
        )

    st.divider()


def show_business_insights(inventory_df):
    section_title("📊 Business Insights")

    col1, col2 = st.columns([1.4, 1])

    with col1:
        subsection_title("Department-wise Units to Order")

        dept_order_qty = (
            inventory_df.groupby("dept_id", as_index=False)["recommended_order_qty"]
            .sum()
            .sort_values(by="recommended_order_qty", ascending=False)
        )

        fig_dept_order = px.bar(
            dept_order_qty,
            x="dept_id",
            y="recommended_order_qty",
            text="recommended_order_qty",
            labels={
                "dept_id": "Department",
                "recommended_order_qty": "Units to Order",
            },
        )

        fig_dept_order.update_layout(
            height=420,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", size=14),
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.12)"),
        )

        fig_dept_order.update_traces(
            textposition="outside",
            marker_line_width=0,
        )

        st.plotly_chart(fig_dept_order, use_container_width=True)

    with col2:
        subsection_title("Stock Status Summary")

        status_counts = inventory_df["stock_status"].value_counts().reset_index()
        status_counts.columns = ["stock_status", "count"]

        fig_status = px.pie(
            status_counts,
            names="stock_status",
            values="count",
            hole=0.55,
        )

        fig_status.update_layout(
            height=420,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", size=14),
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=True,
            legend=dict(
                orientation="h",
                y=-0.15,
                x=0.5,
                xanchor="center",
            ),
        )

        st.plotly_chart(fig_status, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col3, col4 = st.columns([1, 1.4])

    with col3:
        subsection_title("Risk Level Distribution")

        selected_dept_for_risk = st.selectbox(
            "Select Department",
            options=["All"] + sorted(inventory_df["dept_id"].unique().tolist()),
            key="risk_department_filter",
        )

        if selected_dept_for_risk == "All":
            risk_df = inventory_df.copy()
        else:
            risk_df = inventory_df[
                inventory_df["dept_id"] == selected_dept_for_risk
            ]

        risk_counts = risk_df["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["risk_level", "count"]

        fig_risk = px.pie(
            risk_counts,
            names="risk_level",
            values="count",
            hole=0.55,
        )

        fig_risk.update_layout(
            height=420,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", size=14),
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=True,
            legend=dict(
                orientation="h",
                y=-0.15,
                x=0.5,
                xanchor="center",
            ),
        )

        st.plotly_chart(fig_risk, use_container_width=True)

    with col4:
        subsection_title("Department-wise 28-Day Demand")

        selected_risk_for_dept = st.selectbox(
            "Select Risk Level",
            options=["All"] + sorted(inventory_df["risk_level"].unique().tolist()),
            key="dept_demand_risk_filter",
        )

        if selected_risk_for_dept == "All":
            dept_demand_df = inventory_df.copy()
        else:
            dept_demand_df = inventory_df[
                inventory_df["risk_level"] == selected_risk_for_dept
            ]

        dept_demand = (
            dept_demand_df.groupby("dept_id", as_index=False)["expected_28_day_demand"]
            .sum()
            .sort_values(by="expected_28_day_demand", ascending=False)
        )

        fig_dept_demand = px.bar(
            dept_demand,
            x="dept_id",
            y="expected_28_day_demand",
            text="expected_28_day_demand",
            labels={
                "dept_id": "Department",
                "expected_28_day_demand": "Expected 28-Day Demand",
            },
        )

        fig_dept_demand.update_layout(
            height=420,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", size=14),
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.12)"),
        )

        fig_dept_demand.update_traces(
            textposition="outside",
            marker_line_width=0,
        )

        st.plotly_chart(fig_dept_demand, use_container_width=True)

    st.divider()


def show_footer(last_updated):
    st.markdown(
        f"""
        <div class="footer">

        <div class="footer-title">
        📋 Dashboard Information
        </div>

        <div class="footer-grid">

        <div class="footer-item">
        <b>Forecast Model</b>
        <span>LightGBM</span>
        </div>

        <div class="footer-item">
        <b>Forecast Horizon</b>
        <span>28 Days</span>
        </div>

        <div class="footer-item">
        <b>Store</b>
        <span>CA_1</span>
        </div>

        <div class="footer-item">
        <b>Dataset</b>
        <span>M5 Forecasting</span>
        </div>

        <div class="footer-item">
        <b>Last Updated</b>
        <span>{last_updated}</span>
        </div>

        </div>

        </div>
        """,
        unsafe_allow_html=True,
    )