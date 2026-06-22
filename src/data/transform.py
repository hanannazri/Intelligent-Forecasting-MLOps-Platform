from pathlib import Path
import duckdb


RAW_DATA_PATH = Path("data/raw")
PROCESSED_DATA_PATH = Path("data/processed")
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

STORE_ID = "CA_1"


def transform_store_data():
    sales_path = RAW_DATA_PATH / "sales_train_validation.csv"
    calendar_path = RAW_DATA_PATH / "calendar.csv"
    prices_path = RAW_DATA_PATH / "sell_prices.csv"
    output_path = PROCESSED_DATA_PATH / f"sales_{STORE_ID.lower()}_foods.parquet"

    con = duckdb.connect()

    print(f"Transforming data for store: {STORE_ID}")

    con.execute(f"""
        CREATE OR REPLACE TABLE sales_wide AS
        SELECT *
        FROM read_csv_auto('{sales_path.as_posix()}')
        WHERE store_id = '{STORE_ID}'
            AND cat_id = 'FOODS'
    """)

    day_columns = [
        row[0]
        for row in con.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'sales_wide'
                AND regexp_matches(column_name, '^d_[0-9]+$')
            ORDER BY CAST(REPLACE(column_name, 'd_', '') AS INTEGER)
        """).fetchall()
    ]

    day_structs = ", ".join(
        [f"{{'d': '{col}', 'sales': {col}}}" for col in day_columns]
    )

    con.execute(f"""
        COPY (
            SELECT
                s.id,
                s.item_id,
                s.dept_id,
                s.cat_id,
                s.store_id,
                s.state_id,
                x.d,
                x.sales,
                c.date,
                c.wm_yr_wk,
                c.weekday,
                c.wday,
                c.month,
                c.year,
                c.event_name_1,
                c.event_type_1,
                c.event_name_2,
                c.event_type_2,
                c.snap_CA,
                c.snap_TX,
                c.snap_WI,
                p.sell_price
            FROM sales_wide s
            CROSS JOIN UNNEST([{day_structs}]) AS t(x)
            LEFT JOIN read_csv_auto('{calendar_path.as_posix()}') c
                ON x.d = c.d
            LEFT JOIN read_csv_auto('{prices_path.as_posix()}') p
                ON s.store_id = p.store_id
                AND s.item_id = p.item_id
                AND c.wm_yr_wk = p.wm_yr_wk
        )
        TO '{output_path.as_posix()}'
        (FORMAT PARQUET)
    """)

    print(f"Saved transformed data to: {output_path}")


if __name__ == "__main__":
    transform_store_data()