from ingest import (
    load_calendar,
    load_sell_prices,
    load_sales_validation,
    load_sales_evaluation,
    load_sample_submission,
)


def validate_data():
    calendar = load_calendar()
    sell_prices = load_sell_prices()
    sales_validation = load_sales_validation()
    sales_evaluation = load_sales_evaluation()
    sample_submission = load_sample_submission()

    print("Validation started...\n")

    required_calendar_cols = {"date", "wm_yr_wk", "d"}
    required_price_cols = {"store_id", "item_id", "wm_yr_wk", "sell_price"}
    required_sales_cols = {"id", "item_id", "dept_id", "cat_id", "store_id", "state_id"}

    assert required_calendar_cols.issubset(calendar.columns), "Calendar missing required columns"
    assert required_price_cols.issubset(sell_prices.columns), "Sell prices missing required columns"
    assert required_sales_cols.issubset(sales_validation.columns), "Sales validation missing required columns"
    assert required_sales_cols.issubset(sales_evaluation.columns), "Sales evaluation missing required columns"

    assert sales_validation["id"].is_unique, "Duplicate IDs found in sales_validation"
    assert sales_evaluation["id"].is_unique, "Duplicate IDs found in sales_evaluation"

    assert (sell_prices["sell_price"] >= 0).all(), "Negative sell prices found"

    day_cols = [col for col in sales_validation.columns if col.startswith("d_")]
    assert (sales_validation[day_cols] >= 0).all().all(), "Negative sales found"

    print("All validation checks passed!")
    print(f"Calendar rows: {calendar.shape[0]}")
    print(f"Sell price rows: {sell_prices.shape[0]}")
    print(f"Sales validation rows: {sales_validation.shape[0]}")
    print(f"Sales evaluation rows: {sales_evaluation.shape[0]}")
    print(f"Sample submission rows: {sample_submission.shape[0]}")


if __name__ == "__main__":
    validate_data()