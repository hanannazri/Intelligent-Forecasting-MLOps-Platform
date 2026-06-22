from pathlib import Path
import pandas as pd


RAW_DATA_PATH = Path("data/raw")


def load_calendar():
    return pd.read_csv(RAW_DATA_PATH / "calendar.csv")


def load_sell_prices():
    return pd.read_csv(RAW_DATA_PATH / "sell_prices.csv")


def load_sales_validation():
    return pd.read_csv(RAW_DATA_PATH / "sales_train_validation.csv")


def load_sales_evaluation():
    return pd.read_csv(RAW_DATA_PATH / "sales_train_evaluation.csv")


def load_sample_submission():
    return pd.read_csv(RAW_DATA_PATH / "sample_submission.csv")


if __name__ == "__main__":
    calendar = load_calendar()
    sell_prices = load_sell_prices()
    sales_validation = load_sales_validation()
    sales_evaluation = load_sales_evaluation()
    sample_submission = load_sample_submission()

    print("Data loaded successfully!")
    print(f"Calendar shape: {calendar.shape}")
    print(f"Sell prices shape: {sell_prices.shape}")
    print(f"Sales validation shape: {sales_validation.shape}")
    print(f"Sales evaluation shape: {sales_evaluation.shape}")
    print(f"Sample submission shape: {sample_submission.shape}")

    print("\nCalendar columns:")
    print(calendar.columns.tolist())

    print("\nSales validation columns first 10:")
    print(sales_validation.columns[:10].tolist())