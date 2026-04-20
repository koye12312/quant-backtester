"""Download daily close prices for a small universe of tickers into data/."""
from core.data_fetcher import download_stock_data

TICKERS = ["SPY", "AAPL", "MSFT", "XLK", "XLF"]
START_DATE = "2015-01-01"

if __name__ == "__main__":
    download_stock_data(TICKERS, start_date=START_DATE)
    print("\nDone. CSVs are in data/. Run `python example_usage.py` next.")
