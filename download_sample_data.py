from core.data_fetcher import download_stock_data

if __name__ == "__main__":
    tickers = ['SPY', 'AAPL', 'MSFT', 'XLK', 'XLF']
    download_stock_data(tickers, start_date="2015-01-01")
    print("Sample data download complete! Now you can run the backtests.")