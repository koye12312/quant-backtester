import yfinance as yf
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)  # ensure folder exists

def download_stock_data(tickers, start_date='2015-01-01', end_date=None):
    """
    Download stock data from Yahoo Finance and save as CLEAN CSV files
    with just Date and Close (avoids multi-index header issues).
    """
    data = {}
    for t in tickers:
        print(f"Downloading {t}...")
        df = yf.download(
            t, start=start_date, end=end_date,
            progress=False, auto_adjust=True, group_by="column"
        )
        if df.empty:
            print(f"Warning: {t} returned no rows.")
            continue

        # Keep only Date + Close, reset index so 'Date' is a column
        out = df[['Close']].reset_index()
        out.to_csv(DATA_DIR / f"{t}_1d.csv", index=False)
        data[t] = out
    return data

def load_csv_close(symbol: str) -> pd.Series:
    """
    Load a CSV with columns ['Date','Close'] and return a float Series indexed by Date.
    """
    f = DATA_DIR / f"{symbol}_1d.csv"
    if not f.exists():
        raise FileNotFoundError(f"Expected {f} to exist. Put sample CSVs in data/ or run downloader.")
    df = pd.read_csv(f)

    # Normalise
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.set_index('Date')
    if 'Close' not in df.columns:
        raise ValueError("CSV must contain a 'Close' column.")

    s = pd.to_numeric(df['Close'], errors='coerce').dropna()
    s = s.sort_index()
    return s
