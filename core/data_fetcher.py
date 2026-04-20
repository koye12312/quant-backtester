"""Download and load historical price data.

Writes (and reads) clean two-column ``Date,Close`` CSVs in ``data/``.
Handles yfinance's multi-index column output so CSVs never end up with
a stray ticker row that breaks parsing downstream.
"""
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _flatten_close(df: pd.DataFrame, ticker: str) -> pd.Series:
    """Return a 1-D Close series regardless of yfinance's column shape."""
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance can return (field, ticker) or (ticker, field); try both.
        if "Close" in df.columns.get_level_values(0):
            close = df["Close"]
        elif "Close" in df.columns.get_level_values(-1):
            close = df.xs("Close", axis=1, level=-1)
        else:
            raise ValueError(f"No 'Close' column in downloaded data for {ticker}.")
        if isinstance(close, pd.DataFrame):
            if ticker in close.columns:
                close = close[ticker]
            else:
                close = close.iloc[:, 0]
    else:
        if "Close" not in df.columns:
            raise ValueError(f"No 'Close' column in downloaded data for {ticker}.")
        close = df["Close"]
    return pd.to_numeric(close, errors="coerce").dropna()


def download_stock_data(
    tickers: Iterable[str],
    start_date: str = "2015-01-01",
    end_date: Optional[str] = None,
) -> Dict[str, pd.Series]:
    """Download daily prices from Yahoo Finance and save as ``Date,Close`` CSVs.

    Returns a dict mapping ticker -> Close series (indexed by Date).
    """
    out: Dict[str, pd.Series] = {}
    for t in tickers:
        print(f"Downloading {t}...")
        df = yf.download(
            t,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
        )
        if df is None or df.empty:
            print(f"  Warning: {t} returned no rows; skipping.")
            continue

        close = _flatten_close(df, t)
        close.index.name = "Date"
        close.name = "Close"
        close.to_frame().reset_index().to_csv(DATA_DIR / f"{t}_1d.csv", index=False)
        out[t] = close
        print(f"  Saved {len(close)} rows -> data/{t}_1d.csv")
    return out


def load_csv_close(symbol: str) -> pd.Series:
    """Load ``data/<symbol>_1d.csv`` and return a float Close series indexed by Date."""
    f = DATA_DIR / f"{symbol}_1d.csv"
    if not f.exists():
        raise FileNotFoundError(
            f"Expected {f} to exist. Run `python download_sample_data.py` "
            f"or place a Date,Close CSV in data/."
        )

    df = pd.read_csv(f)
    if "Date" not in df.columns or "Close" not in df.columns:
        raise ValueError(
            f"{f.name} must have columns 'Date' and 'Close'. "
            f"Found: {list(df.columns)}. Delete the file and re-run the downloader."
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date")
    s = pd.to_numeric(df["Close"], errors="coerce").dropna().sort_index()
    s.name = symbol
    return s
