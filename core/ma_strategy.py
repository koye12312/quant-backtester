import pandas as pd
import numpy as np

class MovingAverageStrategy:
    """Simple moving-average crossover on a single price series."""
    def __init__(self, short_window=20, long_window=50):
        self.short_window = int(short_window)
        self.long_window = int(long_window)

    def calculate_signals(self, prices: pd.Series) -> pd.DataFrame:
        df = prices.to_frame(name="Close").copy()
        df["SMA_Short"] = df["Close"].rolling(self.short_window).mean()
        df["SMA_Long"]  = df["Close"].rolling(self.long_window).mean()
        # Signal: 1 when short > long, else 0 (flat). Keep it simple.
        df["Signal"] = (df["SMA_Short"] > df["SMA_Long"]).astype(int)
        return df

def generate_positions(signals_df: pd.DataFrame) -> pd.Series:
    """Return a 1/0 exposure series aligned with signals."""
    pos = signals_df["Signal"].fillna(0).astype(int)
    return pos