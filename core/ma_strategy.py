"""Simple moving-average crossover strategy."""
import pandas as pd


class MovingAverageStrategy:
    """SMA crossover on a single price series.

    Signal is 1 when the short SMA is above the long SMA, else 0 (flat/cash).
    """

    def __init__(self, short_window: int = 20, long_window: int = 50):
        if short_window <= 0 or long_window <= 0:
            raise ValueError("Windows must be positive integers.")
        if short_window >= long_window:
            raise ValueError("short_window must be strictly less than long_window.")
        self.short_window = int(short_window)
        self.long_window = int(long_window)

    def calculate_signals(self, prices: pd.Series) -> pd.DataFrame:
        df = prices.to_frame(name="Close").copy()
        df["SMA_Short"] = df["Close"].rolling(self.short_window).mean()
        df["SMA_Long"] = df["Close"].rolling(self.long_window).mean()
        df["Signal"] = (df["SMA_Short"] > df["SMA_Long"]).astype(int)
        # Force signal to 0 before either SMA has enough data.
        df.loc[df["SMA_Long"].isna(), "Signal"] = 0
        return df


def generate_positions(signals_df: pd.DataFrame) -> pd.Series:
    """Return a 1/0 target-exposure series aligned with the signals."""
    return signals_df["Signal"].fillna(0).astype(int)
