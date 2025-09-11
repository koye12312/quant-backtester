import pandas as pd
import numpy as np

def compute_returns_from_prices(prices: pd.Series) -> pd.Series:
    return prices.pct_change().fillna(0.0)

def equity_curve_from_positions(returns: pd.Series, positions: pd.Series, initial_capital=10_000.0) -> pd.Series:
    aligned = pd.concat([returns, positions], axis=1).dropna()
    aligned.columns = ["ret", "pos"]
    # Use yesterday's position to calculate today's return (avoid look-ahead bias)
    strat_ret = aligned["ret"] * aligned["pos"].shift(1).fillna(0)
    equity = (1 + strat_ret).cumprod() * float(initial_capital)
    equity.name = "Equity"
    return equity