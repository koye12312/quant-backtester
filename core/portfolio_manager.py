"""Translate positions + returns into equity curves (no look-ahead)."""
import numpy as np
import pandas as pd


def compute_returns_from_prices(prices: pd.Series) -> pd.Series:
    """Simple daily returns, first row filled with 0."""
    return prices.pct_change().fillna(0.0)


def compute_strategy_returns(
    returns: pd.Series,
    positions: pd.Series,
    cost_per_turn: float = 0.0,
) -> pd.Series:
    """Strategy returns with no look-ahead bias.

    Today's P&L uses *yesterday's* position (``positions.shift(1)``) applied to
    today's market return. Optionally, every change in position incurs a
    proportional cost (e.g. ``cost_per_turn=0.0005`` for 5 bps round-trip
    per flip).
    """
    aligned = pd.concat([returns, positions], axis=1).dropna()
    aligned.columns = ["ret", "pos"]
    lagged_pos = aligned["pos"].shift(1).fillna(0)
    strat_ret = aligned["ret"] * lagged_pos

    if cost_per_turn:
        turnover = lagged_pos.diff().abs().fillna(0)
        strat_ret = strat_ret - turnover * float(cost_per_turn)

    strat_ret.name = "strategy_return"
    return strat_ret


def equity_curve_from_returns(
    strategy_returns: pd.Series,
    initial_capital: float = 10_000.0,
) -> pd.Series:
    """Compound strategy returns into an equity curve."""
    equity = (1.0 + strategy_returns).cumprod() * float(initial_capital)
    equity.name = "Equity"
    return equity


def equity_curve_from_positions(
    returns: pd.Series,
    positions: pd.Series,
    initial_capital: float = 10_000.0,
    cost_per_turn: float = 0.0,
) -> pd.Series:
    """Convenience wrapper: positions + returns -> equity curve."""
    strat_ret = compute_strategy_returns(returns, positions, cost_per_turn=cost_per_turn)
    return equity_curve_from_returns(strat_ret, initial_capital=initial_capital)
