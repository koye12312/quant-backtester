from pathlib import Path
import pandas as pd
import numpy as np

from core.data_fetcher import load_csv_close
from core.ma_strategy import MovingAverageStrategy, generate_positions
from core.portfolio_manager import compute_returns_from_prices, equity_curve_from_positions
from core.performance import compute_cagr, compute_max_drawdown, compute_sharpe_ratio, compute_sortino_ratio, plot_equity_curve, plot_drawdown

DATA_DIR = Path(__file__).parent / "data"

def run_sma_backtest(symbol="SPY", short_window=20, long_window=50, initial_capital=10_000.0):
    prices = load_csv_close(symbol)
    strat = MovingAverageStrategy(short_window=short_window, long_window=long_window)
    signals = strat.calculate_signals(prices)
    positions = generate_positions(signals)  # 1 or 0 exposure

    rets = compute_returns_from_prices(prices)
    equity = equity_curve_from_positions(rets, positions, initial_capital=initial_capital)

    cagr = compute_cagr(equity)
    mdd = compute_max_drawdown(equity)
    sharpe = compute_sharpe_ratio(rets * positions)
    sortino = compute_sortino_ratio(rets * positions)

    # Plots
    plot_equity_curve(equity, title=f"{symbol} SMA({short_window}/{long_window}) Equity Curve")
    plot_drawdown(equity, title=f"{symbol} Drawdown")

    stats = {
        "symbol": symbol,
        "short_window": short_window,
        "long_window": long_window,
        "initial_capital": initial_capital,
        "final_value": float(equity.iloc[-1]),
        "CAGR": float(cagr),
        "MaxDrawdown": float(mdd),
        "Sharpe": float(sharpe),
        "Sortino": float(sortino),
    }
    return stats

if __name__ == "__main__":
    stats = run_sma_backtest("SPY", 20, 50)
    print("Backtest complete. Stats:\n", stats)