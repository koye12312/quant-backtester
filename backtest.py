"""End-to-end SMA backtest: CSV -> signals -> positions -> equity -> metrics."""
from typing import Optional

import pandas as pd

from core.data_fetcher import load_csv_close
from core.ma_strategy import MovingAverageStrategy, generate_positions
from core.performance import (
    plot_drawdown,
    plot_equity_curve,
    summarize,
)
from core.portfolio_manager import (
    compute_returns_from_prices,
    compute_strategy_returns,
    equity_curve_from_returns,
)


def run_sma_backtest(
    symbol: str = "SPY",
    short_window: int = 20,
    long_window: int = 50,
    initial_capital: float = 10_000.0,
    cost_per_turn: float = 0.0,
    show_plots: bool = True,
    save_plots: bool = True,
    prices: Optional[pd.Series] = None,
) -> dict:
    """Run an SMA crossover backtest and return a stats dictionary.

    Parameters
    ----------
    symbol : str
        Ticker used to find ``data/<symbol>_1d.csv``. Ignored if ``prices`` is given.
    short_window, long_window : int
        SMA lookbacks (short must be < long).
    initial_capital : float
        Starting equity.
    cost_per_turn : float
        Proportional transaction cost applied on every change in position.
        E.g. ``0.0005`` = 5 bps per flip.
    show_plots, save_plots : bool
        Whether to display and/or save the equity/drawdown charts to ``plots/``.
    prices : pd.Series, optional
        Pre-loaded close series. If omitted, the CSV for ``symbol`` is loaded.
    """
    if prices is None:
        prices = load_csv_close(symbol)

    strat = MovingAverageStrategy(short_window=short_window, long_window=long_window)
    signals = strat.calculate_signals(prices)
    positions = generate_positions(signals)

    market_rets = compute_returns_from_prices(prices)

    # Strategy: positions are lagged one day inside compute_strategy_returns.
    strat_rets = compute_strategy_returns(market_rets, positions, cost_per_turn=cost_per_turn)
    equity = equity_curve_from_returns(strat_rets, initial_capital=initial_capital)

    # Benchmark: always invested.
    bench_rets = market_rets.reindex(equity.index).fillna(0.0)
    bench_equity = equity_curve_from_returns(bench_rets, initial_capital=initial_capital)

    strat_stats = summarize(equity, strat_rets)
    bench_stats = summarize(bench_equity, bench_rets)

    plot_equity_curve(
        equity,
        benchmark=bench_equity,
        title=f"{symbol} SMA({short_window}/{long_window}) vs Buy & Hold",
        save_path=save_plots,
        show=show_plots,
    )
    plot_drawdown(
        equity,
        title=f"{symbol} Strategy Drawdown",
        save_path=save_plots,
        show=show_plots,
    )

    return {
        "symbol": symbol,
        "short_window": short_window,
        "long_window": long_window,
        "initial_capital": initial_capital,
        "cost_per_turn": cost_per_turn,
        "strategy": strat_stats,
        "benchmark": bench_stats,
    }


def print_stats(stats: dict) -> None:
    """Pretty-print the output of ``run_sma_backtest``."""
    print(
        f"\n=== {stats['symbol']} | SMA({stats['short_window']}/{stats['long_window']}) "
        f"| capital ${stats['initial_capital']:,.0f} | cost/turn {stats['cost_per_turn']:.4%} ==="
    )
    header = f"{'Metric':<15}{'Strategy':>15}{'Buy & Hold':>15}"
    print(header)
    print("-" * len(header))
    rows = [
        ("Final value", "final_value", "${:>13,.2f}"),
        ("CAGR",        "CAGR",        "{:>14.2%}"),
        ("Max DD",      "MaxDrawdown", "{:>14.2%}"),
        ("Sharpe",      "Sharpe",      "{:>14.2f}"),
        ("Sortino",     "Sortino",     "{:>14.2f}"),
    ]
    for label, key, fmt in rows:
        s = fmt.format(stats["strategy"][key])
        b = fmt.format(stats["benchmark"][key])
        print(f"{label:<15}{s:>15}{b:>15}")


if __name__ == "__main__":
    stats = run_sma_backtest("SPY", 20, 50)
    print_stats(stats)
