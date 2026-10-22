"""Backtester: no look-ahead bias, transaction costs, and benchmark behaviour."""
import numpy as np
import pandas as pd
import pytest

from backtest import run_sma_backtest
from core.ma_strategy import MovingAverageStrategy, generate_positions
from core.portfolio_manager import (
    compute_returns_from_prices,
    compute_strategy_returns,
    equity_curve_from_returns,
)


def synthetic_prices(n=400, seed=0):
    """A reproducible random-walk price series with a business-day index."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.01, size=n)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(100 * np.cumprod(1 + rets), index=idx, name="TEST")


def test_todays_return_uses_yesterdays_position():
    rets = pd.Series([0.10, 0.20, 0.30, 0.40])
    positions = pd.Series([0, 1, 1, 0])
    out = compute_strategy_returns(rets, positions)
    # Position taken at the end of day 1 earns day 2's return, not day 1's.
    assert out.tolist() == pytest.approx([0.0, 0.0, 0.30, 0.40])


def test_changing_the_future_does_not_change_the_past():
    """The strongest look-ahead check: rewrite every price after day k and
    confirm the strategy's returns up to day k are identical."""
    prices = synthetic_prices()
    k = 250
    altered = prices.copy()
    altered.iloc[k + 1:] *= np.linspace(0.5, 2.0, len(prices) - k - 1)

    def strategy_returns(p):
        positions = generate_positions(MovingAverageStrategy(20, 50).calculate_signals(p))
        return compute_strategy_returns(compute_returns_from_prices(p), positions)

    original, changed = strategy_returns(prices), strategy_returns(altered)
    pd.testing.assert_series_equal(original.iloc[: k + 1], changed.iloc[: k + 1])
    assert not original.iloc[k + 1:].equals(changed.iloc[k + 1:])


def test_transaction_costs_charged_on_each_position_change():
    rets = pd.Series([0.0] * 6)
    positions = pd.Series([0, 1, 1, 0, 0, 1])
    out = compute_strategy_returns(rets, positions, cost_per_turn=0.001)
    # Lagged positions are 0,0,1,1,0,0 -> changes on days 2 and 4.
    assert out.tolist() == pytest.approx([0, 0, -0.001, 0, -0.001, 0])


def test_signal_is_flat_until_the_long_average_exists():
    signals = MovingAverageStrategy(5, 20).calculate_signals(synthetic_prices(60))
    assert (signals["Signal"].iloc[:19] == 0).all()
    assert signals["SMA_Long"].iloc[19:].notna().all()


@pytest.mark.parametrize("short, long", [(0, 10), (10, 10), (20, 5)])
def test_rejects_invalid_windows(short, long):
    with pytest.raises(ValueError):
        MovingAverageStrategy(short, long)


def test_always_invested_matches_buy_and_hold():
    prices = synthetic_prices()
    rets = compute_returns_from_prices(prices)
    always_long = pd.Series(1, index=prices.index)
    strat = compute_strategy_returns(rets, always_long)
    # Identical except day 0, before the first position is held.
    pd.testing.assert_series_equal(strat.iloc[1:], rets.iloc[1:], check_names=False)


def test_equity_curve_compounds_returns():
    eq = equity_curve_from_returns(pd.Series([0.10, -0.10, 0.05]), initial_capital=1000)
    assert eq.tolist() == pytest.approx([1100, 990, 1039.5])


def test_full_backtest_runs_offline_and_reports_both_sides():
    stats = run_sma_backtest(prices=synthetic_prices(), symbol="TEST",
                             cost_per_turn=0.0005, show_plots=False, save_plots=False)
    for side in ("strategy", "benchmark"):
        assert set(stats[side]) == {"final_value", "CAGR", "MaxDrawdown", "Sharpe", "Sortino"}
        assert stats[side]["MaxDrawdown"] <= 0
        assert stats[side]["final_value"] > 0


def test_costs_never_improve_results():
    prices = synthetic_prices()
    free = run_sma_backtest(prices=prices, cost_per_turn=0.0, show_plots=False, save_plots=False)
    costly = run_sma_backtest(prices=prices, cost_per_turn=0.001, show_plots=False, save_plots=False)
    assert costly["strategy"]["final_value"] < free["strategy"]["final_value"]
    assert costly["benchmark"] == free["benchmark"]  # buy and hold never trades
