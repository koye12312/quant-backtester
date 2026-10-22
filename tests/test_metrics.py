"""Performance metrics checked against values worked out by hand."""
import math

import numpy as np
import pandas as pd
import pytest

from core.performance import (
    compute_cagr,
    compute_max_drawdown,
    compute_sharpe_ratio,
    compute_sortino_ratio,
)


def test_cagr_uses_calendar_time():
    # Doubling over exactly two years -> sqrt(2) - 1 = 41.42% a year
    idx = pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"])
    equity = pd.Series([100.0, 150.0, 200.0], index=idx)
    assert compute_cagr(equity) == pytest.approx(math.sqrt(2) - 1, rel=1e-3)


def test_cagr_without_dates_counts_intervals_not_rows():
    # 253 rows = 252 daily intervals = exactly one year
    equity = pd.Series(np.linspace(100, 110, 253))
    assert compute_cagr(equity) == pytest.approx(0.10)


def test_cagr_edge_cases():
    assert compute_cagr(pd.Series([], dtype=float)) == 0.0
    assert compute_cagr(pd.Series([100.0])) == 0.0


def test_max_drawdown():
    # Peak 120 -> trough 90 is a 25% fall; the later new high doesn't undo it
    assert compute_max_drawdown(pd.Series([100, 120, 90, 130.0])) == pytest.approx(-0.25)
    assert compute_max_drawdown(pd.Series([100, 101, 102.0])) == 0.0


def test_sharpe_hand_calculation():
    r = pd.Series([0.01, -0.005, 0.02, 0.0])
    expected = math.sqrt(252) * r.mean() / r.std()
    assert compute_sharpe_ratio(r) == pytest.approx(expected)
    assert compute_sharpe_ratio(pd.Series([0.001] * 10)) == 0.0  # no volatility


def test_sortino_uses_downside_deviation_over_all_days():
    r = pd.Series([0.02, -0.01, 0.03, -0.02])
    # Downside deviation: sqrt(mean([0, 0.01^2, 0, 0.02^2])) = sqrt(0.000125)
    downside_dev = math.sqrt((0.01**2 + 0.02**2) / 4)
    expected = (r.mean() * 252) / (downside_dev * math.sqrt(252))
    assert compute_sortino_ratio(r) == pytest.approx(expected)


def test_sortino_penalises_deeper_losses():
    # Same average return, and the losing days are equally spread out
    # (std of losses is identical), but the second series loses twice as much.
    # Taking the std of losing days only scores these two as equally risky;
    # downside deviation correctly ranks the deeper losses as worse.
    shallow = pd.Series([0.02, -0.010, -0.011, 0.031] * 10)
    deep = pd.Series([0.04, -0.020, -0.021, 0.031] * 10)
    assert shallow.mean() == pytest.approx(deep.mean())
    assert compute_sortino_ratio(deep) < compute_sortino_ratio(shallow)


def test_sortino_with_no_losing_days_is_zero_by_convention():
    assert compute_sortino_ratio(pd.Series([0.01, 0.02, 0.0])) == 0.0
