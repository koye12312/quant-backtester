"""Black–Scholes pricer: textbook values, put–call parity and Greeks."""
import math

import pytest

from bonus_option import BSParams, bs_greeks, bs_price


def test_matches_hull_textbook_example():
    # Hull, Options, Futures, and Other Derivatives, Example 15.6:
    # S=42, K=40, r=10%, sigma=20%, T=0.5 -> call 4.76, put 0.81
    p = BSParams(S=42, K=40, r=0.10, sigma=0.20, T=0.5)
    assert bs_price(p, "call") == pytest.approx(4.76, abs=0.005)
    assert bs_price(p, "put") == pytest.approx(0.81, abs=0.005)


CASES = [
    BSParams(S=100, K=100, r=0.02, sigma=0.20, T=1.0),   # at the money
    BSParams(S=120, K=100, r=0.05, sigma=0.30, T=0.5),   # in the money call
    BSParams(S=80, K=100, r=0.01, sigma=0.15, T=2.0),    # out of the money call
    BSParams(S=100, K=95, r=0.00, sigma=0.60, T=0.1),    # high vol, short dated
]


@pytest.mark.parametrize("p", CASES)
def test_put_call_parity(p):
    # C - P = S - K e^{-rT} must hold exactly for European options
    lhs = bs_price(p, "call") - bs_price(p, "put")
    rhs = p.S - p.K * math.exp(-p.r * p.T)
    assert lhs == pytest.approx(rhs, abs=1e-10)


@pytest.mark.parametrize("p", CASES)
def test_prices_respect_no_arbitrage_bounds(p):
    disc_k = p.K * math.exp(-p.r * p.T)
    call, put = bs_price(p, "call"), bs_price(p, "put")
    assert max(p.S - disc_k, 0) - 1e-12 <= call <= p.S
    assert max(disc_k - p.S, 0) - 1e-12 <= put <= disc_k


def _bump(p, **changes):
    return BSParams(**{**p.__dict__, **changes})


def _finite_difference_greeks(p, kind):
    """Central differences of the price — an independent check of the formulas."""
    h_s = p.S * 1e-4
    h = 1e-5
    f = lambda q: bs_price(q, kind)  # noqa: E731
    return {
        "delta": (f(_bump(p, S=p.S + h_s)) - f(_bump(p, S=p.S - h_s))) / (2 * h_s),
        "gamma": (f(_bump(p, S=p.S + h_s)) - 2 * f(p) + f(_bump(p, S=p.S - h_s))) / h_s**2,
        "vega": (f(_bump(p, sigma=p.sigma + h)) - f(_bump(p, sigma=p.sigma - h))) / (2 * h),
        # theta is the change as time passes, i.e. as T *decreases*
        "theta": -(f(_bump(p, T=p.T + h)) - f(_bump(p, T=p.T - h))) / (2 * h),
        "rho": (f(_bump(p, r=p.r + h)) - f(_bump(p, r=p.r - h))) / (2 * h),
    }


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("p", CASES)
def test_greeks_match_finite_differences_within_0_01_percent(p, kind):
    analytic = bs_greeks(p, kind)
    numeric = _finite_difference_greeks(p, kind)
    for name, value in analytic.items():
        assert value == pytest.approx(numeric[name], rel=1e-4, abs=1e-8), name


def test_greek_signs_and_ranges():
    p = CASES[0]
    call, put = bs_greeks(p, "call"), bs_greeks(p, "put")
    assert 0 < call["delta"] < 1 and -1 < put["delta"] < 0
    assert call["delta"] - put["delta"] == pytest.approx(1.0)
    assert call["gamma"] == pytest.approx(put["gamma"]) and call["gamma"] > 0
    assert call["vega"] == pytest.approx(put["vega"]) and call["vega"] > 0
    assert call["rho"] > 0 > put["rho"]


@pytest.mark.parametrize("bad", [dict(S=0), dict(K=-1), dict(sigma=0), dict(T=0)])
def test_rejects_invalid_inputs(bad):
    with pytest.raises(ValueError):
        bs_price(_bump(CASES[0], **bad))


def test_rejects_unknown_option_kind():
    with pytest.raises(ValueError):
        bs_price(CASES[0], "straddle")
