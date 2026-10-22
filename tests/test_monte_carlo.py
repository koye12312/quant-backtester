"""GBM Monte Carlo: shape, reproducibility, and agreement with exact results."""
import math

import numpy as np
import pytest
from scipy.stats import norm

from bonus_monte_carlo import est_prob_final_ge_k, gbm_paths

S0, MU, SIGMA, T = 100.0, 0.05, 0.20, 1.0


def test_shape_and_starting_price():
    paths = gbm_paths(S0=S0, steps=50, n_paths=7)
    assert paths.shape == (51, 7)
    assert np.all(paths[0] == S0)
    assert np.all(paths > 0)  # GBM prices can't go negative


def test_same_seed_gives_same_paths():
    a = gbm_paths(n_paths=100, seed=1)
    b = gbm_paths(n_paths=100, seed=1)
    c = gbm_paths(n_paths=100, seed=2)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


@pytest.fixture(scope="module")
def many_paths():
    return gbm_paths(S0=S0, mu=MU, sigma=SIGMA, T=T, steps=50, n_paths=200_000, seed=7)


def test_mean_terminal_price_matches_exact(many_paths):
    # Under GBM, E[S_T] = S0 * exp(mu * T)
    final = many_paths[-1]
    exact = S0 * math.exp(MU * T)
    standard_error = final.std() / math.sqrt(len(final))
    assert abs(final.mean() - exact) < 4 * standard_error


@pytest.mark.parametrize("K", [90.0, 100.0, 110.0, 130.0])
def test_probability_estimate_matches_exact(many_paths, K):
    # ln S_T is normal, so P(S_T >= K) = N(d), d = (ln(S0/K) + (mu - sigma^2/2) T) / (sigma sqrt(T))
    d = (math.log(S0 / K) + (MU - 0.5 * SIGMA**2) * T) / (SIGMA * math.sqrt(T))
    exact = norm.cdf(d)
    estimate = est_prob_final_ge_k(many_paths, K)
    standard_error = math.sqrt(exact * (1 - exact) / many_paths.shape[1])
    assert abs(estimate - exact) < 4 * standard_error


def test_log_returns_have_the_right_volatility(many_paths):
    log_ret = np.log(many_paths[-1] / S0)
    assert log_ret.std() == pytest.approx(SIGMA * math.sqrt(T), rel=0.01)


def test_rejects_invalid_sizes():
    with pytest.raises(ValueError):
        gbm_paths(steps=0)
    with pytest.raises(ValueError):
        gbm_paths(n_paths=0)
