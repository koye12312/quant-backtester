"""Black-Scholes option pricing and Greeks."""
import math
from dataclasses import dataclass
from typing import Literal

from scipy.stats import norm

OptionType = Literal["call", "put"]


@dataclass
class BSParams:
    """Inputs to the Black-Scholes model."""
    S: float      # spot price
    K: float      # strike price
    r: float      # annualised risk-free rate
    sigma: float  # annualised volatility
    T: float      # time to maturity, in years


def _d1_d2(p: BSParams) -> tuple[float, float]:
    if p.S <= 0 or p.K <= 0 or p.sigma <= 0 or p.T <= 0:
        raise ValueError("S, K, sigma, and T must all be positive.")
    vol_sqrt_t = p.sigma * math.sqrt(p.T)
    d1 = (math.log(p.S / p.K) + (p.r + 0.5 * p.sigma ** 2) * p.T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return d1, d2


def bs_price(p: BSParams, kind: OptionType = "call") -> float:
    """Return the Black-Scholes price of a European call or put."""
    d1, d2 = _d1_d2(p)
    disc = math.exp(-p.r * p.T)
    if kind == "call":
        return p.S * norm.cdf(d1) - p.K * disc * norm.cdf(d2)
    if kind == "put":
        return p.K * disc * norm.cdf(-d2) - p.S * norm.cdf(-d1)
    raise ValueError(f"Unknown option kind: {kind!r}")


def bs_greeks(p: BSParams, kind: OptionType = "call") -> dict:
    """Return delta, gamma, vega, theta, rho (annual-time conventions)."""
    d1, d2 = _d1_d2(p)
    pdf_d1 = norm.pdf(d1)
    disc = math.exp(-p.r * p.T)
    sqrt_t = math.sqrt(p.T)

    gamma = pdf_d1 / (p.S * p.sigma * sqrt_t)
    vega = p.S * pdf_d1 * sqrt_t

    if kind == "call":
        delta = norm.cdf(d1)
        theta = -(p.S * pdf_d1 * p.sigma) / (2 * sqrt_t) - p.r * p.K * disc * norm.cdf(d2)
        rho = p.K * p.T * disc * norm.cdf(d2)
    elif kind == "put":
        delta = norm.cdf(d1) - 1.0
        theta = -(p.S * pdf_d1 * p.sigma) / (2 * sqrt_t) + p.r * p.K * disc * norm.cdf(-d2)
        rho = -p.K * p.T * disc * norm.cdf(-d2)
    else:
        raise ValueError(f"Unknown option kind: {kind!r}")

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


if __name__ == "__main__":
    params = BSParams(S=100, K=100, r=0.02, sigma=0.2, T=1.0)
    print(f"Inputs: S={params.S}, K={params.K}, r={params.r}, sigma={params.sigma}, T={params.T}\n")
    for kind in ("call", "put"):
        px = bs_price(params, kind)
        greeks = bs_greeks(params, kind)
        print(f"{kind.title():<4}  price = {px:8.4f}")
        for name, value in greeks.items():
            print(f"       {name:<6} = {value:8.4f}")
        print()
