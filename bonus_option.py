import math
from dataclasses import dataclass
from typing import Literal
from scipy.stats import norm

OptionType = Literal["call", "put"]

@dataclass
class BSParams:
    S: float   # spot
    K: float   # strike
    r: float   # risk-free rate (annual)
    sigma: float  # volatility (annual)
    T: float   # time to maturity (in years)

def d1_d2(p: BSParams):
    if p.S <= 0 or p.K <= 0 or p.sigma <= 0 or p.T <= 0:
        raise ValueError("S, K, sigma, T must be positive.")
    d1 = (math.log(p.S/p.K) + (p.r + 0.5*p.sigma**2)*p.T) / (p.sigma*math.sqrt(p.T))
    d2 = d1 - p.sigma*math.sqrt(p.T)
    return d1, d2

def bs_price(p: BSParams, kind: OptionType = "call") -> float:
    d1, d2 = d1_d2(p)
    if kind == "call":
        return p.S*norm.cdf(d1) - p.K*math.exp(-p.r*p.T)*norm.cdf(d2)
    else:
        return p.K*math.exp(-p.r*p.T)*norm.cdf(-d2) - p.S*norm.cdf(-d1)

def bs_greeks(p: BSParams, kind: OptionType = "call") -> dict:
    d1, d2 = d1_d2(p)
    pdf_d1 = norm.pdf(d1)
    if kind == "call":
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1
    gamma = pdf_d1 / (p.S * p.sigma * math.sqrt(p.T))
    vega = p.S * pdf_d1 * math.sqrt(p.T)
    theta_call = -(p.S*pdf_d1*p.sigma)/(2*math.sqrt(p.T)) - p.r*p.K*math.exp(-p.r*p.T)*norm.cdf(d2)
    theta_put  = -(p.S*pdf_d1*p.sigma)/(2*math.sqrt(p.T)) + p.r*p.K*math.exp(-p.r*p.T)*norm.cdf(-d2)
    theta = theta_call if kind == "call" else theta_put
    rho_call = p.K*p.T*math.exp(-p.r*p.T)*norm.cdf(d2)
    rho_put  = -p.K*p.T*math.exp(-p.r*p.T)*norm.cdf(-d2)
    rho = rho_call if kind == "call" else rho_put
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}

if __name__ == "__main__":
    params = BSParams(S=100, K=100, r=0.02, sigma=0.2, T=1.0)
    for kind in ("call", "put"):
        px = bs_price(params, kind)
        greeks = bs_greeks(params, kind)
        print(f"{kind.title()} Price: {px:.4f}")
        print("Greeks:", {k: round(v, 6) for k, v in greeks.items()})
        print()