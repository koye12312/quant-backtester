import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    if equity.empty:
        return 0.0
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    years = len(equity) / periods_per_year
    if years <= 0:
        return 0.0
    return (1 + total_return) ** (1 / years) - 1

def compute_max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return float(dd.min()) if not dd.empty else 0.0

def compute_sharpe_ratio(returns: pd.Series, risk_free_rate_annual: float = 0.0, periods_per_year: int = 252) -> float:
    if returns.empty or returns.std() == 0:
        return 0.0
    excess = returns - (risk_free_rate_annual / periods_per_year)
    return float(np.sqrt(periods_per_year) * excess.mean() / excess.std())

def compute_sortino_ratio(returns: pd.Series, risk_free_rate_annual: float = 0.0, periods_per_year: int = 252) -> float:
    """Calculate Sortino ratio (only considers downside deviation)"""
    if returns.empty:
        return 0.0
    
    # Calculate excess returns
    excess_returns = returns - (risk_free_rate_annual / periods_per_year)
    
    # Only consider negative returns for downside deviation
    downside_returns = excess_returns[excess_returns < 0]
    
    if downside_returns.empty or downside_returns.std() == 0:
        return 0.0
    
    # Calculate Sortino ratio
    return float((excess_returns.mean() * periods_per_year) / (downside_returns.std() * np.sqrt(periods_per_year)))

def plot_equity_curve(equity: pd.Series, title: str = "Equity Curve") -> None:
    plt.figure()
    equity.plot(title=title)
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.show()

def plot_drawdown(equity: pd.Series, title: str = "Drawdown") -> None:
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    plt.figure()
    dd.plot(title=title)
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.show()