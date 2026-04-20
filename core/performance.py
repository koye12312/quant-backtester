"""Performance metrics and plotting helpers."""
from pathlib import Path
from typing import Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    if equity.empty:
        return 0.0
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    years = len(equity) / periods_per_year
    if years <= 0:
        return 0.0
    return float((1.0 + total_return) ** (1.0 / years) - 1.0)


def compute_max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return float(dd.min())


def compute_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate_annual: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    if returns.empty or returns.std() == 0:
        return 0.0
    excess = returns - (risk_free_rate_annual / periods_per_year)
    return float(np.sqrt(periods_per_year) * excess.mean() / excess.std())


def compute_sortino_ratio(
    returns: pd.Series,
    risk_free_rate_annual: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Sortino uses only downside deviation in the denominator."""
    if returns.empty:
        return 0.0
    excess = returns - (risk_free_rate_annual / periods_per_year)
    downside = excess[excess < 0]
    if downside.empty or downside.std() == 0:
        return 0.0
    return float(
        (excess.mean() * periods_per_year)
        / (downside.std() * np.sqrt(periods_per_year))
    )


def summarize(equity: pd.Series, strategy_returns: pd.Series) -> dict:
    """Bundle the four headline metrics into a dict."""
    return {
        "final_value": float(equity.iloc[-1]) if len(equity) else 0.0,
        "CAGR": compute_cagr(equity),
        "MaxDrawdown": compute_max_drawdown(equity),
        "Sharpe": compute_sharpe_ratio(strategy_returns),
        "Sortino": compute_sortino_ratio(strategy_returns),
    }


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _resolve_save_path(save_path: Union[str, Path, bool, None], default_name: str) -> Optional[Path]:
    if save_path is False or save_path is None:
        return None
    if save_path is True:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        return PLOTS_DIR / default_name
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _show_or_close(show: bool) -> None:
    # If we're on a non-interactive backend (e.g. Agg), plt.show() is a no-op.
    if show and matplotlib.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close()


def plot_equity_curve(
    equity: pd.Series,
    benchmark: Optional[pd.Series] = None,
    title: str = "Equity Curve",
    save_path: Union[str, Path, bool, None] = True,
    show: bool = True,
) -> Optional[Path]:
    """Plot the equity curve (optionally overlaid with a benchmark)."""
    plt.figure(figsize=(10, 5))
    equity.plot(label="Strategy")
    if benchmark is not None and not benchmark.empty:
        benchmark.plot(label="Buy & Hold", linestyle="--")
        plt.legend()
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    path = _resolve_save_path(save_path, "equity_curve.png")
    if path is not None:
        plt.savefig(path, dpi=120)
        print(f"Saved plot -> {path}")
    _show_or_close(show)
    return path


def plot_drawdown(
    equity: pd.Series,
    title: str = "Drawdown",
    save_path: Union[str, Path, bool, None] = True,
    show: bool = True,
) -> Optional[Path]:
    """Plot the running drawdown of an equity curve."""
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0

    plt.figure(figsize=(10, 4))
    dd.plot(color="crimson")
    plt.fill_between(dd.index, dd.values, 0, alpha=0.25, color="crimson")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    path = _resolve_save_path(save_path, "drawdown.png")
    if path is not None:
        plt.savefig(path, dpi=120)
        print(f"Saved plot -> {path}")
    _show_or_close(show)
    return path
