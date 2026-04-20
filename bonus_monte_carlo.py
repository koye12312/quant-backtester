"""Geometric Brownian Motion Monte Carlo simulator."""
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

PLOTS_DIR = Path(__file__).resolve().parent / "plots"


def gbm_paths(
    S0: float = 100.0,
    mu: float = 0.05,
    sigma: float = 0.2,
    T: float = 1.0,
    steps: int = 252,
    n_paths: int = 1000,
    seed: Optional[int] = 42,
) -> np.ndarray:
    """Simulate ``n_paths`` GBM price paths with ``steps`` daily increments.

    Returns an array of shape ``(steps + 1, n_paths)`` where row 0 is ``S0``.
    """
    if steps <= 0 or n_paths <= 0:
        raise ValueError("steps and n_paths must be positive integers.")
    rng = np.random.default_rng(seed)
    dt = T / steps
    Z = rng.standard_normal(size=(steps, n_paths))
    increments = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.vstack([np.zeros((1, n_paths)), np.cumsum(increments, axis=0)])
    return S0 * np.exp(log_paths)


def est_prob_final_ge_k(paths: np.ndarray, K: float) -> float:
    """Estimate P(S_T >= K) from the simulated terminal prices."""
    final = paths[-1]
    return float((final >= K).mean())


def plot_paths(
    paths: np.ndarray,
    n_to_plot: int = 25,
    title: str = "Sample GBM Paths",
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    n_to_plot = min(n_to_plot, paths.shape[1])
    plt.figure(figsize=(10, 5))
    plt.plot(paths[:, :n_to_plot], alpha=0.7)
    plt.title(f"{title} (first {n_to_plot} of {paths.shape[1]})")
    plt.xlabel("Step")
    plt.ylabel("Price")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120)
        print(f"Saved plot -> {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    S0, K = 100.0, 110.0
    paths = gbm_paths(S0=S0, mu=0.05, sigma=0.2, T=1.0, steps=252, n_paths=10_000)

    prob = est_prob_final_ge_k(paths, K)
    final = paths[-1]
    print(f"Simulated {paths.shape[1]} paths over {paths.shape[0] - 1} steps.")
    print(f"Terminal mean:     {final.mean():.2f}")
    print(f"Terminal std dev:  {final.std():.2f}")
    print(f"P(S_T >= {K:g}):       {prob:.4f}")

    plot_paths(paths, save_path=PLOTS_DIR / "gbm_paths.png")
