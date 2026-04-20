"""Return-correlation matrix and heatmap across every CSV in data/."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from core.data_fetcher import load_csv_close

DATA_DIR = Path(__file__).resolve().parent / "data"
PLOTS_DIR = Path(__file__).resolve().parent / "plots"


def build_return_frame() -> pd.DataFrame:
    files = sorted(DATA_DIR.glob("*_1d.csv"))
    if not files:
        raise FileNotFoundError(
            "No CSVs found in data/. Run `python download_sample_data.py` first."
        )
    returns = {}
    for f in files:
        symbol = f.stem.replace("_1d", "")
        returns[symbol] = load_csv_close(symbol).pct_change().dropna()
    return pd.DataFrame(returns).dropna(how="any")


def plot_correlation_heatmap(corr: pd.DataFrame, save_path: Path | None = None, show: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, interpolation="nearest", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    # Annotate each cell with its correlation.
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Return Correlation (daily)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120)
        print(f"Saved plot -> {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    df = build_return_frame()
    corr = df.corr()
    print("\nCorrelation matrix:\n")
    print(corr.round(3).to_string())
    plot_correlation_heatmap(corr, save_path=PLOTS_DIR / "correlation_heatmap.png")


if __name__ == "__main__":
    main()
