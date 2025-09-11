# bonus_correlation.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from core.data_fetcher import load_csv_close

DATA_DIR = Path(__file__).parent / "data"

def main():
    files = sorted(DATA_DIR.glob("*_1d.csv"))
    if not files:
        print("No CSVs found in data/. Add files like SPY_1d.csv, XLK_1d.csv.")
        return

    data = {}
    for f in files:
        name = f.stem.replace("_1d", "")
        s = load_csv_close(name)                 # <- robust parsing
        data[name] = s.pct_change().dropna()

    df = pd.DataFrame(data).dropna(how="any")
    corr = df.corr()
    print("\nCorrelation matrix:\n", corr.round(3))

    plt.figure()
    plt.imshow(corr, interpolation='nearest')
    plt.title("ETF Correlation (daily returns)")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha='right')
    plt.yticks(range(len(corr.index)), corr.index)
    plt.colorbar()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
