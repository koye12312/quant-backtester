# Quant Finance Backtester (SMA + Options + Monte Carlo)

A compact backtesting & quant learning project for students and interns.  
Built with Python (pandas, numpy, matplotlib), it runs a moving-average crossover strategy on daily data and includes a Black–Scholes options pricer, GBM Monte Carlo simulator, and correlation analysis.

## ✨ Features

**Backtesting**  
- Simple Moving Average (SMA) crossover (e.g., 20/50)  
- No look-ahead bias (today's P&L uses yesterday's signal)  
- Runs on clean CSV files (Date,Close) or downloads via Yahoo Finance  

**Performance & Plots**  
- Key metrics: CAGR, Sharpe, Sortino, Max Drawdown  
- Equity curve and drawdown charts  

**Bonus Tools**  
- Options: Black–Scholes prices + Greeks (Δ, Γ, Vega, Θ, ρ)  
- Monte Carlo: GBM path simulation + simple probability estimate  
- Correlation: return correlation matrix & heatmap from your CSVs  

**Student-Friendly**  
- Flat, readable code  
- Clear places to edit parameters  

## 🛠️ Installation (Local)

```bash
# 1) Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

```bash
# (Optional) Download sample data into data/*.csv (Date,Close)
python download_sample_data.py

# Run the main backtest demo (SPY with SMA 20/50)
python example_usage.py

# Bonus scripts
python bonus_option.py          # Black–Scholes prices + Greeks
python bonus_monte_carlo.py     # GBM paths + probability
python bonus_correlation.py     # Correlation matrix/heatmap (uses data/*.csv)
```

## 🔧 Where to Edit Values

**Backtest parameters**  
File: `example_usage.py` – change the symbol, windows, and capital.

```python
from backtest import run_sma_backtest

stats = run_sma_backtest(
    symbol="SPY",      # e.g., "XLK", "XLF", "AAPL"
    short_window=20,   # try 10, 50, etc.
    long_window=50,    # try 100, 200, etc.
    initial_capital=10_000.0
)
```

**Downloader tickers**  
File: `download_sample_data.py` – add/remove tickers to fetch.

```python
tickers = ['SPY', 'AAPL', 'MSFT', 'XLK', 'XLF']
```

**Strategy rule & defaults**  
File: `core/ma_strategy.py` – SMA crossover; edit default windows if you like.

```python
class MovingAverageStrategy:
    def __init__(self, short_window=20, long_window=50):
        ...
# Signal: 1 when SMA_short > SMA_long, else 0 (cash)
```

**Options pricing inputs**  
File: `bonus_option.py` – set S, K, r, sigma, T.

```python
from bonus_option import BSParams, bs_price, bs_greeks
p = BSParams(S=100, K=100, r=0.02, sigma=0.2, T=1.0)
print("Call:", bs_price(p, "call"))
print("Put:",  bs_price(p, "put"))
print("Greeks:", bs_greeks(p, "call"))
```

**Monte Carlo settings**  
File: `bonus_monte_carlo.py` – S0, mu, sigma, T, steps, n_paths.

```python
paths = gbm_paths(S0=100.0, mu=0.05, sigma=0.2, T=1.0, steps=252, n_paths=1000)
```

**Correlation universe**  
File: `bonus_correlation.py` – uses every `data/*_1d.csv`.  
Add/remove CSVs to change the assets included.

## 🗃️ Data Format

Your CSVs must look like this (two columns only):

```csv
Date,Close
2015-01-02,206.38
2015-01-05,204.42
...
```

Tip: `download_sample_data.py` writes clean `Date,Close` files (auto-adjusted for splits/dividends).

## 🧯 Troubleshooting

**"could not convert string to float: 'AAPL'"**  
Your CSV has an extra header row. Remove `data/*.csv` and re-download:

```bash
# Windows
del data\*_1d.csv
# macOS/Linux
rm data/*_1d.csv
python download_sample_data.py
```

**ImportError: No module named 'scipy' (for options)**  
Activate your venv and reinstall:

```bash
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

**Plots not showing**  
Run from a terminal instead of an IDE that blocks GUI:

```bash
python example_usage.py
```



Remove `data/*.csv` if you want to track CSVs in the repo.

