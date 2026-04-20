# Quant Backtester

A compact, readable Python project for learning the mechanics of systematic
trading. It runs a **moving-average crossover** backtest on daily equity data
and ships with three extras — a **Black–Scholes** option pricer, a **Monte
Carlo** GBM simulator, and a **correlation** analyser for a basket of tickers.
=======
A compact backtesting & quant learning project.  
Built with Python (pandas, numpy, matplotlib), it runs a moving-average crossover strategy on daily data and includes a Black–Scholes options pricer, GBM Monte Carlo simulator, and correlation analysis.

Built with `pandas`, `numpy`, `matplotlib`, `scipy`, and `yfinance`.

---

## Features

**Backtester**
- SMA crossover (e.g. 20 / 50) with explicit long/flat positions.
- No look-ahead bias — today's P&L is driven by **yesterday's** signal.
- Optional proportional transaction costs per position change.
- **Buy-and-hold benchmark** reported alongside every run.
- Metrics: CAGR, Sharpe, Sortino, Max Drawdown.
- Equity-curve and drawdown charts saved to `plots/`.

**Options (`bonus_option.py`)**
- Black–Scholes price for European calls and puts.
- All five Greeks: Δ (delta), Γ (gamma), Vega, Θ (theta), ρ (rho).

**Monte Carlo (`bonus_monte_carlo.py`)**
- Geometric Brownian Motion path simulator.
- Terminal-price distribution summary and P(S_T ≥ K) estimate.

**Correlation (`bonus_correlation.py`)**
- Daily-return correlation matrix and annotated heatmap for every CSV in `data/`.

---

## Project structure

```
quant-backtester/
├── backtest.py               # End-to-end SMA backtest + pretty printer
├── example_usage.py          # One-liner entry point
├── download_sample_data.py   # Fetch SPY / AAPL / MSFT / XLK / XLF
├── bonus_option.py           # Black–Scholes prices and Greeks
├── bonus_monte_carlo.py      # GBM path simulator
├── bonus_correlation.py      # Return-correlation matrix and heatmap
├── core/
│   ├── data_fetcher.py       # yfinance download + robust CSV loader
│   ├── ma_strategy.py        # SMA crossover signal generation
│   ├── portfolio_manager.py  # Position → strategy returns → equity curve
│   └── performance.py        # Metrics + plotting helpers
├── data/                     # (generated) Date,Close CSVs
├── plots/                    # (generated) PNG output
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Installation

```bash
# 1) Clone
git clone https://github.com/koye12312/quant-backtester.git
cd quant-backtester

# 2) Create a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3) Install dependencies
pip install -r requirements.txt
```

Requires Python 3.9+.

---

## Quick start

```bash
# 1) Download sample data (writes Date,Close CSVs into data/)
python download_sample_data.py

# 2) Run the default SMA(20/50) backtest on SPY
python example_usage.py
```

Expected output (numbers will vary with the date range):

```
=== SPY | SMA(20/50) | capital $10,000 | cost/turn 0.0500% ===
Metric                Strategy     Buy & Hold
---------------------------------------------
Final value     $    23,104.60 $    34,951.77
CAGR                     8.32%         13.32%
Max DD                 -19.41%        -33.72%
Sharpe                    0.72           0.74
Sortino                   0.98           1.02
```

Equity-curve and drawdown charts land in `plots/`.

Run the bonus scripts standalone:

```bash
python bonus_option.py          # Black–Scholes prices and Greeks
python bonus_monte_carlo.py     # GBM paths + P(S_T ≥ K)
python bonus_correlation.py     # Correlation matrix and heatmap
```

---

## Configuring each module

**Backtest parameters** — `example_usage.py`

```python
from backtest import run_sma_backtest, print_stats

stats = run_sma_backtest(
    symbol="SPY",            # any ticker with a data/<SYMBOL>_1d.csv
    short_window=20,
    long_window=50,
    initial_capital=10_000.0,
    cost_per_turn=0.0005,    # 5 bps per position change; set to 0 to disable
)
print_stats(stats)
```

**Download universe** — `download_sample_data.py`

```python
TICKERS = ["SPY", "AAPL", "MSFT", "XLK", "XLF"]
START_DATE = "2015-01-01"
```

**Options inputs** — `bonus_option.py`

```python
from bonus_option import BSParams, bs_price, bs_greeks

p = BSParams(S=100, K=100, r=0.02, sigma=0.2, T=1.0)
print(bs_price(p, "call"))
print(bs_greeks(p, "put"))
```

**Monte Carlo inputs** — `bonus_monte_carlo.py`

```python
from bonus_monte_carlo import gbm_paths, est_prob_final_ge_k

paths = gbm_paths(S0=100, mu=0.05, sigma=0.2, T=1.0, steps=252, n_paths=10_000)
print(est_prob_final_ge_k(paths, K=110))
```

---

## Data format

CSVs in `data/` must have exactly two columns:

```csv
Date,Close
2015-01-02,206.38
2015-01-05,204.42
```

`download_sample_data.py` writes these directly and auto-adjusts for splits and
dividends. The loader handles yfinance's multi-index column output, so you
should no longer see the old `could not convert string to float: 'AAPL'`
error that happened when the ticker leaked into the header row.

---

## Implementation notes

**No look-ahead bias.** Strategy returns are computed as
`market_returns * positions.shift(1)` — today's P&L is what yesterday's signal
earned on today's price move. All metrics (Sharpe, Sortino, CAGR, MaxDD) are
computed from this same shifted series, so nothing peeks at future information.

**Transaction costs.** If `cost_per_turn > 0`, the absolute change in position
on each day is multiplied by that rate and subtracted from the day's return.
This is a simple proportional model; it does not simulate bid/ask spreads or
market impact.

**Metrics.**
- **CAGR** — compound annual growth rate, assuming 252 trading days/year.
- **Sharpe** — annualised; uses daily strategy returns, zero risk-free rate by default.
- **Sortino** — annualised; denominator uses only negative excess returns.
- **Max Drawdown** — peak-to-trough decline of the equity curve.

---

## Troubleshooting

**`FileNotFoundError: Expected data/SPY_1d.csv to exist`**
Run `python download_sample_data.py` first, or drop a `Date,Close` CSV into
`data/` by hand.

**`ModuleNotFoundError: No module named 'scipy'` (or pandas/yfinance/etc.)**
Your virtual environment isn't active, or deps aren't installed:

```bash
# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

**Plots don't pop up in my IDE.** They're always saved to `plots/*.png` on
every run — open them from there. If you want the interactive window too,
run the script from a terminal rather than an IDE that sandboxes GUI windows.

**Yahoo Finance returns nothing for a ticker.** Yahoo occasionally rate-limits
or changes ticker symbols. Try again after a minute, or swap the symbol for
another ETF.

---

## License

MIT — see [LICENSE](LICENSE).
