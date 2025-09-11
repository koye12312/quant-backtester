from backtest import run_sma_backtest

if __name__ == "__main__":
    stats = run_sma_backtest(symbol="SPY", short_window=20, long_window=50, initial_capital=10_000.0)
    print("\nSummary:")
    for k, v in stats.items():
        print(f"{k}: {v}")