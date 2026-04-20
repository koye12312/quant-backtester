"""Run the default backtest: SPY, SMA(20/50), $10k starting capital."""
from backtest import print_stats, run_sma_backtest

if __name__ == "__main__":
    stats = run_sma_backtest(
        symbol="SPY",
        short_window=20,
        long_window=50,
        initial_capital=10_000.0,
        cost_per_turn=0.0005,   # 5 bps per position flip; set to 0 to disable
    )
    print_stats(stats)
