"""
tests/test_backtesting.py — Test Suite for Backtesting Engine
Validates trade execution simulation, transaction cost deductions, and benchmark accounting.
"""

import unittest
import pandas as pd
import numpy as np

from backtesting.engine import BacktestEngine, BacktestResult


class TestBacktestingEngine(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        n = 200
        dates = pd.date_range("2026-01-01", periods=n, freq="1min")
        close = 150.0 + np.cumsum(np.random.normal(0, 0.4, n))
        high = close + np.random.uniform(0.1, 0.5, n)
        low = close - np.random.uniform(0.1, 0.5, n)
        open_p = close + np.random.uniform(-0.2, 0.2, n)
        volume = np.random.randint(5000, 30000, n)

        self.df = pd.DataFrame({
            "Datetime": dates,
            "Open": open_p,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume
        })

    def test_backtest_execution(self):
        engine = BacktestEngine(
            initial_capital=100000.0,
            take_profit_pct=1.5,
            stop_loss_pct=0.8,
            slippage_pct=0.05,
            brokerage_pct=0.03
        )
        result = engine.run(self.df, ticker="TEST.NS", use_indicator_filter=False)

        self.assertIsInstance(result, BacktestResult)
        self.assertEqual(result.initial_capital, 100000.0)
        self.assertGreaterEqual(result.max_drawdown_pct, 0.0)
        self.assertGreaterEqual(result.win_rate_pct, 0.0)
        self.assertLessEqual(result.win_rate_pct, 100.0)

        # Ensure trade records have slippage/fees accounted
        if result.trades:
            for t in result.trades:
                self.assertGreater(t.fees_and_slippage, 0.0)
                # Gross PnL - Fees = Net PnL
                self.assertAlmostEqual(t.gross_pnl - t.fees_and_slippage, t.net_pnl, places=2)

    def test_benchmark_comparison(self):
        engine = BacktestEngine()
        result = engine.run(self.df, ticker="TEST.NS")
        self.assertIn("benchmark_return_pct", result.to_dict())
        self.assertIn("excess_return_pct", result.to_dict())


if __name__ == "__main__":
    unittest.main()
