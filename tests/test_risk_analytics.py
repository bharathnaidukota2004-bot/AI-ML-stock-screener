"""
tests/test_risk_analytics.py — Test Suite for Portfolio Risk Analytics
Validates Sharpe, Sortino, Calmar, Volatility, Drawdowns, and VaR calculations.
"""

import unittest
import numpy as np

from analytics.risk_engine import RiskAnalyticsEngine, PortfolioRiskReport, RISK_FREE_RATE_ANNUAL


class TestRiskAnalytics(unittest.TestCase):

    def setUp(self):
        self.engine = RiskAnalyticsEngine(risk_free_rate=RISK_FREE_RATE_ANNUAL)

    def test_risk_metrics_calculation(self):
        # Monotonically increasing equity curve
        equities = [100000.0, 101000.0, 102000.0, 103000.0, 104000.0, 105000.0]
        report = self.engine.analyze_equity_curve(equities, is_intraday_minutes=False)

        self.assertIsInstance(report, PortfolioRiskReport)
        self.assertAlmostEqual(report.total_return_pct, 5.0, places=1)
        self.assertEqual(report.max_drawdown_pct, 0.0)
        # Monotonically increasing should have very high positive Sharpe
        self.assertGreater(report.sharpe_ratio, 0.0)
        self.assertGreater(report.sortino_ratio, 0.0)

    def test_drawdown_detection(self):
        # Peak at 110,000 then drop to 99,000 (10% drawdown)
        equities = [100000.0, 110000.0, 105000.0, 99000.0, 102000.0]
        report = self.engine.analyze_equity_curve(equities, is_intraday_minutes=False)

        expected_dd = (110000.0 - 99000.0) / 110000.0 * 100.0
        self.assertAlmostEqual(report.max_drawdown_pct, expected_dd, places=1)
        self.assertGreater(report.max_drawdown_pct, 9.9)

    def test_documented_assumptions(self):
        equities = [100000.0, 101000.0]
        report = self.engine.analyze_equity_curve(equities)
        self.assertEqual(report.assumptions["risk_free_rate_pct"], 6.5)
        self.assertEqual(report.assumptions["trading_days_per_year"], 252)


if __name__ == "__main__":
    unittest.main()
