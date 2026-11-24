"""
tests/test_technical_indicators.py — Test Suite for Technical Indicator Engine
Validates mathematical correctness, boundary constraints, and zero lookahead bias.
"""

import unittest
import numpy as np
import pandas as pd

from core.indicators import (
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_atr,
    add_technical_indicators
)
from core.indicator_engine import TechnicalIndicatorEngine


class TestTechnicalIndicators(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        n = 100
        dates = pd.date_range("2026-01-01", periods=n, freq="1min")
        close = 150.0 + np.cumsum(np.random.normal(0, 0.5, n))
        high = close + np.random.uniform(0.1, 0.8, n)
        low = close - np.random.uniform(0.1, 0.8, n)
        open_p = close + np.random.uniform(-0.3, 0.3, n)
        volume = np.random.randint(1000, 20000, n)

        self.df = pd.DataFrame({
            "Datetime": dates,
            "Open": open_p,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume
        })

    def test_rsi_bounds_and_values(self):
        rsi = calculate_rsi(self.df["Close"], period=14)
        self.assertEqual(len(rsi), len(self.df))
        # Bounds must be strictly between 0 and 100
        self.assertTrue((rsi >= 0.0).all())
        self.assertTrue((rsi <= 100.0).all())
        # Not all values should be NaN
        self.assertFalse(rsi.isna().any())

    def test_macd_relationship(self):
        macd_df = calculate_macd(self.df["Close"], fast=12, slow=26, signal=9)
        self.assertIn("macd", macd_df.columns)
        self.assertIn("macd_signal", macd_df.columns)
        self.assertIn("macd_hist", macd_df.columns)
        # Check: hist == macd - signal (within float tolerance)
        diff = (macd_df["macd"] - macd_df["macd_signal"]) - macd_df["macd_hist"]
        self.assertTrue((diff.abs() < 1e-6).all())

    def test_bollinger_bands_ordering(self):
        bb_df = calculate_bollinger_bands(self.df["Close"], period=20, num_std=2.0)
        # Upper band must be >= Middle band >= Lower band
        self.assertTrue((bb_df["bb_upper"] >= bb_df["bb_middle"] - 1e-6).all())
        self.assertTrue((bb_df["bb_middle"] >= bb_df["bb_lower"] - 1e-6).all())
        self.assertTrue((bb_df["bb_bandwidth"] >= 0.0).all())

    def test_ema_smoothing(self):
        ema_20 = calculate_ema(self.df["Close"], period=20)
        self.assertEqual(len(ema_20), len(self.df))
        self.assertFalse(ema_20.isna().any())

    def test_atr_positive(self):
        atr_df = calculate_atr(self.df["High"], self.df["Low"], self.df["Close"], period=14)
        self.assertTrue((atr_df["atr"] >= 0.0).all())
        self.assertTrue((atr_df["natr"] >= 0.0).all())

    def test_zero_lookahead_bias(self):
        """
        Critical test: Verifies that indicator values at bar t do not change
        when future bars (t+1, t+2...) are added or modified.
        """
        # Calculate indicators on first 50 bars
        df_sub = self.df.iloc[:50].copy()
        ind_sub = add_technical_indicators(df_sub)
        val_at_bar_40_original = ind_sub.loc[40, "RSI_14"]

        # Calculate indicators on all 100 bars (where bars 50-99 have new data)
        ind_full = add_technical_indicators(self.df.copy())
        val_at_bar_40_after_future = ind_full.loc[40, "RSI_14"]

        # Must be mathematically identical!
        self.assertAlmostEqual(val_at_bar_40_original, val_at_bar_40_after_future, places=6)

    def test_indicator_engine_summary(self):
        engine = TechnicalIndicatorEngine()
        res = engine.analyze_stock("TATASTEEL.NS", df=self.df)
        self.assertEqual(res["status"], "OK")
        summary = res["summary"]
        self.assertIn("rsi", summary)
        self.assertIn("macd", summary)
        self.assertIn("bollinger", summary)
        self.assertIn("ema", summary)
        self.assertIn("atr", summary)
        self.assertIn("composite_verdict", summary)


if __name__ == "__main__":
    unittest.main()
