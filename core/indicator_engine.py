"""
core/indicator_engine.py — Advanced Technical Indicator Engine
Calculates RSI, MACD, Bollinger Bands, EMA, and ATR with zero lookahead bias.
Provides analytical summaries for UI and normalized features for ML model training.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from core.indicators import (
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_atr,
    add_technical_indicators
)
from core.data_fetcher import fetch_stock_1min_data


class TechnicalIndicatorEngine:
    """
    Dedicated engine for calculating and interpreting technical indicators:
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - Bollinger Bands
    - EMA (Exponential Moving Averages: 9, 21, 50, 200)
    - ATR (Average True Range & Normalized ATR)
    """

    def __init__(self):
        self._cache: Dict[str, pd.DataFrame] = {}

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enriches OHLCV DataFrame with all indicators chronologically.
        Guarantees no future data is accessed.
        """
        if df is None or df.empty:
            return df
        return add_technical_indicators(df)

    def analyze_stock(self, ticker: str, df: Optional[pd.DataFrame] = None, period_days: int = 3) -> Dict[str, Any]:
        """
        Fetches or takes data for a stock, computes all indicators,
        and generates an analytical summary report.
        """
        if df is None:
            df = fetch_stock_1min_data(ticker, period_days=period_days)
            
        if df is None or df.empty:
            return {
                'ticker': ticker,
                'status': 'NO_DATA',
                'summary': {},
                'df': pd.DataFrame()
            }

        df_enriched = self.compute_indicators(df)
        self._cache[ticker] = df_enriched
        summary = self.summarize_latest(df_enriched, ticker)

        return {
            'ticker': ticker,
            'status': 'OK',
            'summary': summary,
            'df': df_enriched
        }

    def summarize_latest(self, df: pd.DataFrame, ticker: str = "") -> Dict[str, Any]:
        """
        Generates human-readable and quantitative summary of the latest candle's indicators.
        """
        if df is None or df.empty:
            return {}

        latest = df.iloc[-1]
        close = float(latest.get('Close', latest.get('close', 0.0)))
        
        # 1. RSI
        rsi_val = float(latest.get('RSI_14', 50.0))
        if rsi_val >= 70.0:
            rsi_desc = "Overbought (Potential Reversal/Exhaustion)"
            rsi_bias = "BEARISH_REVERSAL"
        elif rsi_val <= 30.0:
            rsi_desc = "Oversold (Potential Bounce)"
            rsi_bias = "BULLISH_REVERSAL"
        elif rsi_val >= 55.0:
            rsi_desc = "Bullish Momentum"
            rsi_bias = "BULLISH"
        elif rsi_val <= 45.0:
            rsi_desc = "Bearish Momentum"
            rsi_bias = "BEARISH"
        else:
            rsi_desc = "Neutral Consolidation"
            rsi_bias = "NEUTRAL"

        # 2. MACD
        macd = float(latest.get('MACD', 0.0))
        macd_sig = float(latest.get('MACD_Signal', 0.0))
        macd_hist = float(latest.get('MACD_Hist', 0.0))
        if macd > macd_sig and macd_hist > 0:
            macd_desc = "Bullish (MACD above Signal)"
            macd_bias = "BULLISH"
        elif macd < macd_sig and macd_hist < 0:
            macd_desc = "Bearish (MACD below Signal)"
            macd_bias = "BEARISH"
        else:
            macd_desc = "Neutral / Converging"
            macd_bias = "NEUTRAL"

        # 3. Bollinger Bands
        bb_upper = float(latest.get('BB_Upper', close))
        bb_middle = float(latest.get('BB_Middle', close))
        bb_lower = float(latest.get('BB_Lower', close))
        bb_pct_b = float(latest.get('BB_Pct_B', 0.5))
        bb_bw = float(latest.get('BB_Bandwidth', 0.0))
        if bb_pct_b >= 1.0:
            bb_desc = "Trading Above Upper Band (Overextended)"
            bb_bias = "OVERBOUGHT"
        elif bb_pct_b <= 0.0:
            bb_desc = "Trading Below Lower Band (Undervalued / Oversold)"
            bb_bias = "OVERSOLD"
        elif bb_pct_b > 0.6:
            bb_desc = "Upper Band Expansion"
            bb_bias = "BULLISH"
        elif bb_pct_b < 0.4:
            bb_desc = "Lower Band Compression"
            bb_bias = "BEARISH"
        else:
            bb_desc = "Middle Band Range"
            bb_bias = "NEUTRAL"

        # 4. EMAs
        ema_9 = float(latest.get('EMA_9', close))
        ema_21 = float(latest.get('EMA_21', close))
        ema_50 = float(latest.get('EMA_50', close))
        ema_200 = float(latest.get('EMA_200', close))
        
        if close > ema_9 > ema_21:
            ema_trend = "Strong Short-Term Uptrend"
            ema_bias = "BULLISH"
        elif close < ema_9 < ema_21:
            ema_trend = "Strong Short-Term Downtrend"
            ema_bias = "BEARISH"
        else:
            ema_trend = "Mixed / Sideways"
            ema_bias = "NEUTRAL"

        # 5. ATR
        atr = float(latest.get('ATR_14', 0.0))
        natr = float(latest.get('NATR_14', 0.0))
        if natr > 1.5:
            vol_desc = f"High Volatility ({natr:.2f}% of price)"
        elif natr < 0.4:
            vol_desc = f"Low Volatility ({natr:.2f}% of price)"
        else:
            vol_desc = f"Moderate Volatility ({natr:.2f}% of price)"

        # Composite Technical Bias
        bull_count = sum(1 for b in [rsi_bias, macd_bias, bb_bias, ema_bias] if "BULLISH" in b)
        bear_count = sum(1 for b in [rsi_bias, macd_bias, bb_bias, ema_bias] if "BEARISH" in b)
        if bull_count >= 3:
            composite_verdict = "STRONG BUY CONFIRMATION"
        elif bull_count > bear_count:
            composite_verdict = "MODERATE BULLISH"
        elif bear_count >= 3:
            composite_verdict = "STRONG SELL CONFIRMATION"
        elif bear_count > bull_count:
            composite_verdict = "MODERATE BEARISH"
        else:
            composite_verdict = "NEUTRAL / INDECISIVE"

        return {
            'ticker': ticker,
            'close': round(close, 2),
            'rsi': {
                'value': round(rsi_val, 2),
                'description': rsi_desc,
                'bias': rsi_bias
            },
            'macd': {
                'macd': round(macd, 3),
                'signal': round(macd_sig, 3),
                'hist': round(macd_hist, 3),
                'description': macd_desc,
                'bias': macd_bias
            },
            'bollinger': {
                'upper': round(bb_upper, 2),
                'middle': round(bb_middle, 2),
                'lower': round(bb_lower, 2),
                'pct_b': round(bb_pct_b, 3),
                'bandwidth': round(bb_bw, 4),
                'description': bb_desc,
                'bias': bb_bias
            },
            'ema': {
                'ema_9': round(ema_9, 2),
                'ema_21': round(ema_21, 2),
                'ema_50': round(ema_50, 2),
                'ema_200': round(ema_200, 2),
                'trend': ema_trend,
                'bias': ema_bias
            },
            'atr': {
                'value': round(atr, 2),
                'natr_pct': round(natr, 2),
                'description': vol_desc
            },
            'composite_verdict': composite_verdict
        }

    @staticmethod
    def extract_ml_features(df: pd.DataFrame, idx: int = -1) -> Dict[str, float]:
        """
        Extracts normalized technical indicator features for ML training or inference.
        Returns normalized floats safe for machine learning models.
        """
        if df is None or df.empty:
            return {
                'rsi_norm': 0.5,
                'macd_hist_ratio': 0.0,
                'bb_pct_b': 0.5,
                'ema_ratio': 0.0,
                'natr': 0.5
            }
        
        row = df.iloc[idx]
        close = float(row.get('Close', 1.0))
        if close <= 0:
            close = 1.0

        rsi_14 = float(row.get('RSI_14', 50.0))
        macd_hist = float(row.get('MACD_Hist', 0.0))
        bb_pct_b = float(row.get('BB_Pct_B', 0.5))
        ema_ratio = float(row.get('EMA_Ratio', 0.0))
        natr = float(row.get('NATR_14', 0.5))

        return {
            'rsi_norm': float(np.clip(rsi_14 / 100.0, 0.0, 1.0)),
            'macd_hist_ratio': float(np.clip(macd_hist / close, -0.05, 0.05) * 20.0), # normalized scale
            'bb_pct_b': float(np.clip(bb_pct_b, -0.5, 1.5)),
            'ema_ratio': float(np.clip(ema_ratio, -0.1, 0.1) * 10.0),
            'natr': float(np.clip(natr / 5.0, 0.0, 2.0))
        }
