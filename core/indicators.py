import numpy as np
import pandas as pd
from collections import deque
import time

def calculate_smma(prices, period):
    """
    Calculate Smoothed Moving Average (SMMA / RMA) for a price series or list.
    Formula:
      First SMMA = Simple Moving Average (SMA) over period
      Subsequent SMMA = (Previous SMMA * (period - 1) + Current Price) / period
    """
    if len(prices) < period:
        return float(np.mean(prices)) if len(prices) > 0 else 0.0
    
    prices = np.array(prices, dtype=float)
    smma = np.zeros(len(prices))
    smma[period - 1] = np.mean(prices[:period])
    
    for i in range(period, len(prices)):
        smma[i] = (smma[i - 1] * (period - 1) + prices[i]) / period
        
    return float(smma[-1])

def calculate_smma_series(series, period):
    """
    Computes a pandas Series of SMMA (Smoothed Moving Average) for 1-minute candles.
    """
    values = series.to_numpy(dtype=float)
    n = len(values)
    smma = np.full(n, np.nan)
    if n < period:
        return pd.Series(smma, index=series.index)
    
    smma[period - 1] = np.mean(values[:period])
    for i in range(period, n):
        smma[i] = (smma[i - 1] * (period - 1) + values[i]) / period
        
    return pd.Series(smma, index=series.index)


class TickAggregator:
    """
    Maintains dynamic sliding window buffers for ticks to compute:
    1. Traded Volume in last 5m, 20m, 60m
    2. Average LTP in last 20m, 60m
    3. SMMA(20) and SMMA(120)
    """
    def __init__(self, max_buffer_seconds=3600):
        self.ticks = deque()  # stores (timestamp, ltp, volume, bid_qty, ask_qty)
        self.prices_history = deque(maxlen=500) # for SMMA calculations

    def add_tick(self, ltp, volume, bid_qty, ask_qty, timestamp=None):
        if timestamp is None:
            timestamp = time.time()
        
        self.ticks.append({
            'ts': timestamp,
            'ltp': float(ltp),
            'vol': float(volume),
            'bid_qty': float(bid_qty),
            'ask_qty': float(ask_qty)
        })
        self.prices_history.append(float(ltp))
        self._prune(timestamp)

    def _prune(self, current_ts):
        cutoff = current_ts - 3600  # keep max 1 hour of ticks
        while self.ticks and self.ticks[0]['ts'] < cutoff:
            self.ticks.popleft()

    def get_metrics(self, current_ts=None):
        if current_ts is None:
            current_ts = time.time()

        if not self.ticks:
            return {
                'smma_20': 0.0,
                'smma_120': 0.0,
                'vol_5m': 0,
                'vol_20m': 0,
                'vol_60m': 0,
                'avg_price_20m': 0.0,
                'avg_price_60m': 0.0
            }

        # Filter ticks by window
        ts_5m = current_ts - 300
        ts_20m = current_ts - 1200
        ts_60m = current_ts - 3600

        prices_20m = []
        prices_60m = []
        vol_5m = 0.0
        vol_20m = 0.0
        vol_60m = 0.0

        for t in self.ticks:
            if t['ts'] >= ts_60m:
                prices_60m.append(t['ltp'])
                vol_60m += t['vol']
            if t['ts'] >= ts_20m:
                prices_20m.append(t['ltp'])
                vol_20m += t['vol']
            if t['ts'] >= ts_5m:
                vol_5m += t['vol']

        smma_20 = calculate_smma(list(self.prices_history), 20)
        smma_120 = calculate_smma(list(self.prices_history), 120)

        avg_p_20m = float(np.mean(prices_20m)) if prices_20m else (self.ticks[-1]['ltp'] if self.ticks else 0.0)
        avg_p_60m = float(np.mean(prices_60m)) if prices_60m else (self.ticks[-1]['ltp'] if self.ticks else 0.0)

        return {
            'smma_20': round(smma_20, 2),
            'smma_120': round(smma_120, 2),
            'vol_5m': int(vol_5m),
            'vol_20m': int(vol_20m),
            'vol_60m': int(vol_60m),
            'avg_price_20m': round(avg_p_20m, 2),
            'avg_price_60m': round(avg_p_60m, 2)
        }


# =============================================================================
# FEATURE 1: ADVANCED TECHNICAL INDICATOR ENGINE
# Pure chronological implementations with zero lookahead bias.
# Handles missing data, edge cases, and provides normalized features for ML.
# =============================================================================

def calculate_ema(series: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate Exponential Moving Average (EMA).
    Formula:
        Multiplier alpha = 2 / (period + 1)
        EMA_t = alpha * Price_t + (1 - alpha) * EMA_{t-1}
    Uses adjust=False for chronological streaming compatibility.
    """
    if series is None or len(series) == 0:
        return pd.Series(dtype=float)
    if len(series) < period:
        # Graceful fallback: expanding mean if series shorter than period
        return series.expanding(min_periods=1).mean()
    return series.ewm(span=period, adjust=False).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI) using Wilder's Smoothing.
    Strict chronological calculation — no forward-looking values used.
    
    Returns:
        pd.Series of RSI values (bounded strictly 0.0 to 100.0).
    """
    if series is None or len(series) == 0:
        return pd.Series(dtype=float)
        
    delta = series.diff()
    
    if len(series) <= period:
        # Not enough data: return neutral 50.0 series
        return pd.Series(50.0, index=series.index)
        
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    
    # Wilder's smoothing equivalent: ewm with alpha=1/period
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    # Fill warm-up periods cleanly with neutral 50.0
    return rsi.fillna(50.0).clip(lower=0.0, upper=100.0)


def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Calculate Moving Average Convergence Divergence (MACD).
    
    Returns:
        pd.DataFrame with columns: ['macd', 'macd_signal', 'macd_hist']
    """
    if series is None or len(series) == 0:
        return pd.DataFrame(columns=['macd', 'macd_signal', 'macd_hist'])
        
    fast_ema = calculate_ema(series, period=fast)
    slow_ema = calculate_ema(series, period=slow)
    macd_line = fast_ema - slow_ema
    signal_line = calculate_ema(macd_line, period=signal)
    hist = macd_line - signal_line
    
    res = pd.DataFrame({
        'macd': macd_line.fillna(0.0),
        'macd_signal': signal_line.fillna(0.0),
        'macd_hist': hist.fillna(0.0)
    }, index=series.index)
    return res


def calculate_bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """
    Calculate Bollinger Bands: Upper, Middle (SMA), Lower, %B, and Bandwidth.
    
    Returns:
        pd.DataFrame with:
            ['bb_middle', 'bb_upper', 'bb_lower', 'bb_pct_b', 'bb_bandwidth']
    """
    if series is None or len(series) == 0:
        return pd.DataFrame(columns=['bb_middle', 'bb_upper', 'bb_lower', 'bb_pct_b', 'bb_bandwidth'])
        
    middle = series.rolling(window=period, min_periods=1).mean()
    std = series.rolling(window=period, min_periods=1).std(ddof=0).fillna(0.0)
    
    upper = middle + (num_std * std)
    lower = middle - (num_std * std)
    
    band_diff = upper - lower
    # %B = (Price - Lower) / (Upper - Lower)
    pct_b = (series - lower) / (band_diff + 1e-10)
    # Bandwidth = (Upper - Lower) / Middle
    bandwidth = band_diff / (middle + 1e-10)
    
    return pd.DataFrame({
        'bb_middle': middle,
        'bb_upper': upper,
        'bb_lower': lower,
        'bb_pct_b': pct_b.fillna(0.5).clip(-0.5, 1.5),
        'bb_bandwidth': bandwidth.fillna(0.0)
    }, index=series.index)


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
    """
    Calculate Average True Range (ATR) and Normalized ATR (NATR = ATR / Close * 100).
    Uses Wilder's smoothing on True Range.
    
    Returns:
        pd.DataFrame with columns: ['atr', 'natr']
    """
    if len(close) == 0:
        return pd.DataFrame(columns=['atr', 'natr'])
        
    prev_close = close.shift(1).fillna(close)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Wilder's smoothing
    atr = tr.ewm(alpha=1.0 / period, min_periods=1, adjust=False).mean()
    natr = (atr / (close + 1e-10)) * 100.0  # normalized percentage of current price
    
    return pd.DataFrame({
        'atr': atr.fillna(0.0),
        'natr': natr.fillna(0.0)
    }, index=close.index)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich an OHLCV DataFrame with all 5 core technical indicators:
      - RSI (14)
      - MACD (12, 26, 9)
      - Bollinger Bands (20, 2.0)
      - EMA (9, 21, 50, 200)
      - ATR & NATR (14)
    
    Safely handles column casing ('close' vs 'Close', etc.) and strictly
    preserves chronological order without forward-looking data leakage.
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    
    # Map column names case-insensitively
    cols = {c.lower(): c for c in out.columns}
    close_col = cols.get('close', 'Close')
    high_col = cols.get('high', close_col)
    low_col = cols.get('low', close_col)
    
    close_series = out[close_col].astype(float)
    high_series = out[high_col].astype(float)
    low_series = out[low_col].astype(float)
    
    # 1. EMAs
    out['EMA_9'] = calculate_ema(close_series, 9)
    out['EMA_21'] = calculate_ema(close_series, 21)
    out['EMA_50'] = calculate_ema(close_series, 50)
    out['EMA_200'] = calculate_ema(close_series, 200)
    # EMA trend ratio (EMA_9 / EMA_21 - 1)
    out['EMA_Ratio'] = (out['EMA_9'] - out['EMA_21']) / (out['EMA_21'] + 1e-10)

    # 2. RSI (14)
    out['RSI_14'] = calculate_rsi(close_series, 14)
    
    # 3. MACD
    macd_df = calculate_macd(close_series, fast=12, slow=26, signal=9)
    out['MACD'] = macd_df['macd']
    out['MACD_Signal'] = macd_df['macd_signal']
    out['MACD_Hist'] = macd_df['macd_hist']
    
    # 4. Bollinger Bands
    bb_df = calculate_bollinger_bands(close_series, period=20, num_std=2.0)
    out['BB_Upper'] = bb_df['bb_upper']
    out['BB_Middle'] = bb_df['bb_middle']
    out['BB_Lower'] = bb_df['bb_lower']
    out['BB_Pct_B'] = bb_df['bb_pct_b']
    out['BB_Bandwidth'] = bb_df['bb_bandwidth']
    
    # 5. ATR & NATR
    atr_df = calculate_atr(high_series, low_series, close_series, period=14)
    out['ATR_14'] = atr_df['atr']
    out['NATR_14'] = atr_df['natr']

    return out

