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
