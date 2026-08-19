"""
data/fetcher.py — Thin wrapper around core/data_fetcher.py.
This abstraction means I can swap the data source (Fyers, AngelOne, yfinance)
without touching anything else. That was the whole point of separating it.

# Replay mode was the hardest part to build — had to refactor fetcher.py to support
# both live and historical modes
"""

import os
import pandas as pd
from typing import Optional

# Reuse existing fetch logic — no point rewriting it
from core.data_fetcher import (
    fetch_stock_1min_data as _fetch_real,
    generate_synthetic_1min_data as _gen_synthetic,
    prepare_dataset_with_indicators
)


def fetch_live_data(ticker: str, period_days: int = 5) -> pd.DataFrame:
    """
    Fetch 1-minute OHLCV data for a ticker.
    Tries yfinance first, falls back to synthetic data if offline.
    This is the function everything else should call — not the core one directly.
    """
    return _fetch_real(ticker, period_days)


def fetch_historical_data(ticker: str, date_str: str) -> Optional[pd.DataFrame]:
    """
    Load saved historical data for replay mode.
    Looks for a CSV in data/history/<date>/<ticker>.csv
    Returns None if file doesn't exist (replay will use synthetic then).
    """
    from config.settings import HISTORY_DIR
    filepath = os.path.join(HISTORY_DIR, date_str, f"{ticker}.csv")
    
    if not os.path.exists(filepath):
        # Fall back to synthetic data for demonstration
        print(f"[Fetcher] No historical data for {ticker} on {date_str}, using synthetic")
        df = _gen_synthetic(ticker, days=1)
        return df
    
    try:
        df = pd.read_csv(filepath, parse_dates=['Datetime'])
        df['Ticker'] = ticker
        print(f"[Fetcher] Loaded {len(df)} rows for {ticker} on {date_str}")
        return df
    except Exception as e:
        print(f"[Fetcher] Error loading {filepath}: {e}")
        return _gen_synthetic(ticker, days=1)


def get_prepared_data(ticker: str, period_days: int = 5) -> pd.DataFrame:
    """
    Fetch + add all indicators in one call.
    Convenient for the ML trainer and replay mode.
    """
    df = fetch_live_data(ticker, period_days)
    return prepare_dataset_with_indicators(df)
