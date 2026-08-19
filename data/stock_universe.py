"""
data/stock_universe.py — Loads the NSE stock list and filters by LTP range.
Currently uses the hardcoded list from config — could be extended to fetch from NSE directly.
# TODO: add support for BSE stocks too
"""

from config.settings import ACTIVE_NSE_TICKERS, LTP_MIN, LTP_MAX


def get_universe() -> list:
    """
    Returns the list of NSE tickers we're monitoring.
    Strips the .NS suffix for display purposes.
    """
    return [t.replace(".NS", "") for t in ACTIVE_NSE_TICKERS]


def is_in_ltp_range(ltp: float) -> bool:
    """Check if the LTP is within ₹30–₹500 screening criteria."""
    return LTP_MIN <= ltp <= LTP_MAX


def filter_by_ltp(stocks: dict) -> dict:
    """
    Filter a dict of {symbol: data} to only include stocks with LTP in range.
    stocks should have structure: {symbol: {'ltp': float, ...}}
    """
    return {
        sym: data
        for sym, data in stocks.items()
        if is_in_ltp_range(data.get('ltp', 0.0))
    }
