"""
data/processor.py — Wrapper around core/indicators.py for SMMA and tick aggregation.
Having this here means the screener layer doesn't need to know about core/ internals.
"""

from core.indicators import (
    calculate_smma,
    calculate_smma_series,
    TickAggregator
)

# Re-export for clean imports
__all__ = ["calculate_smma", "calculate_smma_series", "TickAggregator"]


def get_smma_slope(smma_values: list, lookback: int = 3) -> float:
    """
    Returns the slope of SMMA over the last `lookback` candles.
    Positive = rising, Negative = falling.
    Useful for the signal scorer and ghost detector.
    """
    if len(smma_values) < lookback + 1:
        return 0.0
    
    recent = smma_values[-lookback:]
    # Simple linear slope: last - first over the window
    slope = (recent[-1] - recent[0]) / lookback
    return round(slope, 4)


def get_bid_ask_imbalance(bid_qty: float, ask_qty: float) -> float:
    """
    Returns normalized imbalance: +1.0 = all bids, -1.0 = all asks.
    Formula: (bid - ask) / (bid + ask)
    """
    total = bid_qty + ask_qty
    if total == 0:
        return 0.0
    return round((bid_qty - ask_qty) / total, 4)
