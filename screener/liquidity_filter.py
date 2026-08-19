"""
screener/liquidity_filter.py — Bid/Ask quantity liquidity filter.
Simple check: both bid_qty and ask_qty must be > 10 Lakhs (10,00,000).
This filters out illiquid stocks where large moves would slip badly.
"""

from config.settings import BID_QTY_MIN, ASK_QTY_MIN


def passes_liquidity_filter(bid_qty: float, ask_qty: float) -> bool:
    """
    Returns True if the stock has enough liquidity on both sides.
    10 Lakh = 10,00,000 = 1,000,000.
    """
    bid_ok = bid_qty >= BID_QTY_MIN
    ask_ok  = ask_qty >= ASK_QTY_MIN
    return bid_ok and ask_ok


def get_liquidity_reason(bid_qty: float, ask_qty: float) -> str:
    """
    Returns a human-readable reason if a stock fails the liquidity filter.
    Empty string if it passes.
    """
    if bid_qty < BID_QTY_MIN:
        return f"Bid qty {bid_qty/1e5:.1f}L < 10L threshold"
    if ask_qty < ASK_QTY_MIN:
        return f"Ask qty {ask_qty/1e5:.1f}L < 10L threshold"
    return ""
