"""
ml/ghost_detector.py — Detects fake/manipulated crossover signals.

FEATURE 1: Ghost Signal Detector

What is a Ghost Signal?
    Sometimes you get a SMMA crossover that looks real on the chart, but it's actually
    caused by a sudden bulk order or a market maker playing games. These are "ghost signals"
    — they appear briefly then reverse, and you'd get stopped out immediately.

    How to detect them:
    1. Wide spread: If the bid-ask spread at crossover time is > 2x the average spread
       over the last 20 minutes, something sketchy is happening.
    2. Low conviction: If the trading volume in the last 5 minutes is less than 10% of
       the 60-minute total, nobody's actually participating — it's a thin signal.

Returns:
    "CLEAN"           — looks like a real, tradeable signal
    "GHOST"           — spread manipulation detected
    "LOW_CONVICTION"  — volume is too thin to trust
"""

from config.settings import GHOST_SPREAD_MULTIPLIER, GHOST_ETQ_LOW_THRESHOLD


def detect_ghost(signal_data: dict) -> str:
    """
    Analyze a crossover event for ghost signal characteristics.

    Args:
        signal_data: dict with these keys:
            - spread (float): current bid-ask spread (ask_price - bid_price)
            - avg_spread_20m (float): average spread over last 20 minutes
            - vol_5m (float): traded quantity in last 5 minutes
            - vol_60m (float): traded quantity in last 60 minutes

    Returns:
        "CLEAN", "GHOST", or "LOW_CONVICTION"
    """
    spread      = signal_data.get('spread', 0.0)
    avg_spread  = signal_data.get('avg_spread_20m', spread)  # default to current if no history
    vol_5m      = signal_data.get('vol_5m', 0.0)
    vol_60m     = signal_data.get('vol_60m', 1.0)

    # Safety: avoid divide by zero
    if avg_spread <= 0:
        avg_spread = max(spread, 0.01)
    if vol_60m <= 0:
        vol_60m = 1.0

    # Check 1: Is the spread unusually wide right now?
    spread_ratio = spread / avg_spread
    is_wide_spread = spread_ratio > GHOST_SPREAD_MULTIPLIER

    if is_wide_spread:
        # This is the classic manipulation signal — someone dropped a huge order
        # which spiked the spread just enough to trigger the crossover
        return "GHOST"

    # Check 2: Is volume suspiciously low in the last 5 minutes?
    vol_ratio = vol_5m / vol_60m
    is_low_conviction = vol_ratio < GHOST_ETQ_LOW_THRESHOLD

    if is_low_conviction:
        # The crossover happened but nobody's trading — could reverse immediately
        return "LOW_CONVICTION"

    return "CLEAN"


def ghost_label_with_emoji(ghost_status: str) -> str:
    """Returns the ghost status with a visual indicator for the UI."""
    labels = {
        "CLEAN":          "✅ CLEAN",
        "GHOST":          "👻 GHOST ⚠️",
        "LOW_CONVICTION": "⚡ LOW VOL"
    }
    return labels.get(ghost_status, ghost_status)
