"""
screener/crossover_detector.py — SMMA(20) vs SMMA(120) crossover detection.
Keeps track of the previous relationship so it can fire exactly once per crossover.
"""


def detect_crossover(
    prev_smma_20: float,
    prev_smma_120: float,
    curr_smma_20: float,
    curr_smma_120: float
) -> str:
    """
    Detects a SMMA crossover event by comparing previous and current relationships.

    Returns:
        "BUY"    — 20 crossed above 120 (bullish)
        "SELL"   — 20 crossed below 120 (bearish)
        "NONE"   — no crossover this tick
    """
    was_above = prev_smma_20 > prev_smma_120
    is_above  = curr_smma_20 > curr_smma_120

    if not was_above and is_above:
        return "BUY"    # bullish crossover — the fast line just crossed up
    if was_above and not is_above:
        return "SELL"   # bearish crossover — fast line dropped below slow
    return "NONE"


def get_crossover_label(smma_20: float, smma_120: float) -> str:
    """
    Returns current market state label (not crossover event, just state).
    Used for column display in the screener table.
    """
    if smma_20 > smma_120:
        return "BULLISH (20 > 120)"
    elif smma_20 < smma_120:
        return "BEARISH (20 < 120)"
    return "NEUTRAL"


def is_crossover_bullish(signal_type: str) -> bool:
    """Convenience check — is this a BUY crossover?"""
    return signal_type == "BUY"
