"""
screener/signal_scorer.py — Assigns a quality score (0–100) to each crossover signal.

FEATURE 2: Signal Quality Score
Each crossover gets a score based on 4 components.
This helps filter out weak signals — low score = don't trade it.

Scoring breakdown:
  - Volume confirmation (ETQ spike 5m vs avg)  → 30 pts max
  - SMMA slope angle (20 SMMA momentum)        → 25 pts max
  - Bid/Ask imbalance                          → 25 pts max
  - Historical win rate of this stock          → 20 pts max
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SignalScore:
    """Holds the score and breakdown for a single crossover signal."""
    score: int = 0
    label: str = "🔴 AVOID"
    breakdown: dict = field(default_factory=dict)


def score_volume(vol_5m: float, vol_60m: float) -> int:
    """
    Volume confirmation: is there a spike in the last 5 minutes?
    Compare ETQ 5m against average 5-min volume (60m / 12).
    Max: 30 points.
    """
    expected_5m = (vol_60m / 12.0) + 1e-5
    ratio = vol_5m / expected_5m

    if ratio >= 2.5:
        return 30   # massive spike — very strong confirmation
    elif ratio >= 1.8:
        return 24
    elif ratio >= 1.3:
        return 18
    elif ratio >= 1.0:
        return 10
    else:
        return 0    # below average volume — not great


def score_slope(smma_20_history: list) -> int:
    """
    SMMA slope angle: how fast is the 20 SMMA moving over last 3 candles?
    A steep slope means the crossover has momentum behind it.
    Max: 25 points.
    """
    if len(smma_20_history) < 4:
        return 8  # not enough data — give partial credit

    # Look at change over last 3 candles
    recent = smma_20_history[-4:]
    slope = recent[-1] - recent[0]  # total move over 3 candles
    abs_slope = abs(slope)

    if abs_slope >= 1.0:
        return 25
    elif abs_slope >= 0.5:
        return 18
    elif abs_slope >= 0.2:
        return 12
    elif abs_slope >= 0.05:
        return 7
    else:
        return 2   # basically flat — weak signal


def score_imbalance(bid_qty: float, ask_qty: float, signal_type: str) -> int:
    """
    Bid/Ask imbalance: (Bid - Ask) / (Bid + Ask)
    For BUY: positive imbalance (more bids) is good.
    For SELL: negative imbalance (more asks) is good.
    Max: 25 points.
    """
    total = bid_qty + ask_qty
    if total == 0:
        return 8

    imbalance = (bid_qty - ask_qty) / total  # range: -1 to +1

    # For BUY signals: we want positive imbalance (bid pressure)
    # For SELL signals: we want negative imbalance (ask pressure)
    directional = imbalance if signal_type == "BUY" else -imbalance

    if directional >= 0.30:
        return 25   # strong agreement between order book and signal
    elif directional >= 0.15:
        return 18
    elif directional >= 0.05:
        return 12
    elif directional >= 0.0:
        return 7
    else:
        return 0    # order book disagrees with signal — bad sign


def score_win_rate(symbol: str, closed_trades: list) -> int:
    """
    Historical win rate for this specific stock.
    Based on paper trader's closed trade history.
    Max: 20 points.

    If no history for this stock, give 10 points (neutral).
    """
    symbol_trades = []
    for t in closed_trades:
        sym = getattr(t, 'symbol', t.get('symbol', '') if isinstance(t, dict) else '')
        if sym == symbol:
            symbol_trades.append(t)

    if not symbol_trades:
        return 10  # no data — neutral score

    wins = 0
    for t in symbol_trades:
        st = getattr(t, 'status', t.get('status', '') if isinstance(t, dict) else '')
        if 'SUCCESS' in str(st):
            wins += 1

    win_rate = wins / len(symbol_trades)

    if win_rate >= 0.75:
        return 20
    elif win_rate >= 0.60:
        return 15
    elif win_rate >= 0.50:
        return 10
    elif win_rate >= 0.35:
        return 5
    else:
        return 0   # this stock has been a loser — approach with caution


def get_score_label(score: int) -> str:
    """Return emoji label based on score range."""
    if score >= 80:
        return "🟢 STRONG"
    elif score >= 60:
        return "🟡 MODERATE"
    elif score >= 40:
        return "🟠 WEAK"
    else:
        return "🔴 AVOID"


def score_signal(
    symbol: str,
    signal_type: str,
    vol_5m: float,
    vol_60m: float,
    smma_20_history: list,
    bid_qty: float,
    ask_qty: float,
    closed_trades: Optional[list] = None
) -> SignalScore:
    """
    Master scoring function — combines all 4 sub-scores into a single 0–100 score.
    Returns a SignalScore dataclass with score, label, and breakdown dict.
    """
    if closed_trades is None:
        closed_trades = []

    vol_pts  = score_volume(vol_5m, vol_60m)           # 30 pts max
    slope_pts = score_slope(smma_20_history)            # 25 pts max
    imb_pts  = score_imbalance(bid_qty, ask_qty, signal_type)  # 25 pts max
    wr_pts   = score_win_rate(symbol, closed_trades)    # 20 pts max

    total = vol_pts + slope_pts + imb_pts + wr_pts
    total = min(100, max(0, total))  # clamp just in case

    return SignalScore(
        score=total,
        label=get_score_label(total),
        breakdown={
            "volume_30pts": vol_pts,
            "slope_25pts": slope_pts,
            "imbalance_25pts": imb_pts,
            "win_rate_20pts": wr_pts
        }
    )
