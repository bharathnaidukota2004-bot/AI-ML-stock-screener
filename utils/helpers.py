"""
utils/helpers.py — Small reusable utilities.
Stuff I kept copy-pasting so I just put them here.
"""

import time
from datetime import datetime


def format_inr(amount: float) -> str:
    """Format a number as Indian Rupees with ₹ symbol."""
    sign = "-" if amount < 0 else ""
    return f"{sign}₹{abs(amount):,.2f}"


def format_qty(qty: int) -> str:
    """Format a large quantity in Lakhs (L) if >= 1L, else as plain number."""
    if qty >= 1_00_000:
        return f"{qty / 1_00_000:.1f}L"
    return f"{qty:,}"


def timestamp_now() -> str:
    """Current time as HH:MM:SS string — used everywhere in trade logs."""
    return time.strftime('%H:%M:%S')


def session_filename_stamp() -> str:
    """Returns YYYYMMDD_HHMMSS for use in report filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def pct_change(old: float, new: float) -> float:
    """Safe percentage change — avoids divide by zero."""
    if old == 0:
        return 0.0
    return round((new - old) / old * 100.0, 4)


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp a value between lo and hi. Python has this but I keep forgetting the syntax."""
    return max(lo, min(hi, value))
