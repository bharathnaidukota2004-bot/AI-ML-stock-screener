"""
ui/sparklines.py — Mini Unicode price chart for the terminal / dashboard.

FEATURE 6: Sparkline Charts

# Learned this trick from a Reddit post — looks cool in the terminal!
# The block characters are U+2581 through U+2588, mapping 8 levels of height.

Each stock gets a deque of its last 20 LTP values.
We map those to Unicode block chars to create a tiny histogram.
Arrow at the end shows overall trend direction.

Example output: ▁▂▃▄▅▆▇█▇▆  ↑
"""

from collections import deque
from typing import Dict

# Unicode block chars from lowest to highest (8 levels)
SPARK_CHARS = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]


def make_sparkline(prices: list) -> str:
    """
    Convert a list of prices into a Unicode sparkline string.

    Args:
        prices: list of float LTP values (ideally last 20)

    Returns:
        A string like "▁▂▃▄▅▆▇█▇▆  ↑"
        Returns "  ─  " if not enough data yet.
    """
    if len(prices) < 2:
        return "  ─  "

    min_p = min(prices)
    max_p = max(prices)
    price_range = max_p - min_p

    if price_range < 0.001:
        # Price is flat — show flat line
        trend = " →"
        return "▄" * min(len(prices), 10) + trend

    # Normalize each price to 0–7 range and pick a block char
    chars = []
    for p in prices:
        normalized = (p - min_p) / price_range  # 0.0 to 1.0
        idx = min(7, int(normalized * 8))
        chars.append(SPARK_CHARS[idx])

    # Determine trend arrow
    if prices[-1] > prices[0]:
        trend = " ↑"
    elif prices[-1] < prices[0]:
        trend = " ↓"
    else:
        trend = " →"

    return "".join(chars) + trend


class SparklineStore:
    """
    Maintains a rolling buffer of the last 20 LTP values per stock.
    Call update() on every tick, get_sparkline() to render.
    """

    def __init__(self, max_history: int = 20):
        self.history: Dict[str, deque] = {}
        self.max_history = max_history

    def update(self, symbol: str, ltp: float):
        """Add a new price point for this symbol."""
        if symbol not in self.history:
            self.history[symbol] = deque(maxlen=self.max_history)
        self.history[symbol].append(ltp)

    def get_sparkline(self, symbol: str) -> str:
        """Get the rendered sparkline string for a symbol."""
        if symbol not in self.history or len(self.history[symbol]) < 2:
            return "  ─  "
        return make_sparkline(list(self.history[symbol]))

    def get_prices(self, symbol: str) -> list:
        """Get the raw price history for a symbol."""
        return list(self.history.get(symbol, []))
