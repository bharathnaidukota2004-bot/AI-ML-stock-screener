"""
simulator/paper_trader.py — Enhanced Paper Trading Simulator with persistence.

FEATURE 4: Paper Trading Simulator
Extended version of core/paper_trader.py with:
  - ₹10,000 fixed capital per trade
  - Trade dataclass for clean access
  - JSON persistence — trades survive app restarts
  - Summary string for the dashboard header

# Virtual money only — this helped me validate the ML model accuracy
# I was surprised how much the ghost detector improved the win rate
"""

import json
import time
import math
from dataclasses import dataclass, field, asdict
from typing import Optional, List
from config.settings import PAPER_TRADE_CAPITAL, STOP_LOSS_PCT, TARGET_PROFIT_PCT, TRADES_LOG_PATH
from utils.helpers import timestamp_now, format_inr


@dataclass
class Trade:
    """Represents a single paper trade — entry to exit."""
    symbol:       str
    signal_type:  str        # "BUY" or "SELL"
    entry_time:   str
    entry_price:  float
    quantity:     int
    target_price: float
    stop_loss:    float
    ghost_status: str = "CLEAN"
    score:        int  = 0
    exit_time:    str  = ""
    exit_price:   float = 0.0
    pnl:          float = 0.0
    status:       str  = "ACTIVE"   # ACTIVE / SUCCESS (TARGET) / LOSS (STOP LOSS)
    reason:       str  = ""


class PaperTrader:
    """
    Paper Trading Simulator with ₹10,000 per-trade fixed capital.
    Wraps core/paper_trader.py logic with extended features:
      - Trade dataclass for cleaner code
      - JSON log persistence across restarts
      - Summary line for the dashboard
    
    # Virtual money only — this helped me validate the ML model accuracy
    """

    def __init__(self):
        self.open_trades: dict   = {}     # symbol -> Trade
        self.closed_trades: List[Trade] = []
        self.avoided_trades: list = []    # [{symbol, reason, ...}]
        self._load_from_disk()

    def _load_from_disk(self):
        """Load trade history from trades_log.json if it exists."""
        try:
            with open(TRADES_LOG_PATH, 'r') as f:
                data = json.load(f)
                for td in data.get('closed_trades', []):
                    self.closed_trades.append(Trade(**td))
                self.avoided_trades = data.get('avoided_trades', [])
                print(f"[PaperTrader] Loaded {len(self.closed_trades)} closed trades from disk")
        except FileNotFoundError:
            pass  # first run — that's fine
        except Exception as e:
            print(f"[PaperTrader] Could not load trades_log.json: {e}")

    def _save_to_disk(self):
        """Persist current trade history to trades_log.json."""
        try:
            data = {
                'closed_trades':  [asdict(t) for t in self.closed_trades],
                'avoided_trades': self.avoided_trades
            }
            with open(TRADES_LOG_PATH, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[PaperTrader] Could not save trades_log.json: {e}")

    def open_trade(
        self,
        symbol: str,
        signal_type: str,
        ltp: float,
        ghost_status: str = "CLEAN",
        score: int = 0,
        reason: str = ""
    ) -> Optional[Trade]:
        """
        Open a new paper trade with ₹10,000 capital.
        Returns the Trade object, or None if already has open position for this stock.
        """
        if symbol in self.open_trades:
            return None  # already in position — don't double up

        qty = max(1, math.floor(PAPER_TRADE_CAPITAL / ltp))

        if signal_type == "BUY":
            target = round(ltp * (1.0 + TARGET_PROFIT_PCT / 100.0), 2)
            sl     = round(ltp * (1.0 - STOP_LOSS_PCT / 100.0), 2)
        else:
            target = round(ltp * (1.0 - TARGET_PROFIT_PCT / 100.0), 2)
            sl     = round(ltp * (1.0 + STOP_LOSS_PCT / 100.0), 2)

        trade = Trade(
            symbol=symbol,
            signal_type=signal_type,
            entry_time=timestamp_now(),
            entry_price=ltp,
            quantity=qty,
            target_price=target,
            stop_loss=sl,
            ghost_status=ghost_status,
            score=score,
            reason=reason
        )

        self.open_trades[symbol] = trade
        print(f"[PaperTrader] OPENED {signal_type} {symbol} @ ₹{ltp} x{qty} | T={target} SL={sl}")
        return trade

    def log_avoided(self, symbol: str, signal_type: str, ltp: float, reason: str, score: int = 0):
        """Record a signal that was rejected (ghost, low score, etc.)."""
        self.avoided_trades.append({
            'timestamp':   timestamp_now(),
            'symbol':      symbol,
            'signal_type': signal_type,
            'ltp':         ltp,
            'reason':      reason,
            'score':       score,
        })

    def update_price(self, symbol: str, current_ltp: float) -> Optional[Trade]:
        """
        Called on every price tick for open positions.
        Closes trade if target or stop loss is hit.
        Returns the closed Trade, or None if still open.
        """
        if symbol not in self.open_trades:
            return None

        trade = self.open_trades[symbol]

        if trade.signal_type == "BUY":
            trade.pnl    = round((current_ltp - trade.entry_price) * trade.quantity, 2)
            hit_target   = current_ltp >= trade.target_price
            hit_sl       = current_ltp <= trade.stop_loss
        else:
            trade.pnl    = round((trade.entry_price - current_ltp) * trade.quantity, 2)
            hit_target   = current_ltp <= trade.target_price
            hit_sl       = current_ltp >= trade.stop_loss

        if hit_target or hit_sl:
            trade.exit_time  = timestamp_now()
            trade.exit_price = current_ltp
            trade.status     = "SUCCESS (TARGET)" if hit_target else "LOSS (STOP LOSS)"
            del self.open_trades[symbol]
            self.closed_trades.append(trade)
            self._save_to_disk()
            print(f"[PaperTrader] CLOSED {trade.symbol} @ ₹{current_ltp} | PnL: {format_inr(trade.pnl)} | {trade.status}")
            return trade

        return None

    def get_statistics(self) -> dict:
        """Calculate win rate, P&L summary, and counts."""
        total  = len(self.closed_trades)
        wins   = sum(1 for t in self.closed_trades if "SUCCESS" in t.status)
        losses = total - wins
        win_rate = (wins / total * 100.0) if total > 0 else 0.0
        net_pnl  = sum(t.pnl for t in self.closed_trades)

        return {
            'total_closed':  total,
            'wins':          wins,
            'losses':        losses,
            'win_rate_pct':  round(win_rate, 1),
            'net_pnl':       round(net_pnl, 2),
            'avoided_count': len(self.avoided_trades),
            'open_count':    len(self.open_trades),
        }

    def get_summary_line(self) -> str:
        """
        One-line summary for the dashboard header.
        Example: 📋 Paper Trades: 12 closed | Win Rate: 66.7% | P&L: +₹1,247.50
        """
        stats = self.get_statistics()
        pnl_str = format_inr(stats['net_pnl'])
        if stats['net_pnl'] >= 0:
            pnl_str = f"+{pnl_str}"
        return (
            f"📋 Paper Trades: {stats['total_closed']} closed | "
            f"Win Rate: {stats['win_rate_pct']}% | "
            f"P&L: {pnl_str}"
        )

    def get_all_closed_as_dicts(self) -> list:
        """Returns closed trades as list of dicts — easier for the UI to consume."""
        return [asdict(t) for t in self.closed_trades]
