"""
backtesting/engine.py — Historical Strategy Backtesting Engine
Simulates historical SMMA crossover + indicator trading strategies on 1-min / daily candles.
Models realistic transaction costs, slippage, trade life cycles, and compares with Buy-and-Hold benchmark.
Zero lookahead bias: execution and exit rules strictly reference past and contemporaneous bar data.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from core.data_fetcher import fetch_stock_1min_data, prepare_dataset_with_indicators
from config.settings import INITIAL_PAPER_CAPITAL, TARGET_PROFIT_PCT, STOP_LOSS_PCT


@dataclass
class TradeRecord:
    trade_id: int
    ticker: str
    direction: str          # "BUY" or "SELL"
    entry_time: Any
    exit_time: Any
    entry_price: float
    exit_price: float
    quantity: int
    gross_pnl: float
    net_pnl: float
    fees_and_slippage: float
    return_pct: float
    bars_held: int
    exit_reason: str        # "TARGET_HIT", "STOP_LOSS_HIT", "MAX_HOLDING_REACHED"


@dataclass
class BacktestResult:
    ticker: str
    initial_capital: float
    final_capital: float
    total_net_pnl: float
    total_return_pct: float
    benchmark_return_pct: float
    excess_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    profit_factor: float
    max_drawdown_pct: float
    avg_trade_pnl: float
    avg_win_pnl: float
    avg_loss_pnl: float
    risk_reward_ratio: float
    total_fees_paid: float
    trades: List[TradeRecord]
    equity_curve: List[Dict[str, Any]]
    assumptions: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'ticker': self.ticker,
            'initial_capital': round(self.initial_capital, 2),
            'final_capital': round(self.final_capital, 2),
            'total_net_pnl': round(self.total_net_pnl, 2),
            'total_return_pct': round(self.total_return_pct, 2),
            'benchmark_return_pct': round(self.benchmark_return_pct, 2),
            'excess_return_pct': round(self.excess_return_pct, 2),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate_pct': round(self.win_rate_pct, 1),
            'profit_factor': round(self.profit_factor, 2),
            'max_drawdown_pct': round(self.max_drawdown_pct, 2),
            'avg_trade_pnl': round(self.avg_trade_pnl, 2),
            'avg_win_pnl': round(self.avg_win_pnl, 2),
            'avg_loss_pnl': round(self.avg_loss_pnl, 2),
            'risk_reward_ratio': round(self.risk_reward_ratio, 2),
            'total_fees_paid': round(self.total_fees_paid, 2),
            'assumptions': self.assumptions,
            'trade_count': len(self.trades)
        }


class BacktestEngine:
    """
    Simulates quantitative trading strategy on historical data.
    """

    def __init__(
        self,
        initial_capital: float = INITIAL_PAPER_CAPITAL,
        position_size_pct: float = 10.0,
        take_profit_pct: float = TARGET_PROFIT_PCT,
        stop_loss_pct: float = STOP_LOSS_PCT,
        slippage_pct: float = 0.05,        # 0.05% slippage per side (5 bps)
        brokerage_pct: float = 0.03,       # 0.03% exchange & regulatory fees (STT, GST)
        max_holding_bars: int = 30         # 30-minute max intraday holding
    ):
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.slippage_pct = slippage_pct
        self.brokerage_pct = brokerage_pct
        self.max_holding_bars = max_holding_bars

    def run(
        self,
        df: pd.DataFrame,
        ticker: str = "TATASTEEL.NS",
        use_indicator_filter: bool = True
    ) -> BacktestResult:
        """
        Executes backtest sequentially candle-by-candle.
        Ensures strict chronological validity.
        """
        if df is None or df.empty or len(df) < 50:
            df = fetch_stock_1min_data(ticker, period_days=5)
            df = prepare_dataset_with_indicators(df)

        capital = float(self.initial_capital)
        peak_capital = capital
        max_drawdown = 0.0

        equity_curve = []
        trades: List[TradeRecord] = []
        trade_id = 0

        # Benchmark: Buy and Hold from first bar to last bar
        first_price = float(df.iloc[0]['Close'])
        last_price = float(df.iloc[-1]['Close'])
        benchmark_return_pct = ((last_price - first_price) / first_price) * 100.0

        active_trade = None

        for i in range(1, len(df)):
            curr_row = df.iloc[i]
            prev_row = df.iloc[i - 1]
            ts = curr_row.get('Datetime', i)
            high_p = float(curr_row['High'])
            low_p = float(curr_row['Low'])
            close_p = float(curr_row['Close'])

            # 1. Update & Manage Active Position
            if active_trade is not None:
                active_trade['bars_held'] += 1
                direction = active_trade['direction']
                target = active_trade['target_price']
                stop = active_trade['stop_loss_price']
                qty = active_trade['quantity']
                entry_p = active_trade['entry_price']

                exit_triggered = False
                exit_price = close_p
                exit_reason = ""

                if direction == "BUY":
                    if high_p >= target:
                        exit_price = target
                        exit_reason = "TARGET_HIT (+1.5%)"
                        exit_triggered = True
                    elif low_p <= stop:
                        exit_price = stop
                        exit_reason = "STOP_LOSS_HIT (-0.8%)"
                        exit_triggered = True
                    elif active_trade['bars_held'] >= self.max_holding_bars:
                        exit_price = close_p
                        exit_reason = "MAX_HOLDING_TIMEOUT"
                        exit_triggered = True
                else: # SELL
                    if low_p <= target:
                        exit_price = target
                        exit_reason = "TARGET_HIT (+1.5%)"
                        exit_triggered = True
                    elif high_p >= stop:
                        exit_price = stop
                        exit_reason = "STOP_LOSS_HIT (-0.8%)"
                        exit_triggered = True
                    elif active_trade['bars_held'] >= self.max_holding_bars:
                        exit_price = close_p
                        exit_reason = "MAX_HOLDING_TIMEOUT"
                        exit_triggered = True

                if exit_triggered:
                    # Apply slippage on exit
                    effective_exit = (
                        exit_price * (1.0 - self.slippage_pct / 100.0)
                        if direction == "BUY"
                        else exit_price * (1.0 + self.slippage_pct / 100.0)
                    )
                    gross_pnl = (
                        (effective_exit - entry_p) * qty
                        if direction == "BUY"
                        else (entry_p - effective_exit) * qty
                    )
                    # Fees on turnover (entry + exit)
                    turnover = (entry_p * qty) + (effective_exit * qty)
                    fees = turnover * (self.brokerage_pct / 100.0)
                    net_pnl = gross_pnl - fees
                    ret_pct = (net_pnl / (entry_p * qty)) * 100.0

                    capital += net_pnl
                    if capital > peak_capital:
                        peak_capital = capital
                    dd = (peak_capital - capital) / peak_capital * 100.0
                    if dd > max_drawdown:
                        max_drawdown = dd

                    trades.append(TradeRecord(
                        trade_id=trade_id,
                        ticker=ticker,
                        direction=direction,
                        entry_time=active_trade['entry_time'],
                        exit_time=ts,
                        entry_price=round(entry_p, 2),
                        exit_price=round(effective_exit, 2),
                        quantity=qty,
                        gross_pnl=round(gross_pnl, 2),
                        net_pnl=round(net_pnl, 2),
                        fees_and_slippage=round(fees, 2),
                        return_pct=round(ret_pct, 2),
                        bars_held=active_trade['bars_held'],
                        exit_reason=exit_reason
                    ))
                    active_trade = None

            # 2. Check for New Crossover Entry (Only if no active position)
            if active_trade is None and i < len(df) - 1:
                prev_20 = float(prev_row.get('SMMA_20', 0.0))
                prev_120 = float(prev_row.get('SMMA_120', 0.0))
                curr_20 = float(curr_row.get('SMMA_20', 0.0))
                curr_120 = float(curr_row.get('SMMA_120', 0.0))

                bullish_cross = (prev_20 <= prev_120) and (curr_20 > curr_120)
                bearish_cross = (prev_20 >= prev_120) and (curr_20 < curr_120)

                # Optional Indicator Confirmation Filters
                if use_indicator_filter and (bullish_cross or bearish_cross):
                    rsi = float(curr_row.get('RSI_14', 50.0))
                    ltq = float(curr_row.get('LTQ_Ratio', 1.0))
                    # Avoid overbought longs or oversold shorts; ensure volume support
                    if bullish_cross and (rsi > 75.0 or ltq < 1.05):
                        bullish_cross = False
                    if bearish_cross and (rsi < 25.0 or ltq < 1.05):
                        bearish_cross = False

                if bullish_cross or bearish_cross:
                    direction = "BUY" if bullish_cross else "SELL"
                    trade_id += 1
                    # Effective entry price with slippage
                    effective_entry = (
                        close_p * (1.0 + self.slippage_pct / 100.0)
                        if direction == "BUY"
                        else close_p * (1.0 - self.slippage_pct / 100.0)
                    )
                    # Sizing: allocated capital
                    allocated = capital * (self.position_size_pct / 100.0)
                    qty = max(1, int(allocated / effective_entry))

                    if direction == "BUY":
                        target_p = effective_entry * (1.0 + self.take_profit_pct / 100.0)
                        stop_p = effective_entry * (1.0 - self.stop_loss_pct / 100.0)
                    else:
                        target_p = effective_entry * (1.0 - self.take_profit_pct / 100.0)
                        stop_p = effective_entry * (1.0 + self.stop_loss_pct / 100.0)

                    active_trade = {
                        'direction': direction,
                        'entry_time': ts,
                        'entry_price': effective_entry,
                        'target_price': target_p,
                        'stop_loss_price': stop_p,
                        'quantity': qty,
                        'bars_held': 0
                    }

            # Record point on equity curve
            equity_curve.append({
                'timestamp': ts,
                'equity': round(capital, 2),
                'close': close_p
            })

        # Summary calculations
        total_trades = len(trades)
        wins = [t for t in trades if t.net_pnl > 0]
        losses = [t for t in trades if t.net_pnl <= 0]
        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate = (winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0

        gross_profit = sum(t.net_pnl for t in wins)
        gross_loss = abs(sum(t.net_pnl for t in losses))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 1.0)

        total_net_pnl = capital - self.initial_capital
        total_return_pct = (total_net_pnl / self.initial_capital) * 100.0
        excess_return = total_return_pct - benchmark_return_pct

        avg_trade = (total_net_pnl / total_trades) if total_trades > 0 else 0.0
        avg_win = (gross_profit / winning_trades) if winning_trades > 0 else 0.0
        avg_loss = (gross_loss / losing_trades) if losing_trades > 0 else 0.0
        rr_ratio = (avg_win / avg_loss) if avg_loss > 0 else avg_win

        total_fees = sum(t.fees_and_slippage for t in trades)

        assumptions = {
            'initial_capital': self.initial_capital,
            'position_size_pct': self.position_size_pct,
            'target_profit_pct': self.take_profit_pct,
            'stop_loss_pct': self.stop_loss_pct,
            'slippage_per_side_pct': self.slippage_pct,
            'turnover_fees_pct': self.brokerage_pct,
            'max_holding_bars': self.max_holding_bars,
            'use_indicator_filter': use_indicator_filter,
            'benchmark': 'Buy and Hold (Stock Close Start to End)'
        }

        return BacktestResult(
            ticker=ticker,
            initial_capital=self.initial_capital,
            final_capital=capital,
            total_net_pnl=total_net_pnl,
            total_return_pct=total_return_pct,
            benchmark_return_pct=benchmark_return_pct,
            excess_return_pct=excess_return,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate_pct=win_rate,
            profit_factor=profit_factor,
            max_drawdown_pct=max_drawdown,
            avg_trade_pnl=avg_trade,
            avg_win_pnl=avg_win,
            avg_loss_pnl=avg_loss,
            risk_reward_ratio=rr_ratio,
            total_fees_paid=total_fees,
            trades=trades,
            equity_curve=equity_curve,
            assumptions=assumptions
        )
