"""
ui/tab_backtesting.py — Historical Strategy Backtesting Dashboard Tab
Executes historical strategy simulations on 1-min data with realistic slippage,
transaction costs, trade-by-trade audit log, and Buy-and-Hold benchmark comparison.
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import ttk
import threading
from typing import Dict, Any, Optional
import numpy as np

from backtesting.engine import BacktestEngine, BacktestResult
from config.settings import ACTIVE_NSE_TICKERS, INITIAL_PAPER_CAPITAL


class BacktestingTab:
    """
    UI Tab component for Strategy Backtesting.
    """

    def __init__(self, parent_tab, app_ref):
        self.tab = parent_tab
        self.app = app_ref
        self.engine = BacktestEngine()
        self.last_result: Optional[BacktestResult] = None
        self._build_ui()

    def _build_ui(self):
        # 1. Parameter Control Bar
        ctrl_frame = ctk.CTkFrame(self.tab, fg_color="#11111B", corner_radius=6)
        ctrl_frame.pack(fill="x", padx=10, pady=6)

        # Stock picker
        lbl_stock = ctk.CTkLabel(ctrl_frame, text="Stock:", font=ctk.CTkFont(size=11, weight="bold"), text_color="#A6ADC8")
        lbl_stock.pack(side="left", padx=(10, 4), pady=8)

        tickers = [t.replace(".NS", "") for t in ACTIVE_NSE_TICKERS]
        self.combo_ticker = ctk.CTkComboBox(ctrl_frame, values=tickers, width=120)
        self.combo_ticker.set(tickers[0])
        self.combo_ticker.pack(side="left", padx=4, pady=8)

        # Capital
        lbl_cap = ctk.CTkLabel(ctrl_frame, text="Capital (₹):", font=ctk.CTkFont(size=11), text_color="#A6ADC8")
        lbl_cap.pack(side="left", padx=(10, 4), pady=8)
        self.entry_cap = ctk.CTkEntry(ctrl_frame, width=90)
        self.entry_cap.insert(0, str(int(INITIAL_PAPER_CAPITAL)))
        self.entry_cap.pack(side="left", padx=4, pady=8)

        # Target & Stop Loss
        lbl_target = ctk.CTkLabel(ctrl_frame, text="Target %:", font=ctk.CTkFont(size=11), text_color="#A6ADC8")
        lbl_target.pack(side="left", padx=(10, 4), pady=8)
        self.entry_target = ctk.CTkEntry(ctrl_frame, width=50)
        self.entry_target.insert(0, "1.5")
        self.entry_target.pack(side="left", padx=4, pady=8)

        lbl_sl = ctk.CTkLabel(ctrl_frame, text="Stop Loss %:", font=ctk.CTkFont(size=11), text_color="#A6ADC8")
        lbl_sl.pack(side="left", padx=(10, 4), pady=8)
        self.entry_sl = ctk.CTkEntry(ctrl_frame, width=50)
        self.entry_sl.insert(0, "0.8")
        self.entry_sl.pack(side="left", padx=4, pady=8)

        # Slippage & Commission
        lbl_fees = ctk.CTkLabel(ctrl_frame, text="Slippage+Fees: 0.08%", font=ctk.CTkFont(size=10), text_color="#F9E2AF")
        lbl_fees.pack(side="left", padx=10, pady=8)

        # Filter checkbox
        self.chk_filter = ctk.CTkCheckBox(ctrl_frame, text="Indicator Confirmation", font=ctk.CTkFont(size=10), text_color="#CDD6F4")
        self.chk_filter.select()
        self.chk_filter.pack(side="left", padx=10, pady=8)

        # Run Button
        self.btn_run = ctk.CTkButton(
            ctrl_frame,
            text="🚀 Run Backtest",
            fg_color="#89B4FA",
            text_color="#11111B",
            font=ctk.CTkFont(weight="bold"),
            command=self._on_click_run
        )
        self.btn_run.pack(side="right", padx=10, pady=8)

        # 2. Performance Summary Banner
        self.banner = ctk.CTkFrame(self.tab, fg_color="#1E1E2E", corner_radius=6)
        self.banner.pack(fill="x", padx=10, pady=(2, 6))

        self.lbl_summary_title = ctk.CTkLabel(
            self.banner,
            text="BACKTEST AUDIT: INITIALIZING...",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#A6E3A1"
        )
        self.lbl_summary_title.pack(side="left", padx=15, pady=8)

        self.lbl_assumptions = ctk.CTkLabel(
            self.banner,
            text="Assumptions: 10% Size/Trade | 0.05% Slippage | 0.03% Turnover Fees | No Lookahead Bias",
            font=ctk.CTkFont(size=10),
            text_color="#A6ADC8"
        )
        self.lbl_assumptions.pack(side="right", padx=15, pady=8)

        # 3. KPI Cards
        cards_frame = ctk.CTkFrame(self.tab, fg_color="transparent")
        cards_frame.pack(fill="x", padx=10, pady=4)

        self.card_ret = self._create_card(cards_frame, "Strategy Return", "-- %", "#A6E3A1", "Net of all costs")
        self.card_ret.pack(side="left", fill="x", expand=True, padx=4)

        self.card_bench = self._create_card(cards_frame, "Buy & Hold Benchmark", "-- %", "#89B4FA", "Underlying stock return")
        self.card_bench.pack(side="left", fill="x", expand=True, padx=4)

        self.card_win = self._create_card(cards_frame, "Win Rate", "-- %", "#F9E2AF", "Profitable trades")
        self.card_win.pack(side="left", fill="x", expand=True, padx=4)

        self.card_pf = self._create_card(cards_frame, "Profit Factor", "--", "#CBA6F7", "Gross Profit / Loss")
        self.card_pf.pack(side="left", fill="x", expand=True, padx=4)

        self.card_dd = self._create_card(cards_frame, "Max Drawdown", "-- %", "#F38BA8", "Peak-to-trough drop")
        self.card_dd.pack(side="left", fill="x", expand=True, padx=4)

        self.card_pnl = self._create_card(cards_frame, "Net Profit (₹)", "--", "#A6E3A1", "Starting: ₹1,00,000")
        self.card_pnl.pack(side="left", fill="x", expand=True, padx=4)

        # 4. Split View: Trade Audit Table (Left) & Equity Curve (Right)
        split_frame = ctk.CTkFrame(self.tab, fg_color="#1E1E2E", corner_radius=6)
        split_frame.pack(fill="both", expand=True, padx=10, pady=6)

        left_frame = ctk.CTkFrame(split_frame, fg_color="#181825", corner_radius=6)
        left_frame.pack(side="left", fill="both", expand=True, padx=6, pady=6)

        lbl_tbl = ctk.CTkLabel(
            left_frame,
            text="📑 Simulated Trade Execution History (Chronological Order)",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#CDD6F4"
        )
        lbl_tbl.pack(anchor="w", padx=10, pady=(6, 2))

        cols = ("id", "type", "entry_t", "exit_t", "entry_p", "exit_p", "pnl", "ret", "reason")
        self.tree = ttk.Treeview(left_frame, columns=cols, show="headings", height=8)
        headers = {
            "id": "#",
            "type": "Signal",
            "entry_t": "Entry Time",
            "exit_t": "Exit Time",
            "entry_p": "Entry (₹)",
            "exit_p": "Exit (₹)",
            "pnl": "Net P&L (₹)",
            "ret": "Return %",
            "reason": "Exit Catalyst"
        }
        for c, h in headers.items():
            self.tree.heading(c, text=h)
            self.tree.column(c, anchor="center", width=75 if c in ["id", "type", "ret"] else 90)

        self.tree.tag_configure("win", foreground="#A6E3A1")
        self.tree.tag_configure("loss", foreground="#F38BA8")

        sb = ttk.Scrollbar(left_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=6)
        sb.pack(side="right", fill="y", padx=(0, 6), pady=6)

        # Right: Equity Curve Visual
        right_frame = ctk.CTkFrame(split_frame, fg_color="#181825", corner_radius=6, width=400)
        right_frame.pack(side="right", fill="both", expand=False, padx=6, pady=6)

        lbl_curve = ctk.CTkLabel(
            right_frame,
            text="📈 Strategy Equity Curve vs Starting Capital",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#CDD6F4"
        )
        lbl_curve.pack(anchor="w", padx=10, pady=(6, 2))

        self.canvas = tk.Canvas(right_frame, bg="#11111B", highlightthickness=0, width=380, height=220)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=8)

        # Run default backtest
        self._on_click_run()

    def _create_card(self, parent, title, val, color, subtext=""):
        card = ctk.CTkFrame(parent, fg_color="#1E1E2E", corner_radius=6)
        t_lbl = ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=10, weight="bold"), text_color="#A6ADC8")
        t_lbl.pack(padx=8, pady=(6, 1))

        v_lbl = ctk.CTkLabel(card, text=val, font=ctk.CTkFont(size=16, weight="bold"), text_color=color)
        v_lbl.pack(padx=8, pady=0)

        s_lbl = ctk.CTkLabel(card, text=subtext, font=ctk.CTkFont(size=9), text_color="#6C7086")
        s_lbl.pack(padx=8, pady=(1, 6))

        card.v_lbl = v_lbl
        card.s_lbl = s_lbl
        return card

    def _on_click_run(self):
        ticker_val = self.combo_ticker.get()
        full_ticker = f"{ticker_val}.NS" if not ticker_val.endswith(".NS") else ticker_val
        try:
            cap = float(self.entry_cap.get())
            tp = float(self.entry_target.get())
            sl = float(self.entry_sl.get())
        except ValueError:
            cap, tp, sl = INITIAL_PAPER_CAPITAL, 1.5, 0.8

        use_filter = bool(self.chk_filter.get())
        self.btn_run.configure(text="⏳ Backtesting...", state="disabled")

        threading.Thread(
            target=self._run_backtest_bg,
            args=(full_ticker, cap, tp, sl, use_filter),
            daemon=True
        ).start()

    def _run_backtest_bg(self, ticker, capital, tp, sl, use_filter):
        eng = BacktestEngine(
            initial_capital=capital,
            take_profit_pct=tp,
            stop_loss_pct=sl,
            slippage_pct=0.05,
            brokerage_pct=0.03
        )
        res = eng.run(None, ticker=ticker, use_indicator_filter=use_filter)
        try:
            self.tab.after(0, self._render_results, res)
        except Exception:
            pass

    def _render_results(self, res: BacktestResult):
        self.btn_run.configure(text="🚀 Run Backtest", state="normal")
        self.last_result = res

        # Banner
        ret_color = "#A6E3A1" if res.total_net_pnl >= 0 else "#F38BA8"
        self.lbl_summary_title.configure(
            text=f"BACKTEST RESULTS ({res.ticker.replace('.NS','')}) — TOTAL NET RETURN: {res.total_return_pct:+.2f}% | NET P&L: ₹{res.total_net_pnl:+,.2f}",
            text_color=ret_color
        )

        # Cards
        self.card_ret.v_lbl.configure(text=f"{res.total_return_pct:+.1f}%")
        self.card_bench.v_lbl.configure(text=f"{res.benchmark_return_pct:+.1f}%")
        self.card_win.v_lbl.configure(text=f"{res.win_rate_pct:.1f}%")
        self.card_pf.v_lbl.configure(text=f"{res.profit_factor:.2f}")
        self.card_dd.v_lbl.configure(text=f"{res.max_drawdown_pct:.1f}%")
        self.card_pnl.v_lbl.configure(text=f"₹{res.total_net_pnl:+,.0f}")

        # Trades table
        for r in self.tree.get_children():
            self.tree.delete(r)

        for t in res.trades:
            tag = "win" if t.net_pnl > 0 else "loss"
            e_time = str(t.entry_time)[-8:] if t.entry_time else "--"
            x_time = str(t.exit_time)[-8:] if t.exit_time else "--"
            self.tree.insert("", "end", values=(
                t.trade_id,
                t.direction,
                e_time,
                x_time,
                f"₹{t.entry_price:.2f}",
                f"₹{t.exit_price:.2f}",
                f"₹{t.net_pnl:+.2f}",
                f"{t.return_pct:+.2f}%",
                t.exit_reason
            ), tags=(tag,))

        # Draw equity curve
        self._draw_equity_curve(res.equity_curve, res.initial_capital)

    def _draw_equity_curve(self, eq_data, initial_cap):
        self.canvas.delete("all")
        w = self.canvas.winfo_width() or 380
        h = self.canvas.winfo_height() or 220

        if not eq_data or len(eq_data) < 2:
            return

        equities = [p['equity'] for p in eq_data]
        min_e = min(np.min(equities), initial_cap) * 0.995
        max_e = max(np.max(equities), initial_cap) * 1.005
        e_range = max(1.0, max_e - min_e)

        n = len(equities)
        x_step = (w - 40) / max(1, n - 1)

        # Baseline horizontal line (Initial Capital)
        y_base = h - 20 - ((initial_cap - min_e) / e_range * (h - 40))
        self.canvas.create_line(20, y_base, w - 20, y_base, fill="#45475A", dash=(4, 4))
        self.canvas.create_text(25, y_base - 8, text=f"Start: ₹{initial_cap:,.0f}", fill="#6C7086", font=("Segoe UI", 8), anchor="w")

        # Strategy equity curve line
        for i in range(n - 1):
            x1 = 20 + (i * x_step)
            y1 = h - 20 - ((equities[i] - min_e) / e_range * (h - 40))
            x2 = 20 + ((i + 1) * x_step)
            y2 = h - 20 - ((equities[i+1] - min_e) / e_range * (h - 40))

            color = "#A6E3A1" if equities[-1] >= initial_cap else "#F38BA8"
            self.canvas.create_line(x1, y1, x2, y2, fill=color, width=2)

        # Final equity annotation
        final_eq = equities[-1]
        y_final = h - 20 - ((final_eq - min_e) / e_range * (h - 40))
        self.canvas.create_oval(w - 23, y_final - 3, w - 17, y_final + 3, fill="#89B4FA", outline="")
        self.canvas.create_text(w - 25, y_final - 10, text=f"₹{final_eq:,.0f}", fill="#89B4FA", font=("Segoe UI", 8, "bold"), anchor="e")
