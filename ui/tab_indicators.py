"""
ui/tab_indicators.py — Advanced Technical Indicator Engine Tab
Displays real-time and historical technical indicators (RSI, MACD, Bollinger Bands, EMA, ATR)
with interactive stock selection and visual status gauges.
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import ttk
import threading
from typing import Optional

from core.indicator_engine import TechnicalIndicatorEngine
from config.settings import ACTIVE_NSE_TICKERS


class IndicatorsTab:
    """
    UI Tab component for Technical Indicators analysis.
    """

    def __init__(self, parent_tab, app_ref):
        self.tab = parent_tab
        self.app = app_ref
        self.engine = TechnicalIndicatorEngine()
        self.current_ticker = ACTIVE_NSE_TICKERS[0]
        self._build_ui()

    def _build_ui(self):
        # 1. Control Bar
        ctrl_frame = ctk.CTkFrame(self.tab, fg_color="#11111B", corner_radius=6)
        ctrl_frame.pack(fill="x", padx=10, pady=6)

        lbl = ctk.CTkLabel(
            ctrl_frame,
            text="📐 Technical Indicator Engine: RSI (14) | MACD (12,26,9) | Bollinger Bands (20,2) | EMA (9,21,50,200) | ATR (14)",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#89B4FA"
        )
        lbl.pack(side="left", padx=12, pady=8)

        # Stock dropdown
        tickers = [t.replace(".NS", "") for t in ACTIVE_NSE_TICKERS]
        self.combo_ticker = ctk.CTkComboBox(
            ctrl_frame,
            values=tickers,
            width=130,
            command=self._on_ticker_selected
        )
        self.combo_ticker.set(tickers[0])
        self.combo_ticker.pack(side="right", padx=(5, 12), pady=8)

        btn_analyze = ctk.CTkButton(
            ctrl_frame,
            text="🔄 Analyze Stock",
            fg_color="#89B4FA",
            text_color="#11111B",
            font=ctk.CTkFont(weight="bold"),
            width=110,
            command=self._analyze_current_stock
        )
        btn_analyze.pack(side="right", padx=5, pady=8)

        # 2. Verdict Banner
        self.verdict_frame = ctk.CTkFrame(self.tab, fg_color="#1E1E2E", corner_radius=6)
        self.verdict_frame.pack(fill="x", padx=10, pady=(2, 6))

        self.lbl_verdict = ctk.CTkLabel(
            self.verdict_frame,
            text="COMPOSITE TECHNICAL VERDICT: ANALYZING...",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#A6E3A1"
        )
        self.lbl_verdict.pack(side="left", padx=15, pady=8)

        self.lbl_price = ctk.CTkLabel(
            self.verdict_frame,
            text="LTP: ₹ --",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#F5E0DC"
        )
        self.lbl_price.pack(side="right", padx=15, pady=8)

        # 3. Indicator Metric Cards
        cards_frame = ctk.CTkFrame(self.tab, fg_color="transparent")
        cards_frame.pack(fill="x", padx=10, pady=4)

        self.card_rsi = self._create_card(cards_frame, "RSI (14)", "--", "#A6E3A1", "Neutral (50.0)")
        self.card_rsi.pack(side="left", fill="x", expand=True, padx=4)

        self.card_macd = self._create_card(cards_frame, "MACD (12,26,9)", "--", "#89B4FA", "Histogram: --")
        self.card_macd.pack(side="left", fill="x", expand=True, padx=4)

        self.card_bb = self._create_card(cards_frame, "Bollinger %B", "--", "#F9E2AF", "Bandwidth: --")
        self.card_bb.pack(side="left", fill="x", expand=True, padx=4)

        self.card_ema = self._create_card(cards_frame, "EMA Trend (9/21)", "--", "#CBA6F7", "Trend: --")
        self.card_ema.pack(side="left", fill="x", expand=True, padx=4)

        self.card_atr = self._create_card(cards_frame, "ATR (14)", "--", "#F38BA8", "Vol %: --")
        self.card_atr.pack(side="left", fill="x", expand=True, padx=4)

        # 4. Details Split View: Table on Left, Visual Chart on Right
        detail_box = ctk.CTkFrame(self.tab, fg_color="#1E1E2E", corner_radius=6)
        detail_box.pack(fill="both", expand=True, padx=10, pady=6)

        left_frame = ctk.CTkFrame(detail_box, fg_color="#181825", corner_radius=6)
        left_frame.pack(side="left", fill="both", expand=True, padx=6, pady=6)

        lbl_tbl = ctk.CTkLabel(
            left_frame,
            text="📊 Chronological Indicator Values (Last 10 Candles)",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#CDD6F4"
        )
        lbl_tbl.pack(anchor="w", padx=10, pady=(6, 2))

        cols = ("time", "close", "rsi", "macd", "macd_hist", "bb_pct_b", "ema_9", "ema_21", "atr")
        self.tree = ttk.Treeview(left_frame, columns=cols, show="headings", height=8)
        headers = {
            "time": "Time",
            "close": "Close (₹)",
            "rsi": "RSI",
            "macd": "MACD",
            "macd_hist": "Hist",
            "bb_pct_b": "%B",
            "ema_9": "9 EMA",
            "ema_21": "21 EMA",
            "atr": "ATR"
        }
        for c, h in headers.items():
            self.tree.heading(c, text=h)
            self.tree.column(c, anchor="center", width=75)

        sb = ttk.Scrollbar(left_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=6)
        sb.pack(side="right", fill="y", padx=(0, 6), pady=6)

        # Visual Canvas on Right
        right_frame = ctk.CTkFrame(detail_box, fg_color="#181825", corner_radius=6, width=380)
        right_frame.pack(side="right", fill="both", expand=False, padx=6, pady=6)

        lbl_chart = ctk.CTkLabel(
            right_frame,
            text="📈 Price vs Bollinger Bands & EMAs",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#CDD6F4"
        )
        lbl_chart.pack(anchor="w", padx=10, pady=(6, 2))

        self.canvas = tk.Canvas(right_frame, bg="#11111B", highlightthickness=0, width=360, height=200)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=8)

        # Trigger initial analysis
        self._analyze_current_stock()

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

    def _on_ticker_selected(self, choice):
        self.current_ticker = f"{choice}.NS" if not choice.endswith(".NS") else choice
        self._analyze_current_stock()

    def _analyze_current_stock(self):
        ticker = self.current_ticker
        threading.Thread(target=self._run_analysis_bg, args=(ticker,), daemon=True).start()

    def _run_analysis_bg(self, ticker: str):
        res = self.engine.analyze_stock(ticker)
        try:
            self.tab.after(0, self._render_analysis, res)
        except Exception:
            pass

    def _render_analysis(self, res: dict):
        if res.get('status') != 'OK':
            self.lbl_verdict.configure(text=f"NO DATA FOR {res.get('ticker')}")
            return

        summary = res['summary']
        df = res['df']

        close_p = summary.get('close', 0.0)
        verdict = summary.get('composite_verdict', 'NEUTRAL')
        self.lbl_price.configure(text=f"LTP: ₹{close_p:.2f}")

        # Verdict styling
        if "STRONG BUY" in verdict:
            v_color = "#A6E3A1"
        elif "BUY" in verdict:
            v_color = "#89B4FA"
        elif "STRONG SELL" in verdict or "SELL" in verdict:
            v_color = "#F38BA8"
        else:
            v_color = "#F9E2AF"

        self.lbl_verdict.configure(text=f"COMPOSITE VERDICT ({res['ticker'].replace('.NS','')}): {verdict}", text_color=v_color)

        # 1. RSI
        rsi_info = summary.get('rsi', {})
        self.card_rsi.v_lbl.configure(text=f"{rsi_info.get('value', 50.0):.1f}")
        self.card_rsi.s_lbl.configure(text=rsi_info.get('description', '')[:25])

        # 2. MACD
        macd_info = summary.get('macd', {})
        self.card_macd.v_lbl.configure(text=f"{macd_info.get('macd', 0.0):.2f}")
        self.card_macd.s_lbl.configure(text=f"Hist: {macd_info.get('hist', 0.0):.2f} | {macd_info.get('bias', '')}")

        # 3. Bollinger
        bb_info = summary.get('bollinger', {})
        self.card_bb.v_lbl.configure(text=f"{bb_info.get('pct_b', 0.5):.2f}")
        self.card_bb.s_lbl.configure(text=f"Bandwidth: {bb_info.get('bandwidth', 0.0):.3f}")

        # 4. EMA
        ema_info = summary.get('ema', {})
        self.card_ema.v_lbl.configure(text=f"₹{ema_info.get('ema_9', close_p):.1f}")
        self.card_ema.s_lbl.configure(text=ema_info.get('trend', '')[:25])

        # 5. ATR
        atr_info = summary.get('atr', {})
        self.card_atr.v_lbl.configure(text=f"₹{atr_info.get('value', 0.0):.2f}")
        self.card_atr.s_lbl.configure(text=f"{atr_info.get('natr_pct', 0.0):.2f}% of price")

        # Update Treeview with last 10 candles
        for row in self.tree.get_children():
            self.tree.delete(row)

        tail_df = df.tail(10)
        for _, row in tail_df.iterrows():
            ts = str(row.get('Datetime', ''))[-8:]
            c = round(float(row.get('Close', 0.0)), 2)
            rsi = round(float(row.get('RSI_14', 50.0)), 1)
            macd = round(float(row.get('MACD', 0.0)), 2)
            hist = round(float(row.get('MACD_Hist', 0.0)), 2)
            pct_b = round(float(row.get('BB_Pct_B', 0.5)), 2)
            e9 = round(float(row.get('EMA_9', c)), 1)
            e21 = round(float(row.get('EMA_21', c)), 1)
            atr = round(float(row.get('ATR_14', 0.0)), 2)

            self.tree.insert("", "end", values=(ts, c, rsi, macd, hist, pct_b, e9, e21, atr))

        # Draw chart
        self._draw_chart(tail_df)

    def _draw_chart(self, df):
        self.canvas.delete("all")
        w = self.canvas.winfo_width() or 360
        h = self.canvas.winfo_height() or 200

        if len(df) < 5:
            return

        prices = df['Close'].values
        bb_upper = df.get('BB_Upper', df['Close']).values
        bb_lower = df.get('BB_Lower', df['Close']).values
        ema_9 = df.get('EMA_9', df['Close']).values

        min_val = min(np.min(bb_lower), np.min(prices)) * 0.998
        max_val = max(np.max(bb_upper), np.max(prices)) * 1.002
        val_range = max(1e-4, max_val - min_val)

        n = len(prices)
        x_step = (w - 40) / max(1, n - 1)

        # Draw Bollinger Band Shade
        upper_pts = []
        lower_pts = []
        for i in range(n):
            x = 20 + (i * x_step)
            y_u = h - 20 - ((bb_upper[i] - min_val) / val_range * (h - 40))
            y_l = h - 20 - ((bb_lower[i] - min_val) / val_range * (h - 40))
            upper_pts.append((x, y_u))
            lower_pts.append((x, y_l))

        # Draw upper & lower band lines
        for i in range(n - 1):
            self.canvas.create_line(upper_pts[i][0], upper_pts[i][1], upper_pts[i+1][0], upper_pts[i+1][1], fill="#45475A", dash=(2, 2))
            self.canvas.create_line(lower_pts[i][0], lower_pts[i][1], lower_pts[i+1][0], lower_pts[i+1][1], fill="#45475A", dash=(2, 2))

        # Draw 9 EMA line
        for i in range(n - 1):
            x1 = 20 + (i * x_step)
            y1 = h - 20 - ((ema_9[i] - min_val) / val_range * (h - 40))
            x2 = 20 + ((i + 1) * x_step)
            y2 = h - 20 - ((ema_9[i+1] - min_val) / val_range * (h - 40))
            self.canvas.create_line(x1, y1, x2, y2, fill="#CBA6F7", width=1)

        # Draw Price line
        for i in range(n - 1):
            x1 = 20 + (i * x_step)
            y1 = h - 20 - ((prices[i] - min_val) / val_range * (h - 40))
            x2 = 20 + ((i + 1) * x_step)
            y2 = h - 20 - ((prices[i+1] - min_val) / val_range * (h - 40))
            color = "#A6E3A1" if prices[i+1] >= prices[i] else "#F38BA8"
            self.canvas.create_line(x1, y1, x2, y2, fill=color, width=2)
            self.canvas.create_oval(x2 - 2, y2 - 2, x2 + 2, y2 + 2, fill=color, outline="")

        # Legend
        self.canvas.create_text(25, 15, text="● Price", fill="#A6E3A1", font=("Segoe UI", 9, "bold"), anchor="w")
        self.canvas.create_text(85, 15, text="— 9 EMA", fill="#CBA6F7", font=("Segoe UI", 9), anchor="w")
        self.canvas.create_text(150, 15, text="-- BBands", fill="#6C7086", font=("Segoe UI", 9), anchor="w")
