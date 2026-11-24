"""
ui/tab_risk.py — Portfolio Risk Analytics Dashboard Tab
Visualizes institutional-grade risk metrics: Sharpe Ratio, Sortino Ratio, Calmar Ratio,
Annualized Volatility, Maximum Drawdown, Value at Risk (VaR 95%), and Return Distribution.
"""

import customtkinter as ctk
import tkinter as tk
import threading
from typing import Dict, Any, Optional
import numpy as np

from analytics.risk_engine import RiskAnalyticsEngine, PortfolioRiskReport, RISK_FREE_RATE_ANNUAL


class RiskTab:
    """
    UI Tab component for Portfolio Risk Analytics.
    """

    def __init__(self, parent_tab, app_ref):
        self.tab = parent_tab
        self.app = app_ref
        self.risk_engine = RiskAnalyticsEngine(risk_free_rate=RISK_FREE_RATE_ANNUAL)
        self.last_report: Optional[PortfolioRiskReport] = None
        self._build_ui()

    def _build_ui(self):
        # 1. Top Control Bar & Assumptions
        ctrl_frame = ctk.CTkFrame(self.tab, fg_color="#11111B", corner_radius=6)
        ctrl_frame.pack(fill="x", padx=10, pady=6)

        lbl = ctk.CTkLabel(
            ctrl_frame,
            text="📊 Institutional Portfolio Risk Engine: Sharpe | Sortino | Calmar | Annualized Volatility | 95% VaR & CVaR",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#F9E2AF"
        )
        lbl.pack(side="left", padx=12, pady=8)

        btn_calc = ctk.CTkButton(
            ctrl_frame,
            text="🔄 Compute Risk Analytics",
            fg_color="#F9E2AF",
            text_color="#11111B",
            font=ctk.CTkFont(weight="bold"),
            command=self._compute_risk_metrics
        )
        btn_calc.pack(side="right", padx=12, pady=8)

        # 2. Documented Financial Benchmark Assumptions Banner
        banner = ctk.CTkFrame(self.tab, fg_color="#1E1E2E", corner_radius=6)
        banner.pack(fill="x", padx=10, pady=(2, 6))

        lbl_assump = ctk.CTkLabel(
            banner,
            text=f"📌 Documented Assumptions: Risk-Free Rate = {RISK_FREE_RATE_ANNUAL*100:.1f}% (RBI 91-Day T-Bill Yield) | 252 Trading Days/Year | 95% Confidence VaR/CVaR",
            font=ctk.CTkFont(size=11),
            text_color="#A6ADC8"
        )
        lbl_assump.pack(side="left", padx=15, pady=8)

        self.lbl_ret_summary = ctk.CTkLabel(
            banner,
            text="Total Return: -- % | Ann. Vol: -- %",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#A6E3A1"
        )
        self.lbl_ret_summary.pack(side="right", padx=15, pady=8)

        # 3. KPI Risk Cards (Row 1: Ratios)
        row1_frame = ctk.CTkFrame(self.tab, fg_color="transparent")
        row1_frame.pack(fill="x", padx=10, pady=4)

        self.card_sharpe = self._create_card(row1_frame, "Sharpe Ratio", "--", "#A6E3A1", "Risk-adjusted (Rf=6.5%)")
        self.card_sharpe.pack(side="left", fill="x", expand=True, padx=4)

        self.card_sortino = self._create_card(row1_frame, "Sortino Ratio", "--", "#89B4FA", "Downside risk-adjusted")
        self.card_sortino.pack(side="left", fill="x", expand=True, padx=4)

        self.card_calmar = self._create_card(row1_frame, "Calmar Ratio", "--", "#CBA6F7", "Return / Max Drawdown")
        self.card_calmar.pack(side="left", fill="x", expand=True, padx=4)

        self.card_vol = self._create_card(row1_frame, "Annualized Volatility", "-- %", "#F9E2AF", "Annualized standard dev")
        self.card_vol.pack(side="left", fill="x", expand=True, padx=4)

        # Row 2: Drawdown & Tail Risk
        row2_frame = ctk.CTkFrame(self.tab, fg_color="transparent")
        row2_frame.pack(fill="x", padx=10, pady=4)

        self.card_maxdd = self._create_card(row2_frame, "Maximum Drawdown", "-- %", "#F38BA8", "Worst peak-to-trough drop")
        self.card_maxdd.pack(side="left", fill="x", expand=True, padx=4)

        self.card_var = self._create_card(row2_frame, "Value at Risk (95%)", "-- %", "#F38BA8", "1-day 95% tail threshold")
        self.card_var.pack(side="left", fill="x", expand=True, padx=4)

        self.card_cvar = self._create_card(row2_frame, "Conditional VaR (95%)", "-- %", "#F38BA8", "Expected tail loss")
        self.card_cvar.pack(side="left", fill="x", expand=True, padx=4)

        self.card_skew = self._create_card(row2_frame, "Skew / Kurtosis", "-- / --", "#89B4FA", "Distribution shape")
        self.card_skew.pack(side="left", fill="x", expand=True, padx=4)

        # 4. Charts: Drawdown Curve (Left) & Return Distribution Histogram (Right)
        charts_frame = ctk.CTkFrame(self.tab, fg_color="#1E1E2E", corner_radius=6)
        charts_frame.pack(fill="both", expand=True, padx=10, pady=6)

        left_box = ctk.CTkFrame(charts_frame, fg_color="#181825", corner_radius=6)
        left_box.pack(side="left", fill="both", expand=True, padx=6, pady=6)

        lbl_dd = ctk.CTkLabel(left_box, text="🌊 Underwater Drawdown Curve (% Peak Drop)", font=ctk.CTkFont(size=11, weight="bold"), text_color="#CDD6F4")
        lbl_dd.pack(anchor="w", padx=12, pady=(8, 2))

        self.canvas_dd = tk.Canvas(left_box, bg="#11111B", highlightthickness=0)
        self.canvas_dd.pack(fill="both", expand=True, padx=8, pady=8)

        right_box = ctk.CTkFrame(charts_frame, fg_color="#181825", corner_radius=6)
        right_box.pack(side="right", fill="both", expand=True, padx=6, pady=6)

        lbl_hist = ctk.CTkLabel(right_box, text="📊 Return Distribution Histogram (% Change Frequency)", font=ctk.CTkFont(size=11, weight="bold"), text_color="#CDD6F4")
        lbl_hist.pack(anchor="w", padx=12, pady=(8, 2))

        self.canvas_dist = tk.Canvas(right_box, bg="#11111B", highlightthickness=0)
        self.canvas_dist.pack(fill="both", expand=True, padx=8, pady=8)

        # Initial calculation
        self._compute_risk_metrics()

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

    def _compute_risk_metrics(self):
        # Pull equity curve from paper trader or generate default simulation
        threading.Thread(target=self._run_calc_bg, daemon=True).start()

    def _run_calc_bg(self):
        # Check if paper trader has closed trades
        trades = self.app.paper_trader.closed_trades if hasattr(self.app, 'paper_trader') else []
        if trades:
            equities = [self.app.paper_trader.initial_capital]
            curr = self.app.paper_trader.initial_capital
            for t in trades:
                curr += t.get('pnl', 0.0)
                equities.append(curr)
        else:
            # Generate representative 1-min portfolio progression
            np.random.seed(42)
            n_bars = 180
            ret_sim = np.random.normal(0.0003, 0.002, n_bars)
            equities = [100000.0]
            for r in ret_sim:
                equities.append(equities[-1] * (1.0 + r))

        report = self.risk_engine.analyze_equity_curve(equities, is_intraday_minutes=True)
        try:
            self.tab.after(0, self._render_report, report)
        except Exception:
            pass

    def _render_report(self, report: PortfolioRiskReport):
        self.last_report = report

        self.lbl_ret_summary.configure(
            text=f"Total Net Return: {report.total_return_pct:+.2f}% | Ann. Volatility: {report.annualized_volatility_pct:.1f}%"
        )

        self.card_sharpe.v_lbl.configure(text=f"{report.sharpe_ratio:.2f}")
        self.card_sortino.v_lbl.configure(text=f"{report.sortino_ratio:.2f}")
        self.card_calmar.v_lbl.configure(text=f"{report.calmar_ratio:.2f}")
        self.card_vol.v_lbl.configure(text=f"{report.annualized_volatility_pct:.1f}%")

        self.card_maxdd.v_lbl.configure(text=f"{report.max_drawdown_pct:.2f}%")
        self.card_var.v_lbl.configure(text=f"-{report.var_95_pct:.2f}%")
        self.card_cvar.v_lbl.configure(text=f"-{report.cvar_95_pct:.2f}%")
        self.card_skew.v_lbl.configure(text=f"{report.skewness:.2f} / {report.kurtosis:.1f}")

        # Draw charts
        self._draw_drawdown_chart(report.drawdown_series)
        self._draw_distribution_chart(report.return_distribution)

    def _draw_drawdown_chart(self, dd_series):
        self.canvas_dd.delete("all")
        w = self.canvas_dd.winfo_width() or 380
        h = self.canvas_dd.winfo_height() or 200

        if not dd_series:
            return

        max_dd = max(0.5, float(np.max(dd_series)))
        n = len(dd_series)
        x_step = (w - 40) / max(1, n - 1)

        # Baseline at top (0% drawdown)
        y_zero = 25
        self.canvas_dd.create_line(20, y_zero, w - 20, y_zero, fill="#45475A", dash=(2, 2))
        self.canvas_dd.create_text(25, 12, text="0% Peak", fill="#A6ADC8", font=("Segoe UI", 8), anchor="w")

        # Draw underwater curve
        pts = []
        for i in range(n):
            x = 20 + (i * x_step)
            y = y_zero + (dd_series[i] / max_dd * (h - 50))
            pts.append((x, y))

        for i in range(n - 1):
            self.canvas_dd.create_line(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1], fill="#F38BA8", width=2)

        self.canvas_dd.create_text(w - 25, h - 15, text=f"Max: -{max_dd:.2f}%", fill="#F38BA8", font=("Segoe UI", 9, "bold"), anchor="e")

    def _draw_distribution_chart(self, dist_dict):
        self.canvas_dist.delete("all")
        w = self.canvas_dist.winfo_width() or 380
        h = self.canvas_dist.winfo_height() or 200

        counts = dist_dict.get('counts', [])
        bins = dist_dict.get('bins', [])
        if not counts:
            return

        max_c = max(1, max(counts))
        n = len(counts)
        bar_w = (w - 60) / max(1, n)

        for i in range(n):
            x0 = 30 + (i * bar_w)
            x1 = x0 + bar_w - 2
            bar_h = (counts[i] / max_c) * (h - 50)
            y0 = h - 25 - bar_h
            y1 = h - 25

            color = "#A6E3A1" if i >= n // 2 else "#F38BA8"
            self.canvas_dist.create_rectangle(x0, y0, x1, y1, fill=color, outline="")

        self.canvas_dist.create_text(30, h - 10, text="Negative Returns", fill="#F38BA8", font=("Segoe UI", 8), anchor="w")
        self.canvas_dist.create_text(w - 30, h - 10, text="Positive Returns", fill="#A6E3A1", font=("Segoe UI", 8), anchor="e")
