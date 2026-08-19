"""
ui/main_window.py — Main CustomTkinter GUI with all 8 new features integrated.

Tabs:
  1. 📈 Live 1-Min Screener    — enhanced with Ghost, Score, Sparkline columns
  2. 🤖 Day 1 ML Training       — unchanged from base
  3. 📊 Live Paper Trading      — enhanced stats + summary line
  4. 🗺️ Sector Heatmap          — NEW Feature 5
  5. 📄 Reports & Alerts        — NEW Features 3, 7

New columns in screener table:
  - Sparkline (Feature 6) — mini price chart
  - Ghost (Feature 1)     — CLEAN / GHOST / LOW_CONVICTION
  - Score (Feature 2)     — 0-100 signal quality
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import ttk
import threading
import time
from datetime import datetime

# Existing core imports
from config.settings import load_credentials, get_masked_credentials, LTP_MIN, LTP_MAX, BID_QTY_MIN, ASK_QTY_MIN
from core.broker_streamer import BrokerStreamer
from core.paper_trader import PaperTrader
from ml.crossover_evaluator import AICrossoverEvaluator

# ── New feature imports ──────────────────────────────────────────────────────
from ml.ghost_detector import detect_ghost, ghost_label_with_emoji
from ml.feature_builder import build_ghost_signal_data
from screener.signal_scorer import score_signal
from alerts.telegram_bot import TelegramAlerter
from ui.sparklines import SparklineStore
from ui.sector_heatmap import SectorHeatmap
from reports.pdf_generator import ReportGenerator

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class StockScreenerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AI/ML 1-Min Stock Screener — Day 1 Model Training & Live Paper Trading Engine")
        self.geometry("1500x950")

        self.credentials = load_credentials()
        self.ai_evaluator = AICrossoverEvaluator()
        self.paper_trader = PaperTrader()

        # ── New feature instances ────────────────────────────────────────────
        self.telegram_alerter = TelegramAlerter()
        self.sparkline_store  = SparklineStore(max_history=20)
        self.report_generator = ReportGenerator()
        # sector_heatmap is built after tab creation

        # Session data for report generation
        self.session_start_time = datetime.now().strftime("%H:%M:%S")
        self.session_crossovers: list = []  # all crossovers this session

        self.latest_data     = {}
        self.crossover_states = {}  # Tracks previous SMMA(20) vs SMMA(120) relation
        self._pending_ml_summary = None
        self.smma_20_history = {}   # {symbol: deque of recent smma_20 values}

        self._build_header()
        self._build_tabview()
        self._build_statusbar()

        # Handle window close — auto-generate PDF
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Train initial Day 1 ML Model in background thread
        threading.Thread(target=self._auto_train_ml_model, daemon=True).start()

        # Start Market Feed Streamer
        self.streamer = BrokerStreamer(self.credentials, self.on_tick_update)
        self.streamer.start()

        # UI Auto-refresh timer
        self.after(500, self._periodic_ui_refresh)

        # Sector heatmap refresh (every 30 seconds)
        self.after(30_000, self._refresh_heatmap_periodic)

    # ─────────────────────────────────────────────────────────────────────────
    # Header
    # ─────────────────────────────────────────────────────────────────────────

    def _build_header(self):
        header_frame = ctk.CTkFrame(self, fg_color="#1E1E2E", corner_radius=8)
        header_frame.pack(fill="x", padx=15, pady=(12, 4))

        title_lbl = ctk.CTkLabel(
            header_frame,
            text="📈 AI/ML Stock Market Quantitative Screener & Paper Trading Engine",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#F5E0DC"
        )
        title_lbl.pack(side="left", padx=15, pady=10)

        masked_creds = get_masked_credentials(self.credentials)
        cred_text = f"Timeframe: 1-Min | SMMA: (20/120) | Broker: {masked_creds['broker']} | Features: Ghost+Score+Alerts+PDF"

        mode_lbl = ctk.CTkLabel(
            header_frame,
            text=cred_text,
            font=ctk.CTkFont(size=11),
            text_color="#A6ADC8"
        )
        mode_lbl.pack(side="right", padx=15, pady=10)

    # ─────────────────────────────────────────────────────────────────────────
    # Tab View
    # ─────────────────────────────────────────────────────────────────────────

    def _build_tabview(self):
        self.tabview = ctk.CTkTabview(self, fg_color="#181825", segmented_button_selected_color="#89B4FA")
        self.tabview.pack(fill="both", expand=True, padx=15, pady=5)

        self.tab_screener    = self.tabview.add("📈 Live 1-Min Screener")
        self.tab_training    = self.tabview.add("🤖 Day 1 ML Training & Features")
        self.tab_papertrade  = self.tabview.add("📊 Live Paper Trading & Win/Loss Stats")
        self.tab_heatmap     = self.tabview.add("🗺️ Sector Heatmap")
        self.tab_reports     = self.tabview.add("📄 Reports & Alerts")

        self._build_screener_tab()
        self._build_training_tab()
        self._build_papertrade_tab()
        self._build_heatmap_tab()
        self._build_reports_tab()

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 1: Screener (enhanced with Ghost, Score, Sparkline)
    # ─────────────────────────────────────────────────────────────────────────

    def _build_screener_tab(self):
        filter_bar = ctk.CTkFrame(self.tab_screener, fg_color="#11111B", corner_radius=6)
        filter_bar.pack(fill="x", padx=10, pady=5)

        lbl = ctk.CTkLabel(
            filter_bar,
            text="🎯 1-Min Criteria: LTP ₹30–₹500 | Liquidity >10L | SMMA 20/120 | Ghost Detect | Signal Score | Sparkline",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#89B4FA"
        )
        lbl.pack(padx=10, pady=6, side="left")

        table_container = ctk.CTkFrame(self.tab_screener, fg_color="#1E1E2E", corner_radius=6)
        table_container.pack(fill="both", expand=True, padx=10, pady=5)

        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview", background="#181825", foreground="#CDD6F4", rowheight=30,
                         fieldbackground="#181825", font=("Segoe UI", 10))
        style.configure("Treeview.Heading", background="#313244", foreground="#F5E0DC",
                         font=("Segoe UI", 10, "bold"))
        style.map("Treeview", background=[("selected", "#45475A")])

        # Enhanced columns — added sparkline, ghost_status, score
        columns = (
            "symbol", "ltp", "sparkline", "bid_depth", "ask_depth",
            "smma_20", "smma_120", "vol_5m", "ltq_ratio",
            "ghost_status", "score", "crossover"
        )
        self.tree = ttk.Treeview(table_container, columns=columns, show="headings", selectmode="browse")

        headers = {
            "symbol":      "Symbol",
            "ltp":         "LTP (₹)",
            "sparkline":   "Price 20m",
            "bid_depth":   "Bid Qty (10L+)",
            "ask_depth":   "Ask Qty (10L+)",
            "smma_20":     "20 SMMA",
            "smma_120":    "120 SMMA",
            "vol_5m":      "Vol (5m)",
            "ltq_ratio":   "LTQ Ratio",
            "ghost_status": "Ghost",
            "score":       "Score",
            "crossover":   "20/120 SMMA Crossover"
        }
        col_widths = {
            "symbol": 100, "ltp": 85, "sparkline": 130, "bid_depth": 100,
            "ask_depth": 100, "smma_20": 80, "smma_120": 80, "vol_5m": 80,
            "ltq_ratio": 80, "ghost_status": 100, "score": 100, "crossover": 160
        }
        for col, text in headers.items():
            self.tree.heading(col, text=text)
            self.tree.column(col, anchor="center", width=col_widths.get(col, 90))

        # Color tags for ghost status
        self.tree.tag_configure("ghost", foreground="#F38BA8")        # red for ghost
        self.tree.tag_configure("low_conv", foreground="#F9E2AF")     # yellow for low conviction
        self.tree.tag_configure("clean", foreground="#A6E3A1")        # green for clean

        sb = ttk.Scrollbar(table_container, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        sb.pack(side="right", fill="y", pady=5)
        self.tree.bind("<<TreeviewSelect>>", self._on_row_select)

        # AI Analysis Panel
        panel = ctk.CTkFrame(self.tab_screener, fg_color="#11111B", corner_radius=6, height=160)
        panel.pack(fill="x", padx=10, pady=(5, 10))

        content_box = ctk.CTkFrame(panel, fg_color="transparent")
        content_box.pack(fill="both", expand=True, padx=10, pady=6)

        self.ai_output_box = ctk.CTkTextbox(
            content_box, height=130, fg_color="#181825", text_color="#F5E0DC",
            font=ctk.CTkFont(family="Consolas", size=10), wrap="word"
        )
        self.ai_output_box.pack(side="left", fill="both", expand=True, padx=(0, 10))
        self.ai_output_box.insert("1.0", "Awaiting live 20/120 SMMA crossovers...")
        self.ai_output_box.configure(state="disabled")

        chart_frame = ctk.CTkFrame(content_box, fg_color="#181825", corner_radius=6, width=300)
        chart_frame.pack(side="right", fill="y")
        chart_frame.pack_propagate(False)

        chart_lbl = ctk.CTkLabel(chart_frame, text="📊 Live Order Book Imbalance",
                                  font=ctk.CTkFont(size=10, weight="bold"), text_color="#89B4FA")
        chart_lbl.pack(pady=(4, 2))

        self.canvas = tk.Canvas(chart_frame, bg="#181825", highlightthickness=0, height=95, width=280)
        self.canvas.pack(padx=10, pady=2)
        self._draw_placeholder_chart()

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 2: ML Training (unchanged from base project)
    # ─────────────────────────────────────────────────────────────────────────

    def _build_training_tab(self):
        ctrl_frame = ctk.CTkFrame(self.tab_training, fg_color="#11111B", corner_radius=6)
        ctrl_frame.pack(fill="x", padx=10, pady=10)

        lbl = ctk.CTkLabel(ctrl_frame, text="🤖 Day 1 ML Model Training Pipeline (1-Min Crossovers & Feature Permutations)",
                            font=ctk.CTkFont(size=12, weight="bold"), text_color="#A6E3A1")
        lbl.pack(side="left", padx=15, pady=10)

        btn_train = ctk.CTkButton(ctrl_frame, text="⚡ Retrain Day 1 ML Model",
                                   fg_color="#A6E3A1", text_color="#11111B",
                                   font=ctk.CTkFont(weight="bold"), command=self._on_click_retrain)
        btn_train.pack(side="right", padx=15, pady=10)

        cards_frame = ctk.CTkFrame(self.tab_training, fg_color="transparent")
        cards_frame.pack(fill="x", padx=10, pady=5)

        self.card_crossovers = self._create_card(cards_frame, "Day 1 Crossovers", "0", "#89B4FA")
        self.card_crossovers.pack(side="left", fill="x", expand=True, padx=5)

        self.card_accuracy = self._create_card(cards_frame, "Day 1 Model Accuracy", "-- %", "#A6E3A1")
        self.card_accuracy.pack(side="left", fill="x", expand=True, padx=5)

        self.card_precision = self._create_card(cards_frame, "Precision Score", "-- %", "#F9E2AF")
        self.card_precision.pack(side="left", fill="x", expand=True, padx=5)

        chart_box = ctk.CTkFrame(self.tab_training, fg_color="#11111B", corner_radius=6)
        chart_box.pack(fill="both", expand=True, padx=10, pady=10)

        lbl_feat = ctk.CTkLabel(chart_box, text="📊 Day 1 Feature Permutation Importance Scores (LTQ, Depth Imbalance, SMMA Spread)",
                                 font=ctk.CTkFont(size=11, weight="bold"), text_color="#89B4FA")
        lbl_feat.pack(anchor="w", padx=15, pady=10)

        self.feat_canvas = tk.Canvas(chart_box, bg="#11111B", highlightthickness=0)
        self.feat_canvas.pack(fill="both", expand=True, padx=15, pady=(0, 15))

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 3: Paper Trading (enhanced with summary line)
    # ─────────────────────────────────────────────────────────────────────────

    def _build_papertrade_tab(self):
        # Summary line — new Feature 4 addition
        self.paper_summary_lbl = ctk.CTkLabel(
            self.tab_papertrade,
            text="📋 Paper Trades: 0 closed | Win Rate: 0.0% | P&L: ₹0.00",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#89B4FA"
        )
        self.paper_summary_lbl.pack(fill="x", padx=15, pady=(8, 4))

        # Stats cards row
        stats_frame = ctk.CTkFrame(self.tab_papertrade, fg_color="transparent")
        stats_frame.pack(fill="x", padx=10, pady=5)

        self.card_win_rate = self._create_card(stats_frame, "Successful Trades % (Win Rate)", "0.0%", "#A6E3A1")
        self.card_win_rate.pack(side="left", fill="x", expand=True, padx=4)

        self.card_loss_rate = self._create_card(stats_frame, "Unsuccessful Trades % (Loss Rate)", "0.0%", "#F38BA8")
        self.card_loss_rate.pack(side="left", fill="x", expand=True, padx=4)

        self.card_avoided_pct = self._create_card(stats_frame, "Avoided Trades % (Loss Prevented)", "0.0%", "#F9E2AF")
        self.card_avoided_pct.pack(side="left", fill="x", expand=True, padx=4)

        self.card_pnl = self._create_card(stats_frame, "Net Paper P&L (₹)", "₹0.00", "#89B4FA")
        self.card_pnl.pack(side="left", fill="x", expand=True, padx=4)

        # Trade log table
        log_frame = ctk.CTkFrame(self.tab_papertrade, fg_color="#11111B", corner_radius=6)
        log_frame.pack(fill="both", expand=True, padx=10, pady=10)

        lbl_log = ctk.CTkLabel(log_frame, text="📜 Live Paper Trade Executions & Avoided Crossovers Log",
                                font=ctk.CTkFont(size=11, weight="bold"), text_color="#CDD6F4")
        lbl_log.pack(anchor="w", padx=15, pady=8)

        cols = ("time", "symbol", "signal", "entry_p", "exit_p", "pnl", "ghost", "score", "status", "reason")
        self.trade_tree = ttk.Treeview(log_frame, columns=cols, show="headings", selectmode="browse")

        headers = {
            "time": "Time", "symbol": "Symbol", "signal": "Signal",
            "entry_p": "Entry (₹)", "exit_p": "Exit (₹)", "pnl": "P&L (₹)",
            "ghost": "Ghost", "score": "Score",
            "status": "Trade Status", "reason": "Reason"
        }
        for c, text in headers.items():
            self.trade_tree.heading(c, text=text)
            self.trade_tree.column(c, anchor="center", width=80)

        self.trade_tree.column("reason", width=320, anchor="w")
        self.trade_tree.column("status", width=130)
        self.trade_tree.column("symbol", width=90)

        sb = ttk.Scrollbar(log_frame, orient="vertical", command=self.trade_tree.yview)
        self.trade_tree.configure(yscrollcommand=sb.set)
        self.trade_tree.pack(side="left", fill="both", expand=True, padx=10, pady=(0, 10))
        sb.pack(side="right", fill="y", pady=(0, 10))

        # Seed demo trades
        self.paper_trader.seed_initial_paper_trades(self.ai_evaluator)
        for trade in self.paper_trader.closed_trades:
            self._add_trade_to_table(trade)
        for avoid in self.paper_trader.avoided_trades:
            self._add_trade_to_table({
                'entry_time':   avoid['timestamp'],
                'symbol':       avoid['symbol'],
                'signal_type':  avoid['signal_type'],
                'entry_price':  avoid['ltp'],
                'exit_price':   '-',
                'pnl':          0.0,
                'ghost_status': 'N/A',
                'score':        avoid.get('probability', 0),
                'status':       'TRADE AVOIDED',
                'reason':       avoid['reason']
            })
        self._refresh_paper_trade_stats()

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 4: Sector Heatmap (Feature 5)
    # ─────────────────────────────────────────────────────────────────────────

    def _build_heatmap_tab(self):
        header = ctk.CTkFrame(self.tab_heatmap, fg_color="#11111B", corner_radius=6)
        header.pack(fill="x", padx=10, pady=(10, 5))

        ctk.CTkLabel(
            header,
            text="🗺️ Sector Activity Heatmap — Live Crossover Signals by Sector (refreshes every 30s)",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#A6E3A1"
        ).pack(side="left", padx=15, pady=8)

        btn_clear = ctk.CTkButton(
            header, text="🔄 Clear Signals",
            fg_color="#45475A", text_color="#CDD6F4",
            font=ctk.CTkFont(size=11),
            command=self._clear_heatmap
        )
        btn_clear.pack(side="right", padx=10, pady=8)

        table_frame = ctk.CTkFrame(self.tab_heatmap, fg_color="#1E1E2E", corner_radius=6)
        table_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.sector_heatmap = SectorHeatmap(table_frame)
        self.sector_heatmap.refresh()  # show empty state

    def _clear_heatmap(self):
        self.sector_heatmap.clear_signals()

    def _refresh_heatmap_periodic(self):
        """Refresh heatmap every 30 seconds."""
        try:
            self.sector_heatmap.refresh()
        except Exception as e:
            print(f"[Heatmap] Refresh error: {e}")
        self.after(30_000, self._refresh_heatmap_periodic)

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 5: Reports & Alerts (Features 3, 7)
    # ─────────────────────────────────────────────────────────────────────────

    def _build_reports_tab(self):
        # ── Telegram section ─────────────────────────────────────────────────
        tg_frame = ctk.CTkFrame(self.tab_reports, fg_color="#11111B", corner_radius=6)
        tg_frame.pack(fill="x", padx=10, pady=(10, 5))

        ctk.CTkLabel(
            tg_frame,
            text="🔔 Telegram Alert Settings",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#89B4FA"
        ).pack(anchor="w", padx=15, pady=(10, 4))

        tg_status = "✅ Configured — alerts will fire on score ≥ 70" if self.telegram_alerter.is_enabled \
                    else "⚠️ Not configured — add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to .env"
        tg_color = "#A6E3A1" if self.telegram_alerter.is_enabled else "#F9E2AF"

        ctk.CTkLabel(
            tg_frame,
            text=tg_status,
            font=ctk.CTkFont(size=11),
            text_color=tg_color
        ).pack(anchor="w", padx=15, pady=4)

        btn_row = ctk.CTkFrame(tg_frame, fg_color="transparent")
        btn_row.pack(fill="x", padx=10, pady=(4, 10))

        ctk.CTkButton(
            btn_row, text="📱 Send Test Message",
            fg_color="#89B4FA", text_color="#11111B",
            font=ctk.CTkFont(weight="bold"),
            command=self._send_test_telegram
        ).pack(side="left", padx=5)

        self.tg_status_lbl = ctk.CTkLabel(btn_row, text="", font=ctk.CTkFont(size=10), text_color="#A6ADC8")
        self.tg_status_lbl.pack(side="left", padx=10)

        # ── PDF Report section ───────────────────────────────────────────────
        pdf_frame = ctk.CTkFrame(self.tab_reports, fg_color="#11111B", corner_radius=6)
        pdf_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(
            pdf_frame,
            text="🧾 PDF Session Report",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#A6E3A1"
        ).pack(anchor="w", padx=15, pady=(10, 4))

        ctk.CTkLabel(
            pdf_frame,
            text="Generate a professional PDF report with all crossovers, paper trades, and top signals.\n"
                 "Reports are saved to reports/session_YYYYMMDD_HHMMSS.pdf",
            font=ctk.CTkFont(size=11),
            text_color="#A6ADC8",
            justify="left"
        ).pack(anchor="w", padx=15, pady=4)

        btn_row2 = ctk.CTkFrame(pdf_frame, fg_color="transparent")
        btn_row2.pack(fill="x", padx=10, pady=(4, 10))

        ctk.CTkButton(
            btn_row2, text="📄 Generate Report Now",
            fg_color="#A6E3A1", text_color="#11111B",
            font=ctk.CTkFont(weight="bold"),
            command=self._generate_pdf_now
        ).pack(side="left", padx=5)

        self.pdf_status_lbl = ctk.CTkLabel(btn_row2, text="", font=ctk.CTkFont(size=10), text_color="#A6ADC8")
        self.pdf_status_lbl.pack(side="left", padx=10)

        # ── Recent Signals Log ───────────────────────────────────────────────
        log_frame = ctk.CTkFrame(self.tab_reports, fg_color="#11111B", corner_radius=6)
        log_frame.pack(fill="both", expand=True, padx=10, pady=5)

        ctk.CTkLabel(
            log_frame,
            text="📋 Recent High-Quality Signals (Score ≥ 70, Clean)",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#CDD6F4"
        ).pack(anchor="w", padx=15, pady=8)

        cols = ("time", "symbol", "signal", "score", "ghost", "ltp", "alert_sent")
        self.signal_log_tree = ttk.Treeview(log_frame, columns=cols, show="headings", selectmode="browse")

        for c, txt, w in [
            ("time",       "Time",        80),
            ("symbol",     "Symbol",      90),
            ("signal",     "Signal",      70),
            ("score",      "Score",      100),
            ("ghost",      "Ghost",       100),
            ("ltp",        "LTP (₹)",     90),
            ("alert_sent", "Alert",        80),
        ]:
            self.signal_log_tree.heading(c, text=txt)
            self.signal_log_tree.column(c, anchor="center", width=w)

        sb = ttk.Scrollbar(log_frame, orient="vertical", command=self.signal_log_tree.yview)
        self.signal_log_tree.configure(yscrollcommand=sb.set)
        self.signal_log_tree.pack(side="left", fill="both", expand=True, padx=10, pady=(0, 10))
        sb.pack(side="right", fill="y", pady=(0, 10))

    def _send_test_telegram(self):
        def _send():
            ok = self.telegram_alerter.send_test_message()
            msg = "✅ Test sent!" if ok else "❌ Failed — check .env config"
            self.after(0, lambda: self.tg_status_lbl.configure(text=msg))
        threading.Thread(target=_send, daemon=True).start()

    def _generate_pdf_now(self):
        """Generate PDF report on demand."""
        self.pdf_status_lbl.configure(text="Generating...")
        def _gen():
            try:
                session_data = self._build_session_data()
                path = self.report_generator.generate(session_data)
                msg = f"✅ Saved: {path}" if path else "❌ fpdf2 not installed (pip install fpdf2)"
            except Exception as e:
                msg = f"❌ Error: {e}"
            self.after(0, lambda: self.pdf_status_lbl.configure(text=msg))
        threading.Thread(target=_gen, daemon=True).start()

    # ─────────────────────────────────────────────────────────────────────────
    # Shared helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _create_card(self, parent, title, value, color):
        card = ctk.CTkFrame(parent, fg_color="#11111B", corner_radius=6)
        t_lbl = ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=11), text_color="#A6ADC8")
        t_lbl.pack(padx=10, pady=(8, 2))
        v_lbl = ctk.CTkLabel(card, text=value, font=ctk.CTkFont(size=18, weight="bold"), text_color=color)
        v_lbl.pack(padx=10, pady=(0, 8))
        card.v_lbl = v_lbl
        return card

    def _draw_placeholder_chart(self):
        self.canvas.delete("all")
        self.canvas.create_text(140, 45, text="Select stock row\nto view Order Book depth",
                                fill="#6C7086", font=("Segoe UI", 9), justify="center")

    def _build_statusbar(self):
        self.status_bar = ctk.CTkLabel(
            self,
            text="System Ready | 1-Min Timeframe | Ghost+Score+Telegram+PDF+Heatmap+Sparkline Enabled",
            anchor="w",
            font=ctk.CTkFont(size=11),
            text_color="#A6ADC8"
        )
        self.status_bar.pack(fill="x", padx=15, pady=(0, 5))

    def _build_session_data(self) -> dict:
        """Build the session data dict for the PDF report generator."""
        stats = self.paper_trader.get_statistics()
        return {
            'date':          datetime.now().strftime("%Y-%m-%d"),
            'start_time':    self.session_start_time,
            'stocks_scanned': len(self.latest_data),
            'crossovers':    self.session_crossovers,
            'paper_stats':   stats,
            'closed_trades': [
                {
                    'symbol':       t.get('symbol', ''),
                    'signal_type':  t.get('signal_type', ''),
                    'entry_price':  t.get('entry_price', 0),
                    'exit_price':   t.get('exit_price', 0),
                    'pnl':          t.get('pnl', 0),
                    'status':       t.get('status', ''),
                    'entry_time':   t.get('entry_time', ''),
                }
                for t in self.paper_trader.closed_trades
            ]
        }

    # ─────────────────────────────────────────────────────────────────────────
    # ML training callbacks
    # ─────────────────────────────────────────────────────────────────────────

    def _auto_train_ml_model(self):
        summary = self.ai_evaluator.train_model()
        self._pending_ml_summary = summary

    def _on_click_retrain(self):
        self.status_bar.configure(text="Retraining Day 1 ML Model on 1-min candles...")
        threading.Thread(target=self._auto_train_ml_model, daemon=True).start()

    def _update_training_ui(self, summary):
        self.card_crossovers.v_lbl.configure(text=str(summary.get('total_crossovers', 0)))
        self.card_accuracy.v_lbl.configure(text=f"{summary.get('accuracy', 0)}%")
        self.card_precision.v_lbl.configure(text=f"{summary.get('precision', 0)}%")
        self._draw_feature_importance_chart(summary.get('feature_importances', {}))
        self.status_bar.configure(text=f"Day 1 ML Model Trained | Accuracy: {summary.get('accuracy', 0)}% | All 8 Features Active")

    def _draw_feature_importance_chart(self, importances):
        self.feat_canvas.delete("all")
        if not importances:
            return

        w = self.feat_canvas.winfo_width() or 800
        bar_h = 24
        gap   = 12
        top_y = 20
        max_v = max(importances.values()) if importances else 1.0

        label_map = {
            'ltq_ratio':       'LTQ (Last Traded Qty) Ratio',
            'depth_imbalance': 'Order Book Depth Imbalance',
            'smma_spread':     '20/120 SMMA Spread %',
            'smma_slope':      '20 SMMA Slope Velocity',
            'vol_surge_ratio': 'Volume Surge Acceleration',
            'price_dev':       'Price Deviation'
        }

        for i, (name, val) in enumerate(importances.items()):
            y     = top_y + i * (bar_h + gap)
            pct   = val / max_v
            bar_w = int((w - 240) * pct)
            lbl   = label_map.get(name, name)

            self.feat_canvas.create_text(180, y + bar_h/2, text=lbl, fill="#CDD6F4",
                                          font=("Segoe UI", 9, "bold"), anchor="e")
            self.feat_canvas.create_rectangle(190, y, 190 + bar_w, y + bar_h, fill="#89B4FA", outline="")
            self.feat_canvas.create_text(200 + bar_w, y + bar_h/2, text=f"{val*100:.1f}%",
                                          fill="#A6E3A1", font=("Segoe UI", 9, "bold"), anchor="w")

    # ─────────────────────────────────────────────────────────────────────────
    # Live tick processing
    # ─────────────────────────────────────────────────────────────────────────

    def on_tick_update(self, symbol, data, metrics):
        """Called from BrokerStreamer thread on every tick."""
        ltp = data['ltp']

        # Update sparkline buffer
        self.sparkline_store.update(symbol, ltp)

        # Update paper trader live prices
        closed = self.paper_trader.update_live_price(symbol, ltp)
        if closed:
            self._add_trade_to_table(closed)

        self.latest_data[symbol] = {'data': data, 'metrics': metrics, 'ts': time.time()}

    def _periodic_ui_refresh(self):
        """Main refresh loop — runs every 500ms on the tkinter main thread."""
        # Handle pending ML training results
        if self._pending_ml_summary is not None:
            summary = self._pending_ml_summary
            self._pending_ml_summary = None
            try:
                self._update_training_ui(summary)
            except Exception as e:
                print("Update training UI error:", e)

        # Refresh all stock rows
        for symbol, entry in list(self.latest_data.items()):
            data    = entry['data']
            metrics = entry['metrics']
            ltp     = data['ltp']
            bid_qty = data['total_bid_qty']
            ask_qty = data['total_ask_qty']

            # Skip stocks not in LTP range or lacking liquidity
            if not (LTP_MIN <= ltp <= LTP_MAX) or not (bid_qty >= BID_QTY_MIN and ask_qty >= ASK_QTY_MIN):
                if self.tree.exists(symbol):
                    self.tree.delete(symbol)
                continue

            smma_20  = metrics['smma_20']
            smma_120 = metrics['smma_120']
            prev_rel = self.crossover_states.get(symbol, "NEUTRAL")

            # Crossover state
            if smma_20 > smma_120:
                current_rel   = "ABOVE"
                crossover_text = "BULLISH (20 > 120)"
            elif smma_20 < smma_120:
                current_rel   = "BELOW"
                crossover_text = "BEARISH (20 < 120)"
            else:
                current_rel   = prev_rel
                crossover_text = "NEUTRAL"

            # ── Detect crossover event ────────────────────────────────────────
            if prev_rel not in ("NEUTRAL", "") and current_rel != prev_rel:
                sig_type = "BUY" if current_rel == "ABOVE" else "SELL"

                # ML evaluation (existing)
                eval_res = self.ai_evaluator.evaluate_signal(sig_type, symbol, data, metrics)
                self._display_ai_result(symbol, eval_res)

                # Feature 1: Ghost detection
                ghost_data   = build_ghost_signal_data(data, metrics)
                ghost_status = detect_ghost(ghost_data)

                # Feature 2: Signal quality score
                smma_hist = list(self.sparkline_store.get_prices(symbol))  # reuse price history
                sig_score = score_signal(
                    symbol=symbol,
                    signal_type=sig_type,
                    vol_5m=float(metrics.get('vol_5m', 0)),
                    vol_60m=float(metrics.get('vol_60m', 1)),
                    smma_20_history=smma_hist,
                    bid_qty=float(bid_qty),
                    ask_qty=float(ask_qty),
                    closed_trades=self.paper_trader.closed_trades
                )

                # Paper trade via existing engine
                res = self.paper_trader.process_signal(symbol, sig_type, ltp, eval_res)

                # Build crossover record
                crossover_rec = {
                    'time':         time.strftime('%H:%M:%S'),
                    'symbol':       symbol,
                    'signal_type':  sig_type,
                    'ltp':          ltp,
                    'smma_20':      smma_20,
                    'smma_120':     smma_120,
                    'ghost_status': ghost_status,
                    'score':        sig_score.score,
                    'label':        sig_score.label,
                    'vol_5m':       metrics.get('vol_5m', 0),
                    'reason':       eval_res.get('explanation', '')
                }
                self.session_crossovers.append(crossover_rec)

                # Feature 5: Heatmap
                self.sector_heatmap.add_signal(symbol, sig_score.score, sig_type, ltp)

                # Add to paper trade table
                if res['action'] == 'OPENED':
                    pos = res['position']
                    self._add_trade_to_table({
                        'entry_time':   pos['entry_time'],
                        'symbol':       symbol,
                        'signal_type':  sig_type,
                        'entry_price':  ltp,
                        'exit_price':   '-',
                        'pnl':          0.0,
                        'ghost_status': ghost_status,
                        'score':        sig_score.score,
                        'status':       'ACTIVE POSITION',
                        'reason':       f"ML {eval_res['probability']}% | {sig_score.label}"
                    })
                elif res['action'] == 'AVOIDED':
                    self._add_trade_to_table({
                        'entry_time':   time.strftime('%H:%M:%S'),
                        'symbol':       symbol,
                        'signal_type':  sig_type,
                        'entry_price':  ltp,
                        'exit_price':   '-',
                        'pnl':          0.0,
                        'ghost_status': ghost_status,
                        'score':        sig_score.score,
                        'status':       'TRADE AVOIDED',
                        'reason':       res['reason']
                    })

                # Feature 3: Telegram alert for high-quality clean signals
                self.telegram_alerter.send_alert({
                    'symbol':      symbol,
                    'signal_type': sig_type,
                    'score':       sig_score.score,
                    'score_label': sig_score.label,
                    'ltp':         ltp,
                    'smma_20':     smma_20,
                    'smma_120':    smma_120,
                    'vol_5m':      metrics.get('vol_5m', 0),
                    'ghost_status': ghost_status,
                    'reason':      eval_res.get('explanation', '')
                })

                # Log in Reports tab if score >= 70
                if sig_score.score >= 70:
                    alert_sent = "📱 Sent" if (self.telegram_alerter.is_enabled and ghost_status == "CLEAN") else "-"
                    self._log_signal_report(crossover_rec, alert_sent)

            self.crossover_states[symbol] = current_rel

            # ── Feature 6: Sparkline ─────────────────────────────────────────
            sparkline = self.sparkline_store.get_sparkline(symbol)

            # Ghost status for current state (on every tick, not just crossover)
            ghost_data   = build_ghost_signal_data(data, metrics)
            ghost_status = detect_ghost(ghost_data)
            ghost_label  = ghost_label_with_emoji(ghost_status)

            # Score (keep last computed or compute rough one for display)
            last_cross = next((c for c in reversed(self.session_crossovers) if c['symbol'] == symbol), None)
            score_display = f"{last_cross['score']} {last_cross['label']}" if last_cross else "-"

            ltq_ratio = data.get('ltq_ratio', metrics['vol_5m'] / 10000.0)

            row_values = (
                symbol,
                f"₹{ltp:.2f}",
                sparkline,
                f"{bid_qty:,}",
                f"{ask_qty:,}",
                f"{smma_20:.2f}",
                f"{smma_120:.2f}",
                f"{metrics['vol_5m']:,}",
                f"{ltq_ratio:.2f}x",
                ghost_label,
                score_display,
                crossover_text
            )

            # Choose row tag based on ghost status
            tag = "clean" if ghost_status == "CLEAN" else ("ghost" if ghost_status == "GHOST" else "low_conv")

            if self.tree.exists(symbol):
                self.tree.item(symbol, values=row_values, tags=(tag,))
            else:
                self.tree.insert("", "end", iid=symbol, values=row_values, tags=(tag,))

        self._refresh_paper_trade_stats()
        self.after(500, self._periodic_ui_refresh)

    def _log_signal_report(self, crossover_rec: dict, alert_sent: str):
        """Add a signal to the Reports tab log."""
        try:
            self.signal_log_tree.insert("", 0, values=(
                crossover_rec.get('time', '-'),
                crossover_rec.get('symbol', '-'),
                crossover_rec.get('signal_type', '-'),
                f"{crossover_rec.get('score', 0)}/100 {crossover_rec.get('label', '')}",
                ghost_label_with_emoji(crossover_rec.get('ghost_status', 'CLEAN')),
                f"₹{crossover_rec.get('ltp', 0):.2f}",
                alert_sent
            ))
        except Exception as e:
            print(f"[ReportTab] Log error: {e}")

    def _refresh_paper_trade_stats(self):
        stats = self.paper_trader.get_statistics()

        # Update cards
        self.card_win_rate.v_lbl.configure(text=f"{stats['win_rate_pct']}%")
        self.card_loss_rate.v_lbl.configure(text=f"{stats['loss_rate_pct']}%")
        self.card_avoided_pct.v_lbl.configure(text=f"{stats['avoided_trades_pct']}%")
        pnl_color = "#A6E3A1" if stats['net_pnl'] >= 0 else "#F38BA8"
        self.card_pnl.v_lbl.configure(text=f"₹{stats['net_pnl']:,.2f}", text_color=pnl_color)

        # Update summary line — Feature 4
        total  = stats['total_trades_taken']
        wr     = stats['win_rate_pct']
        pnl    = stats['net_pnl']
        pnl_str = f"+₹{pnl:,.2f}" if pnl >= 0 else f"-₹{abs(pnl):,.2f}"
        self.paper_summary_lbl.configure(
            text=f"📋 Paper Trades: {total} closed | Win Rate: {wr}% | P&L: {pnl_str}"
        )

    def _add_trade_to_table(self, trade):
        vals = (
            trade.get('entry_time', trade.get('exit_time', '-')),
            trade.get('symbol', '-'),
            trade.get('signal_type', '-'),
            f"₹{trade['entry_price']:.2f}",
            f"₹{trade['exit_price']:.2f}" if isinstance(trade.get('exit_price'), (int, float)) else trade.get('exit_price', '-'),
            f"₹{trade['pnl']:.2f}" if trade.get('pnl', 0) != 0 else "-",
            str(trade.get('ghost_status', '-')),
            str(trade.get('score', '-')),
            trade.get('status', '-'),
            trade.get('reason', '-')
        )
        self.trade_tree.insert("", 0, values=vals)

    # ─────────────────────────────────────────────────────────────────────────
    # Stock row selection — AI analysis + order book chart
    # ─────────────────────────────────────────────────────────────────────────

    def _on_row_select(self, event):
        selected = self.tree.selection()
        if not selected:
            return
        symbol = selected[0]
        if symbol in self.latest_data:
            entry   = self.latest_data[symbol]
            metrics = entry['metrics']
            data    = entry['data']
            smma_20 = metrics['smma_20']
            smma_120 = metrics['smma_120']
            sig_type = "BUY" if smma_20 >= smma_120 else "SELL"
            eval_res = self.ai_evaluator.evaluate_signal(sig_type, symbol, data, metrics)
            self._display_ai_result(symbol, eval_res)

    def _display_ai_result(self, symbol, eval_res):
        """Show AI evaluation result in the analysis box with ghost + score info."""
        # Get ghost and score for this symbol
        last_cross = next((c for c in reversed(self.session_crossovers) if c['symbol'] == symbol), None)
        ghost_label = ghost_label_with_emoji(last_cross['ghost_status']) if last_cross else "N/A"
        score_info  = f"{last_cross['score']}/100 {last_cross['label']}" if last_cross else "N/A"

        self.ai_output_box.configure(state="normal")
        self.ai_output_box.delete("1.0", "end")

        status_tag = "✅ APPROVED" if eval_res['is_approved'] else "⚠️ AVOID / REJECTED"
        text  = f"[{time.strftime('%H:%M:%S')}] SYMBOL: {symbol} | SIGNAL: {eval_res['signal_type']} SMMA CROSSOVER | STATUS: {status_tag}\n"
        text += f"DAY 1 ML WIN PROBABILITY: {eval_res['probability']}%\n"
        text += f"SIGNAL QUALITY SCORE: {score_info}\n"
        text += f"GHOST STATUS: {ghost_label}\n"
        text += f"EXPLANATION: {eval_res['explanation']}\n"
        text += f"METRICS: LTQ {eval_res['features']['ltq_ratio']:.2f}x | Imbalance: {eval_res['features']['depth_imbalance']:.2f} | Spread: {eval_res['features']['smma_spread']*100:.2f}%"

        self.ai_output_box.insert("1.0", text)
        self.ai_output_box.configure(state="disabled")
        self._draw_depth_chart(symbol, eval_res)

    def _draw_depth_chart(self, symbol, eval_res):
        self.canvas.delete("all")
        imbalance = eval_res['features']['depth_imbalance']
        prob      = eval_res['probability']

        self.canvas.create_text(140, 12, text=f"Order Book: {symbol}", fill="#CDD6F4", font=("Segoe UI", 9, "bold"))
        bid_pct = max(0.1, min(0.9, (imbalance + 1.0) / 2.0))
        ask_pct = 1.0 - bid_pct
        bid_w   = int(240 * bid_pct)

        self.canvas.create_rectangle(20, 26, 20 + bid_w, 42, fill="#A6E3A1", outline="")
        self.canvas.create_text(25, 34, text=f"Bids: {int(bid_pct*100)}%", fill="#11111B", font=("Segoe UI", 8, "bold"), anchor="w")
        self.canvas.create_rectangle(20 + bid_w, 26, 260, 42, fill="#F38BA8", outline="")
        self.canvas.create_text(255, 34, text=f"Asks: {int(ask_pct*100)}%", fill="#11111B", font=("Segoe UI", 8, "bold"), anchor="e")

        prob_color = "#A6E3A1" if prob >= 60 else "#F38BA8"
        self.canvas.create_text(140, 56, text=f"Day 1 ML Win Prob: {prob}%", fill=prob_color, font=("Segoe UI", 9, "bold"))
        self.canvas.create_rectangle(20, 68, 260, 78, fill="#313244", outline="")
        prob_w = int(240 * (prob / 100.0))
        self.canvas.create_rectangle(20, 68, 20 + prob_w, 78, fill=prob_color, outline="")

    # ─────────────────────────────────────────────────────────────────────────
    # App close — auto-generate PDF
    # ─────────────────────────────────────────────────────────────────────────

    def _on_close(self):
        """Generate PDF report on close if there are crossovers this session."""
        print("Session ended. Generating PDF report...")
        if self.session_crossovers:
            try:
                session_data = self._build_session_data()
                path = self.report_generator.generate(session_data)
                if path:
                    print(f"Report saved: {path}")
            except Exception as e:
                print(f"Could not generate report: {e}")
        self.destroy()


if __name__ == "__main__":
    app = StockScreenerApp()
    app.mainloop()
