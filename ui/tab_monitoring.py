"""
ui/tab_monitoring.py — Model Monitoring & Data Drift Dashboard Tab
Observes inference predictions, evaluates rolling accuracy upon outcome resolution,
and calculates Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) tests for drift.
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import ttk
import threading
from typing import Dict, Any, Optional

from mlops.monitoring import ModelMonitor, DriftReport
from core.data_fetcher import fetch_stock_1min_data, prepare_dataset_with_indicators


class MonitoringTab:
    """
    UI Tab component for Real-Time Model Monitoring & Data Drift Detection.
    """

    def __init__(self, parent_tab, app_ref):
        self.tab = parent_tab
        self.app = app_ref
        self.monitor = ModelMonitor()
        self.monitor.seed_initial_monitoring_data()
        self.last_drift_report: Optional[DriftReport] = None
        self._build_ui()

    def _build_ui(self):
        # 1. Header Control Bar
        ctrl_frame = ctk.CTkFrame(self.tab, fg_color="#11111B", corner_radius=6)
        ctrl_frame.pack(fill="x", padx=10, pady=6)

        lbl = ctk.CTkLabel(
            ctrl_frame,
            text="📡 Real-Time Model Observability: Prediction Audit Store | Outcome Resolution | PSI & KS Drift Tests",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#89B4FA"
        )
        lbl.pack(side="left", padx=12, pady=8)

        btn_drift = ctk.CTkButton(
            ctrl_frame,
            text="🔍 Run Drift Assessment",
            fg_color="#89B4FA",
            text_color="#11111B",
            font=ctk.CTkFont(weight="bold"),
            command=self._on_click_drift
        )
        btn_drift.pack(side="right", padx=10, pady=8)

        # 2. Observability Status Banner
        self.banner = ctk.CTkFrame(self.tab, fg_color="#1E1E2E", corner_radius=6)
        self.banner.pack(fill="x", padx=10, pady=(2, 6))

        self.lbl_status = ctk.CTkLabel(
            self.banner,
            text="MODEL STATUS: HEALTHY & STABLE (Zero Lookahead Outcome Resolution Active)",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#A6E3A1"
        )
        self.lbl_status.pack(side="left", padx=15, pady=8)

        self.lbl_rec = ctk.CTkLabel(
            self.banner,
            text="Recommendation: All feature distributions within acceptable tolerances (PSI < 0.10)",
            font=ctk.CTkFont(size=10),
            text_color="#A6ADC8"
        )
        self.lbl_rec.pack(side="right", padx=15, pady=8)

        # 3. KPI Cards
        cards_frame = ctk.CTkFrame(self.tab, fg_color="transparent")
        cards_frame.pack(fill="x", padx=10, pady=4)

        self.card_drift = self._create_card(cards_frame, "Drift Assessment", "STABLE", "#A6E3A1", "PSI < 0.10 baseline")
        self.card_drift.pack(side="left", fill="x", expand=True, padx=4)

        self.card_total = self._create_card(cards_frame, "Total Predictions", "--", "#89B4FA", "Inference log count")
        self.card_total.pack(side="left", fill="x", expand=True, padx=4)

        self.card_resolved = self._create_card(cards_frame, "Resolved Outcomes", "--", "#A6E3A1", "Verified market ground truth")
        self.card_resolved.pack(side="left", fill="x", expand=True, padx=4)

        self.card_pending = self._create_card(cards_frame, "Pending Outcomes", "--", "#F9E2AF", "Awaiting trade exit")
        self.card_pending.pack(side="left", fill="x", expand=True, padx=4)

        self.card_roll_acc = self._create_card(cards_frame, "Live Rolling Accuracy", "-- %", "#CBA6F7", "Recent verified trades")
        self.card_roll_acc.pack(side="left", fill="x", expand=True, padx=4)

        # 4. Split View: Feature Drift Table (Left) & Predictions Log (Right)
        split_frame = ctk.CTkFrame(self.tab, fg_color="#1E1E2E", corner_radius=6)
        split_frame.pack(fill="both", expand=True, padx=10, pady=6)

        # Left: Feature Drift Table
        left_box = ctk.CTkFrame(split_frame, fg_color="#181825", corner_radius=6)
        left_box.pack(side="left", fill="both", expand=True, padx=6, pady=6)

        lbl_drift_tbl = ctk.CTkLabel(
            left_box,
            text="🔬 Input Feature Distribution Drift (PSI & KS-Test Statistics)",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#CDD6F4"
        )
        lbl_drift_tbl.pack(anchor="w", padx=10, pady=(6, 2))

        cols_drift = ("feature", "ref_m", "live_m", "psi", "ks_stat", "status")
        self.tree_drift = ttk.Treeview(left_box, columns=cols_drift, show="headings", height=8)
        headers_drift = {
            "feature": "Feature",
            "ref_m": "Baseline Mean",
            "live_m": "Live Mean",
            "psi": "PSI Index",
            "ks_stat": "KS Stat (p-val)",
            "status": "Drift Status"
        }
        for c, h in headers_drift.items():
            self.tree_drift.heading(c, text=h)
            self.tree_drift.column(c, anchor="center", width=85 if c != "feature" else 110)

        self.tree_drift.tag_configure("stable", foreground="#A6E3A1")
        self.tree_drift.tag_configure("moderate", foreground="#F9E2AF")
        self.tree_drift.tag_configure("drift", foreground="#F38BA8")

        self.tree_drift.pack(fill="both", expand=True, padx=6, pady=6)

        # Right: Live Predictions Audit Log
        right_box = ctk.CTkFrame(split_frame, fg_color="#181825", corner_radius=6)
        right_box.pack(side="right", fill="both", expand=True, padx=6, pady=6)

        lbl_pred_tbl = ctk.CTkLabel(
            right_box,
            text="📑 Live Prediction Audit Store (Pending vs Resolved)",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#CDD6F4"
        )
        lbl_pred_tbl.pack(anchor="w", padx=10, pady=(6, 2))

        cols_pred = ("id", "ticker", "pred", "prob", "actual", "return", "status")
        self.tree_pred = ttk.Treeview(right_box, columns=cols_pred, show="headings", height=8)
        headers_pred = {
            "id": "Pred ID",
            "ticker": "Symbol",
            "pred": "Model Pred",
            "prob": "Win Prob",
            "actual": "Actual Outcome",
            "return": "Trade Return",
            "status": "Resolution"
        }
        for c, h in headers_pred.items():
            self.tree_pred.heading(c, text=h)
            self.tree_pred.column(c, anchor="center", width=75 if c in ["pred", "actual", "return"] else 95)

        self.tree_pred.tag_configure("resolved", foreground="#A6E3A1")
        self.tree_pred.tag_configure("pending", foreground="#F9E2AF")

        self.tree_pred.pack(fill="both", expand=True, padx=6, pady=6)

        # Trigger initial data load
        self._refresh_display()

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

    def _on_click_drift(self):
        threading.Thread(target=self._run_drift_assessment_bg, daemon=True).start()

    def _run_drift_assessment_bg(self):
        # Fetch baseline historical data
        df = fetch_stock_1min_data("TATASTEEL.NS", period_days=3)
        df = prepare_dataset_with_indicators(df)
        report = self.monitor.evaluate_data_drift(reference_df=df)
        try:
            self.tab.after(0, self._render_drift_report, report)
        except Exception:
            pass

    def _refresh_display(self):
        # Render prediction metrics
        perf = self.monitor.compute_performance_metrics()
        self.card_total.v_lbl.configure(text=str(perf['total_predictions']))
        self.card_resolved.v_lbl.configure(text=str(perf['resolved_count']))
        self.card_pending.v_lbl.configure(text=str(perf['pending_count']))
        self.card_roll_acc.v_lbl.configure(text=f"{perf['current_accuracy']:.1f}%")

        # Update Predictions Treeview
        for r in self.tree_pred.get_children():
            self.tree_pred.delete(r)

        for p in reversed(self.monitor.predictions[-20:]):
            st = p.get('status', 'PENDING_OUTCOME')
            tag = "resolved" if st == "RESOLVED" else "pending"
            pred_text = "BUY (Win)" if p.get('predicted_label') == 1 else "AVOID/SELL"
            act_text = ("PROFIT (1)" if p.get('actual_label') == 1 else "LOSS (0)") if p.get('actual_label') is not None else "⏳ AWAITING"
            ret_text = f"{p.get('actual_return_pct'):+.1f}%" if p.get('actual_return_pct') is not None else "--"

            self.tree_pred.insert("", "end", values=(
                p.get('pred_id', '')[-12:],
                p.get('ticker', ''),
                pred_text,
                f"{p.get('predicted_probability', 0):.1f}%",
                act_text,
                ret_text,
                st
            ), tags=(tag,))

        # Run initial drift test
        self._on_click_drift()

    def _render_drift_report(self, report: DriftReport):
        self.last_drift_report = report
        status = report.overall_status

        if status == "STABLE":
            color = "#A6E3A1"
            self.lbl_status.configure(text="MODEL STATUS: HEALTHY & STABLE (No Significant Data Drift Detected)", text_color=color)
            self.card_drift.v_lbl.configure(text="✅ STABLE", text_color=color)
        elif status == "MODERATE_SHIFT":
            color = "#F9E2AF"
            self.lbl_status.configure(text="MODEL STATUS: MODERATE DISTRIBUTION SHIFT (Monitoring Drift)", text_color=color)
            self.card_drift.v_lbl.configure(text="⚠️ SHIFT", text_color=color)
        else:
            color = "#F38BA8"
            self.lbl_status.configure(text="MODEL STATUS: SIGNIFICANT DATA DRIFT (Retraining Triggered)", text_color=color)
            self.card_drift.v_lbl.configure(text="🚨 DRIFT ALERT", text_color=color)

        if report.recommendations:
            self.lbl_rec.configure(text=f"Action: {report.recommendations[0]}")

        # Update Drift Treeview
        for r in self.tree_drift.get_children():
            self.tree_drift.delete(r)

        for feat, metrics in report.feature_metrics.items():
            st = metrics.get('status', 'STABLE')
            tag = "stable" if st == "STABLE" else ("moderate" if st == "MODERATE_SHIFT" else "drift")
            psi = metrics.get('psi', 0.0)
            ks = metrics.get('ks_statistic', 0.0)
            pval = metrics.get('p_value', 1.0)
            ref_m = metrics.get('ref_mean', 0.0)
            live_m = metrics.get('live_mean', 0.0)

            self.tree_drift.insert("", "end", values=(
                feat,
                f"{ref_m:.3f}",
                f"{live_m:.3f}",
                f"{psi:.4f}",
                f"{ks:.3f} (p={pval:.2f})",
                st
            ), tags=(tag,))
