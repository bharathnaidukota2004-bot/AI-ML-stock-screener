"""
ui/tab_evaluation.py — Machine Learning Evaluation Dashboard Tab
Renders chronological time-series evaluation comparing Random Forest,
Logistic Regression, and Gradient Boosting on held-out test data.
Displays Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix, and Feature Importance.
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import ttk
import threading
from typing import Dict, Any, Optional

from ml.evaluator import ModelEvaluator


class EvaluationTab:
    """
    UI Tab component for ML Model Evaluation.
    """

    def __init__(self, parent_tab, app_ref):
        self.tab = parent_tab
        self.app = app_ref
        self.evaluator = ModelEvaluator(use_technical_indicators=True)
        self.last_results: Dict[str, Any] = {}
        self._build_ui()

    def _build_ui(self):
        # 1. Top Control Bar
        ctrl_frame = ctk.CTkFrame(self.tab, fg_color="#11111B", corner_radius=6)
        ctrl_frame.pack(fill="x", padx=10, pady=6)

        lbl = ctk.CTkLabel(
            ctrl_frame,
            text="🤖 Chronological Machine Learning Model Evaluation & Benchmark Comparison (Zero Data Leakage)",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#A6E3A1"
        )
        lbl.pack(side="left", padx=12, pady=8)

        self.btn_eval = ctk.CTkButton(
            ctrl_frame,
            text="⚡ Run Chronological Evaluation",
            fg_color="#A6E3A1",
            text_color="#11111B",
            font=ctk.CTkFont(weight="bold"),
            command=self._on_click_evaluate
        )
        self.btn_eval.pack(side="right", padx=12, pady=8)

        self.chk_indicators = ctk.CTkCheckBox(
            ctrl_frame,
            text="Include Technical Indicators (RSI, MACD, BB, EMA, ATR)",
            font=ctk.CTkFont(size=10),
            text_color="#CDD6F4",
            onvalue=1,
            offvalue=0
        )
        self.chk_indicators.select()
        self.chk_indicators.pack(side="right", padx=15, pady=8)

        # 2. Best Model Banner
        self.banner = ctk.CTkFrame(self.tab, fg_color="#1E1E2E", corner_radius=6)
        self.banner.pack(fill="x", padx=10, pady=(2, 6))

        self.lbl_best_title = ctk.CTkLabel(
            self.banner,
            text="🏆 BEST VALIDATED MODEL: EVALUATION IN PROGRESS...",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#F9E2AF"
        )
        self.lbl_best_title.pack(side="left", padx=15, pady=8)

        self.lbl_sample_info = ctk.CTkLabel(
            self.banner,
            text="Dataset: 70% Chronological Train | 30% Held-out Test",
            font=ctk.CTkFont(size=11),
            text_color="#A6ADC8"
        )
        self.lbl_sample_info.pack(side="right", padx=15, pady=8)

        # 3. KPI Cards for Best Model
        cards_frame = ctk.CTkFrame(self.tab, fg_color="transparent")
        cards_frame.pack(fill="x", padx=10, pady=4)

        self.card_acc = self._create_card(cards_frame, "Test Accuracy", "-- %", "#89B4FA", "Held-out test split")
        self.card_acc.pack(side="left", fill="x", expand=True, padx=4)

        self.card_prec = self._create_card(cards_frame, "Precision", "-- %", "#A6E3A1", "Positive signal accuracy")
        self.card_prec.pack(side="left", fill="x", expand=True, padx=4)

        self.card_rec = self._create_card(cards_frame, "Recall", "-- %", "#F9E2AF", "Profit opportunity capture")
        self.card_rec.pack(side="left", fill="x", expand=True, padx=4)

        self.card_f1 = self._create_card(cards_frame, "F1-Score", "-- %", "#CBA6F7", "Harmonic balance")
        self.card_f1.pack(side="left", fill="x", expand=True, padx=4)

        self.card_roc = self._create_card(cards_frame, "ROC-AUC", "-- %", "#F38BA8", "Discrimination capability")
        self.card_roc.pack(side="left", fill="x", expand=True, padx=4)

        # 4. Model Comparison Table
        table_box = ctk.CTkFrame(self.tab, fg_color="#1E1E2E", corner_radius=6)
        table_box.pack(fill="x", padx=10, pady=6)

        lbl_tbl = ctk.CTkLabel(
            table_box,
            text="📋 Empirical Model Benchmark Comparison (Out-of-Sample Performance)",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#89B4FA"
        )
        lbl_tbl.pack(anchor="w", padx=12, pady=(6, 2))

        cols = ("model", "accuracy", "precision", "recall", "f1", "roc_auc", "rank")
        self.tree = ttk.Treeview(table_box, columns=cols, show="headings", height=4)
        headers = {
            "model": "Model Architecture",
            "accuracy": "Accuracy (%)",
            "precision": "Precision (%)",
            "recall": "Recall (%)",
            "f1": "F1-Score (%)",
            "roc_auc": "ROC-AUC (%)",
            "rank": "Validation Status"
        }
        for c, h in headers.items():
            self.tree.heading(c, text=h)
            self.tree.column(c, anchor="center", width=120)

        self.tree.pack(fill="x", padx=10, pady=(2, 8))

        # 5. Split Bottom Section: Confusion Matrix on Left, Feature Importance Bar Chart on Right
        bottom_frame = ctk.CTkFrame(self.tab, fg_color="#1E1E2E", corner_radius=6)
        bottom_frame.pack(fill="both", expand=True, padx=10, pady=6)

        # Left: Confusion Matrix & Class Breakdown
        cm_box = ctk.CTkFrame(bottom_frame, fg_color="#181825", corner_radius=6, width=420)
        cm_box.pack(side="left", fill="both", expand=False, padx=6, pady=6)

        lbl_cm = ctk.CTkLabel(
            cm_box,
            text="🎯 Confusion Matrix & Classification Report",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#CDD6F4"
        )
        lbl_cm.pack(anchor="w", padx=12, pady=(8, 4))

        self.cm_grid = ctk.CTkFrame(cm_box, fg_color="#11111B", corner_radius=6)
        self.cm_grid.pack(fill="x", padx=12, pady=6)

        self.lbl_tn = ctk.CTkLabel(self.cm_grid, text="True Neg (Losses Avoided): --", font=ctk.CTkFont(size=11), text_color="#A6E3A1")
        self.lbl_tn.pack(anchor="w", padx=10, pady=3)

        self.lbl_fp = ctk.CTkLabel(self.cm_grid, text="False Pos (False Alarms): --", font=ctk.CTkFont(size=11), text_color="#F38BA8")
        self.lbl_fp.pack(anchor="w", padx=10, pady=3)

        self.lbl_fn = ctk.CTkLabel(self.cm_grid, text="False Neg (Missed Wins): --", font=ctk.CTkFont(size=11), text_color="#F9E2AF")
        self.lbl_fn.pack(anchor="w", padx=10, pady=3)

        self.lbl_tp = ctk.CTkLabel(self.cm_grid, text="True Pos (Winning Trades): --", font=ctk.CTkFont(size=11), text_color="#89B4FA")
        self.lbl_tp.pack(anchor="w", padx=10, pady=3)

        self.lbl_tscv = ctk.CTkLabel(
            cm_box,
            text="Time-Series CV (3-Fold): Pending...",
            font=ctk.CTkFont(size=10),
            text_color="#A6ADC8"
        )
        self.lbl_tscv.pack(anchor="w", padx=12, pady=(4, 8))

        # Right: Feature Importance Bar Chart
        fi_box = ctk.CTkFrame(bottom_frame, fg_color="#181825", corner_radius=6)
        fi_box.pack(side="right", fill="both", expand=True, padx=6, pady=6)

        lbl_fi = ctk.CTkLabel(
            fi_box,
            text="📊 Feature Importance / Information Gain (Best Model)",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#CDD6F4"
        )
        lbl_fi.pack(anchor="w", padx=12, pady=(8, 2))

        self.canvas_fi = tk.Canvas(fi_box, bg="#11111B", highlightthickness=0)
        self.canvas_fi.pack(fill="both", expand=True, padx=8, pady=8)

        # Trigger initial background evaluation
        self._on_click_evaluate()

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

    def _on_click_evaluate(self):
        use_indicators = bool(self.chk_indicators.get())
        self.btn_eval.configure(text="⏳ Evaluating Models...", state="disabled")
        threading.Thread(target=self._run_eval_bg, args=(use_indicators,), daemon=True).start()

    def _run_eval_bg(self, use_indicators: bool):
        self.evaluator.use_technical_indicators = use_indicators
        self.evaluator.feature_names = (
            ["ltq_ratio", "depth_imbalance", "smma_spread", "smma_slope", "vol_surge_ratio", "price_dev",
             "rsi_norm", "macd_hist_ratio", "bb_pct_b", "ema_ratio", "natr"]
            if use_indicators
            else ["ltq_ratio", "depth_imbalance", "smma_spread", "smma_slope", "vol_surge_ratio", "price_dev"]
        )
        try:
            results = self.evaluator.run_chronological_evaluation()
        except Exception as e:
            results = {'error': str(e)}

        try:
            self.tab.after(0, self._render_results, results)
        except Exception:
            pass

    def _render_results(self, res: dict):
        self.btn_eval.configure(text="⚡ Run Chronological Evaluation", state="normal")
        if 'error' in res:
            self.lbl_best_title.configure(text=f"EVALUATION ERROR: {res['error']}")
            return

        self.last_results = res
        best = res.get('best_model', {})
        best_name = best.get('name', 'Random Forest')
        models_data = res.get('models', {})
        best_metrics = models_data.get(best_name, {})

        # Update Banner
        self.lbl_best_title.configure(
            text=f"🏆 BEST MODEL: {best_name.upper()} (F1: {best.get('f1_score', 0):.1f}% | Acc: {best.get('accuracy', 0):.1f}%)"
        )
        ds = res.get('dataset', {})
        self.lbl_sample_info.configure(
            text=f"Samples: {ds.get('train_samples', 0)} Train | {ds.get('test_samples', 0)} Held-Out Test | {len(ds.get('feature_names', []))} Features"
        )

        # Update KPI cards
        self.card_acc.v_lbl.configure(text=f"{best_metrics.get('accuracy', 0.0):.1f}%")
        self.card_prec.v_lbl.configure(text=f"{best_metrics.get('precision', 0.0):.1f}%")
        self.card_rec.v_lbl.configure(text=f"{best_metrics.get('recall', 0.0):.1f}%")
        self.card_f1.v_lbl.configure(text=f"{best_metrics.get('f1_score', 0.0):.1f}%")
        self.card_roc.v_lbl.configure(text=f"{best_metrics.get('roc_auc', 0.0):.1f}%")

        # Update Table
        for row in self.tree.get_children():
            self.tree.delete(row)

        for m_name, m_stats in models_data.items():
            is_best = (m_name == best_name)
            status_text = "⭐ TOP PERFORMER" if is_best else "BASELINE"
            self.tree.insert("", "end", values=(
                m_name,
                f"{m_stats['accuracy']:.1f}%",
                f"{m_stats['precision']:.1f}%",
                f"{m_stats['recall']:.1f}%",
                f"{m_stats['f1_score']:.1f}%",
                f"{m_stats['roc_auc']:.1f}%",
                status_text
            ))

        # Confusion Matrix
        cm = best_metrics.get('confusion_matrix', [[0, 0], [0, 0]])
        tn = cm[0][0] if len(cm) > 0 and len(cm[0]) > 0 else 0
        fp = cm[0][1] if len(cm) > 0 and len(cm[0]) > 1 else 0
        fn = cm[1][0] if len(cm) > 1 and len(cm[1]) > 0 else 0
        tp = cm[1][1] if len(cm) > 1 and len(cm[1]) > 1 else 0

        self.lbl_tn.configure(text=f"True Neg (Loss Avoided): {tn}")
        self.lbl_fp.configure(text=f"False Pos (Bad Signal):   {fp}")
        self.lbl_fn.configure(text=f"False Neg (Missed Win):   {fn}")
        self.lbl_tp.configure(text=f"True Pos (Winning Trade):  {tp}")

        tscv_list = best.get('tscv_3fold_accuracies', [])
        tscv_str = ", ".join([f"{s}%" for s in tscv_list]) if tscv_list else "Completed"
        self.lbl_tscv.configure(text=f"Time-Series CV (3-Fold Out-of-Sample): [{tscv_str}]")

        # Feature Importance Chart
        importances = best_metrics.get('feature_importances', {})
        self._draw_feature_importance(importances)

    def _draw_feature_importance(self, importances: dict):
        self.canvas_fi.delete("all")
        w = self.canvas_fi.winfo_width() or 400
        h = self.canvas_fi.winfo_height() or 220

        if not importances:
            return

        sorted_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:8]
        max_val = max(1e-4, max(v for _, v in sorted_feats))

        bar_h = min(20, int((h - 30) / len(sorted_feats)))
        y_start = 15

        for i, (name, val) in enumerate(sorted_feats):
            y = y_start + (i * (bar_h + 8))
            self.canvas_fi.create_text(10, y + (bar_h / 2), text=name[:14], fill="#CDD6F4", font=("Segoe UI", 8, "bold"), anchor="w")

            bar_w = (val / max_val) * (w - 180)
            x0 = 110
            x1 = x0 + max(4, bar_w)
            color = "#89B4FA" if i < 3 else "#A6E3A1"

            self.canvas_fi.create_rectangle(x0, y, x1, y + bar_h, fill=color, outline="")
            self.canvas_fi.create_text(x1 + 6, y + (bar_h / 2), text=f"{val:.3f}", fill="#F5E0DC", font=("Segoe UI", 8), anchor="w")
