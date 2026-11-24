"""
ui/tab_mlflow.py — MLflow Experiment Tracking & Model Registry Dashboard Tab
Provides experiment management, parameter tracking, metrics comparison,
and safe model versioning with automatic backups.
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import ttk
import threading
from typing import Dict, Any, Optional
import os

from mlops.mlflow_tracker import MLflowExperimentTracker


class MLflowTab:
    """
    UI Tab component for MLflow Experiment Tracking & Model Registry.
    """

    def __init__(self, parent_tab, app_ref):
        self.tab = parent_tab
        self.app = app_ref
        self.tracker = MLflowExperimentTracker()
        self._build_ui()

    def _build_ui(self):
        # 1. Header Control Bar
        ctrl_frame = ctk.CTkFrame(self.tab, fg_color="#11111B", corner_radius=6)
        ctrl_frame.pack(fill="x", padx=10, pady=6)

        mlflow_status = "Connected (mlflow-skinny)" if self.tracker.use_mlflow else "Local JSON Registry (Zero Dependency Mode)"
        lbl = ctk.CTkLabel(
            ctrl_frame,
            text=f"🔬 MLflow Experiment Tracking & Model Versioning Engine — Backend: {mlflow_status}",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#CBA6F7"
        )
        lbl.pack(side="left", padx=12, pady=8)

        btn_refresh = ctk.CTkButton(
            ctrl_frame,
            text="🔄 Refresh Runs",
            fg_color="#CBA6F7",
            text_color="#11111B",
            font=ctk.CTkFont(weight="bold"),
            width=110,
            command=self._load_experiments
        )
        btn_refresh.pack(side="right", padx=10, pady=8)

        # 2. Status Banner
        self.banner = ctk.CTkFrame(self.tab, fg_color="#1E1E2E", corner_radius=6)
        self.banner.pack(fill="x", padx=10, pady=(2, 6))

        self.lbl_active_model = ctk.CTkLabel(
            self.banner,
            text="ACTIVE PRODUCTION MODEL: Random Forest (Version: latest | Zero Overwrite Protection: ENABLED)",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#A6E3A1"
        )
        self.lbl_active_model.pack(side="left", padx=15, pady=8)

        self.lbl_storage_info = ctk.CTkLabel(
            self.banner,
            text="Artifacts: models/active/ & models/archive/ (with auto-backup)",
            font=ctk.CTkFont(size=10),
            text_color="#A6ADC8"
        )
        self.lbl_storage_info.pack(side="right", padx=15, pady=8)

        # 3. KPI Cards
        cards_frame = ctk.CTkFrame(self.tab, fg_color="transparent")
        cards_frame.pack(fill="x", padx=10, pady=4)

        self.card_total = self._create_card(cards_frame, "Total Runs Tracked", "0", "#89B4FA", "All experiments")
        self.card_total.pack(side="left", fill="x", expand=True, padx=4)

        self.card_best_name = self._create_card(cards_frame, "Top Performing Architecture", "--", "#A6E3A1", "Empirical best")
        self.card_best_name.pack(side="left", fill="x", expand=True, padx=4)

        self.card_best_f1 = self._create_card(cards_frame, "Highest F1-Score", "-- %", "#F9E2AF", "Held-out test validation")
        self.card_best_f1.pack(side="left", fill="x", expand=True, padx=4)

        self.card_best_acc = self._create_card(cards_frame, "Top Accuracy", "-- %", "#CBA6F7", "Chronological out-of-sample")
        self.card_best_acc.pack(side="left", fill="x", expand=True, padx=4)

        # 4. Experiments Comparison Table
        table_box = ctk.CTkFrame(self.tab, fg_color="#1E1E2E", corner_radius=6)
        table_box.pack(fill="both", expand=True, padx=10, pady=6)

        lbl_tbl = ctk.CTkLabel(
            table_box,
            text="📑 Tracked Experiment Runs & Validated Model Checkpoints",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#CDD6F4"
        )
        lbl_tbl.pack(anchor="w", padx=12, pady=(6, 2))

        cols = ("run_id", "model", "timestamp", "acc", "prec", "rec", "f1", "roc", "feats", "artifact")
        self.tree = ttk.Treeview(table_box, columns=cols, show="headings", height=7)
        headers = {
            "run_id": "Run ID / Version",
            "model": "Model Architecture",
            "timestamp": "Timestamp",
            "acc": "Accuracy",
            "prec": "Precision",
            "rec": "Recall",
            "f1": "F1-Score",
            "roc": "ROC-AUC",
            "feats": "Features",
            "artifact": "Saved Artifact File"
        }
        for c, h in headers.items():
            self.tree.heading(c, text=h)
            self.tree.column(c, anchor="center", width=100 if c in ["acc", "prec", "rec", "f1", "roc", "feats"] else 140)

        sb = ttk.Scrollbar(table_box, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=(4, 10))
        sb.pack(side="right", fill="y", padx=(0, 10), pady=(4, 10))

        self.tree.bind("<<TreeviewSelect>>", self._on_run_selected)

        # 5. Run Parameters & Metadata Inspector
        self.detail_frame = ctk.CTkFrame(self.tab, fg_color="#181825", corner_radius=6)
        self.detail_frame.pack(fill="x", padx=10, pady=(0, 8))

        self.lbl_detail_title = ctk.CTkLabel(
            self.detail_frame,
            text="🔍 Run Details Inspector: Select a run above to view hyperparameters & dataset provenance",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#CDD6F4"
        )
        self.lbl_detail_title.pack(anchor="w", padx=12, pady=(6, 2))

        self.lbl_detail_params = ctk.CTkLabel(
            self.detail_frame,
            text="Hyperparameters: None selected | Dataset: None | Features: None",
            font=ctk.CTkFont(size=10),
            text_color="#A6ADC8"
        )
        self.lbl_detail_params.pack(anchor="w", padx=12, pady=(0, 8))

        # Initial load
        self._load_experiments()

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

    def _load_experiments(self):
        threading.Thread(target=self._load_exps_bg, daemon=True).start()

    def _load_exps_bg(self):
        exps = self.tracker.get_all_experiments()
        best_run = self.tracker.get_best_experiment('f1_score')
        try:
            self.tab.after(0, self._render_experiments, exps, best_run)
        except Exception:
            pass

    def _render_experiments(self, exps: list, best_run: Optional[dict]):
        self.card_total.v_lbl.configure(text=str(len(exps)))

        if best_run:
            metrics = best_run.get('metrics', {})
            self.card_best_name.v_lbl.configure(text=best_run.get('model_name', 'Random Forest'))
            self.card_best_f1.v_lbl.configure(text=f"{metrics.get('f1_score', 0):.1f}%")
            self.card_best_acc.v_lbl.configure(text=f"{metrics.get('accuracy', 0):.1f}%")
            self.lbl_active_model.configure(
                text=f"ACTIVE PRODUCTION MODEL: {best_run.get('model_name')} (Run ID: {best_run.get('run_id')} | Safe Versioning: ACTIVE)"
            )

        for r in self.tree.get_children():
            self.tree.delete(r)

        for exp in reversed(exps):
            m = exp.get('metrics', {})
            artifact_name = os.path.basename(exp.get('artifact_path', ''))
            self.tree.insert("", "end", values=(
                exp.get('run_id', ''),
                exp.get('model_name', ''),
                exp.get('timestamp', ''),
                f"{m.get('accuracy', 0):.1f}%",
                f"{m.get('precision', 0):.1f}%",
                f"{m.get('recall', 0):.1f}%",
                f"{m.get('f1_score', 0):.1f}%",
                f"{m.get('roc_auc', 0):.1f}%",
                len(exp.get('features', [])),
                artifact_name
            ))

    def _on_run_selected(self, event):
        sel = self.tree.selection()
        if not sel:
            return
        item = self.tree.item(sel[0])
        run_id = item['values'][0]

        exps = self.tracker.get_all_experiments()
        matching = next((e for e in exps if e.get('run_id') == run_id), None)
        if matching:
            params = matching.get('parameters', {})
            feats = matching.get('features', [])
            ds = matching.get('dataset_info', {})
            self.lbl_detail_title.configure(text=f"🔍 Run Details: {run_id} ({matching.get('model_name')})")
            param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
            feat_str = ", ".join(feats)
            self.lbl_detail_params.configure(
                text=f"Hyperparameters: {param_str} | Dataset: {ds} | Features ({len(feats)}): [{feat_str}]"
            )
