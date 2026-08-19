"""
ui/sector_heatmap.py — Sector-grouped signal heatmap panel.

FEATURE 5: Sector Heatmap

Groups all screened stocks by their sector (from SECTOR_MAP in config.py)
and shows which sectors have the most crossover activity and average signal score.

Displayed as a colored Treeview table inside a dedicated tab.
Refreshes every 30 seconds.

# I manually mapped ~200 stocks to sectors — took a while but worth it
"""

import tkinter as tk
from tkinter import ttk
from typing import Dict, List
from collections import defaultdict
from config.settings import SECTOR_MAP, DEFAULT_SECTOR


class SectorHeatmap:
    """
    Manages the sector heatmap display.
    Keeps track of signals per sector and renders them as a color-coded table.
    """

    def __init__(self, parent_frame):
        """
        Args:
            parent_frame: the CTkFrame or Frame to place the heatmap inside.
        """
        self.parent = parent_frame
        self.sector_signals: Dict[str, List[dict]] = defaultdict(list)
        self._build_table()

    def _build_table(self):
        """Create the Treeview table with sector, signals count, and avg score."""
        # Style for the heatmap table
        style = ttk.Style()
        style.configure(
            "Heatmap.Treeview",
            background="#181825",
            foreground="#CDD6F4",
            rowheight=28,
            fieldbackground="#181825",
            font=("Segoe UI", 10)
        )
        style.configure(
            "Heatmap.Treeview.Heading",
            background="#313244",
            foreground="#F5E0DC",
            font=("Segoe UI", 10, "bold")
        )
        style.map("Heatmap.Treeview", background=[("selected", "#45475A")])

        columns = ("sector", "signals", "avg_score", "bar", "top_stock")
        self.tree = ttk.Treeview(
            self.parent,
            columns=columns,
            show="headings",
            style="Heatmap.Treeview",
            selectmode="browse"
        )

        self.tree.heading("sector",    text="🗂️ Sector")
        self.tree.heading("signals",   text="📊 Signals")
        self.tree.heading("avg_score", text="⭐ Avg Score")
        self.tree.heading("bar",       text="Activity Bar")
        self.tree.heading("top_stock", text="🏆 Top Stock")

        self.tree.column("sector",    anchor="w",      width=160)
        self.tree.column("signals",   anchor="center", width=90)
        self.tree.column("avg_score", anchor="center", width=120)
        self.tree.column("bar",       anchor="w",      width=200)
        self.tree.column("top_stock", anchor="center", width=120)

        # Color tags: green for high activity, yellow for medium, grey for low
        self.tree.tag_configure("high",   background="#1E3A2F", foreground="#A6E3A1")
        self.tree.tag_configure("medium", background="#2E2A1B", foreground="#F9E2AF")
        self.tree.tag_configure("low",    background="#1E1E2E", foreground="#A6ADC8")

        sb = ttk.Scrollbar(self.parent, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        sb.pack(side="right", fill="y", pady=5)

    def add_signal(self, symbol: str, score: int, signal_type: str, ltp: float):
        """
        Register a new crossover signal for a stock.
        The sector is looked up automatically from SECTOR_MAP.
        """
        sector = SECTOR_MAP.get(symbol, DEFAULT_SECTOR)
        self.sector_signals[sector].append({
            'symbol':      symbol,
            'score':       score,
            'signal_type': signal_type,
            'ltp':         ltp
        })

    def refresh(self):
        """Re-render the heatmap table with current signal counts."""
        # Clear existing rows
        for row in self.tree.get_children():
            self.tree.delete(row)

        if not self.sector_signals:
            # Show placeholder when no signals yet
            self.tree.insert("", "end", values=(
                "Waiting for crossover signals...", "-", "-", "", "-"
            ), tags=("low",))
            return

        # Find max signals for bar scaling
        max_signals = max(len(sigs) for sigs in self.sector_signals.values())

        # Sort by signal count descending
        sorted_sectors = sorted(
            self.sector_signals.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )

        for sector, signals in sorted_sectors:
            count     = len(signals)
            avg_score = round(sum(s['score'] for s in signals) / count) if signals else 0
            
            # Activity bar (Unicode blocks scaled to signal count)
            bar_len   = max(1, int((count / max_signals) * 12))
            bar       = "█" * bar_len + "░" * (12 - bar_len)

            # Top stock by score
            top = max(signals, key=lambda x: x['score']) if signals else {}
            top_stock = top.get('symbol', '-')

            # Score label
            if avg_score >= 80:
                score_label = f"{avg_score} 🟢"
                tag = "high"
            elif avg_score >= 60:
                score_label = f"{avg_score} 🟡"
                tag = "medium"
            else:
                score_label = f"{avg_score} 🟠"
                tag = "low"

            if count >= 3:
                tag = "high"
            elif count >= 2:
                tag = "medium"

            self.tree.insert("", "end", values=(
                sector, str(count), score_label, bar, top_stock
            ), tags=(tag,))

    def clear_signals(self):
        """Reset all signal data — call at session start."""
        self.sector_signals.clear()
        self.refresh()
