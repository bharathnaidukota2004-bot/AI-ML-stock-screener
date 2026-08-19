"""
reports/pdf_generator.py — Generates a professional PDF report at session end.

FEATURE 7: PDF Report Generator
The report includes:
  - Session date, time, total stocks scanned
  - All crossovers detected with scores and ghost status
  - Paper trading P&L summary
  - Top 3 best signals of the session
"""

import os
import re
from datetime import datetime
from typing import Optional

from config.settings import REPORTS_DIR
from utils.helpers import session_filename_stamp, format_inr


def clean_pdf_text(text: str) -> str:
    """Sanitizes text for standard FPDF Helvetica font (removes emojis, replaces Rupee symbol and unicode dashes)."""
    text = str(text)
    text = text.replace("₹", "Rs.")
    text = text.replace("—", "-")
    text = text.replace("🟢", "[STRONG]").replace("🟡", "[MODERATE]").replace("🟠", "[WEAK]").replace("🔴", "[AVOID]")
    text = text.replace("✅", "[CLEAN]").replace("👻", "[GHOST]").replace("⚠️", "[LOW_CONVICTION]")
    # Strip any remaining non-latin1 characters
    return text.encode("latin-1", "ignore").decode("latin-1")


class ReportGenerator:
    """
    Generates a PDF session report using fpdf2.
    Call generate(session_data) at session end or when the user presses the button.
    """

    def generate(self, session_data: dict) -> Optional[str]:
        try:
            from fpdf import FPDF
        except ImportError:
            print("[ReportGenerator] fpdf2 not installed. Run: pip install fpdf2")
            return None

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        self._draw_header(pdf, session_data)
        self._draw_summary_section(pdf, session_data)
        self._draw_crossover_table(pdf, session_data)
        self._draw_paper_trading_section(pdf, session_data)
        self._draw_footer(pdf)

        os.makedirs(REPORTS_DIR, exist_ok=True)
        filename = f"session_{session_filename_stamp()}.pdf"
        filepath = os.path.join(REPORTS_DIR, filename)
        pdf.output(filepath)
        print(f"[ReportGenerator] Report saved: {filepath}")
        return filepath

    def _draw_header(self, pdf, session_data: dict):
        pdf.set_font("Helvetica", "B", 18)
        pdf.set_text_color(30, 30, 80)
        pdf.cell(0, 12, clean_pdf_text("NSE AI Stock Screener - Session Report"), ln=True, align="C")

        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(80, 80, 80)
        date_str = session_data.get('date', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        pdf.cell(0, 8, clean_pdf_text(f"Generated: {date_str} | Market Universe: {session_data.get('stocks_scanned', 0)} Stocks"), ln=True, align="C")

        pdf.set_draw_color(100, 100, 200)
        pdf.set_line_width(0.8)
        pdf.line(10, pdf.get_y() + 2, 200, pdf.get_y() + 2)
        pdf.ln(6)

    def _draw_summary_section(self, pdf, session_data: dict):
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(30, 30, 80)
        pdf.cell(0, 8, "1. Executive Summary", ln=True)

        stats = session_data.get('paper_stats', {})
        pnl = stats.get('net_pnl', 0.0)
        pnl_str = f"+Rs.{pnl:,.2f}" if pnl >= 0 else f"-Rs.{abs(pnl):,.2f}"

        items = [
            ("Stocks Scanned", str(session_data.get('stocks_scanned', 0))),
            ("Crossovers Detected", str(len(session_data.get('crossovers', [])))),
            ("Paper Trades Closed", str(stats.get('total_closed', 0))),
            ("Win Rate", f"{stats.get('win_rate_pct', 0.0)}%"),
            ("Net P&L", pnl_str),
        ]

        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(40, 40, 40)
        for label, val in items:
            pdf.cell(50, 6, clean_pdf_text(label + ":"), border=0)
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 6, clean_pdf_text(val), ln=True)
            pdf.set_font("Helvetica", "", 10)

        pdf.ln(4)

    def _draw_crossover_table(self, pdf, session_data: dict):
        crossovers = session_data.get('crossovers', [])
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(30, 30, 80)
        pdf.cell(0, 8, "2. Active Market Signals", ln=True)

        if not crossovers:
            pdf.set_font("Helvetica", "I", 10)
            pdf.cell(0, 6, "No crossover signals active at current snapshot.", ln=True)
            pdf.ln(4)
            return

        col_widths = [25, 20, 25, 30, 30, 30, 30]
        headers = ["Symbol", "Signal", "LTP", "SMMA 20", "SMMA 120", "Ghost Status", "Score"]
        
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(60, 60, 160)
        pdf.set_text_color(255, 255, 255)
        for w, h in zip(col_widths, headers):
            pdf.cell(w, 7, clean_pdf_text(h), border=1, fill=True, align="C")
        pdf.ln()

        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(40, 40, 40)
        for i, cross in enumerate(crossovers[:15]):
            fill = (245, 247, 250) if i % 2 == 0 else (255, 255, 255)
            pdf.set_fill_color(*fill)
            row_vals = [
                str(cross.get('symbol', '-')),
                str(cross.get('signal', '-')),
                f"Rs.{cross.get('ltp', 0.0):.2f}",
                f"{cross.get('smma_20', 0.0):.2f}",
                f"{cross.get('smma_120', 0.0):.2f}",
                str(cross.get('ghost', '-')),
                f"{cross.get('score', 0)}/100"
            ]
            for w, val in zip(col_widths, row_vals):
                pdf.cell(w, 6, clean_pdf_text(val), border=1, fill=True, align="C")
            pdf.ln()
        pdf.ln(4)

    def _draw_paper_trading_section(self, pdf, session_data: dict):
        trades = session_data.get('closed_trades', [])
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(30, 30, 80)
        pdf.cell(0, 8, "3. Closed Paper Trades", ln=True)

        if not trades:
            pdf.set_font("Helvetica", "I", 10)
            pdf.cell(0, 6, "No closed paper trades recorded yet.", ln=True)
            pdf.ln(4)
            return

        col_widths = [30, 20, 35, 35, 35, 35]
        headers = ["Symbol", "Side", "Entry Price", "Exit Price", "P&L", "Reason"]
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(60, 60, 160)
        pdf.set_text_color(255, 255, 255)
        for w, h in zip(col_widths, headers):
            pdf.cell(w, 7, clean_pdf_text(h), border=1, fill=True, align="C")
        pdf.ln()

        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(40, 40, 40)
        for i, t in enumerate(trades[:10]):
            fill = (245, 247, 250) if i % 2 == 0 else (255, 255, 255)
            pdf.set_fill_color(*fill)
            row_vals = [
                str(t.get('symbol', '-')),
                str(t.get('signal_type', '-')),
                f"Rs.{t.get('entry_price', 0.0):.2f}",
                f"Rs.{t.get('exit_price', 0.0):.2f}",
                f"Rs.{t.get('pnl', 0.0):.2f}",
                str(t.get('status', '-'))
            ]
            for w, val in zip(col_widths, row_vals):
                pdf.cell(w, 6, clean_pdf_text(val), border=1, fill=True, align="C")
            pdf.ln()
        pdf.ln(4)

    def _draw_footer(self, pdf):
        pdf.set_y(-15)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(0, 10, "Generated automatically by NSE AI Stock Screener Engine", align="C")
