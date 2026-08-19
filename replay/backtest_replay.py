"""
replay/backtest_replay.py — Crossover Replay Mode for historical backtesting.

FEATURE 8: Crossover Replay Mode

# Replay mode was the hardest part to build — had to refactor fetcher.py to support
# both live and historical modes

How it works:
  1. Load 1-minute OHLCV data for the given date (from data/history/ or synthetic)
  2. Replay candle-by-candle at configurable speed
  3. Run the same screener + ML + ghost + scoring logic on each candle
  4. Show live progress in terminal using Rich
  5. Print summary at the end

Usage:
  python replay.py --date 2025-08-15
  python replay.py --date 2025-08-15 --speed 20
  python replay.py --date 2025-08-15 --ticker TATASTEEL
"""

import sys
import os
import time
import argparse
from collections import deque
from datetime import datetime

# Make sure project root is on path when called as script
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_dir)


def run_replay(date_str: str, speed: int = 10, ticker_filter: str = None):
    """
    Main replay function. Loads data and runs the screener logic on historical candles.
    
    Args:
        date_str: Date to replay in YYYY-MM-DD format
        speed:    Playback multiplier — 10 means 10x faster than real-time
        ticker_filter: If set, only replay this one ticker
    """
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
        from rich.live import Live
        from rich.panel import Panel
        from rich import box
    except ImportError:
        print("Rich not installed. Run: pip install rich")
        sys.exit(1)

    from data.fetcher import fetch_historical_data, get_prepared_data
    from ml.ghost_detector import detect_ghost, ghost_label_with_emoji
    from screener.signal_scorer import score_signal, get_score_label
    from screener.crossover_detector import detect_crossover
    from ml.feature_builder import build_ghost_signal_data
    from config.settings import ACTIVE_NSE_TICKERS, SMMA_FAST_PERIOD, SMMA_SLOW_PERIOD

    console = Console()

    console.print(Panel.fit(
        f"[bold cyan]NSE AI Screener — Replay Mode[/bold cyan]\n"
        f"[white]Date: {date_str} | Speed: {speed}x[/white]",
        border_style="cyan"
    ))

    tickers = ACTIVE_NSE_TICKERS
    if ticker_filter:
        tickers = [t for t in tickers if ticker_filter.upper() in t.upper()]
        if not tickers:
            console.print(f"[red]No ticker matching '{ticker_filter}' found.[/red]")
            sys.exit(1)

    # Load and prepare data for all tickers
    console.print(f"\n[yellow]Loading historical data for {len(tickers)} tickers...[/yellow]")
    ticker_dfs = {}
    for t in tickers:
        sym = t.replace(".NS", "")
        df  = fetch_historical_data(sym, date_str)
        if df is not None and not df.empty:
            ticker_dfs[sym] = df
        else:
            console.print(f"  [red]No data for {sym}[/red]")

    if not ticker_dfs:
        console.print("[red]No data available for replay. Exiting.[/red]")
        sys.exit(1)

    console.print(f"[green]Loaded {len(ticker_dfs)} tickers. Starting replay...[/green]\n")

    # Track state per ticker
    price_history = {sym: deque(maxlen=130) for sym in ticker_dfs}
    smma_20_hist  = {sym: deque(maxlen=10) for sym in ticker_dfs}
    crossover_states = {sym: "NEUTRAL" for sym in ticker_dfs}

    # Results collection
    detected_signals = []
    total_candles = 0

    # Determine replay delay — real trading is 1 candle/minute
    # At 10x speed: sleep 6 seconds per candle. At 60x: 1 second per candle
    delay = max(0.1, 60.0 / speed)

    # Find max candles across all tickers
    max_rows = max(len(df) for df in ticker_dfs.values())

    # Create summary table (updated live)
    def make_summary_table() -> Table:
        t = Table(title="Replay Progress", box=box.SIMPLE_HEAD, show_edge=True)
        t.add_column("Signal", style="cyan")
        t.add_column("Stock",  style="white")
        t.add_column("Type",   style="yellow")
        t.add_column("Score",  style="green")
        t.add_column("Ghost",  style="magenta")
        t.add_column("LTP",    style="white")
        t.add_column("Time",   style="dim")
        for sig in detected_signals[-10:]:  # show last 10
            t.add_row(
                str(len(detected_signals)),
                sig['symbol'],
                sig['signal_type'],
                f"{sig['score']}/100 {sig['score_label']}",
                ghost_label_with_emoji(sig['ghost_status']),
                f"₹{sig['ltp']:.2f}",
                sig['candle_time']
            )
        return t

    console.print(f"[dim]Replaying {max_rows} candles at {speed}x speed (press Ctrl+C to stop early)[/dim]\n")

    try:
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[cyan]{task.completed}/{task.total} candles"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Replaying session...", total=max_rows)

            for candle_idx in range(max_rows):
                total_candles += 1

                for sym, df in ticker_dfs.items():
                    if candle_idx >= len(df):
                        continue

                    row = df.iloc[candle_idx]
                    close = float(row['Close'])
                    vol   = float(row.get('Volume', 0))

                    price_history[sym].append(close)
                    prices_list = list(price_history[sym])

                    # Compute SMMA values from accumulated history
                    from core.indicators import calculate_smma
                    smma_20  = calculate_smma(prices_list, SMMA_FAST_PERIOD)
                    smma_120 = calculate_smma(prices_list, SMMA_SLOW_PERIOD)
                    smma_20_hist[sym].append(smma_20)

                    # Check for crossover
                    prev_rel = crossover_states.get(sym, "NEUTRAL")
                    was_above = prev_rel == "ABOVE"
                    is_above  = smma_20 > smma_120

                    if smma_20 > smma_120:
                        crossover_states[sym] = "ABOVE"
                    elif smma_20 < smma_120:
                        crossover_states[sym] = "BELOW"

                    # Detect event
                    if prev_rel != "NEUTRAL" and was_above != is_above:
                        sig_type = "BUY" if is_above else "SELL"
                        candle_time = str(row.get('Datetime', f"Candle {candle_idx}"))[:16]

                        # Ghost detection
                        ghost_data = {
                            'spread':        0.10,
                            'avg_spread_20m': 0.05,
                            'vol_5m':        vol,
                            'vol_60m':       vol * 12 + 1
                        }
                        from ml.ghost_detector import detect_ghost
                        ghost = detect_ghost(ghost_data)

                        # Score
                        from screener.signal_scorer import score_signal
                        sig_score = score_signal(
                            symbol=sym,
                            signal_type=sig_type,
                            vol_5m=vol,
                            vol_60m=vol * 12,
                            smma_20_history=list(smma_20_hist[sym]),
                            bid_qty=1_200_000.0,
                            ask_qty=1_100_000.0,
                            closed_trades=[]
                        )

                        detected_signals.append({
                            'symbol':      sym,
                            'signal_type': sig_type,
                            'ltp':         close,
                            'smma_20':     smma_20,
                            'smma_120':    smma_120,
                            'ghost_status': ghost,
                            'score':       sig_score.score,
                            'score_label': sig_score.label,
                            'candle_time': candle_time,
                            'candle_idx':  candle_idx
                        })

                        console.print(
                            f"  ⚡ [bold]{sig_type}[/bold] signal: [cyan]{sym}[/cyan] @ ₹{close:.2f} "
                            f"| Score: {sig_score.score} {sig_score.label} "
                            f"| Ghost: {ghost_label_with_emoji(ghost)} "
                            f"| {candle_time}"
                        )

                progress.update(task, advance=1)
                time.sleep(delay)

    except KeyboardInterrupt:
        console.print("\n[yellow]Replay stopped early by user.[/yellow]")

    # Final summary
    console.print("\n")
    clean_signals = [s for s in detected_signals if s['ghost_status'] == 'CLEAN']
    strong_signals = [s for s in clean_signals if s['score'] >= 80]

    summary = Table(title="Replay Summary", box=box.ROUNDED, show_header=True)
    summary.add_column("Metric",      style="cyan", width=30)
    summary.add_column("Value",       style="white", width=20)
    summary.add_row("Total Candles Replayed",  str(total_candles))
    summary.add_row("Total Signals Detected",  str(len(detected_signals)))
    summary.add_row("Clean Signals",           str(len(clean_signals)))
    summary.add_row("Ghost/Fake Signals",      str(len(detected_signals) - len(clean_signals)))
    summary.add_row("Strong Signals (80+)",    str(len(strong_signals)))
    if detected_signals:
        best = max(detected_signals, key=lambda x: x['score'])
        summary.add_row("Best Signal",  f"{best['symbol']} {best['signal_type']} (Score: {best['score']})")
    console.print(summary)
    console.print("\n[bold green]Replay complete![/bold green]\n")

    return detected_signals
