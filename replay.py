"""
replay.py — CLI entry point for Crossover Replay Mode.

Usage:
  python replay.py --date 2025-08-15
  python replay.py --date 2025-08-15 --speed 20
  python replay.py --date 2025-08-15 --ticker TATASTEEL
  python replay.py --help
"""

import sys
import os
import argparse

# Make sure the project root is on path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)


def main():
    parser = argparse.ArgumentParser(
        description="NSE AI Stock Screener — Crossover Replay Mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python replay.py --date 2025-08-15
  python replay.py --date 2025-08-15 --speed 30
  python replay.py --date 2025-08-15 --ticker TATASTEEL --speed 5

Notes:
  - Historical data is loaded from data/history/<date>/<ticker>.csv
  - If no data file exists, synthetic data is used for demonstration
  - Ctrl+C stops replay and shows partial summary
        """
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Date to replay in YYYY-MM-DD format (e.g., 2025-08-15)"
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=10,
        help="Playback speed multiplier — 10 = 10x faster than real-time (default: 10)"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Filter to a single ticker (e.g., TATASTEEL). Default: all tickers"
    )

    args = parser.parse_args()

    # Validate date format
    from datetime import datetime
    try:
        datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        print(f"Error: --date must be in YYYY-MM-DD format, got: {args.date}")
        sys.exit(1)

    if args.speed < 1 or args.speed > 1000:
        print("Error: --speed must be between 1 and 1000")
        sys.exit(1)

    print(f"Starting replay for {args.date} at {args.speed}x speed...")

    from replay.backtest_replay import run_replay
    run_replay(
        date_str=args.date,
        speed=args.speed,
        ticker_filter=args.ticker
    )


if __name__ == "__main__":
    main()
