"""
app.py — Main Entry Point for launching the AI/ML Stock Screener Web Dashboard.

Run this file to start the live web server and automatically open the dashboard in your default browser:
    python app.py
"""

import sys
import os
import webbrowser
import threading
import time

# Ensure project root is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from web.server import start_server, log

def open_browser():
    """Wait 1.5 seconds for server to spin up, then open browser."""
    time.sleep(1.5)
    url = "http://localhost:5000"
    log.info(f"Opening Web Dashboard in default browser: {url}")
    webbrowser.open(url)

if __name__ == "__main__":
    print("=" * 72)
    print("LAUNCHING: AI/ML Stock Screener - Live Web Browser Dashboard")
    print("=" * 72)
    print("URL: http://localhost:5000")
    print("Press Ctrl+C in terminal to stop the server.")
    print("=" * 72)
    print()

    # Open browser in a separate timer thread
    threading.Thread(target=open_browser, daemon=True).start()

    # Start Flask Web Server
    start_server(host="127.0.0.1", port=5000)
