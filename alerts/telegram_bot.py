"""
alerts/telegram_bot.py — Sends Telegram alerts for high-quality crossover signals.

FEATURE 3: Telegram Alert Bot

# I added this so I could get alerts on my phone while testing — works great!
# Just set up a bot with @BotFather and put the token in .env

Uses synchronous requests (not async python-telegram-bot) so it plays nicely with tkinter.
Runs in a background thread so the GUI never blocks.

Setup:
  1. Open Telegram, search for @BotFather
  2. Send /newbot, follow prompts, copy the token
  3. Send a message to your bot, then go to:
     https://api.telegram.org/bot<TOKEN>/getUpdates
     to find your chat_id
  4. Add both to your .env file
"""

import time
import threading
import requests
from typing import Optional
from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TELEGRAM_ALERT_MIN_SCORE
from utils.logger import get_logger

log = get_logger("TelegramAlerter")

# Cooldown: don't spam the same stock within 10 minutes
COOLDOWN_SECONDS = 10 * 60


class TelegramAlerter:
    """
    Sends formatted Telegram messages for crossover signals.
    Wraps all API calls in try/except — if Telegram fails, we just log and continue.
    The app should NEVER crash because of a failed notification.
    """

    def __init__(self):
        self.bot_token  = TELEGRAM_BOT_TOKEN
        self.chat_id    = TELEGRAM_CHAT_ID
        self.is_enabled = bool(self.bot_token and self.chat_id)
        self.last_alert_times = {}  # {symbol: timestamp}

        if self.is_enabled:
            log.info("Telegram alerts enabled. Bot token configured.")
        else:
            log.warning("Telegram not configured. Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to .env")

    def _is_on_cooldown(self, symbol: str) -> bool:
        """Check if we sent an alert for this stock within the cooldown window."""
        last = self.last_alert_times.get(symbol, 0)
        return (time.time() - last) < COOLDOWN_SECONDS

    def _build_message(self, signal: dict) -> str:
        """
        Build the formatted Telegram message from signal data.
        Matches the spec format from the assignment.
        """
        sym     = signal.get('symbol', 'UNKNOWN')
        sig     = signal.get('signal_type', 'BUY')
        score   = signal.get('score', 0)
        label   = signal.get('score_label', '🟢 STRONG')
        ltp     = signal.get('ltp', 0.0)
        smma_20 = signal.get('smma_20', 0.0)
        smma_120 = signal.get('smma_120', 0.0)
        vol_5m  = signal.get('vol_5m', 0)
        ghost   = signal.get('ghost_status', 'CLEAN')
        reason  = signal.get('reason', 'Strong signal detected')

        sig_emoji = "🟢" if sig == "BUY" else "🔴"
        ghost_display = "CLEAN ✅" if ghost == "CLEAN" else f"⚠️ {ghost}"

        msg = (
            f"🚨 *CROSSOVER ALERT — {sym}*\n"
            f"─────────────────────────────\n"
            f"Signal   : *{sig}* {sig_emoji}\n"
            f"Score    : {score}/100 ({label})\n"
            f"LTP      : ₹{ltp:.2f}\n"
            f"SMMA 20  : {smma_20:.2f}\n"
            f"SMMA 120 : {smma_120:.2f}\n"
            f"ETQ 5min : {int(vol_5m):,}\n"
            f"Ghost    : {ghost_display}\n"
            f"Reason   : {reason}\n"
            f"─────────────────────────────\n"
            f"⚠️ _This is not financial advice._"
        )
        return msg

    def _send_via_requests(self, message: str) -> bool:
        """
        Calls the Telegram Bot API directly via requests.
        Returns True if successful, False if it fails for any reason.
        """
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id":    self.chat_id,
            "text":       message,
            "parse_mode": "Markdown"
        }
        try:
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                return True
            else:
                log.warning(f"Telegram API returned {resp.status_code}: {resp.text[:100]}")
                return False
        except requests.exceptions.Timeout:
            log.warning("Telegram request timed out — skipping alert")
            return False
        except Exception as e:
            log.warning(f"Telegram request failed: {e}")
            return False

    def send_alert(self, signal: dict) -> None:
        """
        Send an alert for a crossover signal.
        Runs in background thread — never blocks the GUI.
        Respects:
          - Minimum score threshold (default 70)
          - 10-minute cooldown per stock
          - Telegram not configured → silently skip
        """
        if not self.is_enabled:
            return

        symbol = signal.get('symbol', '')
        score  = signal.get('score', 0)

        # Only alert for strong enough signals
        if score < TELEGRAM_ALERT_MIN_SCORE:
            log.debug(f"Score {score} < {TELEGRAM_ALERT_MIN_SCORE} for {symbol}, skipping alert")
            return

        # Cooldown check
        if self._is_on_cooldown(symbol):
            log.debug(f"{symbol} is on cooldown, skipping alert")
            return

        # Send in background so GUI doesn't freeze
        def _send():
            try:
                message = self._build_message(signal)
                success = self._send_via_requests(message)
                if success:
                    self.last_alert_times[symbol] = time.time()
                    log.info(f"[Telegram] Alert sent for {symbol} (score={score})")
            except Exception as e:
                # Never crash the app because of a notification failure
                log.warning(f"Telegram alert thread error: {e}")

        threading.Thread(target=_send, daemon=True).start()

    def send_test_message(self) -> bool:
        """
        Send a test message to verify the bot is configured correctly.
        Returns True if it worked.
        """
        if not self.is_enabled:
            return False
        msg = "✅ NSE AI Screener connected! Telegram alerts are working."
        return self._send_via_requests(msg)
