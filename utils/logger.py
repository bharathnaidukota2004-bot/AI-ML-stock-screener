"""
utils/logger.py — Loguru-based logging setup.
I switched to loguru because standard logging is too verbose to set up every time.
"""

import sys
from loguru import logger

# Remove default loguru handler and set a cleaner format
logger.remove()

# Console: colored, compact (utf-8 safe for Windows console)
logger.add(
    sys.stdout,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
    colorize=True,
    catch=True
)

# File: full debug log, rotating daily
logger.add(
    "screener_debug.log",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
    rotation="1 day",
    retention="7 days",
    compression="zip"
)

def get_logger(name: str):
    """Get a named logger — keeps module names visible in output."""
    return logger.bind(name=name)
