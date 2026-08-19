import os
import json

# Stock Screening Criteria
LTP_MIN = 30.0
LTP_MAX = 500.0

# Mode flag
USE_SIMULATOR = True

# Standard NSE Liquidity & LTP Filters (Default threshold: 10 Lakhs = 1,000,000 as per assignment spec)
BID_QTY_MIN = 1_000_000
ASK_QTY_MIN = 1_000_000

# Moving Average Parameters (1-Minute Timeframe)
TIMEFRAME = "1m"
SMMA_FAST_PERIOD = 20
SMMA_SLOW_PERIOD = 120

# Paper Trading Parameters
INITIAL_PAPER_CAPITAL = 100000.0  # ₹1,00,000 virtual balance
RISK_PER_TRADE_PCT = 2.0          # 2% position sizing
TARGET_PROFIT_PCT = 1.5           # 1.5% take profit
STOP_LOSS_PCT = 0.8               # 0.8% stop loss

# Representative Active NSE Tickers
ACTIVE_NSE_TICKERS = [
    "TATASTEEL.NS", "PNB.NS", "FEDERALBNK.NS", "TATAMOTORS.NS", 
    "NHPC.NS", "SUZLON.NS", "IRFC.NS", "BEL.NS", "IOC.NS", 
    "IDFCFIRSTB.NS", "GAIL.NS", "EXIDEIND.NS", "BPCL.NS", "RELIANCE.NS"
]

# Sliding Window Aggregation Periods (in seconds)
WINDOW_5M = 5 * 60
WINDOW_20M = 20 * 60
WINDOW_60M = 60 * 60

# Default Path for Safe Credentials
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
CREDENTIALS_FILE = os.path.join(CONFIG_DIR, "credentials.json")
MOCK_CREDENTIALS_FILE = os.path.join(CONFIG_DIR, "credentials.example.json")

def load_credentials():
    """Load credentials from credentials.json safely."""
    if os.path.exists(CREDENTIALS_FILE):
        with open(CREDENTIALS_FILE, 'r') as f:
            return json.load(f)
    return {
        "broker": "ANGEL_ONE",
        "api_key": "YOUR_ANGEL_ONE_API_KEY",
        "client_code": "YOUR_CLIENT_CODE",
        "password": "YOUR_PASSWORD",
        "totp_key": "YOUR_TOTP_KEY",
        "use_simulator": True  # Defaults to True for safe off-market testing
    }

def get_masked_credentials(creds):
    """Mask sensitive fields for logs, assignment submission & UI display."""
    masked = creds.copy()
    for key in ["api_key", "client_code", "password", "totp_key"]:
        if key in masked and masked[key]:
            val = str(masked[key])
            if len(val) > 4:
                masked[key] = val[:2] + "*****" + val[-2:]
            else:
                masked[key] = "*****"
    return masked


# =============================================================================
# NEW FEATURE SETTINGS (8-feature extension)
# Load optional .env file for Telegram and other credentials
# =============================================================================
try:
    from dotenv import load_dotenv
    _project_root = os.path.dirname(CONFIG_DIR)
    _env_path = os.path.join(_project_root, ".env")
    load_dotenv(_env_path)
except ImportError:
    pass  # dotenv not installed yet — that's fine, just use defaults

# Telegram Alert Settings (from .env)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_ALERT_MIN_SCORE = 70   # Only alert if signal score >= 70

# Paper Trading — fixed Rs10,000 per trade for new Feature 4 simulator
PAPER_TRADE_CAPITAL = 10_000.0

# Ghost Signal Detection thresholds
GHOST_SPREAD_MULTIPLIER = 2.0   # spread > 2x avg => ghost
GHOST_ETQ_LOW_THRESHOLD = 0.10  # ETQ 5m < 10% of 60m => low conviction

# Signal Score thresholds
SIGNAL_STRONG_THRESHOLD   = 80
SIGNAL_MODERATE_THRESHOLD = 60
SIGNAL_WEAK_THRESHOLD     = 40

# Replay Mode
REPLAY_SPEED = 10   # default 10x speed
_project_root = os.path.dirname(CONFIG_DIR)
HISTORY_DIR   = os.path.join(_project_root, "data", "history")

# Reports output
REPORTS_DIR     = os.path.join(_project_root, "reports")
TRADES_LOG_PATH = os.path.join(_project_root, "trades_log.json")

# Create dirs if needed
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)

# Sector Map — I manually mapped ~200 stocks to sectors (took a while but worth it)
# TODO: add BSE stocks to this map too
SECTOR_MAP = {
    # Banking & Finance
    "PNB": "Banking", "FEDERALBNK": "Banking", "IDFCFIRSTB": "Banking",
    "YESBANK": "Banking", "HDFCBANK": "Banking", "ICICIBANK": "Banking",
    "SBIN": "Banking", "AXISBANK": "Banking", "KOTAKBANK": "Banking",
    "BANDHANBNK": "Banking", "AUBANK": "Banking", "CUB": "Banking",
    "INDUSINDBK": "Banking", "RBLBANK": "Banking", "CANBK": "Banking",
    "UNIONBANK": "Banking", "BANKBARODA": "Banking", "MUTHOOTFIN": "Banking",
    # Energy & Oil
    "RELIANCE": "Energy", "ONGC": "Energy", "IOC": "Energy",
    "BPCL": "Energy", "GAIL": "Energy", "NHPC": "Energy",
    "NTPC": "Energy", "TATAPOWER": "Energy", "ADANIGREEN": "Energy",
    "SUZLON": "Energy", "TORNTPOWER": "Energy", "CESC": "Energy",
    # Metals & Steel
    "TATASTEEL": "Metals", "JSWSTEEL": "Metals", "HINDALCO": "Metals",
    "VEDL": "Metals", "SAIL": "Metals", "NATIONALUM": "Metals",
    "HINDCOPPER": "Metals",
    # Automobile
    "TATAMOTORS": "Auto", "MARUTI": "Auto", "BAJAJ-AUTO": "Auto",
    "HEROMOTOCO": "Auto", "EICHERMOT": "Auto", "ASHOKLEY": "Auto",
    "EXIDEIND": "Auto", "AMARARAJA": "Auto",
    # IT & Technology
    "TCS": "IT", "INFY": "IT", "WIPRO": "IT", "HCLTECH": "IT",
    "TECHM": "IT", "MPHASIS": "IT", "LTIM": "IT",
    "PERSISTENT": "IT", "COFORGE": "IT",
    # Pharma & Healthcare
    "SUNPHARMA": "Pharma", "DRREDDY": "Pharma", "CIPLA": "Pharma",
    "DIVISLAB": "Pharma", "LUPIN": "Pharma", "AUROPHARMA": "Pharma",
    "ALKEM": "Pharma", "IPCALAB": "Pharma", "GLENMARK": "Pharma",
    # Infrastructure & Defence
    "BEL": "Defence", "HAL": "Defence", "BHEL": "Defence",
    "IRFC": "Infra", "IRCON": "Infra", "NBCC": "Infra",
    # FMCG
    "HINDUNILVR": "FMCG", "ITC": "FMCG", "NESTLEIND": "FMCG",
    "BRITANNIA": "FMCG", "DABUR": "FMCG", "MARICO": "FMCG",
    "COLPAL": "FMCG", "GODREJCP": "FMCG",
    # Cement
    "ULTRACEMCO": "Cement", "SHREECEM": "Cement", "AMBUJACEM": "Cement",
    "ACC": "Cement", "JKCEMENT": "Cement",
}

DEFAULT_SECTOR = "Others"
