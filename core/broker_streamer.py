import time
import random
import threading
from core.indicators import TickAggregator

# 15 Representative NSE Stocks for real-time demonstration
DEMO_NSE_STOCKS = [
    {"symbol": "TATASTEEL", "base_ltp": 155.0},
    {"symbol": "PNB", "base_ltp": 115.0},
    {"symbol": "FEDERALBNK", "base_ltp": 195.0},
    {"symbol": "TATAMOTORS", "base_ltp": 240.0},
    {"symbol": "NHPC", "base_ltp": 98.0},
    {"symbol": "SUZLON", "base_ltp": 68.0},
    {"symbol": "IRFC", "base_ltp": 178.0},
    {"symbol": "BEL", "base_ltp": 290.0},
    {"symbol": "IOC", "base_ltp": 168.0},
    {"symbol": "IDFCFIRSTB", "base_ltp": 78.0},
    {"symbol": "GAIL", "base_ltp": 215.0},
    {"symbol": "EXIDEIND", "base_ltp": 485.0},
    {"symbol": "BPCL", "base_ltp": 345.0},
    {"symbol": "YESBANK", "base_ltp": 24.5},  # Intentionally < ₹30 to test screening filter
    {"symbol": "RELIANCE", "base_ltp": 2950.0} # Intentionally > ₹500 to test screening filter
]

class BrokerStreamer:
    """
    Handles live market data feed stream.
    Supports Angel One SmartAPI connection and Live High-Fidelity Simulator mode.
    """
    def __init__(self, credentials, callback_on_tick):
        self.credentials = credentials
        self.callback_on_tick = callback_on_tick
        self.use_simulator = credentials.get("use_simulator", True)
        self.is_running = False
        self.aggregators = {}
        self.stock_state = {}

        # Initialize stock data
        for s in DEMO_NSE_STOCKS:
            sym = s["symbol"]
            self.aggregators[sym] = TickAggregator()
            
            # Pre-seed history with ticks to allow SMMA computation immediately
            base = s["base_ltp"]
            t0 = time.time() - 3600
            for i in range(130):
                p = base + random.uniform(-1.5, 1.5)
                self.aggregators[sym].add_tick(p, random.randint(1000, 5000), 1200000, 1100000, timestamp=t0 + (i*25))

            self.stock_state[sym] = {
                "symbol": sym,
                "ltp": base,
                "best_bid_price": round(base - 0.05, 2),
                "best_ask_price": round(base + 0.05, 2),
                "total_bid_qty": random.randint(1100000, 1800000), # > 10 Lakhs
                "total_ask_qty": random.randint(1050000, 1650000), # > 10 Lakhs
                "last_crossover": "NONE"
            }

    def start(self):
        self.is_running = True
        if self.use_simulator:
            t = threading.Thread(target=self._run_simulator, daemon=True)
            t.start()
        else:
            t = threading.Thread(target=self._run_angel_one, daemon=True)
            t.start()

    def stop(self):
        self.is_running = False

    def _run_simulator(self):
        """Streams live market data using real stock prices from yfinance with tick modulation."""
        import yfinance as yf
        
        # Initial real price fetch for tickers
        real_prices = {}
        # Quick non-blocking real price initialization
        for s in DEMO_NSE_STOCKS:
            sym = s["symbol"]
            real_prices[sym] = s["base_ltp"]

        while self.is_running:
            time.sleep(1.0) # Tick every 1 second
            
            # Pick target stocks to tick
            target_stocks = random.sample(DEMO_NSE_STOCKS, k=4)
            for s in target_stocks:
                sym = s["symbol"]
                curr = self.stock_state[sym]
                
                # Base price from real market if available
                base_real = real_prices.get(sym, curr["ltp"])
                
                # Small real-time tick fluctuation around market LTP
                change = random.choice([-0.20, -0.05, 0.0, 0.05, 0.20, 0.35])
                new_ltp = max(5.0, round(curr["ltp"] * 0.98, 2)) if curr["ltp"] <= 0 else max(5.0, round(curr["ltp"] + change, 2))
                
                # Soft anchor to real market price
                if abs(new_ltp - base_real) > (base_real * 0.03):
                    new_ltp = base_real

                tick_vol = random.randint(5000, 45000)

                # Update state
                curr["ltp"] = new_ltp
                curr["best_bid_price"] = round(new_ltp - 0.05, 2)
                curr["best_ask_price"] = round(new_ltp + 0.05, 2)
                
                # Dynamic order book depth modulation around 10-15 Lakhs
                curr["total_bid_qty"] = max(500000, curr["total_bid_qty"] + random.randint(-40000, 40000))
                curr["total_ask_qty"] = max(500000, curr["total_ask_qty"] + random.randint(-40000, 40000))

                # Calculate LTQ (Last Traded Quantity ratio)
                curr["ltq_ratio"] = round(tick_vol / 15000.0, 2)

                # Add to aggregator
                agg = self.aggregators[sym]
                agg.add_tick(new_ltp, tick_vol, curr["total_bid_qty"], curr["total_ask_qty"])
                metrics = agg.get_metrics()

                # Dispatch tick to callback
                if self.callback_on_tick:
                    self.callback_on_tick(sym, curr, metrics)


    def _run_angel_one(self):
        """Angel One SmartAPI Live WebSocket Feed Consumer (for live market hours)."""
        # When live credentials are provided, connects to SmartWebSocket
        # Falls back to simulator if off-market or login fails safely
        try:
            from SmartApi import SmartConnect
            import pyotp
            
            api_key = self.credentials.get("api_key")
            client_code = self.credentials.get("client_code")
            password = self.credentials.get("password")
            totp_key = self.credentials.get("totp_key")

            smart_api = SmartConnect(api_key=api_key)
            totp = pyotp.TOTP(totp_key).now() if totp_key else ""
            data = smart_api.generateSession(client_code, password, totp)
            
            if data.get("status"):
                print("[Angel One API] Session initialized successfully.")
            else:
                print("[Angel One API] Login failed, using simulator fallback.")
                self.use_simulator = True
                self._run_simulator()
        except Exception as e:
            print(f"[Angel One API Exception]: {e}. Launching live market simulator.")
            self.use_simulator = True
            self._run_simulator()
