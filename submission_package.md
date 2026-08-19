# 🚀 Stock Market AI/ML Screener — Submission Package & Deliverables

> [!NOTE]
> All personal trading account credentials, API keys, client codes, and confidential credentials have been **thoroughly sanitized and masked** across all source code, config files, and build assets before packaging.

---

## 📦 Downloadable Deliverables to Submit

All requested files have been generated, verified, and placed in the project directory as well as the downloadable artifacts folder.

| Deliverable Item | File Name | Size / Format | Direct Download / Access Link |
| :--- | :--- | :--- | :--- |
| **1. Python Source Code** | `StockMarket_AIML_Screener_SourceCode.zip` | `22.6 KB` (.zip) | [Download Source Code ZIP](file:///c:/Users/mail/OneDrive/Desktop/STOCK-PROJECT/StockMarket_AIML_Screener_SourceCode.zip) |
| **2. Standalone Executable** | `StockMarket_AIML_Screener.exe` | `84.8 MB` (.exe) | [Download Executable (.exe)](file:///c:/Users/mail/OneDrive/Desktop/STOCK-PROJECT/dist/StockMarket_AIML_Screener.exe) |
| **3. Screen Recording (MP4 Video)** | `screen_recording.mp4` | `563.9 KB` (.mp4) | [Download Video (MP4)](file:///c:/Users/mail/OneDrive/Desktop/STOCK-PROJECT/screen_recording.mp4) |
| **4. Screen Recording (WebP Video)** | `screen_recording.webp` | `545.0 KB` (.webp) | [Download Video (WebP)](file:///c:/Users/mail/OneDrive/Desktop/STOCK-PROJECT/screen_recording.webp) |
| **5. Screen Recording (Animated GIF)** | `screen_recording.gif` | `374.3 KB` (.gif) | [Download Animation (GIF)](file:///c:/Users/mail/OneDrive/Desktop/STOCK-PROJECT/screen_recording.gif) |

---

## 🎥 Screen Recording Demonstration Overview

The screen recording (`screen_recording.mp4` / `screen_recording.webp` / `screen_recording.gif`) demonstrates the 4 required pillars in action:

1. **Program Execution**:
   - Launches the CustomTkinter dark-mode graphical user interface.
   - Loads timeframe: **1-Minute Candles**, SMMA parameters: **(20 / 120)**, and filtering criteria (LTP ₹30–₹500, Bid/Ask Depth > 10 Lakhs).
2. **Live Stock Screening**:
   - Displays real-time stock updates across representative NSE stocks.
   - Monitors live price fluctuations, bid/ask depth spread, 5-min volume, and LTQ (Last Traded Quantity) volume expansion ratio.
3. **Real-Time Analysis**:
   - Automatically tracks **20 SMMA vs 120 SMMA** crossovers (`BULLISH 20 > 120` or `BEARISH 20 < 120`).
   - Renders live order book depth imbalance charts.
4. **AI/ML Predictions & Results**:
   - **Day 1 Training Workflow**: Evaluates historical 1-min crossover signals, extracts micro-structural features (LTQ ratio, depth imbalance, SMMA spread, volume acceleration), and outputs win probability predictions.
   - **Day 2 Live Inference**: Predicts win probability for live crossovers (Open Position if probability $\ge 60\%$, Avoid Position if $< 60\%$).
   - **Trade Avoidance Explanations**: Details plain-language reasons for trade rejections (e.g. *"Avoided: Insufficient LTQ volume expansion"*, *"Avoided: Heavy ask wall overhead"*).
   - **Performance Stats**: Displays real-time Win Rate %, Loss Rate %, Avoided Trades %, and Net Paper P&L.

---

## 🛡️ Security & Credential Masking Audit Report

A thorough security grep and sanitization pass was performed on the entire codebase prior to archiving:

> [!IMPORTANT]
> - `config/credentials.json`: Masked with standard placeholder tokens (`YOUR_ANGEL_ONE_API_KEY`, `YOUR_CLIENT_CODE`, `YOUR_PASSWORD`, `YOUR_TOTP_KEY`).
> - `config/credentials.example.json`: All personal client codes and credentials replaced with sanitized placeholders.
> - `config/settings.py`: Includes automatic masking helper `get_masked_credentials()` to ensure API keys are never rendered in full on the UI header or log output.
> - `core/broker_streamer.py`: Live Angel One SmartAPI authentication falls back to offline high-fidelity simulator mode when live credentials are empty or invalid.

---

## 📂 Source Code Repository Architecture

```
STOCK-PROJECT/
├── main.py                     # Main application entry point
├── build_exe.py                 # PyInstaller standalone build script
├── requirements.txt            # Python dependencies (CustomTkinter, Scikit-Learn, Pandas, etc.)
├── README.md                   # Full system documentation & usage guide
├── config/
│   ├── credentials.json        # Sanitized user configuration (placeholders)
│   ├── credentials.example.json# Clean credentials template
│   └── settings.py             # System settings & credential masking utilities
├── core/
│   ├── broker_streamer.py      # Real-time market feed streamer & simulator
│   ├── data_fetcher.py         # 1-minute historical data engine
│   ├── indicators.py           # 20/120 SMMA & tick aggregator engine
│   └── paper_trader.py         # Live paper trading & P&L analytics engine
├── ml/
│   ├── crossover_evaluator.py  # AI/ML signal evaluator & feature extraction
│   └── trainer.py              # Day 1 Machine Learning model trainer
└── ui/
    └── main_window.py          # CustomTkinter 3-tab graphical interface
```

---

## 🔨 How to Re-Run or Re-Build Locally

### Running the Python Application
```bash
python main.py
```

### Re-Building the Standalone Executable (.exe)
```bash
python build_exe.py
```
The compiled executable will be output to `dist/StockMarket_AIML_Screener.exe`.
