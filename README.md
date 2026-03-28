#  FreqTrade AI Trading Bot

> A modular, AI-powered cryptocurrency trading bot built on top of [Freqtrade](https://github.com/freqtrade/freqtrade), featuring a native Windows desktop interface built with PyQt6. Designed with a **ML-first, Neural-Networks-ready** architecture that allows seamless model upgrades without touching strategy code.

---

##  Table of Contents

- [About the Project](#about-the-project)
- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Tech Stack](#tech-stack)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [How to Launch](#how-to-launch)
- [Project Structure](#project-structure)
- [Supported Exchanges](#supported-exchanges)
- [Trading Strategies](#trading-strategies)
- [ML Model Upgrade Path](#ml-model-upgrade-path)
- [Backtesting](#backtesting)
- [Dry Run (Paper Trading)](#dry-run-paper-trading)
- [Risk Management](#risk-management)
- [Roadmap](#roadmap)
- [Security](#security)
- [Contributing](#contributing)
- [License](#license)
- [Version History](#version-history)

---

## About the Project

**FreqTrade AI Bot** is a personal algorithmic trading system designed for cryptocurrency markets. It combines the battle-tested Freqtrade engine with machine learning models (starting with classical ML, upgradeable to neural networks) to generate trading signals for **Trend Following** and **Swing Trading** strategies across multiple exchanges.

The key design principle is **future-proofing**: the ML layer is abstracted behind a `BaseMLModel` interface and a `ModelFactory`, meaning you can switch from LightGBM to LSTM or Transformer models with a single config change — no strategy code modifications required.

The project ships with a **PyQt6 Windows desktop application** that provides real-time dashboard, candlestick charts, backtesting viewer, and bot controls — all connected to the Freqtrade backend via REST API and WebSocket.

> ⚠️ **Disclaimer:** This software is for educational and research purposes. Trading cryptocurrencies carries significant financial risk. Always test thoroughly with dry-run before using real funds. Past performance does not guarantee future results.

---

## Features

-  **FreqAI Integration** — Built-in machine learning pipeline using Freqtrade's FreqAI module
-  **Multi-Exchange** — Supports Binance, OKX, Bybit via CCXT (100+ exchanges available)
-  **Dual Strategy** — Trend Following and Swing Trading strategies out of the box
-  **Model Factory Pattern** — Switch ML models (LightGBM → LSTM → Transformer) without touching strategy code
-  **Native Windows GUI** — PyQt6 desktop app with real-time dashboard and candlestick charts
-  **Risk Manager** — Independent risk layer: max drawdown, daily loss limits, position sizing
-  **Hyperopt Ready** — Parameter optimization via Freqtrade's Hyperopt engine
-  **Backtesting Viewer** — Visualize backtest results directly in the GUI
-  **Secure Config** — API keys stored in `.env`, never in source code
-  **Full Logging** — Every trade decision, model prediction, and feature value logged

---

## Architecture Overview

```
┌──────────────────────────────────────────┐
│        PyQt6 Desktop Application         │  ← GUI Layer
│  Dashboard · Charts · Settings · Logs    │
└───────────────┬──────────────────────────┘
                │  REST API + WebSocket
┌───────────────▼──────────────────────────┐
│          Freqtrade Bot Engine            │  ← Bot Layer
│  Strategy · FreqAI · Risk · Hyperopt     │
└───────────────┬──────────────────────────┘
                │  CCXT
┌───────────────▼──────────────────────────┐
│            Exchange Layer                │  ← Data Layer
│      Binance · OKX · Bybit · ...        │
└──────────────────────────────────────────┘
```

**The ML layer sits inside the Bot Layer and is fully swappable:**

```
ModelFactory.get_model("lightgbm")    → LightGBMModel    (Phase 1 - now)
ModelFactory.get_model("xgboost")     → XGBoostModel     (Phase 1 - now)
ModelFactory.get_model("lstm")        → LSTMModel        (Phase 2 - future)
ModelFactory.get_model("transformer") → TransformerModel (Phase 3 - future)
```

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Bot Engine | Freqtrade | Core trading logic, backtesting, hyperopt |
| AI/ML | FreqAI + LightGBM | Signal generation, model training |
| Exchange | CCXT | Multi-exchange connectivity |
| Technical Analysis | pandas-ta | RSI, EMA, MACD, ATR, Bollinger Bands |
| GUI | PyQt6 | Windows desktop application |
| Charts | pyqtgraph | Real-time candlestick charts |
| API Bridge | REST + WebSocket | GUI ↔ Bot communication |
| Database | SQLite | Trade history, OHLCV cache |
| Language | Python 3.11 | Core language |

---

## Requirements

### System Requirements

> 💡 **WSL2 Recommended:** Freqtrade is Linux-native. Running the bot backend inside WSL2 (Windows Subsystem for Linux) provides the most stable experience on Windows. The PyQt6 GUI runs natively on Windows and connects to the WSL2 backend via the REST API.

### Python Dependencies

```
freqtrade[freqai]>=2024.1
lightgbm>=4.0.0
xgboost>=2.0.0
scikit-learn>=1.3.0
pandas>=2.0.0
pandas-ta>=0.3.14b
numpy>=1.24.0
PyQt6>=6.6.0
pyqtgraph>=0.13.0
plotly>=5.18.0
requests>=2.31.0
websockets>=12.0
python-dotenv>=1.0.0
ccxt>=4.0.0
```

## Installation

### Step 1 — Clone the Repository

```bash
git clone https://github.com/yourusername/freqtrade-ai-bot.git
cd freqtrade-ai-bot
```

### Step 2 — Set Up Python Virtual Environment

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\activate

# WSL2 / Linux
python3.11 -m venv .venv
source .venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
# Core bot + FreqAI
pip install freqtrade[freqai]

# ML libraries
pip install lightgbm xgboost scikit-learn

# Technical analysis
pip install pandas-ta

# GUI (Windows native)
pip install PyQt6 pyqtgraph plotly

# Utilities
pip install requests websockets python-dotenv
```

Or install everything at once:

```bash
pip install -r requirements.txt
```

### Step 4 — Initialize Freqtrade User Directory

```bash
freqtrade create-userdir --userdir user_data
```

### Step 5 — Set Up Environment Variables

```bash
cp .env.example .env
```

Edit `.env` with your exchange API keys:

```env
# Binance
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here

# OKX
OKX_API_KEY=your_api_key_here
OKX_API_SECRET=your_api_secret_here
OKX_PASSPHRASE=your_passphrase_here

# Bybit
BYBIT_API_KEY=your_api_key_here
BYBIT_API_SECRET=your_api_secret_here

# Bot API (for GUI connection)
FREQTRADE_API_HOST=127.0.0.1
FREQTRADE_API_PORT=8080
FREQTRADE_API_USERNAME=bot_user
FREQTRADE_API_PASSWORD=strong_password_here
```

> 🔐 **Security:** Never commit `.env` to Git. It is already included in `.gitignore`. Use **read-only** API keys with no withdrawal permissions.

### Step 6 — Download Historical Data

```bash
# Download 1 year of 1h and 4h data for BTC and ETH
freqtrade download-data \
  --pairs BTC/USDT ETH/USDT SOL/USDT \
  --timeframe 1h 4h \
  --days 365 \
  --config config/config_binance.json
```

---

## Configuration

Configuration files are located in the `config/` directory:

| File | Exchange | Purpose |
|------|----------|---------|
| `config_binance.json` | Binance | Binance trading config |
| `config_okx.json` | OKX | OKX trading config |
| `config_bybit.json` | Bybit | Bybit trading config |

### Key Configuration Options

```json
{
  "max_open_trades": 5,
  "stake_currency": "USDT",
  "stake_amount": 100,
  "dry_run": true,
  "dry_run_wallet": 1000,
  "freqai": {
    "enabled": true,
    "model_save_type": "joblib",
    "train_period_days": 30,
    "identifier": "trend_v1",
    "feature_parameters": {
      "include_timeframes": ["1h", "4h"],
      "indicator_periods_candles": [10, 20, 50, 100]
    }
  },
  "api_server": {
    "enabled": true,
    "listen_ip_address": "127.0.0.1",
    "listen_port": 8080,
    "username": "bot_user",
    "password": "strong_password_here"
  }
}
```

---

## How to Launch

### Option A — Launch Everything (Recommended)

The `launcher.py` script starts the bot backend and the GUI together:

```bash
python launcher.py --exchange binance --strategy TrendMLStrategy
```

### Option B — Launch Bot Only (Headless)

```bash
# Dry run (paper trading)
freqtrade trade \
  --strategy TrendMLStrategy \
  --freqaimodel LightGBMRegressor \
  --config config/config_binance.json \
  --dry-run

# Live trading (use with caution!)
freqtrade trade \
  --strategy TrendMLStrategy \
  --freqaimodel LightGBMRegressor \
  --config config/config_binance.json
```

### Option C — Launch GUI Only

If the bot is already running (e.g., in WSL2), launch the GUI separately:

```bash
python gui/main_window.py
```

### Switching Exchanges

```bash
# OKX
python launcher.py --exchange okx --config config/config_okx.json

# Bybit
python launcher.py --exchange bybit --config config/config_bybit.json
```

### Switching ML Models

Edit the strategy config or pass via command line:

```bash
# Use XGBoost instead of LightGBM
python launcher.py --exchange binance --freqaimodel XGBoostRegressor
```

Or in `config_binance.json`:
```json
{
  "freqai": {
    "model_save_type": "joblib"
  }
}
```

---

## Project Structure

```
freqtrade-ai-bot/
│
├── core/                        # Bot engine — stable, rarely modified
│   ├── exchange_manager.py      # CCXT multi-exchange management
│   ├── trade_manager.py         # Position open/close logic
│   ├── risk_manager.py          # Stop-loss, drawdown limits
│   └── data_manager.py          # OHLCV download & SQLite cache
│
├── strategies/                  # Trading strategies
│   ├── base_strategy.py         # Abstract base class (IStrategy)
│   ├── trend_ml_strategy.py     # ML + Trend Following
│   ├── swing_ml_strategy.py     # ML + Swing Trading
│   └── archive/                 # Old strategy versions
│
├── ml/                          # ★ Pluggable ML layer
│   ├── base_model.py            # Abstract model interface
│   ├── model_factory.py         # Registry — swap models here
│   ├── classical/               # Phase 1: Current models
│   │   ├── lightgbm_model.py
│   │   ├── random_forest_model.py
│   │   └── xgboost_model.py
│   └── neural/                  # Phase 2: Future models
│       ├── lstm_model.py        # (planned)
│       ├── transformer_model.py # (planned)
│       └── cnn_model.py         # (planned)
│
├── features/                    # Feature engineering
│   ├── technical.py             # RSI, EMA, MACD, BB, ATR
│   ├── volume.py                # Volume-based features
│   ├── market_regime.py         # ADX trend/ranging detection
│   └── feature_pipeline.py      # Combines all features
│
├── gui/                         # PyQt6 desktop app
│   ├── main_window.py           # Application entry point
│   ├── dashboard.py             # Main screen
│   ├── chart_widget.py          # Candlestick + signals
│   ├── backtest_view.py         # Backtest result viewer
│   ├── settings_panel.py        # Exchange, strategy, config
│   └── components/
│       ├── trade_table.py       # Open/closed trades table
│       └── metric_cards.py      # P&L, win rate, balance
│
├── api/                         # GUI ↔ Bot bridge
│   ├── rest_client.py           # Freqtrade REST API client
│   └── ws_client.py             # WebSocket real-time updates
│
├── backtesting/
│   ├── runner.py                # Backtest orchestration
│   └── reporter.py              # Results formatting
│
├── user_data/                   # Freqtrade data directory
│   ├── data/                    # OHLCV candle data
│   ├── models/                  # Trained ML models
│   ├── backtest_results/        # Backtest outputs
│   └── logs/                    # Trade & bot logs
│
├── config/
│   ├── config_binance.json
│   ├── config_okx.json
│   └── config_bybit.json
│
├── launcher.py                  # One-click: bot + GUI
├── .env.example                 # Environment variable template
├── .gitignore
└── README.md
```

---

## Supported Exchanges

| Exchange | Status | Spot | Futures | Notes |
|----------|--------|------|---------|-------|
| Binance | ✅ Active | ✓ | ✓ | Primary exchange, best FreqAI support |
| OKX | ✅ Active | ✓ | ✓ | |
| Bybit | ✅ Active | ✓ | ✓ | |
| Others | 🔧 Via CCXT | ✓ | Varies | Any CCXT-supported exchange |

---

## Trading Strategies

### TrendMLStrategy

Uses ML model predictions combined with directional trend indicators to enter long positions when a strong trend is confirmed. Designed for 1h–4h timeframes.

**Key indicators:** EMA-20/50/200 alignment, ADX > 25, ML signal > threshold

### SwingMLStrategy

Uses ML model predictions to identify swing entry points at market structure support/resistance levels. Designed for 4h timeframes.

**Key indicators:** RSI divergence, Bollinger Band squeeze, ML probability score

---

## ML Model Upgrade Path

The project is designed so that **upgrading the ML model never requires changes to strategy code**. The `ModelFactory` handles all routing:

```
Phase 1 (Now)      → LightGBM, Random Forest, XGBoost
Phase 2 (2-3 mo.)  → LSTM, GRU, 1D-CNN
Phase 3 (4-6 mo.)  → Temporal Fusion Transformer
Phase 4 (Advanced) → Reinforcement Learning (PPO/SAC)
```

To switch models, change one line in config:

```json
{ "freqai": { "identifier": "lstm_v1" } }
```

---

## Backtesting

```bash
# Run backtest with FreqAI
freqtrade backtesting \
  --strategy TrendMLStrategy \
  --freqaimodel LightGBMRegressor \
  --timerange 20240101-20241231 \
  --config config/config_binance.json

# View results in GUI
python launcher.py --mode backtest-viewer

# Hyperparameter optimization
freqtrade hyperopt \
  --strategy TrendMLStrategy \
  --hyperopt-loss SharpeHyperoptLoss \
  --epochs 200 \
  --config config/config_binance.json
```

---

## Dry Run (Paper Trading)

Always test with dry run before going live:

```bash
# Set in config: "dry_run": true, "dry_run_wallet": 1000
freqtrade trade \
  --strategy TrendMLStrategy \
  --freqaimodel LightGBMRegressor \
  --config config/config_binance.json
```

Recommended dry run period: **minimum 2 weeks** per strategy version.

---

## Risk Management

The `risk_manager.py` module operates independently of strategies:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_open_trades` | 5 | Max simultaneous positions |
| `stake_amount` | 100 USDT | Per-trade investment |
| `stoploss` | -0.05 | 5% stop-loss per trade |
| `max_drawdown` | 0.15 | 15% portfolio drawdown halt |
| `daily_loss_limit` | 0.05 | 5% daily loss circuit breaker |
| `trailing_stop` | true | Dynamic trailing stop-loss |

---

## Roadmap

- [x] Project architecture & folder structure
- [x] Freqtrade + FreqAI integration
- [x] Multi-exchange support (Binance, OKX, Bybit)
- [x] LightGBM model via Model Factory
- [x] TrendMLStrategy implementation
- [ ] SwingMLStrategy implementation
- [ ] PyQt6 GUI — Dashboard
- [ ] PyQt6 GUI — Candlestick chart widget
- [ ] PyQt6 GUI — Backtest viewer
- [ ] PyQt6 GUI — Settings panel
- [ ] Hyperopt optimization pipeline
- [ ] LSTM model (Phase 2)
- [ ] Temporal Fusion Transformer (Phase 3)
- [ ] Reinforcement Learning agent (Phase 4)
- [ ] Multi-timeframe ensemble signals
- [ ] Telegram / Discord alert integration

---

## Security

- **Never commit API keys** — use `.env` file (already in `.gitignore`)
- **Use read-only API keys** — enable only Spot Trading, disable Withdrawal
- **IP whitelist** — restrict API access to your IP in exchange settings
- **Bot API password** — set a strong password in `config.json` → `api_server`
- **Dry run first** — never deploy a new strategy directly to live trading

---

## Contributing

This is a personal project. If you'd like to suggest improvements or report bugs, feel free to open an issue. Pull requests for new ML model implementations (following `BaseMLModel` interface) are especially welcome.

---

## License

This project is for personal educational use. See `LICENSE` for details.

Freqtrade is licensed under [GNU GPLv3](https://github.com/freqtrade/freqtrade/blob/stable/LICENSE).

---

## Version History

| Version | Date | Notes |
|---------|------|-------|
| `0.1.0` | 2025-03 | Initial project structure, architecture design |
| `0.2.0` | TBD | LightGBM model + TrendMLStrategy |
| `0.3.0` | TBD | PyQt6 GUI — Dashboard & Charts |
| `0.4.0` | TBD | Hyperopt pipeline + SwingMLStrategy |
| `1.0.0` | TBD | First stable release — dry run validated |
| `2.0.0` | TBD | Neural Networks (LSTM/Transformer) integration |

---

<div align="center">
  <sub>Built with Freqtrade · FreqAI · PyQt6 · LightGBM · CCXT</sub>
</div>