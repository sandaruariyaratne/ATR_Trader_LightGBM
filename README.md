# AlgoTrader — Event-Driven Quantitative Trading Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/ML-LightGBM%20%7C%20XGBoost-blue.svg?style=flat)](https://lightgbm.readthedocs.io/)
[![AsyncIO](https://img.shields.io/badge/Architecture-AsyncIO%20Event--Driven-purple.svg?style=flat)](https://docs.python.org/3/library/asyncio.html)
[![CCXT Pro](https://img.shields.io/badge/Exchange-CCXT%20Pro-orange.svg?style=flat)](https://ccxt.com/)
[![Vercel](https://img.shields.io/badge/Vercel-Web%20Dashboard-black.svg?style=flat&logo=vercel&logoColor=white)](https://vercel.com/)
[![Docker](https://img.shields.io/badge/Deployment-Docker%20%7C%20Compose-2496ED.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![Google Cloud](https://img.shields.io/badge/Cloud-Google%20Compute%20Engine-4285F4.svg?style=flat&logo=googlecloud&logoColor=white)](https://cloud.google.com/)
[![Tests](https://img.shields.io/badge/Tests-Pytest%20Suite-brightgreen.svg?style=flat&logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat)](LICENSE)

An institutional-grade, asynchronous quantitative trading system designed for high-frequency 1-minute OHLCV crypto market execution on Binance via CCXT. The engine integrates **Marcos López de Prado’s Volatility-Adjusted Triple Barrier Method (TBM)** with calibrated **LightGBM gradient boosting models**, stationary microstructure feature engineering, dynamic risk sizing, and a decoupled event-driven reactive architecture.

---

## 📌 Executive Summary & Key Performance Metrics

The trading model was trained and rigorously validated across **2,551,833 consecutive 1-minute candles (~58.2 months)** of historical tick/bar data (covering market regimes from bull runs, deep bear drawdowns, and high-volatility sideways compression). 

All backtests incorporate a **10 bps (0.10%) round-trip fee model** and realistic execution constraints (zero look-ahead bias, strict chronological training/validation splits, and realistic bid-ask spread assumptions).

### Aligned Strategy Performance (SOL/USDT — 2.55M 1-Minute Bars)

| Metric | Peak Performance Mode | Balanced Growth Mode | High-Conviction Mode |
| :--- | :---: | :---: | :---: |
| **Model Confidence Threshold** | **48%** | **49%** | **53%** |
| **Total Cumulative Return** | **+261.96%** | **+221.30%** | **+55.62%** |
| **Maximum Drawdown** | **5.30%** | **3.89%** | **1.27%** |
| **Profit Factor** | **1.273** | **1.366** | **2.171** |
| **Win Rate** | **50.68%** | **52.56%** | **60.29%** |
| **Average Trades / Day** | 4.89 | 3.19 | 0.51 |
| **Return / Max Drawdown (Calmar)** | **49.4x** | **56.9x** | **43.8x** |

> **Key Quantitative Takeaway:** Enforcing a **15-candle vertical timeout** alongside dynamic ATR barriers boosted win rate by over **+3.3%** and compressed maximum drawdown from **4.54% down to 3.89%**, systematically mitigating capital lockup during low-volatility dead zones.

---

## 🏗️ System Architecture & Event-Driven Engine

The system is built on a **fully asynchronous, non-blocking pub/sub message pipeline** powered by Python's `asyncio` and typed message contracts. Each domain component is isolated as an independent agent communicating strictly over an in-memory `EventBus`.

```
                        ┌─────────────────────────────────────┐
                        │      CCXT Pro / Binance Stream      │
                        │        WebSocket / REST Feeds       │
                        └──────────────────┬──────────────────┘
                                           │ Kline Updates
                                           ▼
                              ┌─────────────────────────┐
                              │    MarketDataAgent      │
                              │   (Ingestion & ATR)     │
                              └────────────┬────────────┘
                                           │ MarketStateEvent
                                           ▼
                        ┌─────────────────────────────────────┐
                        │     FeatureEngineeringAgent         │
                        │ (Stationary Transforms & Indicators) │
                        └──────────────────┬──────────────────┘
                                           │ FeatureVectorEvent
                                           ▼
                              ┌─────────────────────────┐
                              │      DecisionAgent      │
                              │ (LightGBM ML Inference) │
                              └────────────┬────────────┘
                                           │ TradeSignalEvent
                                           ▼
                        ┌─────────────────────────────────────┐
                        │       RiskManagementAgent           │
                        │ (Dynamic ATR Barriers & Drawdown)   │
                        └──────────────────┬──────────────────┘
                                           │ OrderExecutionEvent
                                           ▼
                              ┌─────────────────────────┐
                              │     ExecutionAgent      │
                              │ (Fills & 15-Bar Exit)   │
                              └────────────┬────────────┘
                                           │ FillEvent / CancelEvent
                                           ▼
                        ┌─────────────────────────────────────┐
                        │         StateManager                │
                        │ (Portfolio Equity & Audit Logs)     │
                        └─────────────────────────────────────┘
```

### Modular Component Breakdown

- **`agents/market_data_agent.py`**: Handles low-latency WebSocket connection management, kline ingestion, order book heartbeat checks, and real-time streaming ATR calculation.
- **`agents/feature_engineering_agent.py`**: Computes 36+ streaming features over rolling windows, guaranteeing stationarity (log-returns, multi-scale VWAP deviations, volume force indices, taker imbalances).
- **`agents/decision_agent.py`**: Sub-millisecond inference using trained gradient-boosted trees (LightGBM/XGBoost) with calibrated probabilistic thresholds.
- **`agents/risk_management_agent.py`**: Computes dynamic stops and targets based on real-time ATR, enforces portfolio-level circuit breakers, and verifies margin constraints.
- **`agents/execution_agent.py`**: Manages the full order lifecycle, simulates paper fills with slippage, monitors open positions, and automatically enforces 15-period vertical exits.
- **`core/event_bus.py`**: High-throughput typed asyncio pub-sub event router.
- **`core/state_manager.py`**: Thread-safe state tracking for account balance, open positions, realized PnL, and concurrent CSV trade journaling.
- **`core/logger.py`**: High-performance structured JSON logging (`structlog`) with rotation and colorized developer output.

---

## 📐 Quantitative Methodology: Volatility-Adjusted TBM

Standard fixed-percentage stop-loss and take-profit mechanisms fail in cryptocurrency markets due to severe volatility clustering. This system implements an adaptive **Triple Barrier Method (TBM)**:

$$\text{Upper Barrier (Take Profit)} = P_{\text{entry}} + \beta_{\text{TP}} \times \text{ATR}_{14}(t)$$

$$\text{Lower Barrier (Stop Loss)} = P_{\text{entry}} - \beta_{\text{SL}} \times \text{ATR}_{14}(t)$$

$$\text{Vertical Barrier (Time Exit)} = t_{\text{entry}} + 15 \times \Delta t_{\text{candle}}$$

### Why the Triple Barrier Structure Wins:
1. **Dynamic Volatility Scaling**: During high-volatility expansions, barriers widen to avoid premature stop-outs from noise; during quiet consolidations, barriers tighten to lock in profits.
2. **Path-Dependent Realism**: Captures intraday wick breaches rather than just closing-price checks.
3. **Vertical Time Barrier (15 Bars)**: Eliminates long-tail capital lockup. If a thesis does not realize within 15 minutes, the position is liquidated at market price.

### Microstructure Feature Engineering
To eliminate non-stationarity without losing memory:
- **Stationary Multi-Scale Returns**: Continuous log-returns across 5m, 15m, 30m, 1h, and 4h horizons: $r_t = \ln(P_t / P_{t-k})$.
- **VWAP Deviations**: Normalized deviation from volume-weighted average price across rolling windows: $(P_t - \text{VWAP}_k) / \text{VWAP}_k$.
- **Order Flow & Taker Imbalance**: Net buyer flow ratios $\frac{V_{\text{taker buy}} - V_{\text{taker sell}}}{V_{\text{total}}}$ across 1m, 5m, 15m, 60m, and 240m.
- **Microstructural Liquidity Indicators**: Elder's Force Index, Volume Z-Scores, True Volume Index (TVI), and Candle Wick Ratios.
- **Outlier Winsorization**: Automated quantile clipping ($0.1\%$ to $99.9\%$) via `scripts/treat_anomalies.py` preventing extreme data artifacts from corrupting split nodes during model training.

---

## 📊 Backtest Evaluation & Parameter Sensitivity

### Comprehensive Threshold Sweep (SOL/USDT — 58.2 Months)

| Threshold | Total Return | Final Equity ($10k Start) | Compounded Monthly Return | Max Drawdown | Profit Factor | Win Rate | Trades / Month |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **45%** | +131.53% | $23,153.00 | +1.45% | 31.57% | 1.046 | 45.31% | 562.3 |
| **46%** | +232.43% | $33,243.00 | +2.08% | 15.10% | 1.107 | 46.66% | 363.6 |
| **47%** | +246.52% | $34,652.00 | +2.16% | 10.21% | 1.179 | 48.37% | 234.8 |
| **48% (Peak)** | **+261.96%** | **$36,196.00** | **+2.23%** | **5.30%** | **1.273** | **50.68%** | **149.0** |
| **49% (Balanced)** | **+221.30%** | **$32,130.00** | **+2.03%** | **3.89%** | **1.366** | **52.56%** | **97.1** |
| **50%** | +166.24% | $26,624.00 | +1.70% | 4.36% | 1.462 | 54.21% | 63.3 |
| **51%** | +126.35% | $22,635.00 | +1.41% | 2.99% | 1.673 | 56.34% | 38.9 |
| **52%** | +82.47% | $18,247.00 | +1.04% | 1.64% | 1.841 | 57.76% | 24.6 |
| **53% (Conservative)** | **+55.62%** | **$15,562.00** | **+0.76%** | **1.27%** | **2.171** | **60.29%** | **15.6** |
| **54%** | +34.61% | $13,461.00 | +0.51% | 1.16% | 2.119 | 60.73% | 10.8 |
| **55%** | +24.08% | $12,408.00 | +0.37% | 0.85% | 2.196 | 60.39% | 7.9 |

---

## 📁 Repository Structure

```
ATR_Trader_LightGBM/
├── agents/                         # Decoupled pipeline agents
│   ├── market_data_agent.py        # Real-time WebSocket/REST kline ingestion
│   ├── feature_engineering_agent.py # Stationary feature transform pipeline
│   ├── decision_agent.py           # LightGBM/XGBoost inference engine
│   ├── risk_management_agent.py    # Dynamic ATR barriers & circuit breakers
│   └── execution_agent.py          # Paper fill simulator & 15-candle vertical exit
├── config/                         # Centralized configuration
│   ├── settings.py                 # Pydantic BaseSettings (env-driven validation)
│   └── trading_params.py           # Quantitative parameters & indicator constants
├── core/                           # Asynchronous runtime backbone
│   ├── event_bus.py                # Typed pub-sub event bus
│   ├── pipeline.py                 # Agent lifecycle & orchestrator
│   ├── state_manager.py            # Portfolio accounting & CSV trade audit log
│   └── logger.py                   # Structured JSON logger (structlog)
├── models/                         # Model loading & inference abstraction
│   ├── base_model.py               # Abstract base model contract
│   ├── xgboost_model.py            # LightGBM & XGBoost inference adapter
│   ├── onnx_model.py               # ONNX runtime interface
│   └── torch_model.py              # PyTorch model adapter
├── utils/                          # Financial math & indicator primitives
│   ├── indicators.py               # EMA, SMA, RSI, MACD, ATR, VWAP, ADX
│   ├── features.py                 # Multi-timeframe stationary features & TBM labels
│   ├── normalizer.py               # Z-score & streaming feature normalizers
│   └── time_utils.py               # Timestamp conversions & interval math
├── scripts/                        # Operational & Research CLI tools
│   ├── backtest.py                 # Vectorized backtester with 15-candle timeout
│   ├── train_model.py              # Single-asset model training pipeline
│   ├── train_multi_asset.py        # Universal multi-asset LightGBM trainer
│   ├── label_dataset.py            # Triple Barrier Method labeling generator
│   ├── treat_anomalies.py          # Winsorization & outlier preprocessing
│   ├── filter_dates.py             # Dataset date range slice tool
│   ├── download_data.py            # Binance historical kline downloader
│   ├── deploy.py                   # Automated cloud deployment utility
│   ├── deploy_to_gce.sh            # Production Google Cloud VM deployer
│   └── sync_trades.sh              # Remote trade log synchronizer
├── tests/                          # Automated pytest suite
│   ├── test_indicators.py          # Technical indicator unit tests
│   ├── test_feature_engineering.py # Stationary feature extraction tests
│   ├── test_tbm_labeling.py        # Triple Barrier Method labeling tests
│   ├── test_decision_agent.py      # Inference & probability boundary tests
│   ├── test_risk_management.py     # Position sizing & circuit breaker tests
│   ├── test_execution_fees.py      # Fee calculation & order reconciliation tests
│   ├── test_state_manager_csv.py   # State tracking & CSV persistence tests
│   └── test_pipeline_integration.py# End-to-end event-bus integration tests
├── dashboard/                      # Interactive quantitative web dashboard (Vercel)
│   ├── src/                        # React UI, dynamic equity curve SVG, trade ledger
│   ├── public/                     # Icons & static assets
│   ├── index.html                  # Dashboard entrypoint
│   └── package.json                # Web app dependencies & scripts
├── data/
│   ├── models/                     # Serialized production model artifacts (.pkl)
│   ├── atr_sweep_results.md        # Comprehensive ATR multiplier sweep report
│   └── tp_confidence_sweep_results.md # TBM confidence threshold sweep report
├── Dockerfile                      # Production container image definition
├── docker-compose.yml              # Container orchestration configuration
├── vercel.json                     # Root Vercel serverless deployment config
├── pyproject.toml                  # Pytest and coverage configurations
├── requirements.txt                # Production Python dependencies
└── main.py                         # Application runtime entrypoint
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10 or higher
- Optional: Docker and Docker Compose (for containerized deployment)

### 2. Environment Setup
Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/your-username/ATR_Trader_LightGBM.git
cd ATR_Trader_LightGBM

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configuration
Copy sample configurations or create a `.env` file in the root directory:

```bash
# Environment Mode
PAPER_TRADING=true
TRADING_SYMBOL=SOL/USDT
CANDLE_INTERVAL=1m

# Risk & Capital Management
INITIAL_CAPITAL=10000.0
MAX_POSITION_PCT=0.10
LEVERAGE=2.0
STOP_LOSS_ATR_MULT=1.5
TAKE_PROFIT_ATR_MULT=3.0

# Model Inference
MODEL_TYPE=xgboost
MODEL_PATH=data/models/lightgbm_universal.pkl
CONFIDENCE_THRESHOLD_BUY=0.48
CONFIDENCE_THRESHOLD_SELL=0.42

# Optional Live Exchange Credentials
EXCHANGE_ID=binance
EXCHANGE_API_KEY=your_api_key_here
EXCHANGE_API_SECRET=your_api_secret_here
```

### 4. Running the Engine in Paper Trading Mode
Start streaming live market data from Binance and evaluating inference in real time:

```bash
python main.py
```

### 5. Running Backtests
Execute backtests over historical CSV datasets:

```bash
python scripts/backtest.py \
    --csv data/raw/solusdt_1m.csv \
    --model data/models/lightgbm_SOLUSDT.pkl \
    --capital 10000 \
    --confidence 0.48
```

### 6. Training Multi-Asset Models
Train an asset-agnostic universal LightGBM model across multiple crypto pairs:

```bash
python scripts/train_multi_asset.py \
    --tp-mult 3.0 \
    --sl-mult 1.5 \
    --fee-rate 0.0002
```

---

## 🧪 Testing & Validation

The codebase includes comprehensive unit and integration test suites covering indicator calculations, TBM labeling logic, risk management circuit breakers, execution fee deductions, and event-bus message flow.

Run the test suite:
```bash
pytest
```

Run test suite with coverage report:
```bash
pytest --cov=. --cov-report=term-missing
```

---

## 🌐 Interactive Web Dashboard (Vercel)

The repository includes a web dashboard built with React and Vite in the `dashboard/` directory, designed to showcase strategy metrics, dynamic equity curves, parameter sweeps, and live trade ledgers on your CV or portfolio.

### Run Dashboard Locally
```bash
cd dashboard
npm install
npm run dev
```

### Deploy to Vercel (2 Options)

#### Option 1: One-Click GitHub Integration (Recommended)
1. Push your repository to GitHub.
2. Go to [Vercel Dashboard](https://vercel.com/new) and click **"Add New Project"** &rarr; **"Import"** your repository.
3. Vercel automatically detects `vercel.json` and builds the dashboard using:
   - **Framework Preset**: `Vite`
   - **Root Directory**: `.` (or set to `dashboard` if preferred)
   - **Build Command**: `cd dashboard && npm install && npm run build`
   - **Output Directory**: `dashboard/dist`
4. Click **Deploy**. Your interactive portfolio dashboard will be live at `https://your-project.vercel.app`!

#### Option 2: Deploy via Vercel CLI
```bash
cd dashboard
npx vercel
```

---

## ☁️ Production Trading Engine Deployment (Docker & GCP)

The system is configured for 24/7 cloud execution with zero downtime, graceful signal handling, and auto-restart policies.

### Local Docker Deployment
```bash
docker-compose up -d --build
```

Monitor live logs:
```bash
docker-compose logs -f
```

### Google Cloud Compute Engine (GCE) Automated Provisioning
The included deployment script automates VM provisioning, Docker installation, code sync, and daemon initialization:

```bash
# 1. Authenticate with Google Cloud
gcloud auth login
gcloud config set project <your-gcp-project-id>

# 2. Deploy to Compute Engine
bash scripts/deploy_to_gce.sh
```

### Live Trade Log Synchronization
Download live executed trades from the cloud VM to your local workspace:

```bash
bash scripts/sync_trades.sh
```

---

## ⚙️ Configuration Reference

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `PAPER_TRADING` | `bool` | `true` | When true, executes orders via internal paper fill simulation |
| `TRADING_SYMBOL` | `str` | `"SOL/USDT"` | Target asset trading pair |
| `CANDLE_INTERVAL` | `str` | `"1m"` | Timeframe resolution of OHLCV bars |
| `INITIAL_CAPITAL` | `float` | `10000.0` | Initial balance in quote currency (USDT) |
| `MAX_POSITION_PCT` | `float` | `0.10` | Maximum allocation per trade (10% of equity) |
| `LEVERAGE` | `float` | `2.0` | Leverage multiplier applied to position margin |
| `STOP_LOSS_ATR_MULT` | `float` | `1.5` | Stop loss distance as multiple of real-time ATR |
| `TAKE_PROFIT_ATR_MULT` | `float` | `3.0` | Take profit distance as multiple of real-time ATR |
| `CONFIDENCE_THRESHOLD_BUY`| `float` | `0.48` | Minimum model confidence required to execute long |
| `CONFIDENCE_THRESHOLD_SELL`| `float` | `0.42` | Minimum model confidence required to execute short |
| `MODEL_PATH` | `Path` | `"data/models/lightgbm_universal.pkl"` | Path to serialized model artifact |
| `LOG_LEVEL` | `str` | `"INFO"` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
