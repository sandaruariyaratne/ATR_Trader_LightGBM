# AlgoTrader — Low-Latency Event-Driven Crypto Trading System

A modular, asyncio-based algorithmic trading engine for 1-minute OHLCV candle trading on Binance via CCXT. It is fully aligned with the **Volatility-Adjusted Triple Barrier Method (TBM)** and includes support for live paper trading and automated 24/7 Google Cloud deployment.

---

## 📈 Strategy: Volatility-Adjusted TBM

This system implements a strict Triple Barrier Method (TBM) for exit boundaries, dynamically sized by real-time Average True Range (ATR) volatility:

1.  **Horizontal Profit Barrier (Take Profit)**: $2.0 \times \text{ATR}$
2.  **Horizontal Risk Barrier (Stop Loss)**: $1.0 \times \text{ATR}$
3.  **Vertical Time Barrier (Time Exit)**: **15 candles** (forces an immediate exit at market price if neither SL nor TP is hit within 15 periods)
4.  **Transaction Fees**: 10 bps (0.10%) round-trip fee is fully accounted for in all simulations.

### Aligned Performance Sweeps (SOL/USDT — 2.5M candles)
The bot's performance on 58.2 months of historical SOL/USDT 1-minute bars using the trained LightGBM model:

| Confidence Threshold | Total Return % | Max Drawdown % | Profit Factor | Win Rate % | Average Trades / Day |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **48% (New Peak)** | **+261.96%** | **5.30%** | **1.273** | **50.68%** | **4.89** |
| **49% (Balanced)** | **+221.30%** | **3.89%** | **1.366** | **52.56%** | **3.19** |
| **53% (Conservative)** | **+55.62%** | **1.27%** | **2.171** | **60.29%** | **0.51** |

---

## 🗂️ Architecture

```
ATR_Trader_XG/
├── agents/
│   ├── market_data_agent.py        # WS/REST live candle ingestion + ATR calculation
│   ├── feature_engineering_agent.py # Stationary log-returns & feature compiling
│   ├── decision_agent.py           # LightGBM inference & predictions
│   ├── risk_management_agent.py    # Kelly sizing, circuit breakers, ATR-based SL/TP
│   └── execution_agent.py          # Paper fill simulation & 15-candle vertical exit
├── core/
│   ├── pipeline.py                 # Async agent lifecycle manager
│   ├── event_bus.py                # Typed async event bus
│   ├── state_manager.py            # Thread-safe portfolio state & CSV logging
│   └── logger.py                   # Structured log outputs with standard fallback
├── models/
│   ├── base_model.py               # Abstract model interface
│   └── xgboost_model.py            # LightGBM/XGBoost unpickling wrapper
├── config/
│   └── settings.py                 # Pydantic-based configuration settings
├── utils/
│   ├── indicators.py               # Volatility indicators (ATR)
│   └── time_utils.py               # Time parsing & candle interval utils
├── scripts/
│   ├── label_dataset.py            # TBM training dataset generator
│   ├── train_model.py              # LightGBM training script
│   ├── backtest.py                 # Vectorized backtester with 15-candle timeout
│   └── deploy_to_gce.sh            # GCP VM dockerized deploy script
├── scratch/
│   └── backtest_confidence_sweep.py # Threshold & parameter sweep utility
├── data/
│   ├── models/                     # Saved pkl model artifacts
│   └── trades_log.csv              # Live trade records (local & remote mounted)
├── main.py                         # Application entrypoint
├── requirements.txt                # Production requirements (slim, no PyTorch)
└── docker-compose.yml              # Container runtime configuration
```

---

## 🚀 Quick Start (Local Run)

### 1. Configure the Environment
Ensure your local Python virtual environment is activated, then install the slim trading dependencies:
```bash
pip install -r requirements.txt
```

### 2. Configure Settings
Configuration is driven by Pydantic defaults in [settings.py](file:///Users/sandaruariyaratne/Downloads/ATR_Trader_XG/config/settings.py). You can override them via environment variables or a `.env` file:
*   `PAPER_TRADING=True` (Safe simulation mode)
*   `TRADING_SYMBOL=SOL/USDT`
*   `CONFIDENCE_THRESHOLD=0.48` (Peak performance setting)
*   `INITIAL_CAPITAL=10000.0`

### 3. Run the Bot locally
Stream live market data and evaluate predictions:
```bash
python main.py
```

---

## ☁️ Google Cloud (GCP) 24/7 Deployment

The bot runs on a light, automated Compute Engine VM (`atr-trader-vm`) in a Docker container that is configured to auto-restart on system reboots.

### 1. Initialize and Select Account
Make sure you are logged in to your deployment account:
```bash
gcloud auth login sandaruariyaratne@gmail.com
```

### 2. Create the GCP Project
```bash
gcloud projects create atr-trader-lgbm-v1 --name="ATR-Trader-LGBM"
gcloud config set project atr-trader-lgbm-v1
```
*Note: Make sure to link your billing account in the [GCP console](https://console.cloud.google.com/billing/projects) before proceeding.*

### 3. Enable Compute API and Deploy
```bash
# Enable the Compute Engine API
gcloud services enable compute.googleapis.com --project=atr-trader-lgbm-v1

# Run the automated deployment script
bash scripts/deploy_to_gce.sh
```

---

## 📊 Live Trade Syncing & Monitoring

### Stream VM container logs:
```bash
gcloud compute ssh atr-trader-vm \
  --zone=asia-east1-a \
  --project=atr-trader-lgbm-v1 \
  --command="sudo docker logs -f atr-trader-lgbm"
```

### Synchronize Remote Trades to your local machine:
A dedicated sync script downloads the latest trade logs from the VM to your local `data/trades_log.csv` file:
```bash
bash scripts/sync_trades.sh
```
Use `cat data/trades_log.csv` to view the local copy.
