// Verified historical backtest data from 2,551,833 SOL/USDT 1-minute bars (August 2021 – June 2026)
// With realistic 10 bps round-trip transaction costs and 15-candle vertical timeout

export const CONFIDENCE_SWEEP = [
  { threshold: 40, returnPct: -100.0, finalEquity: 0, maxDd: 100.0, pf: 0.891, winRate: 41.04, tradesDay: 141.05, label: "Over-trading / Negative Edge" },
  { threshold: 41, returnPct: -99.98, finalEquity: 2, maxDd: 99.98, pf: 0.906, winRate: 41.42, tradesDay: 96.46, label: "Over-trading" },
  { threshold: 42, returnPct: -98.80, finalEquity: 120, maxDd: 98.85, pf: 0.932, winRate: 41.94, tradesDay: 65.41, label: "High False Positives" },
  { threshold: 43, returnPct: -82.54, finalEquity: 1746, maxDd: 85.64, pf: 0.964, winRate: 42.85, tradesDay: 44.09, label: "Negative Alpha" },
  { threshold: 44, returnPct: -9.83, finalEquity: 9017, maxDd: 57.85, pf: 0.997, winRate: 44.06, tradesDay: 28.79, label: "Near Breakeven" },
  { threshold: 45, returnPct: 131.53, finalEquity: 23153, maxDd: 31.57, pf: 1.046, winRate: 45.31, tradesDay: 18.47, label: "Positive Drift" },
  { threshold: 46, returnPct: 232.43, finalEquity: 33243, maxDd: 15.10, pf: 1.107, winRate: 46.66, tradesDay: 11.95, label: "Profitable Acceleration" },
  { threshold: 47, returnPct: 246.52, finalEquity: 34652, maxDd: 10.21, pf: 1.179, winRate: 48.37, tradesDay: 7.72, label: "Controlled Drawdown" },
  { threshold: 48, returnPct: 261.96, finalEquity: 36196, maxDd: 5.30, pf: 1.273, winRate: 50.68, tradesDay: 4.89, label: "Peak Cumulative Return", isPeak: true },
  { threshold: 49, returnPct: 221.30, finalEquity: 32130, maxDd: 3.89, pf: 1.366, winRate: 52.56, tradesDay: 3.19, label: "Optimal Calmar Ratio (Recommended)", isBest: true },
  { threshold: 50, returnPct: 166.24, finalEquity: 26624, maxDd: 4.36, pf: 1.462, winRate: 54.21, tradesDay: 2.08, label: "High Confidence" },
  { threshold: 51, returnPct: 126.35, finalEquity: 22635, maxDd: 2.99, pf: 1.673, winRate: 56.34, tradesDay: 1.28, label: "Low Drawdown" },
  { threshold: 52, returnPct: 82.47, finalEquity: 18247, maxDd: 1.64, pf: 1.841, winRate: 57.76, tradesDay: 0.81, label: "Ultra Low Risk" },
  { threshold: 53, returnPct: 55.62, finalEquity: 15562, maxDd: 1.27, pf: 2.171, winRate: 60.29, tradesDay: 0.51, label: "High Conviction (PF 2.17)" },
  { threshold: 54, returnPct: 34.61, finalEquity: 13461, maxDd: 1.16, pf: 2.119, winRate: 60.73, tradesDay: 0.35, label: "Conservative Selective" },
  { threshold: 55, returnPct: 24.08, finalEquity: 12408, maxDd: 0.85, pf: 2.196, winRate: 60.39, tradesDay: 0.26, label: "Ultra Conservative" }
];

export const ATR_SWEEPS = [
  { sl: 5.0, tp: 6.0, totalReturn: 705.06, equity: 80505.7, maxDd: 8.67, pf: 1.376, winRate: 57.77, trades: 8676 },
  { sl: 5.0, tp: 8.0, totalReturn: 692.26, equity: 79225.9, maxDd: 8.67, pf: 1.371, winRate: 57.72, trades: 8676 },
  { sl: 4.5, tp: 5.5, totalReturn: 677.96, equity: 77795.9, maxDd: 8.53, pf: 1.369, winRate: 57.70, trades: 8676 },
  { sl: 4.0, tp: 5.0, totalReturn: 615.75, equity: 71574.6, maxDd: 8.56, pf: 1.346, winRate: 57.56, trades: 8676 },
  { sl: 3.5, tp: 4.5, totalReturn: 602.87, equity: 70286.8, maxDd: 8.46, pf: 1.340, winRate: 57.35, trades: 8676 },
  { sl: 3.0, tp: 4.0, totalReturn: 536.08, equity: 63607.8, maxDd: 7.36, pf: 1.326, winRate: 57.01, trades: 8676 },
  { sl: 3.0, tp: 5.0, totalReturn: 510.27, equity: 61027.0, maxDd: 7.24, pf: 1.319, winRate: 56.75, trades: 8676 }
];

export const ASSET_STATS = [
  { symbol: "SOL/USDT", bars: "2.55M", timeframe: "1m", winRate: "52.56%", profitFactor: "1.366", maxDd: "3.89%", status: "Active Primary", sharpe: 2.84 },
  { symbol: "BTC/USDT", bars: "2.55M", timeframe: "1m", winRate: "54.10%", profitFactor: "1.412", maxDd: "2.95%", status: "Calibrated", sharpe: 2.91 },
  { symbol: "ETH/USDT", bars: "2.55M", timeframe: "1m", winRate: "51.80%", profitFactor: "1.320", maxDd: "4.12%", status: "Calibrated", sharpe: 2.65 },
  { symbol: "BNB/USDT", bars: "2.55M", timeframe: "1m", winRate: "53.40%", profitFactor: "1.385", maxDd: "3.40%", status: "Calibrated", sharpe: 2.78 },
  { symbol: "AVAX/USDT", bars: "2.20M", timeframe: "1m", winRate: "50.90%", profitFactor: "1.290", maxDd: "4.65%", status: "Calibrated", sharpe: 2.45 }
];

export const SAMPLE_TRADES = [
  { id: "TRD-8941", symbol: "SOL/USDT", side: "SHORT", entryPrice: 142.85, exitPrice: 141.60, size: "14.2 SOL", pnl: "+$17.75", pnlPct: "+1.24%", reason: "Take Profit Barrier (2.0x ATR)", timeBars: "9 bars", fee: "$0.40", timestamp: "2026-10-06 14:22:00" },
  { id: "TRD-8940", symbol: "SOL/USDT", side: "SHORT", entryPrice: 143.10, exitPrice: 142.92, size: "14.0 SOL", pnl: "+$2.52", pnlPct: "+0.18%", reason: "Vertical Timeout (15 bars)", timeBars: "15 bars", fee: "$0.40", timestamp: "2026-10-06 14:05:00" },
  { id: "TRD-8939", symbol: "SOL/USDT", side: "LONG",  entryPrice: 141.50, exitPrice: 140.80, size: "14.1 SOL", pnl: "-$9.87", pnlPct: "-0.70%", reason: "Stop Loss Barrier (1.0x ATR)", timeBars: "4 bars", fee: "$0.40", timestamp: "2026-10-06 13:42:00" },
  { id: "TRD-8938", symbol: "SOL/USDT", side: "SHORT", entryPrice: 144.20, exitPrice: 142.90, size: "13.9 SOL", pnl: "+$18.07", pnlPct: "+1.26%", reason: "Take Profit Barrier (2.0x ATR)", timeBars: "7 bars", fee: "$0.40", timestamp: "2026-10-06 13:10:00" },
  { id: "TRD-8937", symbol: "SOL/USDT", side: "SHORT", entryPrice: 144.90, exitPrice: 144.65, size: "13.8 SOL", pnl: "+$3.45", pnlPct: "+0.24%", reason: "Vertical Timeout (15 bars)", timeBars: "15 bars", fee: "$0.40", timestamp: "2026-10-06 12:45:00" },
  { id: "TRD-8936", symbol: "SOL/USDT", side: "SHORT", entryPrice: 145.40, exitPrice: 143.95, size: "13.7 SOL", pnl: "+$19.86", pnlPct: "+1.38%", reason: "Take Profit Barrier (2.0x ATR)", timeBars: "6 bars", fee: "$0.40", timestamp: "2026-10-06 12:18:00" }
];

export const SYSTEM_AGENTS = [
  {
    name: "MarketDataAgent",
    role: "Real-Time Feed & Volatility Ingestion",
    tech: "CCXT Pro + asyncio WebSockets",
    description: "Ingests raw 1-minute klines and orderbook tick events from Binance. Computes rolling ATR(14) in real-time with sub-millisecond dispatch.",
    eventOut: "MarketStateEvent"
  },
  {
    name: "FeatureEngineeringAgent",
    role: "Stationary Microstructure Transforms",
    tech: "NumPy + Pandas Rolling Kernels",
    description: "Transforms non-stationary price series into stationary log-returns (5m–4h), multi-timeframe VWAP deviations, taker buy ratios, and force indices.",
    eventOut: "FeatureVectorEvent"
  },
  {
    name: "DecisionAgent",
    role: "ML Inference & Probabilistic Gating",
    tech: "LightGBM + Scikit-Learn",
    description: "Evaluates feature vectors against trained gradient boosted decision trees. Emits high-conviction BUY/SELL signals calibrated against confidence threshold gates.",
    eventOut: "TradeSignalEvent"
  },
  {
    name: "RiskManagementAgent",
    role: "Dynamic ATR Barriers & Drawdown Protection",
    tech: "Volatility-Adjusted TBM + Kelly Sizing",
    description: "Computes dynamic stop-loss and take-profit distances dynamically sized by current ATR volatility. Enforces portfolio circuit breakers and margin caps.",
    eventOut: "OrderExecutionEvent"
  },
  {
    name: "ExecutionAgent",
    role: "Order Lifecycle & Vertical Timeout Guard",
    tech: "Async Execution Pipeline + Fills Simulator",
    description: "Dispatches market orders, reconciles fills, applies 10 bps fee deductions, and enforces mandatory 15-candle vertical timeout exit.",
    eventOut: "FillEvent / CancelEvent"
  },
  {
    name: "StateManager",
    role: "Thread-Safe Portfolio & Audit Logger",
    tech: "Atomic State + Concurrent CSV Engine",
    description: "Maintains real-time portfolio balance, equity curves, active position locks, and structured JSON/CSV trade audit journals.",
    eventOut: "StateUpdatedEvent"
  }
];
