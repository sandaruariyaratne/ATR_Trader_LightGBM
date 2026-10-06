# AlgoTrader Web Dashboard

An interactive quantitative performance dashboard for the AlgoTrader engine.

## Features
- **Real-Time Simulation**: Interactive confidence threshold slider (40%–55%) showing dynamic equity curve, win rate, and drawdown adjustments.
- **TBM Parameter Sweeps**: Deep analysis over 2.55M 1-minute bars across multiple confidence gates and ATR multiplier bounds.
- **Trade Execution Ledger**: Sample audited trade fills with barrier exit categorization (Take Profit, Stop Loss, 15-Bar Vertical Timeout).
- **Architecture & Agents**: Interactive visualization of the 6 asynchronous domain agents and typed event bus.
- **CV Takeaways**: Key quantitative engineering accomplishments for technical reviewers.

## Local Development
```bash
npm install
npm run dev
```

## Production Build
```bash
npm run build
```

## Vercel Deployment
This dashboard is pre-configured for one-click Vercel deployment with zero additional configuration needed.
