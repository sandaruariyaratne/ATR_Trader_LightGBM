# Aligned TBM Confidence Sweep Results (Vertical Barrier Enforced)

Below are the final, fully aligned backtesting results on the 2.5 million SOL/USDT bars (covering approximately **58.23 months** from August 2021 to June 2026).

These metrics reflect the **complete Volatility-Adjusted Triple Barrier Method (TBM)**:
1.  **Horizontal Profit Barrier**: **$2.0 \times \text{ATR}$ (take-profit)**
2.  **Horizontal Risk Barrier**: **$1.0 \times \text{ATR}$ (stop-loss)**
3.  **Vertical Time Barrier**: **15 candles timeout** (forces trade close if neither TP nor SL is hit in 15 periods)
4.  **Transaction Fees**: **10 bps (0.10%) round-trip fee** fully deducted

*Initial Capital: $10,000.00*
*Allows Overlapping Trades (Max 100 concurrent)*

---

## Sweep Performance Table (Vertical Barrier Active)

| Confidence Threshold | Total Return % | Final Equity ($) | Simple Monthly Return % | Compounded Monthly Return % | Estimated Balance after 1 Month ($) | Max Drawdown % | Profit Factor | Win Rate % | Average Trades / Month | Average Trades / Day |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **40%** | -100.00% | **$0.00** | -1.717% | -100.000% | **$0.00** | 100.00% | 0.891 | 41.04% | 4,293.15 | 141.05 |
| **41%** | -99.98% | **$2.00** | -1.717% | -13.607% | **$8,639.30** | 99.98% | 0.906 | 41.42% | 2,936.01 | 96.46 |
| **42%** | -98.80% | **$120.00** | -1.697% | -7.314% | **$9,268.60** | 98.85% | 0.932 | 41.94% | 1,990.92 | 65.41 |
| **43%** | -82.54% | **$1,746.00** | -1.417% | -2.953% | **$9,704.74** | 85.64% | 0.964 | 42.85% | 1,341.84 | 44.09 |
| **44%** | -9.83% | **$9,017.00** | -0.169% | -0.178% | **$9,982.25** | 57.85% | 0.997 | 44.06% | 876.38 | 28.79 |
| **45%** | **+131.53%** | **$23,153.00** | +2.259% | +1.452% | **$10,145.22** | 31.57% | 1.046 | 45.31% | 562.29 | 18.47 |
| **46%** | **+232.43%** | **$33,243.00** | +3.992% | +2.084% | **$10,208.43** | 15.10% | 1.107 | 46.66% | 363.63 | 11.95 |
| **47%** | **+246.52%** | **$34,652.00** | +4.233% | +2.157% | **$10,215.71** | 10.21% | 1.179 | 48.37% | 234.84 | 7.72 |
| **48%** | **+261.96%** | **$36,196.00** | +4.499% | +2.234% | **$10,223.36** | 5.30% | 1.273 | 50.68% | 148.99 | 4.89 |
| **49%** | **+221.30%** | **$32,130.00** | +3.800% | **+2.025%** | **$10,202.46** | **3.89%** | **1.366** | **52.56%** | **97.11** | **3.19** |
| **50%** | **+166.24%** | **$26,624.00** | +2.855% | +1.696% | **$10,169.58** | 4.36% | 1.462 | 54.21% | 63.30 | 2.08 |
| **51%** | **+126.35%** | **$22,635.00** | +2.170% | +1.413% | **$10,141.27** | 2.99% | 1.673 | 56.34% | 38.90 | 1.28 |
| **52%** | **+82.47%** | **$18,247.00** | +1.416% | +1.038% | **$10,103.81** | 1.64% | 1.841 | 57.76% | 24.56 | 0.81 |
| **53%** | **+55.62%** | **$15,562.00** | +0.955% | +0.762% | **$10,076.24** | 1.27% | **2.171** | **60.29%** | **15.61** | **0.51** |
| **54%** | **+34.61%** | **$13,461.00** | +0.594% | +0.512% | **$10,051.17** | 1.16% | 2.119 | 60.73% | 10.80 | 0.35 |
| **55%** | **+24.08%** | **$12,408.00** | +0.414% | +0.371% | **$10,037.12** | 0.85% | 2.196 | 60.39% | 7.85 | 0.26 |

---

## Performance Enhancements from the Vertical Barrier

Implementing the 15-candle vertical barrier time limit significantly improved almost all strategy metrics compared to leaving trades open indefinitely:

1.  **Win Rate Surge**: 
    At the **49% threshold**, the win rate climbed from **49.69% to 52.56%**. At the **53% threshold**, the win rate crossed a major milestone, rising from **56.99% to 60.29%**.
2.  **Drawdown Reduction**: 
    At the **49% threshold**, the maximum drawdown dropped from **4.54% to 3.89%**. At the **53% threshold**, drawdown decreased from **1.48% to 1.27%**.
3.  **Profit Factor Optimization**: 
    At the **53% threshold**, the profit factor increased from **1.987 to 2.171**, confirming that exiting slow trades early prevents capital lockup and avoids late-trade decay.
