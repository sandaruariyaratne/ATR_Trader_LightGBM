import sys
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

# Add the project root to PYTHONPATH to import utilities
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.trading_params import INDICATOR_PARAMS as P
from utils.features import build_stationary_features
from utils.indicators import atr as atr_fn
from scripts.backtest import BacktestTrade, BacktestResult

def main():
    csv_path = "data/SOLUSDT.csv"
    model_path = "data/models/xgboost_SOLUSDT.pkl"
    capital = 10000.0
    max_pos_pct = 0.10
    
    print(f"Loading data from {csv_path} …")
    with open(csv_path, "r") as f:
        first_line = f.readline()
    delim = "|" if "|" in first_line else ","
    
    col_names = [
        "timestamp", "open", "high", "low", "close", "volume",
        "taker_buy_quote_volume", "taker_buy_base_volume", "quote_volume", "trades"
    ]
    df = pd.read_csv(csv_path, sep=delim, header=None, names=col_names)
    
    print(f"Loading model from {model_path} …")
    with open(model_path, "rb") as f:
        artefact = pickle.load(f)
    model = artefact["model"]
    label_map = artefact.get("label_map", {0: "BUY", 1: "HOLD", 2: "SELL"})
    feature_cols = artefact.get("feature_names", [])

    print("Building features snapshot ...")
    df = build_stationary_features(df)
    
    # Compute atr for risk management/exit logic in backtesting
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    df["atr"] = pd.Series(range(len(c))).rolling(P.atr_period + 1).apply(
        lambda idx: atr_fn(
            h[int(idx[0]) : int(idx[-1]) + 1].tolist(),
            l[int(idx[0]) : int(idx[-1]) + 1].tolist(),
            c[int(idx[0]) : int(idx[-1]) + 1].tolist(),
            P.atr_period,
        ), raw=True,
    )
    
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    available = [col for col in feature_cols if col in df.columns]
    X = df[available].values
    
    print("Computing model probabilities once ...")
    probas = model.predict_proba(X)
    
    # Convert dataframe to fast numpy arrays for iteration speedup (10-50x faster than iterrows)
    close_arr = df["close"].values
    high_arr = df["high"].values
    low_arr = df["low"].values
    
    # Grid search parameters
    tps = [0.006, 0.007, 0.008, 0.009, 0.010, 0.011, 0.012, 0.013, 0.014, 0.015]
    confs = [0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.60]
    
    # We will print tables grouped by Take Profit
    for tp_val in tps:
        tp_pct = tp_val * 100
        print(f"\n### 🎯 Take Profit: {tp_pct:.1f}% (Stop Loss: 0.3%, Commission: 0.05% entry/exit)")
        print("| Confidence | Return % | Trades | Win Rate % | Max DD % | Profit Factor | Final Equity |")
        print("|---|---|---|---|---|---|---|")
        
        for conf in confs:
            # Run simulation
            equity = capital
            peak_equity = capital
            result = BacktestResult(initial_capital=capital, final_equity=capital)
            
            # Use raw lists for positions instead of complex classes inside high-frequency loop
            # [side_is_buy, entry_price, stop_loss, take_profit, quantity, entry_bar]
            open_positions = []
            
            for i in range(len(close_arr)):
                close = close_arr[i]
                
                # exits
                active_positions = []
                for pos in open_positions:
                    is_buy = pos[0]
                    entry_price = pos[1]
                    stop_loss = pos[2]
                    take_profit = pos[3]
                    qty = pos[4]
                    entry_bar = pos[5]
                    
                    exit_reason = None
                    if is_buy:
                        if close <= stop_loss:
                            exit_reason = "sl"
                        elif close >= take_profit:
                            exit_reason = "tp"
                    else:
                        if close >= stop_loss:
                            exit_reason = "sl"
                        elif close <= take_profit:
                            exit_reason = "tp"

                    if exit_reason:
                        fee = (entry_price + close) * qty * 0.0005
                        pnl = (close - entry_price) * qty - fee
                        if not is_buy:
                            pnl = (entry_price - close) * qty - fee
                            
                        # Build trade record
                        trade = BacktestTrade(
                            idx=entry_bar, side="buy" if is_buy else "sell",
                            entry_price=entry_price, exit_price=close,
                            quantity=qty, stop_loss=stop_loss, take_profit=take_profit,
                            entry_bar=entry_bar, exit_bar=i,
                            exit_reason=exit_reason, pnl=pnl,
                            pnl_pct=pnl / (entry_price * qty) * 100
                        )
                        equity += pnl
                        result.trades.append(trade)
                    else:
                        active_positions.append(pos)
                open_positions = active_positions
                
                # entries
                proba = probas[i]
                pred_class = int(np.argmax(proba))
                confidence = float(proba[pred_class])
                action = label_map.get(pred_class, "HOLD")
                
                if confidence >= conf and action in ("BUY", "SELL"):
                    qty = (equity * max_pos_pct) / close
                    is_buy = (action == "BUY")
                    if is_buy:
                        sl = close * (1.0 - 0.003)
                        tp = close * (1.0 + tp_val)
                    else:
                        sl = close * (1.0 + 0.003)
                        tp = close * (1.0 - tp_val)
                    
                    open_positions.append([is_buy, close, sl, tp, qty, i])
                
                peak_equity = max(peak_equity, equity)
                
            # end of data closeout
            last_close = close_arr[-1]
            for pos in open_positions:
                is_buy = pos[0]
                entry_price = pos[1]
                stop_loss = pos[2]
                take_profit = pos[3]
                qty = pos[4]
                entry_bar = pos[5]
                
                fee = (entry_price + last_close) * qty * 0.0005
                pnl = (last_close - entry_price) * qty - fee
                if not is_buy:
                    pnl = (entry_price - last_close) * qty - fee
                    
                trade = BacktestTrade(
                    idx=entry_bar, side="buy" if is_buy else "sell",
                    entry_price=entry_price, exit_price=last_close,
                    quantity=qty, stop_loss=stop_loss, take_profit=take_profit,
                    entry_bar=entry_bar, exit_bar=len(close_arr)-1,
                    exit_reason="end_of_data", pnl=pnl,
                    pnl_pct=pnl / (entry_price * qty) * 100
                )
                equity += pnl
                result.trades.append(trade)
                
            result.final_equity = equity
            summ = result.summary()
            
            print(f"| {int(conf*100)}% | {summ['total_return_pct']:.2f}% | {summ['total_trades']:,} | {summ['win_rate_pct']:.2f}% | {summ['max_drawdown_pct']:.2f}% | {summ['profit_factor']:.3f} | ${summ['final_equity']:.2f} |")

if __name__ == "__main__":
    main()
