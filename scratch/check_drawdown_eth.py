import sys
import os
import pickle
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.features import build_stationary_features

COL_NAMES = [
    "timestamp", "open", "high", "low", "close", "volume",
    "taker_buy_quote_volume", "taker_buy_base_volume", "quote_volume", "trades"
]

def run_simulation(closes, highs, lows, atrs, pred_classes, confidences, label_map, 
                   confidence_threshold, sl_mult, tp_mult, pos_size_pct, leverage, fee_rate, sell_only):
    capital = 10000.0
    equity = capital
    open_positions = []
    trades_log = []
    equity_curve = []
    
    for i in range(len(closes)):
        close = closes[i]
        high = highs[i]
        low = lows[i]
        
        active_positions = []
        for entry_idx, pos_side, pos_entry_price, pos_sl, pos_tp, pos_qty in open_positions:
            exit_reason = None
            if i - entry_idx >= 15:
                exit_reason = "timeout"
                exit_price = close
            elif pos_side == "buy":
                if low <= pos_sl:
                    exit_reason = "sl"
                    exit_price = pos_sl
                elif high >= pos_tp:
                    exit_reason = "tp"
                    exit_price = pos_tp
            else: # sell (short)
                if high >= pos_sl:
                    exit_reason = "sl"
                    exit_price = pos_sl
                elif low <= pos_tp:
                    exit_reason = "tp"
                    exit_price = pos_tp
                    
            if exit_reason:
                fee = (pos_entry_price + exit_price) * pos_qty * (fee_rate / 2.0)
                pnl = (exit_price - pos_entry_price) * pos_qty - fee
                if pos_side == "sell":
                    pnl = (pos_entry_price - exit_price) * pos_qty - fee
                
                equity += pnl
                trades_log.append(pnl)
            else:
                active_positions.append((entry_idx, pos_side, pos_entry_price, pos_sl, pos_tp, pos_qty))
                
        open_positions = active_positions
        
        confidence = confidences[i]
        pred_class = pred_classes[i]
        action = label_map[pred_class]
        
        if confidence >= confidence_threshold:
            should_enter = False
            if sell_only and action == "SELL":
                should_enter = True
                side = "sell"
            elif not sell_only and action in ("BUY", "SELL"):
                should_enter = True
                side = action.lower()
                
            if should_enter and len(open_positions) < 5:
                atr = atrs[i]
                qty = (equity * pos_size_pct * leverage) / close
                
                if side == "buy":
                    sl = close - (sl_mult * atr)
                    tp = close + (tp_mult * atr)
                else: # sell
                    sl = close + (sl_mult * atr)
                    tp = close - (tp_mult * atr)
                    
                open_positions.append((i, side, close, sl, tp, qty))
                
        equity_curve.append(equity)
        
    for entry_idx, pos_side, pos_entry_price, pos_sl, pos_tp, pos_qty in open_positions:
        fee = (pos_entry_price + closes[-1]) * pos_qty * (fee_rate / 2.0)
        pnl = (closes[-1] - pos_entry_price) * pos_qty - fee
        if pos_side == "sell":
            pnl = (pos_entry_price - closes[-1]) * pos_qty - fee
        equity += pnl
        trades_log.append(pnl)
        
    eq_curve = np.array(equity_curve)
    cum_max = np.maximum.accumulate(eq_curve)
    drawdowns = (cum_max - eq_curve) / cum_max * 100
    max_dd = np.max(drawdowns)
    
    return equity, max_dd

def main():
    with open("data/models/lightgbm_universal.pkl", "rb") as f:
        artifact = pickle.load(f)
    model = artifact["model"]
    feature_cols = artifact["feature_names"]
    label_map = artifact["label_map"]
    
    df = pd.read_csv("data/ETHUSDT.csv", sep="|", header=None, names=COL_NAMES)
    df = df[df["volume"] > 0].reset_index(drop=True)
    df = df[df["timestamp"] >= 1640995200].reset_index(drop=True)
    
    df = build_stationary_features(df)
    
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    tr = np.zeros(len(c))
    tr[0] = h[0] - l[0]
    tr[1:] = np.maximum(
        h[1:] - l[1:],
        np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1]))
    )
    df["atr"] = pd.Series(tr).rolling(window=14).mean().values
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    n_rows = len(df)
    split_idx = int(n_rows * 0.8)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    
    X_test = test_df[feature_cols].values
    probas = model.predict_proba(X_test)
    pred_classes = np.argmax(probas, axis=1)
    confidences = probas[np.arange(len(probas)), pred_classes]
    
    closes = test_df["close"].values
    highs = test_df["high"].values
    lows = test_df["low"].values
    atrs = test_df["atr"].values
    
    confidence_threshold = 0.42
    pos_size_pct = 0.10
    leverage = 2.0
    fee_rate = 0.0002
    sell_only = True
    
    # 1. Run Current (1.5 SL, 3.0 TP)
    curr_equity, curr_dd = run_simulation(
        closes, highs, lows, atrs, pred_classes, confidences, label_map,
        confidence_threshold, 1.5, 3.0, pos_size_pct, leverage, fee_rate, sell_only
    )
    
    # 2. Run Top 1 (4.0 SL, 3.0 TP)
    opt1_equity, opt1_dd = run_simulation(
        closes, highs, lows, atrs, pred_classes, confidences, label_map,
        confidence_threshold, 4.0, 3.0, pos_size_pct, leverage, fee_rate, sell_only
    )
    
    # 3. Run Top 5 (3.0 SL, 3.0 TP)
    opt5_equity, opt5_dd = run_simulation(
        closes, highs, lows, atrs, pred_classes, confidences, label_map,
        confidence_threshold, 3.0, 3.0, pos_size_pct, leverage, fee_rate, sell_only
    )
    
    print("\n--- ETH RISK DIAGNOSTIC COMPARISON ---")
    print(f"Current Settings (1.5x SL / 3.0x TP):")
    print(f"  - Final Equity: ${curr_equity:.2f}")
    print(f"  - Maximum Portfolio Drawdown: {curr_dd:.2f}%")
    
    print(f"\nTop 1 Optimal Settings (4.0x SL / 3.0x TP):")
    print(f"  - Final Equity: ${opt1_equity:.2f}")
    print(f"  - Maximum Portfolio Drawdown: {opt1_dd:.2f}%")
    
    print(f"\nTop 5 Optimal Settings (3.0x SL / 3.0x TP):")
    print(f"  - Final Equity: ${opt5_equity:.2f}")
    print(f"  - Maximum Portfolio Drawdown: {opt5_dd:.2f}%")
    
if __name__ == '__main__':
    main()
