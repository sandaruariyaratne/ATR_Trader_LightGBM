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
    open_positions = [] # list of (entry_idx, side, entry_price, sl, tp, qty)
    trades_log = []
    equity_curve = []
    
    for i in range(len(closes)):
        close = closes[i]
        high = highs[i]
        low = lows[i]
        
        # 1. Update existing positions
        active_positions = []
        for entry_idx, pos_side, pos_entry_price, pos_sl, pos_tp, pos_qty in open_positions:
            exit_reason = None
            if i - entry_idx >= 15: # 15-minute timeout
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
        
        # 2. Check new entries
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
                
    # Close remaining positions at last close price
    for entry_idx, pos_side, pos_entry_price, pos_sl, pos_tp, pos_qty in open_positions:
        fee = (pos_entry_price + closes[-1]) * pos_qty * (fee_rate / 2.0)
        pnl = (closes[-1] - pos_entry_price) * pos_qty - fee
        if pos_side == "sell":
            pnl = (pos_entry_price - closes[-1]) * pos_qty - fee
        equity += pnl
        trades_log.append(pnl)
        
    total_trades = len(trades_log)
    win_count = sum(1 for p in trades_log if p > 0)
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0.0
    
    gross_profit = sum(p for p in trades_log if p > 0)
    gross_loss = abs(sum(p for p in trades_log if p < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    
    net_pnl_pct = ((equity - capital) / capital) * 100
    
    eq_curve = np.array(equity_curve)
    cum_max = np.maximum.accumulate(eq_curve)
    drawdowns = (cum_max - eq_curve) / cum_max * 100
    max_dd = np.max(drawdowns)
    
    return equity, net_pnl_pct, total_trades, win_rate, profit_factor, max_dd

def main():
    print("Loading universal model ...", flush=True)
    with open("data/models/lightgbm_universal.pkl", "rb") as f:
        artifact = pickle.load(f)
    model = artifact["model"]
    feature_cols = artifact["feature_names"]
    label_map = artifact["label_map"]
    
    print("Loading BNB/USDT data ...", flush=True)
    df = pd.read_csv("data/BNBUSDT.csv", sep="|", header=None, names=COL_NAMES)
    df = df[df["volume"] > 0].reset_index(drop=True)
    df = df[df["timestamp"] >= 1640995200].reset_index(drop=True) # Post-Jan 2022
    
    print("Building features ...", flush=True)
    df = build_stationary_features(df)
    
    # Calculate ATR on-the-fly
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
    
    # Split 80/20 to extract out-of-sample test set (unseen data)
    n_rows = len(df)
    split_idx = int(n_rows * 0.8)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    
    # Pre-calculate model predictions
    print("Running predictions on test set ...", flush=True)
    X_test = test_df[feature_cols].values
    probas = model.predict_proba(X_test)
    pred_classes = np.argmax(probas, axis=1)
    confidences = probas[np.arange(len(probas)), pred_classes]
    
    closes = test_df["close"].values
    highs = test_df["high"].values
    lows = test_df["low"].values
    atrs = test_df["atr"].values
    
    # Setup the sweep parameters
    sl_options = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    tp_options = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
    
    confidence_threshold = 0.42
    pos_size_pct = 0.10
    leverage = 2.0
    fee_rate = 0.0002
    sell_only = True
    
    results = []
    
    print("\nStarting SL/TP Grid Sweep for BNB ...", flush=True)
    print(f"{'SL Mult':<10}{'TP Mult':<10}{'Net Profit %':<15}{'Trades':<10}{'Win Rate %':<12}{'Profit Factor':<15}{'Max DD %':<10}", flush=True)
    print("-" * 85, flush=True)
    
    for sl in sl_options:
        for tp in tp_options:
            final_equity, pnl_pct, trades, win_rate, pf, max_dd = run_simulation(
                closes, highs, lows, atrs, pred_classes, confidences, label_map,
                confidence_threshold, sl, tp, pos_size_pct, leverage, fee_rate, sell_only
            )
            print(f"{sl:<10.1f}{tp:<10.1f}{pnl_pct:<15.2f}{trades:<10}{win_rate:<12.2f}{pf:<15.3f}{max_dd:<10.2f}", flush=True)
            results.append({
                'sl': sl,
                'tp': tp,
                'pnl_pct': pnl_pct,
                'equity': final_equity,
                'trades': trades,
                'win_rate': win_rate,
                'profit_factor': pf,
                'max_dd': max_dd
            })
            
    # Sort results by profit
    sorted_results = sorted(results, key=lambda x: x['pnl_pct'], reverse=True)
    
    print("\n" + "="*80)
    print("TOP 5 OPTIMAL SL/TP PARAMETERS FOR BNB (INCLUDING HIGH/LOW VALUES)")
    print("="*80)
    for idx, r in enumerate(sorted_results[:5]):
        print(f"{idx+1}. SL Mult: {r['sl']}x ATR | TP Mult: {r['tp']}x ATR")
        print(f"   - Net Profit: {r['pnl_pct']:.2f}% (Final Equity: ${r['equity']:.2f})")
        print(f"   - Trades Executed: {r['trades']} (Win Rate: {r['win_rate']:.2f}%)")
        print(f"   - Profit Factor: {r['profit_factor']:.3f} | Max Drawdown: {r['max_dd']:.2f}%")
        print()

if __name__ == '__main__':
    main()
