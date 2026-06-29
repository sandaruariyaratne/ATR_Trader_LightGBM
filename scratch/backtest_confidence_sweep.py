"""
scratch/backtest_confidence_sweep.py
───────────────────────────────────
Performs a fast parameter sweep of confidence thresholds from 40% to 55% in 1% steps.
"""
from __future__ import annotations
import sys
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.backtest import _build_feature_matrix, BacktestTrade, BacktestResult

def run_fast_simulation(df, probas, label_map, confidence_threshold, initial_capital=10000.0, max_position_pct=0.10, max_open_positions=100, stop_loss_atr_mult=1.0, take_profit_atr_mult=2.0):
    inv_label = {v: k for k, v in label_map.items()}
    buy_idx  = inv_label.get("BUY")
    sell_idx = inv_label.get("SELL")

    equity = initial_capital
    peak_equity = initial_capital
    result = BacktestResult(initial_capital=initial_capital, final_equity=initial_capital)

    open_positions = []
    
    # Pre-extract numpy arrays for maximum iteration speed
    closes = df["close"].values
    atrs = df["atr"].values
    
    for i in range(len(df)):
        close = closes[i]

        # ── Check open positions for SL/TP ──────────────────────────────────
        active_positions = []
        for position in open_positions:
            exit_reason = None
            if i - position.entry_bar >= 15:
                exit_reason = "timeout"
            elif position.side == "buy":
                if close <= position.stop_loss:
                    exit_reason = "sl"
                elif close >= position.take_profit:
                    exit_reason = "tp"
            else:
                if close >= position.stop_loss:
                    exit_reason = "sl"
                elif close <= position.take_profit:
                    exit_reason = "tp"

            if exit_reason:
                fee = (position.entry_price + close) * position.quantity * 0.0005
                pnl = (close - position.entry_price) * position.quantity - fee
                if position.side == "sell":
                    pnl = (position.entry_price - close) * position.quantity - fee
                position.exit_price = close
                position.exit_bar = i
                position.exit_reason = exit_reason
                position.pnl = pnl
                position.pnl_pct = pnl / (position.entry_price * position.quantity) * 100
                equity += pnl
                result.trades.append(position)
            else:
                active_positions.append(position)
        
        open_positions = active_positions

        # ── Entry logic ────────────────────────────────────────────────────
        proba = probas[i]
        pred_class = int(np.argmax(proba))
        confidence = float(proba[pred_class])
        action = label_map.get(pred_class, "HOLD")

        if len(open_positions) < max_open_positions and confidence >= confidence_threshold and action in ("BUY", "SELL"):
            qty = (equity * max_position_pct) / close
            atr = atrs[i]
            if action == "BUY":
                sl = close - (stop_loss_atr_mult * atr)
                tp = close + (take_profit_atr_mult * atr)
            else:
                sl = close + (stop_loss_atr_mult * atr)
                tp = close - (take_profit_atr_mult * atr)

            new_position = BacktestTrade(
                idx=i, side=action.lower(),
                entry_price=close, exit_price=close,
                quantity=qty, stop_loss=sl, take_profit=tp,
                entry_bar=i, exit_bar=i,
                exit_reason="open", pnl=0.0, pnl_pct=0.0,
            )
            open_positions.append(new_position)

        peak_equity = max(peak_equity, equity)

    # Close any open positions at end of data
    last_close = closes[-1]
    for position in open_positions:
        fee = (position.entry_price + last_close) * position.quantity * 0.0005
        pnl = (last_close - position.entry_price) * position.quantity - fee
        if position.side == "sell":
            pnl = (position.entry_price - last_close) * position.quantity - fee
        position.exit_price = last_close
        position.exit_bar = len(df) - 1
        position.exit_reason = "end_of_data"
        position.pnl = pnl
        equity += pnl
        result.trades.append(position)

    result.final_equity = equity
    return result

def main():
    csv_path = "data/SOLUSDT.csv"
    model_path = "data/models/lightgbm_SOLUSDT.pkl"
    
    print(f"Loading data from {csv_path} …")
    # Peek delimiter
    with open(csv_path, "r") as f:
        first_line = f.readline()
    delim = "|" if "|" in first_line else ","
    
    df = pd.read_csv(csv_path, sep=delim, header=None)
    col_names = [
        "timestamp", "open", "high", "low", "close", "volume",
        "taker_buy_quote_volume", "taker_buy_base_volume", "quote_volume", "trades"
    ]
    num_cols = df.shape[1]
    if num_cols > len(col_names):
        col_names.extend([f"col_{i}" for i in range(len(col_names), num_cols)])
    elif num_cols < len(col_names):
        col_names = col_names[:num_cols]
    df.columns = col_names

    print("Building feature matrix (running once) …")
    t0 = time.time()
    df = _build_feature_matrix(df)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"Features built in {time.time() - t0:.2f} seconds.")

    print(f"Loading model from {model_path} …")
    with open(model_path, "rb") as f:
        artefact = pickle.load(f)
    model = artefact["model"]
    label_map = artefact.get("label_map", {0: "BUY", 1: "HOLD", 2: "SELL"})
    feature_cols = artefact.get("feature_names", [])

    available = [c for c in feature_cols if c in df.columns]
    X = df[available].values
    
    print("Generating model probabilities …")
    probas = model.predict_proba(X)

    print("\n| Confidence Threshold | Total Trades | Win Rate % | Return % | Max Drawdown % | Profit Factor |")
    print("|---|---|---|---|---|---|")

    for thresh_pct in range(40, 56):
        thresh = thresh_pct / 100.0
        
        t_start = time.time()
        res = run_fast_simulation(df, probas, label_map, thresh, max_open_positions=100, stop_loss_atr_mult=1.0, take_profit_atr_mult=2.0)
        summary = res.summary()
        
        win_rate = summary["win_rate_pct"]
        ret_pct = summary["total_return_pct"]
        dd_pct = summary["max_drawdown_pct"]
        trades = summary["total_trades"]
        pf = summary["profit_factor"]
        
        pf_str = f"{pf:.3f}" if pf != float("inf") else "inf"
        print(f"| {thresh_pct}% | {trades:,} | {win_rate:.2f}% | {ret_pct:+.2f}% | {dd_pct:.2f}% | {pf_str} |")

if __name__ == "__main__":
    main()
