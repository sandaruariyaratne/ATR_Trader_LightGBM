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
from scripts.backtest import BacktestTrade, BacktestResult, run_backtest

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
    
    print("\n| Confidence Threshold | Return % | Trades | Win Rate % | Max DD % | Profit Factor | Final Equity |")
    print("|---|---|---|---|---|---|---|")
    
    # Define local run_backtest function that uses precomputed features and probas to speed up sweep
    def simulate_backtest(conf):
        equity = capital
        peak_equity = capital
        result = BacktestResult(initial_capital=capital, final_equity=capital)
        
        open_positions = []
        
        for i, row in df.iterrows():
            close = row["close"]
            
            # exits
            active_positions = []
            for position in open_positions:
                exit_reason = None
                if position.side == "buy":
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
            
            # entries
            proba = probas[i]
            pred_class = int(np.argmax(proba))
            confidence = float(proba[pred_class])
            action = label_map.get(pred_class, "HOLD")
            
            if confidence >= conf and action in ("BUY", "SELL"):
                qty = (equity * max_pos_pct) / close
                if action == "BUY":
                    sl = close * (1.0 - 0.003)
                    tp = close * (1.0 + 0.008)
                else:
                    sl = close * (1.0 + 0.003)
                    tp = close * (1.0 - 0.008)
                
                new_position = BacktestTrade(
                    idx=i, side=action.lower(),
                    entry_price=close, exit_price=close,
                    quantity=qty, stop_loss=sl, take_profit=tp,
                    entry_bar=i, exit_bar=i,
                    exit_reason="open", pnl=0.0, pnl_pct=0.0,
                )
                open_positions.append(new_position)
            
            peak_equity = max(peak_equity, equity)
            
        # end of data closeout
        last_close = df["close"].iloc[-1]
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

    for c_pct in range(50, 61):
        conf = c_pct / 100.0
        res = simulate_backtest(conf)
        summ = res.summary()
        print(f"| {c_pct}% | {summ['total_return_pct']:.2f}% | {summ['total_trades']:,} | {summ['win_rate_pct']:.2f}% | {summ['max_drawdown_pct']:.2f}% | {summ['profit_factor']:.3f} | ${summ['final_equity']:.2f} |")

if __name__ == "__main__":
    main()
