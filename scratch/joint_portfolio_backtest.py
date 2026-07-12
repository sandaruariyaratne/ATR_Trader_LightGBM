import sys
import os
import pickle
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timezone
warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.features import build_stationary_features

COL_NAMES = [
    "timestamp", "open", "high", "low", "close", "volume",
    "taker_buy_quote_volume", "taker_buy_base_volume", "quote_volume", "trades"
]

ASSETS = {
    "AVAX": {"file": "data/AVAXUSDT.csv", "sl": 3.0, "tp": 2.0},
    "ETH": {"file": "data/ETHUSDT.csv", "sl": 4.0, "tp": 3.0},
    "BTC": {"file": "data/BTCUSDT.csv", "sl": 2.5, "tp": 4.0},
    "SOL": {"file": "data/SOLUSDT.csv", "sl": 4.0, "tp": 8.0},
    "BNB": {"file": "data/BNBUSDT.csv", "sl": 4.0, "tp": 6.0}
}

def main():
    print("Loading universal model ...", flush=True)
    with open("data/models/lightgbm_universal.pkl", "rb") as f:
        artifact = pickle.load(f)
    model = artifact["model"]
    feature_cols = artifact["feature_names"]
    label_map = artifact["label_map"]
    
    # We will load each asset, compute features and predictions, and extract the test set (last 20%)
    asset_dfs = {}
    test_start_ts = 0
    test_end_ts = float("inf")
    
    for symbol, config in ASSETS.items():
        print(f"Loading and processing {symbol} ...", flush=True)
        df = pd.read_csv(config["file"], sep="|", header=None, names=COL_NAMES)
        df = df[df["volume"] > 0].reset_index(drop=True)
        df = df[df["timestamp"] >= 1640995200].reset_index(drop=True) # Post-Jan 2022
        
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
        
        # Unseen out-of-sample data is the last 20%
        n_rows = len(df)
        split_idx = int(n_rows * 0.8)
        test_df = df.iloc[split_idx:].reset_index(drop=True)
        
        # Run predictions
        X_test = test_df[feature_cols].values
        probas = model.predict_proba(X_test)
        pred_classes = np.argmax(probas, axis=1)
        confidences = probas[np.arange(len(probas)), pred_classes]
        
        test_df["pred_class"] = pred_classes
        test_df["confidence"] = confidences
        
        asset_dfs[symbol] = test_df
        
        # Find the overlap of the test sets
        start_ts = test_df["timestamp"].iloc[0]
        end_ts = test_df["timestamp"].iloc[-1]
        test_start_ts = max(test_start_ts, start_ts)
        test_end_ts = min(test_end_ts, end_ts)
        
    print(f"\nAligned out-of-sample test range:")
    print(f"Start: {datetime.fromtimestamp(test_start_ts, tz=timezone.utc)}")
    print(f"End:   {datetime.fromtimestamp(test_end_ts, tz=timezone.utc)}")
    
    # Align all test sets to the overlapping time range
    aligned_dfs = {}
    for symbol, test_df in asset_dfs.items():
        aligned_df = test_df[(test_df["timestamp"] >= test_start_ts) & (test_df["timestamp"] <= test_end_ts)].reset_index(drop=True)
        aligned_dfs[symbol] = aligned_df
        print(f"  - Aligned {symbol} count: {len(aligned_df)} minutes")
        
    # Get a list of unique timestamps
    timestamps = sorted(aligned_dfs["BTC"]["timestamp"].values)
    
    # Convert aligned_dfs into dictionary mapping timestamp -> data for fast minute-by-minute lookup
    print("\nPreparing historical price maps ...", flush=True)
    price_maps = {symbol: {} for symbol in ASSETS.keys()}
    for symbol, df_aligned in aligned_dfs.items():
        for idx, row in df_aligned.iterrows():
            price_maps[symbol][int(row["timestamp"])] = {
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "atr": row["atr"],
                "pred_class": row["pred_class"],
                "confidence": row["confidence"]
            }
            
    # Joint Backtest Simulation
    print("\nRunning joint portfolio backtest simulation post-alignment ...", flush=True)
    initial_capital = 50000.0 # $10k per asset
    equity = initial_capital
    open_positions = [] # list of dicts: {'symbol', 'entry_ts', 'entry_price', 'sl', 'tp', 'qty', 'side'}
    trades_log = []
    equity_curve = []
    
    confidence_threshold = 0.42
    pos_size_pct = 0.10 # 10% margin sizing per trade
    leverage = 2.0
    fee_rate = 0.0002
    sell_only = True
    
    for ts in timestamps:
        # 1. Update existing positions
        active_positions = []
        for pos in open_positions:
            sym = pos["symbol"]
            curr_data = price_maps[sym].get(ts)
            if not curr_data:
                # If data is missing for this minute, keep it active
                active_positions.append(pos)
                continue
                
            high = curr_data["high"]
            low = curr_data["low"]
            close = curr_data["close"]
            
            exit_reason = None
            exit_price = close
            
            # Check duration limit (15 minutes)
            duration_mins = (ts - pos["entry_ts"]) // 60000
            if duration_mins >= 15:
                exit_reason = "timeout"
            elif pos["side"] == "buy":
                if low <= pos["sl"]:
                    exit_reason = "sl"
                    exit_price = pos["sl"]
                elif high >= pos["tp"]:
                    exit_reason = "tp"
                    exit_price = pos["tp"]
            else: # sell (short)
                if high >= pos["sl"]:
                    exit_reason = "sl"
                    exit_price = pos["sl"]
                elif low <= pos["tp"]:
                    exit_reason = "tp"
                    exit_price = pos["tp"]
                    
            if exit_reason:
                fee = (pos["entry_price"] + exit_price) * pos["qty"] * (fee_rate / 2.0)
                pnl = (exit_price - pos["entry_price"]) * pos["qty"] - fee
                if pos["side"] == "sell":
                    pnl = (pos["entry_price"] - exit_price) * pos["qty"] - fee
                
                equity += pnl
                trades_log.append({
                    "symbol": sym,
                    "pnl": pnl,
                    "pnl_pct": (pnl / (pos["entry_price"] * pos["qty"])) * 100 if pos["qty"] > 0 else 0.0,
                    "exit_reason": exit_reason
                })
            else:
                active_positions.append(pos)
                
        open_positions = active_positions
        
        # 2. Check entries for this minute
        # We check signals for all 5 assets. If there are multiple signals, we sort them by confidence
        signals = []
        for sym in ASSETS.keys():
            data = price_maps[sym].get(ts)
            if not data:
                continue
                
            conf = data["confidence"]
            pred = data["pred_class"]
            action = label_map[pred]
            
            # Check if this asset already has less than 5 open positions
            coin_pos_count = sum(1 for p in open_positions if p["symbol"] == sym)
            
            if conf >= confidence_threshold and coin_pos_count < 5:
                should_enter = False
                if sell_only and action == "SELL":
                    should_enter = True
                    side = "sell"
                elif not sell_only and action in ("BUY", "SELL"):
                    should_enter = True
                    side = action.lower()
                    
                if should_enter:
                    signals.append({
                        "symbol": sym,
                        "side": side,
                        "confidence": conf,
                        "close": data["close"],
                        "atr": data["atr"]
                    })
                    
        # Sort signals by confidence descending
        signals = sorted(signals, key=lambda x: x["confidence"], reverse=True)
        
        # Enter signals if we have slots (limit 25 open positions globally, 5 per coin)
        for sig in signals:
            if len(open_positions) >= 25:
                break
                
            sym = sig["symbol"]
            side = sig["side"]
            close = sig["close"]
            atr = sig["atr"]
            
            sl_mult = ASSETS[sym]["sl"]
            tp_mult = ASSETS[sym]["tp"]
            
            # Sizing: 10% margin of current total equity * 2x leverage
            qty = (equity * pos_size_pct * leverage) / close
            
            if side == "buy":
                sl = close - (sl_mult * atr)
                tp = close + (tp_mult * atr)
            else: # sell
                sl = close + (sl_mult * atr)
                tp = close - (tp_mult * atr)
                
            open_positions.append({
                "symbol": sym,
                "entry_ts": ts,
                "entry_price": close,
                "sl": max(sl, 0.0),
                "tp": tp,
                "qty": qty,
                "side": side
            })
            
        equity_curve.append(equity)
        
    # Close any remaining positions at the last available close price
    for pos in open_positions:
        sym = pos["symbol"]
        last_price = aligned_dfs[sym]["close"].iloc[-1]
        fee = (pos["entry_price"] + last_price) * pos["qty"] * (fee_rate / 2.0)
        pnl = (last_price - pos["entry_price"]) * pos["qty"] - fee
        if pos["side"] == "sell":
            pnl = (pos["entry_price"] - last_price) * pos["qty"] - fee
            
        equity += pnl
        trades_log.append({
            "symbol": sym,
            "pnl": pnl,
            "pnl_pct": (pnl / (pos["entry_price"] * pos["qty"])) * 100 if pos["qty"] > 0 else 0.0,
            "exit_reason": "end_of_data"
        })
        
    # Calculate performance metrics
    total_trades = len(trades_log)
    win_count = sum(1 for t in trades_log if t["pnl"] > 0)
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0.0
    
    gross_profit = sum(t["pnl"] for t in trades_log if t["pnl"] > 0)
    gross_loss = abs(sum(t["pnl"] for t in trades_log if t["pnl"] < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    
    total_return_pct = ((equity - initial_capital) / initial_capital) * 100
    
    eq_curve = np.array(equity_curve)
    cum_max = np.maximum.accumulate(eq_curve)
    drawdowns = (cum_max - eq_curve) / cum_max * 100
    max_dd = np.max(drawdowns)
    
    # Calculate months in the test set
    start_dt = datetime.fromtimestamp(test_start_ts, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(test_end_ts, tz=timezone.utc)
    delta_days = (end_dt - start_dt).total_seconds() / 86400.0
    number_of_months = delta_days / 30.4375 # standard average days in month
    
    # Monthly Compound Rate (CAGR)
    # CAGR_monthly = (Final_Equity / Initial_Equity) ** (1 / number_of_months) - 1
    monthly_cagr = ((equity / initial_capital) ** (1.0 / number_of_months) - 1.0) * 100
    
    # Asset wise performance
    print("\n" + "="*80)
    print("JOINT PORTFOLIO BACKTEST PERFORMANCE (UNSEEN OUT-OF-SAMPLE DATA)")
    print("="*80)
    print(f"Time Period: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')} ({number_of_months:.2f} months)")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Final Portfolio Equity: ${equity:,.2f}")
    print(f"Total Net Return: {total_return_pct:.2f}%")
    print(f"Maximum Portfolio Drawdown: {max_dd:.2f}%")
    print(f"Monthly Compound Growth Rate (CAGR): {monthly_cagr:.2f}%")
    print(f"Total Portfolio Trades Executed: {total_trades}")
    print(f"Portfolio Win Rate: {win_rate:.2f}% ({win_count} Win / {total_trades - win_count} Loss)")
    print(f"Portfolio Profit Factor: {profit_factor:.3f}")
    print("="*80)
    
    print("\nASSET-WISE PERFORMANCE CONTRIBTUTION:")
    print(f"{'Asset':<10}{'Trades':<10}{'Win Rate %':<12}{'Net PnL ($)':<15}{'Net PnL %':<10}")
    print("-" * 60)
    for sym in ASSETS.keys():
        asset_trades = [t for t in trades_log if t["symbol"] == sym]
        t_count = len(asset_trades)
        w_count = sum(1 for t in asset_trades if t["pnl"] > 0)
        w_rate = (w_count / t_count * 100) if t_count > 0 else 0.0
        pnl_val = sum(t["pnl"] for t in asset_trades)
        # Sizing was 10k per asset theoretically
        pnl_pct = (pnl_val / 10000.0) * 100
        print(f"{sym:<10}{t_count:<10}{w_rate:<12.2f}${pnl_val:<14.2f}{pnl_pct:.2f}%")
    print("="*80)

if __name__ == '__main__':
    main()
