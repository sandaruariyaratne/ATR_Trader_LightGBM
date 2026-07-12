import sys
import os
import csv
import json
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def parse_time(time_str):
    try:
        dt = datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S.%f')
    except ValueError:
        try:
            dt = datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S')
        except ValueError:
            dt = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
    return dt.replace(tzinfo=timezone.utc)

def read_workspace_klines(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    parts = content.split('---\n\n')
    if len(parts) > 1:
        json_str = parts[1].strip()
    else:
        json_str = content.strip()
    return json.loads(json_str)

def run_simulation(entries, exits, df, candle_map, sl_mult, tp_mult, n):
    fee_rate = 0.0002
    simulated_trades = []
    
    for e in entries:
        order_id = e['order_id']
        entry_ms = int(e['time'].timestamp() * 1000)
        
        target_ms = (entry_ms // 60000) * 60000
        found_candle_ts = None
        for offset in [0, -60000, 60000, -120000, 120000, -180000, 180000]:
            if target_ms + offset in candle_map:
                found_candle_ts = target_ms + offset
                break
                
        if not found_candle_ts:
            continue
            
        entry_candle = candle_map[found_candle_ts]
        entry_idx = entry_candle['index']
        atr = entry_candle['atr']
        
        entry_price = e['price']
        sl = entry_price + (sl_mult * atr)
        tp = entry_price - (tp_mult * atr)
        qty = e['qty']
        
        exit_reason = "timeout"
        exit_price = entry_price
        
        for offset_idx in range(1, 16):
            curr_idx = entry_idx + offset_idx
            if curr_idx >= n:
                break
                
            curr_candle = df.iloc[curr_idx]
            high = curr_candle['high']
            low = curr_candle['low']
            close = curr_candle['close']
            
            if high >= sl:
                exit_reason = "sl"
                exit_price = sl
                break
            elif low <= tp:
                exit_reason = "tp"
                exit_price = tp
                break
                
            exit_price = close
            
        fee = (entry_price + exit_price) * qty * (fee_rate / 2.0)
        pnl = (entry_price - exit_price) * qty - fee
        pnl_pct = (pnl / (entry_price * qty)) * 100 if qty > 0 else 0.0
        result = "PROFIT" if pnl > 0 else "LOSS"
        
        simulated_trades.append({
            'order_id': order_id,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'exit_reason': exit_reason,
            'result': result
        })
        
    win_count = sum(1 for t in simulated_trades if t['pnl'] > 0)
    loss_count = sum(1 for t in simulated_trades if t['pnl'] < 0)
    total = len(simulated_trades)
    win_rate = (win_count / total * 100) if total > 0 else 0.0
    
    gross_profit = sum(t['pnl'] for t in simulated_trades if t['pnl'] > 0)
    gross_loss = abs(sum(t['pnl'] for t in simulated_trades if t['pnl'] < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    net_pnl = gross_profit - gross_loss
    
    reasons = {}
    for t in simulated_trades:
        reasons[t['exit_reason']] = reasons.get(t['exit_reason'], 0) + 1
        
    return net_pnl, win_rate, win_count, loss_count, pf, reasons

def main():
    file_path = '/Users/sandaruariyaratne/Downloads/ATR_Trader_LightGBM/data/trades_log_BNB_USDT.csv'
    entries = []
    exits = {}
    with open(file_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [name.strip() for name in reader.fieldnames] if reader.fieldnames else []
        for r in reader:
            event = r.get('Event', '').strip().upper()
            order_id = r.get('OrderID', '').strip()
            if event == 'ENTRY':
                entries.append({
                    'order_id': order_id,
                    'time_str': r.get('Timestamp', '').strip(),
                    'time': parse_time(r.get('Timestamp', '').strip()),
                    'price': float(r.get('Price', 0.0)),
                    'qty': float(r.get('Quantity', 0.0)),
                    'balance': float(r.get('Total_Balance', 10000.0))
                })
            elif event == 'EXIT':
                exits[order_id] = {
                    'price': float(r.get('Price', 0.0)),
                    'pnl': float(r.get('PnL_USD') or 0.0),
                    'pnl_pct': float(r.get('PnL_Pct') or 0.0),
                    'reason': r.get('Exit_Reason', '').strip().lower(),
                    'result': r.get('Profit_Loss', '').strip().upper()
                }
                
    print(f"Total entry trades found: {len(entries)}")
    
    workspace_files = [
        '/Users/sandaruariyaratne/Downloads/ATR_Trader_LightGBM/data/bnb_klines_1.json',
        '/Users/sandaruariyaratne/Downloads/ATR_Trader_LightGBM/data/bnb_klines_2.json',
        '/Users/sandaruariyaratne/Downloads/ATR_Trader_LightGBM/data/bnb_klines_3.json',
        '/Users/sandaruariyaratne/Downloads/ATR_Trader_LightGBM/data/bnb_klines_4.json',
        '/Users/sandaruariyaratne/Downloads/ATR_Trader_LightGBM/data/bnb_klines_5.json',
        '/Users/sandaruariyaratne/Downloads/ATR_Trader_LightGBM/data/bnb_klines_6.json',
        '/Users/sandaruariyaratne/Downloads/ATR_Trader_LightGBM/data/bnb_klines_7.json',
        '/Users/sandaruariyaratne/Downloads/ATR_Trader_LightGBM/data/bnb_klines_8.json',
    ]
    
    all_ohlcv = []
    for f_path in workspace_files:
        try:
            klines = read_workspace_klines(f_path)
            all_ohlcv.extend(klines)
        except Exception as e:
            print(f"Error reading {f_path}: {e}")
            
    if not all_ohlcv:
        print("No candle data loaded. Please copy the step files to the workspace first!")
        return
        
    all_ohlcv_dict = {k[0]: k for k in all_ohlcv}
    sorted_ts = sorted(all_ohlcv_dict.keys())
    cleaned_ohlcv = [all_ohlcv_dict[ts] for ts in sorted_ts]
    
    print(f"Total merged candles: {len(cleaned_ohlcv)} 1-minute bars.")
    
    df = pd.DataFrame(cleaned_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
        
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    
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
    
    n = len(df)
    candle_map = {}
    for idx, row in df.iterrows():
        min_ts = (int(row['timestamp']) // 60000) * 60000
        candle_map[min_ts] = {
            'index': idx,
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close'],
            'atr': row['atr']
        }
        
    # Get actual results
    act_win_count = 0
    act_loss_count = 0
    act_net_pnl = 0.0
    act_reasons = {}
    for order_id, a in exits.items():
        act_net_pnl += a['pnl']
        if a['result'] == 'PROFIT':
            act_win_count += 1
        elif a['result'] == 'LOSS':
            act_loss_count += 1
        act_reasons[a['reason']] = act_reasons.get(a['reason'], 0) + 1
    act_total = act_win_count + act_loss_count
    act_win_rate = (act_win_count / act_total * 100) if act_total > 0 else 0.0
    
    # Run Option A: 4.0x SL / 6.0x TP
    optA_pnl, optA_wr, optA_w, optA_l, optA_pf, optA_reasons = run_simulation(entries, exits, df, candle_map, 4.0, 6.0, n)
    
    # Run Option B: 3.0x SL / 5.0x TP
    optB_pnl, optB_wr, optB_w, optB_l, optB_pf, optB_reasons = run_simulation(entries, exits, df, candle_map, 3.0, 5.0, n)
    
    print("\n" + "="*80)
    print("SIMULATING OPTIMAL PARAMETERS ON THE ACTUAL BNB LIVE TRADES (30 Completed)")
    print("="*80)
    print(f"\n--- ACTUAL LIVE RESULTS (Current 1.5x SL / 3.0x TP) ---")
    print(f"- Net Profit (USD): ${act_net_pnl:.2f}")
    print(f"- Win Rate: {act_win_rate:.2f}% ({act_win_count} Win / {act_loss_count} Loss)")
    print(f"- Exit Reasons: {', '.join([f'{k.upper()}: {v}' for k,v in act_reasons.items()])}")
    
    print(f"\n--- OPTION A (4.0x SL / 6.0x TP) ---")
    print(f"- Net Profit (USD): ${optA_pnl:.2f} (Improvement: ${optA_pnl - act_net_pnl:+.2f})")
    print(f"- Win Rate: {optA_wr:.2f}% ({optA_w} Win / {optA_l} Loss) (Improvement: {optA_wr - act_win_rate:+.2f}%)")
    print(f"- Profit Factor: {optA_pf:.3f}")
    print(f"- Exit Reasons: {', '.join([f'{k.upper()}: {v}' for k,v in optA_reasons.items()])}")
    
    print(f"\n--- OPTION B (3.0x SL / 5.0x TP) ---")
    print(f"- Net Profit (USD): ${optB_pnl:.2f} (Improvement: ${optB_pnl - act_net_pnl:+.2f})")
    print(f"- Win Rate: {optB_wr:.2f}% ({optB_w} Win / {optB_l} Loss) (Improvement: {optB_wr - act_win_rate:+.2f}%)")
    print(f"- Profit Factor: {optB_pf:.3f}")
    print(f"- Exit Reasons: {', '.join([f'{k.upper()}: {v}' for k,v in optB_reasons.items()])}")
    print("="*80)

if __name__ == '__main__':
    main()
