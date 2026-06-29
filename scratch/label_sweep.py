import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add the project root to PYTHONPATH to import make_labels
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.features import make_labels

def main():
    csv_p = Path("data/SOLUSDT.csv")
    delimiter = "|"
    
    print(f"Loading dataset from {csv_p} …")
    peek_df = pd.read_csv(csv_p, sep=delimiter, header=None, nrows=2)
    num_cols = peek_df.shape[1]
    
    col_names = [
        "timestamp", "open", "high", "low", "close", "volume",
        "quote_volume", "taker_buy_base_volume", "taker_buy_quote_volume", "trades"
    ]
    
    if num_cols > len(col_names):
        col_names.extend([f"col_{i}" for i in range(len(col_names), num_cols)])
    elif num_cols < len(col_names):
        col_names = col_names[:num_cols]
        
    df = pd.read_csv(csv_p, sep=delimiter, header=None, names=col_names)
    print(f"Loaded {len(df):,} rows.")

    df.sort_values("timestamp", inplace=True)
    df.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)

    min_ts = df["timestamp"].min()
    max_ts = df["timestamp"].max()
    expected_rows = (max_ts - min_ts) // 60 + 1
    if len(df) < expected_rows:
        full_ts = np.arange(min_ts, max_ts + 60, 60)
        df = df.set_index("timestamp").reindex(full_ts)
        df.index.name = "timestamp"
        df = df.reset_index()
        
        price_cols = ["open", "high", "low", "close"]
        df[price_cols] = df[price_cols].ffill()
        
        volume_cols = ["volume", "taker_buy_base_volume", "taker_buy_quote_volume", "quote_volume", "trades"]
        df[volume_cols] = df[volume_cols].fillna(0.0)

    # Replicate the exact row filtering used in label_dataset.py
    # We will compute labels, then chop off the 60 warmup rows and 15 tail rows.
    # We will do this sweep from 0.5% to 2.0% in steps of 0.1% (0.005 to 0.020).
    
    print("\n| Barrier (TP & SL %) | BUY % | SELL % | HOLD % | BUY Count | SELL Count | HOLD Count |")
    print("|---|---|---|---|---|---|---|")
    
    for pct_x10 in range(5, 21):
        pct = pct_x10 / 1000.0  # 0.005, 0.006, ..., 0.020
        
        # Compute labels
        lbls = make_labels(df, tp_pct=pct, sl_pct=pct, window=15)
        
        # Chop edges
        lbls_chopped = lbls.iloc[60:-15]
        
        counts = lbls_chopped.value_counts()
        total = len(lbls_chopped)
        
        buy_cnt = counts.get(2, 0)
        sell_cnt = counts.get(0, 0)
        hold_cnt = counts.get(1, 0)
        
        buy_pct = (buy_cnt / total) * 100
        sell_pct = (sell_cnt / total) * 100
        hold_pct = (hold_cnt / total) * 100
        
        print(f"| {pct*100:.1f}% | {buy_pct:.2f}% | {sell_pct:.2f}% | {hold_pct:.2f}% | {buy_cnt:,} | {sell_cnt:,} | {hold_cnt:,} |")

if __name__ == "__main__":
    main()
