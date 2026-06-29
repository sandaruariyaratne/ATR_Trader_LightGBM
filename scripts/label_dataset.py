"""
scripts/label_dataset.py
────────────────────────
Label historical OHLCV data using forward returns.

Usage:
    python scripts/label_dataset.py --csv data/SOLUSDT.csv --output data/SOLUSDT_labeled.csv
"""
from __future__ import annotations

import time
from pathlib import Path

import click
import numpy as np
import pandas as pd

from utils.features import build_stationary_features, make_dynamic_volatility_labels


@click.command()
@click.option("--csv", "csv_path", required=True, type=click.Path(exists=True), help="Path to raw OHLCV CSV file")
@click.option("--output", "output_path", default=None, help="Path to write the labeled CSV file (auto-named if omitted)")
@click.option("--threshold", default=0.60, show_default=True, type=float, help="Imbalance threshold (deprecated/ignored in TBM)")
@click.option("--label-window", default=15, show_default=True, type=int, help="Time limit window in minutes for TBM barrier evaluation")
@click.option("--tp-mult", default=2.0, show_default=True, type=float, help="ATR multiplier for Upper Profit Barrier (TP)")
@click.option("--sl-mult", default=1.0, show_default=True, type=float, help="ATR multiplier for Lower Risk Barrier (SL)")
@click.option("--fee-rate", default=0.0010, show_default=True, type=float, help="Round-trip commission fee rate (e.g. 0.0010 for 10 bps)")
@click.option("--delimiter", default="|", show_default=True, help="Delimiter of the input CSV file")
def main(
    csv_path: str,
    output_path: str | None,
    threshold: float,
    label_window: int,
    tp_mult: float,
    sl_mult: float,
    fee_rate: float,
    delimiter: str,
) -> None:
    csv_p = Path(csv_path)
    if output_path is None:
        out_p = csv_p.parent / f"{csv_p.stem}_labeled.csv"
    else:
        out_p = Path(output_path)

    print(f"Loading dataset from {csv_p} (delimiter: {repr(delimiter)}) …")
    t0 = time.time()
    
    # Peek at the file to determine columns
    peek_df = pd.read_csv(csv_p, sep=delimiter, header=None, nrows=2)
    num_cols = peek_df.shape[1]
    
    # FIX B: Arranged exactly to match your true source configuration
    col_names = [
        "timestamp", "open", "high", "low", "close", "volume",
        "taker_buy_quote_volume", "taker_buy_base_volume", "quote_volume", "trades"
    ]
    
    if num_cols > len(col_names):
        col_names.extend([f"col_{i}" for i in range(len(col_names), num_cols)])
    elif num_cols < len(col_names):
        col_names = col_names[:num_cols]
        
    df = pd.read_csv(csv_p, sep=delimiter, header=None, names=col_names)
    t1 = time.time()
    print(f"Loaded {len(df):,} rows in {t1 - t0:.2f} seconds.")

    if "close" not in df.columns:
        raise ValueError("The dataset does not contain a 'close' column required for labeling.")

    # Sort and remove duplicate timestamps
    df.sort_values("timestamp", inplace=True)
    num_dups = df["timestamp"].duplicated().sum()
    if num_dups > 0:
        print(f"Detecting {num_dups:,} duplicate timestamps. Removing duplicates (keeping the last observation)...")
        df.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)

    # Detect and treat missing minutes (gaps) in timestamps
    min_ts = df["timestamp"].min()
    max_ts = df["timestamp"].max()
    expected_rows = (max_ts - min_ts) // 60 + 1
    if len(df) < expected_rows:
        print(f"Detecting {expected_rows - len(df):,} missing minute gaps in timestamps. Treating gaps by reindexing and forward-filling...")
        full_ts = np.arange(min_ts, max_ts + 60, 60)
        df = df.set_index("timestamp").reindex(full_ts)
        df.index.name = "timestamp"
        df = df.reset_index()
        
        # Forward fill prices
        price_cols = ["open", "high", "low", "close"]
        df[price_cols] = df[price_cols].ffill()
        
        # Zero fill volume/trade metrics
        volume_cols = ["volume", "taker_buy_base_volume", "taker_buy_quote_volume", "quote_volume", "trades"]
        df[volume_cols] = df[volume_cols].fillna(0.0)

    print("Generating engineered, stationary features...")
    df = build_stationary_features(df)

    print(f"Labeling bars with Volatility-Adjusted TBM (window={label_window}, tp_mult={tp_mult}, sl_mult={sl_mult}, fee_rate={fee_rate}) …")
    t2 = time.time()
    df["label"] = make_dynamic_volatility_labels(df, window=label_window, tp_mult=tp_mult, sl_mult=sl_mult, fee_rate=fee_rate)
    t3 = time.time()
    print(f"Labeling completed in {t3 - t2:.4f} seconds.")

    # FIX A: PURGE BURN-IN WARMUP AND UNLABELLED TERMINAL TAILS CLEANLY
    initial_len = len(df)
    
    # 1. Chop off the top 60 rows where 1-hour rolling metrics are incomplete/0.0 fallback
    df = df.iloc[60:].reset_index(drop=True)
    
    # 2. Chop off the tail window where forward labeling couldn't resolve (-1 flags)
    df = df[df["label"] != -1].reset_index(drop=True)
    
    # 3. Final structural fallback check for any stray NaNs or infinities
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)
    df = df.reset_index(drop=True)
    
    print(f"Removed {initial_len - len(df):,} initialization edge rows (60 warmup rows + {label_window} tail rows).")

    # Drop raw price, volume, and timestamp columns entirely
    raw_cols = ["timestamp", "open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base_volume", "taker_buy_quote_volume", "trades"]
    df.drop(columns=raw_cols, inplace=True, errors="ignore")

    # Cast label to integer type explicitly
    df["label"] = df["label"].astype(int)

    # Show label distribution mapping integers to human classes for inspection
    tbm_name_map = {2: "BUY", 0: "SELL", 1: "HOLD"}
    distribution = df["label"].value_counts()
    print("\nLabel distribution:")
    for label_val, count in distribution.items():
        pct = (count / len(df)) * 100
        class_name = tbm_name_map.get(label_val, f"UNKNOWN_{label_val}")
        print(f"  {class_name:<5} ({label_val}): {count:>10,} ({pct:.2f}%)")

    print(f"\nSaving labeled dataset to {out_p} …")
    t4 = time.time()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_p, index=False)
    t5 = time.time()
    print(f"Saved labeled CSV in {t5 - t4:.2f} seconds.")


if __name__ == "__main__":
    main()