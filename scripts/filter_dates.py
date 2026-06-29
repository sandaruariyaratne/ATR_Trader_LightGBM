"""
scripts/filter_dates.py
───────────────────────
Filters out data before August 12, 2021 00:00:00 UTC (timestamp 1628726400)
from both SOLUSDT.csv and SOLUSDT_labeled.csv.

Usage:
    python scripts/filter_dates.py
"""
from __future__ import annotations

import datetime
from pathlib import Path
import pandas as pd

# Cutoff: 2021-08-12 00:00:00 UTC
CUTOFF_TIMESTAMP = 1628726400


def main() -> None:
    cutoff_dt = datetime.datetime.fromtimestamp(CUTOFF_TIMESTAMP, datetime.timezone.utc)
    print(f"Filtering out all records before {cutoff_dt} (timestamp: {CUTOFF_TIMESTAMP})")

    # 1. Filter SOLUSDT.csv (Raw)
    raw_path = Path("data/SOLUSDT.csv")
    if raw_path.exists():
        print(f"\nProcessing {raw_path} …")
        # Load raw without header, sep='|'
        df_raw = pd.read_csv(raw_path, sep="|", header=None)
        orig_len = len(df_raw)
        
        # Column 0 is the timestamp
        df_raw_filtered = df_raw[df_raw[0] >= CUTOFF_TIMESTAMP]
        new_len = len(df_raw_filtered)
        
        print(f"Original rows: {orig_len:,}")
        print(f"Filtered rows: {new_len:,} (Deleted {orig_len - new_len:,} rows)")
        
        if new_len > 0:
            first_ts = int(df_raw_filtered.iloc[0, 0])
            first_dt = datetime.datetime.fromtimestamp(first_ts, datetime.timezone.utc)
            print(f"New start date: {first_dt} (timestamp: {first_ts})")
        
        df_raw_filtered.to_csv(raw_path, sep="|", header=False, index=False)
        print(f"Saved filtered data back to {raw_path}.")
    else:
        print(f"Warning: {raw_path} not found. Skipping.")

    # 2. Filter SOLUSDT_labeled.csv (Labeled)
    labeled_path = Path("data/SOLUSDT_labeled.csv")
    if labeled_path.exists():
        print(f"\nProcessing {labeled_path} …")
        # Load labeled with header, sep=','
        df_labeled = pd.read_csv(labeled_path)
        orig_len = len(df_labeled)
        
        df_labeled_filtered = df_labeled[df_labeled["timestamp"] >= CUTOFF_TIMESTAMP]
        new_len = len(df_labeled_filtered)
        
        print(f"Original rows: {orig_len:,}")
        print(f"Filtered rows: {new_len:,} (Deleted {orig_len - new_len:,} rows)")
        
        if new_len > 0:
            first_ts = int(df_labeled_filtered["timestamp"].iloc[0])
            first_dt = datetime.datetime.fromtimestamp(first_ts, datetime.timezone.utc)
            print(f"New start date: {first_dt} (timestamp: {first_ts})")
        
        df_labeled_filtered.to_csv(labeled_path, index=False)
        print(f"Saved filtered data back to {labeled_path}.")
    else:
        print(f"Warning: {labeled_path} not found. Skipping.")


if __name__ == "__main__":
    main()
