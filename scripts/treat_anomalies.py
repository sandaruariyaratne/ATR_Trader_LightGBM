"""
scripts/treat_anomalies.py
───────────────────────────
Detect and treat (winsorize) unusual values/outliers in the labeled CSV.

Usage:
    python scripts/treat_anomalies.py --csv data/SOLUSDT_labeled.csv
"""
from __future__ import annotations

from pathlib import Path
import click
import numpy as np
import pandas as pd


@click.command()
@click.option("--csv", "csv_path", default="data/SOLUSDT_labeled.csv", show_default=True, help="Path to the labeled CSV file")
@click.option("--lower-q", default=0.001, show_default=True, type=float, help="Lower quantile for Winsorization")
@click.option("--upper-q", default=0.999, show_default=True, type=float, help="Upper quantile for Winsorization")
def main(csv_path: str, lower_q: float, upper_q: float) -> None:
    csv_p = Path(csv_path)
    if not csv_p.exists():
        raise FileNotFoundError(f"Labeled CSV file not found: {csv_p}")

    print(f"Loading labeled dataset from {csv_p} …")
    df = pd.read_csv(csv_p)
    print(f"Loaded {len(df):,} rows with {df.shape[1]} columns.")

    # Identify numerical columns to treat
    # Bounded columns (already strictly in range like [-1, 1] or [0, 1]) are excluded.
    exclude_cols = {
        "timestamp", "label",
        "candle_body_ratio", "upper_wick_ratio", "lower_wick_ratio",
        "taker_buy_ratio", "quote_vol_dominance",
        "hour_sin", "hour_cos", "day_sin", "day_cos"
    }
    treat_cols = [c for c in df.columns if c not in exclude_cols]

    print(f"\nAnalyzing and treating {len(treat_cols)} columns with Winsorization ({lower_q*100}% to {upper_q*100}%) ...")
    print(f"{'Column':<25} | {'Outliers Treated':<16} | {'Min (Old -> New)':<30} | {'Max (Old -> New)':<30}")
    print("-" * 115)

    for col in treat_cols:
        series = df[col]
        lower_bound = series.quantile(lower_q)
        upper_bound = series.quantile(upper_q)

        # Count outliers before clipping
        outliers_mask = (series < lower_bound) | (series > upper_bound)
        num_outliers = outliers_mask.sum()

        old_min, old_max = series.min(), series.max()
        
        # Apply winsorization (clipping)
        df[col] = np.clip(series, lower_bound, upper_bound)
        
        new_min, new_max = df[col].min(), df[col].max()

        print(f"{col:<25} | {num_outliers:>16,} | {old_min:>.6f} -> {new_min:>.6f} | {old_max:>.6f} -> {new_max:>.6f}")

    # Overwrite the dataset
    print(f"\nSaving treated dataset to {csv_p} …")
    df.to_csv(csv_p, index=False)
    print("Done! Anomalies have been treated.")


if __name__ == "__main__":
    main()
