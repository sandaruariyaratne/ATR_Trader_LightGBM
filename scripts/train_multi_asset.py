"""
scripts/train_multi_asset.py
─────────────────────────────
Universal multi-asset LightGBM training pipeline.

Strategy:
  - Each asset's features and labels are computed INDEPENDENTLY within its own
    time series, preventing any rolling-window or label look-forward contamination
    at asset boundaries.
  - Only after per-asset feature/label generation are all DataFrames concatenated
    and shuffled for training.
  - The resulting model is asset-agnostic and can be deployed to any asset.

Usage:
    python scripts/train_multi_asset.py
    python scripts/train_multi_asset.py --tp-mult 3.0 --sl-mult 1.5 --fee-rate 0.0002
"""
from __future__ import annotations

import os
import sys
# Add project root to path for local imports
sys.path.append(os.getcwd())

import pickle
import time
from pathlib import Path

import click
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

from utils.features import build_stationary_features, make_dynamic_volatility_labels


# ─── Asset Registry ───────────────────────────────────────────────────────────

ASSETS = [
    {"symbol": "SOLUSDT",  "csv": "data/SOLUSDT.csv"},
    {"symbol": "BTCUSDT",  "csv": "data/BTCUSDT.csv"},
    {"symbol": "ETHUSDT",  "csv": "data/ETHUSDT.csv"},
    {"symbol": "BNBUSDT",  "csv": "data/BNBUSDT.csv"},
    {"symbol": "AVAXUSDT", "csv": "data/AVAXUSDT.csv"},
]

FEATURE_COLS = [
    "log_return_5m", "log_return_15m", "log_return_30m", "log_return_1h", "log_return_4h",
    "vwap_dev_5m", "vwap_dev_15m", "vwap_dev_1h", "vwap_dev_4h",
    "relative_volume", "volatility_5m", "net_taker_flow_5m", "net_taker_flow_15m",
    "volume_weighted_return_5m", "whale_buying_factor_5m", "vwap_dev_slope_3m",
    "candle_body_ratio", "upper_wick_ratio", "lower_wick_ratio",
    "extreme_ratio_1h", "extreme_ratio_4h",
    "volatility_1h", "volatility_4h", "volatility_ratio",
    "tvi_5m", "tvi_15m", "tvi_1h",
    "taker_imbalance_1m", "taker_imbalance_5m", "taker_imbalance_15m",
    "taker_imbalance_60m", "taker_imbalance_240m",
    "force_index_1m", "force_index_5m", "force_index_15m",
    "volume_acc_15m",
    "btc_log_return_5m", "btc_log_return_1h", "asset_btc_correlation_1h",
    "distance_to_daily_200ema",
    "funding_rate", "funding_rate_velocity",
]

COL_NAMES = [
    "timestamp", "open", "high", "low", "close", "volume",
    "taker_buy_quote_volume", "taker_buy_base_volume", "quote_volume", "trades"
]

TBM_MAP = {2: "BUY", 0: "SELL", 1: "HOLD"}


# ─── Per-Asset Processing ─────────────────────────────────────────────────────

def process_asset(symbol: str, csv_path: str, tp_mult: float, sl_mult: float, fee_rate: float, start_timestamp: int = 1629000000) -> pd.DataFrame:
    """
    Load raw OHLCV, compute features, compute labels — all within this asset's
    own time series. Returns a clean DataFrame of (features + label) rows.
    """
    t0 = time.time()
    print(f"\n[{symbol}] Loading {csv_path} ...", flush=True)

    df = pd.read_csv(csv_path, sep="|", header=None, names=COL_NAMES, dtype={
        "timestamp": "int64", "open": "float64", "high": "float64",
        "low": "float64", "close": "float64", "volume": "float64",
        "taker_buy_quote_volume": "float64", "taker_buy_base_volume": "float64",
        "quote_volume": "float64", "trades": "float64"
    })
    print(f"[{symbol}] Loaded {len(df):,} rows in {time.time()-t0:.1f}s", flush=True)

    # Drop rows with zero volume (dead/delisted periods, exchange outages)
    df = df[df["volume"] > 0].reset_index(drop=True)
    print(f"[{symbol}] After zero-volume filter: {len(df):,} rows", flush=True)

    # Keep only rows onward from August 15, 2021 (timestamp >= 1629000000)
    df = df[df["timestamp"] >= start_timestamp].reset_index(drop=True)
    print(f"[{symbol}] Filtered to onward {start_timestamp} (Aug 2021): {len(df):,} rows", flush=True)

    # Feature engineering (within this asset's time series only)
    t1 = time.time()
    print(f"[{symbol}] Computing features ...", flush=True)
    df = build_stationary_features(df)
    
    # Load BTC for Category A features alignment
    print(f"[{symbol}] Aligning with BTCUSDT for Category A features ...", flush=True)
    df_btc = pd.read_csv("data/BTCUSDT.csv", sep="|", header=None, names=COL_NAMES, dtype={
        "timestamp": "int64", "open": "float64", "high": "float64",
        "low": "float64", "close": "float64", "volume": "float64",
        "taker_buy_quote_volume": "float64", "taker_buy_base_volume": "float64",
        "quote_volume": "float64", "trades": "float64"
    })
    df_btc = df_btc[df_btc["volume"] > 0].reset_index(drop=True)
    
    # Merge BTC close
    df = pd.merge(df, df_btc[["timestamp", "close"]], on="timestamp", how="left", suffixes=("", "_btc"))
    df["close_btc"] = df["close_btc"].ffill().bfill()
    
    # Calculate returns and correlation
    df["btc_log_return_5m"] = np.log(df["close_btc"] / df["close_btc"].shift(5))
    df["btc_log_return_1h"] = np.log(df["close_btc"] / df["close_btc"].shift(60))
    df["log_ret_1m"] = np.log(df["close"] / df["close"].shift(1))
    df["btc_log_ret_1m"] = np.log(df["close_btc"] / df["close_btc"].shift(1))
    df["asset_btc_correlation_1h"] = df["log_ret_1m"].rolling(60).corr(df["btc_log_ret_1m"]).fillna(1.0)
    
    # Daily 200 EMA distance
    df["date"] = pd.to_datetime(df["timestamp"], unit="s").dt.date
    daily_close = df.groupby("date")["close"].last().reset_index()
    daily_close["daily_200ema"] = daily_close["close"].ewm(span=200, adjust=False).mean()
    df = pd.merge(df, daily_close[["date", "daily_200ema"]], on="date", how="left")
    df["distance_to_daily_200ema"] = (df["close"] - df["daily_200ema"]) / df["daily_200ema"]
    
    # Cleanup temporary columns
    df.drop(columns=["close_btc", "log_ret_1m", "btc_log_ret_1m", "date", "daily_200ema"], inplace=True)
    
    # Load Funding Rates for Category B features
    print(f"[{symbol}] Loading historical funding rates ...", flush=True)
    try:
        df_funding = pd.read_csv("data/futures_funding_rates.csv")
        df_funding = df_funding[df_funding["symbol"] == symbol].copy()
        
        # Merge funding rates
        if len(df_funding) > 0:
            df = pd.merge(df, df_funding[["timestamp", "funding_rate"]], on="timestamp", how="left")
            df["funding_rate"] = df["funding_rate"].ffill().bfill()
        else:
            df["funding_rate"] = 0.0
    except Exception as e:
        print(f"[{symbol}] Error loading funding rates: {e}. Defaulting to 0.0.")
        df["funding_rate"] = 0.0
        
    # Calculate funding rate velocity (diff from 8 hours ago, which is 480 minutes)
    df["funding_rate_velocity"] = df["funding_rate"] - df["funding_rate"].shift(480)
    df["funding_rate_velocity"] = df["funding_rate_velocity"].fillna(0.0)
    
    print(f"[{symbol}] Features, Category A, and Category B features computed in {time.time()-t1:.1f}s", flush=True)

    # Label generation with Triple Barrier Method (lookforward within this asset only)
    t2 = time.time()
    print(f"[{symbol}] Generating TBM labels (TP={tp_mult}x ATR, SL={sl_mult}x ATR, fee={fee_rate}) ...", flush=True)
    df["label"] = make_dynamic_volatility_labels(
        df, window=15, tp_mult=tp_mult, sl_mult=sl_mult, fee_rate=fee_rate
    )

    # Remove warmup rows (label = -1)
    df = df[df["label"] != -1].reset_index(drop=True)

    # Map to string labels
    df["label"] = df["label"].map(TBM_MAP)

    # Drop any NaN rows from rolling window burn-in
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    label_dist = df["label"].value_counts().to_dict()
    print(f"[{symbol}] Labels computed in {time.time()-t2:.1f}s | Distribution: {label_dist}", flush=True)
    print(f"[{symbol}] Final rows: {len(df):,}", flush=True)

    # Keep only feature columns and label
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    df = df[available_features + ["label"]].copy()

    # Add symbol tag (for diagnostics only — NOT used as a feature to avoid leakage)
    df["_symbol"] = symbol

    return df


# ─── Training ─────────────────────────────────────────────────────────────────

def train_lightgbm(X_train, y_train):
    from lightgbm import LGBMClassifier

    model = LGBMClassifier(
        class_weight="balanced",
        n_estimators=300,
        num_leaves=63,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=50,
        random_state=42,
        n_jobs=2,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


# ─── CLI ──────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--tp-mult", default=3.0, show_default=True, type=float, help="ATR multiplier for TP barrier")
@click.option("--sl-mult", default=1.5, show_default=True, type=float, help="ATR multiplier for SL barrier")
@click.option("--fee-rate", default=0.0002, show_default=True, type=float, help="Round-trip commission fee rate")
@click.option("--output-dir", default="data/models", show_default=True)
@click.option("--output-name", default="lightgbm_universal.pkl", show_default=True)
@click.option("--start-timestamp", default=1629000000, show_default=True, type=int, help="Timestamp threshold to filter onwards (default: Aug 15, 2021)")
def main(tp_mult: float, sl_mult: float, fee_rate: float, output_dir: str, output_name: str, start_timestamp: int) -> None:
    total_start = time.time()
    print("=" * 60)
    print("MULTI-ASSET UNIVERSAL LIGHTGBM TRAINING PIPELINE")
    print(f"Assets: {[a['symbol'] for a in ASSETS]}")
    print(f"Labels: TP={tp_mult}x ATR | SL={sl_mult}x ATR | Fee={fee_rate}")
    print(f"Start timestamp: {start_timestamp} (onward)")
    print("=" * 60)

    # Step 1: Process each asset INDEPENDENTLY (no contamination at boundaries)
    asset_dfs = []
    for asset in ASSETS:
        df_asset = process_asset(asset["symbol"], asset["csv"], tp_mult, sl_mult, fee_rate, start_timestamp=start_timestamp)
        asset_dfs.append(df_asset)

    # Step 2: Concatenate all asset DataFrames (safe at this point — all features/labels pre-computed)
    print("\n" + "=" * 60)
    print("COMBINING ALL ASSETS", flush=True)
    combined = pd.concat(asset_dfs, ignore_index=True)
    print(f"Total combined rows: {len(combined):,}", flush=True)

    # Per-asset breakdown
    print("\nPer-asset row counts:")
    for sym, cnt in combined["_symbol"].value_counts().sort_index().items():
        print(f"  {sym}: {cnt:,}")

    # Step 3: Shuffle (critical — prevents temporal autocorrelation in training)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    # Step 4: Train/test split (80/20, time-agnostic since rows are shuffled)
    feature_cols = [c for c in FEATURE_COLS if c in combined.columns]
    print(f"\nTraining on {len(feature_cols)} features", flush=True)

    X = combined[feature_cols].values
    le = LabelEncoder()
    y = le.fit_transform(combined["label"])
    label_map = {i: label for i, label in enumerate(le.classes_)}

    print(f"Label distribution: {pd.Series(combined['label']).value_counts().to_dict()}", flush=True)

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Step 5: Train LightGBM
    print(f"\nTraining LightGBM on {len(X_train):,} samples ...", flush=True)
    t_train = time.time()
    model = train_lightgbm(X_train, y_train)
    print(f"Training completed in {time.time()-t_train:.1f}s", flush=True)

    # Step 6: Evaluate
    y_pred = model.predict(X_test)
    print("\nTest-set performance:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Step 7: Save model artifact
    out_path = Path(output_dir) / output_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "version": "2.0.0-multi-asset",
        "label_map": label_map,
        "feature_names": feature_cols,
        "model_type": "lightgbm",
        "assets_trained_on": [a["symbol"] for a in ASSETS],
        "tp_mult": tp_mult,
        "sl_mult": sl_mult,
        "fee_rate": fee_rate,
    }
    with open(out_path, "wb") as f:
        pickle.dump(artifact, f)

    print(f"\nSaved universal model → {out_path}")
    print(f"Total pipeline time: {time.time()-total_start:.1f}s")


if __name__ == "__main__":
    main()
