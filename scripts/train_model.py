"""
scripts/train_model.py
───────────────────────
Offline model training from historical OHLCV CSV data.

Pipeline:
  1. Load raw OHLCV CSV
  2. Compute all indicators + features (same logic as live pipeline)
  3. Generate labels: 1-bar-forward return → BUY / HOLD / SELL
  4. Train XGBoost classifier
  5. Evaluate on held-out test set
  6. Save artefact to data/models/

Usage:
    python scripts/train_model.py --csv data/raw/btcusdt_1m_90d.csv
    python scripts/train_model.py --csv data/raw/btcusdt_1m_90d.csv --model lightgbm
"""
from __future__ import annotations

import pickle
from pathlib import Path

import click
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from utils.features import build_stationary_features, make_dynamic_volatility_labels



# ─── Training ─────────────────────────────────────────────────────────────────

def train_xgboost(X_train, y_train):
    from xgboost import XGBClassifier
    from sklearn.utils.class_weight import compute_sample_weight

    sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def train_lightgbm(X_train, y_train):
    from lightgbm import LGBMClassifier

    model = LGBMClassifier(
        class_weight="balanced",
        n_estimators=300,
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


# ─── CLI ─────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--csv", "csv_path", required=True, help="Path to OHLCV CSV")
@click.option(
    "--model", "model_type", default="xgboost",
    type=click.Choice(["xgboost", "lightgbm"]), show_default=True,
)
@click.option("--output-dir", default="data/models", show_default=True)
@click.option("--threshold", default=0.60, show_default=True, type=float, help="Imbalance threshold (deprecated/ignored in TBM)")
@click.option("--label-window", default=15, show_default=True, type=int, help="Time limit window in minutes for TBM barrier evaluation")
@click.option("--tp-mult", default=2.0, show_default=True, type=float, help="ATR multiplier for Upper Profit Barrier (TP)")
@click.option("--sl-mult", default=1.0, show_default=True, type=float, help="ATR multiplier for Lower Risk Barrier (SL)")
@click.option("--fee-rate", default=0.0010, show_default=True, type=float, help="Round-trip commission fee rate (e.g. 0.0010 for 10 bps)")
def main(
    csv_path: str,
    model_type: str,
    output_dir: str,
    threshold: float,
    label_window: int,
    tp_mult: float,
    sl_mult: float,
    fee_rate: float,
) -> None:
    print(f"Loading data from {csv_path} …")
    with open(csv_path, "r") as f:
        first_line = f.readline()
    delim = "|" if "|" in first_line else ","
    if delim == "|":
        df = pd.read_csv(csv_path, sep=delim, header=None)
        df.columns = [
            "timestamp", "open", "high", "low", "close", "volume",
            "taker_buy_quote_volume", "taker_buy_base_volume", "quote_volume", "trades"
        ]
    else:
        df = pd.read_csv(csv_path)
        df.columns = [c.lower() for c in df.columns]

    # Map numeric labels to standard strings for system-wide compatibility
    tbm_map = {
        2: "BUY", 0: "SELL", 1: "HOLD",
        2.0: "BUY", 0.0: "SELL", 1.0: "HOLD",
        "2": "BUY", "0": "SELL", "1": "HOLD",
        "2.0": "BUY", "0.0": "SELL", "1.0": "HOLD"
    }
    if "label" in df.columns and df["label"].isin(tbm_map.keys()).any():
        print("Mapping numeric labels (2, 0, 1) back to class strings (BUY, SELL, HOLD) …")
        df["label"] = df["label"].map(tbm_map)

    required_cols = [
        "log_return_5m", "log_return_15m", "log_return_30m", "log_return_1h", "log_return_4h",
        "vwap_dev_15m", "vwap_dev_1h", "vwap_dev_4h"
    ]
    if any(col not in df.columns for col in required_cols):
        print("Required optimal features not found in CSV. Rebuilding on-the-fly...")
        df = build_stationary_features(df)
        df["label"] = make_dynamic_volatility_labels(df, window=label_window, tp_mult=tp_mult, sl_mult=sl_mult, fee_rate=fee_rate)
        df = df[df["label"] != -1].reset_index(drop=True)
        df["label"] = df["label"].map(tbm_map)
        df.dropna(inplace=True)

    feature_cols = [
        "log_return_5m", "log_return_15m", "log_return_30m", "log_return_1h", "log_return_4h",
        "vwap_dev_15m", "vwap_dev_1h", "vwap_dev_4h"
    ]
    print(f"Training on 8 core features: {feature_cols}")
    X = df[feature_cols].values
    le = LabelEncoder()
    y = le.fit_transform(df["label"])

    label_map = {i: label for i, label in enumerate(le.classes_)}
    print(f"Label distribution: {dict(zip(le.classes_, np.bincount(y)))}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    print(f"Training {model_type} on {len(X_train):,} samples …")
    if model_type == "xgboost":
        model = train_xgboost(X_train, y_train)
    else:
        model = train_lightgbm(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\nTest-set performance:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    safe_sym = csv_path.split("/")[-1].split("_")[0]
    out_path = Path(output_dir) / f"{model_type}_{safe_sym}.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    artefact = {
        "model": model,
        "version": "1.0.0",
        "label_map": label_map,
        "feature_names": feature_cols,
        "model_type": model_type,
    }
    with open(out_path, "wb") as f:
        pickle.dump(artefact, f)

    print(f"\nSaved model → {out_path}")


if __name__ == "__main__":
    main()
