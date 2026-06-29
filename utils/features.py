"""
utils/features.py
─────────────────
Feature engineering and labeling utilities for historical datasets.
Fully optimized to eliminate look-ahead bias and handle burn-in lookback safely.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_stationary_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate engineered, scale-invariant, and stationary features from raw OHLCV.
    Ensures absolute stationarity across multiple market regimes for SOL/USDT.
    
    NOTE: The first 60 rows will contain fallback 0.0 values due to the 1h rolling window.
    Ensure `df = df.iloc[60:].reset_index(drop=True)` is called prior to model training.
    """
    df = df.copy()
    epsilon = 1e-8

    # 1. Log Returns (Short-term and mid-term momentum anchors)
    for n in [5, 15]:
        df[f"log_return_{n}m"] = np.log(df["close"] / df["close"].shift(n))

    # 2. Candle Body and Wick Ratios (Microscopic structural geometry)
    hl_range = df["high"] - df["low"]
    body = df["close"] - df["open"]
    upper_wick = df["high"] - np.maximum(df["open"], df["close"])
    lower_wick = np.minimum(df["open"], df["close"]) - df["low"]

    df["candle_body_ratio"] = body / (hl_range + epsilon)
    df["upper_wick_ratio"] = upper_wick / (hl_range + epsilon)
    df["lower_wick_ratio"] = lower_wick / (hl_range + epsilon)

    # 3. Distance from Rolling VWAP (5m, 15m, 1h baseline tracking)
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    tp_vol = tp * df["volume"]
    for name, w in [("5m", 5), ("15m", 15), ("1h", 60)]:
        tp_vol_sum = tp_vol.rolling(window=w).sum()
        vol_sum = df["volume"].rolling(window=w).sum()
        vwap = np.where(vol_sum > 0, tp_vol_sum / vol_sum, df["close"])
        df[f"vwap_dev_{name}"] = np.where(vwap > 0, (df["close"] - vwap) / vwap, 0.0)

    # 4. Volume Microstructure Features (Buyer vs Seller aggression dynamics)
    if "taker_buy_base_volume" in df.columns:
        df["taker_buy_ratio"] = np.where(df["volume"] > 0, df["taker_buy_base_volume"] / df["volume"], 0.5)
    if "quote_volume" in df.columns and "taker_buy_quote_volume" in df.columns:
        df["quote_vol_dominance"] = np.where(df["quote_volume"] > 0, df["taker_buy_quote_volume"] / df["quote_volume"], 0.5)

    # Relative Volume (Volume Z-Score relative to the trailing 30 minutes)
    vol_mean_30 = df["volume"].rolling(window=30).mean()
    vol_std_30 = df["volume"].rolling(window=30).std(ddof=1)
    df["relative_volume"] = np.where(vol_std_30 > 0, (df["volume"] - vol_mean_30) / vol_std_30, 0.0)

    # Average Trade Size Change
    if "trades" in df.columns:
        trade_size = np.where(df["trades"] > 0, df["volume"] / df["trades"], 0.0)
        trade_size_series = pd.Series(trade_size, index=df.index)
        trade_size_lag5 = trade_size_series.shift(5)
        df["avg_trade_size_change_5m"] = np.where(trade_size_lag5 > 0, (trade_size_series - trade_size_lag5) / trade_size_lag5, 0.0)

    # 5. Rolling Noise-Filter Volatility (Retained short window anchor only)
    log_ret_1m = np.log(df["close"] / df["close"].shift(1))
    df["volatility_5m"] = log_ret_1m.rolling(window=5).std(ddof=1)

    # 6. Cumulative Net Taker Flow (Directional Order Flow Volume Pressure)
    if "taker_buy_base_volume" in df.columns:
        net_flow = df["taker_buy_base_volume"] - (df["volume"] - df["taker_buy_base_volume"])
        df["net_taker_flow_5m"] = net_flow.rolling(window=5).sum()
        df["net_taker_flow_15m"] = net_flow.rolling(window=15).sum()

    # 7. Volume-Weighted Price Momentum (Isolates high-volume institutional blocks)
    df["volume_weighted_return_5m"] = df["log_return_5m"] * df["relative_volume"]

    # 8. Institutional Participation (Whale Tracking factor via trade sizes)
    if "trades" in df.columns and "taker_buy_base_volume" in df.columns:
        trade_size = np.where(df["trades"] > 0, df["volume"] / df["trades"], 0.0)
        avg_trade_size_series = pd.Series(trade_size, index=df.index)
        whale_mean_30 = avg_trade_size_series.rolling(30).mean()
        whale_std_30 = avg_trade_size_series.rolling(30).std(ddof=1)
        avg_trade_size_z = np.where(whale_std_30 > 0, (trade_size - whale_mean_30) / whale_std_30, 0.0)
        df["whale_buying_factor_5m"] = avg_trade_size_z * df["taker_buy_ratio"]

    # 9. Trend Acceleration (Velocity change away from volume baseline)
    df["vwap_dev_slope_3m"] = (df["vwap_dev_5m"] - df["vwap_dev_5m"].shift(2)) / 2.0

    return df


def make_labels(
    df: pd.DataFrame,
    threshold: float = 0.60,
    window: int = 15,
) -> pd.Series:
    """
    Label each bar based on the Mass-Imbalance Momentum concept over a look-forward horizon:
      - Compute Positive Mass (P) as the sum of distances for future close prices > reference close.
      - Compute Negative Mass (N) as the absolute sum of distances for future close prices < reference close.
      - If P / (P + N) > threshold -> 2 (BUY)
      - If N / (P + N) > threshold -> 0 (SELL)
      - Otherwise -> 1 (HOLD)
    """
    close = df["close"].values
    n_rows = len(df)
    
    # Initialize labels as 1 (HOLD)
    labels = np.full(n_rows, 1, dtype=int)
    
    # Construct a 2D array of future close prices of shape (n_rows, window)
    close_2d = np.zeros((n_rows, window))
    for k in range(1, window + 1):
        close_2d[:-k, k - 1] = close[k:]
        close_2d[-k:, k - 1] = np.nan
        
    # Calculate differences from the reference price close[i]
    diffs = close_2d - close[:, np.newaxis]
    
    pos_mask = diffs > 0
    neg_mask = diffs < 0
    
    P = np.sum(np.where(pos_mask, diffs, 0.0), axis=1)
    N = np.sum(np.where(neg_mask, -diffs, 0.0), axis=1)
    
    total_mass = P + N
    valid_mask = total_mass > 0
    
    ratio_P = np.divide(P, total_mass, out=np.zeros_like(P), where=valid_mask)
    ratio_N = np.divide(N, total_mass, out=np.zeros_like(N), where=valid_mask)
    
    buy_mask = valid_mask & (ratio_P > threshold)
    sell_mask = valid_mask & (ratio_N > threshold)
    
    labels[buy_mask] = 2
    labels[sell_mask] = 0
    
    labels_series = pd.Series(labels, index=df.index, dtype=int)
    
    # Flag tail window rows as -1 since they run out of historical context
    if n_rows >= window:
        labels_series.iloc[-window:] = -1
        
    return labels_series


def make_dynamic_volatility_labels(
    df: pd.DataFrame,
    window: int = 15,
    tp_mult: float = 2.0,
    sl_mult: float = 1.0,
    fee_rate: float = 0.0,
) -> pd.Series:
    """
    Label each bar based on a Volatility-Adjusted Triple Barrier Method (TBM):
      - Treat every 1-minute candle index i as a fresh entry opportunity.
      - Look forward into a 15-minute time horizon (i+1 to i+15) to evaluate a race condition.
      - Barriers are Close[i] + (tp_mult * ATR[i]) and Close[i] - (sl_mult * ATR[i]).
      - 14-period ATR calculated via Wilder's smoothing if not present in df as 'atr'.
      - Label mapping:
        - If forward high breaches Upper Barrier first -> 2 (BUY)
        - If forward low breaches Lower Barrier first -> 0 (SELL)
        - If window expires, or if a single candle wicks through BOTH barriers simultaneously -> 1 (HOLD)
        - Tail-end / warmup rows -> -1
    """
    n_rows = len(df)
    if n_rows == 0:
        return pd.Series([], dtype=int)

    # 1. Compute ATR if 'atr' column is not already present
    if "atr" in df.columns:
        atr = df["atr"].values
    else:
        # Require 'high', 'low', 'close' columns
        required_cols = ["high", "low", "close"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column '{col}' for ATR calculation.")

        high = df["high"].values
        low = df["low"].values
        close = df["close"].values

        tr = np.zeros(n_rows)
        tr[0] = high[0] - low[0]

        prev_close = close[:-1]
        tr[1:] = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - prev_close),
                np.abs(low[1:] - prev_close)
            )
        )

        period = 14
        if n_rows < period:
            labels = np.full(n_rows, -1, dtype=int)
            return pd.Series(labels, index=df.index)

        tr_series = pd.Series(tr)
        init_series = pd.Series(np.nan, index=df.index)
        sma = tr_series.iloc[:period].mean()
        init_series.iloc[period - 1] = sma
        init_series.iloc[period:] = tr_series.iloc[period:]

        # Wilder's smoothing (equivalent to EMA with alpha = 1 / period)
        atr = init_series.ewm(alpha=1.0 / period, adjust=False).mean().values

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    # 2. Build 2D arrays for forward high and low prices of shape (n_rows, window)
    high_2d = np.zeros((n_rows, window))
    low_2d = np.zeros((n_rows, window))

    for k in range(1, window + 1):
        high_2d[:-k, k - 1] = high[k:]
        high_2d[-k:, k - 1] = np.nan

        low_2d[:-k, k - 1] = low[k:]
        low_2d[-k:, k - 1] = np.nan

    # 3. Compute dynamic upper and lower barriers (including commission fees)
    upper_barrier = close + (tp_mult * atr) + (fee_rate * close)
    lower_barrier = close - (sl_mult * atr) - (fee_rate * close)

    ub_2d = upper_barrier[:, np.newaxis]
    lb_2d = lower_barrier[:, np.newaxis]

    # 4. Check breaches
    high_breach = (high_2d >= ub_2d)
    low_breach = (low_2d <= lb_2d)

    any_high_breach = high_breach.any(axis=1)
    any_low_breach = low_breach.any(axis=1)

    # First index of breach
    high_first_idx = np.argmax(high_breach, axis=1)
    low_first_idx = np.argmax(low_breach, axis=1)

    # Default is HOLD (1)
    labels = np.full(n_rows, 1, dtype=int)

    # BUY (2):
    # - Upper barrier is breached and lower is not
    # - OR both are breached, but upper is breached strictly before lower
    buy_mask = (any_high_breach & ~any_low_breach) | (
        any_high_breach & any_low_breach & (high_first_idx < low_first_idx)
    )

    # SELL (0):
    # - Lower barrier is breached and upper is not
    # - OR both are breached, but lower is breached strictly before upper
    sell_mask = (~any_high_breach & any_low_breach) | (
        any_high_breach & any_low_breach & (low_first_idx < high_first_idx)
    )

    labels[buy_mask] = 2
    labels[sell_mask] = 0

    # In case of warmup rows where atr is NaN: flag them as -1
    nan_atr_mask = np.isnan(atr)
    labels[nan_atr_mask] = -1

    # Tail end edge cases: the last 'window' rows cannot look forward 'window' periods.
    if n_rows >= window:
        labels[-window:] = -1
    else:
        labels[:] = -1

    return pd.Series(labels, index=df.index, dtype=int)