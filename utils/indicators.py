"""
utils/indicators.py
───────────────────
Vectorised technical indicator calculations using NumPy.
All functions operate on plain Python lists or 1-D numpy arrays and
return float values (latest bar) unless otherwise noted.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np


# ─── Moving averages ─────────────────────────────────────────────────────────


def ema(values: Sequence[float], period: int) -> float:
    """Exponential Moving Average — returns latest value."""
    arr = np.asarray(values, dtype=float)
    if len(arr) < period:
        return float(np.mean(arr))
    k = 2.0 / (period + 1)
    result = arr[0]
    for v in arr[1:]:
        result = v * k + result * (1 - k)
    return float(result)


def ema_series(values: Sequence[float], period: int) -> np.ndarray:
    """Full EMA series."""
    arr = np.asarray(values, dtype=float)
    k = 2.0 / (period + 1)
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = arr[i] * k + out[i - 1] * (1 - k)
    return out


def sma(values: Sequence[float], period: int) -> float:
    """Simple Moving Average — returns latest value."""
    arr = np.asarray(values, dtype=float)
    n = min(period, len(arr))
    return float(np.mean(arr[-n:]))


def sma_series(values: Sequence[float], period: int) -> np.ndarray:
    """Full SMA series (valid region only, NaN-padded at start)."""
    arr = np.asarray(values, dtype=float)
    out = np.full_like(arr, np.nan)
    for i in range(period - 1, len(arr)):
        out[i] = np.mean(arr[i - period + 1 : i + 1])
    return out


# ─── RSI ─────────────────────────────────────────────────────────────────────


def rsi(closes: Sequence[float], period: int = 14) -> float:
    """Wilder's RSI — returns latest value in [0, 100]."""
    arr = np.asarray(closes, dtype=float)
    if len(arr) < period + 1:
        return 50.0
    deltas = np.diff(arr)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - 100.0 / (1.0 + rs))


# ─── MACD ────────────────────────────────────────────────────────────────────


def macd(
    closes: Sequence[float],
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> tuple[float, float, float]:
    """
    Returns (macd_line, signal_line, histogram) as of the latest bar.
    """
    arr = np.asarray(closes, dtype=float)
    fast_ema = ema_series(arr, fast)
    slow_ema = ema_series(arr, slow)
    macd_line = fast_ema - slow_ema
    sig_line = ema_series(macd_line, signal_period)
    histogram = macd_line - sig_line
    return float(macd_line[-1]), float(sig_line[-1]), float(histogram[-1])


# ─── ATR ─────────────────────────────────────────────────────────────────────


def atr(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> float:
    """Average True Range — latest value."""
    h = np.asarray(highs, dtype=float)
    l = np.asarray(lows, dtype=float)
    c = np.asarray(closes, dtype=float)

    if len(c) < 2:
        return float(h[-1] - l[-1])

    prev_c = c[:-1]
    tr = np.maximum(
        h[1:] - l[1:],
        np.maximum(np.abs(h[1:] - prev_c), np.abs(l[1:] - prev_c)),
    )
    n = min(period, len(tr))
    # Wilder smoothing
    atr_val = float(np.mean(tr[-n:]))
    return atr_val


# ─── VWAP ────────────────────────────────────────────────────────────────────


def vwap(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    volumes: Sequence[float],
) -> float:
    """Session VWAP — cumulative from the first provided bar."""
    h = np.asarray(highs, dtype=float)
    l = np.asarray(lows, dtype=float)
    c = np.asarray(closes, dtype=float)
    v = np.asarray(volumes, dtype=float)

    typical = (h + l + c) / 3.0
    total_vol = np.sum(v)
    if total_vol == 0:
        return float(c[-1])
    return float(np.sum(typical * v) / total_vol)


# ─── Volume spike ─────────────────────────────────────────────────────────────


def volume_z_score(volumes: Sequence[float], window: int = 20) -> float:
    """Z-score of the latest volume relative to a rolling window."""
    arr = np.asarray(volumes, dtype=float)
    n = min(window, len(arr))
    if n < 2:
        return 0.0
    subset = arr[-n:]
    mean = np.mean(subset)
    std = np.std(subset, ddof=1)
    if std == 0:
        return 0.0
    return float((arr[-1] - mean) / std)


# ─── Momentum ────────────────────────────────────────────────────────────────


def momentum(closes: Sequence[float], period: int = 10) -> float:
    """Rate of change: (close / close_n_periods_ago) - 1."""
    arr = np.asarray(closes, dtype=float)
    if len(arr) <= period:
        return 0.0
    prev = arr[-period - 1]
    if prev == 0:
        return 0.0
    return float((arr[-1] / prev) - 1.0)


# ─── ADX (for regime classification) ─────────────────────────────────────────


def adx(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> float:
    """Average Directional Index — returns latest value."""
    h = np.asarray(highs, dtype=float)
    l = np.asarray(lows, dtype=float)
    c = np.asarray(closes, dtype=float)

    if len(c) < period * 2:
        return 20.0  # default: no trend

    up_move = h[1:] - h[:-1]
    down_move = l[:-1] - l[1:]

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr_vals = np.maximum(
        h[1:] - l[1:],
        np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])),
    )

    # Wilder smoothing
    def wilder_smooth(arr: np.ndarray, p: int) -> np.ndarray:
        out = np.empty_like(arr)
        out[0] = np.mean(arr[:p])
        for i in range(1, len(arr)):
            out[i] = out[i - 1] - out[i - 1] / p + arr[i]
        return out

    n = min(period, len(tr_vals) - 1)
    tr_smooth = wilder_smooth(tr_vals, n)
    pdm_smooth = wilder_smooth(plus_dm, n)
    mdm_smooth = wilder_smooth(minus_dm, n)

    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = np.where(tr_smooth > 0, 100 * pdm_smooth / tr_smooth, 0.0)
        mdi = np.where(tr_smooth > 0, 100 * mdm_smooth / tr_smooth, 0.0)
        dx = np.where(pdi + mdi > 0, 100 * np.abs(pdi - mdi) / (pdi + mdi), 0.0)

    adx_val = float(np.mean(dx[-n:]))
    return adx_val
