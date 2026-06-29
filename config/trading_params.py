"""
config/trading_params.py
────────────────────────
Strategy-level constants and feature-engineering parameters.
These are intentionally separate from env-based settings so they can be
version-controlled independently from secrets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class IndicatorParams:
    # EMA periods
    ema_fast: int = 9
    ema_slow: int = 21
    ema_trend: int = 50

    # SMA periods
    sma_short: int = 20
    sma_long: int = 50

    # RSI
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0

    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # ATR
    atr_period: int = 14

    # VWAP
    vwap_anchored: bool = True   # reset at session open


@dataclass(frozen=True)
class FeatureParams:
    # Lag steps to include as features
    lag_steps: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10])

    # Rolling window sizes for statistics
    rolling_windows: List[int] = field(default_factory=lambda: [5, 10, 20])

    # Regime classification thresholds
    adx_trend_threshold: float = 25.0
    volatility_regime_pct: float = 0.015   # 1.5 % of price = volatile

    # Feature names (must match model training)
    feature_names: List[str] = field(default_factory=lambda: [
        "ema_fast", "ema_slow", "ema_trend",
        "sma_short", "sma_long",
        "rsi", "macd", "macd_signal", "macd_hist",
        "atr", "vwap",
        "volume_spike",    # z-score of volume
        "momentum",        # close / close_n - 1
        "regime_encoded",  # 0 = sideways, 1 = trend, 2 = volatile
    ])


@dataclass(frozen=True)
class RiskParams:
    # Kelly fraction (conservative: half-Kelly)
    kelly_fraction: float = 0.5

    # Minimum order size in quote currency (USDT)
    min_order_value: float = 10.0

    # Maximum consecutive losses before halting
    max_consecutive_losses: int = 5

    # Cooldown seconds after circuit breaker fires
    circuit_breaker_cooldown: int = 300


INDICATOR_PARAMS = IndicatorParams()
FEATURE_PARAMS = FeatureParams()
RISK_PARAMS = RiskParams()
