"""
tests/test_feature_engineering.py
──────────────────────────────────
Tests for the FeatureEngineeringAgent and normalizer utilities.
"""
from __future__ import annotations

import asyncio
import pytest
import numpy as np

from utils.normalizer import RollingZScore, ClipScaler
from core.event_bus import MarketStateEvent


# ─── RollingZScore ───────────────────────────────────────────────────────────

class TestRollingZScore:
    def test_single_feature_normalizes_to_zero_mean(self):
        scaler = RollingZScore(window=20, feature_names=["price"])
        values = [float(i) for i in range(20)]
        for v in values:
            scaler.fit_partial("price", v)
        # After filling window, the last value's z-score should be positive
        z = scaler.transform_one("price", values[-1])
        assert z > 0.0

    def test_constant_feature_returns_zero(self):
        scaler = RollingZScore(window=10, feature_names=["x"])
        for _ in range(10):
            scaler.fit_partial("x", 5.0)
        assert scaler.transform_one("x", 5.0) == pytest.approx(0.0)

    def test_fit_transform_returns_all_keys(self):
        features = {"ema": 100.0, "rsi": 55.0, "atr": 1.2}
        scaler = RollingZScore(window=10, feature_names=list(features.keys()))
        for _ in range(5):
            result = scaler.fit_transform(features)
        assert set(result.keys()) == set(features.keys())

    def test_is_warm_after_half_window(self):
        scaler = RollingZScore(window=20, feature_names=["f"])
        assert not scaler.is_warm
        for i in range(10):
            scaler.fit_partial("f", float(i))
        assert scaler.is_warm

    def test_transform_vector_length(self):
        names = ["a", "b", "c"]
        scaler = RollingZScore(window=10, feature_names=names)
        for _ in range(10):
            for n in names:
                scaler.fit_partial(n, np.random.rand())
        result = scaler.transform_vector(names, [1.0, 2.0, 3.0])
        assert len(result) == 3

    def test_unknown_feature_returns_zero(self):
        scaler = RollingZScore(window=10)
        assert scaler.transform_one("nonexistent", 42.0) == pytest.approx(0.0)


# ─── ClipScaler ──────────────────────────────────────────────────────────────

class TestClipScaler:
    def test_within_bounds(self):
        scaler = ClipScaler(bounds={"rsi": (0.0, 100.0)})
        assert scaler.transform_one("rsi", 50.0) == pytest.approx(0.5)

    def test_clips_above_max(self):
        scaler = ClipScaler(bounds={"x": (0.0, 10.0)})
        assert scaler.transform_one("x", 999.0) == pytest.approx(1.0)

    def test_clips_below_min(self):
        scaler = ClipScaler(bounds={"x": (0.0, 10.0)})
        assert scaler.transform_one("x", -999.0) == pytest.approx(0.0)

    def test_equal_bounds_returns_zero(self):
        scaler = ClipScaler(bounds={"flat": (5.0, 5.0)})
        assert scaler.transform_one("flat", 5.0) == pytest.approx(0.0)

    def test_transform_dict(self):
        scaler = ClipScaler(bounds={"a": (0.0, 1.0), "b": (10.0, 20.0)})
        result = scaler.transform({"a": 0.5, "b": 15.0})
        assert result["a"] == pytest.approx(0.5)
        assert result["b"] == pytest.approx(0.5)

    def test_missing_bounds_passthrough(self):
        scaler = ClipScaler()
        assert scaler.transform_one("unknown", 42.0) == pytest.approx(42.0)


# ─── MarketStateEvent helpers ─────────────────────────────────────────────────

def _make_market_event(close: float = 100.0, n_candles: int = 55) -> MarketStateEvent:
    """Build a synthetic MarketStateEvent for testing."""
    np.random.seed(0)
    closes = (close + np.cumsum(np.random.randn(n_candles) * 0.3)).tolist()
    highs = [c + abs(np.random.randn() * 0.2) for c in closes]
    lows  = [c - abs(np.random.randn() * 0.2) for c in closes]
    vols  = [abs(np.random.randn() * 500 + 2000) for _ in closes]

    from utils.indicators import ema, sma, rsi, macd, atr, vwap, volume_z_score, momentum
    macd_val, macd_sig, macd_hist = macd(closes)

    indicators = {
        "ema_fast":     ema(closes, 9),
        "ema_slow":     ema(closes, 21),
        "ema_trend":    ema(closes, 50),
        "sma_short":    sma(closes, 20),
        "sma_long":     sma(closes, 50),
        "rsi":          rsi(closes, 14),
        "macd":         macd_val,
        "macd_signal":  macd_sig,
        "macd_hist":    macd_hist,
        "atr":          atr(highs, lows, closes),
        "vwap":         vwap(highs, lows, closes, vols),
        "volume_spike": volume_z_score(vols),
        "momentum":     momentum(closes),
    }
    candles = [
        [i * 60000, highs[i] - 0.1, highs[i], lows[i], closes[i], vols[i]]
        for i in range(n_candles)
    ]
    return MarketStateEvent(
        symbol="BTC/USDT",
        timestamp=n_candles * 60000,
        open=closes[-2],
        high=highs[-1],
        low=lows[-1],
        close=closes[-1],
        volume=vols[-1],
        indicators=indicators,
        raw_candles=candles,
    )


# ─── FeatureEngineeringAgent ─────────────────────────────────────────────────

@pytest.mark.asyncio
class TestFeatureEngineeringAgent:
    def _build_agent(self):
        from unittest.mock import AsyncMock, MagicMock
        from agents.feature_engineering_agent import FeatureEngineeringAgent
        from config.settings import Settings

        bus = MagicMock()
        bus.consume = AsyncMock()
        bus.publish = AsyncMock()
        state = MagicMock()
        settings = Settings(feature_window=20)
        return FeatureEngineeringAgent(bus, state, settings)

    async def test_returns_none_while_warming_up(self):
        agent = self._build_agent()
        event = _make_market_event(n_candles=55)
        # First call — history is empty → None
        result = await agent._process(event)
        # After first event, history has 1 item; need max(lag_steps)+1 = 11
        assert result is None

    async def test_emits_after_warmup(self):
        agent = self._build_agent()
        event = _make_market_event(n_candles=55)
        result = None
        for _ in range(15):   # feed enough events to warm up
            result = await agent._process(event)
        assert result is not None

    async def test_feature_names_match_features_length(self):
        agent = self._build_agent()
        event = _make_market_event(n_candles=55)
        result = None
        for _ in range(15):
            result = await agent._process(event)
        assert result is not None
        assert len(result.features) == len(result.feature_names)

    async def test_regime_is_valid_string(self):
        agent = self._build_agent()
        event = _make_market_event(n_candles=55)
        result = None
        for _ in range(15):
            result = await agent._process(event)
        assert result is not None
        assert result.regime in ("trend", "sideways", "volatile")

    async def test_lagged_features_included(self):
        agent = self._build_agent()
        event = _make_market_event(n_candles=55)
        result = None
        for _ in range(15):
            result = await agent._process(event)
        assert result is not None
        lag_names = [n for n in result.feature_names if "lag" in n]
        assert len(lag_names) > 0, "Expected lagged feature names"
