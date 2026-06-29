"""
tests/test_indicators.py
────────────────────────
Unit tests for utils/indicators.py.
All tests use deterministic synthetic data; no network calls.
"""
from __future__ import annotations

import pytest
import numpy as np

from utils.indicators import (
    ema, ema_series, sma, sma_series,
    rsi, macd, atr, vwap,
    volume_z_score, momentum, adx,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def linear_up():
    """Linearly increasing price series."""
    return [float(i) for i in range(1, 101)]


@pytest.fixture
def constant():
    return [50.0] * 50


@pytest.fixture
def ohlcv():
    """Synthetic OHLCV data: 50 bars with slight uptrend."""
    np.random.seed(42)
    closes = 100.0 + np.cumsum(np.random.randn(50) * 0.5)
    highs  = closes + np.abs(np.random.randn(50) * 0.3)
    lows   = closes - np.abs(np.random.randn(50) * 0.3)
    opens  = closes - np.random.randn(50) * 0.2
    vols   = np.abs(np.random.randn(50) * 1000 + 5000)
    return {
        "opens":  opens.tolist(),
        "highs":  highs.tolist(),
        "lows":   lows.tolist(),
        "closes": closes.tolist(),
        "vols":   vols.tolist(),
    }


# ─── EMA ─────────────────────────────────────────────────────────────────────

class TestEMA:
    def test_constant_series_equals_constant(self, constant):
        result = ema(constant, period=10)
        assert abs(result - 50.0) < 1e-9

    def test_ema_single_value(self):
        assert ema([42.0], period=5) == pytest.approx(42.0)

    def test_ema_returns_float(self, linear_up):
        assert isinstance(ema(linear_up, period=9), float)

    def test_ema_tracks_uptrend(self, linear_up):
        fast = ema(linear_up, period=9)
        slow = ema(linear_up, period=21)
        assert fast > slow, "Fast EMA should be above slow EMA in uptrend"

    def test_ema_series_length(self, linear_up):
        series = ema_series(linear_up, period=9)
        assert len(series) == len(linear_up)

    def test_ema_series_last_matches_scalar(self, ohlcv):
        closes = ohlcv["closes"]
        assert ema_series(closes, 9)[-1] == pytest.approx(ema(closes, 9), rel=1e-6)


# ─── SMA ─────────────────────────────────────────────────────────────────────

class TestSMA:
    def test_sma_constant(self, constant):
        assert sma(constant, 10) == pytest.approx(50.0)

    def test_sma_last_n_elements(self):
        data = [1, 2, 3, 4, 5]
        assert sma(data, 3) == pytest.approx(4.0)   # mean of [3, 4, 5]

    def test_sma_series_nan_prefix(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        series = sma_series(data, 3)
        assert np.isnan(series[0])
        assert np.isnan(series[1])
        assert series[2] == pytest.approx(2.0)


# ─── RSI ─────────────────────────────────────────────────────────────────────

class TestRSI:
    def test_rsi_range(self, ohlcv):
        value = rsi(ohlcv["closes"], 14)
        assert 0.0 <= value <= 100.0

    def test_rsi_constant_returns_50(self, constant):
        # No gains or losses → undefined; implementation returns 100 or 50
        value = rsi(constant, 14)
        assert value in (50.0, 100.0)

    def test_rsi_all_gains_near_100(self):
        prices = [float(i) for i in range(1, 30)]
        value = rsi(prices, 14)
        assert value > 90.0

    def test_rsi_all_losses_near_0(self):
        prices = [float(30 - i) for i in range(30)]
        value = rsi(prices, 14)
        assert value < 10.0

    def test_rsi_short_series(self):
        # Should not raise; returns 50 on insufficient data
        value = rsi([1.0, 2.0, 1.5], 14)
        assert value == 50.0


# ─── MACD ────────────────────────────────────────────────────────────────────

class TestMACD:
    def test_macd_returns_three_floats(self, ohlcv):
        line, signal, hist = macd(ohlcv["closes"])
        assert isinstance(line, float)
        assert isinstance(signal, float)
        assert isinstance(hist, float)

    def test_macd_hist_equals_line_minus_signal(self, ohlcv):
        line, signal, hist = macd(ohlcv["closes"])
        assert hist == pytest.approx(line - signal, rel=1e-6)

    def test_macd_constant_near_zero(self, constant):
        line, signal, hist = macd(constant)
        assert abs(line) < 1e-6
        assert abs(hist) < 1e-6


# ─── ATR ─────────────────────────────────────────────────────────────────────

class TestATR:
    def test_atr_positive(self, ohlcv):
        value = atr(ohlcv["highs"], ohlcv["lows"], ohlcv["closes"])
        assert value > 0.0

    def test_atr_constant_ohc_is_zero(self):
        h = [10.0] * 20
        l = [10.0] * 20
        c = [10.0] * 20
        value = atr(h, l, c, period=14)
        assert value == pytest.approx(0.0)

    def test_atr_single_bar(self):
        value = atr([10.0], [8.0], [9.0])
        assert value == pytest.approx(2.0)


# ─── VWAP ────────────────────────────────────────────────────────────────────

class TestVWAP:
    def test_vwap_equal_volumes(self):
        h = [11.0, 12.0]
        l = [9.0, 10.0]
        c = [10.0, 11.0]
        v = [100.0, 100.0]
        # typical prices: 10, 11 — equal weight → 10.5
        result = vwap(h, l, c, v)
        assert result == pytest.approx(10.5)

    def test_vwap_zero_volume_fallback(self):
        h = [10.0]
        l = [10.0]
        c = [10.0]
        v = [0.0]
        result = vwap(h, l, c, v)
        assert result == pytest.approx(10.0)

    def test_vwap_positive(self, ohlcv):
        result = vwap(ohlcv["highs"], ohlcv["lows"], ohlcv["closes"], ohlcv["vols"])
        assert result > 0.0


# ─── Volume Z-Score ───────────────────────────────────────────────────────────

class TestVolumeZScore:
    def test_constant_volume_is_zero(self):
        vols = [1000.0] * 20
        assert volume_z_score(vols) == pytest.approx(0.0)

    def test_spike_is_positive(self):
        vols = [1000.0] * 19 + [5000.0]
        assert volume_z_score(vols) > 2.0

    def test_short_series(self):
        assert volume_z_score([100.0]) == pytest.approx(0.0)


# ─── Momentum ────────────────────────────────────────────────────────────────

class TestMomentum:
    def test_flat_momentum_is_zero(self, constant):
        assert momentum(constant, 10) == pytest.approx(0.0)

    def test_doubling_price_is_one(self):
        closes = [50.0] * 10 + [100.0]
        assert momentum(closes, 10) == pytest.approx(1.0)

    def test_momentum_short_series(self):
        assert momentum([100.0], 10) == pytest.approx(0.0)


# ─── ADX ─────────────────────────────────────────────────────────────────────

class TestADX:
    def test_adx_range(self, ohlcv):
        value = adx(ohlcv["highs"], ohlcv["lows"], ohlcv["closes"])
        assert 0.0 <= value <= 100.0

    def test_adx_insufficient_data_returns_default(self):
        h = [10.0] * 5
        l = [9.0] * 5
        c = [9.5] * 5
        assert adx(h, l, c) == pytest.approx(20.0)
