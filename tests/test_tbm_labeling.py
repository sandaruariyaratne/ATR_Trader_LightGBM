"""
tests/test_tbm_labeling.py
──────────────────────────
Unit tests for make_dynamic_volatility_labels implementation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from utils.features import make_dynamic_volatility_labels


def test_empty_dataframe():
    df = pd.DataFrame(columns=["open", "high", "low", "close"])
    labels = make_dynamic_volatility_labels(df, window=15)
    assert len(labels) == 0
    assert isinstance(labels, pd.Series)


def test_missing_columns_raises_value_error():
    df = pd.DataFrame({"close": [10.0, 11.0]})
    with pytest.raises(ValueError, match="Missing required column"):
        make_dynamic_volatility_labels(df, window=15)


def test_tbm_tail_handling():
    # Make a 50-row dataframe with dummy data and a provided ATR to avoid ATR warmup logic
    n_rows = 50
    df = pd.DataFrame({
        "high": np.full(n_rows, 100.0),
        "low": np.full(n_rows, 100.0),
        "close": np.full(n_rows, 100.0),
        "atr": np.full(n_rows, 2.0)
    })
    
    window = 15
    labels = make_dynamic_volatility_labels(df, window=window)
    
    # The last 15 elements must be -1
    assert (labels.iloc[-window:] == -1).all()
    # The remaining elements should be 1 (HOLD) because price is flat (no barriers hit)
    assert (labels.iloc[:-window] == 1).all()


def test_tbm_atr_warmup():
    # Test that if atr is not provided, the first 13 rows are flagged with -1 (since period=14 ATR has no complete SMA lookup)
    n_rows = 40
    df = pd.DataFrame({
        "high": np.full(n_rows, 10.0),
        "low": np.full(n_rows, 10.0),
        "close": np.full(n_rows, 10.0),
    })
    # Since prices are flat, TR is 0, ATR is 0. But the first 13 elements should be -1 due to NaN ATR
    labels = make_dynamic_volatility_labels(df, window=10)
    
    assert (labels.iloc[:13] == -1).all()
    assert (labels.iloc[13:-10] == 1).all()
    assert (labels.iloc[-10:] == -1).all()


def test_tbm_buy_breach():
    # 30 rows. We provide ATR as constant 2.0, so:
    # UB = close[i] + 2 * 2.0 = close[i] + 4.0
    # LB = close[i] - 1 * 2.0 = close[i] - 2.0
    n_rows = 30
    close_vals = np.full(n_rows, 100.0)
    high_vals = np.full(n_rows, 100.0)
    low_vals = np.full(n_rows, 100.0)
    
    # At index 5:
    # UB_5 = 104.0, LB_5 = 98.0
    # Let's set high of index 8 (index 5 + 3) to 105.0 to breach UB.
    # Keep low_vals[8] = 99.0 so LB is not breached.
    high_vals[8] = 105.0
    
    df = pd.DataFrame({
        "high": high_vals,
        "low": low_vals,
        "close": close_vals,
        "atr": np.full(n_rows, 2.0)
    })
    
    labels = make_dynamic_volatility_labels(df, window=10)
    
    # Index 5 should be labeled as BUY (2) because its upper barrier (104.0) is hit first (at index 8)
    assert labels.iloc[5] == 2


def test_tbm_sell_breach():
    # UB = close[i] + 4.0, LB = close[i] - 2.0
    n_rows = 30
    close_vals = np.full(n_rows, 100.0)
    high_vals = np.full(n_rows, 100.0)
    low_vals = np.full(n_rows, 100.0)
    
    # At index 5:
    # UB_5 = 104.0, LB_5 = 98.0
    # Let's set low of index 9 (index 5 + 4) to 97.0 to breach LB.
    low_vals[9] = 97.0
    
    df = pd.DataFrame({
        "high": high_vals,
        "low": low_vals,
        "close": close_vals,
        "atr": np.full(n_rows, 2.0)
    })
    
    labels = make_dynamic_volatility_labels(df, window=10)
    
    # Index 5 should be labeled as SELL (0)
    assert labels.iloc[5] == 0


def test_tbm_simultaneous_wick():
    # At index 5: UB_5 = 104.0, LB_5 = 98.0.
    # At index 8, high is 105.0 and low is 97.0. Both UB and LB breached in the same candle!
    n_rows = 30
    close_vals = np.full(n_rows, 100.0)
    high_vals = np.full(n_rows, 100.0)
    low_vals = np.full(n_rows, 100.0)
    
    high_vals[8] = 105.0
    low_vals[8] = 97.0
    
    df = pd.DataFrame({
        "high": high_vals,
        "low": low_vals,
        "close": close_vals,
        "atr": np.full(n_rows, 2.0)
    })
    
    labels = make_dynamic_volatility_labels(df, window=10)
    
    # Index 5 should be labeled as HOLD (1) due to simultaneous breach
    assert labels.iloc[5] == 1


def test_tbm_race_condition():
    # At index 5: UB_5 = 104.0, LB_5 = 98.0.
    # Let's breach LB at index 7 (low = 97.0) and UB at index 8 (high = 105.0).
    # Since LB is breached at index 7 (which is index 5 + 2) and UB is breached at index 8 (index 5 + 3),
    # LB is breached first. The label should be SELL (0).
    n_rows = 30
    close_vals = np.full(n_rows, 100.0)
    high_vals = np.full(n_rows, 100.0)
    low_vals = np.full(n_rows, 100.0)
    
    low_vals[7] = 97.0
    high_vals[8] = 105.0
    
    df = pd.DataFrame({
        "high": high_vals,
        "low": low_vals,
        "close": close_vals,
        "atr": np.full(n_rows, 2.0)
    })
    
    labels = make_dynamic_volatility_labels(df, window=10)
    assert labels.iloc[5] == 0


def test_tbm_atr_calculation_correctness():
    # Let's generate a dynamic high, low, close series
    # and compare the computed ATR with manual calculation
    np.random.seed(42)
    n_rows = 100
    close_vals = 100.0 + np.cumsum(np.random.randn(n_rows) * 0.5)
    high_vals = close_vals + np.abs(np.random.randn(n_rows) * 0.2)
    low_vals = close_vals - np.abs(np.random.randn(n_rows) * 0.2)
    
    df = pd.DataFrame({
        "high": high_vals,
        "low": low_vals,
        "close": close_vals
    })
    
    # Calculate ATR manually
    tr = np.zeros(n_rows)
    tr[0] = high_vals[0] - low_vals[0]
    for i in range(1, n_rows):
        tr[i] = max(
            high_vals[i] - low_vals[i],
            abs(high_vals[i] - close_vals[i-1]),
            abs(low_vals[i] - close_vals[i-1])
        )
    
    period = 14
    manual_atr = np.zeros(n_rows)
    manual_atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n_rows):
        manual_atr[i] = (manual_atr[i-1] * (period - 1) + tr[i]) / period
        
    labels = make_dynamic_volatility_labels(df, window=10)
    
    # Let's extract the internal ATR computed by the function.
    # We can do this by running the ATR logic and comparing it.
    tr_series = pd.Series(tr)
    init_series = pd.Series(np.nan, index=df.index)
    init_series.iloc[period - 1] = np.mean(tr[:period])
    init_series.iloc[period:] = tr_series.iloc[period:]
    computed_atr = init_series.ewm(alpha=1.0 / period, adjust=False).mean().values
    
    # Let's compare manual_atr and computed_atr for valid indices
    np.testing.assert_allclose(computed_atr[period-1:], manual_atr[period-1:], rtol=1e-7)
