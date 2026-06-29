"""
utils/time_utils.py
───────────────────
Timestamp helpers and candle-alignment utilities.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional


def now_ms() -> int:
    """Current UTC time in milliseconds."""
    return int(time.time() * 1000)


def ms_to_dt(ms: int) -> datetime:
    """Convert unix milliseconds to UTC datetime."""
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


def dt_to_ms(dt: datetime) -> int:
    """Convert datetime to unix milliseconds."""
    return int(dt.timestamp() * 1000)


def next_candle_close_ms(interval_ms: int = 60_000) -> int:
    """
    Return the unix-ms timestamp of the next 1-minute candle close.
    Useful for sleeping until the next event.
    """
    now = now_ms()
    return ((now // interval_ms) + 1) * interval_ms


def sleep_until_next_candle(interval_ms: int = 60_000) -> float:
    """Return seconds to sleep until the next candle boundary."""
    target = next_candle_close_ms(interval_ms)
    return max(0.0, (target - now_ms()) / 1000.0)


def candle_open_ms(ts_ms: int, interval_ms: int = 60_000) -> int:
    """Snap a timestamp down to the open of its containing candle."""
    return (ts_ms // interval_ms) * interval_ms


def interval_to_ms(interval: str) -> int:
    """Convert a CCXT-style interval string to milliseconds."""
    unit = interval[-1]
    value = int(interval[:-1])
    mapping = {"s": 1_000, "m": 60_000, "h": 3_600_000, "d": 86_400_000}
    return value * mapping.get(unit, 60_000)


def format_ts(ms: int) -> str:
    """Human-readable UTC timestamp string."""
    return ms_to_dt(ms).strftime("%Y-%m-%d %H:%M:%S UTC")
