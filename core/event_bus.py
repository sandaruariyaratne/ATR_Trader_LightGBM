"""
core/event_bus.py
─────────────────
Typed, asyncio-native event bus.

Events flow through typed asyncio.Queue instances so each consumer
can process at its own pace. The bus is not a broadcast bus — each
topic has exactly one consumer queue (point-to-point pipeline).

Usage
─────
    bus = EventBus()
    await bus.publish("market_state", event)
    event = await bus.consume("market_state")
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, Optional, TypeVar

from core.logger import get_logger

T = TypeVar("T")
logger = get_logger("event_bus")

# ─── Event types ─────────────────────────────────────────────────────────────


@dataclass
class MarketStateEvent:
    """Emitted by MarketDataAgent after each candle close."""
    symbol: str
    timestamp: int                  # unix ms of candle open
    open: float
    high: float
    low: float
    close: float
    volume: float
    indicators: Dict[str, float]    # computed indicator values
    raw_candles: list               # last N candles for feature lag


@dataclass
class FeatureVectorEvent:
    """Emitted by FeatureEngineeringAgent — model-ready feature array."""
    symbol: str
    timestamp: int
    features: list[float]           # ordered feature vector
    feature_names: list[str]
    regime: str                     # "trend" | "sideways" | "volatile"


@dataclass
class TradeSignalEvent:
    """Emitted by DecisionAgent after model inference."""
    symbol: str
    timestamp: int
    action: str                     # "BUY" | "SELL" | "HOLD"
    confidence: float               # [0, 1]
    expected_return: float          # predicted % return
    model_version: str
    features_snapshot: Dict[str, float]


@dataclass
class ApprovedOrderEvent:
    """Emitted by RiskManagementAgent — validated, sized order."""
    symbol: str
    timestamp: int
    side: str                       # "buy" | "sell"
    order_type: str                 # "market" | "limit"
    quantity: float
    entry_price: float
    stop_loss: float
    take_profit: float
    rationale: str


@dataclass
class OrderResultEvent:
    """Emitted by ExecutionAgent after order fill/rejection."""
    symbol: str
    timestamp: int
    order_id: str
    side: str
    filled_qty: float
    avg_price: float
    status: str                     # "filled" | "rejected" | "cancelled"
    pnl: float                      # realised PnL (0 for entries)
    error: Optional[str] = None


# ─── Bus ─────────────────────────────────────────────────────────────────────

TOPICS = (
    "market_state",
    "feature_vector",
    "trade_signal",
    "approved_order",
    "order_result",
)


class EventBus:
    """Single-consumer async pipeline bus."""

    def __init__(self, maxsize: int = 500) -> None:
        self._queues: Dict[str, asyncio.Queue] = {
            topic: asyncio.Queue(maxsize=maxsize) for topic in TOPICS
        }
        logger.debug("event_bus.init", topics=TOPICS, maxsize=maxsize)

    async def publish(self, topic: str, event: Any) -> None:
        if topic not in self._queues:
            raise ValueError(f"Unknown topic: {topic!r}")
        try:
            self._queues[topic].put_nowait(event)
        except asyncio.QueueFull:
            logger.warning("event_bus.queue_full", topic=topic, dropping=str(event)[:80])

    async def consume(self, topic: str, timeout: Optional[float] = None) -> Any:
        if topic not in self._queues:
            raise ValueError(f"Unknown topic: {topic!r}")
        q = self._queues[topic]
        if timeout is not None:
            return await asyncio.wait_for(q.get(), timeout=timeout)
        return await q.get()

    def task_done(self, topic: str) -> None:
        self._queues[topic].task_done()

    def qsize(self, topic: str) -> int:
        return self._queues[topic].qsize()

    def stats(self) -> Dict[str, int]:
        return {t: self._queues[t].qsize() for t in TOPICS}
