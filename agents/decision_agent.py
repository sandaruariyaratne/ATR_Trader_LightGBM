"""
agents/decision_agent.py
─────────────────────────
Consumes FeatureVectorEvent, runs model inference, and emits TradeSignalEvent.

Supports hot-reloading: if the model file changes on disk, the agent reloads
it automatically before the next inference (checked every 60 candles).
"""
from __future__ import annotations

import asyncio
import os
from typing import Optional

from config.settings import Settings
from core.event_bus import EventBus, FeatureVectorEvent, TradeSignalEvent
from core.logger import get_logger
from core.state_manager import StateManager
from models import BaseModel, load_model

logger = get_logger("decision_agent")

_HOT_RELOAD_INTERVAL = 60  # candles between mtime checks


class DecisionAgent:
    def __init__(
        self,
        bus: EventBus,
        state: StateManager,
        settings: Settings,
    ) -> None:
        self.bus = bus
        self.state = state
        self.settings = settings
        self._model: Optional[BaseModel] = None
        self._model_mtime: float = 0.0
        self._candle_count: int = 0

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        logger.info("decision_agent.starting")
        await self._load_model()

        while True:
            try:
                event: FeatureVectorEvent = await self.bus.consume("feature_vector")
                signal = await self._infer(event)
                if signal is not None:
                    await self.bus.publish("trade_signal", signal)
            except asyncio.CancelledError:
                logger.info("decision_agent.cancelled")
                raise
            except Exception as exc:
                logger.error("decision_agent.error", error=str(exc))

    # ── Model management ──────────────────────────────────────────────────────

    async def _load_model(self) -> None:
        try:
            self._model = load_model(self.settings)
            self._model_mtime = os.path.getmtime(self.settings.model_path)
            logger.info(
                "decision_agent.model_loaded",
                type=self.settings.model_type,
                path=str(self.settings.model_path),
                version=self._model.version,
            )
        except Exception as exc:
            logger.error("decision_agent.model_load_failed", error=str(exc))
            raise

    async def _maybe_hot_reload(self) -> None:
        """Reload model if file has been updated on disk."""
        try:
            mtime = os.path.getmtime(self.settings.model_path)
            if mtime != self._model_mtime:
                logger.info("decision_agent.hot_reload_triggered")
                await self._load_model()
        except OSError:
            pass  # File temporarily unavailable during write

    # ── Inference ─────────────────────────────────────────────────────────────

    async def _infer(self, event: FeatureVectorEvent) -> Optional[TradeSignalEvent]:
        self._candle_count += 1
        if self._candle_count % _HOT_RELOAD_INTERVAL == 0:
            await self._maybe_hot_reload()

        if self._model is None:
            logger.warning("decision_agent.no_model")
            return None

        try:
            feat_dict = dict(zip(event.feature_names, event.features))
            if self._model.feature_names:
                ordered_features = [feat_dict[name] for name in self._model.feature_names]
            else:
                ordered_features = event.features
            prediction = self._model.predict(ordered_features)
        except Exception as exc:
            logger.error("decision_agent.inference_failed", error=str(exc))
            return None

        logger.info(
            "decision_agent.prediction",
            action=prediction.action,
            confidence=round(prediction.confidence, 3),
            expected_return=round(prediction.expected_return, 4),
            probabilities={k: round(v, 4) for k, v in prediction.probabilities.items()},
            regime=event.regime,
        )

        return TradeSignalEvent(
            symbol=event.symbol,
            timestamp=event.timestamp,
            action=prediction.action,
            confidence=prediction.confidence,
            expected_return=prediction.expected_return,
            model_version=prediction.model_version,
            features_snapshot=dict(
                zip(event.feature_names, event.features)
            ),
        )
