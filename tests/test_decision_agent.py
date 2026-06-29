"""
tests/test_decision_agent.py
────────────────────────────
Tests for DecisionAgent and model wrappers.
Uses a mock model to avoid requiring trained artefacts on disk.
"""
from __future__ import annotations

import asyncio
import pickle
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from core.event_bus import FeatureVectorEvent
from models.base_model import BaseModel, Prediction


# ─── Mock model ──────────────────────────────────────────────────────────────

class _ConstantModel(BaseModel):
    """Always predicts BUY with 0.9 confidence."""

    def __init__(self, action: str = "BUY", confidence: float = 0.9) -> None:
        super().__init__(Path("/dev/null"))
        self._action = action
        self._confidence = confidence
        self.version = "mock-1.0"

    def load(self) -> None:
        pass

    def predict(self, features):
        probs = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}
        probs[self._action] = self._confidence
        hold_share = 1.0 - self._confidence
        for k in probs:
            if k != self._action:
                probs[k] = hold_share / 2
        return Prediction(
            action=self._action,
            confidence=self._confidence,
            expected_return=0.02,
            probabilities=probs,
            model_version=self.version,
        )


def _make_feature_event(n_features: int = 14) -> FeatureVectorEvent:
    names = [f"f{i}" for i in range(n_features)]
    values = np.random.randn(n_features).tolist()
    return FeatureVectorEvent(
        symbol="BTC/USDT",
        timestamp=1_700_000_000_000,
        features=values,
        feature_names=names,
        regime="trend",
    )


def _build_agent(model: BaseModel | None = None):
    from agents.decision_agent import DecisionAgent
    from config.settings import Settings

    bus = MagicMock()
    bus.consume = AsyncMock()
    bus.publish = AsyncMock()
    state = MagicMock()
    settings = Settings(
        model_type="xgboost",
        model_path=Path("data/models/test.pkl"),
        confidence_threshold=0.60,
    )
    agent = DecisionAgent(bus, state, settings)
    agent._model = model or _ConstantModel()
    return agent


# ─── Tests ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestDecisionAgent:

    async def test_buy_signal_emitted(self):
        agent = _build_agent(_ConstantModel("BUY", 0.9))
        event = _make_feature_event()
        result = await agent._infer(event)
        assert result is not None
        assert result.action == "BUY"

    async def test_sell_signal_emitted(self):
        agent = _build_agent(_ConstantModel("SELL", 0.9))
        event = _make_feature_event()
        result = await agent._infer(event)
        assert result is not None
        assert result.action == "SELL"

    async def test_hold_signal_emitted(self):
        agent = _build_agent(_ConstantModel("HOLD", 0.9))
        event = _make_feature_event()
        result = await agent._infer(event)
        assert result is not None
        assert result.action == "HOLD"

    async def test_confidence_propagated(self):
        agent = _build_agent(_ConstantModel("BUY", 0.77))
        event = _make_feature_event()
        result = await agent._infer(event)
        assert result is not None
        assert result.confidence == pytest.approx(0.77)

    async def test_model_version_propagated(self):
        agent = _build_agent(_ConstantModel("BUY", 0.9))
        event = _make_feature_event()
        result = await agent._infer(event)
        assert result.model_version == "mock-1.0"

    async def test_no_model_returns_none(self):
        agent = _build_agent()
        agent._model = None
        result = await agent._infer(_make_feature_event())
        assert result is None

    async def test_inference_error_returns_none(self):
        class _FailModel(_ConstantModel):
            def predict(self, features):
                raise ValueError("intentional failure")

        agent = _build_agent(_FailModel())
        result = await agent._infer(_make_feature_event())
        assert result is None

    async def test_symbol_preserved(self):
        agent = _build_agent()
        event = _make_feature_event()
        result = await agent._infer(event)
        assert result.symbol == "BTC/USDT"

    async def test_timestamp_preserved(self):
        agent = _build_agent()
        event = _make_feature_event()
        result = await agent._infer(event)
        assert result.timestamp == 1_700_000_000_000

    async def test_features_snapshot_non_empty(self):
        agent = _build_agent()
        event = _make_feature_event(n_features=14)
        result = await agent._infer(event)
        assert len(result.features_snapshot) > 0


# ─── XGBoost wrapper (with dummy model) ─────────────────────────────────────

class TestXGBoostModel:
    def _make_pkl(self) -> Path:
        """Create a real scikit-learn compatible dummy classifier."""
        from sklearn.dummy import DummyClassifier

        clf = DummyClassifier(strategy="most_frequent")
        X = np.random.randn(30, 14)
        y = np.array([0, 1, 2] * 10)
        clf.fit(X, y)

        artefact = {
            "model": clf,
            "version": "0.1.0",
            "label_map": {0: "BUY", 1: "HOLD", 2: "SELL"},
        }
        tmp = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
        pickle.dump(artefact, tmp)
        tmp.close()
        return Path(tmp.name)

    def test_load_and_predict(self):
        from models.xgboost_model import XGBoostModel

        path = self._make_pkl()
        model = XGBoostModel(path)
        model.load()
        features = np.random.randn(14).tolist()
        pred = model.predict(features)
        assert pred.action in ("BUY", "SELL", "HOLD")
        assert 0.0 <= pred.confidence <= 1.0
        assert isinstance(pred.expected_return, float)

    def test_version_loaded(self):
        from models.xgboost_model import XGBoostModel

        path = self._make_pkl()
        model = XGBoostModel(path)
        model.load()
        assert model.version == "0.1.0"

    def test_probabilities_sum_to_one(self):
        from models.xgboost_model import XGBoostModel

        path = self._make_pkl()
        model = XGBoostModel(path)
        model.load()
        pred = model.predict(np.random.randn(14).tolist())
        total = sum(pred.probabilities.values())
        assert total == pytest.approx(1.0, abs=1e-4)
