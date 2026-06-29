"""
models/xgboost_model.py
────────────────────────
XGBoost model wrapper.

Expected artefact: a pickle file produced by `scripts/train_model.py`
containing a dict:
    {
        "model": xgb.XGBClassifier,
        "version": "1.0.0",
        "label_map": {0: "BUY", 1: "HOLD", 2: "SELL"}   # or similar
    }
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List

import numpy as np

from models.base_model import BaseModel, Prediction


class XGBoostModel(BaseModel):
    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self._model = None
        self._label_map: dict = {0: "BUY", 1: "HOLD", 2: "SELL"}

    def load(self) -> None:
        with open(self.path, "rb") as f:
            artefact = pickle.load(f)

        if isinstance(artefact, dict):
            self._model = artefact["model"]
            self.version = artefact.get("version", "1.0.0")
            self._label_map = artefact.get(
                "label_map", {0: "BUY", 1: "HOLD", 2: "SELL"}
            )
            self.feature_names = artefact.get("feature_names", [])
        else:
            # Bare model object
            self._model = artefact
            self.version = "1.0.0"
            self.feature_names = []

    def predict(self, features: List[float]) -> Prediction:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        x = np.asarray(features, dtype=float).reshape(1, -1)
        proba = self._model.predict_proba(x)[0]  # shape (n_classes,)

        pred_idx = int(np.argmax(proba))
        action = self._label_map.get(pred_idx, "HOLD")
        confidence = float(proba[pred_idx])

        prob_dict = {
            self._label_map.get(i, str(i)): float(p)
            for i, p in enumerate(proba)
        }

        # Naïve expected return: (P_BUY - P_SELL) as a signal
        p_buy = prob_dict.get("BUY", 0.0)
        p_sell = prob_dict.get("SELL", 0.0)
        expected_return = p_buy - p_sell

        return Prediction(
            action=action,
            confidence=confidence,
            expected_return=expected_return,
            probabilities=prob_dict,
            model_version=self.version,
        )
