"""
models/torch_model.py
─────────────────────
PyTorch model wrapper for inference-only use.

Expects a TorchScript model (.pt) produced by `torch.jit.script/trace`.
The model must output a softmax probability tensor of shape (1, n_classes).
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from models.base_model import BaseModel, Prediction


class TorchModel(BaseModel):
    def __init__(self, path: Path, label_map: dict | None = None) -> None:
        super().__init__(path)
        self._model = None
        self._label_map = label_map or {0: "BUY", 1: "HOLD", 2: "SELL"}
        self.version = "torch-1.0"

    def load(self) -> None:
        import torch

        self._model = torch.jit.load(str(self.path), map_location="cpu")
        self._model.eval()
        self.version = "torch-1.0"

    def predict(self, features: List[float]) -> Prediction:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        import torch

        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            out = self._model(x)
            proba = out.squeeze(0).numpy()

        pred_idx = int(np.argmax(proba))
        action = self._label_map.get(pred_idx, "HOLD")
        confidence = float(proba[pred_idx])

        prob_dict = {
            self._label_map.get(i, str(i)): float(p)
            for i, p in enumerate(proba)
        }

        return Prediction(
            action=action,
            confidence=confidence,
            expected_return=prob_dict.get("BUY", 0.0) - prob_dict.get("SELL", 0.0),
            probabilities=prob_dict,
            model_version=self.version,
        )
