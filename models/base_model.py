"""
models/base_model.py
────────────────────
Abstract interface that all model wrappers must implement.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Prediction:
    action: str          # "BUY" | "SELL" | "HOLD"
    confidence: float    # probability of the predicted class
    expected_return: float
    probabilities: dict  # {"BUY": p, "SELL": p, "HOLD": p}
    model_version: str


class BaseModel(ABC):
    """All model backends must implement this interface."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.version: str = "unknown"
        self.feature_names: List[str] = []

    @abstractmethod
    def load(self) -> None:
        """Load model artefact from self.path."""

    @abstractmethod
    def predict(self, features: List[float]) -> Prediction:
        """
        Run inference on a single feature vector.
        Returns a Prediction dataclass.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}(path={self.path}, version={self.version})"
