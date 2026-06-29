"""
utils/normalizer.py
───────────────────
Rolling normalization utilities for online feature scaling.

RollingZScore: normalizes each feature to zero mean, unit variance
               using a sliding window — suitable for live inference.
MinMaxScaler:  clips to [0, 1] based on observed min/max.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

import numpy as np


class RollingZScore:
    """
    Online rolling z-score normalizer.

    Maintains a deque of recent values per feature name and normalizes
    new observations using that window's mean and std.
    """

    def __init__(self, window: int = 50, feature_names: Optional[List[str]] = None) -> None:
        self.window = window
        self.feature_names = feature_names or []
        self._buffers: Dict[str, deque] = {
            name: deque(maxlen=window) for name in self.feature_names
        }

    def fit_partial(self, feature_name: str, value: float) -> None:
        """Add a new observation to the buffer (does not normalize)."""
        if feature_name not in self._buffers:
            self._buffers[feature_name] = deque(maxlen=self.window)
        self._buffers[feature_name].append(value)

    def transform_one(self, feature_name: str, value: float) -> float:
        """Normalize a single feature value using its rolling buffer."""
        buf = self._buffers.get(feature_name)
        if buf is None or len(buf) < 2:
            return 0.0
        arr = np.asarray(buf, dtype=float)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1))
        if std == 0:
            return 0.0
        return float((value - mean) / std)

    def fit_transform(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Update buffers with new observations and return normalized values.
        Mutates internal state.
        """
        normalized = {}
        for name, value in features.items():
            self.fit_partial(name, value)
            normalized[name] = self.transform_one(name, value)
        return normalized

    def transform_vector(
        self, feature_names: List[str], values: List[float]
    ) -> List[float]:
        """Normalize a list of values matching feature_names order."""
        return [self.transform_one(n, v) for n, v in zip(feature_names, values)]

    @property
    def is_warm(self) -> bool:
        """True when all buffers have at least half the window filled."""
        if not self._buffers:
            return False
        return all(len(buf) >= self.window // 2 for buf in self._buffers.values())


class ClipScaler:
    """
    Clips and scales values to [0, 1] using observed or provided min/max bounds.
    Safe: values outside bounds are clipped, not rejected.
    """

    def __init__(self, bounds: Optional[Dict[str, tuple]] = None) -> None:
        # bounds = {"feature_name": (min_val, max_val)}
        self.bounds: Dict[str, tuple] = bounds or {}

    def set_bounds(self, feature_name: str, min_val: float, max_val: float) -> None:
        self.bounds[feature_name] = (min_val, max_val)

    def transform_one(self, feature_name: str, value: float) -> float:
        if feature_name not in self.bounds:
            return value
        lo, hi = self.bounds[feature_name]
        if hi == lo:
            return 0.0
        return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))

    def transform(self, features: Dict[str, float]) -> Dict[str, float]:
        return {name: self.transform_one(name, v) for name, v in features.items()}
