"""
models/onnx_model.py
────────────────────
ONNX Runtime model wrapper for low-latency inference.
Supports any classification model exported to ONNX with softmax output.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from models.base_model import BaseModel, Prediction


class ONNXModel(BaseModel):
    def __init__(self, path: Path, label_map: dict | None = None) -> None:
        super().__init__(path)
        self._session = None
        self._input_name: str = "input"
        self._label_map = label_map or {0: "BUY", 1: "HOLD", 2: "SELL"}
        self.version = "onnx-1.0"

    def load(self) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            str(self.path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name

        meta = self._session.get_modelmeta()
        self.version = meta.version if meta.version else "onnx-1.0"

    def predict(self, features: List[float]) -> Prediction:
        if self._session is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        x = np.asarray(features, dtype=np.float32).reshape(1, -1)
        outputs = self._session.run(None, {self._input_name: x})

        # Expect outputs[0] to be probabilities, shape (1, n_classes)
        proba = outputs[0][0]

        pred_idx = int(np.argmax(proba))
        action = self._label_map.get(pred_idx, "HOLD")
        confidence = float(proba[pred_idx])

        prob_dict = {
            self._label_map.get(i, str(i)): float(p)
            for i, p in enumerate(proba)
        }

        p_buy = prob_dict.get("BUY", 0.0)
        p_sell = prob_dict.get("SELL", 0.0)

        return Prediction(
            action=action,
            confidence=confidence,
            expected_return=p_buy - p_sell,
            probabilities=prob_dict,
            model_version=self.version,
        )
