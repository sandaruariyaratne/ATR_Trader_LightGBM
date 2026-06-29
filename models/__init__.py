from .base_model import BaseModel, Prediction
from .xgboost_model import XGBoostModel
from .onnx_model import ONNXModel
from .torch_model import TorchModel
from config.settings import Settings


def load_model(settings: Settings) -> BaseModel:
    """Factory: instantiate and load the configured model."""
    model_map = {
        "xgboost": XGBoostModel,
        "onnx": ONNXModel,
        "torch": TorchModel,
    }
    cls = model_map.get(settings.model_type)
    if cls is None:
        raise ValueError(f"Unknown model type: {settings.model_type!r}")
    model = cls(settings.model_path)
    model.load()
    return model


__all__ = [
    "BaseModel", "Prediction",
    "XGBoostModel", "ONNXModel", "TorchModel",
    "load_model",
]
