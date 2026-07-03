"""
config/settings.py
──────────────────
Pydantic-settings based configuration loaded from environment / .env file.
All values can be overridden at runtime via environment variables.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Exchange ──────────────────────────────────────────────────────────────
    exchange_id: str = Field(default="binance", description="CCXT exchange id")
    exchange_api_key: str = Field(default="", description="API key")
    exchange_api_secret: str = Field(default="", description="API secret")

    # ── Trading ───────────────────────────────────────────────────────────────
    trading_symbol: str = Field(default="SOL/USDT")
    candle_interval: str = Field(default="1m")
    paper_trading: bool = Field(default=True, description="Disable real orders when True")
    fee_rate: float = Field(default=0.0010, description="Round-trip commission fee rate (e.g. 0.0010 for 10 bps)")

    # ── Capital & Risk ────────────────────────────────────────────────────────
    initial_capital: float = Field(default=10_000.0, gt=0)
    max_position_pct: float = Field(default=0.10, gt=0, le=1.0)
    max_drawdown_pct: float = Field(default=0.15, gt=0, le=1.0)
    stop_loss_atr_mult: float = Field(default=2.0, gt=0)
    take_profit_atr_mult: float = Field(default=4.0, gt=0)
    max_open_positions: int = Field(default=100, ge=1)

    # ── Model ─────────────────────────────────────────────────────────────────
    model_type: Literal["xgboost", "onnx", "torch"] = Field(default="xgboost")
    model_path: Path = Field(default=Path("data/models/lightgbm_SOLUSDT.pkl"))
    confidence_threshold: float = Field(default=0.50, ge=0.4, le=1.0)
    feature_window: int = Field(default=50, ge=20)

    # ── Pipeline ──────────────────────────────────────────────────────────────
    queue_max_size: int = Field(default=500, ge=10)
    candle_buffer_size: int = Field(default=200, ge=50)

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    log_dir: Path = Field(default=Path("logs"))

    @field_validator("log_dir", "model_path", mode="before")
    @classmethod
    def _coerce_path(cls, v: object) -> Path:
        return Path(v)  # type: ignore[arg-type]

    def ensure_dirs(self) -> None:
        """Create required directories if they don't exist."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)


# Singleton
_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_dirs()
    return _settings
