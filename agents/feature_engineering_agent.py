"""
agents/feature_engineering_agent.py
-------------------------------------
Consumes MarketStateEvent, transforms raw indicators into ML-ready features:
  • Computes stationary features on-the-fly from raw candles
  • Handles regimes (trend / sideways / volatile)
  • Dynamic feature selection mapping

Emits FeatureVectorEvent when warm (buffer has enough history).
"""
from __future__ import annotations

import asyncio
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from config.settings import Settings
from config.trading_params import FEATURE_PARAMS, INDICATOR_PARAMS
from core.event_bus import EventBus, FeatureVectorEvent, MarketStateEvent
from core.logger import get_logger
from core.state_manager import StateManager
from utils.indicators import adx
from utils.features import build_stationary_features

logger = get_logger("feature_eng_agent")

_REGIME_MAP = {"sideways": 0, "trend": 1, "volatile": 2}


class FeatureEngineeringAgent:
    def __init__(
        self,
        bus: EventBus,
        state: StateManager,
        settings: Settings,
    ) -> None:
        self.bus = bus
        self.state = state
        self.settings = settings
        self.fp = FEATURE_PARAMS

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        logger.info("feature_eng_agent.starting")
        while True:
            try:
                event: MarketStateEvent = await self.bus.consume("market_state")
                feat_event = await self._process(event)
                if feat_event is not None:
                    await self.bus.publish("feature_vector", feat_event)
            except asyncio.CancelledError:
                logger.info("feature_eng_agent.cancelled")
                raise
            except Exception as exc:
                logger.error("feature_eng_agent.error", error=str(exc))

    # ── Processing ────────────────────────────────────────────────────────────

    async def _process(self, event: MarketStateEvent) -> Optional[FeatureVectorEvent]:
        # Update shared state with raw close price and volatility
        await self.state.update_market_feed(event.close, event.indicators.get("atr", 0.0))

        candles = event.raw_candles
        if len(candles) < 241:
            logger.debug(
                "feature_eng_agent.warming_up",
                have=len(candles),
                need=241,
            )
            return None

        # Build DataFrame from raw candles list
        df = pd.DataFrame(
            candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

        # Classify market regime using ADX
        regime = self._classify_regime(
            df["high"].tolist(),
            df["low"].tolist(),
            df["close"].tolist(),
            event.close,
        )

        # Build stationary features
        df = build_stationary_features(df)
        df.dropna(inplace=True)
        if len(df) == 0:
            return None

        # Build feature dictionary for the latest closed candle (the last row)
        feature_dict = self._build_feature_dict(df)
        
        # Inject regime encoded feature
        feature_dict["regime_encoded"] = float(_REGIME_MAP[regime])

        feature_names = list(feature_dict.keys())
        features = [feature_dict[k] for k in feature_names]

        logger.debug(
            "feature_eng_agent.emitting",
            regime=regime,
            n_features=len(features),
        )

        return FeatureVectorEvent(
            symbol=event.symbol,
            timestamp=event.timestamp,
            features=features,
            feature_names=feature_names,
            regime=regime,
        )

    def _build_feature_dict(self, df: pd.DataFrame) -> Dict[str, float]:
        exclude_cols = (
            "timestamp", "label", "open", "high", "low", "close", "volume",
            "quote_volume", "taker_buy_base_volume", "taker_buy_quote_volume", "trades",
            "taker_buy_ratio", "quote_vol_dominance", "avg_trade_size_change_5m",
            "net_taker_flow_5m", "net_taker_flow_15m", "whale_buying_factor_5m"
        )
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        last_row = df.iloc[-1]
        return {col: float(last_row[col]) for col in feature_cols}

    def _classify_regime(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        current_close: float,
    ) -> str:
        fp = FEATURE_PARAMS

        if len(closes) < 20:
            return "sideways"

        adx_val = adx(highs, lows, closes, period=INDICATOR_PARAMS.atr_period)

        # Volatility: ATR as % of price
        recent = closes[-14:]
        hi_14 = highs[-14:]
        lo_14 = lows[-14:]
        prev = closes[-15:-1]
        # TR: use hi_14 / lo_14 aligned with prev (14 bars each)
        h_arr = np.array(hi_14)
        l_arr = np.array(lo_14)
        p_arr = np.array(prev)
        n = min(len(h_arr), len(l_arr), len(p_arr))
        if n < 1:
            tr = np.array([0.0])
        else:
            tr = np.maximum(
                h_arr[-n:] - l_arr[-n:],
                np.maximum(np.abs(h_arr[-n:] - p_arr[-n:]), np.abs(l_arr[-n:] - p_arr[-n:])),
            )
        atr_pct = float(np.mean(tr)) / current_close if current_close > 0 else 0.0

        if atr_pct > fp.volatility_regime_pct:
            return "volatile"
        if adx_val >= fp.adx_trend_threshold:
            return "trend"
        return "sideways"
