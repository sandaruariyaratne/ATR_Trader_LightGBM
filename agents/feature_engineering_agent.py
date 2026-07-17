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
import time
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import ccxt.async_support as ccxt

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

        # Initialize async CCXT client for public REST calls
        exchange_class = getattr(ccxt, self.settings.exchange_id)
        self.exchange = exchange_class({
            "apiKey": self.settings.exchange_api_key,
            "secret": self.settings.exchange_api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })
        if self.settings.sandbox:
            self.exchange.enableDemoTrading(True)

        # Cache variables to prevent rate limit bans
        self.btc_closes: List[float] = []
        self.btc_timestamps: List[int] = []
        self.daily_200ema: Optional[float] = None
        self.last_daily_update: float = 0.0
        self.funding_rate: float = 0.0
        self.funding_rate_velocity: float = 0.0
        self.last_funding_update: float = 0.0

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
                if self.exchange:
                    await self.exchange.close()
                raise
            except Exception as exc:
                logger.error("feature_eng_agent.error", error=str(exc))

    # ── REST Cache Updates ───────────────────────────────────────────────────

    async def _update_btc_data(self) -> None:
        try:
            params = {"symbol": "BTCUSDT", "interval": "1m", "limit": 100}
            klines = await self.exchange.fapiPublicGetKlines(params)
            self.btc_timestamps = [int(k[0]) for k in klines]
            self.btc_closes = [float(k[4]) for k in klines]
        except Exception as e:
            logger.error("feature_eng.update_btc_failed", error=str(e))

    async def _update_daily_200ema(self, binance_symbol: str) -> None:
        now = time.time()
        if self.daily_200ema is not None and (now - self.last_daily_update) < 3600:
            return
        try:
            params = {"symbol": binance_symbol, "interval": "1d", "limit": 250}
            klines = await self.exchange.fapiPublicGetKlines(params)
            closes = [float(k[4]) for k in klines]
            if len(closes) >= 200:
                df_close = pd.Series(closes)
                ema_200 = df_close.ewm(span=200, adjust=False).mean()
                self.daily_200ema = float(ema_200.iloc[-1])
                self.last_daily_update = now
                logger.info("feature_eng.updated_daily_200ema", symbol=binance_symbol, ema=self.daily_200ema)
        except Exception as e:
            logger.error("feature_eng.update_daily_ema_failed", error=str(e))

    async def _update_funding_data(self, binance_symbol: str) -> None:
        now = time.time()
        if self.last_funding_update > 0 and (now - self.last_funding_update) < 300:
            return
        try:
            params = {"symbol": binance_symbol, "limit": 5}
            history = await self.exchange.fapiPublicGetFundingRate(params)
            if len(history) >= 2:
                history = sorted(history, key=lambda x: int(x["fundingTime"]))
                self.funding_rate = float(history[-1]["fundingRate"])
                prev_funding_rate = float(history[-2]["fundingRate"])
                self.funding_rate_velocity = self.funding_rate - prev_funding_rate
                self.last_funding_update = now
                logger.info(
                    "feature_eng.updated_funding",
                    symbol=binance_symbol,
                    funding=self.funding_rate,
                    velocity=self.funding_rate_velocity
                )
        except Exception as e:
            logger.error("feature_eng.update_funding_failed", error=str(e))

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

        # Build DataFrame from raw 10-column candles list
        df = pd.DataFrame(
            candles,
            columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "taker_buy_quote_volume", "taker_buy_base_volume", "quote_volume", "trades"
            ]
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

        # Calculate Category A & B features
        binance_symbol = event.symbol.replace("/", "").replace(":", "")
        await self._update_btc_data()
        await self._update_daily_200ema(binance_symbol)
        await self._update_funding_data(binance_symbol)

        # Align BTC close
        if self.btc_timestamps and self.btc_closes:
            ts_to_btc = dict(zip(self.btc_timestamps, self.btc_closes))
            df["close_btc"] = df["timestamp"].map(ts_to_btc).ffill().bfill()
        else:
            df["close_btc"] = df["close"]  # fallback to avoid crash

        df["btc_log_return_5m"] = np.log(df["close_btc"] / df["close_btc"].shift(5))
        df["btc_log_return_1h"] = np.log(df["close_btc"] / df["close_btc"].shift(60))
        
        df["log_ret_1m"] = np.log(df["close"] / df["close"].shift(1))
        df["btc_log_ret_1m"] = np.log(df["close_btc"] / df["close_btc"].shift(1))
        df["asset_btc_correlation_1h"] = df["log_ret_1m"].rolling(60).corr(df["btc_log_ret_1m"]).fillna(1.0)
        
        if self.daily_200ema is not None:
            df["distance_to_daily_200ema"] = (df["close"] - self.daily_200ema) / self.daily_200ema
        else:
            df["distance_to_daily_200ema"] = 0.0
            
        df["funding_rate"] = self.funding_rate
        df["funding_rate_velocity"] = self.funding_rate_velocity

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
            "close_btc", "log_ret_1m", "btc_log_ret_1m"
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
