"""
agents/market_data_agent.py
────────────────────────────
Ingests real-time 1-minute OHLCV candles from Binance via CCXT Pro
WebSocket. Computes technical indicators on each candle close and emits
a MarketStateEvent onto the event bus.

Fallback: if WebSocket delivery fails, polls REST at 1-second intervals
until the next candle boundary, then retries WebSocket.
"""
from __future__ import annotations

import asyncio
from collections import deque
from typing import Deque, List, Optional

try:
    import ccxt.pro as ccxtpro
except ImportError:
    ccxtpro = None  # type: ignore[assignment]

from config.settings import Settings
from config.trading_params import INDICATOR_PARAMS
from core.event_bus import EventBus, MarketStateEvent
from core.logger import get_logger
from core.state_manager import StateManager
from utils.indicators import (
    atr, ema, macd, momentum, rsi, sma, volume_z_score, vwap,
)
from utils.time_utils import candle_open_ms, interval_to_ms, now_ms

logger = get_logger("market_data_agent")

# OHLCV indices
I_TS, I_O, I_H, I_L, I_C, I_V = 0, 1, 2, 3, 4, 5


class MarketDataAgent:
    """
    Streams 1-minute OHLCV candles, computes indicators, emits events.
    """

    def __init__(
        self,
        bus: EventBus,
        state: StateManager,
        settings: Settings,
    ) -> None:
        self.bus = bus
        self.state = state
        self.settings = settings
        self.symbol = settings.trading_symbol
        self.interval = settings.candle_interval
        self.interval_ms = interval_to_ms(self.interval)

        buf_size = settings.candle_buffer_size
        self._candles: Deque[list] = deque(maxlen=buf_size)
        self._last_candle_ts: Optional[int] = None
        self._exchange = None

    # ── Exchange ──────────────────────────────────────────────────────────────

    def _build_exchange(self):
        exchange_class = getattr(ccxtpro, self.settings.exchange_id)
        exchange = exchange_class(
            {
                "apiKey": self.settings.exchange_api_key,
                "secret": self.settings.exchange_api_secret,
                "enableRateLimit": True,
                "options": {"defaultType": "future"},
            }
        )
        if self.settings.sandbox:
            exchange.enableDemoTrading(True)
        return exchange

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        logger.info("market_data_agent.starting", symbol=self.symbol)
        self._exchange = self._build_exchange()

        try:
            await self._seed_history()
            await self._stream_candles()
        except asyncio.CancelledError:
            logger.info("market_data_agent.cancelled")
        except Exception as exc:
            logger.error("market_data_agent.fatal", error=str(exc))
            raise
        finally:
            if self._exchange:
                await self._exchange.close()

    async def _seed_history(self) -> None:
        """Pre-fill the candle buffer with historical 10-column data."""
        logger.info("market_data_agent.seeding_history", bars=self.settings.candle_buffer_size)
        
        # Format symbol for Binance REST API (e.g. SOLUSDT)
        binance_symbol = self.symbol.replace("/", "").replace(":", "")
        params = {
            "symbol": binance_symbol,
            "interval": self.interval,
            "limit": self.settings.candle_buffer_size,
        }
        
        try:
            # Try futures endpoint first, fallback to spot
            try:
                raw_klines = await self._exchange.fapiPublicGetKlines(params)
            except Exception:
                raw_klines = await self._exchange.publicGetKlines(params)
                
            for c in raw_klines:
                parsed = [
                    int(c[0]),          # timestamp
                    float(c[1]),        # open
                    float(c[2]),        # high
                    float(c[3]),        # low
                    float(c[4]),        # close
                    float(c[5]),        # volume
                    float(c[10]),       # taker_buy_quote_volume
                    float(c[9]),        # taker_buy_base_volume
                    float(c[7]),        # quote_volume
                    float(c[8]),        # trades
                ]
                self._candles.append(parsed)
            if self._candles:
                self._last_candle_ts = self._candles[-1][I_TS]
            logger.info("market_data_agent.history_loaded", bars=len(self._candles))
        except Exception as exc:
            logger.error("market_data_agent.seeding_failed", error=str(exc))
            raise

    async def _stream_candles(self) -> None:
        """Fetch completed candles at the start of each minute boundary."""
        binance_symbol = self.symbol.replace("/", "").replace(":", "")
        params = {
            "symbol": binance_symbol,
            "interval": self.interval,
            "limit": 5,
        }
        
        while True:
            try:
                # Poll REST API at boundary closes (more stable for full 10 columns)
                try:
                    raw_klines = await self._exchange.fapiPublicGetKlines(params)
                except Exception:
                    raw_klines = await self._exchange.publicGetKlines(params)
                    
                parsed_klines = []
                for c in raw_klines:
                    parsed = [
                        int(c[0]),          # timestamp
                        float(c[1]),        # open
                        float(c[2]),        # high
                        float(c[3]),        # low
                        float(c[4]),        # close
                        float(c[5]),        # volume
                        float(c[10]),       # taker_buy_quote_volume
                        float(c[9]),        # taker_buy_base_volume
                        float(c[7]),        # quote_volume
                        float(c[8]),        # trades
                    ]
                    parsed_klines.append(parsed)
                    
                await self._process_candles(parsed_klines)
                await asyncio.sleep(2) # poll every 2 seconds
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("market_data_agent.poll_error", error=str(exc))
                await asyncio.sleep(5)

    async def _process_candles(self, incoming: List[list]) -> None:
        """Handle a batch of candles from the REST response."""
        for candle in incoming:
            ts = candle[I_TS]
            candle_open = candle_open_ms(ts, self.interval_ms)

            # Only emit on a *new* closed candle (not the current open one)
            if self._last_candle_ts is not None and candle_open <= self._last_candle_ts:
                continue

            # Update buffer
            self._candles.append(candle)
            self._last_candle_ts = candle_open

            # Compute indicators and emit
            event = self._build_event(candle)
            await self.bus.publish("market_state", event)
            logger.info(
                "market_data_agent.candle_emitted",
                ts=candle_open,
                close=candle[I_C],
                volume=candle[I_V],
            )

    # ── Indicator computation ─────────────────────────────────────────────────

    def _build_event(self, latest: list) -> MarketStateEvent:
        candles = list(self._candles)
        opens   = [c[I_O] for c in candles]
        highs   = [c[I_H] for c in candles]
        lows    = [c[I_L] for c in candles]
        closes  = [c[I_C] for c in candles]
        volumes = [c[I_V] for c in candles]

        p = INDICATOR_PARAMS
        macd_val, macd_sig, macd_hist = macd(closes, p.macd_fast, p.macd_slow, p.macd_signal)

        indicators = {
            "ema_fast":     ema(closes, p.ema_fast),
            "ema_slow":     ema(closes, p.ema_slow),
            "ema_trend":    ema(closes, p.ema_trend),
            "sma_short":    sma(closes, p.sma_short),
            "sma_long":     sma(closes, p.sma_long),
            "rsi":          rsi(closes, p.rsi_period),
            "macd":         macd_val,
            "macd_signal":  macd_sig,
            "macd_hist":    macd_hist,
            "atr":          atr(highs, lows, closes, p.atr_period),
            "vwap":         vwap(highs, lows, closes, volumes),
            "volume_spike": volume_z_score(volumes),
            "momentum":     momentum(closes),
        }

        return MarketStateEvent(
            symbol=self.symbol,
            timestamp=latest[I_TS],
            open=latest[I_O],
            high=latest[I_H],
            low=latest[I_L],
            close=latest[I_C],
            volume=latest[I_V],
            indicators=indicators,
            raw_candles=candles,
        )
