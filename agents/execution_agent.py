"""
agents/execution_agent.py
──────────────────────────
Order execution via CCXT Pro.

Responsibilities:
  • Place market or limit orders on approval
  • Manage open positions: poll for SL/TP hits every 5 seconds
  • Update PnL in state manager
  • Emit OrderResultEvent for each fill
  • Paper-trading mode: simulate fills without real orders

Design note: a production deployment would use WebSocket order updates
instead of polling. The polling fallback here ensures compatibility with
exchanges that don't support order WebSockets.
"""
from __future__ import annotations

import asyncio
import time
from typing import Dict, Optional
from uuid import uuid4

try:
    import ccxt.pro as ccxtpro
except ImportError:
    ccxtpro = None  # type: ignore[assignment]

from config.settings import Settings
from core.event_bus import ApprovedOrderEvent, EventBus, OrderResultEvent
from core.logger import get_logger
from core.state_manager import Position, StateManager
from utils.time_utils import interval_to_ms

logger = get_logger("execution_agent")

_POLL_INTERVAL = 5.0   # seconds between position checks
_PRICE_TOLERANCE = 0.001  # 0.1 % slippage tolerance


class ExecutionAgent:
    def __init__(
        self,
        bus: EventBus,
        state: StateManager,
        settings: Settings,
    ) -> None:
        self.bus = bus
        self.state = state
        self.settings = settings
        self._exchange = None
        self._pending_order: Optional[ApprovedOrderEvent] = None
        self.interval_ms = interval_to_ms(settings.candle_interval)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        logger.info(
            "execution_agent.starting",
            paper=self.settings.paper_trading,
        )
        if not self.settings.paper_trading:
            self._exchange = self._build_exchange()
            try:
                if hasattr(self._exchange, "set_leverage"):
                    await self._exchange.set_leverage(int(self.settings.leverage), self.settings.trading_symbol)
                    logger.info("execution_agent.leverage_configured", leverage=self.settings.leverage)
            except Exception as exc:
                logger.warning("execution_agent.set_leverage_failed", error=str(exc))

        order_listener = asyncio.create_task(self._order_listener())
        position_monitor = asyncio.create_task(self._position_monitor())

        try:
            await asyncio.gather(order_listener, position_monitor)
        except asyncio.CancelledError:
            order_listener.cancel()
            position_monitor.cancel()
            logger.info("execution_agent.cancelled")
            raise
        finally:
            if self._exchange:
                await self._exchange.close()

    def _build_exchange(self):
        cls = getattr(ccxtpro, self.settings.exchange_id)
        exchange = cls(
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

    # ── Order listener ────────────────────────────────────────────────────────

    async def _order_listener(self) -> None:
        while True:
            try:
                order: ApprovedOrderEvent = await self.bus.consume("approved_order")
                result = await self._execute(order)
                await self.bus.publish("order_result", result)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("execution_agent.order_error", error=str(exc))

    async def _execute(self, order: ApprovedOrderEvent) -> OrderResultEvent:
        if self.settings.paper_trading:
            return await self._paper_execute(order)
        return await self._live_execute(order)

    # ── Paper trading ─────────────────────────────────────────────────────────

    async def _paper_execute(self, order: ApprovedOrderEvent) -> OrderResultEvent:
        """Simulate an instant fill at the entry price."""
        await asyncio.sleep(0)   # yield to event loop
        order_id = f"PAPER-{uuid4().hex[:8].upper()}"

        position = Position(
            symbol=order.symbol,
            side="long" if order.side == "buy" else "short",
            entry_price=order.entry_price,
            quantity=order.quantity,
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
            opened_at=order.timestamp,
            order_id=order_id,
        )
        await self.state.add_position(position)

        logger.info(
            "execution_agent.paper_fill",
            order_id=order_id,
            side=order.side,
            qty=round(order.quantity, 6),
            price=order.entry_price,
        )

        return OrderResultEvent(
            symbol=order.symbol,
            timestamp=int(time.time() * 1000),
            order_id=order_id,
            side=order.side,
            filled_qty=order.quantity,
            avg_price=order.entry_price,
            status="filled",
            pnl=0.0,
        )

    # ── Live trading ──────────────────────────────────────────────────────────

    async def _live_execute(self, order: ApprovedOrderEvent) -> OrderResultEvent:
        """Place a real order via CCXT Pro."""
        assert self._exchange is not None
        order_id = "UNKNOWN"
        try:
            response = await self._exchange.create_order(
                symbol=order.symbol,
                type=order.order_type,
                side=order.side,
                amount=order.quantity,
            )
            order_id = response["id"]
            filled_qty = float(response.get("filled", order.quantity))
            avg_price = float(response.get("average", order.entry_price) or order.entry_price)

            position = Position(
                symbol=order.symbol,
                side="long" if order.side == "buy" else "short",
                entry_price=avg_price,
                quantity=filled_qty,
                stop_loss=order.stop_loss,
                take_profit=order.take_profit,
                opened_at=order.timestamp,
                order_id=order_id,
            )
            await self.state.add_position(position)

            logger.info(
                "execution_agent.live_fill",
                order_id=order_id,
                side=order.side,
                qty=filled_qty,
                avg_price=avg_price,
            )

            return OrderResultEvent(
                symbol=order.symbol,
                timestamp=int(time.time() * 1000),
                order_id=order_id,
                side=order.side,
                filled_qty=filled_qty,
                avg_price=avg_price,
                status="filled",
                pnl=0.0,
            )

        except Exception as exc:
            logger.error(
                "execution_agent.live_order_failed",
                error=str(exc),
                order_id=order_id,
            )
            return OrderResultEvent(
                symbol=order.symbol,
                timestamp=int(time.time() * 1000),
                order_id=order_id,
                side=order.side,
                filled_qty=0.0,
                avg_price=0.0,
                status="rejected",
                pnl=0.0,
                error=str(exc),
            )

    # ── Position monitor ──────────────────────────────────────────────────────

    async def _position_monitor(self) -> None:
        """Poll open positions and close on SL / TP breach."""
        while True:
            await asyncio.sleep(_POLL_INTERVAL)
            if not self.state.open_positions:
                continue

            current_price = await self._fetch_price()
            if current_price is None:
                continue

            for order_id, pos in list(self.state.open_positions.items()):
                exit_reason = self._check_exit(pos, current_price)
                if exit_reason:
                    await self._close_position(pos, order_id, current_price, exit_reason)

    def _check_exit(self, pos: Position, price: float) -> Optional[str]:
        # 1. Check vertical barrier (time limit of 15 candle periods)
        now_ms = int(time.time() * 1000)
        if now_ms - pos.opened_at >= 15 * self.interval_ms:
            return "timeout"

        if pos.side == "long":
            if price <= pos.stop_loss:
                return "sl"
            if price >= pos.take_profit:
                return "tp"
        else:  # short
            if price >= pos.stop_loss:
                return "sl"
            if price <= pos.take_profit:
                return "tp"
        return None

    async def _close_position(
        self,
        pos: Position,
        order_id: str,
        price: float,
        reason: str,
    ) -> None:
        fee_rate = getattr(self.settings, "fee_rate", 0.0010)
        fee = (pos.entry_price + price) * pos.quantity * (fee_rate / 2.0)

        if pos.side == "long":
            pnl = (price - pos.entry_price) * pos.quantity - fee
        else:
            pnl = (pos.entry_price - price) * pos.quantity - fee

        record = await self.state.close_position(order_id, price, reason, pnl)

        result = OrderResultEvent(
            symbol=pos.symbol,
            timestamp=int(time.time() * 1000),
            order_id=order_id,
            side="sell" if pos.side == "long" else "buy",
            filled_qty=pos.quantity,
            avg_price=price,
            status="filled",
            pnl=pnl,
        )
        await self.bus.publish("order_result", result)

        logger.info(
            "execution_agent.position_closed",
            order_id=order_id,
            reason=reason,
            pnl=round(pnl, 4),
            fee=round(fee, 4),
            price=price,
        )

    async def _fetch_price(self) -> Optional[float]:
        """Fetch latest ticker price."""
        try:
            if self._exchange is None:
                self._exchange = self._build_exchange()
            ticker = await self._exchange.fetch_ticker(self.settings.trading_symbol)
            return float(ticker["last"])
        except Exception as exc:
            logger.warning("execution_agent.price_fetch_failed", error=str(exc))
            return None
