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

            try:
                balance_resp = await self._exchange.fetch_balance()
                usdt_balance = float(balance_resp.get("USDT", {}).get("total", self.state.equity))
                if usdt_balance > 0.0:
                    async with self.state._lock:
                        self.state.equity = usdt_balance
                    logger.info("execution_agent.balance_synced", balance=usdt_balance)
            except Exception as bal_exc:
                logger.error("execution_agent.sync_balance_failed", error=str(bal_exc))

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
            confidence=order.confidence,
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

    async def _get_actual_fill(self, order_id: str, symbol: str, fallback_price: float) -> tuple[float, float]:
        """
        Fetch order details from the exchange to obtain the actual average filled price and filled quantity.
        """
        if self.settings.paper_trading or not self._exchange:
            return 0.0, fallback_price

        for attempt in range(3):
            try:
                order_info = await self._exchange.fetch_order(order_id, symbol)
                filled_qty = float(order_info.get("filled", 0.0) or 0.0)
                avg_price = float(order_info.get("average", 0.0) or order_info.get("price", 0.0) or fallback_price)
                if filled_qty > 0.0:
                    return filled_qty, avg_price
            except Exception as e:
                logger.warning("execution_agent.fetch_fill_failed", order_id=order_id, attempt=attempt, error=str(e))
            await asyncio.sleep(0.5)

        return 0.0, fallback_price

    # ── Live trading ──────────────────────────────────────────────────────────

    async def _live_execute(self, order: ApprovedOrderEvent) -> OrderResultEvent:
        """Place a real order via CCXT Pro."""
        assert self._exchange is not None
        order_id = "UNKNOWN"
        try:
            is_limit = order.order_type == "limit"
            params = {}
            price = None
            if is_limit:
                params["postOnly"] = True
                price = order.entry_price

            response = None
            try:
                response = await self._exchange.create_order(
                    symbol=order.symbol,
                    type=order.order_type,
                    side=order.side,
                    amount=order.quantity,
                    price=price,
                    params=params,
                )
            except Exception as e:
                # Check for Binance Post-Only rejection code or message
                err_str = str(e)
                if "-5022" in err_str or "executed as maker" in err_str or "Post Only" in err_str:
                    logger.warning(
                        "execution_agent.post_only_rejected_retrying",
                        symbol=order.symbol,
                        side=order.side,
                        price=price,
                        error=err_str,
                    )
                    # Retry without postOnly
                    if "postOnly" in params:
                        del params["postOnly"]
                    response = await self._exchange.create_order(
                        symbol=order.symbol,
                        type=order.order_type,
                        side=order.side,
                        amount=order.quantity,
                        price=price,
                        params=params,
                    )
                else:
                    raise

            order_id = response["id"]
            
            filled_qty = 0.0
            avg_price = 0.0
            status = response.get("status", "open")


            if is_limit and status != "closed":
                timeout_sec = 45.0
                poll_interval = 1.0
                elapsed = 0.0
                logger.info(
                    "execution_agent.waiting_limit_fill",
                    order_id=order_id,
                    side=order.side,
                    price=price,
                    qty=order.quantity,
                )
                while elapsed < timeout_sec:
                    await asyncio.sleep(poll_interval)
                    elapsed += poll_interval
                    try:
                        order_status = await self._exchange.fetch_order(order_id, order.symbol)
                        status = order_status.get("status", "open")
                        filled_qty = float(order_status.get("filled", 0.0))
                        avg_price = float(order_status.get("average", 0.0) or order_status.get("price", 0.0) or price)
                        if status == "closed":
                            break
                        if status == "canceled":
                            break
                    except Exception as poll_exc:
                        logger.warning("execution_agent.poll_order_failed", error=str(poll_exc))
                
                if status not in ("closed", "canceled"):
                    logger.info("execution_agent.limit_timeout_cancelling", order_id=order_id, filled=filled_qty)
                    try:
                        await self._exchange.cancel_order(order_id, order.symbol)
                        # Fetch final status one last time to capture final filled qty
                        final_status = await self._exchange.fetch_order(order_id, order.symbol)
                        filled_qty = float(final_status.get("filled", filled_qty))
                        avg_price = float(final_status.get("average", avg_price) or final_status.get("price", avg_price) or price)
                    except Exception as cancel_exc:
                        logger.error("execution_agent.cancel_order_failed", error=str(cancel_exc))
            else:
                # For market entry orders, query the actual fill details from the exchange
                filled_qty, avg_price = await self._get_actual_fill(order_id, order.symbol, order.entry_price)
                if filled_qty <= 0.0:
                    filled_qty = float(response.get("filled", order.quantity) or order.quantity)
                    avg_price = float(response.get("average", order.entry_price) or order.entry_price)

            if filled_qty <= 0.0:
                logger.info("execution_agent.limit_order_unfilled", order_id=order_id)
                await self.state.log_cancelled_entry(
                    order_id=order_id,
                    symbol=order.symbol,
                    side=order.side,
                    qty=order.quantity,
                    price=price if price is not None else order.entry_price,
                    opened_at=order.timestamp,
                    confidence=order.confidence,
                    reason="unfilled_timeout"
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
                )

            position = Position(
                symbol=order.symbol,
                side="long" if order.side == "buy" else "short",
                entry_price=avg_price,
                quantity=filled_qty,
                stop_loss=order.stop_loss,
                take_profit=order.take_profit,
                opened_at=order.timestamp,
                order_id=order_id,
                confidence=order.confidence,
            )
            await self.state.add_position(position)

            if not self.settings.paper_trading:
                exit_side = "sell" if position.side == "long" else "buy"
                
                # 1. Take Profit resting Limit order
                try:
                    logger.info(
                        "execution_agent.placing_resting_tp",
                        symbol=position.symbol,
                        side=exit_side,
                        price=position.take_profit,
                    )
                    tp_resp = await self._exchange.create_order(
                        symbol=position.symbol,
                        type="limit",
                        side=exit_side,
                        amount=position.quantity,
                        price=position.take_profit,
                        params={"reduceOnly": True}
                    )
                    position.tp_order_id = tp_resp["id"]
                    logger.info("execution_agent.resting_tp_placed", order_id=tp_resp["id"])
                except Exception as tp_exc:
                    logger.error("execution_agent.tp_order_failed", error=str(tp_exc))

                # 2. Stop Loss resting Stop Market order
                try:
                    logger.info(
                        "execution_agent.placing_resting_sl",
                        symbol=position.symbol,
                        side=exit_side,
                        trigger_price=position.stop_loss,
                    )
                    sl_resp = await self._exchange.create_order(
                        symbol=position.symbol,
                        type="STOP_MARKET",
                        side=exit_side,
                        amount=position.quantity,
                        price=None,
                        params={"reduceOnly": True, "stopPrice": position.stop_loss}
                    )
                    position.sl_order_id = sl_resp["id"]
                    logger.info("execution_agent.resting_sl_placed", order_id=sl_resp["id"])
                except Exception as sl_exc:
                    logger.error("execution_agent.sl_order_failed", error=str(sl_exc))

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
                if self.settings.paper_trading:
                    exit_reason = self._check_exit(pos, current_price)
                    if exit_reason:
                        await self._close_position(pos, order_id, current_price, exit_reason)
                else:
                    # In live trading, check time barrier first
                    now_ms = int(time.time() * 1000)
                    if now_ms - pos.opened_at >= 15 * self.interval_ms:
                        await self._close_position(pos, order_id, current_price, "timeout")
                        continue

                    # Check if Take Profit has been filled (standard limit order works perfectly)
                    tp_filled = False
                    tp_price = pos.take_profit
                    if pos.tp_order_id:
                        try:
                            tp_status = await self._exchange.fetch_order(pos.tp_order_id, pos.symbol)
                            if tp_status.get("status") == "closed":
                                tp_filled = True
                                tp_price = float(tp_status.get("average", pos.take_profit) or tp_status.get("price", pos.take_profit) or pos.take_profit)
                        except Exception as e:
                            logger.warning("execution_agent.fetch_tp_failed", order_id=pos.tp_order_id, error=str(e))

                    if tp_filled:
                        await self._close_position(pos, order_id, tp_price, "tp")
                        continue

                    # For Stop Loss, use hybrid price checking to avoid Binance Testnet Algo order fetch bugs.
                    # The exchange will execute the STOP_MARKET order itself if hit.
                    exit_reason = self._check_exit(pos, current_price)
                    if exit_reason == "sl":
                        sl_price = pos.stop_loss
                        if pos.sl_order_id:
                            try:
                                sl_info = await self._exchange.fetch_order(pos.sl_order_id, pos.symbol)
                                sl_price = float(sl_info.get("average", pos.stop_loss) or sl_info.get("price", pos.stop_loss) or pos.stop_loss)
                            except Exception as e:
                                logger.warning("execution_agent.fetch_sl_fill_failed", order_id=pos.sl_order_id, error=str(e))
                        await self._close_position(pos, order_id, sl_price, "sl")
                        continue

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
        if not self.settings.paper_trading:
            assert self._exchange is not None
            
            exit_side = "sell" if pos.side == "long" else "buy"
            
            # 1. Cancel TP order if reason is SL or timeout
            if reason in ("sl", "timeout") and pos.tp_order_id:
                try:
                    logger.info("execution_agent.cancelling_resting_tp", order_id=pos.tp_order_id)
                    await self._exchange.cancel_order(pos.tp_order_id, pos.symbol)
                except Exception as exc:
                    logger.warning("execution_agent.cancel_tp_by_id_failed_trying_lookup", error=str(exc))
                    try:
                        open_orders = await self._exchange.fetch_open_orders(pos.symbol)
                        for o in open_orders:
                            if o.get("side") == exit_side and o.get("type") == "limit":
                                if abs(float(o.get("price", 0.0)) - pos.take_profit) < 1e-4:
                                    logger.info("execution_agent.cancelling_tp_by_lookup", order_id=o["id"])
                                    await self._exchange.cancel_order(o["id"], pos.symbol)
                                    break
                    except Exception as fallback_exc:
                        logger.error("execution_agent.cancel_tp_fallback_failed", error=str(fallback_exc))

            # 2. Cancel SL order if reason is TP or timeout
            if reason in ("tp", "timeout") and pos.sl_order_id:
                try:
                    logger.info("execution_agent.cancelling_resting_sl", order_id=pos.sl_order_id)
                    await self._exchange.cancel_order(pos.sl_order_id, pos.symbol)
                except Exception as exc:
                    logger.warning("execution_agent.cancel_sl_by_id_failed_trying_lookup", error=str(exc))
                    try:
                        open_orders = await self._exchange.fetch_open_orders(pos.symbol)
                        matched = False
                        for o in open_orders:
                            o_side = str(o.get("side") or "").lower()
                            o_type = str(o.get("type") or "").lower()
                            raw_info = o.get("info") or {}
                            o_stop = float(o.get("triggerPrice") or o.get("stopPrice") or raw_info.get("stopPrice") or raw_info.get("triggerPrice") or 0.0)
                            
                            # Match by side and check if trigger price is within 0.005
                            if o_side == exit_side.lower() and ("stop" in o_type or o_stop > 0.0):
                                if abs(o_stop - pos.stop_loss) < 0.005:
                                    logger.info("execution_agent.cancelling_sl_by_lookup", order_id=o["id"])
                                    await self._exchange.cancel_order(o["id"], pos.symbol)
                                    matched = True
                                    break
                        if not matched:
                            logger.warning(
                                "execution_agent.cancel_sl_lookup_no_match",
                                exit_side=exit_side,
                                target_sl=pos.stop_loss,
                                open_orders=[{
                                    "id": o.get("id"),
                                    "side": o.get("side"),
                                    "type": o.get("type"),
                                    "price": o.get("price"),
                                    "triggerPrice": o.get("triggerPrice"),
                                    "stopPrice": o.get("stopPrice"),
                                    "raw_stopPrice": (o.get("info") or {}).get("stopPrice")
                                } for o in open_orders]
                            )
                    except Exception as fallback_exc:
                        logger.error("execution_agent.cancel_sl_fallback_failed", error=str(fallback_exc))

            # Only place a market exit order if it's a timeout exit
            if reason == "timeout":
                try:
                    exit_side = "sell" if pos.side == "long" else "buy"
                    logger.info(
                        "execution_agent.placing_exit_order",
                        symbol=pos.symbol,
                        side=exit_side,
                        qty=pos.quantity,
                        reason=reason,
                    )
                    response = await self._exchange.create_order(
                        symbol=pos.symbol,
                        type="market",
                        side=exit_side,
                        amount=pos.quantity,
                    )
                    exit_order_id = response["id"]
                    _, avg_price = await self._get_actual_fill(exit_order_id, pos.symbol, price)
                    price = avg_price
                except Exception as exc:
                    logger.error(
                        "execution_agent.exit_order_failed",
                        error=str(exc),
                        order_id=order_id,
                    )

        entry_fee = pos.entry_price * pos.quantity * 0.0002
        exit_fee_rate = 0.0002 if reason == "tp" else 0.0004
        exit_fee = price * pos.quantity * exit_fee_rate
        fee = entry_fee + exit_fee

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
