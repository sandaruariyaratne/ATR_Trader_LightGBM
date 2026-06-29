"""
agents/risk_management_agent.py
────────────────────────────────
Strict execution filter between inference and execution.

Checks performed on every TradeSignalEvent:
  1. Confidence threshold gate
  2. Circuit breaker / drawdown check
  3. Max open positions check
  4. Consecutive loss limit (circuit breaker trigger)
  5. Position sizing (fractional Kelly + volatility scaling)
  6. Stop-loss and take-profit calculation (ATR-based)

Emits ApprovedOrderEvent only when all checks pass; drops the signal otherwise.
"""
from __future__ import annotations

import asyncio
from typing import Optional

from config.settings import Settings
from config.trading_params import RISK_PARAMS
from core.event_bus import (
    ApprovedOrderEvent,
    EventBus,
    MarketStateEvent,
    TradeSignalEvent,
)
from core.logger import get_logger
from core.state_manager import StateManager

logger = get_logger("risk_agent")


class RiskManagementAgent:
    def __init__(
        self,
        bus: EventBus,
        state: StateManager,
        settings: Settings,
    ) -> None:
        self.bus = bus
        self.state = state
        self.settings = settings
        self.rp = RISK_PARAMS

        # Cache latest market state for ATR / current price
        self._latest_market: Optional[MarketStateEvent] = None

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        logger.info("risk_agent.starting")
        # Also listen to market_state to keep ATR up to date
        # (We subscribe to a copy via a secondary consumer pattern)
        market_task = asyncio.create_task(self._market_state_listener())

        while True:
            try:
                signal: TradeSignalEvent = await self.bus.consume("trade_signal")
                order = await self._evaluate(signal)
                if order is not None:
                    await self.bus.publish("approved_order", order)
            except asyncio.CancelledError:
                market_task.cancel()
                logger.info("risk_agent.cancelled")
                raise
            except Exception as exc:
                logger.error("risk_agent.error", error=str(exc))

    async def _market_state_listener(self) -> None:
        """Keep a local cache of the latest market state for ATR access."""
        # NOTE: In the full pipeline, MarketStateEvent is consumed by
        # FeatureEngineeringAgent. We piggyback on the trade_signal path
        # which carries all the info we need via the state manager.
        # This task is a no-op placeholder for multi-bus extensions.
        while True:
            await asyncio.sleep(60)

    # ── Evaluation ────────────────────────────────────────────────────────────

    async def _evaluate(self, signal: TradeSignalEvent) -> Optional[ApprovedOrderEvent]:
        # ── 1. HOLD signals never execute ─────────────────────────────────────
        if signal.action == "HOLD":
            return None

        # ── 2. Confidence gate ────────────────────────────────────────────────
        if signal.confidence < self.settings.confidence_threshold:
            logger.debug(
                "risk_agent.low_confidence_rejected",
                confidence=signal.confidence,
                threshold=self.settings.confidence_threshold,
            )
            return None

        # ── 3. Circuit breaker ────────────────────────────────────────────────
        if await self.state.check_circuit_breaker():
            logger.warning("risk_agent.circuit_breaker_active")
            return None

        # ── 4. Drawdown check ─────────────────────────────────────────────────
        if self.state.drawdown >= self.settings.max_drawdown_pct:
            logger.warning(
                "risk_agent.max_drawdown_hit",
                drawdown=round(self.state.drawdown * 100, 2),
            )
            await self.state.trigger_circuit_breaker(self.rp.circuit_breaker_cooldown)
            return None

        # ── 5. Max open positions ─────────────────────────────────────────────
        if len(self.state.open_positions) >= self.settings.max_open_positions:
            logger.debug(
                "risk_agent.max_positions_reached",
                open=len(self.state.open_positions),
            )
            return None

        # ── 6. Consecutive loss limit ─────────────────────────────────────────
        if self.state.consecutive_losses >= self.rp.max_consecutive_losses:
            logger.warning(
                "risk_agent.consecutive_loss_limit",
                losses=self.state.consecutive_losses,
            )
            await self.state.trigger_circuit_breaker(self.rp.circuit_breaker_cooldown)
            return None

        # ── 7. Position sizing ────────────────────────────────────────────────
        atr_val = self.state.latest_atr
        entry_price = self.state.latest_close
        quantity = self._size_position(signal, entry_price, atr_val)

        if quantity * entry_price < self.rp.min_order_value:
            logger.debug("risk_agent.order_too_small", value=quantity * entry_price)
            return None

        # ── 8. Stop / take-profit ─────────────────────────────────────────────
        side = "buy" if signal.action == "BUY" else "sell"
        stop_loss, take_profit = self._compute_sl_tp(side, entry_price, atr_val)

        logger.info(
            "risk_agent.order_approved",
            side=side,
            quantity=round(quantity, 6),
            entry=entry_price,
            sl=stop_loss,
            tp=take_profit,
            confidence=round(signal.confidence, 3),
        )

        return ApprovedOrderEvent(
            symbol=signal.symbol,
            timestamp=signal.timestamp,
            side=side,
            order_type="market",
            quantity=quantity,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            rationale=f"model={signal.model_version}, conf={signal.confidence:.2f}",
        )

    def _estimate_entry_price(self, signal: TradeSignalEvent) -> float:
        """Get the latest raw close price from the shared state manager."""
        return self.state.latest_close

    def _size_position(
        self,
        signal: TradeSignalEvent,
        entry_price: float,
        atr: float,
    ) -> float:
        """
        Fractional Kelly position sizing with volatility scaling.

        quantity = (equity × max_pct × kelly_fraction × volatility_scale) / entry_price
        """
        if entry_price <= 0:
            return 0.0

        equity = self.state.equity
        base_capital = equity * self.settings.max_position_pct
        kelly = self.rp.kelly_fraction

        # Volatility scaling: reduce size when ATR is high
        if atr > 0 and entry_price > 0:
            atr_pct = atr / entry_price
            vol_scale = max(0.25, 1.0 - atr_pct * 10)
        else:
            vol_scale = 1.0

        # Confidence scaling: reduce size below 0.8 confidence
        conf_scale = min(signal.confidence / 0.8, 1.0)

        notional = base_capital * kelly * vol_scale * conf_scale
        return notional / entry_price

    def _compute_sl_tp(
        self,
        side: str,
        entry: float,
        atr: float,
    ) -> tuple[float, float]:
        """Volatility-adjusted stop-loss and take-profit using ATR multipliers."""
        if side == "buy":
            stop_loss = entry - (self.settings.stop_loss_atr_mult * atr)
            take_profit = entry + (self.settings.take_profit_atr_mult * atr)
        else:
            stop_loss = entry + (self.settings.stop_loss_atr_mult * atr)
            take_profit = entry - (self.settings.take_profit_atr_mult * atr)

        return max(stop_loss, 0.0), take_profit
