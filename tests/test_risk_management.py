"""
tests/test_risk_management.py
──────────────────────────────
Unit tests for RiskManagementAgent — all exchange calls are mocked.
"""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.event_bus import TradeSignalEvent
from core.state_manager import StateManager


# ─── Fixtures ────────────────────────────────────────────────────────────────

def _make_signal(
    action: str = "BUY",
    confidence: float = 0.75,
    expected_return: float = 0.02,
) -> TradeSignalEvent:
    return TradeSignalEvent(
        symbol="BTC/USDT",
        timestamp=1_700_000_000_000,
        action=action,
        confidence=confidence,
        expected_return=expected_return,
        model_version="test-1.0",
        features_snapshot={
            "ema_trend": 30000.0,
            "atr": 150.0,
            "vwap": 30050.0,
        },
    )


def _build_agent(initial_capital: float = 10_000.0, confidence_threshold: float = 0.60):
    from agents.risk_management_agent import RiskManagementAgent
    from config.settings import Settings

    bus = MagicMock()
    bus.consume = AsyncMock()
    bus.publish = AsyncMock()
    state = StateManager(initial_capital=initial_capital)
    state.latest_close = 30000.0
    state.latest_atr = 150.0
    settings = Settings(
        confidence_threshold=confidence_threshold,
        confidence_threshold_buy=confidence_threshold,
        confidence_threshold_sell=confidence_threshold,
        max_position_pct=0.10,
        max_drawdown_pct=0.15,
        stop_loss_atr_mult=2.0,
        take_profit_atr_mult=3.0,
        max_open_positions=1,
        paper_trading=True,
        sell_only=False,
    )
    return RiskManagementAgent(bus, state, settings), state


# ─── Tests ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestRiskManagementAgent:

    async def test_hold_signal_returns_none(self):
        agent, _ = _build_agent()
        result = await agent._evaluate(_make_signal(action="HOLD"))
        assert result is None

    async def test_low_confidence_rejected(self):
        agent, _ = _build_agent(confidence_threshold=0.80)
        signal = _make_signal(confidence=0.65)
        result = await agent._evaluate(signal)
        assert result is None

    async def test_high_confidence_buy_approved(self):
        agent, _ = _build_agent()
        signal = _make_signal(action="BUY", confidence=0.85)
        result = await agent._evaluate(signal)
        assert result is not None
        assert result.side == "buy"

    async def test_high_confidence_sell_approved(self):
        agent, _ = _build_agent()
        signal = _make_signal(action="SELL", confidence=0.85)
        result = await agent._evaluate(signal)
        assert result is not None
        assert result.side == "sell"

    async def test_circuit_breaker_blocks_order(self):
        agent, state = _build_agent()
        await state.trigger_circuit_breaker(cooldown_seconds=9999)
        result = await agent._evaluate(_make_signal(action="BUY", confidence=0.90))
        assert result is None

    async def test_max_drawdown_triggers_circuit_breaker(self):
        agent, state = _build_agent(initial_capital=10_000.0)
        # Simulate large drawdown: equity dropped to 84% of peak
        state.equity = 8_400.0
        state.peak_equity = 10_000.0
        result = await agent._evaluate(_make_signal(action="BUY", confidence=0.90))
        assert result is None
        assert state.circuit_breaker_active

    async def test_max_open_positions_blocks_second_order(self):
        from core.state_manager import Position
        agent, state = _build_agent()
        pos = Position(
            symbol="BTC/USDT", side="long", entry_price=30000.0,
            quantity=0.01, stop_loss=29000.0, take_profit=33000.0,
            opened_at=1_700_000_000_000, order_id="TEST-001",
        )
        await state.add_position(pos)
        result = await agent._evaluate(_make_signal(action="BUY", confidence=0.90))
        assert result is None

    async def test_consecutive_loss_limit_triggers_circuit_breaker(self):
        agent, state = _build_agent()
        state.consecutive_losses = 5  # at the limit
        result = await agent._evaluate(_make_signal(action="BUY", confidence=0.90))
        assert result is None
        assert state.circuit_breaker_active

    async def test_stop_loss_below_entry_for_buy(self):
        agent, _ = _build_agent()
        result = await agent._evaluate(_make_signal(action="BUY", confidence=0.85))
        assert result is not None
        assert result.stop_loss < result.entry_price

    async def test_take_profit_above_entry_for_buy(self):
        agent, _ = _build_agent()
        result = await agent._evaluate(_make_signal(action="BUY", confidence=0.85))
        assert result is not None
        assert result.take_profit > result.entry_price

    async def test_stop_loss_above_entry_for_sell(self):
        agent, _ = _build_agent()
        result = await agent._evaluate(_make_signal(action="SELL", confidence=0.85))
        assert result is not None
        assert result.stop_loss > result.entry_price

    async def test_quantity_is_positive(self):
        agent, _ = _build_agent()
        result = await agent._evaluate(_make_signal(action="BUY", confidence=0.85))
        assert result is not None
        assert result.quantity > 0.0

    async def test_order_type_is_market(self):
        agent, _ = _build_agent()
        result = await agent._evaluate(_make_signal(action="BUY", confidence=0.85))
        assert result is not None
        assert result.order_type == "market"


# ─── StateManager ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestStateManager:
    async def test_initial_state(self):
        state = StateManager(10_000.0)
        assert state.equity == 10_000.0
        assert state.drawdown == 0.0
        assert state.total_trades == 0
        assert state.win_rate == 0.0

    async def test_equity_update_tracks_peak(self):
        state = StateManager(10_000.0)
        await state.update_equity(12_000.0)
        assert state.peak_equity == 12_000.0
        await state.update_equity(11_000.0)
        assert state.peak_equity == 12_000.0
        assert abs(state.drawdown - 1 / 12) < 1e-6

    async def test_close_position_updates_wins(self):
        from core.state_manager import Position
        state = StateManager(10_000.0)
        pos = Position(
            symbol="BTC/USDT", side="long", entry_price=30000.0,
            quantity=0.01, stop_loss=29000.0, take_profit=33000.0,
            opened_at=0, order_id="X001",
        )
        await state.add_position(pos)
        record = await state.close_position("X001", 31000.0, "tp", pnl=10.0)
        assert record is not None
        assert state.winning_trades == 1
        assert state.consecutive_losses == 0

    async def test_close_position_updates_losses(self):
        from core.state_manager import Position
        state = StateManager(10_000.0)
        pos = Position(
            symbol="BTC/USDT", side="long", entry_price=30000.0,
            quantity=0.01, stop_loss=29000.0, take_profit=33000.0,
            opened_at=0, order_id="X002",
        )
        await state.add_position(pos)
        await state.close_position("X002", 29500.0, "sl", pnl=-5.0)
        assert state.consecutive_losses == 1

    async def test_circuit_breaker_clears_after_cooldown(self):
        import time
        state = StateManager(10_000.0)
        await state.trigger_circuit_breaker(cooldown_seconds=0)
        # Immediately after cooldown expires it should clear
        await asyncio.sleep(0.01)
        active = await state.check_circuit_breaker()
        assert not active

    async def test_summary_keys(self):
        state = StateManager(10_000.0)
        s = state.summary()
        expected_keys = {
            "equity", "initial_capital", "pnl_total", "pnl_pct",
            "drawdown_pct", "total_trades", "win_rate",
            "consecutive_losses", "open_positions", "circuit_breaker",
        }
        assert expected_keys.issubset(s.keys())
