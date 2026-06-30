import pytest
from unittest.mock import MagicMock, AsyncMock
from core.event_bus import EventBus
from core.state_manager import StateManager, Position
from config.settings import Settings
from agents.execution_agent import ExecutionAgent

@pytest.mark.asyncio
async def test_close_position_calculates_fees():
    # Arrange
    settings = Settings(
        fee_rate=0.0010,  # 10 bps round-trip
        initial_capital=10000.0,
        paper_trading=True
    )
    bus = MagicMock()
    bus.publish = AsyncMock()
    state = StateManager(initial_capital=10000.0)
    exec_agent = ExecutionAgent(bus, state, settings)
    
    # Open a SHORT position: entry_price=100.0, qty=10
    pos = Position(
        symbol="SOL/USDT",
        side="short",
        entry_price=100.0,
        quantity=10.0,
        stop_loss=110.0,
        take_profit=90.0,
        opened_at=0,
        order_id="TEST-FEE",
    )
    await state.add_position(pos)
    
    # Act: close at 98.0 (TP)
    # Expected fee = (100.0 + 98.0) * 10.0 * 0.0005 = 0.99 USD
    # Expected raw PnL = (100.0 - 98.0) * 10.0 = 20.0 USD
    # Net PnL = 20.0 - 0.99 = 19.01 USD
    await exec_agent._close_position(pos, "TEST-FEE", 98.0, "tp")
    
    # Assert
    assert abs(state.equity - 10019.01) < 1e-6
    record = state.trade_history[-1]
    assert abs(record.pnl - 19.01) < 1e-6
