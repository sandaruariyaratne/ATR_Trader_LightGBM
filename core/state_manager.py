"""
core/state_manager.py
─────────────────────
Thread-safe (asyncio-safe) shared system state.

Stores live trading state: equity curve, open positions, trade history,
consecutive losses, and circuit-breaker status. All mutations go through
async methods to prevent race conditions in the event loop.
"""
from __future__ import annotations

import asyncio
import time
import os
import csv
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from core.logger import get_logger

logger = get_logger("state_manager")


@dataclass
class Position:
    symbol: str
    side: str               # "long" | "short"
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    opened_at: int          # unix ms
    order_id: str


@dataclass
class TradeRecord:
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    opened_at: int
    closed_at: int
    exit_reason: str        # "tp" | "sl" | "signal" | "forced"


class StateManager:
    """Central mutable state container for the trading system."""

    def __init__(self, initial_capital: float) -> None:
        self._lock = asyncio.Lock()

        # ── Capital ───────────────────────────────────────────────────────────
        self.initial_capital: float = initial_capital
        self.equity: float = initial_capital
        self.peak_equity: float = initial_capital

        # ── Positions ─────────────────────────────────────────────────────────
        self.open_positions: Dict[str, Position] = {}   # order_id -> Position

        # ── Trade history ─────────────────────────────────────────────────────
        self.trade_history: List[TradeRecord] = []

        # ── Risk counters ─────────────────────────────────────────────────────
        self.consecutive_losses: int = 0
        self.total_trades: int = 0
        self.winning_trades: int = 0

        # ── Circuit breaker ───────────────────────────────────────────────────
        self.circuit_breaker_active: bool = False
        self.circuit_breaker_until: float = 0.0    # unix timestamp

        # ── Latest market feed ────────────────────────────────────────────────
        self.latest_close: float = 1.0
        self.latest_atr: float = 0.0

        # ── CSV Logging ───────────────────────────────────────────────────────
        self.csv_path: str = "data/trades_log.csv"
        self._init_csv()

        logger.info("state_manager.init", initial_capital=initial_capital)

    async def update_market_feed(self, close: float, atr: float) -> None:
        async with self._lock:
            self.latest_close = close
            self.latest_atr = atr

    # ── Equity ────────────────────────────────────────────────────────────────

    async def update_equity(self, new_equity: float) -> None:
        async with self._lock:
            self.equity = new_equity
            if new_equity > self.peak_equity:
                self.peak_equity = new_equity
            logger.debug("state.equity_updated", equity=new_equity)

    @property
    def drawdown(self) -> float:
        """Current drawdown from peak as a fraction [0, 1]."""
        if self.peak_equity == 0:
            return 0.0
        return (self.peak_equity - self.equity) / self.peak_equity

    # ── Positions ─────────────────────────────────────────────────────────────

    async def add_position(self, position: Position) -> None:
        async with self._lock:
            self.open_positions[position.order_id] = position
            self._write_csv_row(
                event="ENTRY",
                order_id=position.order_id,
                symbol=position.symbol,
                side=position.side.upper(),
                qty=position.quantity,
                price=position.entry_price,
                pnl_usd=0.0,
                pnl_pct=0.0,
                opened_at=datetime.fromtimestamp(position.opened_at / 1000).isoformat(),
                closed_at="",
                exit_reason="",
            )
            logger.info(
                "state.position_opened",
                order_id=position.order_id,
                symbol=position.symbol,
                side=position.side,
                qty=position.quantity,
                entry=position.entry_price,
            )

    async def close_position(
        self,
        order_id: str,
        exit_price: float,
        exit_reason: str,
        pnl: float,
    ) -> Optional[TradeRecord]:
        async with self._lock:
            pos = self.open_positions.pop(order_id, None)
            if pos is None:
                logger.warning("state.position_not_found", order_id=order_id)
                return None

            closed_time_ms = int(time.time() * 1000)
            record = TradeRecord(
                symbol=pos.symbol,
                side=pos.side,
                entry_price=pos.entry_price,
                exit_price=exit_price,
                quantity=pos.quantity,
                pnl=pnl,
                opened_at=pos.opened_at,
                closed_at=closed_time_ms,
                exit_reason=exit_reason,
            )
            self.trade_history.append(record)
            self.total_trades += 1

            if pnl > 0:
                self.winning_trades += 1
                self.consecutive_losses = 0
            else:
                self.consecutive_losses += 1

            self.equity += pnl

            pnl_pct = (pnl / (pos.entry_price * pos.quantity)) * 100 if (pos.entry_price * pos.quantity) != 0 else 0.0
            self._write_csv_row(
                event="EXIT",
                order_id=order_id,
                symbol=pos.symbol,
                side="SELL" if pos.side == "long" else "BUY",
                qty=pos.quantity,
                price=exit_price,
                pnl_usd=pnl,
                pnl_pct=pnl_pct,
                opened_at=datetime.fromtimestamp(pos.opened_at / 1000).isoformat(),
                closed_at=datetime.fromtimestamp(closed_time_ms / 1000).isoformat(),
                exit_reason=exit_reason,
            )

            logger.info(
                "state.position_closed",
                order_id=order_id,
                pnl=pnl,
                exit_reason=exit_reason,
                consecutive_losses=self.consecutive_losses,
            )
            return record

    def _init_csv(self) -> None:
        """Initialize the CSV log file with headers if it doesn't exist."""
        try:
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            if not os.path.exists(self.csv_path):
                with open(self.csv_path, mode="w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "Timestamp",
                        "Event",
                        "OrderID",
                        "Symbol",
                        "Side",
                        "Quantity",
                        "Price",
                        "PnL_USD",
                        "PnL_Pct",
                        "Opened_At",
                        "Closed_At",
                        "Exit_Reason",
                        "Total_Balance"
                    ])
        except Exception as e:
            logger.error("state.csv_init_failed", error=str(e))

    def _write_csv_row(
        self,
        event: str,
        order_id: str,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        pnl_usd: float,
        pnl_pct: float,
        opened_at: str,
        closed_at: str,
        exit_reason: str,
    ) -> None:
        try:
            with open(self.csv_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    event,
                    order_id,
                    symbol,
                    side,
                    round(qty, 6),
                    round(price, 4),
                    round(pnl_usd, 4),
                    round(pnl_pct, 4),
                    opened_at,
                    closed_at,
                    exit_reason,
                    round(self.equity, 2),
                ])
        except Exception as e:
            logger.error("state.csv_write_failed", error=str(e))

    # ── Circuit breaker ───────────────────────────────────────────────────────

    async def trigger_circuit_breaker(self, cooldown_seconds: int = 300) -> None:
        async with self._lock:
            self.circuit_breaker_active = True
            self.circuit_breaker_until = time.time() + cooldown_seconds
            logger.warning(
                "state.circuit_breaker_triggered",
                cooldown_seconds=cooldown_seconds,
                drawdown=round(self.drawdown, 4),
            )

    async def check_circuit_breaker(self) -> bool:
        """Return True if trading is currently halted."""
        async with self._lock:
            if self.circuit_breaker_active:
                if time.time() >= self.circuit_breaker_until:
                    self.circuit_breaker_active = False
                    logger.info("state.circuit_breaker_cleared")
                    return False
                return True
            return False

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    def summary(self) -> dict:
        return {
            "equity": round(self.equity, 2),
            "initial_capital": self.initial_capital,
            "pnl_total": round(self.equity - self.initial_capital, 2),
            "pnl_pct": round((self.equity / self.initial_capital - 1) * 100, 2),
            "drawdown_pct": round(self.drawdown * 100, 2),
            "total_trades": self.total_trades,
            "win_rate": round(self.win_rate * 100, 2),
            "consecutive_losses": self.consecutive_losses,
            "open_positions": len(self.open_positions),
            "circuit_breaker": self.circuit_breaker_active,
        }
