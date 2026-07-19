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
    confidence: float = 0.0
    tp_order_id: Optional[str] = None
    sl_order_id: Optional[str] = None


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

    def __init__(self, initial_capital: float, csv_path: Optional[str] = None, fee_rate: float = 0.0002) -> None:
        self._lock = asyncio.Lock()

        # ── Capital ───────────────────────────────────────────────────────────
        self.initial_capital: float = initial_capital
        self.equity: float = initial_capital
        self.peak_equity: float = initial_capital
        self.fee_rate: float = fee_rate

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
        import sys
        if csv_path is not None:
            self.csv_path = csv_path
        elif "pytest" in sys.modules or "unittest" in sys.modules:
            self.csv_path = None
        else:
            self.csv_path = "data/trades_log.csv"

        if self.csv_path:
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
            commission = position.entry_price * position.quantity * 0.0002
            pnl_minus_comm = 0.0 - commission
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
                commission=commission,
                confidence=position.confidence,
                pnl_minus_comm=pnl_minus_comm,
            )
            logger.info(
                "state.position_opened",
                order_id=position.order_id,
                symbol=position.symbol,
                side=position.side,
                qty=position.quantity,
                entry=position.entry_price,
            )

    async def log_cancelled_entry(
        self,
        order_id: str,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        opened_at: int,
        confidence: float,
        reason: str = "unfilled_timeout"
    ) -> None:
        async with self._lock:
            closed_time_ms = int(time.time() * 1000)
            self._write_csv_row(
                event="CANCELLED",
                order_id=order_id,
                symbol=symbol,
                side=side.upper(),
                qty=qty,
                price=price,
                pnl_usd=0.0,
                pnl_pct=0.0,
                opened_at=datetime.fromtimestamp(opened_at / 1000).isoformat(),
                closed_at=datetime.fromtimestamp(closed_time_ms / 1000).isoformat(),
                exit_reason=reason,
                commission=0.0,
                confidence=confidence,
                pnl_minus_comm=0.0,
            )
            logger.info(
                "state.entry_cancelled",
                order_id=order_id,
                symbol=symbol,
                side=side,
                qty=qty,
                price=price,
                reason=reason,
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

            if pos.side == "long":
                gross_pnl = (exit_price - pos.entry_price) * pos.quantity
            else:
                gross_pnl = (pos.entry_price - exit_price) * pos.quantity

            pnl_pct = (gross_pnl / (pos.entry_price * pos.quantity)) * 100 if (pos.entry_price * pos.quantity) != 0 else 0.0
            commission = exit_price * pos.quantity * 0.0004
            pnl_minus_comm = gross_pnl - commission
            self._write_csv_row(
                event="EXIT",
                order_id=order_id,
                symbol=pos.symbol,
                side="SELL" if pos.side == "long" else "BUY",
                qty=pos.quantity,
                price=exit_price,
                pnl_usd=gross_pnl,
                pnl_pct=pnl_pct,
                opened_at=datetime.fromtimestamp(pos.opened_at / 1000).isoformat(),
                closed_at=datetime.fromtimestamp(closed_time_ms / 1000).isoformat(),
                exit_reason=exit_reason,
                commission=commission,
                confidence=pos.confidence,
                pnl_minus_comm=pnl_minus_comm,
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
        if not self.csv_path:
            return
        expected_headers = [
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
            "Total_Balance",
            "Profit_Loss",
            "Commission",
            "Confidence",
            "PnL_USD_Minus_Commission"
        ]
        try:
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
                with open(self.csv_path, mode="w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(expected_headers)
                return

            with open(self.csv_path, mode="r", newline="") as f:
                rows = list(csv.reader(f))

            if not rows:
                with open(self.csv_path, mode="w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(expected_headers)
                return

            first_row = rows[0]
            # Check if the first row contains "Timestamp" (case-insensitive check is safer)
            has_header = any(h.strip().lower() == "timestamp" for h in first_row if h)

            if not has_header:
                # File has no headers at all! All rows are data rows.
                new_rows = [expected_headers]
                for row in rows:
                    if not row or (len(row) == 1 and not row[0].strip()):
                        continue
                    # Pad rows that are shorter than the expected headers
                    while len(row) < len(expected_headers):
                        current_len = len(row)
                        if current_len == 13: # Profit_Loss index
                            event = row[1] if len(row) > 1 else ""
                            pnl_usd_str = row[7] if len(row) > 7 else ""
                            if event == "EXIT":
                                try:
                                    pnl_val = float(pnl_usd_str)
                                    if pnl_val > 0:
                                        row.append("PROFIT")
                                    elif pnl_val < 0:
                                        row.append("LOSS")
                                    else:
                                        row.append("BREAKEVEN")
                                except Exception:
                                    row.append("N/A")
                            else:
                                row.append("N/A")
                        elif current_len == 14: # Commission index
                            try:
                                qty = float(row[5])
                                price = float(row[6])
                                event = row[1] if len(row) > 1 else ""
                                rate = 0.0002 if event == "ENTRY" else 0.0004
                                comm_val = price * qty * rate
                                row.append(str(round(comm_val, 6)))
                            except Exception:
                                row.append("0.0")
                        elif current_len == 15: # Confidence index
                            row.append("0.0")
                        elif current_len == 16: # PnL_USD_Minus_Commission index
                            try:
                                pnl_usd = float(row[7])
                                commission = float(row[14])
                                row.append(str(round(pnl_usd - commission, 6)))
                            except Exception:
                                row.append("0.0")
                        else:
                            row.append("")
                    new_rows.append(row[:len(expected_headers)])

                with open(self.csv_path, mode="w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(new_rows)
                logger.info("state.csv_header_added", path=self.csv_path)
            else:
                # File has headers. Check if any expected headers are missing.
                header_lower = [h.strip().lower() for h in first_row]
                missing_profit_loss = "profit_loss" not in header_lower
                missing_commission = "commission" not in header_lower
                missing_confidence = "confidence" not in header_lower
                missing_pnl_minus_comm = "pnl_usd_minus_commission" not in header_lower

                if missing_profit_loss or missing_commission or missing_confidence or missing_pnl_minus_comm:
                    new_rows = []
                    # Create new headers
                    updated_headers = list(first_row)
                    if missing_profit_loss:
                        updated_headers.append("Profit_Loss")
                    if missing_commission:
                        updated_headers.append("Commission")
                    if missing_confidence:
                        updated_headers.append("Confidence")
                    if missing_pnl_minus_comm:
                        updated_headers.append("PnL_USD_Minus_Commission")
                    new_rows.append(updated_headers)

                    event_idx = -1
                    pnl_idx = -1
                    qty_idx = -1
                    price_idx = -1
                    for idx, h in enumerate(first_row):
                        h_clean = h.strip().lower()
                        if h_clean == "event":
                            event_idx = idx
                        elif h_clean == "pnl_usd":
                            pnl_idx = idx
                        elif h_clean == "quantity":
                            qty_idx = idx
                        elif h_clean == "price":
                            price_idx = idx

                    for row in rows[1:]:
                        if not row or (len(row) == 1 and not row[0].strip()):
                            continue
                        
                        # Pad the row if it's shorter than the original headers
                        while len(row) < len(first_row):
                            row.append("")

                        if missing_profit_loss:
                            try:
                                event = row[event_idx] if event_idx != -1 and len(row) > event_idx else ""
                                if event == "EXIT":
                                    pnl_val = float(row[pnl_idx]) if pnl_idx != -1 and len(row) > pnl_idx else 0.0
                                    if pnl_val > 0:
                                        row.append("PROFIT")
                                    elif pnl_val < 0:
                                        row.append("LOSS")
                                    else:
                                        row.append("BREAKEVEN")
                                else:
                                    row.append("N/A")
                            except Exception:
                                row.append("N/A")

                        if missing_commission:
                            try:
                                qty = float(row[qty_idx]) if qty_idx != -1 and len(row) > qty_idx else 0.0
                                price = float(row[price_idx]) if price_idx != -1 and len(row) > price_idx else 0.0
                                event = row[event_idx] if event_idx != -1 and len(row) > event_idx else ""
                                rate = 0.0002 if event == "ENTRY" else 0.0004
                                comm_val = price * qty * rate
                                row.append(str(round(comm_val, 6)))
                            except Exception:
                                row.append("0.0")

                        if missing_confidence:
                            row.append("0.0")

                        if missing_pnl_minus_comm:
                            try:
                                pnl_val = float(row[updated_headers.index("PnL_USD")])
                                comm_val = float(row[updated_headers.index("Commission")])
                                row.append(str(round(pnl_val - comm_val, 6)))
                            except Exception:
                                row.append("0.0")

                        new_rows.append(row[:len(updated_headers)])

                    with open(self.csv_path, mode="w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerows(new_rows)
                    logger.info("state.csv_migrated_successfully", path=self.csv_path)
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
        commission: float,
        confidence: float,
        pnl_minus_comm: float,
    ) -> None:
        if not self.csv_path:
            return
        try:
            if event == "EXIT":
                if pnl_usd > 0:
                    profit_loss = "PROFIT"
                elif pnl_usd < 0:
                    profit_loss = "LOSS"
                else:
                    profit_loss = "BREAKEVEN"
            else:
                profit_loss = "N/A"

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
                    profit_loss,
                    round(commission, 6),
                    round(confidence, 4),
                    round(pnl_minus_comm, 6),
                ])
        except Exception as e:
            logger.error("state.csv_write_failed", error=str(e))

    # ── Circuit breaker ───────────────────────────────────────────────────────

    async def trigger_circuit_breaker(self, cooldown_seconds: int = 300) -> None:
        async with self._lock:
            self.circuit_breaker_active = True
            self.circuit_breaker_until = time.time() + cooldown_seconds
            self.consecutive_losses = 0
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
