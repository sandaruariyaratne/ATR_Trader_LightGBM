"""
tests/test_state_manager_csv.py
───────────────────────────────
Unit tests for StateManager CSV initialization and migration logic.
"""
from __future__ import annotations

import csv
import os
import tempfile
import pytest

from core.state_manager import StateManager


@pytest.fixture
def temp_csv_path():
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


def test_init_csv_creates_new_file(temp_csv_path):
    # Remove it so it doesn't exist
    if os.path.exists(temp_csv_path):
        os.remove(temp_csv_path)

    # Initialize StateManager with custom path (override the pytest detection by passing it explicitly)
    state = StateManager(initial_capital=10000.0, csv_path=temp_csv_path)

    assert os.path.exists(temp_csv_path)
    with open(temp_csv_path, mode="r", newline="") as f:
        rows = list(csv.reader(f))

    assert len(rows) == 1
    assert rows[0][0] == "Timestamp"
    assert rows[0][-1] == "PnL_USD_Minus_Commission"
    assert rows[0][-2] == "Confidence"
    assert rows[0][-3] == "Commission"
    assert rows[0][-4] == "Profit_Loss"
    assert len(rows[0]) == 17


def test_init_csv_handles_headerless_file(temp_csv_path):
    # Write a headerless file (just 2 data rows, with 13 columns - missing Profit_Loss, Commission, Confidence, and PnL_USD_Minus_Commission)
    row1 = [
        "2026-07-13T13:42:03", "ENTRY", "ORDER1", "ETH/USDT", "SHORT",
        "1.0", "1700.0", "0.0", "0.0", "2026-07-13T13:42:00", "", "", "10000.0"
    ]
    row2 = [
        "2026-07-13T13:57:02", "EXIT", "ORDER1", "ETH/USDT", "BUY",
        "1.0", "1710.0", "-10.0", "-0.5", "2026-07-13T13:42:00", "2026-07-13T13:57:02", "timeout", "9990.0"
    ]
    with open(temp_csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row1)
        writer.writerow(row2)

    # StateManager should detect lack of headers, insert headers, and pad rows (including calculating columns)
    state = StateManager(initial_capital=10000.0, csv_path=temp_csv_path, fee_rate=0.0002)

    with open(temp_csv_path, mode="r", newline="") as f:
        rows = list(csv.reader(f))

    # Total 3 rows now: header + 2 data rows
    assert len(rows) == 3
    assert rows[0][0] == "Timestamp"
    assert rows[0][-1] == "PnL_USD_Minus_Commission"
    assert rows[0][-2] == "Confidence"
    assert rows[0][-3] == "Commission"
    assert rows[0][-4] == "Profit_Loss"

    # Verify first data row (ENTRY) - padded to 17 elements
    assert rows[1][2] == "ORDER1"
    assert len(rows[1]) == 17
    assert rows[1][-4] == "N/A"
    assert float(rows[1][-3]) == 0.34
    assert float(rows[1][-2]) == 0.0
    assert float(rows[1][-1]) == -0.34

    # Verify second data row (EXIT) - padded to 17 elements
    assert rows[2][2] == "ORDER1"
    assert len(rows[2]) == 17
    assert rows[2][-4] == "LOSS"
    assert float(rows[2][-3]) == 0.684
    assert float(rows[2][-2]) == 0.0
    assert float(rows[2][-1]) == -10.684


def test_init_csv_handles_missing_column(temp_csv_path):
    # File has 13 headers (missing columns) and 2 data rows (13 columns)
    headers = [
        "Timestamp", "Event", "OrderID", "Symbol", "Side", "Quantity",
        "Price", "PnL_USD", "PnL_Pct", "Opened_At", "Closed_At", "Exit_Reason", "Total_Balance"
    ]
    row1 = [
        "2026-07-13T13:42:03", "ENTRY", "ORDER1", "ETH/USDT", "SHORT",
        "1.0", "1700.0", "0.0", "0.0", "2026-07-13T13:42:00", "", "", "10000.0"
    ]
    row2 = [
        "2026-07-13T13:57:02", "EXIT", "ORDER1", "ETH/USDT", "BUY",
        "1.0", "1690.0", "10.0", "0.5", "2026-07-13T13:42:00", "2026-07-13T13:57:02", "tp", "10010.0"
    ]
    with open(temp_csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerow(row1)
        writer.writerow(row2)

    # StateManager should detect missing headers, append them, and migrate the rows
    state = StateManager(initial_capital=10000.0, csv_path=temp_csv_path, fee_rate=0.0002)

    with open(temp_csv_path, mode="r", newline="") as f:
        rows = list(csv.reader(f))

    assert len(rows) == 3
    assert rows[0][-1] == "PnL_USD_Minus_Commission"
    assert rows[0][-2] == "Confidence"
    assert rows[0][-3] == "Commission"
    assert rows[0][-4] == "Profit_Loss"
    assert rows[1][-4] == "N/A"
    assert rows[2][-4] == "PROFIT"  # pnl is 10.0 (positive)
    assert float(rows[1][-3]) == 0.34
    assert float(rows[2][-3]) == 0.676
    assert float(rows[1][-2]) == 0.0
    assert float(rows[2][-2]) == 0.0
    assert float(rows[1][-1]) == -0.34
    assert float(rows[2][-1]) == 9.324
