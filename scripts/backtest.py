"""
scripts/backtest.py
────────────────────
Vectorised backtesting harness.

Replays historical OHLCV data through the feature pipeline and a trained
model, then simulates order execution with the risk management rules.
Produces a detailed trade log and equity curve.

Usage:
    python scripts/backtest.py \\
        --csv data/raw/btcusdt_1m_90d.csv \\
        --model data/models/xgboost_btcusdt.pkl \\
        --capital 10000 \\
        --confidence 0.60
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import click
import numpy as np
import pandas as pd

from config.trading_params import INDICATOR_PARAMS as P, RISK_PARAMS
from utils.features import build_stationary_features
from utils.indicators import (
    atr as atr_fn, ema_series, macd as macd_fn, momentum as momentum_fn,
    rsi, sma_series, volume_z_score, vwap as vwap_fn,
)


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class BacktestTrade:
    idx: int
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    entry_bar: int
    exit_bar: int
    exit_reason: str
    pnl: float
    pnl_pct: float


@dataclass
class BacktestResult:
    initial_capital: float
    final_equity: float
    trades: List[BacktestTrade] = field(default_factory=list)

    @property
    def total_return_pct(self) -> float:
        return (self.final_equity / self.initial_capital - 1) * 100

    @property
    def win_rate(self) -> float:
        wins = sum(1 for t in self.trades if t.pnl > 0)
        return wins / len(self.trades) * 100 if self.trades else 0.0

    @property
    def max_drawdown_pct(self) -> float:
        if not self.trades:
            return 0.0
        equity = self.initial_capital
        peak = equity
        max_dd = 0.0
        for t in self.trades:
            equity += t.pnl
            peak = max(peak, equity)
            dd = (peak - equity) / peak * 100
            max_dd = max(max_dd, dd)
        return max_dd

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss   = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        return gross_profit / gross_loss if gross_loss > 0 else float("inf")

    def summary(self) -> dict:
        return {
            "initial_capital":  self.initial_capital,
            "final_equity":     round(self.final_equity, 2),
            "total_return_pct": round(self.total_return_pct, 2),
            "total_trades":     len(self.trades),
            "win_rate_pct":     round(self.win_rate, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "profit_factor":    round(self.profit_factor, 3),
        }


# ─── Feature builder (mirrors train_model.py) ────────────────────────────────

def _build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    # Compute stationary features
    df = build_stationary_features(df)
    
    # Compute atr for risk management/exit logic in backtesting
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    
    # Fully vectorized ATR calculation to avoid slow rolling apply on millions of rows
    tr = np.zeros(len(c))
    tr[0] = h[0] - l[0]
    tr[1:] = np.maximum(
        h[1:] - l[1:],
        np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1]))
    )
    df["atr"] = pd.Series(tr).rolling(window=P.atr_period).mean()
    return df


# ─── Backtester ───────────────────────────────────────────────────────────────

def run_backtest(
    df: pd.DataFrame,
    model,
    label_map: dict,
    feature_cols: List[str],
    initial_capital: float,
    confidence_threshold: float,
    max_position_pct: float,
    stop_loss_atr_mult: float,
    take_profit_atr_mult: float,
    max_open_positions: int = 100,
) -> BacktestResult:
    df = df.copy()
    df = _build_feature_matrix(df)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Filter to only the feature columns the model knows about
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].values

    probas = model.predict_proba(X)
    n_classes = probas.shape[1]
    inv_label = {v: k for k, v in label_map.items()}

    buy_idx  = inv_label.get("BUY")
    sell_idx = inv_label.get("SELL")

    equity = initial_capital
    peak_equity = initial_capital
    result = BacktestResult(initial_capital=initial_capital, final_equity=initial_capital)

    open_positions: List[BacktestTrade] = []

    closes = df["close"].values
    atrs = df["atr"].values

    for i in range(len(df)):
        close = closes[i]

        # ── Check open positions for SL/TP ──────────────────────────────────
        active_positions = []
        for position in open_positions:
            exit_reason = None
            if i - position.entry_bar >= 15:
                exit_reason = "timeout"
            elif position.side == "buy":
                if close <= position.stop_loss:
                    exit_reason = "sl"
                elif close >= position.take_profit:
                    exit_reason = "tp"
            else:
                if close >= position.stop_loss:
                    exit_reason = "sl"
                elif close <= position.take_profit:
                    exit_reason = "tp"

            if exit_reason:
                fee = (position.entry_price + close) * position.quantity * 0.0005
                pnl = (close - position.entry_price) * position.quantity - fee
                if position.side == "sell":
                    pnl = (position.entry_price - close) * position.quantity - fee
                position.exit_price = close
                position.exit_bar = i
                position.exit_reason = exit_reason
                position.pnl = pnl
                position.pnl_pct = pnl / (position.entry_price * position.quantity) * 100
                equity += pnl
                result.trades.append(position)
            else:
                active_positions.append(position)
        
        open_positions = active_positions

        # ── Entry logic ────────────────────────────────────────────────────
        proba = probas[i]
        pred_class = int(np.argmax(proba))
        confidence = float(proba[pred_class])
        action = label_map.get(pred_class, "HOLD")

        if len(open_positions) < max_open_positions and confidence >= confidence_threshold and action in ("BUY", "SELL"):
            qty = (equity * max_position_pct) / close
            atr = atrs[i]
            if action == "BUY":
                sl = close - (stop_loss_atr_mult * atr)
                tp = close + (take_profit_atr_mult * atr)
            else:
                sl = close + (stop_loss_atr_mult * atr)
                tp = close - (take_profit_atr_mult * atr)

            new_position = BacktestTrade(
                idx=i, side=action.lower(),
                entry_price=close, exit_price=close,
                quantity=qty, stop_loss=sl, take_profit=tp,
                entry_bar=i, exit_bar=i,
                exit_reason="open", pnl=0.0, pnl_pct=0.0,
            )
            open_positions.append(new_position)

        peak_equity = max(peak_equity, equity)

    # Close any open positions at end of data
    last_close = df["close"].iloc[-1]
    for position in open_positions:
        fee = (position.entry_price + last_close) * position.quantity * 0.0005
        pnl = (last_close - position.entry_price) * position.quantity - fee
        if position.side == "sell":
            pnl = (position.entry_price - last_close) * position.quantity - fee
        position.exit_price = last_close
        position.exit_bar = len(df) - 1
        position.exit_reason = "end_of_data"
        position.pnl = pnl
        equity += pnl
        result.trades.append(position)

    result.final_equity = equity
    return result


# ─── CLI ─────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--csv", "csv_path", required=True)
@click.option("--model", "model_path", required=True)
@click.option("--capital", default=10_000.0, type=float, show_default=True)
@click.option("--confidence", default=0.75, type=float, show_default=True)
@click.option("--max-pos-pct", default=0.10, type=float, show_default=True)
@click.option("--sl-mult", default=2.0, type=float, show_default=True)
@click.option("--tp-mult", default=3.0, type=float, show_default=True)
@click.option("--max-open-positions", default=100, type=int, show_default=True)
@click.option("--trade-log", default=None, help="Path to save trade log CSV")
def main(
    csv_path: str,
    model_path: str,
    capital: float,
    confidence: float,
    max_pos_pct: float,
    sl_mult: float,
    tp_mult: float,
    max_open_positions: int,
    trade_log: str | None,
) -> None:
    print(f"Loading data from {csv_path} …")
    # Peek to detect delimiter and headers
    with open(csv_path, "r") as f:
        first_line = f.readline()
    
    delim = "|" if "|" in first_line else ","
    has_header = any(col in first_line.lower() for col in ["timestamp", "open", "close", "log_return"])
    
    if has_header:
        df = pd.read_csv(csv_path, sep=delim)
        df.columns = [c.lower() for c in df.columns]
    else:
        col_names = [
            "timestamp", "open", "high", "low", "close", "volume",
            "taker_buy_quote_volume", "taker_buy_base_volume", "quote_volume", "trades"
        ]
        df = pd.read_csv(csv_path, sep=delim, header=None)
        num_cols = df.shape[1]
        if num_cols > len(col_names):
            col_names.extend([f"col_{i}" for i in range(len(col_names), num_cols)])
        elif num_cols < len(col_names):
            col_names = col_names[:num_cols]
        df.columns = col_names

    print(f"Loading model from {model_path} …")
    with open(model_path, "rb") as f:
        artefact = pickle.load(f)
    model = artefact["model"]
    label_map = artefact.get("label_map", {0: "BUY", 1: "HOLD", 2: "SELL"})
    feature_cols = artefact.get("feature_names", [])

    print(f"Running backtest on {len(df):,} bars …")
    result = run_backtest(
        df, model, label_map, feature_cols,
        capital, confidence, max_pos_pct, sl_mult, tp_mult,
        max_open_positions,
    )

    print("\n─── Backtest Results ─────────────────────────────────")
    for k, v in result.summary().items():
        print(f"  {k:<25} {v}")
    print("──────────────────────────────────────────────────────")

    if trade_log:
        trades_df = pd.DataFrame([vars(t) for t in result.trades])
        trades_df.to_csv(trade_log, index=False)
        print(f"\nTrade log saved → {trade_log}")


if __name__ == "__main__":
    main()
