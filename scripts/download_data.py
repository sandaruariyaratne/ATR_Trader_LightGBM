"""
scripts/download_data.py
─────────────────────────
Download historical 1-minute OHLCV data from Binance and save as CSV.

Usage:
    python scripts/download_data.py --symbol BTC/USDT --days 90
    python scripts/download_data.py --symbol ETH/USDT --days 30 --output data/raw/eth.csv
"""
from __future__ import annotations

import asyncio
import csv
from datetime import datetime, timezone
from pathlib import Path

import click
import ccxt.pro as ccxtpro
from tqdm import tqdm


async def download(
    symbol: str,
    days: int,
    output: Path,
    interval: str = "1m",
) -> None:
    exchange = ccxtpro.binance({"enableRateLimit": True})

    try:
        interval_ms = 60_000
        total_bars = days * 24 * 60
        since = int(datetime.now(tz=timezone.utc).timestamp() * 1000) - days * 86_400_000

        print(f"Downloading {total_bars:,} bars of {symbol} {interval} from Binance …")

        all_candles = []
        batch_size = 1000
        pbar = tqdm(total=total_bars, unit="bars", ncols=72)

        while since < int(datetime.now(tz=timezone.utc).timestamp() * 1000):
            candles = await exchange.fetch_ohlcv(
                symbol, interval, since=since, limit=batch_size
            )
            if not candles:
                break
            all_candles.extend(candles)
            since = candles[-1][0] + interval_ms
            pbar.update(len(candles))
            await asyncio.sleep(exchange.rateLimit / 1000)

        pbar.close()

        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
            for c in all_candles:
                writer.writerow(c)

        print(f"Saved {len(all_candles):,} bars → {output}")

    finally:
        await exchange.close()


@click.command()
@click.option("--symbol", default="BTC/USDT", show_default=True)
@click.option("--days", default=90, show_default=True, type=int)
@click.option("--interval", default="1m", show_default=True)
@click.option("--output", default=None, help="Output CSV path (auto-named if omitted)")
def main(symbol: str, days: int, interval: str, output: str | None) -> None:
    if output is None:
        safe_sym = symbol.replace("/", "").lower()
        output_path = Path(f"data/raw/{safe_sym}_{interval}_{days}d.csv")
    else:
        output_path = Path(output)

    asyncio.run(download(symbol, days, output_path, interval))


if __name__ == "__main__":
    main()
