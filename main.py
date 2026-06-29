"""
main.py
───────
Entry point for the live trading system.

Loads configuration, validates credentials (in live mode), then starts
the async pipeline. Press Ctrl+C for a graceful shutdown.
"""
from __future__ import annotations

import asyncio
import sys

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()


def _print_banner(settings) -> None:
    mode = "[bold red]LIVE[/]" if not settings.paper_trading else "[bold green]PAPER[/]"
    text = Text.assemble(
        ("AlgoTrader\n", "bold cyan"),
        (f"Symbol   : {settings.trading_symbol}\n", "white"),
        (f"Interval : {settings.candle_interval}\n", "white"),
        (f"Model    : {settings.model_type} @ {settings.model_path}\n", "white"),
        (f"Capital  : ${settings.initial_capital:,.2f}\n", "white"),
        (f"Mode     : ", "white"), (f"{mode}\n", ""),
    )
    console.print(Panel(text, border_style="cyan"))


def _validate_settings(settings) -> bool:
    errors = []
    if not settings.paper_trading:
        if not settings.exchange_api_key:
            errors.append("EXCHANGE_API_KEY is not set")
        if not settings.exchange_api_secret:
            errors.append("EXCHANGE_API_SECRET is not set")

    if not settings.model_path.exists():
        errors.append(f"Model file not found: {settings.model_path}")

    for err in errors:
        console.print(f"[bold red]✗ {err}[/]")

    return len(errors) == 0


async def _main() -> None:
    from config.settings import get_settings
    from core.pipeline import Pipeline

    settings = get_settings()
    _print_banner(settings)

    if not _validate_settings(settings):
        console.print("\n[bold red]Startup aborted — fix the errors above.[/]")
        sys.exit(1)

    console.print("[bold green]Starting pipeline …[/]\n")
    pipeline = Pipeline()
    await pipeline.start()


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Shutdown requested.[/]")
