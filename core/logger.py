"""
core/logger.py
──────────────
Structured JSON logger built on structlog + Python logging.
Writes JSON-formatted logs to both stdout (coloured in dev) and a rotating
file. Import get_logger() anywhere in the codebase.
"""
from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

import structlog

from config.settings import get_settings

_configured = False


def _configure_logging() -> None:
    global _configured
    if _configured:
        return
    _configured = True

    settings = get_settings()
    level = getattr(logging, settings.log_level, logging.INFO)

    # ── shared processors ─────────────────────────────────────────────────────
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
    ]

    # ── file handler (JSON) ───────────────────────────────────────────────────
    log_file = settings.log_dir / "trader.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(level)

    # ── stdout handler (pretty in TTY, JSON otherwise) ────────────────────────
    if sys.stdout.isatty():
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(level)
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    else:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(level)
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )

    for handler in (file_handler, stdout_handler):
        handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(file_handler)
    root.addHandler(stdout_handler)

    # Suppress noisy third-party loggers
    for noisy in ("ccxt", "asyncio", "websockets", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str = "trader") -> structlog.BoundLogger:
    _configure_logging()
    return structlog.get_logger(name)
