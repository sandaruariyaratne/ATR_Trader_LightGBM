"""
core/pipeline.py
────────────────
Async pipeline that wires all agents together and manages their lifecycles.

Each agent runs as an independent asyncio Task. The pipeline starts them all,
monitors for failures, and performs a clean shutdown on SIGINT / SIGTERM.
"""
from __future__ import annotations

import asyncio
import signal
from typing import List

from core.event_bus import EventBus
from core.logger import get_logger
from core.state_manager import StateManager
from config.settings import get_settings

logger = get_logger("pipeline")


class Pipeline:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.bus = EventBus(maxsize=self.settings.queue_max_size)
        clean_symbol = self.settings.trading_symbol.replace("/", "_").replace(":", "_")
        csv_path = f"data/trades_log_{clean_symbol}_42F.csv"
        self.state = StateManager(
            initial_capital=self.settings.initial_capital,
            csv_path=csv_path,
            fee_rate=self.settings.fee_rate
        )
        self._tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Instantiate agents, wire them to the bus, and run until shutdown."""
        from agents.market_data_agent import MarketDataAgent
        from agents.feature_engineering_agent import FeatureEngineeringAgent
        from agents.decision_agent import DecisionAgent
        from agents.risk_management_agent import RiskManagementAgent
        from agents.execution_agent import ExecutionAgent

        logger.info("pipeline.starting", symbol=self.settings.trading_symbol)

        agents = [
            MarketDataAgent(self.bus, self.state, self.settings),
            FeatureEngineeringAgent(self.bus, self.state, self.settings),
            DecisionAgent(self.bus, self.state, self.settings),
            RiskManagementAgent(self.bus, self.state, self.settings),
            ExecutionAgent(self.bus, self.state, self.settings),
        ]

        self._tasks = [
            asyncio.create_task(agent.run(), name=type(agent).__name__)
            for agent in agents
        ]

        # Stats reporter (every 60 s)
        self._tasks.append(
            asyncio.create_task(self._stats_reporter(), name="StatsReporter")
        )

        # Install signal handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._request_shutdown)

        logger.info("pipeline.running", agents=[type(a).__name__ for a in agents])

        # Wait for shutdown signal or any task to fail
        done, pending = await asyncio.wait(
            self._tasks, return_when=asyncio.FIRST_EXCEPTION
        )

        for task in done:
            if task.exception():
                logger.error(
                    "pipeline.agent_failed",
                    agent=task.get_name(),
                    error=str(task.exception()),
                )

        await self._shutdown()

    def _request_shutdown(self) -> None:
        logger.info("pipeline.shutdown_requested")
        self._shutdown_event.set()
        for task in self._tasks:
            task.cancel()

    async def _shutdown(self) -> None:
        logger.info("pipeline.shutting_down")
        for task in self._tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        logger.info("pipeline.stopped", summary=self.state.summary())

    # ── Monitoring ────────────────────────────────────────────────────────────

    async def _stats_reporter(self) -> None:
        while not self._shutdown_event.is_set():
            await asyncio.sleep(60)
            logger.info(
                "pipeline.stats",
                **self.state.summary(),
                queue_depth=self.bus.stats(),
            )
