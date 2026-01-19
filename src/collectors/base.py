"""Base collector class for data ingestion.

Provides common functionality for all collectors:
- Configurable collection intervals
- Start/stop control
- Error handling and retry logic
- Statistics tracking
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Any

from ..core.utils import get_logger
from ..storage.database import Database

logger = get_logger(__name__)


class CollectorInterval(Enum):
    """Standard collection intervals."""

    SECONDS_10 = 10
    SECONDS_30 = 30
    MINUTE_1 = 60
    MINUTE_5 = 300
    MINUTE_15 = 900
    HOUR_1 = 3600


@dataclass
class CollectorConfig:
    """Configuration for a collector.

    Attributes:
        interval: Collection interval in seconds.
        enabled: Whether the collector is active.
        max_retries: Maximum retry attempts on failure.
        retry_delay: Delay between retries in seconds.
        batch_size: Number of items to process per cycle.
    """

    interval: int = 300  # 5 minutes default
    enabled: bool = True
    max_retries: int = 3
    retry_delay: float = 5.0
    batch_size: int = 100


@dataclass
class CollectorStats:
    """Statistics for a collector.

    Attributes:
        collections: Total number of collection cycles.
        items_collected: Total items collected.
        errors: Total errors encountered.
        last_collection: Timestamp of last successful collection.
        last_error: Last error message, if any.
    """

    collections: int = 0
    items_collected: int = 0
    errors: int = 0
    last_collection: datetime | None = None
    last_error: str | None = None
    started_at: datetime | None = None

    def record_success(self, items: int) -> None:
        """Record a successful collection."""
        self.collections += 1
        self.items_collected += items
        self.last_collection = datetime.now(timezone.utc)
        self.last_error = None

    def record_error(self, error: str) -> None:
        """Record a collection error."""
        self.errors += 1
        self.last_error = error

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "collections": self.collections,
            "items_collected": self.items_collected,
            "errors": self.errors,
            "last_collection": self.last_collection.isoformat() if self.last_collection else None,
            "last_error": self.last_error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "uptime_seconds": (
                (datetime.now(timezone.utc) - self.started_at).total_seconds()
                if self.started_at else 0
            ),
        }


class BaseCollector(ABC):
    """Abstract base class for data collectors.

    Subclasses must implement:
    - collect(): Perform one collection cycle
    - name: Property returning collector name

    Usage:
        collector = MarketCollector(db, adapter, config)
        await collector.start()
        # ... runs continuously ...
        await collector.stop()
    """

    def __init__(
        self,
        database: Database,
        config: CollectorConfig | None = None,
    ):
        """Initialize collector.

        Args:
            database: Database instance for storing collected data.
            config: Collector configuration.
        """
        self.db = database
        self.config = config or CollectorConfig()
        self.stats = CollectorStats()
        self._running = False
        self._task: asyncio.Task | None = None
        self._on_collect_callbacks: list[Callable[[int], Any]] = []

    @property
    @abstractmethod
    def name(self) -> str:
        """Collector name for logging and identification."""
        pass

    @abstractmethod
    async def collect(self) -> int:
        """Perform one collection cycle.

        Returns:
            Number of items collected.

        Raises:
            Exception: On collection failure.
        """
        pass

    def on_collect(self, callback: Callable[[int], Any]) -> None:
        """Register callback for successful collections.

        Args:
            callback: Function called with item count after each collection.
        """
        self._on_collect_callbacks.append(callback)

    async def start(self) -> None:
        """Start the collector."""
        if self._running:
            logger.warning(f"{self.name} collector already running")
            return

        if not self.config.enabled:
            logger.info(f"{self.name} collector is disabled")
            return

        self._running = True
        self.stats.started_at = datetime.now(timezone.utc)
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            f"{self.name} collector started "
            f"(interval: {self.config.interval}s)"
        )

    async def stop(self) -> None:
        """Stop the collector."""
        if not self._running:
            return

        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        logger.info(
            f"{self.name} collector stopped "
            f"(collected: {self.stats.items_collected} items)"
        )

    async def collect_once(self) -> int:
        """Run a single collection cycle with retries.

        Returns:
            Number of items collected.
        """
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                items = await self.collect()
                self.stats.record_success(items)

                # Notify callbacks
                for callback in self._on_collect_callbacks:
                    try:
                        result = callback(items)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as e:
                        logger.error(f"Callback error: {e}")

                logger.debug(f"{self.name} collected {items} items")
                return items

            except Exception as e:
                last_error = str(e)
                self.stats.record_error(last_error)

                if attempt < self.config.max_retries - 1:
                    logger.warning(
                        f"{self.name} collection failed (attempt {attempt + 1}): {e}. "
                        f"Retrying in {self.config.retry_delay}s..."
                    )
                    await asyncio.sleep(self.config.retry_delay)
                else:
                    logger.error(
                        f"{self.name} collection failed after {self.config.max_retries} attempts: {e}"
                    )

        return 0

    async def _run_loop(self) -> None:
        """Main collection loop."""
        while self._running:
            await self.collect_once()
            await asyncio.sleep(self.config.interval)

    @property
    def is_running(self) -> bool:
        """Check if collector is running."""
        return self._running

    def get_stats(self) -> dict:
        """Get collector statistics."""
        return {
            "name": self.name,
            "running": self._running,
            "config": {
                "interval": self.config.interval,
                "enabled": self.config.enabled,
                "batch_size": self.config.batch_size,
            },
            **self.stats.to_dict(),
        }
