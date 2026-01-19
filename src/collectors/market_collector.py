"""Market data collector for scheduled price snapshots.

Collects market prices and metadata at configurable intervals,
storing them in the database for historical analysis.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from .base import BaseCollector, CollectorConfig
from ..core.utils import get_logger
from ..storage.database import Database

if TYPE_CHECKING:
    from ..adapters.polymarket import PolymarketAdapter
    from ..adapters.kalshi import KalshiAdapter

logger = get_logger(__name__)


class MarketCollector(BaseCollector):
    """Collects market price snapshots at scheduled intervals.

    Fetches current prices for all active markets and stores
    snapshots in the database for historical analysis.

    Usage:
        adapter = PolymarketAdapter()
        await adapter.connect()

        async with Database() as db:
            collector = MarketCollector(db, adapter)
            await collector.start()
            # ... runs continuously ...
            await collector.stop()
    """

    def __init__(
        self,
        database: Database,
        adapter: "PolymarketAdapter | KalshiAdapter",
        config: CollectorConfig | None = None,
        min_liquidity: float = 100.0,
        track_categories: list[str] | None = None,
    ):
        """Initialize market collector.

        Args:
            database: Database instance for storing snapshots.
            adapter: Platform adapter for fetching market data.
            config: Collector configuration.
            min_liquidity: Minimum liquidity to include market (default: $100).
            track_categories: Only track these categories (None = all).
        """
        super().__init__(database, config)
        self.adapter = adapter
        self.min_liquidity = min_liquidity
        self.track_categories = track_categories

    @property
    def name(self) -> str:
        return f"market_{self.adapter.platform}"

    async def collect(self) -> int:
        """Collect market snapshots.

        Fetches all active markets from the platform and stores
        a snapshot for each that meets the liquidity threshold.

        Returns:
            Number of snapshots collected.
        """
        # Fetch active markets
        markets = await self.adapter.get_markets(
            active_only=True,
            limit=self.config.batch_size,
        )

        if not markets:
            logger.warning(f"{self.name}: No markets fetched")
            return 0

        now = datetime.now(timezone.utc)
        collected = 0

        for market in markets:
            # Apply filters
            if (market.liquidity or 0) < self.min_liquidity:
                continue

            if self.track_categories:
                if market.category not in self.track_categories:
                    continue

            # Update market in database
            await self.db.upsert_market(
                id=market.id,
                platform=market.platform,
                question=market.question,
                description=market.description,
                category=market.category,
                yes_price=market.yes_price,
                no_price=market.no_price,
                volume=market.volume or 0,
                liquidity=market.liquidity or 0,
                end_date=market.end_date,
                outcomes=market.outcomes,
                is_active=market.is_active,
            )

            # Insert snapshot
            await self.db.insert_snapshot(
                market_id=market.id,
                platform=market.platform,
                yes_price=market.yes_price,
                no_price=market.no_price,
                volume=market.volume,
                liquidity=market.liquidity,
                timestamp=now,
            )

            collected += 1

        logger.info(
            f"{self.name}: collected {collected} snapshots "
            f"(from {len(markets)} markets)"
        )
        return collected


class MultiPlatformMarketCollector:
    """Coordinates market collection across multiple platforms.

    Usage:
        poly_adapter = PolymarketAdapter()
        kalshi_adapter = KalshiAdapter()
        await poly_adapter.connect()
        await kalshi_adapter.connect()

        async with Database() as db:
            collector = MultiPlatformMarketCollector(db, [poly_adapter, kalshi_adapter])
            await collector.start_all()
            # ... runs continuously ...
            await collector.stop_all()
    """

    def __init__(
        self,
        database: Database,
        adapters: list,
        config: CollectorConfig | None = None,
    ):
        """Initialize multi-platform collector.

        Args:
            database: Database instance.
            adapters: List of platform adapters.
            config: Shared collector configuration.
        """
        self.collectors = [
            MarketCollector(database, adapter, config)
            for adapter in adapters
        ]

    async def start_all(self) -> None:
        """Start all collectors."""
        for collector in self.collectors:
            await collector.start()

    async def stop_all(self) -> None:
        """Stop all collectors."""
        for collector in self.collectors:
            await collector.stop()

    async def collect_once_all(self) -> int:
        """Run single collection on all platforms.

        Returns:
            Total items collected across all platforms.
        """
        total = 0
        for collector in self.collectors:
            total += await collector.collect_once()
        return total

    def get_stats(self) -> list[dict]:
        """Get statistics for all collectors."""
        return [c.get_stats() for c in self.collectors]
