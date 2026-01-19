"""Orderbook depth collector for active markets.

Collects orderbook snapshots at higher frequency for markets
with trading opportunities or high activity.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from .base import BaseCollector, CollectorConfig, CollectorInterval
from ..core.utils import get_logger
from ..storage.database import Database

if TYPE_CHECKING:
    from ..adapters.polymarket import PolymarketAdapter
    from ..adapters.kalshi import KalshiAdapter
    from ..adapters.base import OrderBook, Side

logger = get_logger(__name__)


class OrderbookCollector(BaseCollector):
    """Collects orderbook depth snapshots for specified markets.

    Tracks bid/ask depth at multiple price levels for:
    - Liquidity analysis
    - Execution quality estimation
    - Market microstructure research

    Usage:
        adapter = PolymarketAdapter()
        await adapter.connect()

        async with Database() as db:
            collector = OrderbookCollector(
                db, adapter,
                market_ids=["token_123", "token_456"],
            )
            await collector.start()
    """

    def __init__(
        self,
        database: Database,
        adapter: "PolymarketAdapter | KalshiAdapter",
        market_ids: list[str] | None = None,
        config: CollectorConfig | None = None,
        depth_levels: int = 10,
    ):
        """Initialize orderbook collector.

        Args:
            database: Database instance for storing snapshots.
            adapter: Platform adapter for fetching orderbook data.
            market_ids: List of market/token IDs to track.
                       If None, tracks top markets by volume.
            config: Collector configuration.
            depth_levels: Number of price levels to store (default: 10).
        """
        # Default to faster collection for orderbooks
        if config is None:
            config = CollectorConfig(
                interval=CollectorInterval.MINUTE_1.value,
                batch_size=50,
            )
        super().__init__(database, config)

        self.adapter = adapter
        self.market_ids = market_ids or []
        self.depth_levels = depth_levels

    @property
    def name(self) -> str:
        return f"orderbook_{self.adapter.platform}"

    def set_market_ids(self, market_ids: list[str]) -> None:
        """Update the list of markets to track.

        Args:
            market_ids: New list of market/token IDs.
        """
        self.market_ids = market_ids
        logger.info(f"{self.name}: tracking {len(market_ids)} markets")

    async def collect(self) -> int:
        """Collect orderbook snapshots.

        Fetches orderbook for each tracked market and stores
        depth metrics and top levels.

        Returns:
            Number of snapshots collected.
        """
        if not self.market_ids:
            # If no specific markets, get top markets by volume
            markets = await self.adapter.get_markets(
                active_only=True,
                limit=self.config.batch_size,
            )
            # Use YES token IDs from market data
            self.market_ids = [
                m.raw.get("clobTokenIds", [""])[0]
                for m in markets
                if m.raw.get("clobTokenIds")
            ][:self.config.batch_size]

        if not self.market_ids:
            logger.warning(f"{self.name}: No markets to track")
            return 0

        now = datetime.now(timezone.utc)
        collected = 0

        for market_id in self.market_ids:
            if not market_id:
                continue

            try:
                orderbook = await self.adapter.get_order_book(market_id)
                await self._store_snapshot(orderbook, now)
                collected += 1
            except Exception as e:
                logger.warning(
                    f"{self.name}: Failed to collect orderbook for {market_id}: {e}"
                )

        logger.debug(f"{self.name}: collected {collected} orderbook snapshots")
        return collected

    async def _store_snapshot(
        self,
        orderbook: "OrderBook",
        timestamp: datetime,
    ) -> None:
        """Store an orderbook snapshot in the database.

        Args:
            orderbook: Orderbook data from adapter.
            timestamp: Snapshot timestamp.
        """
        from ..adapters.base import Side

        # Calculate depth within 1% of best price
        bid_depth_1pct = orderbook.depth_at_level(0.01, Side.BUY) if orderbook.bids else 0
        ask_depth_1pct = orderbook.depth_at_level(0.01, Side.SELL) if orderbook.asks else 0

        # Extract top N levels for storage
        top_bids = [
            (level.price, level.size)
            for level in orderbook.bids[:self.depth_levels]
        ]
        top_asks = [
            (level.price, level.size)
            for level in orderbook.asks[:self.depth_levels]
        ]

        await self.db.insert_orderbook_snapshot(
            market_id=orderbook.market_id,
            platform=orderbook.platform,
            best_bid=orderbook.best_bid,
            best_ask=orderbook.best_ask,
            spread=orderbook.spread,
            bid_depth_1pct=bid_depth_1pct,
            ask_depth_1pct=ask_depth_1pct,
            bids=top_bids,
            asks=top_asks,
            timestamp=timestamp,
        )


class AdaptiveOrderbookCollector(OrderbookCollector):
    """Orderbook collector that adapts tracking based on opportunity detection.

    Automatically tracks markets where:
    - Arbitrage opportunities were detected
    - Spread is abnormally wide
    - Volume spike detected
    """

    def __init__(
        self,
        database: Database,
        adapter: "PolymarketAdapter | KalshiAdapter",
        config: CollectorConfig | None = None,
        max_tracked: int = 20,
    ):
        """Initialize adaptive collector.

        Args:
            database: Database instance.
            adapter: Platform adapter.
            config: Collector configuration.
            max_tracked: Maximum markets to track at once.
        """
        super().__init__(database, adapter, [], config)
        self.max_tracked = max_tracked
        self._priority_markets: dict[str, float] = {}  # market_id -> priority score

    def add_priority_market(self, market_id: str, priority: float = 1.0) -> None:
        """Add a market to priority tracking.

        Args:
            market_id: Market/token ID to track.
            priority: Priority score (higher = more important).
        """
        self._priority_markets[market_id] = priority
        self._update_tracked_markets()

    def remove_priority_market(self, market_id: str) -> None:
        """Remove a market from priority tracking.

        Args:
            market_id: Market ID to remove.
        """
        self._priority_markets.pop(market_id, None)
        self._update_tracked_markets()

    def _update_tracked_markets(self) -> None:
        """Update the list of tracked markets based on priority."""
        # Sort by priority and take top N
        sorted_markets = sorted(
            self._priority_markets.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        self.market_ids = [m[0] for m in sorted_markets[:self.max_tracked]]
