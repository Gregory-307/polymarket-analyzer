"""Trade history collector for market activity analysis.

Collects trade data from markets for:
- Volume analysis
- Price impact estimation
- Order flow analysis
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
    from ..adapters.base import Trade

logger = get_logger(__name__)


class TradeCollector(BaseCollector):
    """Collects trade history for specified markets.

    Fetches recent trades from the API and stores them in the database,
    handling deduplication automatically.

    Usage:
        adapter = PolymarketAdapter()
        await adapter.connect()

        async with Database() as db:
            collector = TradeCollector(
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
        trades_per_market: int = 50,
    ):
        """Initialize trade collector.

        Args:
            database: Database instance for storing trades.
            adapter: Platform adapter for fetching trade data.
            market_ids: List of market/token IDs to track.
                       If None, tracks top markets by volume.
            config: Collector configuration.
            trades_per_market: Number of recent trades to fetch per market.
        """
        # Default to 5-minute collection for trades
        if config is None:
            config = CollectorConfig(
                interval=CollectorInterval.MINUTE_5.value,
                batch_size=30,
            )
        super().__init__(database, config)

        self.adapter = adapter
        self.market_ids = market_ids or []
        self.trades_per_market = trades_per_market
        self._seen_trade_ids: set[str] = set()  # Track seen trades for dedup

    @property
    def name(self) -> str:
        return f"trades_{self.adapter.platform}"

    def set_market_ids(self, market_ids: list[str]) -> None:
        """Update the list of markets to track.

        Args:
            market_ids: New list of market/token IDs.
        """
        self.market_ids = market_ids
        logger.info(f"{self.name}: tracking {len(market_ids)} markets")

    async def collect(self) -> int:
        """Collect trade history.

        Fetches recent trades for each tracked market and stores
        new trades in the database.

        Returns:
            Number of new trades collected.
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

        new_trades = 0

        for market_id in self.market_ids:
            if not market_id:
                continue

            try:
                trades = await self.adapter.get_trades(
                    market_id,
                    limit=self.trades_per_market,
                )
                new_trades += await self._store_trades(trades)
            except Exception as e:
                logger.warning(
                    f"{self.name}: Failed to collect trades for {market_id}: {e}"
                )

        logger.debug(f"{self.name}: collected {new_trades} new trades")
        return new_trades

    async def _store_trades(self, trades: list["Trade"]) -> int:
        """Store trades in the database, skipping duplicates.

        Args:
            trades: List of Trade objects from adapter.

        Returns:
            Number of new trades stored.
        """
        new_count = 0

        for trade in trades:
            # Skip if we've seen this trade ID before
            if trade.id in self._seen_trade_ids:
                continue

            # Insert into database (will ignore if duplicate)
            await self.db.insert_trade(
                id=trade.id,
                market_id=trade.market_id,
                platform=trade.platform,
                side=trade.side.value,
                price=trade.price,
                size=trade.size,
                timestamp=trade.timestamp,
                fee=trade.fee,
            )

            self._seen_trade_ids.add(trade.id)
            new_count += 1

        # Prevent memory growth - keep last N trade IDs
        max_seen = 10000
        if len(self._seen_trade_ids) > max_seen:
            # Convert to list, keep last half, convert back
            seen_list = list(self._seen_trade_ids)
            self._seen_trade_ids = set(seen_list[-(max_seen // 2):])

        return new_count


class TradeAggregator:
    """Aggregates trade data into time-series metrics.

    Computes:
    - VWAP (Volume-Weighted Average Price)
    - Buy/sell volume imbalance
    - Trade count and size distributions
    """

    def __init__(self, database: Database):
        """Initialize aggregator.

        Args:
            database: Database instance.
        """
        self.db = database

    async def get_vwap(
        self,
        market_id: str,
        since: datetime | None = None,
    ) -> float | None:
        """Calculate VWAP for a market.

        Args:
            market_id: Market ID.
            since: Calculate since this time (default: all trades).

        Returns:
            VWAP or None if no trades.
        """
        trades = await self.db.get_trades(market_id, since=since, limit=1000)

        if not trades:
            return None

        total_value = sum(t["price"] * t["size"] for t in trades)
        total_size = sum(t["size"] for t in trades)

        return total_value / total_size if total_size > 0 else None

    async def get_volume_imbalance(
        self,
        market_id: str,
        since: datetime | None = None,
    ) -> float | None:
        """Calculate buy/sell volume imbalance.

        Args:
            market_id: Market ID.
            since: Calculate since this time.

        Returns:
            Imbalance ratio: (buy_vol - sell_vol) / total_vol
            Positive = more buying, negative = more selling.
            Returns None if no trades.
        """
        trades = await self.db.get_trades(market_id, since=since, limit=1000)

        if not trades:
            return None

        buy_volume = sum(t["size"] for t in trades if t["side"] == "buy")
        sell_volume = sum(t["size"] for t in trades if t["side"] == "sell")
        total_volume = buy_volume + sell_volume

        if total_volume == 0:
            return None

        return (buy_volume - sell_volume) / total_volume

    async def get_trade_summary(
        self,
        market_id: str,
        since: datetime | None = None,
    ) -> dict:
        """Get summary statistics for trades.

        Args:
            market_id: Market ID.
            since: Calculate since this time.

        Returns:
            Dictionary with trade statistics.
        """
        trades = await self.db.get_trades(market_id, since=since, limit=1000)

        if not trades:
            return {
                "trade_count": 0,
                "total_volume": 0,
                "vwap": None,
                "imbalance": None,
                "avg_size": None,
            }

        buy_vol = sum(t["size"] for t in trades if t["side"] == "buy")
        sell_vol = sum(t["size"] for t in trades if t["side"] == "sell")
        total_vol = buy_vol + sell_vol
        total_value = sum(t["price"] * t["size"] for t in trades)

        return {
            "trade_count": len(trades),
            "total_volume": total_vol,
            "buy_volume": buy_vol,
            "sell_volume": sell_vol,
            "vwap": total_value / total_vol if total_vol > 0 else None,
            "imbalance": (buy_vol - sell_vol) / total_vol if total_vol > 0 else None,
            "avg_size": total_vol / len(trades) if trades else None,
        }
