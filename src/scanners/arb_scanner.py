"""Real-time arbitrage opportunity scanner.

Continuously monitors markets across platforms for arbitrage opportunities
and favorite-longshot bias edges.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..adapters import KalshiAdapter, PolymarketAdapter
from ..adapters.base import Market
from ..core.config import Config, Credentials
from ..core.utils import get_logger, utc_now
from ..strategies import (
    FavoriteLongshotStrategy,
    MultiOutcomeArbitrage,
    SingleConditionArbitrage,
)

logger = get_logger(__name__)


@dataclass
class ScanResult:
    """Result of a market scan.

    Attributes:
        timestamp: When the scan was performed.
        single_arb_count: Number of single-condition arb opportunities.
        multi_arb_count: Number of multi-outcome arb opportunities.
        favorite_longshot_count: Number of bias opportunities.
        opportunities: All opportunities found.
        markets_scanned: Total markets scanned.
        errors: Any errors encountered.
    """

    timestamp: datetime = field(default_factory=utc_now)
    single_arb_count: int = 0
    multi_arb_count: int = 0
    favorite_longshot_count: int = 0
    opportunities: list[Any] = field(default_factory=list)
    markets_scanned: int = 0
    errors: list[str] = field(default_factory=list)


class ArbitrageScanner:
    """Continuous scanner for prediction market opportunities.

    Monitors both Polymarket and Kalshi for:
    - Single-condition arbitrage (YES + NO != $1)
    - Multi-outcome bundle arbitrage
    - Favorite-longshot bias opportunities

    Example:
        ```python
        scanner = ArbitrageScanner(config)

        # One-time scan
        result = await scanner.scan_once()

        # Continuous scanning
        async for result in scanner.scan_continuous(interval=5):
            print(f"Found {len(result.opportunities)} opportunities")
        ```
    """

    def __init__(
        self,
        config: Config,
        credentials: Credentials | None = None,
    ):
        """Initialize scanner.

        Args:
            config: Application configuration.
            credentials: Optional credentials for authenticated operations.
        """
        self.config = config
        self.credentials = credentials

        # Initialize adapters
        self.polymarket = PolymarketAdapter(
            config=config.polymarket,
            credentials=credentials,
        )
        self.kalshi = KalshiAdapter(
            config=config.kalshi,
            credentials=credentials,
        )

        # Initialize strategies
        self.single_arb = SingleConditionArbitrage(
            min_profit_pct=config.single_arb.min_profit_usd / 100,
        )
        self.multi_arb = MultiOutcomeArbitrage(
            min_profit_pct=config.multi_arb.min_profit_usd / 100,
        )
        self.favorite_longshot = FavoriteLongshotStrategy(
            min_probability=config.favorite_longshot.min_probability,
        )

        self._connected = False

    async def connect(self) -> bool:
        """Connect to both platforms.

        Returns:
            True if at least one platform connected.
        """
        poly_ok = await self.polymarket.connect()
        kalshi_ok = await self.kalshi.connect()

        self._connected = poly_ok or kalshi_ok

        if self._connected:
            logger.info(
                "scanner_connected",
                polymarket=poly_ok,
                kalshi=kalshi_ok,
            )
        else:
            logger.error("scanner_connection_failed")

        return self._connected

    async def disconnect(self) -> None:
        """Disconnect from platforms."""
        await self.polymarket.disconnect()
        await self.kalshi.disconnect()
        self._connected = False

    async def fetch_markets(
        self,
        limit_per_platform: int = 100,
    ) -> tuple[list[Market], list[str]]:
        """Fetch markets from both platforms.

        Args:
            limit_per_platform: Max markets per platform.

        Returns:
            Tuple of (markets, errors).
        """
        markets: list[Market] = []
        errors: list[str] = []

        # Fetch from Polymarket
        if self.config.polymarket.enabled:
            try:
                poly_markets = await self.polymarket.get_markets(
                    active_only=True,
                    limit=limit_per_platform,
                )
                markets.extend(poly_markets)
                logger.debug("polymarket_markets_fetched", count=len(poly_markets))
            except Exception as e:
                errors.append(f"Polymarket: {e}")
                logger.error("polymarket_fetch_error", error=str(e))

        # Fetch from Kalshi
        if self.config.kalshi.enabled:
            try:
                kalshi_markets = await self.kalshi.get_markets(
                    active_only=True,
                    limit=limit_per_platform,
                )
                markets.extend(kalshi_markets)
                logger.debug("kalshi_markets_fetched", count=len(kalshi_markets))
            except Exception as e:
                errors.append(f"Kalshi: {e}")
                logger.error("kalshi_fetch_error", error=str(e))

        return markets, errors

    async def scan_once(
        self,
        strategies: list[str] | None = None,
    ) -> ScanResult:
        """Perform a single scan of all markets.

        Args:
            strategies: List of strategies to run, or None for all.
                Options: 'single_arb', 'multi_arb', 'favorite_longshot'

        Returns:
            ScanResult with all opportunities found.
        """
        if not self._connected:
            await self.connect()

        result = ScanResult()
        strategies = strategies or ["single_arb", "multi_arb", "favorite_longshot"]

        # Fetch markets
        markets, errors = await self.fetch_markets(
            limit_per_platform=self.config.scanning.max_markets_per_scan,
        )
        result.markets_scanned = len(markets)
        result.errors = errors

        # Run strategies
        if "single_arb" in strategies and self.config.single_arb.enabled:
            opps = self.single_arb.scan(markets)
            result.single_arb_count = len(opps)
            result.opportunities.extend(opps)

        if "favorite_longshot" in strategies and self.config.favorite_longshot.enabled:
            opps = self.favorite_longshot.scan(markets)
            result.favorite_longshot_count = len(opps)
            result.opportunities.extend(opps)

        # Multi-arb requires grouped market data (handled separately)
        # This is a simplified version
        if "multi_arb" in strategies and self.config.multi_arb.enabled:
            result.multi_arb_count = 0  # Requires additional API calls

        logger.info(
            "scan_complete",
            markets=result.markets_scanned,
            single_arb=result.single_arb_count,
            favorite_longshot=result.favorite_longshot_count,
            total=len(result.opportunities),
        )

        return result

    async def scan_continuous(
        self,
        interval: int = 5,
        max_iterations: int | None = None,
    ):
        """Continuously scan for opportunities.

        Args:
            interval: Seconds between scans.
            max_iterations: Maximum number of scans, or None for infinite.

        Yields:
            ScanResult for each scan.
        """
        if not self._connected:
            await self.connect()

        iterations = 0

        while True:
            if max_iterations and iterations >= max_iterations:
                break

            result = await self.scan_once()
            yield result

            iterations += 1
            await asyncio.sleep(interval)

    def format_opportunities(self, result: ScanResult) -> str:
        """Format scan results for display.

        Args:
            result: Scan result to format.

        Returns:
            Formatted string.
        """
        lines = [
            f"=== Scan Results ({result.timestamp.isoformat()}) ===",
            f"Markets Scanned: {result.markets_scanned}",
            f"Single-Condition Arb: {result.single_arb_count}",
            f"Multi-Outcome Arb: {result.multi_arb_count}",
            f"Favorite-Longshot: {result.favorite_longshot_count}",
            "",
        ]

        for opp in result.opportunities:
            if hasattr(opp, "market"):
                lines.append(f"  [{opp.market.platform}] {opp.market.question[:50]}")
                if hasattr(opp, "profit_pct"):
                    lines.append(f"    Profit: {opp.profit_pct:.2%}")
                if hasattr(opp, "price"):
                    lines.append(f"    Price: {opp.price:.2%}")
                if hasattr(opp, "edge"):
                    lines.append(f"    Edge: {opp.edge:.2%}")

        if result.errors:
            lines.append("")
            lines.append("Errors:")
            for error in result.errors:
                lines.append(f"  - {error}")

        return "\n".join(lines)
