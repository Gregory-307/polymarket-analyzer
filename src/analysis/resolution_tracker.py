"""Resolution tracker for market outcome recording.

Tracks when markets resolve (close with YES/NO outcome) for:
- Validating calibration of price predictions
- Computing actual P&L on historical positions
- Building training data for edge estimation
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import TYPE_CHECKING

from ..core.utils import get_logger
from ..storage.database import Database

if TYPE_CHECKING:
    from ..adapters.polymarket import PolymarketAdapter
    from ..adapters.kalshi import KalshiAdapter

logger = get_logger(__name__)


@dataclass
class Resolution:
    """Market resolution record.

    Attributes:
        market_id: Market identifier.
        platform: Platform name.
        question: Market question.
        outcome: Resolved outcome ('YES', 'NO', or specific outcome name).
        final_price: Last price before resolution.
        resolved_at: Resolution timestamp.
    """

    market_id: str
    platform: str
    question: str
    outcome: str
    final_price: float | None
    resolved_at: datetime


class ResolutionTracker:
    """Tracks market resolutions by polling for closed markets.

    Polymarket markets resolve when:
    - Market becomes inactive (active=false, closed=true)
    - Final prices settle to 0 or 1

    Usage:
        adapter = PolymarketAdapter()
        await adapter.connect()

        async with Database() as db:
            tracker = ResolutionTracker(db, adapter)
            resolutions = await tracker.check_for_resolutions()
            await tracker.record_resolutions(resolutions)
    """

    def __init__(
        self,
        database: Database,
        adapter: "PolymarketAdapter | KalshiAdapter",
        check_interval: int = 3600,  # 1 hour default
    ):
        """Initialize resolution tracker.

        Args:
            database: Database instance for storing resolutions.
            adapter: Platform adapter for fetching market data.
            check_interval: Seconds between resolution checks.
        """
        self.db = database
        self.adapter = adapter
        self.check_interval = check_interval
        self._running = False
        self._task: asyncio.Task | None = None

    async def check_for_resolutions(self) -> list[Resolution]:
        """Check for newly resolved markets.

        Fetches markets that have resolved since last check
        and returns their resolution data.

        Returns:
            List of Resolution objects for newly resolved markets.
        """
        resolutions = []

        # Get markets we're tracking that might have resolved
        tracked_markets = await self.db.get_active_markets(
            platform=self.adapter.platform
        )

        if not tracked_markets:
            return []

        # Fetch current market states from API
        for market_data in tracked_markets:
            market_id = market_data["id"]

            try:
                current_market = await self.adapter.get_market(market_id)

                if current_market is None:
                    # Market no longer exists - might be resolved
                    logger.debug(f"Market {market_id} no longer accessible")
                    continue

                # Check if market has resolved
                resolution = self._check_market_resolved(current_market, market_data)
                if resolution:
                    resolutions.append(resolution)
                    logger.info(
                        f"Market resolved: {resolution.question[:50]} -> {resolution.outcome}"
                    )

            except Exception as e:
                logger.warning(f"Failed to check market {market_id}: {e}")

        return resolutions

    def _check_market_resolved(
        self,
        current_market,
        stored_data: dict,
    ) -> Resolution | None:
        """Check if a market has resolved.

        Args:
            current_market: Current market data from API.
            stored_data: Stored market data from database.

        Returns:
            Resolution object if market resolved, None otherwise.
        """
        # Market is not active anymore
        if not current_market.is_active:
            # Determine outcome from final prices
            yes_price = current_market.yes_price
            no_price = current_market.no_price

            # Prices near 0 or 1 indicate resolution
            if yes_price >= 0.99 or no_price <= 0.01:
                outcome = "YES"
                final_price = stored_data.get("yes_price", yes_price)
            elif no_price >= 0.99 or yes_price <= 0.01:
                outcome = "NO"
                final_price = stored_data.get("yes_price", yes_price)
            else:
                # Market closed but not clearly resolved
                logger.debug(
                    f"Market {current_market.id} closed but unclear outcome "
                    f"(YES={yes_price:.2f}, NO={no_price:.2f})"
                )
                return None

            return Resolution(
                market_id=current_market.id,
                platform=current_market.platform,
                question=current_market.question,
                outcome=outcome,
                final_price=final_price,
                resolved_at=datetime.now(timezone.utc),
            )

        return None

    async def record_resolutions(self, resolutions: list[Resolution]) -> int:
        """Record resolutions in the database.

        Args:
            resolutions: List of Resolution objects to record.

        Returns:
            Number of resolutions recorded.
        """
        recorded = 0

        for resolution in resolutions:
            try:
                await self.db.insert_resolution(
                    market_id=resolution.market_id,
                    platform=resolution.platform,
                    question=resolution.question,
                    outcome=resolution.outcome,
                    final_price=resolution.final_price,
                    resolved_at=resolution.resolved_at,
                )

                # Mark market as inactive in database
                market = await self.db.get_market(resolution.market_id)
                if market:
                    await self.db.upsert_market(
                        id=resolution.market_id,
                        platform=resolution.platform,
                        question=resolution.question,
                        is_active=False,
                        **{k: v for k, v in market.items() if k not in [
                            "id", "platform", "question", "is_active",
                            "created_at", "updated_at"
                        ]},
                    )

                recorded += 1

            except Exception as e:
                logger.error(f"Failed to record resolution {resolution.market_id}: {e}")

        return recorded

    async def start(self) -> None:
        """Start continuous resolution tracking."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(f"Resolution tracker started (interval: {self.check_interval}s)")

    async def stop(self) -> None:
        """Stop resolution tracking."""
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

        logger.info("Resolution tracker stopped")

    async def _run_loop(self) -> None:
        """Main tracking loop."""
        while self._running:
            try:
                resolutions = await self.check_for_resolutions()
                if resolutions:
                    await self.record_resolutions(resolutions)
                    logger.info(f"Recorded {len(resolutions)} resolutions")
            except Exception as e:
                logger.error(f"Resolution check failed: {e}")

            await asyncio.sleep(self.check_interval)

    async def get_resolution_stats(self) -> dict:
        """Get resolution statistics.

        Returns:
            Dictionary with resolution counts and rates.
        """
        resolutions = await self.db.get_resolutions(
            platform=self.adapter.platform,
            limit=10000,
        )

        if not resolutions:
            return {
                "total_resolutions": 0,
                "yes_count": 0,
                "no_count": 0,
                "yes_rate": 0.0,
            }

        yes_count = sum(1 for r in resolutions if r["outcome"] == "YES")
        no_count = sum(1 for r in resolutions if r["outcome"] == "NO")
        total = yes_count + no_count

        return {
            "total_resolutions": len(resolutions),
            "yes_count": yes_count,
            "no_count": no_count,
            "yes_rate": yes_count / total if total > 0 else 0.0,
            "earliest_resolution": resolutions[-1]["resolved_at"] if resolutions else None,
            "latest_resolution": resolutions[0]["resolved_at"] if resolutions else None,
        }


class ResolutionMatcher:
    """Matches opportunities with their resolutions for P&L calculation.

    Links detected opportunities to market resolutions to compute
    realized P&L and validate edge estimates.
    """

    def __init__(self, database: Database):
        """Initialize matcher.

        Args:
            database: Database instance.
        """
        self.db = database

    async def match_opportunities(self) -> list[dict]:
        """Match opportunities with their resolutions.

        Returns:
            List of matched opportunity-resolution pairs with P&L.
        """
        # Get all opportunities
        opportunities = await self.db.get_opportunities(limit=10000)

        # Get all resolutions
        resolutions = await self.db.get_resolutions(limit=10000)
        resolution_map = {r["market_id"]: r for r in resolutions}

        matched = []

        for opp in opportunities:
            market_id = opp["market_id"]

            if market_id in resolution_map:
                resolution = resolution_map[market_id]
                pnl = self._calculate_pnl(opp, resolution)

                matched.append({
                    "opportunity": opp,
                    "resolution": resolution,
                    "pnl": pnl,
                })

        return matched

    def _calculate_pnl(self, opportunity: dict, resolution: dict) -> dict:
        """Calculate P&L for an opportunity.

        Args:
            opportunity: Opportunity data from database.
            resolution: Resolution data from database.

        Returns:
            Dictionary with P&L calculations.
        """
        import json

        # Parse opportunity details
        details = {}
        if opportunity.get("details"):
            try:
                details = json.loads(opportunity["details"])
            except (json.JSONDecodeError, TypeError):
                pass

        price = details.get("price", 0.5)
        side = opportunity.get("action", "").replace("buy_", "").upper()
        outcome = resolution.get("outcome", "")

        # Calculate P&L
        # If we bought YES and outcome is YES: profit = 1 - price
        # If we bought YES and outcome is NO: loss = -price
        if side == "YES":
            if outcome == "YES":
                pnl_pct = (1.0 - price) / price if price > 0 else 0
                won = True
            else:
                pnl_pct = -1.0
                won = False
        elif side == "NO":
            if outcome == "NO":
                pnl_pct = (1.0 - (1 - price)) / (1 - price) if price < 1 else 0
                won = True
            else:
                pnl_pct = -1.0
                won = False
        else:
            # Unknown side
            pnl_pct = 0
            won = None

        return {
            "entry_price": price,
            "side": side,
            "outcome": outcome,
            "pnl_pct": pnl_pct,
            "won": won,
        }

    async def get_pnl_summary(self) -> dict:
        """Get P&L summary across all matched opportunities.

        Returns:
            Dictionary with aggregate P&L metrics.
        """
        matched = await self.match_opportunities()

        if not matched:
            return {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "total_pnl_pct": 0.0,
                "avg_pnl_pct": 0.0,
            }

        wins = sum(1 for m in matched if m["pnl"]["won"] is True)
        losses = sum(1 for m in matched if m["pnl"]["won"] is False)
        total_pnl = sum(m["pnl"]["pnl_pct"] for m in matched if m["pnl"]["won"] is not None)

        return {
            "total_trades": len(matched),
            "wins": wins,
            "losses": losses,
            "win_rate": wins / (wins + losses) if (wins + losses) > 0 else 0.0,
            "total_pnl_pct": total_pnl,
            "avg_pnl_pct": total_pnl / len(matched) if matched else 0.0,
        }
