"""Position tracking for real-time P&L.

Tracks open positions with:
- Cost basis
- Unrealized P&L
- Realized P&L
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from ..core.utils import get_logger
from ..storage.database import Database

if TYPE_CHECKING:
    from ..adapters.base import BaseAdapter

logger = get_logger(__name__)


@dataclass
class Position:
    """Single position in a market.

    Attributes:
        market_id: Market identifier.
        platform: Platform name.
        side: 'YES' or 'NO'.
        size: Number of shares held.
        cost_basis: Average entry price.
        current_price: Current market price.
        unrealized_pnl: Unrealized profit/loss.
        realized_pnl: Realized profit/loss (from closes).
        opened_at: When position was opened.
        last_updated: Last price update time.
    """

    market_id: str
    platform: str
    side: str  # 'YES' or 'NO'
    size: float = 0.0
    cost_basis: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def notional_value(self) -> float:
        """Current position value (size * current_price)."""
        return self.size * self.current_price

    @property
    def total_cost(self) -> float:
        """Total cost of position (size * cost_basis)."""
        return self.size * self.cost_basis

    @property
    def total_pnl(self) -> float:
        """Total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl

    @property
    def pnl_pct(self) -> float:
        """P&L as percentage of cost."""
        if self.total_cost == 0:
            return 0.0
        return self.total_pnl / self.total_cost

    def update_price(self, price: float) -> None:
        """Update current price and recalculate unrealized P&L.

        Args:
            price: New market price.
        """
        self.current_price = price
        self.unrealized_pnl = (price - self.cost_basis) * self.size
        self.last_updated = datetime.now(timezone.utc)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "market_id": self.market_id,
            "platform": self.platform,
            "side": self.side,
            "size": self.size,
            "cost_basis": self.cost_basis,
            "current_price": self.current_price,
            "notional_value": self.notional_value,
            "total_cost": self.total_cost,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "total_pnl": self.total_pnl,
            "pnl_pct": self.pnl_pct,
            "opened_at": self.opened_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
        }


class PositionTracker:
    """Tracks all open positions with real-time P&L.

    Usage:
        tracker = PositionTracker(adapter)

        # Record a fill
        tracker.record_fill(
            market_id="token_123",
            side="YES",
            price=0.55,
            size=100,
        )

        # Update prices
        await tracker.update_prices()

        # Get position
        pos = tracker.get_position("token_123", "YES")
        print(f"P&L: ${pos.total_pnl:.2f}")

        # Close position
        tracker.record_fill(
            market_id="token_123",
            side="YES",
            price=0.60,
            size=-100,  # Negative for sell/close
        )
    """

    def __init__(
        self,
        adapter: "BaseAdapter | None" = None,
        database: Database | None = None,
    ):
        """Initialize position tracker.

        Args:
            adapter: Platform adapter for price updates.
            database: Optional database for persistence.
        """
        self.adapter = adapter
        self.db = database

        self._positions: dict[str, Position] = {}  # key: f"{market_id}:{side}"
        self._total_realized_pnl: float = 0.0

    def _position_key(self, market_id: str, side: str) -> str:
        """Generate position key."""
        return f"{market_id}:{side}"

    def record_fill(
        self,
        market_id: str,
        side: str,
        price: float,
        size: float,
        platform: str = "polymarket",
    ) -> Position:
        """Record a fill (buy or sell).

        Args:
            market_id: Market identifier.
            side: 'YES' or 'NO'.
            price: Fill price.
            size: Number of shares (positive=buy, negative=sell).
            platform: Platform name.

        Returns:
            Updated Position.
        """
        key = self._position_key(market_id, side)
        position = self._positions.get(key)

        if position is None:
            # New position
            position = Position(
                market_id=market_id,
                platform=platform,
                side=side,
                size=0,
                cost_basis=0,
                current_price=price,
            )
            self._positions[key] = position

        if size > 0:
            # Buy - increase position, update cost basis
            new_size = position.size + size
            if new_size > 0:
                # Weighted average cost basis
                total_cost = (position.size * position.cost_basis) + (size * price)
                position.cost_basis = total_cost / new_size
            position.size = new_size

            logger.info(
                f"Position opened: {market_id} {side} +{size}@{price:.3f} "
                f"(total: {position.size})"
            )

        else:
            # Sell - reduce position, realize P&L
            sell_size = abs(size)
            if sell_size > position.size:
                sell_size = position.size

            # Calculate realized P&L
            realized = (price - position.cost_basis) * sell_size
            position.realized_pnl += realized
            self._total_realized_pnl += realized

            position.size -= sell_size

            logger.info(
                f"Position closed: {market_id} {side} -{sell_size}@{price:.3f} "
                f"(realized: ${realized:.2f}, remaining: {position.size})"
            )

        # Update current price
        position.current_price = price
        position.update_price(price)

        # Remove if fully closed
        if position.size == 0:
            del self._positions[key]

        return position

    def get_position(self, market_id: str, side: str) -> Position | None:
        """Get position for a market/side.

        Args:
            market_id: Market identifier.
            side: 'YES' or 'NO'.

        Returns:
            Position or None if no position.
        """
        key = self._position_key(market_id, side)
        return self._positions.get(key)

    def get_all_positions(self) -> list[Position]:
        """Get all open positions.

        Returns:
            List of all Position objects.
        """
        return list(self._positions.values())

    def get_positions_by_market(self, market_id: str) -> list[Position]:
        """Get positions for a specific market.

        Args:
            market_id: Market identifier.

        Returns:
            List of Position objects for this market.
        """
        return [
            pos for pos in self._positions.values()
            if pos.market_id == market_id
        ]

    async def update_prices(self) -> None:
        """Update current prices for all positions.

        Requires adapter to be set.
        """
        if not self.adapter:
            logger.warning("No adapter set for price updates")
            return

        for position in self._positions.values():
            try:
                yes_price, no_price = await self.adapter.get_price(position.market_id)
                price = yes_price if position.side == "YES" else no_price
                position.update_price(price)
            except Exception as e:
                logger.warning(
                    f"Failed to update price for {position.market_id}: {e}"
                )

    def get_total_unrealized_pnl(self) -> float:
        """Get total unrealized P&L across all positions."""
        return sum(pos.unrealized_pnl for pos in self._positions.values())

    def get_total_realized_pnl(self) -> float:
        """Get total realized P&L."""
        return self._total_realized_pnl

    def get_total_pnl(self) -> float:
        """Get total P&L (realized + unrealized)."""
        return self._total_realized_pnl + self.get_total_unrealized_pnl()

    def get_total_exposure(self) -> float:
        """Get total notional exposure."""
        return sum(pos.notional_value for pos in self._positions.values())

    def get_summary(self) -> dict:
        """Get position summary.

        Returns:
            Dictionary with aggregate position metrics.
        """
        positions = list(self._positions.values())

        return {
            "num_positions": len(positions),
            "total_exposure": self.get_total_exposure(),
            "total_unrealized_pnl": self.get_total_unrealized_pnl(),
            "total_realized_pnl": self._total_realized_pnl,
            "total_pnl": self.get_total_pnl(),
            "positions": [pos.to_dict() for pos in positions],
        }

    def record_resolution(
        self,
        market_id: str,
        outcome: str,
    ) -> float:
        """Record market resolution and realize P&L.

        Args:
            market_id: Market identifier.
            outcome: Resolution outcome ('YES' or 'NO').

        Returns:
            Realized P&L from this resolution.
        """
        realized = 0.0

        for side in ["YES", "NO"]:
            position = self.get_position(market_id, side)
            if not position:
                continue

            # If our side matches outcome, we get $1 per share
            # Otherwise we get $0
            if side == outcome:
                exit_price = 1.0
            else:
                exit_price = 0.0

            # Close the position
            pnl = self.record_fill(
                market_id=market_id,
                side=side,
                price=exit_price,
                size=-position.size,
                platform=position.platform,
            )
            realized += pnl.realized_pnl

        return realized
