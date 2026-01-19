"""Order management for trade execution.

Handles order lifecycle:
- Submission to exchange
- Status tracking
- Partial fills
- Cancellation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable
import asyncio
import uuid

from ..core.utils import get_logger
from ..storage.database import Database

if TYPE_CHECKING:
    from ..adapters.base import BaseAdapter, Side, OrderType

logger = get_logger(__name__)


class OrderState(Enum):
    """Order lifecycle states."""

    PENDING = "pending"  # Created, not yet submitted
    SUBMITTED = "submitted"  # Sent to exchange
    OPEN = "open"  # Active on exchange
    PARTIAL = "partial"  # Partially filled
    FILLED = "filled"  # Fully filled
    CANCELLED = "cancelled"  # Cancelled by user
    REJECTED = "rejected"  # Rejected by exchange
    FAILED = "failed"  # Submission failed


@dataclass
class ManagedOrder:
    """Internal order tracking.

    Attributes:
        internal_id: Internal order ID.
        exchange_id: Exchange-assigned order ID.
        market_id: Market identifier.
        platform: Platform name.
        side: Buy or sell.
        price: Limit price.
        size: Order size (shares).
        filled_size: Amount filled.
        avg_fill_price: Average fill price.
        state: Current order state.
        created_at: Creation timestamp.
        updated_at: Last update timestamp.
        metadata: Additional tracking data.
    """

    internal_id: str
    exchange_id: str | None
    market_id: str
    platform: str
    side: str  # 'buy' or 'sell'
    price: float
    size: float
    filled_size: float = 0.0
    avg_fill_price: float = 0.0
    state: OrderState = OrderState.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def remaining_size(self) -> float:
        """Unfilled size."""
        return self.size - self.filled_size

    @property
    def is_complete(self) -> bool:
        """Whether order is in a terminal state."""
        return self.state in (
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.REJECTED,
            OrderState.FAILED,
        )

    @property
    def fill_pct(self) -> float:
        """Percentage filled."""
        return self.filled_size / self.size if self.size > 0 else 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "internal_id": self.internal_id,
            "exchange_id": self.exchange_id,
            "market_id": self.market_id,
            "platform": self.platform,
            "side": self.side,
            "price": self.price,
            "size": self.size,
            "filled_size": self.filled_size,
            "avg_fill_price": self.avg_fill_price,
            "state": self.state.value,
            "remaining_size": self.remaining_size,
            "fill_pct": self.fill_pct,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class OrderManager:
    """Manages order lifecycle and tracking.

    Usage:
        manager = OrderManager(adapter, db)

        # Submit order
        order = await manager.submit_order(
            market_id="token_123",
            side="buy",
            price=0.55,
            size=100,
        )

        # Check status
        status = manager.get_order(order.internal_id)

        # Cancel
        await manager.cancel_order(order.internal_id)
    """

    def __init__(
        self,
        adapter: "BaseAdapter",
        database: Database | None = None,
    ):
        """Initialize order manager.

        Args:
            adapter: Platform adapter for order submission.
            database: Optional database for persistence.
        """
        self.adapter = adapter
        self.db = database

        self._orders: dict[str, ManagedOrder] = {}
        self._exchange_to_internal: dict[str, str] = {}
        self._callbacks: list[Callable[[ManagedOrder], Any]] = []

    def on_order_update(self, callback: Callable[[ManagedOrder], Any]) -> None:
        """Register callback for order updates.

        Args:
            callback: Function called when order state changes.
        """
        self._callbacks.append(callback)

    async def submit_order(
        self,
        market_id: str,
        side: str,
        price: float,
        size: float,
        metadata: dict | None = None,
    ) -> ManagedOrder:
        """Submit a new order.

        Args:
            market_id: Market/token ID.
            side: 'buy' or 'sell'.
            price: Limit price.
            size: Number of shares.
            metadata: Additional tracking data.

        Returns:
            ManagedOrder with internal tracking ID.

        Raises:
            RuntimeError: If submission fails.
        """
        from ..adapters.base import Side, OrderType

        # Create internal order
        internal_id = str(uuid.uuid4())[:8]
        order = ManagedOrder(
            internal_id=internal_id,
            exchange_id=None,
            market_id=market_id,
            platform=self.adapter.platform,
            side=side,
            price=price,
            size=size,
            metadata=metadata or {},
        )

        self._orders[internal_id] = order
        logger.info(f"Order created: {internal_id} {side} {size}@{price}")

        # Submit to exchange
        try:
            exchange_side = Side.BUY if side == "buy" else Side.SELL
            exchange_order = await self.adapter.place_order(
                market_id=market_id,
                side=exchange_side,
                price=price,
                size=size,
                order_type=OrderType.LIMIT,
            )

            # Update with exchange ID
            order.exchange_id = exchange_order.id
            order.state = OrderState.SUBMITTED
            self._exchange_to_internal[exchange_order.id] = internal_id

            logger.info(f"Order submitted: {internal_id} -> {exchange_order.id}")
            await self._notify_update(order)

        except Exception as e:
            order.state = OrderState.FAILED
            order.metadata["error"] = str(e)
            logger.error(f"Order submission failed: {internal_id} - {e}")
            await self._notify_update(order)
            raise

        return order

    async def cancel_order(self, internal_id: str) -> bool:
        """Cancel an order.

        Args:
            internal_id: Internal order ID.

        Returns:
            True if cancellation was successful.
        """
        order = self._orders.get(internal_id)
        if not order:
            logger.warning(f"Order not found: {internal_id}")
            return False

        if order.is_complete:
            logger.warning(f"Cannot cancel completed order: {internal_id}")
            return False

        if not order.exchange_id:
            # Not yet submitted
            order.state = OrderState.CANCELLED
            await self._notify_update(order)
            return True

        try:
            success = await self.adapter.cancel_order(order.exchange_id)
            if success:
                order.state = OrderState.CANCELLED
                order.updated_at = datetime.now(timezone.utc)
                logger.info(f"Order cancelled: {internal_id}")
            else:
                logger.warning(f"Cancel failed for: {internal_id}")

            await self._notify_update(order)
            return success

        except Exception as e:
            logger.error(f"Cancel error for {internal_id}: {e}")
            return False

    async def cancel_all(self, market_id: str | None = None) -> int:
        """Cancel all open orders.

        Args:
            market_id: Optional filter by market.

        Returns:
            Number of orders cancelled.
        """
        cancelled = 0

        for order in list(self._orders.values()):
            if order.is_complete:
                continue
            if market_id and order.market_id != market_id:
                continue

            if await self.cancel_order(order.internal_id):
                cancelled += 1

        return cancelled

    def get_order(self, internal_id: str) -> ManagedOrder | None:
        """Get order by internal ID.

        Args:
            internal_id: Internal order ID.

        Returns:
            ManagedOrder or None if not found.
        """
        return self._orders.get(internal_id)

    def get_order_by_exchange_id(self, exchange_id: str) -> ManagedOrder | None:
        """Get order by exchange ID.

        Args:
            exchange_id: Exchange-assigned order ID.

        Returns:
            ManagedOrder or None if not found.
        """
        internal_id = self._exchange_to_internal.get(exchange_id)
        if internal_id:
            return self._orders.get(internal_id)
        return None

    def get_open_orders(self, market_id: str | None = None) -> list[ManagedOrder]:
        """Get all open (non-complete) orders.

        Args:
            market_id: Optional filter by market.

        Returns:
            List of open orders.
        """
        orders = []
        for order in self._orders.values():
            if order.is_complete:
                continue
            if market_id and order.market_id != market_id:
                continue
            orders.append(order)
        return orders

    async def update_order_status(
        self,
        internal_id: str,
        state: OrderState,
        filled_size: float | None = None,
        avg_fill_price: float | None = None,
    ) -> None:
        """Update order status (called by exchange events).

        Args:
            internal_id: Internal order ID.
            state: New state.
            filled_size: Updated fill size.
            avg_fill_price: Updated average fill price.
        """
        order = self._orders.get(internal_id)
        if not order:
            return

        order.state = state
        order.updated_at = datetime.now(timezone.utc)

        if filled_size is not None:
            order.filled_size = filled_size
        if avg_fill_price is not None:
            order.avg_fill_price = avg_fill_price

        await self._notify_update(order)

    async def _notify_update(self, order: ManagedOrder) -> None:
        """Notify callbacks of order update.

        Args:
            order: Updated order.
        """
        for callback in self._callbacks:
            try:
                result = callback(order)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Order callback error: {e}")

    def get_stats(self) -> dict:
        """Get order manager statistics.

        Returns:
            Dictionary with order counts by state.
        """
        stats = {state.value: 0 for state in OrderState}

        for order in self._orders.values():
            stats[order.state.value] += 1

        stats["total"] = len(self._orders)
        return stats
