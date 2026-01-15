"""Base adapter class for prediction market platforms.

This module defines the abstract interface that all platform adapters must implement,
ensuring a unified API across Polymarket, Kalshi, and any future platforms.

Pattern adapted from: C:/dev/libs/exchange-adapters/base.py
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from ..core.utils import get_logger, utc_now

logger = get_logger(__name__)


class Side(str, Enum):
    """Order side."""

    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type."""

    LIMIT = "limit"
    MARKET = "market"


class OrderStatus(str, Enum):
    """Order status."""

    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Market:
    """Unified market representation across platforms.

    Attributes:
        id: Platform-specific market identifier.
        platform: Platform name ('polymarket' or 'kalshi').
        question: The market question/title.
        description: Detailed description of the market.
        category: Market category (e.g., 'politics', 'sports').
        yes_price: Current YES token price (0-1).
        no_price: Current NO token price (0-1).
        volume: Total trading volume in USD.
        liquidity: Available liquidity in USD.
        end_date: Market expiration/resolution date.
        outcomes: List of possible outcomes (for multi-outcome markets).
        is_active: Whether the market is currently trading.
        raw: Raw platform-specific data.
    """

    id: str
    platform: str
    question: str
    description: str = ""
    category: str = ""
    yes_price: float = 0.5
    no_price: float = 0.5
    volume: float = 0.0
    liquidity: float = 0.0
    end_date: datetime | None = None
    outcomes: list[str] = field(default_factory=lambda: ["Yes", "No"])
    is_active: bool = True
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def mid_price(self) -> float:
        """Get the midpoint price."""
        return (self.yes_price + self.no_price) / 2

    @property
    def implied_probability(self) -> float:
        """Get implied probability from YES price."""
        return self.yes_price

    @property
    def is_binary(self) -> bool:
        """Check if this is a simple YES/NO market."""
        return len(self.outcomes) == 2

    @property
    def arb_check(self) -> float:
        """Check for single-condition arbitrage.

        Returns:
            Sum of YES + NO prices. Should equal 1.0 in efficient market.
            < 1.0 indicates buy-all arbitrage opportunity.
            > 1.0 indicates sell-all arbitrage opportunity.
        """
        return self.yes_price + self.no_price


@dataclass
class OrderBookLevel:
    """Single level in the order book."""

    price: float
    size: float
    side: Side


@dataclass
class OrderBook:
    """Order book representation.

    Attributes:
        market_id: Market identifier.
        platform: Platform name.
        bids: List of bid levels (buy orders), sorted price descending.
        asks: List of ask levels (sell orders), sorted price ascending.
        timestamp: Time of snapshot.
    """

    market_id: str
    platform: str
    bids: list[OrderBookLevel] = field(default_factory=list)
    asks: list[OrderBookLevel] = field(default_factory=list)
    timestamp: datetime = field(default_factory=utc_now)

    @property
    def best_bid(self) -> float | None:
        """Get best bid price."""
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> float | None:
        """Get best ask price."""
        return self.asks[0].price if self.asks else None

    @property
    def spread(self) -> float | None:
        """Get bid-ask spread."""
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid

    @property
    def mid_price(self) -> float | None:
        """Get mid price."""
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2

    def depth_at_level(self, price_level: float, side: Side) -> float:
        """Get total size available within price level from best.

        Args:
            price_level: Maximum distance from best price (e.g., 0.01 = 1%).
            side: Which side to check.

        Returns:
            Total size available within price level.
        """
        total = 0.0
        if side == Side.BUY:
            if not self.bids or self.best_bid is None:
                return 0.0
            threshold = self.best_bid - price_level
            for level in self.bids:
                if level.price >= threshold:
                    total += level.size
                else:
                    break
        else:
            if not self.asks or self.best_ask is None:
                return 0.0
            threshold = self.best_ask + price_level
            for level in self.asks:
                if level.price <= threshold:
                    total += level.size
                else:
                    break
        return total


@dataclass
class Order:
    """Order representation.

    Attributes:
        id: Order identifier.
        market_id: Market identifier.
        platform: Platform name.
        side: Buy or sell.
        order_type: Limit or market.
        price: Limit price (None for market orders).
        size: Order size.
        filled_size: Amount filled so far.
        status: Current order status.
        created_at: Order creation time.
        raw: Raw platform-specific data.
    """

    id: str
    market_id: str
    platform: str
    side: Side
    order_type: OrderType
    price: float | None
    size: float
    filled_size: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=utc_now)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def remaining_size(self) -> float:
        """Get unfilled size."""
        return self.size - self.filled_size

    @property
    def is_complete(self) -> bool:
        """Check if order is fully filled or cancelled."""
        return self.status in (OrderStatus.FILLED, OrderStatus.CANCELLED)


@dataclass
class Trade:
    """Trade/fill representation.

    Attributes:
        id: Trade identifier.
        market_id: Market identifier.
        platform: Platform name.
        side: Buy or sell.
        price: Execution price.
        size: Trade size.
        timestamp: Execution time.
        fee: Trading fee paid.
        raw: Raw platform-specific data.
    """

    id: str
    market_id: str
    platform: str
    side: Side
    price: float
    size: float
    timestamp: datetime = field(default_factory=utc_now)
    fee: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def notional(self) -> float:
        """Get trade notional value."""
        return self.price * self.size


class BaseAdapter(ABC):
    """Abstract base class for prediction market platform adapters.

    All platform adapters must implement this interface to ensure
    consistent behavior across Polymarket, Kalshi, and future platforms.

    Pattern adapted from: C:/dev/libs/exchange-adapters/base.py
    """

    PLATFORM_NAME: str = "base"
    TIMEOUT: int = 15

    def __init__(self, base_url: str, timeout: int | None = None):
        """Initialize adapter.

        Args:
            base_url: Platform API base URL.
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout or self.TIMEOUT
        self._authenticated = False

    @property
    def platform(self) -> str:
        """Get platform name."""
        return self.PLATFORM_NAME

    @property
    def is_authenticated(self) -> bool:
        """Check if adapter is authenticated for trading."""
        return self._authenticated

    # =========================================================================
    # Abstract methods - must be implemented by subclasses
    # =========================================================================

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection and authenticate if credentials provided.

        Returns:
            True if connection successful.
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection and cleanup resources."""
        pass

    @abstractmethod
    async def get_markets(
        self,
        active_only: bool = True,
        category: str | None = None,
        limit: int = 100,
    ) -> list[Market]:
        """Fetch available markets.

        Args:
            active_only: Only return active/trading markets.
            category: Filter by category.
            limit: Maximum number of markets to return.

        Returns:
            List of Market objects.
        """
        pass

    @abstractmethod
    async def get_market(self, market_id: str) -> Market | None:
        """Fetch a specific market by ID.

        Args:
            market_id: Platform-specific market identifier.

        Returns:
            Market object or None if not found.
        """
        pass

    @abstractmethod
    async def get_order_book(self, market_id: str) -> OrderBook:
        """Fetch order book for a market.

        Args:
            market_id: Market identifier.

        Returns:
            OrderBook object.
        """
        pass

    @abstractmethod
    async def get_trades(
        self,
        market_id: str,
        limit: int = 100,
    ) -> list[Trade]:
        """Fetch recent trades for a market.

        Args:
            market_id: Market identifier.
            limit: Maximum number of trades to return.

        Returns:
            List of Trade objects.
        """
        pass

    @abstractmethod
    async def get_price(self, market_id: str) -> tuple[float, float]:
        """Get current YES and NO prices for a market.

        Args:
            market_id: Market identifier.

        Returns:
            Tuple of (yes_price, no_price).
        """
        pass

    # =========================================================================
    # Trading methods (require authentication)
    # =========================================================================

    @abstractmethod
    async def place_order(
        self,
        market_id: str,
        side: Side,
        price: float,
        size: float,
        order_type: OrderType = OrderType.LIMIT,
    ) -> Order:
        """Place an order.

        Args:
            market_id: Market identifier.
            side: Buy or sell.
            price: Limit price (ignored for market orders).
            size: Order size.
            order_type: Limit or market order.

        Returns:
            Created Order object.

        Raises:
            AuthenticationError: If not authenticated.
            InsufficientFundsError: If insufficient balance.
        """
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order.

        Args:
            order_id: Order identifier.

        Returns:
            True if cancelled successfully.
        """
        pass

    @abstractmethod
    async def get_open_orders(self, market_id: str | None = None) -> list[Order]:
        """Get open orders.

        Args:
            market_id: Filter by market (optional).

        Returns:
            List of open Order objects.
        """
        pass

    @abstractmethod
    async def get_balance(self) -> float:
        """Get account balance in USD.

        Returns:
            Available balance.
        """
        pass

    # =========================================================================
    # Utility methods
    # =========================================================================

    async def test_connection(self) -> dict[str, Any]:
        """Test connection to the platform.

        Returns:
            Dictionary with connection status and latency.
        """
        import time

        start = time.time()
        try:
            connected = await self.connect()
            latency = (time.time() - start) * 1000  # ms
            return {
                "platform": self.platform,
                "connected": connected,
                "authenticated": self.is_authenticated,
                "latency_ms": round(latency, 2),
            }
        except Exception as e:
            return {
                "platform": self.platform,
                "connected": False,
                "error": str(e),
            }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(platform={self.platform}, authenticated={self.is_authenticated})"
