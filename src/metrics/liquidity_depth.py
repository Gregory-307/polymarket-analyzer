"""Liquidity Depth metric for prediction markets.

Measures the available liquidity at various price levels,
helping identify markets suitable for larger positions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..core.utils import get_logger

if TYPE_CHECKING:
    from ..adapters.base import OrderBook

logger = get_logger(__name__)


@dataclass
class DepthLevel:
    """Liquidity at a specific price distance from best."""

    distance: float  # e.g., 0.01 = 1% from best
    bid_depth: float  # USD available on bid side
    ask_depth: float  # USD available on ask side
    total_depth: float  # Combined depth


@dataclass
class LiquidityProfile:
    """Complete liquidity profile for a market.

    Attributes:
        market_id: Market identifier.
        best_bid: Best bid price.
        best_ask: Best ask price.
        spread: Bid-ask spread.
        levels: Depth at various price levels.
        total_bid_depth: Total bid-side liquidity.
        total_ask_depth: Total ask-side liquidity.
        depth_imbalance: Ratio of bid to total depth.
    """

    market_id: str
    best_bid: float | None
    best_ask: float | None
    spread: float | None
    levels: list[DepthLevel] = field(default_factory=list)
    total_bid_depth: float = 0.0
    total_ask_depth: float = 0.0

    @property
    def total_depth(self) -> float:
        return self.total_bid_depth + self.total_ask_depth

    @property
    def depth_imbalance(self) -> float:
        """Ratio of bid depth to total depth. 0.5 = balanced."""
        if self.total_depth == 0:
            return 0.5
        return self.total_bid_depth / self.total_depth

    @property
    def is_liquid(self) -> bool:
        """Check if market has minimum liquidity."""
        return self.total_depth >= 100  # $100 minimum


class LiquidityDepthMetric:
    """Liquidity depth analyzer for prediction markets.

    Measures how much volume can be executed at various price levels,
    essential for:
    - Position sizing (don't exceed available liquidity)
    - Slippage estimation
    - Identifying illiquid markets to avoid

    Example:
        ```python
        metric = LiquidityDepthMetric(
            levels=[0.01, 0.02, 0.05, 0.10]
        )

        book = await adapter.get_order_book(market_id)
        profile = metric.analyze(book)

        print(f"Total liquidity: ${profile.total_depth:.2f}")
        print(f"Spread: {profile.spread:.4f}")

        for level in profile.levels:
            print(f"  At {level.distance:.1%}: ${level.total_depth:.2f}")
        ```
    """

    def __init__(
        self,
        levels: list[float] | None = None,
        min_depth_usd: float = 100,
    ):
        """Initialize metric.

        Args:
            levels: Price distances to measure (e.g., [0.01, 0.02, 0.05]).
            min_depth_usd: Minimum depth to consider market liquid.
        """
        self.levels = levels or [0.01, 0.02, 0.05, 0.10]
        self.min_depth_usd = min_depth_usd

    def analyze(self, order_book: OrderBook) -> LiquidityProfile:
        """Analyze liquidity profile of an order book.

        Args:
            order_book: Order book to analyze.

        Returns:
            LiquidityProfile with depth at each level.
        """
        profile = LiquidityProfile(
            market_id=order_book.market_id,
            best_bid=order_book.best_bid,
            best_ask=order_book.best_ask,
            spread=order_book.spread,
        )

        # Calculate depth at each level
        for distance in self.levels:
            bid_depth = self._depth_within_distance(
                order_book.bids,
                order_book.best_bid,
                distance,
                is_bid=True,
            )
            ask_depth = self._depth_within_distance(
                order_book.asks,
                order_book.best_ask,
                distance,
                is_bid=False,
            )

            profile.levels.append(DepthLevel(
                distance=distance,
                bid_depth=bid_depth,
                ask_depth=ask_depth,
                total_depth=bid_depth + ask_depth,
            ))

        # Calculate totals
        profile.total_bid_depth = sum(level.size for level in order_book.bids)
        profile.total_ask_depth = sum(level.size for level in order_book.asks)

        return profile

    def _depth_within_distance(
        self,
        levels: list,
        best_price: float | None,
        distance: float,
        is_bid: bool,
    ) -> float:
        """Calculate total depth within a price distance.

        Args:
            levels: Order book levels.
            best_price: Best bid or ask price.
            distance: Maximum distance from best price.
            is_bid: True for bid side, False for ask side.

        Returns:
            Total depth in USD within the distance.
        """
        if best_price is None or not levels:
            return 0.0

        total = 0.0

        for level in levels:
            if is_bid:
                # Bids: check if price is within distance below best
                if level.price >= best_price - distance:
                    total += level.size * level.price
                else:
                    break
            else:
                # Asks: check if price is within distance above best
                if level.price <= best_price + distance:
                    total += level.size * level.price
                else:
                    break

        return total

    def estimate_slippage(
        self,
        order_book: OrderBook,
        order_size_usd: float,
        is_buy: bool,
    ) -> float:
        """Estimate execution slippage for a given order size.

        Args:
            order_book: Order book to analyze.
            order_size_usd: Size of order in USD.
            is_buy: True for buy order, False for sell.

        Returns:
            Estimated slippage as decimal (e.g., 0.01 = 1%).
        """
        levels = order_book.asks if is_buy else order_book.bids
        best_price = order_book.best_ask if is_buy else order_book.best_bid

        if not levels or best_price is None:
            return 1.0  # 100% slippage (no liquidity)

        remaining = order_size_usd
        weighted_price = 0.0
        total_filled = 0.0

        for level in levels:
            available_usd = level.size * level.price
            fill_usd = min(remaining, available_usd)
            fill_shares = fill_usd / level.price

            weighted_price += level.price * fill_shares
            total_filled += fill_shares
            remaining -= fill_usd

            if remaining <= 0:
                break

        if total_filled == 0:
            return 1.0

        avg_price = weighted_price / total_filled
        slippage = abs(avg_price - best_price) / best_price

        return slippage
