"""Order Book Imbalance (OBI) metric for prediction markets.

Adapted from traditional market microstructure analysis.
Measures the balance between buy and sell pressure in the order book.

Pattern adapted from: C:/dev/quant/backtester3/1_metrics/obi.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..core.utils import get_logger, safe_divide

if TYPE_CHECKING:
    from ..adapters.base import OrderBook

logger = get_logger(__name__)


@dataclass
class ImbalanceReading:
    """Single order imbalance reading.

    Attributes:
        imbalance: Value between -1 (all sell) and +1 (all buy).
        bid_volume: Total volume on bid side.
        ask_volume: Total volume on ask side.
        signal: Interpretation ('bullish', 'bearish', 'neutral').
    """

    imbalance: float
    bid_volume: float
    ask_volume: float
    signal: str

    @property
    def is_bullish(self) -> bool:
        return self.signal == "bullish"

    @property
    def is_bearish(self) -> bool:
        return self.signal == "bearish"


class OrderImbalanceMetric:
    """Order Book Imbalance (OBI) metric.

    Calculates the imbalance between bid and ask volumes in the order book.
    A positive imbalance indicates more buy pressure (bullish).
    A negative imbalance indicates more sell pressure (bearish).

    Formula:
        OBI = (Bid Volume - Ask Volume) / (Bid Volume + Ask Volume)

    Range: [-1, +1]
    - +1 = All buy orders, no sell orders
    - -1 = All sell orders, no buy orders
    - 0 = Equal buy and sell pressure

    In prediction markets, this can signal:
    - Bullish OBI: Prices likely to rise (more buyers)
    - Bearish OBI: Prices likely to fall (more sellers)
    - High |OBI|: Potential informed trading

    Example:
        ```python
        metric = OrderImbalanceMetric(threshold_bullish=0.3)

        for market in markets:
            book = await adapter.get_order_book(market.id)
            reading = metric.calculate(book)

            if reading.is_bullish:
                print(f"Bullish signal: {reading.imbalance:.2f}")
        ```
    """

    def __init__(
        self,
        threshold_bullish: float = 0.3,
        threshold_bearish: float = -0.3,
        depth_levels: int = 5,
    ):
        """Initialize OBI metric.

        Args:
            threshold_bullish: OBI above this is bullish signal.
            threshold_bearish: OBI below this is bearish signal.
            depth_levels: Number of price levels to include.
        """
        self.threshold_bullish = threshold_bullish
        self.threshold_bearish = threshold_bearish
        self.depth_levels = depth_levels

    def calculate(self, order_book: OrderBook) -> ImbalanceReading:
        """Calculate order book imbalance.

        Args:
            order_book: Order book to analyze.

        Returns:
            ImbalanceReading with calculated values.
        """
        # Sum volumes at top N levels
        bid_volume = sum(
            level.size for level in order_book.bids[: self.depth_levels]
        )
        ask_volume = sum(
            level.size for level in order_book.asks[: self.depth_levels]
        )

        # Calculate imbalance
        total_volume = bid_volume + ask_volume
        imbalance = safe_divide(
            bid_volume - ask_volume,
            total_volume,
            default=0.0,
        )

        # Determine signal
        if imbalance >= self.threshold_bullish:
            signal = "bullish"
        elif imbalance <= self.threshold_bearish:
            signal = "bearish"
        else:
            signal = "neutral"

        return ImbalanceReading(
            imbalance=imbalance,
            bid_volume=bid_volume,
            ask_volume=ask_volume,
            signal=signal,
        )

    def calculate_weighted(
        self,
        order_book: OrderBook,
        decay_factor: float = 0.5,
    ) -> ImbalanceReading:
        """Calculate volume-weighted order book imbalance.

        Gives more weight to levels closer to the best bid/ask.

        Args:
            order_book: Order book to analyze.
            decay_factor: Weight decay per level (0-1).

        Returns:
            ImbalanceReading with weighted values.
        """
        bid_volume = 0.0
        ask_volume = 0.0

        for i, level in enumerate(order_book.bids[: self.depth_levels]):
            weight = decay_factor ** i
            bid_volume += level.size * weight

        for i, level in enumerate(order_book.asks[: self.depth_levels]):
            weight = decay_factor ** i
            ask_volume += level.size * weight

        total_volume = bid_volume + ask_volume
        imbalance = safe_divide(
            bid_volume - ask_volume,
            total_volume,
            default=0.0,
        )

        if imbalance >= self.threshold_bullish:
            signal = "bullish"
        elif imbalance <= self.threshold_bearish:
            signal = "bearish"
        else:
            signal = "neutral"

        return ImbalanceReading(
            imbalance=imbalance,
            bid_volume=bid_volume,
            ask_volume=ask_volume,
            signal=signal,
        )
