"""Spread Dynamics metric for prediction markets.

Tracks bid-ask spread over time to identify:
- Market efficiency
- Liquidity conditions
- Potential trading opportunities
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from ..core.utils import get_logger, utc_now

if TYPE_CHECKING:
    from ..adapters.base import OrderBook

logger = get_logger(__name__)


@dataclass
class SpreadSnapshot:
    """Single spread observation."""

    timestamp: datetime
    spread: float
    mid_price: float
    best_bid: float
    best_ask: float


@dataclass
class SpreadAnalysis:
    """Analysis of spread dynamics over a window.

    Attributes:
        current_spread: Most recent spread value.
        avg_spread: Average spread in window.
        min_spread: Minimum spread observed.
        max_spread: Maximum spread observed.
        spread_volatility: Standard deviation of spread.
        is_tight: Whether spread is below threshold.
        is_widening: Whether spread is increasing.
        observations: Number of observations in window.
    """

    current_spread: float
    avg_spread: float
    min_spread: float
    max_spread: float
    spread_volatility: float
    is_tight: bool
    is_widening: bool
    observations: int


class SpreadDynamicsMetric:
    """Tracks and analyzes bid-ask spread dynamics.

    The spread is a key indicator of market quality:
    - Tight spreads indicate liquid, efficient markets
    - Wide spreads indicate illiquid or volatile markets
    - Widening spreads may signal increased uncertainty

    Example:
        ```python
        metric = SpreadDynamicsMetric(
            window_seconds=300,  # 5 minute window
            alert_threshold=0.05,  # Alert if spread > 5%
        )

        # Update with new order books
        for _ in range(10):
            book = await adapter.get_order_book(market_id)
            metric.update(market_id, book)
            await asyncio.sleep(30)

        # Get analysis
        analysis = metric.analyze(market_id)
        print(f"Avg spread: {analysis.avg_spread:.2%}")
        ```
    """

    def __init__(
        self,
        window_seconds: int = 300,
        alert_threshold: float = 0.05,
        max_history: int = 1000,
    ):
        """Initialize metric.

        Args:
            window_seconds: Time window for analysis.
            alert_threshold: Spread above this triggers alert.
            max_history: Maximum snapshots to keep per market.
        """
        self.window_seconds = window_seconds
        self.alert_threshold = alert_threshold
        self.max_history = max_history

        # Store history per market
        self._history: dict[str, deque[SpreadSnapshot]] = {}

    def update(self, market_id: str, order_book: OrderBook) -> SpreadSnapshot | None:
        """Record a new spread observation.

        Args:
            market_id: Market identifier.
            order_book: Current order book.

        Returns:
            SpreadSnapshot if valid, None otherwise.
        """
        if order_book.spread is None or order_book.mid_price is None:
            return None

        snapshot = SpreadSnapshot(
            timestamp=utc_now(),
            spread=order_book.spread,
            mid_price=order_book.mid_price,
            best_bid=order_book.best_bid or 0,
            best_ask=order_book.best_ask or 0,
        )

        # Initialize history if needed
        if market_id not in self._history:
            self._history[market_id] = deque(maxlen=self.max_history)

        self._history[market_id].append(snapshot)

        # Check for alert
        if snapshot.spread > self.alert_threshold:
            logger.warning(
                "spread_alert",
                market_id=market_id,
                spread=f"{snapshot.spread:.2%}",
                threshold=f"{self.alert_threshold:.2%}",
            )

        return snapshot

    def analyze(self, market_id: str) -> SpreadAnalysis | None:
        """Analyze spread dynamics for a market.

        Args:
            market_id: Market to analyze.

        Returns:
            SpreadAnalysis or None if insufficient data.
        """
        if market_id not in self._history:
            return None

        history = self._history[market_id]
        if len(history) < 2:
            return None

        # Filter to window
        now = utc_now()
        window_start = now.timestamp() - self.window_seconds
        in_window = [
            s for s in history
            if s.timestamp.timestamp() >= window_start
        ]

        if not in_window:
            in_window = list(history)  # Use all if window is empty

        spreads = [s.spread for s in in_window]
        current = spreads[-1]

        # Calculate statistics
        avg_spread = sum(spreads) / len(spreads)
        min_spread = min(spreads)
        max_spread = max(spreads)

        # Volatility (standard deviation)
        variance = sum((s - avg_spread) ** 2 for s in spreads) / len(spreads)
        volatility = variance ** 0.5

        # Trend detection (simple: compare recent to earlier)
        mid_idx = len(spreads) // 2
        if mid_idx > 0:
            earlier_avg = sum(spreads[:mid_idx]) / mid_idx
            recent_avg = sum(spreads[mid_idx:]) / (len(spreads) - mid_idx)
            is_widening = recent_avg > earlier_avg * 1.1  # 10% increase
        else:
            is_widening = False

        return SpreadAnalysis(
            current_spread=current,
            avg_spread=avg_spread,
            min_spread=min_spread,
            max_spread=max_spread,
            spread_volatility=volatility,
            is_tight=current < self.alert_threshold,
            is_widening=is_widening,
            observations=len(in_window),
        )

    def get_tightest_markets(
        self,
        top_n: int = 10,
    ) -> list[tuple[str, float]]:
        """Get markets with tightest spreads.

        Args:
            top_n: Number of markets to return.

        Returns:
            List of (market_id, spread) tuples sorted by spread.
        """
        current_spreads = []

        for market_id, history in self._history.items():
            if history:
                latest = history[-1]
                current_spreads.append((market_id, latest.spread))

        current_spreads.sort(key=lambda x: x[1])
        return current_spreads[:top_n]

    def clear_history(self, market_id: str | None = None) -> None:
        """Clear historical data.

        Args:
            market_id: Specific market to clear, or None for all.
        """
        if market_id:
            self._history.pop(market_id, None)
        else:
            self._history.clear()
