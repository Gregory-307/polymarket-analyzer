"""Multi-outcome arbitrage detection.

Detects when the sum of all outcome prices in a multi-outcome market
doesn't equal $1.00, creating bundle arbitrage opportunities.

Research: ~$28.3M extracted via this method on Polymarket.

Example: A market with outcomes A, B, C, D (mutually exclusive)
- If price(A) + price(B) + price(C) + price(D) < $1.00
- Buy one share of each outcome
- Guaranteed one wins, paying $1.00
- Profit: $1.00 - sum(prices)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..core.utils import get_logger

if TYPE_CHECKING:
    from ..adapters.base import Market

logger = get_logger(__name__)


@dataclass
class OutcomePrice:
    """Price information for a single outcome."""

    outcome_id: str
    outcome_name: str
    price: float


@dataclass
class MultiArbOpportunity:
    """Represents a multi-outcome arbitrage opportunity.

    Attributes:
        market_id: The market identifier.
        platform: Platform name.
        question: Market question.
        outcomes: List of outcome prices.
        action: 'buy_bundle' or 'sell_bundle'.
        sum_prices: Sum of all outcome prices.
        profit_pct: Expected profit as percentage.
        profit_usd: Expected profit in USD per bundle.
    """

    market_id: str
    platform: str
    question: str
    outcomes: list[OutcomePrice] = field(default_factory=list)
    action: str = "buy_bundle"
    sum_prices: float = 0.0
    profit_pct: float = 0.0
    profit_usd: float = 0.0

    @property
    def num_outcomes(self) -> int:
        """Number of outcomes in this market."""
        return len(self.outcomes)


class MultiOutcomeArbitrage:
    """Detector for multi-outcome arbitrage opportunities.

    In markets with N mutually exclusive outcomes (only one can win),
    the sum of all outcome prices should equal $1.00.

    When sum < $1.00:
    - Buy one share of each outcome (a "bundle")
    - Exactly one outcome will resolve to $1.00
    - Cost: sum of all prices
    - Profit: $1.00 - sum

    When sum > $1.00:
    - Sell one share of each outcome
    - Maximum liability is $1.00
    - Received: sum of all prices
    - Profit: sum - $1.00

    Example:
        ```python
        detector = MultiOutcomeArbitrage(min_profit_pct=0.01)

        # For a market with outcomes priced: 0.30, 0.25, 0.20, 0.15
        # Sum = 0.90, Profit = 10%
        opportunities = detector.scan(multi_outcome_markets)
        ```
    """

    def __init__(
        self,
        min_profit_pct: float = 0.01,
        min_profit_usd: float = 1.00,
        min_outcomes: int = 3,
    ):
        """Initialize detector.

        Args:
            min_profit_pct: Minimum profit percentage to report.
            min_profit_usd: Minimum profit in USD per bundle.
            min_outcomes: Minimum number of outcomes for a valid multi-market.
        """
        self.min_profit_pct = min_profit_pct
        self.min_profit_usd = min_profit_usd
        self.min_outcomes = min_outcomes

    def check_market(
        self,
        market_id: str,
        platform: str,
        question: str,
        outcomes: list[dict[str, Any]],
    ) -> MultiArbOpportunity | None:
        """Check a multi-outcome market for arbitrage.

        Args:
            market_id: Market identifier.
            platform: Platform name.
            question: Market question.
            outcomes: List of outcome dicts with 'id', 'name', 'price' keys.

        Returns:
            MultiArbOpportunity if found, None otherwise.
        """
        if len(outcomes) < self.min_outcomes:
            return None

        # Parse outcomes
        parsed_outcomes = []
        sum_prices = 0.0

        for outcome in outcomes:
            price = float(outcome.get("price", 0))
            parsed_outcomes.append(OutcomePrice(
                outcome_id=outcome.get("id", ""),
                outcome_name=outcome.get("name", ""),
                price=price,
            ))
            sum_prices += price

        # Check for arbitrage
        if sum_prices < 1.0:
            profit_pct = 1.0 - sum_prices
            profit_usd = profit_pct  # Per $1 bundle

            if profit_pct >= self.min_profit_pct:
                return MultiArbOpportunity(
                    market_id=market_id,
                    platform=platform,
                    question=question,
                    outcomes=parsed_outcomes,
                    action="buy_bundle",
                    sum_prices=sum_prices,
                    profit_pct=profit_pct,
                    profit_usd=profit_usd,
                )

        elif sum_prices > 1.0:
            profit_pct = sum_prices - 1.0
            profit_usd = profit_pct

            if profit_pct >= self.min_profit_pct:
                return MultiArbOpportunity(
                    market_id=market_id,
                    platform=platform,
                    question=question,
                    outcomes=parsed_outcomes,
                    action="sell_bundle",
                    sum_prices=sum_prices,
                    profit_pct=profit_pct,
                    profit_usd=profit_usd,
                )

        return None

    def scan_polymarket_group(
        self,
        group_data: dict[str, Any],
    ) -> MultiArbOpportunity | None:
        """Scan a Polymarket market group for multi-outcome arbitrage.

        Polymarket groups related binary markets into "events" where
        outcomes are mutually exclusive.

        Args:
            group_data: Polymarket group/event data with nested markets.

        Returns:
            MultiArbOpportunity if found, None otherwise.
        """
        # Extract markets from group
        markets = group_data.get("markets", [])
        if len(markets) < self.min_outcomes:
            return None

        outcomes = []
        for market in markets:
            # Each market in the group is an outcome
            tokens = market.get("tokens", [])
            yes_token = next((t for t in tokens if t.get("outcome") == "Yes"), None)

            if yes_token:
                outcomes.append({
                    "id": market.get("conditionId", ""),
                    "name": market.get("question", ""),
                    "price": float(yes_token.get("price", 0)),
                })

        return self.check_market(
            market_id=group_data.get("id", ""),
            platform="polymarket",
            question=group_data.get("title", ""),
            outcomes=outcomes,
        )

    def scan_kalshi_event(
        self,
        event_data: dict[str, Any],
        markets: list[dict[str, Any]],
    ) -> MultiArbOpportunity | None:
        """Scan a Kalshi event for multi-outcome arbitrage.

        Kalshi groups related markets under "events" or "series".

        Args:
            event_data: Kalshi event metadata.
            markets: List of markets in this event.

        Returns:
            MultiArbOpportunity if found, None otherwise.
        """
        if len(markets) < self.min_outcomes:
            return None

        outcomes = []
        for market in markets:
            outcomes.append({
                "id": market.get("ticker", ""),
                "name": market.get("title", ""),
                "price": float(market.get("yes_bid", 50)) / 100,
            })

        return self.check_market(
            market_id=event_data.get("event_ticker", ""),
            platform="kalshi",
            question=event_data.get("title", ""),
            outcomes=outcomes,
        )

    def calculate_execution(
        self,
        opportunity: MultiArbOpportunity,
        num_bundles: int = 1,
    ) -> dict:
        """Calculate execution details for an opportunity.

        Args:
            opportunity: The arbitrage opportunity.
            num_bundles: Number of complete bundles to buy/sell.

        Returns:
            Dictionary with execution details.
        """
        if opportunity.action == "buy_bundle":
            total_cost = opportunity.sum_prices * num_bundles
            guaranteed_payout = 1.0 * num_bundles

            orders = []
            for outcome in opportunity.outcomes:
                orders.append({
                    "outcome_id": outcome.outcome_id,
                    "outcome_name": outcome.outcome_name,
                    "side": "buy",
                    "size": num_bundles,
                    "price": outcome.price,
                    "cost": outcome.price * num_bundles,
                })

            return {
                "action": "buy_bundle",
                "num_bundles": num_bundles,
                "orders": orders,
                "total_cost": total_cost,
                "guaranteed_payout": guaranteed_payout,
                "profit": guaranteed_payout - total_cost,
                "profit_pct": opportunity.profit_pct,
            }
        else:
            total_received = opportunity.sum_prices * num_bundles
            max_liability = 1.0 * num_bundles

            orders = []
            for outcome in opportunity.outcomes:
                orders.append({
                    "outcome_id": outcome.outcome_id,
                    "outcome_name": outcome.outcome_name,
                    "side": "sell",
                    "size": num_bundles,
                    "price": outcome.price,
                    "received": outcome.price * num_bundles,
                })

            return {
                "action": "sell_bundle",
                "num_bundles": num_bundles,
                "orders": orders,
                "total_received": total_received,
                "max_liability": max_liability,
                "profit": total_received - max_liability,
                "profit_pct": opportunity.profit_pct,
            }
