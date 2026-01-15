"""Single-condition arbitrage detection.

Detects when YES + NO prices don't sum to $1.00, creating
risk-free arbitrage opportunities.

Types:
- Buy-All: YES + NO < $1.00 -> Buy both, guaranteed $1 payout
- Sell-All: YES + NO > $1.00 -> Sell both, guaranteed profit

Research: 1-3% returns per trade, near-zero risk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..core.utils import get_logger

if TYPE_CHECKING:
    from ..adapters.base import Market

logger = get_logger(__name__)


@dataclass
class SingleArbOpportunity:
    """Represents a single-condition arbitrage opportunity.

    Attributes:
        market: The market with mispricing.
        action: 'buy_all' or 'sell_all'.
        yes_price: Current YES price.
        no_price: Current NO price.
        sum_prices: YES + NO (should be 1.0).
        profit_pct: Expected profit as percentage.
        profit_usd: Expected profit in USD per $1 position.
    """

    market: Market
    action: str
    yes_price: float
    no_price: float
    sum_prices: float
    profit_pct: float
    profit_usd: float

    @property
    def is_buy_all(self) -> bool:
        """Check if this is a buy-all opportunity."""
        return self.action == "buy_all"

    @property
    def is_sell_all(self) -> bool:
        """Check if this is a sell-all opportunity."""
        return self.action == "sell_all"


class SingleConditionArbitrage:
    """Detector for single-condition arbitrage opportunities.

    In an efficient market, YES + NO should equal $1.00.
    When they don't, arbitrage exists:

    - YES + NO < $1.00: Buy both YES and NO tokens.
      One will resolve to $1, other to $0.
      Total payout: $1.00, Cost: YES + NO < $1.00
      Profit: $1.00 - (YES + NO)

    - YES + NO > $1.00: Sell both YES and NO tokens.
      Net received: YES + NO > $1.00
      Max liability: $1.00 (whichever resolves to YES)
      Profit: (YES + NO) - $1.00

    Example:
        ```python
        detector = SingleConditionArbitrage(min_profit_pct=0.01)
        opportunities = detector.scan(markets)

        for opp in opportunities:
            print(f"{opp.action}: {opp.market.question}")
            print(f"  Profit: {opp.profit_pct:.2%}")
        ```
    """

    def __init__(
        self,
        min_profit_pct: float = 0.005,
        min_profit_usd: float = 0.50,
    ):
        """Initialize detector.

        Args:
            min_profit_pct: Minimum profit percentage to report (default 0.5%).
            min_profit_usd: Minimum profit in USD per $100 position.
        """
        self.min_profit_pct = min_profit_pct
        self.min_profit_usd = min_profit_usd

    def check_market(self, market: Market) -> SingleArbOpportunity | None:
        """Check a single market for arbitrage.

        Args:
            market: Market to check.

        Returns:
            SingleArbOpportunity if found, None otherwise.
        """
        if not market.is_binary:
            return None

        sum_prices = market.yes_price + market.no_price

        # Buy-all opportunity: sum < 1
        if sum_prices < 1.0:
            profit_pct = 1.0 - sum_prices
            profit_usd = profit_pct * 100  # Per $100 position

            if profit_pct >= self.min_profit_pct:
                return SingleArbOpportunity(
                    market=market,
                    action="buy_all",
                    yes_price=market.yes_price,
                    no_price=market.no_price,
                    sum_prices=sum_prices,
                    profit_pct=profit_pct,
                    profit_usd=profit_usd,
                )

        # Sell-all opportunity: sum > 1
        elif sum_prices > 1.0:
            profit_pct = sum_prices - 1.0
            profit_usd = profit_pct * 100

            if profit_pct >= self.min_profit_pct:
                return SingleArbOpportunity(
                    market=market,
                    action="sell_all",
                    yes_price=market.yes_price,
                    no_price=market.no_price,
                    sum_prices=sum_prices,
                    profit_pct=profit_pct,
                    profit_usd=profit_usd,
                )

        return None

    def scan(self, markets: list[Market]) -> list[SingleArbOpportunity]:
        """Scan multiple markets for arbitrage opportunities.

        Args:
            markets: List of markets to scan.

        Returns:
            List of opportunities found, sorted by profit descending.
        """
        opportunities = []

        for market in markets:
            opp = self.check_market(market)
            if opp:
                opportunities.append(opp)
                logger.info(
                    "single_arb_found",
                    market_id=market.id,
                    platform=market.platform,
                    action=opp.action,
                    profit_pct=f"{opp.profit_pct:.2%}",
                )

        # Sort by profit descending
        opportunities.sort(key=lambda x: x.profit_pct, reverse=True)

        logger.info("single_arb_scan_complete", count=len(opportunities))
        return opportunities

    def calculate_execution(
        self,
        opportunity: SingleArbOpportunity,
        position_size_usd: float,
    ) -> dict:
        """Calculate execution details for an opportunity.

        Args:
            opportunity: The arbitrage opportunity.
            position_size_usd: How much to allocate.

        Returns:
            Dictionary with execution details.
        """
        if opportunity.is_buy_all:
            # Buy both YES and NO
            cost_per_unit = opportunity.sum_prices
            units = position_size_usd / cost_per_unit
            payout = units  # One unit always pays $1

            return {
                "action": "buy_all",
                "yes_units": units,
                "no_units": units,
                "yes_cost": units * opportunity.yes_price,
                "no_cost": units * opportunity.no_price,
                "total_cost": position_size_usd,
                "guaranteed_payout": payout,
                "profit": payout - position_size_usd,
                "profit_pct": opportunity.profit_pct,
            }
        else:
            # Sell both YES and NO
            received_per_unit = opportunity.sum_prices
            units = position_size_usd  # Liability capped at position size
            received = units * received_per_unit

            return {
                "action": "sell_all",
                "yes_units": units,
                "no_units": units,
                "yes_received": units * opportunity.yes_price,
                "no_received": units * opportunity.no_price,
                "total_received": received,
                "max_liability": units,
                "profit": received - units,
                "profit_pct": opportunity.profit_pct,
            }
