"""Favorite-Longshot Bias Strategy.

Exploits the well-documented behavioral bias where:
- Long shots (low probability) are systematically OVERPRICED
- Favorites (high probability) are systematically UNDERPRICED

This is the PRIMARY EDGE for the DRW application.

Research basis:
- Kahneman & Tversky Prospect Theory (1979)
- NBER Working Paper 15923
- Documented returns: Up to 1800% annualized (ChainCatcher analysis)

Mechanism:
- People overvalue small probabilities (lottery ticket mentality)
- People undervalue near-certainties
- At extreme probabilities (>95% or <5%), the mispricing is largest

Strategy:
- Buy high-probability outcomes (>95%) when priced below fair value
- Avoid low-probability outcomes (lottery tickets)
- Compound small, frequent wins
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from ..core.utils import get_logger, utc_now

if TYPE_CHECKING:
    from ..adapters.base import Market

logger = get_logger(__name__)


@dataclass
class FavoriteLongshotOpportunity:
    """Represents a favorite-longshot bias opportunity.

    Attributes:
        market: The market with potential mispricing.
        side: 'YES' or 'NO' - which outcome is the favorite.
        current_price: Current market price of the favorite.
        estimated_fair_value: Our estimate of true probability.
        edge: Estimated edge (fair_value - current_price).
        confidence: Confidence in our estimate (0-1).
        time_to_resolution: Time until market resolves.
        liquidity: Available liquidity for execution.
    """

    market: Market
    side: str
    current_price: float
    estimated_fair_value: float
    edge: float
    confidence: float = 0.5
    time_to_resolution: float | None = None  # hours
    liquidity: float = 0.0

    @property
    def expected_return(self) -> float:
        """Calculate expected return if our estimate is correct.

        Returns:
            Expected return as decimal (e.g., 0.05 = 5%).
        """
        if self.current_price <= 0:
            return 0.0

        # If we buy at current_price and fair_value is true probability
        # Expected payout = fair_value * $1 + (1-fair_value) * $0
        # Expected profit = fair_value - current_price
        return self.edge / self.current_price

    @property
    def annualized_return(self) -> float | None:
        """Calculate annualized return if time_to_resolution is known.

        Returns:
            Annualized return or None if time unknown.
        """
        if self.time_to_resolution is None or self.time_to_resolution <= 0:
            return None

        # Simple annualization (not compounded)
        hours_per_year = 365.25 * 24
        turns_per_year = hours_per_year / self.time_to_resolution
        return self.expected_return * turns_per_year


class FavoriteLongshotStrategy:
    """Strategy for exploiting favorite-longshot bias.

    This strategy identifies high-probability outcomes that may be
    underpriced due to behavioral biases, then accumulates positions
    for small but consistent gains.

    Key principles:
    1. Only bet on favorites (>95% probability)
    2. Require minimum edge above transaction costs
    3. Size positions conservatively (black swan risk)
    4. Compound gains across many positions

    Example:
        ```python
        strategy = FavoriteLongshotStrategy(
            min_probability=0.95,
            min_edge=0.01,
        )

        opportunities = strategy.scan(markets)

        for opp in opportunities:
            print(f"{opp.side}: {opp.market.question}")
            print(f"  Price: {opp.current_price:.2%}")
            print(f"  Fair Value: {opp.estimated_fair_value:.2%}")
            print(f"  Edge: {opp.edge:.2%}")
        ```
    """

    def __init__(
        self,
        min_probability: float = 0.95,
        min_edge: float = 0.01,
        max_position_usd: float = 1000,
        base_rate_weight: float = 0.3,
    ):
        """Initialize strategy.

        Args:
            min_probability: Minimum probability to consider (default 95%).
            min_edge: Minimum edge required (default 1%).
            max_position_usd: Maximum position size per market.
            base_rate_weight: Weight given to historical base rates.
        """
        self.min_probability = min_probability
        self.min_edge = min_edge
        self.max_position_usd = max_position_usd
        self.base_rate_weight = base_rate_weight

        # Track historical resolution rates for calibration
        self._calibration_data: dict[str, list[tuple[float, bool]]] = {}

    def estimate_fair_value(
        self,
        market: Market,
        current_price: float,
    ) -> tuple[float, float]:
        """Estimate the fair value probability for an outcome.

        This is a simplified estimation. A production system would use:
        - Historical resolution rates at similar price levels
        - Category-specific calibration
        - Time-to-resolution adjustments
        - Liquidity/volume signals

        Args:
            market: The market to evaluate.
            current_price: Current price of the outcome.

        Returns:
            Tuple of (estimated_fair_value, confidence).
        """
        # Base estimate: assume market is somewhat efficient
        base_estimate = current_price

        # Adjustment for favorite-longshot bias
        # At high probabilities, markets tend to underprice favorites
        # Research shows 1-3% underpricing at extreme probabilities
        if current_price >= 0.95:
            # High favorites are typically underpriced by 1-3%
            # Base 1% adjustment + additional for more extreme prices
            bias_adjustment = 0.01 + 0.5 * (current_price - 0.95)
            adjusted = min(0.99, base_estimate + bias_adjustment)
        elif current_price <= 0.05:
            # Long shots are typically overpriced by similar margin
            bias_adjustment = -0.01 - 0.5 * (0.05 - current_price)
            adjusted = max(0.01, base_estimate + bias_adjustment)
        else:
            # Middle range - less bias
            adjusted = base_estimate

        # Confidence based on how extreme the probability is
        # More confident at extremes where bias is well-documented
        if current_price >= 0.95 or current_price <= 0.05:
            confidence = 0.7
        elif current_price >= 0.90 or current_price <= 0.10:
            confidence = 0.5
        else:
            confidence = 0.3

        return adjusted, confidence

    def check_market(self, market: Market) -> FavoriteLongshotOpportunity | None:
        """Check a single market for favorite-longshot opportunity.

        Args:
            market: Market to evaluate.

        Returns:
            FavoriteLongshotOpportunity if found, None otherwise.
        """
        # Check YES outcome (high probability favorite)
        if market.yes_price >= self.min_probability:
            fair_value, confidence = self.estimate_fair_value(market, market.yes_price)
            edge = fair_value - market.yes_price

            if edge >= self.min_edge:
                # Calculate time to resolution
                time_to_resolution = None
                if market.end_date:
                    delta = market.end_date - utc_now()
                    time_to_resolution = delta.total_seconds() / 3600  # hours

                return FavoriteLongshotOpportunity(
                    market=market,
                    side="YES",
                    current_price=market.yes_price,
                    estimated_fair_value=fair_value,
                    edge=edge,
                    confidence=confidence,
                    time_to_resolution=time_to_resolution,
                    liquidity=market.liquidity,
                )

        # Check NO outcome (high probability favorite via low YES price)
        if market.yes_price <= (1 - self.min_probability):
            no_price = market.no_price
            fair_value, confidence = self.estimate_fair_value(market, no_price)
            edge = fair_value - no_price

            if edge >= self.min_edge:
                time_to_resolution = None
                if market.end_date:
                    delta = market.end_date - utc_now()
                    time_to_resolution = delta.total_seconds() / 3600

                return FavoriteLongshotOpportunity(
                    market=market,
                    side="NO",
                    current_price=no_price,
                    estimated_fair_value=fair_value,
                    edge=edge,
                    confidence=confidence,
                    time_to_resolution=time_to_resolution,
                    liquidity=market.liquidity,
                )

        return None

    def scan(self, markets: list[Market]) -> list[FavoriteLongshotOpportunity]:
        """Scan markets for favorite-longshot opportunities.

        Args:
            markets: List of markets to scan.

        Returns:
            List of opportunities sorted by edge descending.
        """
        opportunities = []

        for market in markets:
            if not market.is_active:
                continue

            opp = self.check_market(market)
            if opp:
                opportunities.append(opp)
                logger.info(
                    "favorite_longshot_found",
                    market_id=market.id,
                    platform=market.platform,
                    side=opp.side,
                    price=f"{opp.current_price:.2%}",
                    edge=f"{opp.edge:.2%}",
                    confidence=f"{opp.confidence:.2%}",
                )

        # Sort by edge * confidence (expected value)
        opportunities.sort(
            key=lambda x: x.edge * x.confidence,
            reverse=True,
        )

        logger.info("favorite_longshot_scan_complete", count=len(opportunities))
        return opportunities

    def calculate_position_size(
        self,
        opportunity: FavoriteLongshotOpportunity,
        account_balance: float,
        max_risk_pct: float = 0.05,
    ) -> float:
        """Calculate appropriate position size using Kelly criterion variant.

        Uses a fractional Kelly approach to balance growth vs. risk.

        Args:
            opportunity: The opportunity to size.
            account_balance: Total account balance.
            max_risk_pct: Maximum percentage of account to risk.

        Returns:
            Recommended position size in USD.
        """
        # Kelly fraction: f* = (p*b - q) / b
        # where p = win probability, q = 1-p, b = odds
        p = opportunity.estimated_fair_value
        q = 1 - p
        b = (1 / opportunity.current_price) - 1  # Odds received

        if b <= 0:
            return 0.0

        kelly = (p * b - q) / b

        # Use fractional Kelly (25%) for safety
        fraction = 0.25
        kelly_size = kelly * fraction * account_balance

        # Apply constraints
        max_by_risk = account_balance * max_risk_pct
        max_by_config = self.max_position_usd
        max_by_liquidity = opportunity.liquidity * 0.1 if opportunity.liquidity > 0 else float('inf')

        position_size = min(kelly_size, max_by_risk, max_by_config, max_by_liquidity)

        return max(0, position_size)

    def backtest_calibration(
        self,
        historical_data: list[tuple[float, bool]],
    ) -> dict[str, float]:
        """Calibrate fair value estimates using historical data.

        Args:
            historical_data: List of (market_price, did_resolve_yes) tuples.

        Returns:
            Dictionary of calibration metrics.
        """
        if not historical_data:
            return {"error": "no data"}

        # Group by price buckets
        buckets: dict[str, list[bool]] = {
            "95-100": [],
            "90-95": [],
            "85-90": [],
            "10-15": [],
            "5-10": [],
            "0-5": [],
        }

        for price, resolved_yes in historical_data:
            if price >= 0.95:
                buckets["95-100"].append(resolved_yes)
            elif price >= 0.90:
                buckets["90-95"].append(resolved_yes)
            elif price >= 0.85:
                buckets["85-90"].append(resolved_yes)
            elif price <= 0.05:
                buckets["0-5"].append(resolved_yes)
            elif price <= 0.10:
                buckets["5-10"].append(resolved_yes)
            elif price <= 0.15:
                buckets["10-15"].append(resolved_yes)

        # Calculate actual resolution rates
        calibration = {}
        for bucket, outcomes in buckets.items():
            if outcomes:
                actual_rate = sum(outcomes) / len(outcomes)
                calibration[f"{bucket}_actual"] = actual_rate
                calibration[f"{bucket}_count"] = len(outcomes)

        return calibration
