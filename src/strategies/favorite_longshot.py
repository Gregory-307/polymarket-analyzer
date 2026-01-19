"""Favorite-Longshot Bias Scanner.

Finds high-probability markets (>= threshold) that research suggests
may be underpriced due to behavioral bias.

Research basis:
- Kahneman & Tversky Prospect Theory (1979)
- Snowberg & Wolfers NBER Working Paper 15923

Enhancements:
- Optional calibration-based edge estimation
- Kelly criterion integration for position sizing
- Liquidity filtering
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..core.utils import get_logger, utc_now

if TYPE_CHECKING:
    from ..adapters.base import Market
    from ..analysis.calibration import CalibrationResult
    from ..sizing.kelly import KellyBet

logger = get_logger(__name__)


@dataclass
class FavoriteLongshotOpportunity:
    """A high-probability market found by the scanner.

    Attributes:
        market: The market object.
        side: 'YES' or 'NO'.
        price: Current market price.
        time_to_resolution: Hours until market closes.
        liquidity: Market liquidity in USD.
        edge: Estimated edge from calibration (if available).
        confidence: Confidence in edge estimate.
        kelly: Kelly criterion sizing (if calculated).
    """

    market: "Market"
    side: str  # 'YES' or 'NO'
    price: float
    time_to_resolution: float | None = None  # hours
    liquidity: float = 0.0
    edge: float | None = None
    confidence: float = 0.0
    kelly: "KellyBet | None" = None

    @property
    def market_name(self) -> str:
        """Get market question/name."""
        return self.market.question

    @property
    def expected_profit(self) -> float:
        """Get expected profit (edge or 0)."""
        return self.edge if self.edge is not None else 0.0

    @property
    def volume(self) -> float:
        """Get market volume."""
        return self.market.volume or 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for reporting."""
        return {
            "market_id": self.market.id,
            "market_name": self.market_name,
            "side": self.side,
            "price": self.price,
            "time_to_resolution_hours": self.time_to_resolution,
            "liquidity": self.liquidity,
            "volume": self.volume,
            "edge": self.edge,
            "confidence": self.confidence,
            "kelly_fraction": self.kelly.fraction if self.kelly else None,
            "kelly_bet_size": self.kelly.bet_size if self.kelly else None,
        }


class FavoriteLongshotStrategy:
    """Scanner for high-probability markets.

    Finds markets where one side has probability >= min_probability.
    Research suggests these may be underpriced.

    Can optionally use calibration data to estimate edge and Kelly criterion
    for position sizing.

    Example:
        # Basic usage (no edge estimation)
        strategy = FavoriteLongshotStrategy(min_probability=0.90)
        opportunities = strategy.scan(markets)

        # With calibration-based edge
        calibration = CalibrationAnalyzer(db).analyze()
        strategy = FavoriteLongshotStrategy(
            min_probability=0.90,
            calibration=calibration,
            bankroll=10000,
        )
        opportunities = strategy.scan(markets)
        for opp in opportunities:
            print(f"{opp.side} @ {opp.price:.1%}, edge={opp.edge:.1%}")
            if opp.kelly:
                print(f"  Kelly: bet ${opp.kelly.bet_size:.2f}")
    """

    def __init__(
        self,
        min_probability: float = 0.90,
        calibration: "CalibrationResult | None" = None,
        bankroll: float | None = None,
        min_liquidity: float = 0.0,
        min_edge: float = 0.0,
    ):
        """Initialize scanner.

        Args:
            min_probability: Minimum probability to consider (default 90%).
            calibration: Calibration result for edge estimation.
            bankroll: Bankroll for Kelly sizing (required for Kelly).
            min_liquidity: Minimum liquidity filter (default: no filter).
            min_edge: Minimum edge to include opportunity (default: no filter).
        """
        self.min_probability = min_probability
        self.calibration = calibration
        self.bankroll = bankroll
        self.min_liquidity = min_liquidity
        self.min_edge = min_edge

        # Setup Kelly calculator if we have calibration
        self._kelly = None
        if calibration:
            from ..sizing.kelly import KellyCriterion, KellyFraction
            self._kelly = KellyCriterion(
                calibration=calibration,
                min_edge_for_bet=0.001,  # Very low threshold, we filter later
                fraction=KellyFraction.HALF,  # Conservative half-Kelly
            )

    def _estimate_edge(self, price: float) -> tuple[float, float]:
        """Estimate edge for a price using calibration data.

        Args:
            price: Market price (0-1).

        Returns:
            Tuple of (edge, confidence).
        """
        if not self._kelly:
            return (0.0, 0.0)

        return self._kelly.estimate_edge(price)

    def check_market(self, market: "Market") -> FavoriteLongshotOpportunity | None:
        """Check if market has a high-probability side.

        Args:
            market: Market to evaluate.

        Returns:
            FavoriteLongshotOpportunity if found, None otherwise.
        """
        # Liquidity filter
        if self.min_liquidity > 0 and (market.liquidity or 0) < self.min_liquidity:
            return None

        time_to_resolution = None
        if market.end_date:
            delta = market.end_date - utc_now()
            time_to_resolution = delta.total_seconds() / 3600

        # Check YES side
        if market.yes_price >= self.min_probability:
            return self._create_opportunity(
                market=market,
                side="YES",
                price=market.yes_price,
                time_to_resolution=time_to_resolution,
            )

        # Check NO side
        if market.no_price >= self.min_probability:
            return self._create_opportunity(
                market=market,
                side="NO",
                price=market.no_price,
                time_to_resolution=time_to_resolution,
            )

        return None

    def _create_opportunity(
        self,
        market: "Market",
        side: str,
        price: float,
        time_to_resolution: float | None,
    ) -> FavoriteLongshotOpportunity | None:
        """Create opportunity with edge and Kelly calculation.

        Args:
            market: The market.
            side: YES or NO.
            price: Market price.
            time_to_resolution: Hours to resolution.

        Returns:
            FavoriteLongshotOpportunity or None if filtered.
        """
        # Estimate edge from calibration
        edge, confidence = self._estimate_edge(price)

        # Edge filter
        if self.min_edge > 0 and edge < self.min_edge:
            return None

        # Calculate Kelly sizing if we have bankroll
        kelly_bet = None
        if self._kelly and self.bankroll and self.bankroll > 0:
            kelly_bet = self._kelly.calculate(
                price=price,
                bankroll=self.bankroll,
                edge_override=edge if edge > 0 else None,
            )

        return FavoriteLongshotOpportunity(
            market=market,
            side=side,
            price=price,
            time_to_resolution=time_to_resolution,
            liquidity=market.liquidity or 0.0,
            edge=edge if edge > 0 else None,
            confidence=confidence,
            kelly=kelly_bet,
        )

    def scan(self, markets: list["Market"]) -> list[FavoriteLongshotOpportunity]:
        """Scan markets for high-probability favorites.

        Args:
            markets: List of markets to scan.

        Returns:
            List of opportunities sorted by edge (if available) or price.
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
                    price=f"{opp.price:.2%}",
                    edge=f"{opp.edge:.2%}" if opp.edge else "N/A",
                )

        # Sort by edge if available, otherwise by price
        if any(opp.edge is not None for opp in opportunities):
            opportunities.sort(key=lambda x: x.edge or 0, reverse=True)
        else:
            opportunities.sort(key=lambda x: x.price, reverse=True)

        logger.info("favorite_longshot_scan_complete", count=len(opportunities))
        return opportunities

    async def analyze(self, market: "Market") -> list[FavoriteLongshotOpportunity]:
        """Async interface for compatibility with run command.

        Args:
            market: Market to analyze.

        Returns:
            List with single opportunity if found, empty list otherwise.
        """
        opp = self.check_market(market)
        return [opp] if opp else []
