"""Kelly Criterion implementation for optimal position sizing.

The Kelly Criterion determines the optimal fraction of bankroll to bet:
    f* = (bp - q) / b

Where:
    f* = fraction of bankroll to bet
    b = odds received (payout-to-stake ratio)
    p = probability of winning
    q = probability of losing (1 - p)

For prediction markets:
    If buying YES at price p_yes, payout on win is $1 per share
    b = (1 - p_yes) / p_yes  (odds = profit / stake)
    p = true probability of YES
    q = 1 - p

Edge = p - p_yes (true probability minus market price)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import TYPE_CHECKING

from ..core.utils import get_logger

if TYPE_CHECKING:
    from ..analysis.calibration import CalibrationResult

logger = get_logger(__name__)


class KellyFraction(Enum):
    """Standard Kelly fractions for risk management."""

    FULL = 1.0
    HALF = 0.5
    QUARTER = 0.25
    EIGHTH = 0.125


@dataclass
class KellyBet:
    """Result of Kelly calculation.

    Attributes:
        fraction: Optimal fraction of bankroll to bet.
        bet_size: Dollar amount to bet given bankroll.
        edge: Estimated edge (true_prob - market_price).
        odds: Odds received (payout ratio).
        expected_growth: Expected log growth rate.
        confidence: Confidence in the edge estimate.
    """

    fraction: float
    bet_size: float
    edge: float
    odds: float
    expected_growth: float
    confidence: float = 1.0


def kelly_bet_size(
    edge: float,
    odds: float,
    bankroll: float,
    fraction: float = 1.0,
) -> KellyBet:
    """Calculate Kelly bet size.

    Args:
        edge: True probability minus market price (e.g., 0.02 = 2% edge).
        odds: Payout ratio (profit / stake). For price p: odds = (1-p)/p.
        bankroll: Current bankroll in dollars.
        fraction: Kelly fraction to use (0.5 = half Kelly).

    Returns:
        KellyBet with sizing details.

    Example:
        # Market priced at 90%, true probability 92%
        price = 0.90
        true_prob = 0.92
        edge = true_prob - price  # 0.02
        odds = (1 - price) / price  # 0.111 (10:1 odds)

        result = kelly_bet_size(edge=0.02, odds=0.111, bankroll=1000)
        # result.fraction ≈ 0.18 (18% of bankroll)
        # result.bet_size ≈ $180
    """
    # Kelly formula: f* = (p * b - q) / b
    # Where p = true_prob, q = 1-p, b = odds
    # Rearranging with edge: f* = edge / odds (for small edge approximation)
    # More precisely: f* = p - q/b = edge + price - (1-price-edge)/(odds)

    # Derive true probability from edge and odds
    # odds = (1-price)/price => price = 1/(1+odds)
    implied_price = 1 / (1 + odds) if odds > 0 else 0.5
    true_prob = implied_price + edge

    # Ensure valid probability
    true_prob = max(0.001, min(0.999, true_prob))
    false_prob = 1 - true_prob

    # Full Kelly formula
    kelly_fraction = (true_prob * odds - false_prob) / odds if odds > 0 else 0

    # Apply fractional Kelly
    kelly_fraction *= fraction

    # Clamp to [0, 1]
    kelly_fraction = max(0, min(1, kelly_fraction))

    # Calculate bet size
    bet_size = bankroll * kelly_fraction

    # Expected growth rate: E[log(1 + f*b*X)] where X is outcome
    if kelly_fraction > 0 and odds > 0:
        # Growth = p*log(1 + f*b) + q*log(1 - f)
        win_return = 1 + kelly_fraction * odds
        loss_return = 1 - kelly_fraction
        expected_growth = (
            true_prob * math.log(win_return)
            + false_prob * math.log(loss_return)
        ) if loss_return > 0 and win_return > 0 else 0
    else:
        expected_growth = 0

    return KellyBet(
        fraction=kelly_fraction,
        bet_size=bet_size,
        edge=edge,
        odds=odds,
        expected_growth=expected_growth,
    )


def fractional_kelly(
    edge: float,
    price: float,
    bankroll: float,
    kelly_fraction: KellyFraction = KellyFraction.HALF,
) -> KellyBet:
    """Convenience function for fractional Kelly with price input.

    Args:
        edge: True probability minus market price.
        price: Market price (0-1).
        bankroll: Current bankroll in dollars.
        kelly_fraction: Fraction of Kelly to use.

    Returns:
        KellyBet with sizing details.
    """
    # Convert price to odds
    odds = (1 - price) / price if price > 0 else 0

    return kelly_bet_size(
        edge=edge,
        odds=odds,
        bankroll=bankroll,
        fraction=kelly_fraction.value,
    )


class KellyCriterion:
    """Kelly criterion calculator with calibration-based edge estimation.

    Uses calibration data to estimate edge rather than assuming a fixed edge.

    Usage:
        kelly = KellyCriterion(calibration_result)

        # For a market priced at 95%
        bet = kelly.calculate(price=0.95, bankroll=1000)
        print(f"Bet ${bet.bet_size:.2f} (edge: {bet.edge:.1%})")
    """

    def __init__(
        self,
        calibration: "CalibrationResult | None" = None,
        default_edge: float = 0.0,
        min_edge_for_bet: float = 0.005,  # 0.5% minimum edge
        fraction: KellyFraction = KellyFraction.HALF,
    ):
        """Initialize Kelly calculator.

        Args:
            calibration: Calibration result for edge estimation.
            default_edge: Default edge if no calibration data.
            min_edge_for_bet: Minimum edge to recommend a bet.
            fraction: Kelly fraction to use.
        """
        self.calibration = calibration
        self.default_edge = default_edge
        self.min_edge_for_bet = min_edge_for_bet
        self.fraction = fraction

        # Build edge lookup from calibration buckets
        self._edge_by_price: dict[tuple[float, float], tuple[float, float]] = {}
        if calibration:
            for bucket in calibration.buckets:
                key = (bucket.price_low, bucket.price_high)
                # (edge, confidence based on significance)
                confidence = 0.9 if bucket.significant else 0.5
                self._edge_by_price[key] = (bucket.edge, confidence)

    def estimate_edge(self, price: float) -> tuple[float, float]:
        """Estimate edge for a given price from calibration data.

        Args:
            price: Market price (0-1).

        Returns:
            Tuple of (edge, confidence).
        """
        # Find matching bucket
        for (low, high), (edge, confidence) in self._edge_by_price.items():
            if low <= price < high:
                return (edge, confidence)

        # No calibration data for this price
        return (self.default_edge, 0.1)

    def calculate(
        self,
        price: float,
        bankroll: float,
        edge_override: float | None = None,
    ) -> KellyBet:
        """Calculate Kelly bet size.

        Args:
            price: Market price (0-1).
            bankroll: Current bankroll in dollars.
            edge_override: Override estimated edge (optional).

        Returns:
            KellyBet with sizing details.
        """
        if edge_override is not None:
            edge = edge_override
            confidence = 1.0
        else:
            edge, confidence = self.estimate_edge(price)

        # No bet if edge below threshold
        if edge < self.min_edge_for_bet:
            return KellyBet(
                fraction=0,
                bet_size=0,
                edge=edge,
                odds=(1 - price) / price if price > 0 else 0,
                expected_growth=0,
                confidence=confidence,
            )

        # Calculate Kelly with confidence adjustment
        odds = (1 - price) / price if price > 0 else 0

        result = kelly_bet_size(
            edge=edge,
            odds=odds,
            bankroll=bankroll,
            fraction=self.fraction.value * confidence,  # Scale by confidence
        )

        result.confidence = confidence
        return result

    def calculate_for_opportunity(
        self,
        market_price: float,
        side: str,
        bankroll: float,
    ) -> KellyBet:
        """Calculate Kelly for a trading opportunity.

        Args:
            market_price: Current market price of the side.
            side: 'YES' or 'NO'.
            bankroll: Current bankroll.

        Returns:
            KellyBet with sizing details.
        """
        # For YES side, use price directly
        # For NO side, effective price is (1 - market_price)
        if side.upper() == "NO":
            effective_price = 1 - market_price
        else:
            effective_price = market_price

        return self.calculate(price=effective_price, bankroll=bankroll)


def simulate_kelly_performance(
    edge: float,
    price: float,
    kelly_fractions: list[float],
    num_bets: int = 1000,
    num_simulations: int = 100,
) -> dict[float, dict]:
    """Simulate Kelly performance over multiple bets.

    Args:
        edge: True edge (true_prob - price).
        price: Market price.
        kelly_fractions: List of Kelly fractions to compare.
        num_bets: Number of bets per simulation.
        num_simulations: Number of simulations to run.

    Returns:
        Dictionary mapping Kelly fraction to performance metrics.
    """
    import random

    true_prob = price + edge
    results = {}

    for fraction in kelly_fractions:
        final_values = []
        max_drawdowns = []

        for _ in range(num_simulations):
            bankroll = 1.0
            peak = 1.0
            max_dd = 0.0

            for _ in range(num_bets):
                # Calculate bet using this Kelly fraction
                bet = kelly_bet_size(
                    edge=edge,
                    odds=(1 - price) / price,
                    bankroll=bankroll,
                    fraction=fraction,
                )

                # Simulate outcome
                won = random.random() < true_prob

                if won:
                    bankroll += bet.bet_size * ((1 - price) / price)
                else:
                    bankroll -= bet.bet_size

                # Track drawdown
                if bankroll > peak:
                    peak = bankroll
                dd = (peak - bankroll) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)

                # Bust check
                if bankroll <= 0:
                    break

            final_values.append(bankroll)
            max_drawdowns.append(max_dd)

        # Calculate statistics
        import statistics

        results[fraction] = {
            "mean_final": statistics.mean(final_values),
            "median_final": statistics.median(final_values),
            "std_final": statistics.stdev(final_values) if len(final_values) > 1 else 0,
            "mean_max_dd": statistics.mean(max_drawdowns),
            "bust_rate": sum(1 for v in final_values if v <= 0.01) / len(final_values),
        }

    return results
