"""Calibration analysis for edge validation.

Analyzes the relationship between market prices and actual outcomes
to identify systematic mispricings (edge).

Key metrics:
- Calibration curve: actual win rate vs. implied probability by bucket
- Brier score: probabilistic forecasting accuracy
- Statistical significance tests
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
import math

from ..core.utils import get_logger
from ..storage.database import Database

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


@dataclass
class CalibrationBucket:
    """Single bucket in calibration analysis.

    Attributes:
        price_low: Lower bound of price bucket.
        price_high: Upper bound of price bucket.
        avg_price: Actual average price in this bucket.
        actual_rate: Actual win rate in this bucket.
        sample_size: Number of markets in bucket.
        edge: actual_rate - avg_price (positive = underpriced).
        std_error: Standard error of actual_rate.
        significant: Whether edge is statistically significant.
    """

    price_low: float
    price_high: float
    avg_price: float
    actual_rate: float
    sample_size: int
    edge: float
    std_error: float
    significant: bool


@dataclass
class CalibrationResult:
    """Complete calibration analysis result.

    Attributes:
        buckets: List of calibration buckets.
        total_markets: Total markets analyzed.
        brier_score: Overall Brier score (lower is better).
        mean_edge: Average edge across all buckets.
        significant_edge: Whether overall edge is significant.
    """

    buckets: list[CalibrationBucket]
    total_markets: int
    brier_score: float
    mean_edge: float
    significant_edge: bool


class CalibrationAnalyzer:
    """Analyzes calibration of prediction market prices.

    Uses resolved market outcomes to determine if prices accurately
    reflect true probabilities or if there's systematic edge.

    Usage:
        async with Database() as db:
            analyzer = CalibrationAnalyzer(db)
            result = await analyzer.analyze()

            for bucket in result.buckets:
                print(f"{bucket.price_low:.0%}-{bucket.price_high:.0%}: "
                      f"implied={bucket.implied_prob:.1%}, actual={bucket.actual_rate:.1%}, "
                      f"edge={bucket.edge:+.1%} ({'*' if bucket.significant else ''})")
    """

    def __init__(
        self,
        database: Database,
        bucket_width: float = 0.05,  # 5% buckets
        min_sample_size: int = 20,
        significance_level: float = 0.05,
    ):
        """Initialize calibration analyzer.

        Args:
            database: Database instance with resolution data.
            bucket_width: Width of probability buckets (default: 5%).
            min_sample_size: Minimum samples for bucket to be valid.
            significance_level: Alpha for statistical significance.
        """
        self.db = database
        self.bucket_width = bucket_width
        self.min_sample_size = min_sample_size
        self.significance_level = significance_level

    async def analyze(
        self,
        platform: str | None = None,
        min_price: float = 0.0,
        max_price: float = 1.0,
    ) -> CalibrationResult:
        """Run calibration analysis.

        Args:
            platform: Filter by platform (optional).
            min_price: Minimum price to include.
            max_price: Maximum price to include.

        Returns:
            CalibrationResult with bucket-level and aggregate metrics.
        """
        # Fetch resolutions with final prices
        resolutions = await self.db.get_resolutions(
            platform=platform,
            limit=100000,
        )

        if not resolutions:
            return CalibrationResult(
                buckets=[],
                total_markets=0,
                brier_score=0.0,
                mean_edge=0.0,
                significant_edge=False,
            )

        # Filter to markets with valid final prices
        markets = [
            {
                "price": r["final_price"],
                "outcome": 1 if r["outcome"] == "YES" else 0,
            }
            for r in resolutions
            if r.get("final_price") is not None
            and min_price <= r["final_price"] <= max_price
        ]

        if not markets:
            return CalibrationResult(
                buckets=[],
                total_markets=0,
                brier_score=0.0,
                mean_edge=0.0,
                significant_edge=False,
            )

        # Create buckets
        buckets = self._create_buckets(markets)

        # Calculate Brier score
        brier_score = self._calculate_brier_score(markets)

        # Calculate overall edge significance
        mean_edge = sum(b.edge * b.sample_size for b in buckets) / sum(b.sample_size for b in buckets) if buckets else 0
        significant_edge = any(b.significant for b in buckets)

        return CalibrationResult(
            buckets=buckets,
            total_markets=len(markets),
            brier_score=brier_score,
            mean_edge=mean_edge,
            significant_edge=significant_edge,
        )

    def _create_buckets(self, markets: list[dict]) -> list[CalibrationBucket]:
        """Create calibration buckets from market data.

        Args:
            markets: List of {"price": float, "outcome": int} dicts.

        Returns:
            List of CalibrationBucket objects.
        """
        buckets = []
        price_low = 0.0

        while price_low < 1.0:
            price_high = min(price_low + self.bucket_width, 1.0)

            # Get markets in this bucket
            bucket_markets = [
                m for m in markets
                if price_low <= m["price"] < price_high
            ]

            if len(bucket_markets) >= self.min_sample_size:
                # Calculate statistics using actual average price (not bucket midpoint)
                avg_price = sum(m["price"] for m in bucket_markets) / len(bucket_markets)
                wins = sum(m["outcome"] for m in bucket_markets)
                actual_rate = wins / len(bucket_markets)
                edge = actual_rate - avg_price

                # Standard error for proportion
                std_error = math.sqrt(
                    actual_rate * (1 - actual_rate) / len(bucket_markets)
                ) if actual_rate > 0 and actual_rate < 1 else 0

                # Z-score for significance
                z_score = abs(edge) / std_error if std_error > 0 else 0
                significant = z_score > 1.96  # 95% confidence

                buckets.append(CalibrationBucket(
                    price_low=price_low,
                    price_high=price_high,
                    avg_price=avg_price,
                    actual_rate=actual_rate,
                    sample_size=len(bucket_markets),
                    edge=edge,
                    std_error=std_error,
                    significant=significant,
                ))

            price_low = price_high

        return buckets

    def _calculate_brier_score(self, markets: list[dict]) -> float:
        """Calculate Brier score for probabilistic forecasting accuracy.

        Brier score = mean((forecast - outcome)^2)
        Perfect forecasting = 0, random guessing = 0.25

        Args:
            markets: List of {"price": float, "outcome": int} dicts.

        Returns:
            Brier score (lower is better).
        """
        if not markets:
            return 0.0

        total = sum(
            (m["price"] - m["outcome"]) ** 2
            for m in markets
        )
        return total / len(markets)

    async def analyze_by_category(
        self,
        platform: str | None = None,
    ) -> dict[str, CalibrationResult]:
        """Run calibration analysis broken down by market category.

        Args:
            platform: Filter by platform (optional).

        Returns:
            Dictionary mapping category to CalibrationResult.
        """
        # Get resolutions
        resolutions = await self.db.get_resolutions(
            platform=platform,
            limit=100000,
        )

        # Get market categories from database
        categories: dict[str, list[dict]] = {}

        for r in resolutions:
            if r.get("final_price") is None:
                continue

            market = await self.db.get_market(r["market_id"])
            category = market.get("category", "unknown") if market else "unknown"

            if category not in categories:
                categories[category] = []

            categories[category].append({
                "price": r["final_price"],
                "outcome": 1 if r["outcome"] == "YES" else 0,
            })

        # Analyze each category
        results = {}
        for category, markets in categories.items():
            if len(markets) >= self.min_sample_size:
                buckets = self._create_buckets(markets)
                brier_score = self._calculate_brier_score(markets)
                mean_edge = (
                    sum(b.edge * b.sample_size for b in buckets) /
                    sum(b.sample_size for b in buckets)
                ) if buckets else 0

                results[category] = CalibrationResult(
                    buckets=buckets,
                    total_markets=len(markets),
                    brier_score=brier_score,
                    mean_edge=mean_edge,
                    significant_edge=any(b.significant for b in buckets),
                )

        return results

    async def analyze_favorite_longshot(
        self,
        high_prob_threshold: float = 0.90,
    ) -> dict:
        """Analyze favorite-longshot bias specifically.

        Focuses on high-probability markets to validate the
        favorite-longshot strategy edge.

        Args:
            high_prob_threshold: Threshold for "high probability" (default: 90%).

        Returns:
            Dictionary with favorite-longshot specific analysis.
        """
        resolutions = await self.db.get_resolutions(limit=100000)

        # Filter to high-probability markets
        high_prob_markets = [
            {
                "price": r["final_price"],
                "outcome": 1 if r["outcome"] == "YES" else 0,
            }
            for r in resolutions
            if r.get("final_price") is not None
            and r["final_price"] >= high_prob_threshold
        ]

        if len(high_prob_markets) < self.min_sample_size:
            return {
                "sample_size": len(high_prob_markets),
                "error": "Insufficient sample size for analysis",
            }

        # Calculate metrics
        wins = sum(m["outcome"] for m in high_prob_markets)
        avg_price = sum(m["price"] for m in high_prob_markets) / len(high_prob_markets)
        actual_rate = wins / len(high_prob_markets)
        implied_rate = avg_price
        edge = actual_rate - implied_rate

        # Standard error
        std_error = math.sqrt(
            actual_rate * (1 - actual_rate) / len(high_prob_markets)
        ) if 0 < actual_rate < 1 else 0

        # Confidence interval
        ci_low = actual_rate - 1.96 * std_error
        ci_high = actual_rate + 1.96 * std_error

        # Significance test
        z_score = edge / std_error if std_error > 0 else 0
        significant = abs(z_score) > 1.96

        return {
            "sample_size": len(high_prob_markets),
            "avg_price": avg_price,
            "implied_rate": implied_rate,
            "actual_rate": actual_rate,
            "edge": edge,
            "std_error": std_error,
            "confidence_interval": (ci_low, ci_high),
            "z_score": z_score,
            "significant": significant,
            "interpretation": self._interpret_fl_result(edge, significant),
        }

    def _interpret_fl_result(self, edge: float, significant: bool) -> str:
        """Generate interpretation of favorite-longshot analysis.

        Args:
            edge: Measured edge (actual_rate - implied_rate).
            significant: Whether edge is statistically significant.

        Returns:
            Human-readable interpretation.
        """
        if not significant:
            return (
                "No statistically significant favorite-longshot bias detected. "
                "Market prices appear well-calibrated for high-probability outcomes."
            )

        if edge > 0:
            return (
                f"Significant favorite-longshot bias detected: +{edge:.1%} edge. "
                f"High-probability markets resolve YES more often than prices imply. "
                f"This supports buying favorites at these prices."
            )
        else:
            return (
                f"Reverse favorite-longshot bias: {edge:.1%} edge. "
                f"High-probability markets resolve YES LESS often than prices imply. "
                f"Favorites may be overpriced, not underpriced."
            )

    async def analyze_by_time_to_resolution(
        self,
        platform: str | None = None,
        time_buckets_hours: list[float] | None = None,
    ) -> dict[str, dict]:
        """Analyze how edge varies by time remaining until resolution.

        Tests whether edge decays as resolution approaches (informed traders
        push prices to fair value near expiry).

        Args:
            platform: Filter by platform (optional).
            time_buckets_hours: Time bucket boundaries in hours (default: 24h, 72h, 168h, 720h).

        Returns:
            Dictionary mapping time bucket to calibration metrics.
        """
        if time_buckets_hours is None:
            # Default: 1 day, 3 days, 1 week, 1 month
            time_buckets_hours = [24, 72, 168, 720]

        resolutions = await self.db.get_resolutions(
            platform=platform,
            limit=100000,
        )

        # Group by time-to-resolution at snapshot time
        buckets: dict[str, list[dict]] = {
            f"<{time_buckets_hours[0]}h": [],
        }
        for i, threshold in enumerate(time_buckets_hours[:-1]):
            next_threshold = time_buckets_hours[i + 1]
            buckets[f"{threshold}-{next_threshold}h"] = []
        buckets[f">{time_buckets_hours[-1]}h"] = []

        for r in resolutions:
            if r.get("final_price") is None:
                continue
            if r.get("time_to_resolution_hours") is None:
                continue

            hours = r["time_to_resolution_hours"]
            market_data = {
                "price": r["final_price"],
                "outcome": 1 if r["outcome"] == "YES" else 0,
            }

            # Find correct bucket
            if hours < time_buckets_hours[0]:
                buckets[f"<{time_buckets_hours[0]}h"].append(market_data)
            elif hours > time_buckets_hours[-1]:
                buckets[f">{time_buckets_hours[-1]}h"].append(market_data)
            else:
                for i, threshold in enumerate(time_buckets_hours[:-1]):
                    next_threshold = time_buckets_hours[i + 1]
                    if threshold <= hours < next_threshold:
                        buckets[f"{threshold}-{next_threshold}h"].append(market_data)
                        break

        # Analyze each time bucket
        results = {}
        for bucket_name, markets in buckets.items():
            if len(markets) < self.min_sample_size:
                results[bucket_name] = {
                    "sample_size": len(markets),
                    "error": "Insufficient sample size",
                }
                continue

            # Calculate metrics
            wins = sum(m["outcome"] for m in markets)
            avg_price = sum(m["price"] for m in markets) / len(markets)
            actual_rate = wins / len(markets)
            edge = actual_rate - avg_price

            # Standard error
            std_error = math.sqrt(
                actual_rate * (1 - actual_rate) / len(markets)
            ) if 0 < actual_rate < 1 else 0

            # Significance
            z_score = edge / std_error if std_error > 0 else 0
            significant = abs(z_score) > 1.96

            results[bucket_name] = {
                "sample_size": len(markets),
                "avg_price": avg_price,
                "actual_rate": actual_rate,
                "edge": edge,
                "std_error": std_error,
                "z_score": z_score,
                "significant": significant,
            }

        return results


def print_calibration_report(result: CalibrationResult) -> None:
    """Print a formatted calibration report.

    Args:
        result: CalibrationResult from analyze().
    """
    print("\n" + "=" * 70)
    print("  CALIBRATION ANALYSIS")
    print("=" * 70)

    print(f"\n  Total Markets: {result.total_markets}")
    print(f"  Brier Score: {result.brier_score:.4f} (perfect=0, random=0.25)")
    print(f"  Mean Edge: {result.mean_edge:+.2%}")

    print("\n" + "-" * 70)
    print("  BUCKET ANALYSIS")
    print("-" * 70)

    print(f"\n  {'Price Range':<15} {'Implied':>10} {'Actual':>10} {'Edge':>10} {'N':>8} {'Sig':>5}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*5}")

    for b in result.buckets:
        sig = "*" if b.significant else ""
        print(
            f"  {b.price_low:.0%}-{b.price_high:.0%}  "
            f"{b.implied_prob:>10.1%} "
            f"{b.actual_rate:>10.1%} "
            f"{b.edge:>+10.1%} "
            f"{b.sample_size:>8} "
            f"{sig:>5}"
        )

    print("\n" + "-" * 70)
    print("  * = statistically significant at 95% confidence")
