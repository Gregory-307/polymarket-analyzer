"""Favorite-Longshot Bias Analysis.

Quantifies the magnitude of favorite-longshot bias in historical data
and validates the edge before live trading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class CalibrationResult:
    """Calibration analysis for a probability bucket.

    Attributes:
        bucket: Price range (e.g., "95-100").
        market_price_avg: Average market price in bucket.
        actual_resolution_rate: Actual YES resolution rate.
        count: Number of markets in bucket.
        bias: Difference (actual - market), positive = underpriced.
        edge: Estimated tradeable edge after costs.
    """

    bucket: str
    market_price_avg: float
    actual_resolution_rate: float
    count: int
    bias: float
    edge: float


class BiasAnalyzer:
    """Analyzes favorite-longshot bias in historical prediction market data.

    Compares market prices (implied probabilities) to actual resolution rates
    to quantify systematic mispricings.

    Research basis: Kahneman & Tversky Prospect Theory

    Example:
        ```python
        analyzer = BiasAnalyzer()
        analyzer.load_data(historical_markets)

        # Full analysis
        results = analyzer.analyze()

        for bucket in results:
            print(f"{bucket.bucket}: Market={bucket.market_price_avg:.1%}, "
                  f"Actual={bucket.actual_resolution_rate:.1%}, "
                  f"Bias={bucket.bias:+.1%}")

        # Get edge estimate
        edge = analyzer.estimate_edge(price_threshold=0.95)
        print(f"Estimated edge at 95%+: {edge:.2%}")
        ```
    """

    BUCKETS = [
        ("0-5", 0.00, 0.05),
        ("5-10", 0.05, 0.10),
        ("10-20", 0.10, 0.20),
        ("20-30", 0.20, 0.30),
        ("30-40", 0.30, 0.40),
        ("40-50", 0.40, 0.50),
        ("50-60", 0.50, 0.60),
        ("60-70", 0.60, 0.70),
        ("70-80", 0.70, 0.80),
        ("80-90", 0.80, 0.90),
        ("90-95", 0.90, 0.95),
        ("95-100", 0.95, 1.00),
    ]

    def __init__(self, transaction_cost: float = 0.001):
        """Initialize analyzer.

        Args:
            transaction_cost: Estimated transaction cost (0.001 = 0.1%).
        """
        self.transaction_cost = transaction_cost
        self._data: pd.DataFrame | None = None

    def load_data(self, data: pd.DataFrame | list[dict]) -> None:
        """Load historical market data.

        Expected columns:
        - market_id: str
        - yes_price: float (0-1)
        - resolution: str ('YES' or 'NO')

        Args:
            data: DataFrame or list of resolved markets.
        """
        if isinstance(data, list):
            self._data = pd.DataFrame(data)
        else:
            self._data = data.copy()

        # Filter to resolved markets only
        self._data = self._data[self._data["resolution"].isin(["YES", "NO"])]

        # Add binary resolution column
        self._data["resolved_yes"] = self._data["resolution"] == "YES"

    def analyze(self) -> list[CalibrationResult]:
        """Analyze bias across all probability buckets.

        Returns:
            List of CalibrationResult for each bucket.
        """
        if self._data is None or len(self._data) == 0:
            return []

        results = []

        for bucket_name, low, high in self.BUCKETS:
            # Filter to bucket
            mask = (self._data["yes_price"] >= low) & (self._data["yes_price"] < high)
            bucket_data = self._data[mask]

            if len(bucket_data) == 0:
                continue

            # Calculate metrics
            market_avg = bucket_data["yes_price"].mean()
            actual_rate = bucket_data["resolved_yes"].mean()
            bias = actual_rate - market_avg
            edge = max(0, bias - self.transaction_cost * 2)

            results.append(CalibrationResult(
                bucket=bucket_name,
                market_price_avg=market_avg,
                actual_resolution_rate=actual_rate,
                count=len(bucket_data),
                bias=bias,
                edge=edge,
            ))

        return results

    def estimate_edge(self, price_threshold: float = 0.95) -> float:
        """Estimate tradeable edge for high-probability markets.

        Args:
            price_threshold: Minimum YES price to consider.

        Returns:
            Estimated edge as decimal (e.g., 0.02 = 2%).
        """
        if self._data is None or len(self._data) == 0:
            return 0.0

        # Filter to high probability
        high_prob = self._data[self._data["yes_price"] >= price_threshold]

        if len(high_prob) == 0:
            return 0.0

        market_avg = high_prob["yes_price"].mean()
        actual_rate = high_prob["resolved_yes"].mean()

        bias = actual_rate - market_avg
        edge = max(0, bias - self.transaction_cost * 2)

        return edge

    def estimate_expected_return(
        self,
        price_threshold: float = 0.95,
    ) -> dict[str, float]:
        """Estimate expected return from favorite-longshot strategy.

        Args:
            price_threshold: Minimum probability threshold.

        Returns:
            Dictionary with expected return metrics.
        """
        if self._data is None or len(self._data) == 0:
            return {"error": "no data"}

        high_prob = self._data[self._data["yes_price"] >= price_threshold]

        if len(high_prob) == 0:
            return {"error": "no high-probability markets"}

        # Simulate strategy
        total_invested = 0
        total_payout = 0

        for _, row in high_prob.iterrows():
            price = row["yes_price"]
            won = row["resolved_yes"]

            # Assume $1 bet
            cost = 1.0 + self.transaction_cost
            payout = 1.0 if won else 0.0

            total_invested += cost
            total_payout += payout

        if total_invested == 0:
            return {"error": "no trades"}

        total_return = (total_payout - total_invested) / total_invested
        win_rate = high_prob["resolved_yes"].mean()

        return {
            "threshold": price_threshold,
            "num_markets": len(high_prob),
            "win_rate": win_rate,
            "total_return": total_return,
            "avg_market_price": high_prob["yes_price"].mean(),
            "avg_payout": total_payout / len(high_prob),
        }

    def generate_report(self) -> str:
        """Generate human-readable analysis report.

        Returns:
            Formatted report string.
        """
        results = self.analyze()

        if not results:
            return "No data available for analysis."

        lines = [
            "Favorite-Longshot Bias Analysis",
            "=" * 50,
            "",
            f"{'Bucket':<10} {'Market':>8} {'Actual':>8} {'Bias':>8} {'Edge':>8} {'Count':>8}",
            "-" * 50,
        ]

        for r in results:
            bias_str = f"{r.bias:+.1%}" if r.bias != 0 else "0.0%"
            lines.append(
                f"{r.bucket:<10} {r.market_price_avg:>7.1%} {r.actual_resolution_rate:>7.1%} "
                f"{bias_str:>8} {r.edge:>7.1%} {r.count:>8}"
            )

        lines.extend([
            "",
            "Key Findings:",
            "-" * 50,
        ])

        # Find strongest bias buckets
        high_prob = [r for r in results if r.bucket in ("95-100", "90-95")]
        low_prob = [r for r in results if r.bucket in ("0-5", "5-10")]

        if high_prob:
            avg_bias = sum(r.bias for r in high_prob) / len(high_prob)
            lines.append(f"High-probability (90%+) avg bias: {avg_bias:+.2%}")

        if low_prob:
            avg_bias = sum(r.bias for r in low_prob) / len(low_prob)
            lines.append(f"Low-probability (<10%) avg bias: {avg_bias:+.2%}")

        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Export analysis results as DataFrame.

        Returns:
            DataFrame with calibration results.
        """
        results = self.analyze()
        return pd.DataFrame([
            {
                "bucket": r.bucket,
                "market_price": r.market_price_avg,
                "actual_rate": r.actual_resolution_rate,
                "count": r.count,
                "bias": r.bias,
                "edge": r.edge,
            }
            for r in results
        ])
