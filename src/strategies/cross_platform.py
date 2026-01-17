"""Cross-Platform Arbitrage Strategy.

Finds price discrepancies between Polymarket and Kalshi for the same events.

Unlike sum arbitrage, this is NOT mathematically risk-free. Settlement differences,
timing, and execution create real (though manageable) risk.

Research basis:
- arXiv:2508.03474: $40M+ extracted via cross-platform arbitrage
- Returns of 2-8% per trade documented
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

from ..core.utils import get_logger

if TYPE_CHECKING:
    from ..adapters.base import Market

logger = get_logger(__name__)


@dataclass
class CrossPlatformOpportunity:
    """Represents a cross-platform arbitrage opportunity.

    Attributes:
        poly_market: Market on Polymarket.
        kalshi_market: Market on Kalshi.
        poly_yes_price: YES price on Polymarket.
        kalshi_yes_price: YES price on Kalshi.
        spread: Absolute price difference.
        direction: 'buy_poly' or 'buy_kalshi' - where to buy YES.
        match_confidence: How confident we are markets are the same event (0-1).
    """

    poly_market: Market
    kalshi_market: Market
    poly_yes_price: float
    kalshi_yes_price: float
    spread: float
    direction: str
    match_confidence: float

    @property
    def spread_pct(self) -> float:
        """Spread as percentage."""
        return self.spread * 100

    @property
    def potential_profit_pct(self) -> float:
        """Potential profit percentage (before fees/slippage)."""
        # Buy cheap YES, sell expensive YES (or buy cheap YES + cheap NO on other platform)
        return self.spread * 100


class CrossPlatformStrategy:
    """Strategy for cross-platform arbitrage between Polymarket and Kalshi.

    Matches markets across platforms and identifies price discrepancies.

    Example:
        ```python
        strategy = CrossPlatformStrategy(min_spread=0.02, min_match_confidence=0.8)

        poly_markets = await polymarket.get_markets()
        kalshi_markets = await kalshi.get_markets()

        opportunities = strategy.find_opportunities(poly_markets, kalshi_markets)
        ```
    """

    def __init__(
        self,
        min_spread: float = 0.02,
        min_match_confidence: float = 0.7,
        min_liquidity: float = 1000,
    ):
        """Initialize strategy.

        Args:
            min_spread: Minimum price spread to report (default 2%).
            min_match_confidence: Minimum confidence that markets match (0-1).
            min_liquidity: Minimum liquidity on both platforms.
        """
        self.min_spread = min_spread
        self.min_match_confidence = min_match_confidence
        self.min_liquidity = min_liquidity

    def normalize_question(self, question: str) -> str:
        """Normalize a market question for comparison.

        Removes noise like punctuation, dates, extra whitespace.

        Args:
            question: Raw market question.

        Returns:
            Normalized question string.
        """
        # Lowercase
        q = question.lower()

        # Remove common prefixes/suffixes
        q = re.sub(r"^(will |what |who |when |where |how )", "", q)

        # Remove dates in various formats
        q = re.sub(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*\d{1,2},?\s*\d{0,4}\b", "", q)
        q = re.sub(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", "", q)
        q = re.sub(r"\b\d{4}\b", "", q)  # Years

        # Remove punctuation
        q = re.sub(r"[^\w\s]", " ", q)

        # Normalize whitespace
        q = " ".join(q.split())

        return q.strip()

    def calculate_match_confidence(
        self,
        poly_market: Market,
        kalshi_market: Market,
    ) -> float:
        """Calculate confidence that two markets refer to the same event.

        Uses fuzzy string matching on questions and other heuristics.

        Args:
            poly_market: Market from Polymarket.
            kalshi_market: Market from Kalshi.

        Returns:
            Confidence score from 0 to 1.
        """
        # Normalize questions
        poly_q = self.normalize_question(poly_market.question)
        kalshi_q = self.normalize_question(kalshi_market.question)

        # Base similarity from sequence matching
        similarity = SequenceMatcher(None, poly_q, kalshi_q).ratio()

        # Boost if key terms match
        poly_words = set(poly_q.split())
        kalshi_words = set(kalshi_q.split())

        # Key political/sports terms
        key_terms = {
            "trump", "biden", "harris", "election", "president",
            "fed", "rate", "fomc", "inflation",
            "btc", "bitcoin", "eth", "ethereum", "crypto",
            "super", "bowl", "nfl", "nba", "world", "cup",
        }

        common_key_terms = poly_words & kalshi_words & key_terms
        if common_key_terms:
            similarity += 0.1 * len(common_key_terms)

        # Check category match (if available)
        if poly_market.category and kalshi_market.category:
            if poly_market.category.lower() == kalshi_market.category.lower():
                similarity += 0.1

        # Cap at 1.0
        return min(1.0, similarity)

    def match_markets(
        self,
        poly_markets: list[Market],
        kalshi_markets: list[Market],
    ) -> list[tuple[Market, Market, float]]:
        """Find matching markets between platforms.

        Args:
            poly_markets: Markets from Polymarket.
            kalshi_markets: Markets from Kalshi.

        Returns:
            List of (poly_market, kalshi_market, confidence) tuples.
        """
        matches = []

        for poly in poly_markets:
            if not poly.is_active:
                continue

            best_match = None
            best_confidence = 0.0

            for kalshi in kalshi_markets:
                if not kalshi.is_active:
                    continue

                confidence = self.calculate_match_confidence(poly, kalshi)

                if confidence > best_confidence and confidence >= self.min_match_confidence:
                    best_confidence = confidence
                    best_match = kalshi

            if best_match:
                matches.append((poly, best_match, best_confidence))
                logger.debug(
                    "cross_platform_match",
                    poly_question=poly.question[:50],
                    kalshi_question=best_match.question[:50],
                    confidence=f"{best_confidence:.2%}",
                )

        logger.info("cross_platform_matches_found", count=len(matches))
        return matches

    def check_opportunity(
        self,
        poly_market: Market,
        kalshi_market: Market,
        match_confidence: float,
    ) -> CrossPlatformOpportunity | None:
        """Check if a matched pair has an arbitrage opportunity.

        Args:
            poly_market: Market from Polymarket.
            kalshi_market: Market from Kalshi.
            match_confidence: Confidence the markets match.

        Returns:
            CrossPlatformOpportunity if spread exceeds threshold.
        """
        spread = abs(poly_market.yes_price - kalshi_market.yes_price)

        if spread < self.min_spread:
            return None

        # Determine direction
        if poly_market.yes_price < kalshi_market.yes_price:
            direction = "buy_poly"  # Buy YES on Poly, sell YES on Kalshi
        else:
            direction = "buy_kalshi"  # Buy YES on Kalshi, sell YES on Poly

        return CrossPlatformOpportunity(
            poly_market=poly_market,
            kalshi_market=kalshi_market,
            poly_yes_price=poly_market.yes_price,
            kalshi_yes_price=kalshi_market.yes_price,
            spread=spread,
            direction=direction,
            match_confidence=match_confidence,
        )

    def find_opportunities(
        self,
        poly_markets: list[Market],
        kalshi_markets: list[Market],
    ) -> list[CrossPlatformOpportunity]:
        """Find cross-platform arbitrage opportunities.

        Args:
            poly_markets: Markets from Polymarket.
            kalshi_markets: Markets from Kalshi.

        Returns:
            List of opportunities sorted by spread descending.
        """
        # First, match markets
        matches = self.match_markets(poly_markets, kalshi_markets)

        # Then check each match for opportunities
        opportunities = []
        for poly, kalshi, confidence in matches:
            opp = self.check_opportunity(poly, kalshi, confidence)
            if opp:
                opportunities.append(opp)
                logger.info(
                    "cross_platform_opportunity",
                    poly_id=poly.id,
                    kalshi_id=kalshi.id,
                    poly_price=f"{poly.yes_price:.2%}",
                    kalshi_price=f"{kalshi.yes_price:.2%}",
                    spread=f"{opp.spread:.2%}",
                    direction=opp.direction,
                )

        # Sort by spread
        opportunities.sort(key=lambda x: x.spread, reverse=True)

        logger.info("cross_platform_scan_complete", opportunities=len(opportunities))
        return opportunities

    def scan(
        self,
        poly_markets: list[Market],
        kalshi_markets: list[Market],
    ) -> list[CrossPlatformOpportunity]:
        """Alias for find_opportunities for consistent API."""
        return self.find_opportunities(poly_markets, kalshi_markets)
