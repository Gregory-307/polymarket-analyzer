"""Position sizing module.

This module provides:
- Kelly criterion for optimal bet sizing
- Fractional Kelly (half, quarter) for risk management
- Confidence-weighted Kelly based on statistical significance
"""

from .kelly import (
    KellyCriterion,
    KellyFraction,
    kelly_bet_size,
    fractional_kelly,
)

__all__ = [
    "KellyCriterion",
    "KellyFraction",
    "kelly_bet_size",
    "fractional_kelly",
]
