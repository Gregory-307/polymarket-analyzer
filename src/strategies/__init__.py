"""Trading strategies for prediction markets."""

from .single_arb import SingleConditionArbitrage
from .multi_arb import MultiOutcomeArbitrage
from .favorite_longshot import FavoriteLongshotStrategy
from .cross_platform import CrossPlatformStrategy
from .financial_markets import FinancialMarketsStrategy

__all__ = [
    "SingleConditionArbitrage",
    "MultiOutcomeArbitrage",
    "FavoriteLongshotStrategy",
    "CrossPlatformStrategy",
    "FinancialMarketsStrategy",
]
