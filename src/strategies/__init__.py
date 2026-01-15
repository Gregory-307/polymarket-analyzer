"""Trading strategies for prediction markets."""

from .single_arb import SingleConditionArbitrage
from .multi_arb import MultiOutcomeArbitrage
from .favorite_longshot import FavoriteLongshotStrategy

__all__ = [
    "SingleConditionArbitrage",
    "MultiOutcomeArbitrage",
    "FavoriteLongshotStrategy",
]
