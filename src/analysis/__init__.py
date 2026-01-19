"""Analysis module for edge validation and performance attribution.

This module provides tools for:
- Resolution tracking (market outcomes)
- Calibration analysis (actual vs. predicted)
- Historical backtesting with real data
- Performance attribution (luck vs. skill)
"""

from .resolution_tracker import ResolutionTracker
from .calibration import CalibrationAnalyzer
from .backtester import HistoricalBacktester, BacktestResult

__all__ = [
    "ResolutionTracker",
    "CalibrationAnalyzer",
    "HistoricalBacktester",
    "BacktestResult",
]
