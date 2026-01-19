"""Storage module for data persistence.

This module provides SQLite-based storage for market data, snapshots,
trades, and detected opportunities.
"""

from .database import Database, get_database

__all__ = ["Database", "get_database"]
