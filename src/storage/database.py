"""SQLite database layer for data persistence.

Provides async SQLite wrapper with:
- Schema management and migrations
- Connection pooling
- CRUD operations for markets, snapshots, trades, opportunities
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator

import aiosqlite

from ..core.utils import get_logger

logger = get_logger(__name__)

# Default database location
DEFAULT_DB_PATH = Path("data/polymarket.db")

# Schema version for migrations
SCHEMA_VERSION = 1

# SQL schema definitions
SCHEMA = """
-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

-- Markets: Current state of each market
CREATE TABLE IF NOT EXISTS markets (
    id TEXT PRIMARY KEY,
    platform TEXT NOT NULL,
    question TEXT NOT NULL,
    description TEXT,
    category TEXT,
    yes_price REAL,
    no_price REAL,
    volume REAL,
    liquidity REAL,
    end_date TEXT,
    outcomes TEXT,  -- JSON array
    is_active INTEGER DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Market snapshots: Time-series of market prices
CREATE TABLE IF NOT EXISTS market_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT NOT NULL,
    platform TEXT NOT NULL,
    yes_price REAL NOT NULL,
    no_price REAL NOT NULL,
    volume REAL,
    liquidity REAL,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (market_id) REFERENCES markets(id)
);

-- Orderbook snapshots: Depth snapshots for active markets
CREATE TABLE IF NOT EXISTS orderbook_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT NOT NULL,
    platform TEXT NOT NULL,
    best_bid REAL,
    best_ask REAL,
    spread REAL,
    bid_depth_1pct REAL,  -- Total size within 1% of best bid
    ask_depth_1pct REAL,  -- Total size within 1% of best ask
    bids TEXT,  -- JSON array of [price, size] pairs
    asks TEXT,  -- JSON array of [price, size] pairs
    timestamp TEXT NOT NULL,
    FOREIGN KEY (market_id) REFERENCES markets(id)
);

-- Trades: Historical trades (from API)
CREATE TABLE IF NOT EXISTS trades (
    id TEXT PRIMARY KEY,
    market_id TEXT NOT NULL,
    platform TEXT NOT NULL,
    side TEXT NOT NULL,
    price REAL NOT NULL,
    size REAL NOT NULL,
    timestamp TEXT NOT NULL,
    fee REAL DEFAULT 0,
    FOREIGN KEY (market_id) REFERENCES markets(id)
);

-- Opportunities: Detected trading opportunities
CREATE TABLE IF NOT EXISTS opportunities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT NOT NULL,
    platform TEXT NOT NULL,
    strategy TEXT NOT NULL,  -- 'single_arb', 'multi_arb', 'favorite_longshot', etc.
    action TEXT,  -- 'buy_yes', 'buy_no', 'buy_all', 'sell_all', etc.
    edge REAL,
    confidence REAL,
    details TEXT,  -- JSON with strategy-specific details
    detected_at TEXT NOT NULL,
    executed INTEGER DEFAULT 0,
    execution_details TEXT,  -- JSON with execution results
    FOREIGN KEY (market_id) REFERENCES markets(id)
);

-- Market resolutions: Track outcomes for validation
CREATE TABLE IF NOT EXISTS resolutions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT NOT NULL UNIQUE,
    platform TEXT NOT NULL,
    question TEXT NOT NULL,
    outcome TEXT NOT NULL,  -- 'YES', 'NO', or specific outcome name
    final_price REAL,  -- Final price before resolution
    resolved_at TEXT NOT NULL,
    FOREIGN KEY (market_id) REFERENCES markets(id)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_snapshots_market_time ON market_snapshots(market_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_snapshots_time ON market_snapshots(timestamp);
CREATE INDEX IF NOT EXISTS idx_orderbook_market_time ON orderbook_snapshots(market_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_market_time ON trades(market_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_opportunities_strategy ON opportunities(strategy, detected_at);
CREATE INDEX IF NOT EXISTS idx_opportunities_market ON opportunities(market_id, detected_at);
CREATE INDEX IF NOT EXISTS idx_resolutions_platform ON resolutions(platform, resolved_at);
"""


class Database:
    """Async SQLite database wrapper.

    Usage:
        db = Database()
        await db.connect()

        # Insert market snapshot
        await db.insert_snapshot(market_id, platform, yes_price, no_price, ...)

        # Query snapshots
        snapshots = await db.get_snapshots(market_id, since=datetime(...))

        await db.close()

    Or use as async context manager:
        async with Database() as db:
            await db.insert_snapshot(...)
    """

    def __init__(self, db_path: Path | str | None = None):
        """Initialize database.

        Args:
            db_path: Path to SQLite database file. Defaults to data/polymarket.db
        """
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self._connection: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        """Connect to database and ensure schema exists."""
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._connection = await aiosqlite.connect(
            self.db_path,
            isolation_level=None,  # Autocommit mode
        )
        self._connection.row_factory = aiosqlite.Row

        # Enable foreign keys
        await self._connection.execute("PRAGMA foreign_keys = ON")

        # Apply schema
        await self._apply_schema()

        logger.info(f"Connected to database: {self.db_path}")

    async def close(self) -> None:
        """Close database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None
            logger.info("Database connection closed")

    async def __aenter__(self) -> "Database":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    @property
    def connection(self) -> aiosqlite.Connection:
        """Get active connection."""
        if not self._connection:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._connection

    async def _apply_schema(self) -> None:
        """Apply database schema and run migrations."""
        # Check current schema version
        try:
            async with self.connection.execute(
                "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
            ) as cursor:
                row = await cursor.fetchone()
                current_version = row["version"] if row else 0
        except sqlite3.OperationalError:
            # schema_version table doesn't exist yet
            current_version = 0

        if current_version < SCHEMA_VERSION:
            logger.info(f"Applying schema v{SCHEMA_VERSION} (current: v{current_version})")
            await self.connection.executescript(SCHEMA)

            # Record schema version
            now = datetime.now(timezone.utc).isoformat()
            await self.connection.execute(
                "INSERT OR REPLACE INTO schema_version (version, applied_at) VALUES (?, ?)",
                (SCHEMA_VERSION, now),
            )
            logger.info(f"Schema v{SCHEMA_VERSION} applied successfully")

    # =========================================================================
    # Market operations
    # =========================================================================

    async def upsert_market(
        self,
        id: str,
        platform: str,
        question: str,
        description: str = "",
        category: str = "",
        yes_price: float = 0.5,
        no_price: float = 0.5,
        volume: float = 0.0,
        liquidity: float = 0.0,
        end_date: datetime | None = None,
        outcomes: list[str] | None = None,
        is_active: bool = True,
    ) -> None:
        """Insert or update a market."""
        now = datetime.now(timezone.utc).isoformat()
        end_date_str = end_date.isoformat() if end_date else None
        outcomes_json = json.dumps(outcomes or ["Yes", "No"])

        await self.connection.execute(
            """
            INSERT INTO markets (
                id, platform, question, description, category,
                yes_price, no_price, volume, liquidity, end_date,
                outcomes, is_active, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                question = excluded.question,
                description = excluded.description,
                category = excluded.category,
                yes_price = excluded.yes_price,
                no_price = excluded.no_price,
                volume = excluded.volume,
                liquidity = excluded.liquidity,
                end_date = excluded.end_date,
                outcomes = excluded.outcomes,
                is_active = excluded.is_active,
                updated_at = excluded.updated_at
            """,
            (
                id, platform, question, description, category,
                yes_price, no_price, volume, liquidity, end_date_str,
                outcomes_json, int(is_active), now, now,
            ),
        )

    async def get_market(self, market_id: str) -> dict | None:
        """Get a market by ID."""
        async with self.connection.execute(
            "SELECT * FROM markets WHERE id = ?", (market_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return dict(row)
            return None

    async def get_active_markets(self, platform: str | None = None) -> list[dict]:
        """Get all active markets."""
        if platform:
            query = "SELECT * FROM markets WHERE is_active = 1 AND platform = ?"
            params = (platform,)
        else:
            query = "SELECT * FROM markets WHERE is_active = 1"
            params = ()

        async with self.connection.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    # =========================================================================
    # Snapshot operations
    # =========================================================================

    async def insert_snapshot(
        self,
        market_id: str,
        platform: str,
        yes_price: float,
        no_price: float,
        volume: float | None = None,
        liquidity: float | None = None,
        timestamp: datetime | None = None,
    ) -> int:
        """Insert a market price snapshot."""
        ts = (timestamp or datetime.now(timezone.utc)).isoformat()

        cursor = await self.connection.execute(
            """
            INSERT INTO market_snapshots (
                market_id, platform, yes_price, no_price, volume, liquidity, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (market_id, platform, yes_price, no_price, volume, liquidity, ts),
        )
        return cursor.lastrowid

    async def get_snapshots(
        self,
        market_id: str,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 1000,
    ) -> list[dict]:
        """Get snapshots for a market."""
        conditions = ["market_id = ?"]
        params: list[Any] = [market_id]

        if since:
            conditions.append("timestamp >= ?")
            params.append(since.isoformat())
        if until:
            conditions.append("timestamp <= ?")
            params.append(until.isoformat())

        query = f"""
            SELECT * FROM market_snapshots
            WHERE {' AND '.join(conditions)}
            ORDER BY timestamp DESC
            LIMIT ?
        """
        params.append(limit)

        async with self.connection.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_latest_snapshot(self, market_id: str) -> dict | None:
        """Get most recent snapshot for a market."""
        async with self.connection.execute(
            """
            SELECT * FROM market_snapshots
            WHERE market_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (market_id,),
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None

    # =========================================================================
    # Orderbook operations
    # =========================================================================

    async def insert_orderbook_snapshot(
        self,
        market_id: str,
        platform: str,
        best_bid: float | None,
        best_ask: float | None,
        spread: float | None,
        bid_depth_1pct: float | None,
        ask_depth_1pct: float | None,
        bids: list[tuple[float, float]] | None = None,
        asks: list[tuple[float, float]] | None = None,
        timestamp: datetime | None = None,
    ) -> int:
        """Insert an orderbook snapshot."""
        ts = (timestamp or datetime.now(timezone.utc)).isoformat()
        bids_json = json.dumps(bids) if bids else None
        asks_json = json.dumps(asks) if asks else None

        cursor = await self.connection.execute(
            """
            INSERT INTO orderbook_snapshots (
                market_id, platform, best_bid, best_ask, spread,
                bid_depth_1pct, ask_depth_1pct, bids, asks, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                market_id, platform, best_bid, best_ask, spread,
                bid_depth_1pct, ask_depth_1pct, bids_json, asks_json, ts,
            ),
        )
        return cursor.lastrowid

    # =========================================================================
    # Trade operations
    # =========================================================================

    async def insert_trade(
        self,
        id: str,
        market_id: str,
        platform: str,
        side: str,
        price: float,
        size: float,
        timestamp: datetime,
        fee: float = 0.0,
    ) -> None:
        """Insert a trade (ignores duplicates)."""
        try:
            await self.connection.execute(
                """
                INSERT OR IGNORE INTO trades (
                    id, market_id, platform, side, price, size, timestamp, fee
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (id, market_id, platform, side, price, size, timestamp.isoformat(), fee),
            )
        except sqlite3.IntegrityError:
            pass  # Duplicate trade, ignore

    async def get_trades(
        self,
        market_id: str,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Get trades for a market."""
        if since:
            query = """
                SELECT * FROM trades
                WHERE market_id = ? AND timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT ?
            """
            params = (market_id, since.isoformat(), limit)
        else:
            query = """
                SELECT * FROM trades
                WHERE market_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """
            params = (market_id, limit)

        async with self.connection.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    # =========================================================================
    # Opportunity operations
    # =========================================================================

    async def insert_opportunity(
        self,
        market_id: str,
        platform: str,
        strategy: str,
        action: str | None = None,
        edge: float | None = None,
        confidence: float | None = None,
        details: dict | None = None,
        detected_at: datetime | None = None,
    ) -> int:
        """Insert a detected opportunity."""
        ts = (detected_at or datetime.now(timezone.utc)).isoformat()
        details_json = json.dumps(details) if details else None

        cursor = await self.connection.execute(
            """
            INSERT INTO opportunities (
                market_id, platform, strategy, action, edge,
                confidence, details, detected_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (market_id, platform, strategy, action, edge, confidence, details_json, ts),
        )
        return cursor.lastrowid

    async def get_opportunities(
        self,
        strategy: str | None = None,
        since: datetime | None = None,
        executed: bool | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Get detected opportunities."""
        conditions = []
        params: list[Any] = []

        if strategy:
            conditions.append("strategy = ?")
            params.append(strategy)
        if since:
            conditions.append("detected_at >= ?")
            params.append(since.isoformat())
        if executed is not None:
            conditions.append("executed = ?")
            params.append(int(executed))

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"""
            SELECT * FROM opportunities
            {where_clause}
            ORDER BY detected_at DESC
            LIMIT ?
        """
        params.append(limit)

        async with self.connection.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def mark_opportunity_executed(
        self,
        opportunity_id: int,
        execution_details: dict | None = None,
    ) -> None:
        """Mark an opportunity as executed."""
        details_json = json.dumps(execution_details) if execution_details else None
        await self.connection.execute(
            "UPDATE opportunities SET executed = 1, execution_details = ? WHERE id = ?",
            (details_json, opportunity_id),
        )

    # =========================================================================
    # Resolution operations
    # =========================================================================

    async def insert_resolution(
        self,
        market_id: str,
        platform: str,
        question: str,
        outcome: str,
        final_price: float | None = None,
        resolved_at: datetime | None = None,
    ) -> int:
        """Insert a market resolution."""
        ts = (resolved_at or datetime.now(timezone.utc)).isoformat()

        cursor = await self.connection.execute(
            """
            INSERT OR REPLACE INTO resolutions (
                market_id, platform, question, outcome, final_price, resolved_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (market_id, platform, question, outcome, final_price, ts),
        )
        return cursor.lastrowid

    async def get_resolutions(
        self,
        platform: str | None = None,
        since: datetime | None = None,
        limit: int = 1000,
    ) -> list[dict]:
        """Get market resolutions."""
        conditions = []
        params: list[Any] = []

        if platform:
            conditions.append("platform = ?")
            params.append(platform)
        if since:
            conditions.append("resolved_at >= ?")
            params.append(since.isoformat())

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"""
            SELECT * FROM resolutions
            {where_clause}
            ORDER BY resolved_at DESC
            LIMIT ?
        """
        params.append(limit)

        async with self.connection.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    # =========================================================================
    # Statistics
    # =========================================================================

    async def get_stats(self) -> dict:
        """Get database statistics."""
        stats = {}

        tables = [
            "markets", "market_snapshots", "orderbook_snapshots",
            "trades", "opportunities", "resolutions"
        ]

        for table in tables:
            async with self.connection.execute(f"SELECT COUNT(*) as count FROM {table}") as cursor:
                row = await cursor.fetchone()
                stats[table] = row["count"]

        # Get date range of snapshots
        async with self.connection.execute(
            "SELECT MIN(timestamp) as min_ts, MAX(timestamp) as max_ts FROM market_snapshots"
        ) as cursor:
            row = await cursor.fetchone()
            stats["snapshot_range"] = {
                "start": row["min_ts"],
                "end": row["max_ts"],
            }

        return stats


# Module-level singleton for convenience
_db_instance: Database | None = None


async def get_database(db_path: Path | str | None = None) -> Database:
    """Get or create database instance (singleton pattern).

    Args:
        db_path: Optional path to database file.

    Returns:
        Connected Database instance.
    """
    global _db_instance

    if _db_instance is None or not _db_instance._connection:
        _db_instance = Database(db_path)
        await _db_instance.connect()

    return _db_instance


@asynccontextmanager
async def database_session(db_path: Path | str | None = None) -> AsyncGenerator[Database, None]:
    """Async context manager for database sessions.

    Usage:
        async with database_session() as db:
            await db.insert_snapshot(...)
    """
    db = Database(db_path)
    await db.connect()
    try:
        yield db
    finally:
        await db.close()
