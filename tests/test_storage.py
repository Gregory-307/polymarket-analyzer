"""Tests for storage module."""

import pytest
import pytest_asyncio
from datetime import datetime, timezone, timedelta
from pathlib import Path
import tempfile
import os

from src.storage.database import Database, database_session


@pytest_asyncio.fixture
async def test_db():
    """Create a temporary test database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = Database(db_path)
        await db.connect()
        yield db
        await db.close()


class TestDatabaseConnection:
    """Tests for database connection."""

    @pytest.mark.asyncio
    async def test_connect_creates_file(self):
        """Test that connecting creates the database file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            assert not db_path.exists()

            db = Database(db_path)
            await db.connect()

            assert db_path.exists()
            await db.close()

    @pytest.mark.asyncio
    async def test_schema_applied(self, test_db: Database):
        """Test that schema is applied on connection."""
        # Check that tables exist
        async with test_db.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ) as cursor:
            rows = await cursor.fetchall()
            table_names = {row["name"] for row in rows}

        expected_tables = {
            "schema_version", "markets", "market_snapshots",
            "orderbook_snapshots", "trades", "opportunities", "resolutions"
        }
        assert expected_tables.issubset(table_names)

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager usage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            async with Database(db_path) as db:
                # Should be connected
                assert db._connection is not None
                stats = await db.get_stats()
                assert "markets" in stats

            # Should be disconnected after exiting
            assert db._connection is None


class TestMarketOperations:
    """Tests for market CRUD operations."""

    @pytest.mark.asyncio
    async def test_upsert_market(self, test_db: Database):
        """Test market insert and update."""
        await test_db.upsert_market(
            id="test-market-1",
            platform="polymarket",
            question="Will it rain tomorrow?",
            yes_price=0.65,
            no_price=0.35,
            volume=10000,
            liquidity=5000,
        )

        market = await test_db.get_market("test-market-1")
        assert market is not None
        assert market["question"] == "Will it rain tomorrow?"
        assert market["yes_price"] == 0.65
        assert market["volume"] == 10000

        # Update the market
        await test_db.upsert_market(
            id="test-market-1",
            platform="polymarket",
            question="Will it rain tomorrow?",
            yes_price=0.70,
            no_price=0.30,
            volume=15000,
            liquidity=6000,
        )

        market = await test_db.get_market("test-market-1")
        assert market["yes_price"] == 0.70
        assert market["volume"] == 15000

    @pytest.mark.asyncio
    async def test_get_active_markets(self, test_db: Database):
        """Test fetching active markets."""
        # Insert multiple markets
        await test_db.upsert_market(
            id="market-1", platform="polymarket", question="Q1", is_active=True
        )
        await test_db.upsert_market(
            id="market-2", platform="polymarket", question="Q2", is_active=True
        )
        await test_db.upsert_market(
            id="market-3", platform="kalshi", question="Q3", is_active=True
        )
        await test_db.upsert_market(
            id="market-4", platform="polymarket", question="Q4", is_active=False
        )

        # Get all active
        all_active = await test_db.get_active_markets()
        assert len(all_active) == 3

        # Get polymarket active
        poly_active = await test_db.get_active_markets(platform="polymarket")
        assert len(poly_active) == 2


class TestSnapshotOperations:
    """Tests for market snapshot operations."""

    @pytest.mark.asyncio
    async def test_insert_snapshot(self, test_db: Database):
        """Test inserting a snapshot."""
        # First create a market
        await test_db.upsert_market(
            id="test-market", platform="polymarket", question="Test?"
        )

        snapshot_id = await test_db.insert_snapshot(
            market_id="test-market",
            platform="polymarket",
            yes_price=0.55,
            no_price=0.45,
            volume=1000,
            liquidity=500,
        )

        assert snapshot_id is not None
        assert snapshot_id > 0

    @pytest.mark.asyncio
    async def test_get_snapshots(self, test_db: Database):
        """Test fetching snapshots with time filters."""
        await test_db.upsert_market(
            id="test-market", platform="polymarket", question="Test?"
        )

        now = datetime.now(timezone.utc)

        # Insert multiple snapshots
        for i in range(5):
            await test_db.insert_snapshot(
                market_id="test-market",
                platform="polymarket",
                yes_price=0.50 + i * 0.01,
                no_price=0.50 - i * 0.01,
                timestamp=now - timedelta(hours=i),
            )

        # Get all snapshots
        all_snaps = await test_db.get_snapshots("test-market")
        assert len(all_snaps) == 5

        # Get snapshots since 2 hours ago
        recent_snaps = await test_db.get_snapshots(
            "test-market",
            since=now - timedelta(hours=2, minutes=30),
        )
        assert len(recent_snaps) == 3

    @pytest.mark.asyncio
    async def test_get_latest_snapshot(self, test_db: Database):
        """Test fetching the most recent snapshot."""
        await test_db.upsert_market(
            id="test-market", platform="polymarket", question="Test?"
        )

        now = datetime.now(timezone.utc)

        await test_db.insert_snapshot(
            market_id="test-market", platform="polymarket",
            yes_price=0.50, no_price=0.50,
            timestamp=now - timedelta(hours=1),
        )
        await test_db.insert_snapshot(
            market_id="test-market", platform="polymarket",
            yes_price=0.60, no_price=0.40,
            timestamp=now,
        )

        latest = await test_db.get_latest_snapshot("test-market")
        assert latest is not None
        assert latest["yes_price"] == 0.60


class TestOpportunityOperations:
    """Tests for opportunity tracking."""

    @pytest.mark.asyncio
    async def test_insert_opportunity(self, test_db: Database):
        """Test inserting an opportunity."""
        await test_db.upsert_market(
            id="test-market", platform="polymarket", question="Test?"
        )

        opp_id = await test_db.insert_opportunity(
            market_id="test-market",
            platform="polymarket",
            strategy="favorite_longshot",
            action="buy_yes",
            edge=0.02,
            confidence=0.8,
            details={"price": 0.95, "side": "YES"},
        )

        assert opp_id is not None

        opportunities = await test_db.get_opportunities(strategy="favorite_longshot")
        assert len(opportunities) == 1
        assert opportunities[0]["edge"] == 0.02

    @pytest.mark.asyncio
    async def test_mark_executed(self, test_db: Database):
        """Test marking opportunity as executed."""
        await test_db.upsert_market(
            id="test-market", platform="polymarket", question="Test?"
        )

        opp_id = await test_db.insert_opportunity(
            market_id="test-market",
            platform="polymarket",
            strategy="single_arb",
            action="buy_all",
            edge=0.05,
        )

        await test_db.mark_opportunity_executed(
            opp_id,
            execution_details={"filled_price": 0.48, "slippage": 0.002},
        )

        opps = await test_db.get_opportunities(executed=True)
        assert len(opps) == 1
        assert opps[0]["executed"] == 1


class TestResolutionOperations:
    """Tests for market resolution tracking."""

    @pytest.mark.asyncio
    async def test_insert_resolution(self, test_db: Database):
        """Test inserting a resolution."""
        await test_db.upsert_market(
            id="test-market", platform="polymarket", question="Will X happen?"
        )

        res_id = await test_db.insert_resolution(
            market_id="test-market",
            platform="polymarket",
            question="Will X happen?",
            outcome="YES",
            final_price=0.95,
        )

        assert res_id is not None

        resolutions = await test_db.get_resolutions(platform="polymarket")
        assert len(resolutions) == 1
        assert resolutions[0]["outcome"] == "YES"
        assert resolutions[0]["final_price"] == 0.95

    @pytest.mark.asyncio
    async def test_resolution_unique_market(self, test_db: Database):
        """Test that each market can only have one resolution (upsert)."""
        await test_db.upsert_market(
            id="test-market", platform="polymarket", question="Test?"
        )

        await test_db.insert_resolution(
            market_id="test-market",
            platform="polymarket",
            question="Test?",
            outcome="YES",
        )

        # Insert again - should replace
        await test_db.insert_resolution(
            market_id="test-market",
            platform="polymarket",
            question="Test?",
            outcome="NO",
        )

        resolutions = await test_db.get_resolutions()
        assert len(resolutions) == 1
        assert resolutions[0]["outcome"] == "NO"


class TestDatabaseStats:
    """Tests for database statistics."""

    @pytest.mark.asyncio
    async def test_get_stats(self, test_db: Database):
        """Test getting database statistics."""
        # Insert some data
        await test_db.upsert_market(
            id="market-1", platform="polymarket", question="Q1"
        )
        await test_db.upsert_market(
            id="market-2", platform="polymarket", question="Q2"
        )
        await test_db.insert_snapshot(
            market_id="market-1", platform="polymarket",
            yes_price=0.5, no_price=0.5,
        )

        stats = await test_db.get_stats()

        assert stats["markets"] == 2
        assert stats["market_snapshots"] == 1
        assert "snapshot_range" in stats
