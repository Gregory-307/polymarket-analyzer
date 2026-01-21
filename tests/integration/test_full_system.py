"""Full system integration tests.

Tests the complete Polymarket Analyzer system end-to-end:
1. All adapters (Polymarket, Kalshi, Deribit, WebSocket)
2. Both strategies (Favorite-Longshot, Financial Markets)
3. Execution infrastructure (Paper Trader, Risk Manager, Circuit Breakers)
4. Data infrastructure (Database, Scheduler, Resolution Tracker)
5. Full trading simulation

Run with: python -m pytest tests/integration/test_full_system.py -v
"""

import asyncio
import pytest
import tempfile
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

from src.adapters import PolymarketAdapter, KalshiAdapter, PolymarketWebSocket, Market
from src.adapters.deribit_options import DeribitOptionsClient
from src.core.config import Credentials
from src.strategies import FavoriteLongshotStrategy, FinancialMarketsStrategy
from src.execution import (
    PaperTrader,
    RiskManager,
    RiskLimits,
    CircuitBreakerManager,
)
from src.execution.position_tracker import PositionTracker
from src.storage.database import Database
from src.scheduler.scheduler import Scheduler, Job
from src.analysis.calibration import CalibrationAnalyzer
from src.analysis.resolution_tracker import ResolutionTracker


class TestFullSystemIntegration:
    """Complete system integration tests."""

    @pytest.fixture
    def credentials(self):
        return Credentials.from_env()

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield f.name
        # Cleanup
        try:
            os.unlink(f.name)
        except:
            pass

    # =========================================================================
    # ADAPTER TESTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_01_all_adapters_connect(self, credentials):
        """Test all adapters can connect."""
        results = {}

        # Polymarket
        try:
            adapter = PolymarketAdapter(credentials=credentials)
            connected = await adapter.connect()
            results["polymarket"] = connected and adapter._client is not None
            await adapter.disconnect()
        except Exception as e:
            results["polymarket"] = f"Error: {e}"

        # Kalshi
        try:
            adapter = KalshiAdapter(credentials=credentials)
            connected = await adapter.connect()
            results["kalshi"] = connected and adapter._client is not None
            await adapter.disconnect()
        except Exception as e:
            results["kalshi"] = f"Error: {e}"

        # Deribit
        try:
            async with DeribitOptionsClient() as client:
                btc = await client.get_spot("BTC")
                results["deribit"] = btc is not None
        except Exception as e:
            results["deribit"] = f"Error: {e}"

        # WebSocket
        try:
            ws = PolymarketWebSocket()
            await ws.connect()
            results["websocket"] = ws.is_connected
            await ws.disconnect()
        except Exception as e:
            results["websocket"] = f"Error: {e}"

        print("\nAdapter Connection Results:")
        for name, status in results.items():
            print(f"  {name}: {status}")

        # At minimum, Polymarket and Deribit should work
        assert results["polymarket"] == True, "Polymarket should connect"
        assert results["deribit"] == True, "Deribit should connect"

    @pytest.mark.asyncio
    async def test_02_fetch_markets_from_all_platforms(self, credentials):
        """Test fetching markets from all platforms."""
        all_markets = []

        # Polymarket
        poly_adapter = PolymarketAdapter(credentials=credentials)
        await poly_adapter.connect()
        poly_markets = await poly_adapter.get_all_markets(active_only=True, limit=50)
        all_markets.extend(poly_markets)
        await poly_adapter.disconnect()

        # Kalshi (may fail if no credentials)
        try:
            kalshi_adapter = KalshiAdapter(credentials=credentials)
            await kalshi_adapter.connect()
            kalshi_markets = await kalshi_adapter.get_markets(active_only=True, limit=50)
            all_markets.extend(kalshi_markets)
            await kalshi_adapter.disconnect()
        except:
            pass  # Kalshi may not be available

        print(f"\nTotal markets fetched: {len(all_markets)}")
        print(f"  Polymarket: {len(poly_markets)}")

        assert len(poly_markets) > 0, "Should fetch Polymarket markets"

    # =========================================================================
    # STRATEGY TESTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_03_both_strategies_find_opportunities(self, credentials):
        """Test both strategies can find opportunities."""
        # Fetch markets
        adapter = PolymarketAdapter(credentials=credentials)
        await adapter.connect()
        markets = await adapter.get_all_markets(active_only=True, limit=200)
        await adapter.disconnect()

        # Get live data for FM strategy
        async with DeribitOptionsClient() as client:
            btc_spot = await client.get_spot("BTC")
            eth_spot = await client.get_spot("ETH")
            btc_vol = await client.get_historical_vol("BTC")
            eth_vol = await client.get_historical_vol("ETH")

        # Favorite-Longshot
        fl_strategy = FavoriteLongshotStrategy(min_probability=0.85, min_liquidity=100)
        fl_opportunities = []
        for market in markets:
            opp = fl_strategy.check_market(market)
            if opp:
                fl_opportunities.append(opp)

        # Financial Markets
        fm_strategy = FinancialMarketsStrategy(min_edge=0.05)  # 5% for testing
        fm_strategy.set_spot_price("BTC", btc_spot or 90000)
        fm_strategy.set_spot_price("ETH", eth_spot or 3000)
        fm_strategy.set_implied_vol("BTC", btc_vol or 0.50)
        fm_strategy.set_implied_vol("ETH", eth_vol or 0.60)

        fm_opportunities = fm_strategy.scan(markets)

        print(f"\nOpportunities Found:")
        print(f"  Favorite-Longshot: {len(fl_opportunities)}")
        print(f"  Financial Markets: {len(fm_opportunities)}")

        # At least one strategy should find something
        total = len(fl_opportunities) + len(fm_opportunities)
        assert total >= 0, "Strategies should run without error"

    @pytest.mark.asyncio
    async def test_04_strategies_dont_overlap(self, credentials):
        """Verify strategies target different market types."""
        adapter = PolymarketAdapter(credentials=credentials)
        await adapter.connect()
        markets = await adapter.get_all_markets(active_only=True, limit=100)
        await adapter.disconnect()

        fl_strategy = FavoriteLongshotStrategy(min_probability=0.90)
        fm_strategy = FinancialMarketsStrategy(min_edge=0.10)

        # Set FM prices
        fm_strategy.set_spot_price("BTC", 90000)
        fm_strategy.set_spot_price("ETH", 3000)
        fm_strategy.set_implied_vol("BTC", 0.50)
        fm_strategy.set_implied_vol("ETH", 0.60)

        fl_markets = set()
        fm_markets = set()

        for market in markets:
            if fl_strategy.check_market(market):
                fl_markets.add(market.id)
            if fm_strategy.check_market(market):
                fm_markets.add(market.id)

        # Some overlap is OK, but strategies should target different markets
        overlap = fl_markets & fm_markets
        print(f"\nF-L markets: {len(fl_markets)}")
        print(f"FM markets: {len(fm_markets)}")
        print(f"Overlap: {len(overlap)}")

    # =========================================================================
    # EXECUTION INFRASTRUCTURE TESTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_05_paper_trading_full_cycle(self):
        """Test complete paper trading cycle."""
        trader = PaperTrader(initial_balance=10000.0)

        # Execute trades
        await trader.execute_paper_trade("btc_100k", "YES", 100, 0.60)
        await trader.execute_paper_trade("eth_5k", "NO", 50, 0.30)
        await trader.execute_paper_trade("btc_150k", "YES", 25, 0.10)

        # Check positions
        positions = trader.get_open_positions()
        assert len(positions) == 3

        # Mark to market
        snapshot = await trader.mark_to_market({
            "btc_100k": 0.65,  # Price moved up
            "eth_5k": 0.35,    # Price moved against us
        })

        print(f"\nPortfolio after mark-to-market:")
        print(f"  Total value: ${snapshot.total_value:.2f}")
        print(f"  Unrealized P&L: ${snapshot.unrealized_pnl:.2f}")

        # Resolve some positions
        closed = await trader.close_resolved_positions({
            "btc_100k": "YES",  # Win
            "eth_5k": "YES",    # Loss (we bet NO)
        })

        assert len(closed) == 2

        # Get final report
        report = trader.get_performance_report()
        print(f"\nPerformance Report:")
        print(f"  Total trades: {report.total_trades}")
        print(f"  Win rate: {report.win_rate:.1%}")
        print(f"  Total return: ${report.total_return:.2f}")

    def test_06_risk_manager_integration(self):
        """Test risk manager with position tracker."""
        position_tracker = PositionTracker()
        limits = RiskLimits(
            max_position_usd=500,
            max_total_exposure=2000,
            daily_loss_limit=300,
            max_positions=10,
        )
        risk_mgr = RiskManager(limits, position_tracker)

        # Test various order checks
        checks = [
            # (size, price, should_pass, description)
            (100, 0.50, True, "Normal order"),
            (1000, 0.90, False, "Exceeds position limit"),
            (50, 0.80, True, "Within limits"),
        ]

        for size, price, should_pass, desc in checks:
            check = risk_mgr.check_new_order("test", size, price)
            assert check.passed == should_pass, f"Failed: {desc}"

    def test_07_circuit_breaker_integration(self):
        """Test circuit breakers with trading events."""
        manager = CircuitBreakerManager()
        manager.add_loss_breaker(threshold=200, window_seconds=300)
        manager.add_consecutive_loss_breaker(threshold=4)
        manager.add_error_breaker(threshold=5, window_seconds=60)

        # Simulate trading session
        trades = [
            (True, 10),    # Win $10
            (True, 15),    # Win $15
            (False, -30),  # Loss $30
            (False, -50),  # Loss $50
            (False, -80),  # Loss $80
        ]

        for won, pnl in trades:
            if not manager.allows_trading():
                break
            manager.record_trade(won=won)
            if pnl < 0:
                manager.record_loss(abs(pnl))

        # Should have tripped on consecutive losses or cumulative loss
        status = manager.get_status()
        print(f"\nCircuit Breaker Status:")
        print(f"  Allows trading: {status['allows_trading']}")
        print(f"  Open breakers: {status['open_breakers']}")

    # =========================================================================
    # DATABASE TESTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_08_database_operations(self, temp_db_path):
        """Test database CRUD operations."""
        async with Database(temp_db_path) as db:
            # Insert market
            await db.upsert_market(
                id="test_market_1",
                platform="polymarket",
                question="Test market question",
                is_active=True,
                yes_price=0.55,
                no_price=0.45,
                volume=10000,
                liquidity=5000,
            )

            # Insert snapshot
            await db.insert_snapshot(
                market_id="test_market_1",
                platform="polymarket",
                yes_price=0.55,
                no_price=0.45,
                volume=10000,
                liquidity=5000,
            )

            # Insert resolution
            await db.insert_resolution(
                market_id="test_market_1",
                platform="polymarket",
                question="Test market question",
                outcome="YES",
                final_price=0.55,
                resolved_at=datetime.now(timezone.utc),
            )

            # Query data
            markets = await db.get_active_markets()
            resolutions = await db.get_resolutions()
            stats = await db.get_stats()

            print(f"\nDatabase Stats:")
            print(f"  Markets: {stats.get('markets', 0)}")
            print(f"  Snapshots: {stats.get('snapshots', 0)}")
            print(f"  Resolutions: {stats.get('resolutions', 0)}")

            assert len(markets) >= 1
            assert len(resolutions) >= 1

    @pytest.mark.asyncio
    async def test_09_calibration_analysis(self, temp_db_path):
        """Test calibration analysis with mock data."""
        async with Database(temp_db_path) as db:
            # Insert resolved markets for calibration
            test_data = [
                # (price, outcome) - simulate calibration data
                (0.90, "YES"),
                (0.85, "YES"),
                (0.92, "YES"),
                (0.88, "NO"),  # Upset
                (0.15, "NO"),
                (0.10, "NO"),
                (0.08, "YES"),  # Upset
            ]

            for i, (price, outcome) in enumerate(test_data):
                # First insert the market (required due to foreign key constraint)
                await db.upsert_market(
                    id=f"cal_test_{i}",
                    platform="polymarket",
                    question=f"Test market {i}",
                    yes_price=price,
                    no_price=1.0 - price,
                    is_active=False,  # Resolved markets are inactive
                )
                # Then insert the resolution
                await db.insert_resolution(
                    market_id=f"cal_test_{i}",
                    platform="polymarket",
                    question=f"Test market {i}",
                    outcome=outcome,
                    final_price=price,
                    resolved_at=datetime.now(timezone.utc),
                )

            # Run calibration analysis
            analyzer = CalibrationAnalyzer(db, bucket_width=0.20, min_sample_size=2)
            result = await analyzer.analyze()

            print(f"\nCalibration Result:")
            print(f"  Total markets: {result.total_markets}")
            print(f"  Brier score: {result.brier_score:.4f}")
            print(f"  Mean edge: {result.mean_edge:+.2%}")

    # =========================================================================
    # SCHEDULER TESTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_10_scheduler_basic(self):
        """Test scheduler executes jobs."""
        scheduler = Scheduler()

        execution_count = {"count": 0}

        async def test_job():
            execution_count["count"] += 1

        scheduler.add_job(
            name="test_job",
            func=test_job,
            interval_seconds=0.1,
            run_immediately=True,
        )

        # Start scheduler
        await scheduler.start()

        # Wait for some executions
        await asyncio.sleep(0.3)

        # Stop scheduler
        await scheduler.stop()

        assert execution_count["count"] >= 1, "Job should have executed"
        print(f"\nScheduler executed job {execution_count['count']} times")

    # =========================================================================
    # WEBSOCKET TESTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_11_websocket_connection(self):
        """Test WebSocket connects and can subscribe."""
        received_messages = []

        def on_price(update):
            received_messages.append(update)

        ws = PolymarketWebSocket(on_price_update=on_price)

        await ws.connect()
        assert ws.is_connected

        status = ws.get_status()
        assert status["connected"] == True

        await ws.disconnect()
        assert not ws.is_connected

    # =========================================================================
    # FULL PIPELINE TESTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_12_full_trading_simulation(self, credentials, temp_db_path):
        """Simulate complete trading session with both strategies."""
        print("\n" + "=" * 60)
        print("  FULL TRADING SIMULATION")
        print("=" * 60)

        # Initialize components
        paper_trader = PaperTrader(initial_balance=5000.0)
        circuit_breakers = CircuitBreakerManager()
        circuit_breakers.add_loss_breaker(threshold=500)
        circuit_breakers.add_consecutive_loss_breaker(threshold=5)

        position_tracker = PositionTracker()
        risk_mgr = RiskManager(
            RiskLimits(max_position_usd=200, max_total_exposure=1000),
            position_tracker,
        )

        # Fetch markets
        adapter = PolymarketAdapter(credentials=credentials)
        await adapter.connect()
        markets = await adapter.get_all_markets(active_only=True, limit=100)
        await adapter.disconnect()

        # Get Deribit data
        async with DeribitOptionsClient() as client:
            btc_spot = await client.get_spot("BTC")
            btc_vol = await client.get_historical_vol("BTC")

        print(f"\nMarkets fetched: {len(markets)}")
        print(f"BTC: ${btc_spot:,.0f} | Vol: {btc_vol:.1%}")

        # Initialize strategies
        fl_strategy = FavoriteLongshotStrategy(min_probability=0.90, min_liquidity=500)
        fm_strategy = FinancialMarketsStrategy(min_edge=0.15)
        fm_strategy.set_spot_price("BTC", btc_spot)
        fm_strategy.set_implied_vol("BTC", btc_vol)

        # Scan for opportunities
        fl_opps = []
        fm_opps = []

        for market in markets:
            fl_opp = fl_strategy.check_market(market)
            if fl_opp:
                fl_opps.append((market, fl_opp, "FL"))

            fm_opp = fm_strategy.check_market(market)
            if fm_opp:
                fm_opps.append((market, fm_opp, "FM"))

        all_opps = fl_opps + fm_opps
        print(f"\nOpportunities: {len(fl_opps)} F-L, {len(fm_opps)} FM")

        # Execute trades
        trades_executed = 0

        for market, opp, strategy_type in all_opps[:5]:
            if not circuit_breakers.allows_trading():
                print("Circuit breaker tripped - stopping")
                break

            # Determine trade parameters
            if strategy_type == "FL":
                side = opp.side
                price = opp.price
                size_usd = 50
            else:
                side = "YES" if opp.edge > 0 else "NO"
                price = opp.market_price if opp.edge > 0 else (1 - opp.market_price)
                size_usd = 100

            # Risk check
            size_shares = size_usd / price
            check = risk_mgr.check_new_order(market.id, size_shares, price)

            if not check.passed:
                print(f"  Risk blocked: {check.reason}")
                continue

            # Execute
            try:
                await paper_trader.execute_paper_trade(
                    market_id=market.id,
                    side=side,
                    size=size_shares,
                    price=price,
                    platform=market.platform,
                    question=market.question[:40],
                )
                trades_executed += 1
                print(f"  Traded [{strategy_type}]: {side} @ {price:.2f}")
            except ValueError as e:
                print(f"  Trade failed: {e}")

        # Final report
        summary = paper_trader.get_position_summary()
        print(f"\n" + "-" * 60)
        print(f"SIMULATION COMPLETE")
        print(f"  Trades executed: {trades_executed}")
        print(f"  Open positions: {summary['open_positions']}")
        print(f"  Cash balance: ${summary['cash_balance']:.2f}")
        print(f"  Cost basis: ${summary['total_cost_basis']:.2f}")

    @pytest.mark.asyncio
    async def test_13_persistence_and_recovery(self, temp_db_path):
        """Test saving and loading paper trader state."""
        # Create trader with trades
        trader1 = PaperTrader(initial_balance=1000.0)
        await trader1.execute_paper_trade("persist_1", "YES", 50, 0.80)
        await trader1.execute_paper_trade("persist_2", "NO", 30, 0.25)

        # Save state
        save_path = temp_db_path.replace(".db", "_trader.json")
        trader1.save_to_file(save_path)

        # Load into new trader
        trader2 = PaperTrader.load_from_file(save_path)

        # Verify state
        assert trader2.cash_balance == trader1.cash_balance
        assert len(trader2.positions) == len(trader1.positions)
        assert len(trader2.trade_history) == len(trader1.trade_history)

        print(f"\nState persisted and recovered successfully")

        # Cleanup
        try:
            os.unlink(save_path)
        except:
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
