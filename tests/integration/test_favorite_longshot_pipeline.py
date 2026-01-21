"""End-to-end integration tests for Favorite-Longshot strategy pipeline.

Tests the complete flow:
1. Connect to Polymarket API
2. Fetch real markets
3. Run F-L strategy on markets
4. Filter opportunities by criteria
5. Calculate Kelly sizing
6. Validate paper trading flow
7. Track resolution and P&L

Run with: python -m pytest tests/integration/test_favorite_longshot_pipeline.py -v
"""

import asyncio
import pytest
from datetime import datetime, timezone, timedelta

from src.adapters import PolymarketAdapter, Market
from src.core.config import Credentials
from src.strategies import FavoriteLongshotStrategy
from src.sizing.kelly import KellyCriterion, KellyBet, KellyFraction
from src.execution import PaperTrader, RiskManager, RiskLimits
from src.execution.position_tracker import PositionTracker
from src.execution import CircuitBreakerManager


class TestFavoriteLongshotPipeline:
    """End-to-end tests for F-L strategy."""

    @pytest.fixture
    def credentials(self):
        """Load credentials from environment."""
        return Credentials.from_env()

    @pytest.fixture
    def strategy(self):
        """Create F-L strategy with test parameters."""
        return FavoriteLongshotStrategy(
            min_probability=0.85,  # Lower threshold for testing
            min_liquidity=100.0,   # Lower liquidity for testing
        )

    @pytest.fixture
    def paper_trader(self):
        """Create paper trader for testing."""
        return PaperTrader(initial_balance=1000.0)

    @pytest.fixture
    def circuit_breakers(self):
        """Create circuit breaker manager."""
        manager = CircuitBreakerManager()
        manager.add_loss_breaker(threshold=100, window_seconds=300)
        manager.add_consecutive_loss_breaker(threshold=5)
        manager.add_error_breaker(threshold=10, window_seconds=60)
        return manager

    # =========================================================================
    # PHASE 1: API CONNECTION TESTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_01_polymarket_connection(self, credentials):
        """Test Polymarket API connects successfully."""
        adapter = PolymarketAdapter(credentials=credentials)

        result = await adapter.connect()
        assert result, "connect() should return True"
        assert adapter._client is not None, "Adapter should have HTTP client after connect"

        await adapter.disconnect()
        assert adapter._client is None, "Adapter should not have HTTP client after disconnect"

    @pytest.mark.asyncio
    async def test_02_fetch_markets(self, credentials):
        """Test fetching markets from Polymarket."""
        adapter = PolymarketAdapter(credentials=credentials)
        await adapter.connect()

        markets = await adapter.get_markets(active_only=True, limit=50)

        assert len(markets) > 0, "Should fetch some markets"
        assert all(isinstance(m, Market) for m in markets), "All should be Market objects"

        # Verify market structure
        for market in markets[:5]:
            assert market.id, "Market should have ID"
            assert market.question, "Market should have question"
            assert 0 <= market.yes_price <= 1, "YES price should be 0-1"
            assert 0 <= market.no_price <= 1, "NO price should be 0-1"

        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_03_fetch_all_markets_including_events(self, credentials):
        """Test fetching markets including event-nested markets."""
        adapter = PolymarketAdapter(credentials=credentials)
        await adapter.connect()

        # get_all_markets includes markets from events
        all_markets = await adapter.get_all_markets(active_only=True, limit=100)

        assert len(all_markets) > 0, "Should fetch markets"

        await adapter.disconnect()

    # =========================================================================
    # PHASE 2: STRATEGY IDENTIFICATION TESTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_04_strategy_identifies_opportunities(self, credentials, strategy):
        """Test F-L strategy finds high-probability opportunities."""
        adapter = PolymarketAdapter(credentials=credentials)
        await adapter.connect()

        markets = await adapter.get_all_markets(active_only=True, limit=200)
        await adapter.disconnect()

        # Find opportunities
        opportunities = []
        for market in markets:
            opp = strategy.check_market(market)
            if opp:
                opportunities.append(opp)

        # Verify opportunities structure
        for opp in opportunities[:10]:
            assert opp.market is not None, "Opportunity should have market"
            assert opp.market.id, "Market should have ID"
            assert opp.side in ("YES", "NO"), "Side should be YES or NO"
            assert opp.price >= strategy.min_probability, "Price should meet threshold"
            # time_to_resolution may be None if market has no end_date

        print(f"\nFound {len(opportunities)} F-L opportunities out of {len(markets)} markets")

    @pytest.mark.asyncio
    async def test_05_strategy_filters_by_liquidity(self, credentials):
        """Test that low-liquidity markets are filtered out."""
        # High liquidity requirement
        strict_strategy = FavoriteLongshotStrategy(
            min_probability=0.85,
            min_liquidity=10000.0,  # $10k minimum
        )

        adapter = PolymarketAdapter(credentials=credentials)
        await adapter.connect()
        markets = await adapter.get_all_markets(active_only=True, limit=100)
        await adapter.disconnect()

        # All opportunities should meet liquidity threshold
        for market in markets:
            opp = strict_strategy.check_market(market)
            if opp:
                assert market.liquidity >= 10000.0, "Should meet liquidity requirement"

    @pytest.mark.asyncio
    async def test_06_strategy_calculates_edge(self, strategy):
        """Test edge calculation for high-probability markets."""
        # Create test market with known values
        test_market = Market(
            id="test_fl_001",
            question="Test: Will event happen? (95% likely)",
            platform="polymarket",
            yes_price=0.95,
            no_price=0.05,
            volume=50000,
            liquidity=25000,
            is_active=True,
            end_date=datetime.now(timezone.utc) + timedelta(days=7),
        )

        opp = strategy.check_market(test_market)

        assert opp is not None, "Should find opportunity"
        assert opp.side == "YES", "Should recommend YES side"
        assert opp.price == 0.95, "Price should be 0.95"

    # =========================================================================
    # PHASE 3: KELLY SIZING TESTS
    # =========================================================================

    def test_07_kelly_sizing_positive_edge(self):
        """Test Kelly criterion with positive edge."""
        kelly = KellyCriterion(default_edge=0.02)  # 2% edge

        bet = kelly.calculate(
            price=0.90,
            bankroll=1000.0,
        )

        assert isinstance(bet, KellyBet), "Should return KellyBet"
        assert bet.fraction > 0, "Fraction should be positive"
        assert bet.bet_size > 0, "Bet size should be positive"
        assert bet.bet_size <= 1000.0, "Bet size should not exceed bankroll"

    def test_08_kelly_sizing_respects_half_kelly(self):
        """Test that half-Kelly reduces bet size."""
        kelly_full = KellyCriterion(default_edge=0.03, fraction=KellyFraction.FULL)
        kelly_half = KellyCriterion(default_edge=0.03, fraction=KellyFraction.HALF)

        bet_full = kelly_full.calculate(price=0.90, bankroll=1000.0)
        bet_half = kelly_half.calculate(price=0.90, bankroll=1000.0)

        assert bet_half.bet_size < bet_full.bet_size, "Half-Kelly should be smaller"
        assert bet_half.bet_size == pytest.approx(bet_full.bet_size * 0.5, rel=0.1)

    def test_09_kelly_sizing_zero_for_no_edge(self):
        """Test Kelly returns zero for no edge."""
        kelly = KellyCriterion(default_edge=0.0)

        bet = kelly.calculate(
            price=0.90,
            bankroll=1000.0,
        )

        assert bet.fraction == 0, "Should have zero fraction"
        assert bet.bet_size == 0, "Should have zero bet size"

    # =========================================================================
    # PHASE 4: PAPER TRADING TESTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_10_paper_trade_execution(self, paper_trader):
        """Test paper trade execution."""
        initial_balance = paper_trader.cash_balance

        trade = await paper_trader.execute_paper_trade(
            market_id="test_fl_trade",
            side="YES",
            size=100,
            price=0.90,
            platform="polymarket",
            question="Test F-L trade",
        )

        assert trade is not None, "Should return trade object"
        assert trade.side == "YES", "Should be YES side"
        assert trade.size == 100, "Should have correct size"
        assert trade.price == 0.90, "Should have correct price"

        # Check balance deducted
        expected_balance = initial_balance - (100 * 0.90)
        assert paper_trader.cash_balance == pytest.approx(expected_balance)

    @pytest.mark.asyncio
    async def test_11_paper_position_tracking(self, paper_trader):
        """Test position tracking after trades."""
        await paper_trader.execute_paper_trade(
            market_id="pos_test_1",
            side="YES",
            size=50,
            price=0.85,
        )

        positions = paper_trader.get_open_positions()
        assert len(positions) == 1, "Should have 1 position"

        pos = positions[0]
        assert pos.side == "YES"
        assert pos.size == 50
        assert pos.entry_price == 0.85
        assert pos.cost_basis == 50 * 0.85

    @pytest.mark.asyncio
    async def test_12_paper_position_aggregation(self, paper_trader):
        """Test that multiple trades aggregate into one position."""
        await paper_trader.execute_paper_trade(
            market_id="agg_test",
            side="YES",
            size=50,
            price=0.80,
        )
        await paper_trader.execute_paper_trade(
            market_id="agg_test",
            side="YES",
            size=50,
            price=0.90,
        )

        positions = paper_trader.get_open_positions()
        assert len(positions) == 1, "Should aggregate into 1 position"

        pos = positions[0]
        assert pos.size == 100, "Size should be aggregated"
        assert pos.entry_price == 0.85, "Price should be averaged"
        assert pos.cost_basis == (50 * 0.80) + (50 * 0.90)

    @pytest.mark.asyncio
    async def test_13_paper_resolution_winning(self, paper_trader):
        """Test P&L calculation on winning resolution."""
        await paper_trader.execute_paper_trade(
            market_id="win_test",
            side="YES",
            size=100,
            price=0.90,
        )

        # Resolve as YES (we win)
        closed = await paper_trader.close_resolved_positions(
            resolutions={"win_test": "YES"}
        )

        assert len(closed) == 1, "Should close 1 position"
        pos = closed[0]

        # We paid $90, received $100, profit = $10
        assert pos.pnl == pytest.approx(10.0)
        assert pos.resolution_outcome == "YES"

    @pytest.mark.asyncio
    async def test_14_paper_resolution_losing(self, paper_trader):
        """Test P&L calculation on losing resolution."""
        await paper_trader.execute_paper_trade(
            market_id="loss_test",
            side="YES",
            size=100,
            price=0.90,
        )

        # Resolve as NO (we lose)
        closed = await paper_trader.close_resolved_positions(
            resolutions={"loss_test": "NO"}
        )

        assert len(closed) == 1, "Should close 1 position"
        pos = closed[0]

        # We paid $90, received $0, loss = -$90
        assert pos.pnl == pytest.approx(-90.0)
        assert pos.resolution_outcome == "NO"

    # =========================================================================
    # PHASE 5: RISK MANAGEMENT TESTS
    # =========================================================================

    def test_15_risk_manager_position_limit(self):
        """Test risk manager enforces position limits."""
        position_tracker = PositionTracker()
        limits = RiskLimits(max_position_usd=100.0)
        risk_mgr = RiskManager(limits, position_tracker)

        # Should pass
        check1 = risk_mgr.check_new_order(
            market_id="test",
            size=50,
            price=0.90,  # $45
        )
        assert check1.passed, "Should allow small position"

        # Should fail
        check2 = risk_mgr.check_new_order(
            market_id="test2",
            size=200,
            price=0.90,  # $180 > $100 limit
        )
        assert not check2.passed, "Should block large position"
        assert "exceeds max" in check2.reason.lower()

    def test_16_risk_manager_kill_switch(self):
        """Test kill switch activates on large losses."""
        position_tracker = PositionTracker()
        limits = RiskLimits(daily_loss_limit=50.0)
        risk_mgr = RiskManager(limits, position_tracker)

        assert not risk_mgr.is_kill_switch_active()

        # Record losses
        risk_mgr.update_daily_pnl(-30)
        assert not risk_mgr.is_kill_switch_active()

        risk_mgr.update_daily_pnl(-25)  # Total -55, exceeds -50 limit
        assert risk_mgr.is_kill_switch_active()

        # Should block new orders
        check = risk_mgr.check_new_order("test", 10, 0.50)
        assert not check.passed
        assert "kill switch" in check.reason.lower()

    def test_17_circuit_breaker_loss_threshold(self, circuit_breakers):
        """Test circuit breaker trips on cumulative losses."""
        assert circuit_breakers.allows_trading()

        circuit_breakers.record_loss(30)
        assert circuit_breakers.allows_trading()

        circuit_breakers.record_loss(40)
        assert circuit_breakers.allows_trading()

        circuit_breakers.record_loss(35)  # Total 105 > 100 threshold
        assert not circuit_breakers.allows_trading()
        assert "loss" in circuit_breakers.get_open_breakers()

    def test_18_circuit_breaker_consecutive_losses(self, circuit_breakers):
        """Test circuit breaker trips on consecutive losses."""
        circuit_breakers.reset_all()

        circuit_breakers.record_trade(won=True)
        circuit_breakers.record_trade(won=False)
        circuit_breakers.record_trade(won=False)
        assert circuit_breakers.allows_trading()

        circuit_breakers.record_trade(won=False)
        circuit_breakers.record_trade(won=False)
        circuit_breakers.record_trade(won=False)  # 5 consecutive
        assert not circuit_breakers.allows_trading()

    # =========================================================================
    # PHASE 6: FULL PIPELINE TEST
    # =========================================================================

    @pytest.mark.asyncio
    async def test_19_full_fl_pipeline(self, credentials, strategy, paper_trader, circuit_breakers):
        """Test complete F-L pipeline from scan to paper trade."""
        # Step 1: Connect and fetch markets
        adapter = PolymarketAdapter(credentials=credentials)
        await adapter.connect()
        markets = await adapter.get_all_markets(active_only=True, limit=100)
        await adapter.disconnect()

        assert len(markets) > 0, "Should have markets"

        # Step 2: Find opportunities
        opportunities = []
        for market in markets:
            if not circuit_breakers.allows_trading():
                break
            opp = strategy.check_market(market)
            if opp:
                opportunities.append((market, opp))

        print(f"\nFound {len(opportunities)} opportunities")

        # Step 3: Execute paper trades for top opportunities
        kelly = KellyCriterion(default_edge=0.02, fraction=KellyFraction.HALF)
        trades_executed = 0

        for market, opp in opportunities[:3]:  # Top 3
            # Calculate bet size
            bet = kelly.calculate(
                price=opp.price,
                bankroll=paper_trader.cash_balance,
            )

            if bet.bet_size < 1.0:  # Minimum $1 trade
                continue

            # Execute paper trade
            try:
                await paper_trader.execute_paper_trade(
                    market_id=market.id,
                    side=opp.side,
                    size=bet.bet_size / opp.price,  # shares = dollars / price
                    price=opp.price,
                    platform=market.platform,
                    question=market.question[:50],
                )
                trades_executed += 1
            except ValueError as e:
                print(f"Trade failed: {e}")

        print(f"Executed {trades_executed} paper trades")

        # Step 4: Check final state
        summary = paper_trader.get_position_summary()
        print(f"Final balance: ${summary['cash_balance']:.2f}")
        print(f"Open positions: {summary['open_positions']}")

        assert trades_executed > 0 or len(opportunities) == 0, "Should execute some trades if opportunities exist"

    @pytest.mark.asyncio
    async def test_20_fl_performance_report(self, paper_trader):
        """Test performance reporting after trades."""
        # Execute some test trades
        await paper_trader.execute_paper_trade("perf_1", "YES", 100, 0.90)
        await paper_trader.execute_paper_trade("perf_2", "YES", 50, 0.85)

        # Resolve one winning, one losing
        await paper_trader.close_resolved_positions({
            "perf_1": "YES",  # Win
            "perf_2": "NO",   # Loss
        })

        # Get performance report
        report = paper_trader.get_performance_report()

        assert report.total_trades == 2
        assert report.winning_trades == 1
        assert report.losing_trades == 1
        assert report.win_rate == 0.5
        assert report.total_return != 0  # Should have some P&L


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
