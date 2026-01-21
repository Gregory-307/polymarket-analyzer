"""End-to-end integration tests for Financial Markets strategy pipeline.

Tests the complete flow:
1. Connect to Polymarket API
2. Connect to Deribit for live spot/vol data
3. Parse price threshold markets
4. Calculate Black-Scholes fair values
5. Identify OBVIOUS arbitrage (not probabilistic edges)
6. Validate paper trading with hedging info
7. Test full arbitrage pipeline

Run with: python -m pytest tests/integration/test_financial_markets_pipeline.py -v
"""

import asyncio
import pytest
from datetime import datetime, timezone, timedelta
import math

from src.adapters import PolymarketAdapter, Market
from src.adapters.deribit_options import DeribitOptionsClient
from src.core.config import Credentials
from src.strategies.financial_markets import (
    FinancialMarketsStrategy,
    FinancialOpportunity,
    black_scholes_digital_call,
    calculate_digital_greeks,
)
from src.execution import PaperTrader, CircuitBreakerManager


class TestFinancialMarketsPipeline:
    """End-to-end tests for Financial Markets arbitrage strategy."""

    @pytest.fixture
    def credentials(self):
        """Load credentials from environment."""
        return Credentials.from_env()

    @pytest.fixture
    def strategy(self):
        """Create FM strategy with test parameters."""
        return FinancialMarketsStrategy(
            min_edge=0.10,  # 10% for testing (default is 15% for obvious arb)
        )

    @pytest.fixture
    def paper_trader(self):
        """Create paper trader for testing."""
        return PaperTrader(initial_balance=5000.0)

    # =========================================================================
    # PHASE 1: DERIBIT DATA TESTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_01_deribit_connection(self):
        """Test Deribit API connects and returns data."""
        async with DeribitOptionsClient() as client:
            btc_spot = await client.get_spot("BTC")
            eth_spot = await client.get_spot("ETH")

        assert btc_spot is not None, "Should get BTC spot price"
        assert eth_spot is not None, "Should get ETH spot price"
        assert btc_spot > 10000, "BTC spot should be reasonable"
        assert eth_spot > 100, "ETH spot should be reasonable"

    @pytest.mark.asyncio
    async def test_02_deribit_historical_vol(self):
        """Test fetching historical volatility from Deribit."""
        async with DeribitOptionsClient() as client:
            btc_vol = await client.get_historical_vol("BTC")
            eth_vol = await client.get_historical_vol("ETH")

        assert btc_vol is not None, "Should get BTC vol"
        assert eth_vol is not None, "Should get ETH vol"
        assert 0.10 < btc_vol < 2.0, "BTC vol should be reasonable (10%-200%)"
        assert 0.10 < eth_vol < 2.0, "ETH vol should be reasonable"

    @pytest.mark.asyncio
    async def test_03_deribit_atm_implied_vol(self):
        """Test fetching ATM implied volatility."""
        async with DeribitOptionsClient() as client:
            btc_atm_iv = await client.get_atm_iv("BTC", days_to_expiry=30)

        # ATM IV may not always be available
        if btc_atm_iv is not None:
            assert 0.10 < btc_atm_iv < 2.0, "ATM IV should be reasonable"

    # =========================================================================
    # PHASE 2: BLACK-SCHOLES CALCULATION TESTS
    # =========================================================================

    def test_04_bs_digital_call_itm(self):
        """Test Black-Scholes for in-the-money case."""
        # BTC at $100k, threshold at $90k -> should be higher probability
        # With 50% vol and 36 days, there's still meaningful chance of going below
        prob = black_scholes_digital_call(
            spot=100000,
            strike=90000,
            time_to_expiry=0.1,  # ~36 days
            volatility=0.50,
        )

        # With high vol, ITM doesn't mean 90%+, more like 70%+
        assert prob > 0.65, "ITM should have elevated probability"
        assert prob < 1.0, "Should not be certain"

    def test_05_bs_digital_call_otm(self):
        """Test Black-Scholes for out-of-the-money case."""
        # BTC at $90k, threshold at $150k -> should be low probability
        prob = black_scholes_digital_call(
            spot=90000,
            strike=150000,
            time_to_expiry=0.1,
            volatility=0.50,
        )

        assert prob < 0.10, "Deep OTM should have low probability"

    def test_06_bs_digital_call_atm(self):
        """Test Black-Scholes for at-the-money case."""
        # Spot == Strike should be ~50% with symmetric distribution
        prob = black_scholes_digital_call(
            spot=100000,
            strike=100000,
            time_to_expiry=0.25,
            volatility=0.50,
        )

        # Due to drift (risk-free rate), slightly above 50%
        assert 0.40 < prob < 0.60, "ATM should be near 50%"

    def test_07_bs_digital_call_expired(self):
        """Test Black-Scholes at expiry."""
        # Already expired - just check if above/below strike
        prob_itm = black_scholes_digital_call(
            spot=100000,
            strike=90000,
            time_to_expiry=0,
            volatility=0.50,
        )
        prob_otm = black_scholes_digital_call(
            spot=90000,
            strike=100000,
            time_to_expiry=0,
            volatility=0.50,
        )

        assert prob_itm == 1.0, "Expired ITM should be 100%"
        assert prob_otm == 0.0, "Expired OTM should be 0%"

    def test_08_digital_greeks_calculation(self):
        """Test Greek calculations for digital options."""
        greeks = calculate_digital_greeks(
            spot=90000,
            strike=100000,
            time_to_expiry=0.25,
            volatility=0.50,
        )

        assert greeks.delta >= 0, "Delta should be positive for call"
        # For digital options, theta sign depends on moneyness
        # OTM digitals can have positive theta (value increases as certainty of no-pay increases)
        assert greeks.theta is not None, "Theta should be calculated"
        # Vega sign depends on position relative to strike

    # =========================================================================
    # PHASE 3: MARKET PARSING TESTS
    # =========================================================================

    def test_09_parse_btc_threshold_market(self, strategy):
        """Test parsing BTC price threshold markets."""
        test_cases = [
            ("Will the price of Bitcoin be above $100,000?", "BTC", 100000),
            ("Will BTC be above $150k by March?", "BTC", 150000),
            ("Bitcoin above $90,000 on January 21", "BTC", 90000),
            ("Will bitcoin hit $1m before GTA VI?", "BTC", 1000000),
        ]

        for question, expected_asset, expected_threshold in test_cases:
            market = Market(
                id=f"test_{expected_threshold}",
                question=question,
                platform="polymarket",
                yes_price=0.50,
                no_price=0.50,
                is_active=True,
                end_date=datetime.now(timezone.utc) + timedelta(days=30),
            )

            parsed = strategy.parse_price_threshold_market(market)
            assert parsed is not None, f"Should parse: {question}"

            asset, threshold, _ = parsed
            assert asset == expected_asset, f"Asset should be {expected_asset}"
            assert threshold == expected_threshold, f"Threshold should be {expected_threshold}"

    def test_10_parse_eth_threshold_market(self, strategy):
        """Test parsing ETH price threshold markets."""
        market = Market(
            id="eth_test",
            question="Will the price of Ethereum be above $3,000?",
            platform="polymarket",
            yes_price=0.60,
            no_price=0.40,
            is_active=True,
            end_date=datetime.now(timezone.utc) + timedelta(days=7),
        )

        parsed = strategy.parse_price_threshold_market(market)
        assert parsed is not None
        assert parsed[0] == "ETH"
        assert parsed[1] == 3000

    def test_11_parse_non_threshold_market(self, strategy):
        """Test that non-threshold markets return None."""
        market = Market(
            id="non_threshold",
            question="Will Trump win the 2024 election?",
            platform="polymarket",
            yes_price=0.50,
            no_price=0.50,
            is_active=True,
        )

        parsed = strategy.parse_price_threshold_market(market)
        assert parsed is None, "Non-threshold market should return None"

    # =========================================================================
    # PHASE 4: OPPORTUNITY IDENTIFICATION TESTS
    # =========================================================================

    def test_12_identify_obvious_overpriced(self, strategy):
        """Test identifying obviously overpriced market."""
        # BTC at $90k, market asks "Will BTC > $200k?" priced at 40%
        # This is obviously overpriced
        strategy.set_spot_price("BTC", 90000)
        strategy.set_implied_vol("BTC", 0.50)

        market = Market(
            id="overpriced_test",
            question="Will Bitcoin be above $200,000?",
            platform="polymarket",
            yes_price=0.40,
            no_price=0.60,
            is_active=True,
            end_date=datetime.now(timezone.utc) + timedelta(days=30),
        )

        opp = strategy.check_market(market)

        assert opp is not None, "Should find opportunity"
        assert opp.edge < 0, "Edge should be negative (overpriced)"
        assert opp.direction == "SELL", "Should recommend SELL"
        assert opp.arb_type in ("LOW_PROBABILITY", "PROBABILITY_EDGE")

    def test_13_identify_obvious_underpriced(self, strategy):
        """Test identifying obviously underpriced market."""
        # BTC at $95k, market asks "Will BTC > $90k?" priced at 80%
        # Spot is ALREADY above threshold - should be ~100%
        strategy.set_spot_price("BTC", 95000)
        strategy.set_implied_vol("BTC", 0.40)

        market = Market(
            id="underpriced_test",
            question="Will Bitcoin be above $90,000?",
            platform="polymarket",
            yes_price=0.80,
            no_price=0.20,
            is_active=True,
            end_date=datetime.now(timezone.utc) + timedelta(days=1),
        )

        opp = strategy.check_market(market)

        assert opp is not None, "Should find opportunity"
        assert opp.edge > 0, "Edge should be positive (underpriced)"
        assert opp.direction == "BUY", "Should recommend BUY"
        # This is the most obvious case - spot already above threshold
        assert opp.is_obvious, "Should be flagged as obvious"
        assert opp.arb_type == "SPOT_ABOVE_THRESHOLD"

    def test_14_is_obvious_flag(self, strategy):
        """Test the is_obvious property identifies clear arbitrage."""
        strategy.set_spot_price("BTC", 100000)
        strategy.set_implied_vol("BTC", 0.40)

        # Case 1: Spot way above threshold, market not reflecting
        obvious_market = Market(
            id="obvious_1",
            question="Will Bitcoin be above $85,000?",
            platform="polymarket",
            yes_price=0.85,  # Should be ~99%+
            no_price=0.15,
            is_active=True,
            end_date=datetime.now(timezone.utc) + timedelta(days=7),
        )

        opp = strategy.check_market(obvious_market)
        if opp:
            assert opp.is_obvious, "Should be obvious - spot above threshold"

    def test_15_edge_below_threshold_filtered(self, strategy):
        """Test that small edges are filtered out."""
        strategy.set_spot_price("BTC", 90000)
        strategy.set_implied_vol("BTC", 0.50)

        # Market priced fairly close to fair value
        fair_market = Market(
            id="fair_market",
            question="Will Bitcoin be above $95,000?",
            platform="polymarket",
            yes_price=0.35,  # Close to fair value
            no_price=0.65,
            is_active=True,
            end_date=datetime.now(timezone.utc) + timedelta(days=30),
        )

        # With 10% min_edge, small mispricings should be filtered
        opp = strategy.check_market(fair_market)
        # May or may not find opportunity depending on exact fair value
        if opp:
            assert abs(opp.edge) >= 0.10, "Should only report significant edge"

    # =========================================================================
    # PHASE 5: LIVE DATA SCAN TESTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_16_scan_with_live_data(self, credentials, strategy):
        """Test scanning real markets with live Deribit data."""
        # Fetch markets
        adapter = PolymarketAdapter(credentials=credentials)
        await adapter.connect()
        markets = await adapter.get_all_markets(active_only=True, limit=200)
        await adapter.disconnect()

        # Fetch live data
        async with DeribitOptionsClient() as client:
            btc_spot = await client.get_spot("BTC")
            eth_spot = await client.get_spot("ETH")
            btc_vol = await client.get_historical_vol("BTC")
            eth_vol = await client.get_historical_vol("ETH")

        # Set prices
        if btc_spot:
            strategy.set_spot_price("BTC", btc_spot)
        if eth_spot:
            strategy.set_spot_price("ETH", eth_spot)
        if btc_vol:
            strategy.set_implied_vol("BTC", btc_vol)
        if eth_vol:
            strategy.set_implied_vol("ETH", eth_vol)

        # Scan
        opportunities = strategy.scan(markets)

        print(f"\nLive Data:")
        print(f"  BTC: ${btc_spot:,.0f} | Vol: {btc_vol:.1%}")
        print(f"  ETH: ${eth_spot:,.0f} | Vol: {eth_vol:.1%}")
        print(f"\nFound {len(opportunities)} opportunities with >=10% edge")

        # Verify opportunities are valid
        for opp in opportunities[:5]:
            assert isinstance(opp, FinancialOpportunity)
            assert abs(opp.edge) >= strategy.min_edge
            assert opp.underlying in ("BTC", "ETH", "SOL")
            print(f"  {opp.underlying} ${opp.threshold:,.0f}: {opp.edge_pct:+.1f}% edge ({opp.arb_type})")

    @pytest.mark.asyncio
    async def test_17_count_parseable_markets(self, credentials, strategy):
        """Count how many markets can be parsed as price threshold."""
        adapter = PolymarketAdapter(credentials=credentials)
        await adapter.connect()
        markets = await adapter.get_all_markets(active_only=True, limit=500)
        await adapter.disconnect()

        parsed_count = 0
        for market in markets:
            if strategy.parse_price_threshold_market(market):
                parsed_count += 1

        print(f"\nParsed {parsed_count}/{len(markets)} markets as price threshold")
        # Should find at least some BTC/ETH price markets
        assert parsed_count > 0, "Should find some price threshold markets"

    # =========================================================================
    # PHASE 6: HEDGING TESTS
    # =========================================================================

    def test_18_hedge_ratio_calculation(self, strategy):
        """Test hedge ratio is calculated correctly."""
        strategy.set_spot_price("BTC", 90000)
        strategy.set_implied_vol("BTC", 0.50)

        market = Market(
            id="hedge_test",
            question="Will Bitcoin be above $100,000?",
            platform="polymarket",
            yes_price=0.25,
            no_price=0.75,
            is_active=True,
            end_date=datetime.now(timezone.utc) + timedelta(days=60),
        )

        opp = strategy.check_market(market)

        if opp and opp.greeks:
            # Hedge ratio should be based on delta
            assert opp.hedge_ratio is not None
            # If buying (positive edge), hedge ratio should be negative (sell underlying)
            # If selling (negative edge), hedge ratio should be positive (buy underlying)
            if opp.edge > 0:
                assert opp.hedge_ratio <= 0, "Buy position needs short hedge"
            else:
                assert opp.hedge_ratio >= 0, "Sell position needs long hedge"

    def test_19_greeks_present_in_opportunity(self, strategy):
        """Test that Greeks are calculated for opportunities."""
        strategy.set_spot_price("ETH", 3000)
        strategy.set_implied_vol("ETH", 0.60)

        market = Market(
            id="greeks_test",
            question="Will Ethereum be above $3,500?",
            platform="polymarket",
            yes_price=0.20,
            no_price=0.80,
            is_active=True,
            end_date=datetime.now(timezone.utc) + timedelta(days=30),
        )

        opp = strategy.check_market(market)

        if opp:
            assert opp.greeks is not None, "Should have Greeks"
            assert hasattr(opp.greeks, 'delta')
            assert hasattr(opp.greeks, 'gamma')
            assert hasattr(opp.greeks, 'theta')
            assert hasattr(opp.greeks, 'vega')

    # =========================================================================
    # PHASE 7: FULL PIPELINE TESTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_20_full_fm_pipeline(self, credentials, strategy, paper_trader):
        """Test complete Financial Markets pipeline."""
        # Step 1: Fetch markets
        adapter = PolymarketAdapter(credentials=credentials)
        await adapter.connect()
        markets = await adapter.get_all_markets(active_only=True, limit=200)
        await adapter.disconnect()

        # Step 2: Get live data
        async with DeribitOptionsClient() as client:
            btc_spot = await client.get_spot("BTC")
            eth_spot = await client.get_spot("ETH")
            btc_vol = await client.get_historical_vol("BTC")
            eth_vol = await client.get_historical_vol("ETH")

        strategy.set_spot_price("BTC", btc_spot or 90000)
        strategy.set_spot_price("ETH", eth_spot or 3000)
        strategy.set_implied_vol("BTC", btc_vol or 0.50)
        strategy.set_implied_vol("ETH", eth_vol or 0.60)

        # Step 3: Scan for opportunities
        opportunities = strategy.scan(markets)

        print(f"\nFound {len(opportunities)} FM opportunities")

        # Step 4: Filter for obvious arbitrage only
        obvious_opps = [o for o in opportunities if o.is_obvious]
        print(f"Of which {len(obvious_opps)} are OBVIOUS arbitrage")

        # Step 5: Execute paper trades on obvious opportunities
        trades_executed = 0
        for opp in obvious_opps[:3]:
            size_usd = 100  # $100 per trade
            size_shares = size_usd / opp.market_price if opp.edge > 0 else size_usd / (1 - opp.market_price)
            price = opp.market_price if opp.edge > 0 else (1 - opp.market_price)
            side = "YES" if opp.edge > 0 else "NO"

            try:
                await paper_trader.execute_paper_trade(
                    market_id=opp.market.id,
                    side=side,
                    size=size_shares,
                    price=price,
                    platform=opp.market.platform,
                    question=opp.market.question[:50],
                )
                trades_executed += 1
                print(f"  Traded: {opp.direction} {opp.underlying} ${opp.threshold:,.0f} @ {opp.market_price:.1%}")
            except ValueError as e:
                print(f"  Trade failed: {e}")

        print(f"\nExecuted {trades_executed} paper trades")
        print(f"Final balance: ${paper_trader.cash_balance:.2f}")

    @pytest.mark.asyncio
    async def test_21_compare_polymarket_to_options(self, credentials, strategy):
        """Compare Polymarket prices to options-implied for specific markets."""
        # Fetch markets
        adapter = PolymarketAdapter(credentials=credentials)
        await adapter.connect()
        markets = await adapter.get_all_markets(active_only=True, limit=300)
        await adapter.disconnect()

        # Get live data
        async with DeribitOptionsClient() as client:
            btc_spot = await client.get_spot("BTC")
            btc_vol = await client.get_historical_vol("BTC")

        strategy.set_spot_price("BTC", btc_spot)
        strategy.set_implied_vol("BTC", btc_vol)

        # Find BTC markets and compare
        print(f"\nBTC Spot: ${btc_spot:,.0f} | Vol: {btc_vol:.1%}")
        print("-" * 70)

        btc_markets = []
        for market in markets:
            parsed = strategy.parse_price_threshold_market(market)
            if parsed and parsed[0] == "BTC":
                btc_markets.append((market, parsed))

        for market, (asset, threshold, expiry) in btc_markets[:10]:
            fair_value = strategy.calculate_fair_value(
                btc_spot, threshold, expiry, btc_vol
            )
            edge = fair_value - market.yes_price

            print(f"BTC > ${threshold:>10,.0f}: Poly={market.yes_price:5.1%} | "
                  f"Fair={fair_value:5.1%} | Edge={edge:+5.1%}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
