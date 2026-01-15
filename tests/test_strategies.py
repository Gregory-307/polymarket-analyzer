"""Tests for trading strategies."""

import pytest
from datetime import datetime, timezone

from src.adapters.base import Market
from src.strategies.single_arb import SingleConditionArbitrage
from src.strategies.favorite_longshot import FavoriteLongshotStrategy


class TestSingleConditionArbitrage:
    """Tests for single-condition arbitrage detection."""

    def test_buy_all_opportunity(self):
        """Test detection of buy-all arbitrage (YES + NO < 1)."""
        market = Market(
            id="test-market",
            platform="polymarket",
            question="Will X happen?",
            yes_price=0.45,
            no_price=0.45,  # Sum = 0.90
        )

        detector = SingleConditionArbitrage(min_profit_pct=0.01)
        opp = detector.check_market(market)

        assert opp is not None
        assert opp.action == "buy_all"
        assert opp.profit_pct == pytest.approx(0.10, abs=0.001)

    def test_sell_all_opportunity(self):
        """Test detection of sell-all arbitrage (YES + NO > 1)."""
        market = Market(
            id="test-market",
            platform="polymarket",
            question="Will X happen?",
            yes_price=0.55,
            no_price=0.55,  # Sum = 1.10
        )

        detector = SingleConditionArbitrage(min_profit_pct=0.01)
        opp = detector.check_market(market)

        assert opp is not None
        assert opp.action == "sell_all"
        assert opp.profit_pct == pytest.approx(0.10, abs=0.001)

    def test_no_opportunity_efficient_market(self):
        """Test no opportunity in efficient market (YES + NO = 1)."""
        market = Market(
            id="test-market",
            platform="polymarket",
            question="Will X happen?",
            yes_price=0.60,
            no_price=0.40,  # Sum = 1.00
        )

        detector = SingleConditionArbitrage(min_profit_pct=0.01)
        opp = detector.check_market(market)

        assert opp is None

    def test_below_threshold(self):
        """Test opportunity below minimum threshold is ignored."""
        market = Market(
            id="test-market",
            platform="polymarket",
            question="Will X happen?",
            yes_price=0.499,
            no_price=0.499,  # Sum = 0.998, profit = 0.2%
        )

        detector = SingleConditionArbitrage(min_profit_pct=0.01)  # 1% threshold
        opp = detector.check_market(market)

        assert opp is None


class TestFavoriteLongshotStrategy:
    """Tests for favorite-longshot bias strategy."""

    def test_high_probability_yes(self):
        """Test detection of high-probability YES opportunity."""
        market = Market(
            id="test-market",
            platform="polymarket",
            question="Will the sun rise tomorrow?",
            yes_price=0.96,
            no_price=0.04,
            end_date=datetime(2025, 1, 20, tzinfo=timezone.utc),
        )

        strategy = FavoriteLongshotStrategy(
            min_probability=0.95,
            min_edge=0.005,
        )
        opp = strategy.check_market(market)

        assert opp is not None
        assert opp.side == "YES"
        assert opp.current_price == 0.96
        assert opp.edge > 0

    def test_high_probability_no(self):
        """Test detection of high-probability NO opportunity."""
        market = Market(
            id="test-market",
            platform="polymarket",
            question="Will aliens land tomorrow?",
            yes_price=0.03,
            no_price=0.97,
            end_date=datetime(2025, 1, 20, tzinfo=timezone.utc),
        )

        strategy = FavoriteLongshotStrategy(
            min_probability=0.95,
            min_edge=0.005,
        )
        opp = strategy.check_market(market)

        assert opp is not None
        assert opp.side == "NO"
        assert opp.current_price == 0.97

    def test_mid_probability_ignored(self):
        """Test that mid-probability markets are ignored."""
        market = Market(
            id="test-market",
            platform="polymarket",
            question="Will it rain next week?",
            yes_price=0.50,
            no_price=0.50,
        )

        strategy = FavoriteLongshotStrategy(min_probability=0.95)
        opp = strategy.check_market(market)

        assert opp is None

    def test_position_sizing(self):
        """Test Kelly-based position sizing."""
        market = Market(
            id="test-market",
            platform="polymarket",
            question="Near-certain event",
            yes_price=0.96,
            no_price=0.04,
            liquidity=10000,
        )

        strategy = FavoriteLongshotStrategy(
            min_probability=0.95,
            min_edge=0.005,
            max_position_usd=500,
        )
        opp = strategy.check_market(market)

        assert opp is not None

        size = strategy.calculate_position_size(
            opp,
            account_balance=10000,
            max_risk_pct=0.05,
        )

        # Should be constrained by max_position_usd
        assert size <= 500
        assert size > 0


class TestMarketDataclass:
    """Tests for Market dataclass."""

    def test_arb_check_balanced(self):
        """Test arb_check for balanced market."""
        market = Market(
            id="test",
            platform="test",
            question="Test?",
            yes_price=0.60,
            no_price=0.40,
        )

        assert market.arb_check == pytest.approx(1.0)

    def test_arb_check_buy_all(self):
        """Test arb_check for buy-all opportunity."""
        market = Market(
            id="test",
            platform="test",
            question="Test?",
            yes_price=0.45,
            no_price=0.45,
        )

        assert market.arb_check == pytest.approx(0.90)
        assert market.arb_check < 1.0  # Buy-all opportunity

    def test_implied_probability(self):
        """Test implied probability calculation."""
        market = Market(
            id="test",
            platform="test",
            question="Test?",
            yes_price=0.75,
            no_price=0.25,
        )

        assert market.implied_probability == 0.75

    def test_is_binary(self):
        """Test binary market detection."""
        binary_market = Market(
            id="test",
            platform="test",
            question="Binary?",
            outcomes=["Yes", "No"],
        )

        multi_market = Market(
            id="test",
            platform="test",
            question="Multi?",
            outcomes=["A", "B", "C", "D"],
        )

        assert binary_market.is_binary is True
        assert multi_market.is_binary is False
