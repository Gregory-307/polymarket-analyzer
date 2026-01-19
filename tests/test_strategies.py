"""Tests for trading strategies."""

import pytest
from datetime import datetime, timezone

from src.adapters.base import Market
from src.strategies.single_arb import SingleConditionArbitrage
from src.strategies.multi_arb import MultiOutcomeArbitrage, OutcomePrice
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
    """Tests for favorite-longshot bias scanner."""

    def test_high_probability_yes(self):
        """Test detection of high-probability YES market."""
        market = Market(
            id="test-market",
            platform="polymarket",
            question="Will the sun rise tomorrow?",
            yes_price=0.96,
            no_price=0.04,
            end_date=datetime(2025, 1, 20, tzinfo=timezone.utc),
        )

        strategy = FavoriteLongshotStrategy(min_probability=0.95)
        opp = strategy.check_market(market)

        assert opp is not None
        assert opp.side == "YES"
        assert opp.price == 0.96

    def test_high_probability_no(self):
        """Test detection of high-probability NO market."""
        market = Market(
            id="test-market",
            platform="polymarket",
            question="Will aliens land tomorrow?",
            yes_price=0.03,
            no_price=0.97,
            end_date=datetime(2025, 1, 20, tzinfo=timezone.utc),
        )

        strategy = FavoriteLongshotStrategy(min_probability=0.95)
        opp = strategy.check_market(market)

        assert opp is not None
        assert opp.side == "NO"
        assert opp.price == 0.97

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


class TestMultiOutcomeArbitrage:
    """Tests for multi-outcome arbitrage detection."""

    def test_buy_bundle_opportunity(self):
        """Test detection of buy bundle arbitrage (sum < 1.0)."""
        detector = MultiOutcomeArbitrage(min_profit_pct=0.01, min_outcomes=3)

        outcomes = [
            {"id": "1", "name": "Outcome A", "price": 0.30},
            {"id": "2", "name": "Outcome B", "price": 0.25},
            {"id": "3", "name": "Outcome C", "price": 0.20},
            {"id": "4", "name": "Outcome D", "price": 0.15},
        ]  # Sum = 0.90

        opp = detector.check_market(
            market_id="test-multi",
            platform="polymarket",
            question="Which outcome?",
            outcomes=outcomes,
        )

        assert opp is not None
        assert opp.action == "buy_bundle"
        assert opp.sum_prices == pytest.approx(0.90, abs=0.001)
        assert opp.profit_pct == pytest.approx(0.10, abs=0.001)
        assert opp.num_outcomes == 4

    def test_sell_bundle_opportunity(self):
        """Test detection of sell bundle arbitrage (sum > 1.0)."""
        detector = MultiOutcomeArbitrage(min_profit_pct=0.01, min_outcomes=3)

        outcomes = [
            {"id": "1", "name": "Outcome A", "price": 0.40},
            {"id": "2", "name": "Outcome B", "price": 0.35},
            {"id": "3", "name": "Outcome C", "price": 0.30},
        ]  # Sum = 1.05

        opp = detector.check_market(
            market_id="test-multi",
            platform="polymarket",
            question="Which outcome?",
            outcomes=outcomes,
        )

        assert opp is not None
        assert opp.action == "sell_bundle"
        assert opp.sum_prices == pytest.approx(1.05, abs=0.001)
        assert opp.profit_pct == pytest.approx(0.05, abs=0.001)

    def test_no_opportunity_efficient_market(self):
        """Test no opportunity when sum equals 1.0."""
        detector = MultiOutcomeArbitrage(min_profit_pct=0.01, min_outcomes=3)

        outcomes = [
            {"id": "1", "name": "Outcome A", "price": 0.50},
            {"id": "2", "name": "Outcome B", "price": 0.30},
            {"id": "3", "name": "Outcome C", "price": 0.20},
        ]  # Sum = 1.00

        opp = detector.check_market(
            market_id="test-multi",
            platform="polymarket",
            question="Which outcome?",
            outcomes=outcomes,
        )

        assert opp is None

    def test_below_min_outcomes(self):
        """Test that markets with too few outcomes are ignored."""
        detector = MultiOutcomeArbitrage(min_profit_pct=0.01, min_outcomes=3)

        outcomes = [
            {"id": "1", "name": "Outcome A", "price": 0.40},
            {"id": "2", "name": "Outcome B", "price": 0.40},
        ]  # Only 2 outcomes

        opp = detector.check_market(
            market_id="test-multi",
            platform="polymarket",
            question="Which outcome?",
            outcomes=outcomes,
        )

        assert opp is None

    def test_below_profit_threshold(self):
        """Test that small profits below threshold are ignored."""
        detector = MultiOutcomeArbitrage(min_profit_pct=0.05, min_outcomes=3)

        outcomes = [
            {"id": "1", "name": "Outcome A", "price": 0.34},
            {"id": "2", "name": "Outcome B", "price": 0.33},
            {"id": "3", "name": "Outcome C", "price": 0.31},
        ]  # Sum = 0.98, profit = 2% (below 5% threshold)

        opp = detector.check_market(
            market_id="test-multi",
            platform="polymarket",
            question="Which outcome?",
            outcomes=outcomes,
        )

        assert opp is None

    def test_execution_calculation_buy(self):
        """Test execution details for buy bundle."""
        detector = MultiOutcomeArbitrage(min_profit_pct=0.01, min_outcomes=3)

        outcomes = [
            {"id": "1", "name": "A", "price": 0.30},
            {"id": "2", "name": "B", "price": 0.30},
            {"id": "3", "name": "C", "price": 0.30},
        ]  # Sum = 0.90

        opp = detector.check_market(
            market_id="test",
            platform="test",
            question="Test?",
            outcomes=outcomes,
        )

        execution = detector.calculate_execution(opp, num_bundles=10)

        assert execution["action"] == "buy_bundle"
        assert execution["num_bundles"] == 10
        assert execution["total_cost"] == pytest.approx(9.0, abs=0.01)
        assert execution["guaranteed_payout"] == pytest.approx(10.0, abs=0.01)
        assert execution["profit"] == pytest.approx(1.0, abs=0.01)
        assert len(execution["orders"]) == 3

    def test_execution_calculation_sell(self):
        """Test execution details for sell bundle."""
        detector = MultiOutcomeArbitrage(min_profit_pct=0.01, min_outcomes=3)

        outcomes = [
            {"id": "1", "name": "A", "price": 0.40},
            {"id": "2", "name": "B", "price": 0.35},
            {"id": "3", "name": "C", "price": 0.35},
        ]  # Sum = 1.10

        opp = detector.check_market(
            market_id="test",
            platform="test",
            question="Test?",
            outcomes=outcomes,
        )

        execution = detector.calculate_execution(opp, num_bundles=5)

        assert execution["action"] == "sell_bundle"
        assert execution["num_bundles"] == 5
        assert execution["total_received"] == pytest.approx(5.5, abs=0.01)
        assert execution["max_liability"] == pytest.approx(5.0, abs=0.01)
        assert execution["profit"] == pytest.approx(0.5, abs=0.01)
