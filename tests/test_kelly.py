"""Tests for Kelly criterion position sizing."""

import pytest
from src.sizing.kelly import (
    KellyBet,
    KellyCriterion,
    KellyFraction,
    kelly_bet_size,
    fractional_kelly,
    simulate_kelly_performance,
)


class TestKellyBetSize:
    """Tests for kelly_bet_size function."""

    def test_positive_edge_returns_positive_fraction(self):
        """With positive edge, Kelly suggests betting."""
        result = kelly_bet_size(
            edge=0.02,  # 2% edge
            odds=0.111,  # 90% market (odds = 0.1/0.9)
            bankroll=1000,
        )

        assert result.fraction > 0
        assert result.bet_size > 0
        assert result.edge == 0.02

    def test_zero_edge_returns_zero_fraction(self):
        """With no edge, Kelly suggests not betting."""
        result = kelly_bet_size(
            edge=0.0,
            odds=1.0,  # 50/50 market
            bankroll=1000,
        )

        assert result.fraction == 0
        assert result.bet_size == 0

    def test_negative_edge_returns_zero_fraction(self):
        """With negative edge, Kelly caps at zero (no shorting)."""
        result = kelly_bet_size(
            edge=-0.05,
            odds=1.0,
            bankroll=1000,
        )

        assert result.fraction == 0
        assert result.bet_size == 0

    def test_fractional_kelly_reduces_bet_size(self):
        """Half Kelly should reduce bet by half."""
        full_kelly = kelly_bet_size(
            edge=0.05,
            odds=1.0,
            bankroll=1000,
            fraction=1.0,
        )

        half_kelly = kelly_bet_size(
            edge=0.05,
            odds=1.0,
            bankroll=1000,
            fraction=0.5,
        )

        assert half_kelly.fraction == pytest.approx(full_kelly.fraction * 0.5, rel=0.01)
        assert half_kelly.bet_size == pytest.approx(full_kelly.bet_size * 0.5, rel=0.01)

    def test_high_odds_with_edge(self):
        """High odds (longshot) with small edge."""
        # Market at 10%, true prob 15%
        price = 0.10
        edge = 0.05
        odds = (1 - price) / price  # 9.0

        result = kelly_bet_size(
            edge=edge,
            odds=odds,
            bankroll=1000,
        )

        # Should suggest small bet on longshot
        assert 0 < result.fraction < 0.5
        assert result.odds == odds

    def test_low_odds_with_edge(self):
        """Low odds (favorite) with small edge."""
        # Market at 90%, true prob 92%
        price = 0.90
        edge = 0.02
        odds = (1 - price) / price  # 0.111

        result = kelly_bet_size(
            edge=edge,
            odds=odds,
            bankroll=1000,
        )

        # Should suggest larger fraction for favorites
        assert result.fraction > 0
        assert result.bet_size > 0

    def test_expected_growth_positive_for_positive_edge(self):
        """Positive edge should yield positive expected growth."""
        result = kelly_bet_size(
            edge=0.05,
            odds=1.0,
            bankroll=1000,
        )

        assert result.expected_growth > 0

    def test_fraction_capped_at_one(self):
        """Fraction should never exceed 100% of bankroll."""
        # Very large edge
        result = kelly_bet_size(
            edge=0.50,
            odds=1.0,
            bankroll=1000,
        )

        assert result.fraction <= 1.0
        assert result.bet_size <= 1000


class TestFractionalKelly:
    """Tests for fractional_kelly convenience function."""

    def test_half_kelly(self):
        """Half Kelly should use 0.5 fraction."""
        result = fractional_kelly(
            edge=0.05,
            price=0.50,
            bankroll=1000,
            kelly_fraction=KellyFraction.HALF,
        )

        full_result = kelly_bet_size(
            edge=0.05,
            odds=1.0,  # (1-0.5)/0.5 = 1.0
            bankroll=1000,
            fraction=1.0,
        )

        assert result.bet_size == pytest.approx(full_result.bet_size * 0.5, rel=0.01)

    def test_quarter_kelly(self):
        """Quarter Kelly should use 0.25 fraction."""
        result = fractional_kelly(
            edge=0.05,
            price=0.50,
            bankroll=1000,
            kelly_fraction=KellyFraction.QUARTER,
        )

        full_result = kelly_bet_size(
            edge=0.05,
            odds=1.0,
            bankroll=1000,
            fraction=1.0,
        )

        assert result.bet_size == pytest.approx(full_result.bet_size * 0.25, rel=0.01)

    def test_converts_price_to_odds(self):
        """Should correctly convert price to odds."""
        price = 0.60
        expected_odds = (1 - price) / price  # 0.667

        result = fractional_kelly(
            edge=0.05,
            price=price,
            bankroll=1000,
            kelly_fraction=KellyFraction.FULL,
        )

        assert result.odds == pytest.approx(expected_odds, rel=0.01)


class TestKellyCriterion:
    """Tests for KellyCriterion class."""

    def test_no_calibration_uses_default_edge(self):
        """Without calibration, should use default edge."""
        kelly = KellyCriterion(
            calibration=None,
            default_edge=0.02,
            min_edge_for_bet=0.01,
        )

        edge, confidence = kelly.estimate_edge(price=0.50)

        assert edge == 0.02
        assert confidence == 0.1  # Low confidence without calibration

    def test_edge_below_minimum_returns_zero_bet(self):
        """Edge below threshold should return zero bet."""
        kelly = KellyCriterion(
            calibration=None,
            default_edge=0.005,
            min_edge_for_bet=0.01,
        )

        result = kelly.calculate(price=0.50, bankroll=1000)

        assert result.fraction == 0
        assert result.bet_size == 0

    def test_edge_override(self):
        """Edge override should bypass calibration."""
        kelly = KellyCriterion(
            calibration=None,
            default_edge=0.01,
            min_edge_for_bet=0.005,
        )

        result = kelly.calculate(
            price=0.50,
            bankroll=1000,
            edge_override=0.10,
        )

        # Should use override edge, not default
        assert result.edge == 0.10
        assert result.bet_size > 0

    def test_calculate_for_yes_side(self):
        """YES side should use price directly."""
        kelly = KellyCriterion(
            calibration=None,
            default_edge=0.05,
            min_edge_for_bet=0.01,
        )

        result = kelly.calculate_for_opportunity(
            market_price=0.60,
            side="YES",
            bankroll=1000,
        )

        assert result.bet_size > 0

    def test_calculate_for_no_side(self):
        """NO side should use (1 - price)."""
        kelly = KellyCriterion(
            calibration=None,
            default_edge=0.05,
            min_edge_for_bet=0.01,
        )

        result = kelly.calculate_for_opportunity(
            market_price=0.60,  # NO price is effectively 0.40
            side="NO",
            bankroll=1000,
        )

        assert result.bet_size > 0


class TestKellyFraction:
    """Tests for KellyFraction enum."""

    def test_fraction_values(self):
        """Fractions should have correct values."""
        assert KellyFraction.FULL.value == 1.0
        assert KellyFraction.HALF.value == 0.5
        assert KellyFraction.QUARTER.value == 0.25
        assert KellyFraction.EIGHTH.value == 0.125


class TestKellyBetDataclass:
    """Tests for KellyBet dataclass."""

    def test_default_confidence(self):
        """Default confidence should be 1.0."""
        bet = KellyBet(
            fraction=0.1,
            bet_size=100,
            edge=0.02,
            odds=1.0,
            expected_growth=0.001,
        )

        assert bet.confidence == 1.0

    def test_all_fields(self):
        """All fields should be accessible."""
        bet = KellyBet(
            fraction=0.15,
            bet_size=150,
            edge=0.03,
            odds=0.5,
            expected_growth=0.002,
            confidence=0.8,
        )

        assert bet.fraction == 0.15
        assert bet.bet_size == 150
        assert bet.edge == 0.03
        assert bet.odds == 0.5
        assert bet.expected_growth == 0.002
        assert bet.confidence == 0.8


class TestSimulateKellyPerformance:
    """Tests for Kelly simulation."""

    def test_simulation_returns_results_for_all_fractions(self):
        """Should return results for each Kelly fraction."""
        results = simulate_kelly_performance(
            edge=0.02,
            price=0.50,
            kelly_fractions=[0.25, 0.5, 1.0],
            num_bets=100,
            num_simulations=10,
        )

        assert 0.25 in results
        assert 0.5 in results
        assert 1.0 in results

    def test_simulation_result_structure(self):
        """Each result should have expected metrics."""
        results = simulate_kelly_performance(
            edge=0.02,
            price=0.50,
            kelly_fractions=[0.5],
            num_bets=50,
            num_simulations=5,
        )

        assert "mean_final" in results[0.5]
        assert "median_final" in results[0.5]
        assert "std_final" in results[0.5]
        assert "mean_max_dd" in results[0.5]
        assert "bust_rate" in results[0.5]

    def test_positive_edge_yields_growth(self):
        """With positive edge, mean final should be above 1."""
        results = simulate_kelly_performance(
            edge=0.10,  # Large edge for reliable test
            price=0.50,
            kelly_fractions=[0.5],
            num_bets=100,
            num_simulations=50,
        )

        # With 10% edge and half Kelly, should grow on average
        assert results[0.5]["mean_final"] > 1.0
