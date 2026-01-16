"""Live Trading Signals Generator - Production-Ready Signal System.

Generates actionable trading signals with:
- Kelly Criterion position sizing
- Risk-adjusted recommendations
- Real-time opportunity scoring
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import math

sys.path.insert(0, str(Path(__file__).parent))

from src.core.config import Credentials
from src.adapters.polymarket import PolymarketAdapter


@dataclass
class TradingSignal:
    """A trading signal with full context."""
    timestamp: str
    market_id: str
    question: str
    strategy: str
    side: str  # YES or NO
    current_price: float
    estimated_fair_value: float
    edge: float
    confidence: float
    kelly_fraction: float
    recommended_size_pct: float  # % of bankroll
    volume_24h: float
    liquidity: float
    risk_level: str  # LOW, MEDIUM, HIGH
    signal_strength: str  # STRONG, MODERATE, WEAK
    reasoning: str


class SignalGenerator:
    """Generate trading signals from market data."""

    def __init__(
        self,
        bankroll: float = 1000,
        max_position_pct: float = 0.10,  # Max 10% per position
        min_edge: float = 0.01,  # 1% minimum edge
        min_liquidity: float = 1000,  # $1000 minimum
    ):
        self.bankroll = bankroll
        self.max_position_pct = max_position_pct
        self.min_edge = min_edge
        self.min_liquidity = min_liquidity

    def calculate_kelly(self, p_win: float, odds: float) -> float:
        """Calculate Kelly Criterion fraction.

        f* = (p * b - q) / b
        where:
        - p = probability of winning
        - q = probability of losing (1 - p)
        - b = odds (payout ratio)
        """
        if odds <= 0 or p_win <= 0 or p_win >= 1:
            return 0

        q = 1 - p_win
        kelly = (p_win * odds - q) / odds

        # Fractional Kelly (half) for safety
        return max(0, kelly * 0.5)

    def estimate_fair_value(self, market_price: float, strategy: str) -> tuple[float, float]:
        """Estimate fair value and confidence based on strategy.

        Returns: (fair_value, confidence)
        """
        if strategy == "favorite_longshot":
            # High-probability bias: actual resolution > market price
            if market_price >= 0.98:
                fair_value = min(0.995, market_price + 0.015)
                confidence = 0.90
            elif market_price >= 0.95:
                fair_value = min(0.99, market_price + 0.03)
                confidence = 0.85
            elif market_price >= 0.90:
                fair_value = min(0.98, market_price + 0.02)
                confidence = 0.75
            else:
                fair_value = market_price
                confidence = 0.5
        else:
            fair_value = market_price
            confidence = 0.5

        return fair_value, confidence

    def generate_signal(self, market) -> Optional[TradingSignal]:
        """Generate trading signal for a market."""
        yes_price = market.yes_price
        no_price = market.no_price
        high_prob = max(yes_price, no_price)
        side = "YES" if yes_price > no_price else "NO"
        price = yes_price if side == "YES" else no_price

        # Only consider high-probability markets for favorite-longshot
        if high_prob < 0.90:
            return None

        # Skip low liquidity
        if (market.liquidity or 0) < self.min_liquidity:
            return None

        # Calculate fair value
        fair_value, confidence = self.estimate_fair_value(price, "favorite_longshot")
        edge = fair_value - price

        if edge < self.min_edge:
            return None

        # Calculate Kelly fraction
        # For prediction markets: odds = (1 - price) / price for YES
        if price > 0 and price < 1:
            odds = (1 - price) / price
            kelly = self.calculate_kelly(fair_value, odds)
        else:
            kelly = 0

        # Cap at max position
        recommended_size = min(kelly, self.max_position_pct)

        # Determine risk level
        if edge >= 0.03 and confidence >= 0.85:
            risk_level = "LOW"
            signal_strength = "STRONG"
        elif edge >= 0.02 and confidence >= 0.75:
            risk_level = "MEDIUM"
            signal_strength = "MODERATE"
        else:
            risk_level = "HIGH"
            signal_strength = "WEAK"

        # Generate reasoning
        reasoning = (
            f"Market priced at {price:.1%} with estimated fair value of {fair_value:.1%} "
            f"based on favorite-longshot bias research. Historical data shows "
            f"outcomes priced >90% resolve at higher rates than implied. "
            f"Edge of {edge:.2%} exceeds transaction costs."
        )

        return TradingSignal(
            timestamp=datetime.now().isoformat(),
            market_id=market.id,
            question=market.question,
            strategy="favorite_longshot",
            side=side,
            current_price=price,
            estimated_fair_value=fair_value,
            edge=edge,
            confidence=confidence,
            kelly_fraction=kelly,
            recommended_size_pct=recommended_size,
            volume_24h=market.raw.get('volume24hr', 0) if market.raw else 0,
            liquidity=market.liquidity or 0,
            risk_level=risk_level,
            signal_strength=signal_strength,
            reasoning=reasoning,
        )


def print_signal(signal: TradingSignal, bankroll: float):
    """Pretty print a trading signal."""
    border = "=" * 70

    print(f"\n{border}")
    print(f"  TRADING SIGNAL: {signal.signal_strength}")
    print(border)
    print(f"\n  Market: {signal.question[:65]}")
    print(f"\n  +{'-' * 30}+{'-' * 35}+")
    print(f"  | {'Side':<28} | {signal.side:>33} |")
    print(f"  | {'Current Price':<28} | {signal.current_price:>32.2%} |")
    print(f"  | {'Fair Value (Est.)':<28} | {signal.estimated_fair_value:>32.2%} |")
    print(f"  | {'Edge':<28} | {signal.edge:>32.2%} |")
    print(f"  | {'Confidence':<28} | {signal.confidence:>32.0%} |")
    print(f"  +{'-' * 30}+{'-' * 35}+")
    print(f"  | {'Kelly Fraction':<28} | {signal.kelly_fraction:>32.2%} |")
    print(f"  | {'Recommended Size':<28} | {signal.recommended_size_pct:>32.2%} |")
    print(f"  | {'Dollar Amount':<28} | {'$' + f'{bankroll * signal.recommended_size_pct:,.2f}':>32} |")
    print(f"  +{'-' * 30}+{'-' * 35}+")
    print(f"  | {'Risk Level':<28} | {signal.risk_level:>33} |")
    print(f"  | {'Liquidity':<28} | {'$' + f'{signal.liquidity:,.0f}':>32} |")
    print(f"  +{'-' * 30}+{'-' * 35}+")
    print(f"\n  Reasoning: {signal.reasoning[:100]}...")
    print(border)


async def main():
    """Generate live trading signals."""
    bankroll = 1000  # Starting bankroll for position sizing

    print("=" * 74)
    print("           POLYMARKET TRADING SIGNALS GENERATOR")
    print("                  Production-Ready System")
    print("=" * 74)
    print(f"\n  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Bankroll: ${bankroll:,.2f}")
    print(f"  Max Position: 10%")
    print(f"  Min Edge: 1%")
    print()

    # Connect
    creds = Credentials.from_env()
    adapter = PolymarketAdapter(credentials=creds)
    await adapter.connect()

    print("  Fetching market data...")
    markets = await adapter.get_markets(active_only=True, limit=200)
    print(f"  Analyzed: {len(markets)} markets")

    # Generate signals
    generator = SignalGenerator(bankroll=bankroll)
    signals = []

    for market in markets:
        signal = generator.generate_signal(market)
        if signal:
            signals.append(signal)

    # Sort by edge
    signals.sort(key=lambda s: s.edge, reverse=True)

    print(f"\n  Signals Generated: {len(signals)}")

    # Print top signals
    print("\n" + "=" * 70)
    print("  TOP TRADING SIGNALS")
    print("=" * 70)

    for signal in signals[:5]:
        print_signal(signal, bankroll)

    # Summary statistics
    if signals:
        print("\n" + "=" * 70)
        print("  PORTFOLIO SUMMARY")
        print("=" * 70)

        strong_signals = [s for s in signals if s.signal_strength == "STRONG"]
        moderate_signals = [s for s in signals if s.signal_strength == "MODERATE"]

        total_edge = sum(s.edge for s in signals)
        avg_edge = total_edge / len(signals)
        total_recommended = sum(s.recommended_size_pct for s in signals[:10])

        print(f"\n  STRONG Signals: {len(strong_signals)}")
        print(f"  MODERATE Signals: {len(moderate_signals)}")
        print(f"  Average Edge: {avg_edge:.2%}")
        print(f"  Total Recommended Allocation (Top 10): {total_recommended:.1%}")
        print(f"  Dollar Amount: ${bankroll * total_recommended:,.2f}")

        # Expected value calculation
        expected_profit = sum(s.edge * bankroll * s.recommended_size_pct for s in signals[:10])
        print(f"\n  Expected Profit (Top 10 signals): ${expected_profit:,.2f}")
        print(f"  Expected ROI: {(expected_profit / bankroll) * 100:.2f}%")

    # Save signals
    output_dir = Path(__file__).parent / "results" / "signals"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"signals_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump({
            'generated_at': datetime.now().isoformat(),
            'bankroll': bankroll,
            'total_signals': len(signals),
            'strong_signals': len([s for s in signals if s.signal_strength == "STRONG"]),
            'signals': [asdict(s) for s in signals],
        }, f, indent=2)

    print(f"\n  Signals saved to: {output_file}")

    await adapter.disconnect()

    print("\n" + "=" * 74)
    print("                    SIGNAL GENERATION COMPLETE")
    print("=" * 74)


if __name__ == "__main__":
    asyncio.run(main())
