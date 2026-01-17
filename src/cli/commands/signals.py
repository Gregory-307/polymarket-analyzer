"""Signals command - Generate trading signals with Kelly sizing."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import click

from ..utils import async_command, print_header, print_subheader, format_price, format_usd
from ...core.config import Credentials
from ...adapters.polymarket import PolymarketAdapter
from ...adapters.base import Market


@dataclass
class TradingSignal:
    """A trading signal with full context."""

    timestamp: str
    market_id: str
    question: str
    strategy: str
    side: str
    current_price: float
    estimated_fair_value: float
    edge: float
    confidence: float
    kelly_fraction: float
    recommended_size_pct: float
    volume_24h: float
    liquidity: float
    risk_level: str
    signal_strength: str
    reasoning: str


class SignalGenerator:
    """Generate trading signals from market data."""

    def __init__(
        self,
        bankroll: float = 1000,
        max_position_pct: float = 0.10,
        min_edge: float = 0.01,
        min_liquidity: float = 1000,
    ):
        self.bankroll = bankroll
        self.max_position_pct = max_position_pct
        self.min_edge = min_edge
        self.min_liquidity = min_liquidity

    def calculate_kelly(self, p_win: float, odds: float) -> float:
        """Calculate Kelly Criterion fraction (half Kelly for safety)."""
        if odds <= 0 or p_win <= 0 or p_win >= 1:
            return 0
        q = 1 - p_win
        kelly = (p_win * odds - q) / odds
        return max(0, kelly * 0.5)

    def estimate_fair_value(self, market_price: float) -> tuple[float, float]:
        """Estimate fair value and confidence based on favorite-longshot bias."""
        if market_price >= 0.98:
            return min(0.995, market_price + 0.015), 0.90
        elif market_price >= 0.95:
            return min(0.99, market_price + 0.03), 0.85
        elif market_price >= 0.90:
            return min(0.98, market_price + 0.02), 0.75
        return market_price, 0.5

    def generate_signal(self, market: Market) -> TradingSignal | None:
        """Generate trading signal for a market."""
        yes_price = market.yes_price
        no_price = market.no_price
        high_prob = max(yes_price, no_price)
        side = "YES" if yes_price > no_price else "NO"
        price = yes_price if side == "YES" else no_price

        if high_prob < 0.90:
            return None

        if (market.liquidity or 0) < self.min_liquidity:
            return None

        fair_value, confidence = self.estimate_fair_value(price)
        edge = fair_value - price

        if edge < self.min_edge:
            return None

        # Kelly calculation
        if 0 < price < 1:
            odds = (1 - price) / price
            kelly = self.calculate_kelly(fair_value, odds)
        else:
            kelly = 0

        recommended_size = min(kelly, self.max_position_pct)

        # Risk level
        if edge >= 0.03 and confidence >= 0.85:
            risk_level, signal_strength = "LOW", "STRONG"
        elif edge >= 0.02 and confidence >= 0.75:
            risk_level, signal_strength = "MEDIUM", "MODERATE"
        else:
            risk_level, signal_strength = "HIGH", "WEAK"

        reasoning = (
            f"Market priced at {price:.1%} with estimated fair value of {fair_value:.1%} "
            f"based on favorite-longshot bias research."
        )

        return TradingSignal(
            timestamp=datetime.now().isoformat(),
            market_id=market.id,
            question=market.question or "",
            strategy="favorite_longshot",
            side=side,
            current_price=price,
            estimated_fair_value=fair_value,
            edge=edge,
            confidence=confidence,
            kelly_fraction=kelly,
            recommended_size_pct=recommended_size,
            volume_24h=market.raw.get("volume24hr", 0) if market.raw else 0,
            liquidity=market.liquidity or 0,
            risk_level=risk_level,
            signal_strength=signal_strength,
            reasoning=reasoning,
        )


@click.command()
@click.option("--bankroll", type=float, default=1000, help="Starting bankroll for sizing.")
@click.option("--min-edge", type=float, default=0.01, help="Minimum edge (default: 1%).")
@click.option("--min-liquidity", type=float, default=1000, help="Minimum liquidity ($).")
@click.option("--max-position", type=float, default=0.10, help="Max position (% of bankroll).")
@click.option("--limit", type=int, default=200, help="Markets to analyze.")
@click.option("--output", type=click.Choice(["table", "json"]), default="table")
@async_command
async def signals(
    bankroll: float,
    min_edge: float,
    min_liquidity: float,
    max_position: float,
    limit: int,
    output: str,
) -> None:
    """Generate trading signals with Kelly Criterion position sizing.

    Analyzes markets and generates actionable trading signals with:
    - Fair value estimation based on favorite-longshot bias
    - Kelly Criterion position sizing (half Kelly for safety)
    - Risk level and signal strength classification

    \b
    Examples:
      python -m src signals
      python -m src signals --bankroll 5000 --min-edge 0.02
      python -m src signals --output json
    """
    print_header("TRADING SIGNALS GENERATOR")
    click.echo(f"\n  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    click.echo(f"  Bankroll: {format_usd(bankroll)}")
    click.echo(f"  Min Edge: {format_price(min_edge)}")
    click.echo(f"  Max Position: {format_price(max_position)}")

    creds = Credentials.from_env()
    adapter = PolymarketAdapter(credentials=creds)

    click.echo("\n  Fetching market data...")
    await adapter.connect()
    markets = await adapter.get_markets(active_only=True, limit=limit)
    click.echo(f"  Analyzed: {len(markets)} markets")

    generator = SignalGenerator(
        bankroll=bankroll,
        max_position_pct=max_position,
        min_edge=min_edge,
        min_liquidity=min_liquidity,
    )

    signals_list = []
    for market in markets:
        signal = generator.generate_signal(market)
        if signal:
            signals_list.append(signal)

    signals_list.sort(key=lambda s: s.edge, reverse=True)
    click.echo(f"  Signals Generated: {len(signals_list)}")

    await adapter.disconnect()

    if output == "json":
        data = {
            "generated_at": datetime.now().isoformat(),
            "bankroll": bankroll,
            "total_signals": len(signals_list),
            "signals": [asdict(s) for s in signals_list],
        }
        click.echo(json.dumps(data, indent=2))
    else:
        _print_signals(signals_list, bankroll)

    # Save results
    output_dir = Path("results/signals")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"signals_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(
            {
                "generated_at": datetime.now().isoformat(),
                "bankroll": bankroll,
                "total_signals": len(signals_list),
                "signals": [asdict(s) for s in signals_list],
            },
            f,
            indent=2,
        )
    click.echo(f"\n  Saved to: {output_file}")


def _print_signals(signals_list: list[TradingSignal], bankroll: float) -> None:
    """Print signals in table format."""
    print_subheader("TOP TRADING SIGNALS")

    for signal in signals_list[:5]:
        click.echo("\n" + "=" * 70)
        click.echo(f"  SIGNAL: {signal.signal_strength}")
        click.echo("=" * 70)
        click.echo(f"\n  Market: {signal.question[:60]}")
        click.echo(f"\n  {'Side':<25} {signal.side:>20}")
        click.echo(f"  {'Current Price':<25} {format_price(signal.current_price):>20}")
        click.echo(f"  {'Fair Value (Est.)':<25} {format_price(signal.estimated_fair_value):>20}")
        click.echo(f"  {'Edge':<25} {format_price(signal.edge):>20}")
        click.echo(f"  {'Confidence':<25} {format_price(signal.confidence):>20}")
        click.echo(f"  {'Kelly Fraction':<25} {format_price(signal.kelly_fraction):>20}")
        click.echo(f"  {'Recommended Size':<25} {format_price(signal.recommended_size_pct):>20}")
        click.echo(f"  {'Dollar Amount':<25} {format_usd(bankroll * signal.recommended_size_pct):>20}")
        click.echo(f"  {'Risk Level':<25} {signal.risk_level:>20}")
        click.echo(f"  {'Liquidity':<25} {format_usd(signal.liquidity):>20}")

    if signals_list:
        print_subheader("PORTFOLIO SUMMARY")
        strong = [s for s in signals_list if s.signal_strength == "STRONG"]
        moderate = [s for s in signals_list if s.signal_strength == "MODERATE"]
        avg_edge = sum(s.edge for s in signals_list) / len(signals_list)
        total_rec = sum(s.recommended_size_pct for s in signals_list[:10])

        click.echo(f"\n  STRONG Signals: {len(strong)}")
        click.echo(f"  MODERATE Signals: {len(moderate)}")
        click.echo(f"  Average Edge: {format_price(avg_edge)}")
        click.echo(f"  Total Allocation (Top 10): {format_price(total_rec)}")

        expected_profit = sum(s.edge * bankroll * s.recommended_size_pct for s in signals_list[:10])
        click.echo(f"\n  Expected Profit (Top 10): {format_usd(expected_profit)}")
        click.echo(f"  Expected ROI: {format_price(expected_profit / bankroll)}")
