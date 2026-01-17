"""Cross-platform analysis command."""

from __future__ import annotations

import json
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import click

from ...utils import async_command, print_header, print_subheader, format_price, format_usd
from ....core.config import Credentials
from ....adapters.polymarket import PolymarketAdapter
from ....adapters.kalshi import KalshiAdapter


@click.command("cross-platform")
@click.option("--min-spread", type=float, default=0.02, help="Minimum price spread to report.")
@click.option("--similarity", type=float, default=0.6, help="Minimum question similarity (0-1).")
@click.option("--limit", type=int, default=100, help="Markets to fetch per platform.")
@click.option("--output", type=click.Choice(["table", "json"]), default="table")
@click.option("--visualize/--no-visualize", default=False)
@async_command
async def cross_platform(
    min_spread: float,
    similarity: float,
    limit: int,
    output: str,
    visualize: bool,
) -> None:
    """Analyze price discrepancies between Polymarket and Kalshi.

    Identifies same-event markets across platforms and calculates
    potential cross-platform arbitrage opportunities.

    Research (arXiv:2508.03474) documents $40M+ extracted via this method.

    \b
    Examples:
      python -m src analyze cross-platform
      python -m src analyze cross-platform --min-spread 0.03 --visualize
    """
    print_header("CROSS-PLATFORM ANALYSIS")
    click.echo(f"\n  Min Spread: {format_price(min_spread)}")
    click.echo(f"  Similarity Threshold: {similarity}")

    creds = Credentials.from_env()

    # Fetch from both platforms
    poly_markets = []
    kalshi_markets = []

    click.echo("\n  Fetching Polymarket data...")
    try:
        adapter = PolymarketAdapter(credentials=creds)
        await adapter.connect()
        poly_markets = await adapter.get_markets(active_only=True, limit=limit)
        await adapter.disconnect()
        click.echo(f"    Found {len(poly_markets)} markets")
    except Exception as e:
        click.echo(f"    Error: {e}", err=True)

    click.echo("  Fetching Kalshi data...")
    try:
        adapter = KalshiAdapter(credentials=creds)
        await adapter.connect()
        kalshi_markets = await adapter.get_markets(active_only=True, limit=limit)
        await adapter.disconnect()
        click.echo(f"    Found {len(kalshi_markets)} markets")
    except Exception as e:
        click.echo(f"    Error: {e}", err=True)

    if not poly_markets or not kalshi_markets:
        click.echo("\nNeed data from both platforms for cross-platform analysis.")
        return

    # Find matching markets using string similarity
    matches = []
    for poly in poly_markets:
        poly_q = (poly.question or "").lower()

        for kalshi in kalshi_markets:
            kalshi_q = (kalshi.question or "").lower()

            # Calculate similarity
            sim = SequenceMatcher(None, poly_q, kalshi_q).ratio()

            if sim >= similarity:
                spread = abs(poly.yes_price - kalshi.yes_price)

                if spread >= min_spread:
                    matches.append({
                        "polymarket": {
                            "id": poly.id,
                            "question": poly.question,
                            "yes_price": poly.yes_price,
                            "no_price": poly.no_price,
                            "volume": poly.volume,
                        },
                        "kalshi": {
                            "id": kalshi.id,
                            "question": kalshi.question,
                            "yes_price": kalshi.yes_price,
                            "no_price": kalshi.no_price,
                            "volume": kalshi.volume,
                        },
                        "similarity": sim,
                        "spread": spread,
                        "opportunity": _calculate_arb_opportunity(poly, kalshi),
                    })

    # Sort by spread
    matches.sort(key=lambda x: x["spread"], reverse=True)

    results = {
        "timestamp": datetime.now().isoformat(),
        "polymarket_count": len(poly_markets),
        "kalshi_count": len(kalshi_markets),
        "matches_found": len(matches),
        "min_spread": min_spread,
        "similarity_threshold": similarity,
        "matches": matches[:20],  # Top 20
    }

    if output == "json":
        click.echo(json.dumps(results, indent=2))
    else:
        _print_cross_platform_results(results)

    # Save results
    output_dir = Path("results/opportunities")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"cross_platform_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    click.echo(f"\n  Results saved to: {output_file}")

    if visualize and matches:
        _generate_cross_platform_visualization(results, output_dir, timestamp)


def _calculate_arb_opportunity(poly, kalshi) -> dict:
    """Calculate arbitrage opportunity between two markets."""
    # If Polymarket YES is cheaper, buy there; if Kalshi YES is cheaper, buy there
    poly_yes = poly.yes_price
    kalshi_yes = kalshi.yes_price

    if poly_yes < kalshi_yes:
        # Buy YES on Polymarket, sell YES on Kalshi (or buy NO)
        action = "Buy YES on Polymarket, Buy NO on Kalshi"
        cost = poly_yes + kalshi.no_price
        profit = 1.0 - cost if cost < 1.0 else 0
    else:
        # Buy YES on Kalshi, sell YES on Polymarket (or buy NO)
        action = "Buy YES on Kalshi, Buy NO on Polymarket"
        cost = kalshi_yes + poly.no_price
        profit = 1.0 - cost if cost < 1.0 else 0

    return {
        "action": action,
        "total_cost": cost,
        "guaranteed_profit": max(0, profit),
        "profit_pct": max(0, profit) / cost if cost > 0 else 0,
        "is_arbitrage": profit > 0,
    }


def _print_cross_platform_results(results: dict) -> None:
    """Print cross-platform analysis results."""
    matches = results["matches"]

    print_subheader("CROSS-PLATFORM MATCHES")
    click.echo(f"\n  Polymarket Markets: {results['polymarket_count']}")
    click.echo(f"  Kalshi Markets: {results['kalshi_count']}")
    click.echo(f"  Matches Found: {results['matches_found']}")

    if not matches:
        click.echo("\n  No matching markets found with sufficient spread.")
        return

    print_subheader("TOP OPPORTUNITIES")
    for i, match in enumerate(matches[:10], 1):
        poly = match["polymarket"]
        kalshi = match["kalshi"]
        opp = match["opportunity"]

        click.echo(f"\n  {i}. Similarity: {match['similarity']:.0%}")
        click.echo(f"     Polymarket: {poly['question'][:50]}")
        click.echo(f"       YES: {format_price(poly['yes_price'])} | NO: {format_price(poly['no_price'])}")
        click.echo(f"     Kalshi: {kalshi['question'][:50]}")
        click.echo(f"       YES: {format_price(kalshi['yes_price'])} | NO: {format_price(kalshi['no_price'])}")
        click.echo(f"     Spread: {format_price(match['spread'])}")

        if opp["is_arbitrage"]:
            click.echo(f"     ** ARBITRAGE: {format_price(opp['profit_pct'])} profit **")


def _generate_cross_platform_visualization(results: dict, output_dir: Path, timestamp: str) -> None:
    """Generate cross-platform visualization."""
    try:
        import matplotlib.pyplot as plt

        matches = results["matches"]
        if not matches:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Scatter: Polymarket vs Kalshi prices
        ax1 = axes[0]
        poly_prices = [m["polymarket"]["yes_price"] for m in matches]
        kalshi_prices = [m["kalshi"]["yes_price"] for m in matches]

        ax1.scatter(poly_prices, kalshi_prices, alpha=0.6)
        ax1.plot([0, 1], [0, 1], "r--", label="Equal prices")
        ax1.set_xlabel("Polymarket YES Price")
        ax1.set_ylabel("Kalshi YES Price")
        ax1.set_title("Cross-Platform Price Comparison")
        ax1.legend()

        # Bar: Top spreads
        ax2 = axes[1]
        top_matches = matches[:10]
        questions = [m["polymarket"]["question"][:20] for m in top_matches]
        spreads = [m["spread"] for m in top_matches]

        ax2.barh(questions, spreads, color="steelblue")
        ax2.set_xlabel("Price Spread")
        ax2.set_title("Top 10 Cross-Platform Spreads")

        plt.tight_layout()
        viz_file = output_dir / f"cross_platform_{timestamp}.png"
        plt.savefig(viz_file, dpi=150)
        plt.close()
        click.echo(f"  Visualization saved to: {viz_file}")

    except ImportError:
        click.echo("  (matplotlib not available for visualization)")
