"""Signals command - Find high-probability markets."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import click

from ..utils import async_command, print_header, print_subheader, format_price, format_usd
from ...core.config import Credentials
from ...adapters.polymarket import PolymarketAdapter
from ...strategies.favorite_longshot import FavoriteLongshotStrategy


@click.command()
@click.option("--min-probability", type=float, default=0.90, help="Minimum probability (default: 90%).")
@click.option("--min-liquidity", type=float, default=1000, help="Minimum liquidity ($).")
@click.option("--limit", type=int, default=200, help="Markets to analyze.")
@click.option("--output", type=click.Choice(["table", "json"]), default="table")
@async_command
async def signals(
    min_probability: float,
    min_liquidity: float,
    limit: int,
    output: str,
) -> None:
    """Find high-probability markets.

    Scans for markets where one side has probability >= threshold.
    Research suggests these may be underpriced (favorite-longshot bias).

    \b
    Examples:
      python -m src signals
      python -m src signals --min-probability 0.95
      python -m src signals --output json
    """
    print_header("HIGH-PROBABILITY MARKET SCANNER")
    click.echo(f"\n  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    click.echo(f"  Min Probability: {format_price(min_probability)}")
    click.echo(f"  Min Liquidity: {format_usd(min_liquidity)}")

    creds = Credentials.from_env()
    adapter = PolymarketAdapter(credentials=creds)

    click.echo("\n  Fetching market data...")
    await adapter.connect()
    markets = await adapter.get_markets(active_only=True, limit=limit)
    click.echo(f"  Fetched: {len(markets)} markets")

    strategy = FavoriteLongshotStrategy(min_probability=min_probability)

    results = []
    for market in markets:
        opp = strategy.check_market(market)
        if opp and (market.liquidity or 0) >= min_liquidity:
            results.append({
                "market_id": market.id,
                "question": market.question,
                "side": opp.side,
                "price": opp.price,
                "volume": market.volume or 0,
                "liquidity": market.liquidity or 0,
            })

    results.sort(key=lambda x: x["price"], reverse=True)
    click.echo(f"  Found: {len(results)} high-probability markets")

    await adapter.disconnect()

    if output == "json":
        data = {
            "generated_at": datetime.now().isoformat(),
            "min_probability": min_probability,
            "total_found": len(results),
            "markets": results,
        }
        click.echo(json.dumps(data, indent=2))
    else:
        _print_results(results)

    # Save results
    output_dir = Path("results/signals")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"signals_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(
            {
                "generated_at": datetime.now().isoformat(),
                "min_probability": min_probability,
                "total_found": len(results),
                "markets": results,
            },
            f,
            indent=2,
        )
    click.echo(f"\n  Saved to: {output_file}")


def _print_results(results: list) -> None:
    """Print results in table format."""
    print_subheader("HIGH-PROBABILITY MARKETS")

    for r in results[:10]:
        click.echo(f"\n  {r['question'][:60]}")
        click.echo(f"    {r['side']} @ {format_price(r['price'])} | "
                  f"Vol: {format_usd(r['volume'])} | "
                  f"Liq: {format_usd(r['liquidity'])}")

    if results:
        print_subheader("SUMMARY")
        click.echo(f"\n  Total Found: {len(results)}")
        avg_price = sum(r["price"] for r in results) / len(results)
        click.echo(f"  Average Probability: {format_price(avg_price)}")
