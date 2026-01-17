"""Markets command - List active markets."""

from __future__ import annotations

import json
from pathlib import Path

import click

from ..utils import async_command, print_header, format_price, format_usd, output_json
from ...core.config import Credentials
from ...adapters.polymarket import PolymarketAdapter
from ...adapters.kalshi import KalshiAdapter


@click.command()
@click.option(
    "--platform",
    type=click.Choice(["all", "polymarket", "kalshi"], case_sensitive=False),
    default="polymarket",
    help="Platform to fetch markets from.",
)
@click.option(
    "--limit",
    type=int,
    default=20,
    help="Maximum number of markets to display.",
)
@click.option(
    "--output",
    type=click.Choice(["table", "json"], case_sensitive=False),
    default="table",
    help="Output format.",
)
@click.option(
    "--save",
    type=click.Path(),
    default=None,
    help="Save results to JSON file.",
)
@async_command
async def markets(platform: str, limit: int, output: str, save: str | None) -> None:
    """List active prediction markets.

    Fetches and displays active markets from Polymarket and/or Kalshi,
    sorted by volume.

    \b
    Examples:
      python -m src markets
      python -m src markets --platform kalshi --limit 50
      python -m src markets --output json --save markets.json
    """
    print_header("ACTIVE MARKETS")

    creds = Credentials.from_env()
    all_markets = []

    if platform in ("all", "polymarket"):
        click.echo("\nFetching Polymarket markets...")
        adapter = PolymarketAdapter(credentials=creds)
        try:
            await adapter.connect()
            poly_markets = await adapter.get_markets(active_only=True, limit=limit)
            all_markets.extend(poly_markets)
            click.echo(f"  Found {len(poly_markets)} markets")
            await adapter.disconnect()
        except Exception as e:
            click.echo(f"  Error: {e}", err=True)

    if platform in ("all", "kalshi"):
        click.echo("Fetching Kalshi markets...")
        adapter = KalshiAdapter(credentials=creds)
        try:
            await adapter.connect()
            kalshi_markets = await adapter.get_markets(active_only=True, limit=limit)
            all_markets.extend(kalshi_markets)
            click.echo(f"  Found {len(kalshi_markets)} markets")
            await adapter.disconnect()
        except Exception as e:
            click.echo(f"  Error: {e}", err=True)

    if not all_markets:
        click.echo("\nNo markets found.")
        return

    # Sort by volume
    all_markets.sort(key=lambda m: m.volume or 0, reverse=True)
    display_markets = all_markets[:limit]

    if output == "json":
        data = [
            {
                "id": m.id,
                "platform": m.platform,
                "question": m.question,
                "yes_price": m.yes_price,
                "no_price": m.no_price,
                "volume": m.volume,
                "liquidity": m.liquidity,
            }
            for m in display_markets
        ]
        output_json(data)
    else:
        # Table output
        click.echo(f"\n{'#':<3} {'Question':<50} {'YES':>8} {'NO':>8} {'Volume':>12}")
        click.echo("-" * 85)

        for i, market in enumerate(display_markets, 1):
            question = (market.question or "Unknown")[:48]
            yes = format_price(market.yes_price)
            no = format_price(market.no_price)
            vol = format_usd(market.volume or 0)

            click.echo(f"{i:<3} {question:<50} {yes:>8} {no:>8} {vol:>12}")

    # Summary
    total_volume = sum(m.volume or 0 for m in all_markets)
    total_liquidity = sum(m.liquidity or 0 for m in all_markets)

    click.echo("\n" + "-" * 50)
    click.echo(f"  Total Markets: {len(all_markets)}")
    click.echo(f"  Total Volume: {format_usd(total_volume)}")
    click.echo(f"  Total Liquidity: {format_usd(total_liquidity)}")

    # Save if requested
    if save:
        save_path = Path(save)
        data = [
            {
                "id": m.id,
                "platform": m.platform,
                "question": m.question,
                "yes_price": m.yes_price,
                "no_price": m.no_price,
                "volume": m.volume,
                "liquidity": m.liquidity,
            }
            for m in all_markets
        ]
        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)
        click.echo(f"\n  Saved to: {save_path}")
