"""Data investigation command."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import click

from ...utils import async_command, print_header, print_subheader, format_price, format_usd
from ....core.config import Credentials
from ....adapters.polymarket import PolymarketAdapter


@click.command()
@click.option("--min-prob", type=float, default=0.90, help="Minimum probability to investigate.")
@click.option("--max-prob", type=float, default=0.96, help="Maximum probability to investigate.")
@click.option("--limit", type=int, default=100, help="Markets to fetch.")
@click.option("--fetch-spreads/--no-fetch-spreads", default=True, help="Fetch actual order book spreads.")
@click.option("--edge", type=float, default=0.02, help="Assumed gross edge.")
@click.option("--output", type=click.Choice(["table", "json"]), default="table")
@async_command
async def investigate(
    min_prob: float,
    max_prob: float,
    limit: int,
    fetch_spreads: bool,
    edge: float,
    output: str,
) -> None:
    """Investigate real market data without assumptions.

    Fetches actual order book data to calculate real spreads and
    identify truly tradeable opportunities after spread costs.

    \b
    Examples:
      python -m src analyze investigate
      python -m src analyze investigate --min-prob 0.95 --fetch-spreads
    """
    print_header("DATA INVESTIGATION")
    click.echo(f"\n  Probability Range: {format_price(min_prob)} - {format_price(max_prob)}")
    click.echo(f"  Fetch Spreads: {'Yes' if fetch_spreads else 'No'}")
    click.echo(f"  Assumed Gross Edge: {format_price(edge)}")

    creds = Credentials.from_env()
    adapter = PolymarketAdapter(credentials=creds)

    click.echo("\n  Fetching market data...")
    await adapter.connect()
    markets = await adapter.get_markets(active_only=True, limit=limit)

    # Filter by probability range
    eligible = [
        m for m in markets
        if min_prob <= max(m.yes_price, m.no_price) <= max_prob
    ]

    click.echo(f"  Markets in range: {len(eligible)}")

    results = {
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "min_prob": min_prob,
            "max_prob": max_prob,
            "gross_edge": edge,
            "fetch_spreads": fetch_spreads,
        },
        "markets": [],
        "summary": {},
    }

    # Analyze each market
    tradeable_count = 0
    total_positive_ex = 0

    for market in eligible:
        prob = max(market.yes_price, market.no_price)
        side = "YES" if market.yes_price > market.no_price else "NO"

        analysis = {
            "market_id": market.id,
            "question": market.question[:60] if market.question else "Unknown",
            "platform": market.platform,
            "side": side,
            "probability": prob,
            "volume": market.volume,
            "liquidity": market.liquidity,
        }

        # Fetch actual spread from order book
        if fetch_spreads:
            try:
                # Get order book for the favorable token
                token_id = market.raw.get("clobTokenIds", ["", ""])[0 if side == "YES" else 1] if market.raw else None
                if token_id:
                    order_book = await adapter.get_order_book(token_id)
                    if order_book and order_book.bids and order_book.asks:
                        best_bid = order_book.bids[0].price
                        best_ask = order_book.asks[0].price
                        spread = best_ask - best_bid
                        analysis["spread"] = {
                            "best_bid": best_bid,
                            "best_ask": best_ask,
                            "spread": spread,
                            "spread_pct": spread / prob if prob > 0 else 0,
                        }
            except Exception:
                pass

        # Calculate E(X)
        spread_cost = analysis.get("spread", {}).get("spread", 0.01)  # Default 1% spread
        net_edge = edge - spread_cost

        true_prob = min(0.99, prob + edge)
        win_profit = 25 * (1 - prob) / prob  # $25 bet
        lose_profit = -25
        gross_ex = true_prob * win_profit + (1 - true_prob) * lose_profit
        net_ex = gross_ex - (25 * spread_cost)  # Spread cost

        analysis["expected_value"] = {
            "gross_edge": edge,
            "spread_cost": spread_cost,
            "net_edge": net_edge,
            "gross_ex": gross_ex,
            "net_ex": net_ex,
            "tradeable": net_ex > 0,
        }

        if net_ex > 0:
            tradeable_count += 1
            total_positive_ex += net_ex

        results["markets"].append(analysis)

    await adapter.disconnect()

    # Summary
    results["summary"] = {
        "total_markets": len(eligible),
        "tradeable_markets": tradeable_count,
        "pct_tradeable": tradeable_count / len(eligible) if eligible else 0,
        "total_positive_ex": total_positive_ex,
    }

    if output == "json":
        click.echo(json.dumps(results, indent=2))
    else:
        _print_investigation_results(results)

    # Save results
    output_dir = Path("results/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"investigate_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    click.echo(f"\n  Results saved to: {output_file}")


def _print_investigation_results(results: dict) -> None:
    """Print investigation results."""
    markets = results["markets"]
    summary = results["summary"]

    print_subheader("MARKET ANALYSIS")

    # Group by tradeable
    tradeable = [m for m in markets if m["expected_value"]["tradeable"]]
    not_tradeable = [m for m in markets if not m["expected_value"]["tradeable"]]

    if tradeable:
        click.echo("\n  TRADEABLE (positive E(X) after spreads):")
        for m in tradeable[:10]:
            ev = m["expected_value"]
            spread_info = m.get("spread", {})
            spread_str = f"{spread_info.get('spread', 'N/A'):.1%}" if spread_info else "N/A"

            click.echo(f"\n    {m['question']}")
            click.echo(f"      {m['side']} @ {format_price(m['probability'])} | "
                      f"Spread: {spread_str} | "
                      f"Net E(X): {format_usd(ev['net_ex'])}")

    if not_tradeable:
        click.echo("\n  NOT TRADEABLE (negative E(X) after spreads):")
        for m in not_tradeable[:5]:
            ev = m["expected_value"]
            click.echo(f"    {m['question'][:50]}")
            click.echo(f"      Spread eats edge: {format_price(ev['spread_cost'])} > {format_price(ev['gross_edge'])}")

    print_subheader("SUMMARY")
    click.echo(f"\n  Total Markets Analyzed: {summary['total_markets']}")
    click.echo(f"  Tradeable After Spreads: {summary['tradeable_markets']} ({format_price(summary['pct_tradeable'])})")
    click.echo(f"  Total Positive E(X): {format_usd(summary['total_positive_ex'])}")

    if summary["tradeable_markets"] == 0:
        click.echo("\n  WARNING: No markets are tradeable after accounting for spreads.")
        click.echo("  Consider waiting for tighter spreads or larger edge assumptions.")
