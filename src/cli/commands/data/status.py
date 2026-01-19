"""Data status command."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import click

from ...utils import async_command, print_header, print_subheader, format_usd
from ....storage.database import Database


@click.command()
@click.option(
    "--db-path",
    type=click.Path(),
    default="data/polymarket.db",
    help="Database file path.",
)
@async_command
async def status(db_path: str) -> None:
    """Show data collection statistics.

    Displays counts, date ranges, and health metrics for collected data.

    \b
    Examples:
      python -m src data status
      python -m src data status --db-path data/custom.db
    """
    print_header("DATA STATUS")

    db_file = Path(db_path)
    click.echo(f"\n  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    click.echo(f"  Database: {db_path}")

    if not db_file.exists():
        click.echo("\n  ERROR: Database file not found")
        click.echo("  Run 'python -m src data collect --once' to create it")
        return

    # File size
    size_mb = db_file.stat().st_size / (1024 * 1024)
    click.echo(f"  Size: {size_mb:.2f} MB")

    db = Database(db_file)
    await db.connect()

    try:
        stats = await db.get_stats()

        print_subheader("TABLE COUNTS")
        click.echo(f"\n  Markets:           {stats['markets']:>8,}")
        click.echo(f"  Market Snapshots:  {stats['market_snapshots']:>8,}")
        click.echo(f"  Orderbook Snaps:   {stats['orderbook_snapshots']:>8,}")
        click.echo(f"  Trades:            {stats['trades']:>8,}")
        click.echo(f"  Opportunities:     {stats['opportunities']:>8,}")
        click.echo(f"  Resolutions:       {stats['resolutions']:>8,}")

        # Snapshot date range
        if stats['snapshot_range']['start'] and stats['snapshot_range']['end']:
            print_subheader("SNAPSHOT RANGE")
            click.echo(f"\n  From: {stats['snapshot_range']['start']}")
            click.echo(f"  To:   {stats['snapshot_range']['end']}")

        # Additional metrics
        print_subheader("DETAILED METRICS")

        # Active markets
        active_markets = await db.get_active_markets()
        click.echo(f"\n  Active Markets: {len(active_markets)}")

        # Markets by platform
        poly_markets = await db.get_active_markets(platform="polymarket")
        kalshi_markets = await db.get_active_markets(platform="kalshi")
        click.echo(f"    Polymarket: {len(poly_markets)}")
        click.echo(f"    Kalshi:     {len(kalshi_markets)}")

        # Recent opportunities
        recent_opps = await db.get_opportunities(limit=10)
        if recent_opps:
            print_subheader("RECENT OPPORTUNITIES")
            for opp in recent_opps[:5]:
                status_icon = "[x]" if opp["executed"] else "[ ]"
                click.echo(
                    f"\n  {status_icon} [{opp['strategy']}] {opp['action'] or 'N/A'}"
                )
                if opp['edge']:
                    click.echo(f"    Edge: {opp['edge']*100:.1f}%")
                click.echo(f"    Detected: {opp['detected_at']}")

        # Resolutions
        resolutions = await db.get_resolutions(limit=5)
        if resolutions:
            print_subheader("RECENT RESOLUTIONS")
            for res in resolutions[:5]:
                click.echo(f"\n  {res['question'][:50]}...")
                click.echo(f"    Outcome: {res['outcome']}")
                if res['final_price']:
                    click.echo(f"    Final Price: {res['final_price']:.2%}")

    finally:
        await db.close()

    print_subheader("DATA QUALITY")
    click.echo("\n  Check: Database accessible [OK]")
    if stats['market_snapshots'] > 0:
        click.echo("  Check: Snapshots present [OK]")
    else:
        click.echo("  Check: Snapshots present [MISSING] (run 'data collect --once')")
