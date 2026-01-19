"""Data collection command."""

from __future__ import annotations

import asyncio
import signal
import sys
from datetime import datetime
from pathlib import Path

import click

from ...utils import async_command, print_header, print_subheader
from ....core.config import Credentials
from ....adapters.polymarket import PolymarketAdapter
from ....storage.database import Database
from ....collectors.base import CollectorConfig
from ....collectors.market_collector import MarketCollector


INTERVAL_MAP = {
    "10s": 10,
    "30s": 30,
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
}


@click.command()
@click.option(
    "--interval",
    type=click.Choice(list(INTERVAL_MAP.keys())),
    default="5m",
    help="Collection interval.",
)
@click.option(
    "--db-path",
    type=click.Path(),
    default="data/polymarket.db",
    help="Database file path.",
)
@click.option(
    "--min-liquidity",
    type=float,
    default=100.0,
    help="Minimum liquidity to track ($).",
)
@click.option(
    "--limit",
    type=int,
    default=100,
    help="Maximum markets to collect per cycle.",
)
@click.option(
    "--once",
    is_flag=True,
    help="Run one collection cycle and exit.",
)
@async_command
async def collect(
    interval: str,
    db_path: str,
    min_liquidity: float,
    limit: int,
    once: bool,
) -> None:
    """Start data collection.

    Collects market price snapshots at the specified interval
    and stores them in the SQLite database.

    \b
    Examples:
      python -m src data collect
      python -m src data collect --interval 1m
      python -m src data collect --once
      python -m src data collect --min-liquidity 1000
    """
    print_header("DATA COLLECTOR")

    interval_seconds = INTERVAL_MAP[interval]
    click.echo(f"\n  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    click.echo(f"  Interval: {interval} ({interval_seconds}s)")
    click.echo(f"  Database: {db_path}")
    click.echo(f"  Min Liquidity: ${min_liquidity:,.0f}")

    # Setup
    db = Database(Path(db_path))
    await db.connect()

    creds = Credentials.from_env()
    adapter = PolymarketAdapter(credentials=creds)
    await adapter.connect()

    config = CollectorConfig(
        interval=interval_seconds,
        batch_size=limit,
    )
    collector = MarketCollector(
        db, adapter,
        config=config,
        min_liquidity=min_liquidity,
    )

    if once:
        # Single collection
        print_subheader("COLLECTING")
        items = await collector.collect_once()
        click.echo(f"\n  Collected: {items} snapshots")
    else:
        # Continuous collection
        print_subheader("CONTINUOUS COLLECTION")
        click.echo("\n  Press Ctrl+C to stop\n")

        # Handle graceful shutdown
        stop_event = asyncio.Event()

        def signal_handler(sig, frame):
            click.echo("\n  Stopping...")
            stop_event.set()

        signal.signal(signal.SIGINT, signal_handler)
        if sys.platform != "win32":
            signal.signal(signal.SIGTERM, signal_handler)

        # Start collector
        await collector.start()

        # Wait for stop signal
        while not stop_event.is_set():
            await asyncio.sleep(1)

            # Print periodic status
            stats = collector.get_stats()
            if stats["collections"] > 0 and stats["collections"] % 5 == 0:
                click.echo(
                    f"  [{datetime.now().strftime('%H:%M:%S')}] "
                    f"Collections: {stats['collections']} | "
                    f"Items: {stats['items_collected']} | "
                    f"Errors: {stats['errors']}"
                )

        await collector.stop()

    # Cleanup
    await adapter.disconnect()
    await db.close()

    # Final stats
    print_subheader("FINAL STATISTICS")
    stats = collector.get_stats()
    click.echo(f"\n  Total Collections: {stats['collections']}")
    click.echo(f"  Total Items: {stats['items_collected']}")
    click.echo(f"  Errors: {stats['errors']}")
    if stats['last_collection']:
        click.echo(f"  Last Collection: {stats['last_collection']}")
