"""Run command - starts live monitoring with scheduler.

Starts a live monitoring session that:
- Collects market data at configurable intervals
- Scans for opportunities
- Sends alerts via configured channels
"""

from __future__ import annotations

import asyncio
import signal
import sys

import click

from ...adapters.polymarket import PolymarketAdapter
from ...alerts.base import AlertManager, ConsoleHandler, AlertLevel
from ...scheduler.scheduler import Scheduler
from ...storage.database import Database
from ...collectors.market_collector import MarketCollector
from ...collectors.base import CollectorConfig
from ...core.utils import get_logger

logger = get_logger(__name__)


async def run_live(
    scan_interval: int,
    collect_interval: int,
    strategy: str,
    min_edge: float,
    alert_discord: str | None,
    alert_telegram_token: str | None,
    alert_telegram_chat: str | None,
    db_path: str,
) -> None:
    """Run live monitoring loop.

    Args:
        scan_interval: Seconds between opportunity scans.
        collect_interval: Seconds between market data collection.
        strategy: Strategy to scan with.
        min_edge: Minimum edge to alert on.
        alert_discord: Discord webhook URL.
        alert_telegram_token: Telegram bot token.
        alert_telegram_chat: Telegram chat ID.
        db_path: Database path.
    """
    # Setup components
    adapter = PolymarketAdapter()
    db = Database(db_path)
    await db.connect()

    # Setup alert manager
    alert_manager = AlertManager()
    alert_manager.add_handler(ConsoleHandler(min_level=AlertLevel.INFO, rate_limit_seconds=0))

    if alert_discord:
        from ...alerts.discord import DiscordHandler
        alert_manager.add_handler(
            DiscordHandler(webhook_url=alert_discord, rate_limit_seconds=60)
        )

    if alert_telegram_token and alert_telegram_chat:
        from ...alerts.telegram import TelegramHandler
        alert_manager.add_handler(
            TelegramHandler(
                bot_token=alert_telegram_token,
                chat_id=alert_telegram_chat,
                rate_limit_seconds=30,
            )
        )

    # Setup market collector
    collector = MarketCollector(
        database=db,
        adapter=adapter,
        min_liquidity=1000,
    )

    # Import strategy scanner
    if strategy == "favorite_longshot":
        from ...strategies.favorite_longshot import FavoriteLongshotStrategy
        strategy_obj = FavoriteLongshotStrategy(
            min_probability=0.85,
        )
    elif strategy == "single_arb":
        from ...strategies.single_arb import SingleConditionArbitrage
        strategy_obj = SingleConditionArbitrage(min_profit_threshold=0.005)
    elif strategy == "multi_arb":
        from ...strategies.multi_arb import MultiOutcomeArbitrage
        strategy_obj = MultiOutcomeArbitrage(min_profit_threshold=0.02)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Define jobs
    async def collect_markets():
        """Collect market data."""
        count = await collector.collect()
        logger.info(f"Collected {count} market snapshots")

    async def scan_opportunities():
        """Scan for opportunities and alert."""
        try:
            markets = await adapter.get_markets()

            opportunities = []
            for market in markets:
                opps = await strategy_obj.analyze(market)
                opportunities.extend(opps)

            if opportunities:
                logger.info(f"Found {len(opportunities)} opportunities")

                for opp in opportunities:
                    edge = getattr(opp, 'edge', getattr(opp, 'expected_profit', 0))
                    if edge >= min_edge:
                        await alert_manager.send_opportunity(
                            market_name=opp.market_name,
                            strategy=strategy,
                            side=getattr(opp, 'side', 'N/A'),
                            price=getattr(opp, 'price', 0),
                            profit=edge,
                            volume=getattr(opp, 'volume', None),
                        )

        except Exception as e:
            logger.error(f"Scan error: {e}")
            await alert_manager.send_error(
                title="Scan Failed",
                error=str(e),
                source=strategy,
            )

    # Setup scheduler
    async def on_job_error(job_name: str, error: Exception):
        await alert_manager.send_error(
            title=f"Job Failed: {job_name}",
            error=str(error),
            source="scheduler",
        )

    scheduler = Scheduler(error_callback=on_job_error)

    scheduler.add_job(
        name="collect_markets",
        func=collect_markets,
        interval_seconds=collect_interval,
        run_immediately=True,
    )

    scheduler.add_job(
        name="scan_opportunities",
        func=scan_opportunities,
        interval_seconds=scan_interval,
        run_immediately=True,
    )

    # Handle shutdown
    shutdown_event = asyncio.Event()

    def handle_shutdown(sig, frame):
        logger.info(f"Received signal {sig}, shutting down...")
        shutdown_event.set()

    # Register signal handlers (Unix only)
    if sys.platform != "win32":
        signal.signal(signal.SIGINT, handle_shutdown)
        signal.signal(signal.SIGTERM, handle_shutdown)

    # Start scheduler
    click.echo(f"Starting live monitoring with {strategy} strategy...")
    click.echo(f"  Scan interval: {scan_interval}s")
    click.echo(f"  Collect interval: {collect_interval}s")
    click.echo(f"  Min edge for alerts: {min_edge:.2%}")
    click.echo("Press Ctrl+C to stop.")
    click.echo("-" * 50)

    await scheduler.start()

    try:
        if sys.platform == "win32":
            # Windows: use simple wait loop
            while not shutdown_event.is_set():
                await asyncio.sleep(1)
        else:
            await shutdown_event.wait()
    except KeyboardInterrupt:
        pass
    finally:
        await scheduler.stop()
        await db.close()
        click.echo("Shutdown complete.")


@click.command()
@click.option(
    "--strategy",
    type=click.Choice(["favorite_longshot", "single_arb", "multi_arb"]),
    default="favorite_longshot",
    help="Strategy to scan with.",
)
@click.option(
    "--scan-interval",
    type=int,
    default=60,
    help="Seconds between opportunity scans (default: 60).",
)
@click.option(
    "--collect-interval",
    type=int,
    default=300,
    help="Seconds between market data collection (default: 300).",
)
@click.option(
    "--min-edge",
    type=float,
    default=0.01,
    help="Minimum edge to alert on (default: 0.01 = 1%).",
)
@click.option(
    "--discord-webhook",
    envvar="DISCORD_WEBHOOK_URL",
    help="Discord webhook URL for alerts.",
)
@click.option(
    "--telegram-token",
    envvar="TELEGRAM_BOT_TOKEN",
    help="Telegram bot token for alerts.",
)
@click.option(
    "--telegram-chat",
    envvar="TELEGRAM_CHAT_ID",
    help="Telegram chat ID for alerts.",
)
@click.option(
    "--db-path",
    default="data/polymarket.db",
    help="Database path (default: data/polymarket.db).",
)
def run(
    strategy: str,
    scan_interval: int,
    collect_interval: int,
    min_edge: float,
    discord_webhook: str | None,
    telegram_token: str | None,
    telegram_chat: str | None,
    db_path: str,
) -> None:
    """Start live monitoring with scheduled tasks.

    Runs continuous monitoring that:
    - Collects market data at regular intervals
    - Scans for trading opportunities
    - Sends alerts via configured channels (Discord, Telegram)

    \b
    Examples:
      python -m src run
      python -m src run --strategy single_arb --scan-interval 30
      python -m src run --discord-webhook https://discord.com/api/webhooks/...
    """
    asyncio.run(
        run_live(
            scan_interval=scan_interval,
            collect_interval=collect_interval,
            strategy=strategy,
            min_edge=min_edge,
            alert_discord=discord_webhook,
            alert_telegram_token=telegram_token,
            alert_telegram_chat=telegram_chat,
            db_path=db_path,
        )
    )
