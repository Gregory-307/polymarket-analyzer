"""Polymarket Analyzer - Unified CLI.

Production-quality prediction market analysis toolkit.

Usage:
    python -m src --help
    python -m src connect
    python -m src scan --strategy favorite_longshot
    python -m src report generate --strategy favorite_longshot
"""

from __future__ import annotations

import click

from ..core.utils import setup_logging


@click.group()
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default="INFO",
    help="Set logging level.",
)
@click.option(
    "--log-format",
    type=click.Choice(["console", "json"], case_sensitive=False),
    default="console",
    help="Set logging format.",
)
@click.pass_context
def cli(ctx: click.Context, log_level: str, log_format: str) -> None:
    """Polymarket Analyzer - Prediction Market Analysis Toolkit.

    A comprehensive system for detecting arbitrage opportunities,
    exploiting behavioral biases, and analyzing market microstructure
    in prediction markets.

    \b
    Strategies:
      - favorite_longshot: Buy underpriced high-probability outcomes
      - single_arb: YES + NO arbitrage when sum != $1.00
      - multi_arb: Bundle arbitrage on multi-outcome markets

    \b
    Examples:
      python -m src connect
      python -m src scan --strategy favorite_longshot
      python -m src signals --bankroll 1000
      python -m src report generate --strategy favorite_longshot
    """
    ctx.ensure_object(dict)
    ctx.obj["log_level"] = log_level
    ctx.obj["log_format"] = log_format
    setup_logging(level=log_level, log_format=log_format)


# Import and register command groups
from .commands import connect, markets, scan, signals, backtest
from .commands.analyze import analyze
from .commands.report import report
from .commands.visualize import visualize

cli.add_command(connect.connect)
cli.add_command(markets.markets)
cli.add_command(scan.scan)
cli.add_command(signals.signals)
cli.add_command(backtest.backtest)
cli.add_command(analyze)
cli.add_command(report)
cli.add_command(visualize)


if __name__ == "__main__":
    cli()
