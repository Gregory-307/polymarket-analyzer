"""CLI utility functions.

Provides:
- Async execution helpers for Click commands
- Output formatting utilities
- Common CLI patterns
"""

from __future__ import annotations

import asyncio
import sys
from functools import wraps
from typing import Any, Callable, TypeVar

import click

from ..core.utils import setup_logging, get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def async_command(f: F) -> F:
    """Decorator to run async functions in Click commands.

    Usage:
        @cli.command()
        @async_command
        async def my_command():
            await some_async_operation()
    """
    @wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return asyncio.run(f(*args, **kwargs))
    return wrapper  # type: ignore


def format_price(price: float) -> str:
    """Format price as percentage."""
    return f"{price:.2%}"


def format_usd(amount: float) -> str:
    """Format amount as USD."""
    return f"${amount:,.2f}"


def format_edge(edge: float) -> str:
    """Format edge with color hint."""
    if edge >= 0.03:
        return f"{edge:.2%} (HIGH)"
    elif edge >= 0.02:
        return f"{edge:.2%} (MEDIUM)"
    else:
        return f"{edge:.2%} (LOW)"


def print_header(title: str, width: int = 70) -> None:
    """Print a formatted header."""
    click.echo("=" * width)
    click.echo(f"  {title}")
    click.echo("=" * width)


def print_subheader(title: str, width: int = 70) -> None:
    """Print a formatted subheader."""
    click.echo()
    click.echo("-" * width)
    click.echo(f"  {title}")
    click.echo("-" * width)


def print_table_row(label: str, value: str, width: int = 30) -> None:
    """Print a formatted table row."""
    click.echo(f"  {label:<{width}} {value}")


def print_opportunity(opp: dict, index: int | None = None) -> None:
    """Print a formatted opportunity."""
    prefix = f"{index}. " if index is not None else ""
    question = opp.get("question", "Unknown")[:60]

    click.echo(f"\n{prefix}{question}")

    if "side" in opp:
        click.echo(f"    Side: {opp['side']} @ {format_price(opp.get('price', 0))}")
    if "edge" in opp:
        click.echo(f"    Edge: {format_edge(opp['edge'])}")
    if "profit_pct" in opp:
        click.echo(f"    Profit: {format_price(opp['profit_pct'])}")
    if "volume" in opp:
        click.echo(f"    Volume: {format_usd(opp.get('volume', 0))}")
    if "liquidity" in opp:
        click.echo(f"    Liquidity: {format_usd(opp.get('liquidity', 0))}")


def handle_error(error: Exception, verbose: bool = False) -> None:
    """Handle and display errors consistently."""
    if verbose:
        logger.exception("Command failed", error=str(error))
    click.echo(f"\nError: {error}", err=True)
    sys.exit(1)


class OutputFormat:
    """Output format options."""
    JSON = "json"
    TABLE = "table"
    MARKDOWN = "markdown"


def output_json(data: Any) -> None:
    """Output data as JSON."""
    import json
    click.echo(json.dumps(data, indent=2, default=str))


def output_table(headers: list[str], rows: list[list[Any]]) -> None:
    """Output data as formatted table."""
    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    # Print header
    header_line = " | ".join(f"{h:<{widths[i]}}" for i, h in enumerate(headers))
    click.echo(header_line)
    click.echo("-" * len(header_line))

    # Print rows
    for row in rows:
        row_line = " | ".join(f"{str(cell):<{widths[i]}}" for i, cell in enumerate(row))
        click.echo(row_line)
