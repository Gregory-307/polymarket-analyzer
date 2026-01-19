"""Data management CLI commands.

Subcommands:
- collect: Start data collection
- status: Show collection statistics
- export: Export data to CSV
- query: Run raw SQL queries
"""

import click

from .collect import collect
from .status import status
from .export import export
from .query import query


@click.group()
def data() -> None:
    """Data collection and management.

    Commands for collecting, storing, and querying market data.

    \b
    Examples:
      python -m src data collect --interval 5m
      python -m src data status
      python -m src data export --table snapshots --format csv
      python -m src data query "SELECT COUNT(*) FROM markets"
    """
    pass


data.add_command(collect)
data.add_command(status)
data.add_command(export)
data.add_command(query)


__all__ = ["data"]
