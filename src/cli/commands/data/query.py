"""Raw SQL query command."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import click

from ...utils import async_command, print_header, print_subheader
from ....storage.database import Database


@click.command()
@click.argument("sql", required=True)
@click.option(
    "--db-path",
    type=click.Path(),
    default="data/polymarket.db",
    help="Database file path.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json", "csv"]),
    default="table",
    help="Output format.",
)
@click.option(
    "--limit",
    type=int,
    default=100,
    help="Maximum rows to return (appended to query if not present).",
)
@async_command
async def query(
    sql: str,
    db_path: str,
    output_format: str,
    limit: int,
) -> None:
    """Run a raw SQL query.

    Execute arbitrary SQL against the database and display results.
    Only SELECT queries are allowed for safety.

    \b
    Examples:
      python -m src data query "SELECT COUNT(*) FROM markets"
      python -m src data query "SELECT * FROM market_snapshots ORDER BY timestamp DESC"
      python -m src data query "SELECT platform, COUNT(*) FROM markets GROUP BY platform" --format json
    """
    print_header("SQL QUERY")

    click.echo(f"\n  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Safety check - only allow SELECT queries
    sql_upper = sql.strip().upper()
    if not sql_upper.startswith("SELECT"):
        click.echo("\n  ERROR: Only SELECT queries are allowed")
        click.echo("  Use the database file directly for write operations")
        return

    db_file = Path(db_path)
    if not db_file.exists():
        click.echo(f"\n  ERROR: Database not found: {db_path}")
        return

    # Add LIMIT if not present
    if "LIMIT" not in sql_upper:
        sql = f"{sql.rstrip().rstrip(';')} LIMIT {limit}"

    click.echo(f"\n  Query: {sql[:80]}{'...' if len(sql) > 80 else ''}")

    db = Database(db_file)
    await db.connect()

    try:
        async with db.connection.execute(sql) as cursor:
            rows = await cursor.fetchall()
            data = [dict(row) for row in rows]

        click.echo(f"  Rows: {len(data)}")

        if not data:
            click.echo("\n  No results")
            return

        print_subheader("RESULTS")

        if output_format == "json":
            _print_json(data)
        elif output_format == "csv":
            _print_csv(data)
        else:
            _print_table(data)

    except Exception as e:
        click.echo(f"\n  ERROR: {e}")

    finally:
        await db.close()


def _print_table(data: list[dict]) -> None:
    """Print data as formatted table.

    Args:
        data: List of row dictionaries.
    """
    if not data:
        return

    # Get column widths
    columns = list(data[0].keys())
    widths = {}
    for col in columns:
        max_width = len(col)
        for row in data[:20]:  # Sample first 20 rows
            val = str(row.get(col, ""))[:40]  # Truncate long values
            max_width = max(max_width, len(val))
        widths[col] = min(max_width, 40)

    # Print header
    header = " | ".join(col.ljust(widths[col])[:widths[col]] for col in columns)
    click.echo(f"\n  {header}")
    click.echo(f"  {'-' * len(header)}")

    # Print rows (limit display)
    for row in data[:50]:
        values = []
        for col in columns:
            val = str(row.get(col, ""))[:widths[col]]
            values.append(val.ljust(widths[col]))
        click.echo(f"  {' | '.join(values)}")

    if len(data) > 50:
        click.echo(f"\n  ... ({len(data) - 50} more rows)")


def _print_json(data: list[dict]) -> None:
    """Print data as JSON.

    Args:
        data: List of row dictionaries.
    """
    click.echo()
    click.echo(json.dumps(data, indent=2, default=str))


def _print_csv(data: list[dict]) -> None:
    """Print data as CSV.

    Args:
        data: List of row dictionaries.
    """
    if not data:
        return

    columns = list(data[0].keys())

    # Header
    click.echo()
    click.echo(",".join(columns))

    # Rows
    for row in data:
        values = []
        for col in columns:
            val = str(row.get(col, ""))
            # Escape quotes and wrap if contains comma
            if "," in val or '"' in val:
                val = '"' + val.replace('"', '""') + '"'
            values.append(val)
        click.echo(",".join(values))
