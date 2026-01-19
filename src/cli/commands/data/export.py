"""Data export command."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import click

from ...utils import async_command, print_header, print_subheader
from ....storage.database import Database


EXPORT_TABLES = [
    "markets",
    "snapshots",
    "orderbooks",
    "trades",
    "opportunities",
    "resolutions",
]


@click.command()
@click.option(
    "--table",
    type=click.Choice(EXPORT_TABLES),
    default="snapshots",
    help="Table to export.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["csv", "json"]),
    default="csv",
    help="Output format.",
)
@click.option(
    "--db-path",
    type=click.Path(),
    default="data/polymarket.db",
    help="Database file path.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output file path (default: exports/<table>_<timestamp>.<format>)",
)
@click.option(
    "--limit",
    type=int,
    default=10000,
    help="Maximum rows to export.",
)
@async_command
async def export(
    table: str,
    output_format: str,
    db_path: str,
    output: str | None,
    limit: int,
) -> None:
    """Export data to CSV or JSON.

    Exports collected data from the database to a file for
    analysis in other tools.

    \b
    Examples:
      python -m src data export
      python -m src data export --table markets --format json
      python -m src data export --table snapshots -o my_data.csv
      python -m src data export --table trades --limit 50000
    """
    print_header("DATA EXPORT")

    click.echo(f"\n  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    click.echo(f"  Table: {table}")
    click.echo(f"  Format: {output_format}")

    db_file = Path(db_path)
    if not db_file.exists():
        click.echo("\n  ERROR: Database file not found")
        return

    db = Database(db_file)
    await db.connect()

    try:
        # Determine output path
        if output:
            output_path = Path(output)
        else:
            output_dir = Path("exports")
            output_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"{table}_{timestamp}.{output_format}"

        # Fetch data based on table
        data = await _fetch_table_data(db, table, limit)

        if not data:
            click.echo(f"\n  No data found in {table}")
            return

        click.echo(f"  Rows: {len(data)}")

        # Export
        print_subheader("EXPORTING")

        if output_format == "csv":
            _export_csv(data, output_path)
        else:
            _export_json(data, output_path)

        file_size = output_path.stat().st_size / 1024  # KB
        click.echo(f"\n  Exported to: {output_path}")
        click.echo(f"  File size: {file_size:.1f} KB")

    finally:
        await db.close()


async def _fetch_table_data(db: Database, table: str, limit: int) -> list[dict]:
    """Fetch data from the specified table.

    Args:
        db: Database instance.
        table: Table name (friendly name).
        limit: Maximum rows.

    Returns:
        List of row dictionaries.
    """
    # Map friendly names to actual queries
    if table == "markets":
        query = f"SELECT * FROM markets ORDER BY updated_at DESC LIMIT {limit}"
    elif table == "snapshots":
        query = f"SELECT * FROM market_snapshots ORDER BY timestamp DESC LIMIT {limit}"
    elif table == "orderbooks":
        query = f"SELECT * FROM orderbook_snapshots ORDER BY timestamp DESC LIMIT {limit}"
    elif table == "trades":
        query = f"SELECT * FROM trades ORDER BY timestamp DESC LIMIT {limit}"
    elif table == "opportunities":
        query = f"SELECT * FROM opportunities ORDER BY detected_at DESC LIMIT {limit}"
    elif table == "resolutions":
        query = f"SELECT * FROM resolutions ORDER BY resolved_at DESC LIMIT {limit}"
    else:
        return []

    async with db.connection.execute(query) as cursor:
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


def _export_csv(data: list[dict], output_path: Path) -> None:
    """Export data to CSV file.

    Args:
        data: List of row dictionaries.
        output_path: Output file path.
    """
    if not data:
        return

    fieldnames = list(data[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def _export_json(data: list[dict], output_path: Path) -> None:
    """Export data to JSON file.

    Args:
        data: List of row dictionaries.
        output_path: Output file path.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "exported_at": datetime.now().isoformat(),
                "count": len(data),
                "data": data,
            },
            f,
            indent=2,
            default=str,
        )
