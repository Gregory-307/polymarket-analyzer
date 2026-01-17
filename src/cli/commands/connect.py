"""Connect command - Test platform connections."""

from __future__ import annotations

import click

from ..utils import async_command, print_header, print_table_row
from ...core.config import Credentials
from ...adapters.polymarket import PolymarketAdapter
from ...adapters.kalshi import KalshiAdapter


@click.command()
@click.option(
    "--platform",
    type=click.Choice(["all", "polymarket", "kalshi"], case_sensitive=False),
    default="all",
    help="Platform to test.",
)
@async_command
async def connect(platform: str) -> None:
    """Test connections to prediction market platforms.

    Verifies API connectivity and authentication status for
    Polymarket and/or Kalshi.

    \b
    Examples:
      python -m src connect
      python -m src connect --platform polymarket
    """
    print_header("CONNECTION TEST")

    creds = Credentials.from_env()

    results = {}

    if platform in ("all", "polymarket"):
        click.echo("\nTesting Polymarket...")
        adapter = PolymarketAdapter(credentials=creds)
        try:
            connected = await adapter.connect()
            results["polymarket"] = {
                "connected": connected,
                "authenticated": adapter.is_authenticated,
            }
            await adapter.disconnect()
        except Exception as e:
            results["polymarket"] = {"connected": False, "error": str(e)}

    if platform in ("all", "kalshi"):
        click.echo("Testing Kalshi...")
        adapter = KalshiAdapter(credentials=creds)
        try:
            connected = await adapter.connect()
            results["kalshi"] = {
                "connected": connected,
                "authenticated": adapter.is_authenticated,
            }
            await adapter.disconnect()
        except Exception as e:
            results["kalshi"] = {"connected": False, "error": str(e)}

    # Display results
    click.echo("\n" + "-" * 50)
    click.echo("  RESULTS")
    click.echo("-" * 50)

    for name, result in results.items():
        status = "OK" if result.get("connected") else "FAILED"
        auth = "Yes" if result.get("authenticated") else "No"

        click.echo(f"\n  {name.upper()}")
        print_table_row("Connection", status)
        print_table_row("Authenticated", auth)
        if "error" in result:
            print_table_row("Error", result["error"][:50])

    # Summary
    all_connected = all(r.get("connected", False) for r in results.values())
    click.echo("\n" + "=" * 50)
    if all_connected:
        click.echo("  All platforms connected successfully!")
    else:
        click.echo("  Some connections failed. Check credentials.")
    click.echo("=" * 50)
