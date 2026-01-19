"""Execute command - live arbitrage execution.

Scans for arbitrage opportunities and executes them atomically.

IMPORTANT: This is LIVE TRADING. Real money at risk.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

import click

from ...adapters.polymarket import PolymarketAdapter
from ...core.config import Credentials
from ...core.utils import get_logger
from ...execution.arb_executor import ArbExecutor
from ...strategies.single_arb import SingleConditionArbitrage, SingleArbOpportunity

logger = get_logger(__name__)


def format_opportunity(opp: SingleArbOpportunity, idx: int) -> str:
    """Format opportunity for display."""
    lines = [
        f"\n[{idx}] {opp.market.question[:70]}...",
        f"    Action: {opp.action.upper()}",
        f"    YES: ${opp.yes_price:.3f}  NO: ${opp.no_price:.3f}  Sum: ${opp.sum_prices:.3f}",
        f"    Gross Edge: {opp.profit_pct:.2%} (${opp.profit_usd:.2f} per $100)",
    ]

    # Add net profit estimate
    net = opp.estimate_net_profit(100.0)
    lines.append(f"    Net Profit (after costs): {net['net_profit_pct']:.2%} (${net['net_profit']:.2f} per $100)")
    lines.append(f"    Profitable after costs: {'YES' if net['profitable'] else 'NO'}")

    return "\n".join(lines)


async def scan_and_execute(
    min_profit: float,
    size: float,
    max_slippage: float,
    dry_run: bool,
    auto_execute: bool,
) -> None:
    """Scan for opportunities and optionally execute.

    Args:
        min_profit: Minimum profit percentage to consider.
        size: Position size in USD.
        max_slippage: Maximum acceptable slippage.
        dry_run: If True, don't execute trades.
        auto_execute: If True, execute without confirmation.
    """
    # Load credentials
    credentials = Credentials.from_env()

    if not credentials.has_polymarket and not dry_run:
        click.echo("ERROR: No Polymarket credentials found.")
        click.echo("Set POLYMARKET_PRIVATE_KEY environment variable.")
        click.echo("Use --dry-run to scan without trading.")
        return

    # Initialize adapter
    adapter = PolymarketAdapter(credentials=credentials if not dry_run else None)
    connected = await adapter.connect()

    if not connected:
        click.echo("ERROR: Failed to connect to Polymarket.")
        return

    click.echo("=" * 70)
    click.echo("  POLYMARKET ARBITRAGE SCANNER")
    click.echo("=" * 70)

    if dry_run:
        click.echo("  Mode: DRY RUN (no trades will be executed)")
    else:
        click.echo("  Mode: LIVE TRADING")
        click.echo("  WARNING: Real money at risk!")

        # Show wallet info
        if adapter.wallet_address:
            click.echo(f"  Wallet: {adapter.wallet_address[:10]}...{adapter.wallet_address[-6:]}")
            try:
                balance = await adapter.get_balance()
                click.echo(f"  USDC Balance: ${balance:.2f}")
                if balance < size:
                    click.echo(f"  WARNING: Balance (${balance:.2f}) < position size (${size:.2f})")
            except Exception as e:
                click.echo(f"  Balance check failed: {e}")

    click.echo(f"  Min Profit: {min_profit:.2%}")
    click.echo(f"  Position Size: ${size:.2f}")
    click.echo(f"  Max Slippage: {max_slippage:.2%}")
    click.echo("-" * 70)

    # Fetch markets
    click.echo("\nFetching markets...")
    markets = await adapter.get_markets(active_only=True, limit=500)
    click.echo(f"Found {len(markets)} active markets")

    # Scan for arbitrage
    click.echo("\nScanning for single-condition arbitrage...")
    detector = SingleConditionArbitrage(min_profit_pct=min_profit)
    opportunities = detector.scan(markets)

    if not opportunities:
        click.echo("\nNo arbitrage opportunities found above threshold.")
        await adapter.disconnect()
        return

    # Filter to only net-profitable opportunities
    profitable_opps = []
    for opp in opportunities:
        net = opp.estimate_net_profit(size)
        if net["profitable"]:
            profitable_opps.append(opp)

    click.echo(f"\nFound {len(opportunities)} gross opportunities")
    click.echo(f"  {len(profitable_opps)} profitable after estimated costs")

    if not profitable_opps:
        click.echo("\nNo opportunities profitable after transaction costs.")
        await adapter.disconnect()
        return

    # Display opportunities
    click.echo("\n" + "=" * 70)
    click.echo("  PROFITABLE OPPORTUNITIES")
    click.echo("=" * 70)

    for idx, opp in enumerate(profitable_opps[:10]):  # Show top 10
        click.echo(format_opportunity(opp, idx))

    if len(profitable_opps) > 10:
        click.echo(f"\n... and {len(profitable_opps) - 10} more")

    if dry_run:
        click.echo("\n[DRY RUN] No trades executed.")
        await adapter.disconnect()
        return

    # Execution mode
    if not auto_execute:
        click.echo("\n" + "-" * 70)
        choice = click.prompt(
            "Enter opportunity number to execute (or 'q' to quit)",
            default="q"
        )

        if choice.lower() == "q":
            click.echo("Aborted.")
            await adapter.disconnect()
            return

        try:
            idx = int(choice)
            if idx < 0 or idx >= len(profitable_opps):
                click.echo(f"Invalid selection: {idx}")
                await adapter.disconnect()
                return
        except ValueError:
            click.echo(f"Invalid input: {choice}")
            await adapter.disconnect()
            return

        selected = profitable_opps[idx]
    else:
        # Auto-execute: pick the best opportunity
        selected = profitable_opps[0]
        click.echo(f"\n[AUTO] Selected best opportunity: {selected.market.question[:50]}...")

    # Confirm execution
    click.echo("\n" + "=" * 70)
    click.echo("  EXECUTING TRADE")
    click.echo("=" * 70)
    click.echo(format_opportunity(selected, 0))
    click.echo(f"\n  Execution Size: ${size:.2f}")

    if not auto_execute:
        confirm = click.confirm("Execute this trade?", default=False)
        if not confirm:
            click.echo("Aborted.")
            await adapter.disconnect()
            return

    # Execute via ArbExecutor
    executor = ArbExecutor(
        adapter=adapter,
        max_slippage=max_slippage,
    )

    # Pre-execution check
    click.echo("\nChecking executability...")
    can_execute, reason = await executor.check_executable(selected, size)

    if not can_execute:
        click.echo(f"CANNOT EXECUTE: {reason}")
        await adapter.disconnect()
        return

    click.echo(f"Pre-check passed: {reason}")
    click.echo("\nSubmitting orders...")

    result = await executor.execute_arb(selected, size)

    click.echo("\n" + "-" * 70)
    if result.success:
        click.echo("  EXECUTION SUCCESSFUL")
        click.echo("-" * 70)
        click.echo(f"  YES Leg: {result.yes_leg.status} @ ${result.yes_leg.avg_fill_price:.4f}")
        click.echo(f"  NO Leg:  {result.no_leg.status} @ ${result.no_leg.avg_fill_price:.4f}")
        click.echo(f"  Gross Profit: ${result.gross_profit:.2f}")
        click.echo(f"  Fees Paid: ${result.fees_paid:.2f}")
        click.echo(f"  Slippage Cost: ${result.slippage_cost:.2f}")
        click.echo(f"  Net Profit: ${result.net_profit:.2f}")
    else:
        click.echo("  EXECUTION FAILED")
        click.echo("-" * 70)
        click.echo(f"  Error: {result.error}")
        click.echo(f"  YES Leg: {result.yes_leg.status}")
        click.echo(f"  NO Leg: {result.no_leg.status}")

    await adapter.disconnect()


@click.command()
@click.option(
    "--min-profit",
    type=float,
    default=0.005,
    help="Minimum gross profit percentage (default: 0.5%).",
)
@click.option(
    "--size",
    type=float,
    default=100.0,
    help="Position size in USD (default: $100).",
)
@click.option(
    "--max-slippage",
    type=float,
    default=0.01,
    help="Maximum acceptable slippage (default: 1%).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Scan only, don't execute trades.",
)
@click.option(
    "--auto",
    "auto_execute",
    is_flag=True,
    help="Auto-execute best opportunity without confirmation.",
)
def execute(
    min_profit: float,
    size: float,
    max_slippage: float,
    dry_run: bool,
    auto_execute: bool,
) -> None:
    """Scan and execute arbitrage opportunities.

    LIVE TRADING - Real money at risk!

    This command scans for single-condition arbitrage (YES + NO != $1.00)
    and can execute trades atomically.

    \b
    Examples:
      python -m src execute --dry-run              # Scan only
      python -m src execute --size 50              # Trade $50 positions
      python -m src execute --auto --size 25       # Auto-execute best opp

    \b
    Requirements:
      - POLYMARKET_PRIVATE_KEY environment variable
      - py_clob_client package installed
      - USDC balance on Polygon network
    """
    if auto_execute and not dry_run:
        click.echo("WARNING: Auto-execute mode enabled!")
        click.echo("Trades will execute without confirmation.")
        if not click.confirm("Continue?", default=False):
            return

    asyncio.run(
        scan_and_execute(
            min_profit=min_profit,
            size=size,
            max_slippage=max_slippage,
            dry_run=dry_run,
            auto_execute=auto_execute,
        )
    )
