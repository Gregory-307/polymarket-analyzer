"""Expected value analysis command."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import click
import numpy as np

from ...utils import async_command, print_header, print_subheader, format_price, format_usd
from ....core.config import Credentials
from ....adapters.polymarket import PolymarketAdapter


@click.command()
@click.option("--bet-size", type=float, default=25, help="Bet size per position ($).")
@click.option("--limit", type=int, default=50, help="Markets to analyze.")
@click.option("--output", type=click.Choice(["table", "json"]), default="table")
@click.option("--visualize/--no-visualize", default=False, help="Generate visualization.")
@async_command
async def ev(bet_size: float, limit: int, output: str, visualize: bool) -> None:
    """Calculate expected value across edge scenarios.

    Analyzes E(X) for each trade under different edge assumptions:
    - 0% edge (efficient market hypothesis)
    - 1% edge (conservative)
    - 2% edge (documented bias)
    - 3% edge (optimistic)
    - 4% edge (aggressive)

    \b
    Examples:
      python -m src analyze ev
      python -m src analyze ev --bet-size 50 --visualize
    """
    print_header("EXPECTED VALUE ANALYSIS")
    click.echo(f"\n  Bet Size: {format_usd(bet_size)}")

    creds = Credentials.from_env()
    adapter = PolymarketAdapter(credentials=creds)

    click.echo("  Fetching market data...")
    await adapter.connect()
    markets = await adapter.get_markets(active_only=True, limit=limit)
    await adapter.disconnect()

    # Filter for high-probability markets
    eligible = [m for m in markets if max(m.yes_price, m.no_price) >= 0.90][:10]

    if not eligible:
        click.echo("\nNo eligible markets found.")
        return

    edge_scenarios = [0.00, 0.01, 0.02, 0.03, 0.04]
    results = {
        "timestamp": datetime.now().isoformat(),
        "bet_size": bet_size,
        "trades": [],
        "scenarios": {},
    }

    # Calculate E(X) for each trade under each scenario
    for market in eligible:
        prob = max(market.yes_price, market.no_price)
        trade_results = {
            "market_id": market.id,
            "question": market.question[:50] if market.question else "Unknown",
            "probability": prob,
            "scenarios": {},
        }

        for edge in edge_scenarios:
            true_prob = min(0.99, prob + edge)
            win_profit = bet_size * (1 - prob) / prob
            lose_profit = -bet_size
            ex = true_prob * win_profit + (1 - true_prob) * lose_profit

            # Breakeven probability
            if win_profit > 0:
                breakeven_prob = bet_size / (win_profit + bet_size)
            else:
                breakeven_prob = 1.0

            trade_results["scenarios"][f"edge_{int(edge*100)}pct"] = {
                "edge": edge,
                "true_probability": true_prob,
                "expected_value": ex,
                "breakeven_probability": breakeven_prob,
                "profitable": ex > 0,
            }

        results["trades"].append(trade_results)

    # Calculate portfolio totals per scenario
    for edge in edge_scenarios:
        key = f"edge_{int(edge*100)}pct"
        total_ex = sum(
            t["scenarios"][key]["expected_value"]
            for t in results["trades"]
        )
        profitable_trades = sum(
            1 for t in results["trades"]
            if t["scenarios"][key]["profitable"]
        )

        results["scenarios"][key] = {
            "edge": edge,
            "total_expected_value": total_ex,
            "profitable_trades": profitable_trades,
            "total_trades": len(results["trades"]),
            "portfolio_roi": total_ex / (bet_size * len(results["trades"])),
        }

    if output == "json":
        click.echo(json.dumps(results, indent=2))
    else:
        _print_ev_results(results, edge_scenarios)

    # Save results
    output_dir = Path("results/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"ev_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    click.echo(f"\n  Results saved to: {output_file}")

    if visualize:
        _generate_ev_visualization(results, edge_scenarios, output_dir, timestamp)


def _print_ev_results(results: dict, edge_scenarios: list[float]) -> None:
    """Print E(X) results."""
    print_subheader("E(X) BY EDGE SCENARIO")

    # Header
    header = f"{'Trade':<30}"
    for edge in edge_scenarios:
        header += f" {int(edge*100)}% Edge".rjust(12)
    click.echo(header)
    click.echo("-" * (30 + 12 * len(edge_scenarios)))

    # Per-trade results
    for trade in results["trades"]:
        row = f"{trade['question'][:28]:<30}"
        for edge in edge_scenarios:
            key = f"edge_{int(edge*100)}pct"
            ex = trade["scenarios"][key]["expected_value"]
            row += f" {format_usd(ex)}".rjust(12)
        click.echo(row)

    # Portfolio totals
    click.echo("-" * (30 + 12 * len(edge_scenarios)))
    row = f"{'PORTFOLIO TOTAL':<30}"
    for edge in edge_scenarios:
        key = f"edge_{int(edge*100)}pct"
        total = results["scenarios"][key]["total_expected_value"]
        row += f" {format_usd(total)}".rjust(12)
    click.echo(row)

    print_subheader("PORTFOLIO ROI BY SCENARIO")
    for edge in edge_scenarios:
        key = f"edge_{int(edge*100)}pct"
        scenario = results["scenarios"][key]
        status = "PROFITABLE" if scenario["total_expected_value"] > 0 else "LOSING"
        click.echo(f"  {int(edge*100)}% Edge: {format_price(scenario['portfolio_roi'])} ROI ({status})")


def _generate_ev_visualization(
    results: dict,
    edge_scenarios: list[float],
    output_dir: Path,
    timestamp: str,
) -> None:
    """Generate E(X) visualization."""
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        edges = [int(e * 100) for e in edge_scenarios]
        totals = [
            results["scenarios"][f"edge_{e}pct"]["total_expected_value"]
            for e in edges
        ]

        colors = ["red" if t < 0 else "green" for t in totals]
        ax.bar(edges, totals, color=colors)
        ax.axhline(0, color="black", linestyle="-", linewidth=0.5)

        ax.set_xlabel("Edge Assumption (%)")
        ax.set_ylabel("Portfolio Expected Value ($)")
        ax.set_title("E(X) by Edge Scenario")

        for i, (e, t) in enumerate(zip(edges, totals)):
            ax.text(e, t + (1 if t >= 0 else -3), f"${t:.2f}", ha="center")

        viz_file = output_dir / f"ev_{timestamp}.png"
        plt.savefig(viz_file, dpi=150, bbox_inches="tight")
        plt.close()
        click.echo(f"  Visualization saved to: {viz_file}")

    except ImportError:
        click.echo("  (matplotlib not available for visualization)")
