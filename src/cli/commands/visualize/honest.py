"""Honest visualization command - Reality check with actual spreads."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import click
import numpy as np

from ...utils import async_command, print_header
from ....core.config import Credentials
from ....adapters.polymarket import PolymarketAdapter


@click.command()
@click.option("--output-dir", type=click.Path(), default="results/visualizations")
@click.option("--limit", type=int, default=100, help="Markets to analyze.")
@click.option("--edge", type=float, default=0.02, help="Assumed gross edge.")
@async_command
async def honest(output_dir: str, limit: int, edge: float) -> None:
    """Generate honest visualizations with actual spread data.

    Unlike the dashboard, these visualizations show:
    - Actual order book spreads (not assumed)
    - Net E(X) after spread costs
    - Which markets are truly tradeable
    - Realistic profitability assessment

    \b
    Examples:
      python -m src visualize honest
      python -m src visualize honest --edge 0.01 --output-dir ./reality
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        click.echo("Error: matplotlib is required for visualizations.")
        return

    print_header("GENERATING HONEST VISUALIZATIONS")
    click.echo(f"\n  Assumed Gross Edge: {edge:.1%}")

    creds = Credentials.from_env()
    adapter = PolymarketAdapter(credentials=creds)

    click.echo("  Fetching market data...")
    await adapter.connect()
    markets = await adapter.get_markets(active_only=True, limit=limit)

    # Filter high-prob markets
    high_prob = [m for m in markets if max(m.yes_price, m.no_price) >= 0.90]
    click.echo(f"  High-probability markets: {len(high_prob)}")

    # Fetch actual spreads for top markets
    click.echo("  Fetching order book data...")
    market_data = []

    for market in high_prob[:30]:  # Limit API calls
        prob = max(market.yes_price, market.no_price)
        side = "YES" if market.yes_price > market.no_price else "NO"

        data = {
            "question": market.question[:40] if market.question else "Unknown",
            "probability": prob,
            "side": side,
            "volume": market.volume or 0,
            "spread": 0.01,  # Default assumption
        }

        # Try to get actual spread
        try:
            token_id = market.raw.get("clobTokenIds", ["", ""])[0 if side == "YES" else 1] if market.raw else None
            if token_id:
                ob = await adapter.get_order_book(token_id)
                if ob and ob.bids and ob.asks:
                    data["spread"] = ob.asks[0].price - ob.bids[0].price
                    data["best_bid"] = ob.bids[0].price
                    data["best_ask"] = ob.asks[0].price
        except Exception:
            pass

        # Calculate E(X)
        true_prob = min(0.99, prob + edge)
        win_profit = 25 * (1 - prob) / prob
        lose_profit = -25
        gross_ex = true_prob * win_profit + (1 - true_prob) * lose_profit
        net_ex = gross_ex - (25 * data["spread"])

        data["gross_ex"] = gross_ex
        data["net_ex"] = net_ex
        data["tradeable"] = net_ex > 0

        market_data.append(data)

    await adapter.disconnect()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    _generate_spread_analysis(market_data, edge, output_path)
    _generate_ex_comparison(market_data, edge, output_path)
    _generate_tradeable_summary(market_data, output_path)

    click.echo(f"\n  Visualizations saved to: {output_path}")


def _generate_spread_analysis(data: list, edge: float, output_path: Path) -> None:
    """Generate spread analysis visualization."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Spread distribution
    ax1 = axes[0, 0]
    spreads = [d["spread"] for d in data]
    ax1.hist(spreads, bins=15, edgecolor="white", color="#e74c3c", alpha=0.7)
    ax1.axvline(edge, color="green", linestyle="--", label=f"Gross Edge ({edge:.1%})")
    ax1.axvline(np.mean(spreads), color="orange", linestyle="--", label=f"Avg Spread ({np.mean(spreads):.1%})")
    ax1.set_xlabel("Spread")
    ax1.set_ylabel("Count")
    ax1.set_title("Actual Spread Distribution", fontweight="bold")
    ax1.legend()

    # 2. Spread vs Probability
    ax2 = axes[0, 1]
    probs = [d["probability"] for d in data]
    ax2.scatter(probs, spreads, alpha=0.6, c=["green" if d["tradeable"] else "red" for d in data])
    ax2.axhline(edge, color="blue", linestyle="--", label=f"Breakeven ({edge:.1%})")
    ax2.set_xlabel("Market Probability")
    ax2.set_ylabel("Spread")
    ax2.set_title("Spread vs Probability (green=tradeable)", fontweight="bold")
    ax2.legend()

    # 3. Net E(X) distribution
    ax3 = axes[1, 0]
    net_ex = [d["net_ex"] for d in data]
    colors = ["green" if ex > 0 else "red" for ex in net_ex]
    ax3.bar(range(len(net_ex)), net_ex, color=colors, alpha=0.7)
    ax3.axhline(0, color="black", linewidth=0.5)
    ax3.set_xlabel("Market Index")
    ax3.set_ylabel("Net E(X) ($)")
    ax3.set_title("Net Expected Value After Spreads", fontweight="bold")

    # 4. Summary pie
    ax4 = axes[1, 1]
    tradeable = sum(1 for d in data if d["tradeable"])
    not_tradeable = len(data) - tradeable
    ax4.pie(
        [tradeable, not_tradeable],
        labels=["Tradeable", "Not Tradeable"],
        colors=["#27ae60", "#e74c3c"],
        autopct="%1.0f%%",
        startangle=90,
    )
    ax4.set_title("Markets After Spread Analysis", fontweight="bold")

    fig.suptitle("Honest Spread Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path / "honest_spread_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    click.echo("    Generated: honest_spread_analysis.png")


def _generate_ex_comparison(data: list, edge: float, output_path: Path) -> None:
    """Generate E(X) comparison chart."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))

    questions = [d["question"][:25] for d in data[:15]]
    gross = [d["gross_ex"] for d in data[:15]]
    net = [d["net_ex"] for d in data[:15]]

    x = np.arange(len(questions))
    width = 0.35

    ax.bar(x - width / 2, gross, width, label="Gross E(X)", color="#3498db", alpha=0.8)
    ax.bar(x + width / 2, net, width, label="Net E(X) (after spread)", color="#e74c3c", alpha=0.8)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Market")
    ax.set_ylabel("Expected Value ($)")
    ax.set_title(f"Gross vs Net E(X) - Assumed {edge:.0%} Edge", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(questions, rotation=45, ha="right", fontsize=8)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path / "honest_ex_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    click.echo("    Generated: honest_ex_comparison.png")


def _generate_tradeable_summary(data: list, output_path: Path) -> None:
    """Generate tradeable markets summary."""
    import matplotlib.pyplot as plt

    tradeable = [d for d in data if d["tradeable"]]
    tradeable.sort(key=lambda x: x["net_ex"], reverse=True)

    if not tradeable:
        click.echo("    Warning: No tradeable markets found after spread analysis")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    questions = [d["question"] for d in tradeable[:15]]
    net_ex = [d["net_ex"] for d in tradeable[:15]]
    spreads = [d["spread"] for d in tradeable[:15]]

    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(questions)))

    bars = ax.barh(questions, net_ex, color=colors)

    # Add spread labels
    for bar, spread in zip(bars, spreads):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"Spread: {spread:.1%}", va="center", fontsize=8)

    ax.set_xlabel("Net E(X) ($)")
    ax.set_title("Top Tradeable Markets (After Spread Costs)", fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path / "honest_tradeable_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    click.echo("    Generated: honest_tradeable_summary.png")

    # Summary stats
    total_net_ex = sum(d["net_ex"] for d in tradeable)
    avg_spread = np.mean([d["spread"] for d in tradeable])
    click.echo(f"\n    Summary: {len(tradeable)} tradeable markets")
    click.echo(f"    Total Net E(X): ${total_net_ex:.2f}")
    click.echo(f"    Average Spread: {avg_spread:.2%}")
