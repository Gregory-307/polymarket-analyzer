"""Dashboard visualization command."""

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
@click.option("--limit", type=int, default=200, help="Markets to analyze.")
@click.option("--style", type=click.Choice(["dark", "light"]), default="dark")
@async_command
async def dashboard(output_dir: str, limit: int, style: str) -> None:
    """Generate comprehensive dashboard visualizations.

    Creates publication-quality charts for portfolio/CV:
    - Market probability distribution
    - Volume by category
    - Edge opportunity scatter
    - Strategy comparison infographic
    - Executive dashboard with KPIs

    \b
    Examples:
      python -m src visualize dashboard
      python -m src visualize dashboard --style light --output-dir ./viz
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        click.echo("Error: matplotlib is required for visualizations.")
        click.echo("Install with: pip install matplotlib")
        return

    print_header("GENERATING DASHBOARD")

    # Fetch data
    creds = Credentials.from_env()
    adapter = PolymarketAdapter(credentials=creds)

    click.echo("\n  Fetching market data...")
    await adapter.connect()
    markets = await adapter.get_markets(active_only=True, limit=limit)
    await adapter.disconnect()

    click.echo(f"  Analyzing {len(markets)} markets...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set style
    if style == "dark":
        plt.style.use("dark_background")
        bg_color = "#1a1a2e"
        text_color = "#eee"
        accent_color = "#00d4ff"
    else:
        plt.style.use("default")
        bg_color = "#ffffff"
        text_color = "#333"
        accent_color = "#3498db"

    # Generate visualizations
    _generate_probability_distribution(markets, output_path, style)
    _generate_volume_by_category(markets, output_path, style)
    _generate_edge_opportunities(markets, output_path, style)
    _generate_executive_dashboard(markets, output_path, style)

    click.echo(f"\n  All visualizations saved to: {output_path}")


def _generate_probability_distribution(markets: list, output_path: Path, style: str) -> None:
    """Generate probability distribution chart."""
    import matplotlib.pyplot as plt

    probs = [max(m.yes_price, m.no_price) for m in markets]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Histogram with colored zones
    n, bins, patches = ax.hist(probs, bins=20, edgecolor="white", alpha=0.7)

    # Color by opportunity zone
    for patch, left_edge in zip(patches, bins[:-1]):
        if left_edge >= 0.95:
            patch.set_facecolor("#27ae60")  # Green - high opportunity
        elif left_edge >= 0.90:
            patch.set_facecolor("#f39c12")  # Orange - moderate
        else:
            patch.set_facecolor("#3498db")  # Blue - normal

    ax.axvline(0.95, color="red", linestyle="--", label="High-Opportunity Zone (95%+)")
    ax.axvline(0.90, color="orange", linestyle="--", label="Moderate Zone (90-95%)")

    ax.set_xlabel("Market Probability", fontsize=12)
    ax.set_ylabel("Number of Markets", fontsize=12)
    ax.set_title("Market Probability Distribution", fontsize=14, fontweight="bold")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path / "probability_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    click.echo("    Generated: probability_distribution.png")


def _generate_volume_by_category(markets: list, output_path: Path, style: str) -> None:
    """Generate volume by category chart."""
    import matplotlib.pyplot as plt

    categories = {
        "Extreme Favorites (95%+)": [],
        "Favorites (90-95%)": [],
        "Moderate (70-90%)": [],
        "Uncertain (30-70%)": [],
        "Longshots (<30%)": [],
    }

    for m in markets:
        prob = max(m.yes_price, m.no_price)
        vol = m.volume or 0

        if prob >= 0.95:
            categories["Extreme Favorites (95%+)"].append(vol)
        elif prob >= 0.90:
            categories["Favorites (90-95%)"].append(vol)
        elif prob >= 0.70:
            categories["Moderate (70-90%)"].append(vol)
        elif prob >= 0.30:
            categories["Uncertain (30-70%)"].append(vol)
        else:
            categories["Longshots (<30%)"].append(vol)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Volume chart
    ax1 = axes[0]
    cat_names = list(categories.keys())
    cat_volumes = [sum(v) for v in categories.values()]

    colors = ["#27ae60", "#2ecc71", "#3498db", "#9b59b6", "#e74c3c"]
    ax1.bar(range(len(cat_names)), cat_volumes, color=colors)
    ax1.set_xticks(range(len(cat_names)))
    ax1.set_xticklabels([c.split("(")[0].strip() for c in cat_names], rotation=45, ha="right")
    ax1.set_ylabel("Total Volume ($)")
    ax1.set_title("Volume by Probability Category", fontweight="bold")

    # Format y-axis as millions
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x/1e6:.1f}M"))

    # Count chart
    ax2 = axes[1]
    cat_counts = [len(v) for v in categories.values()]
    ax2.bar(range(len(cat_names)), cat_counts, color=colors)
    ax2.set_xticks(range(len(cat_names)))
    ax2.set_xticklabels([c.split("(")[0].strip() for c in cat_names], rotation=45, ha="right")
    ax2.set_ylabel("Number of Markets")
    ax2.set_title("Market Count by Category", fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path / "volume_by_category.png", dpi=150, bbox_inches="tight")
    plt.close()
    click.echo("    Generated: volume_by_category.png")


def _generate_edge_opportunities(markets: list, output_path: Path, style: str) -> None:
    """Generate edge opportunities scatter plot."""
    import matplotlib.pyplot as plt

    # Calculate edges for high-prob markets
    data = []
    for m in markets:
        prob = max(m.yes_price, m.no_price)
        if prob >= 0.90:
            # Estimate edge based on probability
            if prob >= 0.95:
                edge = 0.03
            else:
                edge = 0.02
            data.append({
                "prob": prob,
                "edge": edge,
                "volume": m.volume or 0,
                "liquidity": m.liquidity or 0,
            })

    if not data:
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    probs = [d["prob"] for d in data]
    edges = [d["edge"] for d in data]
    sizes = [max(20, min(500, d["volume"] / 10000)) for d in data]

    scatter = ax.scatter(probs, edges, s=sizes, alpha=0.6, c=probs, cmap="RdYlGn")

    ax.set_xlabel("Market Probability", fontsize=12)
    ax.set_ylabel("Estimated Edge", fontsize=12)
    ax.set_title("Edge Opportunities (bubble size = volume)", fontsize=14, fontweight="bold")

    ax.axhline(0.02, color="orange", linestyle="--", alpha=0.7, label="Target Edge (2%)")
    ax.legend()

    plt.colorbar(scatter, label="Probability")
    plt.tight_layout()
    plt.savefig(output_path / "edge_opportunities.png", dpi=150, bbox_inches="tight")
    plt.close()
    click.echo("    Generated: edge_opportunities.png")


def _generate_executive_dashboard(markets: list, output_path: Path, style: str) -> None:
    """Generate executive summary dashboard."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 10))

    # Calculate KPIs
    total_volume = sum(m.volume or 0 for m in markets)
    total_liquidity = sum(m.liquidity or 0 for m in markets)
    high_prob_count = sum(1 for m in markets if max(m.yes_price, m.no_price) >= 0.95)
    opportunity_count = sum(1 for m in markets if max(m.yes_price, m.no_price) >= 0.90)

    # KPI boxes
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.text(0.5, 0.6, f"{len(markets)}", fontsize=48, ha="center", va="center", fontweight="bold")
    ax1.text(0.5, 0.2, "Markets Analyzed", fontsize=14, ha="center", va="center")
    ax1.axis("off")

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.text(0.5, 0.6, f"${total_volume/1e6:.1f}M", fontsize=42, ha="center", va="center", fontweight="bold", color="#27ae60")
    ax2.text(0.5, 0.2, "Total Volume", fontsize=14, ha="center", va="center")
    ax2.axis("off")

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.text(0.5, 0.6, f"{high_prob_count}", fontsize=48, ha="center", va="center", fontweight="bold", color="#e74c3c")
    ax3.text(0.5, 0.2, "Extreme Favorites (95%+)", fontsize=14, ha="center", va="center")
    ax3.axis("off")

    # Probability distribution (bottom left)
    ax4 = fig.add_subplot(2, 3, 4)
    probs = [max(m.yes_price, m.no_price) for m in markets]
    ax4.hist(probs, bins=15, edgecolor="white", alpha=0.7, color="#3498db")
    ax4.set_title("Probability Distribution", fontweight="bold")
    ax4.set_xlabel("Probability")

    # Volume pie (bottom center)
    ax5 = fig.add_subplot(2, 3, 5)
    labels = ["95%+", "90-95%", "70-90%", "<70%"]
    sizes = [
        sum(m.volume or 0 for m in markets if max(m.yes_price, m.no_price) >= 0.95),
        sum(m.volume or 0 for m in markets if 0.90 <= max(m.yes_price, m.no_price) < 0.95),
        sum(m.volume or 0 for m in markets if 0.70 <= max(m.yes_price, m.no_price) < 0.90),
        sum(m.volume or 0 for m in markets if max(m.yes_price, m.no_price) < 0.70),
    ]
    colors = ["#27ae60", "#f39c12", "#3498db", "#95a5a6"]
    ax5.pie(sizes, labels=labels, colors=colors, autopct="%1.0f%%", startangle=90)
    ax5.set_title("Volume by Category", fontweight="bold")

    # Opportunity summary (bottom right)
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.text(0.5, 0.8, "Opportunity Summary", fontsize=16, ha="center", va="center", fontweight="bold")
    ax6.text(0.5, 0.55, f"Tradeable Markets: {opportunity_count}", fontsize=12, ha="center", va="center")
    ax6.text(0.5, 0.4, f"Total Liquidity: ${total_liquidity/1e6:.1f}M", fontsize=12, ha="center", va="center")
    ax6.text(0.5, 0.25, f"Avg Edge (est.): 2-3%", fontsize=12, ha="center", va="center")
    ax6.axis("off")

    fig.suptitle("Polymarket Analysis Dashboard", fontsize=20, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path / "dashboard_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    click.echo("    Generated: dashboard_summary.png")
