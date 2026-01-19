"""Generate report command."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import click

from ...utils import async_command, print_header, format_price, format_usd
from ....core.config import Credentials
from ....adapters.polymarket import PolymarketAdapter
from ....strategies.favorite_longshot import FavoriteLongshotStrategy
from ....strategies.single_arb import SingleConditionArbitrage


@click.command()
@click.option(
    "--strategy",
    type=click.Choice(["favorite_longshot", "single_arb", "multi_arb", "all"], case_sensitive=False),
    default="all",
    help="Strategy to report on.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "markdown"], case_sensitive=False),
    default="json",
    help="Output format.",
)
@click.option("--limit", type=int, default=200, help="Markets to analyze.")
@async_command
async def generate(strategy: str, output_format: str, limit: int) -> None:
    """Generate strategy-specific analysis report.

    Creates comprehensive reports including:
    - Current opportunities
    - Market analysis
    - Risk metrics
    - Position sizing recommendations

    \b
    Examples:
      python -m src report generate --strategy favorite_longshot
      python -m src report generate --strategy single_arb --format markdown
    """
    print_header(f"GENERATING REPORT: {strategy.upper()}")

    creds = Credentials.from_env()
    adapter = PolymarketAdapter(credentials=creds)

    click.echo("\n  Fetching market data...")
    await adapter.connect()
    markets = await adapter.get_markets(active_only=True, limit=limit)
    await adapter.disconnect()

    click.echo(f"  Analyzed {len(markets)} markets")

    report_data = {
        "title": f"{strategy.replace('_', ' ').title()} Strategy Report",
        "generated_at": datetime.now().isoformat(),
        "strategy": strategy,
        "total_markets": len(markets),
        "sections": {},
    }

    # Generate strategy-specific sections
    if strategy in ("all", "favorite_longshot"):
        report_data["sections"]["favorite_longshot"] = _analyze_favorite_longshot(markets)

    if strategy in ("all", "single_arb"):
        report_data["sections"]["single_arb"] = _analyze_single_arb(markets)

    if strategy in ("all", "multi_arb"):
        report_data["sections"]["multi_arb"] = {"note": "Multi-arb requires grouped market data (see Phase 3)"}

    # Output
    output_dir = Path("results/reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_format == "json":
        output_file = output_dir / f"report_{strategy}_{timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(report_data, f, indent=2, default=str)
        click.echo(f"\n  Report saved to: {output_file}")

    elif output_format == "markdown":
        md_content = _format_as_markdown(report_data)
        output_file = output_dir / f"report_{strategy}_{timestamp}.md"
        with open(output_file, "w") as f:
            f.write(md_content)
        click.echo(f"\n  Report saved to: {output_file}")
        click.echo("\n" + md_content)


def _analyze_favorite_longshot(markets: list) -> dict:
    """Analyze markets for favorite-longshot opportunities."""
    strategy = FavoriteLongshotStrategy(min_probability=0.90)

    opportunities = []
    for market in markets:
        opp = strategy.check_market(market)
        if opp:
            opportunities.append({
                "market_id": market.id,
                "question": market.question,
                "side": opp.side,
                "price": opp.price,
                "volume": market.volume,
                "liquidity": market.liquidity,
            })

    opportunities.sort(key=lambda x: x["price"], reverse=True)

    return {
        "description": "Find high-probability markets",
        "research_basis": "Prospect Theory (Kahneman & Tversky, 1979), Snowberg & Wolfers NBER Working Paper 15923",
        "total_opportunities": len(opportunities),
        "top_opportunities": opportunities[:10],
        "methodology": {
            "min_probability": 0.90,
        },
    }


def _analyze_single_arb(markets: list) -> dict:
    """Analyze markets for single-condition arbitrage."""
    strategy = SingleConditionArbitrage(min_profit_pct=0.003)

    opportunities = []
    for market in markets:
        opp = strategy.check_market(market)
        if opp:
            opportunities.append({
                "market_id": market.id,
                "question": market.question,
                "action": opp.action,
                "yes_price": opp.yes_price,
                "no_price": opp.no_price,
                "sum_prices": opp.sum_prices,
                "profit_pct": opp.profit_pct,
                "volume": market.volume,
            })

    opportunities.sort(key=lambda x: x["profit_pct"], reverse=True)

    return {
        "description": "Risk-free profit when YES + NO != $1.00",
        "research_basis": "Basic arbitrage theory - market inefficiency exploitation",
        "total_opportunities": len(opportunities),
        "top_opportunities": opportunities[:10],
        "methodology": {
            "min_profit_pct": 0.003,
            "execution_note": "Requires simultaneous execution to lock in profit",
        },
    }


def _format_as_markdown(report_data: dict) -> str:
    """Format report as Markdown."""
    lines = [
        f"# {report_data['title']}",
        "",
        f"**Generated:** {report_data['generated_at']}",
        f"**Markets Analyzed:** {report_data['total_markets']}",
        "",
    ]

    for section_name, section_data in report_data["sections"].items():
        lines.append(f"## {section_name.replace('_', ' ').title()}")
        lines.append("")

        if "description" in section_data:
            lines.append(f"**Description:** {section_data['description']}")
            lines.append("")

        if "research_basis" in section_data:
            lines.append(f"**Research Basis:** {section_data['research_basis']}")
            lines.append("")

        if "total_opportunities" in section_data:
            lines.append(f"**Opportunities Found:** {section_data['total_opportunities']}")
            lines.append("")

        if "top_opportunities" in section_data:
            lines.append("### Top Opportunities")
            lines.append("")
            lines.append("| # | Market | Edge/Profit | Volume |")
            lines.append("|---|--------|-------------|--------|")

            for i, opp in enumerate(section_data["top_opportunities"][:10], 1):
                question = (opp.get("question") or "Unknown")[:40]
                edge = opp.get("edge") or opp.get("profit_pct", 0)
                volume = opp.get("volume", 0)
                lines.append(f"| {i} | {question} | {edge:.2%} | ${volume:,.0f} |")

            lines.append("")

        if "note" in section_data:
            lines.append(f"*Note: {section_data['note']}*")
            lines.append("")

    return "\n".join(lines)
