"""PDF report generation command."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import click

from ...utils import async_command, print_header
from ....core.config import Credentials
from ....adapters.polymarket import PolymarketAdapter
from ....strategies.favorite_longshot import FavoriteLongshotStrategy
from ....strategies.single_arb import SingleConditionArbitrage


@click.command()
@click.option(
    "--strategy",
    type=click.Choice(["favorite_longshot", "single_arb", "all"], case_sensitive=False),
    default="favorite_longshot",
    help="Strategy to report on.",
)
@click.option("--limit", type=int, default=200, help="Markets to analyze.")
@click.option("--output", type=click.Path(), default=None, help="Output file path.")
@async_command
async def pdf(strategy: str, limit: int, output: str | None) -> None:
    """Generate PDF briefing for a strategy.

    Creates a professional PDF report with:
    - Executive summary
    - Opportunity analysis
    - Risk metrics
    - Position recommendations

    Requires weasyprint for PDF generation.

    \b
    Examples:
      python -m src report pdf --strategy favorite_longshot
      python -m src report pdf --strategy single_arb --output briefing.pdf
    """
    print_header(f"GENERATING PDF: {strategy.upper()}")

    # Check for weasyprint
    try:
        from weasyprint import HTML, CSS
        import markdown
    except ImportError:
        click.echo("\nError: PDF generation requires weasyprint and markdown packages.")
        click.echo("Install with: pip install weasyprint markdown")
        return

    creds = Credentials.from_env()
    adapter = PolymarketAdapter(credentials=creds)

    click.echo("\n  Fetching market data...")
    await adapter.connect()
    markets = await adapter.get_markets(active_only=True, limit=limit)
    await adapter.disconnect()

    # Generate Markdown content
    md_content = _generate_briefing_markdown(strategy, markets)

    # Convert to HTML
    html_content = markdown.markdown(
        md_content,
        extensions=["tables", "fenced_code"],
    )

    # Wrap with styling
    styled_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{
                font-family: 'Helvetica Neue', Arial, sans-serif;
                line-height: 1.6;
                max-width: 800px;
                margin: 40px auto;
                padding: 20px;
                color: #333;
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #34495e;
                margin-top: 30px;
            }}
            h3 {{
                color: #7f8c8d;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}
            th {{
                background-color: #3498db;
                color: white;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .metric {{
                display: inline-block;
                padding: 10px 20px;
                margin: 5px;
                background: #ecf0f1;
                border-radius: 5px;
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: #2980b9;
            }}
            .warning {{
                background-color: #fff3cd;
                border: 1px solid #ffc107;
                padding: 15px;
                border-radius: 5px;
                margin: 20px 0;
            }}
            code {{
                background: #f4f4f4;
                padding: 2px 6px;
                border-radius: 3px;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    # Generate PDF
    output_dir = Path("results/reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    if output:
        output_file = Path(output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"{strategy}_briefing_{timestamp}.pdf"

    HTML(string=styled_html).write_pdf(output_file)
    click.echo(f"\n  PDF saved to: {output_file}")


def _generate_briefing_markdown(strategy: str, markets: list) -> str:
    """Generate briefing content as Markdown."""
    lines = [
        f"# {strategy.replace('_', ' ').title()} Strategy Briefing",
        "",
        f"**Date:** {datetime.now().strftime('%B %d, %Y')}",
        "",
        "---",
        "",
    ]

    if strategy in ("favorite_longshot", "all"):
        lines.extend(_generate_fl_section(markets))

    if strategy in ("single_arb", "all"):
        lines.extend(_generate_arb_section(markets))

    # Risk disclaimer
    lines.extend([
        "",
        "---",
        "",
        "## Risk Disclaimer",
        "",
        "This analysis is for educational purposes only. Trading in prediction markets ",
        "involves risk of loss. Past performance does not guarantee future results. ",
        "Always do your own research and never trade with money you cannot afford to lose.",
        "",
    ])

    return "\n".join(lines)


def _generate_fl_section(markets: list) -> list[str]:
    """Generate favorite-longshot section."""
    strategy = FavoriteLongshotStrategy(min_probability=0.90, min_edge=0.01)

    opportunities = []
    for market in markets:
        opp = strategy.check_market(market)
        if opp:
            opportunities.append({
                "question": market.question,
                "side": opp.side,
                "price": opp.current_price,
                "edge": opp.edge,
                "volume": market.volume,
            })

    opportunities.sort(key=lambda x: x["edge"], reverse=True)

    lines = [
        "## Favorite-Longshot Bias Strategy",
        "",
        "### Overview",
        "",
        "The favorite-longshot bias is a well-documented behavioral phenomenon where:",
        "- **Long shots** (low probability) are systematically **overpriced**",
        "- **Favorites** (high probability) are systematically **underpriced**",
        "",
        "**Research Basis:** Prospect Theory (Kahneman & Tversky, 1979), NBER Working Paper 15923",
        "",
        f"### Current Opportunities ({len(opportunities)} found)",
        "",
    ]

    if opportunities:
        lines.extend([
            "| Market | Side | Price | Edge | Volume |",
            "|--------|------|-------|------|--------|",
        ])

        for opp in opportunities[:10]:
            q = (opp["question"] or "Unknown")[:35]
            lines.append(
                f"| {q} | {opp['side']} | {opp['price']:.1%} | {opp['edge']:.2%} | ${opp['volume']:,.0f} |"
            )

        lines.append("")

        avg_edge = sum(o["edge"] for o in opportunities) / len(opportunities)
        lines.extend([
            "### Summary Statistics",
            "",
            f"- **Total Opportunities:** {len(opportunities)}",
            f"- **Average Edge:** {avg_edge:.2%}",
            "",
        ])

    return lines


def _generate_arb_section(markets: list) -> list[str]:
    """Generate arbitrage section."""
    strategy = SingleConditionArbitrage(min_profit_pct=0.003)

    opportunities = []
    for market in markets:
        opp = strategy.check_market(market)
        if opp:
            opportunities.append({
                "question": market.question,
                "action": opp.action,
                "sum_prices": opp.sum_prices,
                "profit_pct": opp.profit_pct,
            })

    opportunities.sort(key=lambda x: x["profit_pct"], reverse=True)

    lines = [
        "## Single-Condition Arbitrage",
        "",
        "### Overview",
        "",
        "When YES + NO prices don't sum to $1.00, risk-free profit exists:",
        "- **Sum < $1.00:** Buy both YES and NO, guaranteed profit",
        "- **Sum > $1.00:** Sell both (if possible), guaranteed profit",
        "",
        f"### Current Opportunities ({len(opportunities)} found)",
        "",
    ]

    if opportunities:
        lines.extend([
            "| Market | Action | Sum | Profit |",
            "|--------|--------|-----|--------|",
        ])

        for opp in opportunities[:10]:
            q = (opp["question"] or "Unknown")[:35]
            lines.append(
                f"| {q} | {opp['action']} | {opp['sum_prices']:.3f} | {opp['profit_pct']:.2%} |"
            )

        lines.append("")

    return lines
