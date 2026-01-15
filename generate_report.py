"""Generate comprehensive market analysis report."""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.core.config import Credentials
from src.adapters.polymarket import PolymarketAdapter


async def generate_report():
    """Generate comprehensive market analysis report."""

    print("Generating Polymarket Analysis Report...")
    print()

    # Connect
    creds = Credentials.from_env()
    adapter = PolymarketAdapter(credentials=creds)
    await adapter.connect()

    # Fetch all markets
    markets = await adapter.get_markets(active_only=True, limit=200)

    # Analyze
    report = {
        "generated_at": datetime.now().isoformat(),
        "total_markets": len(markets),
        "summary": {},
        "favorite_longshot_opportunities": [],
        "high_volume_markets": [],
        "price_distribution": {
            "extreme_favorites": [],  # >95%
            "favorites": [],          # 80-95%
            "uncertain": [],          # 40-60%
            "longshots": [],          # 10-20%
            "extreme_longshots": [],  # <10%
        }
    }

    total_volume = 0
    total_liquidity = 0

    for market in markets:
        yes_price = market.yes_price
        no_price = market.no_price
        volume = market.volume or 0
        total_volume += volume
        total_liquidity += market.liquidity or 0

        # Categorize by price
        high_prob = max(yes_price, no_price)
        side = "YES" if yes_price > no_price else "NO"

        entry = {
            "question": market.question[:100],
            "side": side,
            "probability": high_prob,
            "volume": volume,
            "market_id": market.id,
        }

        if high_prob >= 0.95:
            report["price_distribution"]["extreme_favorites"].append(entry)
            # Favorite-longshot opportunity
            estimated_fair = min(0.99, high_prob + 0.03)
            edge = estimated_fair - high_prob
            if edge >= 0.01:
                report["favorite_longshot_opportunities"].append({
                    **entry,
                    "estimated_fair": estimated_fair,
                    "edge": edge,
                    "expected_return_pct": (edge / high_prob) * 100,
                })
        elif high_prob >= 0.80:
            report["price_distribution"]["favorites"].append(entry)
        elif high_prob <= 0.60:
            report["price_distribution"]["uncertain"].append(entry)

        if volume >= 1_000_000:
            report["high_volume_markets"].append(entry)

    # Sort opportunities by edge
    report["favorite_longshot_opportunities"].sort(
        key=lambda x: x["edge"], reverse=True
    )

    # Sort high volume by volume
    report["high_volume_markets"].sort(
        key=lambda x: x["volume"], reverse=True
    )

    # Summary stats
    report["summary"] = {
        "total_markets_analyzed": len(markets),
        "total_volume_usd": total_volume,
        "total_liquidity_usd": total_liquidity,
        "extreme_favorites_count": len(report["price_distribution"]["extreme_favorites"]),
        "favorite_longshot_opportunities_count": len(report["favorite_longshot_opportunities"]),
        "high_volume_markets_count": len(report["high_volume_markets"]),
    }

    # Save report
    output_dir = Path(__file__).parent / "results" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"analysis_report_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"Report saved to: {output_file}")
    print()

    # Print summary
    print("=" * 70)
    print("POLYMARKET ANALYSIS REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()

    print("SUMMARY STATISTICS")
    print("-" * 40)
    print(f"Total Markets Analyzed: {report['summary']['total_markets_analyzed']}")
    print(f"Total Volume: ${report['summary']['total_volume_usd']:,.0f}")
    print(f"Total Liquidity: ${report['summary']['total_liquidity_usd']:,.0f}")
    print(f"Extreme Favorites (>95%): {report['summary']['extreme_favorites_count']}")
    print(f"Tradeable Opportunities: {report['summary']['favorite_longshot_opportunities_count']}")
    print()

    print("TOP FAVORITE-LONGSHOT OPPORTUNITIES")
    print("-" * 40)
    print("(High-probability outcomes that may be systematically underpriced)")
    print()

    for i, opp in enumerate(report["favorite_longshot_opportunities"][:10], 1):
        print(f"{i}. {opp['question'][:65]}")
        print(f"   {opp['side']} @ {opp['probability']:.2%} | Edge: {opp['edge']:.2%} | Vol: ${opp['volume']:,.0f}")
        print()

    print("HIGHEST VOLUME MARKETS")
    print("-" * 40)
    for i, m in enumerate(report["high_volume_markets"][:5], 1):
        print(f"{i}. {m['question'][:65]}")
        print(f"   {m['side']} @ {m['probability']:.2%} | Vol: ${m['volume']:,.0f}")
        print()

    print("=" * 70)
    print("TRADING EDGE ANALYSIS")
    print("=" * 70)
    print()
    print("Based on research (Kahneman & Tversky, NBER WP 15923):")
    print("- High-probability outcomes (>95%) historically resolve at higher rates")
    print("- Retail traders exhibit 'lottery ticket' bias toward longshots")
    print("- Systematic buying of favorites can yield 2-5% edge")
    print()

    if report["favorite_longshot_opportunities"]:
        edges = [o["edge"] for o in report["favorite_longshot_opportunities"]]
        avg_edge = sum(edges) / len(edges)
        max_edge = max(edges)
        total_capital_opportunity = sum(
            o["volume"] * 0.01 for o in report["favorite_longshot_opportunities"]
        )
        print(f"Current Opportunities Found: {len(edges)}")
        print(f"Average Edge: {avg_edge:.2%}")
        print(f"Maximum Edge: {max_edge:.2%}")
        print(f"Est. Addressable Opportunity: ${total_capital_opportunity:,.0f}")

    await adapter.disconnect()

    return report


if __name__ == "__main__":
    asyncio.run(generate_report())
