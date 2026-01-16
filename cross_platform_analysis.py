"""Cross-Platform Arbitrage Analysis - Polymarket vs Kalshi.

Identifies price discrepancies between platforms for the same events.
Research source: arXiv:2508.03474 - $40M+ extracted via cross-platform arb.
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from difflib import SequenceMatcher

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np

from src.core.config import Credentials
from src.adapters.polymarket import PolymarketAdapter
from src.adapters.kalshi import KalshiAdapter

# Colors
COLORS = {
    'polymarket': '#7c3aed',  # Purple
    'kalshi': '#0ea5e9',      # Blue
    'spread': '#f59e0b',      # Amber
    'bg': '#0d1117',
    'card': '#161b22',
    'text': '#c9d1d9',
}


def similarity(a: str, b: str) -> float:
    """Calculate string similarity ratio."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def find_matching_markets(poly_markets, kalshi_markets, threshold=0.6):
    """Find markets that exist on both platforms."""
    matches = []

    for pm in poly_markets:
        pm_q = pm.question.lower()

        for km in kalshi_markets:
            km_q = km.question.lower()

            # Calculate similarity
            sim = similarity(pm_q, km_q)

            if sim >= threshold:
                # Check for price discrepancy
                poly_yes = pm.yes_price
                kalshi_yes = km.yes_price

                spread = abs(poly_yes - kalshi_yes)

                matches.append({
                    'polymarket': {
                        'question': pm.question,
                        'yes_price': poly_yes,
                        'no_price': pm.no_price,
                        'volume': pm.volume,
                        'market_id': pm.id,
                    },
                    'kalshi': {
                        'question': km.question,
                        'yes_price': kalshi_yes,
                        'no_price': km.no_price,
                        'volume': km.volume,
                        'market_id': km.id,
                    },
                    'similarity': sim,
                    'spread': spread,
                    'arb_opportunity': spread > 0.02,  # 2%+ spread
                })

    # Sort by spread (largest first)
    matches.sort(key=lambda x: x['spread'], reverse=True)
    return matches


async def fetch_both_platforms():
    """Fetch market data from both platforms."""
    creds = Credentials.from_env()

    # Polymarket
    print("Connecting to Polymarket...")
    poly_adapter = PolymarketAdapter(credentials=creds)
    poly_connected = await poly_adapter.connect()
    print(f"  Polymarket: {'Connected' if poly_connected else 'Failed'}")
    print(f"  Authenticated: {poly_adapter.is_authenticated}")

    poly_markets = await poly_adapter.get_markets(active_only=True, limit=200)
    print(f"  Markets fetched: {len(poly_markets)}")

    # Kalshi
    print("\nConnecting to Kalshi...")
    kalshi_adapter = KalshiAdapter(credentials=creds)
    kalshi_connected = await kalshi_adapter.connect()
    print(f"  Kalshi: {'Connected' if kalshi_connected else 'Failed'}")
    print(f"  Authenticated: {kalshi_adapter.is_authenticated}")

    kalshi_markets = await kalshi_adapter.get_markets(active_only=True, limit=200)
    print(f"  Markets fetched: {len(kalshi_markets)}")

    await poly_adapter.disconnect()
    await kalshi_adapter.disconnect()

    return poly_markets, kalshi_markets


def create_comparison_visualization(matches, output_dir):
    """Create cross-platform comparison visualization."""
    if not matches:
        print("No matches found for visualization")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor(COLORS['bg'])

    # Title
    fig.suptitle('CROSS-PLATFORM ARBITRAGE ANALYSIS\nPolymarket vs Kalshi',
                 fontsize=20, fontweight='bold', color=COLORS['text'], y=0.98)

    # 1. Price comparison scatter
    ax1 = axes[0, 0]
    ax1.set_facecolor(COLORS['card'])

    poly_prices = [m['polymarket']['yes_price'] for m in matches[:20]]
    kalshi_prices = [m['kalshi']['yes_price'] for m in matches[:20]]

    ax1.scatter(poly_prices, kalshi_prices, c=COLORS['spread'], s=100, alpha=0.7, edgecolors='white')
    ax1.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Perfect Correlation')
    ax1.set_xlabel('Polymarket YES Price', color=COLORS['text'])
    ax1.set_ylabel('Kalshi YES Price', color=COLORS['text'])
    ax1.set_title('Price Correlation', fontsize=14, fontweight='bold', color=COLORS['text'])
    ax1.tick_params(colors=COLORS['text'])
    ax1.legend(facecolor=COLORS['card'], labelcolor=COLORS['text'])

    # 2. Spread distribution
    ax2 = axes[0, 1]
    ax2.set_facecolor(COLORS['card'])

    spreads = [m['spread'] * 100 for m in matches]  # Convert to percentage
    ax2.hist(spreads, bins=20, color=COLORS['spread'], edgecolor='white', alpha=0.8)
    ax2.axvline(x=2, color='red', linestyle='--', label='2% Arb Threshold')
    ax2.set_xlabel('Price Spread (%)', color=COLORS['text'])
    ax2.set_ylabel('Count', color=COLORS['text'])
    ax2.set_title('Spread Distribution', fontsize=14, fontweight='bold', color=COLORS['text'])
    ax2.tick_params(colors=COLORS['text'])
    ax2.legend(facecolor=COLORS['card'], labelcolor=COLORS['text'])

    # 3. Top arbitrage opportunities
    ax3 = axes[1, 0]
    ax3.set_facecolor(COLORS['card'])
    ax3.axis('off')

    arb_opps = [m for m in matches if m['arb_opportunity']][:10]

    if arb_opps:
        table_data = []
        for m in arb_opps:
            table_data.append([
                m['polymarket']['question'][:30] + '...',
                f"{m['polymarket']['yes_price']:.1%}",
                f"{m['kalshi']['yes_price']:.1%}",
                f"{m['spread']:.2%}",
            ])

        table = ax3.table(cellText=table_data,
                         colLabels=['Market', 'Poly', 'Kalshi', 'Spread'],
                         loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        for (row, col), cell in table.get_celld().items():
            cell.set_facecolor(COLORS['card'])
            cell.set_text_props(color=COLORS['text'])
            if row == 0:
                cell.set_text_props(fontweight='bold', color=COLORS['spread'])

    ax3.set_title('TOP ARBITRAGE OPPORTUNITIES', fontsize=14, fontweight='bold',
                  color=COLORS['text'], pad=20)

    # 4. Platform comparison stats
    ax4 = axes[1, 1]
    ax4.set_facecolor(COLORS['card'])
    ax4.axis('off')

    stats_text = f"""CROSS-PLATFORM ANALYSIS SUMMARY

Total Matched Markets: {len(matches)}
Arbitrage Opportunities (>2%): {len([m for m in matches if m['arb_opportunity']])}

Average Spread: {np.mean([m['spread'] for m in matches]):.2%}
Max Spread: {max(m['spread'] for m in matches):.2%}
Min Spread: {min(m['spread'] for m in matches):.2%}

Research Citation:
"Unravelling the Probabilistic Forest"
arXiv:2508.03474 (2025)
• $40M+ extracted via cross-platform arb
• 2-8% returns per trade
• Polymarket leads Kalshi in price discovery"""

    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace', color=COLORS['text'],
             bbox=dict(boxstyle='round', facecolor=COLORS['card'], edgecolor=COLORS['spread']))

    plt.tight_layout()
    plt.savefig(output_dir / 'cross_platform_analysis.png', dpi=150, facecolor=COLORS['bg'],
                bbox_inches='tight')
    plt.close()
    print(f"Created: cross_platform_analysis.png")


async def main():
    """Run cross-platform analysis."""
    print("=" * 70)
    print("CROSS-PLATFORM ARBITRAGE ANALYSIS")
    print("Polymarket vs Kalshi")
    print("=" * 70)
    print()

    output_dir = Path(__file__).parent / "results" / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        poly_markets, kalshi_markets = await fetch_both_platforms()
    except Exception as e:
        print(f"\nError connecting to platforms: {e}")
        print("\nGenerating sample analysis with available data...")

        # Fall back to Polymarket-only analysis
        creds = Credentials.from_env()
        poly_adapter = PolymarketAdapter(credentials=creds)
        await poly_adapter.connect()
        poly_markets = await poly_adapter.get_markets(active_only=True, limit=200)
        await poly_adapter.disconnect()
        kalshi_markets = []

    print(f"\nPolymarket markets: {len(poly_markets)}")
    print(f"Kalshi markets: {len(kalshi_markets)}")

    if kalshi_markets:
        # Find matches
        print("\nSearching for matching markets...")
        matches = find_matching_markets(poly_markets, kalshi_markets)
        print(f"Found {len(matches)} potential matches")

        # Analyze
        arb_opportunities = [m for m in matches if m['arb_opportunity']]
        print(f"Arbitrage opportunities (>2% spread): {len(arb_opportunities)}")

        if matches:
            print("\n" + "=" * 70)
            print("TOP PRICE DISCREPANCIES")
            print("=" * 70)

            for i, match in enumerate(matches[:10], 1):
                print(f"\n{i}. {match['polymarket']['question'][:60]}")
                print(f"   Polymarket: YES @ {match['polymarket']['yes_price']:.2%}")
                print(f"   Kalshi:     YES @ {match['kalshi']['yes_price']:.2%}")
                print(f"   Spread:     {match['spread']:.2%}")
                if match['arb_opportunity']:
                    print(f"   [ARBITRAGE OPPORTUNITY]")

            # Create visualization
            create_comparison_visualization(matches, output_dir)

            # Save results
            results_file = output_dir.parent / "opportunities" / f"cross_platform_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            results_file.parent.mkdir(parents=True, exist_ok=True)
            with open(results_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'total_matches': len(matches),
                    'arb_opportunities': len(arb_opportunities),
                    'matches': matches[:20],
                }, f, indent=2)
            print(f"\nResults saved to: {results_file}")

    else:
        print("\nKalshi connection not available - creating Polymarket-only analysis")

        # Create Polymarket summary
        print(f"\nPolymarket Summary:")
        print(f"  Total active markets: {len(poly_markets)}")
        print(f"  Total volume: ${sum(m.volume or 0 for m in poly_markets):,.0f}")

        high_prob = [m for m in poly_markets if max(m.yes_price, m.no_price) >= 0.90]
        print(f"  High-probability markets (>90%): {len(high_prob)}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
