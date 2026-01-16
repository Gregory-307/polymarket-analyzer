"""Create HONEST visualizations - no fake edge claims."""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np

from src.core.config import Credentials
from src.adapters.polymarket import PolymarketAdapter

# Style
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = {
    'primary': '#58a6ff',
    'success': '#3fb950',
    'warning': '#d29922',
    'danger': '#f85149',
    'purple': '#a371f7',
    'bg': '#0d1117',
    'card': '#161b22',
    'text': '#c9d1d9',
    'muted': '#8b949e',
}

plt.rcParams['figure.facecolor'] = COLORS['bg']
plt.rcParams['axes.facecolor'] = COLORS['card']
plt.rcParams['axes.edgecolor'] = '#30363d'
plt.rcParams['axes.labelcolor'] = COLORS['text']
plt.rcParams['text.color'] = COLORS['text']
plt.rcParams['xtick.color'] = COLORS['muted']
plt.rcParams['ytick.color'] = COLORS['muted']
plt.rcParams['grid.color'] = '#21262d'


async def fetch_data():
    """Fetch market data."""
    creds = Credentials.from_env()
    adapter = PolymarketAdapter(credentials=creds)
    await adapter.connect()
    markets = await adapter.get_markets(active_only=True, limit=200)
    await adapter.disconnect()
    return markets


def create_honest_distribution(markets, output_dir):
    """Show actual distribution of HIGH probability side (YES or NO)."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Get HIGH probability for each market (regardless of YES/NO side)
    high_probs = []
    for m in markets:
        high_p = max(m.yes_price, m.no_price)
        high_probs.append(high_p)

    # Create histogram
    bins = np.arange(0, 1.05, 0.05)
    counts, edges, patches = ax.hist(high_probs, bins=bins, edgecolor='white', linewidth=0.5)

    # Color by zone
    for i, patch in enumerate(patches):
        mid = (edges[i] + edges[i+1]) / 2
        if mid >= 0.95:
            patch.set_facecolor(COLORS['danger'])  # Too high - no edge possible
            patch.set_alpha(0.8)
        elif mid >= 0.90:
            patch.set_facecolor(COLORS['success'])  # Potential opportunity zone
            patch.set_alpha(0.8)
        elif mid >= 0.80:
            patch.set_facecolor(COLORS['warning'])
            patch.set_alpha(0.7)
        else:
            patch.set_facecolor(COLORS['primary'])
            patch.set_alpha(0.5)

    ax.set_xlabel('Highest Probability Side (YES or NO)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Markets', fontsize=12, fontweight='bold')
    ax.set_title('POLYMARKET: Distribution by Highest Probability\n(Showing the "favorite" side of each market)',
                 fontsize=16, fontweight='bold', color=COLORS['text'], pad=20)

    # Add annotations
    ax.axvline(x=0.95, color=COLORS['danger'], linestyle='--', linewidth=2, alpha=0.7)
    ax.axvline(x=0.90, color=COLORS['success'], linestyle='--', linewidth=2, alpha=0.7)

    ax.text(0.97, ax.get_ylim()[1]*0.9, '99%+ zone\nNO EDGE\n(too certain)',
            fontsize=9, color=COLORS['danger'], ha='center')
    ax.text(0.92, ax.get_ylim()[1]*0.7, '90-95%\nPOTENTIAL\nOPPORTUNITY',
            fontsize=9, color=COLORS['success'], ha='center')

    # Stats
    above_99 = len([p for p in high_probs if p >= 0.99])
    zone_95_99 = len([p for p in high_probs if 0.95 <= p < 0.99])
    zone_90_95 = len([p for p in high_probs if 0.90 <= p < 0.95])

    stats = f"99%+ (no edge): {above_99}\n95-99%: {zone_95_99}\n90-95% (opportunity): {zone_90_95}"
    ax.text(0.02, 0.98, stats, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor=COLORS['card'], edgecolor='#30363d'))

    plt.tight_layout()
    plt.savefig(output_dir / 'honest_distribution.png', dpi=150, facecolor=COLORS['bg'])
    plt.close()
    print("Created: honest_distribution.png")


def create_opportunity_zone_chart(markets, output_dir):
    """Show only the 90-96% zone where edge might exist."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Filter to opportunity zone
    opportunities = []
    for m in markets:
        high_p = max(m.yes_price, m.no_price)
        side = "YES" if m.yes_price > m.no_price else "NO"

        # Only 90-96% range has theoretical edge
        if 0.90 <= high_p < 0.97:
            opportunities.append({
                'question': m.question[:50],
                'high_prob': high_p,
                'side': side,
                'volume': m.volume or 0,
                'liquidity': m.liquidity or 0,
            })

    if not opportunities:
        print("No opportunities in 90-96% zone")
        return

    # Sort by probability
    opportunities.sort(key=lambda x: x['high_prob'])

    probs = [o['high_prob'] for o in opportunities]
    volumes = [o['volume'] for o in opportunities]

    # Size by volume
    max_vol = max(volumes) if volumes else 1
    sizes = [50 + (v / max_vol) * 500 for v in volumes]

    scatter = ax.scatter(range(len(opportunities)), probs, s=sizes,
                        c=COLORS['success'], alpha=0.7, edgecolors='white')

    ax.set_ylabel('Market Probability', fontsize=12, fontweight='bold')
    ax.set_xlabel('Market Index (sorted by probability)', fontsize=12, fontweight='bold')
    ax.set_title(f'OPPORTUNITY ZONE: {len(opportunities)} Markets at 90-96%\n(Bubble size = volume)',
                 fontsize=16, fontweight='bold', color=COLORS['text'], pad=20)

    ax.set_ylim(0.88, 0.98)
    ax.axhline(y=0.95, color=COLORS['warning'], linestyle='--', alpha=0.7, label='95% threshold')
    ax.axhline(y=0.90, color=COLORS['muted'], linestyle='--', alpha=0.7, label='90% threshold')

    # Annotate top volume markets
    top_vol = sorted(opportunities, key=lambda x: x['volume'], reverse=True)[:3]
    for opp in top_vol:
        idx = opportunities.index(opp)
        ax.annotate(opp['question'][:30] + '...',
                   (idx, opp['high_prob']),
                   textcoords="offset points", xytext=(10, 10),
                   fontsize=8, color=COLORS['text'],
                   arrowprops=dict(arrowstyle='->', color=COLORS['muted']))

    ax.legend(loc='lower right', facecolor=COLORS['card'])

    plt.tight_layout()
    plt.savefig(output_dir / 'opportunity_zone.png', dpi=150, facecolor=COLORS['bg'])
    plt.close()
    print("Created: opportunity_zone.png")


def create_honest_summary(markets, output_dir):
    """Create honest summary dashboard."""
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor(COLORS['bg'])

    fig.suptitle('POLYMARKET ANALYSIS - HONEST ASSESSMENT',
                 fontsize=24, fontweight='bold', color=COLORS['primary'], y=0.98)
    fig.text(0.5, 0.94, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | Live Data',
             ha='center', fontsize=12, color=COLORS['muted'])

    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)

    # Categorize markets
    all_markets = []
    for m in markets:
        high_p = max(m.yes_price, m.no_price)
        side = "YES" if m.yes_price > m.no_price else "NO"
        all_markets.append({
            'question': m.question,
            'high_prob': high_p,
            'side': side,
            'volume': m.volume or 0,
            'liquidity': m.liquidity or 0,
        })

    zone_99plus = [m for m in all_markets if m['high_prob'] >= 0.99]
    zone_95_99 = [m for m in all_markets if 0.95 <= m['high_prob'] < 0.99]
    zone_90_95 = [m for m in all_markets if 0.90 <= m['high_prob'] < 0.95]
    zone_other = [m for m in all_markets if m['high_prob'] < 0.90]

    # Panel 1: Market breakdown
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(COLORS['card'])

    categories = ['99%+\n(No Edge)', '95-99%\n(Minimal Edge)', '90-95%\n(Opportunity)', '<90%\n(Uncertain)']
    counts = [len(zone_99plus), len(zone_95_99), len(zone_90_95), len(zone_other)]
    colors = [COLORS['danger'], COLORS['warning'], COLORS['success'], COLORS['primary']]

    bars = ax1.bar(categories, counts, color=colors, edgecolor='white')
    ax1.set_ylabel('Number of Markets', fontsize=11, fontweight='bold')
    ax1.set_title('Market Distribution by Probability Zone', fontsize=14, fontweight='bold', color=COLORS['text'])

    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(count), ha='center', fontsize=12, fontweight='bold', color=COLORS['text'])

    # Panel 2: Volume by zone
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(COLORS['card'])

    volumes = [
        sum(m['volume'] for m in zone_99plus) / 1e6,
        sum(m['volume'] for m in zone_95_99) / 1e6,
        sum(m['volume'] for m in zone_90_95) / 1e6,
        sum(m['volume'] for m in zone_other) / 1e6,
    ]

    bars = ax2.bar(categories, volumes, color=colors, edgecolor='white')
    ax2.set_ylabel('Volume ($ Millions)', fontsize=11, fontweight='bold')
    ax2.set_title('Trading Volume by Zone', fontsize=14, fontweight='bold', color=COLORS['text'])

    for bar, vol in zip(bars, volumes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'${vol:.1f}M', ha='center', fontsize=10, color=COLORS['text'])

    # Panel 3: Opportunity zone details
    ax3 = fig.add_subplot(gs[1, :])
    ax3.set_facecolor(COLORS['card'])
    ax3.axis('off')
    ax3.set_title('OPPORTUNITY ZONE (90-95%) - Potential Trades',
                  fontsize=14, fontweight='bold', color=COLORS['success'], pad=20)

    # Show actual opportunities
    opps = sorted(zone_90_95, key=lambda x: x['volume'], reverse=True)[:8]

    if opps:
        table_data = []
        for m in opps:
            table_data.append([
                m['question'][:45] + '...' if len(m['question']) > 45 else m['question'],
                m['side'],
                f"{m['high_prob']:.1%}",
                f"${m['volume']:,.0f}",
                f"${m['liquidity']:,.0f}",
            ])

        table = ax3.table(cellText=table_data,
                         colLabels=['Market', 'Side', 'Prob', 'Volume', 'Liquidity'],
                         loc='center', cellLoc='left',
                         colWidths=[0.45, 0.08, 0.1, 0.15, 0.15])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        for (row, col), cell in table.get_celld().items():
            cell.set_facecolor(COLORS['card'])
            cell.set_edgecolor('#30363d')
            if row == 0:
                cell.set_text_props(fontweight='bold', color=COLORS['success'])
            else:
                cell.set_text_props(color=COLORS['text'])
    else:
        ax3.text(0.5, 0.5, 'No markets in 90-95% zone currently',
                ha='center', va='center', fontsize=14, color=COLORS['muted'])

    # Panel 4: Honest assessment
    ax4 = fig.add_subplot(gs[2, :])
    ax4.set_facecolor(COLORS['card'])
    ax4.axis('off')

    honest_text = """
HONEST ASSESSMENT
================================================================================

WHAT WE KNOW (FACTS):
- {total} total active markets on Polymarket
- {opp_count} markets in the 90-95% "opportunity zone"
- ${total_vol:,.0f} total volume across all markets
- ${opp_vol:,.0f} volume in opportunity zone

WHAT WE DON'T KNOW (ASSUMPTIONS):
- Whether favorite-longshot bias exists on Polymarket (no historical data)
- Actual edge (our estimates are based on academic research, not measured)
- True resolution rates at each probability level

THE THEORY:
- Research suggests 90-95% markets may resolve at 93-97% (2-3% edge)
- Markets at 99%+ have NO edge - already at ceiling
- This is NOT VALIDATED on Polymarket specifically

RISK:
- If theory is wrong, we're just paying transaction fees
- Black swan events (5-10% at this level) cause 100% loss
- Position sizing is critical

RECOMMENDATION:
- Focus ONLY on 90-95% zone if pursuing this strategy
- Start small to validate the edge exists
- Track all trades to measure actual performance
================================================================================
""".format(
        total=len(markets),
        opp_count=len(zone_90_95),
        total_vol=sum(m['volume'] for m in all_markets),
        opp_vol=sum(m['volume'] for m in zone_90_95),
    )

    ax4.text(0.02, 0.98, honest_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace', color=COLORS['text'])

    plt.savefig(output_dir / 'honest_summary.png', dpi=150, facecolor=COLORS['bg'], bbox_inches='tight')
    plt.close()
    print("Created: honest_summary.png")


def create_specific_trades(markets, output_dir):
    """Create chart of specific actionable trades."""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_facecolor(COLORS['card'])

    # Get 90-95% zone markets with decent volume
    trades = []
    for m in markets:
        high_p = max(m.yes_price, m.no_price)
        side = "YES" if m.yes_price > m.no_price else "NO"

        if 0.90 <= high_p < 0.96 and (m.volume or 0) >= 10000:
            trades.append({
                'question': m.question,
                'high_prob': high_p,
                'side': side,
                'volume': m.volume or 0,
                'liquidity': m.liquidity or 0,
            })

    trades.sort(key=lambda x: x['volume'], reverse=True)
    trades = trades[:10]

    if not trades:
        ax.text(0.5, 0.5, 'No trades meet criteria\n(90-95% prob, >$10K volume)',
                ha='center', va='center', fontsize=16, color=COLORS['muted'])
        plt.savefig(output_dir / 'specific_trades.png', dpi=150, facecolor=COLORS['bg'])
        plt.close()
        return

    # Create horizontal bar chart
    questions = [t['question'][:40] + '...' for t in trades]
    probs = [t['high_prob'] for t in trades]
    volumes = [t['volume'] / 1000 for t in trades]  # In thousands

    y_pos = np.arange(len(trades))

    bars = ax.barh(y_pos, probs, color=COLORS['success'], edgecolor='white', alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(questions, fontsize=9)
    ax.set_xlabel('Probability', fontsize=12, fontweight='bold')
    ax.set_xlim(0.85, 1.0)
    ax.set_title('ACTIONABLE TRADES: 90-95% Zone, >$10K Volume\n(These are the only markets with potential edge)',
                 fontsize=14, fontweight='bold', color=COLORS['text'], pad=20)

    # Add annotations
    for i, (t, bar) in enumerate(zip(trades, bars)):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
               f"{t['side']} @ {t['high_prob']:.1%} | ${t['volume']:,.0f}",
               va='center', fontsize=9, color=COLORS['text'])

    ax.axvline(x=0.95, color=COLORS['warning'], linestyle='--', alpha=0.7)
    ax.axvline(x=0.90, color=COLORS['muted'], linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_dir / 'specific_trades.png', dpi=150, facecolor=COLORS['bg'])
    plt.close()
    print("Created: specific_trades.png")


async def main():
    print("=" * 60)
    print("CREATING HONEST VISUALIZATIONS")
    print("=" * 60)

    output_dir = Path(__file__).parent / "results" / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nFetching data...")
    markets = await fetch_data()
    print(f"Got {len(markets)} markets")

    print("\nGenerating honest charts...")
    create_honest_distribution(markets, output_dir)
    create_opportunity_zone_chart(markets, output_dir)
    create_honest_summary(markets, output_dir)
    create_specific_trades(markets, output_dir)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
