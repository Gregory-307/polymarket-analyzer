"""Create visualizations with REAL spread data from order books."""

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

GROSS_EDGE = 0.02  # 2% assumed from research


async def fetch_data_with_spreads():
    """Fetch market data AND actual spreads from order books."""
    creds = Credentials.from_env()
    adapter = PolymarketAdapter(credentials=creds)
    await adapter.connect()
    markets = await adapter.get_markets(active_only=True, limit=200)

    # Get markets in 90-96% zone and fetch their spreads
    analyzed = []
    for m in markets:
        high_p = max(m.yes_price, m.no_price)
        side = "YES" if m.yes_price > m.no_price else "NO"

        data = {
            'question': m.question,
            'side': side,
            'price': high_p,
            'volume': m.volume or 0,
            'liquidity': m.liquidity or 0,
            'category': m.category or 'unknown',
            'spread_pct': None,
            'net_edge': None,
            'ex_25': None,
        }

        # Fetch order book for 90-96% zone
        if 0.90 <= high_p < 0.97:
            tokens = m.raw.get("tokens", [])
            token_id = None
            for t in tokens:
                if t.get("outcome", "").upper() == side:
                    token_id = t.get("token_id")
                    break

            if not token_id:
                import json
                clob_ids = m.raw.get("clobTokenIds")
                if clob_ids:
                    try:
                        ids = json.loads(clob_ids) if isinstance(clob_ids, str) else clob_ids
                        if side == "YES" and len(ids) > 0:
                            token_id = ids[0]
                        elif side == "NO" and len(ids) > 1:
                            token_id = ids[1]
                    except:
                        pass

            if token_id:
                book = await adapter.get_order_book(token_id)
                if book.bids and book.asks:
                    best_bid = book.bids[0].price
                    best_ask = book.asks[0].price
                    spread = best_ask - best_bid
                    spread_pct = spread / ((best_bid + best_ask) / 2) * 100
                    net_edge = GROSS_EDGE - (spread_pct / 100)
                    ex_25 = 25 * net_edge / high_p

                    data['spread_pct'] = spread_pct
                    data['net_edge'] = net_edge
                    data['ex_25'] = ex_25

                await asyncio.sleep(0.05)  # Rate limit

        analyzed.append(data)

    await adapter.disconnect()
    return analyzed


def create_spread_analysis(markets, output_dir):
    """Show actual spreads vs E(X) for 90-96% zone."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor(COLORS['bg'])
    fig.suptitle('ACTUAL SPREAD ANALYSIS - Real Order Book Data',
                 fontsize=18, fontweight='bold', color=COLORS['primary'], y=0.98)

    # Filter to 90-96% zone with spread data
    zone = [m for m in markets if m['spread_pct'] is not None]
    zone.sort(key=lambda x: x['spread_pct'])

    # Panel 1: Spread by market
    ax1 = axes[0, 0]
    ax1.set_facecolor(COLORS['card'])

    names = [m['question'][:25] + '...' for m in zone]
    spreads = [m['spread_pct'] for m in zone]
    colors = [COLORS['success'] if m['net_edge'] > 0 else COLORS['danger'] for m in zone]

    bars = ax1.barh(range(len(zone)), spreads, color=colors, edgecolor='white', alpha=0.8)
    ax1.set_yticks(range(len(zone)))
    ax1.set_yticklabels(names, fontsize=8)
    ax1.set_xlabel('Spread (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Actual Spreads by Market\n(Green = Positive E(X), Red = Negative)', fontsize=12, fontweight='bold')
    ax1.axvline(x=2.0, color=COLORS['warning'], linestyle='--', label='2% edge threshold')
    ax1.legend(loc='lower right', facecolor=COLORS['card'])

    # Panel 2: E(X) by market
    ax2 = axes[0, 1]
    ax2.set_facecolor(COLORS['card'])

    ex_values = [m['ex_25'] for m in zone]
    colors = [COLORS['success'] if e > 0 else COLORS['danger'] for e in ex_values]

    bars = ax2.barh(range(len(zone)), ex_values, color=colors, edgecolor='white', alpha=0.8)
    ax2.set_yticks(range(len(zone)))
    ax2.set_yticklabels(names, fontsize=8)
    ax2.set_xlabel('E(X) on $25 bet', fontsize=11, fontweight='bold')
    ax2.set_title('Expected Value by Market\n(After spread costs)', fontsize=12, fontweight='bold')
    ax2.axvline(x=0, color='white', linestyle='-', linewidth=2)

    # Panel 3: Tradeable vs Non-tradeable summary
    ax3 = axes[1, 0]
    ax3.set_facecolor(COLORS['card'])

    tradeable = [m for m in zone if m['net_edge'] > 0]
    non_tradeable = [m for m in zone if m['net_edge'] <= 0]

    categories = ['Positive E(X)\n(TRADEABLE)', 'Negative E(X)\n(DO NOT TRADE)']
    counts = [len(tradeable), len(non_tradeable)]
    total_ex = [sum(m['ex_25'] for m in tradeable), sum(m['ex_25'] for m in non_tradeable)]

    x = np.arange(2)
    width = 0.35

    bars1 = ax3.bar(x - width/2, counts, width, label='# Markets', color=COLORS['primary'])
    ax3.set_ylabel('Number of Markets', fontsize=11, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories)
    ax3.set_title('Tradeable vs Non-Tradeable (90-96% Zone)', fontsize=12, fontweight='bold')

    # Add count labels
    for bar, count, tex in zip(bars1, counts, total_ex):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{count} mkts\nE(X): ${tex:.2f}', ha='center', fontsize=10, color=COLORS['text'])

    # Panel 4: Summary stats
    ax4 = axes[1, 1]
    ax4.set_facecolor(COLORS['card'])
    ax4.axis('off')

    total_positive_ex = sum(m['ex_25'] for m in tradeable)
    avg_spread_tradeable = np.mean([m['spread_pct'] for m in tradeable]) if tradeable else 0
    avg_spread_all = np.mean([m['spread_pct'] for m in zone]) if zone else 0

    summary = f"""
SPREAD ANALYSIS SUMMARY
{'='*50}

MARKETS ANALYZED: {len(zone)} (90-96% zone)

TRADEABLE (spread < 2%): {len(tradeable)} markets
  - Total E(X): ${total_positive_ex:.2f}
  - Capital needed: ${25 * len(tradeable)}
  - Expected ROI: {total_positive_ex / (25 * len(tradeable)) * 100:.1f}%
  - Avg spread: {avg_spread_tradeable:.2f}%

NON-TRADEABLE (spread >= 2%): {len(non_tradeable)} markets
  - These have NEGATIVE expected value
  - Spread costs exceed theoretical edge

KEY INSIGHT:
  Gross edge (from research): 2.0%
  Avg spread (tradeable): {avg_spread_tradeable:.2f}%
  Net edge: {2.0 - avg_spread_tradeable:.2f}%

TRADEABLE MARKETS:
"""
    for m in tradeable:
        summary += f"  - {m['question'][:35]}... ({m['side']} @ {m['price']:.1%})\n"
        summary += f"    Spread: {m['spread_pct']:.2f}% | E(X): ${m['ex_25']:.2f}\n"

    ax4.text(0.02, 0.98, summary, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace', color=COLORS['text'])

    plt.tight_layout()
    plt.savefig(output_dir / 'honest_summary.png', dpi=150, facecolor=COLORS['bg'], bbox_inches='tight')
    plt.close()
    print("Created: honest_summary.png")


def create_ex_analysis(markets, output_dir):
    """Create E(X) scenario analysis chart."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor(COLORS['bg'])
    fig.suptitle('EXPECTED VALUE ANALYSIS',
                 fontsize=18, fontweight='bold', color=COLORS['primary'], y=0.98)

    tradeable = [m for m in markets if m['net_edge'] is not None and m['net_edge'] > 0]
    n = len(tradeable)
    total_bet = 25 * n

    if n == 0:
        for ax in axes.flat:
            ax.text(0.5, 0.5, 'No tradeable markets found', ha='center', va='center')
        plt.savefig(output_dir / 'expected_value_analysis.png', dpi=150, facecolor=COLORS['bg'])
        plt.close()
        return

    # Calculate win amounts
    win_amounts = [25 / m['price'] - 25 for m in tradeable]
    total_win = sum(win_amounts)
    avg_price = np.mean([m['price'] for m in tradeable])

    # Panel 1: Profit by number of wins
    ax1 = axes[0, 0]
    ax1.set_facecolor(COLORS['card'])

    wins_range = range(n + 1)
    avg_win = total_win / n
    profits = [w * avg_win + (n - w) * (-25) for w in wins_range]
    colors = [COLORS['success'] if p > 0 else COLORS['danger'] for p in profits]

    bars = ax1.bar(wins_range, profits, color=colors, edgecolor='white', alpha=0.8)
    ax1.axhline(y=0, color='white', linestyle='-', linewidth=2)
    ax1.set_xlabel('Number of Winning Trades', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Profit ($)', fontsize=11, fontweight='bold')
    ax1.set_title(f'Profit by Win Count ({n} trades, ${total_bet} total)', fontsize=12, fontweight='bold')
    ax1.set_xticks(wins_range)

    # Panel 2: E(X) by edge scenario
    ax2 = axes[0, 1]
    ax2.set_facecolor(COLORS['card'])

    edges = np.linspace(0, 0.05, 50)
    evs = []
    for edge in edges:
        total_ev = 0
        for m in tradeable:
            net = edge - (m['spread_pct'] / 100)
            ev = 25 * net / m['price']
            total_ev += ev
        evs.append(total_ev)

    ax2.plot(edges * 100, evs, color=COLORS['primary'], linewidth=3)
    ax2.axhline(y=0, color='white', linestyle='--', linewidth=1)
    ax2.axvline(x=2.0, color=COLORS['warning'], linestyle='--', linewidth=2, label='Research estimate (2%)')
    ax2.fill_between(edges * 100, evs, 0, where=[e > 0 for e in evs], color=COLORS['success'], alpha=0.3)
    ax2.fill_between(edges * 100, evs, 0, where=[e <= 0 for e in evs], color=COLORS['danger'], alpha=0.3)
    ax2.set_xlabel('Assumed Gross Edge (%)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Portfolio E(X) ($)', fontsize=11, fontweight='bold')
    ax2.set_title('E(X) by Edge Assumption', fontsize=12, fontweight='bold')
    ax2.legend(facecolor=COLORS['card'])

    # Panel 3: Probability of profit
    ax3 = axes[1, 0]
    ax3.set_facecolor(COLORS['card'])

    import math

    # Find min wins needed
    for min_wins in range(n + 1):
        test_profit = min_wins * avg_win + (n - min_wins) * (-25)
        if test_profit > 0:
            break

    probs_profit = []
    for edge in edges:
        avg_true_prob = min(0.99, avg_price + edge)
        prob = 0
        for k in range(min_wins, n + 1):
            comb = math.factorial(n) / (math.factorial(k) * math.factorial(n - k))
            prob += comb * (avg_true_prob ** k) * ((1 - avg_true_prob) ** (n - k))
        probs_profit.append(prob * 100)

    ax3.plot(edges * 100, probs_profit, color=COLORS['success'], linewidth=3)
    ax3.axhline(y=50, color='white', linestyle='--', linewidth=1, alpha=0.5)
    ax3.axvline(x=2.0, color=COLORS['warning'], linestyle='--', linewidth=2)
    ax3.set_xlabel('Assumed Gross Edge (%)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('P(Profit) %', fontsize=11, fontweight='bold')
    ax3.set_title(f'Probability of Profit (need {min_wins}+ wins out of {n})', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 100)

    # Panel 4: Summary
    ax4 = axes[1, 1]
    ax4.set_facecolor(COLORS['card'])
    ax4.axis('off')

    # Calculate at 2% edge
    idx_2pct = 20  # 2% in our range
    ex_2pct = evs[idx_2pct] if idx_2pct < len(evs) else 0
    prob_2pct = probs_profit[idx_2pct] if idx_2pct < len(probs_profit) else 0

    summary = f"""
EXPECTED VALUE SUMMARY
{'='*50}

PORTFOLIO
  Trades: {n}
  Capital: ${total_bet}
  Avg price: {avg_price:.1%}
  Win per trade: ~${avg_win:.2f}
  Lose per trade: -$25.00

BREAKEVEN
  Need {min_wins} out of {n} wins to profit
  Breakeven rate: {min_wins/n:.0%}

IF MARKET EFFICIENT (0% edge):
  E(X): $0.00
  P(Profit): {probs_profit[0]:.0f}%

IF 2% EDGE (research estimate):
  E(X): ${ex_2pct:.2f}
  P(Profit): {prob_2pct:.0f}%
  ROI: {ex_2pct/total_bet*100:.1f}%

IF 4% EDGE (optimistic):
  E(X): ${evs[40] if len(evs) > 40 else 0:.2f}
  P(Profit): {probs_profit[40] if len(probs_profit) > 40 else 0:.0f}%

{'='*50}
"""

    ax4.text(0.02, 0.98, summary, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace', color=COLORS['text'])

    plt.tight_layout()
    plt.savefig(output_dir / 'expected_value_analysis.png', dpi=150, facecolor=COLORS['bg'], bbox_inches='tight')
    plt.close()
    print("Created: expected_value_analysis.png")


def create_specific_trades(markets, output_dir):
    """Create chart of specific actionable trades with real spreads."""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_facecolor(COLORS['card'])

    # Get tradeable markets (positive E(X))
    tradeable = [m for m in markets if m['net_edge'] is not None and m['net_edge'] > 0]
    tradeable.sort(key=lambda x: x['ex_25'], reverse=True)

    if not tradeable:
        ax.text(0.5, 0.5, 'No tradeable markets found',
                ha='center', va='center', fontsize=16, color=COLORS['muted'])
        plt.savefig(output_dir / 'specific_trades.png', dpi=150, facecolor=COLORS['bg'])
        plt.close()
        return

    # Create horizontal bar chart showing E(X)
    questions = [f"{m['question'][:35]}..." for m in tradeable]
    ex_values = [m['ex_25'] for m in tradeable]

    y_pos = np.arange(len(tradeable))

    bars = ax.barh(y_pos, ex_values, color=COLORS['success'], edgecolor='white', alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(questions, fontsize=9)
    ax.set_xlabel('E(X) on $25 bet', fontsize=12, fontweight='bold')
    ax.set_title(f'TRADEABLE MARKETS: {len(tradeable)} with Positive E(X)\n(Real spreads from order books, assuming 2% gross edge)',
                 fontsize=14, fontweight='bold', color=COLORS['text'], pad=20)

    # Add annotations
    for i, (m, bar) in enumerate(zip(tradeable, bars)):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
               f"{m['side']} @ {m['price']:.1%} | Spread: {m['spread_pct']:.2f}% | E(X): ${m['ex_25']:.2f}",
               va='center', fontsize=9, color=COLORS['text'])

    # Total E(X)
    total_ex = sum(ex_values)
    ax.text(0.95, 0.05, f"Total E(X): ${total_ex:.2f} on ${25*len(tradeable)}",
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            color=COLORS['success'], ha='right',
            bbox=dict(boxstyle='round', facecolor=COLORS['card'], edgecolor=COLORS['success']))

    plt.tight_layout()
    plt.savefig(output_dir / 'specific_trades.png', dpi=150, facecolor=COLORS['bg'])
    plt.close()
    print("Created: specific_trades.png")


def create_distribution(markets, output_dir):
    """Show distribution of all markets."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_facecolor(COLORS['card'])

    prices = [m['price'] for m in markets]

    bins = np.arange(0.5, 1.02, 0.02)
    counts, edges, patches = ax.hist(prices, bins=bins, edgecolor='white', linewidth=0.5)

    for i, patch in enumerate(patches):
        mid = (edges[i] + edges[i+1]) / 2
        if mid >= 0.97:
            patch.set_facecolor(COLORS['danger'])
        elif mid >= 0.90:
            patch.set_facecolor(COLORS['success'])
        else:
            patch.set_facecolor(COLORS['primary'])
        patch.set_alpha(0.8)

    ax.axvline(x=0.96, color=COLORS['warning'], linestyle='--', linewidth=2, label='96% upper bound')
    ax.axvline(x=0.90, color=COLORS['success'], linestyle='--', linewidth=2, label='90% lower bound')

    ax.set_xlabel('Highest Probability Side', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Markets', fontsize=12, fontweight='bold')
    ax.set_title('Market Distribution by Probability\n(Green = Target Zone 90-96%)', fontsize=14, fontweight='bold')
    ax.legend(facecolor=COLORS['card'])

    # Stats
    zone_90_96 = len([p for p in prices if 0.90 <= p < 0.96])
    above_96 = len([p for p in prices if p >= 0.96])
    below_90 = len([p for p in prices if p < 0.90])

    stats = f"90-96% zone: {zone_90_96}\n96%+: {above_96}\n<90%: {below_90}"
    ax.text(0.02, 0.98, stats, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor=COLORS['card'], edgecolor='#30363d'))

    plt.tight_layout()
    plt.savefig(output_dir / 'honest_distribution.png', dpi=150, facecolor=COLORS['bg'])
    plt.close()
    print("Created: honest_distribution.png")


async def main():
    print("=" * 60)
    print("CREATING VISUALIZATIONS WITH REAL SPREAD DATA")
    print("=" * 60)

    output_dir = Path(__file__).parent / "results" / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nFetching data and order book spreads...")
    markets = await fetch_data_with_spreads()
    print(f"Got {len(markets)} markets")

    zone = [m for m in markets if m['spread_pct'] is not None]
    tradeable = [m for m in zone if m['net_edge'] > 0]
    print(f"90-96% zone: {len(zone)} markets")
    print(f"Tradeable (positive E(X)): {len(tradeable)} markets")

    print("\nGenerating charts...")
    create_distribution(markets, output_dir)
    create_spread_analysis(markets, output_dir)
    create_ex_analysis(markets, output_dir)
    create_specific_trades(markets, output_dir)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
