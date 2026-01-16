"""Expected Value Analysis - Rigorous E(X) Calculations.

Shows:
1. Breakeven analysis for each trade
2. Portfolio expected value under different scenarios
3. How many trades need to win vs lose
"""

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
plt.rcParams['figure.facecolor'] = '#0d1117'
plt.rcParams['axes.facecolor'] = '#161b22'
plt.rcParams['axes.edgecolor'] = '#30363d'
plt.rcParams['axes.labelcolor'] = '#c9d1d9'
plt.rcParams['text.color'] = '#c9d1d9'
plt.rcParams['xtick.color'] = '#8b949e'
plt.rcParams['ytick.color'] = '#8b949e'
plt.rcParams['grid.color'] = '#21262d'

COLORS = {
    'success': '#3fb950',
    'danger': '#f85149',
    'warning': '#d29922',
    'primary': '#58a6ff',
    'text': '#c9d1d9',
    'muted': '#8b949e',
}


def calculate_expected_value(price: float, true_prob: float, bet_size: float = 1.0) -> dict:
    """Calculate expected value for a binary outcome bet.

    For buying the HIGH probability side (YES or NO):
    - Cost: price * bet_size
    - Win payout: $1 * bet_size (you get $1 per share)
    - Lose payout: $0

    Args:
        price: Market price of the side you're buying (e.g., 0.91 for 91%)
        true_prob: Your estimate of true probability
        bet_size: Dollar amount to bet

    Returns:
        Dictionary with E(X) calculations
    """
    shares = bet_size / price  # How many shares you can buy

    # If you win
    win_payout = shares * 1.0  # $1 per share
    win_profit = win_payout - bet_size

    # If you lose
    lose_payout = 0
    lose_profit = -bet_size

    # Expected value
    ev = (true_prob * win_profit) + ((1 - true_prob) * lose_profit)
    ev_pct = ev / bet_size * 100

    # Breakeven probability
    breakeven_prob = price  # You need to win at price% to break even

    # Edge
    edge = true_prob - price

    return {
        'price': price,
        'true_prob': true_prob,
        'bet_size': bet_size,
        'shares': shares,
        'win_profit': win_profit,
        'lose_profit': lose_profit,
        'expected_value': ev,
        'ev_pct': ev_pct,
        'breakeven_prob': breakeven_prob,
        'edge': edge,
    }


def portfolio_scenarios(trades: list, bet_per_trade: float = 25.0) -> dict:
    """Calculate portfolio outcomes under different scenarios.

    Args:
        trades: List of (name, price) tuples
        bet_per_trade: Amount to bet on each trade

    Returns:
        Scenario analysis
    """
    n = len(trades)
    total_invested = n * bet_per_trade

    scenarios = {}

    # Calculate profit/loss for each number of wins
    for wins in range(n + 1):
        losses = n - wins

        total_profit = 0
        for name, price in trades:
            shares = bet_per_trade / price
            win_profit = shares - bet_per_trade  # $1 * shares - cost
            lose_profit = -bet_per_trade

            # This is simplified - assumes wins are distributed proportionally
            # In reality, which specific trade wins matters
            avg_win_profit = sum((bet_per_trade / p) - bet_per_trade for _, p in trades) / n
            avg_lose_profit = -bet_per_trade

        total_profit = wins * avg_win_profit + losses * avg_lose_profit
        win_rate = wins / n

        scenarios[wins] = {
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'roi': total_profit / total_invested * 100,
            'final_value': total_invested + total_profit,
        }

    return scenarios


async def main():
    """Run expected value analysis."""

    print("=" * 70)
    print("EXPECTED VALUE ANALYSIS")
    print("=" * 70)

    # Fetch actual market data
    creds = Credentials.from_env()
    adapter = PolymarketAdapter(credentials=creds)
    await adapter.connect()
    markets = await adapter.get_markets(active_only=True, limit=200)
    await adapter.disconnect()

    # Get 90-95% zone trades with sufficient LIQUIDITY (not just volume)
    # Liquidity > $50K means spread won't eat the edge
    trades = []
    for m in markets:
        high_p = max(m.yes_price, m.no_price)
        side = "YES" if m.yes_price > m.no_price else "NO"
        liquidity = m.liquidity or 0

        if 0.90 <= high_p < 0.96 and liquidity >= 50000:
            trades.append({
                'name': m.question[:50],
                'side': side,
                'price': high_p,
                'volume': m.volume or 0,
                'liquidity': liquidity,
            })

    trades.sort(key=lambda x: x['liquidity'], reverse=True)
    # Only take tradeable markets (sufficient liquidity)

    print(f"\nAnalyzing {len(trades)} trades in 90-95% zone:\n")

    # Individual trade analysis
    print("=" * 70)
    print("INDIVIDUAL TRADE ANALYSIS")
    print("=" * 70)

    bet_size = 25.0  # $25 per trade

    for t in trades:
        print(f"\n{t['name']}")
        print(f"  Side: {t['side']} @ {t['price']:.1%}")
        print(f"  Volume: ${t['volume']:,.0f}")
        print()

        # Scenario 1: Market is efficient (no edge)
        ev_no_edge = calculate_expected_value(t['price'], t['price'], bet_size)

        # Scenario 2: 2% edge (conservative)
        ev_2pct = calculate_expected_value(t['price'], min(0.99, t['price'] + 0.02), bet_size)

        # Scenario 3: 4% edge (optimistic)
        ev_4pct = calculate_expected_value(t['price'], min(0.99, t['price'] + 0.04), bet_size)

        print(f"  Bet: ${bet_size:.2f} buys {ev_no_edge['shares']:.2f} shares")
        print(f"  Win profit: +${ev_no_edge['win_profit']:.2f}")
        print(f"  Lose profit: -${abs(ev_no_edge['lose_profit']):.2f}")
        print()
        print(f"  BREAKEVEN: Need to win {ev_no_edge['breakeven_prob']:.1%} of the time")
        print()
        print(f"  E(X) if NO EDGE (market efficient):")
        print(f"    True prob = {ev_no_edge['true_prob']:.1%}, E(X) = ${ev_no_edge['expected_value']:.2f} ({ev_no_edge['ev_pct']:.1f}%)")
        print()
        print(f"  E(X) if 2% EDGE (conservative theory):")
        print(f"    True prob = {ev_2pct['true_prob']:.1%}, E(X) = ${ev_2pct['expected_value']:.2f} ({ev_2pct['ev_pct']:.1f}%)")
        print()
        print(f"  E(X) if 4% EDGE (optimistic theory):")
        print(f"    True prob = {ev_4pct['true_prob']:.1%}, E(X) = ${ev_4pct['expected_value']:.2f} ({ev_4pct['ev_pct']:.1f}%)")
        print("-" * 50)

    # Portfolio analysis
    print("\n" + "=" * 70)
    print("PORTFOLIO ANALYSIS")
    print(f"${bet_size:.0f} x {len(trades)} trades = ${bet_size * len(trades):.0f} total")
    print("=" * 70)

    total_bet = bet_size * len(trades)

    # Calculate exact outcomes for each win count
    print(f"\nOUTCOME TABLE: How many need to win?")
    print("-" * 60)
    print(f"{'Wins':>6} {'Losses':>6} {'Win Rate':>10} {'Profit':>12} {'ROI':>10}")
    print("-" * 60)

    avg_price = sum(t['price'] for t in trades) / len(trades)
    avg_shares = bet_size / avg_price
    avg_win_profit = avg_shares - bet_size

    for wins in range(len(trades) + 1):
        losses = len(trades) - wins
        profit = wins * avg_win_profit + losses * (-bet_size)
        roi = profit / total_bet * 100
        win_rate = wins / len(trades) * 100

        # Highlight breakeven
        marker = ""
        if profit >= 0 and (wins == 0 or (wins - 1) * avg_win_profit + (losses + 1) * (-bet_size) < 0):
            marker = " <-- BREAKEVEN"

        print(f"{wins:>6} {losses:>6} {win_rate:>9.0f}% ${profit:>10.2f} {roi:>9.1f}%{marker}")

    print("-" * 60)

    # Expected outcomes under different scenarios
    print("\n" + "=" * 70)
    print("EXPECTED PORTFOLIO VALUE BY SCENARIO")
    print("=" * 70)

    scenarios = [
        ("Market is efficient (NO EDGE)", 0.00),
        ("1% edge exists", 0.01),
        ("2% edge exists", 0.02),
        ("3% edge exists", 0.03),
        ("4% edge exists", 0.04),
    ]

    print(f"\nStarting capital: ${total_bet:.2f}")
    print()

    for scenario_name, edge in scenarios:
        # Calculate expected value for portfolio
        total_ev = 0
        for t in trades:
            true_prob = min(0.99, t['price'] + edge)
            ev = calculate_expected_value(t['price'], true_prob, bet_size)
            total_ev += ev['expected_value']

        expected_final = total_bet + total_ev
        expected_roi = total_ev / total_bet * 100

        # Calculate probability of profit (simplified binomial)
        # Need to win enough to overcome losses
        import math

        # How many wins needed to profit?
        for min_wins in range(len(trades) + 1):
            test_profit = min_wins * avg_win_profit + (len(trades) - min_wins) * (-bet_size)
            if test_profit > 0:
                break

        # Probability of getting at least min_wins
        avg_true_prob = avg_price + edge
        prob_profit = 0
        for k in range(min_wins, len(trades) + 1):
            # Binomial probability
            comb = math.factorial(len(trades)) / (math.factorial(k) * math.factorial(len(trades) - k))
            prob_profit += comb * (avg_true_prob ** k) * ((1 - avg_true_prob) ** (len(trades) - k))

        print(f"{scenario_name}:")
        print(f"  Expected Value: ${total_ev:+.2f} ({expected_roi:+.1f}%)")
        print(f"  Expected Final: ${expected_final:.2f}")
        print(f"  Prob of Profit: {prob_profit:.1%} (need {min_wins}+ wins)")
        print()

    # Create visualization
    output_dir = Path(__file__).parent / "results" / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    create_ev_visualization(trades, bet_size, output_dir)

    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


def create_ev_visualization(trades: list, bet_size: float, output_dir: Path):
    """Create expected value visualization."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle('EXPECTED VALUE ANALYSIS\nFavorite-Longshot Bias Strategy',
                 fontsize=18, fontweight='bold', color='#58a6ff', y=0.98)

    n = len(trades)
    total_bet = bet_size * n
    avg_price = sum(t['price'] for t in trades) / n
    avg_shares = bet_size / avg_price
    avg_win_profit = avg_shares - bet_size

    # Panel 1: Profit by number of wins
    ax1 = axes[0, 0]
    ax1.set_facecolor('#161b22')

    wins_range = range(n + 1)
    profits = [w * avg_win_profit + (n - w) * (-bet_size) for w in wins_range]
    colors = [COLORS['success'] if p > 0 else COLORS['danger'] for p in profits]

    bars = ax1.bar(wins_range, profits, color=colors, edgecolor='white', alpha=0.8)
    ax1.axhline(y=0, color='white', linestyle='-', linewidth=2)
    ax1.set_xlabel('Number of Winning Trades', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Profit ($)', fontsize=11, fontweight='bold')
    ax1.set_title(f'Profit by Win Count (${bet_size:.0f} x {n} trades)',
                  fontsize=12, fontweight='bold', color=COLORS['text'])
    ax1.set_xticks(wins_range)

    # Add breakeven annotation
    for i, p in enumerate(profits):
        if p >= 0 and (i == 0 or profits[i-1] < 0):
            ax1.annotate(f'BREAKEVEN\n({i}/{n} wins)', (i, p),
                        textcoords="offset points", xytext=(20, 20),
                        fontsize=9, color=COLORS['warning'],
                        arrowprops=dict(arrowstyle='->', color=COLORS['warning']))

    # Panel 2: Expected value by edge assumption
    ax2 = axes[0, 1]
    ax2.set_facecolor('#161b22')

    edges = np.linspace(0, 0.05, 50)
    evs = []
    for edge in edges:
        total_ev = 0
        for t in trades:
            true_prob = min(0.99, t['price'] + edge)
            ev = calculate_expected_value(t['price'], true_prob, bet_size)
            total_ev += ev['expected_value']
        evs.append(total_ev)

    ax2.plot(edges * 100, evs, color=COLORS['primary'], linewidth=3)
    ax2.axhline(y=0, color='white', linestyle='--', linewidth=1)
    ax2.axvline(x=0, color=COLORS['danger'], linestyle='--', linewidth=1, label='No edge')
    ax2.axvline(x=2, color=COLORS['warning'], linestyle='--', linewidth=1, label='2% edge')
    ax2.fill_between(edges * 100, evs, 0, where=[e > 0 for e in evs],
                     color=COLORS['success'], alpha=0.3)
    ax2.fill_between(edges * 100, evs, 0, where=[e <= 0 for e in evs],
                     color=COLORS['danger'], alpha=0.3)
    ax2.set_xlabel('Assumed Edge (%)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Expected Value ($)', fontsize=11, fontweight='bold')
    ax2.set_title('Portfolio E(X) by Edge Assumption', fontsize=12, fontweight='bold', color=COLORS['text'])
    ax2.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor=COLORS['text'])

    # Panel 3: Probability of profit by edge
    ax3 = axes[1, 0]
    ax3.set_facecolor('#161b22')

    import math

    # Find min wins needed
    for min_wins in range(n + 1):
        test_profit = min_wins * avg_win_profit + (n - min_wins) * (-bet_size)
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
    ax3.axvline(x=0, color=COLORS['danger'], linestyle='--', linewidth=1)
    ax3.set_xlabel('Assumed Edge (%)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Probability of Profit (%)', fontsize=11, fontweight='bold')
    ax3.set_title(f'Probability of Profit (need {min_wins}+ wins out of {n})',
                  fontsize=12, fontweight='bold', color=COLORS['text'])
    ax3.set_ylim(0, 100)

    # Panel 4: Summary table
    ax4 = axes[1, 1]
    ax4.set_facecolor('#161b22')
    ax4.axis('off')

    summary = f"""
EXPECTED VALUE SUMMARY
{'='*50}

PORTFOLIO SETUP
  Trades: {n}
  Bet per trade: ${bet_size:.2f}
  Total invested: ${total_bet:.2f}
  Avg market price: {avg_price:.1%}

BREAKEVEN ANALYSIS
  Win profit (avg): +${avg_win_profit:.2f} per trade
  Lose profit: -${bet_size:.2f} per trade
  Breakeven: {min_wins} out of {n} trades must win
  Breakeven win rate: {min_wins/n:.0%}

IF NO EDGE (market efficient):
  Expected win rate: {avg_price:.1%}
  Expected profit: ~$0 (minus tx fees)
  Prob of profit: {probs_profit[0]:.0f}%

IF 2% EDGE (theory applies):
  Expected win rate: {min(0.99, avg_price + 0.02):.1%}
  Expected profit: ~${evs[20]:.2f}
  Prob of profit: {probs_profit[20]:.0f}%

{'='*50}
"""

    ax4.text(0.05, 0.95, summary, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace', color=COLORS['text'])

    plt.tight_layout()
    plt.savefig(output_dir / 'expected_value_analysis.png', dpi=150, facecolor='#0d1117',
                bbox_inches='tight')
    plt.close()
    print("\nCreated: expected_value_analysis.png")


if __name__ == "__main__":
    asyncio.run(main())
