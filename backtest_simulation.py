"""Historical Backtest Simulation - Strategy Performance Analysis.

Simulates favorite-longshot bias strategy performance based on:
- Research-documented bias patterns (NBER WP 15923)
- Conservative edge estimates
- Real-world transaction costs

NOTE: This uses the live market data as a snapshot for simulation.
Actual historical backtesting would require archived price data.
"""

import asyncio
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
import random
import math

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np

from src.core.config import Credentials
from src.adapters.polymarket import PolymarketAdapter


# Visualization settings
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
}


def simulate_resolution(market_price: float, bias_adjustment: float = 0.03) -> bool:
    """Simulate market resolution with favorite-longshot bias.

    Research shows high-probability outcomes resolve MORE often than priced.
    E.g., 95% priced markets resolve YES ~98% of the time.
    """
    # Apply bias adjustment (higher prob = higher actual resolution)
    if market_price >= 0.95:
        actual_prob = min(0.99, market_price + bias_adjustment)
    elif market_price >= 0.90:
        actual_prob = min(0.98, market_price + bias_adjustment * 0.67)
    elif market_price >= 0.80:
        actual_prob = min(0.95, market_price + bias_adjustment * 0.33)
    else:
        actual_prob = market_price

    return random.random() < actual_prob


def run_backtest(
    markets,
    initial_capital: float = 10000,
    position_size_pct: float = 0.05,
    min_probability: float = 0.90,
    transaction_cost: float = 0.002,  # 0.2%
    num_simulations: int = 100,
) -> dict:
    """Run Monte Carlo backtest simulation.

    Returns performance statistics across multiple simulations.
    """
    # Filter to high-probability markets
    eligible_markets = [
        m for m in markets
        if max(m.yes_price, m.no_price) >= min_probability
        and (m.liquidity or 0) >= 1000
    ]

    if not eligible_markets:
        return {"error": "No eligible markets"}

    all_results = []

    for sim in range(num_simulations):
        capital = initial_capital
        equity_curve = [capital]
        trades = []
        wins = 0
        losses = 0

        for market in eligible_markets:
            # Determine trade
            yes_price = market.yes_price
            no_price = market.no_price

            if yes_price >= min_probability:
                side = "YES"
                price = yes_price
            elif no_price >= min_probability:
                side = "NO"
                price = no_price
            else:
                continue

            # Position sizing
            position = capital * position_size_pct
            shares = position / price

            # Transaction cost
            cost = position * transaction_cost

            # Simulate resolution
            resolved_yes = simulate_resolution(yes_price)
            won = (side == "YES" and resolved_yes) or (side == "NO" and not resolved_yes)

            # Calculate P&L
            if won:
                payout = shares  # $1 per share
                pnl = payout - position - cost
                wins += 1
            else:
                pnl = -position - cost
                losses += 1

            capital += pnl
            equity_curve.append(capital)

            trades.append({
                'market': market.question[:50],
                'side': side,
                'price': price,
                'position': position,
                'won': won,
                'pnl': pnl,
            })

        # Calculate metrics
        total_return = (capital - initial_capital) / initial_capital
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0

        # Max drawdown
        peak = initial_capital
        max_dd = 0
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        # Sharpe ratio (simplified)
        pnls = [t['pnl'] for t in trades]
        if len(pnls) > 1:
            mean_pnl = np.mean(pnls)
            std_pnl = np.std(pnls) if np.std(pnls) > 0 else 1
            sharpe = (mean_pnl / std_pnl) * math.sqrt(252)
        else:
            sharpe = 0

        all_results.append({
            'final_capital': capital,
            'total_return': total_return,
            'win_rate': win_rate,
            'max_drawdown': max_dd,
            'sharpe_ratio': sharpe,
            'num_trades': len(trades),
            'equity_curve': equity_curve,
        })

    # Aggregate statistics
    returns = [r['total_return'] for r in all_results]
    win_rates = [r['win_rate'] for r in all_results]
    drawdowns = [r['max_drawdown'] for r in all_results]
    sharpes = [r['sharpe_ratio'] for r in all_results]

    return {
        'num_simulations': num_simulations,
        'num_markets': len(eligible_markets),
        'initial_capital': initial_capital,
        'statistics': {
            'avg_return': np.mean(returns),
            'median_return': np.median(returns),
            'std_return': np.std(returns),
            'min_return': np.min(returns),
            'max_return': np.max(returns),
            'pct_profitable': len([r for r in returns if r > 0]) / len(returns),
            'avg_win_rate': np.mean(win_rates),
            'avg_max_drawdown': np.mean(drawdowns),
            'avg_sharpe': np.mean(sharpes),
        },
        'percentiles': {
            '5th': np.percentile(returns, 5),
            '25th': np.percentile(returns, 25),
            '50th': np.percentile(returns, 50),
            '75th': np.percentile(returns, 75),
            '95th': np.percentile(returns, 95),
        },
        'all_results': all_results,
    }


def create_backtest_visualizations(results: dict, output_dir: Path):
    """Create backtest result visualizations."""
    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor(COLORS['bg'])

    # Main title
    fig.suptitle('FAVORITE-LONGSHOT BIAS STRATEGY BACKTEST',
                 fontsize=24, fontweight='bold', color=COLORS['primary'], y=0.98)
    fig.text(0.5, 0.94, f'Monte Carlo Simulation | {results["num_simulations"]} Runs | {results["num_markets"]} Markets',
             ha='center', fontsize=12, color=COLORS['text'])

    # Grid
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # 1. Return distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(COLORS['card'])
    returns = [r['total_return'] * 100 for r in results['all_results']]
    ax1.hist(returns, bins=30, color=COLORS['success'], edgecolor='white', alpha=0.8)
    ax1.axvline(x=0, color=COLORS['danger'], linestyle='--', linewidth=2, label='Break-even')
    ax1.axvline(x=np.mean(returns), color=COLORS['warning'], linestyle='-', linewidth=2,
                label=f'Mean: {np.mean(returns):.1f}%')
    ax1.set_xlabel('Return (%)', color=COLORS['text'])
    ax1.set_ylabel('Frequency', color=COLORS['text'])
    ax1.set_title('Return Distribution', fontsize=14, fontweight='bold', color=COLORS['text'])
    ax1.tick_params(colors=COLORS['text'])
    ax1.legend(facecolor=COLORS['card'], labelcolor=COLORS['text'])

    # 2. Equity curves (sample)
    ax2 = fig.add_subplot(gs[0, 1:])
    ax2.set_facecolor(COLORS['card'])
    # Plot sample equity curves
    sample_indices = random.sample(range(len(results['all_results'])), min(20, len(results['all_results'])))
    for idx in sample_indices:
        curve = results['all_results'][idx]['equity_curve']
        color = COLORS['success'] if curve[-1] > results['initial_capital'] else COLORS['danger']
        ax2.plot(curve, alpha=0.3, color=color, linewidth=1)
    # Mean equity curve
    max_len = max(len(r['equity_curve']) for r in results['all_results'])
    padded_curves = []
    for r in results['all_results']:
        curve = r['equity_curve']
        padded = curve + [curve[-1]] * (max_len - len(curve))
        padded_curves.append(padded)
    mean_curve = np.mean(padded_curves, axis=0)
    ax2.plot(mean_curve, color=COLORS['warning'], linewidth=3, label='Mean Equity')
    ax2.axhline(y=results['initial_capital'], color=COLORS['text'], linestyle='--', alpha=0.5)
    ax2.set_xlabel('Trade Number', color=COLORS['text'])
    ax2.set_ylabel('Portfolio Value ($)', color=COLORS['text'])
    ax2.set_title('Equity Curves (20 Sample Runs)', fontsize=14, fontweight='bold', color=COLORS['text'])
    ax2.tick_params(colors=COLORS['text'])
    ax2.legend(facecolor=COLORS['card'], labelcolor=COLORS['text'])

    # 3. Win rate distribution
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(COLORS['card'])
    win_rates = [r['win_rate'] * 100 for r in results['all_results']]
    ax3.hist(win_rates, bins=20, color=COLORS['primary'], edgecolor='white', alpha=0.8)
    ax3.axvline(x=np.mean(win_rates), color=COLORS['warning'], linestyle='-', linewidth=2,
                label=f'Mean: {np.mean(win_rates):.1f}%')
    ax3.set_xlabel('Win Rate (%)', color=COLORS['text'])
    ax3.set_ylabel('Frequency', color=COLORS['text'])
    ax3.set_title('Win Rate Distribution', fontsize=14, fontweight='bold', color=COLORS['text'])
    ax3.tick_params(colors=COLORS['text'])
    ax3.legend(facecolor=COLORS['card'], labelcolor=COLORS['text'])

    # 4. Drawdown distribution
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(COLORS['card'])
    drawdowns = [r['max_drawdown'] * 100 for r in results['all_results']]
    ax4.hist(drawdowns, bins=20, color=COLORS['danger'], edgecolor='white', alpha=0.8)
    ax4.axvline(x=np.mean(drawdowns), color=COLORS['warning'], linestyle='-', linewidth=2,
                label=f'Mean: {np.mean(drawdowns):.1f}%')
    ax4.set_xlabel('Max Drawdown (%)', color=COLORS['text'])
    ax4.set_ylabel('Frequency', color=COLORS['text'])
    ax4.set_title('Drawdown Distribution', fontsize=14, fontweight='bold', color=COLORS['text'])
    ax4.tick_params(colors=COLORS['text'])
    ax4.legend(facecolor=COLORS['card'], labelcolor=COLORS['text'])

    # 5. Sharpe ratio distribution
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor(COLORS['card'])
    sharpes = [r['sharpe_ratio'] for r in results['all_results']]
    ax5.hist(sharpes, bins=20, color=COLORS['purple'], edgecolor='white', alpha=0.8)
    ax5.axvline(x=np.mean(sharpes), color=COLORS['warning'], linestyle='-', linewidth=2,
                label=f'Mean: {np.mean(sharpes):.2f}')
    ax5.axvline(x=1.0, color=COLORS['success'], linestyle='--', linewidth=2, label='Target: 1.0')
    ax5.set_xlabel('Sharpe Ratio', color=COLORS['text'])
    ax5.set_ylabel('Frequency', color=COLORS['text'])
    ax5.set_title('Sharpe Ratio Distribution', fontsize=14, fontweight='bold', color=COLORS['text'])
    ax5.tick_params(colors=COLORS['text'])
    ax5.legend(facecolor=COLORS['card'], labelcolor=COLORS['text'])

    # 6. Summary statistics table
    ax6 = fig.add_subplot(gs[2, :])
    ax6.set_facecolor(COLORS['card'])
    ax6.axis('off')

    stats = results['statistics']
    pcts = results['percentiles']

    summary_text = f"""
BACKTEST SUMMARY STATISTICS
{'=' * 80}

RETURN METRICS                              RISK METRICS
--------------                              ------------
Average Return:     {stats['avg_return']*100:>8.2f}%              Average Max Drawdown: {stats['avg_max_drawdown']*100:>8.2f}%
Median Return:      {stats['median_return']*100:>8.2f}%              Average Sharpe Ratio: {stats['avg_sharpe']:>8.2f}
Std Deviation:      {stats['std_return']*100:>8.2f}%              Probability Profitable: {stats['pct_profitable']*100:>6.1f}%
Best Return:        {stats['max_return']*100:>8.2f}%
Worst Return:       {stats['min_return']*100:>8.2f}%              Average Win Rate: {stats['avg_win_rate']*100:>6.1f}%

RETURN PERCENTILES
------------------
5th Percentile:     {pcts['5th']*100:>8.2f}%      (Worst-case scenario)
25th Percentile:    {pcts['25th']*100:>8.2f}%
50th Percentile:    {pcts['50th']*100:>8.2f}%      (Median outcome)
75th Percentile:    {pcts['75th']*100:>8.2f}%
95th Percentile:    {pcts['95th']*100:>8.2f}%      (Best-case scenario)

STRATEGY PARAMETERS
-------------------
Initial Capital:    ${results['initial_capital']:>10,.2f}
Position Size:      5% of portfolio
Min Probability:    90%+
Transaction Cost:   0.2%
Markets Analyzed:   {results['num_markets']}

{'=' * 80}
Based on research: Kahneman & Tversky (1979), NBER WP 15923
"""

    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace', color=COLORS['text'])

    plt.savefig(output_dir / 'backtest_results.png', dpi=150, facecolor=COLORS['bg'],
                bbox_inches='tight')
    plt.close()
    print(f"Created: backtest_results.png")


async def main():
    """Run backtest simulation."""
    print("=" * 70)
    print("FAVORITE-LONGSHOT BIAS STRATEGY BACKTEST")
    print("Monte Carlo Simulation")
    print("=" * 70)
    print()

    output_dir = Path(__file__).parent / "results" / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fetch market data
    print("Fetching live market data...")
    creds = Credentials.from_env()
    adapter = PolymarketAdapter(credentials=creds)
    await adapter.connect()
    markets = await adapter.get_markets(active_only=True, limit=200)
    await adapter.disconnect()
    print(f"Fetched {len(markets)} markets")

    # Run backtest
    print("\nRunning Monte Carlo simulation (100 runs)...")
    results = run_backtest(
        markets,
        initial_capital=10000,
        position_size_pct=0.05,
        min_probability=0.90,
        num_simulations=100,
    )

    # Print results
    stats = results['statistics']
    pcts = results['percentiles']

    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)

    print(f"\nSimulations: {results['num_simulations']}")
    print(f"Markets: {results['num_markets']}")
    print(f"Initial Capital: ${results['initial_capital']:,.2f}")

    print(f"\nRETURN STATISTICS:")
    print(f"  Average Return:  {stats['avg_return']*100:>7.2f}%")
    print(f"  Median Return:   {stats['median_return']*100:>7.2f}%")
    print(f"  Std Deviation:   {stats['std_return']*100:>7.2f}%")
    print(f"  Best Return:     {stats['max_return']*100:>7.2f}%")
    print(f"  Worst Return:    {stats['min_return']*100:>7.2f}%")

    print(f"\nRISK METRICS:")
    print(f"  Prob. Profitable: {stats['pct_profitable']*100:>6.1f}%")
    print(f"  Average Win Rate: {stats['avg_win_rate']*100:>6.1f}%")
    print(f"  Avg Max Drawdown: {stats['avg_max_drawdown']*100:>6.1f}%")
    print(f"  Avg Sharpe Ratio: {stats['avg_sharpe']:>6.2f}")

    print(f"\nRETURN PERCENTILES:")
    print(f"  5th percentile:  {pcts['5th']*100:>7.2f}%  (worst case)")
    print(f"  25th percentile: {pcts['25th']*100:>7.2f}%")
    print(f"  50th percentile: {pcts['50th']*100:>7.2f}%  (median)")
    print(f"  75th percentile: {pcts['75th']*100:>7.2f}%")
    print(f"  95th percentile: {pcts['95th']*100:>7.2f}%  (best case)")

    # Create visualizations
    print("\nGenerating visualizations...")
    create_backtest_visualizations(results, output_dir)

    # Save results
    results_file = output_dir.parent / "backtests" / f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)

    # Remove equity curves for JSON (too large)
    save_results = {k: v for k, v in results.items() if k != 'all_results'}
    save_results['sample_results'] = [
        {k: v for k, v in r.items() if k != 'equity_curve'}
        for r in results['all_results'][:10]
    ]

    with open(results_file, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"Results saved to: {results_file}")

    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
