"""
Risk Analysis for Polymarket Favorite-Longshot Strategy

METHODOLOGY:
============

1. EDGE ASSUMPTION
   - 0% edge: true_probability = market_price (markets are efficient)
   - 2% edge: true_probability = market_price + 0.02 (based on academic research)

2. E(X) CALCULATION (per trade)
   E(X) = true_prob × profit_if_win + (1 - true_prob) × (-stake) - spread_cost

   Where:
     profit_if_win = stake × (1 - price) / price
     spread_cost = stake × spread / 2 / price  (half spread as execution cost)

3. PORTFOLIO E(X)
   E(X)_total = sum of per-trade E(X)

4. P(ALL WIN)
   P(all win) = product of individual true_probabilities
   (assumes independence between trades)

5. SCENARIO ANALYSIS
   Shows probability and profit for each possible outcome (6W/0L, 5W/1L, etc.)
"""

import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from pathlib import Path

Path("results/visualizations").mkdir(parents=True, exist_ok=True)

# Market data with real spreads from order books (January 16, 2026)
TRADES = [
    {"name": "49ers Super Bowl NO", "price": 0.952, "spread": 0.0011},
    {"name": "Bears Super Bowl NO", "price": 0.945, "spread": 0.0011},
    {"name": "Texans Super Bowl NO", "price": 0.910, "spread": 0.0011},
    {"name": "49ers NFC NO", "price": 0.904, "spread": 0.0022},
    {"name": "Rob Jetten YES", "price": 0.958, "spread": 0.0031},
    {"name": "McMillan OROY YES", "price": 0.923, "spread": 0.0065},
]

STAKE = 25  # dollars per trade


def calculate_per_trade_metrics(trades, edge=0.02):
    """
    Calculate metrics for each trade.

    Formula:
      profit_if_win = stake × (1 - price) / price
      spread_cost = stake × spread / 2 / price
      true_prob = price + edge (capped at 0.999)
      E(X) = true_prob × profit_if_win + (1 - true_prob) × (-stake) - spread_cost
    """
    results = []
    for t in trades:
        price = t["price"]
        spread = t["spread"]

        profit_if_win = STAKE * (1 - price) / price
        spread_cost = STAKE * spread / 2 / price
        true_prob = min(price + edge, 0.999)

        ex = true_prob * profit_if_win + (1 - true_prob) * (-STAKE) - spread_cost

        results.append({
            "name": t["name"],
            "price": price,
            "spread": spread,
            "profit_if_win": profit_if_win,
            "spread_cost": spread_cost,
            "true_prob": true_prob,
            "ex": ex,
            "net_edge": edge - spread,
        })

    return results


def calculate_scenario_probabilities(trades, edge=0.02):
    """
    Calculate probability of each outcome scenario.

    Assumes independence between trades.
    P(k losses) = sum over all combinations of k trades losing
    """
    n = len(trades)
    true_probs = [min(t["price"] + edge, 0.999) for t in trades]

    scenarios = []
    for k in range(n + 1):  # k = number of losses
        # Calculate probability of exactly k losses
        prob = 0
        for combo in combinations(range(n), k):
            p = 1.0
            for i in range(n):
                if i in combo:
                    p *= (1 - true_probs[i])  # this trade loses
                else:
                    p *= true_probs[i]  # this trade wins
            prob += p

        wins = n - k
        losses = k

        # Calculate profit for this scenario
        total_profit_if_all_win = sum(STAKE * (1 - t["price"]) / t["price"] for t in trades)
        avg_profit_per_win = total_profit_if_all_win / n
        profit = wins * avg_profit_per_win - losses * STAKE

        scenarios.append({
            "wins": wins,
            "losses": losses,
            "prob": prob,
            "profit": profit,
        })

    return scenarios


def run_analysis():
    """Run complete analysis and print results."""

    print("=" * 80)
    print("POLYMARKET RISK ANALYSIS")
    print("=" * 80)

    # Analyze for both edge scenarios
    for edge, label in [(0.02, "2% EDGE (if bias exists)"), (0.0, "0% EDGE (if markets efficient)")]:
        print(f"\n{'='*80}")
        print(f"SCENARIO: {label}")
        print("=" * 80)

        metrics = calculate_per_trade_metrics(TRADES, edge=edge)
        scenarios = calculate_scenario_probabilities(TRADES, edge=edge)

        # Per-trade breakdown
        print(f"\nPER-TRADE ANALYSIS (edge = {edge:.0%})")
        print("-" * 80)
        print(f"{'Trade':<25} {'Price':>7} {'Spread':>7} {'TrueProb':>8} {'Win$':>8} {'E(X)':>8}")
        print("-" * 80)

        total_ex = 0
        p_all_win = 1.0

        for m in metrics:
            print(f"{m['name']:<25} {m['price']:>6.1%} {m['spread']:>6.2%} "
                  f"{m['true_prob']:>7.1%} {m['profit_if_win']:>+7.2f} {m['ex']:>+7.2f}")
            total_ex += m['ex']
            p_all_win *= m['true_prob']

        print("-" * 80)
        print(f"{'TOTAL':<25} {'':<7} {'':<7} {'':<8} {'':<8} {total_ex:>+7.2f}")
        print(f"\nP(all 6 win) = {p_all_win:.1%}")
        print(f"E(X) = ${total_ex:+.2f}")

        # Scenario breakdown
        print(f"\nSCENARIO PROBABILITIES")
        print("-" * 50)
        print(f"{'Outcome':<12} {'Probability':>12} {'Profit':>12}")
        print("-" * 50)

        for s in scenarios:
            print(f"{s['wins']}W/{s['losses']}L{'':<6} {s['prob']:>11.1%} {s['profit']:>+11.2f}")

        print("-" * 50)

    # Summary comparison
    metrics_2 = calculate_per_trade_metrics(TRADES, edge=0.02)
    metrics_0 = calculate_per_trade_metrics(TRADES, edge=0.0)

    ex_2 = sum(m['ex'] for m in metrics_2)
    ex_0 = sum(m['ex'] for m in metrics_0)
    p_win_2 = np.prod([m['true_prob'] for m in metrics_2])
    p_win_0 = np.prod([m['true_prob'] for m in metrics_0])

    profit_all_win = sum(STAKE * (1 - t["price"]) / t["price"] for t in TRADES)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"""
PORTFOLIO: 6 trades × ${STAKE} = ${STAKE * 6}

                        IF 2% EDGE      IF 0% EDGE
                        ----------      ----------
E(X):                   ${ex_2:>+6.2f}         ${ex_0:>+6.2f}
P(all 6 win):           {p_win_2:>6.1%}          {p_win_0:>6.1%}
P(at least 1 loss):     {1-p_win_2:>6.1%}          {1-p_win_0:>6.1%}

PAYOFFS:
  All 6 win:            +${profit_all_win:.2f}
  5 win, 1 lose:        -${STAKE - profit_all_win/6:.2f}
  Max loss (all lose):  -${STAKE * 6:.2f}

FORMULAS USED:
  profit_if_win = stake × (1 - price) / price
  spread_cost = stake × spread / 2 / price
  E(X) = true_prob × profit_if_win + (1 - true_prob) × (-stake) - spread_cost
  P(all win) = product of true_probabilities
""")

    return metrics_2, metrics_0, ex_2, ex_0, p_win_2, p_win_0


def create_visualization(metrics_2, metrics_0, ex_2, ex_0, p_win_2, p_win_0):
    """Create simple, clear visualization."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Polymarket Risk Analysis", fontsize=14, fontweight='bold')

    # 1. E(X) comparison
    ax1 = axes[0, 0]
    scenarios = ['2% Edge\n(bias exists)', '0% Edge\n(efficient)']
    ex_values = [ex_2, ex_0]
    colors = ['green' if v > 0 else 'red' for v in ex_values]
    bars = ax1.bar(scenarios, ex_values, color=colors, edgecolor='black')
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.set_ylabel('E(X) ($)')
    ax1.set_title('Expected Value by Scenario')
    for bar, val in zip(bars, ex_values):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.1, f'${val:+.2f}',
                ha='center', va='bottom', fontweight='bold')

    # 2. P(all win) comparison
    ax2 = axes[0, 1]
    p_values = [p_win_2 * 100, p_win_0 * 100]
    bars = ax2.bar(scenarios, p_values, color=['green', 'orange'], edgecolor='black')
    ax2.set_ylabel('P(all 6 win) %')
    ax2.set_title('Probability of Profit')
    ax2.set_ylim(0, 100)
    for bar, val in zip(bars, p_values):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val:.1f}%',
                ha='center', va='bottom', fontweight='bold')

    # 3. Per-trade E(X) (2% edge)
    ax3 = axes[1, 0]
    names = [m['name'].replace(' NO', '').replace(' YES', '')[:15] for m in metrics_2]
    ex_per_trade = [m['ex'] for m in metrics_2]
    ax3.barh(names, ex_per_trade, color='green', edgecolor='black')
    ax3.axvline(x=0, color='black', linewidth=0.5)
    ax3.set_xlabel('E(X) ($)')
    ax3.set_title('E(X) per Trade (2% Edge)')

    # 4. Scenario probabilities (2% edge)
    ax4 = axes[1, 1]
    scenarios_2 = calculate_scenario_probabilities(TRADES, edge=0.02)
    outcomes = [f"{s['wins']}W/{s['losses']}L" for s in scenarios_2]
    probs = [s['prob'] * 100 for s in scenarios_2]
    profits = [s['profit'] for s in scenarios_2]
    colors = ['green' if p > 0 else 'red' for p in profits]
    bars = ax4.bar(outcomes, probs, color=colors, edgecolor='black')
    ax4.set_ylabel('Probability (%)')
    ax4.set_title('Outcome Distribution (2% Edge)')

    plt.tight_layout()
    plt.savefig('results/visualizations/advanced_risk_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: results/visualizations/advanced_risk_analysis.png")


if __name__ == "__main__":
    metrics_2, metrics_0, ex_2, ex_0, p_win_2, p_win_0 = run_analysis()
    create_visualization(metrics_2, metrics_0, ex_2, ex_0, p_win_2, p_win_0)
