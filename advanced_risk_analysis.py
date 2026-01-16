"""Advanced risk analysis with proper correlation model and dynamic filtering."""

import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt

# Market data with real spreads from order books
ALL_TRADES = [
    {"name": "49ers Super Bowl NO", "price": 0.952, "spread": 0.0011, "category": "NFL"},
    {"name": "Bears Super Bowl NO", "price": 0.945, "spread": 0.0011, "category": "NFL"},
    {"name": "Texans Super Bowl NO", "price": 0.910, "spread": 0.0011, "category": "NFL"},
    {"name": "49ers NFC NO", "price": 0.904, "spread": 0.0022, "category": "NFL"},
    {"name": "Rob Jetten YES", "price": 0.958, "spread": 0.0031, "category": "Politics"},
    {"name": "McMillan OROY YES", "price": 0.923, "spread": 0.0065, "category": "NFL"},
    {"name": "Jaxson Dart OROY NO", "price": 0.960, "spread": 0.0177, "category": "NFL"},
]

STAKE = 25
GROSS_EDGE = 0.02  # 2% from research (unvalidated)


def calculate_trade_metrics(trades, gross_edge=GROSS_EDGE):
    """Calculate per-trade metrics."""
    for t in trades:
        t["net_edge"] = gross_edge - t["spread"]
        t["profit_win"] = STAKE * (1 - t["price"]) / t["price"]
        t["loss_lose"] = STAKE
        t["ex_with_edge"] = STAKE * t["net_edge"] / t["price"]
        t["ex_no_edge"] = -STAKE * (t["spread"] / 2) / t["price"]
        t["edge_spread_ratio"] = t["net_edge"] / t["spread"] if t["spread"] > 0 else float('inf')
    return trades


def analyze_portfolio(trades, correlation_factor=1.0, assume_edge=True):
    """
    Analyze portfolio with proper correlation adjustment.

    Correlation INCREASES loss probability and DECREASES E(X).
    """
    n = len(trades)
    if n == 0:
        return None

    probs = [t["price"] for t in trades]
    capital = STAKE * n
    profit_all_win = sum(t["profit_win"] for t in trades)

    # Calculate independent loss probabilities
    loss_probs_indep = []
    for k in range(n + 1):
        prob = 0
        for combo in combinations(range(n), k):
            p = 1.0
            for i in range(n):
                if i in combo:
                    p *= (1 - probs[i])
                else:
                    p *= probs[i]
            prob += p
        loss_probs_indep.append(prob)

    p_all_win_indep = loss_probs_indep[0]
    p_loss_indep = 1 - p_all_win_indep

    # Apply correlation adjustment
    # Correlation increases P(loss) and shifts mass to multi-loss scenarios
    if correlation_factor > 1.0:
        p_loss_corr = min(p_loss_indep * correlation_factor, 0.99)
        p_all_win_corr = 1 - p_loss_corr

        # Redistribute loss scenarios - correlation clusters failures
        loss_probs_corr = [p_all_win_corr]

        # With correlation, multi-loss scenarios become more likely
        # Single loss slightly less likely (failures cluster)
        remaining = p_loss_corr
        weights = []
        for k in range(1, n + 1):
            # Increase weight for multi-loss scenarios
            w = loss_probs_indep[k] * (1 + 0.15 * (k - 1))
            weights.append(w)

        # Normalize
        total_w = sum(weights)
        for k in range(1, n + 1):
            loss_probs_corr.append(weights[k-1] / total_w * remaining)

        loss_probs = loss_probs_corr
        p_all_win = p_all_win_corr
        p_loss = p_loss_corr
    else:
        loss_probs = loss_probs_indep
        p_all_win = p_all_win_indep
        p_loss = p_loss_indep

    # Calculate payoffs and E(X) for each scenario
    scenarios = []
    for losses in range(n + 1):
        wins = n - losses
        prob = loss_probs[losses]

        if losses == 0:
            profit = profit_all_win
        else:
            # Lose full stake on losers, proportional profit on winners
            avg_win_profit = profit_all_win / n
            profit = wins * avg_win_profit - losses * STAKE

        scenarios.append({
            "wins": wins,
            "losses": losses,
            "prob": prob,
            "profit": profit,
            "contribution": prob * profit
        })

    # Calculate E(X)
    ex = sum(s["contribution"] for s in scenarios)

    # If no edge assumed, E(X) is just negative spread cost
    if not assume_edge:
        ex = sum(t["ex_no_edge"] for t in trades)

    # Risk metrics
    # Expected loss given loss occurs
    loss_scenarios = [s for s in scenarios if s["losses"] > 0]
    total_loss_prob = sum(s["prob"] for s in loss_scenarios)
    if total_loss_prob > 0:
        expected_loss_given_loss = sum(s["profit"] * s["prob"] for s in loss_scenarios) / total_loss_prob
    else:
        expected_loss_given_loss = 0

    # VaR calculations (sorted by profit, worst first)
    sorted_scenarios = sorted(scenarios, key=lambda x: x["profit"])
    cumul = 0
    var_95, var_99 = -capital, -capital
    for s in sorted_scenarios:
        cumul += s["prob"]
        if cumul >= 0.01 and var_99 == -capital:
            var_99 = s["profit"]
        if cumul >= 0.05 and var_95 == -capital:
            var_95 = s["profit"]

    return {
        "n": n,
        "capital": capital,
        "p_win": p_all_win,
        "p_loss": p_loss,
        "ex": ex,
        "roi": ex / capital if capital > 0 else 0,
        "profit_all_win": profit_all_win,
        "max_loss": -capital,
        "var_95": var_95,
        "var_99": var_99,
        "expected_loss_given_loss": expected_loss_given_loss,
        "sharpe_like": ex / abs(expected_loss_given_loss) if expected_loss_given_loss != 0 else 0,
        "scenarios": scenarios,
        "loss_probs_indep": loss_probs_indep,
    }


def print_analysis():
    """Print comprehensive analysis."""

    # Calculate metrics for all trades
    trades = calculate_trade_metrics(ALL_TRADES.copy())

    print("=" * 85)
    print("ADVANCED RISK ANALYSIS - Polymarket Favorite-Longshot Strategy")
    print("=" * 85)

    # Per-trade analysis
    print("\n## PER-TRADE RISK/REWARD\n")
    print(f"{'Trade':<25} {'Price':>6} {'Spread':>7} {'NetEdge':>8} {'Win$':>7} {'E(X)':>7} {'Edge/Sprd':>10}")
    print("-" * 85)

    for t in sorted(trades, key=lambda x: -x["net_edge"]):
        status = "INCLUDE" if t["net_edge"] >= 0.005 else "THIN" if t["net_edge"] > 0 else "EXCLUDE"
        print(f"{t['name']:<25} {t['price']:>5.1%} {t['spread']:>6.2%} {t['net_edge']:>7.2%} "
              f"{t['profit_win']:>+6.2f} {t['ex_with_edge']:>+6.2f} {t['edge_spread_ratio']:>9.1f}x  [{status}]")

    print("-" * 85)
    print("Edge/Spread > 2x recommended for margin of safety\n")

    # Filter to included trades
    included = [t for t in trades if t["net_edge"] >= 0.005]
    excluded = [t for t in trades if t["net_edge"] < 0.005]

    print(f"INCLUDED: {len(included)} trades | EXCLUDED: {len(excluded)} trades")
    if excluded:
        print(f"  Excluded: {', '.join(t['name'] for t in excluded)}")

    # Portfolio comparison at different correlation levels
    print("\n" + "=" * 85)
    print("CORRELATION IMPACT ON RISK")
    print("=" * 85)
    print(f"\n5 of {len(included)} trades are NFL-related - correlation is real\n")

    print(f"{'Correlation':>12} {'P(win)':>8} {'P(loss)':>8} {'E(X)':>9} {'ROI':>8} {'E(L|L)':>9} {'Sharpe':>8}")
    print("-" * 70)

    for corr in [1.0, 1.1, 1.2, 1.3]:
        r = analyze_portfolio(included, correlation_factor=corr)
        label = "Independent" if corr == 1.0 else f"+{(corr-1)*100:.0f}% corr"
        print(f"{label:>12} {r['p_win']:>7.1%} {r['p_loss']:>7.1%} "
              f"{r['ex']:>+8.2f} {r['roi']:>+7.1%} {r['expected_loss_given_loss']:>+8.2f} {r['sharpe_like']:>+7.3f}")

    print("-" * 70)
    print("E(L|L) = Expected loss given that at least one trade loses")
    print("Sharpe = E(X) / |E(L|L)| - higher is better risk-adjusted\n")

    # Scenario breakdown with correlation
    print("=" * 85)
    print("SCENARIO BREAKDOWN (20% Correlation)")
    print("=" * 85)

    r_corr = analyze_portfolio(included, correlation_factor=1.2)
    r_indep = analyze_portfolio(included, correlation_factor=1.0)

    print(f"\n{'Wins':>5} {'Loss':>5} {'P(indep)':>9} {'P(corr)':>9} {'Profit':>10} {'E(X)_corr':>10}")
    print("-" * 55)

    for i, s in enumerate(r_corr["scenarios"]):
        p_indep = r_indep["scenarios"][i]["prob"]
        print(f"{s['wins']:>5} {s['losses']:>5} {p_indep:>8.1%} {s['prob']:>8.1%} "
              f"{s['profit']:>+9.2f} {s['contribution']:>+9.2f}")

    print("-" * 55)
    print(f"{'TOTAL E(X):':<32} {r_corr['ex']:>+9.2f}")

    # Final summary table
    print("\n" + "=" * 85)
    print("FINAL RISK SUMMARY")
    print("=" * 85)

    # Calculate all scenarios
    r_edge_indep = analyze_portfolio(included, correlation_factor=1.0, assume_edge=True)
    r_edge_corr = analyze_portfolio(included, correlation_factor=1.2, assume_edge=True)
    r_no_edge_corr = analyze_portfolio(included, correlation_factor=1.2, assume_edge=False)

    n = len(included)
    capital = STAKE * n
    profit_win = sum(t["profit_win"] for t in included)
    one_loss_profit = r_edge_corr["scenarios"][1]["profit"]

    print(f"""
PORTFOLIO: {n} trades x ${STAKE} = ${capital}

PROBABILITY (with 20% correlation for NFL clustering):
  P(all {n} win)      = {r_edge_corr['p_win']:.1%}
  P(at least 1 loss) = {r_edge_corr['p_loss']:.1%}  <- YOUR LOSS RISK

PAYOFF STRUCTURE:
  All {n} win:    +${profit_win:.2f}
  1 loses:       ${one_loss_profit:.2f}  <- ONE loss wipes gains
  2 lose:        ${r_edge_corr['scenarios'][2]['profit']:.2f}
  Max loss:      ${r_edge_corr['max_loss']:.2f}

EXPECTED VALUE:
                            E(X)       ROI      P(profit)
  2% edge, independent:   {r_edge_indep['ex']:>+7.2f}    {r_edge_indep['roi']:>+5.1%}     {r_edge_indep['p_win']:.1%}
  2% edge, correlated:    {r_edge_corr['ex']:>+7.2f}    {r_edge_corr['roi']:>+5.1%}     {r_edge_corr['p_win']:.1%}
  0% edge, correlated:    {r_no_edge_corr['ex']:>+7.2f}    {r_no_edge_corr['roi']:>+5.1%}     {r_edge_corr['p_win']:.1%}

RISK METRICS (with correlation):
  VaR 95%:              ${r_edge_corr['var_95']:.2f}
  VaR 99%:              ${r_edge_corr['var_99']:.2f}
  E(Loss | Loss):       ${r_edge_corr['expected_loss_given_loss']:.2f}
  Risk-Adjusted Ratio:  {r_edge_corr['sharpe_like']:.3f}

BOTTOM LINE:
  {r_edge_corr['p_win']:.0%} / {r_edge_corr['p_loss']:.0%} bet
  Win: +${profit_win:.0f}  |  Lose: ${one_loss_profit:.0f} to ${r_edge_corr['max_loss']:.0f}
""")

    return included, r_edge_corr, r_edge_indep, r_no_edge_corr


def create_visualizations(trades, r_corr, r_indep):
    """Create 4-panel risk visualization."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle("ADVANCED RISK ANALYSIS", fontsize=14, fontweight='bold', color='white')

    for ax in axes.flat:
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('#404040')

    # 1. Net Edge by Trade
    ax1 = axes[0, 0]
    names = [t["name"].replace(" NO", "").replace(" YES", "")[:18] for t in ALL_TRADES]
    net_edges = [t["net_edge"] * 100 for t in calculate_trade_metrics(ALL_TRADES.copy())]
    colors = ['#2ecc71' if e >= 0.5 else '#e74c3c' for e in net_edges]

    bars = ax1.barh(names, net_edges, color=colors)
    ax1.axvline(x=0.5, color='#f39c12', linestyle='--', linewidth=2, label='0.5% threshold')
    ax1.axvline(x=2.0, color='#95a5a6', linestyle=':', linewidth=1, label='2% gross edge')
    ax1.set_xlabel('Net Edge (%)')
    ax1.set_title('Net Edge by Trade (Green = Included)')
    ax1.legend(loc='lower right', facecolor='#16213e', labelcolor='white')

    # 2. Correlation Impact
    ax2 = axes[0, 1]
    corr_factors = [1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3]
    p_losses = []
    exs = []

    for cf in corr_factors:
        r = analyze_portfolio(trades, correlation_factor=cf)
        p_losses.append(r['p_loss'] * 100)
        exs.append(r['ex'])

    color1 = '#e74c3c'
    ax2.plot(corr_factors, p_losses, 'o-', color=color1, linewidth=2, markersize=8, label='P(Loss)')
    ax2.set_xlabel('Correlation Factor')
    ax2.set_ylabel('P(at least 1 loss) %', color=color1)
    ax2.tick_params(axis='y', labelcolor=color1)
    ax2.axhline(y=50, color='#95a5a6', linestyle=':', alpha=0.5)

    ax2b = ax2.twinx()
    color2 = '#2ecc71'
    ax2b.plot(corr_factors, exs, 's-', color=color2, linewidth=2, markersize=8, label='E(X)')
    ax2b.set_ylabel('E(X) $', color=color2)
    ax2b.tick_params(axis='y', labelcolor=color2)
    ax2b.spines['right'].set_color(color2)
    ax2b.spines['left'].set_color(color1)

    ax2.set_title('Correlation Impact: P(Loss) Up, E(X) Down')

    # 3. Outcome Distribution (Correlated)
    ax3 = axes[1, 0]
    n = len(trades)
    labels = [f"{s['wins']}W/{s['losses']}L" for s in r_corr["scenarios"]]
    probs = [s["prob"] * 100 for s in r_corr["scenarios"]]
    profits = [s["profit"] for s in r_corr["scenarios"]]
    colors = ['#2ecc71' if p > 0 else '#e74c3c' for p in profits]

    bars = ax3.bar(labels, probs, color=colors)
    ax3.set_xlabel('Outcome')
    ax3.set_ylabel('Probability (%)')
    ax3.set_title(f'Outcome Distribution (20% Corr)\nP(Win)={r_corr["p_win"]:.1%}, P(Loss)={r_corr["p_loss"]:.1%}')

    # Add profit labels
    for bar, profit in zip(bars, profits):
        height = bar.get_height()
        if height > 1:  # Only label visible bars
            ax3.annotate(f'${profit:+.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, color='white')

    # 4. E(X) Sensitivity to Edge Assumption
    ax4 = axes[1, 1]
    edge_assumptions = np.linspace(0, 0.04, 20)

    ex_indep = []
    ex_corr = []

    for edge in edge_assumptions:
        # Recalculate with this edge assumption
        temp_trades = calculate_trade_metrics([t.copy() for t in trades], gross_edge=edge)
        temp_trades = [t for t in temp_trades if t["net_edge"] >= 0]  # Only positive edge

        if temp_trades:
            r_i = analyze_portfolio(temp_trades, correlation_factor=1.0)
            r_c = analyze_portfolio(temp_trades, correlation_factor=1.2)
            ex_indep.append(r_i['ex'])
            ex_corr.append(r_c['ex'])
        else:
            ex_indep.append(sum(t["ex_no_edge"] for t in trades))
            ex_corr.append(sum(t["ex_no_edge"] for t in trades))

    ax4.plot(edge_assumptions * 100, ex_indep, '-', color='#3498db', linewidth=2, label='Independent')
    ax4.plot(edge_assumptions * 100, ex_corr, '--', color='#e74c3c', linewidth=2, label='20% Correlated')
    ax4.axhline(y=0, color='white', linestyle='-', linewidth=0.5)
    ax4.axvline(x=2.0, color='#f39c12', linestyle=':', linewidth=2, label='2% (research)')
    ax4.fill_between(edge_assumptions * 100, ex_corr, alpha=0.2, color='#e74c3c')
    ax4.set_xlabel('Assumed Gross Edge (%)')
    ax4.set_ylabel('Portfolio E(X) ($)')
    ax4.set_title('E(X) Sensitivity to Edge Assumption')
    ax4.legend(facecolor='#16213e', labelcolor='white')

    plt.tight_layout()
    plt.savefig('results/visualizations/advanced_risk_analysis.png', dpi=150,
                bbox_inches='tight', facecolor='#1a1a2e', edgecolor='none')
    plt.close()
    print("\nSaved: results/visualizations/advanced_risk_analysis.png")


if __name__ == "__main__":
    trades, r_corr, r_indep, r_no_edge = print_analysis()

    plt.style.use('dark_background')
    create_visualizations(trades, r_corr, r_indep)
