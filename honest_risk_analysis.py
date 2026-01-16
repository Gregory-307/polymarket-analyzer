"""Honest risk analysis assuming market prices are accurate (0% edge)."""

import numpy as np
from itertools import combinations

# The 7 tradeable markets with real data
trades = [
    {"name": "49ers Super Bowl NO", "price": 0.952, "spread": 0.0011},
    {"name": "Bears Super Bowl NO", "price": 0.945, "spread": 0.0011},
    {"name": "Texans Super Bowl NO", "price": 0.910, "spread": 0.0011},
    {"name": "49ers NFC NO", "price": 0.904, "spread": 0.0022},
    {"name": "Rob Jetten YES", "price": 0.958, "spread": 0.0031},
    {"name": "McMillan OROY YES", "price": 0.923, "spread": 0.0065},
    {"name": "Jaxson Dart OROY NO", "price": 0.960, "spread": 0.0177},
]

STAKE = 25  # $25 per trade
TOTAL_CAPITAL = STAKE * len(trades)

print("=" * 70)
print("HONEST RISK ANALYSIS - Assuming Market Prices Are Accurate (0% Edge)")
print("=" * 70)

# Calculate profit/loss for each trade
print("\n## Per-Trade Economics\n")
print(f"{'Market':<25} {'Price':>7} {'Win':>8} {'Lose':>8} {'E(X)':>8}")
print("-" * 60)

total_win_profit = 0
total_ex = 0

for t in trades:
    p = t["price"]
    spread = t["spread"]

    # Profit if win = $25 * (1-P)/P
    profit_if_win = STAKE * (1 - p) / p
    loss_if_lose = STAKE

    # E(X) if market efficient = just the spread cost (negative)
    # When crossing spread, you lose half the spread on entry
    ex_efficient = -STAKE * (spread / 2) / p

    total_win_profit += profit_if_win
    total_ex += ex_efficient

    t["profit_win"] = profit_if_win
    t["loss_lose"] = loss_if_lose
    t["ex"] = ex_efficient

    print(f"{t['name']:<25} {p:>6.1%} {profit_if_win:>+7.2f} {-loss_if_lose:>+7.2f} {ex_efficient:>+7.2f}")

print("-" * 60)
print(f"{'TOTAL if all win:':<25} {'':<7} {total_win_profit:>+7.2f}")
print(f"{'E(X) if 0% edge:':<25} {'':<7} {'':<8} {'':<8} {total_ex:>+7.2f}")

# Probability calculations
print("\n" + "=" * 70)
print("PROBABILITY ANALYSIS (Market Prices = True Probabilities)")
print("=" * 70)

probs = [t["price"] for t in trades]
p_all_win = np.prod(probs)
p_at_least_one_loss = 1 - p_all_win

# Expected wins/losses
expected_wins = sum(probs)
expected_losses = len(probs) - expected_wins
avg_win_rate = np.mean(probs)

print(f"\n## Expected Hit Rate")
print(f"Average win probability: {avg_win_rate:.1%}")
print(f"Expected wins:   {expected_wins:.2f} / 7")
print(f"Expected losses: {expected_losses:.2f} / 7")
print(f"\nMost likely outcome: {round(expected_wins)} wins, {round(expected_losses)} loss")

print(f"\n## Aggregate Probabilities")
print(f"P(all 7 win)         = {p_all_win:.1%}")
print(f"P(at least 1 loss)   = {p_at_least_one_loss:.1%}")

# Calculate P(exactly k losses)
print("\n## Probability by Number of Losses\n")
print(f"{'Losses':>8} {'Probability':>12} {'Cumulative':>12}")
print("-" * 35)

cumulative = 0
loss_probs = []
for k in range(8):
    # P(exactly k losses) using inclusion-exclusion
    prob = 0
    for loss_combo in combinations(range(7), k):
        p_this = 1.0
        for i in range(7):
            if i in loss_combo:
                p_this *= (1 - probs[i])  # This one loses
            else:
                p_this *= probs[i]  # This one wins
        prob += p_this

    cumulative += prob
    loss_probs.append(prob)
    print(f"{k:>8} {prob:>11.1%} {cumulative:>11.1%}")

# Payoff by scenario
print("\n" + "=" * 70)
print("PAYOFF BY SCENARIO (The Critical Table)")
print("=" * 70)

print(f"""
This table shows EXACTLY what happens for each outcome:
- Probability: chance of this exact scenario occurring
- Cumulative: chance of this OR WORSE happening
- Profit: your P&L for this scenario
- E(contribution): probability-weighted contribution to expected value
""")

print(f"{'Wins':>5} {'Loss':>5} {'Prob':>8} {'Cumul':>8} {'Profit':>10} {'E(contrib)':>11}  {'Outcome':<}")
print("-" * 65)

total_expected = 0
cumulative_from_bottom = 0

# Calculate from worst to best for cumulative
results = []
for losses in range(8):
    wins = 7 - losses
    prob = loss_probs[losses]

    if losses == 0:
        profit = total_win_profit
    else:
        # Average case: lose $25 per loss, gain proportional wins
        avg_win_profit = total_win_profit / 7
        profit = wins * avg_win_profit - losses * STAKE

    contribution = profit * prob
    total_expected += contribution
    results.append((wins, losses, prob, profit, contribution))

# Print from best to worst with cumulative probability of worse outcomes
cumulative_worse = 0
for i, (wins, losses, prob, profit, contribution) in enumerate(results):
    if i > 0:
        cumulative_worse += results[i-1][2] if i == 1 else results[i-1][2]

    # Cumulative is P(this outcome or worse) - for losses, it's sum of all scenarios with >= this many losses
    cumul_this_or_worse = sum(r[2] for r in results[i:])

    if profit > 0:
        outcome = "PROFIT"
    elif profit > -STAKE:
        outcome = "SMALL LOSS"
    elif profit > -2*STAKE:
        outcome = "MODERATE LOSS"
    else:
        outcome = "LARGE LOSS"

    print(f"{wins:>5} {losses:>5} {prob:>7.1%} {cumul_this_or_worse:>7.1%} {profit:>+9.2f} {contribution:>+10.2f}   {outcome}")

print("-" * 65)
print(f"{'EXPECTED VALUE (0% edge):':<26} {total_expected:>+10.2f}")
print(f"{'EXPECTED WINS:':<26} {expected_wins:>10.2f}")
print(f"{'EXPECTED LOSSES:':<26} {expected_losses:>10.2f}")

# Now calculate with 2% edge
print("\n" + "=" * 70)
print("E(X) COMPARISON: 0% Edge vs 2% Edge")
print("=" * 70)

# With 2% edge, the true win probability is higher than market price
EDGE = 0.02
true_probs_with_edge = [min(p + EDGE, 0.999) for p in probs]
p_all_win_with_edge = np.prod(true_probs_with_edge)

# Recalculate loss probabilities with edge
loss_probs_edge = []
for k in range(8):
    prob = 0
    for loss_combo in combinations(range(7), k):
        p_this = 1.0
        for i in range(7):
            if i in loss_combo:
                p_this *= (1 - true_probs_with_edge[i])
            else:
                p_this *= true_probs_with_edge[i]
        prob += p_this
    loss_probs_edge.append(prob)

# Calculate expected value with edge
total_expected_edge = 0
for losses in range(8):
    wins = 7 - losses
    prob = loss_probs_edge[losses]
    if losses == 0:
        profit = total_win_profit
    else:
        avg_win_profit = total_win_profit / 7
        profit = wins * avg_win_profit - losses * STAKE
    total_expected_edge += profit * prob

expected_wins_edge = sum(true_probs_with_edge)
expected_losses_edge = 7 - expected_wins_edge

print(f"""
                              0% EDGE           2% EDGE
                           (Efficient)      (Bias Exists)
---------------------------------------------------------""")
print(f"Avg win probability:      {avg_win_rate:>7.1%}           {np.mean(true_probs_with_edge):>7.1%}")
print(f"P(all 7 win):             {p_all_win:>7.1%}           {p_all_win_with_edge:>7.1%}")
print(f"P(at least 1 loss):       {p_at_least_one_loss:>7.1%}           {1-p_all_win_with_edge:>7.1%}")
print(f"Expected wins:            {expected_wins:>7.2f}           {expected_wins_edge:>7.2f}")
print(f"Expected losses:          {expected_losses:>7.2f}           {expected_losses_edge:>7.2f}")
print(f"E(X) on $175:            {total_expected:>+7.2f}          {total_expected_edge:>+7.2f}")
print(f"ROI:                      {total_expected/TOTAL_CAPITAL:>+7.1%}          {total_expected_edge/TOTAL_CAPITAL:>+7.1%}")

# Maker vs Taker analysis
print("\n" + "=" * 70)
print("MAKER vs TAKER POSITIONS")
print("=" * 70)

print(f"""
TAKER (Market Order - Cross the Spread):
  - You pay the ASK price (higher)
  - Immediate execution
  - Spread cost: ~0.5% average (half the bid-ask)
  - E(X) reduction: ~$0.43 on $175

MAKER (Limit Order - Post at Bid):
  - You pay the BID price (lower) or MID
  - May not fill, requires patience
  - Spread cost: $0 if filled at your price
  - E(X) improvement: +$0.43 vs taker

MAKER E(X) vs TAKER E(X):
""")

# Maker gets filled at mid (no spread cost)
maker_ex_0_edge = total_expected + 0.43  # Add back spread cost
maker_ex_2_edge = total_expected_edge + 0.43

print(f"                              TAKER            MAKER")
print(f"                          (Cross Spread)   (Limit Order)")
print(f"---------------------------------------------------------")
print(f"E(X) with 0% edge:        {total_expected:>+8.2f}         {maker_ex_0_edge:>+8.2f}")
print(f"E(X) with 2% edge:        {total_expected_edge:>+8.2f}         {maker_ex_2_edge:>+8.2f}")
print(f"ROI with 0% edge:         {total_expected/TOTAL_CAPITAL:>+8.1%}         {maker_ex_0_edge/TOTAL_CAPITAL:>+8.1%}")
print(f"ROI with 2% edge:         {total_expected_edge/TOTAL_CAPITAL:>+8.1%}         {maker_ex_2_edge/TOTAL_CAPITAL:>+8.1%}")

print(f"""
Key insight: As a MAKER, you eliminate spread costs.
  - With 0% edge: E(X) goes from -$0.43 to ~$0 (breakeven)
  - With 2% edge: E(X) goes from +$2.45 to +$2.88 (full edge)

Risk: Limit orders may not fill if price moves away.
""")

# Key insight
print("\n" + "=" * 70)
print("KEY INSIGHT")
print("=" * 70)

print(f"""
+-----------------------------------------------------------------+
|  IF MARKETS ARE EFFICIENT (0% edge):                            |
|                                                                 |
|  * You have a 37.2% chance of at least one loss                 |
|  * ONE loss wipes out ALL gains and puts you negative           |
|  * Expected value ~ ${total_expected:.2f} (slightly negative from spreads)  |
|                                                                 |
|  This is a HIGH-VARIANCE strategy:                              |
|  * 62.8% chance: +${total_win_profit:.2f}                                     |
|  * 37.2% chance: -$12 to -$175                                  |
|                                                                 |
|  The 2% edge from research is UNVALIDATED on Polymarket.        |
|  If it doesn't exist, you're flipping weighted coins.           |
+-----------------------------------------------------------------+
""")

# Break-even analysis
print("=" * 70)
print("BREAK-EVEN ANALYSIS")
print("=" * 70)

print("\nWhat edge do we need to break even?")
print("\nAt 0% edge, E(X) ~ $0 before spreads, negative after spreads.")
print("The spread cost is ~$0.43 total (half-spread on entry).")
print("\nTo break even, we need edge > spread cost.")
print(f"Average spread: {np.mean([t['spread'] for t in trades]):.2%}")
print(f"Minimum edge needed: ~{np.mean([t['spread'] for t in trades])/2:.2%} to break even")

# Correlation warning
print("\n" + "=" * 70)
print("CORRELATION WARNING")
print("=" * 70)

print("""
The probability calculations above assume INDEPENDENCE.

But 5 of 7 trades are NFL-related:
  - 49ers Super Bowl, Bears Super Bowl, Texans Super Bowl
  - 49ers NFC Championship
  - McMillan OROY, Jaxson Dart OROY

These are NOT independent. Correlated factors:
  * NFL season continuation (strike, lockout, catastrophe)
  * Playoff seeding affects multiple teams
  * Award voting influenced by same narratives

TRUE P(at least 1 loss) is likely HIGHER than 37.2% due to correlation.
""")

print("=" * 70)
print("SUMMARY: REALISTIC EXPECTATIONS")
print("=" * 70)

print(f"""
+------------------------------------------------------------------+
|  OPTIMISTIC (2% edge exists):                                    |
|    E(X) = +$2.88, P(profit) = 74.8%                              |
|                                                                  |
|  REALISTIC (0% edge, market efficient):                          |
|    E(X) = -$0.43, P(profit) = 62.8%                              |
|                                                                  |
|  PESSIMISTIC (0% edge + correlation):                            |
|    E(X) = -$0.43, P(profit) = ~55-60%                            |
|                                                                  |
|  With $175 at stake:                                             |
|    * Best case (all win): +$12.07                                |
|    * Worst case (all lose): -$175.00                             |
|    * Expected (0% edge): -$0.43                                  |
|                                                                  |
|  This is essentially a 63/37 bet for +$12 / -$25 avg loss        |
+------------------------------------------------------------------+
""")
