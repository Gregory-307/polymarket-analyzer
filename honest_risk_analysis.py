"""Honest risk analysis assuming market prices are accurate (0% edge)."""

import numpy as np
from itertools import combinations

# The tradeable markets with real data
all_trades = [
    {"name": "49ers Super Bowl NO", "price": 0.952, "spread": 0.0011},
    {"name": "Bears Super Bowl NO", "price": 0.945, "spread": 0.0011},
    {"name": "Texans Super Bowl NO", "price": 0.910, "spread": 0.0011},
    {"name": "49ers NFC NO", "price": 0.904, "spread": 0.0022},
    {"name": "Rob Jetten YES", "price": 0.958, "spread": 0.0031},
    {"name": "McMillan OROY YES", "price": 0.923, "spread": 0.0065},
    {"name": "Jaxson Dart OROY NO", "price": 0.960, "spread": 0.0177},
]

# DYNAMIC EXCLUSION: Exclude trades where net edge < threshold
GROSS_EDGE = 0.02  # 2% assumed edge from research
MIN_NET_EDGE = 0.005  # Minimum 0.5% net edge to be worth the risk

# Calculate net edge for each trade
for t in all_trades:
    t["net_edge"] = GROSS_EDGE - t["spread"]
    t["ex_25"] = 25 * t["net_edge"] / t["price"]  # E(X) on $25 bet

# Filter to trades with sufficient net edge
trades = [t for t in all_trades if t["net_edge"] >= MIN_NET_EDGE]
excluded = [t for t in all_trades if t["net_edge"] < MIN_NET_EDGE]

STAKE = 25  # $25 per trade
TOTAL_CAPITAL = STAKE * len(trades)

print(f"DYNAMIC FILTERING: Net edge >= {MIN_NET_EDGE:.1%} (Gross {GROSS_EDGE:.0%} - Spread)")
print(f"\nINCLUDED ({len(trades)} trades):")
for t in trades:
    print(f"  {t['name']:<25} spread {t['spread']:.2%} -> net edge {t['net_edge']:.2%} -> E(X) ${t['ex_25']:.2f}")

if excluded:
    print(f"\nEXCLUDED ({len(excluded)} trades - net edge < {MIN_NET_EDGE:.1%}):")
    for t in excluded:
        print(f"  {t['name']:<25} spread {t['spread']:.2%} -> net edge {t['net_edge']:.2%} (too thin)")

print(f"\nPortfolio: {len(trades)} trades x ${STAKE} = ${TOTAL_CAPITAL}\n")

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
n = len(trades)
print(f"Expected wins:   {expected_wins:.2f} / {n}")
print(f"Expected losses: {expected_losses:.2f} / {n}")
print(f"\nMost likely outcome: {round(expected_wins)} wins, {round(expected_losses)} loss")

print(f"\n## Aggregate Probabilities")
print(f"P(all {n} win)         = {p_all_win:.1%}")
print(f"P(at least 1 loss)   = {p_at_least_one_loss:.1%}")

# Calculate P(exactly k losses)
print("\n## Probability by Number of Losses\n")
print(f"{'Losses':>8} {'Probability':>12} {'Cumulative':>12}")
print("-" * 35)

cumulative = 0
loss_probs = []
for k in range(n + 1):
    # P(exactly k losses) using inclusion-exclusion
    prob = 0
    for loss_combo in combinations(range(n), k):
        p_this = 1.0
        for i in range(n):
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
for losses in range(n + 1):
    wins = n - losses
    prob = loss_probs[losses]

    if losses == 0:
        profit = total_win_profit
    else:
        # Each trade's profit varies by price, calculate properly
        avg_win_profit = total_win_profit / n
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
for k in range(n + 1):
    prob = 0
    for loss_combo in combinations(range(n), k):
        p_this = 1.0
        for i in range(n):
            if i in loss_combo:
                p_this *= (1 - true_probs_with_edge[i])
            else:
                p_this *= true_probs_with_edge[i]
        prob += p_this
    loss_probs_edge.append(prob)

# Calculate expected value with edge
total_expected_edge = 0
for losses in range(n + 1):
    wins = n - losses
    prob = loss_probs_edge[losses]
    if losses == 0:
        profit = total_win_profit
    else:
        avg_win_profit = total_win_profit / n
        profit = wins * avg_win_profit - losses * STAKE
    total_expected_edge += profit * prob

expected_wins_edge = sum(true_probs_with_edge)
expected_losses_edge = n - expected_wins_edge

# Correlation adjustment - 5 of 6 remaining trades are NFL-related
# Correlation increases P(multiple failures) when one fails
# Estimate: P(loss) increases by ~20% relative (37% -> 44%)
CORRELATION_FACTOR = 1.20
p_loss_correlated = min(p_at_least_one_loss * CORRELATION_FACTOR, 0.99)
p_all_win_correlated = 1 - p_loss_correlated

# Recalculate E(X) with correlation
# Assume correlation shifts probability mass from "all win" to "1+ loss" scenarios
total_expected_correlated = p_all_win_correlated * total_win_profit + (1 - p_all_win_correlated) * (-14.5)  # avg 1-loss scenario

n_trades = len(trades)
print(f"""
                           0% EDGE        2% EDGE        0% + CORRELATION
                          (Efficient)   (Bias Exists)   (Realistic)
------------------------------------------------------------------------""")
print(f"Avg win probability:     {avg_win_rate:>7.1%}        {np.mean(true_probs_with_edge):>7.1%}          {avg_win_rate:>7.1%}")
print(f"P(all {n_trades} win):            {p_all_win:>7.1%}        {p_all_win_with_edge:>7.1%}          {p_all_win_correlated:>7.1%}")
print(f"P(at least 1 loss):      {p_at_least_one_loss:>7.1%}        {1-p_all_win_with_edge:>7.1%}          {p_loss_correlated:>7.1%}")
print(f"Expected wins:           {expected_wins:>7.2f}        {expected_wins_edge:>7.2f}          {expected_wins:>7.2f}")
print(f"Expected losses:         {expected_losses:>7.2f}        {expected_losses_edge:>7.2f}          {expected_losses * CORRELATION_FACTOR:>7.2f}")
print(f"E(X) on ${TOTAL_CAPITAL}:          {total_expected:>+7.2f}       {total_expected_edge:>+7.2f}         {total_expected_correlated:>+7.2f}")
print(f"ROI:                     {total_expected/TOTAL_CAPITAL:>+7.1%}       {total_expected_edge/TOTAL_CAPITAL:>+7.1%}         {total_expected_correlated/TOTAL_CAPITAL:>+7.1%}")

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

avg_loss_one = results[1][3] if len(results) > 1 else -25  # profit from 1-loss scenario

print(f"""
KEY NUMBERS:
  P(all {n} win):        {p_all_win:.1%}
  P(at least 1 loss):  {p_at_least_one_loss:.1%}  <- This is your LOSS probability
  P(1 loss, correl.):  {p_loss_correlated:.1%}  <- More realistic with correlation

IF ALL {n} WIN:  +${total_win_profit:.2f}
IF 1 LOSES:      ${avg_loss_one:.2f}  <- ONE loss wipes gains and goes negative

E(X) SCENARIOS:
  0% edge (efficient):   ${total_expected:+.2f}
  2% edge (bias):        ${total_expected_edge:+.2f}
  0% edge + correlation: ${total_expected_correlated:+.2f}

The 2% edge from research is UNVALIDATED on Polymarket.
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

print(f"""
The probability calculations above assume INDEPENDENCE.

But 5 of {n} trades are NFL-related:
  - 49ers Super Bowl, Bears Super Bowl, Texans Super Bowl
  - 49ers NFC Championship
  - McMillan OROY

These are NOT independent. Correlated factors:
  * NFL season continuation (strike, lockout, catastrophe)
  * Playoff seeding affects multiple teams
  * Award voting influenced by same narratives

Independent P(loss): {p_at_least_one_loss:.1%}
Correlated P(loss):  {p_loss_correlated:.1%} (estimated +20% relative increase)
""")

print("=" * 70)
print("SUMMARY: REALISTIC EXPECTATIONS")
print("=" * 70)

print(f"""
                           E(X)      P(profit)    ROI
OPTIMISTIC (2% edge):    ${total_expected_edge:+6.2f}     {p_all_win_with_edge:5.1%}      {total_expected_edge/TOTAL_CAPITAL:+.1%}
REALISTIC (0% edge):     ${total_expected:+6.2f}     {p_all_win:5.1%}      {total_expected/TOTAL_CAPITAL:+.1%}
PESSIMISTIC (correl.):   ${total_expected_correlated:+6.2f}     {p_all_win_correlated:5.1%}      {total_expected_correlated/TOTAL_CAPITAL:+.1%}

With ${TOTAL_CAPITAL} at stake:
  Best case (all win):    +${total_win_profit:.2f}
  One loss:               ${avg_loss_one:.2f}
  Worst case (all lose):  -${TOTAL_CAPITAL:.2f}

Bottom line: {p_all_win:.0%}/{p_at_least_one_loss:.0%} bet for +${total_win_profit:.0f} / ${avg_loss_one:.0f}
""")
