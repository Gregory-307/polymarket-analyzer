# POLYMARKET INVESTMENT BRIEFING
## January 16, 2026

---

# EXECUTIVE SUMMARY

| Metric | IF 2% Edge | IF 0% Edge |
|--------|------------|------------|
| E(X) | **+$3.02** | **-$0.20** |
| P(all 6 win) | 74.3% | 65.4% |
| P(at least 1 loss) | 25.7% | 34.6% |

**Portfolio:** 6 trades × $25 = $150

**Payoffs:**
- All 6 win: **+$11.02**
- 5 win, 1 lose: **-$15.81**
- Max loss: **-$150**

**Key Question:** Does the 2% favorite-longshot bias exist on Polymarket? This is UNVALIDATED.

---

# METHODOLOGY

## Formulas Used

```
profit_if_win = stake × (1 - price) / price
spread_cost = stake × spread / 2 / price
E(X) = true_prob × profit_if_win + (1 - true_prob) × (-stake) - spread_cost
P(all win) = product of individual true_probabilities
```

## Edge Assumptions

| Scenario | true_probability | Interpretation |
|----------|------------------|----------------|
| 2% edge | market_price + 0.02 | Favorite-longshot bias exists |
| 0% edge | market_price | Markets are efficient |

## Example Calculation (49ers Super Bowl NO)

```
price = 95.2%
spread = 0.11%
stake = $25

profit_if_win = $25 × (1 - 0.952) / 0.952 = $1.26
spread_cost = $25 × 0.0011 / 2 / 0.952 = $0.01

IF 2% EDGE:
  true_prob = 0.952 + 0.02 = 0.972
  E(X) = 0.972 × $1.26 + 0.028 × (-$25) - $0.01
       = $1.22 - $0.70 - $0.01
       = +$0.51

IF 0% EDGE:
  true_prob = 0.952
  E(X) = 0.952 × $1.26 + 0.048 × (-$25) - $0.01
       = $1.20 - $1.20 - $0.01
       = -$0.01
```

---

# MARKET DATA

## 6 Selected Trades (from live order books)

| # | Market | Side | Price | Spread | Profit if Win |
|---|--------|------|-------|--------|---------------|
| 1 | 49ers Super Bowl | NO | 95.2% | 0.11% | $1.26 |
| 2 | Bears Super Bowl | NO | 94.5% | 0.11% | $1.46 |
| 3 | Texans Super Bowl | NO | 91.0% | 0.11% | $2.47 |
| 4 | 49ers NFC | NO | 90.4% | 0.22% | $2.65 |
| 5 | Rob Jetten NL PM | YES | 95.8% | 0.31% | $1.10 |
| 6 | McMillan OROY | YES | 92.3% | 0.65% | $2.09 |

**Total profit if all win:** $11.02

## Per-Trade E(X)

| Trade | E(X) if 2% Edge | E(X) if 0% Edge |
|-------|-----------------|-----------------|
| 49ers Super Bowl | +$0.51 | -$0.01 |
| Bears Super Bowl | +$0.51 | -$0.01 |
| Texans Super Bowl | +$0.53 | -$0.02 |
| 49ers NFC | +$0.52 | -$0.03 |
| Rob Jetten | +$0.48 | -$0.04 |
| McMillan OROY | +$0.45 | -$0.09 |
| **TOTAL** | **+$3.02** | **-$0.20** |

---

# SCENARIO ANALYSIS

## IF 2% Edge Exists

| Outcome | Probability | Profit |
|---------|-------------|--------|
| 6W/0L | **74.3%** | +$11.02 |
| 5W/1L | 22.7% | -$15.81 |
| 4W/2L | 2.8% | -$42.65 |
| 3W/3L | 0.2% | -$69.49 |
| Worse | <0.1% | -$96 to -$150 |

**E(X) = +$3.02**

## IF 0% Edge (Markets Efficient)

| Outcome | Probability | Profit |
|---------|-------------|--------|
| 6W/0L | **65.4%** | +$11.02 |
| 5W/1L | 28.9% | -$15.81 |
| 4W/2L | 5.2% | -$42.65 |
| 3W/3L | 0.5% | -$69.49 |
| Worse | <0.1% | -$96 to -$150 |

**E(X) = -$0.20** (spread cost)

---

# CORRELATION ANALYSIS

## Key Insight: E(X) Does NOT Change with Correlation

This is a mathematical fact: E[X + Y] = E[X] + E[Y], regardless of correlation.

Correlation affects:
- **Variance** (higher correlation = more volatile outcomes)
- **Joint probabilities** (P(all win), P(all lose))
- **Outcome distribution** (more extreme results)

Correlation does **NOT** affect:
- Expected value (E(X) stays constant)

## Methodology: Gaussian Copula Simulation

1. Generate correlated standard normal variables
2. Transform to uniform via CDF (copula step)
3. Convert to binary outcomes by comparing to win probabilities
4. Run 100,000 Monte Carlo simulations

## Results (2% Edge)

| Correlation | P(all win) | P(all lose) | Std Dev | E(X) |
|-------------|------------|-------------|---------|------|
| 0% | 74.4% | 0.000% | $14.14 | **+$3.02** |
| 10% | 75.6% | 0.000% | $14.90 | **+$3.02** |
| 20% | 76.9% | 0.002% | $15.84 | **+$3.02** |
| 30% | 78.3% | 0.006% | $17.04 | **+$3.02** |

**Observation:** E(X) = +$3.02 is CONSTANT regardless of correlation.

## Results (0% Edge)

| Correlation | P(all win) | P(all lose) | Std Dev | E(X) |
|-------------|------------|-------------|---------|------|
| 0% | 65.4% | 0.000% | $16.62 | **-$0.20** |
| 10% | 67.3% | 0.000% | $17.78 | **-$0.20** |
| 20% | 69.5% | 0.002% | $19.12 | **-$0.20** |
| 30% | 71.5% | 0.019% | $20.67 | **-$0.20** |

## What This Means

For this portfolio of high-probability bets (all >90% implied):

1. **Positive correlation HELPS** - P(all win) increases from 74% to 78%
2. **But risk also increases** - P(all lose) goes from 0% to 0.006%
3. **Volatility increases** - Std dev goes from $14 to $17
4. **E(X) is unchanged** - Your expected profit doesn't depend on correlation

## Correlation Sources

Our trades have potential correlation because:
- 4 NFL markets may move together (common league factors)
- Dutch politics market is independent of NFL
- NFL awards market has partial correlation with game outcomes

Estimated portfolio correlation: ~10-20% for NFL-heavy portion

---

# THEORETICAL FOUNDATION

## The Favorite-Longshot Bias

**Kahneman & Tversky (1979)** - Prospect Theory
> People systematically overweight small probabilities and underweight near-certainties.

**Snowberg & Wolfers (2010)** - NBER Working Paper 15923

| Implied Prob | Actual Win Rate | Edge |
|--------------|-----------------|------|
| 90.9% | 93.2% | +2.3% |
| 95.2% | 97.1% | +1.9% |
| 98.0% | 98.9% | +0.9% |

**Caveat:** This research is from horse racing and sports betting. It has NOT been validated on Polymarket.

---

# RISK ASSESSMENT

## Key Risk: You Must Win ALL 6

- Win all 6: +$11.02 profit
- Lose just 1: -$15.81 loss (wipes out gains)
- The strategy is binary: profit only if perfect

## Risk Metrics

| Metric | Value |
|--------|-------|
| Capital at risk | $150 |
| Max loss | -$150 (100%) |
| P(loss) if 2% edge | 25.7% |
| P(loss) if 0% edge | 34.6% |

---

# RECOMMENDATION

## Execute These 6 Trades

| Market | Side | Price | Amount |
|--------|------|-------|--------|
| 49ers Super Bowl | NO | 95.2% | $25 |
| Bears Super Bowl | NO | 94.5% | $25 |
| Texans Super Bowl | NO | 91.0% | $25 |
| 49ers NFC | NO | 90.4% | $25 |
| Rob Jetten NL PM | YES | 95.8% | $25 |
| McMillan OROY | YES | 92.3% | $25 |

**Total: $150**

## Do Not Trade

| Market | Reason |
|--------|--------|
| Jaxson Dart OROY | Spread (1.77%) exceeds net edge |
| McCaffrey Comeback | Spread > edge |
| Protector of Year | Spreads 8-18% |

---

# APPENDIX

## Data Sources

- **Prices/Spreads:** Polymarket CLOB API (live order books, January 16, 2026)
- **Research:** Kahneman & Tversky (1979), Snowberg & Wolfers (2010)

## Code

All calculations can be reproduced by running:
```
python advanced_risk_analysis.py
```

## Outputs

- `results/visualizations/advanced_risk_analysis.png` - Main risk analysis charts
- `results/visualizations/correlation_analysis.png` - Correlation effects visualization

---

*Generated: January 16, 2026*
*Status: 2% edge is UNVALIDATED on Polymarket*
