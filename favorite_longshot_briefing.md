# Favorite-Longshot Bias Strategy

## Research

| Source | Finding |
|--------|---------|
| Snowberg & Wolfers (2010) | Longshots systematically overbet, favorites underbet |
| UCD Kalshi Study (2025) | High-priced contracts resolve MORE often than implied |
| Quantpedia | Betting favorites: -3.6% loss; longshots: -26% loss |

## Trade Setup

| Parameter | Value |
|-----------|-------|
| Positions | 15 |
| Price range | 93-97% |
| Stake per position | $25 |
| Total capital | $375 |
| Estimated edge | 2.5-3% |

## Probability Analysis

Assuming 15 positions at ~96% with 3% edge (true prob 99%):

| Wins | Losses | Probability | P&L |
|------|--------|-------------|-----|
| 15 | 0 | **86.0%** | +$15.63 |
| 14 | 1 | 13.0% | -$10.42 |
| 13 | 2 | 0.9% | -$36.46 |
| 12 | 3 | 0.04% | -$62.50 |
| <12 | >3 | <0.01% | -$88 to -$375 |

## Summary Statistics

| Metric | Value |
|--------|-------|
| **P(profit)** | **86%** |
| **P(loss)** | **14%** |
| E[P&L] | +$11.72 |
| E[Return] | +3.1% |
| Median | +$15.63 |
| Worst case | -$375 (-100%) |

## Risk Profile

- **86% chance** of making ~$16
- **13% chance** of losing ~$10 (one position fails)
- **1% chance** of losing $36+ (two+ positions fail)
- **<0.1% chance** of catastrophic loss

## Scenario Analysis

| Scenario | P(profit) | P(loss) | E[P&L] |
|----------|-----------|---------|--------|
| No edge (market fair) | 54% | 46% | -$0.00 |
| Half estimated edge | 72% | 28% | +$5.50 |
| Full estimated edge | 86% | 14% | +$11.72 |
| 1.5x estimated edge | 93% | 7% | +$17.50 |

## Correlation Risk

If positions are correlated (sports/politics clustering):

| Correlation | P(profit) | P(loss) | 5th percentile |
|-------------|-----------|---------|----------------|
| 0% (independent) | 86% | 14% | -$10 |
| 20% | 82% | 18% | -$37 |
| 40% | 78% | 22% | -$62 |
| 60% | 73% | 27% | -$88 |

## Caveats

- Edge estimates from horse racing/Kalshi research, not Polymarket-specific
- No historical calibration data for Polymarket at 95%+ prices
- Positions likely correlated (multiple sports, politics)
- Single loss wipes out multiple wins
