# Arbitrage Briefing

**Strategy**: Risk-Free Arbitrage on Prediction Markets
**Risk Level**: Near-Zero (when executed correctly)
**Scan Date**: January 17, 2026

---

## Executive Summary

Arbitrage exploits a fundamental constraint: in any market where exactly one outcome must occur, the prices of all outcomes should sum to exactly $1.00. When they don't, guaranteed profit exists.

**Scan Result**: 0 arbitrage opportunities found in 100 markets.

This is expected. Polymarket markets are relatively efficient, and arbitrage windows close quickly.

---

## Scan Results

### January 17, 2026 Scan

| Metric | Value |
|--------|-------|
| Markets scanned | 100 |
| Single-arb opportunities | 0 |
| Platform | Polymarket |
| Threshold | 0.3% minimum profit |

**Interpretation**: At the time of scan, all binary markets had YES + NO prices summing to approximately $1.00. Market makers and arbitrageurs have already closed any price gaps.

---

## The Core Principle

### Why Prices Must Sum to $1

In a complete market with mutually exclusive outcomes:
- Exactly ONE outcome pays $1.00
- All others pay $0.00
- Holding one share of each outcome = guaranteed $1.00

Therefore: `price(YES) + price(NO) = $1.00`

When this equation doesn't hold, arbitrage exists:

```
Cost    = YES price + NO price
Payout  = $1.00 (exactly one wins)
Profit  = $1.00 - Cost
```

### Example (Hypothetical)

```
Market: "Will BTC exceed $150k by Dec 31?"

  YES: $0.42
  NO:  $0.55
  ─────────
  SUM: $0.97

Action: Buy 100 YES ($42) + 100 NO ($55) = $97
Payout: $100 (guaranteed)
Profit: $3 (3.1%)
```

---

## Why No Opportunities Found

1. **Market efficiency** - Polymarket has active arbitrageurs
2. **Narrow spreads** - Most markets have tight bid/ask
3. **Point-in-time snapshot** - Opportunities may exist briefly between scans
4. **API limits** - Only scanned 100 markets

### Historical Evidence

From arXiv:2508.03474 "Unravelling the Probabilistic Forest":
- **$28.3M** extracted via arbitrage on Polymarket (over time)
- Opportunities DO exist but are short-lived
- Multi-outcome markets have larger gaps than binary

---

## Strategy Variants

### Variant 1: Binary Markets (YES + NO)

Simplest case. Two outcomes must sum to $1.00.

**Current scanner**: `python run.py scan --strategy single_arb`

### Variant 2: Multi-Outcome Markets

Same principle, more outcomes. Example: "Who wins the election?" with 5+ candidates.

**Status**: Partially implemented. Scanner exists but grouped market fetching is incomplete.

---

## Execution Requirements

### For Binary Markets

```
1. Check liquidity on both YES and NO
2. Calculate profit after estimated slippage
3. Place both orders simultaneously
4. Verify fills
5. Hold until settlement
```

### Transaction Costs

| Platform | Fee | Impact |
|----------|-----|--------|
| Polymarket | 0% trading fee | Gas only (~$0.01) |
| Kalshi | ~0.1% per side | Need >0.2% gross profit |

### What Can Go Wrong

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Partial fill | Medium | Size to weakest leg |
| Price moves during execution | Medium | Execute fast |
| Settlement dispute | Very Low | Avoid ambiguous markets |

---

## Recommendation

### Current Assessment

Arbitrage on Polymarket is:
- **Theoretically sound** - Math guarantees profit when sum < $1
- **Practically rare** - Markets are efficient, windows close fast
- **Low capital efficiency** - Funds locked until settlement

### Action Items

1. Run scans more frequently to catch opportunities
2. Implement multi-outcome arbitrage scanning
3. Consider automated execution for speed

### When to Use

Best as a **low-risk supplement** to other strategies, not a primary approach.

---

## Reproducibility

### Run the Scan

```bash
python run.py scan --strategy single_arb
```

### Raw Data

- Scan results: `research/arbitrage/data/scan_20260117_012310.json`
- Findings: `research/arbitrage/findings/scan_20260117.md`

---

*Generated: January 17, 2026*
*Based on: Real scan data (0 opportunities found)*
