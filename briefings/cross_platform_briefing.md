# Cross-Platform Arbitrage Briefing

**Strategy**: Exploit Price Discrepancies Between Prediction Market Platforms
**Risk Level**: Low (not zero - settlement risk exists)
**Scan Date**: January 17, 2026

---

## Executive Summary

Cross-platform arbitrage exploits price differences when the same event is traded on multiple platforms. Buy on the cheap platform, sell on the expensive one.

**Scan Result**: 0 cross-platform opportunities found.

**Root Cause**: Kalshi's open markets are 100% sports parlays. Their political markets (which would overlap with Polymarket) are already finalized from the 2024 election cycle.

---

## Scan Results

### January 17, 2026 Scan

| Metric | Value |
|--------|-------|
| Polymarket markets | 100 (politics, crypto) |
| Kalshi markets | 100 (after filtering 5000+ sports) |
| Matched markets | 0 |
| Opportunities found | 0 |

### Why No Matches Found

**Verified Finding**: Kalshi has NO non-sports open markets.

After paginating through thousands of Kalshi markets:
- **100% of open markets** are sports parlays (KXMVE*, KXNFL*, etc.)
- **Political markets** (PRES, SENATE, etc.) are status="finalized"
- **No overlap** exists with Polymarket's political/crypto markets

### Platform Comparison

| Category | Polymarket | Kalshi |
|----------|------------|--------|
| Politics | 130 active markets | 0 open (all finalized) |
| Crypto | 12 active markets | 0 open |
| Sports | ~50 markets | 5000+ open (all parlays) |

### What This Means

Cross-platform arbitrage between Polymarket and Kalshi is **not currently viable** because:
1. Kalshi pivoted heavily to sports betting
2. Their 2024 election markets are resolved
3. No new political markets are open for trading

---

## The Core Principle

When the same event trades on both platforms at different prices:

```
Event: "Fed cuts rates in March 2026"

  Polymarket YES: $0.35
  Kalshi YES:     $0.42

Action:
  Buy 100 YES on Polymarket @ $0.35 = $35
  Sell 100 YES on Kalshi @ $0.42 = $42

Profit: $7 regardless of outcome (if both settle the same)
```

### Critical Risk: Settlement Differences

Unlike sum arbitrage, cross-platform is NOT risk-free. Platforms may:
- Use different resolution criteria
- Settle at different times
- Interpret ambiguous outcomes differently

---

## Implementation Status

| Component | Status |
|-----------|--------|
| Polymarket adapter | Working |
| Kalshi adapter | Working |
| Market matching algorithm | Implemented |
| Cross-platform scanner | Implemented |
| Market filtering | Needs improvement |

### The Market Filtering Problem

```python
# Current: Returns sports parlays
kalshi.get_markets(status='open')  # 100% sports

# Needed: Filter by category or series
kalshi.get_markets(series_ticker='PRES')  # Political
kalshi.get_markets(exclude_prefix='KXMVE')  # No sports
```

---

## Market Overlap Analysis

### Kalshi Categories (7,988 series total)
- Elections
- Politics
- Economics
- Companies
- Sports (dominates "open" markets)
- Entertainment
- Climate

### Polymarket Focus
- US Politics
- Crypto prices
- Sports (some)
- Pop culture

### Overlap Potential

| Category | Polymarket | Kalshi | Match Potential |
|----------|------------|--------|-----------------|
| US Politics | High | Medium | Good |
| Fed/Rates | Medium | Medium | Good |
| Crypto | High | Low | Limited |
| Sports | Medium | High | Possible |

---

## How to Find Opportunities

### Recommended Approach

1. **Filter Kalshi markets**
   ```python
   # Get political events
   events = kalshi.get_events(category='Politics')
   markets = [m for e in events for m in e.markets]
   ```

2. **Match by question similarity**
   - Normalize questions (remove dates, punctuation)
   - Use fuzzy matching (SequenceMatcher)
   - Require >70% confidence

3. **Compare prices**
   - Spread > 2% is actionable
   - Account for fees (~0.2% on Kalshi)

4. **Verify resolution criteria**
   - Read both platforms' rules
   - Confirm identical settlement conditions

---

## Execution Considerations

### Capital Requirements

- Need funded accounts on BOTH platforms
- Polymarket: USDC on Polygon
- Kalshi: USD (US bank, requires residency)

### Timing

```
Polymarket: 24/7 trading
Kalshi: May have trading hours for some markets

Execute the cheaper side first (less likely to move against you)
```

---

## Risk Analysis

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Settlement mismatch | Low | High (lose both sides) | Read resolution rules |
| Execution gap | Medium | Low-Medium | Execute quickly |
| Platform risk | Very Low | High | Don't over-concentrate |

---

## Conclusion

Cross-platform arbitrage is **theoretically viable** but **practically limited** by:

1. Market category differences between platforms
2. API default behavior returning non-overlapping markets
3. Need for better market filtering

### Next Steps

1. Improve Kalshi market filtering to exclude sports parlays
2. Target specific political/financial series
3. Run scans at different times to catch different markets
4. Build historical dataset of matched markets

---

## Reproducibility

### Run the Scan

```bash
python run.py scan --strategy cross_platform
```

### Raw Data

- Scan results: `research/cross_platform/data/scan_20260117_013427.json`
- Findings: `research/cross_platform/findings/scan_20260117.md`

---

*Generated: January 17, 2026*
*Based on: Real scan data (0 matches due to market category mismatch)*
