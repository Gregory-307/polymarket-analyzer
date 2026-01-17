# Financial Markets Briefing

**Strategy**: Compare Prediction Markets to Options-Implied Probabilities
**Risk Level**: Medium (can be hedged)
**Scan Date**: January 17, 2026

---

## Executive Summary

Financial prediction markets (crypto prices, Fed rates) are fundamentally different from political markets because **fair value is calculable**. Options and futures price the same events, allowing comparison.

**Scan Result**: 1 price threshold market found, 1 opportunity identified.

---

## Scan Results

### January 17, 2026 Scan

| Metric | Value |
|--------|-------|
| Markets scanned | 500 |
| Price threshold markets | 1 |
| Opportunities | 1 |

### Market Found

```
"Will bitcoin hit $1m before GTA VI?"

  BTC spot (live): $95,448 (from CoinGecko)
  Threshold: $1,000,000 (10.5x current price)
  Market price: 48.50%
  Fair value (Black-Scholes): ~0%
  Edge: -48.5% (market overpriced)
  Implied vol used: 55%
```

### Analysis

The market prices BTC hitting $1M at 48.5%. Black-Scholes says it's essentially 0%.

**Why the massive discrepancy?**

1. **Unknown expiry** - "Before GTA VI" is not a fixed date
2. **Long time horizon** - Market may be pricing in decades
3. **Fat tails** - BTC has extreme moves not captured by lognormal
4. **Speculation** - 48.5% may reflect hope, not probability

---

## The Core Principle

### Why Financial Markets Are Different

**Political market**: "Will Trump win?"
- No mathematical fair value
- Pure opinion aggregation
- Cannot hedge

**Financial market**: "Will BTC exceed $100k by March?"
- Options price this exactly (digital call)
- Black-Scholes gives probability
- Can hedge with BTC or options

### Fair Value Calculation

For a binary "above X by date T" market:

```
P(S > K) = N(d2)

where:
  d2 = [ln(S/K) + (r - σ²/2)T] / (σ√T)

  S = current price (spot)
  K = strike (threshold)
  r = risk-free rate
  σ = implied volatility
  T = time to expiry
  N() = cumulative normal distribution
```

If prediction market price ≠ N(d2), there's a discrepancy.

---

## Implementation Status

| Component | Status |
|-----------|--------|
| Market parsing | Implemented |
| Black-Scholes calculator | Implemented |
| Spot price feed | **Live from CoinGecko** |
| Implied vol feed | Using typical values (55-80%) |
| Deribit integration | Not implemented |

### What's Working

1. **Pattern recognition** - Identifies price threshold markets ($Xk, $Xm formats)
2. **Fair value math** - Black-Scholes digital call calculation
3. **Edge detection** - Compares market to fair value
4. **Live spot prices** - BTC, ETH, SOL from CoinGecko API

### What's Missing

1. **Live implied volatility** - Using typical values, not real options data
2. **More market patterns** - Only 1 of 500 markets matched (need more patterns)
3. **Fixed-expiry markets** - "Before GTA VI" can't be accurately valued

---

## Caveats

### The GTA VI Market Problem

The found market ("BTC hit $1m before GTA VI") illustrates key limitations:

1. **Unknown expiry** - "Before GTA VI" is not a fixed date
2. **Black-Scholes needs T** - Can't calculate without time to expiry
3. **Fat tails** - BTC may not follow lognormal distribution

### When Fair Value Breaks Down

| Condition | Black-Scholes Assumption | Reality |
|-----------|-------------------------|---------|
| Distribution | Lognormal | Crypto has fat tails |
| Volatility | Constant | Vol changes over time |
| Expiry | Known | Some markets have fuzzy dates |
| Interest rates | Constant | Fed can change rates |

---

## Strategy Variants

### Variant A: Pure Mispricing

Find where prediction market diverges from options-implied probability.

```
Example:
  BTC $100k call option implies: 38% probability
  Polymarket "BTC > $100k": 45% probability

  Edge: 7% (45% - 38%)
  Action: Sell Polymarket YES if options more efficient
```

### Variant B: Hedged Arbitrage

Lock in profit by hedging.

```
  Sell YES on Polymarket (receive $0.45)
  Buy call option to hedge ($0.38 equivalent)

  Net: $0.07 profit locked in, regardless of BTC price
```

### Variant C: Information Edge

Use superior financial analysis to price better than the market.

---

## Data Sources Needed

### For Crypto

| Source | What It Provides | Integration |
|--------|------------------|-------------|
| Deribit | BTC/ETH options, implied vol | Not implemented |
| Binance | Spot prices | Easy to add |
| CME | Regulated BTC futures | API available |

### For Rates / Macro

| Source | What It Provides | Integration |
|--------|------------------|-------------|
| CME FedWatch | Fed meeting probabilities | Not implemented |
| Treasury.gov | Yield curves | Not implemented |

---

## Risk Analysis

| Risk | Description | Mitigation |
|------|-------------|------------|
| Model risk | Black-Scholes may not fit crypto | Use thicker tails, validate |
| Basis risk | Prediction vs options may settle differently | Check resolution criteria |
| Liquidity | Options may be illiquid at strikes | Check order books |
| Timing | Vol and spot change fast | Monitor positions |

---

## Recommendations

### Current Assessment

The financial markets strategy is:
- **Theoretically sound** - Options math is rigorous
- **Data-limited** - Needs live feeds from Deribit/CME
- **Market-limited** - Few price threshold markets found

### Next Steps

1. **Integrate Deribit API** - Get real BTC/ETH options data
2. **Improve pattern matching** - Find more price threshold markets
3. **Add Fed rate markets** - Compare to CME FedWatch
4. **Build hedging module** - Calculate option hedges

### When to Use

Best for traders who:
- Have options/derivatives experience
- Can access real-time options data
- Understand Black-Scholes limitations
- Want to hedge prediction market exposure

---

## Reproducibility

### Run the Scan

```bash
python run.py scan --strategy financial_markets
```

### Raw Data

- Scan results: `research/financial_markets/data/scan_20260117_014100.json`
- Findings: `research/financial_markets/findings/scan_20260117.md`

---

*Generated: January 17, 2026*
*Based on: Real scan data (1 market found, placeholder spot/vol data)*
*Note: Full functionality requires Deribit integration*
