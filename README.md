# Polymarket Analyzer

**Production-quality prediction market analysis toolkit for Polymarket and Kalshi.**

A comprehensive system for detecting arbitrage opportunities, exploiting behavioral biases, and analyzing market microstructure in prediction markets.

## Motivation

Prediction markets are among the most efficient mechanisms for aggregating information, yet systematic inefficiencies persist. This project implements research-backed strategies to identify and exploit these edges:

1. **Favorite-Longshot Bias** - High-probability outcomes are systematically underpriced due to behavioral biases (Kahneman & Tversky, 1979)
2. **Single-Condition Arbitrage** - When YES + NO ≠ $1.00, risk-free profit exists
3. **Multi-Outcome Arbitrage** - Bundle strategies in markets with 3+ mutually exclusive outcomes
4. **Cross-Platform Arbitrage** - Price discrepancies between Polymarket and Kalshi

## Research Background

### Favorite-Longshot Bias

The well-documented behavioral phenomenon where:
- **Long shots** (low probability) are systematically **overpriced**
- **Favorites** (high probability) are systematically **underpriced**

**Source**: Prospect Theory (Kahneman & Tversky, 1979), NBER Working Paper 15923

**Mechanism**: People overvalue small probabilities (lottery ticket mentality) and undervalue near-certainties. At extreme probabilities (>95% or <5%), the mispricing is most pronounced.

**Documented Returns**: Up to 1800% annualized using the "High-Probability Bond Strategy" (ChainCatcher, 2025)

### Arbitrage Opportunities

Recent research ("Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets", arXiv:2508.03474) documented:
- **$28.3M** extracted via multi-outcome bundle arbitrage
- **$40M+** extracted via cross-platform arbitrage
- **2-8%** returns per trade on cross-platform opportunities

## Features

### Strategies

| Strategy | Description | Risk Level | Expected Return |
|----------|-------------|------------|-----------------|
| `favorite_longshot` | Buy underpriced high-probability outcomes | Low-Medium | 5-15% per position |
| `single_arb` | YES + NO < $1 arbitrage | Near-Zero | 1-3% per trade |
| `multi_arb` | Bundle arbitrage on multi-outcome markets | Near-Zero | 1-5% per trade |
| `cross_platform` | Polymarket vs Kalshi price discrepancies | Low | 2-8% per trade |

### Microstructure Metrics

- **Order Book Imbalance (OBI)** - Buy/sell pressure indicator
- **Liquidity Depth** - Available volume at price levels
- **Spread Dynamics** - Bid-ask spread tracking and analysis

### Platform Support

| Platform | Market Data | Trading | WebSocket |
|----------|-------------|---------|-----------|
| Polymarket | ✅ | ✅ | ✅ |
| Kalshi | ✅ | ✅ | ✅ |

## Installation

```bash
# Clone repository
git clone https://github.com/Gregory-307/polymarket-analyzer.git
cd polymarket-analyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Configure credentials (optional, for trading)
cp configs/credentials.env.example .env
# Edit .env with your API keys
```

## Quick Start

### Test Connections

```bash
python run.py test-connection
```

### List Active Markets

```bash
python run.py markets --limit 20
```

### Scan for Opportunities

```bash
# Scan all strategies
python run.py scan

# Scan specific strategy
python run.py scan --strategy favorite_longshot
python run.py scan --strategy single_arb
```

### Programmatic Usage

```python
import asyncio
from src.adapters import PolymarketAdapter
from src.strategies import FavoriteLongshotStrategy, SingleConditionArbitrage

async def main():
    # Connect to Polymarket
    adapter = PolymarketAdapter()
    await adapter.connect()

    # Fetch markets
    markets = await adapter.get_markets(limit=100)

    # Scan for favorite-longshot opportunities
    strategy = FavoriteLongshotStrategy(min_probability=0.95, min_edge=0.01)
    opportunities = strategy.scan(markets)

    for opp in opportunities:
        print(f"[{opp.side}] {opp.market.question}")
        print(f"  Price: {opp.current_price:.2%}")
        print(f"  Fair Value: {opp.estimated_fair_value:.2%}")
        print(f"  Edge: {opp.edge:.2%}")

    # Scan for arbitrage
    arb = SingleConditionArbitrage()
    arb_opps = arb.scan(markets)

    for opp in arb_opps:
        print(f"[{opp.action}] {opp.market.question}")
        print(f"  YES: {opp.yes_price:.2%} + NO: {opp.no_price:.2%} = {opp.sum_prices:.2%}")
        print(f"  Profit: {opp.profit_pct:.2%}")

    await adapter.disconnect()

asyncio.run(main())
```

## Architecture

```
polymarket-analyzer/
├── src/
│   ├── adapters/           # Platform API adapters
│   │   ├── base.py        # Abstract base class
│   │   ├── polymarket.py  # Polymarket CLOB client
│   │   └── kalshi.py      # Kalshi API client
│   │
│   ├── strategies/         # Trading strategies
│   │   ├── favorite_longshot.py  # Bias exploitation
│   │   ├── single_arb.py  # YES+NO arbitrage
│   │   └── multi_arb.py   # Bundle arbitrage
│   │
│   ├── metrics/            # Microstructure analysis
│   │   ├── order_imbalance.py
│   │   ├── liquidity_depth.py
│   │   └── spread_dynamics.py
│   │
│   ├── scanners/           # Real-time monitoring
│   │   └── arb_scanner.py
│   │
│   └── core/               # Configuration & utilities
│       ├── config.py
│       └── utils.py
│
├── analysis/               # Backtesting & research
├── configs/                # Configuration files
├── results/                # Output directory
└── tests/                  # Test suite
```

## Configuration

Edit `configs/default.json` to customize:

```json
{
  "strategies": {
    "favorite_longshot": {
      "enabled": true,
      "min_probability": 0.95,
      "min_edge": 0.01,
      "max_position_usd": 1000
    },
    "single_arb": {
      "enabled": true,
      "min_profit_usd": 0.50
    }
  }
}
```

## API Reference

### Adapters

```python
# Initialize adapter
adapter = PolymarketAdapter(credentials=Credentials.from_env())
await adapter.connect()

# Market data
markets = await adapter.get_markets(active_only=True, limit=100)
market = await adapter.get_market(market_id)
order_book = await adapter.get_order_book(token_id)
yes_price, no_price = await adapter.get_price(token_id)

# Trading (requires authentication)
order = await adapter.place_order(market_id, Side.BUY, 0.55, 10)
await adapter.cancel_order(order_id)
```

### Strategies

```python
# Favorite-Longshot Bias
strategy = FavoriteLongshotStrategy(min_probability=0.95, min_edge=0.01)
opportunities = strategy.scan(markets)
position_size = strategy.calculate_position_size(opp, balance, max_risk=0.05)

# Arbitrage Detection
arb = SingleConditionArbitrage(min_profit_pct=0.005)
opportunities = arb.scan(markets)
execution = arb.calculate_execution(opp, position_size_usd=100)
```

### Metrics

```python
# Order Imbalance
obi = OrderImbalanceMetric(threshold_bullish=0.3)
reading = obi.calculate(order_book)
print(f"Imbalance: {reading.imbalance:.2f} ({reading.signal})")

# Liquidity Depth
depth = LiquidityDepthMetric(levels=[0.01, 0.02, 0.05])
profile = depth.analyze(order_book)
slippage = depth.estimate_slippage(order_book, 1000, is_buy=True)
```

## Risk Considerations

1. **Black Swan Risk** - High-probability events can still fail; position sizing is critical
2. **Execution Risk** - Slippage and partial fills may reduce arbitrage profits
3. **Settlement Risk** - Different platforms may have different resolution criteria
4. **API Rate Limits** - Respect platform rate limits to avoid bans
5. **Platform Terms** - Ensure compliance with Terms of Service

## References

- Kahneman, D., & Tversky, A. (1979). Prospect Theory: An Analysis of Decision under Risk
- Snowberg, E., & Wolfers, J. (2010). Explaining the Favorite-Long Shot Bias. NBER Working Paper 15923
- Saguillo, O., et al. (2025). Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets. arXiv:2508.03474
- Polymarket Documentation: https://docs.polymarket.com/
- Kalshi Documentation: https://docs.kalshi.com/

## License

MIT License - See LICENSE file for details.

## Disclaimer

This software is for educational and research purposes. Trading in prediction markets involves risk of loss. Past performance does not guarantee future results. Always do your own research and never trade with money you cannot afford to lose.
