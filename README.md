# Polymarket Analyzer

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-116%20passed-brightgreen.svg)]()

**Quantitative analysis toolkit for prediction markets.** Detects mispriced contracts across Polymarket and Kalshi using behavioral finance research and options pricing theory.

## Strategies

| Strategy | Mechanism |
|----------|-----------|
| **Favorite-Longshot** | High-probability outcomes underpriced due to behavioral bias |
| **Single-Condition Arb** | YES + NO != $1.00 creates risk-free profit |
| **Multi-Outcome Arb** | Bundle pricing errors on events with 3+ outcomes |
| **Cross-Platform** | Polymarket vs Kalshi price discrepancies |
| **Financial Markets** | Prediction market vs Black-Scholes fair value (live Deribit data) |

## Quick Start

```bash
pip install -r requirements.txt
python -m src scan --strategy all
```

## Project Structure

```
polymarket-analyzer/
├── exploration.ipynb              # Interactive notebook
├── *_briefing.md                  # Strategy analysis documents
├── *_briefing.pdf                 # PDF versions
├── src/                           # Core code
│   ├── adapters/                  # Polymarket, Kalshi, Deribit APIs
│   ├── strategies/                # Strategy implementations
│   └── cli/                       # Command-line interface
└── tests/                         # Unit tests (116 passing)
```

## CLI Reference

```bash
# Scan for opportunities
python -m src scan --strategy all
python -m src scan --strategy favorite_longshot
python -m src scan --strategy financial_markets --atm-iv

# Other commands
python -m src connect              # Test API connections
python -m src markets --limit 50   # List active markets
```

## Sample Output

```
$ python -m src scan --strategy financial_markets

  ==================================================================
  Will bitcoin hit $1m before GTA VI?
  ==================================================================

  Market Question: Will BTC exceed $1,000,000?
  Current Spot:    $95,127 (+951% to target)

  +-----------------------------------------------------------+
  |  PREDICTION MARKET PRICE:   48.5%                       |
  |  OPTIONS-IMPLIED VALUE:      0.0%  (Black-Scholes)      |
  |-----------------------------------------------------------|
  |  MISPRICING:               -48.5%                       |
  +-----------------------------------------------------------+

  TRADE THESIS:
    The prediction market is OVERPRICED vs options fair value.
    -> SELL on Polymarket at 48.5%
    -> Options market implies only 0.0% probability
```

## API Usage

```python
import asyncio
from src.adapters import PolymarketAdapter
from src.strategies import FavoriteLongshotStrategy

async def main():
    adapter = PolymarketAdapter()
    await adapter.connect()

    markets = await adapter.get_markets(limit=100)
    strategy = FavoriteLongshotStrategy(min_probability=0.90)

    for market in markets:
        opp = strategy.check_market(market)
        if opp:
            print(f"{opp.side} @ {opp.price:.1%}")

    await adapter.disconnect()

asyncio.run(main())
```

## Research References

- Kahneman & Tversky (1979). Prospect Theory
- Snowberg & Wolfers (2010). Explaining the Favorite-Long Shot Bias. NBER 15923

## License

MIT License. For educational and research purposes.
