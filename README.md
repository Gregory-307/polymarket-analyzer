# Polymarket Analyzer

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-19%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Quantitative analysis toolkit for prediction markets.** Implements 4 trading strategies to detect mispriced contracts across Polymarket and Kalshi using behavioral finance research and options pricing theory.

## Features

- **Live market data** from Polymarket and Kalshi APIs
- **4 implemented strategies** with real-time scanning
- **Kelly Criterion** position sizing with risk controls
- **Gaussian Copula** correlation modeling for portfolio risk
- **Black-Scholes** fair value for crypto price threshold markets

> **Strategy Briefings**: See [`briefings/`](briefings/) for detailed analysis per strategy.

## Quick Start

```bash
pip install -r requirements.txt
python run.py scan --strategy all
```

## Strategies

| Strategy | Mechanism | Status |
|----------|-----------|--------|
| **Favorite-Longshot** | High-probability outcomes underpriced due to behavioral bias | Implemented |
| **Single-Condition Arb** | YES + NO ≠ $1.00 creates risk-free profit | Implemented |
| **Cross-Platform** | Polymarket vs Kalshi price discrepancies | Implemented |
| **Financial Markets** | Compare prediction markets to Black-Scholes fair value | Implemented |

---

## Sample Output

```
$ python run.py scan --strategy favorite_longshot

======================================================================
  OPPORTUNITY SCANNER
======================================================================
  Strategy: favorite_longshot
  Platform: polymarket
  Min Edge: 1.00%

----------------------------------------------------------------------
  FAVORITE-LONGSHOT OPPORTUNITIES
----------------------------------------------------------------------

  Will tariffs generate >$250b in 2025?
    NO @ 97.00% | Edge: 2.00% (MEDIUM) | Vol: $1,062,205

  Will Trump deport less than 250,000?
    NO @ 96.85% | Edge: 1.92% (LOW) | Vol: $959,251

  Will Elon and DOGE cut less than $50b in federal spending?
    YES @ 96.10% | Edge: 1.55% (LOW) | Vol: $320,416

  Will the San Francisco 49ers win Super Bowl 2026?
    NO @ 95.25% | Edge: 1.12% (LOW) | Vol: $7,080,133

----------------------------------------------------------------------
  Summary: 7 opportunities found
----------------------------------------------------------------------
```

---

## Project Structure

```
polymarket-analyzer/
├── run.py                    # CLI entry point
├── briefings/                # Strategy analysis reports
│   ├── favorite_longshot_briefing.md
│   ├── arbitrage_briefing.md
│   ├── cross_platform_briefing.md
│   └── financial_markets_briefing.md
├── research/                 # Raw scan data and findings
│   ├── arbitrage/
│   ├── cross_platform/
│   ├── favorite_longshot/
│   └── financial_markets/
├── src/
│   ├── cli/                  # Command-line interface
│   ├── adapters/             # Polymarket & Kalshi API clients
│   ├── strategies/           # Trading strategy implementations
│   ├── metrics/              # Order book analysis
│   └── scanners/             # Opportunity detection
├── output/                   # Generated results
├── tests/                    # Unit tests (19 passing)
└── configs/                  # Configuration files
```

---

## CLI Reference

```bash
# Core commands
python run.py connect                    # Test API connections
python run.py markets --limit 50         # List active markets
python run.py scan --strategy all        # Scan for opportunities
python run.py signals --bankroll 1000    # Generate Kelly-sized signals
python run.py backtest --simulations 100 # Monte Carlo backtest

# Analysis
python run.py analyze risk --correlation # Risk analysis with Gaussian copula
python run.py analyze ev --bet-size 25   # Expected value by edge scenario
python run.py analyze investigate        # Reality check with actual spreads

# Reports
python run.py report pdf --strategy favorite_longshot
python run.py visualize dashboard
```

---

## Installation

```bash
git clone https://github.com/Gregory-307/polymarket-analyzer.git
cd polymarket-analyzer
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Optional: Configure API keys for trading
cp configs/credentials.env.example .env
```

---

## API Usage

```python
import asyncio
from src.adapters import PolymarketAdapter, KalshiAdapter
from src.strategies import FavoriteLongshotStrategy, CrossPlatformStrategy

async def main():
    # Connect to prediction markets
    poly = PolymarketAdapter()
    kalshi = KalshiAdapter()

    await poly.connect()
    await kalshi.connect()

    # Scan for favorite-longshot opportunities
    markets = await poly.get_markets(limit=100)
    strategy = FavoriteLongshotStrategy(min_probability=0.95, min_edge=0.01)

    for market in markets:
        opp = strategy.check_market(market)
        if opp:
            print(f"{opp.side} @ {opp.current_price:.1%} | Edge: {opp.edge:.2%}")

            # Calculate position size with Kelly Criterion
            size = strategy.calculate_position_size(opp, account_balance=10000)
            print(f"  Recommended size: ${size:.2f}")

    await poly.disconnect()
    await kalshi.disconnect()

asyncio.run(main())
```

---

## Key Findings (January 2026 Scans)

Real scans with live market data revealed:

| Strategy | Markets Scanned | Result |
|----------|-----------------|--------|
| Favorite-Longshot | 100 | 7 opportunities, 1.5% avg edge |
| Single-Condition Arb | 500 | 0 opportunities (markets efficient) |
| Cross-Platform | 200 | 0 matches (no market overlap) |
| Financial Markets | 500 | 1 market found (BTC $1M) |

**Notable findings:**
- Polymarket and Kalshi have minimal market overlap (Kalshi focuses on sports)
- Single-condition arbitrage opportunities are rare (markets internally consistent)
- Favorite-longshot bias is most actionable strategy with consistent edge
- Crypto price threshold markets are scarce but can be valued with Black-Scholes

See [`research/`](research/) for raw scan data and detailed findings.

---

## Research References

- Kahneman, D., & Tversky, A. (1979). Prospect Theory: An Analysis of Decision under Risk
- Snowberg, E., & Wolfers, J. (2010). Explaining the Favorite-Long Shot Bias. NBER Working Paper 15923
- Saguillo, O., et al. (2025). Unravelling the Probabilistic Forest. arXiv:2508.03474

---

## License

MIT License - See LICENSE file for details.

**Disclaimer**: This software is for educational and research purposes. Trading involves risk of loss. Past performance does not guarantee future results.
