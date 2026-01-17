# Polymarket Analyzer

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Quantitative analysis toolkit for prediction markets.** Detects mispriced contracts using behavioral bias research and correlation modeling across Polymarket and Kalshi.

> **Investment Briefings**: See [`briefings/`](briefings/) for strategy-specific analysis reports and recommendations.

## Quick Start

```bash
pip install -r requirements.txt
python run.py scan --strategy favorite_longshot
```

**Or explore interactively:**
```bash
jupyter notebook exploration.ipynb
```

---

## The Edge

This toolkit exploits documented market inefficiencies:

| Strategy | Mechanism | Research Basis | Documented Returns |
|----------|-----------|----------------|-------------------|
| **Favorite-Longshot Bias** | High-probability outcomes are systematically underpriced | Prospect Theory (Kahneman & Tversky, 1979) | 2-5% edge per position |
| **Single-Condition Arb** | YES + NO ≠ $1.00 creates risk-free profit | arXiv:2508.03474 | 1-3% per trade |
| **Multi-Outcome Arb** | Bundle arbitrage across 3+ outcomes | $28.3M extracted historically | 1-5% per trade |
| **Cross-Platform Arb** | Polymarket vs Kalshi price gaps | $40M+ extracted historically | 2-8% per trade |

---

## Sample Output

```
======================================================================
TRADING SIGNALS - Favorite-Longshot Strategy
======================================================================

Signal: STRONG
  Market: Will the 49ers win Super Bowl 2026?
  Side: NO @ 95.75%
  Fair Value: 98.75%
  Edge: 3.00%
  Kelly Size: $100 (10% of bankroll)
  Risk: LOW

Portfolio Summary:
  STRONG Signals: 5
  Average Edge: 1.87%
  Expected ROI: 2.55%
```

See full analysis: [`output/reports/favorite_longshot_briefing.pdf`](output/reports/)

---

## Project Structure

```
polymarket-analyzer/
├── exploration.ipynb         # Interactive demo - START HERE
├── run.py                    # CLI entry point
├── briefings/                # Investment briefings per strategy
│   ├── favorite_longshot_briefing.pdf
│   └── single_arb_briefing.pdf
├── src/
│   ├── cli/                  # Command-line interface
│   ├── adapters/             # Polymarket & Kalshi API clients
│   ├── strategies/           # Trading strategy implementations
│   ├── metrics/              # Order book analysis (OBI, liquidity, spread)
│   └── scanners/             # Real-time opportunity detection
├── output/                   # Generated artifacts (gitignored)
│   ├── charts/               # Visualizations
│   ├── data/                 # Trading signals
│   └── backtests/            # Monte Carlo results
├── tests/
└── configs/
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

## Backtest Results

Monte Carlo simulation (100 runs, 140 high-probability markets):

| Metric | Value |
|--------|-------|
| Win Rate | 98.9% |
| Prob. Profitable | 54% |
| Median Return | +0.49% |
| Sharpe Ratio | 1.08 |
| 5th Percentile | -14.4% |
| 95th Percentile | +6.25% |

The high win rate validates the favorite-longshot bias. Negative tail risk emphasizes the importance of position sizing (Kelly Criterion).

---

## API Usage

```python
import asyncio
from src.adapters import PolymarketAdapter
from src.strategies import FavoriteLongshotStrategy

async def main():
    adapter = PolymarketAdapter()
    await adapter.connect()

    markets = await adapter.get_markets(limit=100)

    strategy = FavoriteLongshotStrategy(min_probability=0.95)
    for market in markets:
        opp = strategy.check_market(market)
        if opp:
            print(f"{opp.side} @ {opp.current_price:.1%} | Edge: {opp.edge:.2%}")

    await adapter.disconnect()

asyncio.run(main())
```

---

## Research References

- Kahneman, D., & Tversky, A. (1979). Prospect Theory: An Analysis of Decision under Risk
- Snowberg, E., & Wolfers, J. (2010). Explaining the Favorite-Long Shot Bias. NBER Working Paper 15923
- Saguillo, O., et al. (2025). Unravelling the Probabilistic Forest. arXiv:2508.03474

---

## License

MIT License - See LICENSE file for details.

**Disclaimer**: This software is for educational and research purposes. Trading involves risk of loss. Past performance does not guarantee future results.
