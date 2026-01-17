"""CLI command modules.

Core commands:
- connect: Test platform connections
- markets: List active markets
- scan: Scan for trading opportunities
- signals: Generate trading signals
- backtest: Run strategy backtests

Command groups:
- analyze: Analysis workflows (risk, ev, cross-platform)
- report: Generate reports (JSON, PDF)
- visualize: Create visualizations
"""

from . import connect, markets, scan, signals, backtest

__all__ = ["connect", "markets", "scan", "signals", "backtest"]
