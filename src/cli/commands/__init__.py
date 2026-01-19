"""CLI command modules.

Core commands:
- connect: Test platform connections
- markets: List active markets
- scan: Scan for trading opportunities
- signals: Generate trading signals
- backtest: Run strategy backtests
- run: Start live monitoring with scheduler

Command groups:
- analyze: Analysis workflows (risk, ev, cross-platform)
- report: Generate reports (JSON, PDF)
- visualize: Create visualizations
- data: Data collection and export
"""

from . import connect, markets, scan, signals, backtest, run

__all__ = ["connect", "markets", "scan", "signals", "backtest", "run"]
