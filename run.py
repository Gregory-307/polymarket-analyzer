#!/usr/bin/env python3
"""Polymarket Analyzer - Entry Point.

A production-quality prediction market analysis toolkit for
Polymarket and Kalshi platforms.

Usage:
    python run.py --help
    python run.py connect
    python run.py scan --strategy favorite_longshot
    python run.py signals --bankroll 1000
    python run.py report generate --strategy favorite_longshot
    python run.py visualize dashboard

Or run as module:
    python -m src --help
"""

from src.cli.main import cli

if __name__ == "__main__":
    cli()
