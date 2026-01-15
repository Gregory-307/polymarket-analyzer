#!/usr/bin/env python3
"""Polymarket Analyzer - Main Entry Point.

A production-quality prediction market analysis toolkit for
Polymarket and Kalshi platforms.

Usage:
    python run.py test-connection     # Test platform connections
    python run.py markets             # List active markets
    python run.py scan                # Scan for opportunities
    python run.py scan --strategy favorite_longshot
    python run.py analyze             # Run market analysis
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.adapters import KalshiAdapter, PolymarketAdapter
from src.core.config import Credentials, load_config
from src.core.utils import format_percentage, format_usd, get_logger, setup_logging

logger = get_logger(__name__)


async def test_connection() -> None:
    """Test connections to both platforms."""
    config = load_config()
    credentials = Credentials.from_env()

    print("\n=== Testing Platform Connections ===\n")

    # Test Polymarket
    print("Polymarket:")
    poly = PolymarketAdapter(config=config.polymarket, credentials=credentials)
    result = await poly.test_connection()
    print(f"  Connected: {result.get('connected', False)}")
    print(f"  Authenticated: {result.get('authenticated', False)}")
    if result.get("latency_ms"):
        print(f"  Latency: {result['latency_ms']}ms")
    if result.get("error"):
        print(f"  Error: {result['error']}")
    await poly.disconnect()

    print()

    # Test Kalshi
    print("Kalshi:")
    kalshi = KalshiAdapter(config=config.kalshi, credentials=credentials)
    result = await kalshi.test_connection()
    print(f"  Connected: {result.get('connected', False)}")
    print(f"  Authenticated: {result.get('authenticated', False)}")
    if result.get("latency_ms"):
        print(f"  Latency: {result['latency_ms']}ms")
    if result.get("error"):
        print(f"  Error: {result['error']}")
    await kalshi.disconnect()

    print()


async def list_markets(platform: str = "all", limit: int = 20) -> None:
    """List active markets from platforms.

    Args:
        platform: 'polymarket', 'kalshi', or 'all'.
        limit: Maximum markets to show per platform.
    """
    config = load_config()

    print(f"\n=== Active Markets (limit={limit}) ===\n")

    if platform in ("all", "polymarket"):
        print("--- Polymarket ---")
        poly = PolymarketAdapter(config=config.polymarket)
        if await poly.connect():
            markets = await poly.get_markets(limit=limit)
            for m in markets:
                print(f"  [{m.id[:8]}...] {m.question[:60]}")
                print(f"           YES: {format_percentage(m.yes_price)} | "
                      f"NO: {format_percentage(m.no_price)} | "
                      f"Vol: {format_usd(m.volume)}")
            await poly.disconnect()
        print()

    if platform in ("all", "kalshi"):
        print("--- Kalshi ---")
        kalshi = KalshiAdapter(config=config.kalshi)
        if await kalshi.connect():
            markets = await kalshi.get_markets(limit=limit)
            for m in markets:
                print(f"  [{m.id}] {m.question[:60]}")
                print(f"           YES: {format_percentage(m.yes_price)} | "
                      f"NO: {format_percentage(m.no_price)} | "
                      f"Vol: {format_usd(m.volume)}")
            await kalshi.disconnect()
        print()


async def scan_opportunities(strategy: str = "all") -> None:
    """Scan for trading opportunities.

    Args:
        strategy: Strategy name or 'all'.
    """
    config = load_config()

    print(f"\n=== Scanning for Opportunities (strategy={strategy}) ===\n")

    # Initialize adapters
    poly = PolymarketAdapter(config=config.polymarket)
    kalshi = KalshiAdapter(config=config.kalshi)

    await poly.connect()
    await kalshi.connect()

    # Fetch markets
    poly_markets = await poly.get_markets(limit=50)
    kalshi_markets = await kalshi.get_markets(limit=50)

    opportunities = []

    # Single-condition arbitrage (YES + NO != 1)
    if strategy in ("all", "single_arb"):
        print("--- Single-Condition Arbitrage ---")
        for m in poly_markets + kalshi_markets:
            arb = m.arb_check
            if arb < 0.99:  # Buy-all opportunity
                profit = 1.0 - arb
                if profit >= config.single_arb.min_profit_usd / 100:
                    print(f"  BUY-ALL: {m.platform} | {m.question[:50]}")
                    print(f"           YES={format_percentage(m.yes_price)} + "
                          f"NO={format_percentage(m.no_price)} = "
                          f"{format_percentage(arb)} | "
                          f"Profit: {format_percentage(profit)}")
                    opportunities.append({
                        "type": "single_arb",
                        "action": "buy_all",
                        "market": m.id,
                        "platform": m.platform,
                        "profit_pct": profit,
                    })
            elif arb > 1.01:  # Sell-all opportunity
                profit = arb - 1.0
                if profit >= config.single_arb.min_profit_usd / 100:
                    print(f"  SELL-ALL: {m.platform} | {m.question[:50]}")
                    print(f"            YES={format_percentage(m.yes_price)} + "
                          f"NO={format_percentage(m.no_price)} = "
                          f"{format_percentage(arb)} | "
                          f"Profit: {format_percentage(profit)}")
                    opportunities.append({
                        "type": "single_arb",
                        "action": "sell_all",
                        "market": m.id,
                        "platform": m.platform,
                        "profit_pct": profit,
                    })
        print()

    # Favorite-longshot bias
    if strategy in ("all", "favorite_longshot"):
        print("--- Favorite-Longshot Bias (High-Prob Opportunities) ---")
        threshold = config.favorite_longshot.min_probability
        for m in poly_markets + kalshi_markets:
            # High probability YES outcomes
            if m.yes_price >= threshold:
                print(f"  HIGH-PROB YES: {m.platform} | {m.question[:50]}")
                print(f"                 Price: {format_percentage(m.yes_price)} | "
                      f"Potential edge if underpriced")
                opportunities.append({
                    "type": "favorite_longshot",
                    "side": "YES",
                    "market": m.id,
                    "platform": m.platform,
                    "price": m.yes_price,
                })
            # High probability NO outcomes (low YES price)
            elif m.yes_price <= (1 - threshold):
                print(f"  HIGH-PROB NO: {m.platform} | {m.question[:50]}")
                print(f"                Price: {format_percentage(m.no_price)} | "
                      f"Potential edge if underpriced")
                opportunities.append({
                    "type": "favorite_longshot",
                    "side": "NO",
                    "market": m.id,
                    "platform": m.platform,
                    "price": m.no_price,
                })
        print()

    # Cleanup
    await poly.disconnect()
    await kalshi.disconnect()

    # Summary
    print(f"=== Summary: {len(opportunities)} opportunities found ===")

    # Save results
    output_dir = Path("results/opportunities")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "latest_scan.json"
    with open(output_file, "w") as f:
        json.dump(opportunities, f, indent=2)
    print(f"Results saved to: {output_file}")


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Polymarket Analyzer - Prediction Market Analysis Toolkit"
    )
    parser.add_argument(
        "command",
        choices=["test-connection", "markets", "scan", "analyze"],
        help="Command to run",
    )
    parser.add_argument(
        "--platform",
        choices=["all", "polymarket", "kalshi"],
        default="all",
        help="Platform to target",
    )
    parser.add_argument(
        "--strategy",
        choices=["all", "single_arb", "favorite_longshot", "cross_platform"],
        default="all",
        help="Strategy to scan for",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum items to show",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level=args.log_level)

    # Run command
    if args.command == "test-connection":
        asyncio.run(test_connection())
    elif args.command == "markets":
        asyncio.run(list_markets(platform=args.platform, limit=args.limit))
    elif args.command == "scan":
        asyncio.run(scan_opportunities(strategy=args.strategy))
    elif args.command == "analyze":
        print("Analysis command not yet implemented. Coming in Phase 6.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
