"""Scan command - Scan for trading opportunities."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import click

from ..utils import (
    async_command,
    print_header,
    print_subheader,
    format_price,
    format_usd,
    format_edge,
    output_json,
)
from ...core.config import Credentials
from ...adapters.polymarket import PolymarketAdapter
from ...adapters.kalshi import KalshiAdapter
from ...strategies.favorite_longshot import FavoriteLongshotStrategy
from ...strategies.single_arb import SingleConditionArbitrage
from ...strategies.multi_arb import MultiOutcomeArbitrage
from ...strategies.cross_platform import CrossPlatformStrategy
from ...strategies.financial_markets import FinancialMarketsStrategy


@click.command()
@click.option(
    "--strategy",
    type=click.Choice(["all", "favorite_longshot", "single_arb", "multi_arb", "cross_platform", "financial_markets"], case_sensitive=False),
    default="all",
    help="Strategy to scan for.",
)
@click.option(
    "--platform",
    type=click.Choice(["all", "polymarket", "kalshi"], case_sensitive=False),
    default="polymarket",
    help="Platform to scan.",
)
@click.option(
    "--limit",
    type=int,
    default=100,
    help="Maximum markets to scan.",
)
@click.option(
    "--min-edge",
    type=float,
    default=0.01,
    help="Minimum edge to report (default: 1%).",
)
@click.option(
    "--output",
    type=click.Choice(["table", "json"], case_sensitive=False),
    default="table",
    help="Output format.",
)
@click.option(
    "--save/--no-save",
    default=True,
    help="Save results to file.",
)
@async_command
async def scan(
    strategy: str,
    platform: str,
    limit: int,
    min_edge: float,
    output: str,
    save: bool,
) -> None:
    """Scan prediction markets for trading opportunities.

    Analyzes markets for:
    - Favorite-longshot bias (underpriced high-probability outcomes)
    - Single-condition arbitrage (YES + NO != $1.00)
    - Multi-outcome bundle arbitrage

    \b
    Examples:
      python -m src scan
      python -m src scan --strategy favorite_longshot --min-edge 0.02
      python -m src scan --platform kalshi --output json
    """
    print_header("OPPORTUNITY SCANNER")
    click.echo(f"  Time: {datetime.now().isoformat()}")
    click.echo(f"  Strategy: {strategy}")
    click.echo(f"  Platform: {platform}")
    click.echo(f"  Min Edge: {format_price(min_edge)}")

    creds = Credentials.from_env()
    all_markets = []
    poly_markets = []
    kalshi_markets = []

    # Cross-platform requires both platforms
    if strategy == "cross_platform":
        platform = "all"

    # Fetch markets
    if platform in ("all", "polymarket"):
        click.echo("\nConnecting to Polymarket...")
        adapter = PolymarketAdapter(credentials=creds)
        try:
            await adapter.connect()
            markets = await adapter.get_markets(active_only=True, limit=limit)
            poly_markets = markets
            all_markets.extend(markets)
            click.echo(f"  Fetched {len(markets)} markets")
            await adapter.disconnect()
        except Exception as e:
            click.echo(f"  Error: {e}", err=True)

    if platform in ("all", "kalshi"):
        click.echo("\nConnecting to Kalshi...")
        adapter = KalshiAdapter(credentials=creds)
        try:
            await adapter.connect()
            markets = await adapter.get_markets(active_only=True, limit=limit)
            kalshi_markets = markets
            all_markets.extend(markets)
            click.echo(f"  Fetched {len(markets)} markets")
            await adapter.disconnect()
        except Exception as e:
            click.echo(f"  Error: {e}", err=True)

    if not all_markets:
        click.echo("\nNo markets found to scan.")
        return

    # Initialize opportunities
    opportunities = {
        "favorite_longshot": [],
        "single_arb": [],
        "multi_arb": [],
        "cross_platform": [],
        "financial_markets": [],
        "summary": {
            "total_markets": len(all_markets),
            "poly_markets": len(poly_markets),
            "kalshi_markets": len(kalshi_markets),
            "timestamp": datetime.now().isoformat(),
            "strategy": strategy,
            "platform": platform,
        },
    }

    # Scan for favorite-longshot opportunities
    if strategy in ("all", "favorite_longshot"):
        fl_strategy = FavoriteLongshotStrategy(min_probability=0.90, min_edge=min_edge)

        for market in all_markets:
            opp = fl_strategy.check_market(market)
            if opp:
                opportunities["favorite_longshot"].append({
                    "market_id": market.id,
                    "question": market.question,
                    "platform": market.platform,
                    "side": opp.side,
                    "price": opp.current_price,
                    "fair_value": opp.estimated_fair_value,
                    "edge": opp.edge,
                    "confidence": opp.confidence,
                    "volume": market.volume,
                    "liquidity": market.liquidity,
                    "strategy": "favorite_longshot",
                })

    # Scan for single-condition arbitrage
    if strategy in ("all", "single_arb"):
        arb_strategy = SingleConditionArbitrage(min_profit_pct=0.003)

        for market in all_markets:
            opp = arb_strategy.check_market(market)
            if opp:
                opportunities["single_arb"].append({
                    "market_id": market.id,
                    "question": market.question,
                    "platform": market.platform,
                    "action": opp.action,
                    "yes_price": opp.yes_price,
                    "no_price": opp.no_price,
                    "sum_prices": opp.sum_prices,
                    "profit_pct": opp.profit_pct,
                    "volume": market.volume,
                    "strategy": "single_arb",
                })

    # Multi-outcome arbitrage on grouped markets
    if strategy in ("all", "multi_arb"):
        multi_arb = MultiOutcomeArbitrage(min_profit_pct=min_edge)

        # Fetch events (grouped markets) from Polymarket
        if platform in ("all", "polymarket"):
            click.echo("\n  Fetching Polymarket events for multi-arb...")
            try:
                poly_adapter = PolymarketAdapter(credentials=creds)
                await poly_adapter.connect()
                events = await poly_adapter.get_events(limit=limit)
                await poly_adapter.disconnect()

                for event in events:
                    opp = multi_arb.scan_polymarket_group(event)
                    if opp:
                        opportunities["multi_arb"].append({
                            "market_id": opp.market_id,
                            "question": opp.question,
                            "platform": opp.platform,
                            "action": opp.action,
                            "num_outcomes": opp.num_outcomes,
                            "sum_prices": opp.sum_prices,
                            "profit_pct": opp.profit_pct,
                            "profit_usd": opp.profit_usd,
                            "outcomes": [
                                {"name": o.outcome_name, "price": o.price}
                                for o in opp.outcomes
                            ],
                            "strategy": "multi_arb",
                        })

                click.echo(f"  Scanned {len(events)} events")
            except Exception as e:
                click.echo(f"  Error fetching events: {e}", err=True)

    # Scan for cross-platform arbitrage
    if strategy in ("all", "cross_platform"):
        if poly_markets and kalshi_markets:
            cp_strategy = CrossPlatformStrategy(min_spread=min_edge, min_match_confidence=0.7)
            cp_opps = cp_strategy.find_opportunities(poly_markets, kalshi_markets)

            for opp in cp_opps:
                opportunities["cross_platform"].append({
                    "poly_market_id": opp.poly_market.id,
                    "kalshi_market_id": opp.kalshi_market.id,
                    "poly_question": opp.poly_market.question,
                    "kalshi_question": opp.kalshi_market.question,
                    "poly_yes_price": opp.poly_yes_price,
                    "kalshi_yes_price": opp.kalshi_yes_price,
                    "spread": opp.spread,
                    "spread_pct": opp.spread_pct,
                    "direction": opp.direction,
                    "match_confidence": opp.match_confidence,
                    "strategy": "cross_platform",
                })
        else:
            click.echo("\n  Cross-platform requires markets from both Polymarket and Kalshi.")

    # Scan for financial markets mispricing
    if strategy in ("all", "financial_markets"):
        fm_strategy = FinancialMarketsStrategy(min_edge=min_edge)

        import httpx

        # Fetch real spot prices and volatility
        btc_price, eth_price, sol_price = 95000, 3300, 150
        btc_vol, eth_vol, sol_vol = 0.55, 0.65, 0.80

        async with httpx.AsyncClient() as client:
            # Spot prices from Deribit (more accurate than CoinGecko for options)
            try:
                btc_resp = await client.get(
                    'https://www.deribit.com/api/v2/public/get_index_price',
                    params={'index_name': 'btc_usd'}
                )
                btc_data = btc_resp.json()
                btc_price = btc_data.get('result', {}).get('index_price', btc_price)

                eth_resp = await client.get(
                    'https://www.deribit.com/api/v2/public/get_index_price',
                    params={'index_name': 'eth_usd'}
                )
                eth_data = eth_resp.json()
                eth_price = eth_data.get('result', {}).get('index_price', eth_price)
            except Exception:
                pass  # Use defaults

            # Historical volatility from Deribit
            try:
                btc_vol_resp = await client.get(
                    'https://www.deribit.com/api/v2/public/get_historical_volatility',
                    params={'currency': 'BTC'}
                )
                btc_vol_data = btc_vol_resp.json()
                vol_points = btc_vol_data.get('result', [])
                if vol_points:
                    # Get latest volatility (percentage -> decimal)
                    btc_vol = vol_points[-1][1] / 100

                eth_vol_resp = await client.get(
                    'https://www.deribit.com/api/v2/public/get_historical_volatility',
                    params={'currency': 'ETH'}
                )
                eth_vol_data = eth_vol_resp.json()
                vol_points = eth_vol_data.get('result', [])
                if vol_points:
                    eth_vol = vol_points[-1][1] / 100
            except Exception:
                pass  # Use defaults

            # SOL from CoinGecko (Deribit doesn't have SOL)
            try:
                sol_resp = await client.get(
                    'https://api.coingecko.com/api/v3/simple/price',
                    params={'ids': 'solana', 'vs_currencies': 'usd'}
                )
                sol_data = sol_resp.json()
                sol_price = sol_data.get('solana', {}).get('usd', sol_price)
            except Exception:
                pass

        click.echo(f"\n  Live data (Deribit + CoinGecko):")
        click.echo(f"    BTC: ${btc_price:,.0f} | Vol: {btc_vol:.1%}")
        click.echo(f"    ETH: ${eth_price:,.0f} | Vol: {eth_vol:.1%}")
        click.echo(f"    SOL: ${sol_price:,.0f} | Vol: {sol_vol:.1%}")

        fm_strategy.set_spot_price("BTC", btc_price)
        fm_strategy.set_spot_price("ETH", eth_price)
        fm_strategy.set_spot_price("SOL", sol_price)
        fm_strategy.set_implied_vol("BTC", btc_vol)
        fm_strategy.set_implied_vol("ETH", eth_vol)
        fm_strategy.set_implied_vol("SOL", sol_vol)

        fm_opps = fm_strategy.scan(all_markets)

        for opp in fm_opps:
            opportunities["financial_markets"].append({
                "market_id": opp.market.id,
                "question": opp.market.question,
                "platform": opp.market.platform,
                "underlying": opp.underlying,
                "threshold": opp.threshold,
                "spot_price": opp.spot_price,
                "market_price": opp.market_price,
                "fair_value": opp.fair_value,
                "edge": opp.edge,
                "edge_pct": opp.edge_pct,
                "implied_vol": opp.implied_vol,
                "direction": opp.direction,
                "strategy": "financial_markets",
            })

        # Also report how many price threshold markets were found
        parsed_count = sum(
            1 for m in all_markets
            if fm_strategy.parse_price_threshold_market(m) is not None
        )
        opportunities["summary"]["price_threshold_markets"] = parsed_count

    # Output results
    if output == "json":
        output_json(opportunities)
    else:
        # Table output
        _print_scan_results(opportunities, min_edge)

    # Save results
    if save:
        output_dir = Path("results/opportunities")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"scan_{timestamp}.json"

        with open(output_file, "w") as f:
            json.dump(opportunities, f, indent=2, default=str)

        click.echo(f"\n  Results saved to: {output_file}")


def _print_scan_results(opportunities: dict, min_edge: float) -> None:
    """Print scan results in table format."""
    fl_opps = opportunities["favorite_longshot"]
    arb_opps = opportunities["single_arb"]
    multi_opps = opportunities.get("multi_arb", [])
    cp_opps = opportunities.get("cross_platform", [])
    fm_opps = opportunities.get("financial_markets", [])

    # Favorite-Longshot
    if fl_opps:
        print_subheader("FAVORITE-LONGSHOT OPPORTUNITIES")

        # Sort by edge
        fl_opps.sort(key=lambda x: x["edge"], reverse=True)

        for opp in fl_opps[:10]:
            question = opp["question"][:55]
            click.echo(f"\n  {question}")
            click.echo(f"    {opp['side']} @ {format_price(opp['price'])} | "
                      f"Edge: {format_edge(opp['edge'])} | "
                      f"Vol: {format_usd(opp.get('volume', 0))}")

    # Single Arb
    if arb_opps:
        print_subheader("SINGLE-CONDITION ARBITRAGE")

        # Sort by profit
        arb_opps.sort(key=lambda x: x["profit_pct"], reverse=True)

        for opp in arb_opps[:10]:
            question = opp["question"][:55]
            click.echo(f"\n  {question}")
            click.echo(f"    YES + NO = {opp['sum_prices']:.3f} | "
                      f"Profit: {format_price(opp['profit_pct'])} | "
                      f"Vol: {format_usd(opp.get('volume', 0))}")

    # Multi-Outcome Arb
    if multi_opps:
        print_subheader("MULTI-OUTCOME ARBITRAGE")

        # Sort by profit
        multi_opps.sort(key=lambda x: x["profit_pct"], reverse=True)

        for opp in multi_opps[:10]:
            question = opp["question"][:50]
            click.echo(f"\n  {question}")
            click.echo(f"    {opp['num_outcomes']} outcomes | Sum: {opp['sum_prices']:.3f}")
            click.echo(f"    Action: {opp['action']} | Profit: {format_price(opp['profit_pct'])}")

    # Cross-Platform
    if cp_opps:
        print_subheader("CROSS-PLATFORM OPPORTUNITIES")

        # Sort by spread
        cp_opps.sort(key=lambda x: x["spread"], reverse=True)

        for opp in cp_opps[:10]:
            poly_q = opp["poly_question"][:40]
            click.echo(f"\n  Poly: {poly_q}")
            click.echo(f"  Kalshi: {opp['kalshi_question'][:40]}")
            click.echo(f"    Poly YES: {format_price(opp['poly_yes_price'])} | "
                      f"Kalshi YES: {format_price(opp['kalshi_yes_price'])} | "
                      f"Spread: {opp['spread_pct']:.1f}%")
            click.echo(f"    Direction: {opp['direction']} | "
                      f"Match Confidence: {opp['match_confidence']:.0%}")

    # Financial Markets
    if fm_opps:
        print_subheader("FINANCIAL MARKETS OPPORTUNITIES")

        # Sort by edge
        fm_opps.sort(key=lambda x: abs(x["edge"]), reverse=True)

        for opp in fm_opps[:10]:
            question = opp["question"][:50]
            click.echo(f"\n  {question}")
            click.echo(f"    {opp['underlying']} > ${opp['threshold']:,.0f}")
            click.echo(f"    Spot: ${opp['spot_price']:,.0f} | "
                      f"Market: {format_price(opp['market_price'])} | "
                      f"Fair Value: {format_price(opp['fair_value'])}")
            click.echo(f"    Edge: {opp['edge_pct']:.1f}% | "
                      f"Direction: {opp['direction']} | "
                      f"IV: {opp['implied_vol']:.0%}")

    # Summary
    print_subheader("SUMMARY")
    click.echo(f"  Total Markets Scanned: {opportunities['summary']['total_markets']}")
    click.echo(f"  Favorite-Longshot Opportunities: {len(fl_opps)}")
    click.echo(f"  Single Arb Opportunities: {len(arb_opps)}")
    click.echo(f"  Multi-Outcome Arb Opportunities: {len(multi_opps)}")
    click.echo(f"  Cross-Platform Opportunities: {len(cp_opps)}")
    click.echo(f"  Financial Markets Opportunities: {len(fm_opps)}")

    if "price_threshold_markets" in opportunities["summary"]:
        click.echo(f"  Price Threshold Markets Found: {opportunities['summary']['price_threshold_markets']}")

    total_opps = len(fl_opps) + len(arb_opps) + len(multi_opps) + len(cp_opps) + len(fm_opps)
    if total_opps > 0:
        all_edges = [o.get("edge", o.get("profit_pct", o.get("spread", 0))) for o in fl_opps + arb_opps + multi_opps + cp_opps + fm_opps]
        avg_edge = sum(all_edges) / total_opps
        click.echo(f"  Average Edge: {format_price(avg_edge)}")
