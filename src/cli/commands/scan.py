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
from ...adapters.deribit_options import DeribitOptionsClient


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
@click.option(
    "--atm-iv/--historical-iv",
    default=False,
    help="Use ATM implied vol from Deribit options chain (slower but more accurate).",
)
@async_command
async def scan(
    strategy: str,
    platform: str,
    limit: int,
    min_edge: float,
    output: str,
    save: bool,
    atm_iv: bool,
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
        fl_strategy = FavoriteLongshotStrategy(min_probability=0.90)

        for market in all_markets:
            opp = fl_strategy.check_market(market)
            if opp:
                opportunities["favorite_longshot"].append({
                    "market_id": market.id,
                    "question": market.question,
                    "platform": market.platform,
                    "side": opp.side,
                    "price": opp.price,
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

        # Fetch live data from Deribit
        click.echo("\n  Fetching Deribit market data...")
        async with DeribitOptionsClient() as deribit:
            btc_spot = await deribit.get_spot("BTC")
            eth_spot = await deribit.get_spot("ETH")

            # Get volatility (ATM IV if requested, else historical)
            if atm_iv:
                click.echo("  Using ATM implied volatility from options chain...")
                btc_vol = await deribit.get_atm_iv("BTC", days_to_expiry=30)
                eth_vol = await deribit.get_atm_iv("ETH", days_to_expiry=30)
            else:
                btc_vol = await deribit.get_historical_vol("BTC")
                eth_vol = await deribit.get_historical_vol("ETH")

        # Fallback defaults
        btc_spot = btc_spot or 95000
        eth_spot = eth_spot or 3300
        btc_vol = btc_vol or 0.55
        eth_vol = eth_vol or 0.65

        vol_source = "ATM IV" if atm_iv else "Historical"
        click.echo(f"\n  Live data (Deribit) | Vol: {vol_source}")
        click.echo(f"    BTC: ${btc_spot:,.0f} | Vol: {btc_vol:.1%}")
        click.echo(f"    ETH: ${eth_spot:,.0f} | Vol: {eth_vol:.1%}")

        fm_strategy.set_spot_price("BTC", btc_spot)
        fm_strategy.set_spot_price("ETH", eth_spot)
        fm_strategy.set_implied_vol("BTC", btc_vol)
        fm_strategy.set_implied_vol("ETH", eth_vol)

        fm_opps = fm_strategy.scan(all_markets)

        for opp in fm_opps:
            fm_data = {
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
            }
            # Add Greeks if available
            if opp.greeks:
                fm_data["greeks"] = {
                    "delta": opp.greeks.delta,
                    "gamma": opp.greeks.gamma,
                    "theta": opp.greeks.theta,
                    "vega": opp.greeks.vega,
                }
                fm_data["hedge_ratio"] = opp.hedge_ratio
            opportunities["financial_markets"].append(fm_data)

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

        # Sort by price (highest probability first)
        fl_opps.sort(key=lambda x: x["price"], reverse=True)

        for opp in fl_opps[:10]:
            question = opp["question"][:55]
            click.echo(f"\n  {question}")
            click.echo(f"    {opp['side']} @ {format_price(opp['price'])} | "
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
        print_subheader("FINANCIAL MARKETS vs OPTIONS ARBITRAGE")
        click.echo("\n  Compares prediction market prices to Black-Scholes fair value")
        click.echo("  derived from Deribit options implied volatility.\n")

        # Sort by edge
        fm_opps.sort(key=lambda x: abs(x["edge"]), reverse=True)

        for opp in fm_opps[:10]:
            question = opp["question"][:60]
            click.echo(f"  {'='*66}")
            click.echo(f"  {question}")
            click.echo(f"  {'='*66}")

            # Current state
            threshold = opp['threshold']
            spot = opp['spot_price']
            pct_to_target = ((threshold / spot) - 1) * 100
            click.echo(f"\n  Market Question: Will {opp['underlying']} exceed ${threshold:,.0f}?")
            click.echo(f"  Current Spot:    ${spot:,.0f} ({pct_to_target:+.0f}% to target)")

            # The key comparison
            click.echo(f"\n  +-----------------------------------------------------------+")
            click.echo(f"  |  PREDICTION MARKET PRICE:  {opp['market_price']*100:>5.1f}%                       |")
            click.echo(f"  |  OPTIONS-IMPLIED VALUE:    {opp['fair_value']*100:>5.1f}%  (Black-Scholes)      |")
            click.echo(f"  |-----------------------------------------------------------|")
            click.echo(f"  |  MISPRICING:               {opp['edge_pct']:>+5.1f}%                       |")
            click.echo(f"  +-----------------------------------------------------------+")

            # Trade thesis
            click.echo(f"\n  TRADE THESIS:")
            if opp['edge'] > 0:
                click.echo(f"    The prediction market is UNDERPRICED vs options fair value.")
                click.echo(f"    -> BUY on Polymarket at {opp['market_price']*100:.1f}%")
                click.echo(f"    -> Options market implies {opp['fair_value']*100:.1f}% probability")
                click.echo(f"    -> Edge: +{opp['edge_pct']:.1f}% expected value")
            else:
                click.echo(f"    The prediction market is OVERPRICED vs options fair value.")
                click.echo(f"    -> SELL on Polymarket at {opp['market_price']*100:.1f}%")
                click.echo(f"    -> Options market implies only {opp['fair_value']*100:.1f}% probability")
                click.echo(f"    -> Edge: {opp['edge_pct']:.1f}% expected value (by selling)")

            # Hedging explanation
            click.echo(f"\n  HEDGING (Optional):")
            click.echo(f"    IV used: {opp['implied_vol']*100:.0f}% (from Deribit {opp['underlying']} options)")
            if opp.get("greeks") and opp.get("hedge_ratio") is not None:
                g = opp["greeks"]
                hr = opp["hedge_ratio"]
                # Check for valid Greeks (not NaN/Inf)
                if g['delta'] == g['delta'] and abs(g['delta']) < 1e6:  # NaN check
                    click.echo(f"    Delta: {g['delta']:.4f} (price sensitivity to spot)")
                    if hr != 0 and hr == hr:  # NaN check
                        action = "SELL" if hr < 0 else "BUY"
                        click.echo(f"    To delta-hedge: {action} {abs(hr):.6f} {opp['underlying']} per $1 position")
                else:
                    click.echo(f"    Delta: ~0 (far out of the money)")

            # Execution note
            click.echo(f"\n  EXECUTION:")
            click.echo(f"    1. {opp['direction']} this market on Polymarket")
            click.echo(f"    2. If hedging: use {opp['underlying']} futures or spot")
            click.echo(f"    3. Hold until resolution or exit on price movement")
            click.echo("")

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

    # Show average for strategies that have computed edges
    arb_and_fm = arb_opps + multi_opps + cp_opps + fm_opps
    if arb_and_fm:
        all_edges = [o.get("edge", o.get("profit_pct", o.get("spread", 0))) for o in arb_and_fm]
        avg_edge = sum(all_edges) / len(arb_and_fm)
        click.echo(f"  Average Edge (arb strategies): {format_price(avg_edge)}")
