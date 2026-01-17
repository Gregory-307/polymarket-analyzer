"""Risk analysis command."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import click
import numpy as np
from scipy import stats

from ...utils import async_command, print_header, print_subheader, format_price, format_usd
from ....core.config import Credentials
from ....adapters.polymarket import PolymarketAdapter


@click.command()
@click.option(
    "--strategy",
    type=click.Choice(["favorite_longshot", "single_arb"], case_sensitive=False),
    default="favorite_longshot",
)
@click.option("--edge", type=float, default=0.02, help="Assumed edge (0.0 for honest analysis).")
@click.option("--correlation/--no-correlation", default=False, help="Include correlation analysis.")
@click.option("--bet-size", type=float, default=25, help="Bet size per position ($).")
@click.option("--limit", type=int, default=50, help="Markets to analyze.")
@click.option("--simulations", type=int, default=10000, help="Monte Carlo simulations.")
@click.option("--output", type=click.Choice(["table", "json"]), default="table")
@async_command
async def risk(
    strategy: str,
    edge: float,
    correlation: bool,
    bet_size: float,
    limit: int,
    simulations: int,
    output: str,
) -> None:
    """Run risk analysis with optional correlation modeling.

    Calculates expected value and risk metrics for a portfolio of trades:
    - Per-trade E(X) with win/loss breakdown
    - Scenario probabilities (all win, all lose, mixed)
    - Correlation effects using Gaussian Copula (if enabled)

    Use --edge 0 for "honest" analysis assuming markets are efficient.

    \b
    Examples:
      python -m src analyze risk
      python -m src analyze risk --edge 0 --correlation
      python -m src analyze risk --strategy single_arb
    """
    print_header("RISK ANALYSIS")
    click.echo(f"\n  Strategy: {strategy}")
    click.echo(f"  Assumed Edge: {format_price(edge)}")
    click.echo(f"  Bet Size: {format_usd(bet_size)}")
    click.echo(f"  Correlation Analysis: {'Yes' if correlation else 'No'}")

    # Fetch markets
    creds = Credentials.from_env()
    adapter = PolymarketAdapter(credentials=creds)

    click.echo("\n  Fetching market data...")
    await adapter.connect()
    markets = await adapter.get_markets(active_only=True, limit=limit)
    await adapter.disconnect()

    # Filter for strategy
    if strategy == "favorite_longshot":
        eligible = [m for m in markets if max(m.yes_price, m.no_price) >= 0.90]
    else:
        eligible = [m for m in markets if abs((m.yes_price + m.no_price) - 1.0) > 0.003]

    # Take top opportunities
    trades = []
    for market in eligible[:10]:
        if strategy == "favorite_longshot":
            prob = max(market.yes_price, market.no_price)
            side = "YES" if market.yes_price > market.no_price else "NO"
            true_prob = min(0.99, prob + edge)
            win_profit = bet_size * (1 - prob) / prob
            lose_profit = -bet_size
            ex = true_prob * win_profit + (1 - true_prob) * lose_profit
        else:
            sum_prices = market.yes_price + market.no_price
            prob = 1.0  # Arbitrage is certain if executed
            true_prob = 1.0
            win_profit = bet_size * abs(1.0 - sum_prices)
            lose_profit = 0
            ex = win_profit

        trades.append({
            "market_id": market.id,
            "question": market.question[:50] if market.question else "Unknown",
            "probability": prob,
            "true_probability": true_prob,
            "win_profit": win_profit,
            "lose_profit": lose_profit,
            "expected_value": ex,
            "bet_size": bet_size,
        })

    if not trades:
        click.echo("\nNo eligible trades found.")
        return

    # Calculate portfolio metrics
    total_ex = sum(t["expected_value"] for t in trades)
    total_cost = sum(t["bet_size"] for t in trades)
    roi = total_ex / total_cost if total_cost > 0 else 0

    results = {
        "strategy": strategy,
        "edge_assumption": edge,
        "timestamp": datetime.now().isoformat(),
        "trades": trades,
        "portfolio": {
            "total_trades": len(trades),
            "total_cost": total_cost,
            "expected_value": total_ex,
            "expected_roi": roi,
        },
    }

    # Correlation analysis
    if correlation:
        corr_results = _run_correlation_analysis(trades, simulations)
        results["correlation_analysis"] = corr_results

    if output == "json":
        click.echo(json.dumps(results, indent=2))
    else:
        _print_risk_results(results, correlation)

    # Save results
    output_dir = Path("results/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"risk_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    click.echo(f"\n  Results saved to: {output_file}")


def _run_correlation_analysis(trades: list[dict], simulations: int) -> dict:
    """Run Gaussian Copula correlation analysis."""
    n = len(trades)
    probs = np.array([t["true_probability"] for t in trades])
    win_profits = np.array([t["win_profit"] for t in trades])
    lose_profits = np.array([t["lose_profit"] for t in trades])

    results = {}
    for rho in [0.0, 0.1, 0.2, 0.3]:
        # Create correlation matrix
        corr_matrix = np.full((n, n), rho)
        np.fill_diagonal(corr_matrix, 1.0)

        # Generate correlated normals
        try:
            L = np.linalg.cholesky(corr_matrix)
        except np.linalg.LinAlgError:
            # Fallback if not positive definite
            corr_matrix = np.eye(n) * (1 - rho) + np.ones((n, n)) * rho
            np.fill_diagonal(corr_matrix, 1.0)
            L = np.linalg.cholesky(corr_matrix)

        z = np.random.randn(simulations, n)
        correlated_z = z @ L.T

        # Transform to uniforms then to outcomes
        uniforms = stats.norm.cdf(correlated_z)
        outcomes = uniforms < probs  # True = win

        # Calculate profits for each simulation
        sim_profits = np.where(outcomes, win_profits, lose_profits).sum(axis=1)

        results[f"rho_{rho}"] = {
            "correlation": rho,
            "mean_profit": float(np.mean(sim_profits)),
            "std_profit": float(np.std(sim_profits)),
            "prob_all_win": float(np.mean(outcomes.all(axis=1))),
            "prob_all_lose": float(np.mean(~outcomes.any(axis=1))),
            "prob_positive": float(np.mean(sim_profits > 0)),
            "percentile_5": float(np.percentile(sim_profits, 5)),
            "percentile_95": float(np.percentile(sim_profits, 95)),
        }

    return results


def _print_risk_results(results: dict, show_correlation: bool) -> None:
    """Print risk analysis results."""
    trades = results["trades"]
    portfolio = results["portfolio"]

    print_subheader("INDIVIDUAL TRADES")
    for i, t in enumerate(trades, 1):
        click.echo(f"\n  {i}. {t['question']}")
        click.echo(f"     Prob: {format_price(t['probability'])} | "
                  f"E(X): {format_usd(t['expected_value'])} | "
                  f"Win: {format_usd(t['win_profit'])} | "
                  f"Lose: {format_usd(t['lose_profit'])}")

    print_subheader("PORTFOLIO SUMMARY")
    click.echo(f"  Total Trades: {portfolio['total_trades']}")
    click.echo(f"  Total Cost: {format_usd(portfolio['total_cost'])}")
    click.echo(f"  Expected Value: {format_usd(portfolio['expected_value'])}")
    click.echo(f"  Expected ROI: {format_price(portfolio['expected_roi'])}")

    if show_correlation and "correlation_analysis" in results:
        print_subheader("CORRELATION ANALYSIS")
        click.echo("\n  E(X) is constant across correlations; only variance changes.\n")

        for key, data in results["correlation_analysis"].items():
            rho = data["correlation"]
            click.echo(f"  Correlation = {rho}")
            click.echo(f"    Mean: {format_usd(data['mean_profit'])} | "
                      f"Std: {format_usd(data['std_profit'])} | "
                      f"P(profit): {format_price(data['prob_positive'])}")
            click.echo(f"    P(all win): {format_price(data['prob_all_win'])} | "
                      f"P(all lose): {format_price(data['prob_all_lose'])}")
