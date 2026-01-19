"""Backtest command - Run strategy backtests.

Supports two modes:
1. Historical backtest (default): Uses real data from database
2. Monte Carlo simulation: Statistical simulation (legacy)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import click
import numpy as np

from ..utils import async_command, print_header, print_subheader, format_price, format_usd
from ...core.config import Credentials
from ...adapters.polymarket import PolymarketAdapter
from ...storage.database import Database
from ...analysis.backtester import HistoricalBacktester


@click.command()
@click.option(
    "--strategy",
    type=click.Choice(["favorite_longshot", "single_arb"], case_sensitive=False),
    default="favorite_longshot",
    help="Strategy to backtest.",
)
@click.option(
    "--mode",
    type=click.Choice(["historical", "monte_carlo"], case_sensitive=False),
    default="historical",
    help="Backtest mode: historical (real data) or monte_carlo (simulation).",
)
@click.option("--simulations", type=int, default=100, help="Number of Monte Carlo simulations (monte_carlo mode only).")
@click.option("--capital", type=float, default=1000, help="Starting capital.")
@click.option("--bet-size", type=float, default=25, help="Bet size per position ($).")
@click.option("--limit", type=int, default=200, help="Markets to include.")
@click.option("--db-path", type=click.Path(), default="data/polymarket.db", help="Database path (historical mode).")
@click.option(
    "--output",
    type=click.Choice(["table", "json"], case_sensitive=False),
    default="table",
)
@click.option("--visualize/--no-visualize", default=False, help="Generate visualization.")
@async_command
async def backtest(
    strategy: str,
    mode: str,
    simulations: int,
    capital: float,
    bet_size: float,
    limit: int,
    db_path: str,
    output: str,
    visualize: bool,
) -> None:
    """Run strategy backtest.

    \b
    Two modes available:
    - historical: Uses REAL market data and resolutions from database
    - monte_carlo: Statistical simulation (for comparison only)

    The historical mode is preferred as it uses actual market outcomes
    rather than simulated probabilities.

    \b
    Examples:
      python -m src backtest                              # Historical (default)
      python -m src backtest --mode historical            # Explicit historical
      python -m src backtest --mode monte_carlo           # Legacy simulation
      python -m src backtest --capital 5000 --bet-size 50
    """
    print_header("BACKTEST")
    click.echo(f"\n  Mode: {mode.upper()}")
    click.echo(f"  Strategy: {strategy}")
    click.echo(f"  Starting Capital: {format_usd(capital)}")
    click.echo(f"  Bet Size: {format_usd(bet_size)}")

    if mode == "historical":
        # Historical backtest using real data
        await _run_historical_backtest(
            strategy, capital, bet_size, db_path, output, visualize
        )
    else:
        # Legacy Monte Carlo simulation
        click.echo(f"  Simulations: {simulations}")
        await _run_monte_carlo_backtest(
            strategy, simulations, capital, bet_size, limit, output, visualize
        )


async def _run_historical_backtest(
    strategy: str,
    capital: float,
    bet_size: float,
    db_path: str,
    output: str,
    visualize: bool,
) -> None:
    """Run backtest using real historical data."""
    db_file = Path(db_path)
    if not db_file.exists():
        click.echo(f"\n  ERROR: Database not found: {db_path}")
        click.echo("  Run 'python -m src data collect --once' first to collect data")
        return

    db = Database(db_file)
    await db.connect()

    try:
        # Check for resolution data
        resolutions = await db.get_resolutions(limit=1)
        if not resolutions:
            click.echo("\n  ERROR: No resolution data found")
            click.echo("  Historical backtest requires resolved markets")
            click.echo("  Resolution data will accumulate as markets close")
            click.echo("\n  TIP: Use --mode monte_carlo for simulation without historical data")
            return

        click.echo("\n  Running historical backtest...")

        backtester = HistoricalBacktester(db)
        result = await backtester.run(
            strategy=strategy,
            min_probability=0.90,
            bet_size=bet_size,
            capital=capital,
        )

        if result.total_trades == 0:
            click.echo("\n  No trades generated - insufficient historical data")
            click.echo("  Need more resolved markets with matching snapshots")
            return

        if output == "json":
            click.echo(json.dumps({
                "mode": "historical",
                "strategy": result.strategy,
                "total_trades": result.total_trades,
                "wins": result.wins,
                "losses": result.losses,
                "win_rate": result.win_rate,
                "total_pnl": result.total_pnl,
                "total_return": result.total_return,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown,
                "profit_factor": result.profit_factor,
            }, indent=2))
        else:
            _print_historical_results(result)

        # Save results
        output_dir = Path("results/backtests")
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"backtest_historical_{timestamp}.json"

        with open(output_file, "w") as f:
            json.dump({
                "mode": "historical",
                "strategy": result.strategy,
                "start_date": result.start_date.isoformat() if result.start_date else None,
                "end_date": result.end_date.isoformat() if result.end_date else None,
                "total_trades": result.total_trades,
                "wins": result.wins,
                "losses": result.losses,
                "win_rate": result.win_rate,
                "total_pnl": result.total_pnl,
                "total_return": result.total_return,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown,
                "profit_factor": result.profit_factor,
                "avg_win": result.avg_win,
                "avg_loss": result.avg_loss,
                "timestamp": datetime.now().isoformat(),
            }, f, indent=2)
        click.echo(f"\n  Results saved to: {output_file}")

    finally:
        await db.close()


def _print_historical_results(result) -> None:
    """Print historical backtest results."""
    print_subheader("TRADE SUMMARY")
    click.echo(f"  {'Total Trades:':<25} {result.total_trades}")
    click.echo(f"  {'Wins:':<25} {result.wins}")
    click.echo(f"  {'Losses:':<25} {result.losses}")
    click.echo(f"  {'Win Rate:':<25} {format_price(result.win_rate)}")

    print_subheader("P&L SUMMARY")
    click.echo(f"  {'Total P&L:':<25} {format_usd(result.total_pnl)}")
    click.echo(f"  {'Total Return:':<25} {format_price(result.total_return)}")
    click.echo(f"  {'Average Win:':<25} {format_usd(result.avg_win)}")
    click.echo(f"  {'Average Loss:':<25} {format_usd(result.avg_loss)}")
    click.echo(f"  {'Profit Factor:':<25} {result.profit_factor:.2f}")

    print_subheader("RISK METRICS")
    click.echo(f"  {'Sharpe Ratio:':<25} {result.sharpe_ratio:.2f}")
    click.echo(f"  {'Max Drawdown:':<25} {format_price(result.max_drawdown)}")

    if result.start_date and result.end_date:
        print_subheader("PERIOD")
        click.echo(f"  Start: {result.start_date}")
        click.echo(f"  End: {result.end_date}")


async def _run_monte_carlo_backtest(
    strategy: str,
    simulations: int,
    capital: float,
    bet_size: float,
    limit: int,
    output: str,
    visualize: bool,
) -> None:
    """Run Monte Carlo simulation backtest (legacy)."""
    click.echo("\n  [WARNING] Monte Carlo mode uses simulated outcomes, not real data")

    # Fetch market data for backtest
    creds = Credentials.from_env()
    adapter = PolymarketAdapter(credentials=creds)

    click.echo("\n  Fetching market data...")
    await adapter.connect()
    markets = await adapter.get_markets(active_only=True, limit=limit)
    await adapter.disconnect()

    # Filter high-probability markets for favorite-longshot
    if strategy == "favorite_longshot":
        eligible_markets = [
            m for m in markets
            if max(m.yes_price, m.no_price) >= 0.90
        ]
    else:
        # Single arb - look for price inefficiencies
        eligible_markets = [
            m for m in markets
            if abs((m.yes_price + m.no_price) - 1.0) > 0.003
        ]

    click.echo(f"  Eligible Markets: {len(eligible_markets)}")

    if not eligible_markets:
        click.echo("\nNo eligible markets found for backtest.")
        return

    # Run Monte Carlo simulation
    click.echo("\n  Running simulation...")
    results = _run_monte_carlo(
        eligible_markets,
        strategy,
        simulations,
        capital,
        bet_size,
    )

    if output == "json":
        click.echo(json.dumps(results, indent=2))
    else:
        _print_backtest_results(results)

    # Save results
    output_dir = Path("results/backtests")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"backtest_montecarlo_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    click.echo(f"\n  Results saved to: {output_file}")

    # Generate visualization if requested
    if visualize:
        _generate_backtest_visualization(results, output_dir, timestamp)


def _run_monte_carlo(
    markets: list,
    strategy: str,
    num_simulations: int,
    capital: float,
    bet_size: float,
) -> dict:
    """Run Monte Carlo simulation."""
    returns = []
    win_rates = []
    max_drawdowns = []
    equity_curves = []

    for _ in range(num_simulations):
        sim_return, win_rate, max_dd, equity = _simulate_one_run(
            markets, strategy, capital, bet_size
        )
        returns.append(sim_return)
        win_rates.append(win_rate)
        max_drawdowns.append(max_dd)
        equity_curves.append(equity)

    returns_arr = np.array(returns)
    win_rates_arr = np.array(win_rates)
    max_dd_arr = np.array(max_drawdowns)

    # Calculate Sharpe ratio (assuming risk-free rate = 0)
    sharpe = np.mean(returns_arr) / np.std(returns_arr) if np.std(returns_arr) > 0 else 0

    return {
        "strategy": strategy,
        "simulations": num_simulations,
        "markets_tested": len(markets),
        "capital": capital,
        "bet_size": bet_size,
        "timestamp": datetime.now().isoformat(),
        "return_stats": {
            "mean": float(np.mean(returns_arr)),
            "median": float(np.median(returns_arr)),
            "std": float(np.std(returns_arr)),
            "min": float(np.min(returns_arr)),
            "max": float(np.max(returns_arr)),
            "percentile_5": float(np.percentile(returns_arr, 5)),
            "percentile_25": float(np.percentile(returns_arr, 25)),
            "percentile_75": float(np.percentile(returns_arr, 75)),
            "percentile_95": float(np.percentile(returns_arr, 95)),
        },
        "risk_metrics": {
            "sharpe_ratio": float(sharpe),
            "prob_profitable": float(np.mean(returns_arr > 0)),
            "avg_win_rate": float(np.mean(win_rates_arr)),
            "avg_max_drawdown": float(np.mean(max_dd_arr)),
        },
    }


def _simulate_one_run(
    markets: list,
    strategy: str,
    capital: float,
    bet_size: float,
) -> tuple[float, float, float, list[float]]:
    """Simulate one complete run through the markets."""
    balance = capital
    wins = 0
    total_bets = 0
    peak = capital
    max_drawdown = 0.0
    equity_curve = [capital]

    for market in markets:
        if balance < bet_size:
            break

        # Determine bet side and probability
        if strategy == "favorite_longshot":
            if market.yes_price >= 0.90:
                bet_prob = market.yes_price
                # Simulate resolution with bias adjustment
                # High-prob markets resolve YES more often than priced
                true_prob = min(0.99, bet_prob + 0.02)
            elif market.no_price >= 0.90:
                bet_prob = market.no_price
                true_prob = min(0.99, bet_prob + 0.02)
            else:
                continue

            # Simulate outcome
            won = np.random.random() < true_prob
            if won:
                profit = bet_size * (1 - bet_prob) / bet_prob
                wins += 1
            else:
                profit = -bet_size

        else:  # single_arb
            sum_prices = market.yes_price + market.no_price
            if sum_prices < 1.0:
                # Buy both - guaranteed profit
                profit = bet_size * (1.0 - sum_prices)
                wins += 1
            elif sum_prices > 1.0:
                # Sell both (if possible)
                profit = bet_size * (sum_prices - 1.0)
                wins += 1
            else:
                continue

        balance += profit
        total_bets += 1
        equity_curve.append(balance)

        # Track drawdown
        if balance > peak:
            peak = balance
        drawdown = (peak - balance) / peak if peak > 0 else 0
        max_drawdown = max(max_drawdown, drawdown)

    total_return = (balance - capital) / capital
    win_rate = wins / total_bets if total_bets > 0 else 0

    return total_return, win_rate, max_drawdown, equity_curve


def _print_backtest_results(results: dict) -> None:
    """Print backtest results in table format."""
    stats = results["return_stats"]
    risk = results["risk_metrics"]

    print_subheader("RETURN STATISTICS")
    click.echo(f"  {'Average Return:':<25} {format_price(stats['mean'])}")
    click.echo(f"  {'Median Return:':<25} {format_price(stats['median'])}")
    click.echo(f"  {'Best Return:':<25} {format_price(stats['max'])}")
    click.echo(f"  {'Worst Return:':<25} {format_price(stats['min'])}")

    print_subheader("RISK METRICS")
    click.echo(f"  {'Prob. Profitable:':<25} {format_price(risk['prob_profitable'])}")
    click.echo(f"  {'Average Win Rate:':<25} {format_price(risk['avg_win_rate'])}")
    click.echo(f"  {'Avg Max Drawdown:':<25} {format_price(risk['avg_max_drawdown'])}")
    click.echo(f"  {'Sharpe Ratio:':<25} {risk['sharpe_ratio']:.2f}")

    print_subheader("RETURN PERCENTILES")
    click.echo(f"  {'5th percentile:':<25} {format_price(stats['percentile_5'])} (worst case)")
    click.echo(f"  {'50th percentile:':<25} {format_price(stats['median'])} (median)")
    click.echo(f"  {'95th percentile:':<25} {format_price(stats['percentile_95'])} (best case)")


def _generate_backtest_visualization(results: dict, output_dir: Path, timestamp: str) -> None:
    """Generate visualization of backtest results."""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Backtest Results: {results['strategy']}", fontsize=14)

        # Return distribution histogram
        ax1 = axes[0, 0]
        stats = results["return_stats"]
        ax1.set_title("Return Distribution")
        ax1.axvline(stats["mean"], color="red", linestyle="--", label=f"Mean: {stats['mean']:.1%}")
        ax1.axvline(stats["median"], color="green", linestyle="--", label=f"Median: {stats['median']:.1%}")
        ax1.set_xlabel("Return")
        ax1.set_ylabel("Frequency")
        ax1.legend()

        # Risk metrics
        ax2 = axes[0, 1]
        risk = results["risk_metrics"]
        metrics = ["Prob Profit", "Win Rate", "1 - Max DD"]
        values = [risk["prob_profitable"], risk["avg_win_rate"], 1 - risk["avg_max_drawdown"]]
        ax2.bar(metrics, values, color=["green", "blue", "orange"])
        ax2.set_title("Risk Metrics")
        ax2.set_ylim(0, 1)
        for i, v in enumerate(values):
            ax2.text(i, v + 0.02, f"{v:.1%}", ha="center")

        plt.tight_layout()
        viz_file = output_dir / f"backtest_{timestamp}.png"
        plt.savefig(viz_file, dpi=150)
        plt.close()
        click.echo(f"  Visualization saved to: {viz_file}")

    except ImportError:
        click.echo("  (matplotlib not available for visualization)")
