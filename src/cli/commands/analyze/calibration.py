"""Calibration analysis command."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import click

from ...utils import async_command, print_header, print_subheader
from ....storage.database import Database
from ....analysis.calibration import CalibrationAnalyzer


@click.command()
@click.option(
    "--db-path",
    type=click.Path(),
    default="data/polymarket.db",
    help="Database file path.",
)
@click.option(
    "--bucket-width",
    type=float,
    default=0.05,
    help="Width of probability buckets (default: 5%).",
)
@click.option(
    "--min-samples",
    type=int,
    default=20,
    help="Minimum samples per bucket.",
)
@click.option(
    "--platform",
    type=str,
    default=None,
    help="Filter by platform (e.g., 'polymarket').",
)
@click.option(
    "--fl-only",
    is_flag=True,
    help="Run favorite-longshot analysis only.",
)
@async_command
async def calibration(
    db_path: str,
    bucket_width: float,
    min_samples: int,
    platform: str | None,
    fl_only: bool,
) -> None:
    """Run calibration analysis on resolved markets.

    Compares market prices to actual outcomes to identify systematic
    mispricings and validate edge estimates.

    Requires resolution data in the database. Run resolution tracker
    first to collect outcome data.

    \b
    Examples:
      python -m src analyze calibration
      python -m src analyze calibration --fl-only
      python -m src analyze calibration --bucket-width 0.10
    """
    print_header("CALIBRATION ANALYSIS")

    click.echo(f"\n  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    db_file = Path(db_path)
    if not db_file.exists():
        click.echo(f"\n  ERROR: Database not found: {db_path}")
        return

    db = Database(db_file)
    await db.connect()

    try:
        analyzer = CalibrationAnalyzer(
            db,
            bucket_width=bucket_width,
            min_sample_size=min_samples,
        )

        # Check for resolution data
        resolutions = await db.get_resolutions(limit=1)
        if not resolutions:
            click.echo("\n  ERROR: No resolution data found")
            click.echo("  Run the resolution tracker to collect outcomes first")
            click.echo("  Or use 'python -m src data status' to check data")
            return

        if fl_only:
            # Favorite-longshot specific analysis
            print_subheader("FAVORITE-LONGSHOT BIAS ANALYSIS")

            fl_result = await analyzer.analyze_favorite_longshot(
                high_prob_threshold=0.90,
            )

            if "error" in fl_result:
                click.echo(f"\n  {fl_result['error']}")
                click.echo(f"  Sample size: {fl_result['sample_size']}")
                return

            click.echo(f"\n  Markets >= 90%: {fl_result['sample_size']}")
            click.echo(f"  Average Price: {fl_result['avg_price']:.1%}")
            click.echo(f"  Implied Win Rate: {fl_result['implied_rate']:.1%}")
            click.echo(f"  Actual Win Rate: {fl_result['actual_rate']:.1%}")
            click.echo(f"  Edge: {fl_result['edge']:+.2%}")
            click.echo(f"  95% CI: ({fl_result['confidence_interval'][0]:.1%}, {fl_result['confidence_interval'][1]:.1%})")
            click.echo(f"  Significant: {'Yes' if fl_result['significant'] else 'No'}")

            print_subheader("INTERPRETATION")
            click.echo(f"\n  {fl_result['interpretation']}")

        else:
            # Full calibration analysis
            result = await analyzer.analyze(platform=platform)

            if result.total_markets == 0:
                click.echo("\n  No markets with resolution data found")
                return

            click.echo(f"  Total Markets: {result.total_markets}")
            click.echo(f"  Brier Score: {result.brier_score:.4f} (perfect=0, random=0.25)")
            click.echo(f"  Mean Edge: {result.mean_edge:+.2%}")

            print_subheader("BUCKET ANALYSIS")

            # Header
            click.echo(f"\n  {'Price Range':<15} {'Implied':>10} {'Actual':>10} {'Edge':>10} {'N':>8} {'Sig':>5}")
            click.echo(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*5}")

            for b in result.buckets:
                sig = "*" if b.significant else ""
                click.echo(
                    f"  {b.price_low:.0%}-{b.price_high:.0%}  "
                    f"{b.implied_prob:>10.1%} "
                    f"{b.actual_rate:>10.1%} "
                    f"{b.edge:>+10.1%} "
                    f"{b.sample_size:>8} "
                    f"{sig:>5}"
                )

            print_subheader("KEY FINDINGS")

            # Identify significant buckets
            sig_buckets = [b for b in result.buckets if b.significant]

            if sig_buckets:
                click.echo("\n  Statistically significant mispricings found:")
                for b in sig_buckets:
                    direction = "underpriced" if b.edge > 0 else "overpriced"
                    click.echo(
                        f"    {b.price_low:.0%}-{b.price_high:.0%}: "
                        f"{direction} by {abs(b.edge):.1%} (n={b.sample_size})"
                    )
            else:
                click.echo("\n  No statistically significant mispricings found.")
                click.echo("  Markets appear well-calibrated.")

            # Favorite-longshot summary
            fl_buckets = [b for b in result.buckets if b.price_low >= 0.90]
            if fl_buckets:
                fl_edge = sum(b.edge * b.sample_size for b in fl_buckets) / sum(b.sample_size for b in fl_buckets)
                click.echo(f"\n  High-probability (>=90%) edge: {fl_edge:+.2%}")

    finally:
        await db.close()
