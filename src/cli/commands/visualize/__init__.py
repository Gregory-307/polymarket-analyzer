"""Visualize command group - Generate visualizations."""

from __future__ import annotations

import click

from .dashboard import dashboard
from .honest import honest


@click.group()
def visualize() -> None:
    """Generate visualizations.

    \b
    Subcommands:
      dashboard - Full portfolio dashboard (showcase)
      honest    - Reality-check visualizations with actual spreads

    \b
    Examples:
      python -m src visualize dashboard
      python -m src visualize honest --output-dir ./results
    """
    pass


visualize.add_command(dashboard)
visualize.add_command(honest)
