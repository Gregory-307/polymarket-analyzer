"""Analyze command group - Analysis workflows."""

from __future__ import annotations

import click

from .risk import risk
from .ev import ev
from .cross_platform import cross_platform
from .investigate import investigate
from .calibration import calibration


@click.group()
def analyze() -> None:
    """Run analysis workflows.

    \b
    Subcommands:
      risk           - Risk analysis with correlation modeling
      ev             - Expected value calculations
      cross-platform - Cross-platform price comparison
      investigate    - Data investigation and exploration
      calibration    - Calibration analysis on resolved markets

    \b
    Examples:
      python -m src analyze risk --strategy favorite_longshot
      python -m src analyze ev --bet-size 25
      python -m src analyze cross-platform
      python -m src analyze calibration --fl-only
    """
    pass


analyze.add_command(risk)
analyze.add_command(ev)
analyze.add_command(cross_platform)
analyze.add_command(investigate)
analyze.add_command(calibration)
