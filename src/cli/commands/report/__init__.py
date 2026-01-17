"""Report command group - Generate reports."""

from __future__ import annotations

import click

from .generate import generate
from .pdf import pdf


@click.group()
def report() -> None:
    """Generate analysis reports.

    \b
    Subcommands:
      generate - Generate JSON/Markdown reports
      pdf      - Generate PDF briefings

    \b
    Examples:
      python -m src report generate --strategy favorite_longshot
      python -m src report pdf --strategy single_arb
    """
    pass


report.add_command(generate)
report.add_command(pdf)
