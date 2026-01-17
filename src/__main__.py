"""Enable running as: python -m src

Usage:
    python -m src --help
    python -m src connect
    python -m src scan --strategy favorite_longshot
"""

from src.cli.main import cli

if __name__ == "__main__":
    cli()
