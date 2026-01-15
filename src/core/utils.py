"""Utility functions for Polymarket Analyzer."""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import structlog


def setup_logging(level: str = "INFO", format: str = "console") -> None:
    """Configure structured logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        format: Output format ('console' or 'json').
    """
    # Configure standard library logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        stream=sys.stdout,
    )

    # Configure structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Configured logger instance.
    """
    return structlog.get_logger(name)


def utc_now() -> datetime:
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc)


def to_decimal(value: float | str | int) -> Decimal:
    """Convert value to Decimal for precise calculations.

    Args:
        value: Numeric value to convert.

    Returns:
        Decimal representation.
    """
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def format_price(price: float | Decimal, decimals: int = 4) -> str:
    """Format price for display.

    Args:
        price: Price value.
        decimals: Number of decimal places.

    Returns:
        Formatted price string.
    """
    return f"{float(price):.{decimals}f}"


def format_usd(amount: float | Decimal) -> str:
    """Format amount as USD.

    Args:
        amount: Dollar amount.

    Returns:
        Formatted USD string.
    """
    return f"${float(amount):,.2f}"


def format_percentage(value: float | Decimal, decimals: int = 2) -> str:
    """Format value as percentage.

    Args:
        value: Value between 0 and 1.
        decimals: Number of decimal places.

    Returns:
        Formatted percentage string.
    """
    return f"{float(value) * 100:.{decimals}f}%"


def safe_divide(
    numerator: float | Decimal,
    denominator: float | Decimal,
    default: float = 0.0,
) -> float:
    """Safely divide two numbers.

    Args:
        numerator: Top of fraction.
        denominator: Bottom of fraction.
        default: Value to return if denominator is zero.

    Returns:
        Result of division or default.
    """
    if denominator == 0:
        return default
    return float(numerator) / float(denominator)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value to a range.

    Args:
        value: Value to clamp.
        min_val: Minimum allowed value.
        max_val: Maximum allowed value.

    Returns:
        Clamped value.
    """
    return max(min_val, min(max_val, value))


def dict_to_query_string(params: dict[str, Any]) -> str:
    """Convert dictionary to URL query string.

    Args:
        params: Dictionary of parameters.

    Returns:
        URL-encoded query string.
    """
    parts = []
    for key, value in params.items():
        if value is not None:
            parts.append(f"{key}={value}")
    return "&".join(parts)
