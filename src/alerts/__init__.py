"""Alerting system for trading opportunities and events.

This module provides:
- Multi-channel alerting (Discord, Telegram, console)
- Configurable alert thresholds
- Rate limiting to prevent spam
- Alert history tracking
"""

from .base import (
    Alert,
    AlertChannel,
    AlertHandler,
    AlertLevel,
    AlertManager,
)
from .discord import DiscordHandler
from .telegram import TelegramHandler

__all__ = [
    "Alert",
    "AlertChannel",
    "AlertHandler",
    "AlertLevel",
    "AlertManager",
    "DiscordHandler",
    "TelegramHandler",
]
