"""Base alerting infrastructure.

Provides:
- Alert dataclass for structured messages
- AlertHandler abstract base class
- AlertManager for routing alerts to handlers
- Rate limiting to prevent spam
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any

from ..core.utils import get_logger

logger = get_logger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Supported alert channels."""

    CONSOLE = "console"
    DISCORD = "discord"
    TELEGRAM = "telegram"
    EMAIL = "email"


@dataclass
class Alert:
    """Structured alert message.

    Attributes:
        title: Alert title/subject.
        message: Alert body text.
        level: Severity level.
        source: Source system/strategy.
        data: Additional structured data.
        timestamp: When the alert was created.
        alert_id: Unique identifier.
    """

    title: str
    message: str
    level: AlertLevel = AlertLevel.INFO
    source: str = "system"
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    alert_id: str = ""

    def __post_init__(self):
        if not self.alert_id:
            self.alert_id = f"{self.source}_{self.timestamp.timestamp():.0f}"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "title": self.title,
            "message": self.message,
            "level": self.level.value,
            "source": self.source,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }

    def format_text(self) -> str:
        """Format as plain text."""
        level_emoji = {
            AlertLevel.DEBUG: "[DEBUG]",
            AlertLevel.INFO: "[INFO]",
            AlertLevel.WARNING: "[WARN]",
            AlertLevel.ERROR: "[ERROR]",
            AlertLevel.CRITICAL: "[CRITICAL]",
        }

        lines = [
            f"{level_emoji.get(self.level, '')} {self.title}",
            f"{self.message}",
        ]

        if self.data:
            lines.append("")
            for key, value in self.data.items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.4f}")
                else:
                    lines.append(f"  {key}: {value}")

        lines.append(f"\n[{self.source}] {self.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")

        return "\n".join(lines)

    def format_markdown(self) -> str:
        """Format as markdown."""
        level_emoji = {
            AlertLevel.DEBUG: ":grey_question:",
            AlertLevel.INFO: ":information_source:",
            AlertLevel.WARNING: ":warning:",
            AlertLevel.ERROR: ":x:",
            AlertLevel.CRITICAL: ":rotating_light:",
        }

        lines = [
            f"## {level_emoji.get(self.level, '')} {self.title}",
            "",
            self.message,
        ]

        if self.data:
            lines.append("")
            lines.append("```")
            for key, value in self.data.items():
                if isinstance(value, float):
                    lines.append(f"{key}: {value:.4f}")
                else:
                    lines.append(f"{key}: {value}")
            lines.append("```")

        lines.append("")
        lines.append(f"*Source: {self.source} | {self.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}*")

        return "\n".join(lines)


class AlertHandler(ABC):
    """Abstract base class for alert handlers.

    Subclasses must implement send_alert().
    """

    def __init__(
        self,
        min_level: AlertLevel = AlertLevel.INFO,
        rate_limit_seconds: float = 60.0,
    ):
        """Initialize handler.

        Args:
            min_level: Minimum level to send.
            rate_limit_seconds: Minimum seconds between alerts.
        """
        self.min_level = min_level
        self.rate_limit_seconds = rate_limit_seconds
        self._last_alert_time: datetime | None = None
        self._alerts_sent: int = 0
        self._alerts_dropped: int = 0

    @property
    @abstractmethod
    def channel(self) -> AlertChannel:
        """Get the channel type."""
        pass

    @abstractmethod
    async def send_alert(self, alert: Alert) -> bool:
        """Send an alert.

        Args:
            alert: Alert to send.

        Returns:
            True if sent successfully.
        """
        pass

    def should_send(self, alert: Alert) -> bool:
        """Check if alert should be sent.

        Args:
            alert: Alert to check.

        Returns:
            True if alert passes filters.
        """
        # Check level
        level_order = list(AlertLevel)
        if level_order.index(alert.level) < level_order.index(self.min_level):
            return False

        # Check rate limit
        if self._last_alert_time is not None:
            elapsed = (datetime.now(timezone.utc) - self._last_alert_time).total_seconds()
            if elapsed < self.rate_limit_seconds:
                self._alerts_dropped += 1
                return False

        return True

    async def handle(self, alert: Alert) -> bool:
        """Handle an alert with filtering.

        Args:
            alert: Alert to handle.

        Returns:
            True if sent successfully.
        """
        if not self.should_send(alert):
            return False

        success = await self.send_alert(alert)

        if success:
            self._last_alert_time = datetime.now(timezone.utc)
            self._alerts_sent += 1

        return success

    def get_stats(self) -> dict:
        """Get handler statistics."""
        return {
            "channel": self.channel.value,
            "min_level": self.min_level.value,
            "rate_limit_seconds": self.rate_limit_seconds,
            "alerts_sent": self._alerts_sent,
            "alerts_dropped": self._alerts_dropped,
            "last_alert": self._last_alert_time.isoformat() if self._last_alert_time else None,
        }


class ConsoleHandler(AlertHandler):
    """Handler that prints alerts to console."""

    @property
    def channel(self) -> AlertChannel:
        return AlertChannel.CONSOLE

    async def send_alert(self, alert: Alert) -> bool:
        """Print alert to console."""
        print(alert.format_text())
        print("-" * 50)
        return True


class AlertManager:
    """Manages multiple alert handlers and routes alerts.

    Usage:
        manager = AlertManager()
        manager.add_handler(ConsoleHandler())
        manager.add_handler(DiscordHandler(webhook_url="..."))

        await manager.send(Alert(
            title="Opportunity Found",
            message="Buy YES at 0.85 for 15% ROI",
            level=AlertLevel.INFO,
            source="arbitrage",
            data={"market": "BTC 100k", "roi": 0.15},
        ))
    """

    def __init__(self):
        """Initialize alert manager."""
        self._handlers: list[AlertHandler] = []
        self._history: list[Alert] = []
        self._max_history: int = 1000

    def add_handler(self, handler: AlertHandler) -> None:
        """Add an alert handler.

        Args:
            handler: Handler to add.
        """
        self._handlers.append(handler)
        logger.info(f"Added alert handler: {handler.channel.value}")

    def remove_handler(self, channel: AlertChannel) -> None:
        """Remove handlers for a channel.

        Args:
            channel: Channel to remove.
        """
        self._handlers = [h for h in self._handlers if h.channel != channel]

    async def send(self, alert: Alert) -> dict[AlertChannel, bool]:
        """Send alert to all handlers.

        Args:
            alert: Alert to send.

        Returns:
            Dictionary mapping channel to success status.
        """
        results: dict[AlertChannel, bool] = {}

        for handler in self._handlers:
            try:
                success = await handler.handle(alert)
                results[handler.channel] = success
            except Exception as e:
                logger.error(f"Handler {handler.channel.value} failed: {e}")
                results[handler.channel] = False

        # Track history
        self._history.append(alert)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        return results

    async def send_opportunity(
        self,
        market_name: str,
        strategy: str,
        side: str,
        price: float,
        profit: float,
        **extra_data,
    ) -> dict[AlertChannel, bool]:
        """Send an opportunity alert.

        Args:
            market_name: Market name.
            strategy: Strategy that found it.
            side: YES or NO.
            price: Entry price.
            profit: Expected profit/ROI.
            **extra_data: Additional data fields.

        Returns:
            Send results per channel.
        """
        alert = Alert(
            title=f"Opportunity: {market_name}",
            message=f"Buy {side} at {price:.2%} for {profit:.2%} expected profit",
            level=AlertLevel.INFO,
            source=strategy,
            data={
                "market": market_name,
                "side": side,
                "price": price,
                "profit": profit,
                **extra_data,
            },
        )
        return await self.send(alert)

    async def send_error(
        self,
        title: str,
        error: str,
        source: str = "system",
    ) -> dict[AlertChannel, bool]:
        """Send an error alert.

        Args:
            title: Error title.
            error: Error message.
            source: Source system.

        Returns:
            Send results per channel.
        """
        alert = Alert(
            title=title,
            message=error,
            level=AlertLevel.ERROR,
            source=source,
        )
        return await self.send(alert)

    async def send_risk_alert(
        self,
        title: str,
        message: str,
        risk_data: dict,
    ) -> dict[AlertChannel, bool]:
        """Send a risk management alert.

        Args:
            title: Alert title.
            message: Alert message.
            risk_data: Risk metrics data.

        Returns:
            Send results per channel.
        """
        alert = Alert(
            title=title,
            message=message,
            level=AlertLevel.WARNING,
            source="risk_manager",
            data=risk_data,
        )
        return await self.send(alert)

    def get_history(
        self,
        limit: int = 100,
        level: AlertLevel | None = None,
        source: str | None = None,
    ) -> list[Alert]:
        """Get alert history.

        Args:
            limit: Maximum alerts to return.
            level: Filter by level.
            source: Filter by source.

        Returns:
            List of alerts.
        """
        alerts = self._history

        if level:
            alerts = [a for a in alerts if a.level == level]

        if source:
            alerts = [a for a in alerts if a.source == source]

        return alerts[-limit:]

    def get_stats(self) -> dict:
        """Get manager statistics."""
        return {
            "handlers": [h.get_stats() for h in self._handlers],
            "history_size": len(self._history),
            "max_history": self._max_history,
        }
