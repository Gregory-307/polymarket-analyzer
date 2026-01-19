"""Discord webhook alert handler.

Sends alerts to a Discord channel via webhook.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

from .base import Alert, AlertChannel, AlertHandler, AlertLevel
from ..core.utils import get_logger

logger = get_logger(__name__)


class DiscordHandler(AlertHandler):
    """Discord webhook alert handler.

    Usage:
        handler = DiscordHandler(
            webhook_url="https://discord.com/api/webhooks/...",
            username="Polymarket Bot",
        )

        await handler.send_alert(Alert(
            title="Opportunity Found",
            message="Buy YES at 0.85",
            level=AlertLevel.INFO,
        ))
    """

    def __init__(
        self,
        webhook_url: str,
        username: str = "Polymarket Alert",
        avatar_url: str | None = None,
        min_level: AlertLevel = AlertLevel.INFO,
        rate_limit_seconds: float = 5.0,
    ):
        """Initialize Discord handler.

        Args:
            webhook_url: Discord webhook URL.
            username: Bot username to display.
            avatar_url: Bot avatar URL (optional).
            min_level: Minimum level to send.
            rate_limit_seconds: Minimum seconds between alerts.
        """
        super().__init__(min_level=min_level, rate_limit_seconds=rate_limit_seconds)
        self.webhook_url = webhook_url
        self.username = username
        self.avatar_url = avatar_url

    @property
    def channel(self) -> AlertChannel:
        return AlertChannel.DISCORD

    def _level_to_color(self, level: AlertLevel) -> int:
        """Convert alert level to Discord embed color."""
        colors = {
            AlertLevel.DEBUG: 0x808080,  # Gray
            AlertLevel.INFO: 0x3498DB,  # Blue
            AlertLevel.WARNING: 0xF39C12,  # Orange
            AlertLevel.ERROR: 0xE74C3C,  # Red
            AlertLevel.CRITICAL: 0x9B59B6,  # Purple
        }
        return colors.get(level, 0x3498DB)

    def _build_embed(self, alert: Alert) -> dict[str, Any]:
        """Build Discord embed from alert."""
        embed = {
            "title": alert.title,
            "description": alert.message,
            "color": self._level_to_color(alert.level),
            "timestamp": alert.timestamp.isoformat(),
            "footer": {
                "text": f"Source: {alert.source}",
            },
        }

        # Add data as fields
        if alert.data:
            fields = []
            for key, value in alert.data.items():
                if isinstance(value, float):
                    display_value = f"{value:.4f}"
                elif isinstance(value, bool):
                    display_value = "Yes" if value else "No"
                else:
                    display_value = str(value)

                fields.append({
                    "name": key.replace("_", " ").title(),
                    "value": display_value,
                    "inline": True,
                })

            embed["fields"] = fields[:25]  # Discord limit

        return embed

    def _build_payload(self, alert: Alert) -> dict[str, Any]:
        """Build full webhook payload."""
        payload: dict[str, Any] = {
            "username": self.username,
            "embeds": [self._build_embed(alert)],
        }

        if self.avatar_url:
            payload["avatar_url"] = self.avatar_url

        return payload

    async def send_alert(self, alert: Alert) -> bool:
        """Send alert to Discord webhook.

        Args:
            alert: Alert to send.

        Returns:
            True if sent successfully.
        """
        try:
            import aiohttp
        except ImportError:
            logger.error("aiohttp required for Discord alerts: pip install aiohttp")
            return False

        payload = self._build_payload(alert)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    if response.status == 204:
                        logger.debug(f"Discord alert sent: {alert.title}")
                        return True
                    elif response.status == 429:
                        # Rate limited
                        retry_after = response.headers.get("Retry-After", "60")
                        logger.warning(f"Discord rate limited, retry after {retry_after}s")
                        return False
                    else:
                        text = await response.text()
                        logger.error(f"Discord error {response.status}: {text}")
                        return False

        except asyncio.TimeoutError:
            logger.error("Discord webhook timeout")
            return False
        except Exception as e:
            logger.error(f"Discord error: {e}")
            return False

    async def send_opportunity_embed(
        self,
        market_name: str,
        strategy: str,
        side: str,
        price: float,
        profit: float,
        volume: float | None = None,
        market_url: str | None = None,
    ) -> bool:
        """Send a rich opportunity embed.

        Args:
            market_name: Market name.
            strategy: Strategy name.
            side: YES or NO.
            price: Entry price.
            profit: Expected profit.
            volume: Market volume (optional).
            market_url: Link to market (optional).

        Returns:
            True if sent successfully.
        """
        try:
            import aiohttp
        except ImportError:
            logger.error("aiohttp required for Discord alerts")
            return False

        # Build rich embed
        embed = {
            "title": f"Trading Opportunity: {market_name}",
            "description": f"**Strategy:** {strategy}",
            "color": 0x00FF00,  # Green
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "fields": [
                {"name": "Side", "value": side, "inline": True},
                {"name": "Price", "value": f"{price:.2%}", "inline": True},
                {"name": "Expected Profit", "value": f"{profit:.2%}", "inline": True},
            ],
            "footer": {"text": "Polymarket Analyzer"},
        }

        if volume:
            embed["fields"].append({
                "name": "24h Volume",
                "value": f"${volume:,.0f}",
                "inline": True,
            })

        if market_url:
            embed["url"] = market_url

        payload = {
            "username": self.username,
            "embeds": [embed],
        }

        if self.avatar_url:
            payload["avatar_url"] = self.avatar_url

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    return response.status == 204
        except Exception as e:
            logger.error(f"Discord error: {e}")
            return False
