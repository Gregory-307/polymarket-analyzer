"""Telegram bot alert handler.

Sends alerts to a Telegram chat via Bot API.
"""

from __future__ import annotations

import asyncio
from typing import Any

from .base import Alert, AlertChannel, AlertHandler, AlertLevel
from ..core.utils import get_logger

logger = get_logger(__name__)


class TelegramHandler(AlertHandler):
    """Telegram bot alert handler.

    Usage:
        handler = TelegramHandler(
            bot_token="123456:ABC-DEF...",
            chat_id="-100123456789",
        )

        await handler.send_alert(Alert(
            title="Opportunity Found",
            message="Buy YES at 0.85",
            level=AlertLevel.INFO,
        ))

    To get your chat_id:
        1. Create a bot via @BotFather
        2. Add the bot to your chat/channel
        3. Send a message to the bot
        4. Visit https://api.telegram.org/bot<token>/getUpdates
        5. Find the chat.id in the response
    """

    BASE_URL = "https://api.telegram.org"

    def __init__(
        self,
        bot_token: str,
        chat_id: str | int,
        min_level: AlertLevel = AlertLevel.INFO,
        rate_limit_seconds: float = 1.0,
        parse_mode: str = "HTML",
    ):
        """Initialize Telegram handler.

        Args:
            bot_token: Telegram bot token from BotFather.
            chat_id: Target chat/channel ID.
            min_level: Minimum level to send.
            rate_limit_seconds: Minimum seconds between alerts.
            parse_mode: Message parse mode (HTML or Markdown).
        """
        super().__init__(min_level=min_level, rate_limit_seconds=rate_limit_seconds)
        self.bot_token = bot_token
        self.chat_id = str(chat_id)
        self.parse_mode = parse_mode

    @property
    def channel(self) -> AlertChannel:
        return AlertChannel.TELEGRAM

    def _format_html(self, alert: Alert) -> str:
        """Format alert as HTML for Telegram."""
        level_emoji = {
            AlertLevel.DEBUG: "🔍",
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ERROR: "❌",
            AlertLevel.CRITICAL: "🚨",
        }

        lines = [
            f"<b>{level_emoji.get(alert.level, '')} {alert.title}</b>",
            "",
            alert.message,
        ]

        if alert.data:
            lines.append("")
            lines.append("<pre>")
            for key, value in alert.data.items():
                if isinstance(value, float):
                    lines.append(f"{key}: {value:.4f}")
                else:
                    lines.append(f"{key}: {value}")
            lines.append("</pre>")

        lines.append("")
        lines.append(f"<i>Source: {alert.source}</i>")
        lines.append(f"<i>{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</i>")

        return "\n".join(lines)

    def _format_markdown(self, alert: Alert) -> str:
        """Format alert as Markdown for Telegram."""
        level_emoji = {
            AlertLevel.DEBUG: "🔍",
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ERROR: "❌",
            AlertLevel.CRITICAL: "🚨",
        }

        lines = [
            f"*{level_emoji.get(alert.level, '')} {self._escape_markdown(alert.title)}*",
            "",
            self._escape_markdown(alert.message),
        ]

        if alert.data:
            lines.append("")
            lines.append("```")
            for key, value in alert.data.items():
                if isinstance(value, float):
                    lines.append(f"{key}: {value:.4f}")
                else:
                    lines.append(f"{key}: {value}")
            lines.append("```")

        lines.append("")
        lines.append(f"_Source: {alert.source}_")
        lines.append(f"_{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}_")

        return "\n".join(lines)

    def _escape_markdown(self, text: str) -> str:
        """Escape Markdown special characters."""
        chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
        for char in chars:
            text = text.replace(char, f'\\{char}')
        return text

    def _get_api_url(self, method: str) -> str:
        """Get Telegram API URL for method."""
        return f"{self.BASE_URL}/bot{self.bot_token}/{method}"

    async def send_alert(self, alert: Alert) -> bool:
        """Send alert to Telegram.

        Args:
            alert: Alert to send.

        Returns:
            True if sent successfully.
        """
        try:
            import aiohttp
        except ImportError:
            logger.error("aiohttp required for Telegram alerts: pip install aiohttp")
            return False

        if self.parse_mode == "HTML":
            text = self._format_html(alert)
        else:
            text = self._format_markdown(alert)

        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": self.parse_mode,
            "disable_web_page_preview": True,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._get_api_url("sendMessage"),
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    data = await response.json()

                    if data.get("ok"):
                        logger.debug(f"Telegram alert sent: {alert.title}")
                        return True
                    else:
                        error = data.get("description", "Unknown error")
                        logger.error(f"Telegram error: {error}")
                        return False

        except asyncio.TimeoutError:
            logger.error("Telegram API timeout")
            return False
        except Exception as e:
            logger.error(f"Telegram error: {e}")
            return False

    async def send_photo(
        self,
        photo_url: str,
        caption: str = "",
    ) -> bool:
        """Send a photo with caption.

        Args:
            photo_url: URL of the photo.
            caption: Photo caption.

        Returns:
            True if sent successfully.
        """
        try:
            import aiohttp
        except ImportError:
            return False

        payload = {
            "chat_id": self.chat_id,
            "photo": photo_url,
            "caption": caption,
            "parse_mode": self.parse_mode,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._get_api_url("sendPhoto"),
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    data = await response.json()
                    return data.get("ok", False)
        except Exception as e:
            logger.error(f"Telegram photo error: {e}")
            return False

    async def send_document(
        self,
        document_url: str,
        caption: str = "",
        filename: str | None = None,
    ) -> bool:
        """Send a document file.

        Args:
            document_url: URL of the document.
            caption: Document caption.
            filename: Custom filename.

        Returns:
            True if sent successfully.
        """
        try:
            import aiohttp
        except ImportError:
            return False

        payload: dict[str, Any] = {
            "chat_id": self.chat_id,
            "document": document_url,
            "caption": caption,
            "parse_mode": self.parse_mode,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._get_api_url("sendDocument"),
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as response:
                    data = await response.json()
                    return data.get("ok", False)
        except Exception as e:
            logger.error(f"Telegram document error: {e}")
            return False

    async def test_connection(self) -> bool:
        """Test bot connection by getting bot info.

        Returns:
            True if connection successful.
        """
        try:
            import aiohttp
        except ImportError:
            return False

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self._get_api_url("getMe"),
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    data = await response.json()

                    if data.get("ok"):
                        bot_info = data.get("result", {})
                        logger.info(
                            f"Telegram bot connected: @{bot_info.get('username')}"
                        )
                        return True
                    else:
                        logger.error(f"Telegram auth failed: {data.get('description')}")
                        return False

        except Exception as e:
            logger.error(f"Telegram connection error: {e}")
            return False
