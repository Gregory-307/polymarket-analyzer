"""Tests for alerting system."""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, patch, MagicMock

from src.alerts.base import (
    Alert,
    AlertChannel,
    AlertHandler,
    AlertLevel,
    AlertManager,
    ConsoleHandler,
)
from src.alerts.discord import DiscordHandler
from src.alerts.telegram import TelegramHandler


class TestAlert:
    """Tests for Alert dataclass."""

    def test_default_values(self):
        """Alert should have sensible defaults."""
        alert = Alert(title="Test", message="Test message")

        assert alert.level == AlertLevel.INFO
        assert alert.source == "system"
        assert alert.data == {}
        assert alert.alert_id != ""

    def test_to_dict(self):
        """Alert should convert to dictionary."""
        alert = Alert(
            title="Test",
            message="Test message",
            level=AlertLevel.WARNING,
            source="test",
            data={"key": "value"},
        )

        d = alert.to_dict()

        assert d["title"] == "Test"
        assert d["message"] == "Test message"
        assert d["level"] == "warning"
        assert d["source"] == "test"
        assert d["data"] == {"key": "value"}

    def test_format_text(self):
        """Alert should format as plain text."""
        alert = Alert(
            title="Test Alert",
            message="This is a test",
            level=AlertLevel.ERROR,
        )

        text = alert.format_text()

        assert "[ERROR]" in text
        assert "Test Alert" in text
        assert "This is a test" in text

    def test_format_text_with_data(self):
        """Alert text should include data fields."""
        alert = Alert(
            title="Test",
            message="Message",
            data={"price": 0.85, "market": "BTC"},
        )

        text = alert.format_text()

        assert "price:" in text
        assert "0.8500" in text
        assert "market:" in text
        assert "BTC" in text

    def test_format_markdown(self):
        """Alert should format as markdown."""
        alert = Alert(
            title="Test Alert",
            message="This is a test",
            level=AlertLevel.INFO,
        )

        md = alert.format_markdown()

        assert "## " in md
        assert "Test Alert" in md


class TestAlertLevel:
    """Tests for AlertLevel enum."""

    def test_level_values(self):
        """Levels should have correct values."""
        assert AlertLevel.DEBUG.value == "debug"
        assert AlertLevel.INFO.value == "info"
        assert AlertLevel.WARNING.value == "warning"
        assert AlertLevel.ERROR.value == "error"
        assert AlertLevel.CRITICAL.value == "critical"


class TestConsoleHandler:
    """Tests for ConsoleHandler."""

    @pytest.mark.asyncio
    async def test_send_alert(self, capsys):
        """ConsoleHandler should print alert."""
        handler = ConsoleHandler()
        alert = Alert(title="Test", message="Test message")

        result = await handler.send_alert(alert)

        assert result is True
        captured = capsys.readouterr()
        assert "Test" in captured.out
        assert "Test message" in captured.out

    def test_channel_property(self):
        """ConsoleHandler should return CONSOLE channel."""
        handler = ConsoleHandler()
        assert handler.channel == AlertChannel.CONSOLE

    @pytest.mark.asyncio
    async def test_level_filtering(self):
        """Handler should filter by level."""
        handler = ConsoleHandler(min_level=AlertLevel.WARNING)

        # INFO should not pass
        info_alert = Alert(title="Info", message="Info msg", level=AlertLevel.INFO)
        assert handler.should_send(info_alert) is False

        # WARNING should pass
        warn_alert = Alert(title="Warn", message="Warn msg", level=AlertLevel.WARNING)
        assert handler.should_send(warn_alert) is True

    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Handler should rate limit alerts."""
        handler = ConsoleHandler(rate_limit_seconds=10.0)
        alert = Alert(title="Test", message="Test")

        # First should pass
        assert handler.should_send(alert) is True
        await handler.handle(alert)

        # Second should be rate limited
        assert handler.should_send(alert) is False

    def test_stats(self):
        """Handler should track statistics."""
        handler = ConsoleHandler()

        stats = handler.get_stats()

        assert stats["channel"] == "console"
        assert stats["alerts_sent"] == 0
        assert stats["alerts_dropped"] == 0


class TestAlertManager:
    """Tests for AlertManager."""

    @pytest.mark.asyncio
    async def test_add_handler(self):
        """Manager should add handlers."""
        manager = AlertManager()
        handler = ConsoleHandler()

        manager.add_handler(handler)

        assert len(manager._handlers) == 1

    @pytest.mark.asyncio
    async def test_remove_handler(self):
        """Manager should remove handlers by channel."""
        manager = AlertManager()
        manager.add_handler(ConsoleHandler())
        manager.add_handler(ConsoleHandler())

        manager.remove_handler(AlertChannel.CONSOLE)

        assert len(manager._handlers) == 0

    @pytest.mark.asyncio
    async def test_send_to_all_handlers(self, capsys):
        """Manager should send to all handlers."""
        manager = AlertManager()
        manager.add_handler(ConsoleHandler(rate_limit_seconds=0))
        manager.add_handler(ConsoleHandler(rate_limit_seconds=0))

        alert = Alert(title="Test", message="Test")
        results = await manager.send(alert)

        # Both should succeed
        assert AlertChannel.CONSOLE in results

    @pytest.mark.asyncio
    async def test_send_opportunity(self, capsys):
        """Manager should send opportunity alerts."""
        manager = AlertManager()
        manager.add_handler(ConsoleHandler(rate_limit_seconds=0))

        results = await manager.send_opportunity(
            market_name="BTC 100k",
            strategy="arbitrage",
            side="YES",
            price=0.85,
            profit=0.15,
        )

        assert AlertChannel.CONSOLE in results
        captured = capsys.readouterr()
        assert "BTC 100k" in captured.out
        assert "YES" in captured.out

    @pytest.mark.asyncio
    async def test_history_tracking(self):
        """Manager should track alert history."""
        manager = AlertManager()
        manager.add_handler(ConsoleHandler(rate_limit_seconds=0))

        alert = Alert(title="Test", message="Test")
        await manager.send(alert)

        history = manager.get_history()
        assert len(history) == 1
        assert history[0].title == "Test"

    @pytest.mark.asyncio
    async def test_history_filtering(self):
        """Manager should filter history."""
        manager = AlertManager()
        manager.add_handler(ConsoleHandler(rate_limit_seconds=0))

        await manager.send(Alert(title="Info", message="Info", level=AlertLevel.INFO))
        await manager.send(Alert(title="Error", message="Error", level=AlertLevel.ERROR))

        info_only = manager.get_history(level=AlertLevel.INFO)
        assert len(info_only) == 1
        assert info_only[0].title == "Info"

    def test_stats(self):
        """Manager should report statistics."""
        manager = AlertManager()
        manager.add_handler(ConsoleHandler())

        stats = manager.get_stats()

        assert "handlers" in stats
        assert "history_size" in stats
        assert len(stats["handlers"]) == 1


class TestDiscordHandler:
    """Tests for DiscordHandler."""

    def test_init(self):
        """DiscordHandler should initialize correctly."""
        handler = DiscordHandler(
            webhook_url="https://discord.com/api/webhooks/123/abc",
            username="Test Bot",
        )

        assert handler.webhook_url == "https://discord.com/api/webhooks/123/abc"
        assert handler.username == "Test Bot"
        assert handler.channel == AlertChannel.DISCORD

    def test_level_to_color(self):
        """Should convert levels to Discord colors."""
        handler = DiscordHandler(webhook_url="https://example.com")

        assert handler._level_to_color(AlertLevel.INFO) == 0x3498DB
        assert handler._level_to_color(AlertLevel.ERROR) == 0xE74C3C

    def test_build_embed(self):
        """Should build Discord embed structure."""
        handler = DiscordHandler(webhook_url="https://example.com")
        alert = Alert(
            title="Test",
            message="Test message",
            data={"price": 0.85},
        )

        embed = handler._build_embed(alert)

        assert embed["title"] == "Test"
        assert embed["description"] == "Test message"
        assert "fields" in embed
        assert len(embed["fields"]) == 1
        assert embed["fields"][0]["name"] == "Price"

    def test_build_payload(self):
        """Should build full webhook payload."""
        handler = DiscordHandler(
            webhook_url="https://example.com",
            username="Bot",
            avatar_url="https://avatar.com/img.png",
        )
        alert = Alert(title="Test", message="Test")

        payload = handler._build_payload(alert)

        assert payload["username"] == "Bot"
        assert payload["avatar_url"] == "https://avatar.com/img.png"
        assert "embeds" in payload

    @pytest.mark.asyncio
    async def test_send_alert_no_aiohttp(self, monkeypatch):
        """Should handle missing aiohttp gracefully."""
        handler = DiscordHandler(webhook_url="https://example.com")
        alert = Alert(title="Test", message="Test")

        # Mock import to fail
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "aiohttp":
                raise ImportError("No module named 'aiohttp'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        result = await handler.send_alert(alert)

        assert result is False


class TestTelegramHandler:
    """Tests for TelegramHandler."""

    def test_init(self):
        """TelegramHandler should initialize correctly."""
        handler = TelegramHandler(
            bot_token="123:ABC",
            chat_id="-100123",
        )

        assert handler.bot_token == "123:ABC"
        assert handler.chat_id == "-100123"
        assert handler.channel == AlertChannel.TELEGRAM

    def test_format_html(self):
        """Should format alert as HTML."""
        handler = TelegramHandler(bot_token="123:ABC", chat_id="123")
        alert = Alert(
            title="Test",
            message="Test message",
            level=AlertLevel.WARNING,
            data={"price": 0.85},
        )

        html = handler._format_html(alert)

        assert "<b>" in html
        assert "Test" in html
        assert "<pre>" in html
        assert "price:" in html

    def test_get_api_url(self):
        """Should construct correct API URL."""
        handler = TelegramHandler(bot_token="123:ABC", chat_id="123")

        url = handler._get_api_url("sendMessage")

        assert url == "https://api.telegram.org/bot123:ABC/sendMessage"

    def test_escape_markdown(self):
        """Should escape Markdown special characters."""
        handler = TelegramHandler(bot_token="123:ABC", chat_id="123")

        escaped = handler._escape_markdown("Hello_World*Test")

        assert "\\_" in escaped
        assert "\\*" in escaped

    @pytest.mark.asyncio
    async def test_send_alert_no_aiohttp(self, monkeypatch):
        """Should handle missing aiohttp gracefully."""
        handler = TelegramHandler(bot_token="123:ABC", chat_id="123")
        alert = Alert(title="Test", message="Test")

        # Mock import to fail
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "aiohttp":
                raise ImportError("No module named 'aiohttp'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        result = await handler.send_alert(alert)

        assert result is False
