"""WebSocket client for real-time Polymarket data.

Provides streaming price updates and trade notifications
for lower-latency data collection than REST polling.

Key constraints from Polymarket docs:
- Max 500 instruments per WebSocket connection
- Cannot unsubscribe once subscribed
- Need ping/pong for keepalive
- Exponential backoff for reconnection
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Any

import websockets
from websockets.asyncio.client import ClientConnection

from ..core.utils import get_logger

logger = get_logger(__name__)


@dataclass
class PriceUpdate:
    """Real-time price update from WebSocket.

    Attributes:
        token_id: Asset/token identifier.
        price: Updated price.
        timestamp: When update was received.
        side: 'bid' or 'ask'.
        size: Order size at this price.
    """

    token_id: str
    price: float
    timestamp: datetime
    side: str  # 'bid' or 'ask'
    size: float


@dataclass
class TradeUpdate:
    """Real-time trade notification.

    Attributes:
        token_id: Asset/token identifier.
        price: Trade price.
        size: Trade size.
        side: 'buy' or 'sell'.
        timestamp: Trade timestamp.
    """

    token_id: str
    price: float
    size: float
    side: str  # 'buy' or 'sell'
    timestamp: datetime


class PolymarketWebSocket:
    """Real-time price updates via Polymarket WebSocket.

    Usage:
        async def on_price(update: PriceUpdate):
            print(f"Price: {update.token_id} = {update.price}")

        async def on_trade(update: TradeUpdate):
            print(f"Trade: {update.token_id} {update.side} {update.size}@{update.price}")

        ws = PolymarketWebSocket(on_price_update=on_price, on_trade=on_trade)
        await ws.connect()
        await ws.subscribe_to_markets(["token_123", "token_456"])
        await ws.run_forever()
    """

    MARKET_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    USER_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/user"

    MAX_INSTRUMENTS = 500
    PING_INTERVAL = 30  # seconds
    RECONNECT_BASE_DELAY = 1  # seconds
    RECONNECT_MAX_DELAY = 60  # seconds

    def __init__(
        self,
        on_price_update: Callable[[PriceUpdate], Any] | None = None,
        on_trade: Callable[[TradeUpdate], Any] | None = None,
        on_error: Callable[[Exception], Any] | None = None,
    ):
        """Initialize WebSocket client.

        Args:
            on_price_update: Callback for price updates.
            on_trade: Callback for trade notifications.
            on_error: Callback for errors.
        """
        self._on_price = on_price_update
        self._on_trade = on_trade
        self._on_error = on_error

        self._ws: ClientConnection | None = None
        self._subscribed_tokens: set[str] = set()
        self._running = False
        self._reconnect_delay = self.RECONNECT_BASE_DELAY

    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._ws is not None and self._ws.state.name == "OPEN"

    @property
    def subscribed_count(self) -> int:
        """Number of subscribed tokens."""
        return len(self._subscribed_tokens)

    async def connect(self) -> None:
        """Establish WebSocket connection."""
        try:
            self._ws = await websockets.connect(
                self.MARKET_WS,
                ping_interval=self.PING_INTERVAL,
                ping_timeout=10,
            )
            self._reconnect_delay = self.RECONNECT_BASE_DELAY
            logger.info("polymarket_ws_connected")
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            raise

    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        self._running = False
        if self._ws:
            await self._ws.close()
            self._ws = None
        logger.info("polymarket_ws_disconnected")

    async def subscribe_to_markets(self, token_ids: list[str]) -> None:
        """Subscribe to orderbook updates for given token IDs.

        Args:
            token_ids: List of token IDs to subscribe to.

        Raises:
            ValueError: If exceeding max instrument limit.
            RuntimeError: If not connected.
        """
        if not self._ws:
            raise RuntimeError("Not connected. Call connect() first.")

        new_tokens = set(token_ids) - self._subscribed_tokens

        if len(self._subscribed_tokens) + len(new_tokens) > self.MAX_INSTRUMENTS:
            raise ValueError(
                f"Cannot subscribe to {len(new_tokens)} more tokens. "
                f"Already at {len(self._subscribed_tokens)}/{self.MAX_INSTRUMENTS}"
            )

        if not new_tokens:
            return

        # Subscribe message format per Polymarket docs
        # Note: Actual format may differ - this follows common patterns
        for token_id in new_tokens:
            subscribe_msg = {
                "type": "subscribe",
                "channel": "market",
                "assets_id": token_id,
            }
            await self._ws.send(json.dumps(subscribe_msg))

        self._subscribed_tokens.update(new_tokens)
        logger.info(f"polymarket_ws_subscribed", count=len(new_tokens))

    async def run_forever(self, reconnect: bool = True) -> None:
        """Main event loop with automatic reconnection.

        Args:
            reconnect: Whether to auto-reconnect on disconnect.
        """
        self._running = True

        while self._running:
            try:
                if not self.is_connected:
                    await self.connect()
                    # Re-subscribe after reconnect
                    if self._subscribed_tokens:
                        tokens = list(self._subscribed_tokens)
                        self._subscribed_tokens.clear()
                        await self.subscribe_to_markets(tokens)

                async for message in self._ws:
                    await self._handle_message(message)

            except websockets.ConnectionClosed as e:
                logger.warning(f"WebSocket closed: code={e.code}")
                self._ws = None

                if not reconnect or not self._running:
                    break

                # Exponential backoff
                logger.info(f"Reconnecting in {self._reconnect_delay}s...")
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(
                    self._reconnect_delay * 2, self.RECONNECT_MAX_DELAY
                )

            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                if self._on_error:
                    await self._maybe_await(self._on_error(e))

                if not reconnect or not self._running:
                    break

                await asyncio.sleep(self._reconnect_delay)

    async def _handle_message(self, raw_message: str) -> None:
        """Parse and dispatch WebSocket message.

        Args:
            raw_message: Raw JSON message from WebSocket.
        """
        try:
            message = json.loads(raw_message)
            msg_type = message.get("type", message.get("event_type", ""))

            if msg_type in ("price_change", "book"):
                if self._on_price:
                    update = PriceUpdate(
                        token_id=message.get("asset_id", message.get("market", "")),
                        price=float(message.get("price", 0)),
                        timestamp=datetime.now(timezone.utc),
                        side=message.get("side", "unknown"),
                        size=float(message.get("size", 0)),
                    )
                    await self._maybe_await(self._on_price(update))

            elif msg_type == "trade":
                if self._on_trade:
                    update = TradeUpdate(
                        token_id=message.get("asset_id", message.get("market", "")),
                        price=float(message.get("price", 0)),
                        size=float(message.get("size", 0)),
                        side=message.get("side", "unknown"),
                        timestamp=datetime.now(timezone.utc),
                    )
                    await self._maybe_await(self._on_trade(update))

            elif msg_type in ("subscribed", "subscription"):
                logger.debug(f"Subscription confirmed: {message}")

            elif msg_type == "error":
                logger.error(f"WebSocket error message: {message}")

            else:
                # Unknown message type - log for debugging
                logger.debug(f"Unknown message type: {msg_type}")

        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON: {e}")
        except Exception as e:
            logger.error(f"Message handling error: {e}")

    async def _maybe_await(self, result: Any) -> Any:
        """Await result if it's a coroutine."""
        if asyncio.iscoroutine(result):
            return await result
        return result

    def get_status(self) -> dict:
        """Get WebSocket client status.

        Returns:
            Dictionary with connection status.
        """
        return {
            "connected": self.is_connected,
            "subscribed_tokens": self.subscribed_count,
            "running": self._running,
            "reconnect_delay": self._reconnect_delay,
        }


async def create_price_collector(
    token_ids: list[str],
    callback: Callable[[PriceUpdate], Any],
) -> PolymarketWebSocket:
    """Create a WebSocket client for collecting prices.

    Convenience function to quickly set up price collection.

    Args:
        token_ids: Token IDs to subscribe to.
        callback: Function to call with price updates.

    Returns:
        Connected and subscribed WebSocket client.

    Example:
        async def save_price(update: PriceUpdate):
            print(f"{update.token_id}: {update.price}")

        ws = await create_price_collector(["token1", "token2"], save_price)
        await ws.run_forever()
    """
    ws = PolymarketWebSocket(on_price_update=callback)
    await ws.connect()
    await ws.subscribe_to_markets(token_ids)
    return ws
