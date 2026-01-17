"""Kalshi adapter implementation.

Integrates with Kalshi's REST API using the official kalshi-python SDK.

References:
- https://docs.kalshi.com/welcome
- https://docs.kalshi.com/python-sdk
- https://pypi.org/project/kalshi-python/
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from ..core.config import Credentials, KalshiConfig
from ..core.utils import get_logger
from .base import (
    BaseAdapter,
    Market,
    Order,
    OrderBook,
    OrderBookLevel,
    OrderStatus,
    OrderType,
    Side,
    Trade,
)

logger = get_logger(__name__)


class KalshiAdapter(BaseAdapter):
    """Adapter for Kalshi prediction market platform.

    Supports both read-only market data access and authenticated trading.

    Note: Kalshi uses RSA key authentication with 30-minute token expiry.

    Example:
        ```python
        # Read-only
        adapter = KalshiAdapter()
        await adapter.connect()
        markets = await adapter.get_markets()

        # With trading
        adapter = KalshiAdapter(credentials=Credentials.from_env())
        await adapter.connect()
        order = await adapter.place_order(market_id, Side.BUY, 0.55, 10)
        ```
    """

    PLATFORM_NAME = "kalshi"
    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

    def __init__(
        self,
        config: KalshiConfig | None = None,
        credentials: Credentials | None = None,
    ):
        """Initialize Kalshi adapter.

        Args:
            config: Platform configuration.
            credentials: API credentials for trading.
        """
        config = config or KalshiConfig()
        super().__init__(config.base_url, config.timeout_seconds)

        self.credentials = credentials
        self._client: httpx.AsyncClient | None = None
        self._kalshi_client: Any = None  # kalshi.KalshiClient

    async def connect(self) -> bool:
        """Establish connection to Kalshi.

        Initializes HTTP client and optionally authenticates
        if credentials are provided.

        Returns:
            True if connection successful.
        """
        try:
            self._client = httpx.AsyncClient(timeout=self.timeout)

            # Test connection
            response = await self._client.get(f"{self.base_url}/exchange/status")
            if response.status_code != 200:
                logger.warning(
                    "kalshi_connection_warning",
                    status_code=response.status_code,
                )

            # Initialize Kalshi client for trading if credentials provided
            if self.credentials and self.credentials.has_kalshi:
                await self._init_kalshi_client()
                self._authenticated = True
                logger.info("kalshi_authenticated")
            else:
                logger.info("kalshi_connected_readonly")

            return True

        except Exception as e:
            logger.error("kalshi_connection_failed", error=str(e))
            return False

    async def _init_kalshi_client(self) -> None:
        """Initialize the official Kalshi client for trading."""
        try:
            from kalshi import Configuration, KalshiClient

            # Load private key
            key_path = Path(self.credentials.kalshi_private_key_path)
            if not key_path.exists():
                logger.warning("kalshi_private_key_not_found", path=str(key_path))
                return

            private_key = key_path.read_text()

            config = Configuration(host=self.base_url)
            config.api_key_id = self.credentials.kalshi_api_key_id
            config.private_key_pem = private_key

            self._kalshi_client = KalshiClient(config)

        except ImportError:
            logger.warning("kalshi-python not installed, trading disabled")
            self._kalshi_client = None
        except Exception as e:
            logger.error("kalshi_client_init_failed", error=str(e))
            self._kalshi_client = None

    async def disconnect(self) -> None:
        """Close connection and cleanup."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._kalshi_client = None
        self._authenticated = False

    async def get_markets(
        self,
        active_only: bool = True,
        category: str | None = None,
        limit: int = 100,
        exclude_sports_parlays: bool = True,
    ) -> list[Market]:
        """Fetch available markets from Kalshi.

        Args:
            active_only: Only return active markets.
            category: Filter by series ticker prefix.
            limit: Maximum markets to return.
            exclude_sports_parlays: Exclude KXMVE* sports parlay markets.

        Returns:
            List of Market objects.
        """
        if not self._client:
            raise RuntimeError("Not connected. Call connect() first.")

        # Fetch more than needed to account for filtering
        fetch_limit = limit * 5 if exclude_sports_parlays else limit

        params: dict[str, Any] = {"limit": fetch_limit}
        if active_only:
            params["status"] = "open"
        if category:
            params["series_ticker"] = category

        try:
            all_markets = []
            cursor = None

            # Paginate to get enough non-sports markets
            while len(all_markets) < limit:
                if cursor:
                    params["cursor"] = cursor

                response = await self._client.get(
                    f"{self.base_url}/markets",
                    params=params,
                )
                response.raise_for_status()
                data = response.json()

                for item in data.get("markets", []):
                    # Skip sports parlays if requested
                    ticker = item.get("ticker", "")
                    if exclude_sports_parlays and ticker.startswith("KXMVE"):
                        continue

                    market = self._parse_market(item)
                    if market:
                        all_markets.append(market)

                    if len(all_markets) >= limit:
                        break

                cursor = data.get("cursor")
                if not cursor:
                    break

            logger.debug("kalshi_markets_fetched", count=len(all_markets))
            return all_markets[:limit]

        except Exception as e:
            logger.error("kalshi_get_markets_failed", error=str(e))
            return []

    async def get_market(self, market_id: str) -> Market | None:
        """Fetch a specific market by ticker.

        Args:
            market_id: Kalshi market ticker.

        Returns:
            Market object or None if not found.
        """
        if not self._client:
            raise RuntimeError("Not connected. Call connect() first.")

        try:
            response = await self._client.get(
                f"{self.base_url}/markets/{market_id}"
            )

            if response.status_code == 404:
                return None

            response.raise_for_status()
            data = response.json()
            return self._parse_market(data.get("market", {}))

        except Exception as e:
            logger.error("kalshi_get_market_failed", market_id=market_id, error=str(e))
            return None

    async def get_order_book(self, market_id: str) -> OrderBook:
        """Fetch order book for a market.

        Args:
            market_id: Market ticker.

        Returns:
            OrderBook object with bids and asks.
        """
        if not self._client:
            raise RuntimeError("Not connected. Call connect() first.")

        try:
            response = await self._client.get(
                f"{self.base_url}/markets/{market_id}/orderbook"
            )
            response.raise_for_status()
            data = response.json()

            orderbook = data.get("orderbook", {})

            bids = [
                OrderBookLevel(
                    price=float(level[0]) / 100,  # Kalshi uses cents
                    size=float(level[1]),
                    side=Side.BUY,
                )
                for level in orderbook.get("yes", [])
            ]

            asks = [
                OrderBookLevel(
                    price=1.0 - float(level[0]) / 100,  # NO price = 1 - YES price
                    size=float(level[1]),
                    side=Side.SELL,
                )
                for level in orderbook.get("no", [])
            ]

            # Sort: bids descending, asks ascending
            bids.sort(key=lambda x: x.price, reverse=True)
            asks.sort(key=lambda x: x.price)

            return OrderBook(
                market_id=market_id,
                platform=self.platform,
                bids=bids,
                asks=asks,
            )

        except Exception as e:
            logger.error("kalshi_get_orderbook_failed", market_id=market_id, error=str(e))
            return OrderBook(market_id=market_id, platform=self.platform)

    async def get_trades(
        self,
        market_id: str,
        limit: int = 100,
    ) -> list[Trade]:
        """Fetch recent trades for a market.

        Args:
            market_id: Market ticker.
            limit: Maximum trades to return.

        Returns:
            List of Trade objects.
        """
        if not self._client:
            raise RuntimeError("Not connected. Call connect() first.")

        try:
            response = await self._client.get(
                f"{self.base_url}/markets/{market_id}/trades",
                params={"limit": limit},
            )
            response.raise_for_status()
            data = response.json()

            trades = []
            for item in data.get("trades", []):
                trade = Trade(
                    id=item.get("trade_id", ""),
                    market_id=market_id,
                    platform=self.platform,
                    side=Side.BUY if item.get("taker_side") == "yes" else Side.SELL,
                    price=float(item.get("yes_price", 50)) / 100,
                    size=float(item.get("count", 0)),
                    timestamp=datetime.fromisoformat(
                        item.get("created_time", "").replace("Z", "+00:00")
                    ) if item.get("created_time") else datetime.now(timezone.utc),
                    raw=item,
                )
                trades.append(trade)

            return trades

        except Exception as e:
            logger.error("kalshi_get_trades_failed", market_id=market_id, error=str(e))
            return []

    async def get_price(self, market_id: str) -> tuple[float, float]:
        """Get current YES and NO prices.

        Args:
            market_id: Market ticker.

        Returns:
            Tuple of (yes_price, no_price).
        """
        if not self._client:
            raise RuntimeError("Not connected. Call connect() first.")

        try:
            response = await self._client.get(
                f"{self.base_url}/markets/{market_id}"
            )
            response.raise_for_status()
            data = response.json()

            market = data.get("market", {})
            yes_price = float(market.get("yes_bid", 50)) / 100
            no_price = 1.0 - yes_price

            return (yes_price, no_price)

        except Exception as e:
            logger.error("kalshi_get_price_failed", market_id=market_id, error=str(e))
            return (0.5, 0.5)

    async def place_order(
        self,
        market_id: str,
        side: Side,
        price: float,
        size: float,
        order_type: OrderType = OrderType.LIMIT,
    ) -> Order:
        """Place an order on Kalshi.

        Requires authentication.

        Args:
            market_id: Market ticker.
            side: BUY or SELL.
            price: Limit price (0-1).
            size: Number of contracts.
            order_type: LIMIT or MARKET.

        Returns:
            Created Order object.

        Raises:
            RuntimeError: If not authenticated.
        """
        if not self._authenticated or not self._kalshi_client:
            raise RuntimeError("Trading requires authentication. Provide credentials.")

        try:
            # Convert price to cents
            price_cents = int(price * 100)

            order_params = {
                "ticker": market_id,
                "side": "yes" if side == Side.BUY else "no",
                "count": int(size),
                "type": "market" if order_type == OrderType.MARKET else "limit",
            }

            if order_type == OrderType.LIMIT:
                order_params["yes_price"] = price_cents

            result = self._kalshi_client.markets.create_order(**order_params)

            return Order(
                id=result.get("order", {}).get("order_id", ""),
                market_id=market_id,
                platform=self.platform,
                side=side,
                order_type=order_type,
                price=price,
                size=size,
                status=OrderStatus.OPEN,
                raw=result,
            )

        except Exception as e:
            logger.error("kalshi_place_order_failed", error=str(e))
            raise

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order.

        Args:
            order_id: Order ID to cancel.

        Returns:
            True if cancelled successfully.
        """
        if not self._authenticated or not self._kalshi_client:
            raise RuntimeError("Trading requires authentication.")

        try:
            self._kalshi_client.markets.cancel_order(order_id=order_id)
            return True
        except Exception as e:
            logger.error("kalshi_cancel_order_failed", order_id=order_id, error=str(e))
            return False

    async def get_open_orders(self, market_id: str | None = None) -> list[Order]:
        """Get open orders.

        Args:
            market_id: Filter by market (optional).

        Returns:
            List of open orders.
        """
        if not self._authenticated or not self._kalshi_client:
            raise RuntimeError("Trading requires authentication.")

        try:
            params = {"status": "resting"}
            if market_id:
                params["ticker"] = market_id

            result = self._kalshi_client.portfolio.get_orders(**params)

            orders = []
            for item in result.get("orders", []):
                orders.append(Order(
                    id=item.get("order_id", ""),
                    market_id=item.get("ticker", ""),
                    platform=self.platform,
                    side=Side.BUY if item.get("side") == "yes" else Side.SELL,
                    order_type=OrderType.LIMIT,
                    price=float(item.get("yes_price", 0)) / 100,
                    size=float(item.get("remaining_count", 0)),
                    filled_size=float(item.get("filled_count", 0)),
                    status=OrderStatus.OPEN,
                    raw=item,
                ))

            return orders

        except Exception as e:
            logger.error("kalshi_get_orders_failed", error=str(e))
            return []

    async def get_balance(self) -> float:
        """Get account balance.

        Returns:
            Available balance in USD.
        """
        if not self._authenticated or not self._kalshi_client:
            raise RuntimeError("Trading requires authentication.")

        try:
            result = self._kalshi_client.portfolio.get_balance()
            return float(result.get("balance", 0)) / 100  # Cents to dollars
        except Exception as e:
            logger.error("kalshi_get_balance_failed", error=str(e))
            return 0.0

    def _parse_market(self, data: dict[str, Any]) -> Market | None:
        """Parse Kalshi API market response into Market object.

        Args:
            data: Raw market data from API.

        Returns:
            Parsed Market object or None.
        """
        try:
            ticker = data.get("ticker")
            if not ticker:
                return None

            # Parse prices (Kalshi uses cents)
            yes_price = float(data.get("yes_bid", 50)) / 100
            no_price = float(data.get("no_bid", 50)) / 100

            # If no bids, use last price
            if yes_price == 0:
                yes_price = float(data.get("last_price", 50)) / 100

            # Parse end date
            end_date = None
            close_time = data.get("close_time")
            if close_time:
                try:
                    end_date = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    pass

            return Market(
                id=ticker,
                platform=self.platform,
                question=data.get("title", ""),
                description=data.get("subtitle", ""),
                category=data.get("category", ""),
                yes_price=yes_price,
                no_price=no_price,
                volume=float(data.get("volume", 0) or 0),
                liquidity=float(data.get("open_interest", 0) or 0),
                end_date=end_date,
                is_active=data.get("status") == "open",
                raw=data,
            )

        except Exception as e:
            logger.warning("kalshi_parse_market_failed", error=str(e))
            return None
