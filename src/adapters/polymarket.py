"""Polymarket adapter implementation.

Integrates with Polymarket's CLOB (Central Limit Order Book) API
using the official py-clob-client SDK.

References:
- https://docs.polymarket.com/developers/CLOB/introduction
- https://github.com/Polymarket/py-clob-client
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx

from ..core.config import Credentials, PolymarketConfig
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


class PolymarketAdapter(BaseAdapter):
    """Adapter for Polymarket prediction market platform.

    Supports both read-only market data access and authenticated trading.

    Read-only operations (no auth required):
    - get_markets, get_market, get_order_book, get_trades, get_price

    Trading operations (require private key):
    - place_order, cancel_order, get_open_orders, get_balance

    Example:
        ```python
        # Read-only
        adapter = PolymarketAdapter()
        await adapter.connect()
        markets = await adapter.get_markets()

        # With trading
        adapter = PolymarketAdapter(credentials=Credentials.from_env())
        await adapter.connect()
        order = await adapter.place_order(market_id, Side.BUY, 0.55, 10)
        ```
    """

    PLATFORM_NAME = "polymarket"
    CLOB_URL = "https://clob.polymarket.com"
    GAMMA_URL = "https://gamma-api.polymarket.com"

    def __init__(
        self,
        config: PolymarketConfig | None = None,
        credentials: Credentials | None = None,
    ):
        """Initialize Polymarket adapter.

        Args:
            config: Platform configuration.
            credentials: API credentials for trading.
        """
        config = config or PolymarketConfig()
        super().__init__(config.base_url, config.timeout_seconds)

        self.gamma_url = config.gamma_url
        self.chain_id = config.chain_id
        self.credentials = credentials

        self._client: httpx.AsyncClient | None = None
        self._clob_client: Any = None  # py_clob_client.ClobClient

    async def connect(self) -> bool:
        """Establish connection to Polymarket.

        Initializes HTTP client and optionally the CLOB trading client
        if credentials are provided.

        Returns:
            True if connection successful.
        """
        try:
            self._client = httpx.AsyncClient(timeout=self.timeout)

            # Test connection with a simple request
            response = await self._client.get(f"{self.base_url}/")
            if response.status_code != 200:
                logger.warning(
                    "polymarket_connection_warning",
                    status_code=response.status_code,
                )

            # Initialize CLOB client for trading if credentials provided
            if self.credentials and self.credentials.has_polymarket:
                await self._init_clob_client()
                self._authenticated = True
                logger.info("polymarket_authenticated")
            else:
                logger.info("polymarket_connected_readonly")

            return True

        except Exception as e:
            logger.error("polymarket_connection_failed", error=str(e))
            return False

    async def _init_clob_client(self) -> None:
        """Initialize the official CLOB client for trading."""
        try:
            from py_clob_client.client import ClobClient

            self._clob_client = ClobClient(
                self.base_url,
                key=self.credentials.polymarket_private_key,
                chain_id=self.credentials.polymarket_chain_id,
                signature_type=self.credentials.polymarket_signature_type,
                funder=self.credentials.polymarket_funder_address,
            )

            # Derive API credentials
            self._clob_client.set_api_creds(
                self._clob_client.create_or_derive_api_creds()
            )

        except ImportError:
            logger.warning("py_clob_client not installed, trading disabled")
            self._clob_client = None

    async def disconnect(self) -> None:
        """Close connection and cleanup."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._clob_client = None
        self._authenticated = False

    async def get_markets(
        self,
        active_only: bool = True,
        category: str | None = None,
        limit: int = 100,
    ) -> list[Market]:
        """Fetch available markets from Polymarket.

        Uses the Gamma API for market metadata.

        Args:
            active_only: Only return active markets.
            category: Filter by category (e.g., 'politics', 'sports').
            limit: Maximum markets to return.

        Returns:
            List of Market objects.
        """
        if not self._client:
            raise RuntimeError("Not connected. Call connect() first.")

        params: dict[str, Any] = {"limit": limit}
        if active_only:
            params["closed"] = "false"
            params["active"] = "true"
        if category:
            params["tag"] = category

        try:
            response = await self._client.get(
                f"{self.gamma_url}/markets",
                params=params,
            )
            response.raise_for_status()
            data = response.json()

            markets = []
            for item in data:
                market = self._parse_market(item)
                if market:
                    markets.append(market)

            logger.debug("polymarket_markets_fetched", count=len(markets))
            return markets

        except Exception as e:
            logger.error("polymarket_get_markets_failed", error=str(e))
            return []

    async def get_events(
        self,
        active_only: bool = True,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Fetch events (grouped markets) from Polymarket.

        Events contain multiple related markets that are mutually exclusive,
        which is needed for multi-outcome arbitrage detection.

        Args:
            active_only: Only return active events.
            limit: Maximum events to return.

        Returns:
            List of event data dictionaries with nested markets.
        """
        if not self._client:
            raise RuntimeError("Not connected. Call connect() first.")

        params: dict[str, Any] = {"limit": limit}
        if active_only:
            params["closed"] = "false"
            params["active"] = "true"

        try:
            response = await self._client.get(
                f"{self.gamma_url}/events",
                params=params,
            )
            response.raise_for_status()
            data = response.json()

            logger.debug("polymarket_events_fetched", count=len(data))
            return data

        except Exception as e:
            logger.error("polymarket_get_events_failed", error=str(e))
            return []

    async def get_market(self, market_id: str) -> Market | None:
        """Fetch a specific market by ID.

        Args:
            market_id: Polymarket condition ID or slug.

        Returns:
            Market object or None if not found.
        """
        if not self._client:
            raise RuntimeError("Not connected. Call connect() first.")

        try:
            response = await self._client.get(
                f"{self.gamma_url}/markets/{market_id}"
            )

            if response.status_code == 404:
                return None

            response.raise_for_status()
            return self._parse_market(response.json())

        except Exception as e:
            logger.error("polymarket_get_market_failed", market_id=market_id, error=str(e))
            return None

    async def get_order_book(self, market_id: str) -> OrderBook:
        """Fetch order book for a market.

        Args:
            market_id: Token ID for the YES or NO outcome.

        Returns:
            OrderBook object with bids and asks.
        """
        if not self._client:
            raise RuntimeError("Not connected. Call connect() first.")

        try:
            response = await self._client.get(
                f"{self.base_url}/book",
                params={"token_id": market_id},
            )
            response.raise_for_status()
            data = response.json()

            bids = [
                OrderBookLevel(
                    price=float(level["price"]),
                    size=float(level["size"]),
                    side=Side.BUY,
                )
                for level in data.get("bids", [])
            ]

            asks = [
                OrderBookLevel(
                    price=float(level["price"]),
                    size=float(level["size"]),
                    side=Side.SELL,
                )
                for level in data.get("asks", [])
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
            logger.error("polymarket_get_orderbook_failed", market_id=market_id, error=str(e))
            return OrderBook(market_id=market_id, platform=self.platform)

    async def get_trades(
        self,
        market_id: str,
        limit: int = 100,
    ) -> list[Trade]:
        """Fetch recent trades for a market.

        Args:
            market_id: Token ID.
            limit: Maximum trades to return.

        Returns:
            List of Trade objects.
        """
        if not self._client:
            raise RuntimeError("Not connected. Call connect() first.")

        try:
            response = await self._client.get(
                f"{self.base_url}/trades",
                params={"token_id": market_id, "limit": limit},
            )
            response.raise_for_status()
            data = response.json()

            trades = []
            for item in data:
                trade = Trade(
                    id=item.get("id", ""),
                    market_id=market_id,
                    platform=self.platform,
                    side=Side.BUY if item.get("side") == "BUY" else Side.SELL,
                    price=float(item.get("price", 0)),
                    size=float(item.get("size", 0)),
                    timestamp=datetime.fromisoformat(
                        item.get("timestamp", "").replace("Z", "+00:00")
                    ) if item.get("timestamp") else datetime.now(timezone.utc),
                    raw=item,
                )
                trades.append(trade)

            return trades

        except Exception as e:
            logger.error("polymarket_get_trades_failed", market_id=market_id, error=str(e))
            return []

    async def get_price(self, market_id: str) -> tuple[float, float]:
        """Get current YES and NO prices.

        Args:
            market_id: Token ID for YES outcome.

        Returns:
            Tuple of (yes_price, no_price).
        """
        if not self._client:
            raise RuntimeError("Not connected. Call connect() first.")

        try:
            response = await self._client.get(
                f"{self.base_url}/midpoint",
                params={"token_id": market_id},
            )
            response.raise_for_status()
            data = response.json()

            yes_price = float(data.get("mid", 0.5))
            no_price = 1.0 - yes_price

            return (yes_price, no_price)

        except Exception as e:
            logger.error("polymarket_get_price_failed", market_id=market_id, error=str(e))
            return (0.5, 0.5)

    async def place_order(
        self,
        market_id: str,
        side: Side,
        price: float,
        size: float,
        order_type: OrderType = OrderType.LIMIT,
    ) -> Order:
        """Place an order on Polymarket.

        Requires authentication.

        Args:
            market_id: Token ID.
            side: BUY or SELL.
            price: Limit price (0-1).
            size: Number of shares.
            order_type: LIMIT or MARKET.

        Returns:
            Created Order object.

        Raises:
            RuntimeError: If not authenticated.
        """
        if not self._authenticated or not self._clob_client:
            raise RuntimeError("Trading requires authentication. Provide credentials.")

        try:
            from py_clob_client.order_builder.constants import BUY, SELL

            clob_side = BUY if side == Side.BUY else SELL

            if order_type == OrderType.MARKET:
                from py_clob_client.clob_types import MarketOrderArgs, OrderType as ClobOrderType

                order_args = MarketOrderArgs(
                    token_id=market_id,
                    amount=size * price,  # USD amount for market orders
                    side=clob_side,
                )
                signed_order = self._clob_client.create_market_order(order_args)
                result = self._clob_client.post_order(signed_order, ClobOrderType.FOK)
            else:
                from py_clob_client.clob_types import OrderArgs

                order_args = OrderArgs(
                    token_id=market_id,
                    price=price,
                    size=size,
                    side=clob_side,
                )
                signed_order = self._clob_client.create_order(order_args)
                result = self._clob_client.post_order(signed_order)

            return Order(
                id=result.get("orderID", ""),
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
            logger.error("polymarket_place_order_failed", error=str(e))
            raise

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order.

        Args:
            order_id: Order ID to cancel.

        Returns:
            True if cancelled successfully.
        """
        if not self._authenticated or not self._clob_client:
            raise RuntimeError("Trading requires authentication.")

        try:
            result = self._clob_client.cancel(order_id)
            return result.get("canceled", False)
        except Exception as e:
            logger.error("polymarket_cancel_order_failed", order_id=order_id, error=str(e))
            return False

    async def get_open_orders(self, market_id: str | None = None) -> list[Order]:
        """Get open orders.

        Args:
            market_id: Filter by market (optional).

        Returns:
            List of open orders.
        """
        if not self._authenticated or not self._clob_client:
            raise RuntimeError("Trading requires authentication.")

        try:
            params = {}
            if market_id:
                params["asset_id"] = market_id

            result = self._clob_client.get_orders(params)

            orders = []
            for item in result:
                orders.append(Order(
                    id=item.get("id", ""),
                    market_id=item.get("asset_id", ""),
                    platform=self.platform,
                    side=Side.BUY if item.get("side") == "BUY" else Side.SELL,
                    order_type=OrderType.LIMIT,
                    price=float(item.get("price", 0)),
                    size=float(item.get("original_size", 0)),
                    filled_size=float(item.get("size_matched", 0)),
                    status=OrderStatus.OPEN,
                    raw=item,
                ))

            return orders

        except Exception as e:
            logger.error("polymarket_get_orders_failed", error=str(e))
            return []

    async def get_balance(self) -> float:
        """Get USDC balance on Polygon.

        Returns:
            Available balance in USD.
        """
        if not self._authenticated or not self._clob_client:
            raise RuntimeError("Trading requires authentication.")

        # Note: Balance checking requires Web3 integration
        # This is a placeholder - actual implementation would query Polygon
        logger.warning("polymarket_balance_not_implemented")
        return 0.0

    def _parse_market(self, data: dict[str, Any]) -> Market | None:
        """Parse Gamma API market response into Market object.

        Args:
            data: Raw market data from Gamma API.

        Returns:
            Parsed Market object or None.
        """
        try:
            import json as json_mod

            # Handle different response formats
            condition_id = data.get("conditionId") or data.get("condition_id") or data.get("id")
            if not condition_id:
                return None

            # Parse outcomePrices (primary source for prices)
            yes_price = 0.5
            no_price = 0.5

            outcome_prices = data.get("outcomePrices")
            if outcome_prices:
                try:
                    prices = json_mod.loads(outcome_prices) if isinstance(outcome_prices, str) else outcome_prices
                    if isinstance(prices, list) and len(prices) >= 2:
                        yes_price = float(prices[0])
                        no_price = float(prices[1])
                except (json_mod.JSONDecodeError, ValueError, IndexError):
                    pass

            # Fallback to tokens array if outcomePrices not available
            if yes_price == 0.5 and no_price == 0.5:
                tokens = data.get("tokens", [])
                for token in tokens:
                    outcome = token.get("outcome", "").upper()
                    price = float(token.get("price", 0.5))
                    if outcome == "YES":
                        yes_price = price
                    elif outcome == "NO":
                        no_price = price

            # Parse end date
            end_date = None
            end_str = data.get("endDate") or data.get("end_date_iso")
            if end_str:
                try:
                    end_date = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    pass

            return Market(
                id=condition_id,
                platform=self.platform,
                question=data.get("question", ""),
                description=data.get("description", ""),
                category=data.get("groupSlug") or data.get("category", ""),
                yes_price=yes_price,
                no_price=no_price,
                volume=float(data.get("volume", 0) or 0),
                liquidity=float(data.get("liquidity", 0) or 0),
                end_date=end_date,
                is_active=data.get("active", True),
                raw=data,
            )

        except Exception as e:
            logger.warning("polymarket_parse_market_failed", error=str(e))
            return None
