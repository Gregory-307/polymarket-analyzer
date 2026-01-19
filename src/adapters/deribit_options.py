"""Deribit Options API for implied volatility and pricing data.

Provides clean interface to Deribit's options chain for:
- ATM implied volatility by expiry
- Volatility surface interpolation
- Spot/index prices

Used by the financial markets strategy to price prediction market
threshold events against options-implied probabilities.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

from ..core.utils import get_logger

logger = get_logger(__name__)

DERIBIT_BASE = "https://www.deribit.com/api/v2"


@dataclass
class OptionInstrument:
    """Single option instrument from Deribit."""

    name: str
    strike: float
    expiry_ts: int  # milliseconds
    option_type: str  # "call" or "put"
    underlying: str  # "BTC" or "ETH"

    @property
    def expiry(self) -> datetime:
        return datetime.fromtimestamp(self.expiry_ts / 1000, tz=timezone.utc)

    @property
    def days_to_expiry(self) -> float:
        delta = self.expiry - datetime.now(timezone.utc)
        return max(0, delta.total_seconds() / 86400)


@dataclass
class VolPoint:
    """Single point on the volatility surface."""

    strike: float
    expiry_days: float
    iv: float  # Implied volatility as decimal (0.55 = 55%)
    bid_iv: float | None = None
    ask_iv: float | None = None


@dataclass
class VolSurface:
    """Volatility surface for an underlying."""

    underlying: str
    spot: float
    points: list[VolPoint] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def get_atm_iv(self, days_to_expiry: float, tolerance_days: float = 7) -> float | None:
        """Get ATM implied volatility for target expiry.

        Args:
            days_to_expiry: Target days to expiry.
            tolerance_days: Maximum days difference to accept.

        Returns:
            ATM IV as decimal, or None if not found.
        """
        if not self.points:
            return None

        # Find points near ATM (within 5% of spot)
        atm_range = self.spot * 0.05
        atm_points = [
            p for p in self.points
            if abs(p.strike - self.spot) <= atm_range
        ]

        if not atm_points:
            return None

        # Find closest expiry
        closest = min(atm_points, key=lambda p: abs(p.expiry_days - days_to_expiry))

        if abs(closest.expiry_days - days_to_expiry) > tolerance_days:
            return None

        return closest.iv

    def interpolate_iv(self, strike: float, days_to_expiry: float) -> float | None:
        """Interpolate IV for arbitrary strike and expiry.

        Simple bilinear interpolation. For production, use SABR or SVI.

        Args:
            strike: Target strike price.
            days_to_expiry: Target days to expiry.

        Returns:
            Interpolated IV, or None if extrapolation required.
        """
        if not self.points:
            return None

        # Find bracketing points
        lower_expiry = [p for p in self.points if p.expiry_days <= days_to_expiry]
        upper_expiry = [p for p in self.points if p.expiry_days >= days_to_expiry]

        if not lower_expiry or not upper_expiry:
            # Can't interpolate, return closest ATM
            return self.get_atm_iv(days_to_expiry, tolerance_days=30)

        # Get closest expiry on each side
        lower = max(lower_expiry, key=lambda p: p.expiry_days)
        upper = min(upper_expiry, key=lambda p: p.expiry_days)

        # Simple average for now (proper implementation would do full surface interp)
        if lower.expiry_days == upper.expiry_days:
            return lower.iv

        # Linear interpolation on expiry
        weight = (days_to_expiry - lower.expiry_days) / (upper.expiry_days - lower.expiry_days)
        return lower.iv * (1 - weight) + upper.iv * weight


class DeribitOptionsClient:
    """Client for Deribit options data.

    Provides spot prices and implied volatility from the options chain.

    Example:
        client = DeribitOptionsClient()
        await client.connect()

        spot = await client.get_spot("BTC")
        iv = await client.get_atm_iv("BTC", days=30)

        await client.disconnect()
    """

    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._cache: dict[str, tuple[Any, float]] = {}
        self._cache_ttl = 60  # seconds

    async def connect(self) -> None:
        """Initialize HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)

    async def disconnect(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.disconnect()

    def _get_cached(self, key: str) -> Any | None:
        """Get cached value if not expired."""
        if key in self._cache:
            value, ts = self._cache[key]
            if time.time() - ts < self._cache_ttl:
                return value
        return None

    def _set_cached(self, key: str, value: Any) -> None:
        """Cache a value."""
        self._cache[key] = (value, time.time())

    async def _get(self, endpoint: str, params: dict | None = None) -> dict:
        """Make GET request to Deribit API."""
        if self._client is None:
            await self.connect()

        url = f"{DERIBIT_BASE}{endpoint}"
        try:
            resp = await self._client.get(url, params=params)
            data = resp.json()
            if "error" in data:
                logger.warning("deribit_api_error", endpoint=endpoint, error=data["error"])
                return {}
            return data.get("result", {})
        except Exception as e:
            logger.error("deribit_request_failed", endpoint=endpoint, error=str(e))
            return {}

    async def get_spot(self, currency: str) -> float | None:
        """Get current spot/index price.

        Args:
            currency: BTC or ETH.

        Returns:
            Spot price in USD, or None on error.
        """
        cache_key = f"spot_{currency}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        result = await self._get(
            "/public/get_index_price",
            {"index_name": f"{currency.lower()}_usd"},
        )

        price = result.get("index_price")
        if price:
            self._set_cached(cache_key, price)
            return price
        return None

    async def get_historical_vol(self, currency: str) -> float | None:
        """Get historical (realized) volatility.

        Args:
            currency: BTC or ETH.

        Returns:
            Annualized historical vol as decimal, or None on error.
        """
        cache_key = f"hvol_{currency}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        result = await self._get(
            "/public/get_historical_volatility",
            {"currency": currency.upper()},
        )

        if isinstance(result, list) and result:
            # Returns array of [timestamp, vol] pairs
            # Vol is in percentage (e.g., 55 for 55%)
            latest_vol = result[-1][1] / 100
            self._set_cached(cache_key, latest_vol)
            return latest_vol
        return None

    async def get_options_instruments(self, currency: str) -> list[OptionInstrument]:
        """Get all active option instruments.

        Args:
            currency: BTC or ETH.

        Returns:
            List of option instruments.
        """
        result = await self._get(
            "/public/get_instruments",
            {"currency": currency.upper(), "kind": "option", "expired": False},
        )

        if not isinstance(result, list):
            return []

        instruments = []
        for inst in result:
            try:
                instruments.append(OptionInstrument(
                    name=inst["instrument_name"],
                    strike=inst["strike"],
                    expiry_ts=inst["expiration_timestamp"],
                    option_type=inst["option_type"],
                    underlying=currency.upper(),
                ))
            except (KeyError, TypeError):
                continue

        return instruments

    async def get_option_ticker(self, instrument_name: str) -> dict | None:
        """Get ticker data for a specific option.

        Args:
            instrument_name: Full instrument name (e.g., "BTC-28MAR25-100000-C").

        Returns:
            Ticker dict with mark_iv, bid_iv, ask_iv, etc.
        """
        result = await self._get(
            "/public/ticker",
            {"instrument_name": instrument_name},
        )

        if not result:
            return None

        return {
            "mark_iv": result.get("mark_iv"),
            "bid_iv": result.get("bid_iv"),
            "ask_iv": result.get("ask_iv"),
            "mark_price": result.get("mark_price"),
            "underlying_price": result.get("underlying_price"),
            "delta": result.get("greeks", {}).get("delta"),
            "gamma": result.get("greeks", {}).get("gamma"),
            "vega": result.get("greeks", {}).get("vega"),
            "theta": result.get("greeks", {}).get("theta"),
        }

    async def build_vol_surface(self, currency: str) -> VolSurface | None:
        """Build volatility surface from options chain.

        Fetches all active options and their IVs to construct surface.

        Args:
            currency: BTC or ETH.

        Returns:
            VolSurface object, or None on error.
        """
        cache_key = f"volsurf_{currency}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        spot = await self.get_spot(currency)
        if not spot:
            return None

        instruments = await self.get_options_instruments(currency)
        if not instruments:
            return None

        # Filter to calls only (puts should have same IV by put-call parity)
        calls = [i for i in instruments if i.option_type == "call"]

        # Get IV for each option (batch would be better, but Deribit doesn't support it)
        points = []
        for inst in calls[:50]:  # Limit to avoid rate limits
            ticker = await self.get_option_ticker(inst.name)
            if ticker and ticker.get("mark_iv"):
                points.append(VolPoint(
                    strike=inst.strike,
                    expiry_days=inst.days_to_expiry,
                    iv=ticker["mark_iv"] / 100,  # Convert to decimal
                    bid_iv=ticker.get("bid_iv", 0) / 100 if ticker.get("bid_iv") else None,
                    ask_iv=ticker.get("ask_iv", 0) / 100 if ticker.get("ask_iv") else None,
                ))

        surface = VolSurface(
            underlying=currency.upper(),
            spot=spot,
            points=points,
        )

        self._set_cached(cache_key, surface)
        return surface

    async def get_atm_iv(self, currency: str, days_to_expiry: int = 30) -> float | None:
        """Get ATM implied volatility for target expiry.

        This is the key method for pricing digital options.

        Args:
            currency: BTC or ETH.
            days_to_expiry: Target days to expiration.

        Returns:
            ATM IV as decimal (0.55 = 55%), or None on error.
        """
        # Try to get from vol surface
        surface = await self.build_vol_surface(currency)
        if surface:
            iv = surface.get_atm_iv(days_to_expiry)
            if iv:
                return iv

        # Fallback to historical vol
        return await self.get_historical_vol(currency)


async def get_market_data(currencies: list[str] = None) -> dict[str, dict]:
    """Convenience function to fetch market data for multiple currencies.

    Args:
        currencies: List of currencies (default: ["BTC", "ETH"]).

    Returns:
        Dict mapping currency to {spot, historical_vol, atm_iv_30d}.
    """
    if currencies is None:
        currencies = ["BTC", "ETH"]

    result = {}
    async with DeribitOptionsClient() as client:
        for currency in currencies:
            spot = await client.get_spot(currency)
            hvol = await client.get_historical_vol(currency)
            atm_iv = await client.get_atm_iv(currency, days_to_expiry=30)

            result[currency] = {
                "spot": spot,
                "historical_vol": hvol,
                "atm_iv_30d": atm_iv,
            }

    return result
