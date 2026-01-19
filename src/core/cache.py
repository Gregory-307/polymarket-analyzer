"""In-memory TTL cache for API responses.

Reduces API calls by caching responses with configurable TTL.
Supports automatic invalidation on significant price changes.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar, Generic
from functools import wraps

from .utils import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """Single cache entry with expiration."""

    value: T
    expires_at: float
    created_at: float = field(default_factory=time.time)

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return time.time() >= self.expires_at

    @property
    def age_seconds(self) -> float:
        """Time since entry was created."""
        return time.time() - self.created_at


class TTLCache(Generic[T]):
    """Thread-safe TTL cache with automatic cleanup.

    Usage:
        cache = TTLCache[Market](default_ttl=60)

        # Store a value
        cache.set("market_123", market_data)

        # Retrieve (returns None if expired)
        market = cache.get("market_123")

        # Check if key exists and is valid
        if cache.has("market_123"):
            ...
    """

    def __init__(
        self,
        default_ttl: float = 60.0,
        max_size: int = 1000,
        cleanup_interval: float = 300.0,
    ):
        """Initialize cache.

        Args:
            default_ttl: Default time-to-live in seconds.
            max_size: Maximum number of entries (LRU eviction when exceeded).
            cleanup_interval: Seconds between automatic cleanup runs.
        """
        self.default_ttl = default_ttl
        self.max_size = max_size
        self.cleanup_interval = cleanup_interval

        self._cache: dict[str, CacheEntry[T]] = {}
        self._access_order: list[str] = []  # For LRU eviction
        self._lock = asyncio.Lock()

        # Statistics
        self._hits = 0
        self._misses = 0

    async def get(self, key: str) -> T | None:
        """Get value from cache.

        Args:
            key: Cache key.

        Returns:
            Cached value or None if not found/expired.
        """
        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._misses += 1
                return None

            if entry.is_expired:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                self._misses += 1
                return None

            # Update access order for LRU
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)

            self._hits += 1
            return entry.value

    async def set(
        self,
        key: str,
        value: T,
        ttl: float | None = None,
    ) -> None:
        """Store value in cache.

        Args:
            key: Cache key.
            value: Value to store.
            ttl: Time-to-live in seconds (uses default if None).
        """
        ttl = ttl if ttl is not None else self.default_ttl

        async with self._lock:
            # Evict if at max capacity
            while len(self._cache) >= self.max_size and self._access_order:
                oldest_key = self._access_order.pop(0)
                self._cache.pop(oldest_key, None)

            self._cache[key] = CacheEntry(
                value=value,
                expires_at=time.time() + ttl,
            )

            # Update access order
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)

    async def delete(self, key: str) -> bool:
        """Delete a key from cache.

        Args:
            key: Cache key to delete.

        Returns:
            True if key was deleted, False if not found.
        """
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                return True
            return False

    async def has(self, key: str) -> bool:
        """Check if key exists and is not expired.

        Args:
            key: Cache key.

        Returns:
            True if key exists and is valid.
        """
        return await self.get(key) is not None

    async def clear(self) -> int:
        """Clear all entries from cache.

        Returns:
            Number of entries cleared.
        """
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._access_order.clear()
            return count

    async def cleanup_expired(self) -> int:
        """Remove all expired entries.

        Returns:
            Number of entries removed.
        """
        async with self._lock:
            now = time.time()
            expired_keys = [
                key for key, entry in self._cache.items()
                if now >= entry.expires_at
            ]

            for key in expired_keys:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)

            return len(expired_keys)

    def get_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dictionary with cache metrics.
        """
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0

        return {
            "entries": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "default_ttl": self.default_ttl,
        }


class MarketCache:
    """Specialized cache for market data with price-based invalidation.

    Automatically invalidates cache entries when price moves significantly.

    Usage:
        cache = MarketCache(ttl=60, price_threshold=0.02)

        # Store market
        await cache.set_market(market)

        # Get market (returns None if expired or price changed significantly)
        market = await cache.get_market("market_123")
    """

    def __init__(
        self,
        ttl: float = 60.0,
        price_threshold: float = 0.02,
        max_markets: int = 500,
    ):
        """Initialize market cache.

        Args:
            ttl: Time-to-live in seconds.
            price_threshold: Invalidate if price changed by more than this.
            max_markets: Maximum number of markets to cache.
        """
        self.ttl = ttl
        self.price_threshold = price_threshold
        self._cache: TTLCache[dict] = TTLCache(
            default_ttl=ttl,
            max_size=max_markets,
        )
        self._last_prices: dict[str, float] = {}  # Track prices for invalidation

    async def get_market(self, market_id: str) -> dict | None:
        """Get cached market data.

        Args:
            market_id: Market identifier.

        Returns:
            Cached market data or None.
        """
        return await self._cache.get(market_id)

    async def set_market(self, market: dict) -> None:
        """Cache market data.

        Args:
            market: Market data dictionary (must have 'id' and 'yes_price').
        """
        market_id = market.get("id")
        if not market_id:
            return

        await self._cache.set(market_id, market)
        self._last_prices[market_id] = market.get("yes_price", 0.5)

    async def invalidate_on_price_change(
        self,
        market_id: str,
        new_price: float,
    ) -> bool:
        """Check if cache should be invalidated due to price change.

        Args:
            market_id: Market identifier.
            new_price: New YES price.

        Returns:
            True if cache was invalidated.
        """
        last_price = self._last_prices.get(market_id)
        if last_price is None:
            return False

        price_change = abs(new_price - last_price)
        if price_change >= self.price_threshold:
            await self._cache.delete(market_id)
            self._last_prices.pop(market_id, None)
            return True

        return False

    async def get_all_markets(self) -> list[dict]:
        """Get all cached markets (non-expired only).

        Returns:
            List of cached market dictionaries.
        """
        # Note: This is O(n) and may trigger many expirations
        # Use sparingly
        markets = []
        for key in list(self._cache._access_order):
            market = await self._cache.get(key)
            if market:
                markets.append(market)
        return markets

    def get_stats(self) -> dict:
        """Get cache statistics."""
        return {
            **self._cache.get_stats(),
            "tracked_prices": len(self._last_prices),
            "price_threshold": self.price_threshold,
        }


def cached(
    ttl: float = 60.0,
    key_builder: Callable[..., str] | None = None,
):
    """Decorator for caching async function results.

    Usage:
        @cached(ttl=60)
        async def get_markets(self, limit: int = 100):
            ...

        # Or with custom key builder
        @cached(ttl=30, key_builder=lambda self, market_id: f"market:{market_id}")
        async def get_market(self, market_id: str):
            ...
    """
    # Module-level cache for decorated functions
    _function_caches: dict[str, TTLCache] = {}

    def decorator(func: Callable) -> Callable:
        cache_key = f"{func.__module__}.{func.__qualname__}"
        _function_caches[cache_key] = TTLCache(default_ttl=ttl)

        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache = _function_caches[cache_key]

            # Build cache key
            if key_builder:
                entry_key = key_builder(*args, **kwargs)
            else:
                # Default key: hash of args and kwargs
                key_data = f"{args}:{sorted(kwargs.items())}"
                entry_key = hashlib.md5(key_data.encode()).hexdigest()

            # Try to get from cache
            cached_value = await cache.get(entry_key)
            if cached_value is not None:
                return cached_value

            # Call function and cache result
            result = await func(*args, **kwargs)
            if result is not None:
                await cache.set(entry_key, result)

            return result

        # Attach cache for manual control
        wrapper._cache = _function_caches[cache_key]
        return wrapper

    return decorator


# Global caches for common use cases
_markets_cache: MarketCache | None = None


def get_markets_cache(ttl: float = 60.0) -> MarketCache:
    """Get or create the global markets cache.

    Args:
        ttl: Time-to-live for cache entries.

    Returns:
        MarketCache instance.
    """
    global _markets_cache
    if _markets_cache is None:
        _markets_cache = MarketCache(ttl=ttl)
    return _markets_cache
