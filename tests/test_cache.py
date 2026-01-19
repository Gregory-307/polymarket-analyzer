"""Tests for caching module."""

import pytest
import asyncio
import time

from src.core.cache import TTLCache, MarketCache, cached


class TestTTLCache:
    """Tests for TTL cache."""

    @pytest.mark.asyncio
    async def test_set_and_get(self):
        """Test basic set and get operations."""
        cache: TTLCache[str] = TTLCache(default_ttl=60)

        await cache.set("key1", "value1")
        result = await cache.get("key1")

        assert result == "value1"

    @pytest.mark.asyncio
    async def test_expiration(self):
        """Test that entries expire after TTL."""
        cache: TTLCache[str] = TTLCache(default_ttl=0.1)  # 100ms TTL

        await cache.set("key1", "value1")

        # Should exist immediately
        result = await cache.get("key1")
        assert result == "value1"

        # Wait for expiration
        await asyncio.sleep(0.15)

        # Should be gone
        result = await cache.get("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_custom_ttl(self):
        """Test custom TTL per entry."""
        cache: TTLCache[str] = TTLCache(default_ttl=60)

        await cache.set("short", "value", ttl=0.1)
        await cache.set("long", "value", ttl=60)

        await asyncio.sleep(0.15)

        assert await cache.get("short") is None
        assert await cache.get("long") == "value"

    @pytest.mark.asyncio
    async def test_delete(self):
        """Test deleting entries."""
        cache: TTLCache[str] = TTLCache(default_ttl=60)

        await cache.set("key1", "value1")
        assert await cache.has("key1")

        deleted = await cache.delete("key1")
        assert deleted is True
        assert await cache.has("key1") is False

        # Deleting non-existent key
        deleted = await cache.delete("nonexistent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_clear(self):
        """Test clearing all entries."""
        cache: TTLCache[str] = TTLCache(default_ttl=60)

        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")

        count = await cache.clear()
        assert count == 3

        assert await cache.get("key1") is None
        assert await cache.get("key2") is None
        assert await cache.get("key3") is None

    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        """Test LRU eviction when at max capacity."""
        cache: TTLCache[int] = TTLCache(default_ttl=60, max_size=3)

        await cache.set("a", 1)
        await cache.set("b", 2)
        await cache.set("c", 3)

        # Access 'a' to make it recently used
        await cache.get("a")

        # Add new entry - should evict 'b' (least recently used)
        await cache.set("d", 4)

        assert await cache.get("a") == 1  # Still there (accessed recently)
        assert await cache.get("b") is None  # Evicted
        assert await cache.get("c") == 3  # Still there
        assert await cache.get("d") == 4  # New entry

    @pytest.mark.asyncio
    async def test_cleanup_expired(self):
        """Test explicit cleanup of expired entries."""
        cache: TTLCache[str] = TTLCache(default_ttl=0.1)

        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3", ttl=60)  # Long TTL

        await asyncio.sleep(0.15)

        removed = await cache.cleanup_expired()
        assert removed == 2

        # key3 should still exist
        assert await cache.get("key3") == "value3"

    @pytest.mark.asyncio
    async def test_stats(self):
        """Test cache statistics."""
        cache: TTLCache[str] = TTLCache(default_ttl=60)

        await cache.set("key1", "value1")

        # One hit
        await cache.get("key1")
        # One miss
        await cache.get("nonexistent")

        stats = cache.get_stats()

        assert stats["entries"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5


class TestMarketCache:
    """Tests for market-specific cache."""

    @pytest.mark.asyncio
    async def test_set_and_get_market(self):
        """Test storing and retrieving markets."""
        cache = MarketCache(ttl=60)

        market = {
            "id": "market_123",
            "question": "Will it rain?",
            "yes_price": 0.65,
        }

        await cache.set_market(market)
        result = await cache.get_market("market_123")

        assert result == market

    @pytest.mark.asyncio
    async def test_price_invalidation(self):
        """Test that cache is invalidated on significant price change."""
        cache = MarketCache(ttl=60, price_threshold=0.02)

        market = {
            "id": "market_123",
            "yes_price": 0.50,
        }
        await cache.set_market(market)

        # Small price change - should not invalidate
        invalidated = await cache.invalidate_on_price_change("market_123", 0.51)
        assert invalidated is False
        assert await cache.get_market("market_123") is not None

        # Large price change - should invalidate
        invalidated = await cache.invalidate_on_price_change("market_123", 0.55)
        assert invalidated is True
        assert await cache.get_market("market_123") is None

    @pytest.mark.asyncio
    async def test_get_all_markets(self):
        """Test getting all cached markets."""
        cache = MarketCache(ttl=60)

        await cache.set_market({"id": "m1", "yes_price": 0.5})
        await cache.set_market({"id": "m2", "yes_price": 0.6})
        await cache.set_market({"id": "m3", "yes_price": 0.7})

        markets = await cache.get_all_markets()
        assert len(markets) == 3

        market_ids = {m["id"] for m in markets}
        assert market_ids == {"m1", "m2", "m3"}


class TestCachedDecorator:
    """Tests for @cached decorator."""

    @pytest.mark.asyncio
    async def test_cached_function(self):
        """Test caching function results."""
        call_count = 0

        @cached(ttl=60)
        async def expensive_operation(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call - should execute function
        result1 = await expensive_operation(5)
        assert result1 == 10
        assert call_count == 1

        # Second call with same arg - should use cache
        result2 = await expensive_operation(5)
        assert result2 == 10
        assert call_count == 1  # No additional call

        # Call with different arg - should execute function
        result3 = await expensive_operation(10)
        assert result3 == 20
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_cached_expiration(self):
        """Test that cached results expire."""
        call_count = 0

        @cached(ttl=0.1)
        async def operation() -> str:
            nonlocal call_count
            call_count += 1
            return "result"

        await operation()
        assert call_count == 1

        await operation()  # Should use cache
        assert call_count == 1

        await asyncio.sleep(0.15)

        await operation()  # Cache expired, should call again
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_custom_key_builder(self):
        """Test custom key builder for cache."""
        call_count = 0

        @cached(ttl=60, key_builder=lambda self, market_id: f"market:{market_id}")
        async def get_market(self, market_id: str) -> dict:
            nonlocal call_count
            call_count += 1
            return {"id": market_id}

        # Simulate method calls (self is ignored for key)
        await get_market(None, "abc")
        await get_market(None, "abc")  # Should hit cache

        assert call_count == 1

        await get_market(None, "xyz")  # Different key
        assert call_count == 2
