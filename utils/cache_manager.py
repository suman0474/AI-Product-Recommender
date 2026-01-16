"""
TTL Cache Manager
Provides thread-safe in-memory caching with time-to-live (TTL) support
"""
import time
import logging
from typing import Dict, Any, Optional
from threading import Lock
from collections import OrderedDict

logger = logging.getLogger(__name__)


class TTLCache:
    """
    Thread-safe in-memory cache with TTL (time-to-live) support

    Features:
    - Automatic expiration based on TTL
    - LRU eviction when max_size is reached
    - Thread-safe operations with locking
    - Cache hit/miss statistics
    """

    def __init__(self, ttl_seconds: int = 1800, max_size: int = 100):
        """
        Initialize TTL cache

        Args:
            ttl_seconds: Time-to-live for cache entries in seconds (default: 1800 = 30 minutes)
            max_size: Maximum number of entries to store (default: 100)
        """
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.lock = Lock()

        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0
        }

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache if exists and not expired

        Args:
            key: Cache key

        Returns:
            Cached value if exists and valid, None otherwise
        """
        with self.lock:
            if key not in self.cache:
                self.stats["misses"] += 1
                return None

            entry = self.cache[key]
            current_time = time.time()

            # Check if expired
            if current_time > entry["expires_at"]:
                logger.debug(f"[TTL_CACHE] Cache entry expired: {key}")
                del self.cache[key]
                self.stats["expirations"] += 1
                self.stats["misses"] += 1
                return None

            # Move to end (LRU)
            self.cache.move_to_end(key)
            self.stats["hits"] += 1

            logger.debug(f"[TTL_CACHE] Cache hit: {key} (TTL remaining: {entry['expires_at'] - current_time:.0f}s)")
            return entry["value"]

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        """
        Set value in cache with TTL

        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Custom TTL for this entry (optional, uses default if not provided)
        """
        with self.lock:
            ttl = ttl_seconds if ttl_seconds is not None else self.ttl_seconds
            current_time = time.time()

            # Evict oldest entry if at max size and key doesn't exist
            if key not in self.cache and len(self.cache) >= self.max_size:
                oldest_key = next(iter(self.cache))
                logger.debug(f"[TTL_CACHE] Evicting oldest entry: {oldest_key}")
                del self.cache[oldest_key]
                self.stats["evictions"] += 1

            self.cache[key] = {
                "value": value,
                "created_at": current_time,
                "expires_at": current_time + ttl
            }

            # Move to end (most recently used)
            self.cache.move_to_end(key)

            logger.debug(f"[TTL_CACHE] Cache set: {key} (TTL: {ttl}s)")

    def invalidate(self, key: str) -> bool:
        """
        Invalidate (delete) a specific cache entry

        Args:
            key: Cache key to invalidate

        Returns:
            True if key was found and deleted, False otherwise
        """
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                logger.debug(f"[TTL_CACHE] Cache invalidated: {key}")
                return True
            return False

    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all keys matching a pattern (simple substring match)

        Args:
            pattern: Pattern to match (substring)

        Returns:
            Number of keys invalidated
        """
        with self.lock:
            keys_to_delete = [k for k in self.cache.keys() if pattern in k]
            for key in keys_to_delete:
                del self.cache[key]
                logger.debug(f"[TTL_CACHE] Cache invalidated: {key}")

            if keys_to_delete:
                logger.info(f"[TTL_CACHE] Invalidated {len(keys_to_delete)} keys matching pattern: {pattern}")

            return len(keys_to_delete)

    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            count = len(self.cache)
            self.cache.clear()
            logger.info(f"[TTL_CACHE] Cache cleared ({count} entries)")

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries from cache

        Returns:
            Number of expired entries removed
        """
        with self.lock:
            current_time = time.time()
            expired_keys = [
                k for k, v in self.cache.items()
                if current_time > v["expires_at"]
            ]

            for key in expired_keys:
                del self.cache[key]
                self.stats["expirations"] += 1

            if expired_keys:
                logger.info(f"[TTL_CACHE] Cleaned up {len(expired_keys)} expired entries")

            return len(expired_keys)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics

        Returns:
            Dictionary with cache stats
        """
        with self.lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0

            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.stats["hits"],
                "misses": self.stats["misses"],
                "hit_rate": f"{hit_rate:.1f}%",
                "evictions": self.stats["evictions"],
                "expirations": self.stats["expirations"]
            }

    def __len__(self):
        """Return current cache size"""
        with self.lock:
            return len(self.cache)

    def __contains__(self, key: str):
        """Check if key exists and is not expired"""
        return self.get(key) is not None


# Singleton instances for common use cases
_schema_cache = None
_product_cache = None
_advanced_params_cache = None


def get_schema_cache(ttl_seconds: int = 1800, max_size: int = 100) -> TTLCache:
    """
    Get singleton schema cache instance

    Args:
        ttl_seconds: TTL for schema entries (default: 1800 = 30 minutes)
        max_size: Maximum cache size (default: 100)

    Returns:
        TTLCache instance for schemas
    """
    global _schema_cache
    if _schema_cache is None:
        _schema_cache = TTLCache(ttl_seconds=ttl_seconds, max_size=max_size)
        logger.info(f"[CACHE_MANAGER] Initialized schema cache (TTL: {ttl_seconds}s, max_size: {max_size})")
    return _schema_cache


def get_product_cache(ttl_seconds: int = 3600, max_size: int = 200) -> TTLCache:
    """
    Get singleton product cache instance

    Args:
        ttl_seconds: TTL for product entries (default: 3600 = 1 hour)
        max_size: Maximum cache size (default: 200)

    Returns:
        TTLCache instance for products
    """
    global _product_cache
    if _product_cache is None:
        _product_cache = TTLCache(ttl_seconds=ttl_seconds, max_size=max_size)
        logger.info(f"[CACHE_MANAGER] Initialized product cache (TTL: {ttl_seconds}s, max_size: {max_size})")
    return _product_cache


def get_advanced_params_cache(ttl_seconds: int = 600, max_size: int = 50) -> TTLCache:
    """
    Get singleton advanced parameters cache instance

    Args:
        ttl_seconds: TTL for parameter entries (default: 600 = 10 minutes)
        max_size: Maximum cache size (default: 50)

    Returns:
        TTLCache instance for advanced parameters
    """
    global _advanced_params_cache
    if _advanced_params_cache is None:
        _advanced_params_cache = TTLCache(ttl_seconds=ttl_seconds, max_size=max_size)
        logger.info(f"[CACHE_MANAGER] Initialized advanced params cache (TTL: {ttl_seconds}s, max_size: {max_size})")
    return _advanced_params_cache


def invalidate_all_caches():
    """Clear all singleton cache instances"""
    if _schema_cache:
        _schema_cache.clear()
    if _product_cache:
        _product_cache.clear()
    if _advanced_params_cache:
        _advanced_params_cache.clear()
    logger.info("[CACHE_MANAGER] All caches invalidated")
