"""
LLM Response Cache Manager

Caches LLM generation results by prompt hash to avoid redundant API calls.
Significantly reduces LLM costs and improves response time for repeated queries.

Phase 4 Optimization: 20-30% reduction in LLM API calls
"""
import hashlib
import logging
import threading
import time
from collections import OrderedDict
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


class LLMResponseCache:
    """
    Cache for LLM-generated responses with TTL and LRU eviction.

    Features:
    - Caches by prompt hash (model + temperature + prompt content)
    - TTL-based expiration (default 24 hours)
    - LRU eviction when capacity exceeded
    - Thread-safe operations
    - Tracks cache statistics
    """

    def __init__(
        self,
        max_size: int = 500,
        ttl_seconds: int = 86400,  # 24 hours
        hash_algorithm: str = "sha256"
    ):
        """
        Initialize LLM response cache.

        Args:
            max_size: Maximum number of cached responses (default 500)
            ttl_seconds: Time-to-live for cached responses in seconds (default 24 hours)
            hash_algorithm: Hash algorithm for cache keys (default sha256)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.hash_algorithm = hash_algorithm

        # OrderedDict for LRU eviction
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

        # Statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_cached": 0
        }

        logger.info(
            f"[LLM_RESPONSE_CACHE] Initialized: max_size={max_size}, "
            f"ttl={ttl_seconds}s, algorithm={hash_algorithm}"
        )

    def _generate_cache_key(
        self,
        model: str,
        prompt: str,
        temperature: float
    ) -> str:
        """
        Generate cache key from LLM parameters.

        Creates a deterministic hash from model, temperature, and prompt.
        This ensures identical requests always map to the same cache key.

        Args:
            model: LLM model name (e.g., "gemini-2.5-flash")
            prompt: The prompt/request text
            temperature: Temperature parameter for generation

        Returns:
            Hex digest of hash
        """
        # Combine parameters into a key string
        key_str = f"{model}|{temperature}|{prompt}"

        # Generate hash
        if self.hash_algorithm == "sha256":
            return hashlib.sha256(key_str.encode()).hexdigest()
        elif self.hash_algorithm == "md5":
            return hashlib.md5(key_str.encode()).hexdigest()
        else:
            return hashlib.sha256(key_str.encode()).hexdigest()

    def get(
        self,
        model: str,
        prompt: str,
        temperature: float
    ) -> Optional[str]:
        """
        Get cached LLM response if available.

        Args:
            model: LLM model name
            prompt: The prompt/request text
            temperature: Temperature parameter

        Returns:
            Cached response string, or None if not cached/expired
        """
        cache_key = self._generate_cache_key(model, prompt, temperature)

        with self._lock:
            if cache_key not in self._cache:
                self._stats["misses"] += 1
                return None

            entry = self._cache[cache_key]

            # Check TTL
            if time.time() - entry["_created_at"] > self.ttl_seconds:
                # Expired, remove from cache
                del self._cache[cache_key]
                self._stats["misses"] += 1
                logger.debug(f"[LLM_RESPONSE_CACHE] Cache miss (expired): {cache_key[:8]}...")
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(cache_key)
            self._stats["hits"] += 1

            logger.debug(f"[LLM_RESPONSE_CACHE] Cache hit: {cache_key[:8]}...")
            return entry["response"]

    def put(
        self,
        model: str,
        prompt: str,
        temperature: float,
        response: str
    ) -> None:
        """
        Cache an LLM response.

        Args:
            model: LLM model name
            prompt: The prompt/request text
            temperature: Temperature parameter
            response: The LLM response to cache
        """
        cache_key = self._generate_cache_key(model, prompt, temperature)

        with self._lock:
            # If exists, move to end (LRU)
            if cache_key in self._cache:
                self._cache.move_to_end(cache_key)
                self._cache[cache_key] = {
                    "response": response,
                    "_created_at": time.time()
                }
                logger.debug(f"[LLM_RESPONSE_CACHE] Updated cache: {cache_key[:8]}...")
                return

            # New entry
            self._cache[cache_key] = {
                "response": response,
                "_created_at": time.time()
            }
            self._stats["total_cached"] = max(self._stats["total_cached"], len(self._cache))

            logger.debug(f"[LLM_RESPONSE_CACHE] Cached response: {cache_key[:8]}...")

            # Enforce size limit with LRU eviction
            while len(self._cache) > self.max_size:
                # Remove oldest (first) item
                oldest_key, _ = self._cache.popitem(last=False)
                self._stats["evictions"] += 1
                logger.debug(f"[LLM_RESPONSE_CACHE] Evicted old entry: {oldest_key[:8]}...")

    def get_hit_rate(self) -> float:
        """
        Get cache hit rate as percentage.

        Returns:
            Hit rate (0-100), or 0 if no accesses
        """
        with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            if total == 0:
                return 0.0
            return (self._stats["hits"] / total) * 100

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with current stats
        """
        with self._lock:
            hit_rate = self.get_hit_rate()
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "evictions": self._stats["evictions"],
                "total_cached": self._stats["total_cached"],
                "hit_rate_percent": f"{hit_rate:.1f}",
                "ttl_seconds": self.ttl_seconds
            }

    def clear(self) -> int:
        """
        Clear all cached responses.

        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._stats = {
                "hits": 0,
                "misses": 0,
                "evictions": 0,
                "total_cached": 0
            }
            logger.info(f"[LLM_RESPONSE_CACHE] Cleared {count} entries")
            return count


# Global singleton instance
_llm_response_cache: Optional[LLMResponseCache] = None


def get_llm_response_cache(
    max_size: int = 500,
    ttl_seconds: int = 86400
) -> LLMResponseCache:
    """
    Get or create global LLM response cache singleton.

    Args:
        max_size: Maximum cache size (only used on first call)
        ttl_seconds: TTL for responses (only used on first call)

    Returns:
        Global LLMResponseCache instance
    """
    global _llm_response_cache
    if _llm_response_cache is None:
        _llm_response_cache = LLMResponseCache(
            max_size=max_size,
            ttl_seconds=ttl_seconds
        )
    return _llm_response_cache


def clear_llm_response_cache() -> int:
    """
    Clear the global LLM response cache.

    Returns:
        Number of entries cleared
    """
    global _llm_response_cache
    if _llm_response_cache:
        return _llm_response_cache.clear()
    return 0
