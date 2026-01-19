# agentic/caching/validation_cache.py
# =============================================================================
# VALIDATION RESULT CACHE MANAGER
# =============================================================================
#
# Purpose: Thread-safe cache for validation results with TTL support.
# Prevents redundant validation calls for identical product type/spec combinations.
#
# Features:
# - Deterministic key generation from product type + specs
# - TTL-based expiration
# - Thread-safe concurrent access
# - Hit/miss statistics
# - Manual cleanup

import hashlib
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import threading

logger = logging.getLogger(__name__)


class ValidationCacheManager:
    """
    Thread-safe cache for validation results with TTL support.

    Key: Hash of (product_type, specifications)
    Value: Validation result with timestamp

    Example:
        cache = ValidationCacheManager(ttl_seconds=3600)

        # First call: runs validation
        result = cache.get_or_compute(
            product_type="Temperature Sensor",
            specs={"range": "0-100C"},
            compute_func=lambda: run_expensive_validation(...)
        )

        # Second call: returns from cache instantly
        result = cache.get_or_compute(
            product_type="Temperature Sensor",
            specs={"range": "0-100C"},
            compute_func=lambda: run_expensive_validation(...)
        )
    """

    def __init__(self, ttl_seconds: int = 3600, max_size: int = 500):
        """
        Initialize the cache.

        Args:
            ttl_seconds: Time-to-live for cache entries (default 1 hour)
            max_size: Maximum number of cached entries (default 500)
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.RLock()
        self.ttl = timedelta(seconds=ttl_seconds)
        self.max_size = max_size
        self.hit_count = 0
        self.miss_count = 0

    def _generate_key(self, product_type: str, specs: Dict[str, Any]) -> str:
        """
        Generate deterministic cache key from product type and specs.

        Args:
            product_type: Type of product
            specs: Specifications dictionary

        Returns:
            Hex string hash key (first 16 chars for readability)
        """
        # Create normalized spec dict for hashing
        normalized = json.dumps(specs, sort_keys=True, default=str)
        content = f"{product_type.lower().strip()}:{normalized}"
        hash_obj = hashlib.sha256(content.encode())
        return hash_obj.hexdigest()[:16]

    def get(
        self,
        product_type: str,
        specs: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Get validation result from cache if available and not expired.

        Args:
            product_type: Type of product
            specs: Specifications dictionary

        Returns:
            Cached validation result or None if not found/expired
        """
        cache_key = self._generate_key(product_type, specs)

        with self.lock:
            if cache_key not in self.cache:
                self.miss_count += 1
                return None

            cached_entry = self.cache[cache_key]

            # Check if expired
            if datetime.now() > cached_entry['expiry']:
                logger.debug(f"[ValidationCache] Expired entry: {product_type} ({cache_key})")
                del self.cache[cache_key]
                self.miss_count += 1
                return None

            # Cache hit!
            self.hit_count += 1
            logger.debug(f"[ValidationCache] HIT for {product_type} (key: {cache_key})")
            return cached_entry['result'].copy()

    def set(
        self,
        product_type: str,
        specs: Dict[str, Any],
        validation_result: Dict[str, Any]
    ) -> None:
        """
        Store validation result in cache.

        Args:
            product_type: Type of product
            specs: Specifications dictionary
            validation_result: Result from validator
        """
        cache_key = self._generate_key(product_type, specs)

        with self.lock:
            # Check if at max size
            if len(self.cache) >= self.max_size:
                # Evict oldest entry
                oldest_key = min(
                    self.cache.keys(),
                    key=lambda k: self.cache[k]['timestamp']
                )
                logger.debug(f"[ValidationCache] Evicting oldest: {oldest_key}")
                del self.cache[oldest_key]

            self.cache[cache_key] = {
                'result': validation_result.copy(),
                'timestamp': datetime.now(),
                'expiry': datetime.now() + self.ttl,
                'product_type': product_type,
                'spec_count': len(specs)
            }
            logger.debug(f"[ValidationCache] Cached {product_type} (key: {cache_key}, size: {len(self.cache)}/{self.max_size})")

    def get_or_compute(
        self,
        product_type: str,
        specs: Dict[str, Any],
        compute_func
    ) -> Dict[str, Any]:
        """
        Get from cache or compute if missing.

        Args:
            product_type: Type of product
            specs: Specifications dictionary
            compute_func: Function to call if cache miss (should return dict)

        Returns:
            Validation result (from cache or newly computed)
        """
        # Try cache first
        cached = self.get(product_type, specs)
        if cached is not None:
            logger.info(f"[ValidationCache] Using cached result for {product_type}")
            return cached

        # Cache miss - compute
        logger.info(f"[ValidationCache] Cache miss for {product_type}, computing...")
        result = compute_func()

        # Store in cache
        self.set(product_type, specs, result)

        return result

    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            size = len(self.cache)
            self.cache.clear()
            self.hit_count = 0
            self.miss_count = 0
            logger.info(f"[ValidationCache] Cleared {size} entries")

    def cleanup_expired(self) -> int:
        """
        Remove expired entries.

        Returns:
            Number of entries removed
        """
        with self.lock:
            now = datetime.now()
            expired_keys = [
                k for k, v in self.cache.items()
                if now > v['expiry']
            ]
            for k in expired_keys:
                del self.cache[k]

            if expired_keys:
                logger.info(f"[ValidationCache] Cleaned up {len(expired_keys)} expired entries")

            return len(expired_keys)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.

        Returns:
            Dictionary with cache metrics
        """
        with self.lock:
            total = self.hit_count + self.miss_count
            hit_rate = (self.hit_count / total * 100) if total > 0 else 0
            return {
                'hits': self.hit_count,
                'misses': self.miss_count,
                'total_requests': total,
                'hit_rate': f"{hit_rate:.1f}%",
                'cached_items': len(self.cache),
                'max_size': self.max_size,
                'utilization': f"{len(self.cache)/self.max_size*100:.1f}%"
            }

    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_stats()
        return (
            f"ValidationCacheManager("
            f"items={stats['cached_items']}, "
            f"hit_rate={stats['hit_rate']}, "
            f"utilization={stats['utilization']}"
            f")"
        )
