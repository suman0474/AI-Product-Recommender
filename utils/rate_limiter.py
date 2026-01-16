"""
Rate Limiter Utility
Token bucket rate limiter for API calls with thread-safe implementation
"""
import time
import threading
import logging
from collections import deque
from typing import Callable, Any
from functools import wraps

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter for API calls.

    Features:
    - Configurable rate (requests per time window)
    - Thread-safe with locking
    - Automatic token refill
    - Blocking wait for available tokens
    - Statistics tracking
    """

    def __init__(self, max_calls: int, time_window: int, name: str = "unnamed"):
        """
        Initialize rate limiter.

        Args:
            max_calls: Maximum calls allowed in time window
            time_window: Time window in seconds
            name: Name for logging purposes
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.name = name
        self.calls = deque()
        self.lock = threading.Lock()

        # Statistics
        self.stats = {
            "total_calls": 0,
            "total_waits": 0,
            "total_wait_time": 0.0
        }

        logger.info(
            f"[RATE_LIMIT] Initialized '{name}': {max_calls} calls/{time_window}s"
        )

    def __call__(self, func: Callable) -> Callable:
        """Decorator to rate-limit a function"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.wait_if_needed()
            return func(*args, **kwargs)
        return wrapper

    def wait_if_needed(self):
        """
        Wait if rate limit reached.

        Blocks the calling thread until a token becomes available.
        """
        with self.lock:
            now = time.time()

            # Remove old calls outside time window
            while self.calls and self.calls[0] < now - self.time_window:
                self.calls.popleft()

            # If at limit, calculate wait time
            if len(self.calls) >= self.max_calls:
                oldest_call = self.calls[0]
                sleep_time = self.time_window - (now - oldest_call)

                if sleep_time > 0:
                    self.stats["total_waits"] += 1
                    self.stats["total_wait_time"] += sleep_time

                    logger.warning(
                        f"[RATE_LIMIT] '{self.name}' limit reached "
                        f"({self.max_calls}/{self.time_window}s), "
                        f"waiting {sleep_time:.1f}s..."
                    )

                    # Release lock during sleep
                    self.lock.release()
                    try:
                        time.sleep(sleep_time)
                    finally:
                        self.lock.acquire()

                    # Recursively retry after sleeping
                    return self.wait_if_needed()

            # Record this call
            self.calls.append(now)
            self.stats["total_calls"] += 1

    def acquire(self, tokens: int = 1):
        """
        Acquire tokens (alias for wait_if_needed for consistency).

        Args:
            tokens: Number of tokens to acquire (currently only supports 1)
        """
        if tokens != 1:
            raise ValueError("Currently only supports acquiring 1 token at a time")
        self.wait_if_needed()

    def get_stats(self) -> dict:
        """
        Get rate limiter statistics.

        Returns:
            Dictionary with statistics
        """
        with self.lock:
            avg_wait = (
                self.stats["total_wait_time"] / self.stats["total_waits"]
                if self.stats["total_waits"] > 0
                else 0.0
            )

            return {
                "name": self.name,
                "max_calls": self.max_calls,
                "time_window": self.time_window,
                "current_calls": len(self.calls),
                "total_calls": self.stats["total_calls"],
                "total_waits": self.stats["total_waits"],
                "total_wait_time": round(self.stats["total_wait_time"], 2),
                "avg_wait_time": round(avg_wait, 2)
            }

    def reset_stats(self):
        """Reset statistics"""
        with self.lock:
            self.stats = {
                "total_calls": 0,
                "total_waits": 0,
                "total_wait_time": 0.0
            }
            logger.info(f"[RATE_LIMIT] Stats reset for '{self.name}'")

    def get_available_calls(self) -> int:
        """
        Get number of calls currently available without waiting.

        Returns:
            Number of available calls
        """
        with self.lock:
            now = time.time()

            # Remove expired calls
            while self.calls and self.calls[0] < now - self.time_window:
                self.calls.popleft()

            return max(0, self.max_calls - len(self.calls))


# ============================================================================
# Global Rate Limiters
# ============================================================================

# LLM rate limiter: 60 calls/minute (1 per second average)
_llm_rate_limiter = RateLimiter(
    max_calls=60,
    time_window=60,
    name="LLM"
)

# Search API rate limiter: 10 calls/minute
_search_rate_limiter = RateLimiter(
    max_calls=10,
    time_window=60,
    name="Search API"
)

# Image search rate limiter: 30 calls/minute
_image_rate_limiter = RateLimiter(
    max_calls=30,
    time_window=60,
    name="Image Search"
)

# PDF download rate limiter: 20 calls/minute
_pdf_rate_limiter = RateLimiter(
    max_calls=20,
    time_window=60,
    name="PDF Download"
)


def get_llm_rate_limiter() -> RateLimiter:
    """
    Get global LLM rate limiter.

    Returns:
        RateLimiter configured for LLM calls (60/min)
    """
    return _llm_rate_limiter


def get_search_rate_limiter() -> RateLimiter:
    """
    Get global search API rate limiter.

    Returns:
        RateLimiter configured for search calls (10/min)
    """
    return _search_rate_limiter


def get_image_rate_limiter() -> RateLimiter:
    """
    Get global image search rate limiter.

    Returns:
        RateLimiter configured for image calls (30/min)
    """
    return _image_rate_limiter


def get_pdf_rate_limiter() -> RateLimiter:
    """
    Get global PDF download rate limiter.

    Returns:
        RateLimiter configured for PDF downloads (20/min)
    """
    return _pdf_rate_limiter


def get_all_rate_limiter_stats() -> dict:
    """
    Get statistics from all global rate limiters.

    Returns:
        Dictionary with stats for all rate limiters
    """
    return {
        "llm": _llm_rate_limiter.get_stats(),
        "search": _search_rate_limiter.get_stats(),
        "image": _image_rate_limiter.get_stats(),
        "pdf": _pdf_rate_limiter.get_stats()
    }


# ============================================================================
# Convenience Decorators
# ============================================================================

def rate_limit_llm(func: Callable) -> Callable:
    """
    Decorator to rate-limit LLM calls.

    Usage:
        @rate_limit_llm
        def call_llm(...):
            return llm.invoke(...)
    """
    return _llm_rate_limiter(func)


def rate_limit_search(func: Callable) -> Callable:
    """
    Decorator to rate-limit search API calls.

    Usage:
        @rate_limit_search
        def search(...):
            return search_api.query(...)
    """
    return _search_rate_limiter(func)


def rate_limit_image(func: Callable) -> Callable:
    """
    Decorator to rate-limit image search calls.

    Usage:
        @rate_limit_image
        def search_images(...):
            return image_api.search(...)
    """
    return _image_rate_limiter(func)


def rate_limit_pdf(func: Callable) -> Callable:
    """
    Decorator to rate-limit PDF downloads.

    Usage:
        @rate_limit_pdf
        def download_pdf(...):
            return requests.get(...)
    """
    return _pdf_rate_limiter(func)
