"""
Retry utility with exponential backoff for Pinecone operations.

According to CLAUDE.md best practices:
- Retry 429 (rate limit) errors with exponential backoff
- Retry 5xx (server errors) with exponential backoff
- Do NOT retry 4xx errors (except 429) - these are client errors that won't resolve
"""

import time
from typing import Callable, TypeVar, Any

T = TypeVar('T')


def exponential_backoff_retry(
    func: Callable[[], T],
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0
) -> T:
    """
    Execute function with exponential backoff retry for transient errors.

    Args:
        func: Function to execute
        max_retries: Maximum number of retry attempts (default: 5)
        base_delay: Base delay in seconds for exponential backoff (default: 1.0)
        max_delay: Maximum delay in seconds (default: 60.0)

    Returns:
        Result of the function call

    Raises:
        Exception: Re-raises the last exception if all retries are exhausted
                  or if error is not retryable (4xx except 429)

    Example:
        >>> def risky_operation():
        ...     return index.upsert_records(namespace, records)
        >>> result = exponential_backoff_retry(risky_operation)
    """
    last_exception = None

    for attempt in range(max_retries):
        try:
            return func()

        except Exception as e:
            last_exception = e

            # Try to get HTTP status code from the exception
            status_code = None

            # Check for Pinecone-specific exception attributes
            if hasattr(e, 'status'):
                status_code = e.status
            elif hasattr(e, 'status_code'):
                status_code = e.status_code
            # Some exceptions might have it in args or message
            elif hasattr(e, 'args') and e.args:
                # Try to extract status code from error message
                error_str = str(e)
                if '429' in error_str:
                    status_code = 429
                elif any(f'{code}' in error_str for code in range(500, 600)):
                    # Server error 5xx
                    for code in range(500, 600):
                        if f'{code}' in error_str:
                            status_code = code
                            break

            # Determine if error is retryable
            is_retryable = False

            if status_code:
                # Retry on 429 (rate limit) and 5xx (server errors)
                if status_code == 429 or status_code >= 500:
                    is_retryable = True
                # Don't retry on other 4xx errors (client errors)
                elif 400 <= status_code < 500:
                    print(f"Client error {status_code}: {e}")
                    raise  # Don't retry client errors

            # If we couldn't determine status code, be conservative and retry
            # (could be network error, timeout, etc.)
            else:
                is_retryable = True

            # Retry logic
            if is_retryable and attempt < max_retries - 1:
                delay = min(base_delay * (2 ** attempt), max_delay)
                error_type = f"Error {status_code}" if status_code else "Error"
                print(f"{error_type} on attempt {attempt + 1}/{max_retries}: {e}")
                print(f"Retrying in {delay:.1f}s...")
                time.sleep(delay)
            elif attempt >= max_retries - 1:
                print(f"Max retries ({max_retries}) exceeded")
                raise
            else:
                # Non-retryable error
                raise

    # Should never reach here, but just in case
    if last_exception:
        raise last_exception
    raise RuntimeError("Retry logic failed unexpectedly")


def retry_on_rate_limit(func: Callable[[], T], max_retries: int = 3) -> T:
    """
    Simpler retry specifically for rate limit errors (429).

    Args:
        func: Function to execute
        max_retries: Maximum number of retry attempts (default: 3)

    Returns:
        Result of the function call

    Example:
        >>> result = retry_on_rate_limit(lambda: index.search(...))
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            status_code = getattr(e, 'status', getattr(e, 'status_code', None))

            if status_code == 429 and attempt < max_retries - 1:
                delay = 2 ** attempt  # 1s, 2s, 4s
                print(f"Rate limit hit, retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise

    raise RuntimeError("Retry logic failed unexpectedly")
