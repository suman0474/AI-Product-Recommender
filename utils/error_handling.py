"""
Error Handling Utilities for Tools and Workflows

Provides decorators and utilities for consistent error handling,
retry logic, and fallback mechanisms.
"""

import functools
import logging
import time
from typing import Any, Callable, Dict, Optional, List
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ToolError(Exception):
    """Base exception for tool errors"""
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM, context: Optional[Dict] = None):
        super().__init__(message)
        self.severity = severity
        self.context = context or {}


class RetryableError(ToolError):
    """Error that can be retried"""
    pass


class FatalError(ToolError):
    """Error that should not be retried"""
    pass


def with_error_handling(
    default_return: Any = None,
    log_errors: bool = True,
    raise_on_error: bool = False
):
    """
    Decorator to add consistent error handling to functions.

    Args:
        default_return: Value to return on error
        log_errors: Whether to log errors
        raise_on_error: Whether to re-raise exceptions

    Usage:
        @with_error_handling(default_return={"success": False})
        def my_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.error(
                        f"{func.__name__} failed: {str(e)}",
                        exc_info=True,
                        extra={"args": args, "kwargs": kwargs}
                    )

                if raise_on_error:
                    raise

                if isinstance(default_return, dict) and "error" not in default_return:
                    result = default_return.copy()
                    result["error"] = str(e)
                    result["success"] = False
                    return result

                return default_return

        return wrapper
    return decorator


def with_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator to add retry logic with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff_factor: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to retry on

    Usage:
        @with_retry(max_attempts=3, delay=1.0)
        def my_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"{func.__name__} attempt {attempt + 1}/{max_attempts} failed: {str(e)}. "
                            f"Retrying in {current_delay}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {str(e)}"
                        )

            raise last_exception

        return wrapper
    return decorator


def with_fallback(*fallback_funcs: Callable):
    """
    Decorator to add fallback functions if primary fails.

    Args:
        *fallback_funcs: Functions to try in order if primary fails

    Usage:
        @with_fallback(fallback_func1, fallback_func2)
        def primary_func():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Try primary function
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"{func.__name__} failed: {str(e)}, trying fallbacks...")

                # Try fallback functions in order
                for i, fallback in enumerate(fallback_funcs):
                    try:
                        logger.info(f"Attempting fallback {i + 1}/{len(fallback_funcs)}")
                        return fallback(*args, **kwargs)
                    except Exception as fallback_error:
                        logger.warning(f"Fallback {i + 1} failed: {str(fallback_error)}")

                # All failed
                logger.error(f"All fallbacks failed for {func.__name__}")
                raise

        return wrapper
    return decorator


def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: type = Exception
):
    """
    Circuit breaker pattern to prevent cascading failures.

    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before attempting recovery
        expected_exception: Exception type to count as failure

    Usage:
        @circuit_breaker(failure_threshold=5, recovery_timeout=60)
        def my_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        func.failure_count = 0
        func.circuit_open = False
        func.last_failure_time = None

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Check if circuit is open
            if func.circuit_open:
                if time.time() - func.last_failure_time > recovery_timeout:
                    logger.info(f"Circuit breaker attempting recovery for {func.__name__}")
                    func.circuit_open = False
                    func.failure_count = 0
                else:
                    raise ToolError(
                        f"Circuit breaker open for {func.__name__}",
                        severity=ErrorSeverity.HIGH
                    )

            try:
                result = func(*args, **kwargs)
                func.failure_count = 0  # Reset on success
                return result
            except expected_exception as e:
                func.failure_count += 1
                func.last_failure_time = time.time()

                if func.failure_count >= failure_threshold:
                    func.circuit_open = True
                    logger.error(
                        f"Circuit breaker opened for {func.__name__} "
                        f"after {failure_threshold} failures"
                    )

                raise

        return wrapper
    return decorator


def safe_tool_execution(tool_func: Callable, *args, **kwargs) -> Dict[str, Any]:
    """
    Safely execute a tool function with consistent error handling.

    Args:
        tool_func: Tool function to execute
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        Dictionary with success status and result/error
    """
    try:
        result = tool_func(*args, **kwargs)

        # Ensure result is a dict with success field
        if not isinstance(result, dict):
            result = {"result": result}

        if "success" not in result:
            result["success"] = True

        return result

    except ToolError as e:
        logger.error(f"Tool error in {tool_func.__name__}: {str(e)}", extra={"context": e.context})
        return {
            "success": False,
            "error": str(e),
            "severity": e.severity.value,
            "context": e.context
        }

    except Exception as e:
        logger.error(f"Unexpected error in {tool_func.__name__}: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "severity": ErrorSeverity.HIGH.value
        }


def aggregate_errors(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate errors from multiple tool results.

    Args:
        results: List of tool result dictionaries

    Returns:
        Aggregated result with error summary
    """
    errors = []
    successes = []

    for result in results:
        if result.get("success", False):
            successes.append(result)
        else:
            errors.append(result)

    if not errors:
        return {
            "success": True,
            "results": successes,
            "total": len(results),
            "successful": len(successes)
        }

    return {
        "success": False if not successes else "partial",
        "results": successes,
        "errors": errors,
        "total": len(results),
        "successful": len(successes),
        "failed": len(errors)
    }


__all__ = [
    'ErrorSeverity',
    'ToolError',
    'RetryableError',
    'FatalError',
    'with_error_handling',
    'with_retry',
    'with_fallback',
    'circuit_breaker',
    'safe_tool_execution',
    'aggregate_errors',
]
