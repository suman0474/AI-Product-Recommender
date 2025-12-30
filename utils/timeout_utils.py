"""
Timeout Utilities for LLM and Tool Calls

Provides decorators and context managers for timeout handling.
"""

import functools
import signal
import logging
from typing import Any, Callable, Optional
from concurrent.futures import TimeoutError as FuturesTimeoutError
import threading

logger = logging.getLogger(__name__)


class TimeoutException(Exception):
    """Raised when an operation exceeds its timeout"""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout"""
    raise TimeoutException("Operation timed out")


def with_timeout(seconds: int):
    """
    Decorator to add timeout to any function.

    Args:
        seconds: Timeout in seconds

    Usage:
        @with_timeout(30)
        def my_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Use threading for timeout (works on Windows)
            result = [TimeoutException("Function timed out")]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    result[0] = e

            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout=seconds)

            if thread.is_alive():
                logger.error(f"{func.__name__} timed out after {seconds}s")
                raise TimeoutException(f"{func.__name__} exceeded timeout of {seconds}s")

            if isinstance(result[0], Exception):
                raise result[0]

            return result[0]

        return wrapper
    return decorator


def invoke_with_timeout(chain, input_data: dict, timeout: int = 60) -> Any:
    """
    Invoke a LangChain chain with timeout.

    Args:
        chain: LangChain runnable/chain
        input_data: Input dictionary
        timeout: Timeout in seconds

    Returns:
        Chain output

    Raises:
        TimeoutException: If operation exceeds timeout
    """
    @with_timeout(timeout)
    def _invoke():
        return chain.invoke(input_data)

    try:
        return _invoke()
    except TimeoutException as e:
        logger.error(f"Chain invocation timed out after {timeout}s: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Chain invocation failed: {str(e)}")
        raise


def batch_invoke_with_timeout(chain, inputs: list, timeout: int = 120) -> list:
    """
    Batch invoke a LangChain chain with timeout.

    Args:
        chain: LangChain runnable/chain
        inputs: List of input dictionaries
        timeout: Timeout in seconds (total for all)

    Returns:
        List of chain outputs
    """
    @with_timeout(timeout)
    def _batch():
        return chain.batch(inputs)

    try:
        return _batch()
    except TimeoutException as e:
        logger.error(f"Batch invocation timed out after {timeout}s")
        raise
    except Exception as e:
        logger.error(f"Batch invocation failed: {str(e)}")
        raise


class TimeoutContext:
    """
    Context manager for timeout operations.

    Usage:
        with TimeoutContext(30):
            result = expensive_operation()
    """
    def __init__(self, seconds: int):
        self.seconds = seconds
        self.timer = None

    def __enter__(self):
        def timeout_handler():
            raise TimeoutException(f"Operation exceeded timeout of {self.seconds}s")

        self.timer = threading.Timer(self.seconds, timeout_handler)
        self.timer.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.timer:
            self.timer.cancel()
        return False


__all__ = [
    'TimeoutException',
    'with_timeout',
    'invoke_with_timeout',
    'batch_invoke_with_timeout',
    'TimeoutContext',
]
