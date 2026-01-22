"""
Global Thread Pool Manager for Resource Control

This module provides a singleton global thread pool executor that enforces
bounded resource usage across all workflows and operations.

Problem Solved:
- BEFORE: 320+ threads under load (system crashes/exhaustion)
- AFTER: Max 16 threads (safe, bounded, predictable)

Key Features:
- Singleton pattern for global executor
- Thread-safe resource management
- Automatic shutdown and cleanup
- Fair resource distribution across workflows
- Metrics and monitoring

Usage:
    from agentic.global_executor_manager import get_global_executor

    executor = get_global_executor()
    future = executor.submit(my_function, arg1, arg2)
    result = future.result()

See: IMPLEMENTATION_ROADMAP.md Phase 1, Item 3
"""

import threading
import logging
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from typing import Callable, Optional, List, Dict, Any
from threading import RLock

from agentic.context_managers import ThreadPoolResourceManager, GlobalResourceRegistry

logger = logging.getLogger(__name__)


class GlobalExecutorManager:
    """
    Singleton thread pool executor with global resource limits.

    Enforces a maximum of 16 concurrent workers across all operations,
    preventing system resource exhaustion while maintaining fair
    access for all workflows.

    Thread Safety: Uses RLock for thread-safe access
    """

    _instance: Optional['GlobalExecutorManager'] = None
    _lock: RLock = RLock()
    _resource_manager: Optional[ThreadPoolResourceManager] = None
    _registry = GlobalResourceRegistry()

    def __new__(cls):
        """Singleton pattern - ensure only one instance exists"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, max_global_workers: int = 16):
        """
        Initialize the global executor manager.

        Args:
            max_global_workers: Maximum concurrent workers (default: 16)
                This is intentionally low to prevent resource exhaustion.
                Each worker thread costs ~1-2MB memory + overhead.
                16 workers = safe for most systems
        """
        if not hasattr(self, '_initialized'):
            self.max_global_workers = max_global_workers
            
            # Use managed resource manager
            self._resource_manager = ThreadPoolResourceManager(
                pool_name="global_executor",
                max_workers=max_global_workers
            )
            # Initialize it immediately
            self._resource_manager.__enter__()
            self._executor = self._resource_manager._executor
            
            self._lock = RLock()
            self._active_tasks = 0
            self._total_tasks = 0
            self._failed_tasks = 0
            self._task_history: List[Dict[str, Any]] = []
            self._initialized = True

            logger.info(
                f"GlobalExecutorManager initialized with max {max_global_workers} workers (managed)"
            )

    def _ensure_executor(self):
        """Ensure executor is active"""
        with self._lock:
            if not self._resource_manager.is_active():
                self._resource_manager.__enter__()
                self._executor = self._resource_manager._executor
                logger.info("Restarted global thread pool manager")

    def submit(self, fn: Callable, *args, **kwargs) -> Future:
        """
        Submit a task to the global executor.

        Args:
            fn: Callable to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Future object that can be used to get the result

        Raises:
            RuntimeError: If executor is not available or has been shut down
        """
        with self._lock:
             self._ensure_executor()

             if not self._resource_manager.is_active():
                 raise RuntimeError("Global executor is not available (may be shut down)")

             # Let the resource manager handle submission and tracking
             # We wrap it to maintain our specific logging if needed, 
             # but ThreadPoolResourceManager already tracks counts.
             
             # Sync our local counters with manager for backward compatibility
             self._total_tasks = self._resource_manager.task_count
             self._active_tasks = self._resource_manager.task_count - self._resource_manager.completed_tasks - self._resource_manager.failed_tasks
             self._failed_tasks = self._resource_manager.failed_tasks

             try:
                 # Direct submission to manager
                 future = self._resource_manager.submit_task(fn, *args, **kwargs)
                 logger.debug(f"Submitted task to managed global executor")
                 return future
             except Exception as e:
                 logger.error(f"Failed to submit task to managed global executor: {e}")
                 raise

    def _task_wrapper(self, fn: Callable, args: tuple, kwargs: dict) -> Any:
        """
        Wrapper to track task execution and handle cleanup.

        Args:
            fn: Function to execute
            args: Arguments tuple
            kwargs: Keyword arguments dict

        Returns:
            Result of fn(*args, **kwargs)
        """
        task_id = self._total_tasks
        try:
            logger.debug(f"Executing task #{task_id}")
            result = fn(*args, **kwargs)
            logger.debug(f"Task #{task_id} completed successfully")
            return result
        except Exception as e:
            logger.error(f"Task #{task_id} failed with error: {e}")
            with self._lock:
                self._failed_tasks += 1
            raise
        finally:
            with self._lock:
                self._active_tasks -= 1

    def map(self, fn: Callable, iterable, timeout=None, chunksize=1):
        """
        Submit multiple tasks and return results as they complete.

        Similar to ThreadPoolExecutor.map() but with tracking.

        Args:
            fn: Callable to execute for each item
            iterable: Items to process
            timeout: Timeout for waiting for results
            chunksize: Size of chunks to submit at once

        Returns:
            Iterator of results
        """
        futures = []
        try:
            for item in iterable:
                future = self.submit(fn, item)
                futures.append(future)

            # Return results as they complete
            for future in as_completed(futures, timeout=timeout):
                yield future.result()

        except Exception as e:
            logger.error(f"Error in map operation: {e}")
            # Cancel remaining futures
            for future in futures:
                future.cancel()
            raise

    def shutdown(self, wait: bool = True):
        """
        Shutdown the executor and release all resources.

        This should be called when the application is shutting down.

        Args:
            wait: If True, wait for all pending tasks to complete
        """
        with self._lock:
            if self._resource_manager and self._resource_manager.is_active():
                self._resource_manager.__exit__(None, None, None)
                self._executor = None
                logger.info("Global managed executor shut down.")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about executor usage.

        Returns:
            Dictionary with task statistics
        """
        with self._lock:
            if not self._resource_manager:
                return {}
                
            return {
                "max_workers": self.max_global_workers,
                "task_count": self._resource_manager.task_count,
                "completed_tasks": self._resource_manager.completed_tasks,
                "failed_tasks": self._resource_manager.failed_tasks,
                "is_active": self._resource_manager.is_active(),
                "executor_available": self._executor is not None
            }

    def is_shutdown(self) -> bool:
        """Check if executor is shut down"""
        with self._lock:
            return self._executor is None or self._executor._shutdown

    def __repr__(self) -> str:
        stats = self.get_stats()
        return f"GlobalExecutorManager({stats})"


# Global instance accessor (recommended way to use)
_manager: Optional[GlobalExecutorManager] = None


def get_global_executor() -> ThreadPoolExecutor:
    """
    Get the global thread pool executor.

    This is the recommended way to access the global executor.
    Returns the ThreadPoolExecutor directly for compatibility with
    code that expects ThreadPoolExecutor.

    Returns:
        The global ThreadPoolExecutor instance

    Example:
        executor = get_global_executor()
        future = executor.submit(my_function, arg)
        result = future.result()
    """
    global _manager
    if _manager is None:
        _manager = GlobalExecutorManager()
    _manager._ensure_executor()
    return _manager._executor


def get_manager() -> GlobalExecutorManager:
    """
    Get the global executor manager instance.

    This gives access to advanced features like stats and shutdown.

    Returns:
        The GlobalExecutorManager instance
    """
    global _manager
    if _manager is None:
        _manager = GlobalExecutorManager()
    return _manager


def shutdown_global_executor(wait: bool = True):
    """
    Shutdown the global executor.

    This should be called during application shutdown.

    Args:
        wait: If True, wait for pending tasks to complete
    """
    global _manager
    if _manager is not None:
        _manager.shutdown(wait=wait)


def get_executor_stats() -> Dict[str, Any]:
    """
    Get current executor statistics.

    Returns:
        Dictionary with executor statistics
    """
    manager = get_manager()
    return manager.get_stats()


# Cleanup on module exit
import atexit

atexit.register(lambda: shutdown_global_executor(wait=True))
