"""
Workflow Synchronization Utilities

Provides thread-safe execution, transaction semantics, and isolation
for LangGraph workflows to prevent race conditions and ensure data consistency.
"""

import functools
import logging
import threading
import time
import copy
from typing import Any, Callable, Dict, Optional, TypeVar
from contextlib import contextmanager
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

T = TypeVar('T')


# ============================================================================
# SESSION-LEVEL LOCKING
# ============================================================================

class WorkflowLockManager:
    """
    Manages locks per session_id to prevent concurrent workflow executions
    from interfering with each other.
    """

    def __init__(self):
        self._locks: Dict[str, threading.RLock] = {}
        self._lock_times: Dict[str, datetime] = {}
        self._manager_lock = threading.Lock()
        self._cleanup_interval = 3600  # 1 hour
        self._last_cleanup = time.time()

    def get_lock(self, session_id: str) -> threading.RLock:
        """
        Get or create a lock for a specific session.

        Args:
            session_id: Session identifier

        Returns:
            RLock for the session
        """
        with self._manager_lock:
            # Periodic cleanup of old locks
            if time.time() - self._last_cleanup > self._cleanup_interval:
                self._cleanup_old_locks()

            if session_id not in self._locks:
                self._locks[session_id] = threading.RLock()
                self._lock_times[session_id] = datetime.now()
                logger.debug(f"Created new lock for session {session_id}")

            return self._locks[session_id]

    def _cleanup_old_locks(self) -> None:
        """Remove locks that haven't been used in 24 hours"""
        cutoff = datetime.now() - timedelta(hours=24)
        expired = [
            sid for sid, lock_time in self._lock_times.items()
            if lock_time < cutoff
        ]

        for sid in expired:
            del self._locks[sid]
            del self._lock_times[sid]

        if expired:
            logger.info(f"Cleaned up {len(expired)} expired session locks")

        self._last_cleanup = time.time()

    def release_lock(self, session_id: str) -> None:
        """
        Explicitly release a session lock (optional, locks auto-release).

        Args:
            session_id: Session identifier
        """
        with self._manager_lock:
            if session_id in self._locks:
                del self._locks[session_id]
                del self._lock_times[session_id]
                logger.debug(f"Released lock for session {session_id}")


# Global lock manager instance
_lock_manager = WorkflowLockManager()


@contextmanager
def workflow_lock(session_id: str, timeout: float = 30.0):
    """
    Context manager for session-level workflow locking.

    Args:
        session_id: Session identifier
        timeout: Maximum time to wait for lock (seconds)

    Yields:
        None

    Raises:
        TimeoutError: If lock cannot be acquired within timeout

    Usage:
        with workflow_lock(session_id):
            # Execute workflow nodes
            ...
    """
    lock = _lock_manager.get_lock(session_id)

    acquired = lock.acquire(timeout=timeout)
    if not acquired:
        raise TimeoutError(
            f"Could not acquire workflow lock for session {session_id} "
            f"within {timeout} seconds. Another workflow may be running."
        )

    try:
        logger.debug(f"Acquired workflow lock for session {session_id}")
        yield
    finally:
        lock.release()
        logger.debug(f"Released workflow lock for session {session_id}")


# ============================================================================
# TRANSACTION SEMANTICS
# ============================================================================

class StateTransaction:
    """
    Provides transaction semantics for state modifications with rollback support.
    """

    def __init__(self, state: Dict[str, Any], auto_commit: bool = False):
        """
        Initialize transaction.

        Args:
            state: Original state dictionary
            auto_commit: If True, automatically commit on success
        """
        self.original_state = copy.deepcopy(state)
        self.working_state = state  # Reference to original
        self.auto_commit = auto_commit
        self.committed = False
        self.rolled_back = False

    def commit(self) -> None:
        """Commit changes (mark as successful)"""
        if self.rolled_back:
            raise RuntimeError("Cannot commit after rollback")

        self.committed = True
        logger.debug("Transaction committed")

    def rollback(self) -> None:
        """Rollback changes to original state"""
        if self.committed:
            logger.warning("Attempting rollback on committed transaction")
            return

        # Restore original state
        self.working_state.clear()
        self.working_state.update(self.original_state)
        self.rolled_back = True
        logger.info("Transaction rolled back")

    def __enter__(self):
        """Enter transaction context"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit transaction context with automatic rollback on error"""
        if exc_type is not None:
            # Exception occurred - rollback
            logger.error(f"Transaction failed with {exc_type.__name__}: {exc_val}")
            self.rollback()
            return False  # Re-raise exception

        if self.auto_commit and not self.committed:
            self.commit()

        return True


@contextmanager
def state_transaction(state: Dict[str, Any], auto_commit: bool = True):
    """
    Context manager for transactional state modifications.

    Args:
        state: State dictionary to protect
        auto_commit: Auto-commit on success, rollback on error

    Yields:
        StateTransaction object

    Usage:
        with state_transaction(state) as txn:
            state["field"] = "new value"
            # Automatically commits on success
            # Automatically rolls back on exception
    """
    txn = StateTransaction(state, auto_commit=auto_commit)
    try:
        yield txn
    except Exception as e:
        txn.rollback()
        raise


# ============================================================================
# PARALLEL EXECUTION WITH SYNCHRONIZATION
# ============================================================================

class ThreadSafeResultCollector:
    """
    Thread-safe collector for parallel execution results.
    """

    def __init__(self):
        self.results: list = []
        self.errors: list = []
        self._lock = threading.Lock()

    def add_result(self, result: Any) -> None:
        """Add a successful result"""
        with self._lock:
            self.results.append(result)

    def add_error(self, error: Exception, context: Optional[Dict] = None) -> None:
        """Add an error"""
        with self._lock:
            self.errors.append({
                "error": str(error),
                "type": type(error).__name__,
                "context": context or {},
                "timestamp": datetime.now().isoformat()
            })

    def get_results(self) -> list:
        """Get all results (thread-safe)"""
        with self._lock:
            return self.results.copy()

    def get_errors(self) -> list:
        """Get all errors (thread-safe)"""
        with self._lock:
            return self.errors.copy()

    def has_errors(self) -> bool:
        """Check if any errors occurred"""
        with self._lock:
            return len(self.errors) > 0

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        with self._lock:
            return {
                "total_results": len(self.results),
                "total_errors": len(self.errors),
                "success_rate": len(self.results) / (len(self.results) + len(self.errors))
                    if (len(self.results) + len(self.errors)) > 0 else 0.0
            }


# ============================================================================
# DECORATORS
# ============================================================================

def with_workflow_lock(session_id_param: str = "session_id", timeout: float = 30.0):
    """
    Decorator to add workflow-level locking to functions.

    Args:
        session_id_param: Name of parameter containing session_id
        timeout: Lock acquisition timeout

    Usage:
        @with_workflow_lock(session_id_param="session_id")
        def my_workflow(state: dict):
            session_id = state["session_id"]
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract session_id from args or kwargs
            session_id = None

            # Try to get from state dict (first arg for workflow nodes)
            if args and isinstance(args[0], dict):
                session_id = args[0].get(session_id_param)

            # Try kwargs
            if not session_id and session_id_param in kwargs:
                session_id = kwargs[session_id_param]

            if not session_id:
                logger.warning(
                    f"{func.__name__} called without session_id, "
                    "skipping workflow lock"
                )
                return func(*args, **kwargs)

            # Acquire lock and execute
            with workflow_lock(session_id, timeout=timeout):
                return func(*args, **kwargs)

        return wrapper
    return decorator


def with_state_transaction(auto_commit: bool = True):
    """
    Decorator to add transaction semantics to workflow nodes.

    Args:
        auto_commit: Auto-commit on success

    Usage:
        @with_state_transaction()
        def my_node(state: dict) -> dict:
            state["field"] = "value"
            return state
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # First arg should be state dict
            if not args or not isinstance(args[0], dict):
                logger.warning(
                    f"{func.__name__} called without state dict, "
                    "skipping transaction"
                )
                return func(*args, **kwargs)

            state = args[0]

            with state_transaction(state, auto_commit=auto_commit):
                return func(*args, **kwargs)

        return wrapper
    return decorator


__all__ = [
    'WorkflowLockManager',
    'workflow_lock',
    'StateTransaction',
    'state_transaction',
    'ThreadSafeResultCollector',
    'with_workflow_lock',
    'with_state_transaction',
]
