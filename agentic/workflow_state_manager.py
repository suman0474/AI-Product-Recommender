"""
Workflow State Manager with Bounded Memory & Auto-Cleanup

Provides thread-safe, bounded storage for workflow states with automatic
background cleanup and LRU eviction to prevent OOM crashes under load.
"""
import logging
import threading
import time
from collections import OrderedDict
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class BoundedWorkflowStateManager:
    """
    Bounded workflow state storage with automatic cleanup and LRU eviction.

    Features:
    - Bounded memory: Max state count with LRU eviction
    - Automatic cleanup: Background thread removes expired states
    - Thread-safe: All operations protected by lock
    - TTL-based expiration: States older than TTL are removed
    """

    def __init__(
        self,
        max_states: int = 10000,
        ttl_seconds: int = 3600,
        cleanup_interval: int = 300
    ):
        """
        Initialize bounded state manager.

        Args:
            max_states: Maximum number of concurrent states (default 10,000)
            ttl_seconds: Time-to-live for states in seconds (default 1 hour)
            cleanup_interval: How often to cleanup expired states (default 5 min)
        """
        self.max_states = max_states
        self.ttl_seconds = ttl_seconds
        self.cleanup_interval = cleanup_interval

        # OrderedDict for LRU eviction (move_to_end when accessed)
        self._states: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

        # Background cleanup thread
        self._cleanup_thread: Optional[threading.Thread] = None
        self._running = False
        self._stop_event = threading.Event()

        # Statistics
        self._stats = {
            "total_states": 0,
            "total_evictions": 0,
            "total_cleanups": 0,
            "last_cleanup": None
        }

        logger.info(
            f"[WORKFLOW_STATE_MANAGER] Initialized: max_states={max_states}, "
            f"ttl={ttl_seconds}s, cleanup_interval={cleanup_interval}s"
        )

    def start(self) -> None:
        """Start background cleanup thread."""
        if self._running:
            logger.warning("[WORKFLOW_STATE_MANAGER] Cleanup thread already running")
            return

        self._running = True
        self._stop_event.clear()

        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=False,  # Non-daemon for proper shutdown
            name="workflow_state_cleanup"
        )
        self._cleanup_thread.start()
        logger.info("[WORKFLOW_STATE_MANAGER] Background cleanup thread started")

    def stop(self) -> None:
        """Stop background cleanup thread gracefully."""
        if not self._running or not self._cleanup_thread:
            return

        logger.info("[WORKFLOW_STATE_MANAGER] Stopping cleanup thread...")
        self._running = False
        self._stop_event.set()

        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)

        if self._cleanup_thread and self._cleanup_thread.is_alive():
            logger.warning("[WORKFLOW_STATE_MANAGER] Cleanup thread did not stop in timeout")
        else:
            logger.info("[WORKFLOW_STATE_MANAGER] Cleanup thread stopped")

    def _cleanup_loop(self) -> None:
        """Main background cleanup loop."""
        while self._running:
            try:
                self._cleanup_expired_states()
            except Exception as e:
                logger.error(f"[WORKFLOW_STATE_MANAGER] Cleanup error: {e}")

            # Wait for next cleanup or stop signal
            if self._stop_event.wait(self.cleanup_interval):
                break  # Stop event was set, exit loop

    def _cleanup_expired_states(self) -> None:
        """Remove states older than TTL."""
        with self._lock:
            if not self._states:
                return

            now = time.time()
            expired = [
                thread_id for thread_id, state in self._states.items()
                if now - state.get("_created_at", 0) > self.ttl_seconds
            ]

            for thread_id in expired:
                del self._states[thread_id]

            if expired:
                self._stats["total_cleanups"] += 1
                self._stats["last_cleanup"] = time.time()
                logger.info(
                    f"[WORKFLOW_STATE_MANAGER] Cleanup: Removed {len(expired)} "
                    f"expired states (total: {len(self._states)})"
                )

    def get(self, thread_id: str) -> Dict[str, Any]:
        """
        Get workflow state by thread ID.

        Returns a copy of the state. Updates LRU order.

        Args:
            thread_id: Workflow thread ID

        Returns:
            Copy of workflow state, or empty dict if not found
        """
        with self._lock:
            if thread_id not in self._states:
                return {}

            state = self._states[thread_id]

            # Check TTL
            if time.time() - state.get("_created_at", 0) > self.ttl_seconds:
                del self._states[thread_id]
                return {}

            # Move to end for LRU
            self._states.move_to_end(thread_id)

            return state.copy()

    def set(self, thread_id: str, state: Dict[str, Any]) -> None:
        """
        Save or update workflow state.

        Enforces size limit with LRU eviction when necessary.
        Adds timestamps for TTL tracking.

        Args:
            thread_id: Workflow thread ID
            state: State dictionary to save
        """
        with self._lock:
            # Add timestamps
            state["_created_at"] = time.time()
            state["_updated_at"] = time.time()

            # If state exists, move to end (most recently used)
            if thread_id in self._states:
                self._states.move_to_end(thread_id)
                self._states[thread_id] = state
                return

            # New state
            self._states[thread_id] = state
            self._stats["total_states"] = max(
                self._stats["total_states"],
                len(self._states)
            )

            # Enforce size limit with LRU eviction
            while len(self._states) > self.max_states:
                # Remove oldest (first) item
                oldest_id, _ = self._states.popitem(last=False)
                self._stats["total_evictions"] += 1

                logger.warning(
                    f"[WORKFLOW_STATE_MANAGER] Evicted old state: {oldest_id} "
                    f"(capacity limit reached: {self.max_states})"
                )

    def delete(self, thread_id: str) -> bool:
        """
        Delete a workflow state explicitly.

        Args:
            thread_id: Workflow thread ID to delete

        Returns:
            True if state was deleted, False if not found
        """
        with self._lock:
            if thread_id in self._states:
                del self._states[thread_id]
                logger.debug(f"[WORKFLOW_STATE_MANAGER] Deleted state: {thread_id}")
                return True
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get manager statistics.

        Returns:
            Dictionary with current stats and memory info
        """
        with self._lock:
            return {
                "current_states": len(self._states),
                "max_capacity": self.max_states,
                "usage_percent": (len(self._states) / self.max_states * 100),
                "total_states_ever": self._stats["total_states"],
                "total_evictions": self._stats["total_evictions"],
                "total_cleanups": self._stats["total_cleanups"],
                "last_cleanup": self._stats.get("last_cleanup"),
                "ttl_seconds": self.ttl_seconds,
                "cleanup_running": self._running
            }

    def clear(self) -> int:
        """
        Clear all states (useful for testing or reset).

        Returns:
            Number of states cleared
        """
        with self._lock:
            count = len(self._states)
            self._states.clear()
            logger.info(f"[WORKFLOW_STATE_MANAGER] Cleared {count} states")
            return count


# Global singleton instance
_manager: Optional[BoundedWorkflowStateManager] = None


def get_workflow_state_manager(
    max_states: int = 10000,
    ttl_seconds: int = 3600
) -> BoundedWorkflowStateManager:
    """
    Get or create global workflow state manager singleton.

    Args:
        max_states: Maximum states (only used on first call)
        ttl_seconds: TTL for states (only used on first call)

    Returns:
        Global BoundedWorkflowStateManager instance
    """
    global _manager
    if _manager is None:
        _manager = BoundedWorkflowStateManager(
            max_states=max_states,
            ttl_seconds=ttl_seconds
        )
        _manager.start()
    return _manager


def stop_workflow_state_manager() -> None:
    """Stop and cleanup the global workflow state manager."""
    global _manager
    if _manager:
        _manager.stop()
        _manager = None
        logger.info("[WORKFLOW_STATE_MANAGER] Global manager stopped and cleared")
