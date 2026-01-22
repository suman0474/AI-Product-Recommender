"""
Lock Monitor
Provides visibility into thread lock contention for debugging and performance monitoring
"""
import threading
import time
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class MonitoredLock:
    """
    Thread lock with monitoring and contention tracking.

    Wraps threading.Lock to provide visibility into lock wait times,
    acquisition counts, and contention patterns.
    """

    def __init__(self, name: str):
        """
        Initialize monitored lock.

        Args:
            name: Lock name for identification in logs
        """
        self.name = name
        self._lock = threading.Lock()
        self._stats = {
            "acquisitions": 0,
            "total_wait_time": 0.0,
            "max_wait_time": 0.0,
            "contentions": 0  # waits > 1ms
        }
        logger.debug(f"[LOCK_MONITOR] Created lock: {name}")

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        """
        Acquire lock with monitoring.

        Args:
            blocking: Block until acquired
            timeout: Max time to wait (-1 for infinite)

        Returns:
            True if acquired, False otherwise
        """
        start_time = time.time()
        acquired = self._lock.acquire(blocking, timeout)

        if acquired:
            wait_time = time.time() - start_time

            # Track contention (waits > 1ms)
            if wait_time > 0.001:
                self._stats["contentions"] += 1
                logger.debug(
                    f"[LOCK_MONITOR] {self.name} waited {wait_time*1000:.1f}ms (contention)"
                )

            self._stats["acquisitions"] += 1
            self._stats["total_wait_time"] += wait_time
            self._stats["max_wait_time"] = max(self._stats["max_wait_time"], wait_time)

        return acquired

    def release(self):
        """Release lock."""
        self._lock.release()

    def __enter__(self):
        """Context manager entry."""
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get lock contention statistics.

        Returns:
            Dictionary with lock statistics
        """
        avg_wait_ms = 0.0
        contention_rate = 0.0

        if self._stats["acquisitions"] > 0:
            avg_wait_ms = (
                self._stats["total_wait_time"] / self._stats["acquisitions"] * 1000
            )
            contention_rate = (
                self._stats["contentions"] / self._stats["acquisitions"] * 100
            )

        return {
            "name": self.name,
            "acquisitions": self._stats["acquisitions"],
            "avg_wait_ms": f"{avg_wait_ms:.2f}",
            "max_wait_ms": f"{self._stats['max_wait_time']*1000:.2f}",
            "contentions": self._stats["contentions"],
            "contention_rate_percent": f"{contention_rate:.1f}"
        }

    def reset_stats(self) -> None:
        """Reset statistics (for testing)."""
        self._stats = {
            "acquisitions": 0,
            "total_wait_time": 0.0,
            "max_wait_time": 0.0,
            "contentions": 0
        }
        logger.debug(f"[LOCK_MONITOR] Reset stats for {self.name}")


class LockMonitoringManager:
    """Global lock monitoring for system visibility."""

    def __init__(self):
        """Initialize lock monitoring manager."""
        self._locks: Dict[str, MonitoredLock] = {}
        self._manager_lock = threading.Lock()

    def get_or_create_lock(self, name: str) -> MonitoredLock:
        """
        Get or create a monitored lock by name.

        Args:
            name: Lock name

        Returns:
            MonitoredLock instance
        """
        with self._manager_lock:
            if name not in self._locks:
                self._locks[name] = MonitoredLock(name)
                logger.info(f"[LOCK_MONITORING] Created lock: {name}")
            return self._locks[name]

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all monitored locks.

        Returns:
            Dictionary of lock name -> stats
        """
        with self._manager_lock:
            return {
                name: lock.get_stats()
                for name, lock in self._locks.items()
            }

    def get_lock_stats(self, name: str) -> Dict[str, Any]:
        """
        Get statistics for specific lock.

        Args:
            name: Lock name

        Returns:
            Lock statistics or empty dict if not found
        """
        with self._manager_lock:
            if name in self._locks:
                return self._locks[name].get_stats()
            return {}


# Global monitoring manager
_monitoring_manager = None


def get_lock_monitoring_manager() -> LockMonitoringManager:
    """Get global lock monitoring manager."""
    global _monitoring_manager
    if _monitoring_manager is None:
        _monitoring_manager = LockMonitoringManager()
        logger.info("[LOCK_MONITORING] Global manager created")
    return _monitoring_manager


def get_monitored_lock(name: str) -> MonitoredLock:
    """
    Get or create a monitored lock.

    Args:
        name: Lock name

    Returns:
        MonitoredLock instance
    """
    manager = get_lock_monitoring_manager()
    return manager.get_or_create_lock(name)
