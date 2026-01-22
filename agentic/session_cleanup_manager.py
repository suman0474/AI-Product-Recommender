"""
Session Cleanup Manager
Automatic background cleanup of old session files to prevent memory leaks
"""
import os
import time
import threading
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class SessionCleanupManager:
    """Automatic session cleanup to prevent memory leaks."""

    def __init__(
        self,
        session_dir: str,
        cleanup_interval: int = 600,  # 10 minutes
        max_age_hours: int = 24  # 24 hours
    ):
        """
        Initialize session cleanup manager.

        Args:
            session_dir: Directory containing session files
            cleanup_interval: Seconds between cleanup runs (default 600 = 10 min)
            max_age_hours: Max hours to keep session files (default 24)
        """
        self.session_dir = Path(session_dir)
        self.cleanup_interval = cleanup_interval
        self.max_age_seconds = max_age_hours * 3600
        self._cleanup_thread = None
        self._running = False
        self._stop_event = threading.Event()
        self._stats = {
            "total_cleaned": 0,
            "last_cleanup": None,
            "errors": 0
        }

    def start(self):
        """Start background cleanup thread with proper lifecycle management."""
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            logger.warning("[SESSION_CLEANUP] Cleanup already running")
            return

        self._running = True
        self._stop_event.clear()
        # Use daemon=False for graceful shutdown (allows proper cleanup)
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=False,
            name="session_cleanup"
        )
        self._cleanup_thread.start()
        logger.info(
            f"[SESSION_CLEANUP] Started with proper lifecycle management: "
            f"interval={self.cleanup_interval}s, max_age={self.max_age_seconds/3600}h"
        )

    def stop(self):
        """Stop background cleanup thread gracefully."""
        if not self._cleanup_thread or not self._cleanup_thread.is_alive():
            return

        logger.info("[SESSION_CLEANUP] Initiating graceful shutdown...")
        self._running = False
        self._stop_event.set()

        # Wait with timeout for thread to finish
        self._cleanup_thread.join(timeout=5)

        if self._cleanup_thread.is_alive():
            logger.warning("[SESSION_CLEANUP] Thread did not shutdown in timeout")
        else:
            logger.info("[SESSION_CLEANUP] Shutdown complete")

    def _cleanup_loop(self):
        """Main cleanup loop with graceful shutdown support."""
        while self._running and not self._stop_event.is_set():
            try:
                self._cleanup_old_sessions()
            except Exception as e:
                logger.error(f"[SESSION_CLEANUP] Error during cleanup: {e}")
                self._stats["errors"] += 1

            # Wait for next cleanup interval
            self._stop_event.wait(self.cleanup_interval)

    def _cleanup_old_sessions(self):
        """Remove session files older than max_age."""
        if not self.session_dir.exists():
            return

        now = time.time()
        cleaned_count = 0
        errors_count = 0

        try:
            for session_file in self.session_dir.glob("*"):
                if session_file.is_file():
                    try:
                        file_age = now - session_file.stat().st_mtime
                        if file_age > self.max_age_seconds:
                            session_file.unlink()
                            cleaned_count += 1
                            logger.debug(
                                f"[SESSION_CLEANUP] Deleted session: "
                                f"{session_file.name} (age: {file_age/3600:.1f}h)"
                            )
                    except Exception as e:
                        logger.error(
                            f"[SESSION_CLEANUP] Failed to delete session {session_file.name}: {e}"
                        )
                        errors_count += 1

        except Exception as e:
            logger.error(f"[SESSION_CLEANUP] Error scanning session directory: {e}")
            return

        self._stats["total_cleaned"] += cleaned_count
        self._stats["last_cleanup"] = datetime.now().isoformat()

        if cleaned_count > 0 or errors_count > 0:
            logger.info(
                f"[SESSION_CLEANUP] Cleanup complete: "
                f"deleted={cleaned_count}, errors={errors_count}"
            )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cleanup statistics.

        Returns:
            Dictionary with cleanup stats
        """
        return self._stats.copy()


# Global instance
_session_cleanup_manager = None


def start_session_cleanup(
    session_dir: str,
    cleanup_interval: int = 600,
    max_age_hours: int = 24
) -> SessionCleanupManager:
    """
    Start automatic session cleanup.

    Args:
        session_dir: Directory containing session files
        cleanup_interval: Seconds between cleanup runs (default 600)
        max_age_hours: Max hours to keep files (default 24)

    Returns:
        SessionCleanupManager instance
    """
    global _session_cleanup_manager
    if _session_cleanup_manager is None:
        _session_cleanup_manager = SessionCleanupManager(
            session_dir,
            cleanup_interval,
            max_age_hours
        )
        _session_cleanup_manager.start()
    return _session_cleanup_manager


def stop_session_cleanup():
    """Stop automatic session cleanup."""
    global _session_cleanup_manager
    if _session_cleanup_manager:
        _session_cleanup_manager.stop()
