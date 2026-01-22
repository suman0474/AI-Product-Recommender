"""
Orchestrator Utilities - Helper functions and background tasks

Provides:
- Singleton getters for SessionOrchestrator and InstanceManager
- Thread ID validation utilities
- Background cleanup tasks
- Logging configuration
- Coordination between orchestrators

Usage:
    from orchestrator_utils import (
        get_session_orchestrator,
        get_instance_manager,
        start_background_cleanup,
        validate_thread_id
    )
"""

import threading
import time
import re
import logging
from typing import Optional, Dict, Any, Tuple

from .session_orchestrator import SessionOrchestrator, get_session_orchestrator
from .instance_manager import WorkflowInstanceManager, get_instance_manager, InstanceStatus

logger = logging.getLogger(__name__)


# ============================================================================
# THREAD ID VALIDATION
# ============================================================================

# Thread ID patterns
MAIN_THREAD_ID_PATTERN = re.compile(r'^main_[a-zA-Z0-9_]+_[a-f0-9-]+_\d+$')
WORKFLOW_THREAD_ID_PATTERN = re.compile(r'^[a-z_]+_[a-f0-9]+_[a-f0-9-]+_\d+$')
ITEM_THREAD_ID_PATTERN = re.compile(r'^item_[a-f0-9]+_[a-f0-9-]+_\d+$')


def validate_main_thread_id(thread_id: str) -> bool:
    """
    Validate main thread ID format

    Expected format: main_{user_id}_{uuid}_{timestamp}
    Example: main_user123_550e8400-e29b-41d4-a716_1705920000

    Args:
        thread_id: Thread ID to validate

    Returns:
        True if valid format, False otherwise
    """
    if not thread_id:
        return False

    # Check starts with "main_"
    if not thread_id.startswith("main_"):
        return False

    # Basic structure check (at least 4 parts)
    parts = thread_id.split("_")
    if len(parts) < 4:
        return False

    return True


def validate_workflow_thread_id(thread_id: str) -> bool:
    """
    Validate workflow thread ID format

    Expected format: {workflow_type}_{main_ref}_{uuid}_{timestamp}
    Example: instrument_identifier_user1_550e8400_1705920000

    Args:
        thread_id: Thread ID to validate

    Returns:
        True if valid format, False otherwise
    """
    if not thread_id:
        return False

    # Should have at least 3 parts separated by underscore
    parts = thread_id.split("_")
    if len(parts) < 3:
        return False

    # Should not start with "main_"
    if thread_id.startswith("main_"):
        return False

    return True


def validate_item_thread_id(thread_id: str) -> bool:
    """
    Validate item thread ID format

    Expected format: item_{hash}_{uuid}_{timestamp}
    Example: item_a1b2c3d4_550e8400-e29b_1705920000

    Args:
        thread_id: Thread ID to validate

    Returns:
        True if valid format, False otherwise
    """
    if not thread_id:
        return False

    # Check starts with "item_"
    if not thread_id.startswith("item_"):
        return False

    # Should have at least 3 parts
    parts = thread_id.split("_")
    if len(parts) < 3:
        return False

    return True


def validate_thread_id(thread_id: str) -> bool:
    """
    Validate any thread ID format

    Args:
        thread_id: Thread ID to validate

    Returns:
        True if valid format, False otherwise
    """
    if not thread_id:
        return False

    return (
        validate_main_thread_id(thread_id) or
        validate_workflow_thread_id(thread_id) or
        validate_item_thread_id(thread_id)
    )


def extract_user_from_main_thread_id(main_thread_id: str) -> Optional[str]:
    """
    Extract user ID from main thread ID

    Args:
        main_thread_id: Main thread ID (e.g., "main_user123_uuid_ts")

    Returns:
        User ID if extractable, None otherwise
    """
    if not main_thread_id or not main_thread_id.startswith("main_"):
        return None

    parts = main_thread_id.split("_")
    if len(parts) >= 2:
        return parts[1]

    return None


def get_thread_type(thread_id: str) -> Optional[str]:
    """
    Determine thread ID type

    Args:
        thread_id: Thread ID to check

    Returns:
        "main", "workflow", "item", or None if invalid
    """
    if validate_main_thread_id(thread_id):
        return "main"
    elif validate_item_thread_id(thread_id):
        return "item"
    elif validate_workflow_thread_id(thread_id):
        return "workflow"
    return None


# ============================================================================
# ORCHESTRATOR COORDINATION
# ============================================================================

def verify_session_for_request(main_thread_id: str) -> Tuple[bool, Optional[str]]:
    """
    Verify session exists and is active for a request

    Args:
        main_thread_id: Session ID from request

    Returns:
        Tuple of (is_valid, error_message)
        - (True, None) if session is valid
        - (False, error_message) if session is invalid
    """
    if not main_thread_id:
        return False, "main_thread_id is required"

    if not validate_main_thread_id(main_thread_id):
        return False, f"Invalid main_thread_id format: {main_thread_id}"

    orchestrator = get_session_orchestrator()
    session = orchestrator.get_session_context(main_thread_id)

    if session is None:
        return False, f"Session not found: {main_thread_id}"

    if not session.active:
        return False, f"Session is not active: {main_thread_id}"

    return True, None


def get_or_create_instance(
    session_id: str,
    thread_id: str,
    workflow_type: str,
    parent_workflow_id: str,
    trigger_source: str,
    parent_workflow_type: Optional[str] = None,
    metadata: Dict[str, Any] = None
) -> Tuple['InstanceMetadata', bool]:
    """
    Get existing instance or create new one (handles deduplication)

    Args:
        session_id: Main thread ID of the session
        thread_id: LangGraph thread ID for this instance
        workflow_type: Type of workflow
        parent_workflow_id: ID of parent workflow
        trigger_source: What triggered this
        parent_workflow_type: Type of parent workflow
        metadata: Additional metadata

    Returns:
        Tuple of (instance, is_new)
        - instance: The InstanceMetadata (existing or new)
        - is_new: True if newly created, False if existing
    """
    manager = get_instance_manager()

    # Check for existing instance
    existing = manager.get_instance_by_trigger(
        session_id, workflow_type, parent_workflow_id, trigger_source
    )

    if existing:
        # Rerun scenario - increment request count
        manager.increment_request_count(existing.instance_id)
        logger.info(
            f"[ORCHESTRATOR] Rerunning existing instance {existing.instance_id} "
            f"(request_count={existing.request_count})"
        )
        return existing, False

    # New scenario - create new instance
    instance = manager.create_instance(
        session_id=session_id,
        thread_id=thread_id,
        workflow_type=workflow_type,
        parent_workflow_id=parent_workflow_id,
        trigger_source=trigger_source,
        parent_workflow_type=parent_workflow_type,
        metadata=metadata
    )

    logger.info(
        f"[ORCHESTRATOR] Created new instance {instance.instance_id} "
        f"for trigger={trigger_source}"
    )

    return instance, True


def complete_instance_with_result(
    instance_id: str,
    result: Dict[str, Any] = None,
    error: str = None
) -> bool:
    """
    Mark instance as completed or errored

    Args:
        instance_id: Instance ID
        result: Result data (if successful)
        error: Error message (if failed)

    Returns:
        True if successful, False otherwise
    """
    manager = get_instance_manager()

    if error:
        return manager.set_instance_error(instance_id, error)
    else:
        return manager.set_instance_completed(instance_id, result)


# ============================================================================
# BACKGROUND CLEANUP TASKS
# ============================================================================

class BackgroundCleanupTask:
    """
    Background task for cleaning up expired sessions and instances.

    Runs periodically to remove:
    - Sessions that have been inactive for too long
    - Instances that have timed out
    - Workflows that have been inactive
    """

    def __init__(
        self,
        session_cleanup_interval: int = 60,      # 1 minute
        instance_cleanup_interval: int = 300,    # 5 minutes
        session_ttl_minutes: int = 30,
        workflow_ttl_minutes: int = 60
    ):
        self._session_cleanup_interval = session_cleanup_interval
        self._instance_cleanup_interval = instance_cleanup_interval
        self._session_ttl_minutes = session_ttl_minutes
        self._workflow_ttl_minutes = workflow_ttl_minutes

        self._running = False
        self._session_thread: Optional[threading.Thread] = None
        self._instance_thread: Optional[threading.Thread] = None

    def start(self):
        """Start background cleanup threads"""
        if self._running:
            logger.warning("[CLEANUP] Background cleanup already running")
            return

        self._running = True

        # Start session cleanup thread
        self._session_thread = threading.Thread(
            target=self._session_cleanup_loop,
            daemon=True,
            name="session-cleanup"
        )
        self._session_thread.start()

        # Start instance cleanup thread
        self._instance_thread = threading.Thread(
            target=self._instance_cleanup_loop,
            daemon=True,
            name="instance-cleanup"
        )
        self._instance_thread.start()

        logger.info("[CLEANUP] Background cleanup tasks started")

    def stop(self):
        """Stop background cleanup threads"""
        self._running = False
        logger.info("[CLEANUP] Background cleanup tasks stopping...")

    def _session_cleanup_loop(self):
        """Session cleanup loop"""
        while self._running:
            try:
                orchestrator = get_session_orchestrator()

                # Clean up expired sessions
                sessions_cleaned = orchestrator.cleanup_expired_sessions(
                    self._session_ttl_minutes
                )

                # Clean up inactive workflows
                workflows_cleaned = orchestrator.cleanup_inactive_workflows(
                    self._workflow_ttl_minutes
                )

                if sessions_cleaned or workflows_cleaned:
                    logger.debug(
                        f"[CLEANUP] Session cleanup: {sessions_cleaned} sessions, "
                        f"{workflows_cleaned} workflows"
                    )

            except Exception as e:
                logger.error(f"[CLEANUP] Session cleanup error: {e}")

            # Sleep until next interval
            time.sleep(self._session_cleanup_interval)

    def _instance_cleanup_loop(self):
        """Instance cleanup loop"""
        while self._running:
            try:
                manager = get_instance_manager()

                # Clean up timed-out instances
                instances_cleaned = manager.cleanup_all_timed_out()

                if instances_cleaned:
                    logger.debug(
                        f"[CLEANUP] Instance cleanup: {instances_cleaned} instances"
                    )

            except Exception as e:
                logger.error(f"[CLEANUP] Instance cleanup error: {e}")

            # Sleep until next interval
            time.sleep(self._instance_cleanup_interval)


# Global cleanup task instance
_cleanup_task: Optional[BackgroundCleanupTask] = None
_cleanup_lock = threading.Lock()


def start_background_cleanup(
    session_cleanup_interval: int = 60,
    instance_cleanup_interval: int = 300,
    session_ttl_minutes: int = 30,
    workflow_ttl_minutes: int = 60
) -> BackgroundCleanupTask:
    """
    Start background cleanup tasks (singleton)

    Args:
        session_cleanup_interval: Seconds between session cleanups
        instance_cleanup_interval: Seconds between instance cleanups
        session_ttl_minutes: Session TTL in minutes
        workflow_ttl_minutes: Workflow TTL in minutes

    Returns:
        BackgroundCleanupTask instance
    """
    global _cleanup_task

    with _cleanup_lock:
        if _cleanup_task is None:
            _cleanup_task = BackgroundCleanupTask(
                session_cleanup_interval=session_cleanup_interval,
                instance_cleanup_interval=instance_cleanup_interval,
                session_ttl_minutes=session_ttl_minutes,
                workflow_ttl_minutes=workflow_ttl_minutes
            )
            _cleanup_task.start()

    return _cleanup_task


def stop_background_cleanup():
    """Stop background cleanup tasks"""
    global _cleanup_task

    with _cleanup_lock:
        if _cleanup_task:
            _cleanup_task.stop()
            _cleanup_task = None


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_orchestrator_logging(level: int = logging.INFO):
    """
    Configure logging for orchestrators

    Args:
        level: Logging level (default INFO)
    """
    # Configure session orchestrator logger
    session_logger = logging.getLogger('backend.agentic.session_orchestrator')
    session_logger.setLevel(level)

    # Configure instance manager logger
    instance_logger = logging.getLogger('backend.agentic.instance_manager')
    instance_logger.setLevel(level)

    # Configure utils logger
    utils_logger = logging.getLogger('backend.agentic.orchestrator_utils')
    utils_logger.setLevel(level)

    logger.info(f"[ORCHESTRATOR] Logging configured at level {level}")


# ============================================================================
# ORCHESTRATOR STATISTICS
# ============================================================================

def get_orchestrator_stats() -> Dict[str, Any]:
    """
    Get combined statistics from all orchestrators

    Returns:
        Dictionary with combined stats
    """
    session_orchestrator = get_session_orchestrator()
    instance_manager = get_instance_manager()

    return {
        "sessions": session_orchestrator.get_session_stats(),
        "instances": instance_manager.get_stats(),
        "cleanup_task_running": _cleanup_task is not None and _cleanup_task._running
    }


def reset_all_orchestrators():
    """Reset all orchestrators (for testing only)"""
    stop_background_cleanup()
    SessionOrchestrator.reset_instance()
    WorkflowInstanceManager.reset_instance()
    logger.warning("[ORCHESTRATOR] All orchestrators reset!")
