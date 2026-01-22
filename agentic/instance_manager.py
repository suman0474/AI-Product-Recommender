"""
WorkflowInstanceManager - Manages concurrent workflow instances

Thread-safe singleton that tracks:
- Multiple concurrent instances per session
- Instance lifecycle (created, running, completed, error)
- Deduplication via trigger_source
- Parent workflow correlation
- Instance statistics and monitoring

Design:
- Uses RLock (reentrant lock) for thread-safe access
- Holds locks <1ms per operation
- Organized by session → pool → instances
- Trigger index enables O(1) deduplication lookup

Usage:
    manager = WorkflowInstanceManager.get_instance()
    instance = manager.create_instance(session_id, thread_id, "product_search", parent_id, "item_1")
    existing = manager.get_instance_by_trigger(session_id, "product_search", parent_id, "item_1")
    manager.set_instance_running(instance.instance_id)
    manager.set_instance_completed(instance.instance_id, result_data)
"""

import threading
import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================

class InstanceStatus(Enum):
    """Status of a workflow instance"""
    CREATED = "created"      # Instance created, not yet started
    RUNNING = "running"      # Instance is executing
    COMPLETED = "completed"  # Instance finished successfully
    ERROR = "error"          # Instance failed (after max retries)
    CANCELLED = "cancelled"  # Instance was cancelled


class InstancePriority(Enum):
    """Priority level for instance execution"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class InstanceMetadata:
    """
    Represents a single workflow instance.

    Contains all tracking information for one execution of a workflow.
    """
    # Core identifiers
    instance_id: str                          # Unique UUID for this instance
    thread_id: str                            # LangGraph thread ID for state isolation
    workflow_type: str                        # e.g., "product_search", "instrument_identifier"
    parent_workflow_id: Optional[str]         # ID of parent workflow (for child instances)
    parent_workflow_type: Optional[str]       # Type of parent workflow
    trigger_source: str                       # What triggered this (e.g., "item_1", "sol_req_2")
    main_thread_id: str                       # Session ID (main thread)

    # Status tracking
    status: InstanceStatus = InstanceStatus.CREATED
    priority: InstancePriority = InstancePriority.NORMAL

    # Counters
    request_count: int = 1                    # How many times this instance was requested
    error_count: int = 0                      # Number of errors encountered
    max_errors: int = 3                       # Max errors before marking as ERROR

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    last_activity: datetime = field(default_factory=datetime.now)

    # Results
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

    # Configuration
    timeout_seconds: int = 3600               # 1 hour timeout

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now()

    def is_timed_out(self) -> bool:
        """Check if instance has timed out"""
        if self.status == InstanceStatus.COMPLETED:
            return False

        elapsed = (datetime.now() - self.created_at).total_seconds()
        return elapsed > self.timeout_seconds

    def is_active(self) -> bool:
        """Check if instance is in an active state"""
        return self.status in [InstanceStatus.CREATED, InstanceStatus.RUNNING]

    def can_retry(self) -> bool:
        """Check if instance can be retried after error"""
        return self.error_count < self.max_errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "instance_id": self.instance_id,
            "thread_id": self.thread_id,
            "workflow_type": self.workflow_type,
            "parent_workflow_id": self.parent_workflow_id,
            "parent_workflow_type": self.parent_workflow_type,
            "trigger_source": self.trigger_source,
            "main_thread_id": self.main_thread_id,
            "status": self.status.value,
            "priority": self.priority.value,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "last_activity": self.last_activity.isoformat(),
            "error_message": self.error_message,
            "has_result": self.result is not None,
            "metadata": self.metadata
        }


@dataclass
class InstancePool:
    """
    Pool of instances grouped by workflow type and parent workflow.

    Key insight: Instances from different parent workflows go into different pools
    to prevent namespace collisions and enable parent-specific management.

    Example:
        Pool 1: product_search from instrument_identifier
            - item_1 -> instance_uuid_1
            - item_2 -> instance_uuid_2

        Pool 2: product_search from solution_workflow
            - sol_req_1 -> instance_uuid_3
            - sol_req_2 -> instance_uuid_4
    """
    workflow_type: str
    parent_workflow_id: str

    # Instance storage
    _instances: Dict[str, InstanceMetadata] = field(default_factory=dict)  # instance_id -> metadata

    # Trigger index for O(1) deduplication lookup
    _trigger_index: Dict[str, str] = field(default_factory=dict)  # trigger_source -> instance_id

    # Thread lock for this pool
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def add_instance(self, instance: InstanceMetadata) -> None:
        """Add an instance to the pool"""
        with self._lock:
            self._instances[instance.instance_id] = instance
            self._trigger_index[instance.trigger_source] = instance.instance_id

    def get_instance(self, instance_id: str) -> Optional[InstanceMetadata]:
        """Get instance by ID"""
        with self._lock:
            return self._instances.get(instance_id)

    def get_instance_by_trigger(self, trigger_source: str) -> Optional[InstanceMetadata]:
        """Get instance by trigger source (O(1) lookup via index)"""
        with self._lock:
            instance_id = self._trigger_index.get(trigger_source)
            if instance_id:
                return self._instances.get(instance_id)
            return None

    def remove_instance(self, instance_id: str) -> Optional[InstanceMetadata]:
        """Remove instance from pool"""
        with self._lock:
            instance = self._instances.pop(instance_id, None)
            if instance:
                # Remove from trigger index
                if instance.trigger_source in self._trigger_index:
                    if self._trigger_index[instance.trigger_source] == instance_id:
                        del self._trigger_index[instance.trigger_source]
            return instance

    def get_active_instances(self) -> List[InstanceMetadata]:
        """Get all active (created or running) instances"""
        with self._lock:
            return [
                inst for inst in self._instances.values()
                if inst.is_active()
            ]

    def get_all_instances(self) -> List[InstanceMetadata]:
        """Get all instances in the pool"""
        with self._lock:
            return list(self._instances.values())

    def get_instance_count(self) -> int:
        """Get total instance count"""
        with self._lock:
            return len(self._instances)

    def cleanup_timed_out(self) -> int:
        """Remove timed-out instances"""
        with self._lock:
            timed_out = [
                iid for iid, inst in self._instances.items()
                if inst.is_timed_out()
            ]

            for iid in timed_out:
                instance = self._instances.pop(iid, None)
                if instance and instance.trigger_source in self._trigger_index:
                    if self._trigger_index[instance.trigger_source] == iid:
                        del self._trigger_index[instance.trigger_source]

            return len(timed_out)


# ============================================================================
# SINGLETON CLASS
# ============================================================================

class WorkflowInstanceManager:
    """
    Singleton that manages all workflow instances across all sessions.

    Thread-safe through RLock. Can be safely accessed from multiple Flask workers.

    Architecture:
        _instance_pools[session_id][pool_key] = InstancePool
        where pool_key = f"{workflow_type}_{parent_workflow_id}"

    Thread Safety:
        - All public methods use self._lock (RLock)
        - Each InstancePool has its own _lock for fine-grained locking
        - Lock hold time < 1ms (only during dict operations)
        - Workflow execution happens OUTSIDE of locks
    """

    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self):
        """Initialize manager (called once via singleton)"""
        # session_id -> { pool_key -> InstancePool }
        self._instance_pools: Dict[str, Dict[str, InstancePool]] = {}

        # Global instance ID -> (session_id, pool_key) for fast lookup
        self._instance_index: Dict[str, tuple] = {}

        # RLock for thread-safe access
        self._lock = threading.RLock()

        # Statistics
        self._total_instances_created = 0
        self._total_instances_completed = 0
        self._total_instances_errored = 0

        logger.info("[INSTANCE_MANAGER] Initialized")

    @classmethod
    def get_instance(cls) -> 'WorkflowInstanceManager':
        """Get singleton instance (thread-safe double-check locking)"""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = WorkflowInstanceManager()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset singleton (for testing only)"""
        with cls._instance_lock:
            cls._instance = None

    # ========================================================================
    # POOL MANAGEMENT
    # ========================================================================

    def _get_pool_key(self, workflow_type: str, parent_workflow_id: str) -> str:
        """Generate pool key from workflow type and parent"""
        return f"{workflow_type}_{parent_workflow_id}"

    def _get_or_create_pool(
        self,
        session_id: str,
        workflow_type: str,
        parent_workflow_id: str
    ) -> InstancePool:
        """Get existing pool or create new one (must be called with lock held)"""
        if session_id not in self._instance_pools:
            self._instance_pools[session_id] = {}

        pool_key = self._get_pool_key(workflow_type, parent_workflow_id)

        if pool_key not in self._instance_pools[session_id]:
            pool = InstancePool(
                workflow_type=workflow_type,
                parent_workflow_id=parent_workflow_id
            )
            self._instance_pools[session_id][pool_key] = pool
            logger.debug(
                f"[INSTANCE_MANAGER] Created pool: {pool_key} for session {session_id}"
            )

        return self._instance_pools[session_id][pool_key]

    # ========================================================================
    # INSTANCE CREATION
    # ========================================================================

    def create_instance(
        self,
        session_id: str,
        thread_id: str,
        workflow_type: str,
        parent_workflow_id: str,
        trigger_source: str,
        parent_workflow_type: Optional[str] = None,
        priority: InstancePriority = InstancePriority.NORMAL,
        timeout_seconds: int = 3600,
        metadata: Dict[str, Any] = None
    ) -> InstanceMetadata:
        """
        Create a new workflow instance

        Args:
            session_id: Main thread ID of the session
            thread_id: LangGraph thread ID for this instance (for state isolation)
            workflow_type: Type of workflow (e.g., "product_search")
            parent_workflow_id: ID of parent workflow
            trigger_source: What triggered this (e.g., "item_1")
            parent_workflow_type: Type of parent workflow (optional)
            priority: Execution priority
            timeout_seconds: Timeout for this instance
            metadata: Additional metadata

        Returns:
            InstanceMetadata: The created instance

        Note:
            Call get_instance_by_trigger() first to check for duplicates!
            This method always creates a NEW instance.
        """
        with self._lock:
            # Get or create pool
            pool = self._get_or_create_pool(session_id, workflow_type, parent_workflow_id)

            # Create instance with unique ID
            instance_id = str(uuid.uuid4())
            now = datetime.now()

            instance = InstanceMetadata(
                instance_id=instance_id,
                thread_id=thread_id,
                workflow_type=workflow_type,
                parent_workflow_id=parent_workflow_id,
                parent_workflow_type=parent_workflow_type,
                trigger_source=trigger_source,
                main_thread_id=session_id,
                priority=priority,
                timeout_seconds=timeout_seconds,
                created_at=now,
                last_activity=now,
                metadata=metadata or {}
            )

            # Add to pool
            pool.add_instance(instance)

            # Add to global index
            pool_key = self._get_pool_key(workflow_type, parent_workflow_id)
            self._instance_index[instance_id] = (session_id, pool_key)

            # Update statistics
            self._total_instances_created += 1

            logger.info(
                f"[INSTANCE_MANAGER] Created instance {instance_id}: "
                f"type={workflow_type}, trigger={trigger_source}, session={session_id}"
            )

            return instance

    # ========================================================================
    # INSTANCE LOOKUP
    # ========================================================================

    def get_instance_by_trigger(
        self,
        session_id: str,
        workflow_type: str,
        parent_workflow_id: str,
        trigger_source: str
    ) -> Optional[InstanceMetadata]:
        """
        Get existing instance by trigger source (O(1) lookup)

        This is the KEY METHOD for deduplication!

        Args:
            session_id: Session ID
            workflow_type: Type of workflow
            parent_workflow_id: Parent workflow ID
            trigger_source: Trigger source (e.g., "item_1")

        Returns:
            InstanceMetadata if found, None if no existing instance

        Use Case:
            existing = manager.get_instance_by_trigger(session_id, "product_search", parent_id, "item_1")
            if existing:
                # RERUN - use existing instance
                existing.request_count += 1
            else:
                # NEW - create new instance
                instance = manager.create_instance(...)
        """
        with self._lock:
            if session_id not in self._instance_pools:
                return None

            pool_key = self._get_pool_key(workflow_type, parent_workflow_id)
            if pool_key not in self._instance_pools[session_id]:
                return None

            pool = self._instance_pools[session_id][pool_key]
            return pool.get_instance_by_trigger(trigger_source)

    def get_instance_by_id(self, instance_id: str) -> Optional[InstanceMetadata]:
        """
        Get instance by its unique ID

        Args:
            instance_id: The instance's UUID

        Returns:
            InstanceMetadata if found, None otherwise
        """
        with self._lock:
            if instance_id not in self._instance_index:
                return None

            session_id, pool_key = self._instance_index[instance_id]
            if session_id not in self._instance_pools:
                return None

            if pool_key not in self._instance_pools[session_id]:
                return None

            pool = self._instance_pools[session_id][pool_key]
            return pool.get_instance(instance_id)

    def get_instance_by_thread_id(self, thread_id: str) -> Optional[InstanceMetadata]:
        """
        Get instance by its LangGraph thread ID

        Searches across all sessions and pools.

        Args:
            thread_id: The LangGraph thread ID

        Returns:
            InstanceMetadata if found, None otherwise

        Note: This is O(n) - use get_instance_by_id() or get_instance_by_trigger()
        when possible for O(1) lookup.
        """
        with self._lock:
            for session_pools in self._instance_pools.values():
                for pool in session_pools.values():
                    for instance in pool.get_all_instances():
                        if instance.thread_id == thread_id:
                            return instance
            return None

    # ========================================================================
    # INSTANCE STATUS UPDATES
    # ========================================================================

    def set_instance_running(self, instance_id: str) -> bool:
        """
        Mark instance as running

        Args:
            instance_id: The instance's UUID

        Returns:
            True if successful, False if instance not found
        """
        with self._lock:
            instance = self.get_instance_by_id(instance_id)
            if instance:
                instance.status = InstanceStatus.RUNNING
                instance.started_at = datetime.now()
                instance.update_activity()
                logger.debug(f"[INSTANCE_MANAGER] Instance {instance_id} marked RUNNING")
                return True
            return False

    def set_instance_completed(
        self,
        instance_id: str,
        result: Dict[str, Any] = None
    ) -> bool:
        """
        Mark instance as completed

        Args:
            instance_id: The instance's UUID
            result: Result data from workflow execution

        Returns:
            True if successful, False if instance not found
        """
        with self._lock:
            instance = self.get_instance_by_id(instance_id)
            if instance:
                instance.status = InstanceStatus.COMPLETED
                instance.completed_at = datetime.now()
                instance.result = result
                instance.update_activity()
                self._total_instances_completed += 1
                logger.debug(f"[INSTANCE_MANAGER] Instance {instance_id} marked COMPLETED")
                return True
            return False

    def set_instance_error(
        self,
        instance_id: str,
        error_message: str
    ) -> bool:
        """
        Mark instance error and increment error count

        Args:
            instance_id: The instance's UUID
            error_message: Error message

        Returns:
            True if successful, False if instance not found

        Note:
            Instance transitions to ERROR status only after max_errors reached.
            Until then, it can be retried.
        """
        with self._lock:
            instance = self.get_instance_by_id(instance_id)
            if instance:
                instance.error_count += 1
                instance.error_message = error_message
                instance.update_activity()

                if not instance.can_retry():
                    instance.status = InstanceStatus.ERROR
                    self._total_instances_errored += 1
                    logger.warning(
                        f"[INSTANCE_MANAGER] Instance {instance_id} marked ERROR "
                        f"(error_count={instance.error_count}): {error_message}"
                    )
                else:
                    logger.warning(
                        f"[INSTANCE_MANAGER] Instance {instance_id} error "
                        f"(error_count={instance.error_count}, can retry): {error_message}"
                    )

                return True
            return False

    def set_instance_cancelled(self, instance_id: str) -> bool:
        """
        Mark instance as cancelled

        Args:
            instance_id: The instance's UUID

        Returns:
            True if successful, False if instance not found
        """
        with self._lock:
            instance = self.get_instance_by_id(instance_id)
            if instance:
                instance.status = InstanceStatus.CANCELLED
                instance.update_activity()
                logger.debug(f"[INSTANCE_MANAGER] Instance {instance_id} marked CANCELLED")
                return True
            return False

    def increment_request_count(self, instance_id: str) -> int:
        """
        Increment request count for an instance (on rerun)

        Args:
            instance_id: The instance's UUID

        Returns:
            New request count, or -1 if instance not found
        """
        with self._lock:
            instance = self.get_instance_by_id(instance_id)
            if instance:
                instance.request_count += 1
                instance.update_activity()
                return instance.request_count
            return -1

    # ========================================================================
    # INSTANCE QUERIES
    # ========================================================================

    def get_instances_for_session(
        self,
        session_id: str,
        workflow_type: Optional[str] = None,
        status: Optional[InstanceStatus] = None
    ) -> List[InstanceMetadata]:
        """
        Get all instances for a session, optionally filtered

        Args:
            session_id: Session ID
            workflow_type: Optional filter by workflow type
            status: Optional filter by status

        Returns:
            List of InstanceMetadata objects
        """
        with self._lock:
            if session_id not in self._instance_pools:
                return []

            instances = []
            for pool in self._instance_pools[session_id].values():
                instances.extend(pool.get_all_instances())

            # Apply filters
            if workflow_type:
                instances = [i for i in instances if i.workflow_type == workflow_type]

            if status:
                instances = [i for i in instances if i.status == status]

            return instances

    def get_instances_for_pool(
        self,
        session_id: str,
        workflow_type: str,
        parent_workflow_id: str
    ) -> List[InstanceMetadata]:
        """
        Get all instances for a specific pool

        Args:
            session_id: Session ID
            workflow_type: Workflow type
            parent_workflow_id: Parent workflow ID

        Returns:
            List of InstanceMetadata objects
        """
        with self._lock:
            if session_id not in self._instance_pools:
                return []

            pool_key = self._get_pool_key(workflow_type, parent_workflow_id)
            if pool_key not in self._instance_pools[session_id]:
                return []

            pool = self._instance_pools[session_id][pool_key]
            return pool.get_all_instances()

    def get_active_instances_for_session(self, session_id: str) -> List[InstanceMetadata]:
        """
        Get all active (created or running) instances for a session

        Args:
            session_id: Session ID

        Returns:
            List of active InstanceMetadata objects
        """
        with self._lock:
            if session_id not in self._instance_pools:
                return []

            active = []
            for pool in self._instance_pools[session_id].values():
                active.extend(pool.get_active_instances())

            return active

    # ========================================================================
    # STATISTICS & MONITORING
    # ========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """
        Get instance manager statistics

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            total_instances = 0
            active_instances = 0
            completed_instances = 0
            error_instances = 0
            pools_count = 0

            for session_pools in self._instance_pools.values():
                for pool in session_pools.values():
                    pools_count += 1
                    for instance in pool.get_all_instances():
                        total_instances += 1
                        if instance.status == InstanceStatus.COMPLETED:
                            completed_instances += 1
                        elif instance.status == InstanceStatus.ERROR:
                            error_instances += 1
                        elif instance.is_active():
                            active_instances += 1

            return {
                "total_sessions_with_instances": len(self._instance_pools),
                "total_pools": pools_count,
                "total_instances": total_instances,
                "active_instances": active_instances,
                "completed_instances": completed_instances,
                "error_instances": error_instances,
                "lifetime_created": self._total_instances_created,
                "lifetime_completed": self._total_instances_completed,
                "lifetime_errored": self._total_instances_errored
            }

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get summary of instances for a session

        Args:
            session_id: Session ID

        Returns:
            Dictionary with session instance summary
        """
        with self._lock:
            if session_id not in self._instance_pools:
                return {
                    "session_id": session_id,
                    "pools": {},
                    "total_instances": 0
                }

            pools_summary = {}
            total_instances = 0

            for pool_key, pool in self._instance_pools[session_id].items():
                instances = pool.get_all_instances()
                total_instances += len(instances)

                pools_summary[pool_key] = {
                    "workflow_type": pool.workflow_type,
                    "parent_workflow_id": pool.parent_workflow_id,
                    "instance_count": len(instances),
                    "active_count": len(pool.get_active_instances()),
                    "instances": [i.to_dict() for i in instances]
                }

            return {
                "session_id": session_id,
                "pools": pools_summary,
                "total_instances": total_instances
            }

    # ========================================================================
    # CLEANUP
    # ========================================================================

    def cleanup_session(self, session_id: str) -> int:
        """
        Remove all instances for a session

        Args:
            session_id: Session ID to clean up

        Returns:
            Number of instances removed
        """
        with self._lock:
            if session_id not in self._instance_pools:
                return 0

            total_removed = 0

            # Get all instance IDs for this session
            for pool in self._instance_pools[session_id].values():
                for instance in pool.get_all_instances():
                    if instance.instance_id in self._instance_index:
                        del self._instance_index[instance.instance_id]
                    total_removed += 1

            # Remove all pools for this session
            del self._instance_pools[session_id]

            if total_removed:
                logger.info(
                    f"[INSTANCE_MANAGER] Cleaned up {total_removed} instances "
                    f"for session {session_id}"
                )

            return total_removed

    def cleanup_all_timed_out(self) -> int:
        """
        Remove all timed-out instances across all sessions

        Returns:
            Total number of instances removed
        """
        with self._lock:
            total_removed = 0

            for session_id, session_pools in list(self._instance_pools.items()):
                for pool_key, pool in list(session_pools.items()):
                    # Get IDs before cleanup
                    instances_before = set(i.instance_id for i in pool.get_all_instances())

                    # Clean up
                    removed = pool.cleanup_timed_out()
                    total_removed += removed

                    # Update global index
                    instances_after = set(i.instance_id for i in pool.get_all_instances())
                    removed_ids = instances_before - instances_after
                    for iid in removed_ids:
                        if iid in self._instance_index:
                            del self._instance_index[iid]

                    # Remove empty pools
                    if pool.get_instance_count() == 0:
                        del self._instance_pools[session_id][pool_key]

                # Remove empty session entries
                if not self._instance_pools[session_id]:
                    del self._instance_pools[session_id]

            if total_removed:
                logger.info(
                    f"[INSTANCE_MANAGER] Cleaned up {total_removed} timed-out instances"
                )

            return total_removed

    # ========================================================================
    # DEBUG / ADMIN METHODS
    # ========================================================================

    def clear_all(self):
        """Clear all instances (for testing only)"""
        with self._lock:
            self._instance_pools.clear()
            self._instance_index.clear()
            self._total_instances_created = 0
            self._total_instances_completed = 0
            self._total_instances_errored = 0
            logger.warning("[INSTANCE_MANAGER] All instances cleared!")

    def debug_dump(self) -> Dict[str, Any]:
        """Dump all internal state for debugging"""
        with self._lock:
            return {
                "sessions_count": len(self._instance_pools),
                "total_pools": sum(
                    len(pools) for pools in self._instance_pools.values()
                ),
                "index_size": len(self._instance_index),
                "stats": self.get_stats(),
                "sessions": {
                    session_id: {
                        pool_key: {
                            "instance_count": pool.get_instance_count(),
                            "trigger_index_size": len(pool._trigger_index)
                        }
                        for pool_key, pool in pools.items()
                    }
                    for session_id, pools in self._instance_pools.items()
                }
            }


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def get_instance_manager() -> WorkflowInstanceManager:
    """Get the WorkflowInstanceManager singleton instance"""
    return WorkflowInstanceManager.get_instance()
