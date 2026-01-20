# agentic/checkpointing.py
# Workflow Checkpointing and State Persistence

import logging
import json
import time
from typing import Optional, Literal, Dict, Any, List
from datetime import datetime, timedelta
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)


# ============================================================================
# CHECKPOINT METADATA
# ============================================================================

class CheckpointMetadata:
    """Metadata for checkpoint tracking and management"""

    def __init__(
        self,
        thread_id: str,
        user_id: Optional[str] = None,
        workflow_type: Optional[str] = None,
        created_at: Optional[datetime] = None
    ):
        self.thread_id = thread_id
        self.user_id = user_id or "anonymous"
        self.workflow_type = workflow_type or "unknown"
        self.created_at = created_at or datetime.now()
        self.last_accessed = datetime.now()
        self.access_count = 1
        self.size_bytes = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "thread_id": self.thread_id,
            "user_id": self.user_id,
            "workflow_type": self.workflow_type,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "size_bytes": self.size_bytes
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointMetadata':
        """Create from dictionary"""
        metadata = cls(
            thread_id=data["thread_id"],
            user_id=data.get("user_id", "anonymous"),
            workflow_type=data.get("workflow_type", "unknown"),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else None
        )
        metadata.last_accessed = datetime.fromisoformat(data["last_accessed"]) if "last_accessed" in data else datetime.now()
        metadata.access_count = data.get("access_count", 1)
        metadata.size_bytes = data.get("size_bytes", 0)
        return metadata


# ============================================================================
# CHECKPOINTER BACKENDS
# ============================================================================

def get_memory_checkpointer():
    """Get in-memory checkpointer (for development)"""
    logger.info("Using MemorySaver checkpointer")
    return MemorySaver()


def get_sqlite_checkpointer(db_path: str = "checkpoints.db"):
    """Get SQLite-based checkpointer (for local persistence)"""
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        logger.info(f"Using SQLite checkpointer: {db_path}")
        return SqliteSaver.from_conn_string(f"sqlite:///{db_path}")
    except ImportError:
        logger.warning("SQLite checkpointer not available, falling back to memory")
        return get_memory_checkpointer()
    except Exception as e:
        logger.error(f"Failed to create SQLite checkpointer: {e}")
        return get_memory_checkpointer()


def get_mongodb_checkpointer(
    connection_string: Optional[str] = None,
    database: str = "workflow_checkpoints",
    collection: str = "checkpoints"
):
    """Get MongoDB-based checkpointer (for production)"""
    try:
        # Note: This requires langgraph-checkpoint-mongodb package
        # pip install langgraph-checkpoint-mongodb
        from langgraph.checkpoint.mongodb import MongoDBSaver
        import os

        conn_str = connection_string or os.getenv("MONGODB_URI")
        if not conn_str:
            logger.warning("MongoDB URI not found, falling back to memory")
            return get_memory_checkpointer()

        logger.info(f"Using MongoDB checkpointer: {database}.{collection}")
        return MongoDBSaver.from_conn_string(
            conn_str,
            db_name=database,
            collection_name=collection
        )
    except ImportError:
        logger.warning("MongoDB checkpointer not available, falling back to memory")
        return get_memory_checkpointer()
    except Exception as e:
        logger.error(f"Failed to create MongoDB checkpointer: {e}")
        return get_memory_checkpointer()


def get_azure_blob_checkpointer(
    connection_string: Optional[str] = None,
    account_url: Optional[str] = None,
    container_prefix: str = "workflow-checkpoints",
    default_zone: str = "DEFAULT",
    ttl_hours: int = 72,
    use_managed_identity: bool = False,
):
    """
    Get Azure Blob Storage-based checkpointer (for production with zone partitioning).

    This checkpointer supports the hierarchical thread-ID system with:
    - Zone-based container organization
    - Thread tree state persistence
    - TTL-based automatic cleanup
    - Item-level state persistence

    Args:
        connection_string: Azure Storage connection string
        account_url: Azure Storage account URL (for managed identity)
        container_prefix: Prefix for checkpoint containers
        default_zone: Default zone if not specified
        ttl_hours: TTL for checkpoint cleanup
        use_managed_identity: Use Azure managed identity authentication

    Returns:
        AzureBlobCheckpointer instance or fallback to memory
    """
    try:
        from .azure_checkpointing import (
            AzureBlobCheckpointer,
            AzureBlobCheckpointerConfig,
        )
        import os

        # Build config
        config = AzureBlobCheckpointerConfig(
            connection_string=connection_string or os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
            account_url=account_url or os.getenv("AZURE_STORAGE_ACCOUNT_URL"),
            container_prefix=container_prefix,
            default_zone=default_zone,
            ttl_hours=ttl_hours,
            use_managed_identity=use_managed_identity,
        )

        # Check if we have credentials
        if not config.connection_string and not (config.account_url and config.use_managed_identity):
            logger.warning("Azure Blob Storage credentials not found, falling back to memory")
            return get_memory_checkpointer()

        checkpointer = AzureBlobCheckpointer(config)
        logger.info(f"Using Azure Blob Storage checkpointer: {container_prefix}-{default_zone.lower()}")
        return checkpointer

    except ImportError as e:
        logger.warning(f"Azure Blob Storage checkpointer not available: {e}, falling back to memory")
        return get_memory_checkpointer()
    except Exception as e:
        logger.error(f"Failed to create Azure Blob Storage checkpointer: {e}")
        return get_memory_checkpointer()


# ============================================================================
# CHECKPOINTER FACTORY
# ============================================================================

def get_checkpointer(
    backend: Literal["memory", "sqlite", "mongodb", "azure_blob"] = "memory",
    **kwargs
):
    """
    Factory function to get the appropriate checkpointer.

    Args:
        backend: The checkpointing backend to use
            - "memory": In-memory (development only, not persistent)
            - "sqlite": SQLite database (local persistence)
            - "mongodb": MongoDB (production, distributed)
            - "azure_blob": Azure Blob Storage (production, zone-partitioned)
        **kwargs: Backend-specific arguments
            - sqlite: db_path
            - mongodb: connection_string, database, collection
            - azure_blob: connection_string, account_url, container_prefix,
                          default_zone, ttl_hours, use_managed_identity

    Returns:
        A checkpointer instance
    """
    if backend == "memory":
        return get_memory_checkpointer()
    elif backend == "sqlite":
        return get_sqlite_checkpointer(
            db_path=kwargs.get("db_path", "checkpoints.db")
        )
    elif backend == "mongodb":
        return get_mongodb_checkpointer(
            connection_string=kwargs.get("connection_string"),
            database=kwargs.get("database", "workflow_checkpoints"),
            collection=kwargs.get("collection", "checkpoints")
        )
    elif backend == "azure_blob":
        return get_azure_blob_checkpointer(
            connection_string=kwargs.get("connection_string"),
            account_url=kwargs.get("account_url"),
            container_prefix=kwargs.get("container_prefix", "workflow-checkpoints"),
            default_zone=kwargs.get("default_zone", "DEFAULT"),
            ttl_hours=kwargs.get("ttl_hours", 72),
            use_managed_identity=kwargs.get("use_managed_identity", False),
        )
    else:
        logger.warning(f"Unknown backend '{backend}', falling back to memory")
        return get_memory_checkpointer()


# ============================================================================
# WORKFLOW COMPILATION WITH CHECKPOINTING
# ============================================================================

def compile_with_checkpointing(
    workflow: StateGraph,
    backend: Literal["memory", "sqlite", "mongodb", "azure_blob"] = "memory",
    **kwargs
):
    """
    Compile a workflow with checkpointing enabled.

    Args:
        workflow: The StateGraph workflow to compile
        backend: The checkpointing backend to use
            - "memory": In-memory (development only)
            - "sqlite": SQLite database (local persistence)
            - "mongodb": MongoDB (production, distributed)
            - "azure_blob": Azure Blob Storage (production, zone-partitioned)
        **kwargs: Backend-specific arguments

    Returns:
        Compiled workflow with checkpointing
    """
    checkpointer = get_checkpointer(backend, **kwargs)
    compiled = workflow.compile(checkpointer=checkpointer)
    logger.info(f"Workflow compiled with {backend} checkpointing")
    return compiled


# ============================================================================
# CHECKPOINT CLEANUP AND MANAGEMENT
# ============================================================================

class CheckpointManager:
    """
    Manages checkpoint lifecycle including cleanup of old checkpoints.
    """

    def __init__(
        self,
        backend: Literal["memory", "sqlite", "mongodb"] = "memory",
        max_age_hours: int = 72,
        max_checkpoints_per_user: int = 100,
        **backend_kwargs
    ):
        """
        Initialize checkpoint manager.

        Args:
            backend: Checkpointing backend to use
            max_age_hours: Maximum age of checkpoints before cleanup (default 72 hours)
            max_checkpoints_per_user: Maximum checkpoints per user (default 100)
            **backend_kwargs: Backend-specific arguments
        """
        self.backend = backend
        self.max_age_hours = max_age_hours
        self.max_checkpoints_per_user = max_checkpoints_per_user
        self.backend_kwargs = backend_kwargs
        self.checkpointer = get_checkpointer(backend, **backend_kwargs)
        self.metadata_store: Dict[str, CheckpointMetadata] = {}

    def cleanup_old_checkpoints(self) -> Dict[str, Any]:
        """
        Clean up checkpoints older than max_age_hours.

        Returns:
            Cleanup summary with counts
        """
        try:
            cutoff = datetime.now() - timedelta(hours=self.max_age_hours)
            removed_count = 0
            kept_count = 0

            # For memory and sqlite backends, we need to track metadata separately
            if self.backend in ["memory", "sqlite"]:
                expired_threads = [
                    thread_id for thread_id, metadata in self.metadata_store.items()
                    if metadata.created_at < cutoff
                ]

                for thread_id in expired_threads:
                    del self.metadata_store[thread_id]
                    removed_count += 1

                kept_count = len(self.metadata_store)

                logger.info(
                    f"Checkpoint cleanup: removed {removed_count}, "
                    f"kept {kept_count} (cutoff: {cutoff.isoformat()})"
                )

            # For MongoDB, use native TTL indexes
            elif self.backend == "mongodb":
                # MongoDB cleanup is handled by TTL indexes on created_at field
                logger.info("MongoDB checkpoint cleanup handled by TTL indexes")
                removed_count = 0
                kept_count = len(self.metadata_store)

            return {
                "success": True,
                "removed_count": removed_count,
                "kept_count": kept_count,
                "cutoff_time": cutoff.isoformat()
            }

        except Exception as e:
            logger.error(f"Checkpoint cleanup failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }

    def cleanup_user_checkpoints(self, user_id: str) -> Dict[str, Any]:
        """
        Clean up excess checkpoints for a specific user.
        Keeps only the most recent max_checkpoints_per_user checkpoints.

        Args:
            user_id: User identifier

        Returns:
            Cleanup summary
        """
        try:
            # Get all checkpoints for user
            user_checkpoints = [
                (thread_id, metadata)
                for thread_id, metadata in self.metadata_store.items()
                if metadata.user_id == user_id
            ]

            # Sort by last_accessed (most recent first)
            user_checkpoints.sort(
                key=lambda x: x[1].last_accessed,
                reverse=True
            )

            # Remove excess
            removed_count = 0
            if len(user_checkpoints) > self.max_checkpoints_per_user:
                excess = user_checkpoints[self.max_checkpoints_per_user:]
                for thread_id, _ in excess:
                    del self.metadata_store[thread_id]
                    removed_count += 1

            logger.info(
                f"User {user_id} checkpoint cleanup: "
                f"removed {removed_count}, kept {len(user_checkpoints) - removed_count}"
            )

            return {
                "success": True,
                "user_id": user_id,
                "removed_count": removed_count,
                "kept_count": len(user_checkpoints) - removed_count
            }

        except Exception as e:
            logger.error(f"User checkpoint cleanup failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }

    def register_checkpoint(
        self,
        thread_id: str,
        user_id: Optional[str] = None,
        workflow_type: Optional[str] = None,
        state_size: int = 0
    ) -> None:
        """
        Register a checkpoint with metadata.

        Args:
            thread_id: Thread identifier
            user_id: Optional user identifier
            workflow_type: Optional workflow type
            state_size: Size of the state in bytes
        """
        if thread_id in self.metadata_store:
            # Update existing
            metadata = self.metadata_store[thread_id]
            metadata.last_accessed = datetime.now()
            metadata.access_count += 1
            if state_size > 0:
                metadata.size_bytes = state_size
        else:
            # Create new
            metadata = CheckpointMetadata(
                thread_id=thread_id,
                user_id=user_id,
                workflow_type=workflow_type
            )
            metadata.size_bytes = state_size
            self.metadata_store[thread_id] = metadata

    def get_checkpoint_metadata(self, thread_id: str) -> Optional[CheckpointMetadata]:
        """Get metadata for a specific checkpoint"""
        return self.metadata_store.get(thread_id)

    def get_user_checkpoints(self, user_id: str) -> List[CheckpointMetadata]:
        """Get all checkpoints for a specific user"""
        return [
            metadata for metadata in self.metadata_store.values()
            if metadata.user_id == user_id
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """Get checkpoint statistics"""
        total_checkpoints = len(self.metadata_store)
        total_size = sum(m.size_bytes for m in self.metadata_store.values())
        users = set(m.user_id for m in self.metadata_store.values())

        workflow_types = {}
        for metadata in self.metadata_store.values():
            wf_type = metadata.workflow_type
            workflow_types[wf_type] = workflow_types.get(wf_type, 0) + 1

        return {
            "total_checkpoints": total_checkpoints,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "unique_users": len(users),
            "workflow_types": workflow_types,
            "backend": self.backend
        }


# ============================================================================
# WORKFLOW EXECUTION WITH STATE MANAGEMENT
# ============================================================================

class WorkflowExecutor:
    """
    Executes workflows with state management and checkpointing support.
    Enables workflow resume, multi-turn conversations, state inspection, and recovery.
    """

    def __init__(
        self,
        workflow: StateGraph,
        backend: Literal["memory", "sqlite", "mongodb"] = "memory",
        enable_recovery: bool = True,
        max_retry_attempts: int = 3,
        **backend_kwargs
    ):
        """
        Initialize workflow executor.

        Args:
            workflow: StateGraph workflow to execute
            backend: Checkpointing backend
            enable_recovery: Enable automatic recovery from failures
            max_retry_attempts: Maximum retry attempts for failed workflows
            **backend_kwargs: Backend-specific arguments
        """
        self.workflow = workflow
        self.backend = backend
        self.backend_kwargs = backend_kwargs
        self.enable_recovery = enable_recovery
        self.max_retry_attempts = max_retry_attempts
        self.compiled = compile_with_checkpointing(workflow, backend, **backend_kwargs)
        self.checkpoint_manager = CheckpointManager(backend, **backend_kwargs)
        logger.info(f"WorkflowExecutor initialized (backend: {backend}, recovery: {enable_recovery})")
    
    def invoke(
        self,
        initial_state: dict,
        thread_id: str,
        user_id: Optional[str] = None,
        workflow_type: Optional[str] = None,
        **invoke_kwargs
    ) -> dict:
        """
        Execute the workflow with the given initial state.
        Includes automatic checkpointing and optional recovery.

        Args:
            initial_state: The initial workflow state
            thread_id: Unique identifier for this conversation/session
            user_id: Optional user identifier for checkpoint tracking
            workflow_type: Optional workflow type for statistics
            **invoke_kwargs: Additional arguments for invoke

        Returns:
            Final workflow state
        """
        config = {"configurable": {"thread_id": thread_id}}
        config.update(invoke_kwargs)

        logger.info(f"Invoking workflow for thread: {thread_id}")

        # Calculate state size for tracking
        state_size = len(json.dumps(initial_state).encode('utf-8'))

        # Register checkpoint
        self.checkpoint_manager.register_checkpoint(
            thread_id=thread_id,
            user_id=user_id,
            workflow_type=workflow_type,
            state_size=state_size
        )

        # Execute with optional recovery
        attempt = 0
        last_error = None

        while attempt < (self.max_retry_attempts if self.enable_recovery else 1):
            try:
                final_state = self.compiled.invoke(initial_state, config)
                logger.info(f"Workflow completed for thread: {thread_id}")

                # Update checkpoint metadata
                final_state_size = len(json.dumps(final_state).encode('utf-8'))
                self.checkpoint_manager.register_checkpoint(
                    thread_id=thread_id,
                    user_id=user_id,
                    workflow_type=workflow_type,
                    state_size=final_state_size
                )

                return final_state

            except Exception as e:
                attempt += 1
                last_error = e
                logger.error(
                    f"Workflow execution failed (attempt {attempt}/{self.max_retry_attempts}): {e}",
                    exc_info=True
                )

                if attempt < self.max_retry_attempts and self.enable_recovery:
                    logger.info(f"Attempting recovery for thread: {thread_id}")
                    time.sleep(1 * attempt)  # Exponential backoff
                else:
                    break

        # All attempts failed
        logger.error(f"Workflow execution failed after {attempt} attempts for thread: {thread_id}")
        raise RuntimeError(
            f"Workflow execution failed after {attempt} attempts. Last error: {last_error}"
        ) from last_error
    
    def resume(
        self,
        thread_id: str,
        additional_input: dict = None,
        user_id: Optional[str] = None,
        **invoke_kwargs
    ) -> Optional[dict]:
        """
        Resume a previously checkpointed workflow with validation.

        Args:
            thread_id: The thread ID to resume
            additional_input: Additional input to merge with saved state
            user_id: Optional user identifier for validation
            **invoke_kwargs: Additional arguments for invoke

        Returns:
            Final workflow state after resume, or None if no checkpoint found
        """
        config = {"configurable": {"thread_id": thread_id}}
        config.update(invoke_kwargs)

        logger.info(f"Resuming workflow for thread: {thread_id}")

        # Validate checkpoint exists
        if not self.validate_checkpoint(thread_id):
            logger.warning(f"Invalid or missing checkpoint for thread: {thread_id}")
            return None

        # Update checkpoint access metadata
        self.checkpoint_manager.register_checkpoint(
            thread_id=thread_id,
            user_id=user_id
        )

        # Get current state
        current_state = self.get_state(thread_id)

        if current_state and additional_input:
            # Merge additional input
            for key, value in additional_input.items():
                current_state[key] = value

        if current_state:
            try:
                final_state = self.compiled.invoke(current_state, config)
                logger.info(f"Workflow resumed successfully for thread: {thread_id}")
                return final_state
            except Exception as e:
                logger.error(f"Failed to resume workflow for thread {thread_id}: {e}", exc_info=True)
                raise
        else:
            logger.warning(f"No saved state found for thread: {thread_id}")
            return None

    def validate_checkpoint(self, thread_id: str) -> bool:
        """
        Validate that a checkpoint exists and is valid.

        Args:
            thread_id: Thread identifier to validate

        Returns:
            True if checkpoint is valid, False otherwise
        """
        try:
            state = self.get_state(thread_id)
            if not state:
                return False

            # Check if state has required fields
            if not isinstance(state, dict):
                logger.warning(f"Invalid checkpoint state type for thread {thread_id}")
                return False

            # Validate state is not empty
            if len(state) == 0:
                logger.warning(f"Empty checkpoint state for thread {thread_id}")
                return False

            # Check metadata
            metadata = self.checkpoint_manager.get_checkpoint_metadata(thread_id)
            if metadata:
                # Check if checkpoint is too old
                age_hours = (datetime.now() - metadata.created_at).total_seconds() / 3600
                if age_hours > self.checkpoint_manager.max_age_hours:
                    logger.warning(
                        f"Checkpoint for thread {thread_id} is too old ({age_hours:.1f} hours)"
                    )
                    return False

            return True

        except Exception as e:
            logger.error(f"Checkpoint validation failed for thread {thread_id}: {e}")
            return False
    
    def get_state(self, thread_id: str) -> Optional[dict]:
        """
        Get the current state for a thread.
        
        Args:
            thread_id: The thread ID to get state for
        
        Returns:
            Current workflow state or None
        """
        try:
            config = {"configurable": {"thread_id": thread_id}}
            state = self.compiled.get_state(config)
            if state and state.values:
                return dict(state.values)
            return None
        except Exception as e:
            logger.error(f"Failed to get state for thread {thread_id}: {e}")
            return None
    
    def get_state_history(self, thread_id: str, limit: int = 10) -> list:
        """
        Get state history for a thread.
        
        Args:
            thread_id: The thread ID to get history for
            limit: Maximum number of states to return
        
        Returns:
            List of historical states
        """
        try:
            config = {"configurable": {"thread_id": thread_id}}
            history = list(self.compiled.get_state_history(config, limit=limit))
            return [dict(s.values) for s in history if s.values]
        except Exception as e:
            logger.error(f"Failed to get state history for thread {thread_id}: {e}")
            return []
    
    def stream(
        self,
        initial_state: dict,
        thread_id: str,
        **stream_kwargs
    ):
        """
        Stream workflow execution, yielding intermediate states.
        
        Args:
            initial_state: The initial workflow state
            thread_id: Unique identifier for this conversation/session
            **stream_kwargs: Additional arguments for stream
        
        Yields:
            Intermediate workflow states
        """
        config = {"configurable": {"thread_id": thread_id}}
        config.update(stream_kwargs)
        
        logger.info(f"Streaming workflow for thread: {thread_id}")
        for state in self.compiled.stream(initial_state, config):
            yield state
