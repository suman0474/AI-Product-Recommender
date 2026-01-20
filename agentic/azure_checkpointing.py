# agentic/azure_checkpointing.py
# Azure Blob Storage Checkpointer for Hierarchical Thread State Persistence
#
# Provides zone-based container organization and TTL-based cleanup for
# the tree-based thread-ID workflow system.

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Iterator, Tuple
from dataclasses import dataclass, field

from azure.storage.blob import BlobServiceClient, ContainerClient, BlobClient
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ResourceNotFoundError, ResourceExistsError

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class AzureBlobCheckpointerConfig:
    """Configuration for Azure Blob Storage Checkpointer."""

    connection_string: Optional[str] = None
    account_url: Optional[str] = None
    container_prefix: str = "workflow-checkpoints"
    default_zone: str = "DEFAULT"
    ttl_hours: int = 72  # Default TTL for checkpoints
    max_checkpoints_per_thread: int = 100
    use_managed_identity: bool = False

    @classmethod
    def from_env(cls) -> "AzureBlobCheckpointerConfig":
        """Create config from environment variables."""
        return cls(
            connection_string=os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
            account_url=os.getenv("AZURE_STORAGE_ACCOUNT_URL"),
            container_prefix=os.getenv("AZURE_CHECKPOINT_CONTAINER_PREFIX", "workflow-checkpoints"),
            default_zone=os.getenv("AZURE_CHECKPOINT_DEFAULT_ZONE", "DEFAULT"),
            ttl_hours=int(os.getenv("AZURE_CHECKPOINT_TTL_HOURS", "72")),
            max_checkpoints_per_thread=int(os.getenv("AZURE_CHECKPOINT_MAX_PER_THREAD", "100")),
            use_managed_identity=os.getenv("AZURE_USE_MANAGED_IDENTITY", "false").lower() == "true",
        )


# ============================================================================
# AZURE BLOB CHECKPOINTER
# ============================================================================

class AzureBlobCheckpointer(BaseCheckpointSaver):
    """
    Azure Blob Storage-based checkpointer for LangGraph workflows.

    Features:
    - Zone-based container organization (e.g., workflow-checkpoints-us-west)
    - Hierarchical thread state persistence
    - TTL-based automatic cleanup
    - Support for thread tree structures

    Blob Structure:
        container: workflow-checkpoints-{zone}
        blob path: {thread_id}/{checkpoint_id}.json

    Metadata stored in blob metadata:
        - thread_id
        - checkpoint_id
        - parent_thread_id
        - zone
        - created_at
        - user_id
        - workflow_type
    """

    def __init__(self, config: Optional[AzureBlobCheckpointerConfig] = None):
        """
        Initialize Azure Blob Storage checkpointer.

        Args:
            config: Configuration options. If None, loads from environment.
        """
        super().__init__()

        self.config = config or AzureBlobCheckpointerConfig.from_env()
        self._blob_service_client: Optional[BlobServiceClient] = None
        self._container_clients: Dict[str, ContainerClient] = {}

        logger.info(f"[AZURE_CHECKPOINT] Initializing with container prefix: {self.config.container_prefix}")

    @property
    def blob_service_client(self) -> BlobServiceClient:
        """Lazy initialization of blob service client."""
        if self._blob_service_client is None:
            if self.config.use_managed_identity:
                credential = DefaultAzureCredential()
                self._blob_service_client = BlobServiceClient(
                    account_url=self.config.account_url,
                    credential=credential
                )
                logger.info("[AZURE_CHECKPOINT] Using managed identity authentication")
            elif self.config.connection_string:
                self._blob_service_client = BlobServiceClient.from_connection_string(
                    self.config.connection_string
                )
                logger.info("[AZURE_CHECKPOINT] Using connection string authentication")
            else:
                raise ValueError(
                    "Azure Blob Storage credentials not configured. "
                    "Set AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_ACCOUNT_URL with managed identity."
                )
        return self._blob_service_client

    def _get_container_name(self, zone: str) -> str:
        """Get container name for a zone."""
        zone_normalized = zone.lower().replace("_", "-").replace(" ", "-")
        return f"{self.config.container_prefix}-{zone_normalized}"

    def _get_container_client(self, zone: str) -> ContainerClient:
        """Get or create container client for a zone."""
        container_name = self._get_container_name(zone)

        if container_name not in self._container_clients:
            container_client = self.blob_service_client.get_container_client(container_name)

            # Create container if it doesn't exist
            try:
                container_client.create_container()
                logger.info(f"[AZURE_CHECKPOINT] Created container: {container_name}")
            except ResourceExistsError:
                pass  # Container already exists

            self._container_clients[container_name] = container_client

        return self._container_clients[container_name]

    def _get_blob_path(self, thread_id: str, checkpoint_id: str) -> str:
        """Get blob path for a checkpoint."""
        return f"{thread_id}/{checkpoint_id}.json"

    def _serialize_checkpoint(self, checkpoint: Checkpoint) -> str:
        """Serialize checkpoint to JSON string."""
        # Convert checkpoint to serializable format
        checkpoint_data = {
            "v": checkpoint.get("v", 1),
            "id": checkpoint.get("id", ""),
            "ts": checkpoint.get("ts", datetime.now().isoformat()),
            "channel_values": checkpoint.get("channel_values", {}),
            "channel_versions": checkpoint.get("channel_versions", {}),
            "versions_seen": checkpoint.get("versions_seen", {}),
        }
        return json.dumps(checkpoint_data, default=str)

    def _deserialize_checkpoint(self, data: str) -> Checkpoint:
        """Deserialize JSON string to checkpoint."""
        checkpoint_data = json.loads(data)
        return Checkpoint(
            v=checkpoint_data.get("v", 1),
            id=checkpoint_data.get("id", ""),
            ts=checkpoint_data.get("ts", ""),
            channel_values=checkpoint_data.get("channel_values", {}),
            channel_versions=checkpoint_data.get("channel_versions", {}),
            versions_seen=checkpoint_data.get("versions_seen", {}),
        )

    def _extract_zone_from_thread_id(self, thread_id: str) -> str:
        """Extract zone from hierarchical thread ID."""
        # Thread ID format: main_{user_id}_{zone}_{timestamp}
        # or: instrument_identifier_{main_ref}_{timestamp}
        parts = thread_id.split("_")

        if thread_id.startswith("main_") and len(parts) >= 3:
            # Extract zone from main thread ID
            return parts[2] if len(parts) > 2 else self.config.default_zone

        # For workflow/item threads, look up parent or use default
        return self.config.default_zone

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
    ) -> RunnableConfig:
        """
        Save a checkpoint to Azure Blob Storage.

        Args:
            config: Configuration with thread_id
            checkpoint: The checkpoint data to save
            metadata: Checkpoint metadata

        Returns:
            Updated configuration with checkpoint info
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = checkpoint.get("id", str(datetime.now().timestamp()))

        # Extract zone from thread ID or metadata
        zone = metadata.get("zone") or self._extract_zone_from_thread_id(thread_id)

        try:
            container_client = self._get_container_client(zone)
            blob_path = self._get_blob_path(thread_id, checkpoint_id)
            blob_client = container_client.get_blob_client(blob_path)

            # Serialize checkpoint
            checkpoint_data = self._serialize_checkpoint(checkpoint)

            # Prepare metadata
            blob_metadata = {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
                "zone": zone,
                "created_at": datetime.now().isoformat(),
                "user_id": str(metadata.get("user_id", "anonymous")),
                "workflow_type": str(metadata.get("workflow_type", "unknown")),
                "parent_thread_id": str(metadata.get("parent_thread_id", "")),
            }

            # Upload blob with metadata
            blob_client.upload_blob(
                checkpoint_data,
                overwrite=True,
                metadata=blob_metadata
            )

            logger.debug(f"[AZURE_CHECKPOINT] Saved checkpoint: {blob_path} in zone {zone}")

            # Return updated config
            return {
                **config,
                "configurable": {
                    **config.get("configurable", {}),
                    "thread_id": thread_id,
                    "checkpoint_id": checkpoint_id,
                }
            }

        except Exception as e:
            logger.error(f"[AZURE_CHECKPOINT] Failed to save checkpoint: {e}")
            raise

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """
        Get a checkpoint tuple from Azure Blob Storage.

        Args:
            config: Configuration with thread_id and optional checkpoint_id

        Returns:
            CheckpointTuple if found, None otherwise
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = config["configurable"].get("checkpoint_id")

        zone = self._extract_zone_from_thread_id(thread_id)

        try:
            container_client = self._get_container_client(zone)

            if checkpoint_id:
                # Get specific checkpoint
                blob_path = self._get_blob_path(thread_id, checkpoint_id)
                blob_client = container_client.get_blob_client(blob_path)
            else:
                # Get latest checkpoint for thread
                blobs = list(container_client.list_blobs(name_starts_with=f"{thread_id}/"))
                if not blobs:
                    return None

                # Sort by last modified (most recent first)
                blobs.sort(key=lambda b: b.last_modified, reverse=True)
                latest_blob = blobs[0]
                blob_client = container_client.get_blob_client(latest_blob.name)
                checkpoint_id = latest_blob.name.split("/")[-1].replace(".json", "")

            # Download and deserialize
            try:
                blob_data = blob_client.download_blob().readall()
                blob_properties = blob_client.get_blob_properties()
                blob_metadata = blob_properties.metadata or {}
            except ResourceNotFoundError:
                return None

            checkpoint = self._deserialize_checkpoint(blob_data.decode("utf-8"))

            metadata = CheckpointMetadata(
                source="azure_blob",
                step=0,
                writes={},
            )

            return CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_id": checkpoint_id,
                    }
                },
                checkpoint=checkpoint,
                metadata=metadata,
                parent_config=None,
            )

        except Exception as e:
            logger.error(f"[AZURE_CHECKPOINT] Failed to get checkpoint: {e}")
            return None

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """
        List checkpoints from Azure Blob Storage.

        Args:
            config: Optional configuration with thread_id filter
            filter: Optional metadata filters
            before: Optional config to list checkpoints before
            limit: Maximum number of checkpoints to return

        Yields:
            CheckpointTuple for each matching checkpoint
        """
        thread_id = config["configurable"].get("thread_id") if config else None
        zone = self._extract_zone_from_thread_id(thread_id) if thread_id else self.config.default_zone

        try:
            container_client = self._get_container_client(zone)

            prefix = f"{thread_id}/" if thread_id else ""
            blobs = container_client.list_blobs(name_starts_with=prefix)

            # Sort by last modified
            blobs_list = sorted(blobs, key=lambda b: b.last_modified, reverse=True)

            count = 0
            for blob in blobs_list:
                if limit and count >= limit:
                    break

                try:
                    blob_client = container_client.get_blob_client(blob.name)
                    blob_data = blob_client.download_blob().readall()
                    blob_properties = blob_client.get_blob_properties()

                    checkpoint = self._deserialize_checkpoint(blob_data.decode("utf-8"))

                    parts = blob.name.split("/")
                    blob_thread_id = parts[0] if parts else ""
                    checkpoint_id = parts[-1].replace(".json", "") if parts else ""

                    yield CheckpointTuple(
                        config={
                            "configurable": {
                                "thread_id": blob_thread_id,
                                "checkpoint_id": checkpoint_id,
                            }
                        },
                        checkpoint=checkpoint,
                        metadata=CheckpointMetadata(source="azure_blob", step=0, writes={}),
                        parent_config=None,
                    )
                    count += 1

                except Exception as e:
                    logger.warning(f"[AZURE_CHECKPOINT] Failed to read blob {blob.name}: {e}")
                    continue

        except Exception as e:
            logger.error(f"[AZURE_CHECKPOINT] Failed to list checkpoints: {e}")

    # ========================================================================
    # THREAD TREE OPERATIONS
    # ========================================================================

    def get_thread_tree(self, main_thread_id: str) -> Dict[str, Any]:
        """
        Get the complete thread tree for a main thread.

        Args:
            main_thread_id: The main thread ID (root of the tree)

        Returns:
            Dictionary representing the thread tree structure
        """
        zone = self._extract_zone_from_thread_id(main_thread_id)

        try:
            container_client = self._get_container_client(zone)

            # Get all blobs that are part of this tree
            # Main thread blobs start with the main_thread_id
            # Child threads reference the main thread

            tree = {
                "main_thread_id": main_thread_id,
                "zone": zone,
                "workflow_threads": [],
                "item_threads": [],
            }

            # List all blobs in the container
            all_blobs = list(container_client.list_blobs())

            for blob in all_blobs:
                blob_client = container_client.get_blob_client(blob.name)
                try:
                    props = blob_client.get_blob_properties()
                    metadata = props.metadata or {}

                    thread_id = metadata.get("thread_id", "")
                    parent_id = metadata.get("parent_thread_id", "")

                    # Check if this thread belongs to our tree
                    if thread_id.startswith(f"instrument_identifier_{main_thread_id}") or \
                       thread_id.startswith(f"solution_{main_thread_id}"):
                        tree["workflow_threads"].append({
                            "thread_id": thread_id,
                            "workflow_type": metadata.get("workflow_type", ""),
                            "created_at": metadata.get("created_at", ""),
                        })
                    elif thread_id.startswith("item_") and parent_id:
                        # Check if parent is a workflow thread in our tree
                        tree["item_threads"].append({
                            "thread_id": thread_id,
                            "parent_thread_id": parent_id,
                            "created_at": metadata.get("created_at", ""),
                        })

                except Exception as e:
                    logger.debug(f"[AZURE_CHECKPOINT] Skipping blob {blob.name}: {e}")
                    continue

            return tree

        except Exception as e:
            logger.error(f"[AZURE_CHECKPOINT] Failed to get thread tree: {e}")
            return {"main_thread_id": main_thread_id, "zone": zone, "error": str(e)}

    def save_item_state(
        self,
        item_thread_id: str,
        item_state: Dict[str, Any],
        zone: str,
    ) -> bool:
        """
        Save full persistent state for an item sub-thread.

        Args:
            item_thread_id: The item thread ID
            item_state: Complete state dictionary for the item
            zone: Zone for storage

        Returns:
            True if successful, False otherwise
        """
        try:
            container_client = self._get_container_client(zone)
            blob_path = f"item_states/{item_thread_id}.json"
            blob_client = container_client.get_blob_client(blob_path)

            # Add metadata
            item_state["_saved_at"] = datetime.now().isoformat()
            item_state["_zone"] = zone

            blob_metadata = {
                "thread_id": item_thread_id,
                "zone": zone,
                "item_type": item_state.get("item_type", "unknown"),
                "status": item_state.get("status", "pending"),
                "created_at": item_state.get("created_at", datetime.now().isoformat()),
            }

            blob_client.upload_blob(
                json.dumps(item_state, default=str),
                overwrite=True,
                metadata=blob_metadata
            )

            logger.debug(f"[AZURE_CHECKPOINT] Saved item state: {item_thread_id}")
            return True

        except Exception as e:
            logger.error(f"[AZURE_CHECKPOINT] Failed to save item state: {e}")
            return False

    def get_item_state(self, item_thread_id: str, zone: str) -> Optional[Dict[str, Any]]:
        """
        Get full persistent state for an item sub-thread.

        Args:
            item_thread_id: The item thread ID
            zone: Zone for storage

        Returns:
            Item state dictionary if found, None otherwise
        """
        try:
            container_client = self._get_container_client(zone)
            blob_path = f"item_states/{item_thread_id}.json"
            blob_client = container_client.get_blob_client(blob_path)

            blob_data = blob_client.download_blob().readall()
            return json.loads(blob_data.decode("utf-8"))

        except ResourceNotFoundError:
            return None
        except Exception as e:
            logger.error(f"[AZURE_CHECKPOINT] Failed to get item state: {e}")
            return None

    def update_item_state(
        self,
        item_thread_id: str,
        zone: str,
        updates: Dict[str, Any],
    ) -> bool:
        """
        Update specific fields in an item's state.

        Args:
            item_thread_id: The item thread ID
            zone: Zone for storage
            updates: Dictionary of fields to update

        Returns:
            True if successful, False otherwise
        """
        current_state = self.get_item_state(item_thread_id, zone)
        if current_state is None:
            return False

        # Merge updates
        current_state.update(updates)
        current_state["updated_at"] = datetime.now().isoformat()

        return self.save_item_state(item_thread_id, current_state, zone)

    # ========================================================================
    # CLEANUP OPERATIONS
    # ========================================================================

    def cleanup_expired_checkpoints(self, zone: Optional[str] = None) -> Dict[str, Any]:
        """
        Clean up checkpoints older than TTL.

        Args:
            zone: Optional zone to clean up. If None, cleans all zones.

        Returns:
            Cleanup summary
        """
        cutoff = datetime.now() - timedelta(hours=self.config.ttl_hours)
        removed_count = 0
        errors = []

        zones_to_clean = [zone] if zone else self._get_all_zones()

        for z in zones_to_clean:
            try:
                container_client = self._get_container_client(z)
                blobs = list(container_client.list_blobs())

                for blob in blobs:
                    if blob.last_modified and blob.last_modified.replace(tzinfo=None) < cutoff:
                        try:
                            container_client.delete_blob(blob.name)
                            removed_count += 1
                        except Exception as e:
                            errors.append(f"Failed to delete {blob.name}: {e}")

            except Exception as e:
                errors.append(f"Failed to clean zone {z}: {e}")

        logger.info(f"[AZURE_CHECKPOINT] Cleanup complete: removed {removed_count} checkpoints")

        return {
            "success": len(errors) == 0,
            "removed_count": removed_count,
            "cutoff_time": cutoff.isoformat(),
            "errors": errors if errors else None,
        }

    def _get_all_zones(self) -> List[str]:
        """Get list of all zones with containers."""
        zones = []
        try:
            containers = self.blob_service_client.list_containers(
                name_starts_with=self.config.container_prefix
            )
            for container in containers:
                # Extract zone from container name
                zone = container.name.replace(f"{self.config.container_prefix}-", "")
                zones.append(zone)
        except Exception as e:
            logger.error(f"[AZURE_CHECKPOINT] Failed to list zones: {e}")
        return zones

    def get_user_threads(self, user_id: str, zone: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all thread trees for a user.

        Args:
            user_id: User identifier
            zone: Optional zone filter

        Returns:
            List of thread tree summaries
        """
        threads = []
        zones_to_search = [zone] if zone else self._get_all_zones()

        for z in zones_to_search:
            try:
                container_client = self._get_container_client(z)
                blobs = container_client.list_blobs()

                seen_threads = set()
                for blob in blobs:
                    try:
                        blob_client = container_client.get_blob_client(blob.name)
                        props = blob_client.get_blob_properties()
                        metadata = props.metadata or {}

                        if metadata.get("user_id") == user_id:
                            thread_id = metadata.get("thread_id", "")
                            if thread_id and thread_id not in seen_threads:
                                seen_threads.add(thread_id)
                                threads.append({
                                    "thread_id": thread_id,
                                    "zone": z,
                                    "workflow_type": metadata.get("workflow_type", ""),
                                    "created_at": metadata.get("created_at", ""),
                                })
                    except Exception:
                        continue

            except Exception as e:
                logger.warning(f"[AZURE_CHECKPOINT] Failed to search zone {z}: {e}")

        return threads


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def get_azure_blob_checkpointer(
    config: Optional[AzureBlobCheckpointerConfig] = None
) -> AzureBlobCheckpointer:
    """
    Factory function to get Azure Blob Storage checkpointer.

    Args:
        config: Optional configuration. If None, loads from environment.

    Returns:
        Configured AzureBlobCheckpointer instance
    """
    return AzureBlobCheckpointer(config)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "AzureBlobCheckpointer",
    "AzureBlobCheckpointerConfig",
    "get_azure_blob_checkpointer",
]
