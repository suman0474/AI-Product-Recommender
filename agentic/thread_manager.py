# agentic/thread_manager.py
# Thread ID Management for Workflow Isolation
# Ensures each workflow execution has its own unique thread ID for state isolation
#
# HIERARCHICAL THREAD-ID SYSTEM (with UUID enhancement):
# This module implements a tree-based thread-ID structure with UUID segments
# for enterprise-grade collision protection:
#
# main_{user_id}_{zone}_{uuid}_{timestamp}
# ├── instrument_identifier_{main_ref}_{uuid}_{timestamp}
# │   ├── item_{wf_ref}_instrument_{hash}_{uuid}_{timestamp}
# │   └── item_{wf_ref}_accessory_{hash}_{uuid}_{timestamp}
# └── solution_{main_ref}_{uuid}_{timestamp}
#     ├── item_{wf_ref}_instrument_{hash}_{uuid}_{timestamp}
#     └── item_{wf_ref}_accessory_{hash}_{uuid}_{timestamp}
#
# UUID BENEFITS:
# - Zero collision even at same millisecond
# - Unpredictable IDs (security)
# - Enterprise-grade uniqueness for distributed systems

import logging
import hashlib
import re
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ============================================================================
# THREAD ZONE ENUM (re-exported from zone_detector for convenience)
# ============================================================================

class ThreadZone(str, Enum):
    """Available zones for thread partitioning."""
    US_WEST = "US-WEST"
    US_EAST = "US-EAST"
    EU_CENTRAL = "EU-CENTRAL"
    EU_WEST = "EU-WEST"
    ASIA_PACIFIC = "ASIA-PACIFIC"
    ASIA_SOUTH = "ASIA-SOUTH"
    DEFAULT = "DEFAULT"

    @classmethod
    def from_string(cls, value: str) -> "ThreadZone":
        """Convert string to ThreadZone with fallback to DEFAULT."""
        try:
            return cls(value.upper().replace(" ", "-").replace("_", "-"))
        except ValueError:
            return cls.DEFAULT


# ============================================================================
# THREAD TYPE ENUM
# ============================================================================

class ThreadType(str, Enum):
    """Types of threads in the hierarchy."""
    MAIN = "main"
    WORKFLOW = "workflow"
    ITEM = "item"


# ============================================================================
# WORKFLOW TYPE ENUM (for thread generation)
# ============================================================================

class WorkflowThreadType(str, Enum):
    """Workflow types that can have threads."""
    INSTRUMENT_IDENTIFIER = "instrument_identifier"
    SOLUTION = "solution"
    PRODUCT_SEARCH = "product_search"
    COMPARISON = "comparison"
    GROUNDED_CHAT = "grounded_chat"


# ============================================================================
# THREAD HIERARCHY INFO
# ============================================================================

@dataclass
class ThreadHierarchyInfo:
    """Information about a thread's position in the hierarchy."""
    thread_id: str
    thread_type: ThreadType
    zone: str
    user_id: Optional[str] = None
    workflow_type: Optional[str] = None
    parent_thread_id: Optional[str] = None
    main_thread_id: Optional[str] = None
    item_type: Optional[str] = None  # "instrument" or "accessory"
    item_name: Optional[str] = None
    item_number: Optional[int] = None
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "thread_id": self.thread_id,
            "thread_type": self.thread_type.value,
            "zone": self.zone,
            "user_id": self.user_id,
            "workflow_type": self.workflow_type,
            "parent_thread_id": self.parent_thread_id,
            "main_thread_id": self.main_thread_id,
            "item_type": self.item_type,
            "item_name": self.item_name,
            "item_number": self.item_number,
            "timestamp": self.timestamp,
        }


# ============================================================================
# HIERARCHICAL THREAD MANAGER
# ============================================================================

class HierarchicalThreadManager:
    """
    Manages hierarchical thread IDs for workflow executions.

    Thread ID Formats:
    - Main Thread: main_{user_id}_{zone}_{timestamp}
    - Workflow Thread: {workflow_type}_{main_ref}_{timestamp}
    - Item Thread: item_{wf_ref}_{item_type}_{hash}_{timestamp}

    Examples:
    - main_user123_US-WEST_20250120_143052_123
    - instrument_identifier_main123_20250120_143055_456
    - item_iid123_instrument_a1b2c3_20250120_143100_789
    """

    @staticmethod
    def generate_timestamp() -> str:
        """Generate timestamp string for thread IDs."""
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

    @staticmethod
    def generate_hash(value: str, length: int = 8) -> str:
        """Generate short hash for item identification."""
        return hashlib.md5(value.encode()).hexdigest()[:length]

    @staticmethod
    def sanitize_for_thread_id(value: str) -> str:
        """Sanitize a string for use in thread ID (alphanumeric + underscore only)."""
        # Remove special characters, replace spaces with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '', value.replace(' ', '_').replace('-', '_'))
        # Limit length
        return sanitized[:32] if len(sanitized) > 32 else sanitized

    @staticmethod
    def generate_uuid_segment(length: int = 8) -> str:
        """Generate a short UUID segment for thread ID uniqueness."""
        return uuid.uuid4().hex[:length]

    @classmethod
    def generate_main_thread_id(
        cls,
        user_id: str,
        zone: ThreadZone,
    ) -> str:
        """
        Generate a main thread ID for a user session.

        Format: main_{user_id}_{zone}_{uuid}_{timestamp}

        UUID segment ensures:
        - Zero collision even if same user logs in at same millisecond
        - Unpredictable IDs (security)
        - Enterprise-grade uniqueness for distributed systems

        Args:
            user_id: User identifier
            zone: ThreadZone for partitioning

        Returns:
            Main thread ID string
        """
        timestamp = cls.generate_timestamp()
        sanitized_user = cls.sanitize_for_thread_id(user_id)
        zone_str = zone.value.replace("-", "_")
        uuid_segment = cls.generate_uuid_segment()

        thread_id = f"main_{sanitized_user}_{zone_str}_{uuid_segment}_{timestamp}"
        logger.info(f"[THREAD] Generated main thread ID: {thread_id}")

        return thread_id

    @classmethod
    def generate_workflow_thread_id(
        cls,
        workflow_type: WorkflowThreadType,
        main_thread_id: str,
    ) -> str:
        """
        Generate a workflow thread ID.

        Format: {workflow_type}_{main_ref}_{uuid}_{timestamp}

        UUID segment provides extra collision protection for:
        - Same-millisecond requests
        - Distributed systems
        - Enterprise-grade isolation

        Args:
            workflow_type: Type of workflow
            main_thread_id: Parent main thread ID

        Returns:
            Workflow thread ID string
        """
        timestamp = cls.generate_timestamp()
        uuid_segment = cls.generate_uuid_segment()

        # Extract short reference from main thread ID
        # main_user123_US_WEST_uuid_20250120_143052_123 -> last 8 alphanumeric chars
        main_ref = main_thread_id[-12:].replace("_", "")[:8] if main_thread_id else "unknown"

        thread_id = f"{workflow_type.value}_{main_ref}_{uuid_segment}_{timestamp}"
        logger.info(f"[THREAD] Generated workflow thread ID: {thread_id}")

        return thread_id

    @classmethod
    def generate_item_thread_id(
        cls,
        workflow_thread_id: str,
        item_type: str,  # "instrument" or "accessory"
        item_name: str,
        item_number: int,
    ) -> str:
        """
        Generate an item sub-thread ID.

        Format: item_{wf_ref}_{item_type}_{hash}_{uuid}_{timestamp}

        UUID segment ensures uniqueness even for items with same name/number

        Args:
            workflow_thread_id: Parent workflow thread ID
            item_type: "instrument" or "accessory"
            item_name: Name of the item
            item_number: Item number in the list

        Returns:
            Item thread ID string
        """
        timestamp = cls.generate_timestamp()
        uuid_segment = cls.generate_uuid_segment()

        # Extract short reference from workflow thread ID
        wf_parts = workflow_thread_id.split("_")
        wf_ref = wf_parts[1] if len(wf_parts) > 1 else "wf"

        # Generate hash from item name and number for uniqueness
        item_hash = cls.generate_hash(f"{item_name}_{item_number}")

        # Normalize item type
        item_type_normalized = "inst" if item_type.lower() == "instrument" else "acc"

        thread_id = f"item_{wf_ref}_{item_type_normalized}_{item_hash}_{uuid_segment}_{timestamp}"
        logger.info(f"[THREAD] Generated item thread ID: {thread_id}")

        return thread_id

    @classmethod
    def parse_thread_id(cls, thread_id: str) -> ThreadHierarchyInfo:
        """
        Parse a thread ID to extract hierarchy information.

        Args:
            thread_id: Thread ID string

        Returns:
            ThreadHierarchyInfo with parsed components
        """
        parts = thread_id.split("_")

        # Determine thread type
        if thread_id.startswith("main_"):
            return cls._parse_main_thread_id(thread_id, parts)
        elif thread_id.startswith("item_"):
            return cls._parse_item_thread_id(thread_id, parts)
        else:
            # Workflow thread
            return cls._parse_workflow_thread_id(thread_id, parts)

    @classmethod
    def _parse_main_thread_id(cls, thread_id: str, parts: List[str]) -> ThreadHierarchyInfo:
        """Parse a main thread ID."""
        # Format: main_{user_id}_{zone}_{timestamp}
        user_id = parts[1] if len(parts) > 1 else "unknown"

        # Zone might be multi-part (US_WEST)
        zone = "DEFAULT"
        timestamp = None

        if len(parts) >= 4:
            # Try to find zone (look for known zone patterns)
            for i in range(2, len(parts) - 3):
                potential_zone = "_".join(parts[i:i+2]) if i + 1 < len(parts) else parts[i]
                try:
                    zone = ThreadZone.from_string(potential_zone).value
                    timestamp = "_".join(parts[i+2:]) if len(parts) > i + 2 else None
                    break
                except:
                    continue

            if zone == "DEFAULT":
                zone = parts[2] if len(parts) > 2 else "DEFAULT"
                timestamp = "_".join(parts[3:]) if len(parts) > 3 else None

        return ThreadHierarchyInfo(
            thread_id=thread_id,
            thread_type=ThreadType.MAIN,
            zone=zone,
            user_id=user_id,
            timestamp=timestamp,
            main_thread_id=thread_id,
        )

    @classmethod
    def _parse_workflow_thread_id(cls, thread_id: str, parts: List[str]) -> ThreadHierarchyInfo:
        """Parse a workflow thread ID."""
        # Format: {workflow_type}_{main_ref}_{timestamp}
        # Known workflow types with underscores
        known_workflows = ["instrument_identifier", "product_search", "grounded_chat"]

        workflow_type = parts[0]
        main_ref = None
        timestamp = None

        for known_wf in known_workflows:
            if thread_id.startswith(known_wf + "_"):
                workflow_type = known_wf
                remaining_parts = thread_id[len(known_wf) + 1:].split("_")
                main_ref = remaining_parts[0] if remaining_parts else None
                timestamp = "_".join(remaining_parts[1:]) if len(remaining_parts) > 1 else None
                break
        else:
            # Single-word workflow type
            main_ref = parts[1] if len(parts) > 1 else None
            timestamp = "_".join(parts[2:]) if len(parts) > 2 else None

        return ThreadHierarchyInfo(
            thread_id=thread_id,
            thread_type=ThreadType.WORKFLOW,
            zone="DEFAULT",  # Will be resolved from parent
            workflow_type=workflow_type,
            timestamp=timestamp,
        )

    @classmethod
    def _parse_item_thread_id(cls, thread_id: str, parts: List[str]) -> ThreadHierarchyInfo:
        """Parse an item thread ID."""
        # Format: item_{wf_ref}_{item_type}_{hash}_{timestamp}
        wf_ref = parts[1] if len(parts) > 1 else None
        item_type_code = parts[2] if len(parts) > 2 else None
        item_hash = parts[3] if len(parts) > 3 else None
        timestamp = "_".join(parts[4:]) if len(parts) > 4 else None

        item_type = "instrument" if item_type_code == "inst" else "accessory"

        return ThreadHierarchyInfo(
            thread_id=thread_id,
            thread_type=ThreadType.ITEM,
            zone="DEFAULT",  # Will be resolved from parent
            item_type=item_type,
            timestamp=timestamp,
        )

    @classmethod
    def get_thread_hierarchy(cls, thread_id: str) -> List[str]:
        """
        Get the path from root to this thread.

        Args:
            thread_id: Thread ID to get hierarchy for

        Returns:
            List of thread IDs from root (main) to the given thread
        """
        info = cls.parse_thread_id(thread_id)
        hierarchy = [thread_id]

        if info.parent_thread_id:
            parent_hierarchy = cls.get_thread_hierarchy(info.parent_thread_id)
            hierarchy = parent_hierarchy + hierarchy

        return hierarchy

    @classmethod
    def is_legacy_thread_id(cls, thread_id: str) -> bool:
        """
        Check if a thread ID is in the legacy (non-hierarchical) format.

        Legacy format: {workflow_type}_{session_id}_{timestamp}
        Hierarchical format: main_*, item_*, or workflow with main_ref

        Args:
            thread_id: Thread ID to check

        Returns:
            True if legacy format, False if hierarchical
        """
        # Hierarchical IDs have specific prefixes
        if thread_id.startswith("main_"):
            return False
        if thread_id.startswith("item_"):
            return False

        # Check if it looks like hierarchical workflow ID
        # Hierarchical: instrument_identifier_{main_ref}_{timestamp}
        # Legacy: instrument_identifier_{session_id}_{timestamp}

        # Legacy session IDs often contain "session" or are UUIDs
        parts = thread_id.split("_")
        for part in parts:
            if "session" in part.lower():
                return True
            # UUID pattern check (8-4-4-4-12)
            if len(part) == 36 and part.count("-") == 4:
                return True

        return False


# ============================================================================
# THREAD TREE BUILDER
# ============================================================================

@dataclass
class ThreadTreeNode:
    """Node in the thread tree structure."""
    thread_id: str
    thread_type: ThreadType
    zone: str
    user_id: Optional[str] = None
    workflow_type: Optional[str] = None
    parent_thread_id: Optional[str] = None
    item_type: Optional[str] = None
    item_name: Optional[str] = None
    item_number: Optional[int] = None
    created_at: Optional[str] = None
    children: List["ThreadTreeNode"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary including children."""
        return {
            "thread_id": self.thread_id,
            "thread_type": self.thread_type.value,
            "zone": self.zone,
            "user_id": self.user_id,
            "workflow_type": self.workflow_type,
            "parent_thread_id": self.parent_thread_id,
            "item_type": self.item_type,
            "item_name": self.item_name,
            "item_number": self.item_number,
            "created_at": self.created_at,
            "children": [child.to_dict() for child in self.children],
        }


class ThreadTreeBuilder:
    """
    Builds and manages thread tree structures.

    Provides methods to:
    - Create a complete thread tree for a workflow execution
    - Add workflow and item threads to an existing tree
    - Serialize/deserialize trees for storage
    """

    def __init__(self):
        """Initialize the thread tree builder."""
        self._trees: Dict[str, ThreadTreeNode] = {}

    def create_main_thread(
        self,
        user_id: str,
        zone: ThreadZone,
    ) -> ThreadTreeNode:
        """
        Create a new main thread (root of tree).

        Args:
            user_id: User identifier
            zone: ThreadZone for the user

        Returns:
            Root ThreadTreeNode
        """
        thread_id = HierarchicalThreadManager.generate_main_thread_id(user_id, zone)

        root = ThreadTreeNode(
            thread_id=thread_id,
            thread_type=ThreadType.MAIN,
            zone=zone.value,
            user_id=user_id,
            created_at=datetime.now().isoformat(),
        )

        self._trees[thread_id] = root
        logger.info(f"[THREAD_TREE] Created main thread: {thread_id}")

        return root

    def add_workflow_thread(
        self,
        main_thread_id: str,
        workflow_type: WorkflowThreadType,
    ) -> ThreadTreeNode:
        """
        Add a workflow thread to a main thread.

        Args:
            main_thread_id: Parent main thread ID
            workflow_type: Type of workflow

        Returns:
            Workflow ThreadTreeNode
        """
        thread_id = HierarchicalThreadManager.generate_workflow_thread_id(
            workflow_type, main_thread_id
        )

        # Get zone from parent
        parent = self._trees.get(main_thread_id)
        zone = parent.zone if parent else "DEFAULT"

        workflow_node = ThreadTreeNode(
            thread_id=thread_id,
            thread_type=ThreadType.WORKFLOW,
            zone=zone,
            workflow_type=workflow_type.value,
            parent_thread_id=main_thread_id,
            created_at=datetime.now().isoformat(),
        )

        # Add to parent's children
        if parent:
            parent.children.append(workflow_node)

        logger.info(f"[THREAD_TREE] Added workflow thread: {thread_id} under {main_thread_id}")

        return workflow_node

    def add_item_thread(
        self,
        workflow_thread_id: str,
        item_type: str,
        item_name: str,
        item_number: int,
        zone: str = "DEFAULT",
    ) -> ThreadTreeNode:
        """
        Add an item sub-thread to a workflow thread.

        Args:
            workflow_thread_id: Parent workflow thread ID
            item_type: "instrument" or "accessory"
            item_name: Name of the item
            item_number: Item number in the list
            zone: Zone for storage

        Returns:
            Item ThreadTreeNode
        """
        thread_id = HierarchicalThreadManager.generate_item_thread_id(
            workflow_thread_id, item_type, item_name, item_number
        )

        item_node = ThreadTreeNode(
            thread_id=thread_id,
            thread_type=ThreadType.ITEM,
            zone=zone,
            item_type=item_type,
            item_name=item_name,
            item_number=item_number,
            parent_thread_id=workflow_thread_id,
            created_at=datetime.now().isoformat(),
        )

        logger.info(f"[THREAD_TREE] Added item thread: {thread_id} under {workflow_thread_id}")

        return item_node

    def generate_item_threads_for_items(
        self,
        workflow_thread_id: str,
        items: List[Dict[str, Any]],
        zone: str = "DEFAULT",
    ) -> Dict[int, str]:
        """
        Generate item thread IDs for a list of items.

        Args:
            workflow_thread_id: Parent workflow thread ID
            items: List of item dictionaries with 'number', 'type', 'name'
            zone: Zone for storage

        Returns:
            Dictionary mapping item numbers to thread IDs
        """
        item_threads = {}

        for item in items:
            item_number = item.get("number", 0)
            item_type = item.get("type", "instrument")
            item_name = item.get("name", f"item_{item_number}")

            item_node = self.add_item_thread(
                workflow_thread_id=workflow_thread_id,
                item_type=item_type,
                item_name=item_name,
                item_number=item_number,
                zone=zone,
            )

            item_threads[item_number] = item_node.thread_id

        logger.info(f"[THREAD_TREE] Generated {len(item_threads)} item threads for workflow {workflow_thread_id}")

        return item_threads

    def get_tree(self, main_thread_id: str) -> Optional[ThreadTreeNode]:
        """Get the tree for a main thread."""
        return self._trees.get(main_thread_id)

    def serialize_tree(self, main_thread_id: str) -> Optional[Dict[str, Any]]:
        """Serialize a tree to dictionary."""
        tree = self._trees.get(main_thread_id)
        if tree:
            return tree.to_dict()
        return None


# Global thread tree builder instance
_thread_tree_builder: Optional[ThreadTreeBuilder] = None


def get_thread_tree_builder() -> ThreadTreeBuilder:
    """Get or create the global thread tree builder."""
    global _thread_tree_builder
    if _thread_tree_builder is None:
        _thread_tree_builder = ThreadTreeBuilder()
    return _thread_tree_builder


# ============================================================================
# LEGACY THREAD ID MANAGER (for backward compatibility)
# ============================================================================

class ThreadIDManager:
    """
    Manages thread IDs for workflow executions.

    Each workflow gets a unique thread ID to ensure:
    - State isolation between workflows
    - Proper checkpointing and state persistence
    - Better tracking and debugging
    - Avoid state conflicts when workflows chain together

    Thread ID Format:
    {workflow_type}_{session_id}_{timestamp}

    Examples:
    - instrument_identifier_session-123_20250101_120530
    - solution_session-123_20250101_120545
    - product_search_session-123_20250101_120600
    """

    @staticmethod
    def generate_thread_id(
        workflow_type: str,
        session_id: str,
        parent_thread_id: Optional[str] = None
    ) -> str:
        """
        Generate a unique thread ID for a workflow execution.

        Args:
            workflow_type: Type of workflow ("instrument_identifier", "solution", "product_search")
            session_id: User session identifier
            parent_thread_id: Optional parent workflow thread ID (for chaining)

        Returns:
            Unique thread ID string
        """
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds

        # Base thread ID
        thread_id = f"{workflow_type}_{session_id}_{timestamp}"

        # Add parent reference if provided (for workflow chaining)
        if parent_thread_id:
            # Extract parent workflow type from parent_thread_id
            parent_workflow = parent_thread_id.split("_")[0] if "_" in parent_thread_id else "unknown"
            thread_id = f"{thread_id}_from_{parent_workflow}"

        logger.info(f"[THREAD] Generated thread ID: {thread_id}")

        return thread_id

    @staticmethod
    def parse_thread_id(thread_id: str) -> dict:
        """
        Parse a thread ID to extract metadata.

        Thread ID Format: {workflow_type}_{session_id}_{timestamp}
        - workflow_type can contain underscores (e.g., "instrument_identifier")
        - session_id is the part before timestamp (before YYYYMMDD pattern)
        - timestamp is YYYYMMDD_HHMMSS_mmm format

        Args:
            thread_id: Thread ID string

        Returns:
            Dictionary with parsed components
        """
        try:
            parts = thread_id.split("_")

            # Known workflow types (to help with parsing)
            known_workflows = ["instrument_identifier", "product_search", "solution", "comparison", "grounded_chat"]

            # Try to find the workflow type by checking known types
            workflow_type = "unknown"
            workflow_end_index = 0

            for known_wf in known_workflows:
                if thread_id.startswith(known_wf + "_"):
                    workflow_type = known_wf
                    workflow_end_index = len(known_wf.split("_"))
                    break

            # If no known workflow found, assume workflow type is first part
            if workflow_type == "unknown" and len(parts) > 0:
                workflow_type = parts[0]
                workflow_end_index = 1

            # Find timestamp pattern (YYYYMMDD)
            timestamp_start_index = -1
            for i, part in enumerate(parts):
                if len(part) == 8 and part.isdigit():  # YYYYMMDD format
                    timestamp_start_index = i
                    break

            # Session ID is everything between workflow_type and timestamp
            if timestamp_start_index > workflow_end_index:
                session_parts = parts[workflow_end_index:timestamp_start_index]
                session_id = "_".join(session_parts) if session_parts else "unknown"
            else:
                session_id = "unknown"

            # Timestamp is from timestamp_start_index onward
            if timestamp_start_index != -1:
                # Look for "from" keyword in remaining parts
                remaining_parts = parts[timestamp_start_index:]
                if "from" in remaining_parts:
                    from_index = remaining_parts.index("from")
                    timestamp_parts = remaining_parts[:from_index]
                    parent_workflow = remaining_parts[from_index + 1] if from_index + 1 < len(remaining_parts) else None
                else:
                    timestamp_parts = remaining_parts
                    parent_workflow = None

                timestamp = "_".join(timestamp_parts) if timestamp_parts else "unknown"
            else:
                timestamp = "unknown"
                parent_workflow = None

            result = {
                "workflow_type": workflow_type,
                "session_id": session_id,
                "timestamp": timestamp,
                "parent_workflow": parent_workflow
            }

            return result

        except Exception as e:
            logger.error(f"[THREAD] Failed to parse thread ID '{thread_id}': {e}")
            return {
                "workflow_type": "unknown",
                "session_id": "unknown",
                "timestamp": "unknown",
                "parent_workflow": None
            }

    @staticmethod
    def get_workflow_config(thread_id: str) -> dict:
        """
        Generate LangGraph config with thread ID.

        Args:
            thread_id: Thread ID string

        Returns:
            Config dictionary for LangGraph workflow
        """
        return {
            "configurable": {
                "thread_id": thread_id
            }
        }


# Convenience function for direct import
def generate_thread_id(
    workflow_type: str,
    session_id: str,
    parent_thread_id: Optional[str] = None
) -> str:
    """
    Generate a unique thread ID for a workflow execution.

    Args:
        workflow_type: "instrument_identifier", "solution", or "product_search"
        session_id: User session identifier
        parent_thread_id: Optional parent workflow thread ID

    Returns:
        Unique thread ID string
    """
    return ThreadIDManager.generate_thread_id(workflow_type, session_id, parent_thread_id)


def get_workflow_config(thread_id: str) -> dict:
    """
    Generate LangGraph config with thread ID.

    Args:
        thread_id: Thread ID string

    Returns:
        Config dictionary for LangGraph workflow
    """
    return ThreadIDManager.get_workflow_config(thread_id)


# Export
__all__ = [
    # Enums
    'ThreadZone',
    'ThreadType',
    'WorkflowThreadType',

    # Data Classes
    'ThreadHierarchyInfo',
    'ThreadTreeNode',

    # Hierarchical Thread Manager (new)
    'HierarchicalThreadManager',
    'ThreadTreeBuilder',
    'get_thread_tree_builder',

    # Legacy Thread ID Manager (backward compatible)
    'ThreadIDManager',
    'generate_thread_id',
    'get_workflow_config',
]
