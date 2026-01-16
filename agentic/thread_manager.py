# agentic/thread_manager.py
# Thread ID Management for Workflow Isolation
# Ensures each workflow execution has its own unique thread ID for state isolation

import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


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
    'ThreadIDManager',
    'generate_thread_id',
    'get_workflow_config'
]
