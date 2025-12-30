"""
State Management Utilities

Utilities for safely handling state transitions between different workflow
state types and ensuring type consistency.
"""

import logging
from typing import Any, Dict, Optional, TypedDict, Union
from enum import Enum

logger = logging.getLogger(__name__)


class StateConversionError(Exception):
    """Raised when state conversion fails"""
    pass


def safe_str_to_enum(value: Any, enum_class: type, default: Optional[Any] = None) -> Any:
    """
    Safely convert string to enum value.

    Args:
        value: Value to convert (string or enum)
        enum_class: Target enum class
        default: Default value if conversion fails

    Returns:
        Enum value or default
    """
    if value is None:
        return default

    if isinstance(value, enum_class):
        return value

    if isinstance(value, str):
        try:
            # Try direct match
            return enum_class(value)
        except ValueError:
            # Try case-insensitive match
            for enum_item in enum_class:
                if enum_item.value.lower() == value.lower():
                    return enum_item
            logger.warning(f"Could not convert '{value}' to {enum_class.__name__}, using default")
            return default

    return default


def safe_enum_to_str(value: Any, default: str = "") -> str:
    """
    Safely convert enum to string value.

    Args:
        value: Value to convert (enum or string)
        default: Default value if conversion fails

    Returns:
        String value
    """
    if value is None:
        return default

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, str):
        return value

    return default


def normalize_intent(intent: Any) -> Optional[str]:
    """
    Normalize intent field to string.

    Args:
        intent: Intent value (IntentType enum or string)

    Returns:
        Normalized string value or None
    """
    if intent is None:
        return None

    if isinstance(intent, Enum):
        return intent.value

    return str(intent)


def ensure_dict(value: Any, default: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Ensure value is a dictionary.

    Args:
        value: Value to convert
        default: Default dict if conversion fails

    Returns:
        Dictionary value
    """
    if isinstance(value, dict):
        return value

    if value is None:
        return default or {}

    logger.warning(f"Expected dict, got {type(value).__name__}, using default")
    return default or {}


def ensure_list(value: Any, default: Optional[list] = None) -> list:
    """
    Ensure value is a list.

    Args:
        value: Value to convert
        default: Default list if conversion fails

    Returns:
        List value
    """
    if isinstance(value, list):
        return value

    if value is None:
        return default or []

    if isinstance(value, (str, dict)):
        return [value]

    logger.warning(f"Expected list, got {type(value).__name__}, using default")
    return default or []


def merge_states(base_state: Dict[str, Any], update_state: Dict[str, Any],
                  preserve_none: bool = False) -> Dict[str, Any]:
    """
    Safely merge two state dictionaries.

    Args:
        base_state: Base state dictionary
        update_state: State updates to apply
        preserve_none: If False, None values won't overwrite existing values

    Returns:
        Merged state dictionary
    """
    merged = base_state.copy()

    for key, value in update_state.items():
        if preserve_none or value is not None:
            merged[key] = value

    return merged


def extract_common_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract common fields that exist across all state types.

    Args:
        state: Any workflow state dictionary

    Returns:
        Dictionary with common fields only
    """
    common_fields = {
        'user_input', 'session_id', 'product_type', 'schema',
        'provided_requirements', 'missing_requirements', 'is_requirements_valid',
        'current_step', 'messages', 'response', 'error'
    }

    return {
        key: state.get(key)
        for key in common_fields
        if key in state
    }


def convert_workflow_state_to_solution(workflow_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert legacy WorkflowState to SolutionState format.

    Args:
        workflow_state: WorkflowState dictionary

    Returns:
        SolutionState-compatible dictionary
    """
    solution_state = extract_common_state(workflow_state)

    # Convert intent from enum to string
    solution_state['intent'] = normalize_intent(workflow_state.get('intent'))
    solution_state['intent_confidence'] = workflow_state.get('intent_confidence', 0.0)

    # Ensure list/dict types
    solution_state['available_vendors'] = ensure_list(workflow_state.get('available_vendors'))
    solution_state['filtered_vendors'] = ensure_list(workflow_state.get('filtered_vendors'))
    solution_state['rag_context'] = ensure_dict(workflow_state.get('rag_context'))

    # Initialize comparison mode fields
    solution_state['comparison_mode'] = False
    solution_state['mode_confidence'] = 0.0
    solution_state['comparison_output'] = None

    return solution_state


def convert_solution_state_to_workflow(solution_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert SolutionState to legacy WorkflowState format.

    Args:
        solution_state: SolutionState dictionary

    Returns:
        WorkflowState-compatible dictionary
    """
    workflow_state = extract_common_state(solution_state)

    # Keep intent as string (will be converted to enum if needed)
    workflow_state['intent'] = solution_state.get('intent')
    workflow_state['intent_confidence'] = solution_state.get('intent_confidence', 0.0)

    # Copy vendor fields
    workflow_state['available_vendors'] = ensure_list(solution_state.get('available_vendors'))
    workflow_state['filtered_vendors'] = ensure_list(solution_state.get('filtered_vendors'))

    # Handle analysis results
    if 'vendor_analysis' in solution_state:
        workflow_state['vendor_analysis'] = solution_state['vendor_analysis']

    return workflow_state


def validate_state_fields(state: Dict[str, Any], required_fields: list,
                           state_name: str = "State") -> None:
    """
    Validate that state has required fields.

    Args:
        state: State dictionary to validate
        required_fields: List of required field names
        state_name: Name of state for error messages

    Raises:
        StateConversionError: If required fields are missing
    """
    missing = [field for field in required_fields if field not in state]

    if missing:
        raise StateConversionError(
            f"{state_name} missing required fields: {missing}"
        )


def sanitize_state_for_logging(state: Dict[str, Any],
                                exclude_fields: Optional[list] = None) -> Dict[str, Any]:
    """
    Sanitize state dictionary for logging by removing large/sensitive fields.

    Args:
        state: State dictionary
        exclude_fields: Additional fields to exclude

    Returns:
        Sanitized state dictionary safe for logging
    """
    # Default fields to exclude (large data)
    default_excludes = {
        'pdf_content', 'products_data', 'parallel_analysis_results',
        'vendor_analysis', 'messages', 'rag_context'
    }

    if exclude_fields:
        default_excludes.update(exclude_fields)

    sanitized = {}
    for key, value in state.items():
        if key in default_excludes:
            if isinstance(value, (list, dict)):
                sanitized[key] = f"<{type(value).__name__} with {len(value)} items>"
            else:
                sanitized[key] = f"<{type(value).__name__}>"
        else:
            sanitized[key] = value

    return sanitized


__all__ = [
    'StateConversionError',
    'safe_str_to_enum',
    'safe_enum_to_str',
    'normalize_intent',
    'ensure_dict',
    'ensure_list',
    'merge_states',
    'extract_common_state',
    'convert_workflow_state_to_solution',
    'convert_solution_state_to_workflow',
    'validate_state_fields',
    'sanitize_state_for_logging',
]
