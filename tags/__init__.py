# tags/__init__.py
"""
Response tagging system for AIPR backend.

This module provides metadata tags for API responses to help the frontend:
1. Route to the correct UI workflow (solution design, product search, knowledge chat)
2. Determine response completeness (complete, incomplete, invalid)
3. Render appropriate UI components (instruments panel, accessories panel, follow-up prompts)

Based on the Tags.md specification.

Example usage:
    >>> from tags import classify_response, IntentType
    >>> tags = classify_response(
    ...     user_input="Design crude unit",
    ...     response_data={"identified_instruments": [...]}
    ... )
    >>> tags.intent_type
    <IntentType.SOLUTION: 'solution'>
    >>> tags.response_status
    <ResponseStatus.COMPLETE: 'complete'>
"""

# Version
__version__ = "1.0.0"
__author__ = "AI Platform Team"


# ============================================================================
# MODELS
# ============================================================================

from .models import (
    # Enums
    IntentType,
    ResponseStatus,

    # Models
    ContentFlags,
    ResponseTags,
    TaggedResponse,

    # Helper functions
    create_tags,
    create_solution_tags,
    create_product_search_tags,
    create_chat_tags,
    create_invalid_tags
)


# ============================================================================
# CLASSIFIER
# ============================================================================

from .classifier import (
    TagClassifier,
    get_tag_classifier,
    classify_response
)


# ============================================================================
# UTILS
# ============================================================================

from .utils import (
    # Response construction
    add_tags_to_response,
    create_tagged_response,

    # Tag manipulation
    extract_tags_from_response,
    merge_tags,
    update_content_flags,

    # Tag checking
    is_complete,
    is_incomplete,
    is_invalid,
    requires_followup,

    # UI helpers
    should_render_instruments_panel,
    should_render_accessories_panel,
    get_ui_route,

    # Logging
    format_tags_for_logging,

    # Validation
    validate_tags
)


# ============================================================================
# CONSTANTS (selective export)
# ============================================================================

from .constants import (
    # Workflow mapping
    WORKFLOW_TO_INTENT_MAP,
)


# ============================================================================
# ALL EXPORTS
# ============================================================================

__all__ = [
    # Models - Enums
    'IntentType',
    'ResponseStatus',

    # Models - Classes
    'ContentFlags',
    'ResponseTags',
    'TaggedResponse',

    # Models - Helper functions
    'create_tags',
    'create_solution_tags',
    'create_product_search_tags',
    'create_chat_tags',
    'create_invalid_tags',

    # Classifier
    'TagClassifier',
    'get_tag_classifier',
    'classify_response',

    # Utils - Response construction
    'add_tags_to_response',
    'create_tagged_response',

    # Utils - Tag manipulation
    'extract_tags_from_response',
    'merge_tags',
    'update_content_flags',

    # Utils - Tag checking
    'is_complete',
    'is_incomplete',
    'is_invalid',
    'requires_followup',

    # Utils - UI helpers
    'should_render_instruments_panel',
    'should_render_accessories_panel',
    'get_ui_route',

    # Utils - Logging
    'format_tags_for_logging',

    # Utils - Validation
    'validate_tags',

    # Constants
    'WORKFLOW_TO_INTENT_MAP',
]


# ============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# ============================================================================

def quick_classify(user_input: str, response_data: dict, workflow_type: str = None):
    """
    Quick classification helper.

    Args:
        user_input: User input text
        response_data: Response data dictionary
        workflow_type: Optional workflow type

    Returns:
        ResponseTags object

    Example:
        >>> from tags import quick_classify
        >>> tags = quick_classify("Need PT", {"ranked_results": [...]})
        >>> tags.intent_type.value
        'product_search'
    """
    return classify_response(user_input, response_data, workflow_type)


def tag_response(success: bool, data: dict, user_input: str, workflow_type: str = None):
    """
    Automatically classify and create a tagged response in one call.

    Args:
        success: Whether request was successful
        data: Response data
        user_input: Original user input
        workflow_type: Optional workflow type

    Returns:
        Complete tagged response dictionary

    Example:
        >>> from tags import tag_response
        >>> response = tag_response(
        ...     success=True,
        ...     data={"identified_instruments": [...]},
        ...     user_input="Design crude unit",
        ...     workflow_type="solution"
        ... )
        >>> response["tags"]["intent_type"]
        'solution'
    """
    tags = classify_response(user_input, data, workflow_type)
    return create_tagged_response(success, data, tags)


# Add convenience functions to __all__
__all__.extend(['quick_classify', 'tag_response'])
