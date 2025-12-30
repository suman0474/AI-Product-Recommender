# tags/utils.py
"""
Utility functions for tag system.

This module provides helper functions for working with tags,
including response construction and tag manipulation.
"""

from typing import Dict, Any, Optional
from .models import ResponseTags, TaggedResponse, IntentType, ResponseStatus


def add_tags_to_response(
    response_data: Dict[str, Any],
    tags: ResponseTags,
    success: bool = True,
    error: Optional[str] = None
) -> Dict[str, Any]:
    """
    Add tags to response data.

    Args:
        response_data: Original response data
        tags: ResponseTags object
        success: Whether the request was successful
        error: Optional error message

    Returns:
        Response dictionary with tags added

    Example:
        >>> from tags import create_tags, add_tags_to_response
        >>> tags = create_tags("solution", "complete", has_instruments=True)
        >>> response = add_tags_to_response({"instruments": [...]}, tags)
        >>> "tags" in response
        True
    """
    response = {
        "success": success,
        "data": response_data,
        "tags": tags.dict()
    }

    if error:
        response["error"] = error

    return response


def create_tagged_response(
    success: bool,
    data: Optional[Dict[str, Any]],
    tags: ResponseTags,
    error: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a complete tagged response.

    Args:
        success: Whether request was successful
        data: Response data
        tags: ResponseTags object
        error: Error message (if any)

    Returns:
        Complete tagged response dictionary

    Example:
        >>> from tags import create_tags, create_tagged_response
        >>> tags = create_tags("solution", "complete")
        >>> response = create_tagged_response(
        ...     success=True,
        ...     data={"instruments": [...]},
        ...     tags=tags
        ... )
        >>> response["success"]
        True
    """
    response = TaggedResponse(
        success=success,
        data=data,
        tags=tags,
        error=error
    )

    return response.dict(exclude_none=True)


def extract_tags_from_response(response: Dict[str, Any]) -> Optional[ResponseTags]:
    """
    Extract tags from a response dictionary.

    Args:
        response: Response dictionary that may contain tags

    Returns:
        ResponseTags object if found, None otherwise

    Example:
        >>> response = {"success": True, "data": {...}, "tags": {...}}
        >>> tags = extract_tags_from_response(response)
        >>> tags.intent_type
        <IntentType.SOLUTION: 'solution'>
    """
    if "tags" not in response:
        return None

    tags_dict = response["tags"]

    try:
        return ResponseTags(**tags_dict)
    except Exception as e:
        logger.error(f"Failed to parse tags from response: {e}")
        return None


def merge_tags(base_tags: ResponseTags, override_tags: Dict[str, Any]) -> ResponseTags:
    """
    Merge tags with overrides.

    Useful for manually adjusting tags after automatic classification.

    Args:
        base_tags: Base ResponseTags object
        override_tags: Dictionary of fields to override

    Returns:
        New ResponseTags object with overrides applied

    Example:
        >>> from tags import create_tags, merge_tags
        >>> tags = create_tags("solution", "incomplete")
        >>> new_tags = merge_tags(tags, {"response_status": "complete"})
        >>> new_tags.response_status
        <ResponseStatus.COMPLETE: 'complete'>
    """
    tags_dict = base_tags.dict()
    tags_dict.update(override_tags)

    return ResponseTags(**tags_dict)


def update_content_flags(
    tags: ResponseTags,
    has_instruments: Optional[bool] = None,
    has_accessories: Optional[bool] = None,
    has_followup: Optional[bool] = None
) -> ResponseTags:
    """
    Update content flags in tags.

    Args:
        tags: Original ResponseTags object
        has_instruments: New value for has_instruments (None = no change)
        has_accessories: New value for has_accessories (None = no change)
        has_followup: New value for has_followup (None = no change)

    Returns:
        New ResponseTags object with updated content flags

    Example:
        >>> from tags import create_tags, update_content_flags
        >>> tags = create_tags("solution", "complete")
        >>> updated = update_content_flags(tags, has_instruments=True)
        >>> updated.content_flags.has_instruments
        True
    """
    tags_dict = tags.dict()

    if has_instruments is not None:
        tags_dict["content_flags"]["has_instruments"] = has_instruments

    if has_accessories is not None:
        tags_dict["content_flags"]["has_accessories"] = has_accessories

    if has_followup is not None:
        tags_dict["content_flags"]["has_followup"] = has_followup

    return ResponseTags(**tags_dict)


def is_complete(tags: ResponseTags) -> bool:
    """
    Check if response is complete.

    Args:
        tags: ResponseTags object

    Returns:
        True if response_status is COMPLETE, False otherwise

    Example:
        >>> from tags import create_tags, is_complete
        >>> tags = create_tags("solution", "complete")
        >>> is_complete(tags)
        True
    """
    return tags.response_status == ResponseStatus.COMPLETE


def is_incomplete(tags: ResponseTags) -> bool:
    """
    Check if response is incomplete.

    Args:
        tags: ResponseTags object

    Returns:
        True if response_status is INCOMPLETE, False otherwise
    """
    return tags.response_status == ResponseStatus.INCOMPLETE


def is_invalid(tags: ResponseTags) -> bool:
    """
    Check if response is invalid.

    Args:
        tags: ResponseTags object

    Returns:
        True if response_status is INVALID_REQUEST, False otherwise
    """
    return tags.response_status == ResponseStatus.INVALID_REQUEST


def requires_followup(tags: ResponseTags) -> bool:
    """
    Check if response requires follow-up from user.

    Args:
        tags: ResponseTags object

    Returns:
        True if has_followup flag is set or status is incomplete
    """
    return tags.content_flags.has_followup or is_incomplete(tags)


def should_render_instruments_panel(tags: ResponseTags) -> bool:
    """
    Check if UI should render instruments panel.

    Args:
        tags: ResponseTags object

    Returns:
        True if has_instruments flag is set
    """
    return tags.content_flags.has_instruments


def should_render_accessories_panel(tags: ResponseTags) -> bool:
    """
    Check if UI should render accessories panel.

    Args:
        tags: ResponseTags object

    Returns:
        True if has_accessories flag is set
    """
    return tags.content_flags.has_accessories


def get_ui_route(tags: ResponseTags) -> str:
    """
    Get suggested UI route based on intent type.

    Args:
        tags: ResponseTags object

    Returns:
        Suggested UI route/tab name

    Example:
        >>> from tags import create_tags, get_ui_route
        >>> tags = create_tags("solution", "complete")
        >>> get_ui_route(tags)
        'solution-design'
    """
    route_map = {
        IntentType.SOLUTION: "solution-design",
        IntentType.PRODUCT_SEARCH: "product-search",
        IntentType.PRODUCT_INFO: "knowledge-base",
        IntentType.CHAT: "chat",
        IntentType.INVALID: "error"
    }

    return route_map.get(tags.intent_type, "chat")


def format_tags_for_logging(tags: ResponseTags) -> str:
    """
    Format tags for logging.

    Args:
        tags: ResponseTags object

    Returns:
        Formatted string for logging

    Example:
        >>> from tags import create_tags, format_tags_for_logging
        >>> tags = create_tags("solution", "complete", has_instruments=True)
        >>> print(format_tags_for_logging(tags))
        [Intent: solution | Status: complete | Instruments: yes | Accessories: no | Followup: no]
    """
    return (
        f"[Intent: {tags.intent_type.value} | "
        f"Status: {tags.response_status.value} | "
        f"Instruments: {'yes' if tags.content_flags.has_instruments else 'no'} | "
        f"Accessories: {'yes' if tags.content_flags.has_accessories else 'no'} | "
        f"Followup: {'yes' if tags.content_flags.has_followup else 'no'}]"
    )


def validate_tags(tags_dict: Dict[str, Any]) -> bool:
    """
    Validate that a dictionary contains valid tag structure.

    Args:
        tags_dict: Dictionary to validate

    Returns:
        True if valid, False otherwise

    Example:
        >>> validate_tags({
        ...     "intent_type": "solution",
        ...     "response_status": "complete",
        ...     "content_flags": {"has_instruments": True}
        ... })
        True
    """
    try:
        ResponseTags(**tags_dict)
        return True
    except Exception:
        return False


# Import logger
import logging
logger = logging.getLogger(__name__)
