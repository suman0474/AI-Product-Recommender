# tags/models.py
"""
Pydantic models for response tagging system.
Based on Tags.md specification.

This module defines the core data structures for the tag system:
- IntentType: Enum for intent classification
- ResponseStatus: Enum for response completeness
- ContentFlags: Flags for UI component rendering
- ResponseTags: Complete tag structure
- TaggedResponse: Full API response with tags
"""

from pydantic import BaseModel, Field
from typing import Literal, Dict, Any, Optional
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class IntentType(str, Enum):
    """
    Intent type classification.

    Describes what the user is trying to do so the system can route
    to the right workflow/tab.
    """

    SOLUTION = "solution"
    """
    User is describing a project/system/solution requirement involving
    multiple instruments and accessories.

    Example: "Design instrumentation for a crude distillation unit with
    pumps, transmitters, flowmeters."
    """

    PRODUCT_SEARCH = "product_search"
    """
    User is trying to find/select a specific product (one instrument/accessory
    type) based on specs.

    Example: "Need a pressure transmitter for crude oil, SIL2, 0-500 psi."
    """

    PRODUCT_INFO = "product_info"
    """
    User is asking for information about an existing product (knowledge chat),
    usually grounded on product index.

    Example: "What is the outer case color of Honeywell STG74L?" or
    "What is the temperature rating of ST 800?"
    """

    CHAT = "chat"
    """
    Generic, free-form conversation that is still in domain but not directly
    solution/search/info.

    Example: "Explain what SIL2 means." or "How do DP transmitters work?"
    """

    INVALID = "invalid"
    """
    Question is outside domain or cannot be handled by the application.

    Example: "Who is the president of the United States?" or random chit-chat
    not about instruments/solutions.
    """


class ResponseStatus(str, Enum):
    """
    Response completeness status.

    Describes how usable the current response is and whether the system
    expects more input.
    """

    COMPLETE = "complete"
    """
    System has given a full, usable response for this step.
    No mandatory follow-up is required to proceed.

    Example: Valid solution extracted with instruments list and specs; or
    product comparison finished with ranked list.
    """

    INCOMPLETE = "incomplete"
    """
    System could not fully process the request and needs more information
    from the user. Usually accompanied by explicit follow-up questions.

    Example: User gave partial specs, and the model asks: "Please provide
    pressure range and fluid type."
    """

    INVALID_REQUEST = "invalid_request"
    """
    Request is not processable (out of domain, malformed, or violates
    constraints). UI should not try to render solution/instrument panels
    for this.

    Example: Asking for general trivia or a question clearly unrelated to
    instrumentation or products.
    """


# ============================================================================
# MODELS
# ============================================================================

class ContentFlags(BaseModel):
    """
    Content flags indicating what structured data is present.

    Describes what structured content is present so UI knows what to render.
    """

    has_instruments: bool = Field(
        default=False,
        description=(
            "True when the response includes a non-empty list of instruments. "
            "UI should render the instruments panel/table. "
            "False when there are no instruments in this response "
            "(either none detected or not relevant for this intent)."
        )
    )

    has_accessories: bool = Field(
        default=False,
        description=(
            "True when the response includes a non-empty list of accessories. "
            "UI should render accessories panel/table. "
            "False when there are no accessories in this response."
        )
    )

    has_followup: bool = Field(
        default=False,
        description=(
            "True when the message explicitly asks the user one or more "
            "follow-up questions (e.g., to complete specs). "
            "UI should visually highlight the follow-up and keep the "
            "conversation 'open', not show a final/summary state. "
            "Typical with response_status = 'incomplete'."
        )
    )

    class Config:
        """Pydantic config"""
        json_schema_extra = {
            "example": {
                "has_instruments": True,
                "has_accessories": True,
                "has_followup": False
            }
        }


class ResponseTags(BaseModel):
    """
    Complete tag structure for API responses.

    This is the main tag object that gets added to all API responses.
    """

    intent_type: IntentType = Field(
        description=(
            "User intent classification. One of: solution, product_search, "
            "product_info, chat, invalid"
        )
    )

    response_status: ResponseStatus = Field(
        description=(
            "Response completeness. One of: complete, incomplete, invalid_request"
        )
    )

    content_flags: ContentFlags = Field(
        default_factory=ContentFlags,
        description="Flags indicating what UI components to render"
    )

    class Config:
        """Pydantic config"""
        use_enum_values = True  # Return enum values as strings in JSON
        json_schema_extra = {
            "example": {
                "intent_type": "solution",
                "response_status": "complete",
                "content_flags": {
                    "has_instruments": True,
                    "has_accessories": True,
                    "has_followup": False
                }
            }
        }


class TaggedResponse(BaseModel):
    """
    Complete API response with tags.

    This is the full response structure that includes both the data
    and the metadata tags.
    """

    success: bool = Field(
        description="Whether the request was successful"
    )

    data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Response data (workflow results, products, answers, etc.)"
    )

    tags: ResponseTags = Field(
        description="Metadata tags for frontend routing and UI rendering"
    )

    error: Optional[str] = Field(
        default=None,
        description="Error message if success=False"
    )

    class Config:
        """Pydantic config"""
        json_schema_extra = {
            "example": {
                "success": True,
                "data": {
                    "identified_instruments": [
                        {
                            "name": "Pressure Transmitter",
                            "quantity": 5
                        }
                    ]
                },
                "tags": {
                    "intent_type": "solution",
                    "response_status": "complete",
                    "content_flags": {
                        "has_instruments": True,
                        "has_accessories": False,
                        "has_followup": False
                    }
                },
                "error": None
            }
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_tags(
    intent_type: IntentType | str,
    response_status: ResponseStatus | str,
    has_instruments: bool = False,
    has_accessories: bool = False,
    has_followup: bool = False
) -> ResponseTags:
    """
    Helper function to create ResponseTags object.

    This is a convenience function that simplifies tag creation.

    Args:
        intent_type: Intent classification (enum or string)
        response_status: Response completeness (enum or string)
        has_instruments: Whether instruments are present in response
        has_accessories: Whether accessories are present in response
        has_followup: Whether follow-up questions are asked

    Returns:
        ResponseTags object

    Example:
        >>> tags = create_tags(
        ...     intent_type=IntentType.SOLUTION,
        ...     response_status=ResponseStatus.COMPLETE,
        ...     has_instruments=True,
        ...     has_accessories=True
        ... )
        >>> tags.intent_type
        <IntentType.SOLUTION: 'solution'>
    """
    return ResponseTags(
        intent_type=intent_type,
        response_status=response_status,
        content_flags=ContentFlags(
            has_instruments=has_instruments,
            has_accessories=has_accessories,
            has_followup=has_followup
        )
    )


def create_solution_tags(
    has_instruments: bool = True,
    has_accessories: bool = False,
    is_complete: bool = True
) -> ResponseTags:
    """
    Convenience function to create tags for solution workflow.

    Args:
        has_instruments: Whether instruments list is present
        has_accessories: Whether accessories list is present
        is_complete: Whether the solution is complete

    Returns:
        ResponseTags with intent_type=SOLUTION
    """
    return create_tags(
        intent_type=IntentType.SOLUTION,
        response_status=ResponseStatus.COMPLETE if is_complete else ResponseStatus.INCOMPLETE,
        has_instruments=has_instruments,
        has_accessories=has_accessories,
        has_followup=not is_complete
    )


def create_product_search_tags(
    has_results: bool = True,
    is_complete: bool = True
) -> ResponseTags:
    """
    Convenience function to create tags for product search workflow.

    Args:
        has_results: Whether ranked results are present
        is_complete: Whether all required specs are provided

    Returns:
        ResponseTags with intent_type=PRODUCT_SEARCH
    """
    return create_tags(
        intent_type=IntentType.PRODUCT_SEARCH,
        response_status=ResponseStatus.COMPLETE if is_complete else ResponseStatus.INCOMPLETE,
        has_instruments=False,  # Product search doesn't return instruments list
        has_accessories=False,
        has_followup=not is_complete
    )


def create_chat_tags(
    intent_type: IntentType = IntentType.CHAT,
    is_complete: bool = True
) -> ResponseTags:
    """
    Convenience function to create tags for chat/knowledge workflows.

    Args:
        intent_type: Either CHAT or PRODUCT_INFO
        is_complete: Whether the answer is complete

    Returns:
        ResponseTags for chat workflows
    """
    return create_tags(
        intent_type=intent_type,
        response_status=ResponseStatus.COMPLETE if is_complete else ResponseStatus.INCOMPLETE,
        has_instruments=False,
        has_accessories=False,
        has_followup=not is_complete
    )


def create_invalid_tags() -> ResponseTags:
    """
    Convenience function to create tags for invalid requests.

    Returns:
        ResponseTags with intent_type=INVALID, response_status=INVALID_REQUEST
    """
    return create_tags(
        intent_type=IntentType.INVALID,
        response_status=ResponseStatus.INVALID_REQUEST,
        has_instruments=False,
        has_accessories=False,
        has_followup=False
    )
