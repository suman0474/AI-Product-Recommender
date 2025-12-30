# tags/classifier.py
"""
Tag classification logic for API responses.

This module implements the TagClassifier class which analyzes response data
and assigns appropriate tags based on user input, response structure, and
workflow metadata.
"""

import logging
from typing import Dict, Any, List, Optional
from .models import (
    ResponseTags, IntentType, ResponseStatus,
    ContentFlags, create_tags
)
from .constants import (
    SOLUTION_KEYWORDS, PRODUCT_SEARCH_KEYWORDS, PRODUCT_INFO_KEYWORDS,
    CHAT_KEYWORDS, INVALID_KEYWORDS, INCOMPLETE_PHRASES, FOLLOWUP_PHRASES,
    INSTRUMENT_FIELDS, ACCESSORY_FIELDS, PRODUCT_MODEL_PATTERNS, VENDOR_NAMES,
    WORKFLOW_TO_INTENT_MAP, contains_any_keyword, count_keyword_matches,
    normalize_text
)

logger = logging.getLogger(__name__)


# ============================================================================
# MAIN CLASSIFIER
# ============================================================================

class TagClassifier:
    """
    Classifies API responses and assigns tags.

    The classifier uses a multi-priority approach:
    1. Routing metadata (if available from smart-chat router)
    2. Workflow type (if provided explicitly)
    3. Keyword analysis (user input)
    4. Response structure analysis (data fields)
    """

    def __init__(self):
        """Initialize the tag classifier."""
        self.logger = logging.getLogger(__name__)

    def classify(
        self,
        user_input: str,
        response_data: Dict[str, Any],
        workflow_type: Optional[str] = None,
        routing_metadata: Optional[Dict[str, Any]] = None
    ) -> ResponseTags:
        """
        Classify response and generate tags.

        Args:
            user_input: Original user input/question
            response_data: Response data from workflow/API
            workflow_type: Workflow type (if known) - e.g., "solution", "comparison"
            routing_metadata: Routing decision metadata (from smart-chat)

        Returns:
            ResponseTags object with classified tags

        Example:
            >>> classifier = TagClassifier()
            >>> tags = classifier.classify(
            ...     user_input="Design crude unit",
            ...     response_data={"identified_instruments": [...]},
            ...     workflow_type="solution"
            ... )
            >>> tags.intent_type
            <IntentType.SOLUTION: 'solution'>
        """

        # Step 1: Determine intent_type
        intent_type = self._classify_intent(
            user_input, response_data, workflow_type, routing_metadata
        )

        # Step 2: Determine response_status
        response_status = self._classify_response_status(
            response_data, intent_type
        )

        # Step 3: Determine content_flags
        content_flags = self._classify_content_flags(response_data)

        # Create tags
        tags = ResponseTags(
            intent_type=intent_type,
            response_status=response_status,
            content_flags=content_flags
        )

        self.logger.info(
            f"[TAG_CLASSIFIER] Intent: {intent_type.value}, "
            f"Status: {response_status.value}, "
            f"Flags: instruments={content_flags.has_instruments}, "
            f"accessories={content_flags.has_accessories}, "
            f"followup={content_flags.has_followup}"
        )

        return tags


    # ========================================================================
    # INTENT CLASSIFICATION
    # ========================================================================

    def _classify_intent(
        self,
        user_input: str,
        response_data: Dict[str, Any],
        workflow_type: Optional[str] = None,
        routing_metadata: Optional[Dict[str, Any]] = None
    ) -> IntentType:
        """
        Classify user intent.

        Priority order:
        1. Use routing_metadata if available (from smart-chat router)
        2. Use workflow_type if available
        3. Analyze user_input keywords
        4. Analyze response_data structure

        Args:
            user_input: User input text
            response_data: Response data dictionary
            workflow_type: Workflow type string
            routing_metadata: Routing metadata dict

        Returns:
            IntentType enum value
        """

        user_input_lower = normalize_text(user_input)

        # ====================================================================
        # Priority 1: Use router decision if available
        # ====================================================================
        if routing_metadata and 'workflow' in routing_metadata:
            workflow = routing_metadata['workflow'].lower()

            if workflow == 'solution':
                return IntentType.SOLUTION

            elif workflow == 'instrument_detail':
                return IntentType.SOLUTION

            elif workflow in ['comparison', 'product_search']:
                return IntentType.PRODUCT_SEARCH

            elif workflow == 'grounded_chat':
                # Could be product_info or chat, need to analyze further
                if self._has_product_reference(user_input_lower):
                    return IntentType.PRODUCT_INFO
                else:
                    return IntentType.CHAT

        # ====================================================================
        # Priority 2: Use workflow_type if available
        # ====================================================================
        if workflow_type:
            workflow_type_lower = workflow_type.lower()

            # Check workflow mapping
            if workflow_type_lower in WORKFLOW_TO_INTENT_MAP:
                intent_str = WORKFLOW_TO_INTENT_MAP[workflow_type_lower]

                # Special case for grounded_chat
                if intent_str == 'chat':
                    if self._has_product_reference(user_input_lower):
                        return IntentType.PRODUCT_INFO
                    else:
                        return IntentType.CHAT

                # Convert string to enum
                return IntentType(intent_str)

        # ====================================================================
        # Priority 3: Keyword analysis
        # ====================================================================

        # Check INVALID first (highest priority for exclusion)
        if contains_any_keyword(user_input, INVALID_KEYWORDS):
            return IntentType.INVALID

        # Count matches for each intent type
        solution_matches = count_keyword_matches(user_input, SOLUTION_KEYWORDS)
        product_search_matches = count_keyword_matches(user_input, PRODUCT_SEARCH_KEYWORDS)
        product_info_matches = count_keyword_matches(user_input, PRODUCT_INFO_KEYWORDS)
        chat_matches = count_keyword_matches(user_input, CHAT_KEYWORDS)

        # Find the intent with most matches
        max_matches = max(solution_matches, product_search_matches, product_info_matches, chat_matches)

        if max_matches > 0:
            if solution_matches == max_matches:
                return IntentType.SOLUTION
            elif product_search_matches == max_matches:
                return IntentType.PRODUCT_SEARCH
            elif product_info_matches == max_matches:
                return IntentType.PRODUCT_INFO
            elif chat_matches == max_matches:
                return IntentType.CHAT

        # ====================================================================
        # Priority 4: Analyze response structure
        # ====================================================================

        # Has instruments/accessories → likely SOLUTION
        if self._has_instruments(response_data) or self._has_accessories(response_data):
            return IntentType.SOLUTION

        # Has ranked_results → likely PRODUCT_SEARCH
        if 'ranked_results' in response_data or 'ranked_products' in response_data:
            return IntentType.PRODUCT_SEARCH

        # Has answer with product reference → likely PRODUCT_INFO
        if 'answer' in response_data:
            if self._has_product_reference(user_input_lower):
                return IntentType.PRODUCT_INFO
            else:
                return IntentType.CHAT

        # Has error → check if it's invalid
        if response_data.get('error'):
            error_msg = str(response_data.get('error', '')).lower()
            if 'domain' in error_msg or 'cannot' in error_msg:
                return IntentType.INVALID

        # ====================================================================
        # Default: CHAT (safe fallback)
        # ====================================================================
        return IntentType.CHAT


    def _has_product_reference(self, text: str) -> bool:
        """
        Check if text references a specific product model or vendor.

        Args:
            text: Input text (already lowercased)

        Returns:
            True if product reference found, False otherwise
        """
        # Check for model patterns
        for pattern in PRODUCT_MODEL_PATTERNS:
            if pattern in text:
                return True

        # Check for vendor names
        for vendor in VENDOR_NAMES:
            if vendor in text:
                return True

        return False


    # ========================================================================
    # RESPONSE STATUS CLASSIFICATION
    # ========================================================================

    def _classify_response_status(
        self,
        response_data: Dict[str, Any],
        intent_type: IntentType
    ) -> ResponseStatus:
        """
        Classify response completeness status.

        Args:
            response_data: Response data dictionary
            intent_type: Already classified intent type

        Returns:
            ResponseStatus enum value
        """

        # ====================================================================
        # Check for INVALID_REQUEST
        # ====================================================================
        if intent_type == IntentType.INVALID:
            return ResponseStatus.INVALID_REQUEST

        # Check for explicit error
        if response_data.get('error'):
            return ResponseStatus.INVALID_REQUEST

        # ====================================================================
        # Check for INCOMPLETE indicators
        # ====================================================================

        # Check response text for incomplete phrases
        response_text = response_data.get('response', '')
        if isinstance(response_text, str):
            response_lower = normalize_text(response_text)

            # Look for incomplete phrases
            if contains_any_keyword(response_text, INCOMPLETE_PHRASES):
                return ResponseStatus.INCOMPLETE

        # ====================================================================
        # Check for missing data based on intent type
        # ====================================================================

        if intent_type == IntentType.SOLUTION:
            # Solution needs instruments or accessories
            if not self._has_instruments(response_data) and not self._has_accessories(response_data):
                return ResponseStatus.INCOMPLETE

        elif intent_type == IntentType.PRODUCT_SEARCH:
            # Product search needs ranked results
            if not response_data.get('ranked_results') and not response_data.get('ranked_products'):
                # Check if it's asking for more specs
                if response_text and '?' in response_text:
                    return ResponseStatus.INCOMPLETE

        elif intent_type == IntentType.PRODUCT_INFO:
            # Product info needs answer
            if not response_data.get('answer') and not response_text:
                return ResponseStatus.INCOMPLETE

        elif intent_type == IntentType.CHAT:
            # Chat needs answer
            if not response_data.get('answer') and not response_text:
                return ResponseStatus.INCOMPLETE

        # ====================================================================
        # Check validation flags
        # ====================================================================
        is_valid = response_data.get('is_valid')
        if is_valid is not None and not is_valid:
            return ResponseStatus.INCOMPLETE

        # Check for missing mandatory fields
        missing_fields = response_data.get('missingMandatoryFields') or response_data.get('missing_mandatory_fields')
        if missing_fields and len(missing_fields) > 0:
            return ResponseStatus.INCOMPLETE

        # ====================================================================
        # Default: COMPLETE
        # ====================================================================
        return ResponseStatus.COMPLETE


    # ========================================================================
    # CONTENT FLAGS CLASSIFICATION
    # ========================================================================

    def _classify_content_flags(
        self,
        response_data: Dict[str, Any]
    ) -> ContentFlags:
        """
        Classify content flags (what UI components to render).

        Args:
            response_data: Response data dictionary

        Returns:
            ContentFlags object
        """

        has_instruments = self._has_instruments(response_data)
        has_accessories = self._has_accessories(response_data)
        has_followup = self._has_followup(response_data)

        return ContentFlags(
            has_instruments=has_instruments,
            has_accessories=has_accessories,
            has_followup=has_followup
        )


    def _has_instruments(self, response_data: Dict[str, Any]) -> bool:
        """
        Check if response has instruments.

        Args:
            response_data: Response data dictionary

        Returns:
            True if instruments present, False otherwise
        """

        # Check for instrument fields
        for field in INSTRUMENT_FIELDS:
            if field in response_data:
                value = response_data[field]
                if isinstance(value, list) and len(value) > 0:
                    return True

        return False


    def _has_accessories(self, response_data: Dict[str, Any]) -> bool:
        """
        Check if response has accessories.

        Args:
            response_data: Response data dictionary

        Returns:
            True if accessories present, False otherwise
        """

        # Check for accessory fields
        for field in ACCESSORY_FIELDS:
            if field in response_data:
                value = response_data[field]
                if isinstance(value, list) and len(value) > 0:
                    return True

        return False


    def _has_followup(self, response_data: Dict[str, Any]) -> bool:
        """
        Check if response has follow-up questions.

        Args:
            response_data: Response data dictionary

        Returns:
            True if follow-up questions present, False otherwise
        """

        response_text = response_data.get('response', '')
        if isinstance(response_text, str):
            response_lower = normalize_text(response_text)

            # Look for follow-up phrases
            if contains_any_keyword(response_text, FOLLOWUP_PHRASES):
                return True

            # Count question marks (more than 0 likely means follow-up)
            if response_text.count('?') > 0:
                return True

        # Check for explicit followup fields
        if 'suggestedQuestions' in response_data or 'suggested_questions' in response_data:
            questions = response_data.get('suggestedQuestions') or response_data.get('suggested_questions')
            if questions and len(questions) > 0:
                return True

        return False


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

# Create global classifier instance for convenience
_tag_classifier_instance: Optional[TagClassifier] = None


def get_tag_classifier() -> TagClassifier:
    """
    Get the global TagClassifier instance (singleton pattern).

    Returns:
        TagClassifier instance
    """
    global _tag_classifier_instance
    if _tag_classifier_instance is None:
        _tag_classifier_instance = TagClassifier()
    return _tag_classifier_instance


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def classify_response(
    user_input: str,
    response_data: Dict[str, Any],
    workflow_type: Optional[str] = None,
    routing_metadata: Optional[Dict[str, Any]] = None
) -> ResponseTags:
    """
    Classify response and generate tags.

    Convenience function that uses the global classifier instance.

    Args:
        user_input: Original user input/question
        response_data: Response data from workflow/API
        workflow_type: Workflow type (if known)
        routing_metadata: Routing decision metadata

    Returns:
        ResponseTags object

    Example:
        >>> from tags import classify_response
        >>> tags = classify_response(
        ...     user_input="Need pressure transmitter",
        ...     response_data={"ranked_results": [...]}
        ... )
        >>> tags.intent_type.value
        'product_search'
    """
    classifier = get_tag_classifier()
    return classifier.classify(
        user_input, response_data, workflow_type, routing_metadata
    )
