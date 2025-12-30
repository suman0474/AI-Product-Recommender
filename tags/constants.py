# tags/constants.py
"""
Constants for tag classification logic.

This module contains keyword patterns and phrases used by the TagClassifier
to automatically determine intent_type, response_status, and content_flags.
"""

from typing import List, Set


# ============================================================================
# INTENT TYPE PATTERNS
# ============================================================================

# Keywords indicating SOLUTION intent
SOLUTION_KEYWORDS: List[str] = [
    # Design-related
    "design", "instrumentation for", "project", "system", "unit",
    "crude distillation", "crude unit", "refinery", "plant",
    "facility", "installation",

    # Multi-item indicators
    "pump", "pumps", "multiple instruments", "several", "various",
    "instrumentation list", "BOM", "bill of materials",

    # Project-related
    "needs", "requires", "project needs", "system requires",
    "equipment for", "instruments for", "accessories for",

    # Quantity indicators (strong signal for solution)
    "5 pressure", "3 control", "10 flow", "qty", "quantity",

    # Process units
    "distillation", "separation", "reactor", "heat exchanger",
    "compressor", "pipeline", "tank", "vessel"
]

# Keywords indicating PRODUCT_SEARCH intent
PRODUCT_SEARCH_KEYWORDS: List[str] = [
    # Search intent
    "need", "need a", "looking for", "find", "find a", "search for",
    "select", "choose", "recommend", "suggest",

    # Specific product types (singular)
    "transmitter", "valve", "meter", "sensor", "gauge",
    "controller", "actuator", "positioner", "switch",

    # Specifications (strong signal for product search)
    "SIL", "SIL2", "SIL3", "psi", "bar", "mbar",
    "4-20mA", "HART", "foundation fieldbus", "profibus",
    "0-500", "0-100", "range", "accuracy",

    # Certifications
    "ATEX", "IECEx", "FM approved", "CSA", "API",

    # Application details
    "for crude oil", "for steam", "for gas", "for water",
    "specific product", "particular model"
]

# Keywords indicating PRODUCT_INFO intent
PRODUCT_INFO_KEYWORDS: List[str] = [
    # Information requests
    "what is", "what are", "what's", "whats",
    "temperature rating", "temperature rating of",
    "outer case color", "case color", "color of",
    "specifications of", "specs of", "details about",
    "information on", "info on", "tell me about",
    "describe", "explain about",

    # Specific models (strong signal for product_info)
    "ST800", "ST700", "ST3000", "STG", "STD",
    "3051", "3051S", "3051CD", "3051L",
    "SMV", "SMV800", "SMV3000",
    "EJX", "EJX910", "EJX530",
    "266", "266DST", "266HST",

    # Vendor + model pattern
    "Honeywell ST", "Emerson 3051", "ABB 266",
    "Yokogawa EJX", "Siemens SITRANS",

    # Property queries
    "rating of", "range of", "accuracy of",
    "dimensions of", "weight of", "material of",
    "certification of", "compliance of"
]

# Keywords indicating CHAT intent
CHAT_KEYWORDS: List[str] = [
    # Explanation requests
    "explain", "explain what", "how do", "how does",
    "what does", "what is", "what are",

    # Concepts
    "SIL2 mean", "SIL3 mean", "ATEX mean",
    "difference between", "compare", "versus", "vs",
    "tell me about", "teach me",

    # Technical concepts
    "DP transmitters work", "DP transmitter",
    "differential pressure", "gauge pressure",
    "absolute pressure", "hydrostatic level",

    # General questions
    "why", "when", "where", "which",
    "how to", "best practice", "recommendation"
]

# Keywords indicating INVALID intent
INVALID_KEYWORDS: List[str] = [
    # Politics
    "president", "election", "government", "congress",
    "senate", "politics", "political",

    # Entertainment
    "weather", "sports", "movie", "film", "music",
    "celebrity", "actor", "actress",

    # Food
    "recipe", "cooking", "restaurant", "food",

    # General knowledge (non-technical)
    "capital of", "population of", "history of",

    # Off-topic
    "outside domain", "unrelated", "not about",
    "joke", "funny", "meme"
]


# ============================================================================
# RESPONSE STATUS PATTERNS
# ============================================================================

# Phrases indicating INCOMPLETE response
INCOMPLETE_PHRASES: List[str] = [
    # Direct requests for information
    "please provide", "please specify", "please clarify",
    "need more information", "need additional information",
    "could you specify", "could you provide",
    "can you provide", "can you specify",

    # Questions about missing info
    "what is the", "what are the",
    "which", "do you need", "do you require",

    # Explicit gaps
    "missing", "required", "needed", "necessary",
    "incomplete", "partial", "not enough"
]

# Phrases indicating follow-up questions
FOLLOWUP_PHRASES: List[str] = [
    # Question markers
    "?",

    # Polite requests
    "please provide", "please specify", "could you",
    "would you", "can you", "may I ask",

    # Information gathering
    "need to know", "clarify", "confirm",
    "tell me", "let me know",

    # Multiple questions
    "1.", "2.", "3.",  # Numbered list of questions
    "first", "second", "third",
    "also", "additionally", "furthermore"
]


# ============================================================================
# CONTENT DETECTION PATTERNS
# ============================================================================

# Fields indicating instruments are present (check in response_data keys)
INSTRUMENT_FIELDS: Set[str] = {
    # Direct fields
    "instruments",
    "identified_instruments",
    "instrument_list",
    "instrumentation",

    # Specific instrument types
    "pressure_transmitters",
    "flow_meters",
    "control_valves",
    "level_transmitters",
    "temperature_transmitters",
    "positioners",
    "actuators",
    "switches",
    "controllers"
}

# Fields indicating accessories are present (check in response_data keys)
ACCESSORY_FIELDS: Set[str] = {
    # Direct fields
    "accessories",
    "identified_accessories",
    "accessory_list",

    # Specific accessory types
    "manifolds",
    "mounting_brackets",
    "cable_glands",
    "junction_boxes",
    "weather_protection",
    "calibration_tools",
    "spare_parts"
}


# ============================================================================
# PRODUCT REFERENCE PATTERNS
# ============================================================================

# Model patterns to detect product references
PRODUCT_MODEL_PATTERNS: List[str] = [
    # Honeywell
    "st800", "st700", "st3000", "stg", "std",
    "sts", "sta", "sth", "stf",

    # Emerson
    "3051", "3051s", "3051cd", "3051l", "3051t",
    "2051", "1151", "rosemount",

    # ABB
    "266", "266dst", "266hst", "266cst",
    "2600t", "2400",

    # Yokogawa
    "ejx", "ejx910", "ejx530", "ejx110",
    "eja", "eja110", "eja430",

    # Siemens
    "sitrans", "sitrans p", "sitrans l",
    "dsiii", "7mf",

    # Endress+Hauser
    "cerabar", "deltapilot", "promag",
    "prowirl", "prosonic"
]

# Vendor names (lowercase for matching)
VENDOR_NAMES: List[str] = [
    "honeywell", "emerson", "rosemount", "abb",
    "yokogawa", "siemens", "endress+hauser", "endress hauser",
    "krohne", "vega", "magnetrol", "omega",
    "dwyer", "wika", "ashcroft", "fuji"
]


# ============================================================================
# WORKFLOW TYPE MAPPINGS
# ============================================================================

# Map workflow types to intent types
WORKFLOW_TO_INTENT_MAP: dict = {
    # Agentic workflows
    "solution": "solution",
    "instrument_detail": "solution",
    "comparison": "product_search",
    "product_search": "product_search",
    "grounded_chat": "chat",  # Default, may be product_info

    # Traditional workflows
    "procurement": "product_search",
    "validation": "product_search",
    "sales": "product_search"
}


# ============================================================================
# CONFIDENCE THRESHOLDS
# ============================================================================

# Minimum confidence for rule-based classification
MIN_CONFIDENCE_RULE_BASED: float = 0.85

# Minimum confidence for keyword-based classification
MIN_CONFIDENCE_KEYWORD: float = 0.70

# High confidence threshold (use primary classification)
HIGH_CONFIDENCE_THRESHOLD: float = 0.90


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def normalize_text(text: str) -> str:
    """
    Normalize text for pattern matching.

    Args:
        text: Input text

    Returns:
        Normalized text (lowercase, stripped)
    """
    return text.lower().strip()


def contains_any_keyword(text: str, keywords: List[str]) -> bool:
    """
    Check if text contains any of the given keywords.

    Args:
        text: Input text
        keywords: List of keywords to search for

    Returns:
        True if any keyword is found, False otherwise
    """
    text_normalized = normalize_text(text)
    return any(keyword.lower() in text_normalized for keyword in keywords)


def contains_all_keywords(text: str, keywords: List[str]) -> bool:
    """
    Check if text contains all of the given keywords.

    Args:
        text: Input text
        keywords: List of keywords to search for

    Returns:
        True if all keywords are found, False otherwise
    """
    text_normalized = normalize_text(text)
    return all(keyword.lower() in text_normalized for keyword in keywords)


def count_keyword_matches(text: str, keywords: List[str]) -> int:
    """
    Count how many keywords are found in text.

    Args:
        text: Input text
        keywords: List of keywords to search for

    Returns:
        Number of keywords found
    """
    text_normalized = normalize_text(text)
    return sum(1 for keyword in keywords if keyword.lower() in text_normalized)
