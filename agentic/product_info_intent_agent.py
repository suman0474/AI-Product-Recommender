"""
Product Info Intent Classification Agent

Classifies user queries to route to appropriate data source:
- Index RAG (product models, specs, datasheets)
- Standards RAG (IEC, ISO, API standards, SIL, ATEX)
- Strategy RAG (vendor priorities, procurement, approved suppliers)
- Deep Agent (detailed spec extraction from tables/clauses)

Routes queries to Product Info page in frontend when relevant.
"""

import logging
import re
from typing import Dict, Tuple, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Available data sources for Product Info."""
    INDEX_RAG = "index_rag"         # Product database + web search
    STANDARDS_RAG = "standards_rag" # IEC, ISO, API standards
    STRATEGY_RAG = "strategy_rag"   # Vendor procurement strategies
    DEEP_AGENT = "deep_agent"       # Detailed spec extraction
    WEB_SEARCH = "web_search"       # Web search with verification (parallel source)
    HYBRID = "hybrid"               # Multiple sources
    LLM = "llm"                     # General LLM fallback


# =============================================================================
# ENHANCED KEYWORD LISTS (Based on Debugs.md Test Cases)
# =============================================================================

# 1. Index RAG - Product Specifications
INDEX_RAG_KEYWORDS = [
    # Product types (instruments)
    "transmitter", "sensor", "valve", "actuator", "controller",
    "flowmeter", "flow meter", "coriolis", "magnetic flowmeter",
    "level transmitter", "level sensor", "level gauge",
    "pressure transmitter", "pressure sensor", "pressure gauge",
    "temperature sensor", "temperature transmitter", "thermocouple", "rtd",
    "analyzer", "ph analyzer", "conductivity", "oxygen analyzer",
    "positioner", "valve positioner", "indicator", "recorder",
    "gauge", "switch", "limit switch", "proximity switch",
    "control valve", "safety valve", "relief valve", "shutoff valve",
    "actuator", "pneumatic actuator", "electric actuator",

    # Brand names (major vendors)
    "rosemount", "yokogawa", "emerson", "honeywell", "siemens",
    "endress", "endress+hauser", "e+h", "abb", "foxboro",
    "fisher", "masoneilan", "metso", "neles", "samson",
    "krohne", "vega", "ifm", "pepperl", "turck",
    "danfoss", "burkert", "asco", "parker", "swagelok",

    # Product identifiers and series
    "3051", "3051s", "ejx", "ejx series", "ejx110", "ejx310", "ejx510",
    "644", "3144", "5400", "5300", "2088", "3100",
    "dvc6200", "fieldvue", "fisher ez", "ez valve",

    # General product queries
    "model", "product", "specification", "specifications", "specs",
    "datasheet", "data sheet", "catalog", "catalogue",
    "features", "capabilities", "performance", "accuracy", "range",
    "accuracy range", "measurement range", "flow characteristics",

    # Query patterns
    "what is", "what are", "tell me about", "describe", "find",
    "show me", "get me", "information about", "details of",
    "how does", "how do"
]

# 2. Standards RAG - Compliance & Safety
STANDARDS_RAG_KEYWORDS = [
    # Standards organizations
    "iec", "iso", "api", "asme", "astm", "nfpa", "ansi", "din", "en",

    # Specific standard numbers
    "61508", "61511", "62443", "61131", "60079",
    "526", "api 526", "api 520", "api 521",
    "isa", "isa-84", "isa-88", "isa-95",

    # Safety Integrity Levels
    "sil", "sil 1", "sil 2", "sil 3", "sil 4",
    "sil-1", "sil-2", "sil-3", "sil-4",
    "sil1", "sil2", "sil3", "sil4",
    "functional safety", "safety integrity", "safety integrity level",
    "safety instrumented", "sis", "sif", "safety function",

    # Hazardous Area Classifications
    "atex", "iecex", "explosion", "explosion-proof", "explosionproof",
    "hazardous area", "hazardous areas", "hazardous zone",
    "zone 0", "zone 1", "zone 2", "zone 20", "zone 21", "zone 22",
    "class i", "class ii", "class iii",
    "division 1", "division 2", "div 1", "div 2",
    "intrinsic safety", "intrinsically safe", "flameproof",
    "increased safety", "encapsulation", "pressurization",
    "ex d", "ex e", "ex i", "ex ia", "ex ib", "ex ic",

    # Compliance & Certification
    "certification", "certified", "compliance", "compliant",
    "standard", "standards", "regulation", "regulations",
    "requirement", "requirements", "conformance", "approval",
    "installation requirement", "installation requirements",

    # Safety-related queries
    "difference between", "compare", "explain",
    "what is the requirement", "according to"
]

# 3. Strategy RAG - Vendor & Procurement
STRATEGY_RAG_KEYWORDS = [
    # Vendor-related
    "vendor", "vendors", "supplier", "suppliers",
    "manufacturer", "manufacturers", "make", "makes",
    "who manufactures", "who makes", "who supplies",

    # Preference & Approval
    "preferred", "preferred vendor", "preferred supplier",
    "approved", "approved vendor", "approved supplier", "approved suppliers",
    "strategic", "strategic partner", "partner",
    "recommended", "recommended vendor",
    "best vendor", "top vendor",

    # Procurement Strategy
    "procurement", "procurement strategy", "sourcing", "sourcing strategy",
    "selection", "vendor selection", "supplier selection",
    "priority", "priorities", "prioritize",
    "strategy", "strategies",

    # Commercial Terms
    "cost", "price", "pricing", "budget",
    "lead time", "delivery", "delivery time",
    "support", "service", "warranty",
    "relationship", "long-term", "contract",
    "compare suppliers", "compare vendors",

    # Specific procurement queries
    "who is our", "who are our", "which vendor", "which supplier",
    "do they also", "do they make"
]

# 4. Deep Agent - Detailed Extraction
DEEP_AGENT_KEYWORDS = [
    # Extraction commands
    "extract", "extraction", "pull out", "get from",
    "from the standard", "from the table", "from the document",
    "according to standard", "according to the standard",
    "as per standard", "as per the standard",

    # Specific document references
    "table", "tables", "figure", "figures",
    "section", "sections", "clause", "clauses",
    "annex", "annexes", "appendix", "appendices",
    "paragraph", "page",

    # Detailed requirements
    "specific requirement", "specific requirements",
    "detailed specs", "detailed specifications",
    "technical specification", "technical specifications",
    "parameter from", "value from", "limit from",
    "what does the standard say", "what is the specific",

    # Data extraction patterns
    "pressure limits", "temperature limits", "response time requirement",
    "material requirements", "dimensional requirements",
    "tolerance", "tolerances", "allowable", "permissible"
]

# 5. Web Search - External/Current Information
WEB_SEARCH_KEYWORDS = [
    # Recency indicators
    "latest", "recent", "current", "newest", "updated",
    "new", "news", "update", "updates", "announcement",

    # Market/Industry
    "market", "market trend", "industry trend", "industry news",
    "competitor", "competition", "alternative", "alternatives",
    "comparison", "versus", "vs",

    # External information
    "external", "outside", "beyond", "internet",
    "online", "website", "web", "search",

    # General queries not in database
    "general information", "overview", "introduction",
    "what is happening", "what's new", "developments",

    # Price/availability (often needs current web data)
    "price", "pricing", "cost", "availability",
    "where to buy", "purchase", "order",

    # Reviews and opinions
    "review", "reviews", "opinion", "opinions",
    "feedback", "experience", "experiences"
]


# =============================================================================
# PATTERN-BASED CLASSIFICATION (Query Structure Recognition)
# =============================================================================

# Patterns that strongly indicate Index RAG (product queries)
INDEX_RAG_PATTERNS = [
    r"what (is|are) the (specifications?|specs|features|capabilities) (for|of)",
    r"tell me about (the )?[a-z0-9\-\s]+ (transmitter|sensor|valve|meter|analyzer)",
    r"find (the |a )?(datasheet|catalog|specs?) for",
    r"(rosemount|yokogawa|emerson|honeywell|siemens|fisher|abb)\s+[a-z0-9\-]+",
    r"[a-z]+ (series|model|type)\s+[a-z0-9\-]+",
    r"accuracy (range|of|for)",
    r"flow characteristics",
    r"(ejx|3051|644|dvc)\s*[a-z0-9]*"
]

# Patterns that strongly indicate Standards RAG (compliance queries)
STANDARDS_RAG_PATTERNS = [
    r"(installation|safety|certification) requirements? (for|according|per)",
    r"(what|explain).*(difference|comparison) between sil",
    r"(iec|iso|api|atex|iecex)\s*[0-9]+",
    r"sil[- ]?[1-4]",
    r"zone [0-2]",
    r"(hazardous|explosive) area",
    r"(according to|per|as per) (iec|iso|api|atex)",
    r"(certified|compliant|approval) for",
    r"standard requirements?"
]

# Patterns that strongly indicate Strategy RAG (vendor queries)
STRATEGY_RAG_PATTERNS = [
    r"(who is|who are) (our )?(preferred|approved|strategic)",
    r"(preferred|approved|recommended) (vendor|supplier|manufacturer)",
    r"(preferred|approved) vendor for",  # "preferred vendor for control valves"
    r"(our )?(preferred|approved) (vendor|supplier) for [a-z\s]+",
    r"procurement strategy",
    r"(compare|comparison of) (the )?(approved )?(suppliers|vendors)",
    r"long lead time",
    r"who (manufactures|makes|supplies)",
    r"do they (also )?(make|manufacture|supply)",
    r"which (vendor|supplier) (should we|do we|to) use"
]

# Patterns that strongly indicate Deep Agent (extraction queries)
DEEP_AGENT_PATTERNS = [
    r"extract (the )?[a-z\s]+ from (the )?(standard|table|document)",
    r"(pressure|temperature|flow) limits? (for|from|in)",
    r"(specific|detailed) (requirement|specification) (for|in)",
    r"(clause|section|annex|table) [0-9]+",
    r"response time requirement",
    r"what does the standard say about"
]

# Patterns that strongly indicate Web Search (external/current info)
WEB_SEARCH_PATTERNS = [
    r"(latest|recent|current|newest) (news|update|development|trend)",
    r"(market|industry) (trend|news|update)",
    r"(competitor|alternative|comparison).*(product|vendor|supplier)",
    r"(price|pricing|cost|availability) (for|of)",
    r"where (to|can i) (buy|purchase|order)",
    r"(review|feedback|experience) (of|for|with)",
    r"what('s| is) (new|happening|trending)",
    r"(search|find).*(online|web|internet)"
]

# Patterns indicating hybrid queries (multiple systems needed)
# These are queries that GENUINELY need data from multiple RAG systems
HYBRID_PATTERNS = [
    # Product + Standards (e.g., "Is Rosemount 3051S certified for SIL 3?")
    # Asking about a specific product's certification status
    r"(rosemount|yokogawa|emerson|fisher|abb)\s+[a-z0-9\-]+.*(sil|atex|zone|certified|hazardous)",
    r"(sil|atex|zone|certified|hazardous).*(rosemount|yokogawa|emerson|fisher|abb)",
    r"is (the |a )?(rosemount|yokogawa|emerson|fisher|abb).*certified",

    # Vendor + Standards (e.g., "Which preferred vendor supplies for Zone 0?")
    # Asking about vendors that meet specific safety/certification requirements
    r"(preferred|approved) (vendor|supplier).*(suitable for|certified for|rated for).*(sil|atex|zone|hazardous)",
    r"(vendor|supplier).*(suitable for|certified for).*(zone|sil|atex|hazardous)",
    r"(sil|atex|zone|hazardous).*(preferred|approved) (vendor|supplier)",
    r"which.*(vendor|supplier).*supplies.*(zone|sil|atex|hazardous)",

    # Specific combined queries
    r"(coriolis|magnetic|vortex) meter.*(zone|hazardous|sil|atex)",
    r"(zone|hazardous|sil|atex).*(coriolis|magnetic|vortex) meter"
]


# =============================================================================
# MAIN CLASSIFICATION FUNCTION
# =============================================================================

def classify_query(query: str) -> Tuple[DataSource, float, str]:
    """
    Classify a query to determine the best data source.

    Uses a multi-stage classification approach:
    1. Pattern matching for strong structural indicators
    2. Keyword scoring with weights
    3. Hybrid detection for multi-domain queries

    Args:
        query: User query string

    Returns:
        Tuple of (DataSource, confidence, reasoning)
    """
    query_lower = query.lower().strip()

    # Stage 1: Pattern-based classification (highest priority)
    pattern_result = _classify_by_patterns(query_lower)
    if pattern_result:
        source, confidence, reasoning = pattern_result
        logger.info(f"[INTENT] Pattern match: {source.value} (confidence: {confidence:.2f})")
        return source, confidence, reasoning

    # Stage 2: Keyword-based scoring
    scores, matches = _calculate_keyword_scores(query_lower)

    # Stage 3: Determine primary source
    primary_source = DataSource.LLM
    max_score = 0.0

    for source, score in scores.items():
        if score > max_score:
            max_score = score
            primary_source = source

    # Stage 4: Calculate confidence
    total_score = sum(scores.values())
    if total_score > 0:
        confidence = min(max_score / total_score + 0.3, 1.0)  # Base confidence boost
    else:
        confidence = 0.3

    # Stage 5: Check for hybrid queries
    # Only consider hybrid if explicitly detected by pattern matching
    # Simple keyword overlap doesn't make a query hybrid
    hybrid_detected = _detect_hybrid_pattern(query_lower)

    if hybrid_detected:
        # Get all sources with significant scores
        high_score_sources = [
            s for s, score in scores.items()
            if score >= max_score * 0.3 and score > 0
        ]
        primary_source = DataSource.HYBRID
        confidence = 0.85
        involved = [s.value for s in high_score_sources]
        reasoning = f"Hybrid query involving: {', '.join(involved)}"
        logger.info(f"[INTENT] Hybrid detected: {reasoning} (confidence: {confidence:.2f})")
        return primary_source, confidence, reasoning

    # Build reasoning
    if primary_source != DataSource.LLM and matches.get(primary_source):
        matched_keywords = matches[primary_source][:5]
        reasoning = f"Matched keywords: {', '.join(matched_keywords)}"
    else:
        reasoning = "No specific keywords matched, using LLM fallback"

    logger.info(f"[INTENT] Query classified as {primary_source.value} (confidence: {confidence:.2f})")
    return primary_source, confidence, reasoning


def _classify_by_patterns(query_lower: str) -> Optional[Tuple[DataSource, float, str]]:
    """
    Classify query using regex patterns for strong structural indicators.
    Returns None if no strong pattern match found.
    """
    # Check for hybrid patterns first (highest specificity)
    for pattern in HYBRID_PATTERNS:
        if re.search(pattern, query_lower):
            return DataSource.HYBRID, 0.9, f"Hybrid pattern matched: {pattern[:50]}..."

    # Check Deep Agent patterns (specific extraction requests)
    for pattern in DEEP_AGENT_PATTERNS:
        if re.search(pattern, query_lower):
            return DataSource.DEEP_AGENT, 0.9, f"Deep Agent pattern matched"

    # Check Web Search patterns (external/current info)
    for pattern in WEB_SEARCH_PATTERNS:
        if re.search(pattern, query_lower):
            return DataSource.WEB_SEARCH, 0.85, f"Web Search pattern matched"

    # Check Standards RAG patterns
    for pattern in STANDARDS_RAG_PATTERNS:
        if re.search(pattern, query_lower):
            return DataSource.STANDARDS_RAG, 0.9, f"Standards pattern matched"

    # Check Strategy RAG patterns
    for pattern in STRATEGY_RAG_PATTERNS:
        if re.search(pattern, query_lower):
            return DataSource.STRATEGY_RAG, 0.9, f"Strategy pattern matched"

    # Check Index RAG patterns
    for pattern in INDEX_RAG_PATTERNS:
        if re.search(pattern, query_lower):
            return DataSource.INDEX_RAG, 0.85, f"Product pattern matched"

    return None


def _calculate_keyword_scores(query_lower: str) -> Tuple[Dict[DataSource, float], Dict[DataSource, List[str]]]:
    """
    Calculate weighted keyword scores for each data source.
    """
    scores = {
        DataSource.INDEX_RAG: 0.0,
        DataSource.STANDARDS_RAG: 0.0,
        DataSource.STRATEGY_RAG: 0.0,
        DataSource.DEEP_AGENT: 0.0,
        DataSource.WEB_SEARCH: 0.0
    }

    matches = {
        DataSource.INDEX_RAG: [],
        DataSource.STANDARDS_RAG: [],
        DataSource.STRATEGY_RAG: [],
        DataSource.DEEP_AGENT: [],
        DataSource.WEB_SEARCH: []
    }

    # Score Index RAG keywords
    for kw in INDEX_RAG_KEYWORDS:
        if kw in query_lower:
            # Higher weight for specific product models/brands
            if re.match(r'^(rosemount|yokogawa|emerson|fisher|abb|3051|ejx)', kw):
                scores[DataSource.INDEX_RAG] += 2.0
            else:
                scores[DataSource.INDEX_RAG] += 1.0
            matches[DataSource.INDEX_RAG].append(kw)

    # Score Standards RAG keywords (higher weight for explicit standard codes)
    for kw in STANDARDS_RAG_KEYWORDS:
        if kw in query_lower:
            # Highest weight for standard codes and SIL levels
            if re.match(r'^(iec|iso|api|sil|atex|61508|61511|526)', kw):
                scores[DataSource.STANDARDS_RAG] += 3.0
            elif re.match(r'^(zone|hazardous|certification|compliance)', kw):
                scores[DataSource.STANDARDS_RAG] += 2.0
            else:
                scores[DataSource.STANDARDS_RAG] += 1.0
            matches[DataSource.STANDARDS_RAG].append(kw)

    # Score Strategy RAG keywords
    for kw in STRATEGY_RAG_KEYWORDS:
        if kw in query_lower:
            # Higher weight for explicit vendor/procurement terms
            if re.match(r'^(preferred|approved|procurement|who manufactures|who makes)', kw):
                scores[DataSource.STRATEGY_RAG] += 2.5
            else:
                scores[DataSource.STRATEGY_RAG] += 1.0
            matches[DataSource.STRATEGY_RAG].append(kw)

    # Score Deep Agent keywords
    for kw in DEEP_AGENT_KEYWORDS:
        if kw in query_lower:
            # Higher weight for explicit extraction commands
            if re.match(r'^(extract|clause|section|table|annex)', kw):
                scores[DataSource.DEEP_AGENT] += 2.5
            else:
                scores[DataSource.DEEP_AGENT] += 1.0
            matches[DataSource.DEEP_AGENT].append(kw)

    # Score Web Search keywords
    for kw in WEB_SEARCH_KEYWORDS:
        if kw in query_lower:
            # Higher weight for recency and market terms
            if re.match(r'^(latest|recent|current|market|industry|competitor|price)', kw):
                scores[DataSource.WEB_SEARCH] += 2.5
            else:
                scores[DataSource.WEB_SEARCH] += 1.0
            matches[DataSource.WEB_SEARCH].append(kw)

    return scores, matches


def _detect_hybrid_pattern(query_lower: str) -> bool:
    """
    Detect if query requires multiple data sources (hybrid).

    Returns True only when the query genuinely requires combining data
    from multiple sources (e.g., product specs + certification status).
    """
    # Check explicit hybrid patterns (high confidence patterns)
    for pattern in HYBRID_PATTERNS:
        if re.search(pattern, query_lower):
            return True

    # Check for product + standards combination
    # This is hybrid when asking about a specific product's certification
    has_specific_product = any(kw in query_lower for kw in ["rosemount", "yokogawa", "emerson", "fisher", "3051", "ejx"])
    has_standards = any(kw in query_lower for kw in ["sil", "atex", "zone", "certified", "iec", "hazardous"])
    if has_specific_product and has_standards:
        return True

    # Check for vendor + standards combination (but NOT just "vendor for X product")
    # Only hybrid if asking about vendor capabilities in hazardous/certified context
    has_vendor = any(kw in query_lower for kw in ["vendor", "supplier", "preferred", "approved"])
    if has_vendor and has_standards:
        # Additional check: is this truly asking about standards, not just mentioning a product?
        asking_about_capability = any(kw in query_lower for kw in ["suitable for", "certified for", "rated for", "approved for"])
        if asking_about_capability:
            return True

    return False


# =============================================================================
# HYBRID QUERY HANDLING
# =============================================================================

def get_sources_for_hybrid(query: str) -> List[DataSource]:
    """
    Get list of sources to query for hybrid requests.

    Returns sources that have relevance to the query, ordered by relevance.
    """
    query_lower = query.lower().strip()
    sources = []
    source_scores = {}

    # Calculate relevance score for each source
    scores, _ = _calculate_keyword_scores(query_lower)

    for source, score in scores.items():
        if score > 0:
            sources.append(source)
            source_scores[source] = score

    # Sort by score (highest first)
    sources.sort(key=lambda s: source_scores.get(s, 0), reverse=True)

    # Ensure at least Index RAG is included as fallback
    if not sources:
        sources = [DataSource.INDEX_RAG]

    return sources


# =============================================================================
# PRODUCT INFO PAGE ROUTING (Frontend Integration)
# =============================================================================

def is_product_info_intent(query: str) -> Tuple[bool, float]:
    """
    Determine if query should be routed to Product Info page in frontend.

    This function is the primary gate for deciding whether to open
    the Product Info page as a new window/tab in the frontend.

    Returns:
        Tuple of (should_route_to_product_info, confidence)
    """
    query_lower = query.lower().strip()

    # Classify the query first
    data_source, classification_confidence, _ = classify_query(query_lower)

    # All RAG sources and Deep Agent should route to Product Info page
    product_info_sources = [
        DataSource.INDEX_RAG,
        DataSource.STANDARDS_RAG,
        DataSource.STRATEGY_RAG,
        DataSource.DEEP_AGENT,
        DataSource.HYBRID
    ]

    if data_source in product_info_sources:
        # High confidence - route to Product Info
        return True, classification_confidence

    # Additional pattern checks for edge cases
    product_info_patterns = [
        # Product specification queries
        r"what (is|are) the (specifications?|specs|features)",
        r"tell me about.*(transmitter|sensor|valve|meter)",
        r"(datasheet|catalog|specs?) for",

        # Standards queries
        r"(iec|iso|api|sil|atex)",
        r"(hazardous|explosive) area",
        r"(installation|safety) requirement",

        # Vendor queries
        r"(preferred|approved|recommended) (vendor|supplier)",
        r"who (manufactures|makes|supplies)",
        r"procurement strategy",

        # Extraction queries
        r"extract.*(from|the)",
        r"(clause|section|table|annex)",

        # Brand/product mentions
        r"(rosemount|yokogawa|emerson|honeywell|siemens|fisher|abb)",
        r"(transmitter|sensor|valve|flowmeter|analyzer|positioner)",

        # Comparison queries
        r"compare.*(vendor|supplier|product)",
        r"difference between"
    ]

    pattern_matches = 0
    for pattern in product_info_patterns:
        if re.search(pattern, query_lower):
            pattern_matches += 1

    # If multiple patterns match, likely Product Info
    if pattern_matches >= 2:
        return True, min(0.7 + (pattern_matches * 0.1), 0.95)
    elif pattern_matches == 1:
        return True, 0.65

    # Default: not a Product Info query
    return False, 0.3


def get_product_info_route_decision(query: str) -> Dict:
    """
    Get detailed routing decision for Product Info page.

    Returns a structured decision object for frontend routing.

    Returns:
        Dict with routing decision details:
        - should_route: bool - whether to open Product Info page
        - confidence: float - confidence in the decision
        - data_source: str - primary data source to query
        - sources: List[str] - all relevant sources for hybrid
        - reasoning: str - explanation for the decision
    """
    query_lower = query.lower().strip()

    # Get classification
    data_source, confidence, reasoning = classify_query(query)

    # Determine if should route to Product Info
    should_route, route_confidence = is_product_info_intent(query)

    # Get sources for hybrid queries
    if data_source == DataSource.HYBRID:
        sources = [s.value for s in get_sources_for_hybrid(query)]
    else:
        sources = [data_source.value] if data_source != DataSource.LLM else []

    return {
        "should_route": should_route,
        "confidence": max(confidence, route_confidence),
        "data_source": data_source.value,
        "sources": sources,
        "reasoning": reasoning,
        "query_type": _get_query_type_description(data_source)
    }


def _get_query_type_description(data_source: DataSource) -> str:
    """Get human-readable description of query type."""
    descriptions = {
        DataSource.INDEX_RAG: "Product Specifications Query",
        DataSource.STANDARDS_RAG: "Standards & Compliance Query",
        DataSource.STRATEGY_RAG: "Vendor & Procurement Query",
        DataSource.DEEP_AGENT: "Detailed Extraction Query",
        DataSource.HYBRID: "Multi-Domain Query",
        DataSource.LLM: "General Query"
    }
    return descriptions.get(data_source, "Unknown Query Type")


# =============================================================================
# FOLLOW-UP QUERY DETECTION (Memory Support)
# =============================================================================

# Pronouns and references that indicate follow-up queries
FOLLOW_UP_INDICATORS = [
    "it", "its", "they", "their", "them", "this", "that", "these", "those",
    "the same", "same one", "that one", "the one",
    "also", "too", "as well", "in addition",
    "more about", "more details", "tell me more",
    "what about", "how about", "and what",
    "do they", "can they", "does it", "can it"
]


def is_follow_up_query(query: str) -> bool:
    """
    Detect if query is a follow-up that requires context resolution.

    Returns True if query contains pronouns or references that need
    to be resolved from conversation history.
    """
    query_lower = query.lower().strip()

    # Check for follow-up indicators
    for indicator in FOLLOW_UP_INDICATORS:
        if indicator in query_lower:
            # Make sure it's at word boundary
            pattern = r'\b' + re.escape(indicator) + r'\b'
            if re.search(pattern, query_lower):
                return True

    # Check for question starting with follow-up patterns
    follow_up_starts = [
        r"^what about",
        r"^how about",
        r"^and (what|how|does|do|can|is)",
        r"^does it",
        r"^do they",
        r"^can it",
        r"^is it",
        r"^are they"
    ]

    for pattern in follow_up_starts:
        if re.search(pattern, query_lower):
            return True

    return False


# =============================================================================
# TESTING & VALIDATION
# =============================================================================

def test_classification(queries: List[str] = None) -> List[Dict]:
    """
    Test classification against sample queries from Debugs.md.

    Returns list of test results.
    """
    if queries is None:
        # Default test queries from Debugs.md
        queries = [
            # Index RAG tests
            ("What are the specifications for the Rosemount 3051S pressure transmitter?", DataSource.INDEX_RAG),
            ("Tell me about the accuracy range of the Yokogawa EJX series.", DataSource.INDEX_RAG),
            ("Find the datasheet for an Emerson magnetic flowmeter.", DataSource.INDEX_RAG),

            # Standards RAG tests
            ("What are the installation requirements for Zone 1 hazardous areas according to ATEX?", DataSource.STANDARDS_RAG),
            ("Explain the difference between SIL 2 and SIL 3 regarding IEC 61508.", DataSource.STANDARDS_RAG),
            ("What are the API 526 standard requirements for safety valves?", DataSource.STANDARDS_RAG),

            # Strategy RAG tests
            ("Who is our preferred vendor for control valves?", DataSource.STRATEGY_RAG),
            ("What is the procurement strategy for long lead time instruments?", DataSource.STRATEGY_RAG),
            ("Compare the approved suppliers for pressure transmitters.", DataSource.STRATEGY_RAG),

            # Deep Agent tests
            ("Extract the pressure limits for carbon steel flanges from the standard table.", DataSource.DEEP_AGENT),
            ("What is the specific response time requirement in the technical specification for emergency shutoff valves?", DataSource.DEEP_AGENT),

            # Hybrid tests
            ("Is the Rosemount 3051S certified for use in SIL 3 applications?", DataSource.HYBRID),
            ("Which preferred vendor supplies coriolis meters suitable for hazardous Zone 0 areas?", DataSource.HYBRID),

            # Follow-up tests (should detect as follow-up)
            ("Who manufactures the Fisher EZ valve?", DataSource.STRATEGY_RAG),
            ("What are its flow characteristics?", DataSource.INDEX_RAG),  # Follow-up
            ("Do they also make positioners?", DataSource.STRATEGY_RAG),  # Follow-up
        ]

    results = []
    for item in queries:
        if isinstance(item, tuple):
            query, expected = item
        else:
            query, expected = item, None

        source, confidence, reasoning = classify_query(query)
        should_route, route_confidence = is_product_info_intent(query)
        is_followup = is_follow_up_query(query)

        result = {
            "query": query,
            "expected": expected.value if expected else None,
            "actual": source.value,
            "confidence": confidence,
            "reasoning": reasoning,
            "should_route_to_product_info": should_route,
            "route_confidence": route_confidence,
            "is_follow_up": is_followup,
            "passed": expected is None or source == expected
        }
        results.append(result)

        # Log result
        status = "PASS" if result["passed"] else "FAIL"
        logger.info(f"[TEST] {status}: '{query[:50]}...' -> {source.value} (expected: {expected.value if expected else 'N/A'})")

    return results


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'DataSource',
    'classify_query',
    'get_sources_for_hybrid',
    'is_product_info_intent',
    'get_product_info_route_decision',
    'is_follow_up_query',
    'test_classification',
    # Keyword lists (for external inspection)
    'INDEX_RAG_KEYWORDS',
    'STANDARDS_RAG_KEYWORDS',
    'STRATEGY_RAG_KEYWORDS',
    'DEEP_AGENT_KEYWORDS',
    'WEB_SEARCH_KEYWORDS'
]


# =============================================================================
# MAIN (Testing)
# =============================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("INTENT CLASSIFICATION ROUTING AGENT - TEST SUITE")
    print("=" * 80)

    results = test_classification()

    passed = sum(1 for r in results if r["passed"])
    total = len(results)

    print("\n" + "=" * 80)
    print(f"RESULTS: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("=" * 80)

    # Show failures
    failures = [r for r in results if not r["passed"]]
    if failures:
        print("\nFAILED TESTS:")
        for f in failures:
            print(f"  - Query: {f['query'][:60]}...")
            print(f"    Expected: {f['expected']}, Got: {f['actual']}")
            print(f"    Reasoning: {f['reasoning']}")
            print()
