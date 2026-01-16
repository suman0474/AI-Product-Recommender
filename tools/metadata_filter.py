# tools/metadata_filter.py
# Hierarchical Metadata Pre-Filter for Index RAG
# Narrows down vector search space using structured hierarchy

import logging
import re
from typing import Dict, Any, List, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# ============================================================================
# PRODUCT TYPE PATTERNS
# ============================================================================

PRODUCT_TYPE_PATTERNS = {
    "pressure_transmitter": [
        r"pressure\s*transmitter", r"pressure\s*sensor", r"dp\s*transmitter",
        r"differential\s*pressure", r"gauge\s*pressure", r"absolute\s*pressure"
    ],
    "flow_meter": [
        r"flow\s*meter", r"flowmeter", r"flow\s*sensor", r"mass\s*flow",
        r"coriolis", r"magnetic\s*flow", r"vortex\s*flow", r"ultrasonic\s*flow"
    ],
    "temperature_sensor": [
        r"temperature\s*sensor", r"thermocouple", r"rtd", r"temp\s*transmitter",
        r"temperature\s*transmitter", r"pt100", r"thermometer"
    ],
    "level_transmitter": [
        r"level\s*transmitter", r"level\s*sensor", r"level\s*meter",
        r"radar\s*level", r"ultrasonic\s*level", r"guided\s*wave"
    ],
    "control_valve": [
        r"control\s*valve", r"valve\s*actuator", r"positioner",
        r"globe\s*valve", r"butterfly\s*valve", r"ball\s*valve"
    ],
    "analyzer": [
        r"analyzer", r"gas\s*analyzer", r"ph\s*analyzer", r"oxygen\s*analyzer",
        r"conductivity", r"turbidity"
    ]
}

# Known vendors for quick matching
KNOWN_VENDORS = [
    "yokogawa", "emerson", "rosemount", "honeywell", "siemens", "abb",
    "endress+hauser", "e+h", "krohne", "vega", "pepperl+fuchs", "wika",
    "foxboro", "fisher", "masoneilan", "flowserve", "metso", "samson"
]


# ============================================================================
# METADATA EXTRACTION
# ============================================================================

def extract_product_types(query: str) -> List[str]:
    """
    Extract product types from user query using pattern matching.
    
    Args:
        query: User's search query
        
    Returns:
        List of detected product types
    """
    query_lower = query.lower()
    detected_types = []
    
    for product_type, patterns in PRODUCT_TYPE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, query_lower):
                if product_type not in detected_types:
                    detected_types.append(product_type)
                break
    
    logger.info(f"[MetadataFilter] Extracted product types: {detected_types}")
    return detected_types


def extract_vendors(query: str) -> List[str]:
    """
    Extract vendor names from user query.
    
    Args:
        query: User's search query
        
    Returns:
        List of detected vendor names
    """
    query_lower = query.lower()
    detected_vendors = []
    
    for vendor in KNOWN_VENDORS:
        if vendor in query_lower:
            detected_vendors.append(vendor)
    
    logger.info(f"[MetadataFilter] Extracted vendors: {detected_vendors}")
    return detected_vendors


def extract_models(query: str) -> List[str]:
    """
    Extract model numbers/names from user query.
    
    Common patterns: EJA110E, 3051S, DPharp, SITRANS, etc.
    
    Args:
        query: User's search query
        
    Returns:
        List of detected model identifiers
    """
    # Pattern for alphanumeric model codes
    model_patterns = [
        r'\b([A-Z]{2,4}\d{2,4}[A-Z]?)\b',  # EJA110E, 3051S
        r'\b([A-Z]+\s*\d{4})\b',  # SITRANS 7000
        r'\b(DPharp|SITRANS|Prowirl|Promass|Promag)\b',  # Known model families
    ]
    
    detected_models = []
    for pattern in model_patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        for match in matches:
            if match and match not in detected_models:
                detected_models.append(match.upper())
    
    logger.info(f"[MetadataFilter] Extracted models: {detected_models}")
    return detected_models


# ============================================================================
# AZURE BLOB FILTERING
# ============================================================================

def filter_vendors_by_product(product_types: List[str]) -> List[str]:
    """
    Get vendors that have products matching the specified product types.
    
    Args:
        product_types: List of product types to filter by
        
    Returns:
        List of vendor names
    """
    try:
        from azure_blob_utils import get_vendors_for_product_type
        
        all_vendors = set()
        for product_type in product_types:
            # Convert underscore format to space format
            pt_formatted = product_type.replace("_", " ")
            vendors = get_vendors_for_product_type(pt_formatted)
            all_vendors.update(vendors)
        
        result = sorted(list(all_vendors))
        logger.info(f"[MetadataFilter] Found {len(result)} vendors for product types: {product_types}")
        return result
        
    except Exception as e:
        logger.error(f"[MetadataFilter] Error filtering vendors by product: {e}")
        return []


def filter_models_by_vendor(
    vendors: List[str],
    product_type: str,
    model_hints: List[str] = None
) -> Dict[str, List[Dict]]:
    """
    Get models from specified vendors, optionally filtered by model hints.
    
    Args:
        vendors: List of vendor names
        product_type: Product type for filtering
        model_hints: Optional model identifiers to filter by
        
    Returns:
        Dict mapping vendor -> list of model data
    """
    try:
        from azure_blob_utils import get_products_for_vendors
        
        # Get products for all vendors
        products_data = get_products_for_vendors(vendors, product_type)
        
        # If no model hints, return all
        if not model_hints:
            logger.info(f"[MetadataFilter] Retrieved models for {len(products_data)} vendors (no model filter)")
            return products_data
        
        # Filter by model hints
        filtered_data = {}
        model_hints_lower = [m.lower() for m in model_hints]
        
        for vendor, products in products_data.items():
            matching_products = []
            for product in products:
                # Check if any model hint matches
                product_str = str(product).lower()
                for hint in model_hints_lower:
                    if hint in product_str:
                        matching_products.append(product)
                        break
            
            if matching_products:
                filtered_data[vendor] = matching_products
        
        logger.info(f"[MetadataFilter] Filtered to {len(filtered_data)} vendors with matching models")
        return filtered_data
        
    except Exception as e:
        logger.error(f"[MetadataFilter] Error filtering models: {e}")
        return {}


# ============================================================================
# MAIN FILTER FUNCTION
# ============================================================================

def filter_by_hierarchy(
    query: str,
    top_k: int = 7,
    explicit_product_type: str = None,
    explicit_vendors: List[str] = None
) -> Dict[str, Any]:
    """
    Apply hierarchical metadata filter to narrow search space.
    
    This is a PRE-RETRIEVAL filter that uses structured hierarchy:
    Product Type → Vendor → Model
    
    Args:
        query: User's search query
        top_k: Maximum number of results to return
        explicit_product_type: Override product type extraction
        explicit_vendors: Override vendor extraction
        
    Returns:
        {
            "product_types": [...],
            "vendors": [...],
            "models": {...},  # vendor -> model list
            "total_products": int,
            "filter_applied": bool
        }
    """
    logger.info(f"[MetadataFilter] Starting hierarchical filter for: {query[:100]}...")
    
    result = {
        "product_types": [],
        "vendors": [],
        "models": {},
        "total_products": 0,
        "filter_applied": False,
        "query": query
    }
    
    try:
        # Layer 1: Product Type Extraction
        if explicit_product_type:
            product_types = [explicit_product_type.lower().replace(" ", "_")]
        else:
            product_types = extract_product_types(query)
        
        result["product_types"] = product_types
        
        if not product_types:
            logger.warning("[MetadataFilter] No product types detected - filter not applied")
            return result
        
        # Layer 2: Vendor Filtering
        query_vendors = extract_vendors(query)
        
        if explicit_vendors:
            vendors = explicit_vendors
        elif query_vendors:
            # User specified vendors in query - use those
            vendors = query_vendors
        else:
            # Get all vendors for the product type
            vendors = filter_vendors_by_product(product_types)
        
        result["vendors"] = vendors
        
        if not vendors:
            logger.warning("[MetadataFilter] No vendors found for product type")
            return result
        
        # Layer 3: Model Filtering
        model_hints = extract_models(query)
        pt_formatted = product_types[0].replace("_", " ") if product_types else ""
        
        models_data = filter_models_by_vendor(
            vendors=vendors,
            product_type=pt_formatted,
            model_hints=model_hints if model_hints else None
        )
        
        result["models"] = models_data
        
        # Count total products
        total = sum(len(products) for products in models_data.values())
        result["total_products"] = total
        result["filter_applied"] = True
        
        logger.info(f"[MetadataFilter] Hierarchical filter complete:")
        logger.info(f"  Product Types: {product_types}")
        logger.info(f"  Vendors: {len(vendors)}")
        logger.info(f"  Total Products: {total}")
        
        return result
        
    except Exception as e:
        logger.error(f"[MetadataFilter] Error in hierarchical filter: {e}", exc_info=True)
        result["error"] = str(e)
        return result


# ============================================================================
# LLM-ENHANCED EXTRACTION (Flash Lite)
# ============================================================================

def extract_metadata_with_llm(
    query: str,
    use_flash_lite: bool = True
) -> Dict[str, Any]:
    """
    Use LLM (Flash Lite) to extract structured metadata from query.
    
    This provides more accurate extraction than regex patterns.
    
    Args:
        query: User's search query
        use_flash_lite: Use lightweight Flash Lite model
        
    Returns:
        Extracted metadata dict
    """
    try:
        from llm_fallback import create_llm_with_fallback
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import JsonOutputParser
        
        # Use Flash Lite for fast, lightweight extraction
        model = "gemini-2.5-flash" if use_flash_lite else "gemini-2.5-flash"
        
        llm = create_llm_with_fallback(
            model=model,
            temperature=0.1,
            max_tokens=500
        )
        
        prompt = ChatPromptTemplate.from_template("""
Extract structured metadata from this product search query.

Query: {query}

Return ONLY valid JSON:
{{
    "product_type": "<detected product type or null>",
    "vendors": ["<list of mentioned vendor names>"],
    "models": ["<list of mentioned model numbers>"],
    "specifications": {{
        "<spec_name>": "<spec_value>"
    }},
    "intent": "<search|compare|quote|info>"
}}
""")
        
        parser = JsonOutputParser()
        chain = prompt | llm | parser
        
        result = chain.invoke({"query": query})
        
        logger.info(f"[MetadataFilter] LLM extraction result: {result}")
        return result
        
    except Exception as e:
        logger.error(f"[MetadataFilter] LLM extraction failed: {e}")
        # Fallback to regex extraction
        return {
            "product_type": extract_product_types(query)[0] if extract_product_types(query) else None,
            "vendors": extract_vendors(query),
            "models": extract_models(query),
            "specifications": {},
            "intent": "search"
        }


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'extract_product_types',
    'extract_vendors',
    'extract_models',
    'filter_vendors_by_product',
    'filter_models_by_vendor',
    'filter_by_hierarchy',
    'extract_metadata_with_llm',
    'PRODUCT_TYPE_PATTERNS',
    'KNOWN_VENDORS'
]
