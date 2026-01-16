# agentic/deep_agent/llm_specs_generator.py
# =============================================================================
# LLM SPECIFICATION GENERATOR
# =============================================================================
#
# Generates all possible specifications for a product type using LLM.
# These specs fill in gaps not covered by user-specified or standards specs.
#
# PERFORMANCE OPTIMIZATIONS:
# - Singleton LLM instance (avoids re-initialization per call)
# - Parallel batch processing (concurrent item generation)
# - Product type caching with LRU cache
# - Thread-safe initialization with locking
#
# =============================================================================

import logging
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

from llm_fallback import create_llm_with_fallback

load_dotenv()
logger = logging.getLogger(__name__)

# Model for LLM spec generation
LLM_SPECS_MODEL = "gemini-2.5-flash"

# =============================================================================
# SINGLETON LLM INSTANCE (Performance Optimization)
# =============================================================================

_llm_specs_instance = None
_llm_specs_lock = threading.Lock()


def _get_llm_specs_llm():
    """
    Get or create singleton LLM instance for specs generation.
    Thread-safe with double-checked locking pattern.
    """
    global _llm_specs_instance
    
    if _llm_specs_instance is None:
        with _llm_specs_lock:
            # Double-check after acquiring lock
            if _llm_specs_instance is None:
                logger.info("[LLM_SPECS] Initializing singleton LLM instance...")
                _llm_specs_instance = create_llm_with_fallback(
                    model=LLM_SPECS_MODEL,
                    temperature=0.3,  # Some creativity for comprehensive coverage
                    google_api_key=os.getenv("GOOGLE_API_KEY")
                )
                logger.info("[LLM_SPECS] Singleton LLM instance ready")
    
    return _llm_specs_instance


# =============================================================================
# GENERATION PROMPT
# =============================================================================

LLM_SPECS_GENERATION_PROMPT = """
You are an industrial instrumentation expert. Generate technical specifications
for the given product type. Return ONLY clean technical values - NO descriptions.

PRODUCT TYPE: {product_type}
CATEGORY: {category}
CONTEXT: {context}

=== CRITICAL: VALUE FORMAT RULES ===

Return ONLY the technical value. NO explanations, NO descriptions, NO sentences.

FORBIDDEN PATTERNS IN VALUES (NEVER include these):
- Words: "typically", "usually", "generally", "approximately", "may be", "can be"
- Phrases: "depends on", "varies by", "according to", "based on", "provided via"
- Sentences: "The X is", "It has", "This should be", "A value of"
- Explanations: "which means", "for applications", "in order to", "when used"
- Parenthetical notes: "(typically...)", "(usually...)", "(for example...)"

CORRECT VALUE EXAMPLES:
- accuracy: "±0.1%" ✓ (NOT "The accuracy is typically ±0.1% depending on...")
- output_signal: "4-20mA, HART" ✓ (NOT "4-20mA with optional HART protocol support")
- protection_rating: "IP67" ✓ (NOT "IP67 protection suitable for outdoor use")
- temperature_range: "-40 to +85°C" ✓ (NOT "Temperature range varies from -40 to +85°C")
- communication_protocol: "HART, Modbus" ✓ (NOT "N/A (typically uses HART or Modbus)")

WRONG VALUE EXAMPLES - NEVER OUTPUT THESE:
✗ "N/A (typically a transmitter with HART will be used)" → Use "HART" or "N/A"
✗ "SPDT Relay, 4-20mA Input" with note → Just "SPDT Relay, 4-20mA"
✗ "±0.1% of Span (accuracy based on calibration)" → Just "±0.1% of Span"

=== SPECIFICATIONS TO GENERATE ===

Generate these specifications with CLEAN values:
- accuracy, repeatability, rangeability
- temperature_range, ambient_temperature, protection_rating, humidity_range
- output_signal, supply_voltage, power_consumption
- communication_protocol
- material_wetted, material_housing, process_connection, weight
- sil_rating, hazardous_area_approval, certifications
- response_time, stability, calibration_interval

=== OUTPUT FORMAT ===

Return ONLY valid JSON with FLAT key-value structure (NO section grouping):
{{
    "specifications": {{
        "accuracy": {{"value": "±0.1%", "confidence": 0.8}},
        "output_signal": {{"value": "4-20mA, HART", "confidence": 0.9}},
        "temperature_range": {{"value": "-40 to +85°C", "confidence": 0.8}},
        "protection_rating": {{"value": "IP67", "confidence": 0.9}},
        "communication_protocol": {{"value": "HART, Modbus RTU", "confidence": 0.7}},
        "supply_voltage": {{"value": "24 VDC", "confidence": 0.9}},
        "sil_rating": {{"value": "SIL 2", "confidence": 0.8}},
        "material_wetted": {{"value": "316L SS", "confidence": 0.8}}
    }},
    "product_category": "{category}",
    "generation_notes": "Brief note"
}}

IMPORTANT:
1. Each value must be a CLEAN technical specification
2. NO section headers, NO nested categories
3. NO descriptive text or explanations in values
4. Use "N/A" only if truly not applicable - don't add explanations
"""


# =============================================================================
# GENERATION FUNCTION
# =============================================================================

def generate_llm_specs(
    product_type: str,
    category: Optional[str] = None,
    context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate specifications for a product type using LLM.

    These specifications are used to fill gaps not covered by user-specified
    or standards-based specifications.

    Args:
        product_type: The product type to generate specs for
        category: Optional category for context
        context: Optional additional context (e.g., sample_input)

    Returns:
        Dict with:
            - specifications: Dict of generated specs with values and confidence
            - source: "llm_generated"
            - generation_notes: Notes on generation
    """
    logger.info(f"[LLM_SPECS] Generating specs for: {product_type}")

    try:
        # Use singleton LLM instance (PERFORMANCE: avoids re-initialization)
        llm = _get_llm_specs_llm()

        prompt = ChatPromptTemplate.from_template(LLM_SPECS_GENERATION_PROMPT)
        parser = JsonOutputParser()
        chain = prompt | llm | parser

        result = chain.invoke({
            "product_type": product_type,
            "category": category or "Industrial Instrument",
            "context": context or "General industrial application"
        })

        raw_specs = result.get("specifications", {})
        notes = result.get("generation_notes", "")

        # Flatten specs if they have nested structure
        flattened_specs = {}
        for key, value in raw_specs.items():
            if isinstance(value, dict):
                flattened_specs[key] = {
                    "value": value.get("value", str(value)),
                    "confidence": value.get("confidence", 0.7),
                    "note": value.get("note", "")
                }
            else:
                flattened_specs[key] = {
                    "value": str(value),
                    "confidence": 0.7,
                    "note": ""
                }

        # Filter out null/empty values
        clean_specs = {
            k: v for k, v in flattened_specs.items()
            if v.get("value") and str(v.get("value")).lower() not in ["null", "none", "n/a"]
        }

        logger.info(f"[LLM_SPECS] Generated {len(clean_specs)} specs for {product_type}")

        return {
            "specifications": clean_specs,
            "source": "llm_generated",
            "generation_notes": notes,
            "timestamp": datetime.now().isoformat(),
            "product_type": product_type,
            "category": category
        }

    except Exception as e:
        logger.error(f"[LLM_SPECS] Generation failed for {product_type}: {e}")
        return {
            "specifications": {},
            "source": "llm_generated",
            "generation_notes": f"Generation failed: {str(e)}",
            "timestamp": datetime.now().isoformat(),
            "product_type": product_type,
            "error": str(e)
        }


def generate_llm_specs_batch(
    items: List[Dict[str, Any]],
    max_workers: int = 5
) -> List[Dict[str, Any]]:
    """
    Generate LLM specs for multiple items IN PARALLEL.
    
    PERFORMANCE: Uses ThreadPoolExecutor for concurrent generation.
    Reduces batch time from O(n * 2s) to O(n/workers * 2s).

    Args:
        items: List of identified items with 'name', 'category', 'sample_input', etc.
        max_workers: Maximum parallel threads (default: 5)

    Returns:
        List of generation results, one per item
    """
    if not items:
        return []
    
    logger.info(f"[LLM_SPECS] Parallel batch generation for {len(items)} items (max_workers={max_workers})")
    
    # Pre-initialize singleton LLM to avoid race condition
    _get_llm_specs_llm()
    
    def generate_single_item(item: Dict[str, Any]) -> Dict[str, Any]:
        """Generate specs for a single item (thread-safe)."""
        product_type = item.get("name") or item.get("product_name", "Unknown")
        category = item.get("category", "Industrial Instrument")
        context = item.get("sample_input", "")
        
        result = generate_llm_specs(
            product_type=product_type,
            category=category,
            context=context
        )
        
        result["item_name"] = product_type
        result["item_type"] = item.get("type", "instrument")
        return result
    
    results = []
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as executor:
        # Submit all items for parallel processing
        future_to_item = {
            executor.submit(generate_single_item, item): item
            for item in items
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_item):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                item = future_to_item[future]
                item_name = item.get("name", "Unknown")
                logger.error(f"[LLM_SPECS] Parallel generation failed for {item_name}: {e}")
                results.append({
                    "specifications": {},
                    "source": "llm_generated",
                    "generation_notes": f"Generation failed: {str(e)}",
                    "timestamp": datetime.now().isoformat(),
                    "product_type": item_name,
                    "item_name": item_name,
                    "item_type": item.get("type", "instrument"),
                    "error": str(e)
                })
    
    logger.info(f"[LLM_SPECS] Parallel batch complete: {len(results)} items processed")
    return results


# =============================================================================
# TRUE BATCH PROCESSING - Single LLM Call (Phase 2 Optimization)
# =============================================================================

# Cache for product type specs (LRU with max 100 entries)
_product_type_cache: Dict[str, Dict[str, Any]] = {}
_cache_lock = threading.Lock()
MAX_CACHE_SIZE = 100


def _get_cached_specs(product_type: str) -> Optional[Dict[str, Any]]:
    """Get cached specs for a product type if available."""
    normalized_key = product_type.lower().strip()
    with _cache_lock:
        return _product_type_cache.get(normalized_key)


def _cache_specs(product_type: str, specs: Dict[str, Any]) -> None:
    """Cache specs for a product type."""
    normalized_key = product_type.lower().strip()
    with _cache_lock:
        # Simple LRU: remove oldest if cache is full
        if len(_product_type_cache) >= MAX_CACHE_SIZE:
            oldest_key = next(iter(_product_type_cache))
            del _product_type_cache[oldest_key]
        _product_type_cache[normalized_key] = specs


# Prompt for TRUE batch processing (single call for multiple products)
LLM_BATCH_GENERATION_PROMPT = """
You are an industrial instrumentation expert. Generate technical specifications
for MULTIPLE product types in a single response.

=== PRODUCTS TO PROCESS ===
{products_json}

=== CRITICAL: VALUE FORMAT RULES ===
Return ONLY clean technical values - NO descriptions, NO explanations.

For EACH product, generate these specifications:
- accuracy, repeatability, rangeability
- temperature_range, ambient_temperature, protection_rating, humidity_range
- output_signal, supply_voltage, power_consumption
- communication_protocol
- material_wetted, material_housing, process_connection, weight
- sil_rating, hazardous_area_approval, certifications
- response_time, stability, calibration_interval

=== OUTPUT FORMAT ===

Return ONLY valid JSON with specs for ALL products:
{{
    "products": {{
        "Product Name 1": {{
            "specifications": {{
                "accuracy": {{"value": "±0.1%", "confidence": 0.8}},
                "output_signal": {{"value": "4-20mA", "confidence": 0.9}},
                ...
            }}
        }},
        "Product Name 2": {{
            "specifications": {{
                ...
            }}
        }}
    }}
}}

IMPORTANT:
1. Use EXACT product names as keys (case-sensitive)
2. Each value must be a CLEAN technical specification
3. NO descriptive text or explanations in values
"""


def generate_llm_specs_true_batch(
    items: List[Dict[str, Any]],
    use_cache: bool = True
) -> List[Dict[str, Any]]:
    """
    Generate LLM specs for multiple items in a SINGLE LLM call.
    
    MAJOR PERFORMANCE GAIN: Reduces N API calls to 1 call.
    
    Args:
        items: List of items with 'name', 'category', 'sample_input'
        use_cache: Whether to use caching for repeated product types
    
    Returns:
        List of generation results, one per item
    """
    if not items:
        return []
    
    logger.info(f"[LLM_SPECS] TRUE batch generation for {len(items)} items (single LLM call)")
    start_time = time.time()
    
    # Check cache for already-processed product types
    cached_results = {}
    items_to_process = []
    
    if use_cache:
        for item in items:
            product_type = item.get("name") or item.get("product_name", "Unknown")
            cached = _get_cached_specs(product_type)
            if cached:
                logger.debug(f"[LLM_SPECS] Cache hit for: {product_type}")
                cached_results[product_type] = cached
            else:
                items_to_process.append(item)
        
        if cached_results:
            logger.info(f"[LLM_SPECS] Cache hits: {len(cached_results)}, need to process: {len(items_to_process)}")
    else:
        items_to_process = items
    
    # Build batch for uncached items
    llm_results = {}
    
    if items_to_process:
        try:
            llm = _get_llm_specs_llm()
            
            # Build products JSON for prompt
            products_info = []
            for item in items_to_process:
                products_info.append({
                    "name": item.get("name") or item.get("product_name", "Unknown"),
                    "category": item.get("category", "Industrial Instrument"),
                    "context": item.get("sample_input", "")
                })
            
            import json
            products_json = json.dumps(products_info, indent=2)
            
            prompt = ChatPromptTemplate.from_template(LLM_BATCH_GENERATION_PROMPT)
            parser = JsonOutputParser()
            chain = prompt | llm | parser
            
            result = chain.invoke({"products_json": products_json})
            
            # Parse results for each product
            products_data = result.get("products", {})
            
            for item in items_to_process:
                product_type = item.get("name") or item.get("product_name", "Unknown")
                product_specs = products_data.get(product_type, {})
                
                if not product_specs:
                    # Try case-insensitive match
                    for key in products_data:
                        if key.lower() == product_type.lower():
                            product_specs = products_data[key]
                            break
                
                raw_specs = product_specs.get("specifications", product_specs)
                
                # Flatten specs
                flattened_specs = {}
                for key, value in raw_specs.items():
                    if isinstance(value, dict):
                        flattened_specs[key] = {
                            "value": value.get("value", str(value)),
                            "confidence": value.get("confidence", 0.7),
                        }
                    else:
                        flattened_specs[key] = {
                            "value": str(value),
                            "confidence": 0.7,
                        }
                
                # Filter out empty
                clean_specs = {
                    k: v for k, v in flattened_specs.items()
                    if v.get("value") and str(v.get("value")).lower() not in ["null", "none", "n/a"]
                }
                
                llm_results[product_type] = clean_specs
                
                # Cache the result
                if use_cache:
                    _cache_specs(product_type, clean_specs)
            
            logger.info(f"[LLM_SPECS] Batch LLM call processed {len(items_to_process)} products")
            
        except Exception as e:
            logger.error(f"[LLM_SPECS] Batch generation failed: {e}")
            # Fall back to empty results for all items
            for item in items_to_process:
                product_type = item.get("name") or item.get("product_name", "Unknown")
                llm_results[product_type] = {}
    
    # Combine cached and new results
    all_specs = {**cached_results, **llm_results}
    
    # Build final results list
    results = []
    for item in items:
        product_type = item.get("name") or item.get("product_name", "Unknown")
        specs = all_specs.get(product_type, {})
        
        results.append({
            "specifications": specs,
            "source": "llm_generated",
            "generation_notes": "Batch generated" if product_type in llm_results else "From cache",
            "timestamp": datetime.now().isoformat(),
            "product_type": product_type,
            "item_name": product_type,
            "item_type": item.get("type", "instrument"),
            "category": item.get("category", "Industrial Instrument")
        })
    
    elapsed = time.time() - start_time
    logger.info(f"[LLM_SPECS] TRUE batch complete: {len(results)} items in {elapsed:.2f}s ({elapsed/len(items):.2f}s per item)")
    
    return results


def clear_llm_specs_cache() -> int:
    """Clear the product type cache. Returns number of items cleared."""
    global _product_type_cache
    with _cache_lock:
        count = len(_product_type_cache)
        _product_type_cache = {}
        logger.info(f"[LLM_SPECS] Cache cleared: {count} items")
        return count


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    with _cache_lock:
        return {
            "size": len(_product_type_cache),
            "max_size": MAX_CACHE_SIZE,
            "cached_types": list(_product_type_cache.keys())
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "generate_llm_specs",
    "generate_llm_specs_batch",
    "generate_llm_specs_true_batch",
    "clear_llm_specs_cache",
    "get_cache_stats"
]

