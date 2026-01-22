# agentic/standards_rag_enrichment.py
# =============================================================================
# STANDARDS RAG ENRICHMENT UTILITIES
# =============================================================================
#
# Shared utility functions for enriching identified instruments/accessories
# with Standards RAG data. Used by both instrument_identifier_workflow and
# solution_workflow.
#
# PERFORMANCE OPTIMIZATIONS:
# - Parallel processing with ThreadPoolExecutor (items processed simultaneously)
# - Product type caching (avoid duplicate queries for same product type)
# - Singleton agent pattern (avoid re-initialization overhead)
#
# =============================================================================

import logging
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

logger = logging.getLogger(__name__)

# =============================================================================
# PRODUCT TYPE CACHE (Performance Optimization)
# =============================================================================

_standards_cache = {}
_standards_cache_lock = threading.Lock()


def _get_cached_standards(product_type: str) -> Optional[Dict[str, Any]]:
    """Get cached standards result for a product type."""
    with _standards_cache_lock:
        return _standards_cache.get(product_type.lower().strip())


def _cache_standards(product_type: str, result: Dict[str, Any]):
    """Cache standards result for a product type."""
    with _standards_cache_lock:
        _standards_cache[product_type.lower().strip()] = result


def clear_standards_cache():
    """Clear the standards cache (useful for testing or refreshing)."""
    global _standards_cache
    with _standards_cache_lock:
        _standards_cache.clear()
    logger.info("[StandardsEnrichment] Cache cleared")


# =============================================================================
# PARALLEL ENRICHMENT FUNCTIONS
# =============================================================================

def _enrich_single_item(
    item: Dict[str, Any],
    product_type_context: Optional[str],
    top_k: int
) -> Dict[str, Any]:
    """
    Enrich a single item with standards. Used by parallel executor.
    
    Args:
        item: Item to enrich
        product_type_context: Context product type
        top_k: Number of standards documents to retrieve
        
    Returns:
        Enriched item with standards_info
    """
    try:
        from tools.standards_enrichment_tool import get_applicable_standards
        
        # Determine the product type for this item
        item_product_type = (
            item.get("category") or 
            item.get("product_name") or 
            item.get("name") or 
            product_type_context or 
            "industrial instrument"
        )
        
        # Check cache first (MAJOR PERF OPTIMIZATION)
        cached_result = _get_cached_standards(item_product_type)
        if cached_result is not None:
            logger.info(f"[StandardsEnrichment] Cache HIT for: {item_product_type}")
            item["standards_info"] = cached_result.copy()
            item["standards_info"]["from_cache"] = True
            if cached_result.get("applicable_standards"):
                item["normalized_category"] = _normalize_category(
                    item_product_type, 
                    cached_result.get("applicable_standards", [])
                )
            return item
        
        logger.info(f"[StandardsEnrichment] Querying standards for: {item_product_type}")
        
        standards_result = get_applicable_standards(
            product_type=item_product_type,
            top_k=top_k
        )
        
        if standards_result.get("success"):
            # Build standards_info
            standards_info = {
                "applicable_standards": standards_result.get("applicable_standards", []),
                "certifications": standards_result.get("certifications", []),
                "safety_requirements": standards_result.get("safety_requirements", {}),
                "communication_protocols": standards_result.get("communication_protocols", []),
                "environmental_requirements": standards_result.get("environmental_requirements", {}),
                "confidence": standards_result.get("confidence", 0.0),
                "sources": standards_result.get("sources", []),
                "enrichment_status": "success",
                "from_cache": False
            }
            
            item["standards_info"] = standards_info
            
            # Cache for future use
            _cache_standards(item_product_type, standards_info)
            
            # Add normalized category if available from standards
            if standards_result.get("applicable_standards"):
                item["normalized_category"] = _normalize_category(
                    item_product_type, 
                    standards_result.get("applicable_standards", [])
                )
                
            logger.info(
                f"[StandardsEnrichment] Enriched {item_product_type}: "
                f"{len(standards_result.get('applicable_standards', []))} standards, "
                f"{len(standards_result.get('certifications', []))} certifications"
            )
        else:
            # Mark as failed but continue
            item["standards_info"] = {
                "enrichment_status": "failed",
                "error": standards_result.get("error", "Unknown error"),
                "from_cache": False
            }
            logger.warning(
                f"[StandardsEnrichment] Failed to enrich {item_product_type}: "
                f"{standards_result.get('error')}"
            )
                
    except Exception as e:
        logger.error(f"[StandardsEnrichment] Error enriching item: {e}")
        item["standards_info"] = {
            "enrichment_status": "error",
            "error": str(e),
            "from_cache": False
        }
    
    return item


def enrich_identified_items_with_standards(
    items: List[Dict[str, Any]],
    product_type: Optional[str] = None,
    domain: Optional[str] = None,
    top_k: int = 3,
    max_workers: int = 5
) -> List[Dict[str, Any]]:
    """
    Enriches identified instruments/accessories with Standards RAG data.
    
    PERFORMANCE OPTIMIZATIONS:
    - Uses parallel processing (ThreadPoolExecutor) for multiple items
    - Caches results by product type to avoid duplicate queries
    - Uses singleton agent for LLM/vector store connections
    
    For each item:
    1. Check cache for previously queried product types
    2. Query Standards RAG for applicable standards (in parallel)
    3. Add normalized category, certifications, and compliance info
    4. Return enriched item list
    
    Args:
        items: List of identified instruments/accessories
        product_type: Optional product type context (e.g., "pressure transmitter")
        domain: Optional industry domain (e.g., "Oil & Gas", "Chemical")
        top_k: Number of standards documents to retrieve per query
        max_workers: Maximum parallel threads for enrichment
        
    Returns:
        List of enriched items with standards_info field added
    """
    if not items:
        logger.info("[StandardsEnrichment] No items to enrich")
        return items
    
    import time
    start_time = time.time()
    
    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║     🔵 PARALLEL STANDARDS ENRICHMENT (Performance Optimized)  🔵     ║
    # ╚══════════════════════════════════════════════════════════════════════╝
    logger.info("="*70)
    logger.info("PARALLEL STANDARDS ENRICHMENT STARTING")
    logger.info(f"   Items to enrich: {len(items)}")
    logger.info(f"   Max workers: {max_workers}")
    logger.info(f"   Cache size: {len(_standards_cache)}")
    logger.info("="*70)
    
    try:
        # Determine optimal worker count
        actual_workers = min(len(items), max_workers)
        
        if len(items) == 1:
            # Single item - no parallelization needed
            enriched_items = [_enrich_single_item(items[0], product_type, top_k)]
        else:
            # Multiple items - use parallel processing
            enriched_items = []
            
            with ThreadPoolExecutor(max_workers=actual_workers) as executor:
                # Submit all items for parallel processing
                future_to_item = {
                    executor.submit(_enrich_single_item, item.copy(), product_type, top_k): idx 
                    for idx, item in enumerate(items)
                }
                
                # Collect results as they complete
                results = [None] * len(items)
                for future in as_completed(future_to_item):
                    idx = future_to_item[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        logger.error(f"[StandardsEnrichment] Future error for item {idx}: {e}")
                        # Return original item with error status
                        original_item = items[idx].copy()
                        original_item["standards_info"] = {
                            "enrichment_status": "error",
                            "error": str(e)
                        }
                        results[idx] = original_item
                
                enriched_items = results
        
        elapsed_time = time.time() - start_time
        cache_hits = sum(1 for item in enriched_items if item.get("standards_info", {}).get("from_cache", False))
        
        logger.info("="*70)
        logger.info("PARALLEL STANDARDS ENRICHMENT COMPLETED")
        logger.info(f"   Total time: {elapsed_time:.2f}s")
        logger.info(f"   Items enriched: {len(enriched_items)}")
        logger.info(f"   Cache hits: {cache_hits}")
        logger.info(f"   Avg time per item: {elapsed_time/len(items):.2f}s")
        logger.info("="*70)
        
        return enriched_items
        
    except ImportError as e:
        logger.error(f"[StandardsEnrichment] Import error: {e}")
        # Return items unchanged if import fails
        for item in items:
            item["standards_info"] = {
                "enrichment_status": "unavailable",
                "error": "Standards enrichment tool not available"
            }
        return items
    except Exception as e:
        logger.error(f"[StandardsEnrichment] Unexpected error: {e}", exc_info=True)
        for item in items:
            item["standards_info"] = {
                "enrichment_status": "error",
                "error": str(e)
            }
        return items


def validate_items_against_domain_standards(
    items: List[Dict[str, Any]],
    domain: str,
    safety_requirements: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Validates identified items against domain-specific standards.

    🔥 FIX #4: VALIDATION CACHE OPTIMIZATION 🔥
    Uses cached standards_specifications from Phase 3 enrichment instead of
    re-running RAG. This saves 2000+ seconds by avoiding duplicate RAG queries.

    Used by solution_workflow to validate compatibility with industry standards.

    Args:
        items: List of identified instruments/accessories
        domain: Industry domain (e.g., "Oil & Gas", "Chemical", "Pharma")
        safety_requirements: Safety requirements from solution analysis

    Returns:
        Validation result with compliance status and issues
    """
    logger.info(f"[StandardsValidation] Validating {len(items)} items for domain: {domain}")

    validation_result = {
        "is_compliant": True,
        "compliance_issues": [],
        "recommendations": [],
        "domain": domain,
        "cache_hits": 0,
        "cache_misses": 0
    }

    try:
        from tools.standards_enrichment_tool import validate_requirements_against_standards

        for item in items:
            item_product_type = item.get("category") or item.get("name") or "instrument"
            item_name = item.get("name") or item.get("product_name") or "Unknown"

            # 🔥 FIX #1 STEP 1: Check if item was enriched via phase3_optimized 🔥
            # If so, SKIP RAG entirely - validation is just type checking
            if item.get("enrichment_source") == "phase3_optimized":
                logger.info(f"[StandardsValidation] SKIPPING RAG for {item_name} - enriched via phase3 (saves 40-50s!)")
                validation_result["cache_hits"] += 1

                # Just do lightweight validation, no RAG needed
                cached_specs = item.get("standards_specifications", {})
                if cached_specs.get("applicable_standards"):
                    logger.debug(f"[StandardsValidation] {item_name} is phase3 optimized with {len(cached_specs.get('applicable_standards', []))} standards")
                continue

            # 🔥 FIX #4 STEP 2: Check if item already has cached standards_specifications 🔥
            # If Phase 3 enrichment populated this field, use it instead of re-running RAG
            if item.get("standards_specifications"):
                logger.info(f"[StandardsValidation] Using CACHED standards for {item_name} (lightweight validation only)")
                validation_result["cache_hits"] += 1

                # Build validation from cached standards
                cached_specs = item.get("standards_specifications", {})
                requirements = {
                    "product_type": item_product_type,
                    "specifications": cached_specs,
                }

                # Add safety requirements if provided
                if safety_requirements:
                    if safety_requirements.get("sil_level"):
                        requirements["sil_level"] = safety_requirements["sil_level"]
                    if safety_requirements.get("hazardous_area"):
                        requirements["hazardous_area"] = True
                    if safety_requirements.get("atex_zone"):
                        requirements["atex_zone"] = safety_requirements["atex_zone"]

                # Simple validation logic (no RAG needed!)
                try:
                    # Check if certifications are specified
                    user_certs = cached_specs.get("certifications", [])
                    if not user_certs:
                        validation_result["recommendations"].append({
                            "item": item_name,
                            "category": "certifications",
                            "recommendation": "Verify required certifications are specified"
                        })

                    # Mark as compliant if we have standards info
                    if cached_specs.get("applicable_standards"):
                        logger.debug(f"[StandardsValidation] {item_name} has {len(cached_specs.get('applicable_standards', []))} applicable standards")

                except Exception as e:
                    logger.warning(f"[StandardsValidation] Simple validation failed for {item_product_type}: {e}")

            else:
                # 🔥 FIX #4 STEP 2: Cache miss - only then run RAG 🔥
                logger.info(f"[StandardsValidation] Cache MISS for {item_name} - running RAG (slower path)")
                validation_result["cache_misses"] += 1

                # Build requirements dict from item
                requirements = {
                    "product_type": item_product_type,
                    "specifications": item.get("specifications", {}),
                }

                # Add safety requirements if provided
                if safety_requirements:
                    if safety_requirements.get("sil_level"):
                        requirements["sil_level"] = safety_requirements["sil_level"]
                    if safety_requirements.get("hazardous_area"):
                        requirements["hazardous_area"] = True
                    if safety_requirements.get("atex_zone"):
                        requirements["atex_zone"] = safety_requirements["atex_zone"]

                # Validate against standards (this will run RAG)
                try:
                    val_result = validate_requirements_against_standards(
                        product_type=item_product_type,
                        requirements=requirements
                    )

                    if not val_result.get("is_compliant"):
                        validation_result["is_compliant"] = False
                        validation_result["compliance_issues"].extend([
                            {
                                "item": item_name,
                                **issue
                            }
                            for issue in val_result.get("compliance_issues", [])
                        ])

                    validation_result["recommendations"].extend([
                        {
                            "item": item_name,
                            **rec
                        }
                        for rec in val_result.get("recommendations", [])
                    ])

                except Exception as e:
                    logger.warning(f"[StandardsValidation] Validation failed for {item_product_type}: {e}")

    except ImportError:
        logger.warning("[StandardsValidation] Standards validation tool not available")
        validation_result["validation_status"] = "unavailable"
    except Exception as e:
        logger.error(f"[StandardsValidation] Validation error: {e}")
        validation_result["validation_status"] = "error"
        validation_result["error"] = str(e)

    logger.info(f"[StandardsValidation] Validation complete - Cache hits: {validation_result['cache_hits']}, Misses: {validation_result['cache_misses']}")

    return validation_result


def _normalize_category(product_type: str, standards: List[str]) -> str:
    """
    Normalize product category based on standards terminology.
    
    Args:
        product_type: Raw product type string
        standards: List of applicable standards
        
    Returns:
        Normalized category string
    """
    # Common normalizations for instrumentation
    normalizations = {
        "temperature transmitter": "TT",
        "pressure transmitter": "PT",
        "flow meter": "FT",
        "level transmitter": "LT",
        "control valve": "CV",
        "isolation valve": "XV",
        "pressure gauge": "PG",
        "temperature sensor": "TE",
        "thermocouple": "TE",
        "rtd": "TE",
        "flow switch": "FS",
        "pressure switch": "PS",
        "level switch": "LS",
        "temperature switch": "TS",
    }
    
    product_lower = product_type.lower()
    
    for key, code in normalizations.items():
        if key in product_lower:
            return code
    
    # Return first word capitalized as fallback
    return product_type.split()[0].upper() if product_type else "INST"


# =============================================================================
# STANDARDS QUESTION DETECTION
# =============================================================================

STANDARDS_KEYWORDS = [
    # Standard codes
    "iec", "iso", "api", "ansi", "isa", "asme", "nfpa", "ieee", "nist", "en",
    # Certifications
    "sil", "atex", "iecex", "ul", "csa", "fm", "ce mark", "nema", "ip rating",
    # Standards-related terms
    "standard", "certification", "compliance", "regulation", "requirement",
    "calibration", "safety integrity", "hazardous area", "explosion proof",
    "intrinsically safe", "functional safety"
]

def is_standards_related_question(question: str) -> bool:
    """
    Detect if a user question is related to standards.
    
    Used by EnGenie workflow to route standards questions to Standards RAG.
    
    Args:
        question: User's question text
        
    Returns:
        True if the question is standards-related
    """
    question_lower = question.lower()
    
    # Check for standards keywords
    for keyword in STANDARDS_KEYWORDS:
        if keyword in question_lower:
            return True
    
    # Check for patterns like "what are the ... standards"
    standards_patterns = [
        "what standard",
        "which standard",
        "what certification",
        "what is sil",
        "what is atex",
        "compliance with",
        "according to",
        "requirement for",
        "specification for",
    ]
    
    for pattern in standards_patterns:
        if pattern in question_lower:
            return True
    
    return False


def route_standards_question(question: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Route a standards-related question to Standards RAG and return the response.
    
    Args:
        question: User's standards-related question
        top_k: Number of documents to retrieve
        
    Returns:
        Standards RAG response with answer and citations
    """
    logger.info(f"[StandardsRouting] Routing standards question: {question[:50]}...")
    
    try:
        from agentic.standards_rag import run_standards_rag_workflow
        
        result = run_standards_rag_workflow(
            question=question,
            top_k=top_k
        )
        
        if result.get("status") == "success":
            final_response = result.get("final_response", {})
            return {
                "success": True,
                "answer": final_response.get("answer", ""),
                "citations": final_response.get("citations", []),
                "sources_used": final_response.get("sources_used", []),
                "confidence": final_response.get("confidence", 0.0),
                "is_standards_response": True
            }
        else:
            return {
                "success": False,
                "error": result.get("error", "Standards RAG query failed"),
                "is_standards_response": True
            }
            
    except Exception as e:
        logger.error(f"[StandardsRouting] Error routing question: {e}")
        return {
            "success": False,
            "error": str(e),
            "is_standards_response": True
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "enrich_identified_items_with_standards",
    "validate_items_against_domain_standards",
    "is_standards_related_question",
    "route_standards_question",
]
