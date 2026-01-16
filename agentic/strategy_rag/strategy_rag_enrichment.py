# agentic/strategy_rag_enrichment.py
# =============================================================================
# STRATEGY ENRICHMENT - CSV-Based Vendor Strategy Enrichment
# =============================================================================
#
# UPDATED: Simplified from complex RAG enrichment to CSV-based filtering.
#
# PURPOSE: Enrich product search workflow with vendor strategy data from
# the instrumentation_procurement_strategy.csv file.
#
# =============================================================================

import logging
from typing import Dict, Any, List, Optional

from .strategy_csv_filter import (
    filter_vendors_by_strategy,
    get_vendor_strategy_info,
    get_strategy_filter
)
from .strategy_rag_workflow import get_strategy_for_product

logger = logging.getLogger(__name__)


# ============================================================================
# ENRICHMENT FUNCTIONS
# ============================================================================

def enrich_with_strategy_rag(
    product_type: str,
    requirements: Dict[str, Any] = None,
    top_k: int = 7,
    refinery: str = None
) -> Dict[str, Any]:
    """
    Enrich product search with CSV-based Strategy data.

    UPDATED: Uses CSV filtering instead of vector store RAG.

    Args:
        product_type: Product type to get strategy for
        requirements: Additional requirements context
        top_k: Number of vendors to return
        refinery: Optional refinery context

    Returns:
        Strategy enrichment data including:
        - preferred_vendors: List of preferred vendors from CSV
        - forbidden_vendors: Empty list (CSV doesn't have forbidden)
        - neutral_vendors: List of acceptable vendors
        - procurement_priorities: Vendor priority scores
        - strategy_notes: Key strategy insights
        - confidence: Confidence score
        - sources_used: CSV file name
    """
    try:
        logger.info(f"[Strategy Enrichment] Getting CSV strategy for {product_type}")

        result = get_strategy_for_product(
            product_type=product_type,
            requirements=requirements,
            top_k=top_k,
            refinery=refinery
        )

        # Check if we got valid results
        if result.get('preferred_vendors') or result.get('neutral_vendors'):
            logger.info(f"[Strategy Enrichment] CSV filter success - "
                       f"{len(result.get('preferred_vendors', []))} preferred vendors")
            return {
                'success': True,
                'rag_type': 'csv_filter',
                'preferred_vendors': result.get('preferred_vendors', []),
                'forbidden_vendors': result.get('forbidden_vendors', []),
                'neutral_vendors': result.get('neutral_vendors', []),
                'procurement_priorities': result.get('procurement_priorities', {}),
                'strategy_notes': result.get('strategy_notes', ''),
                'confidence': result.get('confidence', 0.5),
                'sources_used': result.get('sources_used', ['instrumentation_procurement_strategy.csv']),
                'citations': [],
                'category': result.get('metadata', {}).get('category'),
                'subcategory': result.get('metadata', {}).get('subcategory')
            }
        else:
            logger.warning("[Strategy Enrichment] No vendors found in CSV for this category")
            return {
                'success': False,
                'rag_type': 'csv_filter',
                'error': 'No strategy entries found for this product category.',
                'preferred_vendors': [],
                'forbidden_vendors': [],
                'neutral_vendors': [],
                'procurement_priorities': {},
                'confidence': 0.0
            }

    except Exception as e:
        logger.error(f"[Strategy Enrichment] Error: {e}")
        return {
            'success': False,
            'rag_type': 'csv_filter',
            'error': str(e),
            'preferred_vendors': [],
            'forbidden_vendors': [],
            'neutral_vendors': [],
            'procurement_priorities': {},
            'confidence': 0.0
        }


def get_strategy_with_auto_fallback(
    product_type: str,
    requirements: Dict[str, Any] = None,
    top_k: int = 7,
    refinery: str = None
) -> Dict[str, Any]:
    """
    Get strategy data (CSV-based, no fallback needed).

    UPDATED: The CSV approach is deterministic - no fallback to LLM needed.

    Args:
        product_type: Product type
        requirements: Additional requirements
        top_k: Number of vendors to return
        refinery: Optional refinery context

    Returns:
        Strategy enrichment data
    """
    return enrich_with_strategy_rag(
        product_type=product_type,
        requirements=requirements,
        top_k=top_k,
        refinery=refinery
    )


# ============================================================================
# VENDOR FILTERING
# ============================================================================

def filter_vendors_by_strategy_data(
    vendors: List[str],
    strategy_data: Dict[str, Any],
    product_type: str = None
) -> Dict[str, Any]:
    """
    Filter and prioritize vendors based on strategy data.
    
    UPDATED: Uses vector store similarity scores for dynamic ranking.

    Args:
        vendors: List of vendor names to filter
        strategy_data: Strategy enrichment result
        product_type: Optional product type for semantic scoring

    Returns:
        Filtered vendors with priority scores
    """
    preferred = [v.lower() for v in strategy_data.get('preferred_vendors', [])]
    priorities = strategy_data.get('procurement_priorities', {})
    
    # Try to use vector store for semantic scoring
    use_semantic = False
    vector_store = None
    
    if product_type:
        try:
            from .strategy_vector_store import get_strategy_vector_store
            vector_store = get_strategy_vector_store()
            use_semantic = True
            logger.debug("[StrategyEnrichment] Using semantic scoring for vendor ranking")
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"[StrategyEnrichment] Semantic scoring unavailable: {e}")

    accepted = []

    for vendor in vendors:
        vendor_lower = vendor.lower()

        # Check if preferred (from CSV data)
        is_preferred = any(p in vendor_lower or vendor_lower in p for p in preferred)

        # Calculate priority score
        priority_score = 0
        similarity_score = 0.0
        
        # Method 1: SEMANTIC SCORING (if available)
        if use_semantic and vector_store and product_type:
            try:
                similarity_score = vector_store.get_vendor_similarity(
                    vendor_name=vendor,
                    product_type=product_type
                )
                # Convert similarity (0-1) to priority scale (0-100)
                priority_score = int(similarity_score * 100)
                
                # Auto-detect preferred based on high similarity
                if similarity_score > 0.7:
                    is_preferred = True
                    
            except Exception as e:
                logger.debug(f"Semantic scoring failed for {vendor}: {e}")
        
        # Method 2: Rule-based fallback (existing logic)
        if priority_score == 0:
            for pv, score in priorities.items():
                if pv.lower() in vendor_lower or vendor_lower in pv.lower():
                    priority_score = score
                    break

        # Boost preferred vendors
        if is_preferred:
            priority_score += 10

        accepted.append({
            'vendor': vendor,
            'is_preferred': is_preferred,
            'priority_score': priority_score,
            'similarity_score': round(similarity_score, 4),
            'scoring_method': 'semantic' if similarity_score > 0 else 'rule_based',
            'status': 'preferred' if is_preferred else 'neutral'
        })

    # Sort by priority (semantic scores will naturally be higher for good matches)
    accepted.sort(key=lambda x: x['priority_score'], reverse=True)

    return {
        'accepted_vendors': accepted,
        'excluded_vendors': [],  # CSV doesn't have forbidden vendors
        'total_accepted': len(accepted),
        'total_excluded': 0,
        'scoring_method': 'semantic' if use_semantic else 'rule_based'
    }


def is_strategy_related_question(question: str) -> bool:
    """
    Check if a question is related to procurement strategy.

    Args:
        question: User's question

    Returns:
        True if question is strategy-related
    """
    strategy_keywords = [
        'prefer', 'preferred', 'recommended',
        'forbidden', 'avoid', 'exclude', 'banned',
        'vendor', 'supplier', 'manufacturer',
        'strategy', 'policy', 'guideline',
        'single-source', 'multi-source', 'sole-source',
        'contract', 'agreement', 'term',
        'lifecycle', 'total cost', 'TCO',
        'lead time', 'delivery',
        'sustainability', 'environmental',
        'local content', 'regional',
        'standardization', 'spare parts'
    ]

    question_lower = question.lower()
    return any(kw in question_lower for kw in strategy_keywords)


def route_strategy_question(question: str, product_type: str = "") -> Dict[str, Any]:
    """
    Route a strategy question through CSV-based strategy lookup.

    Args:
        question: User's strategy question
        product_type: Optional product type context

    Returns:
        Strategy response
    """
    try:
        result = get_strategy_for_product(
            product_type=product_type or "industrial instrument",
            top_k=7
        )

        return {
            'success': True,
            'answer': result.get('answer', 'No strategy information available.'),
            'preferred_vendors': result.get('preferred_vendors', []),
            'neutral_vendors': result.get('neutral_vendors', []),
            'strategy_notes': result.get('strategy_notes', ''),
            'sources_used': result.get('sources_used', [])
        }

    except Exception as e:
        logger.error(f"[Strategy Route] Error: {e}")
        return {
            'success': False,
            'error': str(e),
            'answer': f"Error processing strategy question: {str(e)}"
        }


# ============================================================================
# INTEGRATION WITH SCHEMA ENRICHMENT
# ============================================================================

def enrich_schema_with_strategy(
    product_type: str,
    schema: Dict[str, Any],
    requirements: Dict[str, Any] = None,
    refinery: str = None
) -> Dict[str, Any]:
    """
    Enrich a product schema with strategy information.

    Adds a 'procurement_strategy' section to the schema with:
    - preferred_vendors
    - neutral_vendors
    - procurement_priorities
    - strategy_notes

    Args:
        product_type: Product type
        schema: Schema to enrich
        requirements: Additional requirements
        refinery: Optional refinery context

    Returns:
        Enriched schema
    """
    try:
        strategy_data = get_strategy_with_auto_fallback(
            product_type=product_type,
            requirements=requirements or {},
            top_k=7,
            refinery=refinery
        )

        if strategy_data.get('success'):
            schema['procurement_strategy'] = {
                'preferred_vendors': strategy_data.get('preferred_vendors', []),
                'forbidden_vendors': strategy_data.get('forbidden_vendors', []),
                'neutral_vendors': strategy_data.get('neutral_vendors', []),
                'procurement_priorities': strategy_data.get('procurement_priorities', {}),
                'strategy_notes': strategy_data.get('strategy_notes', ''),
                'confidence': strategy_data.get('confidence', 0.0),
                'rag_type': strategy_data.get('rag_type', 'csv_filter'),
                'sources_used': strategy_data.get('sources_used', []),
                'category': strategy_data.get('category'),
                'subcategory': strategy_data.get('subcategory')
            }

            logger.info(f"[Schema Enrichment] Added CSV strategy for {product_type}: "
                       f"{len(strategy_data.get('preferred_vendors', []))} preferred, "
                       f"{len(strategy_data.get('neutral_vendors', []))} neutral")
        else:
            logger.warning(f"[Schema Enrichment] No strategy data for {product_type}")
            schema['procurement_strategy'] = {
                'error': strategy_data.get('error', 'No strategy available'),
                'preferred_vendors': [],
                'forbidden_vendors': [],
                'neutral_vendors': [],
                'confidence': 0.0
            }

        return schema

    except Exception as e:
        logger.error(f"[Schema Enrichment] Error: {e}")
        schema['procurement_strategy'] = {
            'error': str(e),
            'preferred_vendors': [],
            'forbidden_vendors': [],
            'neutral_vendors': [],
            'confidence': 0.0
        }
        return schema


# ============================================================================
# DEPRECATED FUNCTIONS (for backward compatibility)
# ============================================================================

def enrich_with_strategy_llm_fallback(
    product_type: str,
    requirements: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    DEPRECATED: LLM fallback is no longer needed with CSV-based approach.

    This function now just calls the main CSV filter.
    """
    logger.warning("[Strategy] enrich_with_strategy_llm_fallback is deprecated. "
                   "Using CSV-based filter instead.")
    return enrich_with_strategy_rag(
        product_type=product_type,
        requirements=requirements
    )


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("TESTING STRATEGY ENRICHMENT (CSV-BASED)")
    print("=" * 80)

    test_products = ["pressure transmitter", "flow meter", "control valve"]

    for product in test_products:
        print(f"\n[Test] Product: {product}")
        print("-" * 80)

        result = get_strategy_with_auto_fallback(product_type=product)

        print(f"  Success: {result.get('success')}")
        print(f"  RAG Type: {result.get('rag_type')}")
        print(f"  Category: {result.get('category')}")
        print(f"  Preferred: {result.get('preferred_vendors', [])[:3]}")
        print(f"  Neutral: {result.get('neutral_vendors', [])[:3]}")
        print(f"  Confidence: {result.get('confidence', 0):.2f}")
        print(f"  Sources: {result.get('sources_used', [])}")

    print("=" * 80)

    # Test vendor filtering
    print("\n[Test] Vendor Filtering")
    print("-" * 80)

    test_vendors = ["Emerson", "ABB", "Siemens", "Yokogawa", "Honeywell"]
    strategy_result = get_strategy_with_auto_fallback(product_type="pressure transmitter")

    filter_result = filter_vendors_by_strategy_data(
        vendors=test_vendors,
        strategy_data=strategy_result
    )

    print(f"  Accepted: {len(filter_result['accepted_vendors'])}")
    for v in filter_result['accepted_vendors'][:3]:
        print(f"    - {v['vendor']}: score={v['priority_score']}, preferred={v['is_preferred']}")

    print("=" * 80)
