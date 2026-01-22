# tools/specs_filter.py
# Specifications Filter for Index RAG (Layer 2)
# Applies Standards RAG and Strategy RAG constraints

import logging
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# ============================================================================
# STANDARDS FILTER
# ============================================================================

def apply_standards_filter(
    products: List[Dict],
    requirements: Dict[str, Any] = None,
    top_k: int = 7
) -> Dict[str, Any]:
    """
    Apply Standards RAG constraints to filter products.
    
    Filters based on:
    - SIL levels (Safety Integrity Level)
    - ATEX/IECEx certifications
    - API standards compliance
    - Communication protocols
    
    Args:
        products: List of product dicts from parallel indexing
        requirements: User's structured requirements
        top_k: Number of documents to retrieve from Standards RAG
        
    Returns:
        Filtered products with standards metadata
    """
    logger.info(f"[SpecsFilter] Applying standards filter to {len(products)} products...")
    
    if not products:
        return {
            "success": True,
            "filtered_products": [],
            "standards_applied": False,
            "filter_count": 0
        }
    
    try:
        from tools.standards_enrichment_tool import get_applicable_standards
        
        # Extract product type from first product
        product_type = None
        for product in products:
            pt = product.get("product_type") or product.get("data", {}).get("product_type")
            if pt:
                product_type = pt
                break
        
        if not product_type:
            logger.warning("[SpecsFilter] No product type found - skipping standards filter")
            return {
                "success": True,
                "filtered_products": products,
                "standards_applied": False,
                "reason": "No product type detected"
            }
        
        # Get applicable standards
        standards_result = get_applicable_standards(
            product_type=product_type,
            requirements=requirements or {},
            top_k=top_k
        )
        
        if not standards_result.get("success"):
            logger.warning(f"[SpecsFilter] Standards RAG failed: {standards_result.get('error')}")
            return {
                "success": True,
                "filtered_products": products,
                "standards_applied": False,
                "reason": standards_result.get("error")
            }
        
        # Extract standards constraints
        required_sil = standards_result.get("sil_level")
        required_certs = standards_result.get("certifications", [])
        required_protocols = standards_result.get("communication_protocols", [])
        
        logger.info(f"[SpecsFilter] Standards constraints: SIL={required_sil}, Certs={required_certs}")
        
        # Filter products based on standards (lenient - just add metadata)
        filtered_products = []
        for product in products:
            product_data = product.get("data", {})
            
            # Check SIL compliance
            product_sil = product_data.get("sil_level") or product_data.get("safety_level")
            sil_compliant = True
            if required_sil and product_sil:
                try:
                    sil_compliant = int(product_sil) >= int(required_sil)
                except (ValueError, TypeError):
                    sil_compliant = True  # Can't compare, assume compliant
            
            # Check certification compliance  
            product_certs = product_data.get("certifications", [])
            if isinstance(product_certs, str):
                product_certs = [product_certs]
            
            cert_compliant = True
            if required_certs:
                # Check if any required cert is present
                for req_cert in required_certs:
                    if any(req_cert.lower() in str(c).lower() for c in product_certs):
                        cert_compliant = True
                        break
            
            # Add standards metadata
            enriched_product = product.copy()
            enriched_product["standards_compliance"] = {
                "sil_compliant": sil_compliant,
                "cert_compliant": cert_compliant,
                "required_sil": required_sil,
                "required_certs": required_certs,
                "product_sil": product_sil,
                "product_certs": product_certs
            }
            
            # Calculate compliance score
            compliance_score = 1.0
            if not sil_compliant:
                compliance_score -= 0.3
            if not cert_compliant:
                compliance_score -= 0.2
            enriched_product["compliance_score"] = max(0.0, compliance_score)
            
            filtered_products.append(enriched_product)
        
        # Sort by compliance score
        filtered_products.sort(key=lambda x: x.get("compliance_score", 0), reverse=True)
        
        result = {
            "success": True,
            "filtered_products": filtered_products,
            "standards_applied": True,
            "filter_count": len(filtered_products),
            "standards_context": {
                "required_sil": required_sil,
                "required_certs": required_certs,
                "required_protocols": required_protocols
            }
        }
        
        logger.info(f"[SpecsFilter] Standards filter complete: {len(filtered_products)} products")
        return result
        
    except ImportError as ie:
        logger.warning(f"[SpecsFilter] Standards module not available: {ie}")
        return {
            "success": True,
            "filtered_products": products,
            "standards_applied": False,
            "reason": str(ie)
        }
    except Exception as e:
        logger.error(f"[SpecsFilter] Standards filter error: {e}", exc_info=True)
        return {
            "success": False,
            "filtered_products": products,
            "standards_applied": False,
            "error": str(e)
        }


# ============================================================================
# STRATEGY FILTER
# ============================================================================

def apply_strategy_filter(
    products: List[Dict],
    product_type: str = None,
    requirements: Dict[str, Any] = None,
    top_k: int = 7
) -> Dict[str, Any]:
    """
    Apply Strategy RAG constraints to filter/prioritize products.
    
    Filters based on:
    - Preferred vendors (boost priority)
    - Forbidden vendors (exclude)
    - Procurement priorities
    
    Args:
        products: List of product dicts
        product_type: Product type for strategy lookup
        requirements: User's structured requirements
        top_k: Number of documents to retrieve from Strategy RAG
        
    Returns:
        Filtered and prioritized products with strategy metadata
    """
    logger.info(f"[SpecsFilter] Applying strategy filter to {len(products)} products...")
    
    if not products:
        return {
            "success": True,
            "filtered_products": [],
            "strategy_applied": False,
            "excluded_count": 0
        }
    
    try:
        # FIX: Import from correct path (strategy_rag subdirectory, not stub)
        from agentic.strategy_rag.strategy_rag_enrichment import (
            get_strategy_with_auto_fallback,
            filter_vendors_by_strategy
        )
        
        # Get product type if not provided
        if not product_type:
            for product in products:
                pt = product.get("product_type") or product.get("data", {}).get("product_type")
                if pt:
                    product_type = pt
                    break
        
        if not product_type:
            logger.warning("[SpecsFilter] No product type - skipping strategy filter")
            return {
                "success": True,
                "filtered_products": products,
                "strategy_applied": False,
                "reason": "No product type detected"
            }
        
        # Get strategy context
        strategy_result = get_strategy_with_auto_fallback(
            product_type=product_type,
            requirements=requirements or {},
            top_k=top_k
        )
        
        if not strategy_result.get("success"):
            logger.warning(f"[SpecsFilter] Strategy RAG failed: {strategy_result.get('error')}")
            return {
                "success": True,
                "filtered_products": products,
                "strategy_applied": False,
                "reason": strategy_result.get("error")
            }
        
        # Extract strategy constraints
        preferred_vendors = [v.lower() for v in strategy_result.get("preferred_vendors", [])]
        forbidden_vendors = [v.lower() for v in strategy_result.get("forbidden_vendors", [])]
        priorities = strategy_result.get("procurement_priorities", {})
        
        logger.info(f"[SpecsFilter] Strategy constraints: "
                   f"Preferred={preferred_vendors}, Forbidden={forbidden_vendors}")
        
        # Filter and prioritize products
        filtered_products = []
        excluded_products = []
        
        for product in products:
            vendor = product.get("vendor", "").lower()
            
            # Check if vendor is forbidden
            is_forbidden = any(fv in vendor or vendor in fv for fv in forbidden_vendors)
            
            if is_forbidden:
                product["excluded_reason"] = "Forbidden by strategy"
                excluded_products.append(product)
                continue
            
            # Check if vendor is preferred
            is_preferred = any(pv in vendor or vendor in pv for pv in preferred_vendors)
            
            # Get priority score
            priority_score = 0
            for pv, score in priorities.items():
                if pv.lower() in vendor or vendor in pv.lower():
                    priority_score = score
                    break
            
            # Boost preferred vendors
            if is_preferred:
                priority_score += 7
            
            # Add strategy metadata
            enriched_product = product.copy()
            enriched_product["strategy_context"] = {
                "is_preferred": is_preferred,
                "is_forbidden": False,
                "priority_score": priority_score,
                "rag_type": strategy_result.get("rag_type")
            }
            enriched_product["strategy_priority"] = priority_score
            
            filtered_products.append(enriched_product)
        
        # Sort by strategy priority
        filtered_products.sort(key=lambda x: x.get("strategy_priority", 0), reverse=True)
        
        result = {
            "success": True,
            "filtered_products": filtered_products,
            "excluded_products": excluded_products,
            "strategy_applied": True,
            "filter_count": len(filtered_products),
            "excluded_count": len(excluded_products),
            "strategy_context": {
                "preferred_vendors": preferred_vendors,
                "forbidden_vendors": forbidden_vendors,
                "rag_type": strategy_result.get("rag_type"),
                "confidence": strategy_result.get("confidence", 0.0)
            }
        }
        
        logger.info(f"[SpecsFilter] Strategy filter complete: "
                   f"{len(filtered_products)} accepted, {len(excluded_products)} excluded")
        
        return result
        
    except ImportError as ie:
        logger.warning(f"[SpecsFilter] Strategy module not available: {ie}")
        return {
            "success": True,
            "filtered_products": products,
            "strategy_applied": False,
            "reason": str(ie)
        }
    except Exception as e:
        logger.error(f"[SpecsFilter] Strategy filter error: {e}", exc_info=True)
        return {
            "success": False,
            "filtered_products": products,
            "strategy_applied": False,
            "error": str(e)
        }


# ============================================================================
# COMBINED FILTER
# ============================================================================

def apply_specs_and_strategy_filter(
    products: List[Dict],
    product_type: str = None,
    requirements: Dict[str, Any] = None,
    top_k: int = 7
) -> Dict[str, Any]:
    """
    Apply both Standards and Strategy filters (Layer 2).
    
    Order:
    1. Standards filter (SIL, certifications)
    2. Strategy filter (vendor preferences)
    
    Args:
        products: List of product dicts from parallel indexing
        product_type: Product type for RAG lookups
        requirements: User's structured requirements
        top_k: Number of RAG documents to retrieve
        
    Returns:
        Fully filtered and enriched products
    """
    logger.info(f"[SpecsFilter] Applying combined Layer 2 filter to {len(products)} products...")
    
    # Step 1: Apply Standards filter
    standards_result = apply_standards_filter(
        products=products,
        requirements=requirements,
        top_k=top_k
    )
    
    standards_products = standards_result.get("filtered_products", products)
    
    # Step 2: Apply Strategy filter
    strategy_result = apply_strategy_filter(
        products=standards_products,
        product_type=product_type,
        requirements=requirements,
        top_k=top_k
    )
    
    final_products = strategy_result.get("filtered_products", standards_products)
    
    # Combine metadata
    result = {
        "success": True,
        "filtered_products": final_products,
        "filter_count": len(final_products),
        "standards": {
            "applied": standards_result.get("standards_applied", False),
            "context": standards_result.get("standards_context")
        },
        "strategy": {
            "applied": strategy_result.get("strategy_applied", False),
            "excluded_count": strategy_result.get("excluded_count", 0),
            "context": strategy_result.get("strategy_context")
        }
    }
    
    logger.info(f"[SpecsFilter] Layer 2 filter complete: {len(final_products)} products remain")
    
    return result


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'apply_standards_filter',
    'apply_strategy_filter',
    'apply_specs_and_strategy_filter'
]
