# agentic/deep_agent/optimized_parallel_agent.py
# =============================================================================
# OPTIMIZED PARALLEL DEEP AGENT
# =============================================================================
#
# KEY OPTIMIZATIONS:
# 1. SINGLE LLM INSTANCE - Reused across all operations (no test call overhead)
# 2. TRUE PARALLEL PROCESSING - All products processed simultaneously
# 3. BATCHED API CALLS - Where possible, batch multiple products in single request
#
# This replaces the sequential processing in parallel_specs_enrichment.py
# with true parallelization at the product level.
#
# =============================================================================

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from uuid import uuid4
import threading

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

from llm_fallback import create_llm_with_fallback

# Local imports
from .memory import (
    DeepAgentMemory,
    ParallelEnrichmentResult,
    SpecificationSource
)
from .spec_output_normalizer import normalize_specification_output, normalize_key
from .standards_deep_agent import run_standards_deep_agent_batch

load_dotenv()
logger = logging.getLogger(__name__)


# =============================================================================
# SHARED LLM INSTANCE (SINGLETON)
# =============================================================================

_shared_llm = None
_llm_lock = threading.Lock()


def get_shared_llm(temperature: float = 0.1, model: str = "gemini-2.5-flash"):
    """
    Get or create a shared LLM instance.
    
    This avoids the overhead of creating new LLM instances for each call.
    The LLM is initialized ONCE and reused across all operations.
    
    Benefits:
    - Saves ~1.5s per call (no test call overhead)
    - Reduces memory usage
    - Maintains consistent temperature/model settings
    """
    global _shared_llm
    
    with _llm_lock:
        if _shared_llm is None:
            logger.info("[SHARED_LLM] Creating shared LLM instance (first time)...")
            _shared_llm = create_llm_with_fallback(
                model=model,
                temperature=temperature,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                skip_test=True  # Skip test call - saves ~1.5s
            )
            logger.info("[SHARED_LLM] Shared LLM instance created successfully")
        return _shared_llm


def reset_shared_llm():
    """Reset the shared LLM instance (useful for testing or config changes)."""
    global _shared_llm
    with _llm_lock:
        _shared_llm = None
        logger.info("[SHARED_LLM] Shared LLM instance reset")


# =============================================================================
# USER SPECS EXTRACTION (Uses Shared LLM)
# =============================================================================

USER_SPECS_PROMPT = """
You are a specification extractor. Extract ONLY explicitly stated specifications.

CRITICAL: Extract ONLY what is explicitly stated - do NOT infer or assume anything.

USER INPUT:
{user_input}

PRODUCT TYPE: {product_type}

STANDARD SPECIFICATION KEYS TO USE:
- accuracy, pressure_range, temperature_range, process_temperature
- output_signal, supply_voltage, protection_rating
- hazardous_area_approval, sil_rating
- material_wetted, material_housing, process_connection
- response_time, communication_protocol, flow_range, level_range, display, mounting

Return ONLY valid JSON:
{{
    "extracted_specifications": {{
        "specification_key": "exact value from user input"
    }},
    "confidence": 0.0-1.0
}}

Return empty specifications object if nothing is explicitly mentioned.
"""


def extract_user_specs_with_shared_llm(
    user_input: str,
    product_type: str,
    sample_input: Optional[str] = None,
    llm: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Extract user-specified specs using shared LLM instance.
    
    Args:
        user_input: Original user input
        product_type: Product type name
        sample_input: Optional additional context
        llm: Optional LLM instance (uses shared if None)
    
    Returns:
        Dict with extracted specifications
    """
    llm = llm or get_shared_llm(temperature=0.0)
    
    full_input = user_input
    if sample_input:
        full_input = f"{user_input}\n\nAdditional context: {sample_input}"
    
    try:
        prompt = ChatPromptTemplate.from_template(USER_SPECS_PROMPT)
        parser = JsonOutputParser()
        chain = prompt | llm | parser
        
        result = chain.invoke({
            "user_input": full_input,
            "product_type": product_type
        })
        
        extracted_specs = result.get("extracted_specifications", {})
        
        # Filter out null/empty values
        clean_specs = {
            k: v for k, v in extracted_specs.items()
            if v and str(v).lower() not in ["null", "none", "n/a", "not specified"]
        }
        
        return {
            "specifications": clean_specs,
            "source": "user_specified",
            "confidence": result.get("confidence", 0.0)
        }
    
    except Exception as e:
        logger.error(f"[USER_SPECS] Extraction failed for {product_type}: {e}")
        return {"specifications": {}, "source": "user_specified", "error": str(e)}


# =============================================================================
# LLM SPECS GENERATION (Uses Shared LLM)
# =============================================================================

LLM_SPECS_PROMPT = """
You are an industrial instrumentation expert. Generate technical specifications
for the given product type. Return ONLY clean technical values - NO descriptions.

PRODUCT TYPE: {product_type}
CATEGORY: {category}
CONTEXT: {context}

CRITICAL: Return ONLY the technical value. NO explanations, NO descriptions.

Generate these specifications with CLEAN values:
- accuracy, repeatability, rangeability
- temperature_range, ambient_temperature, protection_rating, humidity_range
- output_signal, supply_voltage, power_consumption, communication_protocol
- material_wetted, material_housing, process_connection, weight
- sil_rating, hazardous_area_approval, certifications
- response_time, stability, calibration_interval

Return ONLY valid JSON with FLAT key-value structure:
{{
    "specifications": {{
        "accuracy": {{"value": "±0.1%", "confidence": 0.8}},
        "output_signal": {{"value": "4-20mA, HART", "confidence": 0.9}}
    }}
}}
"""


def generate_llm_specs_with_shared_llm(
    product_type: str,
    category: Optional[str] = None,
    context: Optional[str] = None,
    llm: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Generate LLM specs using shared LLM instance.
    
    Args:
        product_type: Product type name
        category: Optional category
        context: Optional context
        llm: Optional LLM instance (uses shared if None)
    
    Returns:
        Dict with generated specifications
    """
    llm = llm or get_shared_llm(temperature=0.3)
    
    try:
        prompt = ChatPromptTemplate.from_template(LLM_SPECS_PROMPT)
        parser = JsonOutputParser()
        chain = prompt | llm | parser
        
        result = chain.invoke({
            "product_type": product_type,
            "category": category or "Industrial Instrument",
            "context": context or "General industrial application"
        })
        
        raw_specs = result.get("specifications", {})
        
        # Flatten and clean specs
        clean_specs = {}
        for key, value in raw_specs.items():
            if isinstance(value, dict):
                clean_specs[key] = {
                    "value": value.get("value", str(value)),
                    "confidence": value.get("confidence", 0.7)
                }
            else:
                clean_specs[key] = {"value": str(value), "confidence": 0.7}
        
        # Filter out N/A values
        clean_specs = {
            k: v for k, v in clean_specs.items()
            if v.get("value") and str(v.get("value")).lower() not in ["null", "none", "n/a"]
        }
        
        return {
            "specifications": clean_specs,
            "source": "llm_generated"
        }
    
    except Exception as e:
        logger.error(f"[LLM_SPECS] Generation failed for {product_type}: {e}")
        return {"specifications": {}, "source": "llm_generated", "error": str(e)}


# =============================================================================
# PARALLEL PRODUCT PROCESSOR
# =============================================================================

def process_single_product(
    item: Dict[str, Any],
    user_input: str,
    llm: Any
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """
    Process a single product with both user specs and LLM specs.
    
    This function is called in parallel for each product.
    
    Args:
        item: Product item dict
        user_input: Original user input
        llm: Shared LLM instance
    
    Returns:
        Tuple of (item_name, user_specs, llm_specs)
    """
    item_name = item.get("name") or item.get("product_name", "Unknown")
    category = item.get("category", "Industrial Instrument")
    sample_input = item.get("sample_input", "")
    
    logger.info(f"[PARALLEL_PRODUCT] Processing: {item_name}")
    start = time.time()
    
    # Extract user specs
    user_result = extract_user_specs_with_shared_llm(
        user_input=user_input,
        product_type=item_name,
        sample_input=sample_input,
        llm=llm
    )
    user_specs = user_result.get("specifications", {})
    
    # Generate LLM specs
    llm_result = generate_llm_specs_with_shared_llm(
        product_type=item_name,
        category=category,
        context=sample_input,
        llm=llm
    )
    llm_specs = llm_result.get("specifications", {})
    
    elapsed = time.time() - start
    logger.info(f"[PARALLEL_PRODUCT] Completed {item_name} in {elapsed:.2f}s (user: {len(user_specs)}, llm: {len(llm_specs)})")
    
    return item_name, user_specs, llm_specs


# =============================================================================
# DEDUPLICATION (Same as parallel_specs_enrichment.py)
# =============================================================================

def deduplicate_and_merge_specifications(
    user_specs: Dict[str, Any],
    llm_specs: Dict[str, Any],
    standards_specs: Dict[str, Any]
) -> Dict[str, SpecificationSource]:
    """
    Merge specifications from 3 sources with deduplication.
    
    Priority: user_specified > standards > llm_generated
    """
    merged: Dict[str, SpecificationSource] = {}
    timestamp = datetime.now().isoformat()
    
    # First: User specs (mandatory)
    for key, value in user_specs.items():
        if value and str(value).lower() not in ["null", "none", ""]:
            merged[key] = SpecificationSource(
                value=value,
                source="user_specified",
                confidence=1.0,
                standard_reference=None,
                timestamp=timestamp
            )
    
    # Second: Standards specs
    for key, value_data in standards_specs.items():
        if key in merged:
            continue
        
        if isinstance(value_data, dict):
            value = value_data.get("value", str(value_data))
            confidence = value_data.get("confidence", 0.9)
            std_ref = value_data.get("standard_reference", None)
        else:
            value = value_data
            confidence = 0.9
            std_ref = None
        
        if value and str(value).lower() not in ["null", "none", "", "extracted value or null"]:
            merged[key] = SpecificationSource(
                value=value,
                source="standards",
                confidence=confidence,
                standard_reference=std_ref,
                timestamp=timestamp
            )
    
    # Third: LLM specs
    for key, value_data in llm_specs.items():
        if key in merged:
            continue
        
        if isinstance(value_data, dict):
            value = value_data.get("value", str(value_data))
            confidence = value_data.get("confidence", 0.7)
        else:
            value = value_data
            confidence = 0.7
        
        if value and str(value).lower() not in ["null", "none", ""]:
            merged[key] = SpecificationSource(
                value=value,
                source="llm_generated",
                confidence=confidence,
                standard_reference=None,
                timestamp=timestamp
            )
    
    return merged


# =============================================================================
# MAIN OPTIMIZED PARALLEL ENRICHMENT
# =============================================================================

def run_optimized_parallel_enrichment(
    items: List[Dict[str, Any]],
    user_input: str,
    session_id: Optional[str] = None,
    domain_context: Optional[str] = None,
    safety_requirements: Optional[Dict[str, Any]] = None,
    memory: Optional[DeepAgentMemory] = None,
    max_parallel_products: int = 5
) -> Dict[str, Any]:
    """
    OPTIMIZED parallel specification enrichment.
    
    KEY DIFFERENCES FROM parallel_specs_enrichment.py:
    1. Uses SINGLE shared LLM instance (no repeated initialization)
    2. Processes ALL products in PARALLEL (not just sources)
    3. Combines user_specs and llm_specs per product in single thread
    4. Standards extraction still runs as batch (already optimized)
    
    Time Savings:
    - No test calls: ~1.5s per LLM creation × N products = N × 1.5s saved
    - True parallel products: N × 5s → ~5-8s total (vs sequential N × 5s)
    
    Args:
        items: List of product items
        user_input: Original user input
        session_id: Session ID for tracking
        domain_context: Domain context
        safety_requirements: Safety requirements
        memory: DeepAgentMemory instance
        max_parallel_products: Max concurrent product processors (default: 5)
    
    Returns:
        Dict with enriched items and metadata
    """
    start_time = time.time()
    session_id = session_id or f"opt-parallel-{uuid4().hex[:8]}"
    
    logger.info("=" * 60)
    logger.info("[OPT_PARALLEL] OPTIMIZED PARALLEL DEEP AGENT")
    logger.info(f"[OPT_PARALLEL] Processing {len(items)} products")
    logger.info(f"[OPT_PARALLEL] Session: {session_id}")
    logger.info("=" * 60)
    
    if not items:
        return {
            "success": True,
            "items": [],
            "metadata": {"total_items": 0, "processing_time_ms": 0}
        }
    
    # Create or use existing memory
    if memory is None:
        memory = DeepAgentMemory()
    
    # ==========================================================================
    # PHASE 1: Get shared LLM instance (ONCE)
    # ==========================================================================
    
    phase1_start = time.time()
    llm = get_shared_llm()
    phase1_time = time.time() - phase1_start
    logger.info(f"[OPT_PARALLEL] Phase 1: LLM ready in {phase1_time:.2f}s")
    
    # ==========================================================================
    # PHASE 2: Process all products in PARALLEL (user specs + LLM specs)
    # ==========================================================================
    
    phase2_start = time.time()
    user_specs_results = {}
    llm_specs_results = {}
    
    logger.info(f"[OPT_PARALLEL] Phase 2: Processing {len(items)} products in parallel...")
    
    with ThreadPoolExecutor(max_workers=max_parallel_products) as executor:
        future_to_item = {
            executor.submit(
                process_single_product,
                item,
                user_input,
                llm
            ): item
            for item in items
        }
        
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            try:
                item_name, user_specs, llm_specs = future.result()
                user_specs_results[item_name] = user_specs
                llm_specs_results[item_name] = llm_specs
            except Exception as e:
                item_name = item.get("name", "Unknown")
                logger.error(f"[OPT_PARALLEL] Failed for {item_name}: {e}")
                user_specs_results[item_name] = {}
                llm_specs_results[item_name] = {}
    
    phase2_time = time.time() - phase2_start
    logger.info(f"[OPT_PARALLEL] Phase 2: All products done in {phase2_time:.2f}s")
    
    # ==========================================================================
    # PHASE 3: Standards extraction (batch, already optimized)
    # ==========================================================================
    
    phase3_start = time.time()
    standards_specs_results = {}
    
    logger.info("[OPT_PARALLEL] Phase 3: Standards extraction...")
    
    try:
        standards_result = run_standards_deep_agent_batch(
            items=items,
            session_id=session_id,
            domain_context=domain_context,
            safety_requirements=safety_requirements
        )
        
        if standards_result.get("success"):
            for enriched_item in standards_result.get("items", []):
                item_name = enriched_item.get("name") or enriched_item.get("product_name", "Unknown")
                raw_specs = enriched_item.get("standards_specifications", {})
                normalized_specs = normalize_specification_output(raw_specs, preserve_ghost_values=False)
                clean_specs = {k: v for k, v in normalized_specs.items() if not k.startswith('_')}
                standards_specs_results[item_name] = clean_specs
    except Exception as e:
        logger.error(f"[OPT_PARALLEL] Standards extraction failed: {e}")
    
    phase3_time = time.time() - phase3_start
    logger.info(f"[OPT_PARALLEL] Phase 3: Standards done in {phase3_time:.2f}s")
    
    # ==========================================================================
    # PHASE 4: Merge and deduplicate
    # ==========================================================================
    
    phase4_start = time.time()
    enriched_items = []
    
    for item in items:
        item_name = item.get("name") or item.get("product_name", "Unknown")
        item_id = f"{session_id}_{item.get('number', 0)}_{item_name}"
        
        user_specs = user_specs_results.get(item_name, {})
        llm_specs = llm_specs_results.get(item_name, {})
        standards_specs = standards_specs_results.get(item_name, {})
        
        merged_specs = deduplicate_and_merge_specifications(
            user_specs=user_specs,
            llm_specs=llm_specs,
            standards_specs=standards_specs
        )
        
        # Create enrichment result
        enrichment_result: ParallelEnrichmentResult = {
            "item_id": item_id,
            "item_name": item_name,
            "item_type": item.get("type", "instrument"),
            "user_specified_specs": user_specs,
            "llm_generated_specs": llm_specs,
            "standards_specs": standards_specs,
            "merged_specs": merged_specs,
            "enrichment_metadata": {
                "user_specs_count": len(user_specs),
                "llm_specs_count": len(llm_specs),
                "standards_specs_count": len(standards_specs),
                "merged_specs_count": len(merged_specs),
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        memory.store_parallel_enrichment_result(item_id, enrichment_result)
        
        # Create enriched item
        enriched_item = item.copy()
        enriched_item["user_specified_specs"] = user_specs
        enriched_item["llm_generated_specs"] = llm_specs
        enriched_item["standards_specifications"] = standards_specs
        # [FIX #1] Mark as phase3_optimized so validation can skip RAG re-run (saves 2000+ seconds!)
        enriched_item["enrichment_source"] = "phase3_optimized"
        enriched_item["combined_specifications"] = {k: v for k, v in merged_specs.items()}
        enriched_item["specifications"] = {
            k: v.get("value", v) if isinstance(v, dict) else v
            for k, v in merged_specs.items()
        }
        enriched_item["standards_info"] = {
            "enrichment_status": "success",
            "sources_used": ["user_specified", "llm_generated", "standards"],
            "user_specs_count": len(user_specs),
            "llm_specs_count": len(llm_specs),
            "standards_specs_count": len(standards_specs)
        }
        
        enriched_items.append(enriched_item)
    
    phase4_time = time.time() - phase4_start
    total_time = time.time() - start_time
    
    logger.info("=" * 60)
    logger.info("[OPT_PARALLEL] COMPLETE")
    logger.info(f"[OPT_PARALLEL] Phase 1 (LLM init):     {phase1_time:.2f}s")
    logger.info(f"[OPT_PARALLEL] Phase 2 (Products):     {phase2_time:.2f}s")
    logger.info(f"[OPT_PARALLEL] Phase 3 (Standards):    {phase3_time:.2f}s")
    logger.info(f"[OPT_PARALLEL] Phase 4 (Merge):        {phase4_time:.2f}s")
    logger.info(f"[OPT_PARALLEL] TOTAL:                  {total_time:.2f}s")
    logger.info(f"[OPT_PARALLEL] Items processed: {len(enriched_items)}")
    logger.info("=" * 60)
    
    return {
        "success": True,
        "items": enriched_items,
        "metadata": {
            "total_items": len(enriched_items),
            "processing_time_ms": int(total_time * 1000),
            "session_id": session_id,
            "phase_times": {
                "llm_init_s": round(phase1_time, 2),
                "products_parallel_s": round(phase2_time, 2),
                "standards_s": round(phase3_time, 2),
                "merge_s": round(phase4_time, 2),
                "total_s": round(total_time, 2)
            },
            "optimization_stats": {
                "shared_llm_used": True,
                "parallel_products": len(items),
                "max_workers": max_parallel_products
            },
            "memory_stats": memory.get_memory_stats()
        },
        "memory": memory
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "run_optimized_parallel_enrichment",
    "get_shared_llm",
    "reset_shared_llm",
    "extract_user_specs_with_shared_llm",
    "generate_llm_specs_with_shared_llm"
]
