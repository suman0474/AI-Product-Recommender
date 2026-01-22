# agentic/deep_agent/llm_specs_generator.py
# =============================================================================
# LLM SPECIFICATION GENERATOR WITH DYNAMIC KEY DISCOVERY
# =============================================================================
#
# Generates all possible specifications for a product type using LLM.
# These specs fill in gaps not covered by user-specified or standards specs.
#
# FEATURES:
# 1. DYNAMIC KEY DISCOVERY: Uses reasoning LLM to discover relevant spec keys
#    for unknown/novel product types (replaces hardcoded schema fields)
# 2. ITERATIVE GENERATION: If specs count < MIN_LLM_SPECS_COUNT (30), the generator
#    will iterate and request additional specs until the minimum is reached.
# 3. PARALLEL PROCESSING: Uses multiple workers for faster spec generation.
#
# =============================================================================

import logging
import os
import time
import threading
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

from llm_fallback import create_llm_with_fallback
from ..llm_manager import get_cached_llm
from prompts_library import load_prompt

load_dotenv()
logger = logging.getLogger(__name__)

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Model for standard LLM spec generation (fast)
LLM_SPECS_MODEL = "gemini-2.5-flash"

# Model for deep reasoning / key discovery (more capable)
REASONING_MODEL = "gemini-2.5-pro"

# Singleton instances for efficiency
_specs_llm = None
_reasoning_llm = None
_llm_lock = threading.Lock()


def _get_specs_llm():
    """Get singleton specs generation LLM (Gemini Flash for speed)."""
    global _specs_llm
    if _specs_llm is None:
        with _llm_lock:
            if _specs_llm is None:
                logger.info("[LLM_SPECS] Initializing specs LLM (Gemini Flash)...")
                _specs_llm = create_llm_with_fallback(
                    model=LLM_SPECS_MODEL,
                    temperature=0.3,
                    google_api_key=os.getenv("GOOGLE_API_KEY")
                )
                
                # Register with GlobalResourceRegistry
                from agentic.context_managers import GlobalResourceRegistry, ResourceMetrics, ResourceStatus
                registry = GlobalResourceRegistry()
                registry.register(
                    f"llm_specs_{id(_specs_llm)}",
                    ResourceMetrics(
                        acquired_at=time.time(),
                        resource_type="llm_specs",
                        resource_id=LLM_SPECS_MODEL,
                        status=ResourceStatus.ACTIVE
                    )
                )
    return _specs_llm


def _get_reasoning_llm():
    """Get singleton reasoning LLM (Gemini Pro for deep thinking)."""
    global _reasoning_llm
    if _reasoning_llm is None:
        with _llm_lock:
            if _reasoning_llm is None:
                logger.info("[LLM_SPECS] Initializing reasoning LLM (Gemini Pro)...")
                _reasoning_llm = create_llm_with_fallback(
                    model=REASONING_MODEL,
                    temperature=0.2,  # Low for precise reasoning
                    google_api_key=os.getenv("GOOGLE_API_KEY")
                )
    return _reasoning_llm


# =============================================================================
# ITERATIVE GENERATION CONFIGURATION
# =============================================================================

# Minimum number of specifications that LLM must generate
MIN_LLM_SPECS_COUNT = 30

# Maximum iterations to prevent infinite loops
MAX_LLM_ITERATIONS = 5

# Specs to request per iteration when below minimum
SPECS_PER_ITERATION = 15

# =============================================================================
# PARALLEL PROCESSING CONFIGURATION
# =============================================================================

# Maximum parallel workers for iterative spec generation
MAX_PARALLEL_WORKERS = 4

# Whether to use parallel processing for iterations
ENABLE_PARALLEL_ITERATIONS = True

# Whether to enable dynamic key discovery for unknown product types
ENABLE_DYNAMIC_DISCOVERY = True


# =============================================================================
# DYNAMIC KEY DISCOVERY PROMPT (for unknown product types)
# =============================================================================

KEY_DISCOVERY_PROMPT = load_prompt("key_discovery_prompt")


def discover_specification_keys(
    product_type: str,
    category: Optional[str] = None,
    context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Use deep reasoning to discover relevant specification keys for a product type.
    
    This function uses Gemini Pro for intelligent key discovery, replacing 
    hardcoded PRODUCT_TYPE_SCHEMA_FIELDS with dynamic discovery that works
    for ANY product type.
    
    Args:
        product_type: The product type to discover specs for
        category: Optional category for context
        context: Optional additional context
    
    Returns:
        Dict containing discovered keys organized by importance:
            - mandatory_keys: Critical specs that must be specified
            - optional_keys: Nice-to-have specs
            - safety_keys: Safety-critical specs
            - all_keys: Combined list of all keys
            - product_analysis: Analysis of the product type
    """
    logger.info(f"[KEY_DISCOVERY] Discovering specs for: {product_type}")
    start_time = time.time()
    
    try:
        llm = _get_reasoning_llm()
        prompt = ChatPromptTemplate.from_template(KEY_DISCOVERY_PROMPT)
        parser = JsonOutputParser()
        chain = prompt | llm | parser
        
        result = chain.invoke({
            "product_type": product_type,
            "category": category or "Industrial Instrumentation",
            "context": context or "General industrial application"
        })
        
        elapsed = time.time() - start_time
        logger.info(f"[KEY_DISCOVERY] Completed in {elapsed:.2f}s")
        
        # Extract discovered keys
        spec_keys = result.get("specification_keys", {})
        mandatory = [item.get("key") for item in spec_keys.get("mandatory", []) if item.get("key")]
        optional = [item.get("key") for item in spec_keys.get("optional", []) if item.get("key")]
        safety = [item.get("key") for item in spec_keys.get("safety_critical", []) if item.get("key")]
        
        total_keys = len(mandatory) + len(optional) + len(safety)
        logger.info(f"[KEY_DISCOVERY] Found {len(mandatory)} mandatory, {len(optional)} optional, {len(safety)} safety keys (total: {total_keys})")
        
        return {
            "success": True,
            "product_type": product_type,
            "product_analysis": result.get("product_analysis", {}),
            "mandatory_keys": mandatory,
            "optional_keys": optional,
            "safety_keys": safety,
            "all_keys": mandatory + optional + safety,
            "key_details": spec_keys,
            "discovery_confidence": result.get("discovery_confidence", 0.8),
            "reasoning_notes": result.get("reasoning_notes", ""),
            "discovery_time_ms": int(elapsed * 1000)
        }
        
    except Exception as e:
        logger.error(f"[KEY_DISCOVERY] Failed: {e}")
        # Fallback to minimal generic keys
        return {
            "success": False,
            "product_type": product_type,
            "mandatory_keys": ["temperature_range", "material_housing", "certifications", "accuracy", "output_signal"],
            "optional_keys": ["weight", "protection_rating", "dimensions", "power_consumption"],
            "safety_keys": ["hazardous_area_approval", "sil_rating"],
            "all_keys": ["temperature_range", "material_housing", "certifications", "accuracy", 
                        "output_signal", "weight", "protection_rating", "dimensions", 
                        "power_consumption", "hazardous_area_approval", "sil_rating"],
            "error": str(e),
            "discovery_time_ms": int((time.time() - start_time) * 1000)
        }


# =============================================================================
# GENERATION PROMPT - Loaded from prompts_library
# =============================================================================

LLM_SPECS_GENERATION_PROMPT = load_prompt("llm_specs_generation_prompt")


# =============================================================================
# ITERATIVE GENERATION PROMPT - Loaded from prompts_library
# =============================================================================

LLM_SPECS_ITERATIVE_PROMPT = load_prompt("llm_specs_iterative_prompt")


# =============================================================================
# GENERATION FUNCTION
# =============================================================================

def _flatten_and_clean_specs(raw_specs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Helper function to flatten and clean specifications.

    Args:
        raw_specs: Raw specifications from LLM response

    Returns:
        Cleaned and flattened specifications dict
    """
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
        if v.get("value") and str(v.get("value")).lower() not in ["null", "none", "n/a", ""]
    }

    return clean_specs


def _generate_additional_specs(
    llm,
    product_type: str,
    category: str,
    context: str,
    existing_specs: Dict[str, Any],
    specs_needed: int
) -> Dict[str, Any]:
    """
    Generate additional specifications using iterative prompt.

    Args:
        llm: The LLM instance
        product_type: Product type to generate specs for
        category: Product category
        context: Additional context
        existing_specs: Already generated specifications
        specs_needed: Number of additional specs needed

    Returns:
        Dict of new specifications
    """
    # Format existing specs for the prompt
    existing_specs_list = "\n".join([
        f"- {key}: {value.get('value', value)}"
        for key, value in existing_specs.items()
    ])

    prompt = ChatPromptTemplate.from_template(LLM_SPECS_ITERATIVE_PROMPT)
    parser = JsonOutputParser()
    chain = prompt | llm | parser

    result = chain.invoke({
        "product_type": product_type,
        "category": category,
        "context": context,
        "existing_specs": existing_specs_list,
        "specs_needed": specs_needed
    })

    raw_specs = result.get("specifications", {})
    return _flatten_and_clean_specs(raw_specs)


def _parallel_generate_specs_worker(
    worker_id: int,
    product_type: str,
    category: str,
    context: str,
    existing_specs: Dict[str, Any],
    focus_area: str,
    specs_needed: int
) -> Tuple[int, Dict[str, Any]]:
    """
    Worker function for parallel spec generation by focus area.

    Each worker specializes in a specific specification category to enable
    diverse and comprehensive spec generation.

    Args:
        worker_id: Unique worker ID
        product_type: Product type
        category: Product category
        context: Additional context
        existing_specs: Already generated specs
        focus_area: Specification focus area (e.g., "electrical", "physical")
        specs_needed: Target number of specs

    Returns:
        Tuple of (worker_id, new_specs_dict)
    """
    logger.info(f"[WORKER-{worker_id}] Starting parallel spec generation for focus area: {focus_area}")

    try:
        # Create LLM instance for this thread
        llm = get_cached_llm(
            model=LLM_SPECS_MODEL,
            temperature=0.4  # Slightly higher for diversity
        )

        # Specialized prompt for this worker's focus area
        existing_specs_list = "\n".join([
            f"- {key}: {value.get('value', value) if isinstance(value, dict) else value}"
            for key, value in existing_specs.items()
        ])

        focus_prompt = f"""
You are an expert in {focus_area} specifications for industrial instrumentation.

Your ONLY task is to generate NEW {focus_area.upper()} specifications for {product_type} that are NOT in the existing list.

PRODUCT TYPE: {product_type}
CATEGORY: {category}
CONTEXT: {context}

EXISTING SPECIFICATIONS (DO NOT REPEAT):
{existing_specs_list}

=== GENERATE {specs_needed} NEW {focus_area.upper()} SPECIFICATIONS ===

Focus ONLY on {focus_area} aspects:
- {focus_area}_param_1, {focus_area}_param_2, {focus_area}_param_3, etc.

Return ONLY valid JSON:
{{{{
    "specifications": {{{{
        "new_spec_1": {{{{"value": "clean value", "confidence": 0.8}}}},
        "new_spec_2": {{{{"value": "clean value", "confidence": 0.7}}}}
    }}}},
    "focus_area": "{focus_area}"
}}}}

Do NOT repeat any existing specs! Generate ONLY NEW specifications.
"""

        prompt = ChatPromptTemplate.from_template(focus_prompt)
        parser = JsonOutputParser()
        chain = prompt | llm | parser

        result = chain.invoke({
            "product_type": product_type,
            "category": category,
            "context": context
        })

        new_specs = _flatten_and_clean_specs(result.get("specifications", {}))
        logger.info(f"[WORKER-{worker_id}] Generated {len(new_specs)} specs for focus area: {focus_area}")

        return (worker_id, new_specs)

    except Exception as e:
        logger.error(f"[WORKER-{worker_id}] Error in parallel generation: {e}")
        return (worker_id, {})


def _generate_specs_parallel_by_focus(
    product_type: str,
    category: str,
    context: str,
    existing_specs: Dict[str, Any],
    specs_needed: int
) -> Dict[str, Any]:
    """
    Generate specifications in parallel by focus areas.

    PARALLELIZATION STRATEGY:
    - Divides spec generation into multiple focus areas
    - Each worker generates specs for one focus area concurrently
    - Combines results at the end
    - Achieves ~4x speedup with 4 workers

    Args:
        product_type: Product type
        category: Product category
        context: Additional context
        existing_specs: Already generated specs
        specs_needed: Total specs needed across all workers

    Returns:
        Dict of specifications from all workers
    """
    if not ENABLE_PARALLEL_ITERATIONS:
        # Fallback to sequential generation
        return _generate_additional_specs(
            create_llm_with_fallback(model=LLM_SPECS_MODEL, temperature=0.3,
                                    google_api_key=os.getenv("GOOGLE_API_KEY")),
            product_type, category, context, existing_specs, specs_needed
        )

    # Focus areas for parallel generation
    focus_areas = [
        "Physical & Mechanical",
        "Electrical & Power",
        "Environmental & Safety",
        "Performance & Operational"
    ]

    logger.info(f"[PARALLEL] Generating specs with {len(focus_areas)} parallel workers...")

    all_specs: Dict[str, Any] = {}
    worker_specs_per_focus = specs_needed // len(focus_areas)

    # Execute parallel workers
    with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_WORKERS, len(focus_areas))) as executor:
        future_to_worker = {
            executor.submit(
                _parallel_generate_specs_worker,
                i, product_type, category, context,
                existing_specs, focus_area, worker_specs_per_focus
            ): i for i, focus_area in enumerate(focus_areas)
        }

        for future in as_completed(future_to_worker):
            try:
                worker_id, new_specs = future.result()
                all_specs.update(new_specs)
                logger.info(f"[PARALLEL] Worker {worker_id} returned {len(new_specs)} specs")
            except Exception as e:
                logger.error(f"[PARALLEL] Worker exception: {e}")

    logger.info(f"[PARALLEL] All workers completed, total new specs: {len(all_specs)}")
    return all_specs


def generate_llm_specs(
    product_type: str,
    category: Optional[str] = None,
    context: Optional[str] = None,
    min_specs: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generate specifications for a product type using LLM with ITERATIVE LOOP.

    This function ensures a minimum number of specifications are generated.
    If the initial generation produces fewer than MIN_LLM_SPECS_COUNT (30),
    it will iterate and request additional specs until the minimum is reached.

    Args:
        product_type: The product type to generate specs for
        category: Optional category for context
        context: Optional additional context (e.g., sample_input)
        min_specs: Optional minimum specs count (defaults to MIN_LLM_SPECS_COUNT)

    Returns:
        Dict with:
            - specifications: Dict of generated specs with values and confidence
            - source: "llm_generated"
            - generation_notes: Notes on generation
            - iterations: Number of iterations performed
            - specs_count: Total number of specifications generated
    """
    min_required = min_specs if min_specs is not None else MIN_LLM_SPECS_COUNT
    logger.info(f"[LLM_SPECS] Generating specs for: {product_type} (minimum: {min_required})")

    all_specs: Dict[str, Any] = {}
    all_notes: List[str] = []
    iteration = 0
    category_value = category or "Industrial Instrument"
    context_value = context or "General industrial application"

    try:
        llm = get_cached_llm(
            model=LLM_SPECS_MODEL,
            temperature=0.3  # Some creativity for comprehensive coverage
        )

        # =================================================================
        # ITERATION 1: Initial generation
        # =================================================================
        iteration = 1
        logger.info(f"[LLM_SPECS] Iteration {iteration}: Initial generation...")

        prompt = ChatPromptTemplate.from_template(LLM_SPECS_GENERATION_PROMPT)
        parser = JsonOutputParser()
        chain = prompt | llm | parser

        result = chain.invoke({
            "product_type": product_type,
            "category": category_value,
            "context": context_value
        })

        raw_specs = result.get("specifications", {})
        notes = result.get("generation_notes", "")

        clean_specs = _flatten_and_clean_specs(raw_specs)
        all_specs.update(clean_specs)
        all_notes.append(f"Iteration 1: Generated {len(clean_specs)} specs")

        logger.info(f"[LLM_SPECS] Iteration {iteration}: Generated {len(clean_specs)} specs, total: {len(all_specs)}")

        # =================================================================
        # ITERATIVE LOOP: Continue until minimum is reached
        # =================================================================
        while len(all_specs) < min_required and iteration < MAX_LLM_ITERATIONS:
            iteration += 1
            specs_needed = min(SPECS_PER_ITERATION, min_required - len(all_specs) + 5)  # Request a few extra

            logger.info(f"[LLM_SPECS] Iteration {iteration}: Need {min_required - len(all_specs)} more specs, requesting {specs_needed}...")
            logger.info(f"[LLM_SPECS] Using PARALLEL generation with {MAX_PARALLEL_WORKERS} workers...")

            try:
                # Use parallel generation for faster iteration
                if ENABLE_PARALLEL_ITERATIONS and iteration > 1:  # Start parallel from iteration 2
                    new_specs = _generate_specs_parallel_by_focus(
                        product_type=product_type,
                        category=category_value,
                        context=context_value,
                        existing_specs=all_specs,
                        specs_needed=specs_needed
                    )
                    all_notes.append(f"Iteration {iteration}: Used PARALLEL generation")
                else:
                    # First iteration or parallel disabled - use sequential
                    new_specs = _generate_additional_specs(
                        llm=llm,
                        product_type=product_type,
                        category=category_value,
                        context=context_value,
                        existing_specs=all_specs,
                        specs_needed=specs_needed
                    )

                # Only add specs that don't already exist (avoid duplicates)
                added_count = 0
                for key, value in new_specs.items():
                    normalized_key = key.lower().replace(" ", "_").replace("-", "_")
                    existing_keys = {k.lower().replace(" ", "_").replace("-", "_") for k in all_specs.keys()}

                    if normalized_key not in existing_keys:
                        all_specs[key] = value
                        added_count += 1

                all_notes[-1] += f" - Added {added_count} new specs" if all_notes else None
                logger.info(f"[LLM_SPECS] Iteration {iteration}: Added {added_count} new specs, total: {len(all_specs)}")

                # If no new specs were added, break to avoid infinite loop
                if added_count == 0:
                    logger.warning(f"[LLM_SPECS] Iteration {iteration}: No new specs added, stopping iterations")
                    all_notes.append(f"Iteration {iteration}: Stopped - no new specs could be generated")
                    break

            except Exception as iter_error:
                logger.error(f"[LLM_SPECS] Iteration {iteration} failed: {iter_error}")
                all_notes.append(f"Iteration {iteration}: Failed - {str(iter_error)}")
                break

        # =================================================================
        # FINAL RESULT
        # =================================================================
        final_count = len(all_specs)
        target_reached = final_count >= min_required

        if target_reached:
            logger.info(f"[LLM_SPECS] ✓ Target reached: {final_count} specs (minimum: {min_required}) in {iteration} iterations")
        else:
            logger.warning(f"[LLM_SPECS] ✗ Target NOT reached: {final_count} specs (minimum: {min_required}) after {iteration} iterations")

        return {
            "specifications": all_specs,
            "source": "llm_generated",
            "generation_notes": "; ".join(all_notes),
            "timestamp": datetime.now().isoformat(),
            "product_type": product_type,
            "category": category,
            "iterations": iteration,
            "specs_count": final_count,
            "min_required": min_required,
            "target_reached": target_reached
        }

    except Exception as e:
        logger.error(f"[LLM_SPECS] Generation failed for {product_type}: {e}")
        return {
            "specifications": all_specs if all_specs else {},
            "source": "llm_generated",
            "generation_notes": f"Generation failed: {str(e)}; Specs before failure: {len(all_specs)}",
            "timestamp": datetime.now().isoformat(),
            "product_type": product_type,
            "error": str(e),
            "iterations": iteration,
            "specs_count": len(all_specs),
            "min_required": min_required,
            "target_reached": False
        }


def generate_llm_specs_batch(
    items: List[Dict[str, Any]],
    batch_size: int = 4
) -> List[Dict[str, Any]]:
    """
    [FIX #4] Generate LLM specs for multiple items using BATCHED LLM calls.

    Instead of individual LLM calls per item (15 items = 15 calls × 3s = 45s),
    this batches products into groups (4 items per call × 3s = 12s total).

    SPEEDUP: 45 seconds → 12 seconds (3.75x faster!)

    Args:
        items: List of identified items with 'name', 'category', 'sample_input', etc.
        batch_size: Number of products per LLM call (default 4)

    Returns:
        List of generation results, one per item
    """
    logger.info(f"[LLM_SPECS] [FIX #4] Batch generation for {len(items)} items (batch_size={batch_size})")

    results = []
    llm = get_cached_llm(model=LLM_SPECS_MODEL, temperature=0.1)

    # Process items in batches for more efficient LLM calls
    for batch_start in range(0, len(items), batch_size):
        batch_end = min(batch_start + batch_size, len(items))
        batch_items = items[batch_start:batch_end]

        logger.info(f"[LLM_SPECS] [FIX #4] Processing batch {batch_start//batch_size + 1}/{(len(items) + batch_size - 1)//batch_size}: {len(batch_items)} items")

        # Build request for multiple products
        batch_spec_request = "Generate specifications for these products:\n\n"
        product_list = []

        for idx, item in enumerate(batch_items, 1):
            product_type = item.get("name") or item.get("product_name", "Unknown")
            category = item.get("category", "Industrial Instrument")
            context = item.get("sample_input", "")
            product_list.append(f"{idx}. {product_type} ({category})")
            batch_spec_request += f"{idx}. **{product_type}** ({category})\n   Context: {context}\n\n"

        batch_spec_request += "\nReturn a JSON object with product names as keys:\n{\n"
        batch_spec_request += "  \"product_name_1\": { \"specifications\": {...} },\n"
        batch_spec_request += "  \"product_name_2\": { \"specifications\": {...} },\n"
        batch_spec_request += "  ...\n}\n"
        batch_spec_request += "Each product's value should follow the same format as single generation."

        try:
            # Create batch prompt
            batch_prompt = ChatPromptTemplate.from_template(LLM_SPECS_GENERATION_PROMPT.replace(
                "PRODUCT TYPE: {product_type}",
                f"PRODUCTS IN BATCH:\n{chr(10).join(product_list)}"
            ).replace(
                "CATEGORY: {category}",
                "CATEGORIES: See products above"
            ).replace(
                "CONTEXT: {context}",
                "CONTEXTS: See products above"
            ))

            parser = JsonOutputParser()
            chain = batch_prompt | llm | parser

            # Call LLM ONCE for the entire batch
            batch_result = chain.invoke({
                "product_type": "\n".join(product_list),
                "category": "Multiple",
                "context": batch_spec_request
            })

            logger.info(f"[LLM_SPECS] [FIX #4] Batch result received: {type(batch_result)}")

            # Parse batch results and create individual results
            if isinstance(batch_result, dict):
                for idx, item in enumerate(batch_items):
                    product_type = item.get("name") or item.get("product_name", "Unknown")

                    # Try to find specs for this product in batch result
                    item_specs = None
                    for key in batch_result:
                        if product_type.lower() in key.lower() or key.lower() in product_type.lower():
                            item_specs = batch_result[key]
                            break

                    if item_specs is None:
                        # Fallback: use individual generation
                        logger.warning(f"[LLM_SPECS] [FIX #4] Could not find batch specs for {product_type}, falling back to individual generation")
                        item_specs = generate_llm_specs(
                            product_type=product_type,
                            category=item.get("category", "Industrial Instrument"),
                            context=item.get("sample_input", "")
                        )

                    result = item_specs if isinstance(item_specs, dict) else {"specifications": item_specs}
                    result["item_name"] = product_type
                    result["item_type"] = item.get("type", "instrument")
                    result["batch_processed"] = True
                    results.append(result)
            else:
                # Fallback: if batch result is not a dict, process individually
                logger.warning("[LLM_SPECS] [FIX #4] Batch result is not a dict, falling back to individual generation")
                for item in batch_items:
                    product_type = item.get("name") or item.get("product_name", "Unknown")
                    result = generate_llm_specs(
                        product_type=product_type,
                        category=item.get("category", "Industrial Instrument"),
                        context=item.get("sample_input", "")
                    )
                    result["item_name"] = product_type
                    result["item_type"] = item.get("type", "instrument")
                    results.append(result)

        except Exception as e:
            logger.error(f"[LLM_SPECS] [FIX #4] Batch processing failed: {e}, falling back to individual generation")
            # Fallback: process individually
            for item in batch_items:
                product_type = item.get("name") or item.get("product_name", "Unknown")
                result = generate_llm_specs(
                    product_type=product_type,
                    category=item.get("category", "Industrial Instrument"),
                    context=item.get("sample_input", "")
                )
                result["item_name"] = product_type
                result["item_type"] = item.get("type", "instrument")
                results.append(result)

    logger.info(f"[LLM_SPECS] [FIX #4] Batch complete: {len(results)} items processed (saved ~{(len(items) - (len(items) + batch_size - 1)//batch_size) * 3} seconds)")
    return results


# =============================================================================
# COMBINED: DYNAMIC DISCOVERY + ITERATIVE GENERATION
# =============================================================================

def generate_specs_with_discovery(
    product_type: str,
    category: Optional[str] = None,
    context: Optional[str] = None,
    min_specs: Optional[int] = None,
    use_discovery: bool = True
) -> Dict[str, Any]:
    """
    Generate specifications using optional dynamic key discovery.
    
    This function combines the best of both approaches:
    1. Uses Gemini Pro to DISCOVER relevant specification keys (optional)
    2. Uses Gemini Flash to GENERATE values with iterative loop
    
    For known product types, skip discovery and use standard generation.
    For unknown/novel product types, enable discovery for best results.
    
    Args:
        product_type: The product type to generate specs for
        category: Optional category for context
        context: Optional additional context
        min_specs: Minimum specs to generate (defaults to MIN_LLM_SPECS_COUNT)
        use_discovery: Whether to use dynamic key discovery (default True)
    
    Returns:
        Dict containing:
            - specifications: Dict of generated specs with values and confidence
            - discovered_keys: Keys discovered (if discovery enabled)
            - discovery_metadata: Discovery details (timing, confidence)
            - generation_metadata: Generation details (iterations, timing)
    """
    logger.info(f"[SPECS_WITH_DISCOVERY] Generating specs for: {product_type} (discovery: {use_discovery})")
    total_start = time.time()
    
    discovery_result = None
    discovered_keys_list = []
    
    # Phase 1: Dynamic Key Discovery (optional)
    if use_discovery and ENABLE_DYNAMIC_DISCOVERY:
        logger.info(f"[SPECS_WITH_DISCOVERY] Phase 1: Discovering keys...")
        discovery_result = discover_specification_keys(
            product_type=product_type,
            category=category,
            context=context
        )
        discovered_keys_list = discovery_result.get("all_keys", [])
        logger.info(f"[SPECS_WITH_DISCOVERY] Discovered {len(discovered_keys_list)} keys")
    
    # Phase 2: Generate specs using standard iterative approach
    logger.info(f"[SPECS_WITH_DISCOVERY] Phase 2: Generating values...")
    
    # Add discovered keys to context for better generation
    enhanced_context = context or ""
    if discovered_keys_list:
        key_hints = ", ".join(discovered_keys_list[:30])  # Top 30 keys
        enhanced_context += f"\n\nRelevant specification keys to include: {key_hints}"
    
    generation_result = generate_llm_specs(
        product_type=product_type,
        category=category,
        context=enhanced_context if enhanced_context else None,
        min_specs=min_specs
    )
    
    total_elapsed = time.time() - total_start
    
    # Combine results
    final_result = {
        "success": generation_result.get("target_reached", False),
        "product_type": product_type,
        "category": category,
        
        # Specifications
        "specifications": generation_result.get("specifications", {}),
        "specs_count": generation_result.get("specs_count", 0),
        
        # Discovery metadata
        "discovery_used": use_discovery and ENABLE_DYNAMIC_DISCOVERY,
        "discovered_keys": {
            "mandatory": discovery_result.get("mandatory_keys", []) if discovery_result else [],
            "optional": discovery_result.get("optional_keys", []) if discovery_result else [],
            "safety_critical": discovery_result.get("safety_keys", []) if discovery_result else [],
            "total": len(discovered_keys_list)
        } if discovery_result else None,
        "discovery_confidence": discovery_result.get("discovery_confidence", 0.0) if discovery_result else None,
        "product_analysis": discovery_result.get("product_analysis", {}) if discovery_result else None,
        
        # Generation metadata
        "iterations": generation_result.get("iterations", 1),
        "target_reached": generation_result.get("target_reached", False),
        "generation_notes": generation_result.get("generation_notes", ""),
        
        # Timing
        "discovery_time_ms": discovery_result.get("discovery_time_ms", 0) if discovery_result else 0,
        "generation_time_ms": int((time.time() - total_start) * 1000) - (discovery_result.get("discovery_time_ms", 0) if discovery_result else 0),
        "total_time_ms": int(total_elapsed * 1000),
        
        # Source info
        "source": "llm_with_discovery" if use_discovery else "llm_generated",
        "reasoning_model": REASONING_MODEL if use_discovery else None,
        "generation_model": LLM_SPECS_MODEL,
        "timestamp": datetime.now().isoformat()
    }
    
    logger.info(f"[SPECS_WITH_DISCOVERY] Complete: {final_result['specs_count']} specs in {total_elapsed:.2f}s")
    return final_result


# =============================================================================
# USER-SPECIFIED SPECIFICATIONS EXTRACTION - Loaded from prompts_library
# =============================================================================
# Merged from user_specs_extractor.py
# Extracts EXPLICIT specifications from user input. These are MANDATORY.
# =============================================================================

USER_SPECS_EXTRACTION_PROMPT = load_prompt("user_specs_extraction_prompt")


def extract_user_specified_specs(
    user_input: str,
    product_type: str,
    sample_input: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extract EXPLICIT specifications from user input.

    These specifications are MANDATORY and will never be overwritten by
    LLM-generated or standards-based specifications.

    Args:
        user_input: The original user input text
        product_type: The identified product type
        sample_input: Optional sample input that may contain additional specs

    Returns:
        Dict with:
            - specifications: Dict of extracted key-value specs
            - source: "user_specified"
            - confidence: Extraction confidence score
            - extraction_notes: Notes on what was extracted
    """
    logger.info(f"[USER_SPECS] Extracting specs for: {product_type}")

    # Combine user input with sample_input if available
    full_input = user_input
    if sample_input:
        full_input = f"{user_input}\n\nAdditional context: {sample_input}"

    try:
        llm = create_llm_with_fallback(
            model=LLM_SPECS_MODEL,  # Use same model as spec generation
            temperature=0.0,  # Zero temperature for deterministic extraction
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        prompt = ChatPromptTemplate.from_template(USER_SPECS_EXTRACTION_PROMPT)
        parser = JsonOutputParser()
        chain = prompt | llm | parser

        result = chain.invoke({
            "user_input": full_input,
            "product_type": product_type
        })

        extracted_specs = result.get("extracted_specifications", {})
        confidence = result.get("confidence", 0.0)
        notes = result.get("extraction_notes", "")

        # Filter out null/empty values
        clean_specs = {
            k: v for k, v in extracted_specs.items()
            if v and str(v).lower() not in ["null", "none", "n/a", "not specified"]
        }

        logger.info(f"[USER_SPECS] Extracted {len(clean_specs)} specs for {product_type}")
        if clean_specs:
            logger.info(f"[USER_SPECS] Specs: {list(clean_specs.keys())}")

        return {
            "specifications": clean_specs,
            "source": "user_specified",
            "confidence": confidence,
            "extraction_notes": notes,
            "timestamp": datetime.now().isoformat(),
            "product_type": product_type
        }

    except Exception as e:
        logger.error(f"[USER_SPECS] Extraction failed for {product_type}: {e}")
        return {
            "specifications": {},
            "source": "user_specified",
            "confidence": 0.0,
            "extraction_notes": f"Extraction failed: {str(e)}",
            "timestamp": datetime.now().isoformat(),
            "product_type": product_type,
            "error": str(e)
        }


def extract_user_specs_batch(
    items: List[Dict[str, Any]],
    user_input: str
) -> List[Dict[str, Any]]:
    """
    Extract user-specified specs for multiple items.

    Args:
        items: List of identified items with 'name', 'sample_input', etc.
        user_input: Original user input

    Returns:
        List of extraction results, one per item
    """
    logger.info(f"[USER_SPECS] Batch extraction for {len(items)} items")

    results = []
    for item in items:
        product_type = item.get("name") or item.get("product_name", "Unknown")
        sample_input = item.get("sample_input", "")

        result = extract_user_specified_specs(
            user_input=user_input,
            product_type=product_type,
            sample_input=sample_input
        )

        result["item_name"] = product_type
        result["item_type"] = item.get("type", "instrument")
        results.append(result)

    logger.info(f"[USER_SPECS] Batch complete: {len(results)} items processed")
    return results


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main generation functions
    "generate_llm_specs",
    "generate_llm_specs_batch",
    "generate_specs_with_discovery",
    
    # Dynamic key discovery
    "discover_specification_keys",
    
    # User specification extraction
    "extract_user_specified_specs",
    "extract_user_specs_batch",
    
    # Configuration constants
    "MIN_LLM_SPECS_COUNT",
    "MAX_LLM_ITERATIONS",
    "SPECS_PER_ITERATION",
    "MAX_PARALLEL_WORKERS",
    "ENABLE_PARALLEL_ITERATIONS",
    "ENABLE_DYNAMIC_DISCOVERY",
    
    # Model names
    "LLM_SPECS_MODEL",
    "REASONING_MODEL"
]
