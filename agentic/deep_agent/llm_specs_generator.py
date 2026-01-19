# agentic/deep_agent/llm_specs_generator.py
# =============================================================================
# LLM SPECIFICATION GENERATOR
# =============================================================================
#
# Generates all possible specifications for a product type using LLM.
# These specs fill in gaps not covered by user-specified or standards specs.
#
# ITERATIVE GENERATION: If specs count < MIN_LLM_SPECS_COUNT (30), the generator
# will iterate and request additional specs until the minimum is reached.
#
# =============================================================================

import logging
import os
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

from llm_fallback import create_llm_with_fallback

load_dotenv()
logger = logging.getLogger(__name__)

# Model for LLM spec generation
LLM_SPECS_MODEL = "gemini-2.5-flash"

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
# ITERATIVE GENERATION PROMPT - For requesting additional specs
# =============================================================================

LLM_SPECS_ITERATIVE_PROMPT = """
You are an industrial instrumentation expert. Generate ADDITIONAL technical specifications
for the given product type that are NOT already provided. Return ONLY clean technical values.

PRODUCT TYPE: {product_type}
CATEGORY: {category}
CONTEXT: {context}

=== SPECIFICATIONS ALREADY GENERATED (DO NOT REPEAT THESE) ===
{existing_specs}

=== CRITICAL: VALUE FORMAT RULES ===

Return ONLY the technical value. NO explanations, NO descriptions, NO sentences.

FORBIDDEN PATTERNS IN VALUES:
- Words: "typically", "usually", "generally", "approximately", "may be", "can be"
- Phrases: "depends on", "varies by", "according to", "based on", "provided via"
- Sentences or explanations

CORRECT VALUE EXAMPLES:
- accuracy: "±0.1%" ✓
- output_signal: "4-20mA, HART" ✓
- protection_rating: "IP67" ✓

=== GENERATE {specs_needed} NEW SPECIFICATIONS ===

Focus on these ADDITIONAL specification categories (NOT already covered):
- Physical: dimensions, weight, mounting_type, enclosure_material, display_type
- Electrical: power_consumption, loop_resistance, load_impedance, isolation_voltage
- Performance: turndown_ratio, zero_stability, span_stability, hysteresis, linearity
- Environmental: vibration_resistance, shock_resistance, altitude_rating, corrosion_resistance
- Safety: overpressure_limit, burst_pressure, failure_mode, diagnostic_coverage
- Connectivity: cable_entry, terminal_type, wireless_capability, fieldbus_options
- Maintenance: mtbf, service_interval, warranty_period, spare_parts_availability
- Compliance: marine_approval, food_grade_certification, railway_approval, nuclear_qualification
- Application-specific: media_compatibility, viscosity_range, density_range, conductivity_range

=== OUTPUT FORMAT ===

Return ONLY valid JSON with NEW specifications not in the existing list:
{{
    "specifications": {{
        "new_spec_key_1": {{"value": "clean value", "confidence": 0.8}},
        "new_spec_key_2": {{"value": "clean value", "confidence": 0.7}},
        ...
    }},
    "generation_notes": "Brief note about additional specs generated"
}}

CRITICAL: Do NOT repeat any specification keys from the existing list above!
"""


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
        llm = create_llm_with_fallback(
            model=LLM_SPECS_MODEL,
            temperature=0.4,  # Slightly higher for diversity
            google_api_key=os.getenv("GOOGLE_API_KEY")
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
{{
    "specifications": {{
        "new_spec_1": {{"value": "clean value", "confidence": 0.8}},
        "new_spec_2": {{"value": "clean value", "confidence": 0.7}}
    }},
    "focus_area": "{focus_area}"
}}

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
        llm = create_llm_with_fallback(
            model=LLM_SPECS_MODEL,
            temperature=0.3,  # Some creativity for comprehensive coverage
            google_api_key=os.getenv("GOOGLE_API_KEY")
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
    llm = create_llm_with_fallback(model=LLM_SPECS_MODEL, temperature=0.1, google_api_key=os.getenv("GOOGLE_API_KEY"), skip_test=True)

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
# EXPORTS
# =============================================================================

__all__ = [
    "generate_llm_specs",
    "generate_llm_specs_batch",
    "MIN_LLM_SPECS_COUNT",
    "MAX_LLM_ITERATIONS",
    "SPECS_PER_ITERATION",
    "MAX_PARALLEL_WORKERS",
    "ENABLE_PARALLEL_ITERATIONS"
]
