# agentic/deep_agent/llm_specs_generator.py
# =============================================================================
# LLM SPECIFICATION GENERATOR
# =============================================================================
#
# Generates all possible specifications for a product type using LLM.
# These specs fill in gaps not covered by user-specified or standards specs.
#
# =============================================================================

import logging
import os
from typing import Dict, Any, List, Optional
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
        llm = create_llm_with_fallback(
            model=LLM_SPECS_MODEL,
            temperature=0.3,  # Some creativity for comprehensive coverage
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

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
    "generate_llm_specs_batch"
]
