# agentic/deep_agent/user_specs_extractor.py
# =============================================================================
# USER-SPECIFIED SPECIFICATIONS EXTRACTOR
# =============================================================================
#
# Extracts EXPLICIT specifications from user input.
# These are MANDATORY - they will NEVER be overwritten by other sources.
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

# Model for user spec extraction
USER_SPECS_LLM_MODEL = "gemini-2.5-flash"


# =============================================================================
# EXTRACTION PROMPT
# =============================================================================

USER_SPECS_EXTRACTION_PROMPT = """
You are a specification extractor. Your task is to extract ONLY the specifications
that are EXPLICITLY mentioned in the user's input.

CRITICAL RULES:
1. Extract ONLY what is explicitly stated - do NOT infer or assume anything
2. If a value is not explicitly mentioned, do NOT include it
3. Be precise with the values - use exact text from user input
4. Convert user language to standardized specification keys

USER INPUT:
{user_input}

PRODUCT TYPE: {product_type}

STANDARD SPECIFICATION KEYS TO USE:
- accuracy: Measurement accuracy (e.g., "±0.1%", "0.5% of span")
- pressure_range: Pressure measurement range (e.g., "0-100 bar", "0-1000 psi")
- temperature_range: Operating temperature range (e.g., "-40 to 85°C")
- process_temperature: Process medium temperature (e.g., "0 to 350°C")
- output_signal: Signal type (e.g., "4-20mA", "0-10V", "HART")
- supply_voltage: Power supply (e.g., "24 VDC", "12-36 VDC")
- protection_rating: IP rating (e.g., "IP66", "IP67", "NEMA 4X")
- hazardous_area_approval: Zone certification (e.g., "ATEX Zone 1", "IECEx Zone 0")
- sil_rating: Safety integrity level (e.g., "SIL 2", "SIL 3")
- material_wetted: Wetted parts material (e.g., "SS316L", "Hastelloy C-276")
- material_housing: Housing material (e.g., "Aluminum", "Stainless Steel 316")
- process_connection: Connection type (e.g., "1/2 NPT", "DN50 Flange", "Tri-Clamp")
- response_time: Response time (e.g., "< 250ms", "T90 < 5s")
- communication_protocol: Protocol (e.g., "HART", "Modbus RTU", "Profibus PA")
- flow_range: Flow measurement range (e.g., "0-1000 m³/h")
- level_range: Level measurement range (e.g., "0-10m", "0-30ft")
- display: Display type (e.g., "LCD", "LED", "No display")
- mounting: Mounting type (e.g., "Panel mount", "Pipe mount", "Wall mount")

Return ONLY valid JSON:
{{
    "extracted_specifications": {{
        "specification_key": "exact value from user input"
    }},
    "extraction_notes": "Brief note on what was extracted",
    "confidence": 0.0-1.0
}}

IMPORTANT: If no specifications are explicitly mentioned, return an empty specifications object.
"""


# =============================================================================
# EXTRACTION FUNCTION
# =============================================================================

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
            model=USER_SPECS_LLM_MODEL,
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
    "extract_user_specified_specs",
    "extract_user_specs_batch"
]
