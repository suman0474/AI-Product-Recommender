# tools/instrument_tools.py
# Instrument Identification and Accessory Tools

import json
import logging
from typing import Dict, Any, List, Optional
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
from llm_fallback import create_llm_with_fallback

load_dotenv()
logger = logging.getLogger(__name__)


# ============================================================================
# TOOL INPUT SCHEMAS
# ============================================================================

class IdentifyInstrumentsInput(BaseModel):
    """Input for instrument identification"""
    requirements: str = Field(description="Process requirements or problem statement")


class IdentifyAccessoriesInput(BaseModel):
    """Input for accessory identification"""
    instruments: List[Dict[str, Any]] = Field(description="List of identified instruments")
    process_context: Optional[str] = Field(default=None, description="Process context")


# ============================================================================
# PROMPTS
# ============================================================================

INSTRUMENT_IDENTIFICATION_PROMPT = """
You are Engenie - an expert assistant in Industrial Process Control Systems. Analyze the given requirements and identify the Bill of Materials (instruments) needed.
**IMPORTANT: Think step-by-step through your identification process.**

Before providing the final JSON:
1. First, read the requirements and extract a concise project name (1-2 words) that best represents the objective.
2. Then, identify every instrument required by the problem statement. For each instrument, determine category, generic product name, quantity (explicit or inferred), and all specifications mentioned.
3. For any specification not explicitly present but required for sensible procurement (e.g., typical ranges, common materials), mark it with the tag `[INFERRED]` and explain briefly in an internal analysis (see Validation step).
4. Create a clear `sample_input` field for each instrument that contains every specification key/value exactly as listed in the `specifications` object.
5. **ACCESSORY INFERENCE RULE:**
   - If user explicitly says "no accessories", "without accessories", "only the instrument", "just the [instrument]", or similar → Do NOT add any accessories
   - If user does NOT mention anything about accessories → Auto-infer relevant accessories of that instrument(impulse lines, isolation valves, manifolds, mounting brackets, junction boxes, power supplies, calibration kits, connectors)
6. **VENDOR EXTRACTION RULE (CRITICAL):**
   - If user mentions specific vendor/manufacturer names (e.g., "Honeywell", "ABB", "Emerson", "Siemens", "Yokogawa", "Endress+Hauser", "Rosemount", etc.), extract them for EACH instrument/accessory they apply to.
   - Example: "I need a pressure transmitter from Honeywell and flow meter from Emerson" → PT gets ["Honeywell"], Flow Meter gets ["Emerson"]
   - Example: "I need Honeywell instruments" → ALL instruments get ["Honeywell"]
   - If NO vendor is mentioned for an instrument, leave specified_vendors as an empty array []

=== CRITICAL: SPECIFICATION VALUE FORMAT ===

All specification values must be CLEAN technical values only - NO descriptions or explanations.

CORRECT: "4-20mA", "0-100 psi", "316L SS", "-40 to +85°C", "IP67"
WRONG: "4-20mA output signal", "typically 0-100 psi", "316L SS for corrosion resistance"

Requirements:
{requirements}

Instructions:
1. Extract a unique, descriptive project name (1-2 words) from the requirements that best represents the objective of the industrial system or process described. This should be concise and professional.
2. Identify all instruments required for the given Industrial Process Control System Problem Statement 
3. For each instrument, provide:
   - Category (e.g., Pressure Transmitter, Temperature Transmitter, Flow Meter, etc.)
   - Product Name (generic name based on the requirements)
   - Quantity
   - Specifications (extract from requirements or infer based on industry standards - CLEAN VALUES ONLY)
   - Strategy (analyze user requirements to identify procurement approach: budget constraints suggest "Cost optimization", quality emphasis suggests "Life-cycle cost evaluation", sustainability mentions suggest "Sustainability and green procurement", critical applications suggest "Dual sourcing", standard applications suggest "Framework agreement", or leave empty if none identified)
   - Specified Vendors (extract vendor/manufacturer names mentioned by user for THIS specific instrument, empty array if none)
   - Specified Model Families (extract model/series names if user mentions them, e.g., "Rosemount 3051" → ["3051"], "Honeywell STT850" → ["STT850"], empty array if none)
   - Sample Input(must include all specification details exactly as listed in the specifications field (no field should be missing)).
   - Ensure every parameter appears explicitly in the sample input text.
4. Mark inferred requirements explicitly with [INFERRED] tag

Return ONLY valid JSON:
{{
  "project_name": "<unique project name describing the system>",
  "instruments": [
    {{
      "category": "<category>",
      "product_name": "<product name>",
      "quantity": "<quantity>",
      "specifications": {{
        "<spec_field>": "<CLEAN spec_value - no descriptions>",
        "<spec_field>": "<CLEAN spec_value - no descriptions>"
      }},
      "strategy": "<procurement strategy from user requirements or empty string>",
      "specified_vendors": ["<vendor1>", "<vendor2>"],
      "specified_model_families": ["<model_family1>", "<model_family2>"],
      "sample_input": "<category> with <key specifications>",
      "inferred_specs": ["<list of specifications that were inferred with [INFERRED] prefix>"]
    }}
  ],
  "summary": "Brief summary of identified instruments"
}}

Respond ONLY with valid JSON, no additional text.
"""


ACCESSORIES_IDENTIFICATION_PROMPT = """
You are Engenie - an expert in Industrial Process Control Systems.
Identify all accessories and ancillary items needed for the instruments.

**Identified Instruments:**
{instruments}

**Process Context:**
{process_context}

**Accessory Categories to Consider:**
- Impulse lines and tubing
- Isolation valves and manifolds
- Mounting brackets and supports
- Junction boxes and enclosures
- Cable/connector types
- Power supplies
- Calibration kits
- Thermowells (for temperature instruments)
- Protective accessories

=== CRITICAL: SPECIFICATION VALUE FORMAT ===

All specification values must be CLEAN technical values only - NO descriptions.

CORRECT: "1/2 NPT", "316L SS", "Class 150", "IP66"
WRONG: "1/2 NPT for easy connection", "316L SS material for corrosion resistance"

**VENDOR PRIORITY RULE:**
1. If user EXPLICITLY mentions a vendor for THIS accessory → use that vendor
2. If no explicit accessory vendor BUT parent instrument has a vendor → INHERIT from parent
3. Only leave specified_vendors empty if neither condition applies

For each accessory provide:
- Category (e.g., Impulse Line, Isolation Valve, Mounting Bracket, Junction Box)
- Accessory Name (generic)
- Quantity
- Specifications (size, material, pressure rating, connector type - CLEAN VALUES ONLY)
- Strategy (same as parent instrument or empty)
- Specified Vendors (inherit from parent instrument if not explicitly mentioned)
- Specified Model Families (inherit from parent instrument if applicable)
- Parent Instrument Category (the instrument this accessory supports)
- Sample Input (include every specification field)

Return ONLY valid JSON:
{{
    "accessories": [
        {{
            "category": "<accessory category>",
            "accessory_name": "<accessory name>",
            "quantity": <number>,
            "specifications": {{
                "<spec_field>": "<CLEAN spec_value>"
            }},
            "strategy": "<procurement strategy from parent instrument or empty>",
            "specified_vendors": ["<inherited from parent instrument or empty>"],
            "specified_model_families": ["<inherited from parent instrument or empty>"],
            "parent_instrument_category": "<category of parent instrument>",
            "related_instrument": "<parent instrument name>",
            "sample_input": "<accessory category> for <instrument> with <specs>"
        }}
    ],
    "summary": "<brief summary of accessories>"
}}

Respond ONLY with valid JSON, no additional text.
"""




# ============================================================================
# TOOLS
# ============================================================================

@tool("identify_instruments", args_schema=IdentifyInstrumentsInput)
def identify_instruments_tool(requirements: str) -> Dict[str, Any]:
    """
    Identify instruments needed from process requirements.
    Extracts product types, specifications, and creates a Bill of Materials.
    """
    try:
        llm = create_llm_with_fallback(
            model="gemini-2.5-flash",
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        prompt = ChatPromptTemplate.from_template(INSTRUMENT_IDENTIFICATION_PROMPT)
        parser = JsonOutputParser()

        chain = prompt | llm | parser

        result = chain.invoke({"requirements": requirements})

        return {
            "success": True,
            "project_name": result.get("project_name"),
            "instruments": result.get("instruments", []),
            "instrument_count": len(result.get("instruments", [])),
            "summary": result.get("summary")
        }

    except Exception as e:
        logger.error(f"Instrument identification failed: {e}")
        return {
            "success": False,
            "instruments": [],
            "error": str(e)
        }


@tool("identify_accessories", args_schema=IdentifyAccessoriesInput)
def identify_accessories_tool(
    instruments: List[Dict[str, Any]],
    process_context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Identify accessories needed for the identified instruments.
    Returns list of accessories with specifications.
    """
    try:
        if not instruments:
            return {
                "success": False,
                "accessories": [],
                "error": "No instruments provided"
            }

        llm = create_llm_with_fallback(
            model="gemini-2.5-flash",
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        prompt = ChatPromptTemplate.from_template(ACCESSORIES_IDENTIFICATION_PROMPT)
        parser = JsonOutputParser()

        chain = prompt | llm | parser

        result = chain.invoke({
            "instruments": json.dumps(instruments, indent=2),
            "process_context": process_context or "General industrial process"
        })

        return {
            "success": True,
            "accessories": result.get("accessories", []),
            "accessory_count": len(result.get("accessories", [])),
            "summary": result.get("summary")
        }

    except Exception as e:
        logger.error(f"Accessory identification failed: {e}")
        return {
            "success": False,
            "accessories": [],
            "error": str(e)
        }
