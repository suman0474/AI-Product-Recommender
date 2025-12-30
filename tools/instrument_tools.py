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
You are Engenie - an expert in Industrial Process Control Systems.
Analyze the requirements and identify all instruments needed for the project.

**IMPORTANT: Think step-by-step through your analysis.**

Before providing your final list:
1. First, understand the overall process or system being described
2. Then, identify each measurement or control point required
3. Determine the appropriate instrument type for each point
4. Extract or infer specifications for each instrument
5. Finally, organize into a comprehensive bill of materials

**Requirements:**
{requirements}

**Tasks:**
1. Extract a project name (1-2 words) describing the system
2. Identify every instrument required
3. For each instrument provide:
   - Category (e.g., Pressure Transmitter, Flow Meter, Temperature Sensor)
   - Product Name (generic name, not brand-specific)
   - Quantity (count how many needed)
   - Specifications (extract explicitly mentioned specs, infer critical ones with [INFERRED] tag)
   - Procurement Strategy (detect from context if mentioned)
   - Sample Input (comprehensive description including ALL specifications)

**Strategy Detection Rules:**
Analyze the requirements for procurement approach hints:
- Budget constraints, cost-focused language → "Cost optimization"
- Quality emphasis, reliability focus → "Life-cycle cost evaluation"
- Sustainability, environmental concerns → "Sustainability and green procurement"
- Safety-critical, high-risk applications → "Dual sourcing"
- Standard equipment, routine applications → "Framework agreement"
- Innovation, cutting-edge needs → "Technology partnership"
- Local suppliers, regional focus → "Regional sourcing"

**Specification Extraction Guidelines:**
- **ALWAYS extract explicitly mentioned specifications** (e.g., "4-20mA", "0-100 psi", "Class 150")
- **Infer critical specifications** when essential for operation but not mentioned:
  - Pressure range if pressure application mentioned
  - Temperature range if temperature application mentioned
  - Output signal type for transmitters (default: 4-20mA)
  - Process connection if not specified (default: based on industry standards)
- **Mark inferred specs** with [INFERRED] tag in the inferred_specs list
- **Do not over-infer** - only add specifications that are truly necessary

**Instrument Categorization:**
Use standard industrial categories:
- Pressure: Pressure Transmitter, Pressure Gauge, Pressure Switch
- Flow: Flow Meter, Flow Transmitter, Flow Switch
- Level: Level Transmitter, Level Gauge, Level Switch
- Temperature: Temperature Transmitter, Temperature Sensor, Thermocouple, RTD
- Analytical: pH Sensor, Conductivity Sensor, Dissolved Oxygen Sensor
- Control: Control Valve, Actuator, Positioner, I/P Converter
- Safety: Pressure Relief Valve, Rupture Disc, Safety Instrumented System

Return ONLY valid JSON:
{{
    "project_name": "<unique 1-3 word project name>",
    "instruments": [
        {{
            "category": "<standard instrument category>",
            "product_name": "<generic product name>",
            "quantity": <number>,
            "specifications": {{
                "<spec_field>": "<spec_value>"
            }},
            "strategy": "<procurement strategy or empty string>",
            "sample_input": "<category> with <all key specifications including inferred>",
            "inferred_specs": ["<list of specifications that were inferred with [INFERRED] prefix>"]
        }}
    ],
    "summary": "<brief 2-3 sentence summary of identified instruments and project scope>"
}}
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
- Protective accessories

For each accessory provide:
- Category
- Accessory Name
- Quantity
- Specifications
- Related Instrument

Return ONLY valid JSON:
{{
    "accessories": [
        {{
            "category": "<accessory category>",
            "accessory_name": "<accessory name>",
            "quantity": <number>,
            "specifications": {{
                "<spec_field>": "<spec_value>"
            }},
            "for_instrument": "<related instrument>",
            "sample_input": "<accessory category> for <instrument> with <specs>"
        }}
    ],
    "summary": "<brief summary of accessories>"
}}
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
            model="gemini-2.0-flash-exp",
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
            model="gemini-2.0-flash-exp",
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
