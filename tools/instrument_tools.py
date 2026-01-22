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
from prompts_library import load_prompt

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
# PROMPTS - Loaded from prompts_library
# ============================================================================

INSTRUMENT_IDENTIFICATION_PROMPT = load_prompt("instrument_identification_prompt")

ACCESSORIES_IDENTIFICATION_PROMPT = load_prompt("accessories_identification_prompt")



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
