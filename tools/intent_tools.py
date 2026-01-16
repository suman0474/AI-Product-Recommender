# tools/intent_tools.py
# Intent Classification and Requirements Extraction Tools

import json
import logging
from typing import Dict, Any, Optional
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

class ClassifyIntentInput(BaseModel):
    """Input for intent classification"""
    user_input: str = Field(description="User's input message to classify")
    current_step: Optional[str] = Field(default=None, description="Current workflow step")
    context: Optional[str] = Field(default=None, description="Conversation context")


class ExtractRequirementsInput(BaseModel):
    """Input for requirements extraction"""
    user_input: str = Field(description="User's input containing requirements")


# ============================================================================
# PROMPTS
# ============================================================================

INTENT_CLASSIFICATION_PROMPT = """
You are Engenie - a smart assistant that classifies user input for an industrial procurement workflow.

Analyze the user's input and classify it into ONE of these categories:

1. "greeting" - Simple greeting (hi, hello, hey) with NO other content
2. "solution" - Complex engineering challenge/problem requiring MULTIPLE instruments or a complete measurement system
3. "requirements" - Technical requirements for a SINGLE product type or simple specification
4. "question" - Asking about industrial topics, products, or processes
5. "additional_specs" - Adding more specifications to existing requirements
6. "confirm" - User confirms or agrees (yes, proceed, continue)
7. "reject" - User rejects or disagrees (no, cancel, stop)
8. "chitchat" - Casual conversation not related to procurement
9. "unrelated" - Content unrelated to industrial automation

Current workflow step: {current_step}
Context: {context}

User Input: "{user_input}"

**Classification Rules (Priority Order):**

**RULE 1 - SOLUTION Detection (Highest Priority):**
A "solution" is identified when ANY of these patterns are present:
- Contains "Problem Statement", "Challenge", "Design a system", "Implement a system"
- Describes a complete measurement/control SYSTEM with multiple measurement points
- Mentions multiple LOCATIONS for measurements (inlet, outlet, reactor, tubes, zones)
- References REDUNDANT sensors or safety-critical systems
- Contains system integration requirements (DCS, HART, data logging, alarm systems)
- Specifies measurements for DIFFERENT parameters across a process (temperature AND pressure AND flow)
- References industrial standards compliance (ASME, Class I Div 2, SIL, ATEX, hazardous area)
- Total measurement points > 3 or monitoring multiple process stages
- Describes a complete reactor, vessel, or process unit instrumentation

**EXAMPLES of SOLUTION inputs:**
- "Design a temperature measurement system for a chemical reactor with hot oil heating..."
- "Implement temperature profiling for a multi-tube catalytic reactor with 32 measurement points..."
- "Need complete instrumentation for a distillation column: temperature, pressure, level..."
- "Safety instrumented system for reactor with redundant sensors and SIL requirements..."

**RULE 2 - REQUIREMENTS Detection:**
Use "requirements" only for SIMPLE, single-product requests:
- Single product type specification (e.g., "I need a pressure transmitter 0-100 PSI")
- Adding specs to an existing product search
- Does NOT contain system-level architecture or multiple measurement locations

**Other Rules:**
3. If asking "what is", "how does", "explain" about industrial topics → "question"
4. If only greeting words with no other content → "greeting"
5. If says yes/proceed/continue → "confirm"
6. If says no/cancel/stop → "reject"

Return ONLY valid JSON:
{{
    "intent": "<intent_type>",
    "confidence": <0.0-1.0>,
    "next_step": "<suggested_next_step or null>",
    "extracted_info": {{<any extracted information>}},
    "is_solution": <true if solution detected, false otherwise>,
    "solution_indicators": ["<list of patterns that triggered solution detection>"]
}}
"""


REQUIREMENTS_EXTRACTION_PROMPT = """
You are Engenie - an expert assistant for industrial requisitioners and buyers.
Extract and structure the key requirements from this user input so a procurement professional can quickly understand what is needed and why.

User Input: "{user_input}"

Focus on:
- Technical specifications (pressure ranges, accuracy, measurement ranges, etc.)
- Connection types and standards (NPT, flanged, threaded, etc.)
- Application context and environment (steam, water, chemical service, etc.)
- Performance requirements (response time, turndown ratio, etc.)
- Compliance or certification needs (ATEX, SIL, FM, CSA, etc.)
- Any business or operational considerations relevant to buyers

**CRITICAL:**
- Return a clear, structured summary of requirements
- Use language that is actionable and easy for buyers to use in procurement
- **Only include sections and details for which information is explicitly present in the user's input**
- **Do not add any inferred requirements or placeholders for missing information**
- Focus on what the user actually said, not what might be needed

Return ONLY valid JSON:
{{
    "product_type": "<detected product type>",
    "specifications": {{
        "<spec_name>": "<spec_value>"
    }},
    "raw_requirements_text": "<structured summary in buyer-friendly language>"
}}
"""


# ============================================================================
# TOOLS
# ============================================================================

@tool("classify_intent", args_schema=ClassifyIntentInput)
def classify_intent_tool(
    user_input: str,
    current_step: Optional[str] = None,
    context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Classify user intent for routing in the procurement workflow.
    Returns intent type, confidence, and suggested next step.
    """
    try:
        llm = create_llm_with_fallback(
            model="gemini-2.5-flash",
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        prompt = ChatPromptTemplate.from_template(INTENT_CLASSIFICATION_PROMPT)
        parser = JsonOutputParser()

        chain = prompt | llm | parser

        result = chain.invoke({
            "user_input": user_input,
            "current_step": current_step or "start",
            "context": context or "New conversation"
        })

        return {
            "success": True,
            "intent": result.get("intent", "unrelated"),
            "confidence": result.get("confidence", 0.5),
            "next_step": result.get("next_step"),
            "extracted_info": result.get("extracted_info", {}),
            "is_solution": result.get("is_solution", False),
            "solution_indicators": result.get("solution_indicators", [])
        }


    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        return {
            "success": False,
            "intent": "unrelated",
            "confidence": 0.0,
            "next_step": None,
            "error": str(e)
        }


@tool("extract_requirements", args_schema=ExtractRequirementsInput)
def extract_requirements_tool(user_input: str) -> Dict[str, Any]:
    """
    Extract structured technical requirements from user input.
    Identifies product type, specifications, and infers missing common specs.
    """
    try:
        llm = create_llm_with_fallback(
            model="gemini-2.5-flash",
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        prompt = ChatPromptTemplate.from_template(REQUIREMENTS_EXTRACTION_PROMPT)
        parser = JsonOutputParser()

        chain = prompt | llm | parser

        result = chain.invoke({"user_input": user_input})

        return {
            "success": True,
            "product_type": result.get("product_type"),
            "specifications": result.get("specifications", {}),
            "inferred_specs": result.get("inferred_specs", {}),
            "raw_requirements_text": result.get("raw_requirements_text", user_input)
        }

    except Exception as e:
        logger.error(f"Requirements extraction failed: {e}")
        return {
            "success": False,
            "product_type": None,
            "specifications": {},
            "error": str(e)
        }
