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
2. "solution" - Complex engineering challenge requiring MULTIPLE instruments as a complete measurement SYSTEM
3. "requirements" - User provides VERY SPECIFIC technical specifications (ranges, accuracy, protocols)
4. "question" - User wants to SEARCH, FIND, or LEARN about products (including with standards/certifications)
5. "additional_specs" - Adding more specifications to existing requirements
6. "confirm" - User confirms or agrees (yes, proceed, continue)
7. "reject" - User rejects or disagrees (no, cancel, stop)
8. "chitchat" - Casual conversation not related to procurement
9. "unrelated" - Content unrelated to industrial automation

Current workflow step: {current_step}
Context: {context}

User Input: "{user_input}"

**CRITICAL Classification Rules (Priority Order):**

**RULE 1 - SOLUTION Detection (ONLY for MULTI-INSTRUMENT SYSTEMS):**
Classify as "solution" ONLY when ALL of these are true:
- Requires MULTIPLE (3+) different instruments working together as a SYSTEM
- Mentions multiple PHYSICAL LOCATIONS (inlet AND outlet, reactor zones, multiple vessels)
- Describes a complete process/plant instrumentation (not just one product)
- Contains explicit system design language ("system", "complete instrumentation", "profiling")

**SOLUTION Examples (→ solution workflow):**
- "Design a temperature measurement SYSTEM for a chemical reactor with inlet, outlet, AND jacket monitoring" → solution (3+ locations)
- "Complete instrumentation for distillation column: temperature, pressure, AND level transmitters" → solution (3+ different instruments)
- "Reactor temperature profiling with 32 measurement points across 8 zones" → solution (multi-point system)

**NOT SOLUTION (→ question instead):**
- "I need a pressure transmitter for SIL 2 application" → question (single product)
- "Differential pressure transmitter meeting IEC 61508 for Zone 1" → question (single product with standards)
- "Flow meter with ATEX certification" → question (single product with certification)
- "Two pressure transmitters for hazardous area" → question (small quantity, not a system)

**RULE 2 - QUESTION Detection (Product Search - DEFAULT for product requests):**
Classify as "question" when user:
- Searches for a SINGLE product type: "I need a...", "I want a...", "Looking for..."
- Mentions a vendor: "...from Yokogawa", "...from Emerson"
- Asks about standards/certifications: "SIL 2", "ATEX", "IEC 61508", "Zone 1"
- Wants to find products for a specific application
- Asks knowledge questions: "What is...", "How does...", "Tell me about..."

**QUESTION Examples (→ ProductInfo/Index RAG):**
- "I need a pressure transmitter from Yokogawa" → question
- "I need a differential pressure transmitter meeting IEC 61508 SIL 2 for Zone 1" → question
- "Flow meter for chemical applications with ATEX certification" → question
- "What is ISA 84?" → question
- "Looking for SIL 2 rated temperature sensor" → question

**RULE 3 - REQUIREMENTS Detection (Very Detailed Specs):**
Classify as "requirements" ONLY when user provides MULTIPLE specific technical values:
- Measurement ranges: "0-100 PSI", "-40 to 200°C"
- Accuracy: "±0.1%", "0.05% accuracy"
- Protocols: "HART 7", "Modbus RTU"
- Materials: "316L wetted parts"
- Connections: "1/2 NPT", "DN50 flange"

**Other Rules:**
4. If only greeting words → "greeting"
5. If says yes/proceed → "confirm"
6. If says no/cancel → "reject"

**DEFAULT RULE:** When in doubt, prefer "question" over "solution" for product searches.


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
