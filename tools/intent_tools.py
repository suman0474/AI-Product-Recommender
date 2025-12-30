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
2. "requirements" - Technical requirements, specifications, or product requests
3. "question" - Asking about industrial topics, products, or processes
4. "additional_specs" - Adding more specifications to existing requirements
5. "confirm" - User confirms or agrees (yes, proceed, continue)
6. "reject" - User rejects or disagrees (no, cancel, stop)
7. "chitchat" - Casual conversation not related to procurement
8. "unrelated" - Content unrelated to industrial automation

Current workflow step: {current_step}
Context: {context}

User Input: "{user_input}"

**Classification Rules (Priority Order):**
1. If contains technical specs (pressure, temperature, 4-20mA, PSI, etc.) → "requirements"
2. If asking "what is", "how does", "explain" about industrial topics → "question"
3. If only greeting words with no other content → "greeting"
4. If says yes/proceed/continue → "confirm"
5. If says no/cancel/stop → "reject"

Return ONLY valid JSON:
{{
    "intent": "<intent_type>",
    "confidence": <0.0-1.0>,
    "next_step": "<suggested_next_step or null>",
    "extracted_info": {{<any extracted information>}}
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
            model="gemini-2.0-flash-exp",
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
            "extracted_info": result.get("extracted_info", {})
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
            model="gemini-2.0-flash-exp",
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
