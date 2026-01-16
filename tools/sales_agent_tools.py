# tools/sales_agent_tools.py
# Sales Agent Tool - User Interaction for Product Search Workflow

import json
import logging
import re
from typing import Dict, Any, Optional, List
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

class SalesAgentInput(BaseModel):
    """Input for sales agent workflow step"""
    step: str = Field(description="Current workflow step (e.g., 'initialInput', 'awaitAdditionalAndLatestSpecs', 'showSummary')")
    user_message: str = Field(description="User's message or input")
    product_type: Optional[str] = Field(default=None, description="Detected product type")
    data_context: Optional[Dict[str, Any]] = Field(default=None, description="Context data for the current step")
    intent: Optional[str] = Field(default=None, description="Classified intent from intent tool")
    session_id: Optional[str] = Field(default="default", description="Session ID for workflow isolation")


class CollectRequirementsInput(BaseModel):
    """Input for collecting additional requirements"""
    user_message: str = Field(description="User's input with additional requirements")
    product_type: str = Field(description="Product type being searched")
    existing_requirements: Optional[Dict[str, Any]] = Field(default=None, description="Already collected requirements")


# ============================================================================
# PROMPTS
# ============================================================================

SALES_AGENT_PROMPT = """
You are Engenie - a professional AI sales agent for industrial procurement.

Current conversation step: {step}
Product type: {product_type}

User Input: "{user_message}"

Based on the step and user input, generate an appropriate response:

**Step Guide:**
- initialInput: Acknowledge requirements, ask for missing details
- awaitAdditionalAndLatestSpecs: Ask if user wants to add optional specs (if yes/no response, handle accordingly)
- awaitAdvancedSpecs: Offer advanced parameter selection
- showSummary: Present requirements summary, ask to proceed
- finalAnalysis: Confirm analysis is starting/complete

For yes/no questions, detect affirmative (yes, y, yeah, sure, ok) or negative (no, n, nope, skip) responses.

Return ONLY valid JSON:
{{
    "response": "<Your professional response to the user>",
    "next_step": "<Next workflow step>",
    "maintain_workflow": true,
    "user_confirmed": true | false | null,
    "action": "<action like 'proceed', 'collect_specs', 'skip', null>"
}}
"""

REQUIREMENTS_COLLECTION_PROMPT = """
You are Engenie - an expert assistant for industrial product requirements.

The user is adding requirements for a {product_type}.

User Input: "{user_message}"

Existing Requirements:
{existing_requirements}

Extract and structure the NEW requirements from the user's input.
Only include requirements explicitly mentioned in this message.

Return ONLY valid JSON:
{{
    "new_requirements": {{
        "<spec_name>": "<spec_value>"
    }},
    "merged_requirements": {{
        "<complete spec_name>": "<spec_value including existing and new>"
    }},
    "summary": "<Brief summary of what was added>"
}}
"""


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def detect_yes_no_response(message: str) -> Optional[str]:
    """
    Detect if user message is a yes/no response.
    
    Returns:
        'yes', 'no', or None if not a yes/no response
    """
    if not message:
        return None
    
    user_lower = message.lower().strip()
    
    affirmative_keywords = ['yes', 'y', 'yeah', 'yep', 'sure', 'ok', 'okay', 'proceed', 'continue', 'go ahead']
    negative_keywords = ['no', 'n', 'nope', 'skip', 'none', 'not needed', 'done', 'not interested']
    
    for keyword in affirmative_keywords:
        if keyword in user_lower:
            return 'yes'
    
    for keyword in negative_keywords:
        if keyword in user_lower:
            return 'no'
    
    return None


def format_parameters_display(params: List) -> str:
    """Format parameters list for display."""
    if not params:
        return "No parameters available."
    
    formatted = []
    for param in params:
        if isinstance(param, dict):
            name = param.get('name') or param.get('key') or (list(param.keys())[0] if param else '')
        else:
            name = str(param).strip()
        
        name = name.replace('_', ' ')
        name = re.split(r'[\(\[\{]', name, 1)[0].strip()
        name = " ".join(name.split())
        name = name.title()
        formatted.append(f"- {name}")
    
    return "\n".join(formatted)


# ============================================================================
# TOOLS
# ============================================================================

@tool("sales_agent_interact", args_schema=SalesAgentInput)
def sales_agent_interact_tool(
    step: str,
    user_message: str,
    product_type: Optional[str] = None,
    data_context: Optional[Dict[str, Any]] = None,
    intent: Optional[str] = None,
    session_id: Optional[str] = "default"
) -> Dict[str, Any]:
    """
    Handle user interaction in the product search workflow.
    Processes user input based on current step and returns appropriate response
    with next step guidance.
    """
    try:
        logger.info(f"[SALES_AGENT_TOOL] Step: {step}, Message: {user_message[:50]}...")
        
        llm = create_llm_with_fallback(
            model="gemini-2.5-flash",
            temperature=0.3,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        data_context = data_context or {}
        product_type = product_type or data_context.get('productType', 'industrial instrument')
        
        # Detect yes/no for workflow routing
        yes_no = detect_yes_no_response(user_message)
        
        # Handle specific steps with deterministic logic
        if step == 'awaitAdditionalAndLatestSpecs':
            if yes_no == 'no':
                return {
                    "success": True,
                    "response": "Understood. I'll proceed to the summary without additional specifications.",
                    "next_step": "showSummary",
                    "maintain_workflow": True,
                    "user_confirmed": False,
                    "action": "skip"
                }
            elif yes_no == 'yes':
                available_params = data_context.get('availableParameters', [])
                params_display = format_parameters_display(available_params)
                return {
                    "success": True,
                    "response": f"Great! Here are the additional specifications you can add:\n\n{params_display}\n\nPlease enter the specifications you'd like to include.",
                    "next_step": "awaitAdditionalAndLatestSpecs",
                    "maintain_workflow": True,
                    "user_confirmed": True,
                    "action": "collect_specs"
                }
        
        elif step == 'awaitAdvancedSpecs':
            available_params = data_context.get('availableParameters', [])
            
            if yes_no == 'no':
                return {
                    "success": True,
                    "response": "Understood. Let me prepare the summary of your requirements.",
                    "next_step": "showSummary",
                    "maintain_workflow": True,
                    "user_confirmed": False,
                    "action": "skip"
                }
            elif yes_no == 'yes' and available_params:
                params_display = format_parameters_display(available_params)
                return {
                    "success": True,
                    "response": f"Here are the advanced parameters available:\n\n{params_display}\n\nPlease specify the values for the parameters you need.",
                    "next_step": "awaitAdvancedSpecs",
                    "maintain_workflow": True,
                    "user_confirmed": True,
                    "action": "collect_specs"
                }
        
        elif step == 'showSummary':
            if yes_no == 'yes':
                return {
                    "success": True,
                    "response": "Starting the product analysis now. Please wait while I search for matching products...",
                    "next_step": "finalAnalysis",
                    "maintain_workflow": True,
                    "user_confirmed": True,
                    "action": "proceed"
                }
            elif yes_no == 'no':
                # User wants to modify requirements
                return {
                    "success": True,
                    "response": "What would you like to change? You can modify the product type, specifications, or requirements.",
                    "next_step": "showSummary",
                    "maintain_workflow": True,
                    "user_confirmed": False,
                    "action": "modify"
                }
            else:
                # Check for proceed-like messages that aren't strict yes/no
                proceed_patterns = ['run', 'analyze', 'search', 'find', 'start', 'do it', 'let\'s go', 'confirm']
                if any(pattern in user_message.lower() for pattern in proceed_patterns):
                    return {
                        "success": True,
                        "response": "Starting the product analysis now...",
                        "next_step": "finalAnalysis",
                        "maintain_workflow": True,
                        "user_confirmed": True,
                        "action": "proceed"
                    }
        
        # Fall back to LLM for complex interactions
        prompt = ChatPromptTemplate.from_template(SALES_AGENT_PROMPT)
        parser = JsonOutputParser()
        
        chain = prompt | llm | parser
        
        result = chain.invoke({
            "step": step,
            "user_message": user_message,
            "product_type": product_type or "industrial instrument"
        })
        
        return {
            "success": True,
            "response": result.get("response", "I understand. How can I help you further?"),
            "next_step": result.get("next_step", step),
            "maintain_workflow": result.get("maintain_workflow", True),
            "user_confirmed": result.get("user_confirmed"),
            "action": result.get("action")
        }
    
    except Exception as e:
        logger.error(f"[SALES_AGENT_TOOL] Error: {e}")
        return {
            "success": False,
            "response": "I apologize, but I'm having technical difficulties. Please try again.",
            "next_step": step,
            "maintain_workflow": True,
            "error": str(e)
        }


@tool("collect_requirements", args_schema=CollectRequirementsInput)
def collect_requirements_tool(
    user_message: str,
    product_type: str,
    existing_requirements: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Collect and merge additional requirements from user input.
    Extracts new specifications and combines with existing requirements.
    """
    try:
        logger.info(f"[COLLECT_REQUIREMENTS] Collecting requirements for {product_type}")
        
        llm = create_llm_with_fallback(
            model="gemini-2.5-flash",
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        prompt = ChatPromptTemplate.from_template(REQUIREMENTS_COLLECTION_PROMPT)
        parser = JsonOutputParser()
        
        chain = prompt | llm | parser
        
        existing_reqs_str = json.dumps(existing_requirements, indent=2) if existing_requirements else "{}"
        
        result = chain.invoke({
            "user_message": user_message,
            "product_type": product_type,
            "existing_requirements": existing_reqs_str
        })
        
        return {
            "success": True,
            "new_requirements": result.get("new_requirements", {}),
            "merged_requirements": result.get("merged_requirements", existing_requirements or {}),
            "summary": result.get("summary", "Requirements collected.")
        }
    
    except Exception as e:
        logger.error(f"[COLLECT_REQUIREMENTS] Error: {e}")
        return {
            "success": False,
            "new_requirements": {},
            "merged_requirements": existing_requirements or {},
            "error": str(e)
        }


@tool("generate_requirements_summary")
def generate_requirements_summary_tool(
    product_type: str,
    requirements: Dict[str, Any],
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a formatted summary of collected requirements.
    """
    try:
        logger.info(f"[SUMMARY] Generating summary for {product_type}")
        
        llm = create_llm_with_fallback(
            model="gemini-2.5-flash",
            temperature=0.2,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        summary_prompt = f"""
You are Engenie. Generate a clear, professional summary of the collected product requirements.

Product Type: {product_type}
Requirements:
{json.dumps(requirements, indent=2)}

Format the summary in a user-friendly way with:
1. Product type header
2. Key specifications (bullet points)
3. Optional specifications (if any)
4. A confirmation question asking if they want to proceed

Keep it concise but complete.
"""
        
        response = llm.invoke(summary_prompt)
        
        # Handle different response types
        if hasattr(response, 'content'):
            summary_text = response.content
        else:
            summary_text = str(response)
        
        return {
            "success": True,
            "summary": summary_text,
            "product_type": product_type,
            "requirements_count": len(requirements) if requirements else 0
        }
    
    except Exception as e:
        logger.error(f"[SUMMARY] Error: {e}")
        return {
            "success": False,
            "summary": f"Requirements Summary for {product_type}\n\n" + 
                      "\n".join([f"• {k}: {v}" for k, v in (requirements or {}).items()]) +
                      "\n\nWould you like to proceed with the product search?",
            "error": str(e)
        }


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "sales_agent_interact_tool",
    "collect_requirements_tool",
    "generate_requirements_summary_tool",
    "detect_yes_no_response",
    "format_parameters_display"
]
