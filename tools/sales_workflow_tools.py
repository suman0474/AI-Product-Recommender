# tools/sales_workflow_tools.py
# Sales Agent Workflow Tools - Extracted from sales_agent_workflow.py
# 
# This module provides LangChain tools for the Sales Agent FSM workflow:
# greeting → initialInput → awaitMissingInfo → awaitAdditionalAndLatestSpecs → 
# awaitAdvancedSpecs → showSummary → finalAnalysis

import json
import logging
import os
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field

from llm_fallback import create_llm_with_fallback

load_dotenv()
logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class SalesWorkflowStep(str, Enum):
    """Sales Agent workflow step identifiers - the FSM states"""
    GREETING = "greeting"
    INITIAL_INPUT = "initialInput"
    AWAIT_MISSING_INFO = "awaitMissingInfo"
    AWAIT_ADDITIONAL_SPECS = "awaitAdditionalAndLatestSpecs"
    AWAIT_ADVANCED_SPECS = "awaitAdvancedSpecs"
    SHOW_SUMMARY = "showSummary"
    FINAL_ANALYSIS = "finalAnalysis"
    END = "end"
    ERROR = "error"


class SalesWorkflowIntent(str, Enum):
    """Intent types for sales agent workflow"""
    GREETING = "greeting"
    PRODUCT_REQUIREMENTS = "productRequirements"
    KNOWLEDGE_QUESTION = "knowledgeQuestion"
    WORKFLOW = "workflow"
    CONFIRM = "confirm"
    REJECT = "reject"
    CHITCHAT = "chitchat"
    OTHER = "other"


# ============================================================================
# INPUT SCHEMAS
# ============================================================================

class IntentClassificationInput(BaseModel):
    """Input for sales workflow intent classification"""
    user_input: str = Field(description="User's input message")
    current_step: str = Field(default="greeting", description="Current workflow step")
    current_intent: Optional[str] = Field(default=None, description="Previous intent if any")


class GreetingGenerationInput(BaseModel):
    """Input for generating greeting message"""
    search_session_id: Optional[str] = Field(default=None, description="Session ID for personalization")


class RequirementsExtractionInput(BaseModel):
    """Input for extracting requirements from user input"""
    user_input: str = Field(description="User message containing requirements")
    product_type: Optional[str] = Field(default=None, description="Known product type if any")


class MergeRequirementsInput(BaseModel):
    """Input for merging new requirements with existing ones"""
    user_input: str = Field(description="User's input with new requirements")
    product_type: str = Field(description="Product type being searched")
    existing_requirements: Dict[str, Any] = Field(default_factory=dict, description="Already collected requirements")


class ParameterSelectionInput(BaseModel):
    """Input for advanced parameter selection"""
    user_input: str = Field(description="User's parameter selections")
    product_type: str = Field(description="Product type")
    available_parameters: List[Dict[str, Any]] = Field(description="List of available parameters")


class SummaryGenerationInput(BaseModel):
    """Input for generating requirements summary"""
    product_type: str = Field(description="Product type")
    provided_requirements: Dict[str, Any] = Field(default_factory=dict)
    additional_params: Dict[str, Any] = Field(default_factory=dict)
    advanced_params: Dict[str, Any] = Field(default_factory=dict)


class SalesAgentResponseInput(BaseModel):
    """Input for generating sales agent response"""
    step: str = Field(description="Current workflow step")
    context: Dict[str, Any] = Field(default_factory=dict, description="Current state context")
    user_input: Optional[str] = Field(default=None, description="User's input if any")


# ============================================================================
# PROMPTS
# ============================================================================

INTENT_CLASSIFIER_PROMPT = """
You are Engenie - an intelligent assistant for industrial product procurement.

Analyze the user's input and determine their intent within the workflow context.

Current workflow step: {current_step}
Current intent: {current_intent}
User Input: "{user_input}"

**Intent Categories:**
1. "greeting" - Simple greeting (hi, hello, hey)
2. "productRequirements" - Technical specs or product requests
3. "knowledgeQuestion" - Questions about industrial topics
4. "confirm" - User confirms (yes, proceed, continue, ok)
5. "reject" - User rejects (no, cancel, stop)
6. "chitchat" - Casual conversation
7. "other" - Unclassified

Return ONLY valid JSON:
{{
    "intent": "<intent_type>",
    "nextStep": "<suggested_next_workflow_step or null>",
    "resumeWorkflow": <true if should resume after handling, false otherwise>,
    "confidence": <0.0-1.0>
}}
"""

GREETING_PROMPT = """
You are Engenie - a professional, warm, and helpful AI sales agent for industrial automation.

Generate a brief, welcoming greeting for a new customer session.
Session ID: {search_session_id}

Keep it:
- Professional but friendly
- Brief (1-2 sentences)
- Inviting them to describe what they're looking for

Do NOT use bullet points or lists in greeting. Keep it conversational.
"""

REQUIREMENTS_EXTRACTION_PROMPT = """
Extract the product type and detailed specifications from this user input.

User Input: "{user_input}"
{product_type_hint}

Focus on:
- Product type (pressure transmitter, flow meter, level sensor, etc.)
- Technical specifications (pressure range, accuracy, output signal, etc.)
- Connection types (NPT, flanged, etc.)
- Environmental requirements (temperature, hazardous area, etc.)
- Certifications (ATEX, SIL, FM, etc.)

Return ONLY valid JSON:
{{
    "product_type": "<detected product type>",
    "specifications": {{
        "<spec_name>": "<spec_value>"
    }},
    "missing_mandatory": ["<list of commonly required specs not provided>"],
    "confidence": <0.0-1.0>
}}
"""

MERGE_REQUIREMENTS_PROMPT = """
Extract NEW specifications from this user input for a {product_type}.

User Input: "{user_input}"

Existing Requirements:
{existing_requirements}

Only include specifications that are NEW or UPDATED from this specific message.

Return ONLY valid JSON:
{{
    "new_requirements": {{
        "<spec_name>": "<spec_value>"
    }},
    "merged_requirements": {{
        "<complete spec_name>": "<spec_value including existing and new>"
    }},
    "summary": "<brief description of what was added/changed>"
}}
"""

PARAMETER_SELECTION_PROMPT = """
Match the user's selections against available parameters for a {product_type}.

Available Parameters:
{available_parameters}

User Input: "{user_input}"

Identify which parameters the user is selecting and their values.

Return ONLY valid JSON:
{{
    "selected_parameters": {{
        "<param_name>": "<user_specified_value>"
    }},
    "unmatched_input": ["<any input that didn't match available params>"]
}}
"""

SUMMARY_GENERATION_PROMPT = """
Generate a clear, professional requirements summary.

Product Type: {product_type}

Core Requirements:
{provided_requirements}

Additional Specifications:
{additional_params}

Advanced Parameters:
{advanced_params}

Format the summary with:
1. Product type header
2. Key specifications (bullet points)
3. Optional specifications (if any)
4. Ask for confirmation to proceed

Keep it professional and concise.
"""

SALES_RESPONSE_PROMPTS = {
    "greeting": """
You are Engenie. Generate a brief, professional greeting.
Keep it conversational and invite them to describe their needs.
""",
    "initial_input_ack": """
You are Engenie. The user has provided requirements for a {product_type}.
Acknowledge their requirements briefly and ask if they want to add additional specifications.
Keep response to 2-3 sentences maximum.
""",
    "missing_info": """
You are Engenie. We need more information about {missing_fields} for the {product_type}.
Ask for this information in a helpful, non-pushy way.
""",
    "additional_specs_yes": """
You are Engenie. The user wants to add additional specifications.
Here are available options:
{params_display}

Present these options and ask which they'd like to add.
""",
    "additional_specs_no": """
You are Engenie. The user doesn't want additional specs.
Acknowledge and transition to asking about advanced parameters.
One sentence only.
""",
    "advanced_specs_yes": """
You are Engenie. The user wants to add advanced parameters.
Here are available options:
{params_display}

Present these options and ask which they'd like to configure.
""",
    "advanced_specs_no": """
You are Engenie. The user doesn't need advanced parameters.
Acknowledge briefly and indicate you'll prepare the summary.
One sentence only.
""",
    "summary_proceed": """
You are Engenie. Generate a brief acknowledgment that we're starting the product search.
Keep it to one encouraging sentence.
""",
    "final_analysis": """
You are Engenie. We found {count} matching products.
Generate a brief, enthusiastic summary indicating analysis is complete.
""",
    "error": """
You are Engenie. Something went wrong but we want to help.
Apologize briefly and offer to try again.
""",
    "knowledge_question": """
You are Engenie. Answer the user's question about industrial topics.
User Question: "{user_message}"

After answering, remind them:
{context_hint}

Keep answer concise but informative.
""",
    "default": """
You are Engenie. The user said: "{user_message}"
This doesn't seem to match our current workflow.
Politely guide them back to the product search task.
"""
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_llm(temperature: float = 0.1):
    """Get LLM with fallback configuration"""
    return create_llm_with_fallback(
        model="gemini-2.5-flash",
        temperature=temperature,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )


def detect_yes_no_response(message: str) -> Optional[str]:
    """
    Detect if user message is a yes/no response.
    Returns 'yes', 'no', or None if not a yes/no response
    """
    if not message:
        return None
    
    message_lower = message.strip().lower()
    
    # Yes patterns
    yes_patterns = [
        "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "proceed",
        "continue", "go ahead", "let's go", "do it", "affirmative",
        "sounds good", "perfect", "great", "absolutely", "definitely",
        "of course", "please", "y"
    ]
    
    # No patterns
    no_patterns = [
        "no", "nope", "nah", "cancel", "stop", "nevermind", "never mind",
        "skip", "not now", "not yet", "hold", "wait", "n", "negative",
        "don't", "dont", "no thanks", "no thank you"
    ]
    
    for pattern in yes_patterns:
        if pattern in message_lower or message_lower == pattern:
            return "yes"
    
    for pattern in no_patterns:
        if pattern in message_lower or message_lower == pattern:
            return "no"
    
    return None


def format_parameters_display(params: List[Dict[str, Any]]) -> str:
    """Format parameters list for display to user"""
    if not params:
        return "No parameters available."
    
    lines = []
    for i, param in enumerate(params, 1):
        name = param.get("name", param.get("field", param.get("key", "Unknown")))
        description = param.get("description", "")
        if description:
            lines.append(f"{i}. **{name}**: {description}")
        else:
            lines.append(f"{i}. **{name}**")
    
    return "\n".join(lines)


def format_requirements_summary(product_type: str, requirements: Dict[str, Any]) -> str:
    """Format requirements for summary display"""
    lines = [f"**Product Type:** {product_type}", "", "**Specifications:**"]
    
    for key, value in requirements.items():
        if value:
            display_key = key.replace("_", " ").title()
            lines.append(f"• {display_key}: {value}")
    
    return "\n".join(lines)


# ============================================================================
# LANGCHAIN TOOLS
# ============================================================================

@tool("classify_sales_intent", args_schema=IntentClassificationInput)
def classify_sales_intent_tool(
    user_input: str,
    current_step: str = "greeting",
    current_intent: Optional[str] = None
) -> Dict[str, Any]:
    """
    Classify user intent within the sales workflow context.
    Returns intent type, suggested next step, and whether to resume workflow.
    """
    try:
        logger.info(f"[SALES_INTENT] Classifying: {user_input[:50]}... step={current_step}")
        
        llm = get_llm(temperature=0.1)
        
        prompt = ChatPromptTemplate.from_template(INTENT_CLASSIFIER_PROMPT)
        chain = prompt | llm | JsonOutputParser()
        
        result = chain.invoke({
            "current_step": current_step,
            "current_intent": current_intent or "none",
            "user_input": user_input
        })
        
        return {
            "success": True,
            "intent": result.get("intent", "other"),
            "next_step": result.get("nextStep"),
            "resume_workflow": result.get("resumeWorkflow", False),
            "confidence": result.get("confidence", 0.5)
        }
        
    except Exception as e:
        logger.error(f"[SALES_INTENT] Error: {e}")
        return {
            "success": False,
            "intent": "other",
            "next_step": None,
            "resume_workflow": False,
            "error": str(e)
        }


@tool("generate_sales_greeting", args_schema=GreetingGenerationInput)
def generate_sales_greeting_tool(
    search_session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a professional greeting for the sales agent.
    """
    try:
        logger.info("[SALES_GREETING] Generating greeting")
        
        llm = get_llm(temperature=0.7)
        
        prompt = ChatPromptTemplate.from_template(GREETING_PROMPT)
        chain = prompt | llm | StrOutputParser()
        
        response = chain.invoke({
            "search_session_id": search_session_id or str(uuid.uuid4())[:8]
        })
        
        return {
            "success": True,
            "response": response,
            "next_step": SalesWorkflowStep.INITIAL_INPUT.value
        }
        
    except Exception as e:
        logger.error(f"[SALES_GREETING] Error: {e}")
        return {
            "success": True,
            "response": "Hello! I'm Engenie, your industrial sales assistant. How can I help you find the right product today?",
            "next_step": SalesWorkflowStep.INITIAL_INPUT.value,
            "error": str(e)
        }


@tool("extract_product_requirements", args_schema=RequirementsExtractionInput)
def extract_product_requirements_tool(
    user_input: str,
    product_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extract structured product requirements from user input.
    Returns product type, specifications, and commonly missing mandatory fields.
    """
    try:
        logger.info(f"[EXTRACT_REQUIREMENTS] From: {user_input[:50]}...")
        
        llm = get_llm(temperature=0.1)
        
        product_type_hint = f"Hint: Product appears to be a {product_type}" if product_type else ""
        
        prompt = ChatPromptTemplate.from_template(REQUIREMENTS_EXTRACTION_PROMPT)
        chain = prompt | llm | JsonOutputParser()
        
        result = chain.invoke({
            "user_input": user_input,
            "product_type_hint": product_type_hint
        })
        
        return {
            "success": True,
            "product_type": result.get("product_type", "Unknown product"),
            "specifications": result.get("specifications", {}),
            "missing_mandatory": result.get("missing_mandatory", []),
            "confidence": result.get("confidence", 0.7)
        }
        
    except Exception as e:
        logger.error(f"[EXTRACT_REQUIREMENTS] Error: {e}")
        return {
            "success": False,
            "product_type": product_type or "Unknown",
            "specifications": {},
            "missing_mandatory": [],
            "error": str(e)
        }


@tool("merge_sales_requirements", args_schema=MergeRequirementsInput)
def merge_sales_requirements_tool(
    user_input: str,
    product_type: str,
    existing_requirements: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Extract new requirements from user input and merge with existing requirements.
    """
    try:
        logger.info(f"[MERGE_REQUIREMENTS] Merging for {product_type}")
        
        existing_requirements = existing_requirements or {}
        llm = get_llm(temperature=0.1)
        
        prompt = ChatPromptTemplate.from_template(MERGE_REQUIREMENTS_PROMPT)
        chain = prompt | llm | JsonOutputParser()
        
        result = chain.invoke({
            "user_input": user_input,
            "product_type": product_type,
            "existing_requirements": json.dumps(existing_requirements, indent=2)
        })
        
        return {
            "success": True,
            "new_requirements": result.get("new_requirements", {}),
            "merged_requirements": result.get("merged_requirements", existing_requirements),
            "summary": result.get("summary", "Requirements updated")
        }
        
    except Exception as e:
        logger.error(f"[MERGE_REQUIREMENTS] Error: {e}")
        return {
            "success": False,
            "new_requirements": {},
            "merged_requirements": existing_requirements or {},
            "error": str(e)
        }


@tool("select_advanced_parameters", args_schema=ParameterSelectionInput)
def select_advanced_parameters_tool(
    user_input: str,
    product_type: str,
    available_parameters: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Match user's parameter selections against available advanced parameters.
    """
    try:
        logger.info(f"[PARAM_SELECTION] Selecting for {product_type}")
        
        llm = get_llm(temperature=0.1)
        
        params_display = format_parameters_display(available_parameters)
        
        prompt = ChatPromptTemplate.from_template(PARAMETER_SELECTION_PROMPT)
        chain = prompt | llm | JsonOutputParser()
        
        result = chain.invoke({
            "user_input": user_input,
            "product_type": product_type,
            "available_parameters": params_display
        })
        
        return {
            "success": True,
            "selected_parameters": result.get("selected_parameters", {}),
            "unmatched_input": result.get("unmatched_input", [])
        }
        
    except Exception as e:
        logger.error(f"[PARAM_SELECTION] Error: {e}")
        return {
            "success": False,
            "selected_parameters": {},
            "unmatched_input": [],
            "error": str(e)
        }


@tool("generate_sales_summary", args_schema=SummaryGenerationInput)
def generate_sales_summary_tool(
    product_type: str,
    provided_requirements: Dict[str, Any] = None,
    additional_params: Dict[str, Any] = None,
    advanced_params: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Generate a formatted summary of all collected requirements.
    """
    try:
        logger.info(f"[GENERATE_SUMMARY] For {product_type}")
        
        provided_requirements = provided_requirements or {}
        additional_params = additional_params or {}
        advanced_params = advanced_params or {}
        
        llm = get_llm(temperature=0.2)
        
        prompt = ChatPromptTemplate.from_template(SUMMARY_GENERATION_PROMPT)
        chain = prompt | llm | StrOutputParser()
        
        response = chain.invoke({
            "product_type": product_type,
            "provided_requirements": json.dumps(provided_requirements, indent=2),
            "additional_params": json.dumps(additional_params, indent=2) if additional_params else "None",
            "advanced_params": json.dumps(advanced_params, indent=2) if advanced_params else "None"
        })
        
        # Combine all requirements for data return
        all_requirements = {
            **provided_requirements,
            **additional_params,
            **advanced_params
        }
        
        return {
            "success": True,
            "summary": response,
            "product_type": product_type,
            "all_requirements": all_requirements,
            "requirements_count": len(all_requirements)
        }
        
    except Exception as e:
        logger.error(f"[GENERATE_SUMMARY] Error: {e}")
        # Fallback to basic formatting
        all_reqs = {**(provided_requirements or {}), **(additional_params or {}), **(advanced_params or {})}
        fallback_summary = format_requirements_summary(product_type, all_reqs)
        return {
            "success": True,
            "summary": fallback_summary + "\n\nWould you like to proceed with the analysis?",
            "product_type": product_type,
            "all_requirements": all_reqs,
            "error": str(e)
        }


@tool("generate_sales_response", args_schema=SalesAgentResponseInput)
def generate_sales_response_tool(
    step: str,
    context: Dict[str, Any] = None,
    user_input: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate an appropriate sales agent response based on workflow step.
    
    Steps: greeting, initial_input_ack, missing_info, additional_specs_yes,
    additional_specs_no, advanced_specs_yes, advanced_specs_no, summary_proceed,
    final_analysis, error, knowledge_question, default
    """
    try:
        logger.info(f"[SALES_RESPONSE] Step: {step}")
        
        context = context or {}
        llm = get_llm(temperature=0.5)
        
        # Get the appropriate prompt template
        prompt_template = SALES_RESPONSE_PROMPTS.get(step, SALES_RESPONSE_PROMPTS["default"])
        
        # Build context variables based on step
        prompt_vars = {
            "product_type": context.get("product_type", "industrial product"),
            "missing_fields": context.get("missing_fields", "required specifications"),
            "params_display": context.get("params_display", "No parameters available"),
            "count": context.get("count", 0),
            "user_message": user_input or "",
            "context_hint": context.get("context_hint", "How else can I help you?")
        }
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm | StrOutputParser()
        
        response = chain.invoke(prompt_vars)
        
        # Determine next step based on current step
        next_step_map = {
            "greeting": SalesWorkflowStep.INITIAL_INPUT.value,
            "initial_input_ack": SalesWorkflowStep.AWAIT_ADDITIONAL_SPECS.value,
            "missing_info": SalesWorkflowStep.AWAIT_MISSING_INFO.value,
            "additional_specs_yes": SalesWorkflowStep.AWAIT_ADDITIONAL_SPECS.value,
            "additional_specs_no": SalesWorkflowStep.AWAIT_ADVANCED_SPECS.value,
            "advanced_specs_yes": SalesWorkflowStep.AWAIT_ADVANCED_SPECS.value,
            "advanced_specs_no": SalesWorkflowStep.SHOW_SUMMARY.value,
            "summary_proceed": SalesWorkflowStep.FINAL_ANALYSIS.value,
            "final_analysis": SalesWorkflowStep.END.value,
            "error": SalesWorkflowStep.ERROR.value,
        }
        
        return {
            "success": True,
            "response": response,
            "step": step,
            "next_step": next_step_map.get(step, step)
        }
        
    except Exception as e:
        logger.error(f"[SALES_RESPONSE] Error: {e}")
        return {
            "success": False,
            "response": "I'm here to help you find the right industrial product. What are you looking for?",
            "step": step,
            "error": str(e)
        }


@tool("detect_user_confirmation")
def detect_user_confirmation_tool(user_input: str) -> Dict[str, Any]:
    """
    Detect if user's response is a confirmation (yes/no) or something else.
    Returns 'yes', 'no', or 'other' with suggested action.
    """
    result = detect_yes_no_response(user_input)
    
    if result == "yes":
        return {
            "response_type": "yes",
            "action": "proceed",
            "is_confirmation": True
        }
    elif result == "no":
        return {
            "response_type": "no", 
            "action": "skip",
            "is_confirmation": True
        }
    else:
        return {
            "response_type": "other",
            "action": "process_input",
            "is_confirmation": False,
            "original_input": user_input
        }


@tool("get_workflow_step_info")
def get_workflow_step_info_tool(step: str) -> Dict[str, Any]:
    """
    Get information about a workflow step including valid transitions.
    """
    step_info = {
        SalesWorkflowStep.GREETING.value: {
            "description": "Initial greeting to user",
            "next_steps": [SalesWorkflowStep.INITIAL_INPUT.value],
            "awaits_input": True
        },
        SalesWorkflowStep.INITIAL_INPUT.value: {
            "description": "Processing user's initial product requirements",
            "next_steps": [
                SalesWorkflowStep.AWAIT_MISSING_INFO.value,
                SalesWorkflowStep.AWAIT_ADDITIONAL_SPECS.value
            ],
            "awaits_input": False
        },
        SalesWorkflowStep.AWAIT_MISSING_INFO.value: {
            "description": "Waiting for missing mandatory information",
            "next_steps": [SalesWorkflowStep.AWAIT_ADDITIONAL_SPECS.value],
            "awaits_input": True
        },
        SalesWorkflowStep.AWAIT_ADDITIONAL_SPECS.value: {
            "description": "Asking about additional specifications",
            "next_steps": [
                SalesWorkflowStep.AWAIT_ADDITIONAL_SPECS.value,  # If collecting
                SalesWorkflowStep.AWAIT_ADVANCED_SPECS.value
            ],
            "awaits_input": True
        },
        SalesWorkflowStep.AWAIT_ADVANCED_SPECS.value: {
            "description": "Offering advanced parameter selection",
            "next_steps": [
                SalesWorkflowStep.AWAIT_ADVANCED_SPECS.value,  # If collecting
                SalesWorkflowStep.SHOW_SUMMARY.value
            ],
            "awaits_input": True
        },
        SalesWorkflowStep.SHOW_SUMMARY.value: {
            "description": "Displaying requirements summary for confirmation",
            "next_steps": [
                SalesWorkflowStep.FINAL_ANALYSIS.value,
                SalesWorkflowStep.AWAIT_ADDITIONAL_SPECS.value  # If user wants changes
            ],
            "awaits_input": True
        },
        SalesWorkflowStep.FINAL_ANALYSIS.value: {
            "description": "Performing product search and analysis",
            "next_steps": [SalesWorkflowStep.END.value],
            "awaits_input": False
        },
        SalesWorkflowStep.END.value: {
            "description": "Workflow complete",
            "next_steps": [],
            "awaits_input": False
        },
        SalesWorkflowStep.ERROR.value: {
            "description": "Error occurred in workflow",
            "next_steps": [SalesWorkflowStep.GREETING.value],  # Can restart
            "awaits_input": True
        }
    }
    
    info = step_info.get(step, {
        "description": "Unknown step",
        "next_steps": [],
        "awaits_input": True
    })
    
    return {
        "step": step,
        **info
    }


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "SalesWorkflowStep",
    "SalesWorkflowIntent",
    # Tools
    "classify_sales_intent_tool",
    "generate_sales_greeting_tool",
    "extract_product_requirements_tool",
    "merge_sales_requirements_tool", 
    "select_advanced_parameters_tool",
    "generate_sales_summary_tool",
    "generate_sales_response_tool",
    "detect_user_confirmation_tool",
    "get_workflow_step_info_tool",
    # Helpers
    "detect_yes_no_response",
    "format_parameters_display",
    "format_requirements_summary"
]
