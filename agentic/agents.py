# agentic/agents.py
# LangChain Agents for Procurement Workflow

import json
import logging
from typing import Dict, Any, List, Optional, Type
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

# Import tools
from tools.intent_tools import classify_intent_tool, extract_requirements_tool
from tools.schema_tools import load_schema_tool, validate_requirements_tool, get_missing_fields_tool
from tools.vendor_tools import search_vendors_tool, get_vendor_products_tool, fuzzy_match_vendors_tool
from tools.analysis_tools import analyze_vendor_match_tool, calculate_match_score_tool, extract_specifications_tool
from tools.search_tools import search_product_images_tool, search_pdf_datasheets_tool, web_search_tool
from tools.ranking_tools import rank_products_tool, judge_analysis_tool
from tools.instrument_tools import identify_instruments_tool, identify_accessories_tool

from .models import WorkflowState, IntentType, WorkflowStep
from llm_fallback import create_llm_with_fallback

load_dotenv()
logger = logging.getLogger(__name__)


# ============================================================================
# BASE AGENT CLASS
# ============================================================================

class BaseAgent:
    """Base class for all agents using LangGraph's create_react_agent"""

    def __init__(
        self,
        name: str,
        description: str,
        tools: List[BaseTool],
        system_prompt: str,
        model_name: str = "gemini-2.0-flash-exp",
        temperature: float = 0.1
    ):
        self.name = name
        self.description = description
        self.tools = tools
        self.system_prompt = system_prompt

        # Initialize LLM
        self.llm = create_llm_with_fallback(
            model=model_name,
            temperature=temperature,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        # Create agent
        self._create_agent()

    def _create_agent(self):
        """Create the agent with tools using LangGraph's create_react_agent"""
        if self.tools:
            # Create a react agent using LangGraph prebuilt
            self.agent = create_react_agent(
                model=self.llm,
                tools=self.tools,
                prompt=self.system_prompt
            )
        else:
            self.agent = None

    def run(self, input_data: Dict[str, Any], chat_history: List = None, 
            user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run the agent with authentication and audit logging.
        
        Args:
            input_data: Input data for the agent
            chat_history: Optional chat history
            user_context: Optional user context with user_id, session_id, etc.
            
        Returns:
            Dict with success status and output/error
        """
        # Extract user context
        if user_context is None:
            user_context = {}
            
        # Try to get user_id from Flask session if in request context
        if has_request_context() and 'user_id' in session:
            user_context['user_id'] = session.get('user_id')
            user_context['session_id'] = session.get('session_id', 'unknown')
            
        user_id = user_context.get('user_id', 'anonymous')
        session_id = user_context.get('session_id', 'unknown')
        
        # Audit log start
        logger.info(
            f"Agent {self.name} invoked by user {user_id}",
            extra={
                'agent': self.name,
                'user_id': user_id,
                'session_id': session_id,
                'timestamp': datetime.now().isoformat()
            }
        )
        
        try:
            if self.agent:
                # Prepare messages for the react agent
                messages = []
                if chat_history:
                    messages.extend(chat_history)
                
                user_message = json.dumps(input_data) if isinstance(input_data, dict) else str(input_data)
                messages.append(HumanMessage(content=user_message))
                
                # Invoke the LangGraph react agent
                result = self.agent.invoke({"messages": messages})
                
                # Extract output from result
                output = result.get("messages", [])[-1].content if result.get("messages") else ""
                
                # Audit log success
                logger.info(
                    f"Agent {self.name} completed successfully",
                    extra={
                        'agent': self.name,
                        'user_id': user_id,
                        'session_id': session_id,
                        'output_length': len(str(output))
                    }
                )
                
                return {
                    "success": True,
                    "output": output,
                    "agent": self.name,
                    "user_context": user_context
                }
            else:
                # Direct LLM call for agents without tools
                response = self.llm.invoke(str(input_data))
                
                # Audit log success
                logger.info(
                    f"Agent {self.name} (direct LLM) completed successfully",
                    extra={
                        'agent': self.name,
                        'user_id': user_id,
                        'output_length': len(response.content)
                    }
                )
                
                return {
                    "success": True,
                    "output": response.content,
                    "agent": self.name,
                    "user_context": user_context
                }
        except Exception as e:
            # Audit log failure
            logger.error(
                f"Agent {self.name} failed: {e}",
                exc_info=True,
                extra={
                    'agent': self.name,
                    'user_id': user_id,
                    'session_id': session_id,
                    'error': str(e)
                }
            )
            
            return {
                "success": False,
                "error": str(e),
                "agent": self.name,
                "user_context": user_context
            }


# ============================================================================
# SPECIALIZED AGENTS
# ============================================================================

class IntentClassifierAgent(BaseAgent):
    """Agent for classifying user intent"""

    SYSTEM_PROMPT = """
You are Engenie's Intent Classifier - a smart assistant that understands user intentions
in an industrial procurement context.

Your job is to:
1. Classify user input into categories (greeting, requirements, question, etc.)
2. Extract any relevant information from the input
3. Suggest the next workflow step

Use the classify_intent and extract_requirements tools to analyze user input.
Always return structured JSON with your classification results.
"""

    def __init__(self):
        super().__init__(
            name="IntentClassifierAgent",
            description="Classifies user intent and extracts initial information",
            tools=[classify_intent_tool, extract_requirements_tool],
            system_prompt=self.SYSTEM_PROMPT
        )

    def classify(self, user_input: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Classify user intent"""
        result = classify_intent_tool.invoke({
            "user_input": user_input,
            "current_step": "start",
            "context": context
        })
        return result


class ValidationAgent(BaseAgent):
    """Agent for validating user requirements"""

    SYSTEM_PROMPT = """
You are Engenie's Validation Agent - an expert in industrial instrumentation requirements.

Your job is to:
1. Load the appropriate schema for the product type
2. Validate user requirements against the schema
3. Identify missing mandatory and optional fields
4. Provide clear feedback on what's needed

Use the load_schema, validate_requirements, and get_missing_fields tools.
Be helpful and guide users to provide complete requirements.
"""

    def __init__(self):
        super().__init__(
            name="ValidationAgent",
            description="Validates requirements against product schemas",
            tools=[load_schema_tool, validate_requirements_tool, get_missing_fields_tool],
            system_prompt=self.SYSTEM_PROMPT
        )

    def validate(self, user_input: str, product_type: str) -> Dict[str, Any]:
        """Validate user requirements"""
        # Load schema
        schema_result = load_schema_tool.invoke({"product_type": product_type})
        schema = schema_result.get("schema", {})

        # Validate requirements
        validation_result = validate_requirements_tool.invoke({
            "user_input": user_input,
            "product_type": product_type,
            "schema": schema
        })

        return {
            "schema": schema,
            "validation": validation_result
        }


class VendorSearchAgent(BaseAgent):
    """Agent for searching and matching vendors"""

    SYSTEM_PROMPT = """
You are Engenie's Vendor Search Agent - an expert in finding the right industrial vendors.

Your job is to:
1. Search for vendors that offer the required product type
2. Filter vendors based on user preferences or CSV uploads
3. Perform fuzzy matching to find the best vendor matches
4. Return a prioritized list of vendors to analyze

Use the search_vendors, get_vendor_products, and fuzzy_match_vendors tools.
"""

    def __init__(self):
        super().__init__(
            name="VendorSearchAgent",
            description="Searches and filters vendors for product requirements",
            tools=[search_vendors_tool, get_vendor_products_tool, fuzzy_match_vendors_tool],
            system_prompt=self.SYSTEM_PROMPT
        )

    def search(self, product_type: str, requirements: Dict[str, Any], vendor_filter: List[str] = None) -> Dict[str, Any]:
        """Search for vendors"""
        result = search_vendors_tool.invoke({
            "product_type": product_type,
            "requirements": requirements
        })

        if vendor_filter and result.get("success"):
            # Apply fuzzy matching filter
            match_result = fuzzy_match_vendors_tool.invoke({
                "vendor_names": result.get("vendors", []),
                "allowed_vendors": vendor_filter,
                "threshold": 70
            })
            result["filtered_vendors"] = [m["original"] for m in match_result.get("matched_vendors", [])]

        return result


class ProductAnalysisAgent(BaseAgent):
    """Agent for analyzing vendor products against requirements"""

    SYSTEM_PROMPT = """
You are Engenie's Product Analysis Agent - a meticulous technical matching expert.

Your job is to:
1. Analyze vendor products against user requirements
2. Extract specifications from PDF datasheets
3. Calculate match scores
4. Identify strengths and limitations

Use the analyze_vendor_match, extract_specifications, and calculate_match_score tools.
Provide detailed parameter-by-parameter analysis.
"""

    def __init__(self):
        super().__init__(
            name="ProductAnalysisAgent",
            description="Analyzes vendor products against requirements",
            tools=[analyze_vendor_match_tool, extract_specifications_tool, calculate_match_score_tool],
            system_prompt=self.SYSTEM_PROMPT
        )

    def analyze(self, vendor: str, requirements: Dict[str, Any], pdf_content: str = None, product_data: Dict = None) -> Dict[str, Any]:
        """Analyze vendor product match"""
        result = analyze_vendor_match_tool.invoke({
            "vendor": vendor,
            "requirements": requirements,
            "pdf_content": pdf_content,
            "product_data": product_data
        })
        return result


class RankingAgent(BaseAgent):
    """Agent for ranking products"""

    SYSTEM_PROMPT = """
You are Engenie's Ranking Agent - a product ranking specialist.

Your job is to:
1. Review all vendor analysis results
2. Rank products based on match scores and requirements
3. Identify the top picks with clear reasoning
4. Provide actionable recommendations

Use the rank_products and judge_analysis tools.
Create clear, business-relevant rankings.
"""

    def __init__(self):
        super().__init__(
            name="RankingAgent",
            description="Ranks products based on analysis results",
            tools=[rank_products_tool, judge_analysis_tool],
            system_prompt=self.SYSTEM_PROMPT
        )

    def rank(self, vendor_matches: List[Dict], requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Rank products"""
        result = rank_products_tool.invoke({
            "vendor_matches": vendor_matches,
            "requirements": requirements
        })
        return result


class SalesAgent(BaseAgent):
    """Conversational agent for guiding users through the workflow"""

    SYSTEM_PROMPT = """
You are Engenie - a friendly and professional industrial sales consultant.

Your job is to:
1. Guide users through the procurement workflow
2. Ask clarifying questions when needed
3. Explain technical concepts in simple terms
4. Provide helpful suggestions and recommendations

Be conversational, helpful, and professional.
Keep responses concise but informative.
"""

    def __init__(self):
        super().__init__(
            name="SalesAgent",
            description="Conversational agent for user guidance",
            tools=[],  # No tools, uses direct LLM
            system_prompt=self.SYSTEM_PROMPT
        )

    def respond(self, context: str, user_message: str) -> str:
        """Generate a response"""
        prompt = f"""
Context: {context}

User: {user_message}

Respond helpfully and professionally:
"""
        response = self.llm.invoke(prompt)
        return response.content


class InstrumentIdentifierAgent(BaseAgent):
    """Agent for identifying instruments from process descriptions"""

    SYSTEM_PROMPT = """
You are Engenie's Instrument Identifier - an expert in Industrial Process Control Systems.

Your job is to:
1. Analyze process descriptions and requirements
2. Identify all instruments needed
3. Extract specifications for each instrument
4. Identify required accessories

Use the identify_instruments and identify_accessories tools.
Create a complete Bill of Materials.
"""

    def __init__(self):
        super().__init__(
            name="InstrumentIdentifierAgent",
            description="Identifies instruments from process requirements",
            tools=[identify_instruments_tool, identify_accessories_tool],
            system_prompt=self.SYSTEM_PROMPT
        )

    def identify(self, requirements: str) -> Dict[str, Any]:
        """Identify instruments and accessories"""
        # Identify instruments
        instruments_result = identify_instruments_tool.invoke({"requirements": requirements})

        # Identify accessories
        if instruments_result.get("success") and instruments_result.get("instruments"):
            accessories_result = identify_accessories_tool.invoke({
                "instruments": instruments_result["instruments"],
                "process_context": requirements
            })
            instruments_result["accessories"] = accessories_result.get("accessories", [])

        return instruments_result


class ImageSearchAgent(BaseAgent):
    """Agent for searching product images"""

    SYSTEM_PROMPT = """
You are Engenie's Image Search Agent - finding the best product images.

Your job is to:
1. Search for product images from multiple sources
2. Prioritize manufacturer official images
3. Return high-quality images with metadata

Use the search_product_images tool.
"""

    def __init__(self):
        super().__init__(
            name="ImageSearchAgent",
            description="Searches for product images",
            tools=[search_product_images_tool],
            system_prompt=self.SYSTEM_PROMPT
        )

    def search(self, vendor: str, product_name: str, product_type: str) -> Dict[str, Any]:
        """Search for product images"""
        result = search_product_images_tool.invoke({
            "vendor": vendor,
            "product_name": product_name,
            "product_type": product_type
        })
        return result


class PDFSearchAgent(BaseAgent):
    """Agent for searching PDF datasheets"""

    SYSTEM_PROMPT = """
You are Engenie's PDF Search Agent - finding product datasheets.

Your job is to:
1. Search for PDF datasheets from multiple sources
2. Prioritize manufacturer official documents
3. Return relevant PDFs with metadata

Use the search_pdf_datasheets tool.
"""

    def __init__(self):
        super().__init__(
            name="PDFSearchAgent",
            description="Searches for PDF datasheets",
            tools=[search_pdf_datasheets_tool],
            system_prompt=self.SYSTEM_PROMPT
        )

    def search(self, vendor: str, product_type: str, model_family: str = None) -> Dict[str, Any]:
        """Search for PDF datasheets"""
        result = search_pdf_datasheets_tool.invoke({
            "vendor": vendor,
            "product_type": product_type,
            "model_family": model_family
        })
        return result


# ============================================================================
# AGENT FACTORY
# ============================================================================

class AgentFactory:
    """Factory for creating agents"""

    _agents = {
        "intent_classifier": IntentClassifierAgent,
        "validation": ValidationAgent,
        "vendor_search": VendorSearchAgent,
        "product_analysis": ProductAnalysisAgent,
        "ranking": RankingAgent,
        "sales": SalesAgent,
        "instrument_identifier": InstrumentIdentifierAgent,
        "image_search": ImageSearchAgent,
        "pdf_search": PDFSearchAgent
    }

    @classmethod
    def create(cls, agent_type: str) -> BaseAgent:
        """Create an agent by type"""
        if agent_type not in cls._agents:
            raise ValueError(f"Unknown agent type: {agent_type}")
        return cls._agents[agent_type]()

    @classmethod
    def list_agents(cls) -> List[str]:
        """List available agent types"""
        return list(cls._agents.keys())
