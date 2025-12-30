# agentic/grounded_chat_workflow.py
# Grounded Knowledge Chat Workflow
# Instrument/Accessory Q&A with RAG-grounded responses

import json
import logging
from typing import Dict, Any, List, Literal, Optional, TypedDict
from datetime import datetime
from langgraph.graph import StateGraph, END

from .chat_agents import ChatAgent, ResponseValidatorAgent, SessionManagerAgent
from .rag_components import RAGAggregator, create_rag_aggregator
from .checkpointing import compile_with_checkpointing
from .potential_product_index import run_potential_product_index_workflow

from tools.intent_tools import classify_intent_tool
from tools.schema_tools import load_schema_tool

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from llm_fallback import create_llm_with_fallback

load_dotenv()
logger = logging.getLogger(__name__)


# ============================================================================
# WORKFLOW STATE
# ============================================================================

class GroundedChatState(TypedDict):
    """State for the Grounded Knowledge Chat Workflow"""
    # Input
    user_question: str
    session_id: str
    user_id: Optional[str]
    
    # Question classification
    question_type: str
    product_type: Optional[str]
    entities: List[str]
    is_valid_question: bool
    
    # PPI check
    ppi_needed: bool
    ppi_result: Optional[Dict[str, Any]]
    schema_available: bool
    
    # RAG retrieval
    rag_context: Dict[str, Any]
    constraint_context: Optional[Dict[str, Any]]
    preferred_vendors: List[str]
    required_standards: List[str]
    installed_series: List[str]
    
    # Answer generation
    generated_answer: str
    citations: List[Dict[str, str]]
    rag_sources_used: List[str]
    confidence: float
    
    # Validation
    is_valid: bool
    validation_score: float
    validation_issues: List[str]
    hallucination_detected: bool
    
    # Retry loop
    retry_count: int
    max_retries: int
    validation_feedback: str
    
    # Session
    total_interactions: int
    conversation_context: str
    
    # Output
    final_response: Dict[str, Any]
    
    # Control
    current_node: str
    messages: List[Dict[str, str]]
    error: Optional[str]
    completed: bool


def create_grounded_chat_state(
    user_question: str,
    session_id: str = "default",
    user_id: str = None
) -> GroundedChatState:
    """Create initial state for Grounded Chat workflow"""
    return GroundedChatState(
        # Input
        user_question=user_question,
        session_id=session_id,
        user_id=user_id,
        
        # Question classification
        question_type="",
        product_type=None,
        entities=[],
        is_valid_question=True,
        
        # PPI check
        ppi_needed=False,
        ppi_result=None,
        schema_available=False,
        
        # RAG retrieval
        rag_context={},
        constraint_context=None,
        preferred_vendors=[],
        required_standards=[],
        installed_series=[],
        
        # Answer generation
        generated_answer="",
        citations=[],
        rag_sources_used=[],
        confidence=0.0,
        
        # Validation
        is_valid=False,
        validation_score=0.0,
        validation_issues=[],
        hallucination_detected=False,
        
        # Retry loop
        retry_count=0,
        max_retries=2,
        validation_feedback="",
        
        # Session
        total_interactions=0,
        conversation_context="",
        
        # Output
        final_response={},
        
        # Control
        current_node="start",
        messages=[],
        error=None,
        completed=False
    )


# ============================================================================
# PROMPTS
# ============================================================================

QUESTION_CLASSIFICATION_PROMPT = """
You are an expert in industrial instrumentation. Classify this user question.

Question: {question}

Conversation Context: {conversation_context}

Determine:
1. question_type: One of ["product_inquiry", "specification_query", "comparison_query", 
                          "troubleshooting", "general_knowledge", "follow_up", "out_of_scope"]
2. product_type: Detected product type (e.g., "pressure transmitter", "flow meter") or null
3. entities: List of mentioned vendors, models, standards (e.g., ["Honeywell", "SIL2", "ST800"])
4. is_valid: true if this is a valid industrial instrumentation question
5. needs_context: true if this is a follow-up that needs previous conversation

Return ONLY valid JSON:
{{
    "question_type": "...",
    "product_type": "...",
    "entities": [...],
    "is_valid": true/false,
    "needs_context": true/false,
    "reasoning": "Brief explanation"
}}
"""


# ============================================================================
# WORKFLOW NODES
# ============================================================================

# Agent instances (shared across nodes)
_chat_agent: Optional[ChatAgent] = None
_validator_agent: Optional[ResponseValidatorAgent] = None
_session_manager: Optional[SessionManagerAgent] = None
_rag_aggregator: Optional[RAGAggregator] = None


def _get_chat_agent() -> ChatAgent:
    global _chat_agent
    if _chat_agent is None:
        _chat_agent = ChatAgent()
    return _chat_agent


def _get_validator_agent() -> ResponseValidatorAgent:
    global _validator_agent
    if _validator_agent is None:
        _validator_agent = ResponseValidatorAgent()
    return _validator_agent


def _get_session_manager() -> SessionManagerAgent:
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManagerAgent()
    return _session_manager


def _get_rag_aggregator() -> RAGAggregator:
    global _rag_aggregator
    if _rag_aggregator is None:
        _rag_aggregator = create_rag_aggregator()
    return _rag_aggregator


def classify_question_node(state: GroundedChatState) -> GroundedChatState:
    """
    Node 1: Classify the user's question.
    Determines question type, product type, and validity.
    """
    logger.info("[CHAT] Node 1: Classifying question...")
    state["current_node"] = "classify_question"
    
    try:
        # Get conversation context for follow-up detection
        session_manager = _get_session_manager()
        conversation_context = session_manager.get_conversation_context(
            state["session_id"], 
            max_turns=3
        )
        state["conversation_context"] = conversation_context
        
        # Classify using LLM
        llm = create_llm_with_fallback(
            model="gemini-2.0-flash-exp",
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        prompt = ChatPromptTemplate.from_template(QUESTION_CLASSIFICATION_PROMPT)
        parser = JsonOutputParser()
        chain = prompt | llm | parser
        
        result = chain.invoke({
            "question": state["user_question"],
            "conversation_context": conversation_context or "No previous conversation"
        })
        
        state["question_type"] = result.get("question_type", "general_knowledge")
        state["product_type"] = result.get("product_type")
        state["entities"] = result.get("entities", [])
        state["is_valid_question"] = result.get("is_valid", True)
        
        state["messages"].append({
            "role": "system",
            "content": f"Question classified: type={state['question_type']}, product={state['product_type']}"
        })
        
        logger.info(f"[CHAT] Question type: {state['question_type']}, Product: {state['product_type']}")
        
    except Exception as e:
        logger.error(f"[CHAT] Question classification failed: {e}")
        state["error"] = str(e)
        state["is_valid_question"] = True  # Default to valid
    
    return state


def check_ppi_node(state: GroundedChatState) -> GroundedChatState:
    """
    Node 2: Check if PPI workflow is needed.
    Triggered when schema doesn't exist for the product type.
    """
    logger.info("[CHAT] Node 2: Checking PPI requirement...")
    state["current_node"] = "check_ppi"
    
    try:
        product_type = state.get("product_type")
        
        if not product_type:
            state["ppi_needed"] = False
            state["schema_available"] = True
            return state
        
        # Check if schema exists
        schema_result = load_schema_tool.invoke({
            "product_type": product_type
        })
        
        if schema_result.get("success") and schema_result.get("schema"):
            state["schema_available"] = True
            state["ppi_needed"] = False
            logger.info(f"[CHAT] Schema found for {product_type}")
        else:
            state["schema_available"] = False
            state["ppi_needed"] = True
            logger.info(f"[CHAT] No schema for {product_type}, PPI needed")
        
        state["messages"].append({
            "role": "system",
            "content": f"Schema check: available={state['schema_available']}, PPI needed={state['ppi_needed']}"
        })
        
    except Exception as e:
        logger.error(f"[CHAT] PPI check failed: {e}")
        state["schema_available"] = True  # Assume available on error
        state["ppi_needed"] = False
    
    return state


def run_ppi_node(state: GroundedChatState) -> GroundedChatState:
    """
    Node 2b: Run PPI workflow if needed.
    Discovers vendors and creates schema for new product types.
    """
    logger.info("[CHAT] Node 2b: Running PPI workflow...")
    state["current_node"] = "run_ppi"
    
    try:
        if not state.get("ppi_needed"):
            return state
        
        product_type = state.get("product_type", "industrial instrument")
        
        # Run PPI workflow via internal API
        from .internal_api import api_client
        
        ppi_result = api_client.call_potential_product_index(
            product_type=product_type,
            session_id=state["session_id"]
        )
        
        state["ppi_result"] = ppi_result
        
        if ppi_result.get("success"):
            state["schema_available"] = ppi_result.get("schema_saved", False)
            logger.info(f"[CHAT] PPI completed: schema_saved={state['schema_available']}")
        
        state["messages"].append({
            "role": "system",
            "content": f"PPI workflow: success={ppi_result.get('success')}"
        })
        
    except Exception as e:
        logger.error(f"[CHAT] PPI workflow failed: {e}")
        state["error"] = str(e)
    
    return state


def retrieve_data_node(state: GroundedChatState) -> GroundedChatState:
    """
    Node 3: Retrieve data from RAG systems.
    Queries Strategy, Standards, and Inventory RAGs.
    """
    logger.info("[CHAT] Node 3: Retrieving RAG context...")
    state["current_node"] = "retrieve_data"
    
    try:
        rag_aggregator = _get_rag_aggregator()
        
        product_type = state.get("product_type") or "industrial instrument"
        requirements = {
            "question": state["user_question"],
            "entities": state.get("entities", [])
        }
        
        # Query all RAG systems
        rag_results = rag_aggregator.query_all_parallel(
            product_type=product_type,
            requirements=requirements
        )
        
        state["rag_context"] = rag_results
        
        # Extract key information
        strategy_data = rag_results.get("strategy", {}).get("data", {})
        standards_data = rag_results.get("standards", {}).get("data", {})
        inventory_data = rag_results.get("inventory", {}).get("data", {})
        
        state["preferred_vendors"] = strategy_data.get("preferred_vendors", [])
        state["required_standards"] = standards_data.get("required_standards", [])
        state["installed_series"] = inventory_data.get("installed_series", [])
        
        # Build constraint context
        state["constraint_context"] = {
            "preferred_vendors": state["preferred_vendors"],
            "required_standards": state["required_standards"],
            "installed_series": state["installed_series"]
        }
        
        state["messages"].append({
            "role": "system",
            "content": f"RAG retrieved: {len(state['preferred_vendors'])} vendors, {len(state['required_standards'])} standards"
        })
        
        logger.info(f"[CHAT] RAG context retrieved with {len(state['preferred_vendors'])} preferred vendors")
        
    except Exception as e:
        logger.error(f"[CHAT] RAG retrieval failed: {e}")
        state["rag_context"] = {}
        state["error"] = str(e)
    
    return state


def generate_answer_node(state: GroundedChatState) -> GroundedChatState:
    """
    Node 4: Generate grounded answer using ChatAgent.
    Uses RAG context to create response with citations.
    """
    logger.info(f"[CHAT] Node 4: Generating answer (attempt {state['retry_count'] + 1})...")
    state["current_node"] = "generate_answer"
    
    try:
        chat_agent = _get_chat_agent()
        
        # Include validation feedback if retrying
        question = state["user_question"]
        if state["retry_count"] > 0 and state.get("validation_feedback"):
            question = f"{question}\n\n[REVISION NEEDED: {state['validation_feedback']}]"
        
        # Generate answer
        result = chat_agent.run(
            user_question=question,
            product_type=state.get("product_type", ""),
            rag_context=state.get("rag_context", {}),
            specifications={
                "preferred_vendors": state.get("preferred_vendors", []),
                "required_standards": state.get("required_standards", []),
                "installed_series": state.get("installed_series", [])
            }
        )
        
        if result.get("success"):
            state["generated_answer"] = result.get("answer", "")
            state["citations"] = result.get("citations", [])
            state["rag_sources_used"] = result.get("rag_sources_used", [])
            state["confidence"] = result.get("confidence", 0.0)
        else:
            state["generated_answer"] = result.get("answer", "I couldn't generate an answer.")
            state["confidence"] = 0.0
        
        state["messages"].append({
            "role": "system",
            "content": f"Answer generated: confidence={state['confidence']:.2f}"
        })
        
        logger.info(f"[CHAT] Answer generated with confidence: {state['confidence']:.2f}")
        
    except Exception as e:
        logger.error(f"[CHAT] Answer generation failed: {e}")
        state["generated_answer"] = "I apologize, but I encountered an error."
        state["error"] = str(e)
    
    return state


def validate_response_node(state: GroundedChatState) -> GroundedChatState:
    """
    Node 5: Validate the generated response.
    Checks for relevance, accuracy, grounding, citations, and hallucination.
    """
    logger.info("[CHAT] Node 5: Validating response...")
    state["current_node"] = "validate_response"
    
    try:
        validator = _get_validator_agent()
        
        # Build context for validation
        context = json.dumps({
            "preferred_vendors": state.get("preferred_vendors", []),
            "required_standards": state.get("required_standards", []),
            "installed_series": state.get("installed_series", []),
            "rag_context": state.get("rag_context", {})
        })
        
        result = validator.run(
            response=state["generated_answer"],
            user_question=state["user_question"],
            context=context
        )
        
        state["is_valid"] = result.get("is_valid", False)
        state["validation_score"] = result.get("overall_score", 0.0)
        state["validation_issues"] = result.get("issues_found", [])
        state["hallucination_detected"] = result.get("hallucination_detected", False)
        state["validation_feedback"] = result.get("suggestions", "")
        
        state["messages"].append({
            "role": "system",
            "content": f"Validation: valid={state['is_valid']}, score={state['validation_score']:.2f}, hallucination={state['hallucination_detected']}"
        })
        
        logger.info(f"[CHAT] Validation: valid={state['is_valid']}, score={state['validation_score']:.2f}")
        
    except Exception as e:
        logger.error(f"[CHAT] Validation failed: {e}")
        state["is_valid"] = True  # Assume valid on error
        state["validation_score"] = 0.5
    
    return state


def retry_decision_node(state: GroundedChatState) -> GroundedChatState:
    """
    Node 6: Decide whether to retry answer generation.
    Implements max 2 retries as per diagram.
    """
    logger.info("[CHAT] Node 6: Retry decision...")
    state["current_node"] = "retry_decision"
    
    if not state["is_valid"] and state["retry_count"] < state["max_retries"]:
        state["retry_count"] += 1
        logger.info(f"[CHAT] Retrying: attempt {state['retry_count'] + 1}/{state['max_retries'] + 1}")
    else:
        if not state["is_valid"]:
            logger.warning(f"[CHAT] Max retries reached, proceeding with best effort")
    
    return state


def update_session_node(state: GroundedChatState) -> GroundedChatState:
    """
    Node 7: Update session with Q&A pair.
    Stores conversation for multi-turn support.
    """
    logger.info("[CHAT] Node 7: Updating session...")
    state["current_node"] = "update_session"
    
    try:
        session_manager = _get_session_manager()
        
        result = session_manager.run(
            session_id=state["session_id"],
            question=state["user_question"],
            answer=state["generated_answer"],
            validation_score=state["validation_score"],
            citations=state["citations"],
            operation="update"
        )
        
        state["total_interactions"] = result.get("total_interactions", 0)
        
        state["messages"].append({
            "role": "system",
            "content": f"Session updated: {state['total_interactions']} interactions"
        })
        
        logger.info(f"[CHAT] Session updated: {state['total_interactions']} interactions")
        
    except Exception as e:
        logger.error(f"[CHAT] Session update failed: {e}")
    
    return state


def finalize_response_node(state: GroundedChatState) -> GroundedChatState:
    """
    Node 8: Finalize and format the response.
    Prepares final output with all metadata.
    """
    logger.info("[CHAT] Node 8: Finalizing response...")
    state["current_node"] = "finalize"
    
    state["final_response"] = {
        "answer": state["generated_answer"],
        "citations": state["citations"],
        "rag_sources_used": state["rag_sources_used"],
        "confidence": state["confidence"],
        "validation_score": state["validation_score"],
        "is_validated": state["is_valid"],
        "product_type": state["product_type"],
        "question_type": state["question_type"],
        "retry_count": state["retry_count"],
        "session_interactions": state["total_interactions"],
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Add warnings if applicable
    if state["hallucination_detected"]:
        state["final_response"]["warnings"] = ["Potential hallucination detected - verify information"]
    if state["validation_issues"]:
        state["final_response"]["validation_issues"] = state["validation_issues"]
    
    state["completed"] = True
    
    logger.info(f"[CHAT] Response finalized: confidence={state['confidence']:.2f}, validated={state['is_valid']}")
    
    return state


def handle_invalid_question_node(state: GroundedChatState) -> GroundedChatState:
    """Handle out-of-scope or invalid questions"""
    logger.info("[CHAT] Handling invalid question...")
    state["current_node"] = "invalid_question"
    
    state["final_response"] = {
        "answer": "I'm specialized in industrial instrumentation and can't help with that question. Please ask about instruments, accessories, vendors, or specifications.",
        "citations": [],
        "rag_sources_used": [],
        "confidence": 1.0,
        "validation_score": 1.0,
        "is_validated": True,
        "question_type": state["question_type"],
        "timestamp": datetime.utcnow().isoformat()
    }
    
    state["completed"] = True
    return state


def handle_escalation_node(state: GroundedChatState) -> GroundedChatState:
    """Handle escalation after max retries"""
    logger.info("[CHAT] Handling escalation...")
    state["current_node"] = "escalation"
    
    state["final_response"] = {
        "answer": "I apologize, but I wasn't able to generate a reliable answer for your question. The information may not be available in our knowledge base. Please contact technical support for assistance.",
        "citations": [],
        "rag_sources_used": state.get("rag_sources_used", []),
        "confidence": 0.0,
        "validation_score": state["validation_score"],
        "is_validated": False,
        "escalated": True,
        "validation_issues": state["validation_issues"],
        "timestamp": datetime.utcnow().isoformat()
    }
    
    state["completed"] = True
    return state


# ============================================================================
# ROUTING FUNCTIONS
# ============================================================================

def route_after_classification(state: GroundedChatState) -> Literal["check_ppi", "invalid_question"]:
    """Route based on question validity"""
    if not state.get("is_valid_question") or state.get("question_type") == "out_of_scope":
        return "invalid_question"
    return "check_ppi"


def route_after_ppi_check(state: GroundedChatState) -> Literal["run_ppi", "retrieve_data"]:
    """Route based on PPI need"""
    if state.get("ppi_needed"):
        return "run_ppi"
    return "retrieve_data"


def route_after_validation(state: GroundedChatState) -> Literal["update_session", "retry", "escalate"]:
    """Route based on validation result"""
    if state.get("is_valid"):
        return "update_session"
    elif state["retry_count"] < state["max_retries"]:
        return "retry"
    else:
        return "escalate"


# ============================================================================
# WORKFLOW CREATION
# ============================================================================

def create_grounded_chat_workflow() -> StateGraph:
    """
    Create the Grounded Knowledge Chat workflow.
    
    8-Node Architecture:
    1. Question Classification
    2. PPI Check
    2b. Run PPI (conditional)
    3. Data Retrieval (RAG)
    4. Answer Generation (ChatAgent)
    5. Response Validation
    6. Retry Decision
    7. Session Update
    8. Finalize Response
    
    With:
    - Retry loop (max 2 retries)
    - Escalation path
    - Invalid question handling
    """
    
    workflow = StateGraph(GroundedChatState)
    
    # Add nodes
    workflow.add_node("classify_question", classify_question_node)
    workflow.add_node("check_ppi", check_ppi_node)
    workflow.add_node("run_ppi", run_ppi_node)
    workflow.add_node("retrieve_data", retrieve_data_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("validate_response", validate_response_node)
    workflow.add_node("retry_decision", retry_decision_node)
    workflow.add_node("update_session", update_session_node)
    workflow.add_node("finalize", finalize_response_node)
    workflow.add_node("invalid_question", handle_invalid_question_node)
    workflow.add_node("escalation", handle_escalation_node)
    
    # Entry point
    workflow.set_entry_point("classify_question")
    
    # Edges from classification
    workflow.add_conditional_edges(
        "classify_question",
        route_after_classification,
        {
            "check_ppi": "check_ppi",
            "invalid_question": "invalid_question"
        }
    )
    
    # Edges from PPI check
    workflow.add_conditional_edges(
        "check_ppi",
        route_after_ppi_check,
        {
            "run_ppi": "run_ppi",
            "retrieve_data": "retrieve_data"
        }
    )
    
    # PPI to retrieve
    workflow.add_edge("run_ppi", "retrieve_data")
    
    # Main flow
    workflow.add_edge("retrieve_data", "generate_answer")
    workflow.add_edge("generate_answer", "validate_response")
    workflow.add_edge("validate_response", "retry_decision")
    
    # Retry decision routing
    workflow.add_conditional_edges(
        "retry_decision",
        route_after_validation,
        {
            "update_session": "update_session",
            "retry": "generate_answer",
            "escalate": "escalation"
        }
    )
    
    # Session to finalize
    workflow.add_edge("update_session", "finalize")
    
    # Terminal nodes
    workflow.add_edge("finalize", END)
    workflow.add_edge("invalid_question", END)
    workflow.add_edge("escalation", END)
    
    return workflow


# ============================================================================
# WORKFLOW EXECUTION
# ============================================================================

def run_grounded_chat_workflow(
    user_question: str,
    session_id: str = "default",
    user_id: str = None,
    checkpointing_backend: str = "memory"
) -> Dict[str, Any]:
    """
    Run the Grounded Knowledge Chat workflow.
    
    Args:
        user_question: User's question about instruments/accessories
        session_id: Session identifier for conversation continuity
        user_id: Optional user identifier
        checkpointing_backend: Backend for state persistence
    
    Returns:
        Workflow result with answer, citations, and metadata
    """
    try:
        # Create initial state
        initial_state = create_grounded_chat_state(
            user_question=user_question,
            session_id=session_id,
            user_id=user_id
        )
        
        # Create and compile workflow
        workflow = create_grounded_chat_workflow()
        compiled = compile_with_checkpointing(workflow, checkpointing_backend)
        
        # Run workflow
        config = {"configurable": {"thread_id": f"{session_id}_chat"}}
        final_state = compiled.invoke(initial_state, config)
        
        return {
            "success": True,
            "response": final_state.get("final_response", {}),
            "answer": final_state.get("generated_answer", ""),
            "citations": final_state.get("citations", []),
            "confidence": final_state.get("confidence", 0.0),
            "is_validated": final_state.get("is_valid", False),
            "session_id": session_id,
            "error": final_state.get("error")
        }
        
    except Exception as e:
        logger.error(f"Grounded chat workflow failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "answer": "I encountered an error processing your question."
        }


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "create_grounded_chat_workflow",
    "run_grounded_chat_workflow",
    "GroundedChatState",
    "create_grounded_chat_state"
]
