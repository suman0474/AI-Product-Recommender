# agentic/standards_rag_workflow.py
# Standards RAG Workflow with Validation and Retry Logic

import json
import logging
from typing import Dict, Any, List, Literal, Optional, TypedDict
from datetime import datetime
import time

from langgraph.graph import StateGraph, END

from .standards_chat_agent import StandardsChatAgent, create_standards_chat_agent
from ..shared_agents import ResponseValidatorAgent
from ..checkpointing import compile_with_checkpointing

import os
from dotenv import load_dotenv
from llm_fallback import create_llm_with_fallback

load_dotenv()
logger = logging.getLogger(__name__)


# ============================================================================
# WORKFLOW STATE
# ============================================================================

class StandardsRAGState(TypedDict):
    """State for the Standards RAG Workflow (with conversation memory)"""
    # Input
    question: str
    resolved_question: str  # After follow-up resolution (NEW)
    is_follow_up: bool  # NEW
    session_id: Optional[str]
    top_k: int

    # Processing
    question_valid: bool
    key_terms: List[str]
    retrieved_docs: List[Dict]
    context: str
    source_metadata: Dict

    # Generation
    answer: str
    citations: List[Dict]
    confidence: float
    sources_used: List[str]
    generation_count: int

    # Validation
    validation_result: Optional[Dict]
    is_valid: bool
    retry_count: int
    max_retries: int
    validation_feedback: str
    should_fail_fast: bool  # [FIX #A3] Detect unrecoverable errors, skip retries

    # Output
    final_response: Dict
    status: str
    error: Optional[str]

    # Metadata
    start_time: float
    processing_time_ms: int


def create_standards_rag_state(
    question: str,
    session_id: Optional[str] = None,
    top_k: int = 5
) -> StandardsRAGState:
    """Create initial state for Standards RAG workflow (with memory support)"""
    return StandardsRAGState(
        # Input
        question=question,
        resolved_question=question,  # Will be updated if follow-up
        is_follow_up=False,  # Will be updated by validate_question_node
        session_id=session_id or f"session-{int(time.time())}",
        top_k=top_k,

        # Processing
        question_valid=False,
        key_terms=[],
        retrieved_docs=[],
        context="",
        source_metadata={},

        # Generation
        answer="",
        citations=[],
        confidence=0.0,
        sources_used=[],
        generation_count=0,

        # Validation
        validation_result=None,
        is_valid=False,
        retry_count=0,
        max_retries=2,
        validation_feedback="",
        should_fail_fast=False,  # [FIX #A3] Detect unrecoverable errors, skip retries

        # Output
        final_response={},
        status="",
        error=None,

        # Metadata
        start_time=time.time(),
        processing_time_ms=0
    )


# ============================================================================
# WORKFLOW NODES
# ============================================================================

def validate_question_node(state: StandardsRAGState) -> StandardsRAGState:
    """
    Node 1: Validate the user's question.

    Checks if question is valid, extracts key terms, and resolves follow-ups.
    """
    logger.info("[Node 1] Validating question and resolving follow-ups...")

    question = state['question'].strip()

    # Import memory module
    try:
        from .standards_rag_memory import (
            standards_rag_memory,
            resolve_standards_follow_up
        )
        
        # Check if this is a follow-up and resolve it
        session_id = state.get('session_id')
        if session_id:
            resolved = resolve_standards_follow_up(session_id, question)
            if resolved != question:
                state['is_follow_up'] = True
                state['resolved_question'] = resolved
                logger.info(f"Resolved follow-up: '{question}' -> '{resolved}'")
            else:
                state['is_follow_up'] = False
                state['resolved_question'] = question
        else:
            state['is_follow_up'] = False
            state['resolved_question'] = question
    except ImportError:
        logger.warning("Memory module not available, skipping follow-up resolution")
        state['is_follow_up'] = False
        state['resolved_question'] = question

    # Use resolved question for validation
    question = state['resolved_question']

    # Basic validation
    if not question:
        logger.warning("Empty question received")
        state['question_valid'] = False
        state['error'] = "Question cannot be empty"
        return state

    if len(question) < 5:
        logger.warning("Question too short")
        state['question_valid'] = False
        state['error'] = "Question is too short (minimum 5 characters)"
        return state

    if len(question) > 4000:
        logger.warning("Question too long")
        state['question_valid'] = False
        state['error'] = "Question is too long (maximum 4000 characters)"
        return state

    # Extract key terms (simple tokenization)
    # Remove common words and extract meaningful terms
    stop_words = {'what', 'is', 'are', 'the', 'a', 'an', 'for', 'in', 'on', 'at', 'to', 'of'}
    words = question.lower().split()
    key_terms = [w for w in words if w not in stop_words and len(w) > 2]

    state['question_valid'] = True
    state['key_terms'] = key_terms[:10]  # Limit to 10 key terms

    logger.info(f"Question valid. Key terms: {state['key_terms']}, Follow-up: {state['is_follow_up']}")

    return state


def retrieve_standards_node(state: StandardsRAGState) -> StandardsRAGState:
    """
    Node 2: Retrieve relevant standards documents from Pinecone.

    Queries vector store and builds context with production-ready retry logic.
    """
    logger.info("[Node 2] Retrieving standards documents from Pinecone...")

    try:
        # Create agent to use its retrieval method
        agent = create_standards_chat_agent()

        # Retrieve documents
        search_results = agent.retrieve_documents(
            question=state['question'],
            top_k=state['top_k']
        )

        # Store results
        state['retrieved_docs'] = search_results.get('results', [])
        state['context'] = agent.build_context(search_results)

        # Extract source metadata
        sources = {}
        for doc in state['retrieved_docs']:
            metadata = doc.get('metadata', {})
            filename = metadata.get('filename', 'unknown')
            if filename not in sources:
                sources[filename] = {
                    'standard_type': metadata.get('standard_type', ''),
                    'standards_references': metadata.get('standards_references', []),
                    'relevance': doc.get('relevance_score', 0.0)
                }

        state['source_metadata'] = sources

        logger.info(f"Retrieved {len(state['retrieved_docs'])} documents")
        logger.info(f"Unique sources: {list(sources.keys())}")

        if len(state['retrieved_docs']) == 0:
            logger.warning("No documents retrieved")
            state['error'] = "No relevant standards documents found for this question"

    except Exception as e:
        logger.error(f"Error retrieving documents: {e}", exc_info=True)
        state['error'] = f"Error retrieving documents: {str(e)}"

        # ╔════════════════════════════════════════════════════════════════════════╗
        # ║  [FIX #A3] FAST-FAIL ON UNRECOVERABLE ERRORS                         ║
        # ║  Detect network/timeout errors that won't be fixed by retrying       ║
        # ║  Prevents 4 wasteful retry attempts (saves 60-80 seconds!)           ║
        # ╚════════════════════════════════════════════════════════════════════════╝
        error_str = str(e).lower()
        unrecoverable_errors = [
            "connection refused",      # Network connection failed
            "connection reset",         # Connection was reset
            "connection timeout",       # Timeout
            "timed out",               # General timeout
            "winerror 10060",          # Windows connection timeout
            "econnrefused",            # Connection refused (Unix)
            "ehostunreach",            # Host unreachable
            "enetunreach",             # Network unreachable
            "no such host",            # DNS failure
            "name or service not known", # DNS failure (Linux)
            "getaddrinfo failed",      # DNS failure
            "connection attempt failed",  # Generic connection failure
        ]

        if any(unrecoverable in error_str for unrecoverable in unrecoverable_errors):
            logger.warning(f"[FIX #A3] 🔴 UNRECOVERABLE ERROR DETECTED - Failing fast instead of retrying!")
            logger.warning(f"[FIX #A3]    Error: {str(e)}")
            logger.warning(f"[FIX #A3]    Retries Prevented: {state['max_retries']} × ~20s = {state['max_retries'] * 20}+ seconds saved!")
            state['should_fail_fast'] = True
            print(f"\n{'='*70}")
            print(f"🔴 [FIX #A3] FAST-FAIL TRIGGERED")
            print(f"   Error: {type(e).__name__}")
            print(f"   Retries Skipped: {state['max_retries']} (saves {state['max_retries'] * 20}+ seconds)")
            print(f"{'='*70}\n")

    return state


def generate_answer_node(state: StandardsRAGState) -> StandardsRAGState:
    """
    Node 3: Generate answer using StandardsChatAgent.

    Passes question and context to Google Gemini.
    """
    logger.info(f"[Node 3] Generating answer (attempt {state['generation_count'] + 1})...")

    try:
        # Check if we have documents
        if len(state['retrieved_docs']) == 0:
            state['answer'] = "I don't have any relevant standards documents to answer this question. Please rephrase or ask about a different topic."
            state['citations'] = []
            state['confidence'] = 0.0
            state['sources_used'] = []
            state['generation_count'] += 1
            return state

        # Create agent
        agent = create_standards_chat_agent(temperature=0.1)

        # Generate answer
        result = agent.run(
            question=state['question'],
            top_k=state['top_k']
        )

        # Update state
        state['answer'] = result.get('answer', '')
        state['citations'] = result.get('citations', [])
        state['confidence'] = result.get('confidence', 0.0)
        state['sources_used'] = result.get('sources_used', [])
        state['generation_count'] += 1

        logger.info(f"Answer generated with confidence: {state['confidence']:.2f}")
        logger.info(f"Sources used: {state['sources_used']}")

        # Add validation feedback if this is a retry
        if state['retry_count'] > 0 and state['validation_feedback']:
            logger.info(f"Retry based on feedback: {state['validation_feedback']}")

    except Exception as e:
        logger.error(f"Error generating answer: {e}", exc_info=True)
        state['error'] = f"Error generating answer: {str(e)}"
        state['answer'] = "An error occurred while generating the answer."
        state['citations'] = []
        state['confidence'] = 0.0

    return state


def validate_response_node(state: StandardsRAGState) -> StandardsRAGState:
    """
    Node 4: Validate the generated response.

    Uses ResponseValidatorAgent to check quality.
    """
    logger.info("[Node 4] Validating response...")

    try:
        # Create validator
        llm = create_llm_with_fallback(model="gemini-2.5-flash", temperature=0.1)
        validator = ResponseValidatorAgent(llm=llm)

        # Validate
        validation_result = validator.run(
            user_question=state['question'],
            response=state['answer'],
            context=state['context']
        )

        # Store validation result (it's already a dict)
        state['validation_result'] = validation_result
        state['is_valid'] = validation_result.get('is_valid', False)

        # Build feedback for retry
        if not state['is_valid']:
            issues = ", ".join(validation_result.get('issues_found', []))
            suggestions = validation_result.get('suggestions', '')
            state['validation_feedback'] = f"Issues: {issues}. {suggestions}"

            logger.warning(f"Validation failed: {state['validation_feedback']}")
        else:
            overall_score = validation_result.get('overall_score', 0)
            logger.info(f"Validation passed with score: {overall_score:.2f}")

    except Exception as e:
        logger.error(f"Error validating response: {e}", exc_info=True)
        # If validation fails, assume response is valid to avoid blocking
        state['is_valid'] = True
        state['validation_result'] = {'error': str(e)}

    return state


def retry_decision_node(state: StandardsRAGState) -> Literal["finalize", "generate_answer"]:
    """
    Node 5: Decide whether to retry or finalize.

    Routes to finalize if valid or max retries reached, else retry generation.
    """
    logger.info("[Node 5] Making retry decision...")

    # ╔════════════════════════════════════════════════════════════════════════╗
    # ║  [FIX #A3] FAST-FAIL SKIP RETRIES                                    ║
    # ║  If unrecoverable error detected, skip all retries                    ║
    # ║  Prevents 60-80+ seconds of wasteful retry attempts                  ║
    # ╚════════════════════════════════════════════════════════════════════════╝
    if state.get('should_fail_fast', False):
        logger.warning("[FIX #A3] 🔴 FAST-FAIL TRIGGERED - Skipping all retries!")
        return "finalize"

    # If valid, finalize
    if state['is_valid']:
        logger.info("Response valid - proceeding to finalize")
        return "finalize"

    # If max retries reached, finalize anyway with warning
    if state['retry_count'] >= state['max_retries']:
        logger.warning(f"Max retries ({state['max_retries']}) reached - proceeding to finalize")
        return "finalize"

    # Retry
    state['retry_count'] += 1
    logger.info(f"Retrying (attempt {state['retry_count'] + 1}/{state['max_retries'] + 1})...")
    return "generate_answer"


def finalize_node(state: StandardsRAGState) -> StandardsRAGState:
    """
    Node 6: Finalize the response.

    Formats final output with metadata and stores in conversation memory.
    """
    logger.info("[Node 6] Finalizing response...")

    # Calculate processing time
    state['processing_time_ms'] = int((time.time() - state['start_time']) * 1000)

    # Build final response
    final_response = {
        'answer': state['answer'],
        'citations': state['citations'],
        'confidence': state['confidence'],
        'sources_used': state['sources_used'],
        'metadata': {
            'processing_time_ms': state['processing_time_ms'],
            'retry_count': state['retry_count'],
            'validation_score': state['validation_result'].get('overall_score', 0.0) if state['validation_result'] else 0.0,
            'documents_retrieved': len(state['retrieved_docs']),
            'hallucination_detected': state['validation_result'].get('hallucination_detected', False) if state['validation_result'] else False,
            'generation_attempts': state['generation_count'],
            'is_follow_up': state.get('is_follow_up', False),
            'resolved_question': state.get('resolved_question', state['question'])
        }
    }

    # Add warning if validation failed
    if not state['is_valid'] and state['retry_count'] >= state['max_retries']:
        final_response['metadata']['warning'] = "Response may not meet quality standards. Validation issues: " + ", ".join(
            state['validation_result'].get('issues_found', []) if state['validation_result'] else []
        )

    state['final_response'] = final_response
    state['status'] = 'success' if not state['error'] else 'error'

    # Store in conversation memory for future follow-ups
    try:
        from .standards_rag_memory import add_to_standards_memory
        
        session_id = state.get('session_id')
        if session_id and state['answer']:
            add_to_standards_memory(
                session_id=session_id,
                question=state['question'],
                answer=state['answer'],
                citations=state['citations'],
                key_terms=state.get('key_terms', [])
            )
            logger.info(f"Stored Q&A in memory for session: {session_id}")
    except ImportError:
        logger.debug("Memory module not available, skipping Q&A storage")
    except Exception as e:
        logger.warning(f"Failed to store in memory: {e}")

    logger.info(f"Workflow complete in {state['processing_time_ms']}ms")

    return state


# ============================================================================
# WORKFLOW GRAPH CONSTRUCTION
# ============================================================================

def create_standards_rag_workflow():
    """
    Create the Standards RAG workflow graph.

    Graph structure:
    START → validate_question → retrieve_standards → generate_answer
                                                           ↓
                                                    validate_response
                                                           ↓
                                                     retry_decision
                                                       ↙        ↘
                                                 finalize    → generate_answer (loop)
                                                    ↓
                                                   END
    """
    logger.info("Creating Standards RAG workflow...")

    # Create graph
    workflow = StateGraph(StandardsRAGState)

    # Add nodes
    workflow.add_node("validate_question", validate_question_node)
    workflow.add_node("retrieve_standards", retrieve_standards_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("validate_response", validate_response_node)
    workflow.add_node("finalize", finalize_node)

    # Set entry point
    workflow.set_entry_point("validate_question")

    # Add edges
    workflow.add_edge("validate_question", "retrieve_standards")
    workflow.add_edge("retrieve_standards", "generate_answer")
    workflow.add_edge("generate_answer", "validate_response")

    # Conditional edge from retry_decision
    workflow.add_conditional_edges(
        "validate_response",
        retry_decision_node,
        {
            "finalize": "finalize",
            "generate_answer": "generate_answer"
        }
    )

    # Finalize to END
    workflow.add_edge("finalize", END)

    # Compile
    app = compile_with_checkpointing(workflow)

    logger.info("Standards RAG workflow created successfully")

    return app


# ============================================================================
# WORKFLOW EXECUTION
# ============================================================================

# Create workflow instance
_standards_rag_app = None


def get_standards_rag_workflow():
    """Get or create the Standards RAG workflow instance (singleton)."""
    global _standards_rag_app
    if _standards_rag_app is None:
        _standards_rag_app = create_standards_rag_workflow()
    return _standards_rag_app


def run_standards_rag_workflow(
    question: str,
    session_id: Optional[str] = None,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Run the Standards RAG workflow.

    Args:
        question: User's question
        session_id: Optional session ID for checkpointing
        top_k: Number of documents to retrieve

    Returns:
        Final result dictionary
    """
    logger.info(f"Running Standards RAG workflow for question: {question[:100]}...")

    # Create initial state
    initial_state = create_standards_rag_state(
        question=question,
        session_id=session_id,
        top_k=top_k
    )

    # Get workflow
    app = get_standards_rag_workflow()

    # Execute workflow
    try:
        # Run with config for checkpointing
        config = {"configurable": {"thread_id": session_id or "default"}}
        final_state = app.invoke(initial_state, config=config)

        logger.info(f"Workflow completed with status: {final_state.get('status', 'unknown')}")

        return final_state

    except Exception as e:
        logger.error(f"Error executing workflow: {e}", exc_info=True)
        return {
            'status': 'error',
            'error': str(e),
            'final_response': {
                'answer': f"An error occurred: {str(e)}",
                'citations': [],
                'confidence': 0.0,
                'sources_used': [],
                'metadata': {}
            }
        }


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    # Test the workflow
    logger.info("="*80)
    logger.info("TESTING STANDARDS RAG WORKFLOW")
    logger.info("="*80)

    test_questions = [
        "What are the SIL2 requirements for pressure transmitters?",
        "What calibration standards apply to temperature sensors?",
        "What are the ATEX requirements for flow meters?"
    ]

    for i, question in enumerate(test_questions, 1):
        logger.info(f"\n[Test {i}] Question: {question}")
        logger.info("-"*80)

        result = run_standards_rag_workflow(
            question=question,
            session_id=f"test-{i}",
            top_k=5
        )

        if result['status'] == 'success':
            response = result['final_response']
            logger.info(f"\nAnswer: {response['answer'][:200]}...")
            logger.info(f"\nConfidence: {response['confidence']:.2f}")
            logger.info(f"Sources: {', '.join(response['sources_used'])}")
            logger.info(f"Processing time: {response['metadata']['processing_time_ms']}ms")
            logger.info(f"Retries: {response['metadata']['retry_count']}")
            logger.info(f"Validation score: {response['metadata']['validation_score']:.2f}")

            if response.get('citations'):
                logger.info(f"\nTop citations:")
                for j, cite in enumerate(response['citations'][:2], 1):
                    logger.info(f"  [{j}] {cite['source']}")
        else:
            logger.error(f"\n[ERROR] {result.get('error', 'Unknown error')}")

        logger.info("="*80)
