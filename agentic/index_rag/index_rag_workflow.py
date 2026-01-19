# agentic/index_rag_workflow.py
# Index RAG Workflow with LangGraph
# Orchestrates the full Index RAG pipeline with state management

import logging
import time
from typing import Dict, Any, List, Literal, Optional, TypedDict

from langgraph.graph import StateGraph, END

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# ============================================================================
# WORKFLOW STATE
# ============================================================================

class IndexRAGState(TypedDict):
    """State for the Index RAG Workflow (6-node version with PDF + memory + validation)"""
    # Input
    query: str
    resolved_query: str  # After follow-up resolution
    is_follow_up: bool
    requirements: Dict[str, Any]
    session_id: Optional[str]
    top_k: int
    enable_web_search: bool
    
    # Intent Classification (Node 1)
    intent: str
    product_type: Optional[str]
    extracted_vendors: List[str]
    extracted_models: List[str]
    specifications: Dict[str, Any]
    intent_confidence: float
    
    # Parallel Indexing with Metadata Filter + PDF (Node 2)
    database_results: List[Dict]
    web_results: List[Dict]
    pdf_results: List[Dict]  # PDF document results
    merged_results: List[Dict]
    filtered_results: List[Dict]
    filter_applied: bool
    json_count: int  # JSON product count
    pdf_count: int   # PDF document count
    
    # Output (Node 3)
    structured_output: Dict[str, Any]
    
    # Validation (Node 4 & 5) - NEW
    validation_result: Optional[Dict]
    is_valid: bool
    validation_score: float
    validation_issues: List[str]
    retry_count: int
    max_retries: int
    
    # Final (Node 6)
    final_response: Dict[str, Any]
    
    # Metadata
    status: str
    error: Optional[str]
    start_time: float
    processing_time_ms: int


def create_index_rag_state(
    query: str,
    requirements: Dict[str, Any] = None,
    session_id: Optional[str] = None,
    top_k: int = 7,
    enable_web_search: bool = True
) -> IndexRAGState:
    """Create initial state for Index RAG workflow (6 nodes with PDF + memory + validation)"""
    return IndexRAGState(
        # Input
        query=query,
        resolved_query=query,  # Will be updated by memory resolution
        is_follow_up=False,
        requirements=requirements or {},
        session_id=session_id or f"index-rag-{int(time.time())}",
        top_k=top_k,
        enable_web_search=enable_web_search,
        
        # Intent Classification
        intent="",
        product_type=None,
        extracted_vendors=[],
        extracted_models=[],
        specifications={},
        intent_confidence=0.0,
        
        # Parallel Indexing (JSON + PDF)
        database_results=[],
        web_results=[],
        pdf_results=[],
        merged_results=[],
        filtered_results=[],
        filter_applied=False,
        json_count=0,
        pdf_count=0,
        
        # Output
        structured_output={},
        
        # Validation - NEW
        validation_result=None,
        is_valid=False,
        validation_score=0.0,
        validation_issues=[],
        retry_count=0,
        max_retries=2,
        
        # Final
        final_response={},
        
        # Metadata
        status="",
        error=None,
        start_time=time.time(),
        processing_time_ms=0
    )


# ============================================================================
# WORKFLOW NODES
# ============================================================================

def classify_intent_node(state: IndexRAGState) -> IndexRAGState:
    """
    Node 1: Classify user intent using LLM Flash Lite.
    
    Also handles follow-up resolution using conversation memory.
    Extracts product type, vendors, models, and specifications.
    """
    logger.info("[IndexRAG Node 1] Classifying intent...")
    
    try:
        # Step 1: Resolve follow-up queries using memory
        try:
            from .index_rag_memory import resolve_follow_up_query, index_rag_memory
            
            resolved_query = resolve_follow_up_query(
                session_id=state['session_id'],
                query=state['query']
            )
            
            if resolved_query != state['query']:
                state['resolved_query'] = resolved_query
                state['is_follow_up'] = True
                logger.info(f"[IndexRAG Node 1] Resolved follow-up: '{state['query']}' -> '{resolved_query}'")
            else:
                state['resolved_query'] = state['query']
                state['is_follow_up'] = False
                
        except ImportError:
            logger.debug("[IndexRAG Node 1] Memory module not available, skipping follow-up resolution")
            state['resolved_query'] = state['query']
            state['is_follow_up'] = False
        
        # Step 2: Classify intent using the resolved query
        from .index_rag_agent import create_index_rag_agent
        
        agent = create_index_rag_agent()
        result = agent.classify_intent(state['resolved_query'])
        
        state['intent'] = result.get('intent', 'search')
        state['product_type'] = result.get('product_type')
        state['extracted_vendors'] = result.get('vendors', [])
        state['extracted_models'] = result.get('models', [])
        state['specifications'] = result.get('specifications', {})
        state['intent_confidence'] = result.get('confidence', 0.0)
        
        # Step 3: Store in conversation memory for future follow-ups
        try:
            from .index_rag_memory import add_to_conversation_memory
            
            add_to_conversation_memory(
                session_id=state['session_id'],
                query=state['resolved_query'],
                intent=state['intent'],
                product_type=state['product_type'],
                vendors=state['extracted_vendors'],
                models=state['extracted_models']
            )
        except ImportError:
            pass
        
        logger.info(f"Intent: {state['intent']}, Product: {state['product_type']}, "
                   f"Confidence: {state['intent_confidence']:.2f}, Follow-up: {state['is_follow_up']}")
        
    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        state['intent'] = 'search'
        state['resolved_query'] = state['query']
        state['error'] = str(e)
    
    return state


def parallel_index_node(state: IndexRAGState) -> IndexRAGState:
    """
    Node 2: Run parallel database and web search indexing.
    
    This node includes:
    - Metadata filter inside the database thread
    - PDF document indexing from Azure Blob
    - Concurrent web search with LLM
    """
    logger.info("[IndexRAG Node 2] Running parallel indexing (DB + PDF + Web)...")
    
    try:
        from tools.parallel_indexer import run_parallel_indexing
        
        # Use resolved_query for indexing (handles follow-ups)
        query_to_use = state.get('resolved_query', state['query'])
        
        # Run parallel indexing - metadata filter + PDF is now inside database thread
        result = run_parallel_indexing(
            query=query_to_use,
            product_type=state['product_type'],
            vendors=state['extracted_vendors'] if state['extracted_vendors'] else None,
            top_k=state['top_k'],
            enable_web_search=state['enable_web_search'],
            timeout_seconds=30
        )
        
        # Extract results
        db_result = result.get('database', {})
        state['database_results'] = db_result.get('results', [])
        state['web_results'] = result.get('web_search', {}).get('results', []) if state['enable_web_search'] else []
        state['merged_results'] = result.get('merged_results', [])
        
        # Extract PDF-specific results (filter from database results)
        state['pdf_results'] = [r for r in state['database_results'] if r.get('source') == 'pdf']
        
        # Extract counts
        state['json_count'] = db_result.get('json_count', len(state['database_results']) - len(state['pdf_results']))
        state['pdf_count'] = db_result.get('pdf_count', len(state['pdf_results']))
        
        # Extract metadata filter info
        metadata_info = db_result.get('metadata_filter', {})
        state['filter_applied'] = metadata_info.get('applied', False)
        
        # Update product type if detected by metadata filter
        detected_types = metadata_info.get('product_types', [])
        if not state['product_type'] and detected_types:
            state['product_type'] = detected_types[0]
        
        # Use merged_results directly as filtered_results
        state['filtered_results'] = state['merged_results']
        
        logger.info(f"Parallel indexing: JSON={state['json_count']}, PDF={state['pdf_count']}, "
                   f"Web={len(state['web_results'])}, Merged={len(state['merged_results'])}")
        
    except Exception as e:
        logger.error(f"Parallel indexing failed: {e}")
        state['merged_results'] = []
        state['filtered_results'] = []
        state['pdf_results'] = []
        state['json_count'] = 0
        state['pdf_count'] = 0
    
    return state


def structure_output_node(state: IndexRAGState) -> IndexRAGState:
    """
    Node 3: Structure output using LLM Flash Lite.
    
    Formats results for frontend consumption.
    """
    logger.info("[IndexRAG Node 3] Structuring output...")
    
    try:
        from .index_rag_agent import create_index_rag_agent
        
        agent = create_index_rag_agent()
        result = agent.structure_output(
            query=state['query'],
            search_results=state['filtered_results']
        )
        
        state['structured_output'] = result
        
        logger.info(f"Structured output: {len(result.get('recommended_products', []))} recommendations")
        
    except Exception as e:
        logger.error(f"Output structuring failed: {e}")
        state['structured_output'] = {
            "summary": f"Found {len(state['filtered_results'])} products",
            "recommended_products": state['filtered_results'][:10],
            "total_found": len(state['filtered_results'])
        }
    
    return state


def validate_response_node(state: IndexRAGState) -> IndexRAGState:
    """
    Node 4: Validate structured output quality.
    
    Checks for:
    - Response relevance to query
    - Data completeness
    - Hallucination indicators
    
    NOTE: Only logs warnings, does not block responses.
    """
    logger.info("[IndexRAG Node 4] Validating response quality...")
    
    try:
        output = state.get('structured_output', {})
        query = state.get('query', '')
        products = output.get('recommended_products', [])
        
        issues = []
        score = 1.0
        
        # Check 1: Results exist
        if not products:
            issues.append("No products found")
            score -= 0.3
        
        # Check 2: Product type relevance
        product_type = state.get('product_type', '')
        if product_type and products:
            relevant_count = sum(
                1 for p in products 
                if product_type.lower() in str(p).lower()
            )
            relevance_ratio = relevant_count / len(products) if products else 0
            if relevance_ratio < 0.5:
                issues.append(f"Low relevance: only {relevant_count}/{len(products)} match product type")
                score -= 0.2
        
        # Check 3: Summary quality
        summary = output.get('summary', '')
        if len(summary) < 20:
            issues.append("Summary too brief")
            score -= 0.1
        
        # Check 4: Query terms in response
        query_terms = query.lower().split()
        summary_lower = summary.lower()
        matching_terms = sum(1 for term in query_terms if term in summary_lower)
        if len(query_terms) > 2 and matching_terms / len(query_terms) < 0.3:
            issues.append("Response may not address query")
            score -= 0.1
        
        # Normalize score
        score = max(0.0, min(1.0, score))
        is_valid = score >= 0.6
        
        state['validation_result'] = {
            "score": score,
            "is_valid": is_valid,
            "issues": issues,
            "products_checked": len(products)
        }
        state['is_valid'] = is_valid
        state['validation_score'] = score
        state['validation_issues'] = issues
        
        # Log warnings only (per user requirement)
        if issues:
            logger.warning(f"[IndexRAG] Validation issues (score={score:.2f}): {issues}")
        else:
            logger.info(f"[IndexRAG] Validation passed with score={score:.2f}")
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        state['is_valid'] = True  # Don't block on validation errors
        state['validation_score'] = 0.5
        state['validation_issues'] = [f"Validation error: {str(e)}"]
    
    return state


def retry_decision_node(state: IndexRAGState) -> IndexRAGState:
    """
    Node 5: Decide whether to retry or finalize.
    
    Routes to finalize - retries logged as warnings only.
    """
    logger.info("[IndexRAG Node 5] Retry decision...")
    
    is_valid = state.get('is_valid', True)
    retry_count = state.get('retry_count', 0)
    max_retries = state.get('max_retries', 2)
    
    if not is_valid and retry_count < max_retries:
        # Log retry suggestion but don't actually retry (warning-only mode)
        logger.warning(
            f"[IndexRAG] Response could benefit from retry "
            f"(attempt {retry_count + 1}/{max_retries}), proceeding anyway"
        )
        state['retry_count'] = retry_count + 1
    
    return state


def finalize_node(state: IndexRAGState) -> IndexRAGState:
    """
    Node 6: Finalize response with all metadata.
    """
    logger.info("[IndexRAG Node 6] Finalizing response...")
    
    state['processing_time_ms'] = int((time.time() - state['start_time']) * 1000)
    
    state['final_response'] = {
        # Output
        "success": True if state['filtered_results'] else False,
        "output": state['structured_output'],
        
        # Query info
        "query": state['query'],
        "resolved_query": state.get('resolved_query', state['query']),
        "is_follow_up": state.get('is_follow_up', False),
        "intent": state['intent'],
        "product_type": state['product_type'],
        
        # Statistics
        "stats": {
            "database_results": len(state['database_results']),
            "json_count": state.get('json_count', 0),
            "pdf_count": state.get('pdf_count', 0),
            "web_results": len(state['web_results']),
            "merged_results": len(state['merged_results']),
            "filtered_results": len(state['filtered_results'])
        },
        
        # Filters applied
        "filters": {
            "metadata_filter": state['filter_applied']
        },
        
        # Metadata
        "metadata": {
            "session_id": state['session_id'],
            "processing_time_ms": state['processing_time_ms'],
            "top_k": state['top_k'],
            "web_search_enabled": state['enable_web_search']
        },
        
        # Validation results (NEW)
        "validation": {
            "is_valid": state.get('is_valid', True),
            "score": state.get('validation_score', 1.0),
            "issues": state.get('validation_issues', []),
            "retry_count": state.get('retry_count', 0)
        }
    }
    
    if state['error']:
        state['final_response']['error'] = state['error']
    
    state['status'] = 'success'
    
    logger.info(f"Index RAG complete in {state['processing_time_ms']}ms "
               f"(JSON: {state.get('json_count', 0)}, PDF: {state.get('pdf_count', 0)})")
    
    return state


# ============================================================================
# WORKFLOW GRAPH
# ============================================================================

def create_index_rag_workflow():
    """
    Create the Index RAG workflow graph.
    
    6-node graph structure with validation:
    START → classify_intent → parallel_index → structure_output 
          → validate_response → retry_decision → finalize → END
    
    Note: Validation logs warnings only, does not block responses.
    """
    logger.info("Creating Index RAG workflow (6 nodes with validation)...")
    
    workflow = StateGraph(IndexRAGState)
    
    # Add 6 nodes
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("parallel_index", parallel_index_node)
    workflow.add_node("structure_output", structure_output_node)
    workflow.add_node("validate_response", validate_response_node)  # NEW
    workflow.add_node("retry_decision", retry_decision_node)  # NEW
    workflow.add_node("finalize", finalize_node)
    
    # Set entry point
    workflow.set_entry_point("classify_intent")
    
    # Add edges (linear flow - 6 nodes)
    workflow.add_edge("classify_intent", "parallel_index")
    workflow.add_edge("parallel_index", "structure_output")
    workflow.add_edge("structure_output", "validate_response")
    workflow.add_edge("validate_response", "retry_decision")
    workflow.add_edge("retry_decision", "finalize")
    workflow.add_edge("finalize", END)
    
    # Compile
    from ..checkpointing import compile_with_checkpointing
    app = compile_with_checkpointing(workflow)
    
    logger.info("Index RAG workflow created successfully")
    
    return app


# ============================================================================
# WORKFLOW EXECUTION
# ============================================================================

_index_rag_app = None


def get_index_rag_workflow():
    """Get or create the Index RAG workflow instance (singleton)."""
    global _index_rag_app
    if _index_rag_app is None:
        _index_rag_app = create_index_rag_workflow()
    return _index_rag_app


def run_index_rag_workflow(
    query: str = None,
    question: str = None,  # Alias for 'query' for backward compatibility
    requirements: Dict[str, Any] = None,
    session_id: Optional[str] = None,
    top_k: int = 7,
    enable_web_search: bool = True
) -> Dict[str, Any]:
    """
    Run the Index RAG workflow.
    
    This is the main entry point for Index RAG.
    
    Args:
        query: User's search query
        question: Alias for query (backward compatibility)
        requirements: Optional structured requirements
        session_id: Optional session ID
        top_k: Max results per stage (default 7)
        enable_web_search: Enable parallel web search
        
    Returns:
        Final response with structured results
    """
    # Support both 'query' and 'question' parameter names
    query = query or question
    
    if not query:
        return {"success": False, "error": "No query provided"}
    
    logger.info(f"Running Index RAG workflow for: {query[:100]}...")
    
    # Create initial state
    initial_state = create_index_rag_state(
        query=query,
        requirements=requirements,
        session_id=session_id,
        top_k=top_k,
        enable_web_search=enable_web_search
    )
    
    # Get workflow
    app = get_index_rag_workflow()
    
    # Execute
    try:
        config = {"configurable": {"thread_id": session_id or "default"}}
        final_state = app.invoke(initial_state, config=config)
        
        logger.info(f"Workflow completed: {final_state.get('status', 'unknown')}")
        
        return final_state.get('final_response', {})
        
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "query": query
        }


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'IndexRAGState',
    'create_index_rag_state',
    'create_index_rag_workflow',
    'get_index_rag_workflow',
    'run_index_rag_workflow'
]


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("TESTING INDEX RAG WORKFLOW")
    print("=" * 80)
    
    test_query = "I need a pressure transmitter from Yokogawa for hazardous area SIL 2"
    
    result = run_index_rag_workflow(
        query=test_query,
        top_k=5,
        enable_web_search=False  # Disable for faster testing
    )
    
    print(f"\nQuery: {test_query}")
    print(f"Success: {result.get('success')}")
    print(f"Intent: {result.get('intent')}")
    print(f"Product Type: {result.get('product_type')}")
    print(f"Stats: {result.get('stats')}")
    print(f"Processing Time: {result.get('metadata', {}).get('processing_time_ms')}ms")
    
    if result.get('output'):
        print(f"\nSummary: {result['output'].get('summary')}")
        print(f"Total Found: {result['output'].get('total_found')}")
    
    print("=" * 80)
