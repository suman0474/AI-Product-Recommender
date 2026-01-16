# agentic/strategy_rag_workflow.py
# =============================================================================
# STRATEGY WORKFLOW - CSV-Based Vendor Strategy Filter with LangGraph
# =============================================================================
#
# UPDATED: Converted to LangGraph workflow with 6 nodes for feature parity.
# Still uses CSV-based filtering (no LLM generation per user requirement).
#
# PURPOSE: Filter and prioritize vendors based on procurement strategy statements
# from the instrumentation_procurement_strategy.csv file.
#
# NODES:
# 1. validate_input - Validate product type and resolve follow-ups
# 2. load_strategy - Load and filter CSV strategy data
# 3. structure_output - Format results for frontend
# 4. validate_response - Validate strategy results (warning-only)
# 5. retry_decision - Log retry suggestions
# 6. finalize - Build final response and store in memory
#
# =============================================================================

import logging
import time
from typing import Dict, Any, List, Optional, TypedDict, Literal

from langgraph.graph import StateGraph, END

from .strategy_csv_filter import (
    StrategyCSVFilter,
    filter_vendors_by_strategy,
    get_vendor_strategy_info,
    get_strategy_filter
)

logger = logging.getLogger(__name__)


# ============================================================================
# STATE DEFINITION (6-node LangGraph version)
# ============================================================================

class StrategyRAGState(TypedDict):
    """State for the Strategy LangGraph Workflow"""
    # Input
    question: str
    product_type: str
    resolved_product_type: str  # After follow-up resolution
    is_follow_up: bool
    requirements: Dict[str, Any]
    session_id: Optional[str]
    top_k: int
    refinery: Optional[str]

    # Results (CSV-based)
    preferred_vendors: List[str]
    forbidden_vendors: List[str]
    neutral_vendors: List[str]
    procurement_priorities: Dict[str, int]
    strategy_notes: str
    category: str
    subcategory: str
    all_vendors: List[Dict]
    
    # Confidence and sources
    confidence: float
    sources_used: List[str]

    # Structured Output
    structured_output: Dict[str, Any]
    
    # Validation
    validation_result: Optional[Dict]
    is_valid: bool
    validation_score: float
    validation_issues: List[str]
    retry_count: int
    max_retries: int

    # Output
    final_response: Dict
    status: str
    error: Optional[str]

    # Metadata
    start_time: float
    processing_time_ms: int


def create_strategy_rag_state(
    question: str = "",
    product_type: str = "",
    requirements: Dict[str, Any] = None,
    session_id: Optional[str] = None,
    top_k: int = 7,
    refinery: str = None
) -> StrategyRAGState:
    """Create initial state for Strategy LangGraph workflow"""
    return StrategyRAGState(
        question=question,
        product_type=product_type,
        resolved_product_type=product_type,
        is_follow_up=False,
        requirements=requirements or {},
        session_id=session_id or f"strategy-{int(time.time())}",
        top_k=top_k,
        refinery=refinery,
        preferred_vendors=[],
        forbidden_vendors=[],
        neutral_vendors=[],
        procurement_priorities={},
        strategy_notes="",
        category="",
        subcategory="",
        all_vendors=[],
        confidence=0.0,
        sources_used=[],
        structured_output={},
        validation_result=None,
        is_valid=False,
        validation_score=0.0,
        validation_issues=[],
        retry_count=0,
        max_retries=2,
        final_response={},
        status="",
        error=None,
        start_time=time.time(),
        processing_time_ms=0
    )


# ============================================================================
# WORKFLOW NODES
# ============================================================================

def validate_input_node(state: StrategyRAGState) -> StrategyRAGState:
    """
    Node 1: Validate input and resolve follow-up queries.
    """
    logger.info("[Strategy Node 1] Validating input...")

    product_type = state.get('product_type', '').strip()
    question = state.get('question', '').strip()

    # Use product_type if provided, else try to extract from question
    if not product_type and question:
        product_type = question

    if not product_type:
        state['error'] = "Product type cannot be empty"
        state['status'] = 'error'
        return state

    # Try to resolve follow-ups using memory
    try:
        from .strategy_rag_memory import strategy_rag_memory
        
        session_id = state.get('session_id')
        if session_id:
            resolution = strategy_rag_memory.resolve_follow_up(session_id, product_type)
            state['resolved_product_type'] = resolution.get('product_type', product_type)
            state['is_follow_up'] = resolution.get('is_follow_up', False)
            
            if state['is_follow_up']:
                logger.info(f"Resolved follow-up: '{product_type}' -> '{state['resolved_product_type']}'")
        else:
            state['resolved_product_type'] = product_type
            state['is_follow_up'] = False
    except ImportError:
        state['resolved_product_type'] = product_type
        state['is_follow_up'] = False

    logger.info(f"Input validated. Product type: {state['resolved_product_type']}")
    return state


def load_strategy_node(state: StrategyRAGState) -> StrategyRAGState:
    """
    Node 2: Load and filter CSV strategy data.
    """
    logger.info("[Strategy Node 2] Loading strategy from CSV...")

    if state.get('error'):
        return state

    try:
        strategy_filter = get_strategy_filter()

        # Get vendors from CSV for this category
        result = strategy_filter.filter_vendors_for_product(
            product_type=state['resolved_product_type'],
            available_vendors=None,
            refinery=state.get('refinery')
        )

        filtered_vendors = result.get("filtered_vendors", [])
        state['all_vendors'] = filtered_vendors
        state['category'] = result.get('category', '')
        state['subcategory'] = result.get('subcategory', '')

        # Categorize vendors by priority score
        preferred = []
        neutral = []

        for vendor_info in filtered_vendors:
            vendor_name = vendor_info.get("vendor", "")
            priority = vendor_info.get("priority_score", 0)

            if priority >= 20:
                preferred.append(vendor_name)
            else:
                neutral.append(vendor_name)

        state['preferred_vendors'] = preferred
        state['neutral_vendors'] = neutral
        state['forbidden_vendors'] = []

        # Build procurement priorities
        state['procurement_priorities'] = {
            v.get("vendor", ""): v.get("priority_score", 0)
            for v in filtered_vendors[:state['top_k']]
        }

        # Build strategy notes
        notes_parts = []
        for vendor_info in filtered_vendors[:5]:
            strategy = vendor_info.get("strategy", "")
            vendor = vendor_info.get("vendor", "")
            if strategy and strategy != "No specific strategy defined":
                notes_parts.append(f"{vendor}: {strategy}")

        state['strategy_notes'] = "; ".join(notes_parts) if notes_parts else \
            "No specific strategy defined for this product category."

        state['sources_used'] = ['instrumentation_procurement_strategy.csv']
        state['confidence'] = 0.85 if preferred else 0.5

        logger.info(f"Loaded {len(preferred)} preferred, {len(neutral)} neutral vendors")

    except Exception as e:
        logger.error(f"Failed to load strategy: {e}")
        state['error'] = str(e)
        state['confidence'] = 0.0

    return state


def structure_output_node(state: StrategyRAGState) -> StrategyRAGState:
    """
    Node 3: Structure output for frontend consumption.
    """
    logger.info("[Strategy Node 3] Structuring output...")

    if state.get('error'):
        return state

    top_k = state.get('top_k', 7)

    state['structured_output'] = {
        'preferred_vendors': state['preferred_vendors'][:top_k],
        'forbidden_vendors': state['forbidden_vendors'],
        'neutral_vendors': state['neutral_vendors'][:top_k],
        'procurement_priorities': state['procurement_priorities'],
        'strategy_notes': state['strategy_notes'],
        'answer': f"Based on procurement strategy for {state['category'] or state['product_type']}: {state['strategy_notes']}",
        'citations': [],
        'confidence': state['confidence'],
        'sources_used': state['sources_used'],
        'category': state['category'],
        'subcategory': state['subcategory']
    }

    logger.info(f"Structured output with {len(state['preferred_vendors'])} preferred vendors")
    return state


def validate_response_node(state: StrategyRAGState) -> StrategyRAGState:
    """
    Node 4: Validate strategy results (warning-only, no blocking).
    """
    logger.info("[Strategy Node 4] Validating response quality...")

    if state.get('error'):
        state['is_valid'] = True
        state['validation_score'] = 0.0
        return state

    issues = []
    score = 1.0

    output = state.get('structured_output', {})
    preferred = output.get('preferred_vendors', [])
    neutral = output.get('neutral_vendors', [])
    strategy_notes = output.get('strategy_notes', '')

    # Check 1: Results exist
    if not preferred and not neutral:
        issues.append("No vendors found")
        score -= 0.4

    # Check 2: Strategy notes quality
    if not strategy_notes or strategy_notes == "No specific strategy defined for this product category.":
        issues.append("No specific strategy defined")
        score -= 0.2

    # Check 3: Category matching
    if not state.get('category'):
        issues.append("Product category not identified")
        score -= 0.2

    # Normalize score
    score = max(0.0, min(1.0, score))
    is_valid = score >= 0.5

    state['validation_result'] = {
        "score": score,
        "is_valid": is_valid,
        "issues": issues
    }
    state['is_valid'] = is_valid
    state['validation_score'] = score
    state['validation_issues'] = issues

    # Log warnings only (per user requirement)
    if issues:
        logger.warning(f"[Strategy] Validation issues (score={score:.2f}): {issues}")
    else:
        logger.info(f"[Strategy] Validation passed with score={score:.2f}")

    return state


def retry_decision_node(state: StrategyRAGState) -> StrategyRAGState:
    """
    Node 5: Log retry suggestions (warning-only, no actual retry).
    """
    logger.info("[Strategy Node 5] Retry decision...")

    is_valid = state.get('is_valid', True)
    retry_count = state.get('retry_count', 0)
    max_retries = state.get('max_retries', 2)

    if not is_valid and retry_count < max_retries:
        logger.warning(
            f"[Strategy] Response could benefit from retry "
            f"(attempt {retry_count + 1}/{max_retries}), proceeding anyway"
        )
        state['retry_count'] = retry_count + 1

    return state


def finalize_node(state: StrategyRAGState) -> StrategyRAGState:
    """
    Node 6: Finalize response and store in memory.
    """
    logger.info("[Strategy Node 6] Finalizing response...")

    state['processing_time_ms'] = int((time.time() - state['start_time']) * 1000)

    if state.get('error'):
        state['final_response'] = {
            'answer': f"Error processing strategy: {state['error']}",
            'preferred_vendors': [],
            'forbidden_vendors': [],
            'neutral_vendors': [],
            'procurement_priorities': {},
            'strategy_notes': '',
            'citations': [],
            'confidence': 0.0,
            'sources_used': [],
            'metadata': {
                'rag_type': 'csv_filter_langgraph',
                'processing_time_ms': state['processing_time_ms'],
                'error': state['error']
            }
        }
        state['status'] = 'error'
    else:
        output = state['structured_output']
        state['final_response'] = {
            **output,
            'metadata': {
                'processing_time_ms': state['processing_time_ms'],
                'retry_count': state['retry_count'],
                'validation_score': state['validation_score'],
                'documents_retrieved': len(state['all_vendors']),
                'hallucination_detected': False,
                'generation_attempts': 1,
                'product_type': state['product_type'],
                'resolved_product_type': state['resolved_product_type'],
                'is_follow_up': state['is_follow_up'],
                'category': state['category'],
                'subcategory': state['subcategory'],
                'rag_type': 'csv_filter_langgraph',
                'validation_issues': state['validation_issues']
            }
        }
        state['status'] = 'success'

    # Store in memory for future follow-ups
    try:
        from .strategy_rag_memory import add_to_strategy_memory

        session_id = state.get('session_id')
        if session_id and state['status'] == 'success':
            add_to_strategy_memory(
                session_id=session_id,
                product_type=state['product_type'],
                preferred_vendors=state['preferred_vendors'],
                strategy_notes=state['strategy_notes']
            )
            logger.info(f"Stored in strategy memory for session: {session_id}")
    except ImportError:
        logger.debug("Memory module not available")
    except Exception as e:
        logger.warning(f"Failed to store in memory: {e}")

    logger.info(f"Strategy workflow complete in {state['processing_time_ms']}ms")
    return state


# ============================================================================
# WORKFLOW GRAPH CONSTRUCTION
# ============================================================================

def create_strategy_rag_workflow():
    """
    Create the Strategy RAG LangGraph workflow.

    6-node graph structure:
    START → validate_input → load_strategy → structure_output
          → validate_response → retry_decision → finalize → END
    """
    logger.info("Creating Strategy RAG workflow (6 nodes with LangGraph)...")

    workflow = StateGraph(StrategyRAGState)

    # Add 6 nodes
    workflow.add_node("validate_input", validate_input_node)
    workflow.add_node("load_strategy", load_strategy_node)
    workflow.add_node("structure_output", structure_output_node)
    workflow.add_node("validate_response", validate_response_node)
    workflow.add_node("retry_decision", retry_decision_node)
    workflow.add_node("finalize", finalize_node)

    # Set entry point
    workflow.set_entry_point("validate_input")

    # Add edges (linear flow - 6 nodes)
    workflow.add_edge("validate_input", "load_strategy")
    workflow.add_edge("load_strategy", "structure_output")
    workflow.add_edge("structure_output", "validate_response")
    workflow.add_edge("validate_response", "retry_decision")
    workflow.add_edge("retry_decision", "finalize")
    workflow.add_edge("finalize", END)

    # Compile with checkpointing
    try:
        from ..checkpointing import compile_with_checkpointing
        app = compile_with_checkpointing(workflow)
    except ImportError:
        logger.warning("Checkpointing not available, compiling without it")
        app = workflow.compile()

    logger.info("Strategy RAG workflow created successfully")
    return app


# ============================================================================
# WORKFLOW EXECUTION
# ============================================================================

_strategy_rag_app = None


def get_strategy_rag_workflow():
    """Get or create the Strategy RAG workflow instance (singleton)."""
    global _strategy_rag_app
    if _strategy_rag_app is None:
        _strategy_rag_app = create_strategy_rag_workflow()
    return _strategy_rag_app


def run_strategy_rag_workflow(
    product_type: str = "",
    question: str = "",
    requirements: Dict[str, Any] = None,
    session_id: Optional[str] = None,
    top_k: int = 7,
    refinery: str = None
) -> Dict[str, Any]:
    """
    Run the Strategy LangGraph Workflow.

    Args:
        product_type: Product type to get strategy for
        question: Optional specific question
        requirements: Additional requirements context
        session_id: Optional session ID for memory
        top_k: Number of vendors to return
        refinery: Optional refinery to filter by

    Returns:
        Result dictionary with strategy data
    """
    logger.info(f"[Strategy] Running LangGraph workflow for: {product_type}")

    try:
        # Get or create workflow
        app = get_strategy_rag_workflow()

        # Create initial state
        initial_state = create_strategy_rag_state(
            question=question,
            product_type=product_type,
            requirements=requirements,
            session_id=session_id,
            top_k=top_k,
            refinery=refinery
        )

        # Run workflow
        config = {"configurable": {"thread_id": session_id or "default"}}
        final_state = app.invoke(initial_state, config=config)

        # Return result
        return {
            'status': final_state.get('status', 'error'),
            'final_response': final_state.get('final_response', {}),
            'preferred_vendors': final_state.get('preferred_vendors', []),
            'forbidden_vendors': final_state.get('forbidden_vendors', []),
            'neutral_vendors': final_state.get('neutral_vendors', []),
            'procurement_priorities': final_state.get('procurement_priorities', {}),
            'strategy_notes': final_state.get('strategy_notes', ''),
            'confidence': final_state.get('confidence', 0.0),
            'sources_used': final_state.get('sources_used', []),
            'processing_time_ms': final_state.get('processing_time_ms', 0),
            'error': final_state.get('error')
        }

    except Exception as e:
        logger.error(f"[Strategy] Workflow error: {e}", exc_info=True)
        return {
            'status': 'error',
            'error': str(e),
            'final_response': {
                'answer': f"Error: {str(e)}",
                'preferred_vendors': [],
                'forbidden_vendors': [],
                'neutral_vendors': [],
                'procurement_priorities': {},
                'strategy_notes': '',
                'citations': [],
                'confidence': 0.0,
                'sources_used': [],
                'metadata': {'rag_type': 'csv_filter_langgraph'}
            },
            'preferred_vendors': [],
            'forbidden_vendors': [],
            'neutral_vendors': [],
            'procurement_priorities': {},
            'processing_time_ms': 0
        }


# ============================================================================
# CONVENIENCE FUNCTIONS (Backward Compatible)
# ============================================================================

def get_strategy_for_product(
    product_type: str,
    requirements: Dict[str, Any] = None,
    top_k: int = 7,
    refinery: str = None
) -> Dict[str, Any]:
    """
    Quick function to get strategy for a product type.

    Returns just the strategy data, not the full workflow state.
    """
    result = run_strategy_rag_workflow(
        product_type=product_type,
        requirements=requirements,
        top_k=top_k,
        refinery=refinery
    )

    if result.get('status') == 'success':
        return result.get('final_response', {})
    else:
        return {
            'error': result.get('error', 'Unknown error'),
            'preferred_vendors': [],
            'forbidden_vendors': [],
            'neutral_vendors': [],
            'procurement_priorities': {},
            'confidence': 0.0
        }


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'StrategyRAGState',
    'create_strategy_rag_state',
    'create_strategy_rag_workflow',
    'get_strategy_rag_workflow',
    'run_strategy_rag_workflow',
    'get_strategy_for_product',
    # Node functions for testing
    'validate_input_node',
    'load_strategy_node',
    'structure_output_node',
    'validate_response_node',
    'retry_decision_node',
    'finalize_node'
]


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("TESTING STRATEGY RAG WORKFLOW (LANGGRAPH CSV-BASED)")
    print("=" * 80)

    test_products = [
        "pressure transmitter",
        "flow meter",
        "temperature sensor",
        "control valve"
    ]

    for i, product in enumerate(test_products, 1):
        print(f"\n[Test {i}] Product: {product}")
        print("-" * 80)

        result = run_strategy_rag_workflow(
            product_type=product,
            session_id=f"test-strategy-{i}",
            top_k=5
        )

        if result['status'] == 'success':
            response = result['final_response']
            print(f"\nCategory: {response['metadata'].get('category', 'N/A')}")
            print(f"Preferred Vendors: {response.get('preferred_vendors', [])}")
            print(f"Neutral Vendors: {response.get('neutral_vendors', [])}")
            print(f"Procurement Priorities: {response.get('procurement_priorities', {})}")
            print(f"Strategy Notes: {response.get('strategy_notes', '')[:200]}...")
            print(f"\nConfidence: {response.get('confidence', 0):.2f}")
            print(f"Validation Score: {response['metadata'].get('validation_score', 0):.2f}")
            print(f"Processing time: {response['metadata']['processing_time_ms']}ms")
        else:
            print(f"\n[ERROR] {result.get('error', 'Unknown error')}")

        print("=" * 80)
