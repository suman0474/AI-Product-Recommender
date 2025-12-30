"""
Streaming wrappers for all agentic workflows
Provides progress-emitting versions of all workflow functions
"""

import logging
from typing import Dict, Any, Optional, Callable

from .streaming_utils import ProgressEmitter

# Import all workflow functions
from .solution_workflow import run_solution_workflow
from .comparison_workflow import run_comparison_workflow, run_comparison_from_spec
from .instrument_detail_workflow import run_instrument_detail_workflow
from .instrument_identifier_workflow import run_instrument_identifier_workflow
from .grounded_chat_workflow import run_grounded_chat_workflow

logger = logging.getLogger(__name__)


# ============================================================================
# COMPARISON WORKFLOW STREAMING
# ============================================================================

def run_comparison_workflow_stream(
    user_input: str,
    session_id: str = "default",
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
) -> Dict[str, Any]:
    """
    Run comparison workflow with streaming progress updates.

    Args:
        user_input: User's comparison request
        session_id: Session identifier
        progress_callback: Callback function to emit progress updates

    Returns:
        Workflow result with ranked comparison
    """
    emitter = ProgressEmitter(progress_callback)

    try:
        # Step 1: Initialize
        emitter.emit(
            'initialize',
            'Starting comparison workflow...',
            5,
            data={'query': user_input[:100]}
        )

        # Step 2: Parse comparison request
        emitter.emit(
            'parse_request',
            'Analyzing comparison request...',
            20
        )

        # Step 3: Search vendors
        emitter.emit(
            'search_vendors',
            'Finding vendors and products...',
            40
        )

        # Step 4: Gather product data
        emitter.emit(
            'gather_data',
            'Collecting product specifications...',
            60
        )

        # Step 5: Compare products
        emitter.emit(
            'compare',
            'Running detailed comparison analysis...',
            80
        )

        # Execute workflow
        logger.info(f"[COMPARISON-STREAM] Starting for session {session_id}")
        result = run_comparison_workflow(user_input, session_id)

        # Step 6: Complete
        if result.get("success", True):
            emitter.emit(
                'complete',
                'Comparison completed successfully',
                100,
                data={
                    'vendor_count': len(result.get('ranked_products', [])),
                }
            )
        else:
            emitter.error(result.get('error', 'Unknown error'))

        return result

    except Exception as e:
        logger.error(f"[COMPARISON-STREAM] Failed: {e}", exc_info=True)
        emitter.error(f'Comparison failed: {str(e)}')
        return {
            "success": False,
            "error": str(e)
        }


def run_comparison_from_spec_stream(
    spec_object: Dict[str, Any],
    comparison_type: str = "full",
    session_id: str = "default",
    user_id: Optional[str] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
) -> Dict[str, Any]:
    """
    Run comparison from spec with streaming progress updates.

    Args:
        spec_object: Product specification object
        comparison_type: Type of comparison (vendor, series, model, full)
        session_id: Session identifier
        user_id: Optional user ID
        progress_callback: Callback function to emit progress updates

    Returns:
        Multi-level comparison results
    """
    emitter = ProgressEmitter(progress_callback)

    try:
        product_type = spec_object.get('product_type', 'product')

        # Step 1: Initialize
        emitter.emit(
            'initialize',
            f'Starting {comparison_type} comparison for {product_type}...',
            5,
            data={'product_type': product_type, 'comparison_type': comparison_type}
        )

        # Step 2: Load specifications
        emitter.emit(
            'load_spec',
            'Loading product specifications...',
            15
        )

        # Step 3: Vendor-level comparison
        emitter.emit(
            'vendor_comparison',
            'Comparing vendors...',
            30
        )

        # Step 4: Series-level comparison
        if comparison_type in ['series', 'full']:
            emitter.emit(
                'series_comparison',
                'Comparing product series...',
                55
            )

        # Step 5: Model-level comparison
        if comparison_type in ['model', 'full']:
            emitter.emit(
                'model_comparison',
                'Comparing specific models...',
                75
            )

        # Execute workflow
        logger.info(f"[COMPARISON-SPEC-STREAM] Starting {comparison_type} comparison")
        result = run_comparison_from_spec(spec_object, comparison_type, session_id, user_id)

        # Step 6: Complete
        if result.get("success", True):
            emitter.emit(
                'complete',
                'Multi-level comparison completed',
                100,
                data={
                    'top_vendor': result.get('top_recommendation', {}).get('vendor'),
                }
            )
        else:
            emitter.error(result.get('error', 'Unknown error'))

        return result

    except Exception as e:
        logger.error(f"[COMPARISON-SPEC-STREAM] Failed: {e}", exc_info=True)
        emitter.error(f'Comparison failed: {str(e)}')
        return {
            "success": False,
            "error": str(e)
        }


# ============================================================================
# INSTRUMENT DETAIL WORKFLOW STREAMING
# ============================================================================

def run_instrument_detail_workflow_stream(
    user_input: str,
    session_id: str = "default",
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
) -> Dict[str, Any]:
    """
    Run instrument detail workflow with streaming progress updates.

    Args:
        user_input: Requirements with instruments/accessories
        session_id: Session identifier
        progress_callback: Callback function to emit progress updates

    Returns:
        Identified items and rankings
    """
    emitter = ProgressEmitter(progress_callback)

    try:
        # Step 1: Initialize
        emitter.emit(
            'initialize',
            'Starting instrument detail capture...',
            5
        )

        # Step 2: Identify instruments
        emitter.emit(
            'identify_instruments',
            'Identifying required instruments...',
            25
        )

        # Step 3: Identify accessories
        emitter.emit(
            'identify_accessories',
            'Identifying required accessories...',
            45
        )

        # Step 4: Validate specifications
        emitter.emit(
            'validate_specs',
            'Validating specifications...',
            65
        )

        # Step 5: Rank products
        emitter.emit(
            'rank_products',
            'Ranking matching products...',
            85
        )

        # Execute workflow
        logger.info(f"[INSTRUMENT-DETAIL-STREAM] Starting for session {session_id}")
        result = run_instrument_detail_workflow(user_input, session_id)

        # Step 6: Complete
        if result.get("success", True):
            emitter.emit(
                'complete',
                'Instrument detail capture completed',
                100,
                data={
                    'instrument_count': len(result.get('identified_instruments', [])),
                    'accessory_count': len(result.get('identified_accessories', [])),
                }
            )
        else:
            emitter.error(result.get('error', 'Unknown error'))

        return result

    except Exception as e:
        logger.error(f"[INSTRUMENT-DETAIL-STREAM] Failed: {e}", exc_info=True)
        emitter.error(f'Instrument detail workflow failed: {str(e)}')
        return {
            "success": False,
            "error": str(e)
        }


# ============================================================================
# INSTRUMENT IDENTIFIER WORKFLOW STREAMING
# ============================================================================

def run_instrument_identifier_workflow_stream(
    user_input: str,
    session_id: str = "default",
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
) -> Dict[str, Any]:
    """
    Run instrument identifier workflow with streaming progress updates.

    Args:
        user_input: Project description or requirements
        session_id: Session identifier
        progress_callback: Callback function to emit progress updates

    Returns:
        List of identified instruments for selection
    """
    emitter = ProgressEmitter(progress_callback)

    try:
        # Step 1: Initialize
        emitter.emit(
            'initialize',
            'Analyzing project requirements...',
            10
        )

        # Step 2: Parse requirements
        emitter.emit(
            'parse_requirements',
            'Parsing technical requirements...',
            30
        )

        # Step 3: Identify instruments
        emitter.emit(
            'identify_items',
            'Identifying required instruments and equipment...',
            60
        )

        # Step 4: Generate list
        emitter.emit(
            'generate_list',
            'Generating instrument list...',
            85
        )

        # Execute workflow
        logger.info(f"[IDENTIFIER-STREAM] Starting for session {session_id}")
        result = run_instrument_identifier_workflow(user_input, session_id)

        # Step 5: Complete
        if result.get("success", True):
            emitter.emit(
                'complete',
                'Instrument list generated',
                100,
                data={
                    'item_count': len(result.get('items', [])),
                    'awaiting_selection': result.get('awaiting_selection', False)
                }
            )
        else:
            emitter.error(result.get('error', 'Unknown error'))

        return result

    except Exception as e:
        logger.error(f"[IDENTIFIER-STREAM] Failed: {e}", exc_info=True)
        emitter.error(f'Instrument identifier failed: {str(e)}')
        return {
            "success": False,
            "error": str(e)
        }


# ============================================================================
# GROUNDED CHAT WORKFLOW STREAMING
# ============================================================================

def run_grounded_chat_workflow_stream(
    user_question: str,
    session_id: str = "default",
    user_id: Optional[str] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
) -> Dict[str, Any]:
    """
    Run grounded chat workflow with streaming progress updates.

    Args:
        user_question: User's question
        session_id: Session identifier
        user_id: Optional user ID
        progress_callback: Callback function to emit progress updates

    Returns:
        Grounded answer with citations
    """
    emitter = ProgressEmitter(progress_callback)

    try:
        # Step 1: Initialize
        emitter.emit(
            'initialize',
            'Processing your question...',
            10
        )

        # Step 2: Search knowledge base
        emitter.emit(
            'search_knowledge',
            'Searching knowledge base...',
            30
        )

        # Step 3: Retrieve relevant documents
        emitter.emit(
            'retrieve_docs',
            'Retrieving relevant information...',
            50
        )

        # Step 4: Generate answer
        emitter.emit(
            'generate_answer',
            'Generating grounded response...',
            70
        )

        # Step 5: Validate answer
        emitter.emit(
            'validate',
            'Validating answer accuracy...',
            90
        )

        # Execute workflow
        logger.info(f"[GROUNDED-CHAT-STREAM] Starting for session {session_id}")
        result = run_grounded_chat_workflow(user_question, session_id, user_id)

        # Step 6: Complete
        if result.get("success", True):
            emitter.emit(
                'complete',
                'Answer generated successfully',
                100,
                data={
                    'citation_count': len(result.get('citations', [])),
                    'confidence': result.get('confidence', 0)
                }
            )
        else:
            emitter.error(result.get('error', 'Unknown error'))

        return result

    except Exception as e:
        logger.error(f"[GROUNDED-CHAT-STREAM] Failed: {e}", exc_info=True)
        emitter.error(f'Grounded chat failed: {str(e)}')
        return {
            "success": False,
            "error": str(e)
        }
