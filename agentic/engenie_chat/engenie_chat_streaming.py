"""
EnGenie Chat Streaming Module

Provides streaming wrapper for EnGenie Chat queries with progress updates
for classification, query execution, validation, and completion.
"""

import logging
from typing import Dict, Any, Optional, Callable, List

from ..streaming_utils import ProgressEmitter, ProgressStep, with_streaming
from .engenie_chat_orchestrator import run_engenie_chat_query

logger = logging.getLogger(__name__)


# ============================================================================
# ENGENIE CHAT WORKFLOW STREAMING
# ============================================================================

@with_streaming(
    progress_steps=[
        ProgressStep('initialize', 'Processing your question...', 5),
        ProgressStep('classify', 'Classifying intent and selecting data sources...', 15),
        ProgressStep('query_sources', 'Querying knowledge bases...', 40),
        ProgressStep('web_search', 'Searching web for additional information...', 55),
        ProgressStep('validate', 'Validating response accuracy...', 80),
        ProgressStep('finalize', 'Finalizing response...', 95),
        ProgressStep('complete', 'Query completed successfully', 100)
    ],
    workflow_name="engenie_chat",
    extract_result_metadata=lambda r: {
        'source': r.get('source', 'unknown'),
        'sources_used': r.get('sources_used', []),
        'is_validated': r.get('is_validated', False),
        'validation_score': r.get('validation_score'),
        'found_in_database': r.get('found_in_database', False),
        'retry_count': r.get('retry_count', 0),
        'escalated': r.get('escalated', False)
    }
)
def run_engenie_chat_query_stream(
    query: str,
    session_id: str = "default",
    rag_type: str = None,
    validate: bool = True,
    entities: List[str] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
) -> Dict[str, Any]:
    """
    Run EnGenie Chat query with streaming progress updates.

    This is the streaming version of run_engenie_chat_query that emits
    progress events for:
    - Intent classification and routing
    - Parallel source querying (Index RAG, Standards RAG, Strategy RAG, Deep Agent, Web Search)
    - Response validation with 5-dimensional scoring
    - Retry logic and escalation

    Args:
        query: User's question or query
        session_id: Session identifier for memory management
        rag_type: Optional RAG type to force specific routing
                  (index_rag, standards_rag, strategy_rag, deep_agent, web_search, hybrid)
        validate: Whether to validate response (default: True)
        entities: Optional list of entities for verification
        progress_callback: Callback function to emit progress updates

    Returns:
        Query result with answer, sources, validation metadata

    Decorator handles all streaming - this function just executes the workflow.
    """
    # Execute the orchestrated query
    result = run_engenie_chat_query(
        query=query,
        session_id=session_id,
        rag_type=rag_type,
        validate=validate,
        entities=entities
    )
    return result


# ============================================================================
# ENHANCED STREAMING WITH MANUAL PROGRESS CONTROL
# ============================================================================

def run_engenie_chat_query_stream_detailed(
    query: str,
    session_id: str = "default",
    rag_type: str = None,
    validate: bool = True,
    entities: List[str] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
) -> Dict[str, Any]:
    """
    Run EnGenie Chat query with detailed manual progress control.

    This version provides more granular progress updates by manually
    emitting events at each stage of the workflow. Use this when you
    need finer control over progress reporting.

    Args:
        query: User's question or query
        session_id: Session identifier
        rag_type: Optional RAG type override
        validate: Whether to validate response
        entities: Optional entity list for verification
        progress_callback: Callback function to emit progress updates

    Returns:
        Query result with answer, sources, validation metadata
    """
    emitter = ProgressEmitter(progress_callback)

    try:
        # Step 1: Initialize
        emitter.emit('initialize', 'Processing your question...', 5)

        # Step 2: Intent Classification
        emitter.emit('classify', 'Analyzing query intent...', 15)

        # Step 3: Execute Query
        emitter.emit('query_sources', f'Querying {rag_type or "auto-selected"} sources...', 40)

        # Execute the main query
        result = run_engenie_chat_query(
            query=query,
            session_id=session_id,
            rag_type=rag_type,
            validate=validate,
            entities=entities
        )

        # Step 4: Report sources queried
        sources_used = result.get('sources_used', [])
        if sources_used:
            emitter.emit(
                'sources_found',
                f'Retrieved data from: {", ".join(sources_used)}',
                60,
                data={'sources': sources_used}
            )

        # Step 5: Validation status
        if result.get('is_validated'):
            score = result.get('validation_score', 0)
            emitter.emit(
                'validated',
                f'Response validated (score: {score:.2f})',
                80,
                data={'validation_score': score}
            )
        elif result.get('retry_count', 0) > 0:
            emitter.emit(
                'retried',
                f'Response improved after {result.get("retry_count")} retries',
                80,
                data={'retry_count': result.get('retry_count')}
            )

        # Step 6: Escalation check
        if result.get('escalated'):
            emitter.emit(
                'escalated',
                'Response flagged for human review',
                90,
                data={'escalated': True}
            )

        # Step 7: Complete
        emitter.emit(
            'complete',
            'Query completed successfully',
            100,
            data={
                'source': result.get('source', 'unknown'),
                'found_in_database': result.get('found_in_database', False),
                'is_validated': result.get('is_validated', False)
            }
        )

        return result

    except Exception as e:
        logger.error(f"[ENGENIE_CHAT_STREAM] Error: {e}", exc_info=True)
        emitter.error(f'Query failed: {str(e)}', error_details=e)
        return {
            'success': False,
            'error': str(e),
            'answer': 'Sorry, something went wrong. Please try again.',
            'source': 'error',
            'found_in_database': False,
            'sources_used': [],
            'is_validated': False,
            'validation_score': None,
            'retry_count': 0,
            'escalated': False
        }


# Export both streaming versions
__all__ = [
    'run_engenie_chat_query_stream',
    'run_engenie_chat_query_stream_detailed'
]
