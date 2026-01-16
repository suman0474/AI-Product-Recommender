"""
Streaming wrappers for all agentic workflows
Provides progress-emitting versions of all workflow functions

REFACTORED: Now uses @with_streaming decorator to eliminate code duplication
"""

import logging
from typing import Dict, Any, Optional, Callable

from .streaming_utils import ProgressEmitter, ProgressStep, with_streaming

# Import all workflow functions
from .solution_workflow import run_solution_workflow, run_solution_workflow_stream
from .instrument_identifier_workflow import run_instrument_identifier_workflow
# EnGenie workflow has been migrated to Product Info
# See product_info_streaming.py for the new streaming implementation

from .potential_product_index import run_potential_product_index_workflow

logger = logging.getLogger(__name__)


# ============================================================================
# COMPARISON WORKFLOW STREAMING
# ============================================================================











# ============================================================================
# INSTRUMENT IDENTIFIER WORKFLOW STREAMING
# ============================================================================

@with_streaming(
    progress_steps=[
        ProgressStep('initialize', 'Analyzing project requirements...', 10),
        ProgressStep('parse_requirements', 'Parsing technical requirements...', 30),
        ProgressStep('identify_items', 'Identifying required instruments and equipment...', 60),
        ProgressStep('generate_list', 'Generating instrument list...', 85),
        ProgressStep('complete', 'Instrument list generated', 100)
    ],
    workflow_name="identifier",
    extract_result_metadata=lambda r: {
        'item_count': len(r.get('items', [])),
        'awaiting_selection': r.get('awaiting_selection', False)
    }
)
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

    Decorator handles all streaming - this function just executes the workflow.
    """
    # Execute workflow
    result = run_instrument_identifier_workflow(user_input, session_id)
    return result


# ============================================================================
# GROUNDED CHAT WORKFLOW STREAMING
# ============================================================================

# EnGenie Chat Streaming has been migrated to Product Info
# See product_info_streaming.py for the new implementation:
# - run_product_info_query_stream()
# - run_product_info_query_stream_detailed()


# ============================================================================
# PRODUCT SEARCH WORKFLOW STREAMING
# ============================================================================




# ============================================================================
# POTENTIAL PRODUCT INDEX WORKFLOW STREAMING
# ============================================================================

@with_streaming(
    progress_steps=[
        ProgressStep('initialize', 'Starting schema discovery workflow...', 5),
        ProgressStep('discover_vendors', 'Discovering top vendors for this product type...', 15),
        ProgressStep('discover_models', 'Identifying model families from each vendor...', 30),
        ProgressStep('search_pdfs', 'Searching for product datasheets...', 45),
        ProgressStep('process_pdfs', 'Downloading and extracting PDF content...', 60),
        ProgressStep('index_content', 'Indexing content for RAG retrieval...', 75),
        ProgressStep('generate_schema', 'Generating product schema from indexed data...', 85),
        ProgressStep('validate_schema', 'Validating schema structure and quality...', 92),
        ProgressStep('save_schema', 'Saving schema to database...', 98),
        ProgressStep('complete', 'Schema discovery completed successfully!', 100)
    ],
    workflow_name="ppi",
    extract_result_metadata=lambda r: {
        'vendor_count': len(r.get('discovered_vendors', [])),
        'schema_saved': r.get('schema_saved', False),
        'product_type': r.get('product_type', '')
    }
)
def run_potential_product_index_workflow_stream(
    product_type: str,
    session_id: str = "default",
    parent_workflow: str = "solution",
    rag_context: Dict[str, Any] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
) -> Dict[str, Any]:
    """
    Run Potential Product Index workflow with streaming progress updates.

    This workflow discovers vendors, fetches datasheets, and generates schemas
    for new product types. Execution time: 45-60 seconds.

    Args:
        product_type: Product type to index (e.g., "pressure transmitter")
        session_id: Session identifier
        parent_workflow: Parent workflow that triggered this
        rag_context: Optional RAG context data
        progress_callback: Callback function to emit progress updates

    Returns:
        Schema discovery result with vendors, model families, and generated schema

    Decorator handles all streaming - this function just executes the workflow.
    """
    # Execute workflow
    result = run_potential_product_index_workflow(
        product_type=product_type,
        session_id=session_id,
        parent_workflow=parent_workflow,
        rag_context=rag_context
    )
    return result
