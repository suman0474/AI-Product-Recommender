# agentic/api_streaming.py
"""
Streaming API Endpoints for Agentic Workflows

This module contains all SSE (Server-Sent Events) streaming endpoints
that provide real-time progress updates during workflow execution.

All streaming endpoints use StreamingWorkflowRunner to emit progress
events to the frontend.
"""

import logging
from flask import request

# Import blueprint and utilities from main api module
from .api import agentic_bp, api_response, get_session_id, login_required, handle_errors

# Import streaming utilities
from .streaming_utils import StreamingWorkflowRunner
from .workflow_streaming import (
    run_instrument_identifier_workflow_stream,
    run_potential_product_index_workflow_stream
)
# EnGenie streaming has been migrated to Product Info
# See product_info_api.py: /query/stream
# Import solution streaming directly from solution_workflow
from .solution_workflow import run_solution_workflow_stream

logger = logging.getLogger(__name__)


# ============================================================================
# STREAMING ENDPOINTS (SSE)
# ============================================================================

@agentic_bp.route('/solution/stream', methods=['POST'])
@login_required
@handle_errors
def solution_workflow_stream():
    """
    Solution Workflow with SSE Streaming
    ---
    tags:
      - Streaming Workflows
    summary: Stream real-time progress updates for solution workflow
    description: |
      Streams real-time progress updates during solution workflow execution.

      **Progress Steps:**
      - Initialize workflow (5%)
      - Classify intent (15%)
      - Identify instruments (30%)
      - Aggregate RAG constraints (45%)
      - Search vendors (60%)
      - Analyze products (75%)
      - Rank products (90%)
      - Finalize results (100%)

      **SSE Events:** Each step emits progress updates with:
      - `step`: Current step identifier
      - `message`: Human-readable status
      - `progress`: Percentage (0-100)
      - `data`: Step-specific data

    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - message
          properties:
            message:
              type: string
              description: Product requirements
              example: "Need SIL2 pressure transmitters for crude oil unit"
            session_id:
              type: string
              description: Session identifier (optional)

    responses:
      200:
        description: SSE stream with progress updates
        content:
          text/event-stream:
            schema:
              type: object
              properties:
                step:
                  type: string
                  example: "search_vendors"
                message:
                  type: string
                  example: "Searching vendor database..."
                progress:
                  type: integer
                  example: 60
                data:
                  type: object
    """
    data = request.get_json()

    if not data or 'message' not in data:
        return api_response(False, error="Message is required", status_code=400)

    message = data['message']
    session_id = data.get('session_id') or get_session_id()

    logger.info(f"[SOLUTION/STREAM] Starting streaming for session {session_id}")

    runner = StreamingWorkflowRunner(
        run_solution_workflow_stream,
        user_input=message,
        session_id=session_id
    )

    return runner.create_sse_response()








# EnGenie chat streaming endpoint has been removed and migrated to Product Info
# Use /api/product-info/query/stream instead


@agentic_bp.route('/instrument-identifier/stream', methods=['POST'])
@login_required
@handle_errors
def instrument_identifier_stream():
    """
    Instrument Identifier Workflow with SSE Streaming
    ---
    tags:
      - Streaming Workflows
    summary: Stream real-time progress for instrument identification
    """
    data = request.get_json()

    if not data or 'message' not in data:
        return api_response(False, error="Message is required", status_code=400)

    message = data['message']
    session_id = data.get('session_id') or get_session_id()

    logger.info(f"[INSTRUMENT-IDENTIFIER/STREAM] Starting streaming for session {session_id}")

    runner = StreamingWorkflowRunner(
        run_instrument_identifier_workflow_stream,
        user_input=message,
        session_id=session_id
    )

    return runner.create_sse_response()


@agentic_bp.route('/potential-product-index/stream', methods=['POST'])
@login_required
@handle_errors
def potential_product_index_stream():
    """
    Potential Product Index with SSE Streaming
    ---
    tags:
      - Streaming Workflows
    summary: Stream real-time progress for schema discovery (PPI)
    description: |
      Streams real-time progress during PPI workflow execution.
      This is a long-running workflow (45-60 seconds) that discovers
      vendors, fetches datasheets, and generates product schemas.

      **Progress Steps:**
      - Vendor discovery (15%)
      - Model family discovery (30%)
      - PDF search (45%)
      - PDF processing (60%)
      - Content indexing (75%)
      - Schema generation (85%)
      - Validation & save (100%)
    """
    data = request.get_json()

    if not data or 'product_type' not in data:
        return api_response(False, error="product_type is required", status_code=400)

    product_type = data['product_type']
    session_id = data.get('session_id') or get_session_id()
    parent_workflow = data.get('parent_workflow', 'direct')

    logger.info(f"[PPI/STREAM] Starting streaming for {product_type}, session {session_id}")

    runner = StreamingWorkflowRunner(
        run_potential_product_index_workflow_stream,
        product_type=product_type,
        session_id=session_id,
        parent_workflow=parent_workflow
    )

    return runner.create_sse_response()
