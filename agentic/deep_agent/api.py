# agentic/deep_agent_api.py
# =============================================================================
# DEEP AGENT API ENDPOINTS
# =============================================================================
#
# Flask Blueprint for Deep Agent Workflow Endpoints
#
# =============================================================================

import logging
import json
import os
from typing import Dict, Any, Optional
from datetime import datetime
from flask import Blueprint, request, jsonify, Response

from .workflow import (
    run_deep_agent_workflow,
    get_deep_agent_workflow,
    get_memory  # Memory manager function
)
from .memory import (
    PRODUCT_TYPE_DOCUMENT_MAP,
    get_relevant_documents_for_product
)
from .document_loader import (
    STANDARDS_DIRECTORY,
    analyze_user_input
)

logger = logging.getLogger(__name__)

deep_agent_bp = Blueprint("deep_agent", __name__, url_prefix="/deep-agent")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_standards_files() -> list:
    """Get list of all standards documents available"""
    if not os.path.exists(STANDARDS_DIRECTORY):
        return []

    files = []
    try:
        for filename in os.listdir(STANDARDS_DIRECTORY):
            if filename.endswith(('.docx', '.doc', '.pdf', '.txt')):
                filepath = os.path.join(STANDARDS_DIRECTORY, filename)
                file_size = os.path.getsize(filepath)
                files.append({
                    "filename": filename,
                    "size_bytes": file_size,
                    "path": filepath
                })
    except Exception as e:
        logger.error(f"Error reading standards directory: {e}")

    return sorted(files, key=lambda x: x["filename"])


# =============================================================================
# ENDPOINT: Health Check
# =============================================================================

@deep_agent_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for Deep Agent"""
    try:
        standards_count = len(get_standards_files())
        product_types_count = len(PRODUCT_TYPE_DOCUMENT_MAP)

        return jsonify({
            "status": "healthy",
            "service": "deep_agent",
            "timestamp": datetime.now().isoformat(),
            "standards_documents_available": standards_count,
            "product_types_mapped": product_types_count
        })
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500


# =============================================================================
# ENDPOINT: Generate Specifications
# =============================================================================

@deep_agent_bp.route("/generate-specs", methods=["POST"])
def generate_specs():
    """
    Generate standard specifications for identified instruments/accessories

    Request Body:
    {
        "user_input": "string - user requirements",
        "identified_instruments": [
            {
                "product_type": "string",
                "quantity": int,
                "user_specifications": {}
            }
        ],
        "identified_accessories": [
            {
                "product_type": "string",
                "quantity": int,
                "user_specifications": {}
            }
        ],
        "session_id": "string (optional)",
        "cleanup_after": bool (optional, default: True)
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "error": "Request body is required"
            }), 400

        user_input = data.get("user_input", "")
        identified_instruments = data.get("identified_instruments", [])
        identified_accessories = data.get("identified_accessories", [])
        session_id = data.get("session_id")
        cleanup_after = data.get("cleanup_after", True)

        if not user_input:
            return jsonify({
                "success": False,
                "error": "user_input is required"
            }), 400

        # Run the workflow
        result = run_deep_agent_workflow(
            user_input=user_input,
            identified_instruments=identified_instruments,
            identified_accessories=identified_accessories,
            session_id=session_id,
            cleanup_after=cleanup_after
        )

        # Extract response data from workflow result
        response_data = result.get("response_data", {})

        if response_data.get("success"):
            return jsonify({
                "success": True,
                "data": response_data.get("standard_specifications_json", {}),
                "session_id": response_data.get("session_id"),
                "processing_time_ms": response_data.get("processing_time_ms", 0)
            })
        else:
            return jsonify({
                "success": False,
                "error": result.get("error", "Workflow execution failed")
            }), 500

    except Exception as e:
        logger.error(f"Error in generate_specs: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# =============================================================================
# ENDPOINT: Analyze User Input
# =============================================================================

@deep_agent_bp.route("/analyze-input", methods=["POST"])
def analyze_input_endpoint():
    """
    Analyze user input to extract domain, process type, safety requirements

    Request Body:
    {
        "user_input": "string - user requirements"
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "error": "Request body is required"
            }), 400

        user_input = data.get("user_input", "")

        if not user_input:
            return jsonify({
                "success": False,
                "error": "user_input is required"
            }), 400

        # Analyze input
        context = analyze_user_input(user_input)

        return jsonify({
            "success": True,
            "data": context
        })

    except Exception as e:
        logger.error(f"Error in analyze_input_endpoint: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# =============================================================================
# ENDPOINT: Get Relevant Documents
# =============================================================================

@deep_agent_bp.route("/get-relevant-docs", methods=["POST"])
def get_relevant_docs():
    """
    Get list of relevant standards documents for product type(s)

    Request Body:
    {
        "product_types": ["string", "string", ...] or "string"
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "error": "Request body is required"
            }), 400

        product_types = data.get("product_types")

        if not product_types:
            return jsonify({
                "success": False,
                "error": "product_types is required"
            }), 400

        # Handle both string and list inputs
        if isinstance(product_types, str):
            product_types = [product_types]

        # Gather relevant documents
        all_docs = set()
        mapping = {}

        for product_type in product_types:
            docs = get_relevant_documents_for_product(product_type)
            mapping[product_type] = docs
            all_docs.update(docs)

        return jsonify({
            "success": True,
            "data": {
                "product_type_mapping": mapping,
                "all_relevant_documents": sorted(list(all_docs)),
                "total_documents": len(all_docs)
            }
        })

    except Exception as e:
        logger.error(f"Error in get_relevant_docs: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# =============================================================================
# ENDPOINT: List Standards Documents
# =============================================================================

@deep_agent_bp.route("/list-standards-documents", methods=["GET"])
def list_standards_documents():
    """
    Get list of all available standards documents
    """
    try:
        files = get_standards_files()

        return jsonify({
            "success": True,
            "data": {
                "documents": files,
                "total_count": len(files),
                "directory": STANDARDS_DIRECTORY
            }
        })

    except Exception as e:
        logger.error(f"Error in list_standards_documents: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# =============================================================================
# ENDPOINT: Analyze Single Document
# =============================================================================

@deep_agent_bp.route("/analyze-document", methods=["POST"])
def analyze_document():
    """
    Analyze a specific standards document (retrieve from memory if already analyzed)

    Request Body:
    {
        "filename": "string",
        "session_id": "string (optional)"
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "error": "Request body is required"
            }), 400

        filename = data.get("filename")
        session_id = data.get("session_id")

        if not filename:
            return jsonify({
                "success": False,
                "error": "filename is required"
            }), 400

        # Try to get from memory if session_id provided
        if session_id:
            memory = get_memory(session_id)
            if memory:
                analysis = memory.get_standards_analysis(filename)
                if analysis:
                    return jsonify({
                        "success": True,
                        "data": {
                            "filename": filename,
                            "document_type": analysis.get("document_type"),
                            "total_sections": analysis.get("total_sections"),
                            "all_standard_codes": analysis.get("all_standard_codes", []),
                            "all_certifications": analysis.get("all_certifications", []),
                            "all_equipment_types": analysis.get("all_equipment_types", []),
                            "source": "memory"
                        }
                    })

        return jsonify({
            "success": False,
            "error": f"Document analysis not found for {filename}. Run generate-specs first to load documents."
        }), 404

    except Exception as e:
        logger.error(f"Error in analyze_document: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# =============================================================================
# ENDPOINT: Get Memory Statistics
# =============================================================================

@deep_agent_bp.route("/memory-stats", methods=["GET"])
def get_memory_stats():
    """
    Get memory statistics for a session

    Query Parameters:
    - session_id: string (required)
    """
    try:
        session_id = request.args.get("session_id")

        if not session_id:
            return jsonify({
                "success": False,
                "error": "session_id query parameter is required"
            }), 400

        memory = get_memory(session_id)

        if not memory:
            return jsonify({
                "success": False,
                "error": f"No memory found for session: {session_id}"
            }), 404

        stats = memory.get_memory_stats()

        return jsonify({
            "success": True,
            "data": {
                "session_id": session_id,
                "statistics": stats
            }
        })

    except Exception as e:
        logger.error(f"Error in get_memory_stats: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# =============================================================================
# ENDPOINT: Generate Specifications with Streaming
# =============================================================================

@deep_agent_bp.route("/generate-specs-stream", methods=["POST"])
def generate_specs_stream():
    """
    Generate specifications with streaming response (Server-Sent Events)

    Request Body:
    {
        "user_input": "string - user requirements",
        "identified_instruments": [
            {
                "product_type": "string",
                "quantity": int,
                "user_specifications": {}
            }
        ],
        "identified_accessories": [
            {
                "product_type": "string",
                "quantity": int,
                "user_specifications": {}
            }
        ],
        "session_id": "string (optional)",
        "cleanup_after": bool (optional, default: False for streaming)
    }
    """
    try:
        data = request.get_json()

        if not data:
            return Response(
                json.dumps({"error": "Request body is required"}) + "\n",
                mimetype="text/event-stream",
                status=400
            )

        user_input = data.get("user_input", "")
        identified_instruments = data.get("identified_instruments", [])
        identified_accessories = data.get("identified_accessories", [])
        session_id = data.get("session_id")
        cleanup_after = data.get("cleanup_after", False)

        if not user_input:
            return Response(
                json.dumps({"error": "user_input is required"}) + "\n",
                mimetype="text/event-stream",
                status=400
            )

        def generate():
            """Generator for streaming responses"""
            try:
                # Send initial event
                yield f"data: {json.dumps({'status': 'starting', 'message': 'Deep Agent workflow starting...'})}\n\n"

                # Run the workflow
                result = run_deep_agent_workflow(
                    user_input=user_input,
                    identified_instruments=identified_instruments,
                    identified_accessories=identified_accessories,
                    session_id=session_id,
                    cleanup_after=cleanup_after
                )

                response_data = result.get("response_data", {})
                if response_data.get("success"):
                    yield f"data: {json.dumps({'status': 'processing', 'message': 'Documents loaded and analyzed'})}\n\n"
                    yield f"data: {json.dumps({'status': 'processing', 'message': 'Items identified and specs generated'})}\n\n"

                    yield f"data: {json.dumps({'status': 'complete', 'result': response_data.get('standard_specifications_json', {}), 'processing_time_ms': response_data.get('processing_time_ms', 0)})}\n\n"
                else:
                    yield f"data: {json.dumps({'status': 'error', 'error': response_data.get('error', 'Workflow execution failed')})}\n\n"

            except Exception as e:
                logger.error(f"Error in streaming: {e}", exc_info=True)
                yield f"data: {json.dumps({'status': 'error', 'error': str(e)})}\n\n"

        return Response(generate(), mimetype="text/event-stream")

    except Exception as e:
        logger.error(f"Error in generate_specs_stream: {e}", exc_info=True)
        return Response(
            json.dumps({"error": str(e)}) + "\n",
            mimetype="text/event-stream",
            status=500
        )


# =============================================================================
# ENDPOINT: Product Type Mapping
# =============================================================================

@deep_agent_bp.route("/product-type-mapping", methods=["GET"])
def product_type_mapping():
    """
    Get the static product type to document mapping
    """
    try:
        return jsonify({
            "success": True,
            "data": {
                "product_type_document_map": PRODUCT_TYPE_DOCUMENT_MAP,
                "total_product_types": len(PRODUCT_TYPE_DOCUMENT_MAP)
            }
        })

    except Exception as e:
        logger.error(f"Error in product_type_mapping: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "deep_agent_bp"
]
