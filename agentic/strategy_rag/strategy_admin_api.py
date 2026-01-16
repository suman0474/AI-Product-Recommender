# agentic/strategy_rag/strategy_admin_api.py
# =============================================================================
# STRATEGY ADMIN API - Real-time strategy updates without file edits
# =============================================================================
#
# PURPOSE: API endpoints for adding/updating strategy data dynamically
# without requiring CSV file modifications.
#
# =============================================================================

import logging
from flask import Blueprint, request, jsonify
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Create Blueprint
strategy_admin_bp = Blueprint('strategy_admin', __name__, url_prefix='/api/strategy-admin')


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def api_response(success: bool, data: Any = None, error: str = None, status_code: int = 200):
    """Standard API response format."""
    response = {
        "success": success,
        "data": data,
        "error": error
    }
    return jsonify(response), status_code


# =============================================================================
# VECTOR STORE MANAGEMENT
# =============================================================================

@strategy_admin_bp.route('/index/status', methods=['GET'])
def get_index_status():
    """
    Get vector store index status.
    
    Returns:
        Index statistics including document count, availability
    """
    try:
        from .strategy_vector_store import get_strategy_vector_store
        
        store = get_strategy_vector_store()
        stats = store.get_stats()
        
        return api_response(True, data=stats)
        
    except ImportError as e:
        return api_response(False, error=f"Vector store not available: {e}")
    except Exception as e:
        logger.error(f"[StrategyAdmin] Error getting index status: {e}")
        return api_response(False, error=str(e), status_code=500)


@strategy_admin_bp.route('/index/refresh', methods=['POST'])
def refresh_index():
    """
    Force refresh the vector store index from CSV.
    
    Re-indexes all strategy data from the CSV file.
    """
    try:
        from .strategy_vector_store import get_strategy_vector_store
        
        store = get_strategy_vector_store()
        result = store.refresh_index()
        
        return api_response(True, data={
            "message": "Index refreshed successfully",
            "documents_indexed": result.get('indexed', 0),
            "status": result.get('status', 'unknown')
        })
        
    except ImportError as e:
        return api_response(False, error=f"Vector store not available: {e}")
    except Exception as e:
        logger.error(f"[StrategyAdmin] Error refreshing index: {e}")
        return api_response(False, error=str(e), status_code=500)


# =============================================================================
# VENDOR MANAGEMENT
# =============================================================================

@strategy_admin_bp.route('/vendor/add', methods=['POST'])
def add_vendor_strategy():
    """
    Add a new vendor strategy entry.
    
    Request body:
    {
        "category": "Pressure Instruments",
        "vendor": "New Vendor Name",
        "strategy": "Long-term partnership agreement",
        "subcategory": "Pressure Transmitters",  // optional
        "notes": "Additional comments"  // optional
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return api_response(False, error="Request body required", status_code=400)
        
        # Validate required fields
        required = ['category', 'vendor', 'strategy']
        missing = [f for f in required if not data.get(f)]
        
        if missing:
            return api_response(False, error=f"Missing required fields: {missing}", status_code=400)
        
        from .strategy_vector_store import get_strategy_vector_store
        
        store = get_strategy_vector_store()
        result = store.add_strategy_entry(
            category=data['category'],
            vendor=data['vendor'],
            strategy=data['strategy'],
            subcategory=data.get('subcategory', ''),
            notes=data.get('notes', '')
        )
        
        logger.info(f"[StrategyAdmin] Added vendor strategy: {data['vendor']} / {data['category']}")
        
        return api_response(True, data={
            "message": f"Vendor strategy added for {data['vendor']}",
            "id": result.get('id'),
            "category": data['category']
        })
        
    except ImportError as e:
        return api_response(False, error=f"Vector store not available: {e}")
    except Exception as e:
        logger.error(f"[StrategyAdmin] Error adding vendor: {e}")
        return api_response(False, error=str(e), status_code=500)


@strategy_admin_bp.route('/vendor/search', methods=['GET'])
def search_vendor_strategy():
    """
    Search vendor strategy using semantic search.
    
    Query params:
        q: Search query (vendor name, product type, etc.)
        top_k: Number of results (default 5)
    """
    try:
        query = request.args.get('q', '')
        top_k = int(request.args.get('top_k', 5))
        
        if not query:
            return api_response(False, error="Query parameter 'q' required", status_code=400)
        
        from .strategy_vector_store import get_strategy_vector_store
        
        store = get_strategy_vector_store()
        results = store.search_similar(query=query, top_k=top_k)
        
        return api_response(True, data={
            "query": query,
            "results": results,
            "count": len(results)
        })
        
    except ImportError as e:
        return api_response(False, error=f"Vector store not available: {e}")
    except Exception as e:
        logger.error(f"[StrategyAdmin] Error searching: {e}")
        return api_response(False, error=str(e), status_code=500)


@strategy_admin_bp.route('/vendor/preferred', methods=['GET'])
def get_preferred_vendors():
    """
    Get preferred vendors for a product type.
    
    Query params:
        product_type: Product type (e.g., "pressure transmitter")
        top_k: Number of results (default 5)
    """
    try:
        product_type = request.args.get('product_type', '')
        top_k = int(request.args.get('top_k', 5))
        
        if not product_type:
            return api_response(False, error="Query parameter 'product_type' required", status_code=400)
        
        from .strategy_vector_store import get_strategy_vector_store
        
        store = get_strategy_vector_store()
        vendors = store.get_preferred_vendors(product_type=product_type, top_k=top_k)
        
        return api_response(True, data={
            "product_type": product_type,
            "preferred_vendors": vendors,
            "count": len(vendors)
        })
        
    except ImportError as e:
        return api_response(False, error=f"Vector store not available: {e}")
    except Exception as e:
        logger.error(f"[StrategyAdmin] Error getting preferred vendors: {e}")
        return api_response(False, error=str(e), status_code=500)


# =============================================================================
# CATEGORY MANAGEMENT
# =============================================================================

@strategy_admin_bp.route('/category/match', methods=['GET'])
def match_category():
    """
    Match a product type to strategy category using semantic search.
    
    Query params:
        product_type: Product type to match
    """
    try:
        product_type = request.args.get('product_type', '')
        
        if not product_type:
            return api_response(False, error="Query parameter 'product_type' required", status_code=400)
        
        from .strategy_vector_store import get_strategy_vector_store
        
        store = get_strategy_vector_store()
        category, subcategory = store.match_product_to_category(product_type)
        
        return api_response(True, data={
            "product_type": product_type,
            "matched_category": category,
            "matched_subcategory": subcategory,
            "match_found": category is not None
        })
        
    except ImportError as e:
        return api_response(False, error=f"Vector store not available: {e}")
    except Exception as e:
        logger.error(f"[StrategyAdmin] Error matching category: {e}")
        return api_response(False, error=str(e), status_code=500)


# =============================================================================
# MEMORY MANAGEMENT
# =============================================================================

@strategy_admin_bp.route('/memory/session/<session_id>', methods=['GET'])
def get_session_memory(session_id: str):
    """Get memory contents for a session."""
    try:
        from .strategy_rag_memory import get_strategy_rag_memory
        
        memory = get_strategy_rag_memory()
        history = memory.get_history(session_id)
        context = memory.get_last_context(session_id)
        
        return api_response(True, data={
            "session_id": session_id,
            "history": history,
            "context": context,
            "exists": len(history) > 0
        })
        
    except Exception as e:
        logger.error(f"[StrategyAdmin] Error getting session memory: {e}")
        return api_response(False, error=str(e), status_code=500)


@strategy_admin_bp.route('/memory/session/<session_id>', methods=['DELETE'])
def clear_session_memory(session_id: str):
    """Clear memory for a specific session."""
    try:
        from .strategy_rag_memory import clear_strategy_memory
        
        clear_strategy_memory(session_id)
        
        return api_response(True, data={
            "message": f"Session {session_id} memory cleared"
        })
        
    except Exception as e:
        logger.error(f"[StrategyAdmin] Error clearing session memory: {e}")
        return api_response(False, error=str(e), status_code=500)


@strategy_admin_bp.route('/memory/cleanup', methods=['POST'])
def cleanup_expired_sessions():
    """Remove all expired sessions."""
    try:
        from .strategy_rag_memory import get_strategy_rag_memory
        
        memory = get_strategy_rag_memory()
        count = memory.cleanup_expired_sessions()
        
        return api_response(True, data={
            "message": f"Cleaned up {count} expired sessions",
            "expired_count": count
        })
        
    except Exception as e:
        logger.error(f"[StrategyAdmin] Error cleaning up sessions: {e}")
        return api_response(False, error=str(e), status_code=500)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = ['strategy_admin_bp']
