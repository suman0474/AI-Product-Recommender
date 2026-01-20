"""
Product Info API

Unified API endpoint for EnGenie Chat queries with intelligent routing
to Index RAG, Standards RAG, Strategy RAG, and Deep Agent.
"""

import logging
from functools import wraps
from flask import Blueprint, request, jsonify, session

from .engenie_chat_orchestrator import run_engenie_chat_query
from .engenie_chat_memory import (
    get_session_summary, clear_session, cleanup_expired_sessions
)

logger = logging.getLogger(__name__)

# Create blueprint
engenie_chat_bp = Blueprint('engenie_chat', __name__, url_prefix='/api/engenie-chat')


# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated_function


@engenie_chat_bp.route('/query', methods=['POST'])
@login_required
def query_endpoint():
    """
    Main query endpoint for EnGenie Chat.
    
    Request:
        {
            "query": "What is Rosemount 3051S?",
            "session_id": "optional_session_id"
        }
        
    Response:
        {
            "success": true,
            "answer": "The Rosemount 3051S is...",
            "source": "database" | "llm",
            "found_in_database": true,
            "awaiting_confirmation": false,
            "sources_used": ["index_rag", "standards_rag"],
            "results_count": 5,
            "metadata": {...}
        }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'Request body required',
                'answer': 'Please provide a query.'
            }), 400
        
        query = data.get('query', '').strip()
        if not query:
            return jsonify({
                'success': False,
                'error': 'Query is required',
                'answer': 'Please enter your question.'
            }), 400
        
        # Get or generate session ID
        session_id = data.get('session_id') or session.get('engenie_chat_session_id')
        if not session_id:
            session_id = f"ec_{session.get('user_id', 'anon')}_{id(request)}"
            session['engenie_chat_session_id'] = session_id
        
        logger.info(f"[ENGENIE_CHAT_API] Query from session {session_id}: {query[:100]}...")
        
        # Run the orchestrated query
        result = run_engenie_chat_query(
            query=query,
            session_id=session_id
        )
        
        # Format response for frontend
        response = {
            'success': result.get('success', False),
            'answer': result.get('answer', ''),
            'source': result.get('source', 'unknown'),
            'found_in_database': result.get('found_in_database', False),
            'awaiting_confirmation': False,  # Reserved for LLM fallback confirmation
            'sources_used': result.get('sources_used', []),
            'results_count': len(result.get('sources_used', [])),
            'is_follow_up': result.get('is_follow_up', False),
            'classification': result.get('classification', {}),
            'note': ''
        }
        
        # Add note if multiple sources used
        if len(response['sources_used']) > 1:
            response['note'] = f"Combined information from: {', '.join(response['sources_used'])}"
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"[ENGENIE_CHAT_API] Error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e),
            'answer': 'Sorry, something went wrong. Please try again.',
            'source': 'unknown',
            'found_in_database': False,
            'awaiting_confirmation': False,
            'sources_used': []
        }), 500


@engenie_chat_bp.route('/session', methods=['GET'])
@login_required
def get_session_info():
    """Get current session summary."""
    try:
        session_id = session.get('engenie_chat_session_id')
        if not session_id:
            return jsonify({
                'success': True,
                'session': None,
                'message': 'No active session'
            })
        
        summary = get_session_summary(session_id)
        return jsonify({
            'success': True,
            'session': summary
        })
        
    except Exception as e:
        logger.error(f"[ENGENIE_CHAT_API] Session info error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@engenie_chat_bp.route('/session', methods=['DELETE'])
@login_required
def clear_session_endpoint():
    """Clear current session."""
    try:
        session_id = session.get('engenie_chat_session_id')
        if session_id:
            clear_session(session_id)
            session.pop('engenie_chat_session_id', None)
        
        return jsonify({
            'success': True,
            'message': 'Session cleared'
        })
        
    except Exception as e:
        logger.error(f"[ENGENIE_CHAT_API] Clear session error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@engenie_chat_bp.route('/cleanup', methods=['POST'])
@login_required
def cleanup_sessions():
    """Cleanup expired sessions (admin only)."""
    try:
        removed = cleanup_expired_sessions()
        return jsonify({
            'success': True,
            'removed_count': removed,
            'message': f'Cleaned up {removed} expired sessions'
        })
        
    except Exception as e:
        logger.error(f"[ENGENIE_CHAT_API] Cleanup error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@engenie_chat_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'engenie_chat'
    })
