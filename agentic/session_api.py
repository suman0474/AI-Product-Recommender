"""
Session & Instance API Endpoints

Provides REST API endpoints for:
- Session lifecycle management (start, heartbeat, end)
- Instance tracking and monitoring
- Orchestrator statistics

These endpoints are used by the frontend to:
1. Notify backend when sessions start/end
2. Send heartbeats to keep sessions alive
3. Track concurrent workflow instances
4. Monitor system health

Usage:
    Register blueprint in your Flask app:
    from session_api import session_bp
    app.register_blueprint(session_bp)
"""

import logging
from typing import Dict, Any
from flask import Blueprint, request, jsonify, session
from functools import wraps

from .session_orchestrator import get_session_orchestrator
from .instance_manager import get_instance_manager, InstanceStatus
from .orchestrator_utils import (
    get_orchestrator_stats,
    verify_session_for_request,
    get_or_create_instance,
    complete_instance_with_result,
    start_background_cleanup,
    validate_main_thread_id
)

logger = logging.getLogger(__name__)

# Create Blueprint
session_bp = Blueprint('sessions', __name__, url_prefix='/api/agentic/sessions')
instances_bp = Blueprint('instances', __name__, url_prefix='/api/agentic/instances')


# ============================================================================
# DECORATORS
# ============================================================================

def login_required(func):
    """Decorator to require login for endpoints"""
    @wraps(func)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({"success": False, "error": "Unauthorized: Please log in"}), 401
        return func(*args, **kwargs)
    return decorated_function


def handle_errors(f):
    """Decorator for error handling"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Session API Error: {e}", exc_info=True)
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    return decorated_function


def api_response(success: bool, data: Any = None, error: str = None, status_code: int = 200):
    """Create standardized API response"""
    response = {
        "success": success,
        "data": data,
        "error": error
    }
    return jsonify(response), status_code


# ============================================================================
# SESSION ENDPOINTS
# ============================================================================

@session_bp.route('/start', methods=['POST'])
@handle_errors
def start_session():
    """
    Start a new session (called on user login)

    Request Body:
        {
            "user_id": str,          # Required: User ID
            "main_thread_id": str,   # Required: Main thread ID from frontend
            "zone": str,             # Optional: Geographic zone
            "is_saved": bool         # Optional: Whether session is for saved workflow
        }

    Returns:
        {
            "success": bool,
            "data": {
                "session_id": str,
                "main_thread_id": str,
                "created_at": str,
                "message": str
            }
        }
    """
    data = request.get_json()
    if not data:
        return api_response(False, error="Request body is required", status_code=400)

    user_id = data.get('user_id')
    main_thread_id = data.get('main_thread_id')
    zone = data.get('zone', 'default')
    is_saved = data.get('is_saved', False)

    # Validate required fields
    if not user_id:
        return api_response(False, error="user_id is required", status_code=400)

    if not main_thread_id:
        return api_response(False, error="main_thread_id is required", status_code=400)

    # Validate main_thread_id format
    if not validate_main_thread_id(main_thread_id):
        return api_response(
            False,
            error=f"Invalid main_thread_id format: {main_thread_id}",
            status_code=400
        )

    # Create session
    orchestrator = get_session_orchestrator()

    # Check if session already exists
    existing = orchestrator.get_session_context(main_thread_id)
    if existing:
        logger.info(f"[SESSION_API] Session already exists: {main_thread_id}")
        return api_response(True, data={
            "session_id": existing.main_thread_id,
            "main_thread_id": existing.main_thread_id,
            "created_at": existing.created_at.isoformat(),
            "message": "Session already exists"
        })

    # Create new session
    session_ctx = orchestrator.create_session(
        user_id=user_id,
        main_thread_id=main_thread_id,
        is_saved=is_saved,
        zone=zone
    )

    logger.info(f"[SESSION_API] Session started: {main_thread_id} for user {user_id}")

    return api_response(True, data={
        "session_id": session_ctx.main_thread_id,
        "main_thread_id": session_ctx.main_thread_id,
        "created_at": session_ctx.created_at.isoformat(),
        "message": "Session started successfully"
    })


@session_bp.route('/heartbeat', methods=['POST'])
@handle_errors
def heartbeat():
    """
    Send heartbeat to keep session alive (called every 5 minutes)

    Request Body:
        {
            "main_thread_id": str    # Required: Main thread ID
        }

    Returns:
        {
            "success": bool,
            "data": {
                "main_thread_id": str,
                "last_activity": str,
                "message": str
            }
        }
    """
    data = request.get_json()
    if not data:
        return api_response(False, error="Request body is required", status_code=400)

    main_thread_id = data.get('main_thread_id')

    if not main_thread_id:
        return api_response(False, error="main_thread_id is required", status_code=400)

    orchestrator = get_session_orchestrator()
    success = orchestrator.heartbeat(main_thread_id)

    if success:
        session_ctx = orchestrator.get_session_context(main_thread_id)
        return api_response(True, data={
            "main_thread_id": main_thread_id,
            "last_activity": session_ctx.last_activity.isoformat() if session_ctx else None,
            "message": "Heartbeat received"
        })
    else:
        return api_response(
            False,
            error=f"Session not found: {main_thread_id}",
            status_code=404
        )


@session_bp.route('/end', methods=['POST'])
@handle_errors
def end_session():
    """
    End a session (called on user logout)

    Request Body:
        {
            "main_thread_id": str    # Required: Main thread ID
        }

    Returns:
        {
            "success": bool,
            "data": {
                "main_thread_id": str,
                "message": str
            }
        }
    """
    data = request.get_json()
    if not data:
        return api_response(False, error="Request body is required", status_code=400)

    main_thread_id = data.get('main_thread_id')

    if not main_thread_id:
        return api_response(False, error="main_thread_id is required", status_code=400)

    # End session
    orchestrator = get_session_orchestrator()
    session_ctx = orchestrator.end_session(main_thread_id)

    # Also clean up instances for this session
    instance_manager = get_instance_manager()
    instances_cleaned = instance_manager.cleanup_session(main_thread_id)

    if session_ctx:
        logger.info(
            f"[SESSION_API] Session ended: {main_thread_id} "
            f"({instances_cleaned} instances cleaned)"
        )
        return api_response(True, data={
            "main_thread_id": main_thread_id,
            "instances_cleaned": instances_cleaned,
            "message": "Session ended successfully"
        })
    else:
        return api_response(
            False,
            error=f"Session not found: {main_thread_id}",
            status_code=404
        )


@session_bp.route('/stats', methods=['GET'])
@handle_errors
def get_stats():
    """
    Get session and instance statistics (admin endpoint)

    Returns:
        {
            "success": bool,
            "data": {
                "sessions": { ... },
                "instances": { ... },
                "cleanup_task_running": bool
            }
        }
    """
    stats = get_orchestrator_stats()

    return api_response(True, data=stats)


@session_bp.route('/<main_thread_id>', methods=['GET'])
@handle_errors
def get_session(main_thread_id: str):
    """
    Get session details

    Args:
        main_thread_id: Session ID

    Returns:
        {
            "success": bool,
            "data": {
                "session": { ... session details ... },
                "instances": { ... instance summary ... }
            }
        }
    """
    orchestrator = get_session_orchestrator()
    session_ctx = orchestrator.get_session_context(main_thread_id)

    if not session_ctx:
        return api_response(
            False,
            error=f"Session not found: {main_thread_id}",
            status_code=404
        )

    # Get instance summary for this session
    instance_manager = get_instance_manager()
    instance_summary = instance_manager.get_session_summary(main_thread_id)

    return api_response(True, data={
        "session": session_ctx.to_dict(),
        "instances": instance_summary
    })


@session_bp.route('/<main_thread_id>/workflows', methods=['GET'])
@handle_errors
def get_session_workflows(main_thread_id: str):
    """
    Get workflows for a session

    Args:
        main_thread_id: Session ID

    Returns:
        {
            "success": bool,
            "data": {
                "workflows": [ ... workflow details ... ]
            }
        }
    """
    orchestrator = get_session_orchestrator()
    workflows = orchestrator.get_workflows_for_session(main_thread_id)

    return api_response(True, data={
        "workflows": [w.to_dict() for w in workflows]
    })


# ============================================================================
# INSTANCE ENDPOINTS
# ============================================================================

@instances_bp.route('/summary/<main_thread_id>', methods=['GET'])
@handle_errors
def get_instance_summary(main_thread_id: str):
    """
    Get instance summary for a session

    Args:
        main_thread_id: Session ID

    Query Params:
        workflow_type: Optional filter by workflow type
        status: Optional filter by status (created, running, completed, error)

    Returns:
        {
            "success": bool,
            "data": {
                "session_id": str,
                "pools": { ... },
                "total_instances": int
            }
        }
    """
    workflow_type = request.args.get('workflow_type')
    status_filter = request.args.get('status')

    # Convert status string to enum
    status = None
    if status_filter:
        try:
            status = InstanceStatus(status_filter)
        except ValueError:
            return api_response(
                False,
                error=f"Invalid status: {status_filter}",
                status_code=400
            )

    instance_manager = get_instance_manager()

    if workflow_type or status:
        # Get filtered instances
        instances = instance_manager.get_instances_for_session(
            main_thread_id,
            workflow_type=workflow_type,
            status=status
        )
        return api_response(True, data={
            "session_id": main_thread_id,
            "instances": [i.to_dict() for i in instances],
            "total_instances": len(instances)
        })
    else:
        # Get full summary
        summary = instance_manager.get_session_summary(main_thread_id)
        return api_response(True, data=summary)


@instances_bp.route('/<instance_id>', methods=['GET'])
@handle_errors
def get_instance(instance_id: str):
    """
    Get instance details by ID

    Args:
        instance_id: Instance UUID

    Returns:
        {
            "success": bool,
            "data": { ... instance details ... }
        }
    """
    instance_manager = get_instance_manager()
    instance = instance_manager.get_instance_by_id(instance_id)

    if not instance:
        return api_response(
            False,
            error=f"Instance not found: {instance_id}",
            status_code=404
        )

    return api_response(True, data=instance.to_dict())


@instances_bp.route('/by-trigger', methods=['POST'])
@handle_errors
def get_instance_by_trigger():
    """
    Get instance by trigger source (for deduplication check)

    Request Body:
        {
            "session_id": str,           # Required: Main thread ID
            "workflow_type": str,        # Required: Workflow type
            "parent_workflow_id": str,   # Required: Parent workflow ID
            "trigger_source": str        # Required: Trigger source
        }

    Returns:
        {
            "success": bool,
            "data": {
                "exists": bool,
                "instance": { ... } or null
            }
        }
    """
    data = request.get_json()
    if not data:
        return api_response(False, error="Request body is required", status_code=400)

    session_id = data.get('session_id')
    workflow_type = data.get('workflow_type')
    parent_workflow_id = data.get('parent_workflow_id')
    trigger_source = data.get('trigger_source')

    # Validate required fields
    if not all([session_id, workflow_type, parent_workflow_id, trigger_source]):
        return api_response(
            False,
            error="session_id, workflow_type, parent_workflow_id, and trigger_source are required",
            status_code=400
        )

    instance_manager = get_instance_manager()
    instance = instance_manager.get_instance_by_trigger(
        session_id, workflow_type, parent_workflow_id, trigger_source
    )

    return api_response(True, data={
        "exists": instance is not None,
        "instance": instance.to_dict() if instance else None
    })


@instances_bp.route('/stats', methods=['GET'])
@handle_errors
def get_instance_stats():
    """
    Get instance manager statistics

    Returns:
        {
            "success": bool,
            "data": { ... stats ... }
        }
    """
    instance_manager = get_instance_manager()
    stats = instance_manager.get_stats()

    return api_response(True, data=stats)


@instances_bp.route('/active/<main_thread_id>', methods=['GET'])
@handle_errors
def get_active_instances(main_thread_id: str):
    """
    Get all active (created or running) instances for a session

    Args:
        main_thread_id: Session ID

    Returns:
        {
            "success": bool,
            "data": {
                "instances": [ ... ],
                "count": int
            }
        }
    """
    instance_manager = get_instance_manager()
    instances = instance_manager.get_active_instances_for_session(main_thread_id)

    return api_response(True, data={
        "instances": [i.to_dict() for i in instances],
        "count": len(instances)
    })


# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize_session_api():
    """
    Initialize session API and start background tasks

    Call this during Flask app initialization:
        from session_api import initialize_session_api
        initialize_session_api()
    """
    # Start background cleanup tasks
    start_background_cleanup(
        session_cleanup_interval=60,      # 1 minute
        instance_cleanup_interval=300,    # 5 minutes
        session_ttl_minutes=30,
        workflow_ttl_minutes=60
    )

    logger.info("[SESSION_API] Session API initialized with background cleanup tasks")


def register_session_blueprints(app):
    """
    Register session and instance blueprints with Flask app

    Args:
        app: Flask application instance
    """
    app.register_blueprint(session_bp)
    app.register_blueprint(instances_bp)

    # Initialize background tasks
    initialize_session_api()

    logger.info("[SESSION_API] Registered session and instance blueprints")
