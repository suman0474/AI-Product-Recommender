"""Resource monitoring API endpoints."""
from flask import Blueprint, jsonify
from agentic.context_managers import get_resource_metrics, get_active_resources

resource_bp = Blueprint('resources', __name__, url_prefix='/api/resources')

@resource_bp.route('/metrics', methods=['GET'])
def get_metrics():
    """Get global resource metrics summary."""
    return jsonify(get_resource_metrics())

@resource_bp.route('/active', methods=['GET'])
def get_active():
    """Get all currently active resources."""
    resources = get_active_resources()
    # Convert ResourceMetrics objects to dicts
    return jsonify({k: v.to_dict() for k, v in resources.items()})

@resource_bp.route('/health', methods=['GET'])
def health_check():
    """Health check for resource system."""
    metrics = get_resource_metrics()
    
    # Calculate some aggregates
    total_active = 0
    total_errors = 0
    
    for resource_type, data in metrics.items():
        if isinstance(data, dict):
            total_active += data.get("active", 0)
            total_errors += data.get("errors", 0)
            
    return jsonify({
        "status": "healthy",
        "active_resource_types": list(metrics.keys()),
        "total_active_resources": total_active,
        "total_historical_errors": total_errors,
        "system_status": "degraded" if total_errors > 100 else "optimal"
    })
