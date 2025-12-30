"""
Production-Ready Celery + WebSocket Setup for Agentic Workflows

Setup Instructions:
1. Install dependencies:
   pip install celery redis flask-socketio eventlet

2. Start Redis:
   redis-server

3. Start Celery worker:
   celery -A celery_production_example.celery worker --loglevel=info --pool=solo

4. Start Flask app:
   python celery_production_example.py

5. Open frontend and test
"""

from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit, join_room
from celery import Celery, Task
import json
import time
from datetime import datetime
import os

# ===========================
# Configuration
# ===========================

class Config:
    CELERY_BROKER_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    CELERY_RESULT_BACKEND = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    CELERY_TASK_SERIALIZER = 'json'
    CELERY_RESULT_SERIALIZER = 'json'
    CELERY_ACCEPT_CONTENT = ['json']
    CELERY_TIMEZONE = 'UTC'
    CELERY_ENABLE_UTC = True
    SECRET_KEY = 'your-secret-key-change-in-production'

# ===========================
# Flask App Setup
# ===========================

app = Flask(__name__)
app.config.from_object(Config)

# Socket.IO for real-time updates
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    message_queue=Config.CELERY_BROKER_URL,  # Use Redis for multi-worker support
    async_mode='eventlet'
)

# ===========================
# Celery Setup
# ===========================

def make_celery(app):
    celery = Celery(
        app.import_name,
        broker=app.config['CELERY_BROKER_URL'],
        backend=app.config['CELERY_RESULT_BACKEND']
    )
    celery.conf.update(app.config)

    class ContextTask(Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery

celery = make_celery(app)

# ===========================
# Celery Task with Progress Updates
# ===========================

@celery.task(bind=True, name='agentic_workflow')
def agentic_workflow_task(self, params, task_id):
    """
    Agentic workflow that emits progress via WebSocket

    Args:
        self: Celery task instance
        params: Workflow parameters
        task_id: Unique task identifier
    """

    def emit_progress(step, message, progress, data=None):
        """Emit progress update via WebSocket"""
        update = {
            'task_id': task_id,
            'step': step,
            'message': message,
            'progress': progress,
            'timestamp': datetime.utcnow().isoformat(),
            'data': data
        }

        # Update task state (for polling fallback)
        self.update_state(
            state='PROGRESS',
            meta=update
        )

        # Emit via WebSocket (real-time)
        socketio.emit('task_progress', update, room=task_id, namespace='/')

    try:
        # Step 1: Initialize
        emit_progress(
            'initialize',
            'Initializing agentic workflow...',
            10
        )
        time.sleep(2)

        # Step 2: Data Collection
        emit_progress(
            'data_collection',
            'Collecting data from sources...',
            25,
            {'sources': ['API', 'Database', 'Cache']}
        )
        time.sleep(3)

        # Step 3: LLM Agent 1 - Analysis
        emit_progress(
            'llm_analysis',
            'Running analysis agent...',
            40,
            {'agent': 'analyzer', 'model': 'gpt-4'}
        )
        time.sleep(4)

        # Step 4: LLM Agent 2 - Planning
        emit_progress(
            'llm_planning',
            'Running planning agent...',
            60,
            {'agent': 'planner', 'steps_generated': 5}
        )
        time.sleep(3)

        # Step 5: LLM Agent 3 - Execution
        emit_progress(
            'llm_execution',
            'Executing agent chain...',
            75,
            {'agent': 'executor', 'tasks_completed': 3}
        )
        time.sleep(4)

        # Step 6: Validation
        emit_progress(
            'validation',
            'Validating results...',
            90
        )
        time.sleep(2)

        # Final Result
        result = {
            'status': 'success',
            'output': {
                'recommendations': [
                    'Recommendation 1: Optimize data pipeline',
                    'Recommendation 2: Implement caching',
                    'Recommendation 3: Add monitoring'
                ],
                'confidence': 0.95,
                'execution_time': 18,
                'params': params
            },
            'metadata': {
                'agents_used': 3,
                'total_tokens': 5420,
                'cost': 0.32
            }
        }

        emit_progress(
            'complete',
            'Workflow completed successfully!',
            100,
            result
        )

        return result

    except Exception as e:
        error_data = {
            'error': str(e),
            'task_id': task_id
        }

        emit_progress(
            'error',
            f'Error: {str(e)}',
            0,
            error_data
        )

        raise

# ===========================
# Flask Routes
# ===========================

@app.route('/api/workflow/start', methods=['POST'])
def start_workflow():
    """
    Start an agentic workflow

    Returns task ID for tracking
    """
    params = request.json
    task_id = f"task_{int(time.time() * 1000)}"

    # Start Celery task
    task = agentic_workflow_task.apply_async(
        args=[params, task_id],
        task_id=task_id
    )

    return jsonify({
        'task_id': task.id,
        'status': 'started',
        'message': 'Workflow started successfully'
    })


@app.route('/api/workflow/status/<task_id>')
def get_workflow_status(task_id):
    """
    Get workflow status (polling fallback)

    Useful for clients that can't use WebSocket
    """
    task = celery.AsyncResult(task_id)

    response = {
        'task_id': task_id,
        'status': task.state,
        'current': 0,
        'total': 100,
        'result': None
    }

    if task.state == 'PENDING':
        response['message'] = 'Task is waiting to start...'
    elif task.state == 'PROGRESS':
        response.update(task.info)
    elif task.state == 'SUCCESS':
        response['result'] = task.result
        response['message'] = 'Task completed successfully'
        response['progress'] = 100
    elif task.state == 'FAILURE':
        response['message'] = str(task.info)
        response['error'] = str(task.info)

    return jsonify(response)


@app.route('/api/workflow/cancel/<task_id>', methods=['POST'])
def cancel_workflow(task_id):
    """
    Cancel a running workflow
    """
    celery.control.revoke(task_id, terminate=True)

    return jsonify({
        'task_id': task_id,
        'status': 'cancelled',
        'message': 'Workflow cancelled successfully'
    })


@app.route('/api/workflow/list')
def list_workflows():
    """
    List all active workflows

    Production: Use Celery Flower or custom tracking
    """
    # This is a simplified version
    # In production, track tasks in Redis/Database
    inspect = celery.control.inspect()
    active = inspect.active()

    tasks = []
    if active:
        for worker, task_list in active.items():
            tasks.extend(task_list)

    return jsonify({
        'active_tasks': len(tasks),
        'tasks': tasks
    })


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'celery': 'connected',
        'redis': 'connected'
    })

# ===========================
# WebSocket Events
# ===========================

@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    print(f'Client connected: {request.sid}')
    emit('connected', {'message': 'Connected to workflow server'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection"""
    print(f'Client disconnected: {request.sid}')


@socketio.on('subscribe_task')
def handle_subscribe(data):
    """
    Subscribe to task updates

    Client joins a room for the specific task ID
    """
    task_id = data.get('task_id')
    if task_id:
        join_room(task_id)
        emit('subscribed', {
            'task_id': task_id,
            'message': f'Subscribed to task {task_id}'
        })

        # Send current status if available
        task = celery.AsyncResult(task_id)
        if task.state == 'PROGRESS':
            emit('task_progress', task.info, room=task_id)

# ===========================
# CORS Support
# ===========================

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS')
    return response

# ===========================
# Main Entry Point
# ===========================

if __name__ == '__main__':
    print("=" * 80)
    print("PRODUCTION AGENTIC WORKFLOW SERVER")
    print("=" * 80)
    print("\nSetup Instructions:")
    print("1. Start Redis: redis-server")
    print("2. Start Celery worker:")
    print("   celery -A celery_production_example.celery worker --loglevel=info --pool=solo")
    print("3. This Flask app will start automatically")
    print("\nEndpoints:")
    print("  POST /api/workflow/start       - Start workflow")
    print("  GET  /api/workflow/status/:id  - Get status (polling)")
    print("  POST /api/workflow/cancel/:id  - Cancel workflow")
    print("  GET  /api/workflow/list        - List active workflows")
    print("  GET  /health                   - Health check")
    print("\nWebSocket:")
    print("  ws://localhost:5002")
    print("  Event: 'subscribe_task' - Subscribe to task updates")
    print("  Event: 'task_progress' - Receive progress updates")
    print("\n" + "=" * 80)

    # Use eventlet for WebSocket support
    socketio.run(
        app,
        debug=True,
        host='0.0.0.0',
        port=5002,
        use_reloader=False  # Disable reloader with Celery
    )
