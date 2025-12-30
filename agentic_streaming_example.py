"""
Example: Streaming Agentic Workflow with Server-Sent Events (SSE)
Simple approach that works with existing Flask app
"""

from flask import Flask, Response, request, jsonify, stream_with_context
import json
import time
import queue
import threading
from datetime import datetime

app = Flask(__name__)

# Simulate your agentic workflow
def agentic_workflow_with_streaming(params, progress_callback):
    """
    Your agentic workflow that emits progress updates

    Args:
        params: Input parameters
        progress_callback: Function to call with progress updates
    """

    try:
        # Step 1: Initialize
        progress_callback({
            'step': 'initialize',
            'message': 'Initializing agentic workflow...',
            'progress': 10,
            'timestamp': datetime.now().isoformat()
        })
        time.sleep(2)  # Simulate work

        # Step 2: Data gathering (your LLM calls, API calls, etc.)
        progress_callback({
            'step': 'data_gathering',
            'message': 'Gathering data from sources...',
            'progress': 30,
            'timestamp': datetime.now().isoformat()
        })
        time.sleep(3)

        # Step 3: LLM Processing
        progress_callback({
            'step': 'llm_processing',
            'message': 'Processing with LLM agents...',
            'progress': 50,
            'data': {
                'agent': 'analyst',
                'status': 'analyzing requirements'
            },
            'timestamp': datetime.now().isoformat()
        })
        time.sleep(4)

        # Step 4: Chain execution
        progress_callback({
            'step': 'chain_execution',
            'message': 'Executing agent chain...',
            'progress': 70,
            'data': {
                'chains_completed': 2,
                'chains_total': 3
            },
            'timestamp': datetime.now().isoformat()
        })
        time.sleep(3)

        # Step 5: Finalization
        progress_callback({
            'step': 'finalization',
            'message': 'Finalizing results...',
            'progress': 90,
            'timestamp': datetime.now().isoformat()
        })
        time.sleep(2)

        # Complete
        result = {
            'status': 'success',
            'output': {
                'recommendations': ['Item 1', 'Item 2', 'Item 3'],
                'confidence': 0.95,
                'metadata': params
            }
        }

        progress_callback({
            'step': 'complete',
            'message': 'Workflow completed successfully',
            'progress': 100,
            'result': result,
            'timestamp': datetime.now().isoformat()
        })

        return result

    except Exception as e:
        progress_callback({
            'step': 'error',
            'message': f'Error: {str(e)}',
            'progress': 0,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        })
        raise


# SSE endpoint with streaming
@app.route('/api/workflow/stream', methods=['POST'])
def stream_workflow():
    """
    Streaming endpoint that returns Server-Sent Events
    Frontend can connect and receive real-time updates
    """
    params = request.json
    progress_queue = queue.Queue()

    def run_workflow():
        """Background thread that runs the workflow"""
        def emit_progress(data):
            # Format as SSE
            sse_data = f"data: {json.dumps(data)}\n\n"
            progress_queue.put(sse_data)

        try:
            # Run your agentic workflow
            agentic_workflow_with_streaming(params, emit_progress)
        except Exception as e:
            emit_progress({
                'step': 'error',
                'message': str(e),
                'error': True
            })
        finally:
            # Signal completion
            progress_queue.put(None)

    # Start workflow in background thread
    workflow_thread = threading.Thread(target=run_workflow, daemon=True)
    workflow_thread.start()

    def event_stream():
        """Generator that yields SSE events"""
        while True:
            message = progress_queue.get()
            if message is None:
                break
            yield message

    return Response(
        stream_with_context(event_stream()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'  # Disable nginx buffering
        }
    )


# Alternative: Traditional polling endpoint
@app.route('/api/workflow/start', methods=['POST'])
def start_workflow_async():
    """
    Start workflow in background, return task ID for polling
    Use this if SSE doesn't work in your environment
    """
    import uuid
    task_id = str(uuid.uuid4())
    params = request.json

    # Store task status in memory (use Redis in production)
    task_status = {
        'id': task_id,
        'status': 'running',
        'progress': 0,
        'result': None
    }

    # In production, use Celery or similar
    # For now, use thread (not recommended for production)
    def run_task():
        def update_progress(data):
            task_status['progress'] = data.get('progress', 0)
            task_status['current_step'] = data.get('step')

            if data.get('step') == 'complete':
                task_status['status'] = 'complete'
                task_status['result'] = data.get('result')

        try:
            agentic_workflow_with_streaming(params, update_progress)
        except Exception as e:
            task_status['status'] = 'failed'
            task_status['error'] = str(e)

    thread = threading.Thread(target=run_task, daemon=True)
    thread.start()

    # Store in global dict (use Redis in production)
    if not hasattr(app, 'tasks'):
        app.tasks = {}
    app.tasks[task_id] = task_status

    return jsonify({
        'task_id': task_id,
        'status': 'started'
    })


@app.route('/api/workflow/status/<task_id>')
def get_workflow_status(task_id):
    """
    Poll endpoint to check workflow status
    """
    if not hasattr(app, 'tasks') or task_id not in app.tasks:
        return jsonify({'error': 'Task not found'}), 404

    return jsonify(app.tasks[task_id])


# CORS support for development
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response


if __name__ == '__main__':
    print("="*80)
    print("AGENTIC WORKFLOW STREAMING DEMO")
    print("="*80)
    print("\nEndpoints:")
    print("  POST /api/workflow/stream     - SSE streaming endpoint")
    print("  POST /api/workflow/start      - Start async workflow (polling)")
    print("  GET  /api/workflow/status/:id - Check workflow status")
    print("\n" + "="*80)

    app.run(debug=True, port=5001, threaded=True)
