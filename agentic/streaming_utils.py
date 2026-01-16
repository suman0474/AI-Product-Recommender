"""
Streaming utilities for agentic workflows
Provides helper functions for SSE (Server-Sent Events) streaming
"""

import json
import queue
import threading
from functools import wraps
from typing import Callable, Dict, Any, Optional, List
from datetime import datetime
from flask import Response, stream_with_context
import logging

logger = logging.getLogger(__name__)


class ProgressEmitter:
    """
    Helper class to emit progress updates during workflow execution

    Usage:
        emitter = ProgressEmitter()
        emitter.emit('step1', 'Processing...', 20)
        emitter.emit('step2', 'Analyzing...', 50, data={'count': 5})
    """

    def __init__(self, callback: Optional[Callable] = None):
        """
        Initialize progress emitter

        Args:
            callback: Optional callback function to receive progress updates
        """
        self.callback = callback
        self.step_count = 0

    def emit(self, step: str, message: str, progress: int, data: Any = None):
        """
        Emit a progress update

        Args:
            step: Current step identifier
            message: Human-readable status message
            progress: Progress percentage (0-100)
            data: Optional additional data
        """
        self.step_count += 1

        update = {
            'step': step,
            'message': message,
            'progress': progress,
            'step_number': self.step_count,
            'timestamp': datetime.utcnow().isoformat(),
        }

        if data is not None:
            update['data'] = data

        if self.callback:
            try:
                self.callback(update)
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")

    def error(self, message: str, error_details: Any = None):
        """Emit an error update"""
        update = {
            'step': 'error',
            'message': message,
            'progress': 0,
            'error': True,
            'timestamp': datetime.utcnow().isoformat(),
        }

        if error_details:
            update['error_details'] = str(error_details)

        if self.callback:
            try:
                self.callback(update)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")


class StreamingWorkflowRunner:
    """
    Manages workflow execution in a background thread with SSE streaming

    Usage:
        runner = StreamingWorkflowRunner(workflow_func, params)
        return runner.create_sse_response()
    """

    def __init__(self, workflow_func: Callable, **workflow_params):
        """
        Initialize streaming workflow runner

        Args:
            workflow_func: The workflow function to execute
            **workflow_params: Parameters to pass to the workflow function
        """
        self.workflow_func = workflow_func
        self.workflow_params = workflow_params
        self.progress_queue = queue.Queue()
        self.workflow_thread = None

    def _emit_to_queue(self, progress_data: Dict[str, Any]):
        """Emit progress data to the queue as SSE formatted message"""
        sse_message = f"data: {json.dumps(progress_data)}\n\n"
        self.progress_queue.put(sse_message)

    def _run_workflow(self):
        """Execute workflow in background thread"""
        try:
            # Add progress_callback to workflow params
            self.workflow_params['progress_callback'] = self._emit_to_queue

            # Execute workflow
            logger.info(f"Starting workflow: {self.workflow_func.__name__}")
            result = self.workflow_func(**self.workflow_params)

            # If workflow didn't emit completion, emit it now
            if result:
                completion_message = {
                    'step': 'complete',
                    'message': 'Workflow completed successfully',
                    'progress': 100,
                    'timestamp': datetime.utcnow().isoformat()
                }
                # Check if result is dict-like and add it
                if isinstance(result, dict):
                    completion_message['result'] = result

                self._emit_to_queue(completion_message)

            logger.info(f"Workflow completed: {self.workflow_func.__name__}")

        except Exception as e:
            logger.error(f"Workflow error in {self.workflow_func.__name__}: {e}")
            error_data = {
                'step': 'error',
                'message': f'Workflow failed: {str(e)}',
                'error': True,
                'error_type': type(e).__name__,
                'timestamp': datetime.utcnow().isoformat()
            }
            self._emit_to_queue(error_data)

        finally:
            # Signal completion
            self.progress_queue.put(None)

    def _event_stream(self):
        """Generator that yields SSE events from the queue"""
        while True:
            message = self.progress_queue.get()
            if message is None:
                # Workflow completed
                break
            yield message

    def start(self):
        """Start the workflow in a background thread"""
        self.workflow_thread = threading.Thread(
            target=self._run_workflow,
            daemon=True
        )
        self.workflow_thread.start()

    def create_sse_response(self) -> Response:
        """
        Create and return Flask SSE response

        Returns:
            Flask Response object with SSE stream
        """
        # Start workflow
        self.start()

        # Return SSE response
        return Response(
            stream_with_context(self._event_stream()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',  # Disable nginx buffering
                'Connection': 'keep-alive'
            }
        )


def create_streaming_endpoint(workflow_func: Callable):
    """
    Decorator to convert a workflow function into a streaming endpoint

    Usage:
        @create_streaming_endpoint
        def my_workflow(user_input, session_id, progress_callback=None):
            emitter = ProgressEmitter(progress_callback)
            emitter.emit('step1', 'Starting...', 10)
            # ... workflow code ...
            return result

    Args:
        workflow_func: Workflow function that accepts progress_callback parameter

    Returns:
        Decorated function that returns SSE Response
    """
    def wrapper(**kwargs):
        runner = StreamingWorkflowRunner(workflow_func, **kwargs)
        return runner.create_sse_response()

    return wrapper


# Convenience functions for common patterns

def emit_start(emitter: ProgressEmitter, workflow_name: str = "workflow"):
    """Emit workflow start message"""
    emitter.emit(
        'initialize',
        f'Starting {workflow_name}...',
        5
    )


def emit_extraction(emitter: ProgressEmitter, product_type: str = None):
    """Emit requirements extraction message"""
    msg = 'Extracting requirements...'
    data = None
    if product_type:
        msg = f'Extracting {product_type} requirements...'
        data = {'product_type': product_type}

    emitter.emit('extract_requirements', msg, 15, data=data)


def emit_search(emitter: ProgressEmitter, vendor_count: int = None):
    """Emit vendor search message"""
    msg = 'Searching for vendors...'
    data = None
    if vendor_count:
        msg = f'Found {vendor_count} potential vendors...'
        data = {'vendor_count': vendor_count}

    emitter.emit('search_vendors', msg, 30, data=data)


def emit_analysis(emitter: ProgressEmitter, current: int = None, total: int = None):
    """Emit analysis progress"""
    msg = 'Analyzing products...'
    data = None
    if current and total:
        msg = f'Analyzing products ({current}/{total})...'
        data = {'current': current, 'total': total}

    emitter.emit('analyze_products', msg, 50, data=data)


def emit_ranking(emitter: ProgressEmitter):
    """Emit ranking message"""
    emitter.emit('rank_results', 'Ranking results...', 70)


def emit_formatting(emitter: ProgressEmitter):
    """Emit formatting message"""
    emitter.emit('format_output', 'Formatting output...', 85)


def emit_complete(emitter: ProgressEmitter, result: Any = None):
    """Emit completion message"""
    emitter.emit(
        'complete',
        'Workflow completed successfully',
        100,
        data=result
    )


# ============================================================================
# STREAMING DECORATOR
# ============================================================================

class ProgressStep:
    """Represents a single progress step in a workflow"""

    def __init__(self, step_id: str, message: str, progress: int, data: Any = None):
        """
        Initialize a progress step.

        Args:
            step_id: Unique identifier for this step (e.g., 'initialize', 'search_vendors')
            message: Human-readable message (e.g., 'Searching vendor database...')
            progress: Progress percentage (0-100)
            data: Optional additional data to include in the event
        """
        self.step_id = step_id
        self.message = message
        self.progress = progress
        self.data = data


def with_streaming(
    progress_steps: List[ProgressStep],
    workflow_name: str = "workflow",
    extract_result_metadata: Callable[[Dict[str, Any]], Dict[str, Any]] = None
):
    """
    Decorator to add streaming progress updates to any workflow function.

    This decorator eliminates code duplication by:
    - Automatically creating ProgressEmitter
    - Emitting progress steps at the right time
    - Handling success/error cases
    - Wrapping exceptions with proper error emission

    Usage:
        @with_streaming(
            progress_steps=[
                ProgressStep('initialize', 'Starting workflow...', 10),
                ProgressStep('process', 'Processing data...', 50),
                ProgressStep('finalize', 'Finalizing results...', 90)
            ],
            workflow_name="solution",
            extract_result_metadata=lambda r: {'product_count': len(r.get('ranked_results', []))}
        )
        def run_solution_workflow(user_input, session_id, progress_callback=None):
            # Original workflow code - decorator handles all streaming
            return result

    Args:
        progress_steps: List of ProgressStep objects defining the workflow stages
        workflow_name: Name of the workflow for logging
        extract_result_metadata: Optional function to extract metadata from result
                                for the completion event

    Returns:
        Decorated function with streaming support
    """
    def decorator(workflow_func: Callable):
        @wraps(workflow_func)
        def wrapper(*args, progress_callback: Optional[Callable] = None, **kwargs):
            # Create emitter
            emitter = ProgressEmitter(progress_callback)

            # Emit initial progress steps (everything except the last one)
            # The last step will be emitted after workflow completion
            for step in progress_steps[:-1]:
                emitter.emit(step.step_id, step.message, step.progress, step.data)

            try:
                # Execute the actual workflow
                logger.info(f"[{workflow_name.upper()}-STREAM] Starting workflow")
                result = workflow_func(*args, **kwargs)

                # Check if result indicates success
                is_success = result.get("success", True) if isinstance(result, dict) else True

                if is_success:
                    # Extract metadata for completion event
                    completion_data = None
                    if extract_result_metadata and isinstance(result, dict):
                        try:
                            completion_data = extract_result_metadata(result)
                        except Exception as e:
                            logger.warning(f"[{workflow_name.upper()}-STREAM] Failed to extract metadata: {e}")

                    # Emit final progress step (completion)
                    final_step = progress_steps[-1]
                    emitter.emit(
                        final_step.step_id,
                        final_step.message,
                        final_step.progress,
                        completion_data or final_step.data
                    )
                else:
                    # Workflow returned error
                    error_message = result.get('error', 'Unknown error') if isinstance(result, dict) else 'Workflow failed'
                    emitter.error(error_message)

                return result

            except Exception as e:
                logger.error(f"[{workflow_name.upper()}-STREAM] Failed: {e}", exc_info=True)
                emitter.error(f'{workflow_name} failed: {str(e)}')
                return {
                    "success": False,
                    "error": str(e)
                }

        return wrapper
    return decorator
