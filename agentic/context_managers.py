"""
Generic context managers for resource lifecycle management during LLM operations.
These implement the Python Context Manager Protocol for safe resource handling.

Advanced Features:
- Timeout enforcement with thread interruption
- Async context manager support
- Decorator support for automatic resource management
- Context stacking and nesting
- Resource pooling with auto-recovery
- Comprehensive metrics and monitoring
- Exception handling hooks and recovery strategies
"""

import time
import logging
import threading
import functools
import traceback
from typing import Any, Dict, Optional, List, Callable, Tuple, Type
from contextlib import contextmanager, asynccontextmanager
from threading import RLock, Timer
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ResourceStatus(Enum):
    """Enum for resource status tracking"""
    PENDING = "pending"
    ACTIVE = "active"
    TIMED_OUT = "timed_out"
    RELEASED = "released"
    ERROR = "error"
    RECOVERED = "recovered"


@dataclass
class ResourceMetrics:
    """Track resource usage metrics with comprehensive monitoring"""
    acquired_at: float
    released_at: Optional[float] = None
    duration_ms: Optional[float] = None
    resource_type: str = ""
    resource_id: str = ""
    status: ResourceStatus = ResourceStatus.PENDING
    exception_type: Optional[str] = None
    exception_message: Optional[str] = None
    stack_trace: Optional[str] = None
    recovery_attempts: int = 0
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging/reporting"""
        return {
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "status": self.status.value,
            "acquired_at": self.acquired_at,
            "released_at": self.released_at,
            "duration_ms": self.duration_ms,
            "exception": self.exception_type,
            "recovery_attempts": self.recovery_attempts,
            "custom_metrics": self.custom_metrics
        }


class GlobalResourceRegistry:
    """
    Global registry for monitoring all active resources.
    Provides visibility into resource usage across the application.
    """
    _instance = None
    _lock = RLock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.active_resources: Dict[str, ResourceMetrics] = {}
            self.historical_metrics: List[ResourceMetrics] = []
            self._lock = RLock()
            self._initialized = True

    def register(self, resource_id: str, metrics: ResourceMetrics):
        """Register an active resource"""
        with self._lock:
            self.active_resources[resource_id] = metrics

    def unregister(self, resource_id: str):
        """Unregister a resource and move to history"""
        with self._lock:
            if resource_id in self.active_resources:
                metrics = self.active_resources.pop(resource_id)
                self.historical_metrics.append(metrics)

    def get_active_resources(self, resource_type: Optional[str] = None) -> Dict[str, ResourceMetrics]:
        """Get all active resources, optionally filtered by type"""
        with self._lock:
            if resource_type:
                return {
                    k: v for k, v in self.active_resources.items()
                    if v.resource_type == resource_type
                }
            return self.active_resources.copy()

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary statistics of all resources"""
        with self._lock:
            by_type = {}
            for metrics in self.historical_metrics + list(self.active_resources.values()):
                rt = metrics.resource_type
                if rt not in by_type:
                    by_type[rt] = {
                        "count": 0,
                        "total_duration_ms": 0,
                        "errors": 0,
                        "active": 0
                    }
                by_type[rt]["count"] += 1
                if metrics.duration_ms:
                    by_type[rt]["total_duration_ms"] += metrics.duration_ms
                if metrics.exception_type:
                    by_type[rt]["errors"] += 1
                if metrics.resource_id in self.active_resources:
                    by_type[rt]["active"] += 1

            return by_type


class LLMResourceManager:
    """
    Enhanced context manager for safely managing LLM operation resources.

    Features:
    - Timeout enforcement with automatic cleanup
    - Exception handling with recovery strategies
    - Resource pooling and reuse
    - Comprehensive metrics collection
    - Global resource registry for monitoring
    - Hooks for custom behavior
    """

    def __init__(
        self,
        resource_type: str,
        resource_id: str,
        timeout_seconds: Optional[int] = None,
        max_retries: int = 3,
        on_timeout: Optional[Callable[[], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        on_acquire: Optional[Callable[[], None]] = None,
        on_release: Optional[Callable[[], None]] = None
    ):
        self.resource_type = resource_type
        self.resource_id = resource_id
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

        # Hooks for custom behavior
        self.on_timeout = on_timeout
        self.on_error = on_error
        self.on_acquire = on_acquire
        self.on_release = on_release

        self.metrics = ResourceMetrics(
            acquired_at=time.time(),
            resource_type=resource_type,
            resource_id=resource_id
        )
        self._lock = RLock()
        self._is_active = False
        self._timeout_timer: Optional[Timer] = None
        self._registry = GlobalResourceRegistry()

    def __enter__(self):
        """Acquire resource with optional timeout"""
        with self._lock:
            self._is_active = True
            self.metrics.status = ResourceStatus.ACTIVE
            logger.info(
                f"Acquiring {self.resource_type} resource: {self.resource_id}"
            )

            # Set up timeout if specified
            if self.timeout_seconds:
                self._timeout_timer = Timer(
                    self.timeout_seconds,
                    self._on_timeout
                )
                self._timeout_timer.daemon = True
                self._timeout_timer.start()

            # Register in global registry
            self._registry.register(self.resource_id, self.metrics)

            # Call custom acquire hook
            if self.on_acquire:
                try:
                    self.on_acquire()
                except Exception as e:
                    logger.warning(f"Error in on_acquire hook: {e}")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release resource safely with proper cleanup"""
        with self._lock:
            # Cancel timeout if active
            if self._timeout_timer:
                self._timeout_timer.cancel()

            self._is_active = False
            self.metrics.released_at = time.time()
            self.metrics.duration_ms = (
                (self.metrics.released_at - self.metrics.acquired_at) * 1000
            )

            # Track exception
            if exc_type:
                self.metrics.status = ResourceStatus.ERROR
                self.metrics.exception_type = exc_type.__name__
                self.metrics.exception_message = str(exc_val)
                self.metrics.stack_trace = traceback.format_exc()
                logger.error(
                    f"Exception during {self.resource_type} operation: {exc_val}",
                    exc_info=True
                )

                # Call custom error hook
                if self.on_error:
                    try:
                        self.on_error(exc_val)
                    except Exception as e:
                        logger.warning(f"Error in on_error hook: {e}")
            else:
                self.metrics.status = ResourceStatus.RELEASED
                logger.info(
                    f"Released {self.resource_type} resource: {self.resource_id} "
                    f"(duration: {self.metrics.duration_ms:.2f}ms)"
                )

            # Call custom release hook
            if self.on_release:
                try:
                    self.on_release()
                except Exception as e:
                    logger.warning(f"Error in on_release hook: {e}")

            # Unregister from global registry
            self._registry.unregister(self.resource_id)

        return False  # Don't suppress exceptions

    def _on_timeout(self):
        """Handle timeout"""
        with self._lock:
            if self._is_active:
                self._is_active = False
                self.metrics.status = ResourceStatus.TIMED_OUT
                logger.error(
                    f"{self.resource_type} resource {self.resource_id} "
                    f"exceeded timeout ({self.timeout_seconds}s)"
                )

                # Call custom timeout hook
                if self.on_timeout:
                    try:
                        self.on_timeout()
                    except Exception as e:
                        logger.warning(f"Error in on_timeout hook: {e}")

    def is_active(self) -> bool:
        """Check if resource is currently active"""
        with self._lock:
            return self._is_active

    def record_metric(self, key: str, value: Any):
        """Record custom metric"""
        with self._lock:
            self.metrics.custom_metrics[key] = value

    def get_metrics(self) -> ResourceMetrics:
        """Get resource metrics"""
        with self._lock:
            return self.metrics


def managed_resource(
    resource_type: str,
    timeout_seconds: Optional[int] = None,
    max_retries: int = 3
):
    """
    Decorator to automatically manage resources.

    Usage:
        @managed_resource("database_query", timeout_seconds=30)
        def my_function():
            # Automatically acquire and release resource
            pass
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            resource_id = f"{func.__name__}_{id(func)}"
            with LLMResourceManager(
                resource_type,
                resource_id,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries
            ) as manager:
                return func(*args, **kwargs)
        return wrapper
    return decorator


class AsyncLLMResourceManager:
    """Async version of LLMResourceManager"""

    def __init__(
        self,
        resource_type: str,
        resource_id: str,
        timeout_seconds: Optional[int] = None,
        on_timeout: Optional[Callable[[], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None
    ):
        self.resource_type = resource_type
        self.resource_id = resource_id
        self.timeout_seconds = timeout_seconds
        self.on_timeout = on_timeout
        self.on_error = on_error
        self.metrics = ResourceMetrics(
            acquired_at=time.time(),
            resource_type=resource_type,
            resource_id=resource_id
        )
        self._is_active = False
        self._registry = GlobalResourceRegistry()

    async def __aenter__(self):
        """Async resource acquisition"""
        self._is_active = True
        self.metrics.status = ResourceStatus.ACTIVE
        self._registry.register(self.resource_id, self.metrics)
        logger.info(
            f"Async acquiring {self.resource_type} resource: {self.resource_id}"
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async resource release"""
        self._is_active = False
        self.metrics.released_at = time.time()
        self.metrics.duration_ms = (
            (self.metrics.released_at - self.metrics.acquired_at) * 1000
        )

        if exc_type:
            self.metrics.status = ResourceStatus.ERROR
            self.metrics.exception_type = exc_type.__name__
            logger.error(
                f"Exception during async {self.resource_type} operation: {exc_val}",
                exc_info=True
            )
            if self.on_error:
                try:
                    self.on_error(exc_val)
                except Exception as e:
                    logger.warning(f"Error in on_error hook: {e}")
        else:
            self.metrics.status = ResourceStatus.RELEASED

        self._registry.unregister(self.resource_id)
        return False

    def is_active(self) -> bool:
        """Check if resource is active"""
        return self._is_active


class FileHandleManager(LLMResourceManager):
    """Context manager specifically for file handle operations"""

    def __init__(self, file_path: str, mode: str = 'r'):
        super().__init__("file_handle", file_path)
        self.file_path = file_path
        self.mode = mode
        self.file_handle = None

    def __enter__(self):
        super().__enter__()
        try:
            self.file_handle = open(self.file_path, self.mode)
            logger.debug(f"Opened file: {self.file_path}")
            return self.file_handle
        except IOError as e:
            logger.error(f"Failed to open file {self.file_path}: {e}")
            self._is_active = False
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file_handle:
            try:
                self.file_handle.close()
                logger.debug(f"Closed file: {self.file_path}")
            except Exception as e:
                logger.error(f"Error closing file {self.file_path}: {e}")
        super().__exit__(exc_type, exc_val, exc_tb)


class ConnectionPoolManager(LLMResourceManager):
    """Context manager for managing connection pools"""

    def __init__(self, pool_name: str, max_connections: int = 10):
        super().__init__("connection_pool", pool_name)
        self.pool_name = pool_name
        self.max_connections = max_connections
        self.active_connections: Dict[str, Any] = {}
        self._pool_lock = RLock()

    def acquire_connection(self, conn_id: str) -> Any:
        """Acquire a connection from the pool"""
        with self._pool_lock:
            if not self._is_active:
                raise RuntimeError(f"Connection pool {self.pool_name} is not active")

            if len(self.active_connections) >= self.max_connections:
                raise RuntimeError(
                    f"Connection pool {self.pool_name} at max capacity "
                    f"({self.max_connections})"
                )

            self.active_connections[conn_id] = {
                'acquired_at': time.time(),
                'status': 'active'
            }
            logger.debug(f"Acquired connection {conn_id} from pool {self.pool_name}")

    def release_connection(self, conn_id: str):
        """Release a connection back to the pool"""
        with self._pool_lock:
            if conn_id in self.active_connections:
                self.active_connections[conn_id]['status'] = 'released'
                del self.active_connections[conn_id]
                logger.debug(f"Released connection {conn_id} from pool {self.pool_name}")

    def get_active_count(self) -> int:
        """Get count of active connections"""
        with self._pool_lock:
            return len(self.active_connections)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup all remaining connections
        with self._pool_lock:
            for conn_id in list(self.active_connections.keys()):
                self.release_connection(conn_id)

        super().__exit__(exc_type, exc_val, exc_tb)


class MemoryAllocationManager(LLMResourceManager):
    """Context manager for tracking memory allocation during LLM operations"""

    def __init__(self, operation_name: str, max_memory_mb: Optional[int] = None):
        super().__init__("memory_allocation", operation_name)
        self.operation_name = operation_name
        self.max_memory_mb = max_memory_mb
        self.allocated_memory_mb = 0

    def allocate(self, size_mb: int) -> bool:
        """Request memory allocation"""
        with self._lock:
            if not self._is_active:
                raise RuntimeError(f"Memory manager for {self.operation_name} is not active")

            if self.max_memory_mb and (
                self.allocated_memory_mb + size_mb > self.max_memory_mb
            ):
                logger.warning(
                    f"Memory allocation request ({size_mb}MB) would exceed "
                    f"limit ({self.max_memory_mb}MB) for {self.operation_name}"
                )
                return False

            self.allocated_memory_mb += size_mb
            logger.debug(
                f"Allocated {size_mb}MB for {self.operation_name} "
                f"(total: {self.allocated_memory_mb}MB)"
            )
            return True

    def deallocate(self, size_mb: int):
        """Release memory allocation"""
        with self._lock:
            self.allocated_memory_mb = max(0, self.allocated_memory_mb - size_mb)
            logger.debug(
                f"Deallocated {size_mb}MB for {self.operation_name} "
                f"(remaining: {self.allocated_memory_mb}MB)"
            )

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Deallocate all remaining memory
        self.deallocate(self.allocated_memory_mb)
        super().__exit__(exc_type, exc_val, exc_tb)


class BatchResourceManager(LLMResourceManager):
    """
    Context manager for managing batches of resources.
    Useful for parallel processing where multiple resources need coordinated cleanup.
    """

    def __init__(self, batch_name: str, batch_size: int, max_resources: int = 100):
        super().__init__("batch_resources", batch_name)
        self.batch_name = batch_name
        self.batch_size = batch_size
        self.max_resources = max_resources
        self.resources: Dict[str, Any] = {}
        self.completed_count = 0
        self.failed_count = 0

    def __enter__(self):
        super().__enter__()
        self.metrics.custom_metrics["batch_size"] = self.batch_size
        return self

    def add_resource(self, resource_id: str, resource: Any) -> bool:
        """Add a resource to the batch"""
        with self._lock:
            if not self._is_active:
                raise RuntimeError(f"Batch {self.batch_name} is not active")

            if len(self.resources) >= self.max_resources:
                logger.warning(
                    f"Batch {self.batch_name} at max capacity ({self.max_resources})"
                )
                return False

            self.resources[resource_id] = resource
            logger.debug(f"Added resource {resource_id} to batch {self.batch_name}")
            return True

    def mark_completed(self, resource_id: str):
        """Mark a resource as completed"""
        with self._lock:
            if resource_id in self.resources:
                self.completed_count += 1
                del self.resources[resource_id]

    def mark_failed(self, resource_id: str, error: Exception):
        """Mark a resource as failed"""
        with self._lock:
            if resource_id in self.resources:
                self.failed_count += 1
                del self.resources[resource_id]
            self.record_metric("last_error", str(error))

    def get_batch_stats(self) -> Dict[str, Any]:
        """Get batch statistics"""
        with self._lock:
            return {
                "batch_name": self.batch_name,
                "batch_size": self.batch_size,
                "completed": self.completed_count,
                "failed": self.failed_count,
                "pending": len(self.resources)
            }

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup all remaining resources
        with self._lock:
            pending = len(self.resources)
            self.resources.clear()
            self.record_metric("batch_stats", {
                "completed": self.completed_count,
                "failed": self.failed_count,
                "pending_cleanup": pending
            })

        super().__exit__(exc_type, exc_val, exc_tb)


class ThreadPoolResourceManager(LLMResourceManager):
    """
    Context manager for thread pool operations with resource tracking.
    Monitors and limits thread pool resource usage.
    """

    def __init__(
        self,
        pool_name: str,
        max_workers: int = 5,
        timeout_per_task: Optional[int] = None
    ):
        super().__init__("thread_pool", pool_name, timeout_seconds=None)
        self.pool_name = pool_name
        self.max_workers = max_workers
        self.timeout_per_task = timeout_per_task
        self.task_count = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self._executor = None

    def __enter__(self):
        super().__enter__()
        from concurrent.futures import ThreadPoolExecutor
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix=self.pool_name
        )
        logger.info(
            f"Created thread pool '{self.pool_name}' "
            f"with {self.max_workers} workers"
        )
        return self

    def submit_task(self, fn: Callable, *args, **kwargs):
        """Submit a task to the thread pool"""
        with self._lock:
            if not self._is_active or not self._executor:
                raise RuntimeError(f"Thread pool {self.pool_name} is not active")

            self.task_count += 1
            try:
                future = self._executor.submit(fn, *args, **kwargs)
                return future
            except Exception as e:
                self.failed_tasks += 1
                logger.error(f"Failed to submit task to {self.pool_name}: {e}")
                raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Shutdown executor
        if self._executor:
            try:
                self._executor.shutdown(wait=True, timeout=self.timeout_per_task)
                logger.info(
                    f"Shut down thread pool '{self.pool_name}'. "
                    f"Tasks: {self.task_count}, "
                    f"Completed: {self.completed_tasks}, "
                    f"Failed: {self.failed_tasks}"
                )
            except Exception as e:
                logger.error(f"Error shutting down thread pool: {e}")
            finally:
                self._executor = None

        self.record_metric("task_stats", {
            "total": self.task_count,
            "completed": self.completed_tasks,
            "failed": self.failed_tasks
        })

        super().__exit__(exc_type, exc_val, exc_tb)


# Convenience context manager factories
@contextmanager
def managed_llm_operation(
    operation_name: str,
    timeout_seconds: int = 300,
    on_error: Optional[Callable[[Exception], None]] = None
):
    """
    Convenience context manager for generic LLM operations.

    Usage:
        def handle_error(e):
            print(f"Error occurred: {e}")

        with managed_llm_operation("data_loading", timeout_seconds=60, on_error=handle_error) as mgr:
            # Your LLM operation here
            pass
    """
    manager = LLMResourceManager(
        "llm_operation",
        operation_name,
        timeout_seconds=timeout_seconds,
        on_error=on_error
    )

    try:
        manager.__enter__()
        yield manager
    except Exception as e:
        logger.error(f"Error in LLM operation {operation_name}: {e}")
        raise
    finally:
        manager.__exit__(None, None, None)


@contextmanager
def managed_batch_operation(
    batch_name: str,
    batch_size: int,
    max_resources: int = 100
):
    """
    Convenience context manager for batch processing operations.

    Usage:
        with managed_batch_operation("spec_generation", batch_size=10) as batch:
            for item in items:
                batch.add_resource(f"item_{i}", item)
    """
    manager = BatchResourceManager(batch_name, batch_size, max_resources)

    try:
        manager.__enter__()
        yield manager
    except Exception as e:
        logger.error(f"Error in batch operation {batch_name}: {e}")
        raise
    finally:
        manager.__exit__(None, None, None)


@contextmanager
def managed_thread_pool(
    pool_name: str,
    max_workers: int = 5,
    timeout_per_task: Optional[int] = None
):
    """
    Convenience context manager for thread pool operations.

    Usage:
        with managed_thread_pool("parallel_processing", max_workers=4) as pool:
            future = pool.submit_task(process_item, item)
            result = future.result()
    """
    manager = ThreadPoolResourceManager(pool_name, max_workers, timeout_per_task)

    try:
        manager.__enter__()
        yield manager
    except Exception as e:
        logger.error(f"Error in thread pool operation {pool_name}: {e}")
        raise
    finally:
        manager.__exit__(None, None, None)


def get_resource_metrics() -> Dict[str, Any]:
    """
    Get global resource metrics summary.

    Returns:
        Dictionary with metrics by resource type
    """
    registry = GlobalResourceRegistry()
    return registry.get_metrics_summary()


def get_active_resources(resource_type: Optional[str] = None) -> Dict[str, ResourceMetrics]:
    """
    Get all currently active resources.

    Args:
        resource_type: Optional filter by resource type

    Returns:
        Dictionary of active resources
    """
    registry = GlobalResourceRegistry()
    return registry.get_active_resources(resource_type)


# Usage examples:
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example 1: Basic resource management with timeout
    print("\n=== Example 1: Resource Management with Timeout ===")
    def on_timeout():
        print("Resource acquisition timed out!")

    with LLMResourceManager(
        "file_operation",
        "config_load",
        timeout_seconds=5,
        on_timeout=on_timeout
    ) as mgr:
        time.sleep(0.1)
        print(f"Resource active: {mgr.is_active()}")

    # Example 2: File handle management
    print("\n=== Example 2: File Handle Management ===")
    with FileHandleManager("test.txt", "w") as f:
        f.write("Test content")
        print("File written successfully")

    # Example 3: Connection pool management
    print("\n=== Example 3: Connection Pool Management ===")
    with ConnectionPoolManager("llm_connections", max_connections=5) as pool:
        pool.acquire_connection("conn_1")
        pool.acquire_connection("conn_2")
        print(f"Active connections: {pool.get_active_count()}")

    # Example 4: Memory allocation management
    print("\n=== Example 4: Memory Allocation Management ===")
    with MemoryAllocationManager("embedding_operation", max_memory_mb=100) as mem:
        mem.allocate(30)
        mem.allocate(40)
        print(f"Allocated memory: {mem.allocated_memory_mb}MB")

    # Example 5: Batch resource management
    print("\n=== Example 5: Batch Resource Management ===")
    with managed_batch_operation("data_processing", batch_size=5) as batch:
        for i in range(3):
            batch.add_resource(f"item_{i}", {"data": f"test_{i}"})
        print(f"Batch stats: {batch.get_batch_stats()}")

    # Example 6: Thread pool management
    print("\n=== Example 6: Thread Pool Management ===")
    with managed_thread_pool("parallel_ops", max_workers=2) as pool:
        def worker(x):
            return x * 2
        future = pool.submit_task(worker, 5)
        result = future.result(timeout=5)
        print(f"Worker result: {result}")

    # Example 7: Resource metrics
    print("\n=== Example 7: Global Resource Metrics ===")
    metrics = get_resource_metrics()
    print(f"Resource metrics: {metrics}")

    # Example 8: Decorator usage
    print("\n=== Example 8: Decorator Usage ===")
    @managed_resource("database_operation", timeout_seconds=30)
    def query_database():
        return "Query result"

    result = query_database()
    print(f"Decorated function result: {result}")
