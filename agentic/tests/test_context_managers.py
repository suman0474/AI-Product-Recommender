"""Unit tests for context managers."""
import unittest
import time
import threading
from agentic.context_managers import (
    LLMResourceManager,
    GlobalResourceRegistry,
    ThreadPoolResourceManager,
    managed_thread_pool,
    get_resource_metrics
)

class TestLLMResourceManager(unittest.TestCase):
    def test_basic_lifecycle(self):
        """Test enter/exit lifecycle."""
        with LLMResourceManager("test_llm", "test_id") as mgr:
            self.assertTrue(mgr.is_active())
        self.assertFalse(mgr.is_active())
        # Check if it was unregistered
        registry = GlobalResourceRegistry()
        active = registry.get_active_resources()
        self.assertNotIn("test_id", active)
    
    def test_timeout_callback(self):
        """Test timeout triggers callback."""
        timeout_called = [False]
        def on_timeout(): timeout_called[0] = True
        
        # Use a very short timeout for testing
        with LLMResourceManager("test", "id", timeout_seconds=1, on_timeout=on_timeout):
            time.sleep(1.5)  # Exceed timeout
        
        self.assertTrue(timeout_called[0])
    
    def test_metrics_collection(self):
        """Test metrics are collected."""
        with LLMResourceManager("test", "id") as mgr:
            mgr.record_metric("custom_key", "custom_value")
        
        metrics = mgr.get_metrics()
        self.assertIsNotNone(metrics.duration_ms)
        self.assertEqual(metrics.custom_metrics["custom_key"], "custom_value")

class TestGlobalResourceRegistry(unittest.TestCase):
    def test_singleton_pattern(self):
        """Test registry is singleton."""
        r1 = GlobalResourceRegistry()
        r2 = GlobalResourceRegistry()
        self.assertIs(r1, r2)
    
    def test_register_unregister(self):
        """Test resource registration lifecycle."""
        registry = GlobalResourceRegistry()
        from agentic.context_managers import ResourceMetrics
        
        metrics = ResourceMetrics(acquired_at=time.time(), resource_type="test")
        registry.register("test_id_reg", metrics)
        self.assertIn("test_id_reg", registry.get_active_resources())
        
        registry.unregister("test_id_reg")
        self.assertNotIn("test_id_reg", registry.get_active_resources())

class TestThreadPoolResourceManager(unittest.TestCase):
    def test_thread_pool_execution(self):
        """Test basic thread pool execution."""
        with ThreadPoolResourceManager("test_pool", max_workers=2) as pool:
            future = pool.submit_task(lambda x: x*2, 5)
            result = future.result(timeout=5)
            self.assertEqual(result, 10)
            self.assertEqual(pool.task_count, 1)
            
    def test_managed_context_manager(self):
        """Test convenience context manager."""
        with managed_thread_pool("test_managed", max_workers=2) as pool:
            future = pool.submit_task(lambda x: x*2, 10)
            result = future.result(timeout=5)
            self.assertEqual(result, 20)

if __name__ == '__main__':
    unittest.main()
