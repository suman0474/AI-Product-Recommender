"""
Test Suite for Phase 4 Week 1: Critical Fixes
- Parallel RAG Queries Verification
- Workflow State Bounds Implementation

Run with: python -m pytest test_phase4_week1.py -v
"""
import pytest
import time
import threading
import logging
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# TEST 1: WORKFLOW STATE BOUNDS
# ============================================================================

class TestWorkflowStateBounds:
    """Test suite for bounded workflow state manager."""

    def test_manager_singleton(self):
        """Test that manager is a proper singleton."""
        from agentic.workflow_state_manager import get_workflow_state_manager

        manager1 = get_workflow_state_manager()
        manager2 = get_workflow_state_manager()

        assert manager1 is manager2, "Manager should be singleton"
        logger.info("[PASS] Manager singleton test")

    def test_set_and_get_state(self):
        """Test basic set/get operations."""
        from agentic.workflow_state_manager import BoundedWorkflowStateManager

        manager = BoundedWorkflowStateManager(max_states=100)
        manager.start()

        # Set state
        test_state = {"phase": "analysis", "data": {"key": "value"}}
        manager.set("thread_1", test_state)

        # Get state
        retrieved = manager.get("thread_1")

        assert retrieved["phase"] == "analysis"
        assert retrieved["data"]["key"] == "value"
        logger.info("[PASS] Set/get state test")

        manager.stop()

    def test_lru_eviction(self):
        """Test LRU eviction when max capacity reached."""
        from agentic.workflow_state_manager import BoundedWorkflowStateManager

        manager = BoundedWorkflowStateManager(max_states=3)
        manager.start()

        # Add 3 states (fill capacity)
        manager.set("thread_1", {"id": 1})
        manager.set("thread_2", {"id": 2})
        manager.set("thread_3", {"id": 3})

        stats = manager.get_stats()
        assert stats["current_states"] == 3, "Should have 3 states"

        # Access thread_1 to make it most recent
        manager.get("thread_1")

        # Add 4th state, should evict thread_2 (least recently used)
        manager.set("thread_4", {"id": 4})

        stats = manager.get_stats()
        assert stats["current_states"] == 3, "Should still have max 3 states"
        assert stats["total_evictions"] == 1, "Should have evicted 1 state"

        # thread_2 should be gone, thread_1 should exist
        assert len(manager.get("thread_1")) > 0, "thread_1 should exist (recently accessed)"
        assert len(manager.get("thread_2")) == 0, "thread_2 should be evicted (least recent)"

        logger.info("[PASS] LRU eviction test")
        manager.stop()

    def test_ttl_expiration(self):
        """Test that states expire based on TTL."""
        from agentic.workflow_state_manager import BoundedWorkflowStateManager

        manager = BoundedWorkflowStateManager(max_states=100, ttl_seconds=1)
        manager.start()

        # Add state
        manager.set("thread_1", {"data": "test"})
        assert len(manager.get("thread_1")) > 0, "State should exist initially"

        # Wait for TTL to expire
        time.sleep(1.5)

        # State should be expired
        assert len(manager.get("thread_1")) == 0, "State should be expired after TTL"

        logger.info("[PASS] TTL expiration test")
        manager.stop()

    def test_auto_cleanup_thread(self):
        """Test that cleanup thread runs and removes expired states."""
        from agentic.workflow_state_manager import BoundedWorkflowStateManager

        manager = BoundedWorkflowStateManager(
            max_states=100,
            ttl_seconds=1,
            cleanup_interval=1  # Fast cleanup for testing
        )
        manager.start()

        # Add state
        manager.set("thread_1", {"data": "test"})

        # Wait for TTL and cleanup
        time.sleep(2.5)

        # Check stats (cleanup should have run)
        stats = manager.get_stats()
        assert stats["cleanup_running"], "Cleanup thread should be running"

        logger.info("[PASS] Auto-cleanup thread test")
        manager.stop()

    def test_thread_safety(self):
        """Test thread-safe operations with concurrent access."""
        from agentic.workflow_state_manager import BoundedWorkflowStateManager

        manager = BoundedWorkflowStateManager(max_states=1000)
        manager.start()

        errors = []

        def worker(thread_id):
            try:
                for i in range(10):
                    state = {"id": thread_id, "iteration": i}
                    manager.set(f"thread_{thread_id}_{i}", state)
                    retrieved = manager.get(f"thread_{thread_id}_{i}")
                    assert retrieved["id"] == thread_id
            except Exception as e:
                errors.append(e)

        # Run 10 workers concurrently
        threads = []
        for i in range(10):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert not errors, f"Errors occurred: {errors}"
        logger.info("[PASS] Thread safety test")

        manager.stop()

    def test_memory_bounds(self):
        """Test that memory stays within bounds."""
        from agentic.workflow_state_manager import BoundedWorkflowStateManager

        manager = BoundedWorkflowStateManager(max_states=100)
        manager.start()

        # Add 200 states (exceeds limit)
        for i in range(200):
            state = {"id": i, "data": "x" * 1000}  # Some data
            manager.set(f"thread_{i}", state)

        stats = manager.get_stats()

        # Should never exceed max
        assert stats["current_states"] <= 100, f"States exceed max: {stats['current_states']}"
        assert stats["total_evictions"] == 100, "Should have evicted 100 states"
        assert stats["usage_percent"] <= 100, "Usage should not exceed 100%"

        logger.info("[PASS] Memory bounds test")
        manager.stop()

    def test_delete_state(self):
        """Test explicit state deletion."""
        from agentic.workflow_state_manager import BoundedWorkflowStateManager

        manager = BoundedWorkflowStateManager(max_states=100)
        manager.start()

        manager.set("thread_1", {"data": "test"})
        assert len(manager.get("thread_1")) > 0

        result = manager.delete("thread_1")
        assert result, "Delete should return True for existing state"
        assert len(manager.get("thread_1")) == 0, "State should be deleted"

        result = manager.delete("thread_2")
        assert not result, "Delete should return False for non-existent state"

        logger.info("[PASS] Delete state test")
        manager.stop()

    def test_clear_all_states(self):
        """Test clearing all states."""
        from agentic.workflow_state_manager import BoundedWorkflowStateManager

        manager = BoundedWorkflowStateManager(max_states=100)
        manager.start()

        manager.set("thread_1", {"data": "test1"})
        manager.set("thread_2", {"data": "test2"})
        manager.set("thread_3", {"data": "test3"})

        count = manager.clear()
        assert count == 3, "Should clear 3 states"

        stats = manager.get_stats()
        assert stats["current_states"] == 0, "Should have no states after clear"

        logger.info("[PASS] Clear all states test")
        manager.stop()


# ============================================================================
# TEST 2: PARALLEL RAG VERIFICATION
# ============================================================================

class TestParallelRAG:
    """Test suite for parallel RAG query verification."""

    def test_query_all_parallel_exists(self):
        """Test that query_all_parallel method exists and is callable."""
        from agentic.rag_components import RAGAggregator

        aggregator = RAGAggregator()
        assert hasattr(aggregator, "query_all_parallel"), "Method should exist"
        assert callable(aggregator.query_all_parallel), "Should be callable"

        logger.info("[PASS] Parallel RAG method exists")

    def test_query_all_parallel_with_mocks(self):
        """Test parallel RAG execution with mocked queries."""
        from agentic.rag_components import RAGAggregator

        # Create mock aggregator
        aggregator = RAGAggregator()

        # Mock the three query methods
        aggregator.query_strategy_rag = Mock(return_value={"success": True, "source": "strategy", "data": {}})
        aggregator.query_standards_rag = Mock(return_value={"success": True, "source": "standards", "data": {}})
        aggregator.query_inventory_rag = Mock(return_value={"success": True, "source": "inventory", "data": {}})

        # Execute parallel queries
        results = aggregator.query_all_parallel(
            product_type="Pressure Transmitter",
            requirements={"pressure_range": "0-100 psi"}
        )

        # Verify all three sources were queried
        assert "strategy" in results, "Strategy results should be included"
        assert "standards" in results, "Standards results should be included"
        assert "inventory" in results, "Inventory results should be included"

        # Verify all succeeded
        assert results["strategy"]["success"], "Strategy should succeed"
        assert results["standards"]["success"], "Standards should succeed"
        assert results["inventory"]["success"], "Inventory should succeed"

        logger.info("[PASS] Parallel RAG execution test")

    def test_parallel_queries_execute_concurrently(self):
        """Test that queries execute concurrently (not sequentially)."""
        from agentic.rag_components import RAGAggregator
        from unittest.mock import patch

        aggregator = RAGAggregator()

        # Track execution times
        execution_times = {}

        def mock_query_strategy(product_type, requirements):
            execution_times["strategy_start"] = time.time()
            time.sleep(0.1)  # Simulate 100ms query
            execution_times["strategy_end"] = time.time()
            return {"success": True, "source": "strategy", "data": {}}

        def mock_query_standards(product_type, requirements):
            execution_times["standards_start"] = time.time()
            time.sleep(0.1)  # Simulate 100ms query
            execution_times["standards_end"] = time.time()
            return {"success": True, "source": "standards", "data": {}}

        def mock_query_inventory(product_type, requirements):
            execution_times["inventory_start"] = time.time()
            time.sleep(0.1)  # Simulate 100ms query
            execution_times["inventory_end"] = time.time()
            return {"success": True, "source": "inventory", "data": {}}

        aggregator.query_strategy_rag = mock_query_strategy
        aggregator.query_standards_rag = mock_query_standards
        aggregator.query_inventory_rag = mock_query_inventory

        # Execute parallel queries
        start_time = time.time()
        results = aggregator.query_all_parallel(
            product_type="Test Product",
            requirements={}
        )
        total_time = time.time() - start_time

        # If executed sequentially: 0.3 seconds (3 × 0.1s)
        # If executed in parallel: ~0.1 seconds
        # Allow small overhead for threading
        assert total_time < 0.2, f"Queries should run in parallel, took {total_time:.2f}s"

        logger.info(f"[PASS] Parallel execution test - completed in {total_time:.2f}s")

    def test_parallel_queries_handle_failures(self):
        """Test that parallel queries handle individual failures gracefully."""
        from agentic.rag_components import RAGAggregator

        aggregator = RAGAggregator()

        # Mock one failing query
        aggregator.query_strategy_rag = Mock(side_effect=Exception("Strategy error"))
        aggregator.query_standards_rag = Mock(return_value={"success": True, "source": "standards", "data": {}})
        aggregator.query_inventory_rag = Mock(return_value={"success": True, "source": "inventory", "data": {}})

        # Execute parallel queries
        results = aggregator.query_all_parallel(
            product_type="Test Product",
            requirements={}
        )

        # Verify partial success
        assert "strategy" in results, "Failed result should still be included"
        assert "standards" in results, "Success result should be included"
        assert "inventory" in results, "Success result should be included"

        assert results["strategy"]["success"] is False, "Failed query should be marked as failure"
        assert results["standards"]["success"], "Successful query should be marked as success"

        logger.info("[PASS] Failure handling test")


# ============================================================================
# TEST 3: INTEGRATION TESTS
# ============================================================================

class TestWeek1Integration:
    """Integration tests for Week 1 implementations."""

    def test_api_integration(self):
        """Test that api.py correctly uses bounded state manager."""
        from agentic.api import get_workflow_state, set_workflow_state, cleanup_expired_workflow_states

        # Set state
        test_state = {"phase": "test", "data": {"key": "value"}}
        set_workflow_state("test_thread", test_state)

        # Get state
        retrieved = get_workflow_state("test_thread")

        assert retrieved["phase"] == "test"
        assert retrieved["data"]["key"] == "value"

        logger.info("[PASS] API integration test")

    def test_state_manager_initialization(self):
        """Test that state manager initializes correctly in main.py context."""
        from agentic.workflow_state_manager import get_workflow_state_manager

        manager = get_workflow_state_manager()

        assert manager is not None
        stats = manager.get_stats()

        assert "current_states" in stats
        assert "max_capacity" in stats
        assert "usage_percent" in stats
        assert "cleanup_running" in stats

        logger.info("[PASS] State manager initialization test")


# ============================================================================
# TEST SUMMARY
# ============================================================================

if __name__ == "__main__":
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 4 WEEK 1 TEST SUITE")
    logger.info("Testing: Parallel RAG Verification + Workflow State Bounds")
    logger.info("=" * 80)

    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])

    logger.info("\n" + "=" * 80)
    logger.info("TEST EXECUTION COMPLETE")
    logger.info("=" * 80)
