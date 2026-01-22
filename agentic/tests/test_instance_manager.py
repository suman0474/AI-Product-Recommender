"""
Unit tests for WorkflowInstanceManager

Tests:
- Instance creation
- Instance lookup by trigger (deduplication)
- Instance lookup by ID and thread ID
- Instance status updates
- Instance pools and organization
- Concurrent access
- Cleanup functionality
- Statistics

Run with: python -m pytest backend/agentic/tests/test_instance_manager.py -v
"""

import unittest
import threading
import time
from datetime import datetime, timedelta

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from backend.agentic.instance_manager import (
    WorkflowInstanceManager,
    InstanceMetadata,
    InstancePool,
    InstanceStatus,
    InstancePriority,
    get_instance_manager
)


class TestWorkflowInstanceManager(unittest.TestCase):
    """Test WorkflowInstanceManager functionality"""

    def setUp(self):
        """Set up test fixtures"""
        # Reset singleton before each test
        WorkflowInstanceManager.reset_instance()
        self.manager = WorkflowInstanceManager.get_instance()

    def tearDown(self):
        """Clean up after each test"""
        self.manager.clear_all()
        WorkflowInstanceManager.reset_instance()

    # ========================================================================
    # BASIC FUNCTIONALITY TESTS
    # ========================================================================

    def test_singleton_pattern(self):
        """Test that WorkflowInstanceManager follows singleton pattern"""
        manager1 = WorkflowInstanceManager.get_instance()
        manager2 = WorkflowInstanceManager.get_instance()
        self.assertIs(manager1, manager2)

    def test_get_instance_manager_function(self):
        """Test convenience function returns singleton"""
        manager = get_instance_manager()
        self.assertIs(manager, self.manager)

    def test_create_instance(self):
        """Test creating a new instance"""
        instance = self.manager.create_instance(
            session_id="main_user1_abc123_1705920000",
            thread_id="item_hash1_xyz_1705920001",
            workflow_type="product_search",
            parent_workflow_id="iid_ref_abc_1705920000",
            trigger_source="item_1"
        )

        self.assertIsNotNone(instance)
        self.assertIsNotNone(instance.instance_id)
        self.assertEqual(instance.thread_id, "item_hash1_xyz_1705920001")
        self.assertEqual(instance.workflow_type, "product_search")
        self.assertEqual(instance.trigger_source, "item_1")
        self.assertEqual(instance.status, InstanceStatus.CREATED)
        self.assertEqual(instance.request_count, 1)

    def test_create_instance_with_priority(self):
        """Test creating instance with priority"""
        instance = self.manager.create_instance(
            session_id="main_user1_abc123_1705920000",
            thread_id="item_hash1_xyz_1705920001",
            workflow_type="product_search",
            parent_workflow_id="iid_ref_abc_1705920000",
            trigger_source="item_1",
            priority=InstancePriority.HIGH
        )

        self.assertEqual(instance.priority, InstancePriority.HIGH)

    def test_create_instance_with_metadata(self):
        """Test creating instance with metadata"""
        instance = self.manager.create_instance(
            session_id="main_user1_abc123_1705920000",
            thread_id="item_hash1_xyz_1705920001",
            workflow_type="product_search",
            parent_workflow_id="iid_ref_abc_1705920000",
            trigger_source="item_1",
            metadata={"item_name": "pressure sensor"}
        )

        self.assertEqual(instance.metadata["item_name"], "pressure sensor")

    # ========================================================================
    # DEDUPLICATION TESTS (MOST IMPORTANT!)
    # ========================================================================

    def test_get_instance_by_trigger_new(self):
        """Test get_instance_by_trigger returns None for new triggers"""
        result = self.manager.get_instance_by_trigger(
            session_id="main_user1_abc123_1705920000",
            workflow_type="product_search",
            parent_workflow_id="iid_ref_abc_1705920000",
            trigger_source="item_1"
        )

        self.assertIsNone(result)

    def test_get_instance_by_trigger_existing(self):
        """Test get_instance_by_trigger returns existing instance"""
        # Create instance first
        original = self.manager.create_instance(
            session_id="main_user1_abc123_1705920000",
            thread_id="item_hash1_xyz_1705920001",
            workflow_type="product_search",
            parent_workflow_id="iid_ref_abc_1705920000",
            trigger_source="item_1"
        )

        # Now try to get by trigger
        existing = self.manager.get_instance_by_trigger(
            session_id="main_user1_abc123_1705920000",
            workflow_type="product_search",
            parent_workflow_id="iid_ref_abc_1705920000",
            trigger_source="item_1"
        )

        self.assertIsNotNone(existing)
        self.assertEqual(existing.instance_id, original.instance_id)

    def test_deduplication_different_triggers(self):
        """Test that different triggers create different instances"""
        instance1 = self.manager.create_instance(
            session_id="main_user1_abc123_1705920000",
            thread_id="item_hash1_xyz_1705920001",
            workflow_type="product_search",
            parent_workflow_id="iid_ref_abc_1705920000",
            trigger_source="item_1"
        )

        instance2 = self.manager.create_instance(
            session_id="main_user1_abc123_1705920000",
            thread_id="item_hash2_xyz_1705920002",
            workflow_type="product_search",
            parent_workflow_id="iid_ref_abc_1705920000",
            trigger_source="item_2"  # Different trigger
        )

        self.assertNotEqual(instance1.instance_id, instance2.instance_id)

        # Both should be retrievable
        retrieved1 = self.manager.get_instance_by_trigger(
            "main_user1_abc123_1705920000",
            "product_search",
            "iid_ref_abc_1705920000",
            "item_1"
        )
        retrieved2 = self.manager.get_instance_by_trigger(
            "main_user1_abc123_1705920000",
            "product_search",
            "iid_ref_abc_1705920000",
            "item_2"
        )

        self.assertEqual(retrieved1.instance_id, instance1.instance_id)
        self.assertEqual(retrieved2.instance_id, instance2.instance_id)

    def test_deduplication_different_parent_workflows(self):
        """Test that same trigger in different parent workflows are separate"""
        instance1 = self.manager.create_instance(
            session_id="main_user1_abc123_1705920000",
            thread_id="item_hash1_xyz_1705920001",
            workflow_type="product_search",
            parent_workflow_id="iid_ref_abc_1705920000",  # From instrument_identifier
            trigger_source="item_1"
        )

        instance2 = self.manager.create_instance(
            session_id="main_user1_abc123_1705920000",
            thread_id="item_hash2_xyz_1705920002",
            workflow_type="product_search",
            parent_workflow_id="sol_ref_def_1705920000",  # From solution workflow
            trigger_source="item_1"  # Same trigger name, but different parent!
        )

        self.assertNotEqual(instance1.instance_id, instance2.instance_id)

    # ========================================================================
    # INSTANCE LOOKUP TESTS
    # ========================================================================

    def test_get_instance_by_id(self):
        """Test getting instance by UUID"""
        instance = self.manager.create_instance(
            session_id="main_user1_abc123_1705920000",
            thread_id="item_hash1_xyz_1705920001",
            workflow_type="product_search",
            parent_workflow_id="iid_ref_abc_1705920000",
            trigger_source="item_1"
        )

        retrieved = self.manager.get_instance_by_id(instance.instance_id)

        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.instance_id, instance.instance_id)

    def test_get_instance_by_id_nonexistent(self):
        """Test getting non-existent instance by ID"""
        retrieved = self.manager.get_instance_by_id("nonexistent-uuid")
        self.assertIsNone(retrieved)

    def test_get_instance_by_thread_id(self):
        """Test getting instance by LangGraph thread_id"""
        instance = self.manager.create_instance(
            session_id="main_user1_abc123_1705920000",
            thread_id="item_hash1_xyz_1705920001",
            workflow_type="product_search",
            parent_workflow_id="iid_ref_abc_1705920000",
            trigger_source="item_1"
        )

        retrieved = self.manager.get_instance_by_thread_id("item_hash1_xyz_1705920001")

        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.instance_id, instance.instance_id)

    def test_get_instance_by_thread_id_nonexistent(self):
        """Test getting non-existent instance by thread_id"""
        retrieved = self.manager.get_instance_by_thread_id("nonexistent")
        self.assertIsNone(retrieved)

    # ========================================================================
    # STATUS UPDATE TESTS
    # ========================================================================

    def test_set_instance_running(self):
        """Test marking instance as running"""
        instance = self.manager.create_instance(
            session_id="main_user1_abc123_1705920000",
            thread_id="item_hash1_xyz_1705920001",
            workflow_type="product_search",
            parent_workflow_id="iid_ref_abc_1705920000",
            trigger_source="item_1"
        )

        result = self.manager.set_instance_running(instance.instance_id)

        self.assertTrue(result)

        # Verify status changed
        retrieved = self.manager.get_instance_by_id(instance.instance_id)
        self.assertEqual(retrieved.status, InstanceStatus.RUNNING)
        self.assertIsNotNone(retrieved.started_at)

    def test_set_instance_completed(self):
        """Test marking instance as completed"""
        instance = self.manager.create_instance(
            session_id="main_user1_abc123_1705920000",
            thread_id="item_hash1_xyz_1705920001",
            workflow_type="product_search",
            parent_workflow_id="iid_ref_abc_1705920000",
            trigger_source="item_1"
        )

        result_data = {"products": ["product1", "product2"]}
        result = self.manager.set_instance_completed(instance.instance_id, result_data)

        self.assertTrue(result)

        # Verify status changed
        retrieved = self.manager.get_instance_by_id(instance.instance_id)
        self.assertEqual(retrieved.status, InstanceStatus.COMPLETED)
        self.assertIsNotNone(retrieved.completed_at)
        self.assertEqual(retrieved.result, result_data)

    def test_set_instance_error_increments_count(self):
        """Test that errors increment error count"""
        instance = self.manager.create_instance(
            session_id="main_user1_abc123_1705920000",
            thread_id="item_hash1_xyz_1705920001",
            workflow_type="product_search",
            parent_workflow_id="iid_ref_abc_1705920000",
            trigger_source="item_1"
        )

        # First error
        self.manager.set_instance_error(instance.instance_id, "Error 1")
        retrieved = self.manager.get_instance_by_id(instance.instance_id)
        self.assertEqual(retrieved.error_count, 1)
        self.assertNotEqual(retrieved.status, InstanceStatus.ERROR)  # Not yet ERROR

        # Second error
        self.manager.set_instance_error(instance.instance_id, "Error 2")
        retrieved = self.manager.get_instance_by_id(instance.instance_id)
        self.assertEqual(retrieved.error_count, 2)

        # Third error - should transition to ERROR status
        self.manager.set_instance_error(instance.instance_id, "Error 3")
        retrieved = self.manager.get_instance_by_id(instance.instance_id)
        self.assertEqual(retrieved.error_count, 3)
        self.assertEqual(retrieved.status, InstanceStatus.ERROR)

    def test_increment_request_count(self):
        """Test incrementing request count (rerun scenario)"""
        instance = self.manager.create_instance(
            session_id="main_user1_abc123_1705920000",
            thread_id="item_hash1_xyz_1705920001",
            workflow_type="product_search",
            parent_workflow_id="iid_ref_abc_1705920000",
            trigger_source="item_1"
        )

        self.assertEqual(instance.request_count, 1)

        new_count = self.manager.increment_request_count(instance.instance_id)

        self.assertEqual(new_count, 2)

        retrieved = self.manager.get_instance_by_id(instance.instance_id)
        self.assertEqual(retrieved.request_count, 2)

    # ========================================================================
    # POOL ORGANIZATION TESTS
    # ========================================================================

    def test_instances_grouped_by_pool(self):
        """Test that instances are organized in pools by workflow type and parent"""
        # Create instances from instrument_identifier
        self.manager.create_instance(
            session_id="main_user1_abc123_1705920000",
            thread_id="item_hash1_xyz_1705920001",
            workflow_type="product_search",
            parent_workflow_id="iid_ref_abc_1705920000",
            trigger_source="item_1"
        )
        self.manager.create_instance(
            session_id="main_user1_abc123_1705920000",
            thread_id="item_hash2_xyz_1705920002",
            workflow_type="product_search",
            parent_workflow_id="iid_ref_abc_1705920000",
            trigger_source="item_2"
        )

        # Create instances from solution workflow
        self.manager.create_instance(
            session_id="main_user1_abc123_1705920000",
            thread_id="item_hash3_xyz_1705920003",
            workflow_type="product_search",
            parent_workflow_id="sol_ref_def_1705920000",
            trigger_source="sol_req_1"
        )

        # Get instances for specific pool
        iid_instances = self.manager.get_instances_for_pool(
            "main_user1_abc123_1705920000",
            "product_search",
            "iid_ref_abc_1705920000"
        )

        sol_instances = self.manager.get_instances_for_pool(
            "main_user1_abc123_1705920000",
            "product_search",
            "sol_ref_def_1705920000"
        )

        self.assertEqual(len(iid_instances), 2)
        self.assertEqual(len(sol_instances), 1)

    def test_get_instances_for_session(self):
        """Test getting all instances for a session"""
        # Create multiple instances
        for i in range(5):
            self.manager.create_instance(
                session_id="main_user1_abc123_1705920000",
                thread_id=f"item_hash{i}_xyz_170592000{i}",
                workflow_type="product_search",
                parent_workflow_id="iid_ref_abc_1705920000",
                trigger_source=f"item_{i}"
            )

        instances = self.manager.get_instances_for_session("main_user1_abc123_1705920000")

        self.assertEqual(len(instances), 5)

    def test_get_instances_filtered_by_status(self):
        """Test filtering instances by status"""
        # Create instances with different statuses
        instance1 = self.manager.create_instance(
            session_id="main_user1_abc123_1705920000",
            thread_id="item_hash1_xyz_1705920001",
            workflow_type="product_search",
            parent_workflow_id="iid_ref_abc_1705920000",
            trigger_source="item_1"
        )
        instance2 = self.manager.create_instance(
            session_id="main_user1_abc123_1705920000",
            thread_id="item_hash2_xyz_1705920002",
            workflow_type="product_search",
            parent_workflow_id="iid_ref_abc_1705920000",
            trigger_source="item_2"
        )

        # Mark one as completed
        self.manager.set_instance_completed(instance1.instance_id, {})

        # Get only running/created instances
        active = self.manager.get_active_instances_for_session("main_user1_abc123_1705920000")

        self.assertEqual(len(active), 1)
        self.assertEqual(active[0].instance_id, instance2.instance_id)

    # ========================================================================
    # CONCURRENCY TESTS
    # ========================================================================

    def test_concurrent_instance_creation(self):
        """Test that concurrent instance creation doesn't lose instances"""
        created_instances = []
        lock = threading.Lock()

        def create_instance(i):
            instance = self.manager.create_instance(
                session_id="main_user1_abc123_1705920000",
                thread_id=f"item_hash{i}_xyz_170592000{i}",
                workflow_type="product_search",
                parent_workflow_id="iid_ref_abc_1705920000",
                trigger_source=f"item_{i}"
            )
            with lock:
                created_instances.append(instance)

        threads = []
        for i in range(10):
            t = threading.Thread(target=create_instance, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Should have all 10 instances
        instances = self.manager.get_instances_for_session("main_user1_abc123_1705920000")
        self.assertEqual(len(instances), 10)

    def test_concurrent_status_updates(self):
        """Test concurrent status updates don't cause issues"""
        instance = self.manager.create_instance(
            session_id="main_user1_abc123_1705920000",
            thread_id="item_hash1_xyz_1705920001",
            workflow_type="product_search",
            parent_workflow_id="iid_ref_abc_1705920000",
            trigger_source="item_1"
        )

        def update_status():
            for _ in range(10):
                self.manager.increment_request_count(instance.instance_id)

        threads = []
        for _ in range(5):
            t = threading.Thread(target=update_status)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Should have incremented 50 times (5 threads * 10 each) + initial 1
        retrieved = self.manager.get_instance_by_id(instance.instance_id)
        self.assertEqual(retrieved.request_count, 51)

    # ========================================================================
    # CLEANUP TESTS
    # ========================================================================

    def test_cleanup_session(self):
        """Test cleaning up all instances for a session"""
        # Create instances
        for i in range(5):
            self.manager.create_instance(
                session_id="main_user1_abc123_1705920000",
                thread_id=f"item_hash{i}_xyz_170592000{i}",
                workflow_type="product_search",
                parent_workflow_id="iid_ref_abc_1705920000",
                trigger_source=f"item_{i}"
            )

        # Clean up session
        cleaned = self.manager.cleanup_session("main_user1_abc123_1705920000")

        self.assertEqual(cleaned, 5)

        # All instances should be gone
        instances = self.manager.get_instances_for_session("main_user1_abc123_1705920000")
        self.assertEqual(len(instances), 0)

    def test_cleanup_timed_out_instances(self):
        """Test cleaning up timed-out instances"""
        instance = self.manager.create_instance(
            session_id="main_user1_abc123_1705920000",
            thread_id="item_hash1_xyz_1705920001",
            workflow_type="product_search",
            parent_workflow_id="iid_ref_abc_1705920000",
            trigger_source="item_1",
            timeout_seconds=1  # Very short timeout for test
        )

        # Wait for timeout
        time.sleep(1.5)

        # Run cleanup
        cleaned = self.manager.cleanup_all_timed_out()

        self.assertEqual(cleaned, 1)

        # Instance should be gone
        retrieved = self.manager.get_instance_by_id(instance.instance_id)
        self.assertIsNone(retrieved)

    # ========================================================================
    # STATISTICS TESTS
    # ========================================================================

    def test_get_stats(self):
        """Test getting manager statistics"""
        # Create instances with different statuses
        instance1 = self.manager.create_instance(
            session_id="main_user1_abc123_1705920000",
            thread_id="item_hash1_xyz_1705920001",
            workflow_type="product_search",
            parent_workflow_id="iid_ref_abc_1705920000",
            trigger_source="item_1"
        )
        instance2 = self.manager.create_instance(
            session_id="main_user1_abc123_1705920000",
            thread_id="item_hash2_xyz_1705920002",
            workflow_type="product_search",
            parent_workflow_id="iid_ref_abc_1705920000",
            trigger_source="item_2"
        )

        self.manager.set_instance_completed(instance1.instance_id, {})
        self.manager.set_instance_running(instance2.instance_id)

        stats = self.manager.get_stats()

        self.assertEqual(stats["total_instances"], 2)
        self.assertEqual(stats["completed_instances"], 1)
        self.assertEqual(stats["active_instances"], 1)
        self.assertEqual(stats["lifetime_created"], 2)

    def test_get_session_summary(self):
        """Test getting session summary"""
        # Create instances
        for i in range(3):
            self.manager.create_instance(
                session_id="main_user1_abc123_1705920000",
                thread_id=f"item_hash{i}_xyz_170592000{i}",
                workflow_type="product_search",
                parent_workflow_id="iid_ref_abc_1705920000",
                trigger_source=f"item_{i}"
            )

        summary = self.manager.get_session_summary("main_user1_abc123_1705920000")

        self.assertEqual(summary["session_id"], "main_user1_abc123_1705920000")
        self.assertEqual(summary["total_instances"], 3)
        self.assertIn("pools", summary)


class TestInstanceMetadata(unittest.TestCase):
    """Test InstanceMetadata dataclass"""

    def test_instance_metadata_creation(self):
        """Test creating InstanceMetadata"""
        now = datetime.now()
        instance = InstanceMetadata(
            instance_id="uuid1",
            thread_id="item_hash1_xyz_1705920001",
            workflow_type="product_search",
            parent_workflow_id="iid_ref_abc_1705920000",
            parent_workflow_type="instrument_identifier",
            trigger_source="item_1",
            main_thread_id="main_user1_abc123_1705920000"
        )

        self.assertEqual(instance.instance_id, "uuid1")
        self.assertEqual(instance.status, InstanceStatus.CREATED)
        self.assertEqual(instance.request_count, 1)
        self.assertEqual(instance.error_count, 0)
        self.assertGreaterEqual(instance.created_at, now)

    def test_instance_is_active(self):
        """Test checking if instance is active"""
        instance = InstanceMetadata(
            instance_id="uuid1",
            thread_id="item_hash1_xyz_1705920001",
            workflow_type="product_search",
            parent_workflow_id="iid_ref_abc_1705920000",
            parent_workflow_type="instrument_identifier",
            trigger_source="item_1",
            main_thread_id="main_user1_abc123_1705920000"
        )

        # CREATED status should be active
        self.assertTrue(instance.is_active())

        # RUNNING status should be active
        instance.status = InstanceStatus.RUNNING
        self.assertTrue(instance.is_active())

        # COMPLETED status should not be active
        instance.status = InstanceStatus.COMPLETED
        self.assertFalse(instance.is_active())

    def test_instance_can_retry(self):
        """Test checking if instance can retry"""
        instance = InstanceMetadata(
            instance_id="uuid1",
            thread_id="item_hash1_xyz_1705920001",
            workflow_type="product_search",
            parent_workflow_id="iid_ref_abc_1705920000",
            parent_workflow_type="instrument_identifier",
            trigger_source="item_1",
            main_thread_id="main_user1_abc123_1705920000"
        )

        # Can retry with 0 errors
        self.assertTrue(instance.can_retry())

        # Can retry with 1 error
        instance.error_count = 1
        self.assertTrue(instance.can_retry())

        # Can retry with 2 errors
        instance.error_count = 2
        self.assertTrue(instance.can_retry())

        # Cannot retry with 3+ errors (default max)
        instance.error_count = 3
        self.assertFalse(instance.can_retry())

    def test_instance_is_timed_out(self):
        """Test checking if instance has timed out"""
        instance = InstanceMetadata(
            instance_id="uuid1",
            thread_id="item_hash1_xyz_1705920001",
            workflow_type="product_search",
            parent_workflow_id="iid_ref_abc_1705920000",
            parent_workflow_type="instrument_identifier",
            trigger_source="item_1",
            main_thread_id="main_user1_abc123_1705920000",
            timeout_seconds=1  # 1 second timeout
        )

        # Fresh instance not timed out
        self.assertFalse(instance.is_timed_out())

        # Completed instance not timed out
        instance.status = InstanceStatus.COMPLETED
        time.sleep(1.5)
        self.assertFalse(instance.is_timed_out())

        # Active instance past timeout
        instance.status = InstanceStatus.RUNNING
        instance.created_at = datetime.now() - timedelta(seconds=2)
        self.assertTrue(instance.is_timed_out())

    def test_instance_to_dict(self):
        """Test converting instance to dict"""
        instance = InstanceMetadata(
            instance_id="uuid1",
            thread_id="item_hash1_xyz_1705920001",
            workflow_type="product_search",
            parent_workflow_id="iid_ref_abc_1705920000",
            parent_workflow_type="instrument_identifier",
            trigger_source="item_1",
            main_thread_id="main_user1_abc123_1705920000"
        )

        data = instance.to_dict()

        self.assertEqual(data["instance_id"], "uuid1")
        self.assertEqual(data["thread_id"], "item_hash1_xyz_1705920001")
        self.assertEqual(data["workflow_type"], "product_search")
        self.assertEqual(data["trigger_source"], "item_1")
        self.assertEqual(data["status"], "created")
        self.assertIn("created_at", data)


class TestInstancePool(unittest.TestCase):
    """Test InstancePool class"""

    def test_pool_creation(self):
        """Test creating an instance pool"""
        pool = InstancePool(
            workflow_type="product_search",
            parent_workflow_id="iid_ref_abc_1705920000"
        )

        self.assertEqual(pool.workflow_type, "product_search")
        self.assertEqual(pool.parent_workflow_id, "iid_ref_abc_1705920000")
        self.assertEqual(pool.get_instance_count(), 0)

    def test_pool_add_and_get_instance(self):
        """Test adding and getting instances from pool"""
        pool = InstancePool(
            workflow_type="product_search",
            parent_workflow_id="iid_ref_abc_1705920000"
        )

        instance = InstanceMetadata(
            instance_id="uuid1",
            thread_id="item_hash1_xyz_1705920001",
            workflow_type="product_search",
            parent_workflow_id="iid_ref_abc_1705920000",
            parent_workflow_type="instrument_identifier",
            trigger_source="item_1",
            main_thread_id="main_user1_abc123_1705920000"
        )

        pool.add_instance(instance)

        self.assertEqual(pool.get_instance_count(), 1)

        # Get by ID
        retrieved = pool.get_instance("uuid1")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.instance_id, "uuid1")

        # Get by trigger
        retrieved = pool.get_instance_by_trigger("item_1")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.trigger_source, "item_1")

    def test_pool_trigger_index(self):
        """Test that trigger index enables O(1) lookup"""
        pool = InstancePool(
            workflow_type="product_search",
            parent_workflow_id="iid_ref_abc_1705920000"
        )

        # Add multiple instances
        for i in range(100):
            instance = InstanceMetadata(
                instance_id=f"uuid{i}",
                thread_id=f"item_hash{i}_xyz_170592000{i}",
                workflow_type="product_search",
                parent_workflow_id="iid_ref_abc_1705920000",
                parent_workflow_type="instrument_identifier",
                trigger_source=f"item_{i}",
                main_thread_id="main_user1_abc123_1705920000"
            )
            pool.add_instance(instance)

        # Lookup should be fast via trigger index
        retrieved = pool.get_instance_by_trigger("item_50")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.trigger_source, "item_50")

    def test_pool_remove_instance(self):
        """Test removing instance from pool"""
        pool = InstancePool(
            workflow_type="product_search",
            parent_workflow_id="iid_ref_abc_1705920000"
        )

        instance = InstanceMetadata(
            instance_id="uuid1",
            thread_id="item_hash1_xyz_1705920001",
            workflow_type="product_search",
            parent_workflow_id="iid_ref_abc_1705920000",
            parent_workflow_type="instrument_identifier",
            trigger_source="item_1",
            main_thread_id="main_user1_abc123_1705920000"
        )

        pool.add_instance(instance)
        self.assertEqual(pool.get_instance_count(), 1)

        # Remove
        removed = pool.remove_instance("uuid1")
        self.assertIsNotNone(removed)
        self.assertEqual(pool.get_instance_count(), 0)

        # Trigger index should also be cleared
        retrieved = pool.get_instance_by_trigger("item_1")
        self.assertIsNone(retrieved)


if __name__ == "__main__":
    unittest.main()
