"""
Unit tests for SessionOrchestrator

Tests:
- Session creation and retrieval
- Heartbeat functionality
- Session ending
- User index management
- Workflow tracking
- Concurrent access
- Session expiration
- Statistics

Run with: python -m pytest backend/agentic/tests/test_session_orchestrator.py -v
"""

import unittest
import threading
import time
from datetime import datetime, timedelta

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from backend.agentic.session_orchestrator import (
    SessionOrchestrator,
    SessionContext,
    WorkflowContext,
    get_session_orchestrator
)


class TestSessionOrchestrator(unittest.TestCase):
    """Test SessionOrchestrator functionality"""

    def setUp(self):
        """Set up test fixtures"""
        # Reset singleton before each test
        SessionOrchestrator.reset_instance()
        self.orchestrator = SessionOrchestrator.get_instance()

    def tearDown(self):
        """Clean up after each test"""
        self.orchestrator.clear_all()
        SessionOrchestrator.reset_instance()

    # ========================================================================
    # BASIC FUNCTIONALITY TESTS
    # ========================================================================

    def test_singleton_pattern(self):
        """Test that SessionOrchestrator follows singleton pattern"""
        orchestrator1 = SessionOrchestrator.get_instance()
        orchestrator2 = SessionOrchestrator.get_instance()
        self.assertIs(orchestrator1, orchestrator2)

    def test_get_session_orchestrator_function(self):
        """Test convenience function returns singleton"""
        orchestrator = get_session_orchestrator()
        self.assertIs(orchestrator, self.orchestrator)

    def test_create_session(self):
        """Test creating a new session"""
        session = self.orchestrator.create_session(
            user_id="user1",
            main_thread_id="main_user1_abc123_1705920000"
        )

        self.assertIsNotNone(session)
        self.assertEqual(session.user_id, "user1")
        self.assertEqual(session.main_thread_id, "main_user1_abc123_1705920000")
        self.assertTrue(session.active)
        self.assertFalse(session.is_saved)

    def test_create_saved_session(self):
        """Test creating a saved session"""
        session = self.orchestrator.create_session(
            user_id="user1",
            main_thread_id="main_user1_abc123_1705920000",
            is_saved=True
        )

        self.assertTrue(session.is_saved)

    def test_create_session_with_zone(self):
        """Test creating a session with zone"""
        session = self.orchestrator.create_session(
            user_id="user1",
            main_thread_id="main_user1_abc123_1705920000",
            zone="US-WEST"
        )

        self.assertEqual(session.zone, "US-WEST")

    def test_get_session_context(self):
        """Test retrieving session by thread ID"""
        self.orchestrator.create_session(
            user_id="user1",
            main_thread_id="main_user1_abc123_1705920000"
        )

        session = self.orchestrator.get_session_context("main_user1_abc123_1705920000")

        self.assertIsNotNone(session)
        self.assertEqual(session.user_id, "user1")

    def test_get_nonexistent_session(self):
        """Test getting a session that doesn't exist"""
        session = self.orchestrator.get_session_context("nonexistent")
        self.assertIsNone(session)

    def test_heartbeat_updates_activity(self):
        """Test that heartbeat updates last_activity"""
        self.orchestrator.create_session(
            user_id="user1",
            main_thread_id="main_user1_abc123_1705920000"
        )

        session = self.orchestrator.get_session_context("main_user1_abc123_1705920000")
        first_activity = session.last_activity
        first_count = session.request_count

        time.sleep(0.1)  # Small delay

        result = self.orchestrator.heartbeat("main_user1_abc123_1705920000")

        self.assertTrue(result)
        session = self.orchestrator.get_session_context("main_user1_abc123_1705920000")
        self.assertGreater(session.last_activity, first_activity)
        self.assertGreater(session.request_count, first_count)

    def test_heartbeat_nonexistent_session(self):
        """Test heartbeat on non-existent session returns False"""
        result = self.orchestrator.heartbeat("nonexistent")
        self.assertFalse(result)

    def test_end_session(self):
        """Test ending a session"""
        self.orchestrator.create_session(
            user_id="user1",
            main_thread_id="main_user1_abc123_1705920000"
        )

        # Verify it exists
        session = self.orchestrator.get_session_context("main_user1_abc123_1705920000")
        self.assertIsNotNone(session)

        # End it
        ended = self.orchestrator.end_session("main_user1_abc123_1705920000")

        self.assertIsNotNone(ended)
        self.assertEqual(ended.user_id, "user1")

        # Verify it's gone
        session = self.orchestrator.get_session_context("main_user1_abc123_1705920000")
        self.assertIsNone(session)

    def test_end_nonexistent_session(self):
        """Test ending a non-existent session"""
        ended = self.orchestrator.end_session("nonexistent")
        self.assertIsNone(ended)

    # ========================================================================
    # USER INDEX TESTS
    # ========================================================================

    def test_user_sessions_index(self):
        """Test that user → sessions index is maintained"""
        self.orchestrator.create_session("user1", "main_user1_ts1_1705920000")
        self.orchestrator.create_session("user1", "main_user1_ts2_1705920001")
        self.orchestrator.create_session("user2", "main_user2_ts1_1705920002")

        user1_sessions = self.orchestrator.get_user_sessions("user1")
        user2_sessions = self.orchestrator.get_user_sessions("user2")

        self.assertEqual(len(user1_sessions), 2)
        self.assertEqual(len(user2_sessions), 1)

    def test_user_index_cleanup_on_end_session(self):
        """Test that user index is cleaned up when session ends"""
        self.orchestrator.create_session("user1", "main_user1_ts1_1705920000")
        self.orchestrator.create_session("user1", "main_user1_ts2_1705920001")

        self.orchestrator.end_session("main_user1_ts1_1705920000")

        user1_sessions = self.orchestrator.get_user_sessions("user1")
        self.assertEqual(len(user1_sessions), 1)

    def test_max_sessions_per_user(self):
        """Test that old sessions are removed when max is exceeded"""
        # Create sessions up to the limit
        for i in range(15):  # Default limit is 10
            self.orchestrator.create_session(
                "user1",
                f"main_user1_ts{i}_170592000{i}"
            )

        user_sessions = self.orchestrator.get_user_sessions("user1")
        self.assertLessEqual(len(user_sessions), 10)

    # ========================================================================
    # WORKFLOW TRACKING TESTS
    # ========================================================================

    def test_add_workflow_to_session(self):
        """Test adding a workflow to a session"""
        self.orchestrator.create_session("user1", "main_user1_abc123_1705920000")

        workflow = self.orchestrator.add_workflow_to_session(
            main_thread_id="main_user1_abc123_1705920000",
            workflow_id="iid_ref_xyz_1705920001",
            workflow_type="instrument_identifier"
        )

        self.assertIsNotNone(workflow)
        self.assertEqual(workflow.workflow_id, "iid_ref_xyz_1705920001")
        self.assertEqual(workflow.workflow_type, "instrument_identifier")

    def test_add_workflow_to_nonexistent_session(self):
        """Test adding workflow to non-existent session returns None"""
        workflow = self.orchestrator.add_workflow_to_session(
            main_thread_id="nonexistent",
            workflow_id="workflow_id",
            workflow_type="product_search"
        )

        self.assertIsNone(workflow)

    def test_get_workflows_for_session(self):
        """Test getting workflows for a session"""
        self.orchestrator.create_session("user1", "main_user1_abc123_1705920000")

        self.orchestrator.add_workflow_to_session(
            "main_user1_abc123_1705920000",
            "wf1",
            "instrument_identifier"
        )
        self.orchestrator.add_workflow_to_session(
            "main_user1_abc123_1705920000",
            "wf2",
            "product_search"
        )

        workflows = self.orchestrator.get_workflows_for_session("main_user1_abc123_1705920000")
        self.assertEqual(len(workflows), 2)

    def test_get_workflows_filtered_by_type(self):
        """Test filtering workflows by type"""
        self.orchestrator.create_session("user1", "main_user1_abc123_1705920000")

        self.orchestrator.add_workflow_to_session(
            "main_user1_abc123_1705920000",
            "wf1",
            "instrument_identifier"
        )
        self.orchestrator.add_workflow_to_session(
            "main_user1_abc123_1705920000",
            "wf2",
            "product_search"
        )
        self.orchestrator.add_workflow_to_session(
            "main_user1_abc123_1705920000",
            "wf3",
            "product_search"
        )

        workflows = self.orchestrator.get_workflows_for_session(
            "main_user1_abc123_1705920000",
            workflow_type="product_search"
        )
        self.assertEqual(len(workflows), 2)

    def test_remove_workflow_from_session(self):
        """Test removing a workflow from session"""
        self.orchestrator.create_session("user1", "main_user1_abc123_1705920000")

        self.orchestrator.add_workflow_to_session(
            "main_user1_abc123_1705920000",
            "wf1",
            "product_search"
        )

        removed = self.orchestrator.remove_workflow_from_session(
            "main_user1_abc123_1705920000",
            "wf1"
        )

        self.assertIsNotNone(removed)
        workflows = self.orchestrator.get_workflows_for_session("main_user1_abc123_1705920000")
        self.assertEqual(len(workflows), 0)

    # ========================================================================
    # CONCURRENCY TESTS
    # ========================================================================

    def test_concurrent_session_creation(self):
        """Test that concurrent session creation doesn't lose sessions"""
        created_sessions = []

        def create_session(user_id):
            session = self.orchestrator.create_session(
                user_id=user_id,
                main_thread_id=f"main_{user_id}_uuid_1705920000"
            )
            created_sessions.append(session)

        threads = []
        for i in range(10):
            t = threading.Thread(target=create_session, args=(f"user{i}",))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Should have all 10 sessions
        stats = self.orchestrator.get_session_stats()
        self.assertEqual(stats["active_sessions"], 10)
        self.assertEqual(stats["active_users"], 10)

    def test_concurrent_heartbeat(self):
        """Test concurrent heartbeat doesn't cause issues"""
        self.orchestrator.create_session("user1", "main_user1_abc123_1705920000")

        def send_heartbeat():
            for _ in range(10):
                self.orchestrator.heartbeat("main_user1_abc123_1705920000")

        threads = []
        for _ in range(5):
            t = threading.Thread(target=send_heartbeat)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Session should still exist and be healthy
        session = self.orchestrator.get_session_context("main_user1_abc123_1705920000")
        self.assertIsNotNone(session)

    def test_concurrent_mixed_operations(self):
        """Test concurrent create, heartbeat, and end operations"""
        # Create initial sessions
        for i in range(5):
            self.orchestrator.create_session(f"user{i}", f"main_user{i}_abc_{i}")

        results = {"created": 0, "heartbeat": 0, "ended": 0}
        lock = threading.Lock()

        def create_sessions():
            for i in range(5, 10):
                self.orchestrator.create_session(f"user{i}", f"main_user{i}_abc_{i}")
                with lock:
                    results["created"] += 1

        def send_heartbeats():
            for i in range(5):
                if self.orchestrator.heartbeat(f"main_user{i}_abc_{i}"):
                    with lock:
                        results["heartbeat"] += 1

        def end_sessions():
            for i in range(3):
                if self.orchestrator.end_session(f"main_user{i}_abc_{i}"):
                    with lock:
                        results["ended"] += 1

        threads = [
            threading.Thread(target=create_sessions),
            threading.Thread(target=send_heartbeats),
            threading.Thread(target=end_sessions)
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Verify results
        self.assertEqual(results["created"], 5)
        self.assertGreater(results["heartbeat"], 0)
        self.assertEqual(results["ended"], 3)

    # ========================================================================
    # EXPIRATION TESTS
    # ========================================================================

    def test_session_expiration(self):
        """Test that expired sessions are cleaned up"""
        session = self.orchestrator.create_session("user1", "main_user1_abc123_1705920000")

        # Manually set last_activity to past (simulate inactivity)
        session.last_activity = datetime.now() - timedelta(hours=1)

        # Run cleanup with 0 minute TTL (everything expires)
        cleaned = self.orchestrator.cleanup_expired_sessions(ttl_minutes=0)

        self.assertEqual(cleaned, 1)

        # Session should be gone
        session = self.orchestrator.get_session_context("main_user1_abc123_1705920000")
        self.assertIsNone(session)

    def test_saved_sessions_dont_expire(self):
        """Test that saved sessions are not cleaned up"""
        session = self.orchestrator.create_session(
            "user1",
            "main_user1_abc123_1705920000",
            is_saved=True
        )

        # Manually set to very old
        session.last_activity = datetime.now() - timedelta(days=365)

        # Run cleanup
        cleaned = self.orchestrator.cleanup_expired_sessions(ttl_minutes=0)

        self.assertEqual(cleaned, 0)

        # Session should still exist
        session = self.orchestrator.get_session_context("main_user1_abc123_1705920000")
        self.assertIsNotNone(session)

    def test_workflow_cleanup(self):
        """Test cleaning up inactive workflows"""
        self.orchestrator.create_session("user1", "main_user1_abc123_1705920000")

        workflow = self.orchestrator.add_workflow_to_session(
            "main_user1_abc123_1705920000",
            "wf1",
            "product_search"
        )

        # Manually set workflow to old
        workflow.last_activity = datetime.now() - timedelta(hours=2)

        # Run cleanup
        cleaned = self.orchestrator.cleanup_inactive_workflows(ttl_minutes=0)

        self.assertEqual(cleaned, 1)

        # Workflow should be gone
        workflows = self.orchestrator.get_workflows_for_session("main_user1_abc123_1705920000")
        self.assertEqual(len(workflows), 0)

    # ========================================================================
    # STATS TESTS
    # ========================================================================

    def test_get_session_stats(self):
        """Test getting session statistics"""
        self.orchestrator.create_session("user1", "main_user1_ts1_1705920000")
        self.orchestrator.create_session("user1", "main_user1_ts2_1705920001")
        self.orchestrator.create_session("user2", "main_user2_ts1_1705920002")

        self.orchestrator.add_workflow_to_session(
            "main_user1_ts1_1705920000",
            "wf1",
            "product_search"
        )
        self.orchestrator.add_workflow_to_session(
            "main_user1_ts1_1705920000",
            "wf2",
            "product_search"
        )

        stats = self.orchestrator.get_session_stats()

        self.assertEqual(stats["active_sessions"], 3)
        self.assertEqual(stats["active_users"], 2)
        self.assertEqual(stats["total_workflows"], 2)
        self.assertIn("sessions", stats)
        self.assertEqual(len(stats["sessions"]), 3)

    def test_get_active_session_count(self):
        """Test getting active session count"""
        self.orchestrator.create_session("user1", "main_user1_ts1_1705920000")
        self.orchestrator.create_session("user2", "main_user2_ts1_1705920001")

        self.assertEqual(self.orchestrator.get_active_session_count(), 2)

    def test_get_active_user_count(self):
        """Test getting active user count"""
        self.orchestrator.create_session("user1", "main_user1_ts1_1705920000")
        self.orchestrator.create_session("user1", "main_user1_ts2_1705920001")
        self.orchestrator.create_session("user2", "main_user2_ts1_1705920002")

        self.assertEqual(self.orchestrator.get_active_user_count(), 2)

    # ========================================================================
    # REQUEST LOGGING TESTS
    # ========================================================================

    def test_log_request(self):
        """Test logging requests to a session"""
        self.orchestrator.create_session("user1", "main_user1_abc123_1705920000")

        self.orchestrator.log_request(
            "main_user1_abc123_1705920000",
            "product_search",
            {"item": "pressure sensor"}
        )

        logs = self.orchestrator.get_request_log("main_user1_abc123_1705920000")
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["action"], "product_search")

    def test_request_log_limit(self):
        """Test that request log is limited"""
        self.orchestrator.create_session("user1", "main_user1_abc123_1705920000")

        # Log more than the limit
        for i in range(150):
            self.orchestrator.log_request(
                "main_user1_abc123_1705920000",
                f"action_{i}",
                {}
            )

        logs = self.orchestrator.get_request_log("main_user1_abc123_1705920000")
        self.assertLessEqual(len(logs), 100)  # Default limit


class TestSessionContext(unittest.TestCase):
    """Test SessionContext dataclass"""

    def test_session_context_creation(self):
        """Test creating SessionContext"""
        now = datetime.now()
        session = SessionContext(
            user_id="user1",
            main_thread_id="main_user1_abc123_1705920000"
        )

        self.assertEqual(session.user_id, "user1")
        self.assertEqual(session.main_thread_id, "main_user1_abc123_1705920000")
        self.assertTrue(session.active)
        self.assertFalse(session.is_saved)
        self.assertEqual(session.request_count, 0)
        self.assertGreaterEqual(session.created_at, now)

    def test_session_context_update_activity(self):
        """Test updating session activity"""
        session = SessionContext(
            user_id="user1",
            main_thread_id="main_user1_abc123_1705920000"
        )

        old_activity = session.last_activity
        old_count = session.request_count

        time.sleep(0.1)
        session.update_activity()

        self.assertGreater(session.last_activity, old_activity)
        self.assertEqual(session.request_count, old_count + 1)

    def test_session_context_expiration(self):
        """Test session expiration check"""
        session = SessionContext(
            user_id="user1",
            main_thread_id="main_user1_abc123_1705920000"
        )

        # Fresh session should not be expired
        self.assertFalse(session.is_expired(ttl_minutes=30))

        # Manually set old activity
        session.last_activity = datetime.now() - timedelta(hours=1)
        self.assertTrue(session.is_expired(ttl_minutes=30))

    def test_session_context_to_dict(self):
        """Test converting SessionContext to dict"""
        session = SessionContext(
            user_id="user1",
            main_thread_id="main_user1_abc123_1705920000",
            zone="US-WEST"
        )

        data = session.to_dict()

        self.assertEqual(data["user_id"], "user1")
        self.assertEqual(data["main_thread_id"], "main_user1_abc123_1705920000")
        self.assertEqual(data["zone"], "US-WEST")
        self.assertIn("created_at", data)
        self.assertIn("last_activity", data)


class TestWorkflowContext(unittest.TestCase):
    """Test WorkflowContext dataclass"""

    def test_workflow_context_creation(self):
        """Test creating WorkflowContext"""
        workflow = WorkflowContext(
            workflow_id="wf_abc123",
            workflow_type="product_search"
        )

        self.assertEqual(workflow.workflow_id, "wf_abc123")
        self.assertEqual(workflow.workflow_type, "product_search")
        self.assertIsNone(workflow.parent_workflow_id)

    def test_workflow_context_with_parent(self):
        """Test creating WorkflowContext with parent"""
        workflow = WorkflowContext(
            workflow_id="wf_abc123",
            workflow_type="product_search",
            parent_workflow_id="parent_xyz"
        )

        self.assertEqual(workflow.parent_workflow_id, "parent_xyz")

    def test_workflow_context_is_active(self):
        """Test workflow active check"""
        workflow = WorkflowContext(
            workflow_id="wf_abc123",
            workflow_type="product_search"
        )

        # Fresh workflow should be active
        self.assertTrue(workflow.is_active(ttl_minutes=60))

        # Old workflow should not be active
        workflow.last_activity = datetime.now() - timedelta(hours=2)
        self.assertFalse(workflow.is_active(ttl_minutes=60))

    def test_workflow_context_to_dict(self):
        """Test converting WorkflowContext to dict"""
        workflow = WorkflowContext(
            workflow_id="wf_abc123",
            workflow_type="product_search",
            metadata={"item": "sensor"}
        )

        data = workflow.to_dict()

        self.assertEqual(data["workflow_id"], "wf_abc123")
        self.assertEqual(data["workflow_type"], "product_search")
        self.assertEqual(data["metadata"]["item"], "sensor")


if __name__ == "__main__":
    unittest.main()
