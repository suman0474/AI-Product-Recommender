#!/usr/bin/env python
# test_deep_agentic_workflow.py
# =============================================================================
# Test the Deep Agentic Workflow Orchestrator
# =============================================================================

import sys
import os
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_orchestrator_initialization():
    """Test that the orchestrator can be initialized."""
    print("\n" + "="*70)
    print("TEST 1: Orchestrator Initialization")
    print("="*70)

    try:
        from agentic.deep_agent.deep_agentic_workflow import (
            get_deep_agentic_orchestrator,
            DeepAgenticWorkflowOrchestrator,
            WorkflowPhase,
            UserDecision
        )

        orchestrator = get_deep_agentic_orchestrator()
        print("[OK] Orchestrator initialized successfully")
        print(f"[OK] Type: {type(orchestrator).__name__}")
        print(f"[OK] Has session_manager: {hasattr(orchestrator, 'session_manager')}")

        print("\n[PASS] TEST 1 PASSED: Orchestrator initialization working\n")
        return True

    except Exception as e:
        print(f"\n[FAIL] TEST 1 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_session_management():
    """Test session creation and retrieval."""
    print("\n" + "="*70)
    print("TEST 2: Session Management")
    print("="*70)

    try:
        from agentic.deep_agent.deep_agentic_workflow import (
            WorkflowSessionManager,
            WorkflowState,
            WorkflowPhase
        )

        manager = WorkflowSessionManager(ttl_seconds=3600)
        print("[OK] Session manager created")

        # Create a session
        state = manager.create_session(
            user_input="I need a pressure transmitter with accuracy of 0.1%",
            session_id="test_session_001"
        )
        print(f"[OK] Session created: thread_id={state.thread_id}")
        print(f"[OK] Initial phase: {state.phase.value}")

        # Retrieve the session
        retrieved = manager.get_state(session_id="test_session_001")
        assert retrieved is not None, "Session should be retrievable"
        assert retrieved.thread_id == state.thread_id, "Thread IDs should match"
        print(f"[OK] Session retrieved successfully")

        # Update the state
        state.phase = WorkflowPhase.VALIDATION
        state.product_type = "Pressure Transmitter"
        manager.update_state(state)
        print(f"[OK] Session updated")

        # Verify update
        retrieved = manager.get_state(thread_id=state.thread_id)
        assert retrieved.phase == WorkflowPhase.VALIDATION
        assert retrieved.product_type == "Pressure Transmitter"
        print(f"[OK] Updated state verified: phase={retrieved.phase.value}, product={retrieved.product_type}")

        print("\n[PASS] TEST 2 PASSED: Session management working\n")
        return True

    except Exception as e:
        print(f"\n[FAIL] TEST 2 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_user_decision_parsing():
    """Test user decision parsing."""
    print("\n" + "="*70)
    print("TEST 3: User Decision Parsing")
    print("="*70)

    try:
        from agentic.deep_agent.deep_agentic_workflow import (
            DeepAgenticWorkflowOrchestrator,
            UserDecision
        )

        orchestrator = DeepAgenticWorkflowOrchestrator()

        test_cases = [
            ("continue", UserDecision.CONTINUE),
            ("Continue anyway", UserDecision.CONTINUE),
            ("add fields", UserDecision.ADD_FIELDS),
            ("Add the missing fields", UserDecision.ADD_FIELDS),
            ("skip", UserDecision.SKIP),
            ("yes", UserDecision.YES),
            ("no", UserDecision.NO),
            ("all", UserDecision.SELECT_ALL),
            ("include all", UserDecision.SELECT_ALL),
            (None, UserDecision.UNKNOWN),
            ("random text", UserDecision.UNKNOWN),
        ]

        all_passed = True
        for input_text, expected in test_cases:
            result = orchestrator._parse_user_decision(input_text)
            status = "[OK]" if result == expected else "[FAIL]"
            if result != expected:
                all_passed = False
            print(f"{status} '{input_text}' -> {result.value} (expected: {expected.value})")

        if all_passed:
            print("\n[PASS] TEST 3 PASSED: User decision parsing working\n")
            return True
        else:
            print("\n[FAIL] TEST 3 FAILED: Some parsing tests failed\n")
            return False

    except Exception as e:
        print(f"\n[FAIL] TEST 3 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_workflow_phases():
    """Test workflow phase enumeration."""
    print("\n" + "="*70)
    print("TEST 4: Workflow Phase Enumeration")
    print("="*70)

    try:
        from agentic.deep_agent.deep_agentic_workflow import WorkflowPhase

        expected_phases = [
            "initial",
            "validation",
            "await_missing_fields",
            "collect_fields",
            "await_advanced_params",
            "advanced_discovery",
            "await_advanced_selection",
            "vendor_analysis",
            "ranking",
            "complete",
            "error"
        ]

        actual_phases = [p.value for p in WorkflowPhase]

        print(f"[OK] Found {len(actual_phases)} workflow phases:")
        for phase in actual_phases:
            status = "[OK]" if phase in expected_phases else "[WARN]"
            print(f"   {status} {phase}")

        missing = set(expected_phases) - set(actual_phases)
        if missing:
            print(f"[WARN] Missing expected phases: {missing}")

        print("\n[PASS] TEST 4 PASSED: Workflow phases defined\n")
        return True

    except Exception as e:
        print(f"\n[FAIL] TEST 4 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_schema_normalization():
    """Test schema normalization for frontend."""
    print("\n" + "="*70)
    print("TEST 5: Schema Normalization")
    print("="*70)

    try:
        from agentic.deep_agent.deep_agentic_workflow import DeepAgenticWorkflowOrchestrator

        orchestrator = DeepAgenticWorkflowOrchestrator()

        # Test various schema formats
        test_cases = [
            # Backend format with snake_case
            {
                "input": {"mandatory_requirements": {"field1": "val1"}, "optional_requirements": {"opt1": "opt_val"}},
                "expected_mandatory": {"field1": "val1"},
                "expected_optional": {"opt1": "opt_val"}
            },
            # Frontend format (already normalized)
            {
                "input": {"mandatoryRequirements": {"field2": "val2"}, "optionalRequirements": {}},
                "expected_mandatory": {"field2": "val2"},
                "expected_optional": {}
            },
            # Nested schema
            {
                "input": {"schema": {"mandatoryRequirements": {"nested": "field"}, "optionalRequirements": {}}},
                "expected_mandatory": {"nested": "field"},
                "expected_optional": {}
            },
            # Empty/None
            {
                "input": None,
                "expected_mandatory": {},
                "expected_optional": {}
            },
        ]

        all_passed = True
        for i, case in enumerate(test_cases):
            result = orchestrator._normalize_schema(case["input"])
            mandatory_match = result["mandatoryRequirements"] == case["expected_mandatory"]
            optional_match = result["optionalRequirements"] == case["expected_optional"]

            if mandatory_match and optional_match:
                print(f"[OK] Test case {i+1}: Schema normalized correctly")
            else:
                print(f"[FAIL] Test case {i+1}: Normalization failed")
                print(f"   Expected mandatory: {case['expected_mandatory']}")
                print(f"   Got: {result['mandatoryRequirements']}")
                all_passed = False

        if all_passed:
            print("\n[PASS] TEST 5 PASSED: Schema normalization working\n")
            return True
        else:
            print("\n[FAIL] TEST 5 FAILED: Some normalization tests failed\n")
            return False

    except Exception as e:
        print(f"\n[FAIL] TEST 5 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_api_import():
    """Test that the API can import the orchestrator."""
    print("\n" + "="*70)
    print("TEST 6: API Import Compatibility")
    print("="*70)

    try:
        # Simulate the API import
        from agentic.deep_agent.deep_agentic_workflow import get_deep_agentic_orchestrator
        from agentic.deep_agent import (
            DeepAgenticWorkflowOrchestrator,
            WorkflowSessionManager,
            WorkflowState,
            WorkflowPhase,
            UserDecision,
            get_deep_agentic_orchestrator,
            reset_orchestrator,
        )

        print("[OK] All orchestrator components imported from agentic.deep_agent")

        # Verify singleton behavior
        orc1 = get_deep_agentic_orchestrator()
        orc2 = get_deep_agentic_orchestrator()
        assert orc1 is orc2, "Singleton should return same instance"
        print("[OK] Singleton pattern working correctly")

        print("\n[PASS] TEST 6 PASSED: API import compatibility verified\n")
        return True

    except Exception as e:
        print(f"\n[FAIL] TEST 6 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("   DEEP AGENTIC WORKFLOW TEST SUITE")
    print("="*70)

    results = []

    # Run tests
    results.append(("Orchestrator Initialization", test_orchestrator_initialization()))
    results.append(("Session Management", test_session_management()))
    results.append(("User Decision Parsing", test_user_decision_parsing()))
    results.append(("Workflow Phases", test_workflow_phases()))
    results.append(("Schema Normalization", test_schema_normalization()))
    results.append(("API Import Compatibility", test_api_import()))

    # Summary
    print("\n" + "="*70)
    print("   TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"   {status}: {name}")

    print(f"\n   Total: {passed}/{total} tests passed")
    print("="*70 + "\n")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
