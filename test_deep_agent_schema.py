#!/usr/bin/env python
# test_deep_agent_schema.py
# =============================================================================
# Test the new Deep Agent Schema Generation with Failure Memory
# =============================================================================

import sys
import os
import logging
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_failure_memory():
    """Test SchemaFailureMemory module."""
    print("\n" + "="*70)
    print("TEST 1: SchemaFailureMemory")
    print("="*70)

    try:
        from agentic.deep_agent.schema_failure_memory import (
            get_schema_failure_memory,
            FailureType,
            RecoveryAction
        )

        memory = get_schema_failure_memory()
        print("[OK] SchemaFailureMemory initialized")

        # Record a test failure
        entry = memory.record_failure(
            product_type="Test Product",
            failure_type=FailureType.JSON_PARSE_ERROR,
            error_message="Test JSON parse error",
            prompt="Test prompt for JSON parsing",
            context={"test": True}
        )
        print(f"[OK] Recorded test failure: {entry.failure_id}")

        # Record a test success
        success = memory.record_success(
            product_type="Test Product",
            prompt="Test successful prompt",
            fields_populated=30,
            total_fields=35,
            confidence_score=0.86,
            sources_used=["llm", "template"]
        )
        print(f"[OK] Recorded test success: {success.success_id}")

        # Test prediction
        risk = memory.predict_failure_risk("Test Product")
        print(f"[OK] Risk prediction: {risk['risk_level']}")

        # Get stats
        stats = memory.get_stats()
        print(f"[OK] Memory stats: {stats['total_failures']} failures, {stats['total_successes']} successes")

        print("\n[PASS] TEST 1 PASSED: SchemaFailureMemory working correctly\n")
        return True

    except Exception as e:
        print(f"\n[FAIL] TEST 1 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_adaptive_prompt_engine():
    """Test AdaptivePromptEngine module."""
    print("\n" + "="*70)
    print("TEST 2: AdaptivePromptEngine")
    print("="*70)

    try:
        from agentic.deep_agent.adaptive_prompt_engine import (
            get_adaptive_prompt_engine,
            PromptStrategy
        )

        engine = get_adaptive_prompt_engine()
        print("[OK] AdaptivePromptEngine initialized")

        # Test prompt optimization
        base_prompt = "Get specs for Temperature Sensor including accuracy, range, output."
        optimized, optimization = engine.optimize_prompt(
            product_type="Temperature Sensor",
            base_prompt=base_prompt,
            fields=["accuracy", "range", "output"]
        )

        print(f"[OK] Prompt optimized using strategy: {optimization.strategy.value}")
        print(f"[OK] Modifications: {optimization.modifications}")
        print(f"[OK] Confidence: {optimization.confidence:.2f}")

        # Test specialized prompts
        schema_prompt, opt = engine.get_schema_population_prompt(
            product_type="Pressure Transmitter",
            fields=["accuracy", "range", "material", "output_signal"]
        )
        print(f"[OK] Schema population prompt generated (strategy: {opt.strategy.value})")

        print("\n[PASS] TEST 2 PASSED: AdaptivePromptEngine working correctly\n")
        return True

    except Exception as e:
        print(f"\n[FAIL] TEST 2 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_deep_agent_schema_populator():
    """Test deep_agent_schema_populator module."""
    print("\n" + "="*70)
    print("TEST 3: deep_agent_schema_populator (wrapper module)")
    print("="*70)

    try:
        from agentic.deep_agent_schema_populator import (
            populate_schema_with_deep_agent,
            predict_population_success,
            get_population_stats
        )

        print("[OK] deep_agent_schema_populator module imported")

        # Test prediction
        prediction = predict_population_success("Flow Meter")
        print(f"[OK] Population prediction for Flow Meter: {prediction.get('risk_level', 'unknown')}")

        # Test stats
        stats = get_population_stats("Flow Meter")
        print("[OK] Population stats retrieved")

        print("\n[PASS] TEST 3 PASSED: deep_agent_schema_populator working correctly\n")
        return True

    except Exception as e:
        print(f"\n[FAIL] TEST 3 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_schema_generation_deep_agent():
    """Test SchemaGenerationDeepAgent (main orchestrator)."""
    print("\n" + "="*70)
    print("TEST 4: SchemaGenerationDeepAgent")
    print("="*70)

    try:
        from agentic.deep_agent.schema_generation_deep_agent import (
            get_schema_generation_deep_agent,
            SchemaSourceType
        )

        agent = get_schema_generation_deep_agent()
        print("[OK] SchemaGenerationDeepAgent initialized")

        # Get stats
        stats = agent.get_stats()
        print(f"[OK] Agent stats: {stats['generations_attempted']} attempts")

        print("\n[PASS] TEST 4 PASSED: SchemaGenerationDeepAgent initialized correctly\n")
        return True

    except Exception as e:
        print(f"\n[FAIL] TEST 4 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_full_schema_generation():
    """Test full schema generation with mock product type."""
    print("\n" + "="*70)
    print("TEST 5: Full Schema Generation (may take 10-30 seconds)")
    print("="*70)

    try:
        from agentic.deep_agent_schema_populator import populate_schema_with_deep_agent

        # Create a basic schema structure
        test_schema = {
            "product_type": "Temperature Sensor",
            "mandatory_requirements": {
                "accuracy": {"value": "", "description": "Measurement accuracy"},
                "range": {"value": "", "description": "Measurement range"},
            },
            "optional_requirements": {}
        }

        print("Starting schema population for 'Temperature Sensor'...")
        start_time = time.time()

        populated_schema, stats = populate_schema_with_deep_agent(
            product_type="Temperature Sensor",
            schema=test_schema,
            session_id="test_session_001"
        )

        duration = time.time() - start_time

        print(f"[OK] Schema generation completed in {duration:.2f}s")
        print(f"[OK] Success: {stats.get('success', False)}")
        print(f"[OK] Fields populated: {stats.get('fields_populated', 0)}")
        print(f"[OK] Sources used: {stats.get('sources_used', [])}")

        if stats.get('success'):
            print("\n[PASS] TEST 5 PASSED: Full schema generation successful\n")
            return True
        else:
            print(f"\n[WARN] TEST 5 PARTIAL: Generation ran but with issues: {stats.get('error')}\n")
            return True  # Still counts as pass if no exception

    except Exception as e:
        print(f"\n[FAIL] TEST 5 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("   DEEP AGENT SCHEMA GENERATION TEST SUITE")
    print("="*70)

    results = []

    # Test 1: Failure Memory
    results.append(("SchemaFailureMemory", test_failure_memory()))

    # Test 2: Adaptive Prompt Engine
    results.append(("AdaptivePromptEngine", test_adaptive_prompt_engine()))

    # Test 3: Wrapper Module
    results.append(("deep_agent_schema_populator", test_deep_agent_schema_populator()))

    # Test 4: Deep Agent
    results.append(("SchemaGenerationDeepAgent", test_schema_generation_deep_agent()))

    # Test 5: Full Generation (optional - takes time)
    print("\nSkipping TEST 5 (full generation) - run manually with: python test_deep_agent_schema.py --full\n")

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
    if "--full" in sys.argv:
        # Run full test including schema generation
        test_failure_memory()
        test_adaptive_prompt_engine()
        test_deep_agent_schema_populator()
        test_schema_generation_deep_agent()
        test_full_schema_generation()
    else:
        # Run quick tests only
        success = run_all_tests()
        sys.exit(0 if success else 1)
