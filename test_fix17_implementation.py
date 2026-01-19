"""
Test Suite for FIX #17: 60+ Specification Enrichment

This module tests:
1. SpecificationAggregator integration in PPI workflow
2. LLM specs generation
3. UnifiedEnrichmentEngine functionality
4. 60+ specification guarantee across all sources
"""

import logging
import sys
import json
from typing import Dict, Any, List
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Color codes for test output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


class TestFix17:
    """Test suite for FIX #17 implementation"""

    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0

    def log_test(self, name: str, passed: bool, message: str = ""):
        """Log test result"""
        status = f"{GREEN}✅ PASS{RESET}" if passed else f"{RED}❌ FAIL{RESET}"
        self.results.append({
            "name": name,
            "passed": passed,
            "message": message
        })

        if passed:
            self.passed += 1
        else:
            self.failed += 1

        print(f"{status} | {name}")
        if message:
            print(f"     {message}")

    def test_imports(self) -> bool:
        """Test 1: Verify all required modules can be imported"""
        logger.info("[TEST 1] Verifying module imports...")

        try:
            # Test SpecificationAggregator import
            from agentic.deep_agent.phase3_spec_aggregator import SpecificationAggregator
            logger.info("  ✓ SpecificationAggregator imported")

            # Test LLM specs generator import
            from agentic.deep_agent.llm_specs_generator import generate_llm_specs
            logger.info("  ✓ LLM specs generator imported")

            # Test UnifiedEnrichmentEngine import
            from agentic.unified_enrichment_engine import UnifiedEnrichmentEngine
            logger.info("  ✓ UnifiedEnrichmentEngine imported")

            # Test augmented templates import
            from agentic.deep_agent.augmented_product_templates import AugmentedProductTemplates
            logger.info("  ✓ AugmentedProductTemplates imported")

            self.log_test(
                "Module Imports",
                True,
                "All required modules imported successfully"
            )
            return True
        except Exception as e:
            logger.error(f"  ✗ Import failed: {e}")
            self.log_test(
                "Module Imports",
                False,
                f"Import error: {str(e)}"
            )
            return False

    def test_aggregator_basic(self) -> bool:
        """Test 2: Basic SpecificationAggregator functionality"""
        logger.info("[TEST 2] Testing SpecificationAggregator...")

        try:
            from agentic.deep_agent.phase3_spec_aggregator import SpecificationAggregator

            aggregator = SpecificationAggregator("temperature_sensor")
            logger.info("  ✓ SpecificationAggregator instantiated")

            # Test with minimal input
            result = aggregator.aggregate(
                item_id="test_temp_1",
                item_name="Test Temperature Sensor",
                user_specs={},
                standards_specs={},
                llm_specs={},
                session_id="test_session"
            )

            spec_count = len(result.get("specifications", {}))
            logger.info(f"  ✓ Generated {spec_count} specifications")

            # Check minimum guarantee
            if spec_count >= 60:
                self.log_test(
                    "Aggregator - 60+ Specs",
                    True,
                    f"Generated {spec_count} specs (target: 60+)"
                )
                return True
            else:
                self.log_test(
                    "Aggregator - 60+ Specs",
                    False,
                    f"Only {spec_count} specs (target: 60+)"
                )
                return False
        except Exception as e:
            logger.error(f"  ✗ Aggregator test failed: {e}")
            self.log_test(
                "Aggregator - Basic",
                False,
                f"Error: {str(e)}"
            )
            return False

    def test_user_specs_preservation(self) -> bool:
        """Test 3: User specs are never overwritten"""
        logger.info("[TEST 3] Testing user specs preservation...")

        try:
            from agentic.deep_agent.phase3_spec_aggregator import SpecificationAggregator

            user_specs = {
                "accuracy": {"value": "±0.1%", "confidence": 1.0},
                "sil_rating": {"value": "SIL2", "confidence": 1.0},
                "output_signal": {"value": "4-20mA", "confidence": 1.0}
            }

            aggregator = SpecificationAggregator("pressure_transmitter")
            result = aggregator.aggregate(
                item_id="test_press_1",
                item_name="Test Transmitter",
                user_specs=user_specs,
                standards_specs={},
                llm_specs={},
                session_id="test_session"
            )

            specs = result.get("specifications", {})

            # Verify user specs are preserved
            all_preserved = True
            for key, val in user_specs.items():
                if key in specs:
                    spec_val = specs[key].get("value") if isinstance(specs[key], dict) else specs[key]
                    expected_val = val.get("value") if isinstance(val, dict) else val
                    if spec_val != expected_val:
                        logger.warning(f"  ✗ {key}: expected {expected_val}, got {spec_val}")
                        all_preserved = False
                else:
                    logger.warning(f"  ✗ {key} missing from aggregated specs")
                    all_preserved = False

            if all_preserved:
                logger.info(f"  ✓ All {len(user_specs)} user specs preserved")
                self.log_test(
                    "User Specs Preservation",
                    True,
                    "All user-specified values preserved with confidence 1.0"
                )
                return True
            else:
                self.log_test(
                    "User Specs Preservation",
                    False,
                    "Some user specs were lost or modified"
                )
                return False
        except Exception as e:
            logger.error(f"  ✗ Preservation test failed: {e}")
            self.log_test(
                "User Specs Preservation",
                False,
                f"Error: {str(e)}"
            )
            return False

    def test_multi_source_aggregation(self) -> bool:
        """Test 4: Multi-source aggregation with priorities"""
        logger.info("[TEST 4] Testing multi-source aggregation...")

        try:
            from agentic.deep_agent.phase3_spec_aggregator import SpecificationAggregator

            user_specs = {
                "accuracy": {"value": "±0.1%", "confidence": 1.0}
            }

            standards_specs = {
                "sil_rating": {"value": "SIL2", "confidence": 0.9},
                "accuracy": {"value": "±0.5%", "confidence": 0.9}  # Should NOT override user
            }

            llm_specs = {
                "response_time": {"value": "100ms", "confidence": 0.8},
                "accuracy": {"value": "±1%", "confidence": 0.8}  # Should NOT override
            }

            aggregator = SpecificationAggregator("flow_meter")
            result = aggregator.aggregate(
                item_id="test_flow_1",
                item_name="Test Flow Meter",
                user_specs=user_specs,
                standards_specs=standards_specs,
                llm_specs=llm_specs,
                session_id="test_session"
            )

            specs = result.get("specifications", {})

            # Check priority: user accuracy should be ±0.1%, not overridden
            accuracy = specs.get("accuracy")
            if isinstance(accuracy, dict):
                accuracy_val = accuracy.get("value")
            else:
                accuracy_val = accuracy

            if accuracy_val == "±0.1%":
                logger.info("  ✓ User spec not overridden by standards or LLM")
                self.log_test(
                    "Multi-source Priority",
                    True,
                    "User specs have priority over standards and LLM"
                )
                return True
            else:
                logger.warning(f"  ✗ User spec overridden: {accuracy_val}")
                self.log_test(
                    "Multi-source Priority",
                    False,
                    f"User spec was overridden (got {accuracy_val})"
                )
                return False
        except Exception as e:
            logger.error(f"  ✗ Multi-source test failed: {e}")
            self.log_test(
                "Multi-source Priority",
                False,
                f"Error: {str(e)}"
            )
            return False

    def test_unified_engine(self) -> bool:
        """Test 5: UnifiedEnrichmentEngine end-to-end"""
        logger.info("[TEST 5] Testing UnifiedEnrichmentEngine...")

        try:
            from agentic.unified_enrichment_engine import enrich_product_with_60_plus_specs

            result = enrich_product_with_60_plus_specs(
                product_type="temperature_sensor",
                user_specs={"accuracy": "±0.1%"},
                vendor_data=None,
                standards_context="Industrial temperature measurement"
            )

            spec_count = len(result.get("specifications", {}))
            logger.info(f"  ✓ Generated {spec_count} specifications")

            aggregation = result.get("aggregation", {})
            sources = aggregation.get("spec_sources", {})
            logger.info(f"  ✓ Sources: {sources}")

            if spec_count >= 60:
                self.log_test(
                    "UnifiedEnrichmentEngine",
                    True,
                    f"Generated {spec_count} specs from sources: {sources}"
                )
                return True
            else:
                self.log_test(
                    "UnifiedEnrichmentEngine",
                    False,
                    f"Only {spec_count} specs (need 60+)"
                )
                return False
        except Exception as e:
            logger.error(f"  ✗ UnifiedEnrichmentEngine test failed: {e}")
            self.log_test(
                "UnifiedEnrichmentEngine",
                False,
                f"Error: {str(e)}"
            )
            return False

    def test_different_product_types(self) -> bool:
        """Test 6: Multiple product types reach 60+ specs"""
        logger.info("[TEST 6] Testing different product types...")

        product_types = [
            "temperature_sensor",
            "pressure_transmitter",
            "flow_meter",
            "level_transmitter"
        ]

        try:
            from agentic.deep_agent.phase3_spec_aggregator import SpecificationAggregator

            all_passed = True
            type_results = {}

            for product_type in product_types:
                try:
                    aggregator = SpecificationAggregator(product_type)
                    result = aggregator.aggregate(
                        item_id=f"test_{product_type}",
                        item_name=f"Test {product_type.replace('_', ' ').title()}",
                        user_specs={},
                        standards_specs={},
                        llm_specs={},
                        session_id="test_session"
                    )

                    spec_count = len(result.get("specifications", {}))
                    type_results[product_type] = spec_count
                    logger.info(f"  ✓ {product_type}: {spec_count} specs")

                    if spec_count < 60:
                        all_passed = False
                except Exception as e:
                    logger.warning(f"  ✗ {product_type}: {e}")
                    type_results[product_type] = 0
                    all_passed = False

            if all_passed:
                self.log_test(
                    "Multiple Product Types",
                    True,
                    f"All types ≥60 specs: {type_results}"
                )
                return True
            else:
                self.log_test(
                    "Multiple Product Types",
                    False,
                    f"Some types <60 specs: {type_results}"
                )
                return False
        except Exception as e:
            logger.error(f"  ✗ Product types test failed: {e}")
            self.log_test(
                "Multiple Product Types",
                False,
                f"Error: {str(e)}"
            )
            return False

    def test_consistency_with_without_user(self) -> bool:
        """Test 7: Consistency with and without user input"""
        logger.info("[TEST 7] Testing consistency with/without user input...")

        try:
            from agentic.deep_agent.phase3_spec_aggregator import SpecificationAggregator

            aggregator = SpecificationAggregator("pressure_transmitter")

            # With user input
            result_with = aggregator.aggregate(
                item_id="test_with",
                item_name="With User",
                user_specs={"accuracy": "±0.1%"},
                standards_specs={},
                llm_specs={},
                session_id="test_session"
            )

            # Without user input
            result_without = aggregator.aggregate(
                item_id="test_without",
                item_name="Without User",
                user_specs={},
                standards_specs={},
                llm_specs={},
                session_id="test_session"
            )

            count_with = len(result_with.get("specifications", {}))
            count_without = len(result_without.get("specifications", {}))
            variance = abs(count_with - count_without)

            logger.info(f"  ✓ With user: {count_with} specs")
            logger.info(f"  ✓ Without user: {count_without} specs")
            logger.info(f"  ✓ Variance: ±{variance} specs")

            if variance <= 5 and count_with >= 60 and count_without >= 60:
                self.log_test(
                    "Consistency",
                    True,
                    f"Variance: ±{variance} specs (both ≥60)"
                )
                return True
            else:
                self.log_test(
                    "Consistency",
                    False,
                    f"Variance too high (±{variance}) or counts too low"
                )
                return False
        except Exception as e:
            logger.error(f"  ✗ Consistency test failed: {e}")
            self.log_test(
                "Consistency",
                False,
                f"Error: {str(e)}"
            )
            return False

    def run_all_tests(self):
        """Run all tests"""
        print(f"\n{BLUE}{'='*60}")
        print(f"FIX #17 IMPLEMENTATION TEST SUITE")
        print(f"{'='*60}{RESET}\n")

        # Run tests
        self.test_imports()
        self.test_aggregator_basic()
        self.test_user_specs_preservation()
        self.test_multi_source_aggregation()
        self.test_unified_engine()
        self.test_different_product_types()
        self.test_consistency_with_without_user()

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print test summary"""
        total = self.passed + self.failed
        pass_rate = (self.passed / total * 100) if total > 0 else 0

        print(f"\n{BLUE}{'='*60}")
        print(f"TEST SUMMARY")
        print(f"{'='*60}{RESET}\n")

        print(f"Total Tests: {total}")
        print(f"{GREEN}Passed: {self.passed}{RESET}")
        print(f"{RED}Failed: {self.failed}{RESET}")
        print(f"Pass Rate: {pass_rate:.1f}%\n")

        # Detailed results
        print(f"{BLUE}Detailed Results:{RESET}\n")
        for i, result in enumerate(self.results, 1):
            status = f"{GREEN}✅{RESET}" if result['passed'] else f"{RED}❌{RESET}"
            print(f"{status} {i}. {result['name']}")
            if result['message']:
                print(f"   {result['message']}")

        # Overall status
        print(f"\n{BLUE}{'='*60}{RESET}")
        if self.failed == 0:
            print(f"{GREEN}ALL TESTS PASSED ✅{RESET}")
            print(f"FIX #17 implementation is ready for deployment!")
        else:
            print(f"{RED}SOME TESTS FAILED ❌{RESET}")
            print(f"Please fix issues before deployment")
        print(f"{BLUE}{'='*60}{RESET}\n")


if __name__ == "__main__":
    tester = TestFix17()
    tester.run_all_tests()

    # Exit with appropriate code
    sys.exit(0 if tester.failed == 0 else 1)
