#!/usr/bin/env python3
"""
TEST SCRIPT: Validate 60+ Specification Key Generation
This script tests the updated LLM specification generation to ensure it generates
at least 60 specification keys for each product type.
"""

import json
import logging
import sys
from datetime import datetime

# Add backend to path
sys.path.insert(0, '/d/AI PR/AIPR/backend')

from agentic.deep_agent.dynamic_specs_generator import (
    generate_dynamic_specs,
    discover_specification_keys
)
from agentic.deep_agent.llm_specs_generator import (
    generate_llm_specs,
    generate_llm_specs_true_batch
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test product types representing different categories
TEST_PRODUCTS = [
    {
        "name": "Pressure Transmitter",
        "category": "Instrumentation",
        "sample_input": "A smart pressure transmitter for industrial process monitoring with HART communication"
    },
    {
        "name": "Temperature Sensor",
        "category": "Instrumentation",
        "sample_input": "Thermowell-mounted temperature sensor for high-pressure applications"
    },
    {
        "name": "Junction Box",
        "category": "Electrical Equipment",
        "sample_input": "Industrial junction box for signal conditioning and wiring"
    },
    {
        "name": "Power Supply",
        "category": "Electrical Equipment",
        "sample_input": "DIN-rail mounted power supply for industrial control systems"
    },
    {
        "name": "Thermowell",
        "category": "Process Equipment",
        "sample_input": "Flanged thermowell for process temperature measurement"
    }
]


def test_discover_specification_keys():
    """Test Phase 1: Specification key discovery."""
    logger.info("=" * 80)
    logger.info("TEST 1: SPECIFICATION KEY DISCOVERY (Phase 1)")
    logger.info("=" * 80)

    results = []

    for product in TEST_PRODUCTS:
        logger.info(f"\nDiscovering keys for: {product['name']}")
        logger.info("-" * 60)

        try:
            discovery_result = discover_specification_keys(
                product_type=product['name'],
                category=product['category'],
                context=product['sample_input']
            )

            if discovery_result.get('success'):
                mandatory = discovery_result.get('mandatory_keys', [])
                optional = discovery_result.get('optional_keys', [])
                safety = discovery_result.get('safety_keys', [])
                total = len(mandatory) + len(optional) + len(safety)

                logger.info(f"✓ DISCOVERY SUCCESSFUL")
                logger.info(f"  - Mandatory keys: {len(mandatory)}")
                logger.info(f"  - Optional keys: {len(optional)}")
                logger.info(f"  - Safety-critical keys: {len(safety)}")
                logger.info(f"  - TOTAL KEYS: {total}")

                if total >= 60:
                    logger.info(f"  ✓ PASS: Generated {total} keys (required: 60+)")
                else:
                    logger.warning(f"  ✗ FAIL: Generated only {total} keys (required: 60+)")

                results.append({
                    'product': product['name'],
                    'phase': 'discovery',
                    'success': True,
                    'mandatory_count': len(mandatory),
                    'optional_count': len(optional),
                    'safety_count': len(safety),
                    'total_keys': total,
                    'passed_60_plus': total >= 60
                })
            else:
                logger.error(f"✗ DISCOVERY FAILED: {discovery_result.get('error', 'Unknown error')}")
                results.append({
                    'product': product['name'],
                    'phase': 'discovery',
                    'success': False,
                    'error': discovery_result.get('error')
                })

        except Exception as e:
            logger.error(f"✗ EXCEPTION: {str(e)}")
            results.append({
                'product': product['name'],
                'phase': 'discovery',
                'success': False,
                'error': str(e)
            })

    return results


def test_generate_llm_specs():
    """Test direct LLM specification generation."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: DIRECT LLM SPECIFICATION GENERATION")
    logger.info("=" * 80)

    results = []

    for product in TEST_PRODUCTS[:2]:  # Test first 2 products to save API calls
        logger.info(f"\nGenerating specs for: {product['name']}")
        logger.info("-" * 60)

        try:
            gen_result = generate_llm_specs(
                product_type=product['name'],
                category=product['category'],
                context=product['sample_input']
            )

            specs = gen_result.get('specifications', {})
            spec_count = len(specs)

            logger.info(f"✓ GENERATION SUCCESSFUL")
            logger.info(f"  - Generated specifications: {spec_count}")

            if spec_count >= 60:
                logger.info(f"  ✓ PASS: Generated {spec_count} specifications (required: 60+)")
            else:
                logger.warning(f"  ✗ FAIL: Generated only {spec_count} specifications (required: 60+)")

            # Show sample specs
            if specs:
                logger.info(f"  - Sample specs:")
                for i, (key, val) in enumerate(list(specs.items())[:5]):
                    confidence = val.get('confidence', 0.7) if isinstance(val, dict) else 0.7
                    value = val.get('value', val) if isinstance(val, dict) else val
                    logger.info(f"    {i+1}. {key}: {value} (confidence: {confidence})")

            results.append({
                'product': product['name'],
                'phase': 'llm_generation',
                'success': True,
                'spec_count': spec_count,
                'passed_60_plus': spec_count >= 60
            })

        except Exception as e:
            logger.error(f"✗ EXCEPTION: {str(e)}")
            results.append({
                'product': product['name'],
                'phase': 'llm_generation',
                'success': False,
                'error': str(e)
            })

    return results


def test_batch_generation():
    """Test batch specification generation."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: BATCH SPECIFICATION GENERATION")
    logger.info("=" * 80)

    logger.info(f"\nGenerating specs for {len(TEST_PRODUCTS)} products in batch...")
    logger.info("-" * 60)

    try:
        batch_results = generate_llm_specs_true_batch(
            items=TEST_PRODUCTS,
            use_cache=True
        )

        results = []

        for i, result in enumerate(batch_results):
            product_name = result.get('product_type', f'Product {i+1}')
            specs = result.get('specifications', {})
            spec_count = len(specs)

            logger.info(f"\n{product_name}:")
            logger.info(f"  - Generated specifications: {spec_count}")

            if spec_count >= 60:
                logger.info(f"  ✓ PASS: Generated {spec_count} specifications (required: 60+)")
                passed = True
            else:
                logger.warning(f"  ✗ FAIL: Generated only {spec_count} specifications (required: 60+)")
                passed = False

            results.append({
                'product': product_name,
                'phase': 'batch_generation',
                'success': result.get('success', False),
                'spec_count': spec_count,
                'passed_60_plus': passed
            })

        return results

    except Exception as e:
        logger.error(f"✗ BATCH EXCEPTION: {str(e)}")
        return [{
            'phase': 'batch_generation',
            'success': False,
            'error': str(e)
        }]


def print_summary(discovery_results, llm_results, batch_results):
    """Print comprehensive summary of all tests."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)

    # Discovery phase
    logger.info("\n[PHASE 1] KEY DISCOVERY RESULTS:")
    logger.info("-" * 60)
    discovery_passed = 0
    for result in discovery_results:
        if result.get('success'):
            status = "✓ PASS" if result.get('passed_60_plus') else "✗ FAIL"
            logger.info(f"{status} - {result['product']}: {result.get('total_keys', 0)} keys")
            if result.get('passed_60_plus'):
                discovery_passed += 1
        else:
            logger.info(f"✗ ERROR - {result['product']}: {result.get('error', 'Unknown error')}")

    # LLM generation phase
    logger.info("\n[PHASE 2] LLM SPECIFICATION GENERATION RESULTS:")
    logger.info("-" * 60)
    llm_passed = 0
    for result in llm_results:
        if result.get('success'):
            status = "✓ PASS" if result.get('passed_60_plus') else "✗ FAIL"
            logger.info(f"{status} - {result['product']}: {result.get('spec_count', 0)} specs")
            if result.get('passed_60_plus'):
                llm_passed += 1
        else:
            logger.info(f"✗ ERROR - {result['product']}: {result.get('error', 'Unknown error')}")

    # Batch generation phase
    logger.info("\n[PHASE 3] BATCH SPECIFICATION GENERATION RESULTS:")
    logger.info("-" * 60)
    batch_passed = 0
    for result in batch_results:
        if result.get('success') or 'product' in result:
            status = "✓ PASS" if result.get('passed_60_plus') else "✗ FAIL"
            logger.info(f"{status} - {result.get('product', 'Unknown')}: {result.get('spec_count', 0)} specs")
            if result.get('passed_60_plus'):
                batch_passed += 1
        else:
            logger.info(f"✗ ERROR: {result.get('error', 'Unknown error')}")

    # Overall summary
    logger.info("\n" + "=" * 80)
    logger.info("OVERALL SUMMARY:")
    logger.info("=" * 80)
    logger.info(f"Discovery Phase (60+ keys):")
    logger.info(f"  - Passed: {discovery_passed}/{len([r for r in discovery_results if r.get('success')])}")
    logger.info(f"LLM Generation Phase (60+ specs):")
    logger.info(f"  - Passed: {llm_passed}/{len([r for r in llm_results if r.get('success')])}")
    logger.info(f"Batch Generation Phase (60+ specs each):")
    logger.info(f"  - Passed: {batch_passed}/{len([r for r in batch_results if 'product' in r])}")

    total_passed = discovery_passed + llm_passed + batch_passed
    total_tests = len([r for r in discovery_results if r.get('success')]) + \
                  len([r for r in llm_results if r.get('success')]) + \
                  len([r for r in batch_results if 'product' in r])

    if total_tests > 0:
        pass_rate = (total_passed / total_tests) * 100
        logger.info(f"\nOVERALL PASS RATE: {pass_rate:.1f}% ({total_passed}/{total_tests})")

    if total_passed == total_tests:
        logger.info("\n✓ ALL TESTS PASSED!")
    else:
        logger.info("\n✗ SOME TESTS FAILED - Review logs above for details")


if __name__ == "__main__":
    logger.info(f"START TEST: 60+ Specification Key Generation Validation")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")

    try:
        # Run all tests
        discovery_results = test_discover_specification_keys()
        llm_results = test_generate_llm_specs()
        batch_results = test_batch_generation()

        # Print summary
        print_summary(discovery_results, llm_results, batch_results)

    except Exception as e:
        logger.error(f"FATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    logger.info("\nTest completed successfully!")
