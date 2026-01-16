# test_dynamic_specs.py
# =============================================================================
# TEST: Dynamic Specification Generation with Deep Reasoning
# =============================================================================
#
# This script tests the 2-phase deep reasoning approach:
# - Phase 1: Discover relevant specification keys for any product type
# - Phase 2: Generate values for those discovered keys
#
# Run: python test_dynamic_specs.py
# =============================================================================

import sys
import os
import json
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_single_product():
    """Test dynamic spec generation for a single product."""
    from agentic.deep_agent.dynamic_specs_generator import generate_dynamic_specs
    
    print("\n" + "="*80)
    print("TEST 1: Dynamic Specs for THERMOWELL")
    print("="*80)
    
    result = generate_dynamic_specs(
        product_type="Thermowell",
        category="Temperature Measurement Accessory",
        context="Thermowell with High pressure rating and ASME Section VIII compliance for 200-350°C temperature"
    )
    
    print(f"\nSuccess: {result['success']}")
    print(f"\nProduct Analysis:")
    print(json.dumps(result.get('product_analysis', {}), indent=2))
    
    print(f"\nDiscovered Keys:")
    print(f"  Mandatory: {result.get('discovered_keys', {}).get('mandatory', [])}")
    print(f"  Optional: {result.get('discovered_keys', {}).get('optional', [])}")
    print(f"  Safety: {result.get('discovered_keys', {}).get('safety_critical', [])}")
    
    print(f"\nGenerated Specifications:")
    for key, spec in result.get('specifications', {}).items():
        value = spec.get('value', spec) if isinstance(spec, dict) else spec
        conf = spec.get('confidence', 'N/A') if isinstance(spec, dict) else 'N/A'
        print(f"  {key}: {value} (confidence: {conf})")
    
    print(f"\nTiming:")
    print(f"  Discovery: {result.get('discovery_time_ms', 0)}ms")
    print(f"  Generation: {result.get('generation_time_ms', 0)}ms")
    print(f"  Total: {result.get('total_time_ms', 0)}ms")
    
    print(f"\nReasoning Notes: {result.get('reasoning_notes', 'N/A')}")
    
    return result


def test_junction_box():
    """Test for Junction Box - should NOT have accuracy/measurement specs."""
    from agentic.deep_agent.dynamic_specs_generator import generate_dynamic_specs
    
    print("\n" + "="*80)
    print("TEST 2: Dynamic Specs for JUNCTION BOX")
    print("="*80)
    
    result = generate_dynamic_specs(
        product_type="Junction Box",
        category="Electrical Enclosure",
        context="Junction Box for 200-350°C temperature environment with explosion-proof requirements"
    )
    
    print(f"\nSuccess: {result['success']}")
    
    print(f"\nDiscovered Keys (should NOT include accuracy/measurement_range):")
    all_keys = (
        result.get('discovered_keys', {}).get('mandatory', []) +
        result.get('discovered_keys', {}).get('optional', []) +
        result.get('discovered_keys', {}).get('safety_critical', [])
    )
    print(f"  All Keys: {all_keys}")
    
    # Check if inappropriate keys were excluded
    inappropriate = ['accuracy', 'measurement_range', 'repeatability', 'rangeability']
    found_inappropriate = [k for k in inappropriate if k in all_keys]
    
    if found_inappropriate:
        print(f"\n  ⚠️  WARNING: Found inappropriate keys: {found_inappropriate}")
    else:
        print(f"\n  ✅ GOOD: No inappropriate keys discovered!")
    
    print(f"\nGenerated Specifications:")
    for key, spec in result.get('specifications', {}).items():
        value = spec.get('value', spec) if isinstance(spec, dict) else spec
        print(f"  {key}: {value}")
    
    return result


def test_comparison():
    """Compare dynamic vs static approach for multiple products."""
    from agentic.deep_agent.dynamic_specs_generator import generate_dynamic_specs
    from agentic.deep_agent.llm_specs_generator import generate_llm_specs
    
    print("\n" + "="*80)
    print("TEST 3: COMPARISON - Dynamic vs Static Approach")
    print("="*80)
    
    products = [
        {"product_type": "Thermowell", "context": "High pressure 200-350°C"},
        {"product_type": "Power Supply", "context": "24VDC industrial"},
        {"product_type": "Cable Gland", "context": "Explosion-proof"}
    ]
    
    for product in products:
        pt = product["product_type"]
        ctx = product["context"]
        
        print(f"\n--- {pt} ---")
        
        # Dynamic approach
        dynamic_result = generate_dynamic_specs(
            product_type=pt,
            context=ctx
        )
        dynamic_keys = list(dynamic_result.get('specifications', {}).keys())
        
        # Static approach (current)
        static_result = generate_llm_specs(
            product_type=pt,
            context=ctx
        )
        static_keys = list(static_result.get('specifications', {}).keys())
        
        print(f"  Dynamic Approach ({len(dynamic_keys)} keys): {dynamic_keys[:8]}...")
        print(f"  Static Approach ({len(static_keys)} keys): {static_keys[:8]}...")
        
        # Unique to each
        only_dynamic = set(dynamic_keys) - set(static_keys)
        only_static = set(static_keys) - set(dynamic_keys)
        
        if only_dynamic:
            print(f"  ✅ Dynamic discovered unique keys: {list(only_dynamic)[:5]}")
        if only_static:
            print(f"  ⚠️  Static has generic keys missing from dynamic: {list(only_static)[:5]}")


def test_batch():
    """Test batch processing."""
    from agentic.deep_agent.dynamic_specs_generator import generate_dynamic_specs_batch
    
    print("\n" + "="*80)
    print("TEST 4: BATCH Processing (3 products)")
    print("="*80)
    
    items = [
        {"name": "Multi-point Thermocouple", "category": "Temperature Sensor", 
         "sample_input": "Internal catalyst bed with multiple measurement points for 200-350°C"},
        {"name": "Temperature Transmitter", "category": "Transmitter",
         "sample_input": "Thermocouple input for 200-350°C temperature"},
        {"name": "HART Communicator", "category": "Accessory",
         "sample_input": "For Temperature Transmitter configuration"}
    ]
    
    import time
    start = time.time()
    results = generate_dynamic_specs_batch(items, max_workers=2)
    elapsed = time.time() - start
    
    print(f"\nBatch completed in {elapsed:.2f}s")
    
    for result in results:
        pt = result.get('product_type', 'Unknown')
        specs = result.get('specifications', {})
        success = result.get('success', False)
        
        print(f"\n  {pt}:")
        print(f"    Success: {success}")
        print(f"    Specs Generated: {len(specs)}")
        print(f"    Keys: {list(specs.keys())[:6]}...")


if __name__ == "__main__":
    print("="*80)
    print("TESTING DYNAMIC SPECIFICATION GENERATION WITH DEEP REASONING")
    print("="*80)
    
    try:
        # Test 1: Single product
        test_single_product()
        
        # Test 2: Junction Box (should not have measurement specs)
        test_junction_box()
        
        # Test 3: Comparison
        test_comparison()
        
        # Test 4: Batch processing
        test_batch()
        
        print("\n" + "="*80)
        print("ALL TESTS COMPLETED!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
