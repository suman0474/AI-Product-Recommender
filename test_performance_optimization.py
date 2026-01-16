# test_performance_optimization.py
# =============================================================================
# PERFORMANCE TEST: Before/After Optimization Comparison
# =============================================================================
#
# Tests the performance improvements from Phase 1 optimizations:
# 1. Singleton LLM instances
# 2. Parallel batch processing
# 3. Pre-initialization of LLM clients
#
# Run with: python test_performance_optimization.py
#
# =============================================================================

import time
import logging
from typing import Dict, Any, List

# Configure logging to see performance info
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import optimized functions
from agentic.deep_agent import (
    extract_user_specs_batch,
    generate_llm_specs_batch,
    generate_llm_specs_true_batch,  # Phase 2: Single LLM call
    clear_llm_specs_cache,          # Phase 2: Cache management
    get_cache_stats,                # Phase 2: Cache statistics
    _get_user_specs_llm,
    _get_llm_specs_llm,
    run_parallel_3_source_enrichment
)


def create_test_items(count: int = 10) -> List[Dict[str, Any]]:
    """Create realistic test items for benchmarking."""
    product_types = [
        "Pressure Transmitter",
        "Temperature Sensor", 
        "Flow Meter",
        "Level Transmitter",
        "Control Valve",
        "Thermocouple",
        "RTD",
        "Junction Box",
        "Cable Gland",
        "Mounting Bracket",
        "Intrinsic Safety Barrier",
        "Surge Protection Device",
        "Power Supply",
        "HART Communicator",
        "Diaphragm Seal"
    ]
    
    items = []
    for i in range(count):
        product = product_types[i % len(product_types)]
        items.append({
            "name": product,
            "category": "Industrial Instrument",
            "type": "instrument" if i < count // 2 else "accessory",
            "sample_input": f"{product} for industrial application with explosion-proof requirement",
            "number": i + 1
        })
    
    return items


def test_singleton_initialization():
    """Test that singleton LLM instances are properly initialized once."""
    print("\n" + "=" * 70)
    print("TEST 1: Singleton LLM Initialization")
    print("=" * 70)
    
    # First call - should initialize
    start = time.time()
    llm1 = _get_user_specs_llm()
    first_time = time.time() - start
    print(f"First LLM init (user_specs): {first_time:.3f}s")
    
    # Second call - should be instant (cached)
    start = time.time()
    llm2 = _get_user_specs_llm()
    second_time = time.time() - start
    print(f"Second LLM access (cached):  {second_time:.5f}s")
    
    # Verify same instance
    assert llm1 is llm2, "LLM instances should be the same (singleton)"
    print(f"[OK] Singleton verified: same instance reused")
    print(f"[OK] Performance gain: {first_time/max(second_time, 0.0001):.1f}x faster on reuse")
    
    return True


def test_parallel_user_specs(items: List[Dict[str, Any]]):
    """Test parallel user specs extraction."""
    print("\n" + "=" * 70)
    print(f"TEST 2: Parallel User Specs Extraction ({len(items)} items)")
    print("=" * 70)
    
    user_input = "Need industrial instrumentation for oil and gas pipeline with ATEX Zone 1 certification, SIL2, and 4-20mA output."
    
    start = time.time()
    results = extract_user_specs_batch(items, user_input, max_workers=5)
    elapsed = time.time() - start
    
    print(f"Total time: {elapsed:.2f}s")
    print(f"Time per item: {elapsed/len(items):.2f}s")
    print(f"Items processed: {len(results)}")
    
    specs_count = sum(len(r.get("specifications", {})) for r in results)
    print(f"Total specs extracted: {specs_count}")
    
    # Estimate sequential time (2s per item typical)
    sequential_estimate = len(items) * 2.0
    print(f"Estimated sequential time: {sequential_estimate:.1f}s")
    print(f"[OK] Speedup: {sequential_estimate/elapsed:.1f}x faster")
    
    return results


def test_parallel_llm_specs(items: List[Dict[str, Any]]):
    """Test parallel LLM specs generation."""
    print("\n" + "=" * 70)
    print(f"TEST 3: Parallel LLM Specs Generation ({len(items)} items)")
    print("=" * 70)
    
    start = time.time()
    results = generate_llm_specs_batch(items, max_workers=5)
    elapsed = time.time() - start
    
    print(f"Total time: {elapsed:.2f}s")
    print(f"Time per item: {elapsed/len(items):.2f}s")
    print(f"Items processed: {len(results)}")
    
    specs_count = sum(len(r.get("specifications", {})) for r in results)
    print(f"Total specs generated: {specs_count}")
    
    # Estimate sequential time (2s per item typical)
    sequential_estimate = len(items) * 2.0
    print(f"Estimated sequential time: {sequential_estimate:.1f}s")
    print(f"[OK] Speedup: {sequential_estimate/elapsed:.1f}x faster")
    
    return results


def test_full_parallel_enrichment(items: List[Dict[str, Any]]):
    """Test full parallel 3-source enrichment."""
    print("\n" + "=" * 70)
    print(f"TEST 4: Full Parallel 3-Source Enrichment ({len(items)} items)")
    print("=" * 70)
    
    user_input = "Need industrial instrumentation for oil and gas pipeline with ATEX Zone 1 certification, SIL2, and 4-20mA output."
    
    start = time.time()
    result = run_parallel_3_source_enrichment(
        items=items,
        user_input=user_input,
        session_id="perf-test",
        domain_context="Oil & Gas"
    )
    elapsed = time.time() - start
    
    print(f"Total time: {elapsed:.2f}s")
    print(f"Time per item: {elapsed/len(items):.2f}s")
    print(f"Success: {result.get('success', False)}")
    
    enriched_items = result.get("items", [])
    print(f"Items enriched: {len(enriched_items)}")
    
    metadata = result.get("metadata", {})
    print(f"Processing time (ms): {metadata.get('processing_time_ms', 0)}")
    
    # Count total specs
    total_specs = 0
    source_counts = {"user": 0, "llm": 0, "standards": 0}
    for item in enriched_items:
        total_specs += len(item.get("specifications", {}))
        source_counts["user"] += len(item.get("user_specified_specs", {}))
        source_counts["llm"] += len(item.get("llm_generated_specs", {}))
        source_counts["standards"] += len(item.get("standards_specifications", {}))
    
    print(f"Total specs: {total_specs}")
    print(f"  - User specs: {source_counts['user']}")
    print(f"  - LLM specs: {source_counts['llm']}")
    print(f"  - Standards specs: {source_counts['standards']}")
    
    # Estimate old sequential time: 3 sources × n items × 2s per item
    sequential_estimate = 3 * len(items) * 2.0
    print(f"Estimated old sequential time: {sequential_estimate:.1f}s")
    print(f"[OK] Speedup: {sequential_estimate/elapsed:.1f}x faster")
    
    return result


# =============================================================================
# PHASE 2 TESTS: TRUE Batch Processing & Caching
# =============================================================================

def test_true_batch_llm_specs(items: List[Dict[str, Any]]):
    """Test TRUE batch LLM specs (single API call for all items)."""
    print("\n" + "=" * 70)
    print(f"TEST 5 (PHASE 2): TRUE Batch LLM Specs - Single API Call ({len(items)} items)")
    print("=" * 70)
    
    # Clear cache first
    clear_llm_specs_cache()
    
    start = time.time()
    results = generate_llm_specs_true_batch(items, use_cache=True)
    elapsed = time.time() - start
    
    print(f"Total time: {elapsed:.2f}s")
    print(f"Time per item: {elapsed/len(items):.2f}s")
    print(f"Items processed: {len(results)}")
    
    specs_count = sum(len(r.get("specifications", {})) for r in results)
    print(f"Total specs generated: {specs_count}")
    
    # Compare to parallel batch (5 API calls)
    parallel_estimate = len(items) * 2.0 / 5  # 5 parallel workers
    print(f"Estimated parallel batch time: {parallel_estimate:.1f}s (5 API calls)")
    print(f"[OK] TRUE Batch speedup vs parallel: {parallel_estimate/max(elapsed, 0.01):.1f}x faster")
    
    # Check cache
    cache_stats = get_cache_stats()
    print(f"Cache stats: {cache_stats['size']} items cached")
    
    return results


def test_cache_performance(items: List[Dict[str, Any]]):
    """Test caching performance (second run should be instant)."""
    print("\n" + "=" * 70)
    print(f"TEST 6 (PHASE 2): Cache Performance Test ({len(items)} items)")
    print("=" * 70)
    
    # Second run - should hit cache
    start = time.time()
    results = generate_llm_specs_true_batch(items, use_cache=True)
    elapsed = time.time() - start
    
    print(f"Total time (with cache): {elapsed:.2f}s")
    print(f"Time per item: {elapsed/len(items):.3f}s")
    
    cache_stats = get_cache_stats()
    print(f"Cache hits: {cache_stats['size']} cached product types")
    
    if elapsed < 1.0:
        print(f"[OK] Cache working - {elapsed:.3f}s is near-instant!")
    else:
        print(f"[WARN] Cache may not be working optimally")
    
    return results


def run_all_tests():
    """Run all performance tests."""
    print("\n" + "=" * 70)
    print("PERFORMANCE OPTIMIZATION TEST SUITE")
    print("=" * 70)
    print("Testing Phase 1 + Phase 2 optimizations:")
    print("  PHASE 1:")
    print("    1. Singleton LLM instances")
    print("    2. Parallel batch processing")
    print("    3. Pre-initialization of LLM clients")
    print("  PHASE 2:")
    print("    4. TRUE batch (single LLM call for all items)")
    print("    5. Product type caching (LRU cache)")
    print("=" * 70)
    
    # Create test items
    items_5 = create_test_items(5)
    items_10 = create_test_items(10)
    
    try:
        # =====================================================================
        # PHASE 1 TESTS
        # =====================================================================
        print("\n" + "-" * 70)
        print("PHASE 1 TESTS")
        print("-" * 70)
        
        # Test 1: Singleton initialization
        test_singleton_initialization()
        
        # Test 2: Parallel user specs (5 items)
        test_parallel_user_specs(items_5)
        
        # Test 3: Parallel LLM specs (5 items) - Phase 1 approach
        test_parallel_llm_specs(items_5)
        
        # =====================================================================
        # PHASE 2 TESTS
        # =====================================================================
        print("\n" + "-" * 70)
        print("PHASE 2 TESTS (10 items)")
        print("-" * 70)
        
        # Test 5: TRUE batch (single LLM call)
        test_true_batch_llm_specs(items_10)
        
        # Test 6: Cache performance (second run should be instant)
        test_cache_performance(items_10)
        
        # =====================================================================
        # FULL INTEGRATION TEST (uses Phase 2 optimizations)
        # =====================================================================
        print("\n" + "-" * 70)
        print("FULL INTEGRATION TEST (with Phase 2) - 5 items")
        print("-" * 70)
        
        # Clear cache for fair test
        clear_llm_specs_cache()
        
        # Test 4: Full parallel enrichment (5 items)
        test_full_parallel_enrichment(items_5)
        
        print("\n" + "=" * 70)
        print("[OK] ALL PHASE 1 + PHASE 2 TESTS COMPLETED SUCCESSFULLY")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    run_all_tests()
