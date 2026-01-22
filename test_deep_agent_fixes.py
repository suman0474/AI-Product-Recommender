"""
Test script to verify Deep Agent prompt fixes.

This script:
1. Clears the prompt cache to ensure updated prompts are loaded
2. Runs the Standards Deep Agent with a test requirement
3. Displays the extracted specifications
4. Checks if the "Not specified" issue is resolved
"""

import sys
import os
import logging
import json
from datetime import datetime

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompts_library import clear_prompt_cache, list_available_prompts
from agentic.deep_agent.standards_deep_agent import run_standards_deep_agent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def count_valid_specs(specs_dict):
    """Count non-null, non-empty specifications."""
    if not specs_dict:
        return 0

    count = 0
    for key, value in specs_dict.items():
        if value is not None:
            str_value = str(value).lower().strip()
            if str_value and str_value not in ["null", "none", "n/a", "", "not specified"]:
                count += 1

    return count


def display_specs(specs_dict, max_display=50):
    """Display specifications in a readable format."""
    if not specs_dict:
        print("  [EMPTY - No specifications]")
        return

    displayed = 0
    for key, value in sorted(specs_dict.items()):
        if displayed >= max_display:
            print(f"  ... and {len(specs_dict) - displayed} more")
            break

        # Check if value is valid
        str_value = str(value).lower().strip()
        if str_value in ["null", "none", "n/a", "", "not specified"]:
            status = "[FAIL]"
        else:
            status = "[OK]"

        print(f"  {status} {key}: {value}")
        displayed += 1


def main():
    """Main test function."""
    print("=" * 100)
    print("DEEP AGENT PROMPT FIX VERIFICATION TEST")
    print("=" * 100)
    print()

    # Step 1: Clear prompt cache
    print("[STEP 1] Clearing prompt cache...")
    clear_prompt_cache()
    print("[OK] Prompt cache cleared")
    print()

    # Step 2: Verify updated prompts are available
    print("[STEP 2] Verifying updated prompts...")
    available_prompts = list_available_prompts()
    deep_agent_prompts = [p for p in available_prompts if 'deep_agent' in p.lower()]
    print(f"Found {len(deep_agent_prompts)} deep agent prompts:")
    for prompt in deep_agent_prompts:
        print(f"  - {prompt}")
    print()

    # Step 3: Test with a sample requirement
    print("[STEP 3] Running Deep Agent with test requirement...")
    print("-" * 100)

    test_requirement = (
        "I need a multi-point thermocouple assembly for measuring temperature in a "
        "chemical reactor. Operating temperature range: -50°C to +400°C. Must have "
        "ATEX Zone 1 certification and SIL 2 capability. Prefer 4-20mA output with HART protocol."
    )

    print(f"Test Requirement: {test_requirement}")
    print()

    # Sample inferred specs (simulating database)
    sample_inferred_specs = {
        "product_type": "Multi-point Thermocouple Assembly",
        "material": "Stainless Steel 316",
        "output": "4-20mA / HART",
        "ip_rating": "IP65",
        "certification": "CE, ATEX (optional)",
        "range": "Consult datasheet",
        "accuracy": "Consult datasheet"
    }

    print("Sample Database Inferred Specs:")
    for key, value in sample_inferred_specs.items():
        print(f"  - {key}: {value}")
    print()

    # Run the deep agent
    print("[EXECUTING] Standards Deep Agent workflow...")
    print("-" * 100)

    result = run_standards_deep_agent(
        user_requirement=test_requirement,
        session_id=f"test-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        inferred_specs=sample_inferred_specs,
        min_specs=30  # Minimum 30 specifications
    )

    print()
    print("-" * 100)
    print("[RESULTS] Deep Agent Execution Complete")
    print("=" * 100)
    print()

    # Display results
    print(f"Status: {result.get('status', 'unknown').upper()}")
    print(f"Success: {result.get('success', False)}")
    print(f"Processing Time: {result.get('processing_time_ms', 0)}ms")
    print(f"Iterations: {result.get('iterations', 0)}")
    print(f"Standards Analyzed: {', '.join(result.get('standards_analyzed', []))}")
    print(f"Specs Count: {result.get('specs_count', 0)} / {result.get('min_required', 30)} (target)")
    print(f"Target Reached: {'[OK] YES' if result.get('target_reached', False) else '[FAIL] NO'}")
    print()

    if result.get('iteration_notes'):
        print(f"Iteration Notes: {result['iteration_notes']}")
        print()

    if result.get('error'):
        print(f"[WARN]  ERROR: {result['error']}")
        print()

    # Display final specifications
    final_specs = result.get('final_specifications', {})
    if isinstance(final_specs, dict) and 'specifications' in final_specs:
        specs_dict = final_specs['specifications']
    elif isinstance(final_specs, dict):
        specs_dict = final_specs
    else:
        specs_dict = {}

    valid_count = count_valid_specs(specs_dict)
    total_count = len(specs_dict)
    not_specified_count = total_count - valid_count

    print(f"FINAL SPECIFICATIONS SUMMARY:")
    print(f"  Total Entries: {total_count}")
    print(f"  Valid Specs: {valid_count} [OK]")
    print(f"  Not Specified: {not_specified_count} {'[FAIL]' if not_specified_count > 0 else '[OK]'}")
    print(f"  Completeness: {(valid_count / total_count * 100) if total_count > 0 else 0:.1f}%")
    print()

    print("DETAILED SPECIFICATIONS:")
    display_specs(specs_dict, max_display=100)
    print()

    # Display constraints if available
    if final_specs.get('constraints_applied'):
        print(f"CONSTRAINTS APPLIED: {len(final_specs['constraints_applied'])}")
        for i, constraint in enumerate(final_specs['constraints_applied'][:5], 1):
            print(f"  {i}. {constraint.get('description', 'N/A')}")
            if constraint.get('value'):
                print(f"     Value: {constraint['value']}")
            if constraint.get('standard_reference'):
                print(f"     Reference: {constraint['standard_reference']}")
        print()

    # Display warnings
    if final_specs.get('warnings'):
        print(f"WARNINGS: {len(final_specs['warnings'])}")
        for i, warning in enumerate(final_specs['warnings'][:5], 1):
            print(f"  {i}. {warning}")
        print()

    # Final assessment
    print("=" * 100)
    print("ASSESSMENT:")
    print("=" * 100)

    if valid_count >= 30:
        print("[OK] SUCCESS: Deep Agent extracted 30+ valid specifications!")
        print("[OK] The 'Not specified' issue appears to be RESOLVED!")
    elif valid_count >= 20:
        print("[WARN]  PARTIAL: Deep Agent extracted 20+ specifications, but below target")
        print("   The prompts are working, but may need more iterations or better data")
    elif valid_count >= 10:
        print("[WARN]  LIMITED: Only 10-20 specifications extracted")
        print("   Check if standards documents contain relevant data")
    else:
        print("[FAIL] FAILED: Very few specifications extracted")
        print("   There may still be issues with prompts or data availability")

    print()
    print(f"Confidence: {final_specs.get('confidence', 0.0):.2f}")
    print()

    # Save detailed results to file
    output_file = "test_deep_agent_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"[FILE] Detailed results saved to: {output_file}")
    print()
    print("=" * 100)


if __name__ == "__main__":
    main()
