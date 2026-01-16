"""
Debug Script: Specification Source Tracker
============================================
This script hooks into the workflow to extract and display:
1. Key-value pair specifications being sent to UI
2. Source breakdown (standards / database)
3. Verification of Standards RAG enrichment

Run this after starting the backend server.
"""

import json
import logging
from typing import Dict, Any, List

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("SPEC_TRACKER")


def analyze_item_specifications(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze a single item's specifications and source breakdown.
    
    Returns detailed breakdown of:
    - All specifications (key-value pairs)
    - Source for each specification
    - Percentage breakdown
    """
    item_name = item.get("name", "Unknown")
    item_number = item.get("number", 0)
    item_category = item.get("category", "Unknown")
    
    # Get combined specifications (with source tracking)
    combined_specs = item.get("combined_specifications", {})
    
    # Get specification source breakdown
    spec_source = item.get("specification_source", {})
    
    # Get raw standards specs and database specs
    standards_specs = item.get("standards_specifications", {})
    db_specs = item.get("specifications", {})
    
    # Standards info
    standards_info = item.get("standards_info", {})
    
    result = {
        "item_number": item_number,
        "item_name": item_name,
        "item_category": item_category,
        "specifications": {},
        "source_breakdown": {
            "standards_pct": spec_source.get("standards_pct", 0),
            "database_pct": spec_source.get("database_pct", 0),
            "standards_count": spec_source.get("standards_count", 0),
            "database_count": spec_source.get("database_count", 0),
            "total_count": spec_source.get("total_count", 0)
        },
        "enrichment_status": standards_info.get("enrichment_status", "unknown"),
        "enrichment_method": standards_info.get("enrichment_method", "unknown"),
        "standards_analyzed": standards_info.get("standards_analyzed", [])
    }
    
    # Process combined specifications
    if combined_specs:
        for key, value_info in combined_specs.items():
            if isinstance(value_info, dict):
                result["specifications"][key] = {
                    "value": value_info.get("value", "N/A"),
                    "source": value_info.get("source", "unknown"),
                    "confidence": value_info.get("confidence", 0.0)
                }
            else:
                result["specifications"][key] = {
                    "value": value_info,
                    "source": "unknown",
                    "confidence": 0.0
                }
    else:
        # Fallback: Check regular specifications
        for key, value in db_specs.items():
            result["specifications"][key] = {
                "value": value,
                "source": "database",
                "confidence": 0.7
            }
    
    return result


def print_specification_report(items: List[Dict[str, Any]]) -> None:
    """
    Print a detailed specification report for all items.
    """
    print("\n" + "=" * 80)
    print("📊 SPECIFICATION SOURCE ANALYSIS REPORT")
    print("=" * 80)
    
    total_standards = 0
    total_db = 0
    total_specs = 0
    
    for item in items:
        analysis = analyze_item_specifications(item)
        
        print(f"\n{'─' * 80}")
        print(f"📦 Item #{analysis['item_number']}: {analysis['item_name']}")
        print(f"   Category: {analysis['item_category']}")
        print(f"   Enrichment: {analysis['enrichment_method']} ({analysis['enrichment_status']})")
        print(f"   Standards Analyzed: {', '.join(analysis['standards_analyzed']) or 'None'}")
        print(f"{'─' * 80}")
        
        # Print specifications table
        specs = analysis["specifications"]
        if specs:
            print(f"\n   {'KEY':<35} {'VALUE':<30} {'SOURCE':<12} {'CONF'}")
            print(f"   {'─' * 35} {'─' * 30} {'─' * 12} {'─' * 5}")
            
            for key, info in specs.items():
                value = str(info['value'])[:28]
                source = info['source']
                conf = f"{info['confidence']:.0%}"
                
                # Color-code source
                source_marker = "📗" if source == "standards" else "📘" if source == "database" else "❓"
                
                print(f"   {key:<35} {value:<30} {source_marker} {source:<10} {conf}")
        else:
            print("   ⚠️ No specifications found")
        
        # Print source breakdown
        breakdown = analysis["source_breakdown"]
        print(f"\n   📊 Source Breakdown:")
        print(f"      📗 Standards: {breakdown['standards_count']} specs ({breakdown['standards_pct']}%)")
        print(f"      📘 Database:  {breakdown['database_count']} specs ({breakdown['database_pct']}%)")
        print(f"      📈 Total:     {breakdown['total_count']} specs")
        
        # Verify 80/20 rule
        if breakdown['standards_pct'] >= 80:
            print(f"      ✅ 80/20 Rule: PASSED ({breakdown['standards_pct']}% from standards)")
        elif breakdown['standards_pct'] > 0:
            print(f"      ⚠️ 80/20 Rule: PARTIAL ({breakdown['standards_pct']}% from standards)")
        else:
            print(f"      ❌ 80/20 Rule: FAILED (0% from standards)")
        
        total_standards += breakdown['standards_count']
        total_db += breakdown['database_count']
        total_specs += breakdown['total_count']
    
    # Overall summary
    print(f"\n{'=' * 80}")
    print("📊 OVERALL SUMMARY")
    print("=" * 80)
    
    if total_specs > 0:
        overall_standards_pct = round((total_standards / total_specs) * 100)
        overall_db_pct = round((total_db / total_specs) * 100)
    else:
        overall_standards_pct = 0
        overall_db_pct = 0
    
    print(f"   Total Items Analyzed: {len(items)}")
    print(f"   Total Specifications: {total_specs}")
    print(f"   📗 From Standards:    {total_standards} ({overall_standards_pct}%)")
    print(f"   📘 From Database:     {total_db} ({overall_db_pct}%)")
    
    if overall_standards_pct >= 80:
        print(f"\n   ✅ OVERALL 80/20 COMPLIANCE: PASSED")
    else:
        print(f"\n   ⚠️ OVERALL 80/20 COMPLIANCE: {overall_standards_pct}% from standards")
    
    print("=" * 80 + "\n")


def export_specifications_json(items: List[Dict[str, Any]], output_file: str = "specs_analysis.json") -> None:
    """
    Export specifications to JSON file for further analysis.
    """
    analysis_results = []
    
    for item in items:
        analysis = analyze_item_specifications(item)
        analysis_results.append(analysis)
    
    with open(output_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"📁 Specifications exported to: {output_file}")


# =============================================================================
# HOOK INTO WORKFLOW - Add this to format_selection_list_node or format_solution_list_node
# =============================================================================

def hook_workflow_output(state: Dict[str, Any]) -> None:
    """
    Call this function from the workflow to analyze specifications.
    
    Usage:
        from debug_spec_tracker import hook_workflow_output
        hook_workflow_output(state)
    """
    all_items = state.get("all_items", [])
    
    if all_items:
        print_specification_report(all_items)
        export_specifications_json(all_items)


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    # Test with sample data structure
    sample_items = [
        {
            "number": 1,
            "name": "Temperature Sensor",
            "category": "Temperature Measurement",
            "specifications": {
                "accuracy": "[INFERRED]",
                "temperature_range": "[INFERRED]"
            },
            "combined_specifications": {
                "accuracy": {"value": "±0.5°C", "source": "standards", "confidence": 0.85},
                "temperature_range": {"value": "200-350°C", "source": "standards", "confidence": 0.9},
                "response_time": {"value": "< 5s", "source": "standards", "confidence": 0.8},
                "type": {"value": "Thermocouple", "source": "database", "confidence": 0.7},
                "application": {"value": "In-tube measurement", "source": "standards", "confidence": 0.85}
            },
            "specification_source": {
                "standards_pct": 80,
                "database_pct": 20,
                "standards_count": 4,
                "database_count": 1,
                "total_count": 5
            },
            "standards_info": {
                "enrichment_status": "success",
                "enrichment_method": "standards_rag",
                "standards_analyzed": ["temperature", "safety", "calibration"]
            }
        }
    ]
    
    print("Running test with sample data...")
    print_specification_report(sample_items)
