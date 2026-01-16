#!/usr/bin/env python
"""
Comprehensive debug script to trace deep agent schema population 
and find where the metadata format breaks.
"""

import json
import logging
import sys

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

print("=" * 80)
print("COMPREHENSIVE DEEP AGENT TRACE")
print("=" * 80)

# Stage 1: Test schema_field_extractor defaults
print("\n" + "=" * 60)
print("STAGE 1: Testing schema_field_extractor.get_default_value_for_field()")
print("=" * 60)

from agentic.deep_agent.schema_field_extractor import (
    get_default_value_for_field,
    extract_standards_from_value
)

test_field = "Hazardous Area Certifications"
product = "temperature sensor"
default_value = get_default_value_for_field(product, test_field)
standards = extract_standards_from_value(default_value) if default_value else []

print(f"Field: {test_field}")
print(f"Product: {product}")
print(f"Default Value: {default_value}")
print(f"Extracted Standards: {standards}")

# Stage 2: Test deep_agent_schema_populator.extract_field_value_from_standards()
print("\n" + "=" * 60)
print("STAGE 2: Testing extract_field_value_from_standards()")
print("=" * 60)

from agentic.deep_agent_schema_populator import extract_field_value_from_standards

field_info = {"description": "Test field", "required": True}
standards_content = {
    "documents": ["test_doc.docx"],
    "standards_codes": ["IEC 60079", "ATEX"],
    "certifications": ["ATEX", "IECEx"],
    "content_snippets": []
}

result = extract_field_value_from_standards(
    field_name=test_field,
    field_info=field_info,
    product_type=product,
    standards_content=standards_content
)

print(f"Function returned: {type(result)}")
print(f"Result keys: {result.keys() if isinstance(result, dict) else 'N/A'}")
if isinstance(result, dict):
    for k, v in result.items():
        print(f"  {k}: {type(v).__name__} = {v}")
else:
    print(f"  VALUE (not dict): {result}")

# Stage 3: Test populate_schema_with_deep_agent()
print("\n" + "=" * 60) 
print("STAGE 3: Testing populate_schema_with_deep_agent()")
print("=" * 60)

from agentic.deep_agent_schema_populator import populate_schema_with_deep_agent

mini_schema = {
    "mandatory_requirements": {
        "Performance": {
            "Accuracy": ""
        },
        "Compliance": {
            "Hazardous Area Certifications": ""
        }
    }
}

try:
    populated = populate_schema_with_deep_agent(
        product_type="temperature sensor",
        schema=mini_schema,
        max_workers=2
    )
    
    print("Schema population completed!")
    
    # Check the structure
    mandatory = populated.get("mandatory_requirements", {})
    compliance = mandatory.get("Compliance", {})
    hazardous = compliance.get("Hazardous Area Certifications", {})
    
    print(f"\nField 'Hazardous Area Certifications' structure:")
    print(f"  Type: {type(hazardous)}")
    
    if isinstance(hazardous, dict):
        for k, v in hazardous.items():
            print(f"  {k}: {type(v).__name__} = {str(v)[:60]}")
        
        # Check if this matches the errors.md bug
        confidence = hazardous.get("confidence")
        source = hazardous.get("source")
        value = hazardous.get("value")
        
        if confidence == value:
            print("\n  ❌ BUG DETECTED: confidence == value!")
        elif source == value:
            print("\n  ❌ BUG DETECTED: source == value!")
        elif isinstance(confidence, (int, float)) and isinstance(source, str) and source != value:
            print("\n  ✅ STRUCTURE IS CORRECT")
        else:
            print(f"\n  ⚠️ UNEXPECTED: confidence={confidence}, source={source}")
    else:
        print(f"  ❌ NOT A DICT - This is the bug!")
        print(f"  Value: {hazardous}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# Stage 4: Save output for comparison
print("\n" + "=" * 60)
print("STAGE 4: Saving outputs for inspection")
print("=" * 60)

output_data = {
    "stage1_default_value": default_value,
    "stage1_standards": standards,
    "stage2_extract_result": result,
    "stage3_compliance_field": hazardous if 'hazardous' in dir() else "N/A"
}

with open("deep_agent_trace_output.json", "w") as f:
    json.dump(output_data, f, indent=2, default=str)

print("Saved to: deep_agent_trace_output.json")

# Stage 5: Format check - what does the errors.md format look like?
print("\n" + "=" * 60)
print("STAGE 5: Simulating errors.md format issue")
print("=" * 60)

def show_what_formatter_might_do(field_name, field_data):
    """Show what a buggy formatter might output."""
    print(f"\nIf field_data is a DICT with correct structure:")
    if isinstance(field_data, dict):
        print(f"  → Confidence: {field_data.get('confidence')}")
        print(f"  → Source: {field_data.get('source')}")
        print(f"  → Standards Referenced: {field_data.get('standards_referenced')}")
        print(f"  → Value: {field_data.get('value')}")
    
    print(f"\nIf formatter treats field_data as STRING:")
    val_str = str(field_data.get('value', field_data) if isinstance(field_data, dict) else field_data)
    print(f"  → Confidence: {val_str}")  # BUG
    print(f"  → Source: {val_str}")       # BUG
    print(f"  → Standards Referenced: Not specified")
    print(f"  → Value: {val_str}")

if isinstance(hazardous, dict):
    show_what_formatter_might_do("Hazardous Area Certifications", hazardous)

print("\n" + "=" * 80)
print("TRACE COMPLETE")
print("=" * 80)
print("\nConclusion: If Stage 3 shows correct structure but errors.md shows wrong format,")
print("then there is a TEXT FORMATTER somewhere converting dict to text incorrectly.")
print("\nLook for code that writes to errors.md or generates similar text output.")
