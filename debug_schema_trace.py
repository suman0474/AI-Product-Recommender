#!/usr/bin/env python
"""
Debug script to trace schema creation and find where metadata is incorrectly formatted.
This script simulates the validation_tool.py flow to capture the actual schema output.
"""

import json
import logging

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

print("=" * 80)
print("SCHEMA TRACE DEBUG - Finding Malformed Metadata Source")
print("=" * 80)

# Step 1: Test schema_field_extractor directly
print("\n" + "-" * 60)
print("STEP 1: Testing schema_field_extractor.populate_section()")
print("-" * 60)

from agentic.deep_agent.schema_field_extractor import (
    extract_schema_field_values_from_standards,
    get_default_value_for_field,
    extract_standards_from_value,
    THERMOCOUPLE_DEFAULTS
)

# Create a minimal test schema
test_schema = {
    "mandatory_requirements": {
        "Compliance": {
            "Hazardous Area Certifications": "",
        },
        "Electrical": {
            "Output Signal": "",
        }
    }
}

result = extract_schema_field_values_from_standards("Type K Thermocouple", test_schema)
compliance = result.get("mandatory_requirements", {}).get("Compliance", {})
hazardous = compliance.get("Hazardous Area Certifications", {})

print(f"\nField: Hazardous Area Certifications")
print(f"Type: {type(hazardous)}")
if isinstance(hazardous, dict):
    for key, val in hazardous.items():
        print(f"  {key}: {type(val).__name__} = {str(val)[:60]}")
else:
    print(f"  VALUE (not dict): {hazardous}")

# Step 2: Test deep_agent_schema_populator
print("\n" + "-" * 60)
print("STEP 2: Testing deep_agent_schema_populator")
print("-" * 60)

try:
    from agentic.deep_agent_schema_populator import populate_schema_with_deep_agent
    
    test_schema2 = {
        "mandatory_requirements": {
            "Compliance": {
                "Hazardous Area Certifications": "",
            }
        }
    }
    
    # This is the function called by validation_tool.py
    populated = populate_schema_with_deep_agent(
        product_type="RTD Temperature Sensor",
        schema=test_schema2,
        max_workers=2
    )
    
    compliance2 = populated.get("mandatory_requirements", {}).get("Compliance", {})
    hazardous2 = compliance2.get("Hazardous Area Certifications", {})
    
    print(f"\nField: Hazardous Area Certifications")
    print(f"Type: {type(hazardous2)}")
    if isinstance(hazardous2, dict):
        for key, val in hazardous2.items():
            print(f"  {key}: {type(val).__name__} = {str(val)[:60]}")
    else:
        print(f"  VALUE (not dict): {hazardous2}")
        
except Exception as e:
    print(f"Error: {e}")

# Step 3: Check how the data is being formatted for output
print("\n" + "-" * 60)
print("STEP 3: Checking for text formatters")
print("-" * 60)

# Simulate what might be generating the errors.md format
def format_field_to_text(field_name: str, field_data: any) -> str:
    """Simulate the potential incorrect formatter."""
    lines = [field_name]
    
    if isinstance(field_data, dict):
        # Correct approach - extract each metadata key
        value = field_data.get("value", "Not specified")
        confidence = field_data.get("confidence", "Not specified")
        source = field_data.get("source", "Not specified")
        standards = field_data.get("standards_referenced", "Not specified")
        
        lines.append(f"Confidence:")
        lines.append(str(confidence))
        lines.append(f"Source:")
        lines.append(str(source))
        lines.append(f"Standards Referenced:")
        lines.append(str(standards))
        lines.append(f"Value:")
        lines.append(str(value))
    else:
        # If field_data is just a string
        lines.append(f"Confidence:")
        lines.append(str(field_data))  # BUG: this would output value as confidence
        lines.append(f"Source:")
        lines.append(str(field_data))  # BUG: this would output value as source
        lines.append(f"Standards Referenced:")
        lines.append("Not specified")
        lines.append(f"Value:")
        lines.append(str(field_data))
    
    return "\n".join(lines)

# Test with correct dict format
correct_field = {
    "value": "ATEX II 1/2 G Ex ia/d, IECEx, Class I Div 1/2 per IEC 60079",
    "source": "standards_specifications",
    "confidence": 0.9,
    "standards_referenced": ["IEC 60079", "ATEX", "IECEx"]
}

print("\nWith CORRECT dict format:")
print(format_field_to_text("Hazardous Area Certifications", correct_field))

# Test with incorrect string format (the bug)
incorrect_field = "ATEX II 1/2 G Ex ia/d, IECEx, Class I Div 1/2 per IEC 60079"

print("\n\nWith INCORRECT string format (THE BUG):")
print(format_field_to_text("Hazardous Area Certifications", incorrect_field))

# Step 4: Save actual schema output for inspection
print("\n" + "-" * 60)
print("STEP 4: Saving full schema output to inspect structure")
print("-" * 60)

with open("schema_debug_output.json", "w") as f:
    json.dump({
        "schema_field_extractor_result": result,
        "test_correct_field": correct_field,
        "test_incorrect_field": incorrect_field
    }, f, indent=2, default=str)

print("Saved to: schema_debug_output.json")
print("\nCheck this file to see the actual structure of schema fields.")
print("If fields are STRINGS instead of DICTS, that's the bug source!")

print("\n" + "=" * 80)
print("DEBUG COMPLETE")
print("=" * 80)
