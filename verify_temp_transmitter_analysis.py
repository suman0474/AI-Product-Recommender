
import logging
import sys
import os
import json
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Add backend directory to path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

try:
    from product_search_workflow.vendor_analysis_tool import VendorAnalysisTool
    from agentic_advanced_parameters_tool import advanced_parameters_tool
    from tools.schema_tools import load_schema_tool
except ImportError as e:
    logger.error(f"Import failed: {e}")
    logger.error("Please ensure you are running this script from the backend root directory (d:/AI PR/AIPR/backend)")
    sys.exit(1)

def verify_analysis_flow(product_type="Temperature Transmitter"):
    print("\n" + "="*80)
    print(f"VERIFYING ANALYSIS FLOW FOR: {product_type}")
    print("="*80)

    # 1. GENERATE SCHEMA
    print("\n[Step 1] Loading Schema...")
    try:
        schema_result = load_schema_tool.invoke({"product_type": product_type})
        schema = schema_result.get("schema", {})
        print(f"Schema loaded with {len(schema.get('mandatory', {}))} mandatory and {len(schema.get('optional', {}))} optional fields.")
    except Exception as e:
        logger.error(f"Failed to load schema: {e}")
        # Fallback schema for testing if tool fails
        schema = {
            "mandatory": {"input_type": "Input Type", "output_signal": "Output Signal"},
            "optional": {"mounting": "Mounting Type"}
        }
        print("Using fallback schema.")

    # 2. GENERATE ADVANCED SPECIFICATIONS
    print("\n[Step 2] Discovering Advanced Specifications...")
    try:
        adv_params_result = advanced_parameters_tool.discover(product_type=product_type)
        unique_specs = adv_params_result.get("unique_specifications", [])
        print(f"Discovered {len(unique_specs)} advanced specifications.")
    except Exception as e:
        logger.error(f"Failed to discover advanced specs: {e}")
        unique_specs = []

    # 3. PREPARE STRUCTURED REQUIREMENTS
    # Simulate user selection (selecting top 3 advanced specs if available)
    selected_advanced = {}
    for spec in unique_specs[:3]:
        key = spec.get("key")
        name = spec.get("name")
        selected_advanced[name] = "Yes" # User selects "Yes" for these

    # Construct the payload exactly as the frontend would send it
    structured_requirements = {
        "productType": product_type,
        "mandatoryRequirements": {
            "input_type": "RTD Pt100", # Example value
            "output_signal": "4-20mA"
        },
        "optionalRequirements": {
            "mounting": "Head Mount"
        },
        "selectedAdvancedParams": selected_advanced
    }
    
    print("\n[Step 3] Constructed Structured Requirements Payload:")
    print(json.dumps(structured_requirements, indent=2))

    # 4. RUN FINAL VENDOR ANALYSIS
    print("\n[Step 4] Running Final Vendor Analysis...")
    analysis_tool = VendorAnalysisTool(max_workers=3)
    
    # We call analyze() with the format exactly as expected by the tool
    # Note: In the actual API endpoint, `structured_requirements` is passed as the first arg
    try:
        result = analysis_tool.analyze(
            structured_requirements=structured_requirements,
            product_type=product_type,
            session_id="verify_script_session",
            schema=schema
        )
        
        print("\n" + "="*80)
        print("ANALYSIS RESULT FORMAT")
        print("="*80)
        
        # Print keys and summary to verify structure
        print(f"Result Keys: {list(result.keys())}")
        print(f"Success: {result.get('success')}")
        print(f"Vendors Analyzed: {result.get('vendors_analyzed')}")
        print(f"Total Matches: {result.get('total_matches')}")
        
        if result.get('vendor_matches'):
            print(f"\nFirst Match Example:")
            print(json.dumps(result['vendor_matches'][0], indent=2))
        else:
            print("\nNo matches found (expected if data is missing in Azure Blob).")
            print(f"Analysis Summary: {result.get('analysis_summary')}")
            
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)

if __name__ == "__main__":
    verify_analysis_flow("Temperature Transmitter")
