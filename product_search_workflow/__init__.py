"""
Product Search Workflow Package
================================

A modular, agentic workflow for product search that includes:
1. Validation Tool - Product type detection & schema generation (with PPI)
2. Advanced Parameters Tool - Discovery of latest specifications from vendors
3. Workflow Orchestrator - Complete end-to-end workflow management

Usage:
------

## Quick Start (using convenience function):
```python
from product_search_workflow import product_search_workflow

result = product_search_workflow(
    user_input="I need a pressure transmitter with 4-20mA output",
    enable_ppi=True
)

print(f"Product Type: {result['product_type']}")
print(f"Success: {result['success']}")
```

## Using the workflow class:
```python
from product_search_workflow import ProductSearchWorkflow

workflow = ProductSearchWorkflow(enable_ppi_workflow=True)

result = workflow.run_single_product_workflow(
    user_input="I need a flow meter for water"
)
```

## Using individual tools:
```python
from product_search_workflow import ValidationTool, AdvancedParametersTool

# Validation only
validator = ValidationTool(enable_ppi=True)
validation_result = validator.validate(
    user_input="I need a pH sensor",
    session_id="my_session"
)

# Advanced parameters only
params_tool = AdvancedParametersTool()
params_result = params_tool.discover(
    product_type="Pressure Transmitter",
    session_id="my_session"
)
```

## Integration with other workflows:
```python
from product_search_workflow import ProductSearchWorkflow

workflow = ProductSearchWorkflow()

# From Instruments Identifier
identifier_output = {
    "identified_instruments": [
        {"product_type": "Pressure Transmitter", "quantity": 2}
    ]
}
result = workflow.process_from_instrument_identifier(identifier_output)

# From Solution Workflow
solution_output = {
    "solution_name": "Water Treatment Plant",
    "required_products": [
        {"product_type": "pH Meter", "application": "Monitoring"}
    ]
}
result = workflow.process_from_solution_workflow(solution_output)
```

Workflow Steps:
---------------
1. **Validation Tool** (validation_tool.py)
   - Extracts product type from user input
   - Loads or generates schema (invokes PPI workflow if needed)
   - Validates requirements against schema
   - Returns structured validation result

2. **Advanced Parameters Tool** (advanced_parameters_tool.py)
   - Discovers latest specifications from top vendors
   - Includes series numbers and model information
   - Filters against existing schema (no duplicates)
   - Returns deduplicated list of specifications

3. **Workflow Orchestrator** (workflow.py)
   - Coordinates execution of all workflow steps
   - Manages session state and data flow
   - Provides integration points for other workflows
   - Supports both single product and batch processing

Features:
---------
- **Modular Design**: Each tool can be used independently or as part of workflow
- **PPI Integration**: Automatic schema generation from vendor PDFs
- **Caching**: MongoDB and in-memory caching for performance
- **Multi-Source Input**: Supports direct input, instrument identifier, and solution workflows
- **Flexible Configuration**: Enable/disable PPI, skip steps, custom sessions
- **Comprehensive Logging**: Detailed logging at each step for debugging
- **Error Handling**: Graceful error handling with detailed error reporting

Version: 1.0.0
"""

from .validation_tool import ValidationTool
from .advanced_parameters_tool import AdvancedParametersTool
from .sales_agent_tool import SalesAgentTool, WorkflowStep
from .vendor_analysis_tool import VendorAnalysisTool, analyze_vendors
from .ranking_tool import RankingTool, rank_products
from .workflow import ProductSearchWorkflow, product_search_workflow

# PPI (Potential Product Index) LangGraph Workflow
from .ppi_workflow import run_ppi_workflow, create_ppi_workflow, compile_ppi_workflow
from .ppi_tools import (
    discover_vendors_tool,
    search_pdfs_tool,
    download_and_store_pdf_tool,
    extract_pdf_data_tool,
    generate_schema_tool,
    store_vendor_data_tool,
    get_ppi_tools,
    PPI_TOOLS
)

__version__ = "2.1.0"

__all__ = [
    # Main workflow
    "ProductSearchWorkflow",
    "product_search_workflow",

    # Individual tools (Steps 1-5)
    "ValidationTool",           # Step 1: Product type detection & schema
    "AdvancedParametersTool",   # Step 2: Advanced parameters discovery
    "SalesAgentTool",           # Step 3: Requirements collection workflow
    "VendorAnalysisTool",       # Step 4: Parallel vendor analysis
    "RankingTool",              # Step 5: Final product ranking

    # PPI LangGraph Workflow
    "run_ppi_workflow",
    "create_ppi_workflow",
    "compile_ppi_workflow",
    
    # PPI Tools (LangChain tools for LangGraph)
    "discover_vendors_tool",
    "search_pdfs_tool",
    "download_and_store_pdf_tool",
    "extract_pdf_data_tool",
    "generate_schema_tool",
    "store_vendor_data_tool",
    "get_ppi_tools",
    "PPI_TOOLS",

    # Workflow step enumeration
    "WorkflowStep",

    # Convenience functions
    "analyze_vendors",
    "rank_products",
]

# Module metadata
__author__ = "AI Product Recommender Team"
__description__ = "Modular agentic workflow for product search with LangGraph PPI integration"
