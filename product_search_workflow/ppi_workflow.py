"""
PPI (Potential Product Index) LangGraph Workflow
==================================================

A LangGraph-based workflow that generates schemas and vendor data
when product data is not found in the database.

Workflow Steps:
1. Discover top 5 vendors
2. For each vendor: Search PDFs → Download → Extract → Store
3. Generate schema from vendor data
4. Store schema in database

This workflow is invoked as a sub-graph when load_schema_tool
detects that no schema exists for a product type.
"""

import logging
from typing import Dict, Any, List, Optional, TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


# ============================================================================
# STATE DEFINITION
# ============================================================================

class PPIState(TypedDict):
    """State for the PPI workflow"""
    # Input
    product_type: str
    session_id: Optional[str]
    
    # Vendor Discovery
    vendors: List[Dict[str, Any]]
    vendor_count: int
    
    # PDF Processing
    current_vendor_index: int
    pdfs_found: List[Dict[str, Any]]
    pdfs_downloaded: List[Dict[str, Any]]
    extracted_data: List[Dict[str, Any]]
    
    # Schema Generation
    schema: Optional[Dict[str, Any]]
    schema_saved: bool
    
    # Vendor Data Storage
    vendor_data_stored: List[str]
    
    # Status
    success: bool
    error: Optional[str]
    messages: Annotated[List[BaseMessage], operator.add]


# ============================================================================
# WORKFLOW NODES
# ============================================================================

def discover_vendors_node(state: PPIState) -> Dict[str, Any]:
    """
    Node 1: Discover top 5 vendors for the product type
    """
    product_type = state["product_type"]
    logger.info(f"[PPI_WORKFLOW] Step 1: Discovering vendors for {product_type}")
    
    try:
        from .ppi_tools import discover_vendors_tool
        
        result = discover_vendors_tool.invoke({"product_type": product_type})
        
        if result.get("success"):
            vendors = result.get("vendors", [])
            logger.info(f"[PPI_WORKFLOW] Discovered {len(vendors)} vendors")
            
            return {
                "vendors": vendors,
                "vendor_count": len(vendors),
                "current_vendor_index": 0,
                "messages": [AIMessage(content=f"Discovered {len(vendors)} vendors for {product_type}")]
            }
        else:
            logger.warning(f"[PPI_WORKFLOW] Vendor discovery failed: {result.get('error')}")
            return {
                "vendors": [],
                "vendor_count": 0,
                "success": False,
                "error": result.get("error", "Vendor discovery failed"),
                "messages": [AIMessage(content=f"Failed to discover vendors: {result.get('error')}")]
            }
            
    except Exception as e:
        logger.error(f"[PPI_WORKFLOW] Vendor discovery error: {e}")
        return {
            "vendors": [],
            "vendor_count": 0,
            "success": False,
            "error": str(e),
            "messages": [AIMessage(content=f"Error discovering vendors: {e}")]
        }


def search_pdfs_node(state: PPIState) -> Dict[str, Any]:
    """
    Node 2: Search for PDFs for the current vendor
    """
    vendors = state.get("vendors", [])
    current_index = state.get("current_vendor_index", 0)
    
    if current_index >= len(vendors):
        logger.info("[PPI_WORKFLOW] All vendors processed")
        return {"messages": [AIMessage(content="All vendors processed")]}
    
    vendor_info = vendors[current_index]
    vendor_name = vendor_info.get("vendor", "Unknown")
    product_type = state["product_type"]
    
    logger.info(f"[PPI_WORKFLOW] Step 2: Searching PDFs for {vendor_name}")
    
    try:
        from .ppi_tools import search_pdfs_tool
        
        # Search for PDFs
        result = search_pdfs_tool.invoke({
            "vendor": vendor_name,
            "product_type": product_type,
            "model_family": None
        })
        
        pdfs_found = state.get("pdfs_found", [])
        
        if result.get("success"):
            new_pdfs = result.get("pdfs", [])
            # Tag each PDF with vendor info
            for pdf in new_pdfs:
                pdf["vendor"] = vendor_name
            pdfs_found.extend(new_pdfs)
            
            logger.info(f"[PPI_WORKFLOW] Found {len(new_pdfs)} PDFs for {vendor_name}")
            return {
                "pdfs_found": pdfs_found,
                "messages": [AIMessage(content=f"Found {len(new_pdfs)} PDFs for {vendor_name}")]
            }
        else:
            logger.warning(f"[PPI_WORKFLOW] No PDFs found for {vendor_name}")
            return {
                "pdfs_found": pdfs_found,
                "messages": [AIMessage(content=f"No PDFs found for {vendor_name}")]
            }
            
    except Exception as e:
        logger.error(f"[PPI_WORKFLOW] PDF search error: {e}")
        return {
            "messages": [AIMessage(content=f"Error searching PDFs: {e}")]
        }


def download_pdfs_node(state: PPIState) -> Dict[str, Any]:
    """
    Node 3: Download and store PDFs for the current vendor
    """
    pdfs_found = state.get("pdfs_found", [])
    vendors = state.get("vendors", [])
    current_index = state.get("current_vendor_index", 0)
    product_type = state["product_type"]
    
    if current_index >= len(vendors):
        return {"messages": [AIMessage(content="No more vendors to process")]}
    
    vendor_info = vendors[current_index]
    vendor_name = vendor_info.get("vendor", "Unknown")
    
    # Filter PDFs for current vendor
    vendor_pdfs = [p for p in pdfs_found if p.get("vendor") == vendor_name][:2]  # Top 2 per vendor
    
    logger.info(f"[PPI_WORKFLOW] Step 3: Downloading {len(vendor_pdfs)} PDFs for {vendor_name}")
    
    try:
        from .ppi_tools import download_and_store_pdf_tool
        
        pdfs_downloaded = state.get("pdfs_downloaded", [])
        
        for pdf_info in vendor_pdfs:
            pdf_url = pdf_info.get("url")
            if not pdf_url:
                continue
                
            result = download_and_store_pdf_tool.invoke({
                "pdf_url": pdf_url,
                "vendor": vendor_name,
                "product_type": product_type,
                "model_family": None
            })
            
            if result.get("success"):
                pdfs_downloaded.append({
                    "vendor": vendor_name,
                    "file_id": result.get("file_id"),
                    "pdf_bytes": result.get("pdf_bytes")
                })
                logger.info(f"[PPI_WORKFLOW] Downloaded PDF for {vendor_name}")
        
        return {
            "pdfs_downloaded": pdfs_downloaded,
            "messages": [AIMessage(content=f"Downloaded {len(pdfs_downloaded)} PDFs")]
        }
        
    except Exception as e:
        logger.error(f"[PPI_WORKFLOW] PDF download error: {e}")
        return {
            "messages": [AIMessage(content=f"Error downloading PDFs: {e}")]
        }


def extract_data_node(state: PPIState) -> Dict[str, Any]:
    """
    Node 4: Extract structured data from downloaded PDFs
    """
    pdfs_downloaded = state.get("pdfs_downloaded", [])
    vendors = state.get("vendors", [])
    current_index = state.get("current_vendor_index", 0)
    product_type = state["product_type"]
    
    if current_index >= len(vendors):
        return {"messages": [AIMessage(content="No more vendors to process")]}
    
    vendor_info = vendors[current_index]
    vendor_name = vendor_info.get("vendor", "Unknown")
    
    # Filter PDFs for current vendor
    vendor_pdfs = [p for p in pdfs_downloaded if p.get("vendor") == vendor_name]
    
    logger.info(f"[PPI_WORKFLOW] Step 4: Extracting data from {len(vendor_pdfs)} PDFs for {vendor_name}")
    
    try:
        from .ppi_tools import extract_pdf_data_tool
        
        extracted_data = state.get("extracted_data", [])
        
        for pdf_info in vendor_pdfs:
            pdf_bytes = pdf_info.get("pdf_bytes")
            if not pdf_bytes:
                continue
                
            result = extract_pdf_data_tool.invoke({
                "pdf_bytes": pdf_bytes,
                "vendor": vendor_name,
                "product_type": product_type
            })
            
            if result.get("success"):
                extracted_data.append({
                    "vendor": vendor_name,
                    "product_type": product_type,
                    "data": result.get("extracted_data", []),
                    "success": True
                })
                logger.info(f"[PPI_WORKFLOW] Extracted {len(result.get('extracted_data', []))} entries for {vendor_name}")
        
        return {
            "extracted_data": extracted_data,
            "messages": [AIMessage(content=f"Extracted data from {len(vendor_pdfs)} PDFs")]
        }
        
    except Exception as e:
        logger.error(f"[PPI_WORKFLOW] Data extraction error: {e}")
        return {
            "messages": [AIMessage(content=f"Error extracting data: {e}")]
        }


def store_vendor_data_node(state: PPIState) -> Dict[str, Any]:
    """
    Node 5: Store extracted vendor data in Azure Blob
    """
    extracted_data = state.get("extracted_data", [])
    vendors = state.get("vendors", [])
    current_index = state.get("current_vendor_index", 0)
    product_type = state["product_type"]
    
    if current_index >= len(vendors):
        return {"messages": [AIMessage(content="No more vendors to process")]}
    
    vendor_info = vendors[current_index]
    vendor_name = vendor_info.get("vendor", "Unknown")
    
    # Filter extracted data for current vendor
    vendor_data = [d for d in extracted_data if d.get("vendor") == vendor_name]
    
    logger.info(f"[PPI_WORKFLOW] Step 5: Storing data for {vendor_name}")
    
    try:
        from .ppi_tools import store_vendor_data_tool
        from test import aggregate_results
        
        vendor_data_stored = state.get("vendor_data_stored", [])
        
        for data in vendor_data:
            if data.get("success") and data.get("data"):
                # Aggregate extracted data
                aggregated = aggregate_results(data["data"], product_type)
                
                vendor_payload = {
                    "vendor": vendor_name,
                    "product_type": product_type,
                    "models": aggregated.get("models", [])
                }
                
                result = store_vendor_data_tool.invoke({
                    "vendor_data": vendor_payload,
                    "product_type": product_type
                })
                
                if result.get("success"):
                    vendor_data_stored.append(vendor_name)
                    logger.info(f"[PPI_WORKFLOW] Stored data for {vendor_name}")
        
        # Move to next vendor
        return {
            "vendor_data_stored": vendor_data_stored,
            "current_vendor_index": current_index + 1,
            "messages": [AIMessage(content=f"Stored data for {vendor_name}")]
        }
        
    except Exception as e:
        logger.error(f"[PPI_WORKFLOW] Data storage error: {e}")
        return {
            "current_vendor_index": current_index + 1,
            "messages": [AIMessage(content=f"Error storing data: {e}")]
        }


def generate_schema_node(state: PPIState) -> Dict[str, Any]:
    """
    Node 6: Generate schema from all extracted vendor data
    """
    extracted_data = state.get("extracted_data", [])
    product_type = state["product_type"]

    logger.info(f"[PPI_WORKFLOW] Step 6: Generating schema from {len(extracted_data)} vendor data sets")

    try:
        from .ppi_tools import generate_schema_tool

        result = generate_schema_tool.invoke({
            "product_type": product_type,
            "vendor_data": extracted_data
        })

        if result.get("success"):
            schema = result.get("schema")
            logger.info(f"[PPI_WORKFLOW] Schema generated successfully")

            return {
                "schema": schema,
                "schema_saved": False,  # Will be saved after standards enrichment
                "success": True,
                "messages": [AIMessage(content=f"Schema generated with {result.get('mandatory_count', 0)} mandatory and {result.get('optional_count', 0)} optional fields")]
            }
        else:
            logger.warning(f"[PPI_WORKFLOW] Schema generation failed: {result.get('error')}")
            return {
                "schema": None,
                "schema_saved": False,
                "success": False,
                "error": result.get("error", "Schema generation failed"),
                "messages": [AIMessage(content=f"Schema generation failed: {result.get('error')}")]
            }

    except Exception as e:
        logger.error(f"[PPI_WORKFLOW] Schema generation error: {e}")
        return {
            "schema": None,
            "schema_saved": False,
            "success": False,
            "error": str(e),
            "messages": [AIMessage(content=f"Error generating schema: {e}")]
        }


def enrich_schema_with_standards_node(state: PPIState) -> Dict[str, Any]:
    """
    Node 7: Enrich generated schema with Standards RAG information.

    This node performs TWO levels of enrichment:
    1. POPULATE field values: Fills empty schema fields with standards-based values
       (allowed ranges, units, typical specifications from standards)
    2. ADD standards section: Adds applicable standards, certifications, safety requirements

    Queries Standards RAG to add:
    - Field values from standards (accuracy ranges, output signals, etc.)
    - Applicable engineering standards (ISO, IEC, API, etc.)
    - Required certifications (SIL, ATEX, CE, etc.)
    - Safety requirements
    - Calibration standards
    - Environmental requirements
    - Communication protocols
    """
    schema = state.get("schema")
    product_type = state["product_type"]

    # Skip if no schema was generated
    if not schema:
        logger.warning("[PPI_WORKFLOW] No schema to enrich with standards")
        return {
            "messages": [AIMessage(content="Skipping standards enrichment - no schema available")]
        }

    logger.info(f"[PPI_WORKFLOW] Step 7: Enriching schema with Standards RAG for {product_type}")

    try:
        from tools.standards_enrichment_tool import (
            enrich_schema_with_standards,
            populate_schema_fields_from_standards
        )

        # Step 1: POPULATE field values from Standards RAG
        logger.info(f"[PPI_WORKFLOW] Step 7a: Populating schema field values from standards")
        populated_schema = populate_schema_fields_from_standards(product_type, schema)
        
        fields_populated = populated_schema.get("_standards_population", {}).get("fields_populated", 0)
        logger.info(f"[PPI_WORKFLOW] Populated {fields_populated} fields with standards values")

        # Step 2: ADD standards section with applicable standards, certifications, etc.
        logger.info(f"[PPI_WORKFLOW] Step 7b: Adding standards section")
        enriched_schema = enrich_schema_with_standards(product_type, populated_schema)

        # Get statistics for logging
        standards_section = enriched_schema.get('standards', {})
        num_standards = len(standards_section.get('applicable_standards', []))
        num_certs = len(standards_section.get('certifications', []))
        confidence = standards_section.get('confidence', 0.0)

        logger.info(f"[PPI_WORKFLOW] Schema enriched: {fields_populated} fields populated, "
                   f"{num_standards} standards, {num_certs} certifications")
        logger.info(f"[PPI_WORKFLOW] Standards enrichment confidence: {confidence:.2f}")

        return {
            "schema": enriched_schema,
            "schema_saved": True,
            "messages": [AIMessage(content=f"Schema enriched: {fields_populated} fields populated, "
                                          f"{num_standards} standards, {num_certs} certifications "
                                          f"(confidence: {confidence:.2f})")]
        }

    except Exception as e:
        logger.error(f"[PPI_WORKFLOW] Standards enrichment error: {e}", exc_info=True)
        # On error, keep original schema without standards
        logger.warning("[PPI_WORKFLOW] Proceeding with schema without standards enrichment")
        return {
            "schema_saved": True,  # Save original schema
            "messages": [AIMessage(content=f"Standards enrichment failed (using original schema): {e}")]
        }



# ============================================================================
# CONDITIONAL EDGES
# ============================================================================

def should_continue_vendors(state: PPIState) -> str:
    """
    Decide whether to continue processing vendors or move to schema generation
    """
    current_index = state.get("current_vendor_index", 0)
    vendor_count = state.get("vendor_count", 0)
    
    if current_index < vendor_count:
        # More vendors to process
        return "search_pdfs"
    else:
        # All vendors processed, generate schema
        return "generate_schema"


def check_vendors_found(state: PPIState) -> str:
    """
    Check if vendors were found
    """
    vendor_count = state.get("vendor_count", 0)
    
    if vendor_count > 0:
        return "search_pdfs"
    else:
        return "end_error"


# ============================================================================
# WORKFLOW BUILDER
# ============================================================================

def create_ppi_workflow() -> StateGraph:
    """
    Create the PPI LangGraph workflow

    Flow:
    discover_vendors → [if vendors found] → search_pdfs → download_pdfs →
    extract_data → store_vendor_data → [loop for each vendor] →
    generate_schema → enrich_with_standards → END

    The enrich_with_standards node queries Standards RAG to add:
    - Applicable engineering standards (ISO, IEC, API, etc.)
    - Required certifications (SIL, ATEX, CE, etc.)
    - Safety, calibration, and environmental requirements
    """
    workflow = StateGraph(PPIState)

    # Add nodes
    workflow.add_node("discover_vendors", discover_vendors_node)
    workflow.add_node("search_pdfs", search_pdfs_node)
    workflow.add_node("download_pdfs", download_pdfs_node)
    workflow.add_node("extract_data", extract_data_node)
    workflow.add_node("store_vendor_data", store_vendor_data_node)
    workflow.add_node("generate_schema", generate_schema_node)
    workflow.add_node("enrich_with_standards", enrich_schema_with_standards_node)  # NEW

    # Set entry point
    workflow.set_entry_point("discover_vendors")

    # Add edges
    workflow.add_conditional_edges(
        "discover_vendors",
        check_vendors_found,
        {
            "search_pdfs": "search_pdfs",
            "end_error": END
        }
    )

    workflow.add_edge("search_pdfs", "download_pdfs")
    workflow.add_edge("download_pdfs", "extract_data")
    workflow.add_edge("extract_data", "store_vendor_data")

    # After storing vendor data, either process next vendor or generate schema
    workflow.add_conditional_edges(
        "store_vendor_data",
        should_continue_vendors,
        {
            "search_pdfs": "search_pdfs",
            "generate_schema": "generate_schema"
        }
    )

    # After generating schema, enrich with standards
    workflow.add_edge("generate_schema", "enrich_with_standards")  # NEW
    workflow.add_edge("enrich_with_standards", END)  # NEW

    return workflow


def compile_ppi_workflow():
    """Compile the PPI workflow for execution"""
    workflow = create_ppi_workflow()
    return workflow.compile()


# ============================================================================
# WORKFLOW EXECUTION
# ============================================================================

def run_ppi_workflow(product_type: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Execute the PPI workflow for a product type.
    
    Args:
        product_type: Product type to generate schema for
        session_id: Optional session identifier
        
    Returns:
        Workflow result with schema and status
    """
    logger.info(f"[PPI_WORKFLOW] Starting workflow for: {product_type}")
    
    try:
        # Compile workflow
        app = compile_ppi_workflow()
        
        # Initialize state
        initial_state = {
            "product_type": product_type,
            "session_id": session_id,
            "vendors": [],
            "vendor_count": 0,
            "current_vendor_index": 0,
            "pdfs_found": [],
            "pdfs_downloaded": [],
            "extracted_data": [],
            "schema": None,
            "schema_saved": False,
            "vendor_data_stored": [],
            "success": False,
            "error": None,
            "messages": [HumanMessage(content=f"Generate schema for {product_type}")]
        }
        
        # Execute workflow
        final_state = app.invoke(initial_state)
        
        logger.info(f"[PPI_WORKFLOW] Workflow completed. Success: {final_state.get('success')}")
        
        return {
            "success": final_state.get("success", False),
            "product_type": product_type,
            "schema": final_state.get("schema"),
            "vendors_processed": len(final_state.get("vendor_data_stored", [])),
            "pdfs_processed": len(final_state.get("pdfs_downloaded", [])),
            "error": final_state.get("error")
        }
        
    except Exception as e:
        logger.error(f"[PPI_WORKFLOW] Workflow execution failed: {e}", exc_info=True)
        return {
            "success": False,
            "product_type": product_type,
            "schema": None,
            "error": str(e)
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*70)
    print("PPI LANGGRAPH WORKFLOW TEST")
    print("="*70)
    
    result = run_ppi_workflow("Humidity Transmitter")
    
    print(f"\nSuccess: {result['success']}")
    print(f"Vendors Processed: {result.get('vendors_processed', 0)}")
    print(f"PDFs Processed: {result.get('pdfs_processed', 0)}")
    
    if result.get('schema'):
        print(f"Schema Generated: Yes")
        print(f"  Mandatory: {len(result['schema'].get('mandatory_requirements', {}))}")
        print(f"  Optional: {len(result['schema'].get('optional_requirements', {}))}")
    else:
        print(f"Schema Generated: No")
        if result.get('error'):
            print(f"Error: {result['error']}")
