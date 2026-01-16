"""
PPI (Potential Product Index) Tools for LangGraph Workflow
============================================================

LangChain tools for the PPI workflow that generates schemas and vendor data
when product data is not found in the database.

Tools:
1. discover_vendors_tool - Discovers top 5 vendors for a product type
2. search_pdfs_tool - Searches for PDF datasheets with tier fallback
3. download_and_store_pdf_tool - Downloads PDF and stores in Azure Blob
4. extract_pdf_data_tool - Extracts structured data from PDF
5. generate_schema_tool - Generates schema from vendor data
6. store_vendor_data_tool - Stores vendor JSON data in Azure Blob
"""

import io
import json
import logging
import os
import requests
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


# ============================================================================
# INPUT SCHEMAS
# ============================================================================

class DiscoverVendorsInput(BaseModel):
    """Input for discovering vendors"""
    product_type: str = Field(description="Product type to find vendors for")


class SearchPdfsInput(BaseModel):
    """Input for PDF search"""
    vendor: str = Field(description="Vendor name")
    product_type: str = Field(description="Product type")
    model_family: Optional[str] = Field(default=None, description="Optional model family")


class DownloadPdfInput(BaseModel):
    """Input for PDF download"""
    pdf_url: str = Field(description="URL of the PDF to download")
    vendor: str = Field(description="Vendor name")
    product_type: str = Field(description="Product type")
    model_family: Optional[str] = Field(default=None, description="Model family")


class ExtractPdfDataInput(BaseModel):
    """Input for PDF extraction"""
    pdf_bytes: bytes = Field(description="PDF file bytes")
    vendor: str = Field(description="Vendor name")
    product_type: str = Field(description="Product type")


class GenerateSchemaInput(BaseModel):
    """Input for schema generation"""
    product_type: str = Field(description="Product type")
    vendor_data: List[Dict[str, Any]] = Field(description="List of vendor extracted data")


class StoreVendorDataInput(BaseModel):
    """Input for storing vendor data"""
    vendor_data: Dict[str, Any] = Field(description="Vendor data to store")
    product_type: str = Field(description="Product type")


# ============================================================================
# PPI TOOLS
# ============================================================================

@tool("discover_vendors", args_schema=DiscoverVendorsInput)
def discover_vendors_tool(product_type: str) -> Dict[str, Any]:
    """
    Discover top 5 vendors for a given product type using LLM.
    
    Returns vendor names and their model families.
    """
    try:
        logger.info(f"[PPI_TOOL] Discovering vendors for: {product_type}")
        
        from loading import discover_top_vendors
        from llm_fallback import create_llm_with_fallback
        
        llm = create_llm_with_fallback(
            model="gemini-2.5-flash",
            temperature=0.3,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        vendors = discover_top_vendors(product_type, llm)
        
        if not vendors:
            logger.warning(f"[PPI_TOOL] No vendors discovered for {product_type}")
            return {
                "success": False,
                "product_type": product_type,
                "vendors": [],
                "error": "No vendors found"
            }
        
        logger.info(f"[PPI_TOOL] Discovered {len(vendors)} vendors")
        return {
            "success": True,
            "product_type": product_type,
            "vendors": vendors[:5],  # Top 5
            "vendor_count": len(vendors[:5])
        }
        
    except Exception as e:
        logger.error(f"[PPI_TOOL] Vendor discovery failed: {e}")
        return {
            "success": False,
            "product_type": product_type,
            "vendors": [],
            "error": str(e)
        }


@tool("search_pdfs", args_schema=SearchPdfsInput)
def search_pdfs_tool(vendor: str, product_type: str, model_family: Optional[str] = None) -> Dict[str, Any]:
    """
    Search for PDF datasheets using tier-based fallback.
    
    Tier 1: Serper API
    Tier 2: SerpAPI
    Tier 3: Google CSE
    """
    try:
        logger.info(f"[PPI_TOOL] Searching PDFs for {vendor} - {product_type}")
        
        # Import the search tool
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))
        from search_tools import search_pdf_datasheets_tool
        
        # Use .invoke() for LangChain StructuredTool
        result = search_pdf_datasheets_tool.invoke({
            "vendor": vendor,
            "product_type": product_type,
            "model_family": model_family
        })
        
        if result.get('success'):
            pdfs = result.get('pdfs', [])
            logger.info(f"[PPI_TOOL] Found {len(pdfs)} PDFs for {vendor}")
            return {
                "success": True,
                "vendor": vendor,
                "product_type": product_type,
                "pdfs": pdfs[:5],  # Top 5 PDFs
                "pdf_count": len(pdfs[:5]),
                "tier_used": result.get('tier_used', 'unknown')
            }
        else:
            return {
                "success": False,
                "vendor": vendor,
                "product_type": product_type,
                "pdfs": [],
                "error": result.get('error', 'Search failed')
            }
            
    except Exception as e:
        logger.error(f"[PPI_TOOL] PDF search failed: {e}")
        return {
            "success": False,
            "vendor": vendor,
            "product_type": product_type,
            "pdfs": [],
            "error": str(e)
        }


@tool("download_and_store_pdf")
def download_and_store_pdf_tool(
    pdf_url: str,
    vendor: str,
    product_type: str,
    model_family: Optional[str] = None
) -> Dict[str, Any]:
    """
    Download a PDF from URL and store it in Azure Blob Storage.
    
    Returns the file ID and PDF bytes for further processing.
    """
    try:
        logger.info(f"[PPI_TOOL] Downloading PDF: {pdf_url[:60]}...")
        
        # Download PDF
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(pdf_url, timeout=30, headers=headers, stream=True)
        response.raise_for_status()
        
        pdf_data = b''
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                pdf_data += chunk
        
        # Validate PDF
        if len(pdf_data) < 1024:
            return {
                "success": False,
                "vendor": vendor,
                "error": "PDF too small"
            }
        
        if not pdf_data.startswith(b'%PDF'):
            return {
                "success": False,
                "vendor": vendor,
                "error": "Invalid PDF format"
            }
        
        logger.info(f"[PPI_TOOL] Downloaded {len(pdf_data)} bytes")
        
        # Store in Azure Blob
        from azure_blob_utils import azure_blob_file_manager
        
        # Generate filename
        import re
        filename = os.path.basename(pdf_url.split('?')[0]) or f"{vendor}_{product_type}.pdf"
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        metadata = {
            'collection_type': 'documents',
            'file_type': 'pdf',
            'product_type': product_type.replace(' ', '_'),
            'vendor_name': vendor.replace(' ', '_'),
            'model_family': model_family or '',
            'filename': filename,
            'source_url': pdf_url,
            'file_size': len(pdf_data)
        }
        
        file_id = azure_blob_file_manager.upload_to_azure(pdf_data, metadata)
        
        logger.info(f"[PPI_TOOL] PDF stored with ID: {file_id}")
        
        return {
            "success": True,
            "vendor": vendor,
            "product_type": product_type,
            "file_id": file_id,
            "filename": filename,
            "file_size": len(pdf_data),
            "pdf_bytes": pdf_data  # Include bytes for extraction
        }
        
    except Exception as e:
        logger.error(f"[PPI_TOOL] PDF download/store failed: {e}")
        return {
            "success": False,
            "vendor": vendor,
            "error": str(e)
        }


@tool("extract_pdf_data")
def extract_pdf_data_tool(
    pdf_bytes: bytes,
    vendor: str,
    product_type: str
) -> Dict[str, Any]:
    """
    Extract structured product data from PDF using LLM.
    
    Returns extracted specifications and model information.
    """
    try:
        logger.info(f"[PPI_TOOL] Extracting data from PDF for {vendor}")
        
        from test import extract_data_from_pdf, send_to_language_model
        
        # Convert to BytesIO
        pdf_file = io.BytesIO(pdf_bytes)
        
        # Extract text chunks
        text_chunks = extract_data_from_pdf(pdf_file)
        
        if not text_chunks or len(text_chunks) == 0:
            return {
                "success": False,
                "vendor": vendor,
                "extracted_data": [],
                "error": "No text extracted from PDF"
            }
        
        logger.info(f"[PPI_TOOL] Extracted {len(text_chunks)} text chunks")
        
        # Use LLM to extract structured data
        extracted_data = send_to_language_model(text_chunks)
        
        # Flatten results
        flattened = []
        for item in extracted_data:
            if isinstance(item, list):
                flattened.extend(item)
            else:
                flattened.append(item)
        
        logger.info(f"[PPI_TOOL] Extracted {len(flattened)} product entries")
        
        return {
            "success": True,
            "vendor": vendor,
            "product_type": product_type,
            "extracted_data": flattened,
            "entry_count": len(flattened)
        }
        
    except Exception as e:
        logger.error(f"[PPI_TOOL] PDF extraction failed: {e}")
        return {
            "success": False,
            "vendor": vendor,
            "extracted_data": [],
            "error": str(e)
        }


@tool("generate_schema", args_schema=GenerateSchemaInput)
def generate_schema_tool(product_type: str, vendor_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate a product schema from extracted vendor data using LLM.
    
    Analyzes vendor specifications to create mandatory and optional requirements.
    """
    try:
        logger.info(f"[PPI_TOOL] Generating schema for {product_type} from {len(vendor_data)} vendors")
        
        from loading import create_schema_from_vendor_data, _save_schema_to_specs
        from llm_fallback import create_llm_with_fallback
        
        llm = create_llm_with_fallback(
            model="gemini-2.5-flash",
            temperature=0.2,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Prepare vendor info for schema generation
        vendors_with_families = []
        for data in vendor_data:
            if data.get('success'):
                vendors_with_families.append({
                    'vendor': data.get('vendor', 'Unknown'),
                    'model_families': []
                })
        
        if not vendors_with_families:
            return {
                "success": False,
                "product_type": product_type,
                "schema": None,
                "error": "No successful vendor data for schema generation"
            }
        
        # Generate schema
        schema = create_schema_from_vendor_data(product_type, vendors_with_families, llm)
        
        if schema and schema.get('mandatory_requirements'):
            # Save schema to Azure Blob
            schema_path = _save_schema_to_specs(product_type, schema)
            
            logger.info(f"[PPI_TOOL] Schema generated and saved: {schema_path}")
            
            return {
                "success": True,
                "product_type": product_type,
                "schema": schema,
                "schema_path": schema_path,
                "mandatory_count": len(schema.get('mandatory_requirements', {})),
                "optional_count": len(schema.get('optional_requirements', {}))
            }
        else:
            return {
                "success": False,
                "product_type": product_type,
                "schema": None,
                "error": "Generated schema is empty or invalid"
            }
            
    except Exception as e:
        logger.error(f"[PPI_TOOL] Schema generation failed: {e}")
        return {
            "success": False,
            "product_type": product_type,
            "schema": None,
            "error": str(e)
        }


@tool("store_vendor_data", args_schema=StoreVendorDataInput)
def store_vendor_data_tool(vendor_data: Dict[str, Any], product_type: str) -> Dict[str, Any]:
    """
    Store vendor product data in Azure Blob Storage.
    
    Stores as JSON in the vendors collection for later analysis.
    """
    try:
        vendor_name = vendor_data.get('vendor', 'Unknown')
        logger.info(f"[PPI_TOOL] Storing vendor data: {vendor_name}")
        
        from azure_blob_utils import azure_blob_file_manager
        
        # Use underscores for filename (Azure Blob paths)
        safe_vendor = vendor_name.replace(' ', '_').replace('+', '_')
        safe_product_type_path = product_type.lower().replace(' ', '_')
        
        # Keep original format for metadata (for search matching)
        metadata = {
            'collection_type': 'vendors',
            'product_type': product_type.lower(),  # Keep spaces for matching
            'vendor_name': safe_vendor,
            'filename': f"{safe_vendor}_{safe_product_type_path}.json",
            'file_type': 'json'
        }
        
        # Store in Azure Blob
        doc_id = azure_blob_file_manager.upload_json_data(vendor_data, metadata)
        
        logger.info(f"[PPI_TOOL] Vendor data stored with ID: {doc_id}")
        
        return {
            "success": True,
            "vendor": vendor_name,
            "product_type": product_type,
            "document_id": doc_id
        }
        
    except Exception as e:
        logger.error(f"[PPI_TOOL] Vendor data storage failed: {e}")
        return {
            "success": False,
            "vendor": vendor_data.get('vendor', 'Unknown'),
            "error": str(e)
        }


# ============================================================================
# TOOL REGISTRY
# ============================================================================

PPI_TOOLS = [
    discover_vendors_tool,
    search_pdfs_tool,
    download_and_store_pdf_tool,
    extract_pdf_data_tool,
    generate_schema_tool,
    store_vendor_data_tool
]


def get_ppi_tools() -> List:
    """Get all PPI tools for use in LangGraph workflow"""
    return PPI_TOOLS
