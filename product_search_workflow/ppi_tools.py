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
import time
import threading
import random
import requests
import httpx
import socket
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# =============================================================================
# FIX #16: CONNECTION TIMEOUT CONFIGURATION
# =============================================================================
# Resolves network hang issues causing 436+ second delays
# Default httpx timeout was unlimited, causing thread stalls

_HTTP_CLIENT = None
_HTTP_CLIENT_LOCK = threading.Lock()

def get_http_client() -> httpx.Client:
    """Get or create HTTP client with proper timeout configuration"""
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None:
        with _HTTP_CLIENT_LOCK:
            if _HTTP_CLIENT is None:
                _HTTP_CLIENT = httpx.Client(
                    timeout=60.0,  # 60 second timeout per request
                    limits=httpx.Limits(
                        max_connections=10,
                        max_keepalive_connections=5,
                    ),
                    follow_redirects=True,
                    verify=False  # SSL verification disabled for PDFs
                )
                logger.info("[FIX16] Created HTTP client: timeout=60s, max_connections=10")
    return _HTTP_CLIENT


def validate_dns_resolution(url: str, timeout: int = 5) -> bool:
    """
    FIX #4: Validate DNS resolution before making network request.

    Prevents DNS resolution failures and connection timeouts by checking
    if the domain can be resolved before attempting the actual request.

    Args:
        url: URL to validate
        timeout: DNS lookup timeout (seconds)

    Returns:
        bool: True if DNS resolves, False otherwise
    """
    try:
        # Extract hostname from URL
        from urllib.parse import urlparse
        parsed = urlparse(url)
        hostname = parsed.netloc.split(':')[0]  # Remove port if present

        if not hostname:
            logger.warning(f"[FIX4] Invalid URL format: {url}")
            return False

        # Attempt DNS resolution
        logger.debug(f"[FIX4] Validating DNS for hostname: {hostname}")
        socket.setdefaulttimeout(timeout)
        try:
            ip = socket.gethostbyname(hostname)
            logger.info(f"[FIX4] ✅ DNS resolved: {hostname} -> {ip}")
            return True
        except socket.gaierror as e:
            logger.warning(f"[FIX4] ❌ DNS resolution failed for {hostname}: {e}")
            return False

    except Exception as e:
        logger.warning(f"[FIX4] DNS validation error: {e}")
        return False
    finally:
        socket.setdefaulttimeout(None)


def download_with_retry(
    url: str,
    headers: Dict[str, str],
    max_retries: int = 2,
    timeout: int = 60
) -> Optional[bytes]:
    """
    FIX #16: Download with retry logic and exponential backoff.
    FIX #4: Enhanced with DNS validation before requests.

    Replaces the simple requests.get() with intelligent retry handling
    to avoid 200+ second hangs on network failures.

    Args:
        url: PDF URL to download
        headers: HTTP headers
        max_retries: Number of retry attempts (2 = 3 total attempts)
        timeout: Timeout per attempt (seconds)

    Returns:
        PDF bytes or None if all attempts fail
    """
    # FIX #4: Validate DNS before attempting download
    logger.info(f"[FIX4] Performing pre-request DNS validation...")
    if not validate_dns_resolution(url):
        logger.error(f"[FIX4] DNS validation failed for URL: {url[:60]}...")
        return None

    client = get_http_client()
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            logger.info(f"[FIX16] Downloading (attempt {attempt+1}/{max_retries+1}): {url[:60]}...")

            response = client.get(
                url,
                headers=headers,
                timeout=timeout
            )
            response.raise_for_status()

            # Collect response bytes
            pdf_data = b''
            for chunk in response.iter_bytes(chunk_size=8192):
                if chunk:
                    pdf_data += chunk

            logger.info(f"[FIX16] ✅ Downloaded {len(pdf_data)} bytes on attempt {attempt+1}")
            return pdf_data

        except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as e:
            last_error = e
            if attempt < max_retries:
                # FIX #4: More aggressive backoff for network errors
                wait_time = (2 ** attempt) + (time.time() % 1.0)  # 1-2s, 2-3s, 4-5s with jitter
                logger.warning(
                    f"[FIX4] Network error on attempt {attempt+1}: {type(e).__name__}. "
                    f"Retrying in {wait_time:.1f}s..."
                )
                time.sleep(wait_time)
            else:
                logger.error(f"[FIX16] ❌ All {max_retries+1} attempts failed: {last_error}")

        except Exception as e:
            logger.error(f"[FIX16] Unexpected error on attempt {attempt+1}: {e}")
            last_error = e
            break

    return None


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


def rank_pdfs_by_relevance(pdfs: List[Dict[str, Any]], product_type: str) -> List[Dict[str, Any]]:
    """
    🔥 FIX #14: Rank PDFs by relevance to product type

    Scoring factors:
    - Filename/URL match with product_type (high weight)
    - Title relevance
    - File size (50KB-5MB is optimal)
    - URL domain trust (vendor domain better)

    Returns PDFs sorted by relevance score (highest first)
    """
    scored_pdfs = []
    product_lower = product_type.lower()

    for pdf in pdfs:
        score = 0.0
        url = pdf.get("url", "").lower()
        title = pdf.get("title", "").lower()

        # Exact product name match in URL (high weight)
        if product_lower in url:
            score += 3.0
        elif any(word in url for word in product_lower.split()):
            score += 2.0

        # Title match (medium weight)
        if product_lower in title:
            score += 2.0
        elif any(word in title for word in product_lower.split()):
            score += 1.0

        # File size preference: 50KB-5MB is good for datasheets
        file_size = pdf.get("file_size", 0)
        if 50000 < file_size < 5000000:
            score += 1.0
        elif file_size < 50000:
            score -= 1.0  # Too small, probably not detailed

        # URL domain quality
        if "datasheet" in url or "specification" in url:
            score += 1.0
        if "pdf" in url:
            score += 0.5

        pdf["relevance_score"] = score
        scored_pdfs.append(pdf)

    # Sort by relevance score descending
    scored_pdfs.sort(key=lambda x: x["relevance_score"], reverse=True)

    logger.info(f"[FIX14] Ranked {len(scored_pdfs)} PDFs by relevance for {product_type}")
    for i, pdf in enumerate(scored_pdfs[:3]):
        logger.info(f"  [FIX14] #{i+1}: {pdf.get('title', 'Unknown')[:50]}... (score: {pdf['relevance_score']:.1f})")

    return scored_pdfs


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

    Uses FIX #16 retry logic to handle network failures gracefully.
    Returns the file ID and PDF bytes for further processing.
    """
    try:
        logger.info(f"[PPI_TOOL] Downloading PDF: {pdf_url[:60]}...")

        # Download PDF with FIX #16 retry logic
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        # Use new retry logic instead of simple requests.get()
        pdf_data = download_with_retry(pdf_url, headers, max_retries=2, timeout=60)

        if not pdf_data:
            return {
                "success": False,
                "vendor": vendor,
                "error": "PDF download failed after 3 attempts"
            }

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
    
    FIX #18: Added retry logic with exponential backoff for network errors.
    
    Returns extracted specifications and model information.
    """
    # FIX #18: Retry configuration
    MAX_RETRIES = 2
    BACKOFF_BASE = 2  # seconds
    
    from test import extract_data_from_pdf, send_to_language_model
    
    for attempt in range(MAX_RETRIES + 1):
        try:
            logger.info(f"[PPI_TOOL] Extracting data from PDF for {vendor} (attempt {attempt + 1}/{MAX_RETRIES + 1})")
            
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
            
        except (ConnectionError, TimeoutError, OSError) as e:
            # FIX #18: Network errors get retried with exponential backoff
            if attempt < MAX_RETRIES:
                wait_time = (BACKOFF_BASE ** attempt) + random.uniform(0, 1)
                logger.warning(
                    f"[FIX18] Network error on attempt {attempt + 1}/{MAX_RETRIES + 1}: {type(e).__name__}: {e}. "
                    f"Retrying in {wait_time:.1f}s..."
                )
                time.sleep(wait_time)
            else:
                logger.error(f"[FIX18] PDF extraction failed after {MAX_RETRIES + 1} attempts: {e}")
                return {
                    "success": False,
                    "vendor": vendor,
                    "extracted_data": [],
                    "error": f"Network failure after {MAX_RETRIES + 1} retries: {str(e)}"
                }
                
        except Exception as e:
            logger.error(f"[PPI_TOOL] PDF extraction failed: {e}")
            return {
                "success": False,
                "vendor": vendor,
                "extracted_data": [],
                "error": str(e)
            }
    
    # Should not reach here, but just in case
    return {
        "success": False,
        "vendor": vendor,
        "extracted_data": [],
        "error": "Unknown error during PDF extraction"
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
        
        # FIX #19: Accept partial vendor data - check for actual content, not 'success' flag
        # The extracted data entries contain specifications, not vendor status objects
        vendors_with_families = []
        seen_vendors = set()
        
        for data in vendor_data:
            vendor_name = data.get('vendor', 'Unknown')
            
            # Skip duplicate vendors
            if vendor_name in seen_vendors:
                continue
                
            # Check for actual extracted content (specs, model info, etc.)
            # instead of relying on 'success' flag which may not be present
            has_content = (
                data.get('specifications') or 
                data.get('model') or 
                data.get('product_name') or
                data.get('data') or  # Sometimes data is nested under 'data' key
                data.get('extracted_data') or
                len([k for k in data.keys() if k not in ['vendor', 'success', 'error', 'product_type']]) > 0
            )
            
            if has_content:
                vendors_with_families.append({
                    'vendor': vendor_name,
                    'model_families': []
                })
                seen_vendors.add(vendor_name)
                logger.info(f"[FIX19] ✅ Vendor '{vendor_name}' has extractable content")
        
        # FIX #19: Lower threshold - proceed with at least 1 vendor with data
        MIN_VENDORS = 1
        if len(vendors_with_families) < MIN_VENDORS:
            logger.warning(f"[FIX19] Insufficient vendor data ({len(vendors_with_families)} < {MIN_VENDORS} required)")
            return {
                "success": False,
                "product_type": product_type,
                "schema": None,
                "error": f"Insufficient vendor data ({len(vendors_with_families)} vendors with content, need {MIN_VENDORS})"
            }
        
        logger.info(f"[FIX19] Proceeding with {len(vendors_with_families)} vendors (partial data OK)")
        
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
