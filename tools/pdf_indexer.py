# tools/pdf_indexer.py
# PDF Document Indexer for Index RAG
# Extracts and indexes PDF documents from Azure Blob Storage

import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# ============================================================================
# PDF TEXT CACHE
# ============================================================================

# Cache for extracted PDF text to avoid re-extraction
_pdf_text_cache: Dict[str, str] = {}
_pdf_metadata_cache: Dict[str, Dict[str, Any]] = {}


def get_cached_pdf_text(blob_path: str) -> Optional[str]:
    """Get cached PDF text if available."""
    return _pdf_text_cache.get(blob_path)


def cache_pdf_text(blob_path: str, text: str, metadata: Dict[str, Any] = None):
    """Cache extracted PDF text."""
    _pdf_text_cache[blob_path] = text
    if metadata:
        _pdf_metadata_cache[blob_path] = metadata
    logger.debug(f"[PDFIndexer] Cached PDF text for: {blob_path}")


def clear_pdf_cache():
    """Clear the PDF text cache."""
    global _pdf_text_cache, _pdf_metadata_cache
    _pdf_text_cache = {}
    _pdf_metadata_cache = {}
    logger.info("[PDFIndexer] PDF cache cleared")


# ============================================================================
# PDF LISTING
# ============================================================================

def list_pdf_blobs(
    product_types: List[str],
    vendors: List[str] = None,
    max_pdfs: int = 50
) -> List[Dict[str, Any]]:
    """
    List PDF files in Azure Blob Storage for given product types and vendors.
    
    Searches in:
    - {base_path}/files/{product_type}/{vendor}/*.pdf
    - {base_path}/vendors/{product_type}/{vendor}/datasheets/*.pdf
    
    Args:
        product_types: Product types to search
        vendors: Optional vendor filter
        max_pdfs: Maximum PDFs to return
        
    Returns:
        List of PDF blob metadata
    """
    logger.info(f"[PDFIndexer] Listing PDFs for product_types={product_types}, vendors={vendors}")
    
    try:
        from azure_blob_config import get_azure_blob_connection, Collections
        
        conn = get_azure_blob_connection()
        container_client = conn['container_client']
        base_path = conn['base_path']
        
        pdf_blobs = []
        
        # Search paths to check
        search_paths = []
        
        for pt in product_types:
            pt_normalized = pt.replace("_", " ").lower()
            
            if vendors:
                for vendor in vendors:
                    vendor_normalized = vendor.lower().strip()
                    # Path 1: files collection
                    search_paths.append(f"{base_path}/{Collections.FILES}/{pt_normalized}/{vendor_normalized}")
                    # Path 2: vendors collection with datasheets subfolder
                    search_paths.append(f"{base_path}/{Collections.VENDORS}/{pt_normalized}/{vendor_normalized}/datasheets")
                    # Path 3: vendors collection (PDFs at vendor level)
                    search_paths.append(f"{base_path}/{Collections.VENDORS}/{pt_normalized}/{vendor_normalized}")
            else:
                # Search all vendors for this product type
                search_paths.append(f"{base_path}/{Collections.FILES}/{pt_normalized}")
                search_paths.append(f"{base_path}/{Collections.VENDORS}/{pt_normalized}")
        
        # List blobs from each path
        for prefix in search_paths:
            try:
                blobs = container_client.list_blobs(
                    name_starts_with=prefix,
                    include=['metadata']
                )
                
                for blob in blobs:
                    # Only include PDF files
                    if blob.name.endswith('.pdf'):
                        # Extract metadata from path
                        path_parts = blob.name.replace(base_path + "/", "").split("/")
                        
                        pdf_info = {
                            "blob_path": blob.name,
                            "filename": blob.name.split("/")[-1],
                            "size": blob.size,
                            "last_modified": blob.last_modified.isoformat() if blob.last_modified else None,
                            "metadata": dict(blob.metadata) if blob.metadata else {},
                            "collection": path_parts[0] if len(path_parts) > 0 else "unknown",
                            "product_type": path_parts[1] if len(path_parts) > 1 else "unknown",
                            "vendor": path_parts[2] if len(path_parts) > 2 else "unknown",
                        }
                        
                        pdf_blobs.append(pdf_info)
                        
                        if len(pdf_blobs) >= max_pdfs:
                            break
                            
            except Exception as e:
                logger.debug(f"[PDFIndexer] No blobs at {prefix}: {e}")
                continue
            
            if len(pdf_blobs) >= max_pdfs:
                break
        
        # Deduplicate by blob_path
        seen = set()
        unique_pdfs = []
        for pdf in pdf_blobs:
            if pdf["blob_path"] not in seen:
                seen.add(pdf["blob_path"])
                unique_pdfs.append(pdf)
        
        logger.info(f"[PDFIndexer] Found {len(unique_pdfs)} unique PDFs")
        return unique_pdfs[:max_pdfs]
        
    except Exception as e:
        logger.error(f"[PDFIndexer] Error listing PDFs: {e}", exc_info=True)
        return []


# ============================================================================
# PDF TEXT EXTRACTION
# ============================================================================

def extract_pdf_text(blob_path: str) -> Optional[str]:
    """
    Extract text from a PDF in Azure Blob Storage.
    
    Uses cache to avoid re-extraction.
    
    Args:
        blob_path: Full blob path to PDF
        
    Returns:
        Extracted text or None if extraction fails
    """
    # Check cache first
    cached = get_cached_pdf_text(blob_path)
    if cached:
        logger.debug(f"[PDFIndexer] Using cached text for: {blob_path}")
        return cached
    
    logger.info(f"[PDFIndexer] Extracting text from: {blob_path}")
    
    try:
        from azure_blob_config import get_azure_blob_connection
        from file_extraction_utils import extract_text_from_pdf
        
        conn = get_azure_blob_connection()
        container_client = conn['container_client']
        
        # Download PDF bytes
        blob_client = container_client.get_blob_client(blob_path)
        pdf_bytes = blob_client.download_blob().readall()
        
        # Extract text using existing utility
        extracted_text = extract_text_from_pdf(pdf_bytes)
        
        if extracted_text and len(extracted_text.strip()) > 100:
            # Cache the result
            cache_pdf_text(blob_path, extracted_text, {
                "extraction_time": datetime.utcnow().isoformat(),
                "text_length": len(extracted_text)
            })
            
            logger.info(f"[PDFIndexer] Extracted {len(extracted_text)} chars from PDF")
            return extracted_text
        else:
            logger.warning(f"[PDFIndexer] PDF extraction returned minimal text: {blob_path}")
            return None
            
    except Exception as e:
        logger.error(f"[PDFIndexer] PDF extraction failed for {blob_path}: {e}")
        return None


# ============================================================================
# PDF CONTENT SEARCH
# ============================================================================

def search_pdf_content(
    keywords: List[str],
    product_types: List[str],
    vendors: List[str] = None,
    max_pdfs_per_vendor: int = 3
) -> List[Dict[str, Any]]:
    """
    Search PDF content for keywords.
    
    Steps:
    1. List PDFs matching product type/vendor
    2. Extract text from each PDF (with caching)
    3. Score PDFs based on keyword matches
    4. Return top matches per vendor
    
    Args:
        keywords: Search keywords
        product_types: Product types to search
        vendors: Optional vendor filter
        max_pdfs_per_vendor: Max PDFs to return per vendor
        
    Returns:
        List of matching PDFs with extracted text
    """
    start_time = time.time()
    logger.info(f"[PDFIndexer] Searching PDFs for keywords: {keywords}")
    
    if not keywords:
        return []
    
    # Step 1: List relevant PDFs
    pdf_blobs = list_pdf_blobs(
        product_types=product_types,
        vendors=vendors,
        max_pdfs=50  # Get more to filter
    )
    
    if not pdf_blobs:
        logger.info("[PDFIndexer] No PDFs found for search")
        return []
    
    # Step 2: Extract and score PDFs
    scored_pdfs = []
    keywords_lower = [kw.lower() for kw in keywords]
    
    for pdf_info in pdf_blobs:
        blob_path = pdf_info["blob_path"]
        
        # Extract text
        text = extract_pdf_text(blob_path)
        if not text:
            continue
        
        # Score based on keyword matches
        text_lower = text.lower()
        matches = sum(1 for kw in keywords_lower if kw in text_lower)
        
        if matches > 0:
            pdf_info["extracted_text"] = text
            pdf_info["text_length"] = len(text)
            pdf_info["keyword_matches"] = matches
            pdf_info["relevance_score"] = matches / len(keywords_lower)
            scored_pdfs.append(pdf_info)
    
    # Step 3: Sort by score
    scored_pdfs.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    
    # Step 4: Get top N per vendor
    vendor_counts: Dict[str, int] = {}
    filtered_pdfs = []
    
    for pdf in scored_pdfs:
        vendor = pdf.get("vendor", "unknown").lower()
        current_count = vendor_counts.get(vendor, 0)
        
        if current_count < max_pdfs_per_vendor:
            vendor_counts[vendor] = current_count + 1
            filtered_pdfs.append(pdf)
    
    elapsed = time.time() - start_time
    logger.info(f"[PDFIndexer] PDF search complete: {len(filtered_pdfs)} matches in {elapsed:.2f}s")
    
    return filtered_pdfs


# ============================================================================
# MAIN INDEX FUNCTION
# ============================================================================

def index_pdf_documents(
    query: str,
    product_types: List[str],
    vendors: List[str] = None,
    keywords: List[str] = None,
    max_pdfs_per_vendor: int = 3,
    top_k: int = 10
) -> Dict[str, Any]:
    """
    Index PDF documents from Azure Blob Storage.
    
    This is the main entry point for PDF indexing in Index RAG.
    
    Args:
        query: User's search query
        product_types: Product types to search
        vendors: Optional vendor filter
        keywords: Optional pre-extracted keywords
        max_pdfs_per_vendor: Max PDFs per vendor
        top_k: Total max PDFs to return
        
    Returns:
        Indexed PDF results
    """
    start_time = time.time()
    logger.info(f"[PDFIndexer] Starting PDF indexing for: {query[:50]}...")
    
    try:
        # Extract keywords if not provided
        if not keywords:
            from tools.metadata_filter import extract_product_types, extract_vendors, extract_models
            
            # Combine extraction results as keywords
            extracted_types = extract_product_types(query)
            extracted_vendors = extract_vendors(query)
            extracted_models = extract_models(query)
            
            keywords = set()
            for pt in extracted_types:
                keywords.update(pt.replace("_", " ").split())
            for v in extracted_vendors:
                keywords.add(v)
            for m in extracted_models:
                keywords.add(m.lower())
            
            # Add query words as additional keywords
            query_words = [w.lower() for w in query.split() if len(w) > 2]
            keywords.update(query_words[:10])  # Limit
            
            keywords = list(keywords)
        
        # Search PDFs
        matching_pdfs = search_pdf_content(
            keywords=keywords,
            product_types=product_types,
            vendors=vendors,
            max_pdfs_per_vendor=max_pdfs_per_vendor
        )
        
        # Format results for Index RAG
        indexed_results = []
        for pdf in matching_pdfs[:top_k]:
            indexed_item = {
                "source": "pdf",
                "vendor": pdf.get("vendor", "Unknown"),
                "product_type": pdf.get("product_type", "Unknown"),
                "filename": pdf.get("filename", ""),
                "blob_path": pdf.get("blob_path", ""),
                "data": {
                    "extracted_text": pdf.get("extracted_text", ""),
                    "text_length": pdf.get("text_length", 0),
                    "keyword_matches": pdf.get("keyword_matches", 0),
                    "relevance_score": pdf.get("relevance_score", 0),
                },
                "relevance_score": pdf.get("relevance_score", 0),
                "indexed_at": datetime.utcnow().isoformat()
            }
            indexed_results.append(indexed_item)
        
        elapsed = time.time() - start_time
        
        result = {
            "success": True,
            "source": "pdf",
            "results": indexed_results,
            "count": len(indexed_results),
            "keywords_used": keywords,
            "elapsed_ms": int(elapsed * 1000)
        }
        
        logger.info(f"[PDFIndexer] PDF indexing complete: {len(indexed_results)} PDFs in {elapsed:.2f}s")
        return result
        
    except Exception as e:
        logger.error(f"[PDFIndexer] PDF indexing failed: {e}", exc_info=True)
        return {
            "success": False,
            "source": "pdf",
            "error": str(e),
            "results": [],
            "count": 0
        }


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'list_pdf_blobs',
    'extract_pdf_text',
    'search_pdf_content',
    'index_pdf_documents',
    'get_cached_pdf_text',
    'cache_pdf_text',
    'clear_pdf_cache'
]
