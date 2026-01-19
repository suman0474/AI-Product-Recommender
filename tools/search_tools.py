# tools/search_tools.py
# Image Search, PDF Search, and Web Search Tools

import json
import logging
import requests
from typing import Dict, Any, List, Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# ============================================================================
# TOOL INPUT SCHEMAS
# ============================================================================

class SearchProductImagesInput(BaseModel):
    """Input for product image search"""
    vendor: str = Field(description="Vendor name")
    product_name: str = Field(description="Product name/model")
    product_type: str = Field(description="Product type")
    model_family: Optional[str] = Field(default=None, description="Model family/series")


class SearchPDFDatasheetsInput(BaseModel):
    """Input for PDF datasheet search"""
    vendor: str = Field(description="Vendor name")
    product_type: str = Field(description="Product type")
    model_family: Optional[str] = Field(default=None, description="Model family/series")


class WebSearchInput(BaseModel):
    """Input for general web search"""
    query: str = Field(description="Search query")
    num_results: int = Field(default=5, description="Number of results to return")


class GetGenericProductImageInput(BaseModel):
    """Input for generic product type image generation"""
    product_type: str = Field(description="Product type name (e.g., 'Pressure Transmitter', 'Temperature Sensor')")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def search_with_serper(query: str, search_type: str = "search") -> List[Dict[str, Any]]:
    """Search using Serper API"""
    try:
        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            return []

        url = f"https://google.serper.dev/{search_type}"
        headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json"
        }
        payload = {"q": query, "num": 10}

        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()

        if search_type == "images":
            return data.get("images", [])
        return data.get("organic", [])

    except Exception as e:
        logger.warning(f"Serper search failed: {e}")
        return []


def search_with_serpapi(query: str, search_type: str = "google") -> List[Dict[str, Any]]:
    """Search using SerpAPI"""
    try:
        from serpapi import GoogleSearch

        api_key = os.getenv("SERPAPI_KEY")
        if not api_key:
            return []

        params = {
            "q": query,
            "api_key": api_key,
            "num": 10
        }

        if search_type == "images":
            params["tbm"] = "isch"

        search = GoogleSearch(params)
        results = search.get_dict()

        if search_type == "images":
            return results.get("images_results", [])
        return results.get("organic_results", [])

    except Exception as e:
        logger.warning(f"SerpAPI search failed: {e}")
        return []


def search_with_google_cse(query: str, search_type: str = "web") -> List[Dict[str, Any]]:
    """Search using Google Custom Search Engine"""
    try:
        from googleapiclient.discovery import build

        api_key = os.getenv("GOOGLE_API_KEY")
        cse_id = os.getenv("GOOGLE_CSE_ID", "066b7345f94f64897")

        if not api_key:
            return []

        service = build("customsearch", "v1", developerKey=api_key)

        params = {
            "q": query,
            "cx": cse_id,
            "num": 10
        }

        if search_type == "images":
            params["searchType"] = "image"

        result = service.cse().list(**params).execute()
        return result.get("items", [])

    except Exception as e:
        logger.warning(f"Google CSE search failed: {e}")
        return []


# ============================================================================
# TOOLS
# ============================================================================

@tool("search_product_images", args_schema=SearchProductImagesInput)
def search_product_images_tool(
    vendor: str,
    product_name: str,
    product_type: str,
    model_family: Optional[str] = None
) -> Dict[str, Any]:
    """
    Search for product images using multi-tier fallback:
    1. Google Custom Search (manufacturer domains)
    2. Serper API
    3. SerpAPI

    Returns best quality images with metadata.
    """
    try:
        # Build search query
        search_terms = [vendor, product_name, product_type]
        if model_family:
            search_terms.append(model_family)
        query = " ".join(search_terms) + " product image"

        images = []

        # Tier 1: Google CSE with manufacturer domain
        vendor_domain = vendor.lower().replace(" ", "") + ".com"
        domain_query = f"site:{vendor_domain} {product_name} {product_type}"
        cse_results = search_with_google_cse(domain_query, "images")

        for img in cse_results[:3]:
            images.append({
                "url": img.get("link"),
                "thumbnail": img.get("image", {}).get("thumbnailLink"),
                "title": img.get("title"),
                "source": "google_cse",
                "quality_score": 90  # Higher score for manufacturer images
            })

        # Tier 2: Serper API
        if len(images) < 3:
            serper_results = search_with_serper(query, "images")
            for img in serper_results[:3]:
                images.append({
                    "url": img.get("imageUrl"),
                    "thumbnail": img.get("thumbnailUrl"),
                    "title": img.get("title"),
                    "source": "serper",
                    "quality_score": 70
                })

        # Tier 3: SerpAPI
        if len(images) < 3:
            serpapi_results = search_with_serpapi(query, "images")
            for img in serpapi_results[:3]:
                images.append({
                    "url": img.get("original"),
                    "thumbnail": img.get("thumbnail"),
                    "title": img.get("title"),
                    "source": "serpapi",
                    "quality_score": 60
                })

        # Sort by quality score
        images.sort(key=lambda x: x.get("quality_score", 0), reverse=True)

        return {
            "success": True,
            "vendor": vendor,
            "product_name": product_name,
            "images": images[:5],  # Return top 5
            "total_found": len(images)
        }

    except Exception as e:
        logger.error(f"Image search failed: {e}")
        return {
            "success": False,
            "vendor": vendor,
            "images": [],
            "error": str(e)
        }


@tool("search_pdf_datasheets", args_schema=SearchPDFDatasheetsInput)
def search_pdf_datasheets_tool(
    vendor: str,
    product_type: str,
    model_family: Optional[str] = None
) -> Dict[str, Any]:
    """
    Search for PDF datasheets using multi-tier fallback.
    Returns PDF URLs with metadata.
    """
    try:
        # Build search query
        search_terms = [vendor, product_type, "datasheet", "filetype:pdf"]
        if model_family:
            search_terms.insert(1, model_family)
        query = " ".join(search_terms)

        pdfs = []

        # Tier 1: Serper API
        serper_results = search_with_serper(query)
        for result in serper_results:
            link = result.get("link", "")
            if ".pdf" in link.lower():
                pdfs.append({
                    "url": link,
                    "title": result.get("title"),
                    "snippet": result.get("snippet"),
                    "source": "serper"
                })

        # Tier 2: SerpAPI
        if len(pdfs) < 3:
            serpapi_results = search_with_serpapi(query)
            for result in serpapi_results:
                link = result.get("link", "")
                if ".pdf" in link.lower() and link not in [p["url"] for p in pdfs]:
                    pdfs.append({
                        "url": link,
                        "title": result.get("title"),
                        "snippet": result.get("snippet"),
                        "source": "serpapi"
                    })

        # Tier 3: Google CSE
        if len(pdfs) < 3:
            cse_results = search_with_google_cse(query)
            for result in cse_results:
                link = result.get("link", "")
                if ".pdf" in link.lower() and link not in [p["url"] for p in pdfs]:
                    pdfs.append({
                        "url": link,
                        "title": result.get("title"),
                        "snippet": result.get("snippet"),
                        "source": "google_cse"
                    })

        return {
            "success": True,
            "vendor": vendor,
            "product_type": product_type,
            "pdfs": pdfs[:5],
            "total_found": len(pdfs)
        }

    except Exception as e:
        logger.error(f"PDF search failed: {e}")
        return {
            "success": False,
            "vendor": vendor,
            "pdfs": [],
            "error": str(e)
        }


@tool("web_search", args_schema=WebSearchInput)
def web_search_tool(query: str, num_results: int = 5) -> Dict[str, Any]:
    """
    General web search using multi-tier fallback.
    """
    try:
        results = []

        # Try Serper first
        serper_results = search_with_serper(query)
        for r in serper_results[:num_results]:
            results.append({
                "title": r.get("title"),
                "link": r.get("link"),
                "snippet": r.get("snippet"),
                "source": "serper"
            })

        # Fall back to SerpAPI if needed
        if len(results) < num_results:
            serpapi_results = search_with_serpapi(query)
            for r in serpapi_results:
                if len(results) >= num_results:
                    break
                if r.get("link") not in [x["link"] for x in results]:
                    results.append({
                        "title": r.get("title"),
                        "link": r.get("link"),
                        "snippet": r.get("snippet"),
                        "source": "serpapi"
                    })

        return {
            "success": True,
            "query": query,
            "results": results,
            "total_found": len(results)
        }

    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return {
            "success": False,
            "query": query,
            "results": [],
            "error": str(e)
        }


# ============================================================================
# VENDOR TOOL INPUT SCHEMAS
# ============================================================================

class SearchVendorsInput(BaseModel):
    """Input for vendor search"""
    product_type: str = Field(description="Product type to search vendors for")
    requirements: Optional[Dict[str, Any]] = Field(default=None, description="User requirements")


class GetVendorProductsInput(BaseModel):
    """Input for getting vendor products"""
    vendor: str = Field(description="Vendor name")
    product_type: str = Field(description="Product type")


class FuzzyMatchVendorsInput(BaseModel):
    """Input for fuzzy vendor matching"""
    vendor_names: List[str] = Field(description="List of vendor names to match")
    allowed_vendors: List[str] = Field(description="List of allowed vendors to match against")
    threshold: int = Field(default=70, description="Matching threshold (0-100)")


# ============================================================================
# VENDOR HELPER FUNCTIONS
# ============================================================================

def get_vendors_from_mongodb(product_type: str) -> List[str]:
    """Get vendors from Azure Blob Storage for a product type (MongoDB API compatible)"""
    try:
        from azure_blob_utils import get_vendors_for_product_type
        vendors = get_vendors_for_product_type(product_type)
        return vendors if vendors else []
    except Exception as e:
        logger.warning(f"Failed to get vendors from Azure Blob: {e}")
        return []


def get_all_vendors_from_mongodb() -> List[str]:
    """Get all available vendors from Azure Blob Storage (MongoDB API compatible)"""
    try:
        from azure_blob_utils import get_available_vendors_from_mongodb
        vendors = get_available_vendors_from_mongodb()
        return vendors if vendors else []
    except Exception as e:
        logger.warning(f"Failed to get all vendors: {e}")
        return []


def get_vendor_products_from_mongodb(vendor: str, product_type: str) -> List[Dict[str, Any]]:
    """Get products for a vendor from Azure Blob Storage (MongoDB API compatible)"""
    try:
        from azure_blob_utils import azure_blob_file_manager
        # Query products from Azure Blob
        products = azure_blob_file_manager.list_files(
            'vendors',
            {'vendor_name': vendor, 'product_type': product_type}
        )
        return products
    except Exception as e:
        logger.warning(f"Failed to get vendor products: {e}")
        return []


# ============================================================================
# VENDOR TOOLS
# ============================================================================

@tool("search_vendors", args_schema=SearchVendorsInput)
def search_vendors_tool(
    product_type: str,
    requirements: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Search for vendors that offer products matching the specified type.
    Returns list of vendors with their available product families.
    """
    try:
        # Get vendors for product type
        vendors = get_vendors_from_mongodb(product_type)

        if not vendors:
            # Fall back to all vendors
            vendors = get_all_vendors_from_mongodb()
            logger.info(f"No specific vendors for {product_type}, using all vendors")

        # Get product families for each vendor
        vendor_details = []
        for vendor in vendors:
            vendor_info = {
                "vendor": vendor,
                "product_type": product_type,
                "has_products": True  # Would be checked against DB
            }
            vendor_details.append(vendor_info)

        return {
            "success": True,
            "product_type": product_type,
            "vendors": vendors,
            "vendor_count": len(vendors),
            "vendor_details": vendor_details
        }

    except Exception as e:
        logger.error(f"Vendor search failed: {e}")
        return {
            "success": False,
            "product_type": product_type,
            "vendors": [],
            "error": str(e)
        }


@tool("get_vendor_products", args_schema=GetVendorProductsInput)
def get_vendor_products_tool(vendor: str, product_type: str) -> Dict[str, Any]:
    """
    Get products from a specific vendor for the given product type.
    Returns product list with specifications.
    """
    try:
        products = get_vendor_products_from_mongodb(vendor, product_type)

        return {
            "success": True,
            "vendor": vendor,
            "product_type": product_type,
            "products": products,
            "product_count": len(products)
        }

    except Exception as e:
        logger.error(f"Failed to get vendor products: {e}")
        return {
            "success": False,
            "vendor": vendor,
            "products": [],
            "error": str(e)
        }


@tool("fuzzy_match_vendors", args_schema=FuzzyMatchVendorsInput)
def fuzzy_match_vendors_tool(
    vendor_names: List[str],
    allowed_vendors: List[str],
    threshold: int = 70
) -> Dict[str, Any]:
    """
    Perform fuzzy matching between vendor names and allowed vendors list.
    Used to filter vendors based on CSV uploads or user preferences.
    """
    try:
        from fuzzywuzzy import fuzz, process
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utils.validation_utils import validate_list, validate_numeric, ValidationException

        matched_vendors = []
        unmatched_vendors = []

        for vendor in vendor_names:
            # Find best match
            result = process.extractOne(vendor, allowed_vendors)

            if result and result[1] >= threshold:
                matched_vendors.append({
                    "original": vendor,
                    "matched_to": result[0],
                    "score": result[1]
                })
            else:
                unmatched_vendors.append({
                    "original": vendor,
                    "best_match": result[0] if result else None,
                    "score": result[1] if result else 0
                })

        return {
            "success": True,
            "matched_vendors": matched_vendors,
            "unmatched_vendors": unmatched_vendors,
            "match_count": len(matched_vendors),
            "threshold_used": threshold
        }

    except Exception as e:
        logger.error(f"Fuzzy matching failed: {e}")
        return {
            "success": False,
            "matched_vendors": [],
            "unmatched_vendors": vendor_names,
            "error": str(e)
        }


# ============================================================================
# GENERIC PRODUCT IMAGE GENERATION TOOL
# ============================================================================

@tool("get_generic_product_image", args_schema=GetGenericProductImageInput)
def get_generic_product_image_tool(product_type: str) -> Dict[str, Any]:
    """
    Get or generate a generic product type image using Gemini Imagen 4.0.

    Architecture:
    1. Check Azure Blob Storage cache first
    2. If not cached, generate using Gemini Imagen 4.0 LLM
    3. Store in Azure Blob Storage for future use
    4. Return image URL and metadata

    Args:
        product_type: Product type name (e.g., "Pressure Transmitter", "Temperature Sensor")

    Returns:
        Dict containing:
        - success: bool - Whether image was retrieved/generated
        - image_url: str - URL to the image
        - product_type: str - Product type
        - cached: bool - Whether image was cached or newly generated
        - source: str - Source of image ("gemini_imagen")
        - generation_method: str - How image was created ("llm")
        - error: str - Error message if failed
    """
    try:
        from generic_image_utils import fetch_generic_product_image

        logger.info(f"[TOOL] Fetching generic product image for: {product_type}")

        # Fetch or generate the image
        result = fetch_generic_product_image(product_type)

        if result:
            logger.info(f"[TOOL] ✓ Successfully retrieved image for {product_type}")
            return {
                "success": True,
                "image_url": result.get("url"),
                "product_type": product_type,
                "cached": result.get("cached", False),
                "source": result.get("source", "gemini_imagen"),
                "generation_method": result.get("generation_method", "llm")
            }
        else:
            logger.warning(f"[TOOL] Failed to retrieve/generate image for {product_type}")
            return {
                "success": False,
                "image_url": None,
                "product_type": product_type,
                "error": "Failed to retrieve or generate image"
            }

    except Exception as e:
        logger.error(f"[TOOL] Error in get_generic_product_image_tool: {e}")
        logger.exception(e)
        return {
            "success": False,
            "image_url": None,
            "product_type": product_type,
            "error": str(e)
        }
