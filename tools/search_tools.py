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
