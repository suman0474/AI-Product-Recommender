
# loading.py
from difflib import get_close_matches
import os
import io
import json
import re
import time
import threading
from typing import Any, List, Dict, Optional, Tuple
from urllib.parse import urlparse
from glob import glob
import logging
import requests
from serpapi.google_search import GoogleSearch
from googleapiclient.discovery import build
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from test import (
    extract_data_from_pdf,
    send_to_language_model,
    aggregate_results,
    split_product_types,
    save_json,
)

# Azure Blob imports (MongoDB API compatible)
# Azure Blob imports
from azure_blob_utils import azure_blob_file_manager

# LLM import (LangChain Google Gemini with OpenAI fallback)
from langchain_google_genai import ChatGoogleGenerativeAI
from llm_fallback import create_llm_with_fallback

# ----------------- Config -----------------
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_API_KEY1 = os.getenv("GOOGLE_API_KEY1", "")
GOOGLE_CSE_ID = "066b7345f94f64897"  # You'll need to create this

# ----------------- Retry and Caching Utilities -----------------
class RetryConfig:
    """Configuration for retry logic"""
    def __init__(self, max_retries=3, base_delay=1.0, max_delay=60.0, exponential_base=2.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base

def retry_with_backoff(config: RetryConfig = None):
    """Decorator for exponential backoff retry logic"""
    if config is None:
        config = RetryConfig()
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt == config.max_retries:
                        logger.error(f"Function {func.__name__} failed after {config.max_retries} retries: {e}")
                        break
                    
                    # Calculate delay with exponential backoff
                    delay = min(config.base_delay * (config.exponential_base ** attempt), config.max_delay)
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
            
            raise last_exception
        return wrapper
    return decorator

class SimpleCache:
    """Simple in-memory cache with TTL"""
    def __init__(self, default_ttl=3600):  # 1 hour default
        self.cache = {}
        self.timestamps = {}
        self.default_ttl = default_ttl
    
    def get(self, key):
        if key in self.cache:
            if time.time() - self.timestamps[key] < self.default_ttl:
                logger.info(f"Cache hit for key: {key[:50]}...")
                return self.cache[key]
            else:
                # Expired
                del self.cache[key]
                del self.timestamps[key]
        return None
    
    def set(self, key, value):
        self.cache[key] = value
        self.timestamps[key] = time.time()
        logger.info(f"Cache set for key: {key[:50]}...")

# Global cache instance
search_cache = SimpleCache(default_ttl=3600)  # 1 hour cache
# Cache for generated requirement schemas (longer TTL)
schema_cache = SimpleCache(default_ttl=60 * 60 * 24)  # 24 hours

# Locks to prevent concurrent builds for the same product_type
schema_build_locks = {}

class ProgressTracker:
    """Track progress of long-running operations"""
    def __init__(self, total_steps, operation_name="Operation"):
        self.total_steps = total_steps
        self.current_step = 0
        self.operation_name = operation_name
        self.start_time = time.time()
        self.step_details = []
    
    def update(self, step_name="", details=""):
        self.current_step += 1
        self.step_details.append({
            "step": self.current_step,
            "name": step_name,
            "details": details,
            "timestamp": time.time()
        })
        
        elapsed = time.time() - self.start_time
        progress_pct = (self.current_step / self.total_steps) * 100
        
        logger.info(f"[{self.operation_name}] Step {self.current_step}/{self.total_steps} ({progress_pct:.1f}%) - {step_name}")
        if details:
            logger.info(f"  Details: {details}")
    
    def get_progress(self):
        return {
            "operation": self.operation_name,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "progress_percentage": (self.current_step / self.total_steps) * 100,
            "elapsed_time": time.time() - self.start_time,
            "recent_steps": self.step_details[-3:]  # Last 3 steps
        }

# ----------------- Vendor Discovery -----------------
def _extract_json(text: str) -> str:
    """Extract the first JSON array or object from text."""
    match = re.search(r'(\[.*\]|\{.*\})', text, re.DOTALL)
    if match:
        return match.group(1)
    return text  # fallback

def discover_top_vendors(product_type: str, llm=None) -> List[Dict[str, Any]]:
    """
    Uses LLM to discover the top 5 vendors for a specific product type,
    then queries LLM to find model families for each vendor.
    """
    print(f"[DISCOVER] Discovering top 5 vendors and their model families for product_type='{product_type}'")

    if llm is None:
        llm = create_llm_with_fallback(
            model="gemini-2.5-flash",
            temperature=0.1,
            google_api_key=GOOGLE_API_KEY,
        )

    # First, discover the top 5 vendors for this product type using LLM
    vendor_discovery_prompt = f"""
List the top 5 most prominent and widely recognized vendors/manufacturers for "{product_type}" in industrial instrumentation.

Focus on established companies that are known for manufacturing high-quality {product_type.lower()} used in industrial applications.

Return only a valid JSON array of vendor names. Do not include any other text or explanations.

Example format:
["Emerson", "ABB", "Yokogawa", "Endress+Hauser", "Honeywell"]
"""

    try:
        print(f"[DISCOVER] Invoking LLM to discover top 5 vendors for '{product_type}'")
        vendor_response = llm.invoke(vendor_discovery_prompt)
        vendor_content = ""
        if isinstance(vendor_response.content, list):
            vendor_content = "".join([c.get("text", "") for c in vendor_response.content if isinstance(c, dict)])
        else:
            vendor_content = str(vendor_response.content or "")
        
        # Clean and extract JSON
        vendor_cleaned = vendor_content.strip()
        vendor_cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", vendor_cleaned)
        vendor_cleaned = re.sub(r"\n?```$", "", vendor_cleaned)
        
        vendor_extracted_json = _extract_json(vendor_cleaned)
        discovered_vendors = json.loads(vendor_extracted_json)

        # Basic validation
        if not isinstance(discovered_vendors, list):
            logger.warning(f"Expected a list for vendors from LLM for product type '{product_type}', but got {type(discovered_vendors)}. Using fallback vendors.")
            discovered_vendors = ["Emerson", "ABB", "Yokogawa", "Endress+Hauser", "Honeywell"]
        
        # Ensure we have exactly 5 vendors (truncate if more, pad if less)
        if len(discovered_vendors) > 5:
            discovered_vendors = discovered_vendors[:5]
        elif len(discovered_vendors) < 5:
            # Fallback vendors to fill the gap
            fallback_vendors = ["Emerson", "ABB", "Yokogawa", "Endress+Hauser", "Honeywell", "Siemens", "Holykell"]
            for fallback in fallback_vendors:
                if fallback not in discovered_vendors and len(discovered_vendors) < 5:
                    discovered_vendors.append(fallback)

        print(f"[DISCOVER] Discovered top 5 vendors: {discovered_vendors}")

    except Exception as e:
        logger.warning(f"LLM failed to discover vendors for '{product_type}': {e}. Using fallback vendors.")
        discovered_vendors = ["Emerson", "ABB", "Yokogawa", "Endress+Hauser", "Honeywell"]

    # Check if schema already exists locally (this logic remains the same)
    specs_dir = "specs"
    existing_schema = _load_existing_schema(specs_dir, product_type)

    if existing_schema and existing_schema.get("mandatory_requirements") and existing_schema.get("optional_requirements"):
        print(f"[DISCOVER] Schema already exists for '{product_type}', skipping schema generation")
    else:
        print(f"[DISCOVER] No existing schema found for '{product_type}', will generate after model discovery")

    vendors_with_families = []
    
    # --- UPDATED: Loop through discovered vendors to find model families for each ---
    for vendor_name in discovered_vendors:
        prompt = f"""
List the most popular model families for the product type "{product_type}" from the vendor "{vendor_name}".
Return only a valid JSON array of strings. Do not include any other text or explanations.

Example for vendor "Emerson" and product type "Pressure Transmitter":
["Rosemount 3051", "Rosemount 2051", "Rosemount 2088"]
"""
        
        try:
            print(f"[DISCOVER] Invoking LLM to find model families for '{vendor_name}'")
            response = llm.invoke(prompt)
            content = ""
            if isinstance(response.content, list):
                content = "".join([c.get("text", "") for c in response.content if isinstance(c, dict)])
            else:
                content = str(response.content or "")
            
            # Clean and extract JSON
            cleaned = content.strip()
            cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)
            
            extracted_json = _extract_json(cleaned)
            model_families = json.loads(extracted_json)

            # Basic validation
            if not isinstance(model_families, list):
                logger.warning(f"Expected a list for model families from LLM for vendor '{vendor_name}', but got {type(model_families)}. Using empty list.")
                model_families = []

        except Exception as e:
            logger.warning(f"LLM failed to get model families for '{vendor_name}': {e}. Using an empty list.")
            model_families = []
        
        # Append the result for the current vendor
        vendors_with_families.append({
            "vendor": vendor_name,
            "model_families": model_families
        })

    # Generate schema if it doesn't exist, using the discovered model families
    if not existing_schema or not existing_schema.get("mandatory_requirements") or not existing_schema.get("optional_requirements"):
        try:
            print("[DISCOVER] Generating schema from vendor data using LLM")
            schema = create_schema_from_vendor_data(product_type, vendors_with_families, llm)
            
            # Validate schema before saving
            if schema and isinstance(schema, dict):
                if schema.get("mandatory_requirements") or schema.get("optional_requirements"):
                    schema_path = _save_schema_to_specs(product_type, schema)
                    print(f"[DISCOVER] Schema saved to MongoDB: {schema_path}")
                    
                    # Also cache it immediately
                    try:
                        schema_cache.set(product_type, schema)
                        print(f"[DISCOVER] Schema cached in memory for '{product_type}'")
                    except Exception as cache_error:
                        print(f"[WARN] Failed to cache schema: {cache_error}")
                else:
                    print(f"[WARN] Generated schema is empty for '{product_type}', not saving")
            else:
                print(f"[WARN] Generated schema is invalid for '{product_type}'")
        except Exception as e:
            print(f"[ERROR] Failed to generate/save schema for '{product_type}': {e}")
            logger.error(f"Schema generation failed: {e}", exc_info=True)

    return vendors_with_families


def _search_vendor_pdfs(
    vendor: str,
    product_type: str,
    model_families: List[str] = None
) -> List[Dict[str, Any]]:
    print(f"[SEARCH] Multi-engine search for vendor='{vendor}', product_type='{product_type}'")
    pdfs = []
    
    try:
        search_models = model_families or [None]
        
        for model in search_models:
            model_filter = model if model else ""
            
            # Create enhanced search query with specification terms
            spec_terms = "(specification OR datasheet OR manual OR technical OR brochure OR guide)"
            
            if model_filter:
                query = f"{vendor} {product_type} {model_filter} {spec_terms} filetype:pdf"
            else:
                query = f"{vendor} {product_type} {spec_terms} filetype:pdf"
            
            print(f"[SEARCH] Enhanced query: {query}")
            
            all_matched_urls = []
            search_results_count = 0
            
            # Implement three-tier fallback mechanism: Try Serper first, then SERP API, then Google Custom Search
            serper_results = []
            try:
                serper_results = _search_with_serper(query)
                search_results_count += len(serper_results)
                
                if serper_results:
                    # Filter and score Serper results
                    filtered_serper = _filter_and_score_results(
                        serper_results, vendor, product_type, [model_filter] if model_filter else []
                    )
                    all_matched_urls.extend(filtered_serper)
                
            except Exception as e:
                print(f"[WARN] Serper API search failed: {e}")
            
            # If Serper didn't return sufficient results, try SERP API as fallback
            if len(all_matched_urls) == 0:
                serpapi_results = []
                try:
                    serpapi_results = _search_with_serpapi(query)
                    search_results_count += len(serpapi_results)
                    
                    if serpapi_results:
                        # Filter and score SerpAPI results
                        filtered_serpapi = _filter_and_score_results(
                            serpapi_results, vendor, product_type, [model_filter] if model_filter else []
                        )
                        all_matched_urls.extend(filtered_serpapi)
                    
                except Exception as e:
                    print(f"[WARN] SerpAPI search failed: {e}")
            
            # If both Serper and SerpAPI didn't return sufficient results, try Google Custom Search as final fallback
            if len(all_matched_urls) == 0:
                try:
                    google_results = _search_with_google_custom(query)
                    search_results_count += len(google_results)
                    
                    if google_results:
                        # Filter and score Google Custom Search results
                        filtered_google = _filter_and_score_results(
                            google_results, vendor, product_type, [model_filter] if model_filter else []
                        )
                        all_matched_urls.extend(filtered_google)
                    
                except Exception as e:
                    print(f"[WARN] Google Custom Search fallback failed: {e}")
            
            # Remove duplicates and rank by score
            unique_urls = _deduplicate_and_rank_results(all_matched_urls)
            
            if unique_urls:
                # Determine which search engine(s) were used based on the new three-tier system
                sources_used = []
                fallback_used = False
                
                if serper_results and any(result.get("source") == "serper" for result in all_matched_urls):
                    sources_used.append("serper")
                elif serpapi_results and any(result.get("source") == "serpapi" for result in all_matched_urls):
                    sources_used.append("serpapi")
                    fallback_used = True
                elif any(result.get("source") == "google_custom" for result in all_matched_urls):
                    sources_used.append("google_custom")
                    fallback_used = True
                
                pdfs.append({
                    "vendor": vendor,
                    "product_type": product_type,
                    "model_family": model,
                    "pdfs": unique_urls[:3],
                    "sources_used": sources_used,
                    "fallback_used": fallback_used,
                    "total_results_found": len(unique_urls)
                })

    except Exception as e:
        print(f"[WARN] PDF search failed for {vendor}: {e}")

    return pdfs


# Query generation function removed - using simple multi-search approach instead

@retry_with_backoff(RetryConfig(max_retries=2, base_delay=2.0))
def _search_with_serper(query: str) -> List[Dict[str, str]]:
    """Search using Serper API with timeout handling and caching"""
    
    # Check cache first
    cache_key = f"serper_{query}"
    cached_result = search_cache.get(cache_key)
    if cached_result is not None:
        return cached_result
    
    if not SERPER_API_KEY:
        logger.warning("SERPER_API_KEY not available")
        return []
    
    try:
        url = "https://google.serper.dev/search"
        payload = {
            "q": query,
            "num": 3
        }
        headers = {
            "X-API-KEY": SERPER_API_KEY,
            "Content-Type": "application/json"
        }
        
        # Use threading for timeout on Windows
        import threading
        result_container = [None]
        exception_container = [None]
        
        def serper_request():
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=30)
                response.raise_for_status()
                data = response.json()
                result_container[0] = data.get("organic", [])
            except Exception as e:
                exception_container[0] = e
        
        thread = threading.Thread(target=serper_request)
        thread.daemon = True
        thread.start()
        thread.join(timeout=30)
        
        if thread.is_alive():
            logger.warning(f"Serper API request timed out for query: {query}")
            return []
        
        if exception_container[0]:
            raise exception_container[0]
        
        items = result_container[0] or []
        
        results = []
        for item in items:
            title = item.get("title", "")
            link = item.get("link", "")
            snippet = item.get("snippet", "")
            
            if link:
                results.append({
                    "title": title.strip(),
                    "url": link,
                    "snippet": snippet,
                    "source": "serper"
                })
        
        # Cache the result
        search_cache.set(cache_key, results)
        logger.info(f"Serper API returned {len(results)} results for: {query}")
        return results
        
    except Exception as e:
        logger.error(f"Serper API search failed for query '{query}': {e}")
        return []


@retry_with_backoff(RetryConfig(max_retries=2, base_delay=2.0))
def _search_with_serpapi(query: str) -> List[Dict[str, str]]:
    """Search using SerpAPI with timeout handling and caching"""
    
    # Check cache first
    cache_key = f"serpapi_{query}"
    cached_result = search_cache.get(cache_key)
    if cached_result is not None:
        return cached_result
    
    if not SERPAPI_KEY:
        logger.warning("SERPAPI_KEY not available")
        return []
    
    try:
        search = GoogleSearch({
            "q": query,
            "api_key": SERPAPI_KEY,
            "num": 3,
            "gl": "us",
            "hl": "en",
        })
        
        # Use threading for timeout on Windows
        import threading
        result_container = [None]
        exception_container = [None]
        
        def serpapi_request():
            try:
                result_container[0] = search.get_dict().get("organic_results", [])
            except Exception as e:
                exception_container[0] = e
        
        thread = threading.Thread(target=serpapi_request)
        thread.daemon = True
        thread.start()
        thread.join(timeout=30)  # Reduced from 120s to 30s
        
        if thread.is_alive():
            logger.warning(f"SerpAPI request timed out for query: {query}")
            return []
        
        if exception_container[0]:
            raise exception_container[0]
        
        items = result_container[0] or []
        
        results = []
        for item in items:
            title = item.get("title", "")
            link = item.get("link", "")
            snippet = item.get("snippet", "")
            
            if link:
                results.append({
                    "title": title.strip(),
                    "url": link,
                    "snippet": snippet,
                    "source": "serpapi"
                })
        
        # Cache the result
        search_cache.set(cache_key, results)
        logger.info(f"SerpAPI returned {len(results)} results for: {query}")
        return results
        
    except Exception as e:
        logger.error(f"SerpAPI search failed for query '{query}': {e}")
        return []


@retry_with_backoff(RetryConfig(max_retries=2, base_delay=2.0))
def _search_with_google_custom(query: str) -> List[Dict[str, str]]:
    """Search using Google Custom Search API with timeout handling and caching"""
    
    # Check cache first
    cache_key = f"google_custom_{query}"
    cached_result = search_cache.get(cache_key)
    if cached_result is not None:
        return cached_result
    
    if not GOOGLE_API_KEY1:
        logger.warning("GOOGLE_API_KEY1 not available")
        return []
    
    # Check if CSE ID is configured (not default placeholder)
    if GOOGLE_CSE_ID == "066b7345f94f64897" or not GOOGLE_CSE_ID:
        cse_id = GOOGLE_CSE_ID
    else:
        return []
    
    try:
        import threading
        import socket
        
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY1)
        
        result_container = [None]
        exception_container = [None]
        
        def google_request():
            try:
                result = service.cse().list(
                    q=query,
                    cx=cse_id,
                    num=3,
                    fileType='pdf'
                ).execute()
                result_container[0] = result.get('items', [])
            except Exception as e:
                exception_container[0] = e
        
        thread = threading.Thread(target=google_request)
        thread.daemon = True
        thread.start()
        thread.join(timeout=30)  # Reduced from 60s to 30s
        
        if thread.is_alive():
            logger.warning(f"Google Custom Search timed out for query: {query}")
            return []
        
        if exception_container[0]:
            raise exception_container[0]
        
        items = result_container[0] or []
        
        results = []
        for item in items:
            title = item.get('title', '')
            link = item.get('link', '')
            snippet = item.get('snippet', '')
            
            if link:
                results.append({
                    "title": title.strip(),
                    "url": link,
                    "snippet": snippet,
                    "source": "google_custom"
                })
        
        # Cache the result
        search_cache.set(cache_key, results)
        logger.info(f"Google Custom Search returned {len(results)} results for: {query}")
        return results
        
    except Exception as e:
        logger.error(f"Google Custom Search failed for query '{query}': {e}")
        return []


def _filter_and_score_results(
    results: List[Dict[str, str]], 
    vendor: str, 
    product_type: str, 
    model_families: List[str]
) -> List[Dict[str, Any]]:
    """Filter and score search results for relevance"""
    
    # Enhanced keywords for filtering
    spec_keywords = [
        "specification", "specifications", "spec", "specs",
        "datasheet", "data sheet", "spec sheet", "data-sheet",
        "technical data", "technical specification", "tech spec",
        "manual", "user manual", "installation manual", "product manual",
        "brochure", "catalog", "catalogue", "guide", "documentation",
        product_type.lower(), "pdf"
    ]
    
    # Vendor name variations for matching
    vendor_keywords = [vendor.lower()]
    vendor_parts = vendor.lower().replace('(', ' ').replace(')', ' ').split()
    vendor_keywords.extend([part for part in vendor_parts if len(part) > 2])
    
    # Model-specific keywords from all model families
    model_keywords = []
    if model_families:
        for family in model_families:
            if isinstance(family, str):
                model_keywords.append(family.lower())
                model_keywords.extend(family.lower().split())
    
    filtered_results = []
    
    for result in results:
        title = result.get("title", "").lower()
        url = result.get("url", "").lower() 
        snippet = result.get("snippet", "").lower()
        
        # Combine all text for analysis
        full_text = f"{title} {url} {snippet}"
        
        # Score calculation
        score = 0
        
        # Check for spec keywords (higher weight)
        spec_matches = sum(1 for kw in spec_keywords if kw in full_text)
        score += spec_matches * 2
        
        # Check for vendor name (essential)
        vendor_matches = sum(1 for vk in vendor_keywords if vk in full_text)
        if vendor_matches == 0:
            continue  # Skip if no vendor match
        score += vendor_matches * 3
        
        # Check for model keywords (bonus points)
        if model_keywords:
            model_matches = sum(1 for mk in model_keywords if mk in full_text)
            score += model_matches * 2
        
        # Bonus for PDF in URL
        if '.pdf' in url:
            score += 3
        
        # Bonus for vendor-related domains (dynamic detection)
        vendor_name_parts = vendor.lower().replace('(', '').replace(')', '').split()
        vendor_domain_indicators = [part for part in vendor_name_parts if len(part) > 3]
        
        # Check if URL contains vendor name indicators
        if any(indicator in url for indicator in vendor_domain_indicators):
            score += 3  # Reduced from 5 since it's less certain than hardcoded domains
        
        # Penalty for irrelevant content
        irrelevant_terms = ['news', 'press release', 'blog', 'forum', 'wikipedia', 'linkedin', 
                            'facebook', 'twitter', 'instagram', 'youtube', 'careers', 'jobs',
                            'company profile', 'about us', 'contact', 'privacy policy']
        irrelevant_matches = sum(1 for term in irrelevant_terms if term in full_text)
        score -= irrelevant_matches * 2
        
        # URL structure penalties (likely not technical docs)
        if any(bad_pattern in url for bad_pattern in ['/news/', '/blog/', '/careers/', '/about/']):
            score -= 3
        
        # Length-based quality indicator (very short titles often not useful)
        if len(result.get("title", "").strip()) < 10:
            score -= 1
        
        # Enhanced minimum score threshold for better quality
        if score >= 8:  # Increased from 5 to 8 for better filtering
            filtered_results.append({
                "matched_title": result["title"],
                "pdf_url": result["url"],
                "snippet": result.get("snippet", ""),
                "source": result.get("source", "Unknown"),
                "relevance_score": score,
                "quality_indicators": {
                    "vendor_matches": vendor_matches,
                    "spec_matches": spec_matches,
                    "model_matches": model_matches if model_keywords else 0,
                    "has_pdf": '.pdf' in url,
                    "domain_relevance": any(indicator in url for indicator in vendor_domain_indicators)
                }
            })
    
    # Sort by relevance score
    filtered_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    
    logger.info(f"Filtered {len(results)} results to {len(filtered_results)} high-quality matches for {vendor}")
    return filtered_results


def _deduplicate_and_rank_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicates and rank results by relevance score"""
    
    # Remove duplicates based on URL
    seen_urls = set()
    unique_results = []
    
    for result in results:
        url = result.get("pdf_url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(result)
    
    # Sort by relevance score (descending)
    unique_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    
    # Remove the score from final results (keep it clean)
    final_results = []
    for result in unique_results:
        final_result = {
            "matched_title": result["matched_title"],
            "pdf_url": result["pdf_url"],
            "source": result.get("source", "Unknown")
        }
        if result.get("snippet"):
            final_result["snippet"] = result["snippet"]
        
        final_results.append(final_result)
    
    return final_results


# ----------------- LLM-based Schema Generation -----------------

def create_schema_from_vendor_data(product_type: str, vendors: List[Dict[str, Any]], llm=None) -> Dict[str, Any]:
    """
    Create schema using LLM analysis of vendor data based on classification rules.
    Rules:
    1. If model selection guide exists → those specs go into mandatory; all others go into optional
    2. If no model selection guide → specs classified by name (core functional → mandatory, extras → optional)
    """
    print(f"[SCHEMA] Creating schema for '{product_type}' from {len(vendors)} vendors using LLM")

    if llm is None:
        llm = create_llm_with_fallback(
            model="gemini-2.5-flash",
            temperature=0.1,
            google_api_key=GOOGLE_API_KEY,
        )

    # Build vendor information for LLM prompt
    vendor_info = []
    for vendor in vendors:
        vendor_name = vendor.get("vendor", "")
        model_families = vendor.get("model_families", [])
        vendor_info.append(f"- {vendor_name}: {', '.join(model_families) if model_families else 'General product line'}")

    vendor_list_text = "\n".join(vendor_info)

    prompt = f"""Analyze the technical specifications for "{product_type}" across these major vendors and their model families:

{vendor_list_text}

Create a comprehensive technical specification schema following these classification rules:

**Rule 1 (Per Model Family):**
- For each vendor's model family, check if a model selection guide exists in its PDF documents.
- If a guide exists, the specifications mentioned in that guide must be classified as **MANDATORY**.
- All other specifications not in the guide should be classified as **OPTIONAL**.

**Rule 2**: If no model selection guides exist, classify by functional importance:
- MANDATORY: Core functional parameters needed to select/specify the product (accuracy, measurement range, output signals, power requirements, process connections, etc.)
- OPTIONAL: Enhancement features, advanced options, accessories, special configurations, diagnostics, etc.

**CRITICAL LIMITS:**
- Maximum 15 specifications in mandatory_requirements (focus on most critical)
- Maximum 20 specifications in optional_requirements (focus on most common)
- If you identify more, prioritize the most important and commonly specified parameters
- Avoid redundant or rarely-used specifications

Structure the output as a hierarchical JSON with exactly these two top-level keys:
- "mandatory_requirements"
- "optional_requirements"

Group specifications into logical categories like:
- Performance (measurement type, accuracy, range, response time, etc.)
- Electrical (output signals, power supply, communication protocols, etc.)
- Mechanical (sensor type, process connections, materials, mounting, etc.)
- Compliance (certifications, standards, safety ratings, etc.)
- MechanicalOptions (housing options, display, mounting variations, etc.)
- Environmental (temperature ranges, ingress protection, hazardous area ratings, etc.)
- Features (diagnostics, advanced processing, connectivity, etc.)
- ServiceAndSupport (warranties, calibration, maintenance, etc.)
- Integration (fieldbus, wireless, cloud connectivity, etc.)

Return ONLY the JSON structure with empty string values for all specification fields (no actual values, just the keys).

Example format:
{{
  "mandatory_requirements": {{
    "Performance": {{
      "measurementType": "",
      "accuracy": "",
      "repeatability": "",
      "responseTime": "",
      "turnDownRatio": "",
      "temperatureRange": "",
      "pressureRange": "",
      "flowRange": ""
    }},
    "Electrical": {{
      "outputSignal": "",
      "powerSupply": "",
      "communicationProtocol": "",
      "signalType": "",
      "cableEntry": ""
    }}
  }},
  "optional_requirements": {{
    "MechanicalOptions": {{
      "housingMaterial": "",
      "enclosureType": "",
      "displayOptions": "",
      "liningMaterial": ""
    }},
    "Environmental": {{
      "ingressProtection": "",
      "ambientTemperatureRange": "",
      "hazardousAreaRating": ""
    }}
  }}
}}
"""

    try:
        print("[SCHEMA] Invoking LLM to generate schema structure")
        response = llm.invoke(prompt)
        content = ""
        if isinstance(response.content, list):
            content = "".join([c.get("text", "") for c in response.content if isinstance(c, dict)])
        else:
            content = str(response.content or "")

        # Clean and extract JSON
        cleaned = content.strip()
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)

        extracted_json = _extract_json(cleaned)
        schema = json.loads(extracted_json)

        # Validate schema structure
        if not isinstance(schema, dict):
            raise ValueError("Schema must be a dictionary")

        if "mandatory_requirements" not in schema or "optional_requirements" not in schema:
            raise ValueError("Schema must have exactly 'mandatory_requirements' and 'optional_requirements' keys")

        print(f"[SCHEMA] Successfully generated schema with {len(schema.get('mandatory_requirements', {}))} mandatory groups and {len(schema.get('optional_requirements', {}))} optional groups")
        return schema

    except Exception as e:
        print(f"[WARN] LLM schema generation failed: {e}")



def _load_existing_schema(specs_dir: str, product_type: str) -> Dict[str, Any]:
    """Load existing schema from Azure Blob specs collection."""
    try:
        print(f"[SCHEMA] Loading existing schema for product type: {product_type}")
        # Use Azure Blob Storage directly
        schema = azure_blob_file_manager.get_schema_from_azure(product_type)
        if schema:
            print(f"[SCHEMA] Found existing schema in Azure for: {product_type}")
            return schema
        else:
            print(f"[SCHEMA] No existing schema found in Azure for: {product_type}")
            return {}
    except Exception as e:
        print(f"[WARN] Failed to load existing schema from Azure: {e}")
        return {}

def _save_schema_to_specs(product_type: str, schema_dict: Dict[str, Any]) -> str:
    """Save schema to Azure Blob specs collection."""
    try:
        print(f"[SCHEMA] Saving schema to Azure for product type: {product_type}")
        
        # Prepare metadata
        metadata = {
            'collection_type': 'specs',
            'product_type': product_type,
            'normalized_product_type': product_type.lower().replace(' ', '').replace('_', ''),
            'filename': f"{product_type.lower().replace(' ', '_')}.json",
            'file_type': 'json',
            'schema_version': '1.0'
        }
        
        # Upload to Azure Blob
        doc_id = azure_blob_file_manager.upload_json_data(schema_dict, metadata)
        
        print(f"[SCHEMA] Schema saved to Azure with ID: {doc_id}")
        return f"azure://{doc_id}"
        
    except Exception as e:
        print(f"[ERROR] Failed to save schema to Azure: {e}")
        raise

# ----------------- Process PDFs -----------------
@retry_with_backoff(RetryConfig(max_retries=2, base_delay=3.0))
def _download_pdf_to_mongodb(pdf_url: str, metadata: Dict[str, Any]) -> Tuple[bool, str, Optional[str]]:
    """Download PDF directly to MongoDB with retry logic and validation"""
    try:
        # Add headers to appear more like a regular browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/pdf,application/octet-stream,*/*',
        }
        
        response = requests.get(pdf_url, timeout=60, headers=headers, stream=True)
        response.raise_for_status()
        
        # Collect PDF data
        pdf_data = b''
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                pdf_data += chunk
        
        # Validate it's actually a PDF
        content_type = response.headers.get('content-type', '').lower()
        if 'pdf' not in content_type and not pdf_url.lower().endswith('.pdf'):
            if not pdf_data.startswith(b'%PDF'):
                raise ValueError(f"URL does not appear to contain a valid PDF: {pdf_url}")
        
        # Verify file size
        file_size = len(pdf_data)
        if file_size < 1024:  # Less than 1KB is suspicious
            raise ValueError(f"Downloaded file is too small ({file_size} bytes), likely not a valid PDF")
        
        # Upload to Azure
        upload_metadata = {
            'collection_type': 'documents',
            'file_type': 'pdf',
            'source_url': pdf_url,
            'file_size': file_size,
            **metadata
        }
        
        file_id = azure_blob_file_manager.upload_to_azure(pdf_data, upload_metadata)
        
        logger.info(f"Successfully downloaded and stored PDF in Azure: {pdf_url} ({file_size} bytes, ID: {file_id})")
        return True, f"Success ({file_size} bytes)", file_id
        
    except Exception as e:
        logger.error(f"Failed to download PDF {pdf_url}: {e}")
        return False, str(e), None

def process_pdfs_from_urls(product_type: str, vendor_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process PDFs from URLs and store in MongoDB with improved error handling"""
    logger.info(f"[INGEST] Start processing PDFs for product_type='{product_type}' (MongoDB storage)")

    all_results: List[Dict[str, Any]] = []
    processing_stats = {
        "total_pdfs_attempted": 0,
        "successful_downloads": 0,
        "failed_downloads": 0,
        "processing_errors": 0
    }

    for vendor_entry in vendor_data:
        vendor_name = vendor_entry.get("vendor", "").strip().replace(" ", " ")
        models = vendor_entry.get("models", [])
        logger.info(f"[INGEST] Vendor: {vendor_name}, models_count={len(models)}")

        vendor_results: List[Dict[str, Any]] = []

        for model in models:
            model_family = (model.get("model_family") or "").strip().replace(" ", " ")
            pdfs = model.get("pdfs", [])
            logger.info(f"[INGEST]  Model family: '{model_family or 'unknown'}', pdf_count={len(pdfs)}")

            # Only process top 2 PDFs per model to avoid overwhelming the system
            for pdf_info in pdfs[:2]:
                pdf_url = pdf_info.get("pdf_url")
                if not pdf_url:
                    continue

                processing_stats["total_pdfs_attempted"] += 1

                try:
                    logger.info(f"[DOWNLOAD] Fetching PDF: {pdf_url}")
                    
                    # Prepare metadata for MongoDB storage
                    filename = os.path.basename(pdf_url.split("?")[0]) or f"{vendor_name}_{model_family}.pdf"
                    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
                    
                    pdf_metadata = {
                        'product_type': product_type,
                        'vendor_name': vendor_name,
                        'model_family': model_family,
                        'filename': filename,
                        'pdf_title': pdf_info.get('matched_title', ''),
                        'source': pdf_info.get('source', 'unknown')
                    }
                    
                    # Download and store in MongoDB
                    download_success, download_message, file_id = _download_pdf_to_mongodb(pdf_url, pdf_metadata)
                    
                    if not download_success:
                        logger.warning(f"[DOWNLOAD] Failed to download {pdf_url}: {download_message}")
                        processing_stats["failed_downloads"] += 1
                        continue
                    
                    processing_stats["successful_downloads"] += 1
                    logger.info(f"[MONGODB] Saved PDF to MongoDB with ID: {file_id}")

                    # ---- extract + LLM ----
                    # ---- extract + LLM ----
                    try:
                        # Retrieve PDF from Azure for processing
                        pdf_data = azure_blob_file_manager.get_file_from_azure(
                            'documents', 
                            {'file_id': file_id}
                        )
                        
                        if not pdf_data:
                            logger.error(f"[EXTRACT] Could not retrieve PDF from MongoDB: {file_id}")
                            continue
                            
                        pdf_bytes = io.BytesIO(pdf_data)
                        logger.info(f"[EXTRACT] Extracting text from MongoDB PDF: {file_id}")
                        text_chunks = extract_data_from_pdf(pdf_bytes)

                        if not text_chunks or all(len(chunk.strip()) < 50 for chunk in text_chunks):
                            logger.warning(f"[EXTRACT] PDF appears to have minimal text content: {file_id}")
                            continue

                        logger.info(f"[LLM] Sending {len(text_chunks)} chunks to LLM for JSON extraction")
                        pdf_results = send_to_language_model(text_chunks)
                        pdf_results = [item for r in pdf_results for item in (r if isinstance(r, list) else [r])]
                        vendor_results.extend(pdf_results)
                        logger.info(f"[AGGREGATE] Vendor '{vendor_name}' collected {len(vendor_results)} items so far")

                        # ---- extract product image and store in MongoDB ----
                        pdf_bytes.seek(0)
                        if vendor_name and model_family:
                            logger.info("[PROCESSING] PDF content processing completed")

                    except Exception as e:
                        logger.error(f"[PROCESSING] Failed to process PDF content {file_id}: {e}")
                        processing_stats["processing_errors"] += 1
                        continue

                except Exception as e:
                    logger.error(f"[WARN] Failed to process PDF {pdf_url}: {e}")
                    processing_stats["failed_downloads"] += 1

        # ----------------- aggregate + save for this vendor to MongoDB -----------------
        if vendor_results:
            print(f"[SAVE] Aggregating + saving results for vendor '{vendor_name}' to MongoDB")
            final_result = aggregate_results(vendor_results, product_type)
            split_results = split_product_types([final_result])

            # Save per-family JSONs to MongoDB
            for result in split_results:
                vendor = (result.get("vendor") or "").strip()
                for model in result.get("models", []):
                    if not model.get("model_series"):
                        continue

                    single_family_payload = {
                        "product_type": result.get("product_type", product_type),
                        "vendor": vendor,
                        "models": [model],
                    }

                    safe_vendor = vendor.replace(" ", " ") or "UnknownVendor"
                    safe_ptype = (result.get("product_type") or product_type).replace(" ", " ") or "UnknownProduct"
                    safe_series = (model.get("model_series") or "unknown_model").replace("/", " ").replace("\\", " ")

                    # Prepare metadata for MongoDB storage
                    vendor_metadata = {
                        'collection_type': 'vendors',
                        'product_type': safe_ptype,
                        'vendor_name': safe_vendor,
                        'model_series': safe_series,
                        'filename': f"{safe_series}.json",
                        'file_type': 'json'
                    }
                    
                    # Save to Azure
                    try:
                        doc_id = azure_blob_file_manager.upload_json_data(single_family_payload, vendor_metadata)
                        print(f"[OUTPUT] Saved family JSON to Azure: {safe_vendor}/{safe_ptype}/{safe_series} (ID: {doc_id})")
                        all_results.append(single_family_payload)  # keep global record too
                    except Exception as e:
                        logger.error(f"[ERROR] Failed to save vendor data to Azure: {e}")
        else:
            logger.warning(f"[SAVE] No results to save for vendor '{vendor_name}'")

    # Log final processing statistics
    logger.info(f"[INGEST] PDF processing completed for '{product_type}' (MongoDB storage):")
    logger.info(f"  - Total PDFs attempted: {processing_stats['total_pdfs_attempted']}")
    logger.info(f"  - Successful downloads: {processing_stats['successful_downloads']}")
    logger.info(f"  - Failed downloads: {processing_stats['failed_downloads']}")
    logger.info(f"  - Processing errors: {processing_stats['processing_errors']}")
    logger.info(f"  - Total results generated: {len(all_results)}")
    logger.info(f"  - All data stored in MongoDB collections")

    return all_results



# ----------------- Build Requirements Schema -----------------
def build_requirements_schema_from_web(product_type: str) -> Dict[str, Any]:
    """Build requirements schema from web with parallel processing and progress tracking"""
    logger.info(f"[BUILD] Building requirements schema from web for '{product_type}'")
    # If another thread/process has recently built this schema, return cached copy
    cached = schema_cache.get(product_type)
    if cached:
        logger.info(f"[BUILD] Returning cached schema for '{product_type}' (cache hit)")
        return cached
    
    # Step 1: Discover vendors
    progress = ProgressTracker(4, f"Schema Discovery for {product_type}")
    progress.update("Discovering vendors", f"Finding top vendors for {product_type}")
    
    vendors = discover_top_vendors(product_type)
    if not vendors:
        logger.warning(f"No vendors discovered for {product_type}")
        return {"product_type": product_type, "vendors": [], "combined": {}}
    
    progress.update("Processing vendors", f"Found {len(vendors)} vendors, searching for PDFs in parallel")
    
    # Step 2: Search for PDFs in parallel
    vendors_data = []
    
    def search_single_vendor(vendor_info):
        """Search PDFs for a single vendor"""
        try:
            vendor_name = vendor_info.get("vendor")
            model_families = vendor_info.get("model_families", [])
            logger.info(f"[BUILD] Processing vendor '{vendor_name}' with {len(model_families)} model families")
            
            models = _search_vendor_pdfs(vendor_name, product_type, model_families)
            return {
                "vendor": vendor_name,
                "models": [m for m in models if isinstance(m, dict)],
                "success": True
            }
        except Exception as e:
            logger.error(f"Failed to process vendor {vendor_info.get('vendor', 'unknown')}: {e}")
            return {
                "vendor": vendor_info.get("vendor", "unknown"),
                "models": [],
                "success": False,
                "error": str(e)
            }
    
    # Use ThreadPoolExecutor for parallel vendor processing
    max_workers = min(len(vendors), 3)  # Limit to 3 concurrent searches for balanced performance
    
    from agentic.context_managers import managed_thread_pool
    with managed_thread_pool("loading_csv", max_workers=max_workers) as pool:
        executor = pool._executor
        # Submit all vendor search tasks
        future_to_vendor = {
            executor.submit(search_single_vendor, vendor): vendor.get("vendor", "unknown") 
            for vendor in vendors
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_vendor):
            vendor_name = future_to_vendor[future]
            try:
                result = future.result(timeout=180)  # 3 minutes max per vendor
                vendors_data.append(result)
                
                if result["success"]:
                    logger.info(f"✓ Successfully processed vendor: {vendor_name} ({len(result['models'])} models)")
                else:
                    logger.warning(f"✗ Failed to process vendor: {vendor_name}")
                    
            except Exception as e:
                logger.error(f"✗ Vendor {vendor_name} processing failed with exception: {e}")
                vendors_data.append({
                    "vendor": vendor_name,
                    "models": [],
                    "success": False,
                    "error": str(e)
                })
    
    # Filter out failed vendors but keep partial results
    successful_vendors = [v for v in vendors_data if v["success"]]
    failed_vendors = [v for v in vendors_data if not v["success"]]
    
    if failed_vendors:
        logger.warning(f"Failed to process {len(failed_vendors)} vendors: {[v['vendor'] for v in failed_vendors]}")
    
    progress.update("Processing PDFs", f"Successfully processed {len(successful_vendors)} vendors, processing PDFs")
    
    # Step 3: Process PDFs from successful vendors
    try:
        processed_results = process_pdfs_from_urls(product_type, successful_vendors)
    except Exception as e:
        logger.error(f"PDF processing failed: {e}")
        processed_results = []
    
    progress.update("Finalizing schema", f"Processed {len(processed_results)} PDF results")
    
    # Wrap list under 'combined' to prevent .get() errors downstream
    combined_results = {product_type: processed_results}
    
    # Include processing statistics
    result = {
        "product_type": product_type,
        "vendors": successful_vendors,
        "combined": combined_results,
        "processing_stats": {
            "total_vendors_attempted": len(vendors),
            "successful_vendors": len(successful_vendors),
            "failed_vendors": len(failed_vendors),
            "total_pdf_results": len(processed_results),
            "processing_time": progress.get_progress()["elapsed_time"]
        }
    }
    
    logger.info(f"[BUILD] Schema building completed for '{product_type}': {len(successful_vendors)}/{len(vendors)} vendors successful")
    
    # CRITICAL FIX: Load the schema that was saved during discover_top_vendors and return it
    # The schema was already saved to MongoDB in discover_top_vendors() at line 291
    # We need to retrieve it and return it in the proper format
    try:
        final_schema = get_schema_from_mongodb(product_type)
        if final_schema and final_schema.get("mandatory_requirements"):
            logger.info(f"[BUILD] Retrieved saved schema from MongoDB for '{product_type}'")
            # Cache the proper schema format
            schema_cache.set(product_type, final_schema)
            return final_schema
        else:
            logger.warning(f"[BUILD] Schema was not found in MongoDB after automation for '{product_type}', returning empty schema")
            # Return empty but valid schema structure
            empty_schema = {
                "product_type": product_type,
                "mandatory_requirements": {},
                "optional_requirements": {}
            }
            schema_cache.set(product_type, empty_schema)
            return empty_schema
    except Exception as e:
        logger.error(f"[BUILD] Failed to retrieve schema from MongoDB: {e}")
        # Fallback: return empty schema
        empty_schema = {
            "product_type": product_type,
            "mandatory_requirements": {},
            "optional_requirements": {}
        }
        return empty_schema


# ----------------- Load Products Runnable (MongoDB Version) -----------------
def load_products_runnable(vendors_base_path: str = None):
    """Load products from MongoDB instead of local files"""
    def load_products(input_dict):
        detected_product_type = input_dict.get('detected_product_type')
        print(f"[PRODUCTS] Loading products from MongoDB for detected_product_type='{detected_product_type}'")
        
        products = []
        
        # Get smart analysis search categories based on detection
        from standardization_utils import get_analysis_search_categories
        search_categories = get_analysis_search_categories(detected_product_type)
        
        print(f"[PRODUCTS] Detected: '{detected_product_type}'")
        print(f"[PRODUCTS] Will search categories: {search_categories}")
        
        # DEBUG: Log what's actually in the database
        try:
            vendors_collection = mongodb_file_manager.collections.get('vendors')
            if vendors_collection:
                all_types = vendors_collection.distinct('product_type')
                print(f"[DEBUG] Total unique product_types in DB: {len(all_types)}")
                print(f"[DEBUG] Sample product_types in DB: {all_types[:10]}")
                
                # Check if any match our search
                matching_types = [pt for pt in all_types if any(cat.lower() in pt.lower() for cat in search_categories)]
                print(f"[DEBUG] Product types that might match our search: {matching_types[:10]}")
        except Exception as e:
            print(f"[DEBUG] Could not check DB contents: {e}")
        
        try:
            # Query MongoDB vendors collection directly for vendor JSONs
            # All vendor JSONs are stored in the "vendors" collection
            
            vendors_collection = mongodb_file_manager.collections.get('vendors')
            
            if vendors_collection is not None:
                print(f"[PRODUCTS] Querying vendors collection for product categories: {search_categories}")
                
                # Query the vendors collection for documents matching the search categories
                # Look for vendor JSONs that match the product type categories
                query = {
                    '$or': [
                        {'product_type': {'$regex': category, '$options': 'i'}}
                        for category in search_categories
                    ]
                }
                
                print(f"[PRODUCTS] Querying vendors collection with: {query}")
                
                # DEBUG: Check query results before iterating
                query_count = vendors_collection.count_documents(query)
                print(f"[DEBUG] Query matched {query_count} documents")
                
                if query_count == 0:
                    print(f"[DEBUG] ⚠️ ZERO DOCUMENTS MATCHED!")
                    print(f"[DEBUG] Search categories: {search_categories}")
                    print(f"[DEBUG] Query: {query}")
                    
                    # Try to find similar product types
                    all_types = vendors_collection.distinct('product_type')
                    similar = [pt for pt in all_types if 'pressure' in pt.lower() or 'transmitter' in pt.lower()]
                    print(f"[DEBUG] Similar product types in DB: {similar[:10]}")
                
                cursor = vendors_collection.find(query)
                doc_count = 0
                
                for doc in cursor:
                    doc_count += 1
                    
                    try:
                        # Extract vendor data from the vendors collection
                        # Handle the document structure as stored in vendors collection
                        if 'data' in doc:
                            # If data is wrapped in a 'data' field
                            product_data = doc['data']
                        else:
                            # If data is stored directly (excluding MongoDB metadata)
                            product_data = {k: v for k, v in doc.items() if k not in ['_id']}
                        
                        # Extract vendor name and product type
                        vendor = doc.get('vendor_name') or product_data.get('vendor', 'Unknown')
                        prod_type = doc.get('product_type') or product_data.get('product_type', detected_product_type)
                        
                        # Ensure these fields exist in product_data for consistency
                        product_data.setdefault('vendor', vendor)
                        product_data.setdefault('product_type', prod_type)
                        
                        products.append(product_data)
                        
                        # Extract product name for logging
                        product_name = 'Unknown'
                        
                        # Check different possible name fields in order of preference
                        if product_data.get('product_name'):
                            product_name = product_data['product_name']
                        elif product_data.get('name'):
                            product_name = product_data['name']
                        elif product_data.get('model'):
                            product_name = product_data['model']
                        elif product_data.get('models') and len(product_data['models']) > 0:
                            # Get model_series from first model (most common case)
                            first_model = product_data['models'][0]
                            if isinstance(first_model, dict) and first_model.get('model_series'):
                                product_name = first_model['model_series']
                        
                        print(f"[PRODUCTS]   Loaded from vendors collection: {vendor} - {product_name}")
                        
                    except Exception as e:
                        logging.warning(f"[PRODUCTS] Failed to process vendor document: {e}")
                        continue
                
                print(f"[PRODUCTS] Found {doc_count} documents in vendors collection, loaded {len(products)} product entries")
                
                # If no products found with category search, try a broader search
                if len(products) == 0:
                    print(f"[PRODUCTS] No products found with category search, trying broader search on vendors collection...")
                    
                    # Try to get any vendor documents to see what's available
                    cursor = vendors_collection.find({}).limit(10)  # Limit for debugging
                    for doc in cursor:
                        try:
                            if 'data' in doc:
                                product_data = doc['data']
                            else:
                                product_data = {k: v for k, v in doc.items() if k not in ['_id']}
                            
                            vendor = doc.get('vendor_name') or product_data.get('vendor', 'Unknown')
                            prod_type = doc.get('product_type') or product_data.get('product_type', detected_product_type)
                            
                            product_data.setdefault('vendor', vendor)
                            product_data.setdefault('product_type', prod_type)
                            products.append(product_data)
                            
                            print(f"[PRODUCTS]   Broad search loaded: {vendor} - {prod_type}")
                            
                        except Exception as e:
                            logging.warning(f"[PRODUCTS] Failed to process broad search document: {e}")
                    
                    print(f"[PRODUCTS] Broad search found {len(products)} total products in vendors collection")
                
            else:
                logging.warning("[PRODUCTS] Vendors collection not found in MongoDB")
                
        except Exception as e:
            logging.error(f"[PRODUCTS] Failed to load products from vendors collection: {e}")
            # Fallback to empty list
            products = []
        
        input_dict['products_json'] = json.dumps(products, ensure_ascii=False, indent=2)
        return input_dict

    return RunnableLambda(load_products)





def load_pdf_content_runnable(documents_base_path: str = None):
    """
    Load PDFs from MongoDB GridFS instead of local files.
    Extracts text and adds it to the chain's context.
    """
    def load_pdf_content(input_dict):
        product_type = input_dict.get('detected_product_type')
        if not product_type:
            input_dict['pdf_content_json'] = json.dumps({})
            return input_dict

        pdf_texts = {}
        
        # Get smart analysis search categories
        from standardization_utils import get_analysis_search_categories
        search_categories = get_analysis_search_categories(product_type)
        
        logging.info(f"[PDF_LOADER] Loading PDFs from MongoDB for product type: {product_type}")
        logging.info(f"[PDF_LOADER] Search categories: {search_categories}")

        try:
            # Query MongoDB GridFS fs.files table directly for all documents
            # All documents (PDFs) are stored in fs.files table
            
            logging.info(f"[PDF_LOADER] Querying fs.files collection for documents matching product categories")
            
            # Access the fs.files collection directly (GridFS metadata table)
            database = mongodb_file_manager.conn['database'] 
            fs_files_collection = database['fs.files']
            
            for category in search_categories:
                # Query fs.files table for PDFs matching the category
                # Look for documents that match the product type category
                query = {
                    'file_type': 'pdf',
                    'product_type': {'$regex': category, '$options': 'i'}
                }
                
                logging.info(f"[PDF_LOADER] Querying fs.files with: {query}")
                
                try:
                    pdf_files = list(fs_files_collection.find(query))
                    logging.info(f"[PDF_LOADER] Found {len(pdf_files)} PDF files for category '{category}'")
                    
                    for pdf_metadata in pdf_files:
                        try:
                            # Extract metadata from fs.files document
                            vendor = pdf_metadata.get('vendor_name', 'Unknown')
                            filename = pdf_metadata.get('filename', 'unknown.pdf')
                            file_id = pdf_metadata.get('_id')
                            
                            if not file_id:
                                continue
                            
                            logging.info(f"[PDF_LOADER]   Loading PDF from fs.files: {vendor} - {filename}")
                            
                            # Retrieve PDF bytes from GridFS using the file_id
                            try:
                                pdf_bytes_data = mongodb_file_manager.gridfs.get(file_id).read()
                                pdf_bytes = io.BytesIO(pdf_bytes_data)
                                
                                # Extract text chunks from PDF
                                text_chunks = extract_data_from_pdf(pdf_bytes)
                                if text_chunks and len(text_chunks) > 0:
                                    # Filter out empty or very short chunks
                                    valid_chunks = [chunk for chunk in text_chunks if len(chunk.strip()) > 20]
                                    if valid_chunks:
                                        full_text = "\n\n".join(valid_chunks)
                                        
                                        # Store the full text, keyed by vendor
                                        if vendor not in pdf_texts:
                                            pdf_texts[vendor] = ""
                                        pdf_texts[vendor] += full_text + "\n\n--- End of Document ---\n\n"
                                        
                                        logging.info(f"[PDF_LOADER]   Successfully extracted {len(valid_chunks)} text chunks for {vendor}")
                                else:
                                    logging.warning(f"[PDF_LOADER]   No text content extracted from {filename}")
                                    
                            except Exception as extract_error:
                                logging.warning(f"[PDF_LOADER] Failed to extract content from PDF {filename}: {extract_error}")
                                continue
                            
                        except Exception as e:
                            logging.warning(f"[PDF_LOADER] Failed to process PDF from fs.files: {e}")
                            continue
                            
                except Exception as e:
                    logging.warning(f"[PDF_LOADER] Failed to query fs.files for category {category}: {e}")
                    continue
            
            logging.info(f"[PDF_LOADER] Loaded PDF content for {len(pdf_texts)} vendors from fs.files.")
            
        except Exception as e:
            logging.error(f"[PDF_LOADER] Failed to load PDFs from fs.files: {e}")
            # Fallback to empty dict
            pdf_texts = {}
        
        input_dict['pdf_content_json'] = json.dumps(pdf_texts, ensure_ascii=False)
        return input_dict

    return RunnableLambda(load_pdf_content)

# ----------------- MongoDB Schema Loading -----------------
def load_requirements_schema(product_type: str = None) -> Dict[str, Any]:
    """
    Load requirements schema from MongoDB, with fallback to web building if not found.
    This replaces the original local file-based schema loading.
    
    Args:
        product_type: Product type to load schema for. If None, returns a generic schema.
    """
    try:
        # If no product type specified, return a generic schema for initial validation
        if not product_type:
            print("[SCHEMA] Loading generic schema for product type detection")
            return {
                "product_type": "Generic",
                "mandatory_requirements": {
                    "basic_info": {
                        "product_type": {"type": "string", "description": "Type of product"}
                    }
                },
                "optional_requirements": {},
                "description": "Generic schema for product type detection"
            }
        
        print(f"[SCHEMA] Loading requirements schema for '{product_type}' from Azure")

        # First check in-memory cache to avoid repeated DB/LLM calls
        cached = schema_cache.get(product_type)
        if cached:
            print(f"[SCHEMA] Using cached schema for '{product_type}'")
            return cached

        # Try to load from Azure first (with timeout protection)
        try:
            # Use a shorter timeout for schema loading to fail fast
            schema = azure_blob_file_manager.get_schema_from_azure(product_type)
        except Exception as db_error:
            print(f"[SCHEMA] Azure Blob error for '{product_type}': {str(db_error)}")
            schema = None

        if schema and schema.get("mandatory_requirements") and schema.get("optional_requirements"):
            print(f"[SCHEMA] Successfully loaded schema from Azure for '{product_type}'")
            # cache it
            try:
                schema_cache.set(product_type, schema)
            except Exception:
                pass
            return schema

        print(f"[SCHEMA] No schema found in MongoDB for '{product_type}', attempting to build from web...")
        # Prevent concurrent builds for the same product_type
        lock = schema_build_locks.setdefault(product_type, threading.Lock())
        acquired = lock.acquire(blocking=False)
        if not acquired:
            # Another thread is building it; wait for it to finish (with timeout)
            print(f"[SCHEMA] Wait: another build in progress for '{product_type}', waiting for completion")
            got = lock.acquire(timeout=120)
            if not got:
                # Timeout - fall back to empty schema to avoid hanging
                print(f"[SCHEMA] Timeout waiting for schema build for '{product_type}', returning empty schema")
                return {"product_type": product_type, "mandatory_requirements": {}, "optional_requirements": {}}
            else:
                lock.release()
                # Check cache again
                cached_after = schema_cache.get(product_type)
                if cached_after:
                    print(f"[SCHEMA] Build completed by other thread; using cached schema for '{product_type}'")
                    return cached_after
                else:
                    return {"product_type": product_type, "mandatory_requirements": {}, "optional_requirements": {}}
        try:
            # Only the thread that acquired the lock will reach here
            built = build_requirements_schema_from_web(product_type)
            
            # CRITICAL FIX: Ensure the schema is saved to Azure after automation
            # This is a safety net in case the schema wasn't saved during discover_top_vendors
            if built and built.get("mandatory_requirements"):
                try:
                    # Check if schema exists in Azure
                    existing = azure_blob_file_manager.get_schema_from_azure(product_type)
                    if not existing or not existing.get("mandatory_requirements"):
                        # Schema not in Azure, save it now
                        logger.info(f"[SCHEMA] Schema not found in Azure after automation, saving now for '{product_type}'")
                        _save_schema_to_specs(product_type, built)
                        logger.info(f"[SCHEMA] Successfully saved schema to Azure for '{product_type}'")
                except Exception as save_error:
                    logger.error(f"[SCHEMA] Failed to save schema to Azure after automation: {save_error}")
            
            # Cache the schema
            try:
                schema_cache.set(product_type, built)
            except Exception:
                pass
            return built
        finally:
            try:
                lock.release()
            except Exception:
                pass
            
    except Exception as e:
        print(f"[ERROR] Failed to load schema for '{product_type or 'Generic'}': {e}")
        # Return a basic fallback schema
        return {
            "product_type": product_type or "Generic",
            "mandatory_requirements": {},
            "optional_requirements": {},
            "error": f"Failed to load schema: {str(e)}"
        }
