# tools/parallel_indexer.py
# Parallel Indexing for Index RAG
# Executes database indexing and web search in parallel threads

import logging
import json
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from datetime import datetime
import time

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# ============================================================================
# DATABASE INDEXING
# ============================================================================

def index_database(
    query: str,
    product_types: List[str],
    vendors: List[str],
    model_hints: List[str] = None,
    top_k: int = 7
) -> Dict[str, Any]:
    """
    Index products from Azure Blob Storage database.
    
    This function now includes the metadata filter internally (Layer 1).
    
    Args:
        query: User's search query
        product_types: Product types to search
        vendors: Vendors to include (can be empty for auto-discovery)
        model_hints: Optional model identifiers
        top_k: Max results per vendor
        
    Returns:
        Database indexing results with metadata filter applied
    """
    start_time = time.time()
    logger.info(f"[ParallelIndexer] Starting database indexing...")
    
    try:
        from azure_blob_utils import get_products_for_vendors
        from tools.metadata_filter import filter_by_hierarchy
        
        # Step 1: Apply metadata filter (Layer 1) - INSIDE the thread
        logger.info(f"[ParallelIndexer] Applying metadata filter for: {query[:50]}...")
        
        metadata_result = filter_by_hierarchy(
            query=query,
            top_k=top_k,
            explicit_product_type=product_types[0] if product_types else None,
            explicit_vendors=vendors if vendors else None
        )
        
        # Use filtered vendors from metadata filter
        filtered_vendors = metadata_result.get('vendors', vendors)
        detected_product_types = metadata_result.get('product_types', product_types)
        filter_applied = metadata_result.get('filter_applied', False)
        
        logger.info(f"[ParallelIndexer] Metadata filter: {len(filtered_vendors)} vendors, "
                   f"product_types={detected_product_types}, applied={filter_applied}")
        
        if not filtered_vendors:
            logger.warning("[ParallelIndexer] No vendors found after metadata filter")
            return {
                "success": True,
                "source": "database",
                "results": [],
                "count": 0,
                "pdf_count": 0,
                "metadata_filter": {
                    "applied": filter_applied,
                    "product_types": detected_product_types,
                    "vendors_count": 0
                }
            }
        
        # Step 2: Get product type in proper format
        product_type = detected_product_types[0].replace("_", " ") if detected_product_types else None
        
        # Step 3: Fetch JSON products from Azure Blob using filtered vendors
        products_data = get_products_for_vendors(filtered_vendors, product_type)
        
        # Build indexed results from JSON
        indexed_results = []
        
        for vendor, products in products_data.items():
            for product in products[:top_k]:  # Limit per vendor
                # Extract key fields
                if isinstance(product, dict):
                    indexed_item = {
                        "source": "database",
                        "vendor": vendor,
                        "product_type": product_type,
                        "data": product,
                        "indexed_at": datetime.utcnow().isoformat()
                    }
                    
                    # Try to extract model info
                    if "model" in product:
                        indexed_item["model"] = product["model"]
                    elif "models" in product and isinstance(product["models"], list):
                        indexed_item["models"] = product["models"]
                    
                    indexed_results.append(indexed_item)
        
        logger.info(f"[ParallelIndexer] JSON products indexed: {len(indexed_results)}")
        
        # Step 4: Index PDF documents from Azure Blob (NEW)
        pdf_results = []
        try:
            from tools.pdf_indexer import index_pdf_documents
            
            pdf_result = index_pdf_documents(
                query=query,
                product_types=detected_product_types,
                vendors=filtered_vendors,
                max_pdfs_per_vendor=3,
                top_k=top_k
            )
            
            if pdf_result.get("success") and pdf_result.get("results"):
                pdf_results = pdf_result["results"]
                logger.info(f"[ParallelIndexer] PDF documents indexed: {len(pdf_results)}")
        except ImportError:
            logger.warning("[ParallelIndexer] PDF indexer not available, skipping PDF indexing")
        except Exception as pdf_error:
            logger.warning(f"[ParallelIndexer] PDF indexing failed: {pdf_error}")
        
        # Step 5: Merge JSON + PDF results
        all_results = indexed_results + pdf_results
        
        elapsed = time.time() - start_time
        
        result = {
            "success": True,
            "source": "database",
            "results": all_results,
            "count": len(all_results),
            "json_count": len(indexed_results),
            "pdf_count": len(pdf_results),
            "vendors_found": list(products_data.keys()),
            "elapsed_ms": int(elapsed * 1000),
            "metadata_filter": {
                "applied": filter_applied,
                "product_types": detected_product_types,
                "vendors_count": len(filtered_vendors)
            }
        }
        
        logger.info(f"[ParallelIndexer] Database indexing complete: {len(all_results)} results "
                   f"({len(indexed_results)} JSON + {len(pdf_results)} PDF) in {elapsed:.2f}s")
        return result
        
    except Exception as e:
        logger.error(f"[ParallelIndexer] Database indexing failed: {e}", exc_info=True)
        return {
            "success": False,
            "source": "database",
            "error": str(e),
            "results": [],
            "count": 0
        }


# ============================================================================
# WEB SEARCH INDEXING
# ============================================================================

def index_web_search(
    query: str,
    product_types: List[str],
    vendors: List[str] = None,
    max_results: int = 5
) -> Dict[str, Any]:
    """
    Perform LLM-powered web search for product information.
    
    Uses Gemini's grounding capabilities for web search.
    
    Args:
        query: Search query
        product_types: Product types being searched
        vendors: Optional vendor filter
        max_results: Maximum results to return
        
    Returns:
        Web search indexing results
    """
    start_time = time.time()
    logger.info(f"[ParallelIndexer] Starting web search indexing...")
    
    try:
        from llm_fallback import create_llm_with_fallback
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import JsonOutputParser
        
        # Build search context
        vendor_context = f" from vendors: {', '.join(vendors)}" if vendors else ""
        product_context = f" for {', '.join(pt.replace('_', ' ') for pt in product_types)}" if product_types else ""
        
        search_prompt = f"""
You are a product search assistant for industrial instrumentation.

Search Query: {query}

Context:
- Product Type: {product_context}
- Vendors: {vendor_context}

Based on your knowledge of industrial instrumentation products, provide information about matching products.

Return ONLY valid JSON with this structure:
{{
    "products": [
        {{
            "vendor": "<vendor name>",
            "model": "<model number>",
            "product_type": "<product type>",
            "description": "<brief description>",
            "key_specs": {{
                "<spec_name>": "<spec_value>"
            }},
            "relevance_score": <0.0-1.0>
        }}
    ],
    "search_summary": "<brief summary of findings>",
    "total_found": <number>
}}

Limit to {max_results} most relevant products.
"""
        
        llm = create_llm_with_fallback(
            model="gemini-2.5-flash",
            temperature=0.2,
            max_tokens=2000
        )
        
        prompt = ChatPromptTemplate.from_template("{search_prompt}")
        parser = JsonOutputParser()
        chain = prompt | llm | parser
        
        response = chain.invoke({"search_prompt": search_prompt})
        
        # Convert to indexed format
        indexed_results = []
        for product in response.get("products", []):
            indexed_item = {
                "source": "web_search",
                "vendor": product.get("vendor", "Unknown"),
                "model": product.get("model", ""),
                "product_type": product.get("product_type", ""),
                "data": product,
                "relevance_score": product.get("relevance_score", 0.5),
                "indexed_at": datetime.utcnow().isoformat()
            }
            indexed_results.append(indexed_item)
        
        elapsed = time.time() - start_time
        
        result = {
            "success": True,
            "source": "web_search",
            "results": indexed_results,
            "count": len(indexed_results),
            "search_summary": response.get("search_summary", ""),
            "elapsed_ms": int(elapsed * 1000)
        }
        
        logger.info(f"[ParallelIndexer] Web search complete: {len(indexed_results)} results in {elapsed:.2f}s")
        return result
        
    except Exception as e:
        logger.error(f"[ParallelIndexer] Web search failed: {e}", exc_info=True)
        return {
            "success": False,
            "source": "web_search",
            "error": str(e),
            "results": [],
            "count": 0
        }


# ============================================================================
# PARALLEL EXECUTION
# ============================================================================

def run_parallel_indexing(
    query: str,
    product_type: str = None,
    vendors: List[str] = None,
    top_k: int = 7,
    timeout_seconds: int = 30,
    enable_web_search: bool = True
) -> Dict[str, Any]:
    """
    Run database indexing and web search in parallel.
    
    Note: Metadata filter is now applied INSIDE the database indexing thread.
    
    Args:
        query: User's search query
        product_type: Optional product type (can be None for auto-detection)
        vendors: Optional vendor list (can be None for auto-discovery)
        top_k: Max results per source
        timeout_seconds: Timeout for parallel execution
        enable_web_search: Whether to enable web search thread
        
    Returns:
        Combined results from both indexing methods
    """
    start_time = time.time()
    logger.info("[ParallelIndexer] Starting parallel indexing...")
    
    product_types = [product_type] if product_type else []
    vendors_list = vendors or []
    model_hints = []
    
    results = {
        "success": True,
        "database": None,
        "web_search": None,
        "merged_results": [],
        "total_count": 0,
        "elapsed_ms": 0
    }
    
    # Define tasks
    tasks = []
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Task 1: Database indexing (includes metadata filter now)
        db_future = executor.submit(
            index_database,
            query=query,
            product_types=product_types,
            vendors=vendors_list,
            model_hints=model_hints,
            top_k=top_k
        )
        tasks.append(("database", db_future))
        
        # Task 2: Web search (optional)
        if enable_web_search:
            web_future = executor.submit(
                index_web_search,
                query=query,
                product_types=product_types,
                vendors=vendors_list[:5] if vendors_list else None,
                max_results=top_k
            )
            tasks.append(("web_search", web_future))
        
        # Collect results with timeout
        for task_name, future in tasks:
            try:
                task_result = future.result(timeout=timeout_seconds)
                results[task_name] = task_result
                
                if task_result.get("success") and task_result.get("results"):
                    results["merged_results"].extend(task_result["results"])
                    
            except TimeoutError:
                logger.warning(f"[ParallelIndexer] {task_name} timed out after {timeout_seconds}s")
                results[task_name] = {
                    "success": False,
                    "error": "Timeout",
                    "results": []
                }
            except Exception as e:
                logger.error(f"[ParallelIndexer] {task_name} failed: {e}")
                results[task_name] = {
                    "success": False,
                    "error": str(e),
                    "results": []
                }
    
    # Deduplicate merged results
    results["merged_results"] = deduplicate_results(results["merged_results"])
    results["total_count"] = len(results["merged_results"])
    results["elapsed_ms"] = int((time.time() - start_time) * 1000)
    
    # Check overall success
    db_success = results.get("database", {}).get("success", False)
    web_success = results.get("web_search", {}).get("success", False) if enable_web_search else True
    results["success"] = db_success or web_success
    
    logger.info(f"[ParallelIndexer] Parallel indexing complete: "
                f"{results['total_count']} total results in {results['elapsed_ms']}ms")
    
    return results


def deduplicate_results(results: List[Dict]) -> List[Dict]:
    """
    Remove duplicate products from merged results.
    
    Deduplication key: vendor + model (if available)
    """
    seen = set()
    deduplicated = []
    
    for result in results:
        vendor = result.get("vendor", "").lower()
        model = result.get("model", "").lower()
        
        # Create dedup key
        if model:
            key = f"{vendor}:{model}"
        else:
            # Use data hash for items without model
            data_str = json.dumps(result.get("data", {}), sort_keys=True)
            key = f"{vendor}:{hash(data_str)}"
        
        if key not in seen:
            seen.add(key)
            deduplicated.append(result)
    
    logger.info(f"[ParallelIndexer] Deduplicated: {len(results)} -> {len(deduplicated)} results")
    return deduplicated


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'index_database',
    'index_web_search',
    'run_parallel_indexing',
    'deduplicate_results'
]
