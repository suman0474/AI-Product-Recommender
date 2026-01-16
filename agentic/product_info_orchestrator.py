"""
Product Info Orchestrator

Handles parallel query execution across multiple RAG sources
and merges results into unified response.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

from .product_info_intent_agent import DataSource, classify_query, get_sources_for_hybrid
from .product_info_memory import (
    get_session, add_to_history, is_follow_up_query, 
    resolve_follow_up, set_context, get_context
)

logger = logging.getLogger(__name__)

# Thread pool for parallel execution
_executor = ThreadPoolExecutor(max_workers=4)


def query_index_rag(query: str, session_id: str) -> Dict[str, Any]:
    """Query Index RAG for product information."""
    try:
        from .index_rag.index_rag_workflow import run_index_rag_workflow
        
        result = run_index_rag_workflow(
            question=query,
            session_id=session_id
        )
        
        return {
            "success": True,
            "source": "index_rag",
            "answer": result.get("response", ""),
            "found_in_database": result.get("found_in_database", False),
            "data": result
        }
    except ImportError:
        logger.warning("[ORCHESTRATOR] Index RAG not available")
        return {"success": False, "source": "index_rag", "error": "Index RAG not available"}
    except Exception as e:
        logger.error(f"[ORCHESTRATOR] Index RAG error: {e}")
        return {"success": False, "source": "index_rag", "error": str(e)}


def query_standards_rag(query: str, session_id: str) -> Dict[str, Any]:
    """Query Standards RAG for standards information."""
    try:
        from .standards_rag.standards_rag_workflow import run_standards_rag_workflow
        
        result = run_standards_rag_workflow(
            question=query,
            session_id=session_id
        )
        
        return {
            "success": True,
            "source": "standards_rag",
            "answer": result.get("response", ""),
            "standards_cited": result.get("standards_cited", []),
            "data": result
        }
    except ImportError:
        logger.warning("[ORCHESTRATOR] Standards RAG not available")
        return {"success": False, "source": "standards_rag", "error": "Standards RAG not available"}
    except Exception as e:
        logger.error(f"[ORCHESTRATOR] Standards RAG error: {e}")
        return {"success": False, "source": "standards_rag", "error": str(e)}


def query_strategy_rag(query: str, session_id: str) -> Dict[str, Any]:
    """Query Strategy RAG for vendor strategy information."""
    try:
        from .strategy_rag.strategy_rag_workflow import run_strategy_rag_workflow
        
        result = run_strategy_rag_workflow(
            question=query,
            session_id=session_id
        )
        
        return {
            "success": True,
            "source": "strategy_rag",
            "answer": result.get("response", ""),
            "preferred_vendors": result.get("preferred_vendors", []),
            "data": result
        }
    except ImportError:
        logger.warning("[ORCHESTRATOR] Strategy RAG not available")
        return {"success": False, "source": "strategy_rag", "error": "Strategy RAG not available"}
    except Exception as e:
        logger.error(f"[ORCHESTRATOR] Strategy RAG error: {e}")
        return {"success": False, "source": "strategy_rag", "error": str(e)}


def query_deep_agent(query: str, session_id: str) -> Dict[str, Any]:
    """Query Deep Agent for detailed spec extraction."""
    try:
        from .deep_agent.standards_deep_agent import run_standards_deep_agent
        
        result = run_standards_deep_agent(
            query=query,
            context={"session_id": session_id}
        )
        
        return {
            "success": True,
            "source": "deep_agent",
            "answer": result.get("response", ""),
            "extracted_specs": result.get("extracted_specs", {}),
            "data": result
        }
    except ImportError:
        logger.warning("[ORCHESTRATOR] Deep Agent not available")
        return {"success": False, "source": "deep_agent", "error": "Deep Agent not available"}
    except Exception as e:
        logger.error(f"[ORCHESTRATOR] Deep Agent error: {e}")
        return {"success": False, "source": "deep_agent", "error": str(e)}


def query_llm_fallback(query: str, session_id: str) -> Dict[str, Any]:
    """Use LLM directly for general questions."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from dotenv import load_dotenv
        import os
        
        load_dotenv()
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3
        )
        
        response = llm.invoke(query)
        
        return {
            "success": True,
            "source": "llm",
            "answer": response.content if hasattr(response, 'content') else str(response),
            "data": {}
        }
    except Exception as e:
        logger.error(f"[ORCHESTRATOR] LLM fallback error: {e}")
        return {
            "success": False,
            "source": "llm",
            "error": str(e),
            "answer": "I'm sorry, I couldn't process your request. Please try again."
        }


def query_sources_parallel(
    query: str,
    sources: List[DataSource],
    session_id: str
) -> Dict[str, Dict[str, Any]]:
    """
    Query multiple sources in parallel.
    
    Args:
        query: User query
        sources: List of DataSource to query
        session_id: Session ID for memory
        
    Returns:
        Dict mapping source name to result
    """
    source_funcs = {
        DataSource.INDEX_RAG: query_index_rag,
        DataSource.STANDARDS_RAG: query_standards_rag,
        DataSource.STRATEGY_RAG: query_strategy_rag,
        DataSource.DEEP_AGENT: query_deep_agent,
        DataSource.LLM: query_llm_fallback
    }
    
    results = {}
    futures = {}
    
    for source in sources:
        if source in source_funcs:
            future = _executor.submit(source_funcs[source], query, session_id)
            futures[future] = source
    
    for future in as_completed(futures, timeout=30):
        source = futures[future]
        try:
            result = future.result()
            results[source.value] = result
        except Exception as e:
            logger.error(f"[ORCHESTRATOR] Error querying {source.value}: {e}")
            results[source.value] = {
                "success": False,
                "source": source.value,
                "error": str(e)
            }
    
    return results


def merge_results(
    results: Dict[str, Dict[str, Any]],
    primary_source: DataSource
) -> Dict[str, Any]:
    """
    Merge results from multiple sources into unified response.
    
    Args:
        results: Dict of source results
        primary_source: The primary source to prioritize
        
    Returns:
        Merged response dict
    """
    merged = {
        "success": False,
        "answer": "",
        "source": "unknown",
        "found_in_database": False,
        "sources_used": [],
        "metadata": {}
    }
    
    answer_parts = []
    sources_used = []
    
    # Process primary source first
    primary_key = primary_source.value
    if primary_key in results and results[primary_key].get("success"):
        primary_result = results[primary_key]
        answer_parts.append(primary_result.get("answer", ""))
        sources_used.append(primary_key)
        merged["found_in_database"] = primary_result.get("found_in_database", False)
        merged["metadata"][primary_key] = primary_result.get("data", {})
    
    # Add supplementary information from other sources
    for source_key, result in results.items():
        if source_key == primary_key:
            continue
        if result.get("success") and result.get("answer"):
            # Add as supplementary info
            sources_used.append(source_key)
            merged["metadata"][source_key] = result.get("data", {})
    
    # Build final answer
    if answer_parts:
        merged["answer"] = "\n\n".join(answer_parts)
        merged["success"] = True
        merged["source"] = "database" if merged["found_in_database"] else "llm"
    else:
        # No successful results - use first error or default message
        merged["answer"] = "I couldn't find specific information about your query."
        for result in results.values():
            if result.get("error"):
                logger.warning(f"[ORCHESTRATOR] Source error: {result['error']}")
    
    merged["sources_used"] = sources_used
    
    return merged


def run_product_info_query(
    query: str,
    session_id: str
) -> Dict[str, Any]:
    """
    Main entry point for Product Info queries.
    
    Handles:
    1. Memory resolution (follow-up detection)
    2. Intent classification
    3. Source selection
    4. Parallel querying
    5. Result merging
    
    Args:
        query: User query
        session_id: Session ID for memory tracking
        
    Returns:
        Unified response dict
    """
    logger.info(f"[ORCHESTRATOR] Processing query: {query[:100]}...")
    
    # Step 1: Memory resolution
    is_follow_up = is_follow_up_query(query, session_id)
    resolved_query = query
    
    if is_follow_up:
        logger.info(f"[ORCHESTRATOR] Detected follow-up query")
        resolved_query = resolve_follow_up(query, session_id)
    
    # Step 2: Intent classification
    primary_source, confidence, reasoning = classify_query(resolved_query)
    logger.info(f"[ORCHESTRATOR] Classification: {primary_source.value} ({confidence:.2f}) - {reasoning}")
    
    # Step 3: Determine sources to query
    if primary_source == DataSource.HYBRID:
        sources = get_sources_for_hybrid(resolved_query)
    else:
        sources = [primary_source]
    
    # Add LLM fallback if confidence is low
    if confidence < 0.5 and DataSource.LLM not in sources:
        sources.append(DataSource.LLM)
    
    # Step 4: Query sources in parallel
    results = query_sources_parallel(resolved_query, sources, session_id)
    
    # Step 5: Merge results
    merged = merge_results(results, primary_source)
    
    # Step 6: Save to memory
    add_to_history(
        session_id=session_id,
        query=query,
        response=merged.get("answer", ""),
        sources_used=merged.get("sources_used", [])
    )
    
    # Add metadata
    merged["is_follow_up"] = is_follow_up
    merged["classification"] = {
        "primary_source": primary_source.value,
        "confidence": confidence,
        "reasoning": reasoning
    }
    
    return merged
