"""
Common Workflow Nodes - Comprehensive Deduplication
Shared node functions used across multiple workflows to eliminate code duplication.

This module contains node functions that appear across different workflows:
- Instrument Identifier, Product Search, Solution, Comparison, Grounded Chat

REVISION: 2025-12-30
STATUS: Comprehensive deduplication with 10+ shared node patterns

Author: AI Product Recommender Backend Team
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import AgenticConfig

# Import global executor manager for bounded thread pool
from agentic.global_executor_manager import get_global_executor

from tools.intent_tools import classify_intent_tool, extract_requirements_tool
from tools.instrument_tools import identify_instruments_tool, identify_accessories_tool
from tools.schema_tools import load_schema_tool, validate_requirements_tool
from tools.search_tools import search_vendors_tool
from tools.analysis_tools import analyze_vendor_match_tool
from tools.ranking_tools import judge_analysis_tool, rank_products_tool
from tools.search_tools import search_product_images_tool, search_pdf_datasheets_tool

logger = logging.getLogger(__name__)


# ============================================================================
# PATTERN 1: INTENT CLASSIFICATION
# Used by: Instrument Identifier, Product Search, Solution, Comparison
# ============================================================================

def classify_initial_intent_node(state: Dict[str, Any], workflow_name: str = "WORKFLOW") -> Dict[str, Any]:
    """
    Shared Node: Initial Intent Classification.

    Determines if this is an instrument/accessory identification request.
    Used by: Instrument Identifier, Product Search workflows.

    Args:
        state: Workflow state dictionary
        workflow_name: Name of workflow for logging (e.g., "IDENTIFIER", "INSTRUMENT")

    Returns:
        Updated state with initial_intent field
    """
    logger.info(f"[{workflow_name}] Node: Initial intent classification...")

    try:
        result = classify_intent_tool.invoke({
            "user_input": state["user_input"],
            "context": None
        })

        if result.get("success"):
            state["initial_intent"] = result.get("intent", "requirements")
        else:
            state["initial_intent"] = "requirements"

        state["current_step"] = "identify_instruments"

        state["messages"] = state.get("messages", []) + [{
            "role": "system",
            "content": f"Initial intent: {state['initial_intent']}"
        }]

        logger.info(f"[{workflow_name}] Initial intent: {state['initial_intent']}")

    except Exception as e:
        logger.error(f"[{workflow_name}] Initial intent classification failed: {e}")
        state["error"] = str(e)

    return state


def classify_intent_with_comparison_detection_node(
    state: Dict[str, Any],
    workflow_name: str = "WORKFLOW",
    comparison_keywords: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Shared Node: Intent Classification + Comparison Detection.

    Extended version that also detects if the request is a comparison request.
    Used by: Solution workflow with comparison mode detection.

    Args:
        state: Workflow state dictionary
        workflow_name: Name of workflow for logging
        comparison_keywords: Custom comparison keywords (optional)

    Returns:
        Updated state with intent, comparison_mode, and mode_confidence fields
    """
    logger.info(f"[{workflow_name}] Node: Intent classification with comparison detection...")

    try:
        # Step 1: Classify general intent
        result = classify_intent_tool.invoke({
            "user_input": state["user_input"],
            "context": None
        })

        if result.get("success"):
            state["intent"] = result.get("intent", "requirements")
            state["intent_confidence"] = result.get("confidence", 0.0)
        else:
            state["intent"] = "requirements"
            state["intent_confidence"] = 0.5

        # Step 2: Detect comparison mode
        user_input_lower = state["user_input"].lower()

        # Default comparison keywords
        if comparison_keywords is None:
            comparison_keywords = [
                "compare", "comparison", "versus", "vs", "vs.",
                "difference between", "better", "which is better",
                "pros and cons", "side by side", " or ",
                "which one", "what's the difference"
            ]

        # Check for comparison intent
        is_comparison = any(kw in user_input_lower for kw in comparison_keywords)

        # Check for multiple vendor/model mentions
        vendors_mentioned = sum(1 for v in ["honeywell", "emerson", "abb", "yokogawa", "siemens", "rosemount"]
                                if v in user_input_lower)
        if vendors_mentioned >= 2:
            is_comparison = True

        # Set comparison mode
        if is_comparison:
            state["comparison_mode"] = True
            state["mode_confidence"] = 0.95 if vendors_mentioned >= 2 else 0.75
            logger.info(f"[{workflow_name}] ✓ Comparison mode detected (confidence: {state['mode_confidence']:.2f})")
        else:
            state["comparison_mode"] = False
            state["mode_confidence"] = 0.85

        state["current_step"] = "validate_requirements"

        state["messages"] = state.get("messages", []) + [{
            "role": "system",
            "content": f"Intent: {state['intent']}, Comparison mode: {state['comparison_mode']} (confidence: {state['mode_confidence']:.2f})"
        }]

        logger.info(f"[{workflow_name}] Intent: {state['intent']}, Comparison: {state['comparison_mode']}")

    except Exception as e:
        logger.error(f"[{workflow_name}] Intent classification failed: {e}")
        state["error"] = str(e)
        state["intent"] = "requirements"
        state["comparison_mode"] = False

    return state


# ============================================================================
# PATTERN 2: INSTRUMENT/ACCESSORY IDENTIFICATION
# Used by: Instrument Identifier, Product Search
# ============================================================================

def identify_instruments_and_accessories_node(
    state: Dict[str, Any],
    workflow_name: str = "WORKFLOW",
    include_sample_input: bool = True,
    build_item_list: bool = True
) -> Dict[str, Any]:
    """
    Shared Node: Instrument/Accessory Identifier.

    Identifies ALL instruments and accessories from user requirements.
    Optionally generates sample_input for each item (for Solution workflow routing).

    Used by: Instrument Identifier, Product Search workflows.

    Args:
        state: Workflow state dictionary
        workflow_name: Name of workflow for logging
        include_sample_input: If True, generates sample_input field for each item
        build_item_list: If True, builds unified all_items list

    Returns:
        Updated state with identified_instruments, identified_accessories fields
    """
    logger.info(f"[{workflow_name}] Node: Identifying instruments and accessories...")

    try:
        # Identify instruments
        inst_result = identify_instruments_tool.invoke({
            "requirements": state["user_input"]
        })

        if inst_result.get("success"):
            state["identified_instruments"] = inst_result.get("instruments", [])
            # Also capture accessories from the first pass (Unified Prompt)
            state["identified_accessories"] = inst_result.get("accessories", [])
            state["project_name"] = inst_result.get("project_name", "Untitled Project")

        # Identify accessories (Fallback: only if none found in first pass)
        # This aligns with main.py which uses a single unified pass, but keeps the
        # agentic capability to find them if the first pass missed them.
        if not state.get("identified_accessories"):
            acc_result = identify_accessories_tool.invoke({
                "instruments": state["identified_instruments"],
                "process_context": state["user_input"]
            })

            if acc_result.get("success"):
                state["identified_accessories"] = acc_result.get("accessories", [])

        # If no items found, create a default based on the input
        if not state["identified_instruments"] and not state["identified_accessories"]:
            state["identified_instruments"] = [{
                "category": "Industrial Instrument",
                "product_name": "General Instrument",
                "quantity": 1,
                "specifications": {},
                "sample_input": "Industrial instrument based on requirements" if include_sample_input else None
            }]

        # Build unified item list (if requested)
        if build_item_list:
            all_items = []
            item_number = 1

            # Add instruments
            for instrument in state["identified_instruments"]:
                item = {
                    "number": item_number,
                    "type": "instrument",
                    "name": instrument.get("product_name", "Unknown Instrument"),
                    "category": instrument.get("category", "Instrument"),
                    "quantity": instrument.get("quantity", 1),
                    "specifications": instrument.get("specifications", {}),
                    "strategy": instrument.get("strategy", "")
                }

                if include_sample_input:
                    item["sample_input"] = instrument.get("sample_input", "")

                all_items.append(item)
                item_number += 1

            # Add accessories
            for accessory in state["identified_accessories"]:
                item = {
                    "number": item_number,
                    "type": "accessory",
                    "name": accessory.get("accessory_name", "Unknown Accessory"),
                    "category": accessory.get("category", "Accessory"),
                    "quantity": accessory.get("quantity", 1),
                    "related_instrument": accessory.get("related_instrument", "")
                }

                if include_sample_input:
                    # Construct sample_input for accessories
                    acc_sample_input = f"{accessory.get('category', 'Accessory')} for {accessory.get('related_instrument', 'instruments')}"
                    item["sample_input"] = acc_sample_input

                all_items.append(item)
                item_number += 1

            state["all_items"] = all_items
            state["total_items"] = len(all_items)

        state["current_step"] = "format_list"

        total_items = len(state["identified_instruments"]) + len(state["identified_accessories"])

        state["messages"] = state.get("messages", []) + [{
            "role": "system",
            "content": f"Identified {len(state['identified_instruments'])} instruments and {len(state['identified_accessories'])} accessories"
        }]

        logger.info(f"[{workflow_name}] Found {total_items} items total")

    except Exception as e:
        logger.error(f"[{workflow_name}] Identification failed: {e}")
        state["error"] = str(e)

    return state


# ============================================================================
# PATTERN 3: REQUIREMENT EXTRACTION & VALIDATION
# Used by: Solution, Product Search
# ============================================================================

def extract_requirements_node(
    state: Dict[str, Any],
    workflow_name: str = "WORKFLOW"
) -> Dict[str, Any]:
    """
    Shared Node: Extract Requirements.

    Extracts product type and requirements from user input.
    Used by multiple workflows for initial requirement gathering.

    Args:
        state: Workflow state dictionary
        workflow_name: Name of workflow for logging

    Returns:
        Updated state with product_type, provided_requirements fields
    """
    logger.info(f"[{workflow_name}] Node: Extracting requirements...")

    try:
        result = extract_requirements_tool.invoke({
            "user_input": state["user_input"]
        })

        if result.get("success"):
            state["product_type"] = result.get("product_type", "")
            state["provided_requirements"] = result.get("requirements", {})
        else:
            state["product_type"] = ""
            state["provided_requirements"] = {}

        state["messages"] = state.get("messages", []) + [{
            "role": "system",
            "content": f"Extracted product type: {state.get('product_type', 'Unknown')}"
        }]

        logger.info(f"[{workflow_name}] Product type: {state.get('product_type')}")

    except Exception as e:
        logger.error(f"[{workflow_name}] Requirement extraction failed: {e}")
        state["error"] = str(e)

    return state


def validate_requirements_with_schema_node(
    state: Dict[str, Any],
    workflow_name: str = "WORKFLOW",
    user_input_key: str = "user_input",
    product_type_key: str = "product_type"
) -> Dict[str, Any]:
    """
    Shared Node: Validate Requirements with Schema.

    Extracts requirements, loads schema, and validates against schema.
    Used by: Solution, Product Search workflows.

    Args:
        state: Workflow state dictionary
        workflow_name: Name of workflow for logging
        user_input_key: Key for user input in state
        product_type_key: Key for product type in state

    Returns:
        Updated state with schema, is_requirements_valid, missing_requirements fields
    """
    logger.info(f"[{workflow_name}] Node: Validating requirements with schema...")

    try:
        # Extract requirements
        extract_result = extract_requirements_tool.invoke({
            "user_input": state[user_input_key]
        })

        if extract_result.get("success"):
            state[product_type_key] = extract_result.get("product_type", "")
            state["provided_requirements"] = extract_result.get("requirements", {})

        # Load schema if product type detected
        if state[product_type_key]:
            schema_result = load_schema_tool.invoke({
                "product_type": state[product_type_key]
            })

            if schema_result.get("success"):
                state["schema"] = schema_result.get("schema")
                state["schema_source"] = schema_result.get("source", "mongodb")

        # Validate against schema
        if state.get("schema"):
            validate_result = validate_requirements_tool.invoke({
                "user_input": state[user_input_key],
                "product_type": state[product_type_key],
                "schema": state["schema"]
            })

            if validate_result.get("success"):
                state["is_requirements_valid"] = validate_result.get("is_valid", False)
                state["missing_requirements"] = validate_result.get("missing_fields", [])

        state["current_step"] = "aggregate_data"

        state["messages"] = state.get("messages", []) + [{
            "role": "system",
            "content": f"Validation: {'Valid' if state.get('is_requirements_valid') else 'Missing info'}, Product: {state[product_type_key]}"
        }]

        logger.info(f"[{workflow_name}] Product type: {state[product_type_key]}, Valid: {state.get('is_requirements_valid')}")

    except Exception as e:
        logger.error(f"[{workflow_name}] Validation failed: {e}")
        state["error"] = str(e)

    return state


# ============================================================================
# PATTERN 4: RAG AGGREGATION
# Used by: Solution, Product Search, Comparison
# ============================================================================

def aggregate_rag_data_node(
    state: Dict[str, Any],
    workflow_name: str = "WORKFLOW",
    product_type_key: str = "product_type",
    requirements_key: str = "provided_requirements",
    output_context_key: str = "rag_context"
) -> Dict[str, Any]:
    """
    Shared Node: RAG Data Aggregator.

    Queries all three RAG systems in parallel (Strategy, Standards, Inventory)
    and merges results into unified constraint context.

    Used by: Solution, Product Search, Comparison workflows.

    Args:
        state: Workflow state dictionary
        workflow_name: Name of workflow for logging
        product_type_key: Key for product type in state
        requirements_key: Key for requirements in state
        output_context_key: Key to store RAG results in state

    Returns:
        Updated state with rag_context, strategy_present, allowed_vendors fields
    """
    logger.info(f"[{workflow_name}] Node: Aggregating RAG data...")

    try:
        from .rag_components import create_rag_aggregator

        rag_aggregator = create_rag_aggregator()

        # Query all RAGs in parallel
        rag_results = rag_aggregator.query_all_parallel(
            product_type=state.get(product_type_key) or "industrial instrument",
            requirements=state.get(requirements_key, {})
        )

        state[output_context_key] = rag_results

        # Check if strategy is present
        strategy_data = rag_results.get("strategy", {}).get("data", {})
        has_preferred = bool(strategy_data.get("preferred_vendors", []))
        has_forbidden = bool(strategy_data.get("forbidden_vendors", []))
        state["strategy_present"] = has_preferred or has_forbidden

        # Get allowed vendors from strategy
        if state["strategy_present"]:
            allowed = strategy_data.get("preferred_vendors", []) + strategy_data.get("neutral_vendors", [])
            forbidden = strategy_data.get("forbidden_vendors", [])
            state["allowed_vendors"] = [v for v in allowed if v not in forbidden]

        state["current_step"] = "lookup_schema"

        state["messages"] = state.get("messages", []) + [{
            "role": "system",
            "content": f"RAG data aggregated. Strategy present: {state['strategy_present']}"
        }]

        logger.info(f"[{workflow_name}] Strategy present: {state['strategy_present']}")

    except Exception as e:
        logger.error(f"[{workflow_name}] RAG aggregation failed: {e}")
        state["error"] = str(e)
        state[output_context_key] = {}

    return state


# ============================================================================
# PATTERN 5: SCHEMA LOOKUP
# Used by: Solution, Product Search, Grounded Chat
# ============================================================================

def lookup_schema_node(
    state: Dict[str, Any],
    workflow_name: str = "WORKFLOW",
    product_type_key: str = "product_type"
) -> Dict[str, Any]:
    """
    Shared Node: Schema Lookup.

    Looks up or generates schema for product type from MongoDB.
    Used by: Solution, Product Search workflows.

    Args:
        state: Workflow state dictionary
        workflow_name: Name of workflow for logging
        product_type_key: Key for product type in state

    Returns:
        Updated state with schema, schema_source fields
    """
    logger.info(f"[{workflow_name}] Node: Schema lookup...")

    try:
        if not state.get("schema"):
            # Attempt schema lookup from MongoDB
            schema_result = load_schema_tool.invoke({
                "product_type": state.get(product_type_key) or "industrial instrument"
            })

            if schema_result.get("success") and schema_result.get("schema"):
                state["schema"] = schema_result["schema"]
                state["schema_source"] = "mongodb"
            else:
                # No schema found - may trigger Potential Product Index
                state["schema_source"] = "not_found"
                logger.info(f"[{workflow_name}] Schema not found - may need Potential Product Index")

        state["current_step"] = "apply_strategy"

        state["messages"] = state.get("messages", []) + [{
            "role": "system",
            "content": f"Schema source: {state.get('schema_source', 'unknown')}"
        }]

    except Exception as e:
        logger.error(f"[{workflow_name}] Schema lookup failed: {e}")
        state["error"] = str(e)

    return state


# ============================================================================
# PATTERN 6: STRATEGY APPLICATION & VENDOR FILTERING
# Used by: Solution, Product Search, Comparison
# ============================================================================

def apply_strategy_and_filter_vendors_node(
    state: Dict[str, Any],
    workflow_name: str = "WORKFLOW",
    product_type_key: str = "product_type",
    requirements_key: str = "provided_requirements",
    max_vendors: int = 5,
    refinery: str = None
) -> Dict[str, Any]:
    """
    Shared Node: Strategy Application & Vendor Filtering.

    Searches for vendors, applies CSV-based strategy filter rules,
    and returns filtered/prioritized vendor list.

    UPDATED: Now uses CSV-based strategy filtering instead of RAG.
    The CSV file (instrumentation_procurement_strategy.csv) contains
    vendor strategy statements that are used to prioritize vendors.

    Used by: Solution, Product Search, Comparison workflows.

    Args:
        state: Workflow state dictionary
        workflow_name: Name of workflow for logging
        product_type_key: Key for product type in state
        requirements_key: Key for requirements in state
        max_vendors: Maximum number of vendors to return
        refinery: Optional refinery context for filtering

    Returns:
        Updated state with available_vendors, filtered_vendors fields
    """
    logger.info(f"[{workflow_name}] Node: Applying CSV-based strategy filter...")

    try:
        from .strategy_csv_filter import filter_vendors_by_strategy

        product_type = state.get(product_type_key) or "industrial instrument"

        # Search for vendors
        vendor_result = search_vendors_tool.invoke({
            "product_type": product_type,
            "requirements": state.get(requirements_key, {})
        })

        available_vendors = vendor_result.get("vendors", [])
        if not available_vendors:
            available_vendors = ["Honeywell", "Emerson", "Yokogawa", "ABB", "Siemens"]

        state["available_vendors"] = available_vendors

        # Apply CSV-based strategy filter
        filter_result = filter_vendors_by_strategy(
            product_type=product_type,
            available_vendors=available_vendors,
            refinery=refinery
        )

        filtered_vendors = filter_result.get("filtered_vendors", [])

        # Extract vendor names sorted by priority score
        state["filtered_vendors"] = [v["vendor"] for v in filtered_vendors][:max_vendors]

        # Store strategy info for later use
        state["strategy_filter_result"] = {
            "category": filter_result.get("category"),
            "subcategory": filter_result.get("subcategory"),
            "vendor_strategies": {
                v["vendor"]: {
                    "strategy": v.get("strategy", ""),
                    "priority_score": v.get("priority_score", 0),
                    "comments": v.get("comments", "")
                }
                for v in filtered_vendors[:max_vendors]
            }
        }

        # Mark strategy as present if we found category matches
        state["strategy_present"] = any(v.get("category_match") for v in filtered_vendors)

        state["current_step"] = "parallel_analysis"

        state["messages"] = state.get("messages", []) + [{
            "role": "system",
            "content": f"CSV Strategy filter applied: {len(state['filtered_vendors'])}/{len(state['available_vendors'])} vendors for {filter_result.get('category', 'unknown')} category"
        }]

        logger.info(f"[{workflow_name}] Filtered vendors: {state['filtered_vendors']}")
        if state["strategy_present"]:
            logger.info(f"[{workflow_name}] Category: {filter_result.get('category')}")

    except Exception as e:
        logger.error(f"[{workflow_name}] Strategy filtering failed: {e}")
        state["error"] = str(e)
        # Fallback: use available vendors without filtering
        state["filtered_vendors"] = state.get("available_vendors", [])[:max_vendors]

    return state


# ============================================================================
# PATTERN 7: PARALLEL VENDOR ANALYSIS (Thread-safe)
# Used by: Solution, Product Search, Comparison
# ============================================================================

def parallel_vendor_analysis_node(
    state: Dict[str, Any],
    workflow_name: str = "WORKFLOW",
    vendors_key: str = "filtered_vendors",
    requirements_key: str = "provided_requirements",
    product_type_key: str = "product_type",
    max_workers: int = None,
    include_pdf_search: bool = True,
    include_image_search: bool = True
) -> Dict[str, Any]:
    """
    Shared Node: Parallel Vendor Analysis with Thread-Safety.

    Performs parallel analysis of multiple vendors using ThreadPoolExecutor
    with ThreadSafeResultCollector for safe result accumulation.

    Optionally searches for PDFs and images in parallel.

    Used by: Solution, Product Search, Comparison workflows.

    Args:
        state: Workflow state dictionary
        workflow_name: Name of workflow for logging
        vendors_key: Key for vendor list in state
        requirements_key: Key for requirements in state
        product_type_key: Key for product type in state
        max_workers: Maximum concurrent workers
        include_pdf_search: Whether to search for PDF datasheets
        include_image_search: Whether to search for product images

    Returns:
        Updated state with parallel_analysis_results field
    """
    logger.info(f"[{workflow_name}] Node: Parallel vendor analysis...")

    try:
        # Use config default for max_workers if not specified
        max_workers = max_workers or AgenticConfig.MAX_WORKERS

        # Import thread-safe collector
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utils.workflow_sync import ThreadSafeResultCollector

        vendors = state.get(vendors_key, [])
        requirements = state.get(requirements_key, {})
        product_type = state.get(product_type_key, "industrial instrument")

        # Thread-safe collectors
        vendor_data_collector = ThreadSafeResultCollector()
        analysis_collector = ThreadSafeResultCollector()

        # Phase 1: Search for PDFs and images in parallel (if enabled)
        vendor_data = {}
        if include_pdf_search or include_image_search:
            # Use global executor for bounded thread pool (max 16 workers globally)
            executor = get_global_executor()
            futures = {}

            for vendor in vendors:
                if include_pdf_search:
                    # Search for PDF datasheets
                    pdf_future = executor.submit(
                        search_pdf_datasheets_tool.invoke,
                        {
                            "vendor": vendor,
                            "product_type": product_type,
                            "model_family": None
                        }
                    )
                    futures[pdf_future] = ("pdf", vendor)

                if include_image_search:
                    # Search for product images
                    img_future = executor.submit(
                        search_product_images_tool.invoke,
                        {
                            "vendor": vendor,
                            "product_name": product_type,
                            "product_type": product_type
                        }
                    )
                    futures[img_future] = ("image", vendor)

            # Collect PDF and image results (thread-safe) with timeout
            for future in as_completed(futures, timeout=AgenticConfig.PARALLEL_ANALYSIS_TIMEOUT):
                result_type, vendor = futures[future]
                try:
                    result = future.result(timeout=AgenticConfig.DEFAULT_TIMEOUT)
                    if vendor not in vendor_data:
                        vendor_data[vendor] = {"vendor": vendor}

                    if result_type == "pdf" and result.get("success"):
                        vendor_data[vendor]["pdf_url"] = result.get("pdf_urls", [None])[0]
                    elif result_type == "image" and result.get("success"):
                        vendor_data[vendor]["image_url"] = result.get("images", [None])[0]
                except TimeoutError:
                    logger.error(f"Search timeout for {vendor}")
                    vendor_data_collector.add_error(TimeoutError("Search timeout"), {"vendor": vendor, "type": result_type})
                except Exception as e:
                    logger.error(f"Search failed for {vendor}: {e}")
                    vendor_data_collector.add_error(e, {"vendor": vendor, "type": result_type})

        # Phase 2: Analyze each vendor in parallel
        # Use global executor for bounded thread pool (max 16 workers globally)
        executor = get_global_executor()
        analyze_futures = {}

        for vendor in vendors:
            analyze_futures[executor.submit(
                analyze_vendor_match_tool.invoke,
                {
                    "vendor": vendor,
                    "requirements": requirements,
                    "pdf_content": None,
                    "product_data": vendor_data.get(vendor, {})
                }
            )] = vendor

        for future in as_completed(analyze_futures, timeout=AgenticConfig.PARALLEL_ANALYSIS_TIMEOUT):
            vendor = analyze_futures[future]
            try:
                result = future.result(timeout=AgenticConfig.LONG_TIMEOUT)
                if result.get("success"):
                    analysis_result = {
                        "vendor": vendor,
                        "product_name": result.get("product_name", ""),
                        "model_family": result.get("model_family", ""),
                        "match_score": result.get("match_score", 0),
                        "requirements_match": result.get("requirements_match", False),
                        "matched_requirements": result.get("matched_requirements", {}),
                        "unmatched_requirements": result.get("unmatched_requirements", []),
                        "reasoning": result.get("reasoning", ""),
                        "limitations": result.get("limitations"),
                        "pdf_source": vendor_data.get(vendor, {}).get("pdf_url"),
                        "image_url": vendor_data.get(vendor, {}).get("image_url")
                    }
                    analysis_collector.add_result(analysis_result)
            except TimeoutError:
                logger.error(f"Analysis timeout for {vendor}")
                analysis_collector.add_error(TimeoutError("Analysis timeout"), {"vendor": vendor})
            except Exception as e:
                logger.error(f"Analysis failed for {vendor}: {e}")
                analysis_collector.add_error(e, {"vendor": vendor})

        # Get thread-safe results
        analysis_results = analysis_collector.get_results()

        # Log summary
        summary = analysis_collector.summary()
        logger.info(
            f"[{workflow_name}] Parallel analysis complete: "
            f"{summary['total_results']} successes, {summary['total_errors']} errors"
        )

        # Update state
        state["parallel_analysis_results"] = analysis_results
        state["current_step"] = "summarize"

        state["messages"] = state.get("messages", []) + [{
            "role": "system",
            "content": f"Analyzed {len(analysis_results)} vendors"
        }]

        logger.info(f"[{workflow_name}] Analyzed {len(analysis_results)} vendors")

    except Exception as e:
        logger.error(f"[{workflow_name}] Parallel analysis failed: {e}")
        state["error"] = str(e)
        state["parallel_analysis_results"] = []

    return state


# ============================================================================
# PATTERN 8: JUDGE/VALIDATION
# Used by: Solution, Product Search, Comparison
# ============================================================================

def judge_analysis_results_node(
    state: Dict[str, Any],
    workflow_name: str = "WORKFLOW",
    analysis_results_key: str = "parallel_analysis_results",
    requirements_key: str = "provided_requirements"
) -> Dict[str, Any]:
    """
    Shared Node: Judge/Validate Analysis Results.

    Validates vendor analysis results to ensure quality.
    Used by: Solution, Instrument Detail, Comparison workflows.

    Args:
        state: Workflow state dictionary
        workflow_name: Name of workflow for logging
        analysis_results_key: Key for analysis results in state
        requirements_key: Key for requirements in state

    Returns:
        Updated state with judge_validation field
    """
    logger.info(f"[{workflow_name}] Node: Judging results...")

    try:
        analysis_results = state.get(analysis_results_key, [])

        # Skip judging if no analysis results
        if not analysis_results:
            logger.warning(f"[{workflow_name}] No analysis results to judge, skipping...")
            state["judge_validation"] = {"passed": [], "failed": [], "skipped": True}
            state["current_step"] = "rank"
            return state

        # Convert list to dict format expected by JudgeAnalysisInput
        vendor_analysis_dict = {
            "vendor_matches": analysis_results,
            "total_analyzed": len(analysis_results),
            "analysis_type": "parallel_vendor_analysis"
        }

        judge_result = judge_analysis_tool.invoke({
            "original_requirements": state.get(requirements_key, {}),
            "vendor_analysis": vendor_analysis_dict,
            "strategy_rules": None
        })

        if judge_result.get("success"):
            state["judge_validation"] = {
                "passed": judge_result.get("valid_matches", []),
                "failed": judge_result.get("invalid_matches", []),
                "summary": judge_result.get("summary", "")
            }

        state["current_step"] = "rank"

        state["messages"] = state.get("messages", []) + [{
            "role": "system",
            "content": f"Judge validation: {len(state.get('judge_validation', {}).get('passed', []))} passed"
        }]

    except Exception as e:
        logger.error(f"[{workflow_name}] Judging failed: {e}")
        state["error"] = str(e)
        state["judge_validation"] = {"passed": [], "failed": []}

    return state


# ============================================================================
# PATTERN 9: RANKING
# Used by: Solution, Instrument Detail, Comparison
# ============================================================================

def rank_products_node(
    state: Dict[str, Any],
    workflow_name: str = "WORKFLOW",
    analysis_results_key: str = "parallel_analysis_results",
    requirements_key: str = "provided_requirements",
    top_n: int = 5
) -> Dict[str, Any]:
    """
    Shared Node: Rank Products.

    Ranks vendor products based on analysis results and requirements.
    Used by: Solution, Instrument Detail, Comparison workflows.

    Args:
        state: Workflow state dictionary
        workflow_name: Name of workflow for logging
        analysis_results_key: Key for analysis results in state
        requirements_key: Key for requirements in state
        top_n: Number of top products to include in response

    Returns:
        Updated state with ranked_results, response, response_data fields
    """
    logger.info(f"[{workflow_name}] Node: Ranking products...")

    try:
        analysis_results = state.get(analysis_results_key, [])

        rank_result = rank_products_tool.invoke({
            "vendor_matches": analysis_results,
            "requirements": state.get(requirements_key, {})
        })

        if rank_result.get("success"):
            state["ranked_results"] = rank_result.get("ranked_products", [])

            # Generate response
            response_lines = ["**🎯 Product Recommendations**\n"]

            for i, product in enumerate(state["ranked_results"][:top_n], 1):
                response_lines.append(f"**#{i} {product.get('vendor', '')} - {product.get('product_name', '')}**")
                response_lines.append(f"   Match Score: {product.get('overall_score', 0)}/100")

                strengths = product.get("key_strengths", [])
                if strengths:
                    response_lines.append(f"   ✓ {', '.join(strengths[:2])}")

                concerns = product.get("concerns", [])
                if concerns:
                    response_lines.append(f"   ⚠ {concerns[0]}")

                response_lines.append("")

            state["response"] = "\n".join(response_lines)
            state["response_data"] = {
                "ranked_products": state["ranked_results"],
                "total_analyzed": len(analysis_results),
                "product_type": state.get("product_type")
            }

        state["current_step"] = "complete"

        state["messages"] = state.get("messages", []) + [{
            "role": "system",
            "content": f"Ranked {len(state.get('ranked_results', []))} products"
        }]

        logger.info(f"[{workflow_name}] Ranking complete")

    except Exception as e:
        logger.error(f"[{workflow_name}] Ranking failed: {e}")
        state["error"] = str(e)
        state["ranked_results"] = []

    return state


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def has_identification_results(state: Dict[str, Any]) -> bool:
    """
    Check if state has valid identification results.

    Args:
        state: Workflow state dictionary

    Returns:
        True if instruments or accessories were identified
    """
    instruments = state.get("identified_instruments", [])
    accessories = state.get("identified_accessories", [])
    return len(instruments) > 0 or len(accessories) > 0


def get_total_items(state: Dict[str, Any]) -> int:
    """
    Get total number of identified items.

    Args:
        state: Workflow state dictionary

    Returns:
        Total count of instruments + accessories
    """
    instruments = state.get("identified_instruments", [])
    accessories = state.get("identified_accessories", [])
    return len(instruments) + len(accessories)


def format_item_summary(state: Dict[str, Any]) -> str:
    """
    Format a summary of identified items.

    Args:
        state: Workflow state dictionary

    Returns:
        Human-readable summary string
    """
    instruments = state.get("identified_instruments", [])
    accessories = state.get("identified_accessories", [])
    project_name = state.get("project_name", "Project")

    total = len(instruments) + len(accessories)

    return (
        f"**{project_name}**: "
        f"{len(instruments)} instrument(s), "
        f"{len(accessories)} accessory(ies) "
        f"({total} total items)"
    )


# ============================================================================
# NODE FACTORY - Create workflow-specific wrappers
# ============================================================================

def create_node_wrapper(
    node_func: Callable,
    workflow_name: str,
    **kwargs
) -> Callable:
    """
    Create a workflow-specific wrapper for a shared node function.

    This factory function creates lightweight wrappers that call shared nodes
    with workflow-specific parameters.

    Args:
        node_func: The shared node function to wrap
        workflow_name: Name of the workflow (for logging)
        **kwargs: Additional keyword arguments to pass to the node function

    Returns:
        Wrapper function that can be used as a workflow node

    Example:
        >>> classify_wrapper = create_node_wrapper(
        ...     classify_initial_intent_node,
        ...     workflow_name="IDENTIFIER"
        ... )
        >>> # Use classify_wrapper as a node in your workflow
    """
    def wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
        return node_func(state, workflow_name=workflow_name, **kwargs)

    # Preserve function metadata
    wrapper.__name__ = f"{node_func.__name__}_for_{workflow_name}"
    wrapper.__doc__ = f"Wrapper for {workflow_name} workflow: {node_func.__doc__}"

    return wrapper


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Pattern 1: Intent Classification
    "classify_initial_intent_node",
    "classify_intent_with_comparison_detection_node",

    # Pattern 2: Instrument/Accessory Identification
    "identify_instruments_and_accessories_node",

    # Pattern 3: Requirement Extraction & Validation
    "extract_requirements_node",
    "validate_requirements_with_schema_node",

    # Pattern 4: RAG Aggregation
    "aggregate_rag_data_node",

    # Pattern 5: Schema Lookup
    "lookup_schema_node",

    # Pattern 6: Strategy Application & Vendor Filtering
    "apply_strategy_and_filter_vendors_node",

    # Pattern 7: Parallel Vendor Analysis
    "parallel_vendor_analysis_node",

    # Pattern 8: Judge/Validation
    "judge_analysis_results_node",

    # Pattern 9: Ranking
    "rank_products_node",

    # Utilities
    "has_identification_results",
    "get_total_items",
    "format_item_summary",
    "create_node_wrapper"
]
