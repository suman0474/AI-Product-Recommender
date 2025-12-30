# agentic/instrument_detail_workflow.py
# Instrument/Accessory Detail Capture Workflow
# Integrated with Comparison Workflow for vendor comparison

import json
import logging
from typing import Dict, Any, List, Literal, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from langgraph.graph import StateGraph, END

from .models import (
    InstrumentDetailState,
    create_instrument_detail_state,
    SpecObject
)
from .rag_components import RAGAggregator, StrategyFilter, create_rag_aggregator, create_strategy_filter
from .checkpointing import compile_with_checkpointing

from tools.intent_tools import classify_intent_tool, extract_requirements_tool
from tools.instrument_tools import identify_instruments_tool, identify_accessories_tool
from tools.schema_tools import load_schema_tool, validate_requirements_tool
from tools.vendor_tools import search_vendors_tool
from tools.analysis_tools import analyze_vendor_match_tool
from tools.ranking_tools import judge_analysis_tool, rank_products_tool
from tools.search_tools import search_product_images_tool, search_pdf_datasheets_tool

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()
logger = logging.getLogger(__name__)


# ============================================================================
# PROMPTS
# ============================================================================

INSTRUMENT_LIST_PROMPT = """
You are Engenie's Instrument Presenter. Format the identified instruments and accessories for user selection.

Identified Instruments:
{instruments}

Identified Accessories:
{accessories}

Project Name: {project_name}

Create a numbered list for user selection. Format:

Return ONLY valid JSON:
{{
    "formatted_list": [
        {{
            "number": 1,
            "type": "instrument" | "accessory",
            "name": "<product name>",
            "category": "<category>",
            "quantity": <quantity>,
            "key_specs": "<brief specs>"
        }}
    ],
    "total_items": <count>,
    "message": "<brief message asking user to select an item>"
}}
"""


# ============================================================================
# WORKFLOW NODES
# ============================================================================

def classify_initial_intent_node(state: InstrumentDetailState) -> InstrumentDetailState:
    """
    Agent 1: Initial Intent Classification.
    Determines if this is an instrument/accessory identification request.
    """
    logger.info("[INSTRUMENT] Node 1: Initial intent classification...")
    
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
        
        logger.info(f"[INSTRUMENT] Initial intent: {state['initial_intent']}")
        
    except Exception as e:
        logger.error(f"[INSTRUMENT] Initial intent classification failed: {e}")
        state["error"] = str(e)
    
    return state


def identify_instruments_node(state: InstrumentDetailState) -> InstrumentDetailState:
    """
    Agent 2: Instrument/Accessory Identifier.
    Identifies instruments and accessories from user input.
    """
    logger.info("[INSTRUMENT] Node 2: Identifying instruments and accessories...")
    
    try:
        # Identify instruments
        inst_result = identify_instruments_tool.invoke({
            "requirements": state["user_input"]
        })
        
        if inst_result.get("success"):
            state["identified_instruments"] = inst_result.get("instruments", [])
            state["project_name"] = inst_result.get("project_name", "Untitled Project")
        
        # Identify accessories
        acc_result = identify_accessories_tool.invoke({
            "instruments": state["identified_instruments"],
            "process_context": state["user_input"]
        })
        
        if acc_result.get("success"):
            state["identified_accessories"] = acc_result.get("accessories", [])
        
        # If no items found, create a default based on the input
        if not state["identified_instruments"] and not state["identified_accessories"]:
            # Attempt to extract at least one item
            state["identified_instruments"] = [{
                "category": "Industrial Instrument",
                "product_name": "General Instrument",
                "quantity": 1,
                "specifications": {}
            }]
        
        state["current_step"] = "present_selection"
        state["requires_user_input"] = True
        
        total_items = len(state["identified_instruments"]) + len(state["identified_accessories"])
        
        state["messages"] = state.get("messages", []) + [{
            "role": "system",
            "content": f"Identified {len(state['identified_instruments'])} instruments and {len(state['identified_accessories'])} accessories"
        }]
        
        logger.info(f"[INSTRUMENT] Found {total_items} items")
        
    except Exception as e:
        logger.error(f"[INSTRUMENT] Identification failed: {e}")
        state["error"] = str(e)
    
    return state


def present_selection_node(state: InstrumentDetailState) -> InstrumentDetailState:
    """
    Present identified items for user selection.
    """
    logger.info("[INSTRUMENT] Node 2b: Presenting items for selection...")
    
    try:
        llm = create_llm_with_fallback(
            model="gemini-2.0-flash-exp",
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        prompt = ChatPromptTemplate.from_template(INSTRUMENT_LIST_PROMPT)
        parser = JsonOutputParser()
        chain = prompt | llm | parser
        
        result = chain.invoke({
            "instruments": json.dumps(state["identified_instruments"], indent=2),
            "accessories": json.dumps(state["identified_accessories"], indent=2),
            "project_name": state.get("project_name", "Project")
        })
        
        # Format response for user
        response_lines = [f"**📋 {state.get('project_name', 'Project')} - Identified Items**\n"]
        
        formatted_list = result.get("formatted_list", [])
        for item in formatted_list:
            emoji = "🔧" if item.get("type") == "instrument" else "🔩"
            response_lines.append(
                f"{item['number']}. {emoji} **{item.get('name', 'Unknown')}** ({item.get('category', '')})"
            )
            response_lines.append(f"   Quantity: {item.get('quantity', 1)} | {item.get('key_specs', '')}")
            response_lines.append("")
        
        response_lines.append(result.get("message", "Please select an item number to view details."))
        
        state["response"] = "\n".join(response_lines)
        state["response_data"] = {
            "formatted_list": formatted_list,
            "total_items": result.get("total_items", 0),
            "awaiting_selection": True
        }
        
        state["current_step"] = "await_selection"
        
    except Exception as e:
        logger.error(f"[INSTRUMENT] Selection presentation failed: {e}")
        state["error"] = str(e)
    
    return state


def process_selection_node(state: InstrumentDetailState) -> InstrumentDetailState:
    """
    Process user's item selection.
    This would be called after user provides selection input.
    """
    logger.info("[INSTRUMENT] Node 3: Processing user selection...")
    
    try:
        # In a real implementation, this would parse the user's selection
        # For now, default to first item
        
        if state["identified_instruments"]:
            state["selected_item"] = state["identified_instruments"][0]
            state["selected_type"] = "instrument"
            state["product_type"] = state["selected_item"].get("category", "")
        elif state["identified_accessories"]:
            state["selected_item"] = state["identified_accessories"][0]
            state["selected_type"] = "accessory"
            state["product_type"] = state["selected_item"].get("category", "")
        
        state["requires_user_input"] = False
        state["current_step"] = "detail_intent"
        
        state["messages"] = state.get("messages", []) + [{
            "role": "system",
            "content": f"Selected: {state.get('product_type', 'Unknown')} ({state.get('selected_type', 'item')})"
        }]
        
        logger.info(f"[INSTRUMENT] Selected: {state.get('product_type')}")
        
    except Exception as e:
        logger.error(f"[INSTRUMENT] Selection processing failed: {e}")
        state["error"] = str(e)
    
    return state


def classify_detail_intent_node(state: InstrumentDetailState) -> InstrumentDetailState:
    """
    Agent 3: Intent Classifier - Detail Mode.
    Classifies intent for the selected item.
    """
    logger.info("[INSTRUMENT] Node 4: Detail intent classification...")
    
    try:
        selected_item = state.get("selected_item", {})
        context = f"User selected: {selected_item.get('product_name', 'item')} for detailed procurement"
        
        result = classify_intent_tool.invoke({
            "user_input": state["user_input"],
            "context": context
        })
        
        if result.get("success"):
            state["detail_intent"] = result.get("intent", "requirements")
        else:
            state["detail_intent"] = "requirements"
        
        state["current_step"] = "validate_requirements"
        
    except Exception as e:
        logger.error(f"[INSTRUMENT] Detail intent classification failed: {e}")
        state["error"] = str(e)
    
    return state


def validate_requirements_node(state: InstrumentDetailState) -> InstrumentDetailState:
    """
    Agent 4: Validation Agent.
    Validates requirements for selected item.
    """
    logger.info("[INSTRUMENT] Node 5: Validating requirements...")
    
    try:
        selected_item = state.get("selected_item", {})
        
        # Get specifications from selected item
        state["provided_requirements"] = selected_item.get("specifications", {})
        
        # Load schema for product type
        if state["product_type"]:
            schema_result = load_schema_tool.invoke({
                "product_type": state["product_type"]
            })
            
            if schema_result.get("success"):
                state["schema"] = schema_result.get("schema")
        
        # Validate
        if state["schema"]:
            validate_result = validate_requirements_tool.invoke({
                "user_input": json.dumps(selected_item),
                "product_type": state["product_type"],
                "schema": state["schema"]
            })
            
            if validate_result.get("success"):
                state["is_requirements_valid"] = validate_result.get("is_valid", False)
                state["missing_requirements"] = validate_result.get("missing_fields", [])
        
        state["current_step"] = "aggregate_data"
        
        state["messages"] = state.get("messages", []) + [{
            "role": "system",
            "content": f"Requirements valid: {state['is_requirements_valid']}"
        }]
        
    except Exception as e:
        logger.error(f"[INSTRUMENT] Validation failed: {e}")
        state["error"] = str(e)
    
    return state


def aggregate_data_node(state: InstrumentDetailState) -> InstrumentDetailState:
    """
    Agent 5: RAG Data Aggregator.
    Queries RAG systems for the selected item.
    """
    logger.info("[INSTRUMENT] Node 6: Aggregating RAG data...")
    
    try:
        rag_aggregator = create_rag_aggregator()
        
        rag_results = rag_aggregator.query_all_parallel(
            product_type=state["product_type"] or "industrial instrument",
            requirements=state["provided_requirements"]
        )
        
        state["rag_context"] = rag_results
        
        strategy_data = rag_results.get("strategy", {}).get("data", {})
        state["strategy_present"] = bool(
            strategy_data.get("preferred_vendors") or 
            strategy_data.get("forbidden_vendors")
        )
        
        if state["strategy_present"]:
            state["allowed_vendors"] = (
                strategy_data.get("preferred_vendors", []) + 
                strategy_data.get("neutral_vendors", [])
            )
        
        state["current_step"] = "lookup_schema"
        
    except Exception as e:
        logger.error(f"[INSTRUMENT] RAG aggregation failed: {e}")
        state["error"] = str(e)
        state["rag_context"] = {}
    
    return state


def lookup_schema_node(state: InstrumentDetailState) -> InstrumentDetailState:
    """
    Agent 6: Schema Agent.
    Looks up schema for selected item type.
    """
    logger.info("[INSTRUMENT] Node 7: Schema lookup...")
    
    try:
        if not state.get("schema"):
            schema_result = load_schema_tool.invoke({
                "product_type": state["product_type"] or "industrial instrument"
            })
            
            if schema_result.get("success") and schema_result.get("schema"):
                state["schema"] = schema_result["schema"]
        
        state["current_step"] = "apply_strategy"
        
    except Exception as e:
        logger.error(f"[INSTRUMENT] Schema lookup failed: {e}")
        state["error"] = str(e)
    
    return state


def apply_strategy_node(state: InstrumentDetailState) -> InstrumentDetailState:
    """
    Agent 7: Strategy Agent.
    Applies strategy filtering for selected item.
    """
    logger.info("[INSTRUMENT] Node 8: Applying strategy...")
    
    try:
        # Search for vendors
        vendor_result = search_vendors_tool.invoke({
            "product_type": state["product_type"] or "industrial instrument",
            "requirements": state["provided_requirements"]
        })
        
        available_vendors = vendor_result.get("vendors", [])
        if not available_vendors:
            available_vendors = ["Honeywell", "Emerson", "Yokogawa", "ABB", "Siemens"]
        
        state["available_vendors"] = available_vendors
        
        if state["strategy_present"]:
            strategy_filter = create_strategy_filter()
            strategy_data = state["rag_context"].get("strategy", {}).get("data", {})
            
            constraint_context = {
                "preferred_vendors": strategy_data.get("preferred_vendors", []),
                "forbidden_vendors": strategy_data.get("forbidden_vendors", []),
                "neutral_vendors": strategy_data.get("neutral_vendors", [])
            }
            
            filter_result = strategy_filter.apply_strategy_rules(
                vendors=available_vendors,
                constraint_context=constraint_context
            )
            
            state["filtered_vendors"] = [v["vendor"] for v in filter_result["filtered_vendors"]]
        else:
            state["filtered_vendors"] = available_vendors
        
        state["current_step"] = "parallel_analysis"
        
    except Exception as e:
        logger.error(f"[INSTRUMENT] Strategy application failed: {e}")
        state["error"] = str(e)
    
    return state


def parallel_analysis_node(state: InstrumentDetailState) -> InstrumentDetailState:
    """
    Agent 8: Analysis Coordinator.
    Parallel vendor analysis for selected item.
    """
    logger.info("[INSTRUMENT] Node 9: Parallel vendor analysis...")
    
    try:
        vendors = state.get("filtered_vendors", [])[:5]
        requirements = state["provided_requirements"]
        analysis_results = []
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {}
            
            for vendor in vendors:
                futures[executor.submit(
                    analyze_vendor_match_tool.invoke,
                    {
                        "vendor": vendor,
                        "requirements": requirements,
                        "pdf_content": None,
                        "product_data": None
                    }
                )] = vendor
            
            for future in as_completed(futures):
                vendor = futures[future]
                try:
                    result = future.result()
                    if result.get("success"):
                        analysis_results.append({
                            "vendor": vendor,
                            "product_name": result.get("product_name", ""),
                            "model_family": result.get("model_family", ""),
                            "match_score": result.get("match_score", 0),
                            "requirements_match": result.get("requirements_match", False),
                            "reasoning": result.get("reasoning", "")
                        })
                except Exception as e:
                    logger.error(f"Analysis failed for {vendor}: {e}")
        
        state["parallel_analysis_results"] = analysis_results
        state["current_step"] = "judge"
        
    except Exception as e:
        logger.error(f"[INSTRUMENT] Parallel analysis failed: {e}")
        state["error"] = str(e)
    
    return state


def judge_results_node(state: InstrumentDetailState) -> InstrumentDetailState:
    """
    Agent 9: Judge Agent.
    Validates analysis results.
    """
    logger.info("[INSTRUMENT] Node 10: Judging results...")
    
    try:
        analysis_results = state.get("parallel_analysis_results", [])
        
        # Skip judging if no analysis results
        if not analysis_results:
            logger.warning("[INSTRUMENT] No analysis results to judge, skipping...")
            state["judge_validation"] = {"passed": [], "failed": [], "skipped": True}
            state["current_step"] = "rank"
            return state
        
        # Convert list to dict format expected by JudgeAnalysisInput
        # The tool expects vendor_analysis as Dict[str, Any], not List
        vendor_analysis_dict = {
            "vendor_matches": analysis_results,
            "total_analyzed": len(analysis_results),
            "analysis_type": "parallel_vendor_analysis"
        }
        
        judge_result = judge_analysis_tool.invoke({
            "original_requirements": state["provided_requirements"],
            "vendor_analysis": vendor_analysis_dict,
            "strategy_rules": None
        })
        
        if judge_result.get("success"):
            state["judge_validation"] = {
                "passed": judge_result.get("valid_matches", []),
                "failed": judge_result.get("invalid_matches", [])
            }
        
        state["current_step"] = "rank"
        
    except Exception as e:
        logger.error(f"[INSTRUMENT] Judging failed: {e}")
        state["error"] = str(e)
        state["judge_validation"] = {}
    
    return state


def rank_products_node(state: InstrumentDetailState) -> InstrumentDetailState:
    """
    Agent 10: Ranking Agent.
    Ranks products for selected item.
    """
    logger.info("[INSTRUMENT] Node 11: Ranking products...")
    
    try:
        analysis_results = state.get("parallel_analysis_results", [])
        selected_item = state.get("selected_item", {})
        
        rank_result = rank_products_tool.invoke({
            "vendor_matches": analysis_results,
            "requirements": state["provided_requirements"]
        })
        
        if rank_result.get("success"):
            state["ranked_results"] = rank_result.get("ranked_products", [])
            
            # Generate response
            item_name = selected_item.get("product_name", state.get("product_type", "Item"))
            response_lines = [f"**🎯 Recommendations for: {item_name}**\n"]
            
            for i, product in enumerate(state["ranked_results"][:5], 1):
                response_lines.append(f"**#{i} {product.get('vendor', '')} - {product.get('product_name', '')}**")
                response_lines.append(f"   Match Score: {product.get('overall_score', 0)}/100")
                response_lines.append(f"   Model Family: {product.get('model_family', 'N/A')}")
                
                if product.get("recommendation"):
                    response_lines.append(f"   💡 {product['recommendation']}")
                
                response_lines.append("")
            
            state["response"] = "\n".join(response_lines)
            state["response_data"] = {
                "selected_item": selected_item,
                "ranked_products": state["ranked_results"],
                "total_analyzed": len(analysis_results)
            }
        
        state["current_step"] = "complete"
        
    except Exception as e:
        logger.error(f"[INSTRUMENT] Ranking failed: {e}")
        state["error"] = str(e)
    
    return state


# ============================================================================
# ROUTING FUNCTIONS
# ============================================================================

def route_after_identification(state: InstrumentDetailState) -> Literal["present", "end"]:
    """Route after item identification."""
    if state.get("error"):
        return "end"
    
    total_items = len(state.get("identified_instruments", [])) + len(state.get("identified_accessories", []))
    
    if total_items > 0:
        return "present"
    return "end"


def route_after_selection(state: InstrumentDetailState) -> Literal["continue", "await"]:
    """Route based on whether selection is complete."""
    if state.get("selected_item"):
        return "continue"
    return "await"


# ============================================================================
# WORKFLOW CREATION
# ============================================================================

def create_instrument_detail_workflow() -> StateGraph:
    """
    Create the Instrument/Accessory Detail Capture workflow.
    
    Flow:
    1. Initial Intent Classification
    2. Instrument/Accessory Identification
    3. Present Selection (requires user input)
    4. Process Selection
    5. Detail Intent Classification
    6. Validation
    7. RAG Aggregation
    8. Schema Lookup
    9. Strategy Application
    10. Parallel Analysis
    11. Judge Validation
    12. Ranking
    """
    
    workflow = StateGraph(InstrumentDetailState)
    
    # Add nodes
    workflow.add_node("classify_initial_intent", classify_initial_intent_node)
    workflow.add_node("identify_instruments", identify_instruments_node)
    workflow.add_node("present_selection", present_selection_node)
    workflow.add_node("process_selection", process_selection_node)
    workflow.add_node("classify_detail_intent", classify_detail_intent_node)
    workflow.add_node("validate_requirements", validate_requirements_node)
    workflow.add_node("aggregate_data", aggregate_data_node)
    workflow.add_node("lookup_schema", lookup_schema_node)
    workflow.add_node("apply_strategy", apply_strategy_node)
    workflow.add_node("parallel_analysis", parallel_analysis_node)
    workflow.add_node("judge_results", judge_results_node)
    workflow.add_node("rank_products", rank_products_node)
    
    # Set entry point
    workflow.set_entry_point("classify_initial_intent")
    
    # Add edges
    workflow.add_edge("classify_initial_intent", "identify_instruments")
    workflow.add_edge("identify_instruments", "present_selection")
    workflow.add_edge("present_selection", "process_selection")
    workflow.add_edge("process_selection", "classify_detail_intent")
    workflow.add_edge("classify_detail_intent", "validate_requirements")
    workflow.add_edge("validate_requirements", "aggregate_data")
    workflow.add_edge("aggregate_data", "lookup_schema")
    workflow.add_edge("lookup_schema", "apply_strategy")
    workflow.add_edge("apply_strategy", "parallel_analysis")
    workflow.add_edge("parallel_analysis", "judge_results")
    workflow.add_edge("judge_results", "rank_products")
    workflow.add_edge("rank_products", END)
    
    return workflow


# ============================================================================
# WORKFLOW EXECUTION
# ============================================================================

def build_spec_object_from_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a SpecObject from workflow state for comparison workflow integration.
    
    Args:
        state: Final workflow state
    
    Returns:
        SpecObject dict ready for comparison workflow
    """
    selected_item = state.get("selected_item", {})
    provided_requirements = state.get("provided_requirements", {})
    rag_context = state.get("rag_context", {})
    
    # Extract certifications from RAG
    standards_data = rag_context.get("standards", {}).get("data", {})
    required_certs = standards_data.get("required_certifications", [])
    
    # Build specifications dict
    specifications = {**provided_requirements}
    if selected_item.get("specifications"):
        specifications.update(selected_item["specifications"])
    
    return {
        "product_type": state.get("product_type", selected_item.get("category", "industrial instrument")),
        "category": selected_item.get("category", ""),
        "subcategory": selected_item.get("subcategory"),
        "specifications": specifications,
        "required_certifications": required_certs,
        "sil_rating": standards_data.get("required_sil_rating"),
        "atex_zone": standards_data.get("atex_zone"),
        "environment": selected_item.get("environment"),
        "source_workflow": "instrument_detail",
        "session_id": state.get("session_id")
    }


def run_instrument_detail_workflow(
    user_input: str,
    session_id: str = "default",
    checkpointing_backend: str = "memory"
) -> Dict[str, Any]:
    """
    Run the instrument detail workflow.
    
    Args:
        user_input: User's requirements with instruments/accessories
        session_id: Session identifier
        checkpointing_backend: Backend for state persistence
    
    Returns:
        Workflow result with identified items, rankings, and spec_object for comparison
    """
    try:
        # Create initial state
        initial_state = create_instrument_detail_state(user_input, session_id)
        
        # Create and compile workflow
        workflow = create_instrument_detail_workflow()
        compiled = compile_with_checkpointing(workflow, checkpointing_backend)
        
        # Run workflow
        config = {"configurable": {"thread_id": session_id}}
        final_state = compiled.invoke(initial_state, config)
        
        # Build spec_object for comparison workflow integration
        spec_object = build_spec_object_from_state(final_state)
        
        return {
            "success": True,
            "response": final_state.get("response"),
            "response_data": final_state.get("response_data"),
            "identified_instruments": final_state.get("identified_instruments", []),
            "identified_accessories": final_state.get("identified_accessories", []),
            "selected_item": final_state.get("selected_item"),
            "ranked_results": final_state.get("ranked_results", []),
            "requires_user_input": final_state.get("requires_user_input", False),
            # NEW: SpecObject for comparison workflow
            "spec_object": spec_object,
            "compare_vendors_available": True,
            "error": final_state.get("error")
        }
        
    except Exception as e:
        logger.error(f"Instrument detail workflow failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def run_instrument_detail_with_comparison(
    user_input: str,
    session_id: str = "default",
    auto_compare: bool = False,
    comparison_type: str = "full",
    checkpointing_backend: str = "memory"
) -> Dict[str, Any]:
    """
    Run instrument detail workflow with optional automatic comparison.
    
    This function chains the Instrument Detail Capture workflow with the
    Comparison workflow, simulating the complete flow:
    
    Instrument Detail → [COMPARE VENDORS] → Comparison Workflow
    
    Args:
        user_input: User's requirements with instruments/accessories
        session_id: Session identifier
        auto_compare: If True, automatically run comparison after detail capture
        comparison_type: Type of comparison ("vendor", "series", "model", "full")
        checkpointing_backend: Backend for state persistence
    
    Returns:
        Combined results from both workflows
    """
    logger.info("=" * 60)
    logger.info("INSTRUMENT DETAIL WITH COMPARISON - CHAINED WORKFLOW")
    logger.info("=" * 60)
    
    try:
        # Step 1: Run instrument detail workflow via internal API
        logger.info("[STEP 1] Running Instrument Detail Capture workflow...")
        from .internal_api import api_client
        
        detail_result = api_client.call_instrument_detail(
            message=user_input,
            session_id=session_id
        )
        
        if not detail_result.get("success"):
            return {
                "success": False,
                "phase": "detail_capture",
                "error": detail_result.get("error")
            }
        
        logger.info("[STEP 1 COMPLETE] Detail capture finished")
        
        # If auto_compare is enabled, run comparison workflow
        if auto_compare and detail_result.get("spec_object"):
            logger.info("[STEP 2] Running Comparison workflow (auto_compare=True)...")

            # Use internal API client for workflow chaining
            from .internal_api import api_client

            comparison_result = api_client.call_comparison_from_spec(
                spec_object=detail_result["spec_object"],
                comparison_type=comparison_type,
                session_id=session_id,
                checkpointing_backend=checkpointing_backend
            )
            
            if comparison_result.get("success"):
                logger.info("[STEP 2 COMPLETE] Comparison finished")
                
                return {
                    "success": True,
                    "phase": "comparison_complete",
                    # Detail results
                    "detail_capture": {
                        "response": detail_result.get("response"),
                        "identified_instruments": detail_result.get("identified_instruments", []),
                        "identified_accessories": detail_result.get("identified_accessories", []),
                        "selected_item": detail_result.get("selected_item"),
                        "ranked_results": detail_result.get("ranked_results", []),
                        "spec_object": detail_result.get("spec_object")
                    },
                    # Comparison results
                    "comparison": {
                        "vendor_ranking": comparison_result.get("vendor_ranking", []),
                        "series_comparisons": comparison_result.get("series_comparisons", {}),
                        "top_recommendation": comparison_result.get("top_recommendation"),
                        "formatted_output": comparison_result.get("formatted_output"),
                        "total_analyzed": comparison_result.get("total_analyzed", 0)
                    },
                    # Combined output
                    "top_vendor": comparison_result.get("top_recommendation", {}).get("vendor"),
                    "top_product": comparison_result.get("top_recommendation", {}).get("model"),
                    "session_id": session_id
                }
            else:
                return {
                    "success": False,
                    "phase": "comparison",
                    "detail_capture": detail_result,
                    "error": comparison_result.get("error")
                }
        
        # Return detail result only (for manual comparison trigger)
        return {
            "success": True,
            "phase": "detail_capture_complete",
            "detail_capture": detail_result,
            "spec_object": detail_result.get("spec_object"),
            "compare_vendors_available": True,
            "message": "Click [COMPARE VENDORS] to run comparison workflow",
            "session_id": session_id
        }
        
    except Exception as e:
        logger.error(f"Chained workflow failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "create_instrument_detail_workflow",
    "run_instrument_detail_workflow",
    "run_instrument_detail_with_comparison",
    "build_spec_object_from_state"
]
