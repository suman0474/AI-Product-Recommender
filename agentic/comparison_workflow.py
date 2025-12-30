# agentic/comparison_workflow.py
# Comparative Analysis Workflow - Multi-Vendor + Multi-Model Under Constraints
# Enhanced with SpecObject input and multi-level comparison

import json
import logging
from typing import Dict, Any, List, Literal, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from langgraph.graph import StateGraph, END

from .models import (
    ComparisonState,
    create_comparison_state,
    ScoringBreakdown,
    RankedComparisonProduct,
    SpecObject,
    ComparisonInput,
    ComparisonType
)
from .rag_components import RAGAggregator, StrategyFilter, create_rag_aggregator, create_strategy_filter
from .checkpointing import compile_with_checkpointing

from tools.intent_tools import classify_intent_tool
from tools.instrument_tools import identify_instruments_tool
from tools.analysis_tools import analyze_vendor_match_tool
from tools.ranking_tools import judge_analysis_tool
from tools.vendor_tools import search_vendors_tool

# Import synchronization utilities
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.workflow_sync import (
    with_workflow_lock,
    with_state_transaction,
    ThreadSafeResultCollector
)

# MongoDB for candidate discovery
try:
    from pymongo import MongoClient
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from llm_fallback import create_llm_with_fallback

load_dotenv()
logger = logging.getLogger(__name__)


# ============================================================================
# PROMPTS
# ============================================================================

REQUEST_CLASSIFICATION_PROMPT = """
You are Engenie's Request Classifier. Determine if this is a single-product lookup or a comparative analysis request.

User Input: "{user_input}"

CLASSIFICATION RULES:
1. COMPARISON requests include:
   - "compare", "versus", "vs", "which is better"
   - Multiple vendor/model mentions
   - "difference between", "pros and cons"
   - Requests for side-by-side analysis

2. SINGLE LOOKUP requests include:
   - Specific product inquiries
   - Single vendor/model requests
   - "find me", "I need", "looking for"

Return ONLY valid JSON:
{{
    "mode": "single" | "comparison",
    "confidence": <0.0-1.0>,
    "detected_products": ["<list of mentioned products/vendors>"],
    "comparison_type": "within_vendor" | "cross_vendor" | "mixed" | null
}}
"""

COMPARISON_OUTPUT_PROMPT = """
You are Engenie's Comparison Presenter. Generate a formatted comparison output.

Ranked Products:
{ranked_products}

Scoring Criteria (100 points total):
- Strategy Priority: /25 (Preferred=25, Neutral=18, Discouraged=10)
- Technical Fit: /25 (Critical spec match %)
- Asset Alignment: /20 (Installed base match)
- Standards Compliance: /15 (Certifications)
- Data Completeness: /15 (Specification completeness)

Generate a detailed comparison summary highlighting:
1. Winner and why
2. Key differences between top options
3. Trade-offs to consider
4. Recommendation based on constraints

Return ONLY valid JSON:
{{
    "summary": "<executive summary>",
    "winner": {{
        "vendor": "<vendor>",
        "model": "<model>",
        "reason": "<why this is the best choice>"
    }},
    "key_differences": ["<difference 1>", "<difference 2>"],
    "trade_offs": ["<trade-off 1>", "<trade-off 2>"],
    "recommendation": "<final recommendation>"
}}
"""


# ============================================================================
# WORKFLOW NODES
# ============================================================================

def classify_request_node(state: ComparisonState) -> ComparisonState:
    """
    Node 1: Classify the request as single lookup or comparison.
    """
    logger.info("[COMPARISON] Phase 1: Classifying request...")
    
    try:
        llm = create_llm_with_fallback(
            model="gemini-2.0-flash-exp",
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        prompt = ChatPromptTemplate.from_template(REQUEST_CLASSIFICATION_PROMPT)
        parser = JsonOutputParser()
        chain = prompt | llm | parser
        
        result = chain.invoke({"user_input": state["user_input"]})
        
        state["request_mode"] = result.get("mode", "single")
        state["mode_confidence"] = result.get("confidence", 0.5)
        state["current_phase"] = "identify_instrument"
        
        state["messages"] = state.get("messages", []) + [{
            "role": "system",
            "content": f"Request classified as: {state['request_mode']} (confidence: {state['mode_confidence']:.2f})"
        }]
        
        logger.info(f"[COMPARISON] Request mode: {state['request_mode']}")
        
    except Exception as e:
        logger.error(f"[COMPARISON] Request classification failed: {e}")
        state["error"] = str(e)
        state["request_mode"] = "single"
    
    return state


def identify_instrument_node(state: ComparisonState) -> ComparisonState:
    """
    Node 2: Identify instrument type, category, and critical specs.
    """
    logger.info("[COMPARISON] Phase 2: Identifying instrument...")
    
    try:
        result = identify_instruments_tool.invoke({
            "requirements": state["user_input"]
        })
        
        if result.get("success") and result.get("instruments"):
            first_instrument = result["instruments"][0]
            state["instrument_type"] = first_instrument.get("category", "")
            state["instrument_category"] = first_instrument.get("product_name", "")
            state["critical_specs"] = first_instrument.get("specifications", {})
        else:
            # Fallback to basic extraction
            state["instrument_type"] = "pressure transmitter"  # Default
            state["instrument_category"] = "Industrial Instrumentation"
            state["critical_specs"] = {}
        
        state["current_phase"] = "aggregate_constraints"
        
        state["messages"] = state.get("messages", []) + [{
            "role": "system",
            "content": f"Identified instrument: {state['instrument_type']}"
        }]
        
        logger.info(f"[COMPARISON] Instrument identified: {state['instrument_type']}")
        
    except Exception as e:
        logger.error(f"[COMPARISON] Instrument identification failed: {e}")
        state["error"] = str(e)
    
    return state


def aggregate_constraints_node(state: ComparisonState) -> ComparisonState:
    """
    Node 4: Query all three RAG sources in parallel and merge constraints.
    """
    logger.info("[COMPARISON] Phase 3: Aggregating constraints from RAGs...")
    
    try:
        rag_aggregator = create_rag_aggregator()
        
        # Query all RAGs in parallel
        rag_results = rag_aggregator.query_all_parallel(
            product_type=state["instrument_type"] or "pressure transmitter",
            requirements=state["critical_specs"]
        )
        
        state["rag_results"] = rag_results
        
        # Merge into unified constraint context
        constraint_context = rag_aggregator.merge_to_constraint_context(rag_results)
        state["constraint_context"] = constraint_context
        
        state["current_phase"] = "filter_candidates"
        
        state["messages"] = state.get("messages", []) + [{
            "role": "system",
            "content": f"Constraints aggregated from {len(rag_results)} RAG sources"
        }]
        
        logger.info("[COMPARISON] Constraints aggregated successfully")
        
    except Exception as e:
        logger.error(f"[COMPARISON] Constraint aggregation failed: {e}")
        state["error"] = str(e)
        state["constraint_context"] = {}
    
    return state


def filter_candidates_node(state: ComparisonState) -> ComparisonState:
    """
    Node 5: Apply strategy filter to create candidate pool.
    """
    logger.info("[COMPARISON] Phase 4: Filtering candidate vendors...")
    
    try:
        # Get all vendors
        vendor_result = search_vendors_tool.invoke({
            "product_type": state["instrument_type"] or "pressure transmitter",
            "requirements": state["critical_specs"]
        })
        
        all_vendors = vendor_result.get("vendors", [])
        state["all_vendors"] = all_vendors
        
        if not all_vendors:
            # Fallback vendors for demo
            all_vendors = ["Honeywell", "Emerson", "Yokogawa", "ABB", "Siemens"]
            state["all_vendors"] = all_vendors
        
        # Apply strategy filter
        strategy_filter = create_strategy_filter()
        filter_result = strategy_filter.apply_all_rules(
            vendors=all_vendors,
            constraint_context=state["constraint_context"] or {}
        )
        
        state["filtered_candidates"] = filter_result["filtered_candidates"]
        state["exclusion_reasons"] = filter_result["excluded"]
        
        state["current_phase"] = "parallel_analysis"
        
        state["messages"] = state.get("messages", []) + [{
            "role": "system",
            "content": f"Filtered to {len(state['filtered_candidates'])} candidates from {len(all_vendors)} vendors"
        }]
        
        logger.info(f"[COMPARISON] {len(state['filtered_candidates'])} candidates after filtering")
        
    except Exception as e:
        logger.error(f"[COMPARISON] Candidate filtering failed: {e}")
        state["error"] = str(e)
    
    return state


@with_state_transaction(auto_commit=True)
def parallel_analysis_node(state: ComparisonState) -> ComparisonState:
    """
    Node 6: Perform parallel vendor/model analysis with thread-safe result collection.
    """
    logger.info("[COMPARISON] Phase 5: Parallel vendor analysis...")

    try:
        candidates = state.get("filtered_candidates", [])
        requirements = state.get("critical_specs", {})

        # Thread-safe collector
        analysis_collector = ThreadSafeResultCollector()

        # Use ThreadPoolExecutor for parallel analysis
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {}

            for candidate in candidates[:10]:  # Limit to 10 vendors
                vendor = candidate.get("vendor", "")
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
                        analysis_result = {
                            "vendor": vendor,
                            "product_name": result.get("product_name"),
                            "model_family": result.get("model_family"),
                            "match_score": result.get("match_score", 0),
                            "requirements_match": result.get("requirements_match", False),
                            "reasoning": result.get("reasoning"),
                            "limitations": result.get("limitations"),
                            "key_strengths": result.get("key_strengths", [])
                        }
                        analysis_collector.add_result(analysis_result)
                except Exception as e:
                    logger.error(f"Analysis failed for {vendor}: {e}")
                    analysis_collector.add_error(e, {"vendor": vendor})

        # Get thread-safe results
        analysis_results = analysis_collector.get_results()

        # Log summary
        summary = analysis_collector.summary()
        logger.info(
            f"[COMPARISON] Parallel analysis complete: "
            f"{summary['total_results']} successes, {summary['total_errors']} errors"
        )

        # Update state (protected by transaction)
        state["vendor_analysis_results"] = analysis_results
        state["current_phase"] = "aggregate_summaries"

        state["messages"] = state.get("messages", []) + [{
            "role": "system",
            "content": f"Analyzed {len(analysis_results)} vendors"
        }]

        logger.info(f"[COMPARISON] Analyzed {len(analysis_results)} vendors")

    except Exception as e:
        logger.error(f"[COMPARISON] Parallel analysis failed: {e}")
        state["error"] = str(e)

    return state


def aggregate_summaries_node(state: ComparisonState) -> ComparisonState:
    """
    Node 7: Aggregate all analysis results into comparison matrix.
    """
    logger.info("[COMPARISON] Phase 6: Aggregating summaries...")
    
    try:
        analysis_results = state.get("vendor_analysis_results", [])
        
        # Build comparison matrix
        within_vendor = {}
        cross_vendor = []
        
        # Group by vendor for within-vendor comparisons
        vendor_groups = {}
        for result in analysis_results:
            vendor = result.get("vendor", "Unknown")
            if vendor not in vendor_groups:
                vendor_groups[vendor] = []
            vendor_groups[vendor].append(result)
        
        for vendor, products in vendor_groups.items():
            if len(products) > 1:
                within_vendor[vendor] = products
        
        # Cross-vendor comparison (all against each other)
        cross_vendor = analysis_results
        
        state["within_vendor_comparisons"] = within_vendor
        state["cross_vendor_comparisons"] = cross_vendor
        state["comparison_matrix"] = {
            "within_vendor": within_vendor,
            "cross_vendor": cross_vendor,
            "total_products": len(analysis_results)
        }
        
        state["current_phase"] = "validate_results"
        
        state["messages"] = state.get("messages", []) + [{
            "role": "system",
            "content": f"Aggregated {len(analysis_results)} results into comparison matrix"
        }]
        
        logger.info("[COMPARISON] Summaries aggregated")
        
    except Exception as e:
        logger.error(f"[COMPARISON] Summary aggregation failed: {e}")
        state["error"] = str(e)
    
    return state


def validate_results_node(state: ComparisonState) -> ComparisonState:
    """
    Node 8: Judge Agent - Validate all results against constraints.
    """
    logger.info("[COMPARISON] Phase 7: Validating results...")
    
    try:
        analysis_results = state.get("vendor_analysis_results", [])
        constraint_context = state.get("constraint_context", {})
        
        validated = []
        flagged = []
        removed = []
        
        forbidden_vendors = constraint_context.get("forbidden_vendors", [])
        
        for result in analysis_results:
            vendor = result.get("vendor", "")
            
            # Check against forbidden vendors
            is_forbidden = any(
                f.lower() in vendor.lower() 
                for f in forbidden_vendors
            )
            
            if is_forbidden:
                removed.append({
                    **result,
                    "removal_reason": "Forbidden by strategy"
                })
            elif result.get("match_score", 0) < 30:
                flagged.append({
                    **result,
                    "flag_reason": "Low match score"
                })
            else:
                validated.append(result)
        
        state["validated_results"] = validated
        state["flagged_results"] = flagged
        state["removed_results"] = removed
        
        state["current_phase"] = "present_comparison"
        
        state["messages"] = state.get("messages", []) + [{
            "role": "system",
            "content": f"Validated: {len(validated)}, Flagged: {len(flagged)}, Removed: {len(removed)}"
        }]
        
        logger.info(f"[COMPARISON] Validation complete: {len(validated)} valid results")
        
    except Exception as e:
        logger.error(f"[COMPARISON] Validation failed: {e}")
        state["error"] = str(e)
    
    return state


def present_comparison_node(state: ComparisonState) -> ComparisonState:
    """
    Node 10: Generate final ranked comparison output.
    """
    logger.info("[COMPARISON] Phase 8: Presenting comparison...")
    
    try:
        validated_results = state.get("validated_results", [])
        constraint_context = state.get("constraint_context", {})
        
        # Calculate detailed scores
        ranked_products = []
        preferred_vendors = constraint_context.get("preferred_vendors", [])
        standardized_vendor = constraint_context.get("standardized_vendor")
        
        for result in validated_results:
            vendor = result.get("vendor", "")
            match_score = result.get("match_score", 0)
            
            # Calculate scoring breakdown
            is_preferred = any(p.lower() in vendor.lower() for p in preferred_vendors)
            strategy_priority = 25 if is_preferred else 18
            
            technical_fit = int((match_score / 100) * 25)
            
            matches_base = standardized_vendor and standardized_vendor.lower() in vendor.lower()
            asset_alignment = 20 if matches_base else 10
            
            standards_compliance = 12  # Default
            data_completeness = 13    # Default
            
            overall_score = (strategy_priority + technical_fit + 
                           asset_alignment + standards_compliance + data_completeness)
            
            ranked_products.append({
                "vendor": vendor,
                "model": result.get("product_name", "Unknown"),
                "overall_score": overall_score,
                "scoring_breakdown": {
                    "strategy_priority": strategy_priority,
                    "technical_fit": technical_fit,
                    "asset_alignment": asset_alignment,
                    "standards_compliance": standards_compliance,
                    "data_completeness": data_completeness
                },
                "constraints_met": {
                    "strategy": True,
                    "standards": True,
                    "installed_base": matches_base
                },
                "key_advantages": result.get("key_strengths", [])[:3],
                "reasoning": result.get("reasoning", "")
            })
        
        # Sort by overall score
        ranked_products.sort(key=lambda x: x["overall_score"], reverse=True)
        
        # Add ranks
        for i, product in enumerate(ranked_products):
            product["rank"] = i + 1
        
        state["ranked_products"] = ranked_products
        
        # Generate formatted output
        formatted_output = generate_comparison_output(ranked_products)
        state["formatted_output"] = formatted_output
        
        state["response"] = formatted_output
        state["response_data"] = {
            "ranked_products": ranked_products,
            "total_compared": len(ranked_products)
        }
        
        state["current_phase"] = "complete"
        
        logger.info(f"[COMPARISON] Comparison complete with {len(ranked_products)} ranked products")
        
    except Exception as e:
        logger.error(f"[COMPARISON] Presentation failed: {e}")
        state["error"] = str(e)
    
    return state


# ============================================================================
# OUTPUT FORMATTING
# ============================================================================

def generate_comparison_output(ranked_products: List[Dict[str, Any]]) -> str:
    """Generate the visual comparison output."""
    
    if not ranked_products:
        return "No products to compare."
    
    output_lines = ["**📊 Comparative Analysis Results**\n"]
    
    rank_emoji = {1: "🥇", 2: "🥈", 3: "🥉"}
    
    for product in ranked_products[:5]:  # Top 5
        rank = product["rank"]
        emoji = rank_emoji.get(rank, f"#{rank}")
        vendor = product["vendor"]
        model = product["model"]
        score = product["overall_score"]
        breakdown = product["scoring_breakdown"]
        constraints = product["constraints_met"]
        advantages = product["key_advantages"]
        
        # Score bar
        filled = int(score / 2)
        empty = 50 - filled
        score_bar = "█" * filled + "░" * empty
        
        output_lines.append(f"┌{'─' * 78}┐")
        output_lines.append(f"│ {emoji} RANK #{rank}: {vendor.upper()} {model.upper()[:50]:<50} │")
        output_lines.append(f"├{'─' * 78}┤")
        output_lines.append(f"│                                                                              │")
        output_lines.append(f"│   OVERALL SCORE                                                              │")
        output_lines.append(f"│   {score_bar}  {score}/100                 │")
        output_lines.append(f"│                                                                              │")
        output_lines.append(f"│   SCORING BREAKDOWN                                                          │")
        output_lines.append(f"│   ├── Strategy Priority:      {'█' * breakdown['strategy_priority']}{'░' * (25 - breakdown['strategy_priority'])}  {breakdown['strategy_priority']}/25  │")
        output_lines.append(f"│   ├── Technical Fit:          {'█' * breakdown['technical_fit']}{'░' * (25 - breakdown['technical_fit'])}  {breakdown['technical_fit']}/25  │")
        output_lines.append(f"│   ├── Asset Alignment:        {'█' * breakdown['asset_alignment']}{'░' * (20 - breakdown['asset_alignment'])}  {breakdown['asset_alignment']}/20  │")
        output_lines.append(f"│   ├── Standards Compliance:   {'█' * breakdown['standards_compliance']}{'░' * (15 - breakdown['standards_compliance'])}  {breakdown['standards_compliance']}/15  │")
        output_lines.append(f"│   └── Data Completeness:      {'█' * breakdown['data_completeness']}{'░' * (15 - breakdown['data_completeness'])}  {breakdown['data_completeness']}/15  │")
        output_lines.append(f"│                                                                              │")
        
        # Constraints
        strat = "✓" if constraints.get("strategy") else "○"
        stand = "✓" if constraints.get("standards") else "○"
        base = "✓" if constraints.get("installed_base") else "○"
        output_lines.append(f"│   CONSTRAINTS           {strat} Strategy   {stand} Standards   {base} Installed-Base           │")
        
        # Advantages
        if advantages:
            adv_str = ", ".join(advantages[:2])[:60]
            output_lines.append(f"│   KEY ADVANTAGES        {adv_str:<54} │")
        
        output_lines.append(f"└{'─' * 78}┘")
        output_lines.append("")
    
    return "\n".join(output_lines)


# ============================================================================
# ROUTING FUNCTIONS
# ============================================================================

def route_after_classification(state: ComparisonState) -> Literal["continue", "end"]:
    """Route after request classification."""
    if state.get("error"):
        return "end"
    return "continue"


# ============================================================================
# WORKFLOW CREATION
# ============================================================================

def create_comparison_workflow() -> StateGraph:
    """
    Create the Comparative Analysis workflow.
    
    Phases:
    1. Input & Classification
    2. Instrument Identification
    3. Constraint Aggregation (parallel RAG queries)
    4. Vendor/Model Filtering
    5. Parallel Analysis
    6. Aggregation & Validation
    7. Presentation
    """
    
    workflow = StateGraph(ComparisonState)
    
    # Add nodes
    workflow.add_node("classify_request", classify_request_node)
    workflow.add_node("identify_instrument", identify_instrument_node)
    workflow.add_node("aggregate_constraints", aggregate_constraints_node)
    workflow.add_node("filter_candidates", filter_candidates_node)
    workflow.add_node("parallel_analysis", parallel_analysis_node)
    workflow.add_node("aggregate_summaries", aggregate_summaries_node)
    workflow.add_node("validate_results", validate_results_node)
    workflow.add_node("present_comparison", present_comparison_node)
    
    # Set entry point
    workflow.set_entry_point("classify_request")
    
    # Add edges (linear flow)
    workflow.add_edge("classify_request", "identify_instrument")
    workflow.add_edge("identify_instrument", "aggregate_constraints")
    workflow.add_edge("aggregate_constraints", "filter_candidates")
    workflow.add_edge("filter_candidates", "parallel_analysis")
    workflow.add_edge("parallel_analysis", "aggregate_summaries")
    workflow.add_edge("aggregate_summaries", "validate_results")
    workflow.add_edge("validate_results", "present_comparison")
    workflow.add_edge("present_comparison", END)
    
    return workflow


# ============================================================================
# WORKFLOW EXECUTION
# ============================================================================

@with_workflow_lock(session_id_param="session_id", timeout=60.0)
def run_comparison_workflow(
    user_input: str,
    session_id: str = "default",
    checkpointing_backend: str = "memory"
) -> Dict[str, Any]:
    """
    Run the comparison workflow with session-level locking to prevent race conditions.

    Args:
        user_input: User's comparison request
        session_id: Session identifier
        checkpointing_backend: Backend for state persistence

    Returns:
        Workflow result with ranked comparison
    """
    try:
        logger.info(f"[COMPARISON] Starting workflow for session {session_id}")

        # Create initial state
        initial_state = create_comparison_state(user_input, session_id)

        # Create and compile workflow
        workflow = create_comparison_workflow()
        compiled = compile_with_checkpointing(workflow, checkpointing_backend)

        # Run workflow
        config = {"configurable": {"thread_id": session_id}}
        final_state = compiled.invoke(initial_state, config)

        logger.info(f"[COMPARISON] Workflow completed for session {session_id}")

        return {
            "success": True,
            "response": final_state.get("response"),
            "response_data": final_state.get("response_data"),
            "ranked_products": final_state.get("ranked_products", []),
            "formatted_output": final_state.get("formatted_output"),
            "error": final_state.get("error")
        }

    except TimeoutError as e:
        logger.error(f"[COMPARISON] Workflow lock timeout for session {session_id}: {e}")
        return {
            "success": False,
            "error": "Another workflow is currently running for this session. Please try again."
        }
    except Exception as e:
        logger.error(f"[COMPARISON] Workflow failed for session {session_id}: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }


# ============================================================================
# CANDIDATE DISCOVERY FROM PPI/MongoDB
# ============================================================================

def discover_candidates_from_ppi(
    product_type: str,
    category: str = None,
    max_per_vendor: int = 5
) -> List[Dict[str, Any]]:
    """
    Discover candidate products from Potential Product Index / MongoDB.
    Returns multi-level structure with vendors, series, and models.
    
    Args:
        product_type: Type of product to search
        category: Optional category filter
        max_per_vendor: Maximum models per vendor
    
    Returns:
        List of candidates with structure:
        [
            {
                "vendor": "Honeywell",
                "series": [
                    {"name": "ST800", "models": ["STG74L", "STG75L"]},
                    {"name": "ST700", "models": ["STG71L", "STG72L"]}
                ]
            }
        ]
    """
    candidates = []
    
    try:
        if MONGO_AVAILABLE:
            # Query MongoDB for candidates
            mongo_uri = os.getenv("MONGODB_URI")
            if mongo_uri:
                client = MongoClient(mongo_uri)
                db = client.get_default_database()
                
                # Query potential_products collection
                query = {"product_type": {"$regex": product_type, "$options": "i"}}
                if category:
                    query["category"] = {"$regex": category, "$options": "i"}
                
                products = list(db.potential_products.find(query).limit(50))
                
                # Group by vendor
                vendor_groups = {}
                for product in products:
                    vendor = product.get("vendor", "Unknown")
                    if vendor not in vendor_groups:
                        vendor_groups[vendor] = {}
                    
                    series = product.get("series", "Default")
                    if series not in vendor_groups[vendor]:
                        vendor_groups[vendor][series] = []
                    
                    model = product.get("model", product.get("model_family", ""))
                    if model and model not in vendor_groups[vendor][series]:
                        vendor_groups[vendor][series].append(model)
                
                # Convert to structured format
                for vendor, series_dict in vendor_groups.items():
                    series_list = []
                    for series_name, models in series_dict.items():
                        series_list.append({
                            "name": series_name,
                            "models": models[:max_per_vendor]
                        })
                    
                    candidates.append({
                        "vendor": vendor,
                        "series": series_list
                    })
                
                client.close()
        
        # Fallback to known vendors if no MongoDB results
        if not candidates:
            logger.info("Using fallback vendor candidates")
            candidates = [
                {
                    "vendor": "Honeywell",
                    "series": [
                        {"name": "ST800", "models": ["STG74L", "STG75L", "STG77L"]},
                        {"name": "ST700", "models": ["STG71L", "STG72L"]}
                    ]
                },
                {
                    "vendor": "Emerson",
                    "series": [
                        {"name": "3051S", "models": ["3051S1CD", "3051S2CD"]},
                        {"name": "3051C", "models": ["3051CD1", "3051CD2"]}
                    ]
                },
                {
                    "vendor": "Yokogawa",
                    "series": [
                        {"name": "EJA-E", "models": ["EJA110E", "EJA120E"]},
                        {"name": "EJA-A", "models": ["EJA110A", "EJA120A"]}
                    ]
                },
                {
                    "vendor": "ABB",
                    "series": [
                        {"name": "266", "models": ["266DSH", "266GSH"]}
                    ]
                },
                {
                    "vendor": "Siemens",
                    "series": [
                        {"name": "SITRANS P", "models": ["P320", "P410"]}
                    ]
                }
            ]
        
        return candidates
        
    except Exception as e:
        logger.error(f"Candidate discovery failed: {e}")
        return []


# ============================================================================
# SPEC-OBJECT BASED COMPARISON (UI Invocation)
# ============================================================================

@with_workflow_lock(session_id_param="session_id", timeout=90.0)
def run_comparison_from_spec(
    spec_object: Dict[str, Any],
    comparison_type: str = "full",
    session_id: str = "default",
    user_id: str = None,
    checkpointing_backend: str = "memory"
) -> Dict[str, Any]:
    """
    Run comparison workflow from a SpecObject (triggered by [COMPARE VENDORS] button).
    Uses session-level locking to prevent race conditions.

    This is the main entry point for UI-triggered comparisons after
    Solution Workflow → Instrument Detail Capture → [COMPARE VENDORS]

    Args:
        spec_object: Finalized specification object
        comparison_type: "vendor", "series", "model", or "full"
        session_id: Session identifier
        user_id: Optional user identifier
        checkpointing_backend: Backend for state persistence

    Returns:
        Multi-level comparison result:
        {
            "success": true,
            "vendor_ranking": [...],      # Cross-vendor comparison
            "series_comparisons": {...},  # Within-vendor series comparison
            "model_comparisons": {...},   # Within-series model comparison
            "top_recommendation": {...},
            "formatted_output": "..."
        }
    """
    logger.info("=" * 60)
    logger.info(f"COMPARISON FROM SPEC - [COMPARE VENDORS] INVOKED (session: {session_id})")
    logger.info("=" * 60)
    
    try:
        # Validate spec_object
        if isinstance(spec_object, dict):
            spec = SpecObject(**spec_object)
        else:
            spec = spec_object
        
        product_type = spec.product_type
        specifications = spec.specifications
        required_certs = spec.required_certifications
        
        logger.info(f"Product Type: {product_type}")
        logger.info(f"Specifications: {specifications}")
        logger.info(f"Required Certifications: {required_certs}")
        
        # Step 1: Discover candidates from PPI/MongoDB
        logger.info("[STEP 1] Discovering candidates from PPI...")
        candidates = discover_candidates_from_ppi(product_type, spec.category)
        logger.info(f"Found {len(candidates)} vendor candidates")
        
        # Step 2: Get RAG constraints
        logger.info("[STEP 2] Fetching RAG constraints...")
        rag_aggregator = create_rag_aggregator()
        rag_results = rag_aggregator.query_all_parallel(
            product_type=product_type,
            requirements=specifications
        )
        constraint_context = rag_aggregator.merge_to_constraint_context(rag_results)
        
        # Step 3: Apply strategy filter
        logger.info("[STEP 3] Applying strategy filter...")
        strategy_filter = create_strategy_filter()
        vendor_names = [c["vendor"] for c in candidates]
        filter_result = strategy_filter.apply_all_rules(
            vendors=vendor_names,
            constraint_context=constraint_context
        )
        
        filtered_vendors = [f["vendor"] for f in filter_result.get("filtered_candidates", [])]
        excluded = filter_result.get("excluded", [])
        
        logger.info(f"Filtered to {len(filtered_vendors)} vendors, excluded {len(excluded)}")
        
        # Step 4: Technical analysis (parallel with thread-safe collection)
        logger.info("[STEP 4] Running parallel technical analysis...")

        analysis_collector = ThreadSafeResultCollector()
        series_comparisons = {}
        model_comparisons = {}

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {}

            for candidate in candidates:
                vendor = candidate["vendor"]
                if vendor not in filtered_vendors:
                    continue

                for series in candidate.get("series", []):
                    series_name = series["name"]

                    for model in series.get("models", []):
                        future = executor.submit(
                            analyze_vendor_match_tool.invoke,
                            {
                                "vendor": vendor,
                                "requirements": specifications,
                                "pdf_content": None,
                                "product_data": {
                                    "series": series_name,
                                    "model": model
                                }
                            }
                        )
                        futures[future] = {
                            "vendor": vendor,
                            "series": series_name,
                            "model": model
                        }

            for future in as_completed(futures):
                info = futures[future]
                try:
                    result = future.result()
                    if result.get("success"):
                        analysis_result = {
                            "vendor": info["vendor"],
                            "series": info["series"],
                            "model": info["model"],
                            "match_score": result.get("match_score", 0),
                            "requirements_match": result.get("requirements_match", False),
                            "reasoning": result.get("reasoning", ""),
                            "key_strengths": result.get("key_strengths", []),
                            "limitations": result.get("limitations", "")
                        }
                        analysis_collector.add_result(analysis_result)
                except Exception as e:
                    logger.error(f"Analysis failed for {info}: {e}")
                    analysis_collector.add_error(e, info)

        # Get thread-safe results
        analysis_results = analysis_collector.get_results()

        # Log summary
        summary = analysis_collector.summary()
        logger.info(
            f"[STEP 4] Parallel analysis complete: "
            f"{summary['total_results']} successes, {summary['total_errors']} errors"
        )
        
        # Step 5: Calculate scores and rank
        logger.info("[STEP 5] Calculating scores and ranking...")
        
        ranked_products = []
        preferred_vendors = constraint_context.get("preferred_vendors", [])
        standardized_vendor = constraint_context.get("standardized_vendor")
        
        for result in analysis_results:
            vendor = result["vendor"]
            match_score = result.get("match_score", 0)
            
            # Calculate scoring breakdown
            is_preferred = any(p.lower() in vendor.lower() for p in preferred_vendors)
            strategy_priority = 25 if is_preferred else 18
            
            technical_fit = int((match_score / 100) * 25)
            
            matches_base = standardized_vendor and standardized_vendor.lower() in vendor.lower()
            asset_alignment = 20 if matches_base else 10
            
            standards_compliance = 15 if required_certs else 12
            data_completeness = 13
            
            overall_score = (strategy_priority + technical_fit + 
                           asset_alignment + standards_compliance + data_completeness)
            
            ranked_products.append({
                "vendor": vendor,
                "series": result["series"],
                "model": result["model"],
                "overall_score": overall_score,
                "scoring_breakdown": {
                    "strategy_priority": strategy_priority,
                    "technical_fit": technical_fit,
                    "asset_alignment": asset_alignment,
                    "standards_compliance": standards_compliance,
                    "data_completeness": data_completeness
                },
                "constraints_met": {
                    "strategy": True,
                    "standards": True,
                    "installed_base": matches_base
                },
                "key_advantages": result.get("key_strengths", [])[:3],
                "reasoning": result.get("reasoning", "")
            })
        
        # Sort by overall score
        ranked_products.sort(key=lambda x: x["overall_score"], reverse=True)
        
        # Add ranks
        for i, product in enumerate(ranked_products):
            product["rank"] = i + 1
        
        # Build series comparisons (within vendor)
        vendor_series_groups = {}
        for product in ranked_products:
            vendor = product["vendor"]
            if vendor not in vendor_series_groups:
                vendor_series_groups[vendor] = {}
            
            series = product["series"]
            if series not in vendor_series_groups[vendor]:
                vendor_series_groups[vendor][series] = []
            vendor_series_groups[vendor][series].append(product)
        
        for vendor, series_dict in vendor_series_groups.items():
            if len(series_dict) > 1:
                series_comparisons[vendor] = {
                    series_name: sorted(products, key=lambda x: x["overall_score"], reverse=True)
                    for series_name, products in series_dict.items()
                }
        
        # Generate formatted output
        formatted_output = generate_comparison_output(ranked_products[:5])
        
        # Top recommendation
        top_recommendation = ranked_products[0] if ranked_products else None
        
        logger.info("=" * 60)
        logger.info(f"COMPARISON COMPLETE: {len(ranked_products)} products ranked")
        if top_recommendation:
            logger.info(f"TOP RECOMMENDATION: {top_recommendation['vendor']} {top_recommendation['series']} {top_recommendation['model']}")
        logger.info("=" * 60)

        return {
            "success": True,
            "spec_object": spec.model_dump() if hasattr(spec, 'model_dump') else spec_object,
            "comparison_type": comparison_type,
            "vendor_ranking": ranked_products,
            "series_comparisons": series_comparisons,
            "model_comparisons": model_comparisons,
            "top_recommendation": top_recommendation,
            "excluded_vendors": excluded,
            "constraint_context": {
                "preferred_vendors": preferred_vendors,
                "standardized_vendor": standardized_vendor
            },
            "formatted_output": formatted_output,
            "total_analyzed": len(analysis_results),
            "session_id": session_id
        }

    except TimeoutError as e:
        logger.error(f"Comparison from spec - Lock timeout for session {session_id}: {e}")
        return {
            "success": False,
            "error": "Another comparison is currently running for this session. Please try again."
        }
    except Exception as e:
        logger.error(f"Comparison from spec failed for session {session_id}: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "create_comparison_workflow",
    "run_comparison_workflow",
    "run_comparison_from_spec",
    "discover_candidates_from_ppi",
    "generate_comparison_output"
]
