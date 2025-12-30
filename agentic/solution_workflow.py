# agentic/solution_workflow.py
# Solution-Based Workflow for Design Requests

import json
import logging
from typing import Dict, Any, List, Literal, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from langgraph.graph import StateGraph, END

from .models import (
    SolutionState,
    create_solution_state,
    WorkflowStep
)
from .rag_components import RAGAggregator, StrategyFilter, create_rag_aggregator, create_strategy_filter
from .checkpointing import compile_with_checkpointing

from tools.intent_tools import classify_intent_tool, extract_requirements_tool
from tools.schema_tools import load_schema_tool, validate_requirements_tool, get_missing_fields_tool
from tools.vendor_tools import search_vendors_tool
from tools.analysis_tools import analyze_vendor_match_tool
from tools.ranking_tools import judge_analysis_tool, rank_products_tool
from tools.search_tools import search_product_images_tool, search_pdf_datasheets_tool

# Import synchronization utilities
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.workflow_sync import (
    with_workflow_lock,
    with_state_transaction,
    ThreadSafeResultCollector
)

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

SUMMARY_AGENT_PROMPT = """
You are Engenie's Summary Agent. Summarize the vendor analysis results.

Vendor Analysis Results:
{analysis_results}

Create a concise summary that:
1. Highlights top performing vendors
2. Identifies key differentiators
3. Notes any concerns or limitations
4. Provides recommendation rationale

Return ONLY valid JSON:
{{
    "top_vendors": ["<top 3 vendors>"],
    "key_differentiators": ["<main differences>"],
    "concerns": ["<notable concerns>"],
    "summary": "<executive summary in 2-3 sentences>"
}}
"""


# ============================================================================
# WORKFLOW NODES
# ============================================================================

def classify_intent_node(state: SolutionState) -> SolutionState:
    """
    Agent 1: Intent Classification + Comparison Detection.
    Determines the type of user request and detects comparison mode.
    """
    logger.info("[SOLUTION] Node 1: Classifying intent and detecting comparison mode...")

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

        # Step 2: NEW - Detect comparison mode
        user_input_lower = state["user_input"].lower()

        # Comparison keywords
        comparison_keywords = [
            "compare", "comparison", "versus", "vs", "vs.",
            "difference between", "better", "which is better",
            "pros and cons", "side by side", " or ",
            "which one", "what's the difference"
        ]

        # Check for comparison intent
        is_comparison = any(kw in user_input_lower for kw in comparison_keywords)

        # Check for multiple vendor/model mentions (e.g., "Honeywell vs Emerson")
        vendors_mentioned = sum(1 for v in ["honeywell", "emerson", "abb", "yokogawa", "siemens", "rosemount"]
                                if v in user_input_lower)
        if vendors_mentioned >= 2:
            is_comparison = True

        # Set comparison mode
        if is_comparison:
            state["comparison_mode"] = True
            state["mode_confidence"] = 0.95 if vendors_mentioned >= 2 else 0.75
            logger.info(f"[SOLUTION] ✓ Comparison mode detected (confidence: {state['mode_confidence']:.2f})")
        else:
            state["comparison_mode"] = False
            state["mode_confidence"] = 0.85

        state["current_step"] = "validate_requirements"

        state["messages"] = state.get("messages", []) + [{
            "role": "system",
            "content": f"Intent: {state['intent']}, Comparison mode: {state['comparison_mode']} (confidence: {state['mode_confidence']:.2f})"
        }]

        logger.info(f"[SOLUTION] Intent: {state['intent']}, Comparison: {state['comparison_mode']}")

    except Exception as e:
        logger.error(f"[SOLUTION] Intent classification failed: {e}")
        state["error"] = str(e)
        state["intent"] = "requirements"
        state["comparison_mode"] = False

    return state


def validate_requirements_node(state: SolutionState) -> SolutionState:
    """
    Agent 2: Validation Agent.
    Validates user requirements and extracts product type.
    """
    logger.info("[SOLUTION] Node 2: Validating requirements...")
    
    try:
        # Extract requirements
        extract_result = extract_requirements_tool.invoke({
            "user_input": state["user_input"]
        })
        
        if extract_result.get("success"):
            state["product_type"] = extract_result.get("product_type", "")
            state["provided_requirements"] = extract_result.get("requirements", {})
        
        # Load schema if product type detected
        if state["product_type"]:
            schema_result = load_schema_tool.invoke({
                "product_type": state["product_type"]
            })
            
            if schema_result.get("success"):
                state["schema"] = schema_result.get("schema")
                state["schema_source"] = schema_result.get("source", "mongodb")
        
        # Validate against schema
        if state["schema"]:
            validate_result = validate_requirements_tool.invoke({
                "user_input": state["user_input"],
                "product_type": state["product_type"],
                "schema": state["schema"]
            })
            
            if validate_result.get("success"):
                state["is_requirements_valid"] = validate_result.get("is_valid", False)
                state["missing_requirements"] = validate_result.get("missing_fields", [])
        
        state["current_step"] = "aggregate_data"
        
        state["messages"] = state.get("messages", []) + [{
            "role": "system",
            "content": f"Validation: {'Valid' if state['is_requirements_valid'] else 'Missing info'}, Product: {state['product_type']}"
        }]
        
        logger.info(f"[SOLUTION] Product type: {state['product_type']}, Valid: {state['is_requirements_valid']}")
        
    except Exception as e:
        logger.error(f"[SOLUTION] Validation failed: {e}")
        state["error"] = str(e)
    
    return state


def aggregate_data_node(state: SolutionState) -> SolutionState:
    """
    Agent 3: RAG Data Aggregator.
    Queries RAG systems for context data.
    """
    logger.info("[SOLUTION] Node 3: Aggregating RAG data...")
    
    try:
        rag_aggregator = create_rag_aggregator()
        
        # Query all RAGs
        rag_results = rag_aggregator.query_all_parallel(
            product_type=state["product_type"] or "industrial instrument",
            requirements=state["provided_requirements"]
        )
        
        state["rag_context"] = rag_results
        
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
        
        logger.info(f"[SOLUTION] Strategy present: {state['strategy_present']}")
        
    except Exception as e:
        logger.error(f"[SOLUTION] RAG aggregation failed: {e}")
        state["error"] = str(e)
        state["rag_context"] = {}
    
    return state


def lookup_schema_node(state: SolutionState) -> SolutionState:
    """
    Agent 4: Schema Agent.
    Looks up or generates schema for product type.
    """
    logger.info("[SOLUTION] Node 4: Schema lookup...")
    
    try:
        if not state.get("schema"):
            # Attempt schema lookup from MongoDB
            schema_result = load_schema_tool.invoke({
                "product_type": state["product_type"] or "industrial instrument"
            })
            
            if schema_result.get("success") and schema_result.get("schema"):
                state["schema"] = schema_result["schema"]
                state["schema_source"] = "mongodb"
            else:
                # No schema found - trigger Potential Product Index
                state["schema_source"] = "not_found"
                logger.info("[SOLUTION] Schema not found - may need Potential Product Index")
        
        state["current_step"] = "apply_strategy"
        
        state["messages"] = state.get("messages", []) + [{
            "role": "system",
            "content": f"Schema source: {state.get('schema_source', 'unknown')}"
        }]
        
    except Exception as e:
        logger.error(f"[SOLUTION] Schema lookup failed: {e}")
        state["error"] = str(e)
    
    return state


def apply_strategy_node(state: SolutionState) -> SolutionState:
    """
    Agent 5: Strategy Agent.
    Applies procurement strategy rules.
    """
    logger.info("[SOLUTION] Node 5: Applying strategy...")
    
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
            # Apply strategy filter
            strategy_filter = create_strategy_filter()
            strategy_data = state["rag_context"].get("strategy", {}).get("data", {})
            
            constraint_context = {
                "preferred_vendors": strategy_data.get("preferred_vendors", []),
                "forbidden_vendors": strategy_data.get("forbidden_vendors", []),
                "neutral_vendors": strategy_data.get("neutral_vendors", []),
                "procurement_priorities": strategy_data.get("procurement_priorities", {})
            }
            
            filter_result = strategy_filter.apply_strategy_rules(
                vendors=available_vendors,
                constraint_context=constraint_context
            )
            
            state["filtered_vendors"] = [v["vendor"] for v in filter_result["filtered_vendors"]]
        else:
            # No strategy - use all vendors
            state["filtered_vendors"] = available_vendors
        
        state["current_step"] = "parallel_analysis"
        
        state["messages"] = state.get("messages", []) + [{
            "role": "system",
            "content": f"Strategy applied: {len(state['filtered_vendors'])}/{len(state['available_vendors'])} vendors"
        }]
        
        logger.info(f"[SOLUTION] Filtered vendors: {len(state['filtered_vendors'])}")
        
    except Exception as e:
        logger.error(f"[SOLUTION] Strategy application failed: {e}")
        state["error"] = str(e)
    
    return state


@with_state_transaction(auto_commit=True)
def parallel_analysis_node(state: SolutionState) -> SolutionState:
    """
    Agent 6: Analysis Coordinator.
    Spawns parallel vendor analysis with thread-safe result collection.
    """
    logger.info("[SOLUTION] Node 6: Parallel vendor analysis...")

    try:
        vendors = state.get("filtered_vendors", [])[:5]  # Top 5 vendors
        requirements = state["provided_requirements"]

        # Thread-safe collectors
        vendor_data_collector = ThreadSafeResultCollector()
        analysis_collector = ThreadSafeResultCollector()

        # Phase 1: Search for PDFs and images in parallel
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {}

            for vendor in vendors:
                # Search for PDF datasheets
                pdf_future = executor.submit(
                    search_pdf_datasheets_tool.invoke,
                    {
                        "vendor": vendor,
                        "product_type": state["product_type"] or "industrial instrument",
                        "model_family": None
                    }
                )
                futures[pdf_future] = ("pdf", vendor)

                # Search for product images
                img_future = executor.submit(
                    search_product_images_tool.invoke,
                    {
                        "vendor": vendor,
                        "product_name": state["product_type"] or "industrial instrument",
                        "product_type": state["product_type"] or "instrument"
                    }
                )
                futures[img_future] = ("image", vendor)

            # Collect PDF and image results (thread-safe)
            vendor_data = {}
            for future in as_completed(futures):
                result_type, vendor = futures[future]
                try:
                    result = future.result()
                    if vendor not in vendor_data:
                        vendor_data[vendor] = {"vendor": vendor}

                    if result_type == "pdf" and result.get("success"):
                        vendor_data[vendor]["pdf_url"] = result.get("pdf_urls", [None])[0]
                    elif result_type == "image" and result.get("success"):
                        vendor_data[vendor]["image_url"] = result.get("images", [None])[0]
                except Exception as e:
                    logger.error(f"Search failed for {vendor}: {e}")
                    vendor_data_collector.add_error(e, {"vendor": vendor, "type": result_type})

        # Phase 2: Analyze each vendor in parallel
        with ThreadPoolExecutor(max_workers=5) as executor:
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

            for future in as_completed(analyze_futures):
                vendor = analyze_futures[future]
                try:
                    result = future.result()
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
                except Exception as e:
                    logger.error(f"Analysis failed for {vendor}: {e}")
                    analysis_collector.add_error(e, {"vendor": vendor})

        # Get thread-safe results
        analysis_results = analysis_collector.get_results()

        # Log summary
        summary = analysis_collector.summary()
        logger.info(
            f"[SOLUTION] Parallel analysis complete: "
            f"{summary['total_results']} successes, {summary['total_errors']} errors"
        )

        # Update state (protected by transaction)
        state["parallel_analysis_results"] = analysis_results
        state["current_step"] = "summarize"

        state["messages"] = state.get("messages", []) + [{
            "role": "system",
            "content": f"Analyzed {len(analysis_results)} vendors"
        }]

        logger.info(f"[SOLUTION] Analyzed {len(analysis_results)} vendors")
        
    except Exception as e:
        logger.error(f"[SOLUTION] Parallel analysis failed: {e}")
        state["error"] = str(e)
    
    return state


def summarize_results_node(state: SolutionState) -> SolutionState:
    """
    Summary Agent: Summarize analysis results.
    """
    logger.info("[SOLUTION] Node 6b: Summarizing results...")
    
    try:
        analysis_results = state.get("parallel_analysis_results", [])
        
        llm = create_llm_with_fallback(
            model="gemini-2.0-flash-exp",
            temperature=0.2,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        prompt = ChatPromptTemplate.from_template(SUMMARY_AGENT_PROMPT)
        parser = JsonOutputParser()
        chain = prompt | llm | parser
        
        result = chain.invoke({
            "analysis_results": json.dumps(analysis_results, indent=2)
        })
        
        state["summarized_results"] = [result]
        state["current_step"] = "judge"
        
        state["messages"] = state.get("messages", []) + [{
            "role": "system",
            "content": f"Summary: {result.get('summary', 'No summary available')[:100]}..."
        }]
        
    except Exception as e:
        logger.error(f"[SOLUTION] Summarization failed: {e}")
        state["error"] = str(e)
        state["summarized_results"] = []
    
    return state


def judge_results_node(state: SolutionState) -> SolutionState:
    """
    Agent 7: Judge Agent.
    Validates analysis results.
    """
    logger.info("[SOLUTION] Node 7: Judging results...")
    
    try:
        analysis_results = state.get("parallel_analysis_results", [])
        
        # Use judge_analysis_tool
        judge_result = judge_analysis_tool.invoke({
            "original_requirements": state["provided_requirements"],
            "vendor_analysis": analysis_results,
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
            "content": f"Judge validation: {len(state['judge_validation'].get('passed', []))} passed"
        }]
        
    except Exception as e:
        logger.error(f"[SOLUTION] Judging failed: {e}")
        state["error"] = str(e)
        state["judge_validation"] = {"passed": [], "failed": []}
    
    return state


def rank_products_node(state: SolutionState) -> SolutionState:
    """
    Agent 8: Ranking Agent.
    Ranks products and generates final output.
    """
    logger.info("[SOLUTION] Node 8: Ranking products...")
    
    try:
        analysis_results = state.get("parallel_analysis_results", [])
        
        # Use rank_products_tool
        rank_result = rank_products_tool.invoke({
            "vendor_matches": analysis_results,
            "requirements": state["provided_requirements"]
        })
        
        if rank_result.get("success"):
            state["ranked_results"] = rank_result.get("ranked_products", [])
            
            # Generate response
            response_lines = ["**🎯 Product Recommendations**\n"]
            
            for i, product in enumerate(state["ranked_results"][:5], 1):
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
                "product_type": state["product_type"]
            }
        
        state["current_step"] = "complete"
        
        state["messages"] = state.get("messages", []) + [{
            "role": "system",
            "content": f"Ranked {len(state.get('ranked_results', []))} products"
        }]
        
        logger.info(f"[SOLUTION] Ranking complete")
        
    except Exception as e:
        logger.error(f"[SOLUTION] Ranking failed: {e}")
        state["error"] = str(e)
        state["ranked_results"] = []
    
    return state


def format_comparison_node(state: SolutionState) -> SolutionState:
    """
    NEW Node: Format results as detailed comparison if in comparison mode.
    Generates side-by-side comparison with winner, key differences, and trade-offs.
    """
    logger.info("[SOLUTION] Node 9: Formatting comparison output...")

    try:
        # Only run if in comparison mode
        if not state.get("comparison_mode", False):
            logger.info("[SOLUTION] Not in comparison mode - skipping comparison formatting")
            return state

        ranked_results = state.get("ranked_results", [])

        if not ranked_results:
            state["comparison_output"] = {
                "summary": "No products found for comparison",
                "winner": None,
                "key_differences": [],
                "trade_offs": [],
                "recommendation": "Unable to complete comparison - no matching products found"
            }
            logger.warning("[SOLUTION] No ranked results for comparison")
            return state

        # Get top 3-5 products for comparison
        top_products = ranked_results[:min(5, len(ranked_results))]

        logger.info(f"[SOLUTION] Generating comparison analysis for {len(top_products)} products")

        # Use LLM to generate comprehensive comparison analysis
        llm = create_llm_with_fallback(
            model="gemini-2.0-flash-exp",
            temperature=0.2,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        comparison_prompt = """
You are Engenie's Comparison Presenter. Generate a detailed product comparison analysis.

Ranked Products:
{ranked_products}

Original User Request: {user_input}

Scoring Breakdown (100 points total):
- Overall Match Score: Vendor-specific technical fit
- Key Strengths: Standout features
- Concerns: Limitations or risks

Generate a comprehensive comparison analysis with:
1. **Winner** - Which product is the BEST choice overall and WHY (be specific about technical/business reasons)
2. **Key Differences** - 3-4 main technical or business differences between top options
3. **Trade-offs** - 2-3 important trade-offs to consider when choosing between options
4. **Recommendation** - Final recommendation with clear rationale based on requirements

Return ONLY valid JSON:
{{
    "summary": "<executive summary in 2-3 sentences explaining the comparison result>",
    "winner": {{
        "vendor": "<vendor name>",
        "model": "<model name>",
        "reason": "<specific reason why this is the best choice - mention key advantages>"
    }},
    "key_differences": [
        "<difference 1: e.g., Honeywell has better SIL rating but Emerson has wider range>",
        "<difference 2>",
        "<difference 3>"
    ],
    "trade_offs": [
        "<trade-off 1: e.g., Higher accuracy vs lower cost>",
        "<trade-off 2>"
    ],
    "recommendation": "<final recommendation with rationale - be specific about use case fit>"
}}
"""

        prompt = ChatPromptTemplate.from_template(comparison_prompt)
        parser = JsonOutputParser()
        chain = prompt | llm | parser

        result = chain.invoke({
            "ranked_products": json.dumps(top_products, indent=2),
            "user_input": state["user_input"]
        })

        state["comparison_output"] = result

        # Generate formatted response for comparison mode
        response_lines = ["**🔍 Product Comparison Analysis**\n"]

        # Summary
        summary = result.get("summary", "")
        if summary:
            response_lines.append(f"{summary}\n")

        # Winner
        winner = result.get("winner", {})
        if winner and winner.get("vendor"):
            response_lines.append(f"**✅ Recommended Winner: {winner.get('vendor')} {winner.get('model', '')}**")
            response_lines.append(f"_{winner.get('reason', '')}_\n")

        # Top products comparison table
        response_lines.append("**📊 Top Options Compared:**")
        for i, product in enumerate(top_products, 1):
            response_lines.append(f"\n**#{i} {product.get('vendor', '')} - {product.get('product_name', '')}**")
            response_lines.append(f"   Match Score: {product.get('overall_score', 0)}/100")

            strengths = product.get("key_strengths", [])
            if strengths:
                response_lines.append(f"   ✓ Strengths: {', '.join(strengths[:2])}")

            concerns = product.get("concerns", [])
            if concerns:
                response_lines.append(f"   ⚠ Concerns: {concerns[0]}")

        # Key Differences
        differences = result.get("key_differences", [])
        if differences:
            response_lines.append("\n**🔑 Key Differences:**")
            for diff in differences:
                response_lines.append(f"• {diff}")

        # Trade-offs
        trade_offs = result.get("trade_offs", [])
        if trade_offs:
            response_lines.append("\n**⚖️ Trade-offs to Consider:**")
            for trade_off in trade_offs:
                response_lines.append(f"• {trade_off}")

        # Final Recommendation
        recommendation = result.get("recommendation", "")
        if recommendation:
            response_lines.append(f"\n**💡 Final Recommendation:**")
            response_lines.append(recommendation)

        state["response"] = "\n".join(response_lines)
        state["response_data"] = {
            "comparison_output": result,
            "ranked_products": top_products,
            "total_analyzed": len(state.get("parallel_analysis_results", [])),
            "product_type": state.get("product_type"),
            "comparison_mode": True
        }

        state["current_step"] = "complete"

        state["messages"] = state.get("messages", []) + [{
            "role": "system",
            "content": f"Comparison formatting complete - Winner: {winner.get('vendor', 'N/A')}"
        }]

        logger.info(f"[SOLUTION] Comparison formatting complete - Winner: {winner.get('vendor', 'N/A')}")

    except Exception as e:
        logger.error(f"[SOLUTION] Comparison formatting failed: {e}")
        state["error"] = str(e)
        # Fallback to regular ranking output if comparison fails
        state["comparison_mode"] = False

    return state


# ============================================================================
# ROUTING FUNCTIONS
# ============================================================================

def route_after_intent(state: SolutionState) -> Literal["validate", "end"]:
    """Route after intent classification."""
    if state.get("error"):
        return "end"
    
    intent = state.get("intent", "")
    if intent in ["requirements", "additional_specs", "question"]:
        return "validate"
    
    return "validate"


def route_after_schema(state: SolutionState) -> Literal["strategy", "potential_index"]:
    """Route based on schema availability."""
    if state.get("schema_source") == "not_found":
        # Would trigger Potential Product Index sub-workflow
        # For now, continue with strategy
        logger.info("[SOLUTION] Schema not found - continuing without full schema")
    
    return "strategy"


def route_after_strategy(state: SolutionState) -> Literal["normal", "strategy_path"]:
    """Route based on strategy presence."""
    if state.get("strategy_present"):
        return "strategy_path"
    return "normal"


def route_after_ranking(state: SolutionState) -> Literal["comparison", "end"]:
    """
    NEW: Route after ranking based on comparison mode.
    If comparison mode is enabled, format as detailed comparison.
    Otherwise, end workflow with standard ranking output.
    """
    if state.get("comparison_mode", False):
        logger.info("[SOLUTION] Routing to comparison formatting")
        return "comparison"
    logger.info("[SOLUTION] Routing to end (standard mode)")
    return "end"


# ============================================================================
# WORKFLOW CREATION
# ============================================================================

def create_solution_workflow() -> StateGraph:
    """
    Create the Solution-Based workflow with Comparison Mode support.

    Flow:
    1. Intent Classifier (+ Comparison Detection)
    2. Validation Agent + RAG Aggregation
    3. Schema Agent
    4. Strategy Agent
    5. Analysis Coordinator (Parallel)
    6. Summary Agents
    7. Judge Agent
    8. Ranking Agent
    9. [CONDITIONAL] Comparison Formatter (if comparison_mode=True)

    Supports two modes:
    - Regular Search Mode: "I need pressure transmitters"
    - Comparison Mode: "Compare Honeywell vs Emerson transmitters"
    """

    workflow = StateGraph(SolutionState)

    # Add nodes
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("validate_requirements", validate_requirements_node)
    workflow.add_node("aggregate_data", aggregate_data_node)
    workflow.add_node("lookup_schema", lookup_schema_node)
    workflow.add_node("apply_strategy", apply_strategy_node)
    workflow.add_node("parallel_analysis", parallel_analysis_node)
    workflow.add_node("summarize_results", summarize_results_node)
    workflow.add_node("judge_results", judge_results_node)
    workflow.add_node("rank_products", rank_products_node)
    workflow.add_node("format_comparison", format_comparison_node)  # NEW

    # Set entry point
    workflow.set_entry_point("classify_intent")

    # Add edges (sequential flow until ranking)
    workflow.add_edge("classify_intent", "validate_requirements")
    workflow.add_edge("validate_requirements", "aggregate_data")
    workflow.add_edge("aggregate_data", "lookup_schema")
    workflow.add_edge("lookup_schema", "apply_strategy")
    workflow.add_edge("apply_strategy", "parallel_analysis")
    workflow.add_edge("parallel_analysis", "summarize_results")
    workflow.add_edge("summarize_results", "judge_results")
    workflow.add_edge("judge_results", "rank_products")

    # NEW: Conditional routing after ranking
    # If comparison_mode=True, format as detailed comparison
    # If comparison_mode=False, end with standard ranking output
    workflow.add_conditional_edges(
        "rank_products",
        route_after_ranking,
        {
            "comparison": "format_comparison",  # Comparison mode path
            "end": END  # Standard mode path
        }
    )

    # Comparison formatter ends the workflow
    workflow.add_edge("format_comparison", END)

    return workflow


# ============================================================================
# WORKFLOW EXECUTION
# ============================================================================

@with_workflow_lock(session_id_param="session_id", timeout=60.0)
def run_solution_workflow(
    user_input: str,
    session_id: str = "default",
    checkpointing_backend: str = "memory"
) -> Dict[str, Any]:
    """
    Run the solution workflow with session-level locking to prevent race conditions.

    Args:
        user_input: User's product requirements
        session_id: Session identifier
        checkpointing_backend: Backend for state persistence

    Returns:
        Workflow result with ranked products
    """
    try:
        logger.info(f"[SOLUTION] Starting workflow for session {session_id}")

        # Create initial state
        initial_state = create_solution_state(user_input, session_id)

        # Create and compile workflow
        workflow = create_solution_workflow()
        compiled = compile_with_checkpointing(workflow, checkpointing_backend)

        # Run workflow
        config = {"configurable": {"thread_id": session_id}}
        final_state = compiled.invoke(initial_state, config)

        logger.info(f"[SOLUTION] Workflow completed for session {session_id}")

        return {
            "success": True,
            "response": final_state.get("response"),
            "response_data": final_state.get("response_data"),
            "ranked_results": final_state.get("ranked_results", []),
            "product_type": final_state.get("product_type"),
            "strategy_present": final_state.get("strategy_present"),
            "error": final_state.get("error")
        }

    except TimeoutError as e:
        logger.error(f"[SOLUTION] Workflow lock timeout for session {session_id}: {e}")
        return {
            "success": False,
            "error": "Another workflow is currently running for this session. Please try again."
        }
    except Exception as e:
        logger.error(f"[SOLUTION] Workflow failed for session {session_id}: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }


def run_solution_workflow_stream(
    user_input: str,
    session_id: str = "default",
    checkpointing_backend: str = "memory",
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
) -> Dict[str, Any]:
    """
    Run the solution workflow with streaming progress updates.

    This is a wrapper around run_solution_workflow that emits progress updates
    during execution. Designed for SSE streaming endpoints.

    Args:
        user_input: User's product requirements
        session_id: Session identifier
        checkpointing_backend: Backend for state persistence
        progress_callback: Callback function to emit progress updates

    Returns:
        Workflow result with ranked products
    """
    from .streaming_utils import ProgressEmitter

    # Create emitter
    emitter = ProgressEmitter(progress_callback)

    try:
        # Step 1: Initialize (5%)
        emitter.emit(
            'initialize',
            'Initializing solution workflow...',
            5,
            data={'user_input_preview': user_input[:100]}
        )

        # Step 2: Intent Classification (15%)
        emitter.emit(
            'classify_intent',
            'Analyzing your requirements...',
            15
        )

        # Step 3: Extract requirements (25%)
        emitter.emit(
            'extract_requirements',
            'Extracting product specifications...',
            25
        )

        # Step 4: Search vendors (40%)
        emitter.emit(
            'search_vendors',
            'Searching for matching vendors...',
            40
        )

        # Step 5: Analyze products (60%)
        emitter.emit(
            'analyze_products',
            'Analyzing vendor products and capabilities...',
            60
        )

        # Step 6: Rank results (80%)
        emitter.emit(
            'rank_results',
            'Ranking products based on your requirements...',
            80
        )

        # Execute the actual workflow
        logger.info(f"[SOLUTION-STREAM] Starting workflow for session {session_id}")
        result = run_solution_workflow(user_input, session_id, checkpointing_backend)

        # Step 7: Complete (100%)
        if result.get("success"):
            emitter.emit(
                'complete',
                'Workflow completed successfully',
                100,
                data={
                    'product_count': len(result.get('ranked_results', [])),
                    'product_type': result.get('product_type')
                }
            )
        else:
            emitter.error(
                result.get('error', 'Unknown error'),
                error_details=result
            )

        return result

    except Exception as e:
        logger.error(f"[SOLUTION-STREAM] Workflow failed: {e}", exc_info=True)
        emitter.error(f'Workflow failed: {str(e)}', error_details=str(e))
        return {
            "success": False,
            "error": str(e)
        }
