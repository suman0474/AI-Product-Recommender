# agentic/workflow.py
# LangGraph Workflow Orchestration

import json
import logging
from typing import Dict, Any, List, Optional, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage

from .models import (
    WorkflowState,
    IntentType,
    WorkflowStep,
    create_initial_state,
    VendorAnalysis,
    VendorMatch,
    OverallRanking,
    ProductRanking
)


from tools.intent_tools import classify_intent_tool, extract_requirements_tool
from tools.schema_tools import load_schema_tool, validate_requirements_tool
from tools.search_tools import search_vendors_tool
from tools.analysis_tools import analyze_vendor_match_tool
from tools.ranking_tools import rank_products_tool

import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# ============================================================================
# WORKFLOW NODES (State Transformations)
# ============================================================================

def classify_intent_node(state: WorkflowState) -> WorkflowState:
    """
    Node: Classify user intent
    Updates: intent, intent_confidence, next_step
    """
    logger.info("[WORKFLOW] Classifying intent...")

    try:
        result = classify_intent_tool.invoke({
            "user_input": state["user_input"],
            "current_step": state["current_step"].value if state["current_step"] else "start",
            "context": None
        })

        # Update state
        state["intent"] = IntentType(result.get("intent", "unrelated"))
        state["intent_confidence"] = result.get("confidence", 0.0)

        # Determine next step based on intent
        intent = state["intent"]
        if intent == IntentType.GREETING:
            state["next_step"] = WorkflowStep.GENERATE_RESPONSE
        elif intent == IntentType.REQUIREMENTS:
            state["next_step"] = WorkflowStep.VALIDATE_REQUIREMENTS
        elif intent == IntentType.QUESTION:
            state["next_step"] = WorkflowStep.GENERATE_RESPONSE
        elif intent == IntentType.CONFIRM:
            state["next_step"] = WorkflowStep.SEARCH_VENDORS
        elif intent == IntentType.ADDITIONAL_SPECS:
            state["next_step"] = WorkflowStep.VALIDATE_REQUIREMENTS
        else:
            state["next_step"] = WorkflowStep.GENERATE_RESPONSE

        state["current_step"] = WorkflowStep.CLASSIFY_INTENT

        # Add message
        state["messages"] = state.get("messages", []) + [{
            "role": "system",
            "content": f"Intent classified as: {state['intent'].value} (confidence: {state['intent_confidence']:.2f})"
        }]

        logger.info(f"[WORKFLOW] Intent: {state['intent'].value}, Next: {state['next_step'].value}")

    except Exception as e:
        logger.error(f"[WORKFLOW] Intent classification failed: {e}")
        state["error"] = str(e)
        state["next_step"] = WorkflowStep.GENERATE_RESPONSE

    return state


def validate_requirements_node(state: WorkflowState) -> WorkflowState:
    """
    Node: Validate user requirements
    Updates: product_type, schema, provided_requirements, missing_requirements, is_requirements_valid
    """
    logger.info("[WORKFLOW] Validating requirements...")

    try:
        # Extract requirements first
        extract_result = extract_requirements_tool.invoke({
            "user_input": state["user_input"]
        })

        product_type = extract_result.get("product_type", "")
        state["product_type"] = product_type

        # Load schema
        schema_result = load_schema_tool.invoke({
            "product_type": product_type
        })
        state["schema"] = schema_result.get("schema", {})

        # Validate requirements
        validation_result = validate_requirements_tool.invoke({
            "user_input": state["user_input"],
            "product_type": product_type,
            "schema": state["schema"]
        })

        state["provided_requirements"] = validation_result.get("provided_requirements", {})
        state["missing_requirements"] = validation_result.get("missing_fields", [])
        state["is_requirements_valid"] = validation_result.get("is_valid", False)

        # Determine next step
        if state["is_requirements_valid"]:
            state["next_step"] = WorkflowStep.SEARCH_VENDORS
        else:
            state["next_step"] = WorkflowStep.COLLECT_MISSING_INFO
            state["requires_user_input"] = True

        state["current_step"] = WorkflowStep.VALIDATE_REQUIREMENTS

        # Add message
        state["messages"] = state.get("messages", []) + [{
            "role": "system",
            "content": f"Validated requirements for {product_type}. Valid: {state['is_requirements_valid']}"
        }]

        logger.info(f"[WORKFLOW] Validation complete. Valid: {state['is_requirements_valid']}")

    except Exception as e:
        logger.error(f"[WORKFLOW] Validation failed: {e}")
        state["error"] = str(e)
        state["next_step"] = WorkflowStep.GENERATE_RESPONSE

    return state


def collect_missing_info_node(state: WorkflowState) -> WorkflowState:
    """
    Node: Handle missing information collection
    Updates: response, requires_user_input
    """
    logger.info("[WORKFLOW] Collecting missing info...")

    try:
        missing = state.get("missing_requirements", [])
        product_type = state.get("product_type", "product")

        if missing:
            # Generate response asking for missing info
            missing_str = ", ".join(missing[:5])  # Limit to 5 fields
            response = f"I've identified that you're looking for a {product_type}. " \
                      f"To find the best match, I need a few more details:\n\n" \
                      f"**Missing specifications:** {missing_str}\n\n" \
                      f"Would you like to provide these, or should I proceed with what we have?"

            state["response"] = response
            state["requires_user_input"] = True
        else:
            state["next_step"] = WorkflowStep.SEARCH_VENDORS

        state["current_step"] = WorkflowStep.COLLECT_MISSING_INFO

    except Exception as e:
        logger.error(f"[WORKFLOW] Collect missing info failed: {e}")
        state["error"] = str(e)

    return state


def search_vendors_node(state: WorkflowState) -> WorkflowState:
    """
    Node: Search for vendors
    Updates: available_vendors, filtered_vendors
    """
    logger.info("[WORKFLOW] Searching vendors...")

    try:
        product_type = state.get("product_type", "")

        result = search_vendors_tool.invoke({
            "product_type": product_type,
            "requirements": state.get("provided_requirements", {})
        })

        state["available_vendors"] = result.get("vendors", [])
        state["filtered_vendors"] = result.get("vendors", [])  # Could be filtered by CSV

        if state["available_vendors"]:
            state["next_step"] = WorkflowStep.ANALYZE_PRODUCTS
        else:
            state["next_step"] = WorkflowStep.GENERATE_RESPONSE
            state["response"] = f"No vendors found for {product_type}. Please try a different product type."

        state["current_step"] = WorkflowStep.SEARCH_VENDORS

        # Add message
        state["messages"] = state.get("messages", []) + [{
            "role": "system",
            "content": f"Found {len(state['available_vendors'])} vendors for {product_type}"
        }]

        logger.info(f"[WORKFLOW] Found {len(state['available_vendors'])} vendors")

    except Exception as e:
        logger.error(f"[WORKFLOW] Vendor search failed: {e}")
        state["error"] = str(e)
        state["next_step"] = WorkflowStep.GENERATE_RESPONSE

    return state


def analyze_products_node(state: WorkflowState) -> WorkflowState:
    """
    Node: Analyze products from vendors
    Updates: vendor_analysis
    """
    logger.info("[WORKFLOW] Analyzing products...")

    try:
        vendors = state.get("filtered_vendors", [])[:5]  # Limit to 5 vendors
        requirements = state.get("provided_requirements", {})
        pdf_content = state.get("pdf_content", {})

        vendor_matches = []

        for vendor in vendors:
            logger.info(f"[WORKFLOW] Analyzing vendor: {vendor}")

            result = analyze_vendor_match_tool.invoke({
                "vendor": vendor,
                "requirements": requirements,
                "pdf_content": pdf_content.get(vendor, ""),
                "product_data": None
            })

            if result.get("success"):
                vendor_matches.append({
                    "vendor": vendor,
                    "product_name": result.get("product_name"),
                    "model_family": result.get("model_family"),
                    "match_score": result.get("match_score", 0),
                    "requirements_match": result.get("requirements_match", False),
                    "reasoning": result.get("reasoning"),
                    "limitations": result.get("limitations"),
                    "key_strengths": result.get("key_strengths", [])
                })

        state["vendor_analysis"] = VendorAnalysis(
            vendor_matches=[VendorMatch(**m) for m in vendor_matches],
            total_vendors_analyzed=len(vendor_matches)
        )

        if vendor_matches:
            state["next_step"] = WorkflowStep.RANK_PRODUCTS
        else:
            state["next_step"] = WorkflowStep.GENERATE_RESPONSE
            state["response"] = "Could not analyze any vendor products. Please try again."

        state["current_step"] = WorkflowStep.ANALYZE_PRODUCTS

        # Add message
        state["messages"] = state.get("messages", []) + [{
            "role": "system",
            "content": f"Analyzed {len(vendor_matches)} vendor products"
        }]

        logger.info(f"[WORKFLOW] Analyzed {len(vendor_matches)} vendors")

    except Exception as e:
        logger.error(f"[WORKFLOW] Product analysis failed: {e}")
        state["error"] = str(e)
        state["next_step"] = WorkflowStep.GENERATE_RESPONSE

    return state


def rank_products_node(state: WorkflowState) -> WorkflowState:
    """
    Node: Rank products
    Updates: ranking
    """
    logger.info("[WORKFLOW] Ranking products...")

    try:
        vendor_analysis = state.get("vendor_analysis")
        requirements = state.get("provided_requirements", {})

        if vendor_analysis and vendor_analysis.vendor_matches:
            # Convert to dict for tool
            vendor_matches = [
                {
                    "vendor": m.vendor,
                    "product_name": m.product_name,
                    "model_family": m.model_family,
                    "match_score": m.match_score,
                    "requirements_match": m.requirements_match,
                    "reasoning": m.reasoning,
                    "limitations": m.limitations
                }
                for m in vendor_analysis.vendor_matches
            ]

            result = rank_products_tool.invoke({
                "vendor_matches": vendor_matches,
                "requirements": requirements
            })

            if result.get("success"):
                ranked_products = [
                    ProductRanking(**p) for p in result.get("ranked_products", [])
                ]
                state["ranking"] = OverallRanking(
                    ranked_products=ranked_products,
                    ranking_summary=result.get("ranking_summary")
                )

        state["next_step"] = WorkflowStep.GENERATE_RESPONSE
        state["current_step"] = WorkflowStep.RANK_PRODUCTS

        # Add message
        ranking = state.get("ranking")
        count = len(ranking.ranked_products) if ranking else 0
        state["messages"] = state.get("messages", []) + [{
            "role": "system",
            "content": f"Ranked {count} products"
        }]

        logger.info(f"[WORKFLOW] Ranked {count} products")

    except Exception as e:
        logger.error(f"[WORKFLOW] Ranking failed: {e}")
        state["error"] = str(e)
        state["next_step"] = WorkflowStep.GENERATE_RESPONSE

    return state


def generate_response_node(state: WorkflowState) -> WorkflowState:
    """
    Node: Generate final response
    Updates: response, response_data
    """
    logger.info("[WORKFLOW] Generating response...")

    try:
        intent = state.get("intent")
        ranking = state.get("ranking")
        error = state.get("error")

        # Handle errors
        if error:
            state["response"] = f"I encountered an issue: {error}. Please try again."
            state["current_step"] = WorkflowStep.GENERATE_RESPONSE
            return state

        # Generate response based on context
        if intent == IntentType.GREETING:
            state["response"] = "Hello! I'm Engenie, your industrial procurement assistant. " \
                               "I can help you find the right instruments and equipment. " \
                               "What type of product are you looking for today?"

        elif ranking and ranking.ranked_products:
            # Format ranking response
            products = ranking.ranked_products[:5]
            response_lines = [
                f"I found {len(products)} matching products for your requirements:\n"
            ]

            for p in products:
                response_lines.append(
                    f"**{p.rank}. {p.vendor} - {p.product_name}** (Score: {p.overall_score:.0f}/100)\n"
                    f"   Strengths: {', '.join(p.key_strengths[:2])}\n"
                )

            state["response"] = "\n".join(response_lines)
            state["response_data"] = {
                "ranked_products": [p.dict() for p in products],
                "summary": ranking.ranking_summary
            }

        elif state.get("response"):
            # Keep existing response
            pass

        else:
            state["response"] = "I understand you're looking for industrial products. " \
                               "Could you please provide more details about your requirements?"

        state["current_step"] = WorkflowStep.GENERATE_RESPONSE
        state["next_step"] = WorkflowStep.END

        logger.info("[WORKFLOW] Response generated")

    except Exception as e:
        logger.error(f"[WORKFLOW] Response generation failed: {e}")
        state["response"] = "I apologize, but I encountered an issue. Please try again."
        state["error"] = str(e)

    return state


# ============================================================================
# ROUTING FUNCTIONS
# ============================================================================

def route_after_intent(state: WorkflowState) -> Literal["validate", "respond", "search"]:
    """Route after intent classification"""
    intent = state.get("intent")
    next_step = state.get("next_step")

    if next_step == WorkflowStep.VALIDATE_REQUIREMENTS:
        return "validate"
    elif next_step == WorkflowStep.SEARCH_VENDORS:
        return "search"
    else:
        return "respond"


def route_after_validation(state: WorkflowState) -> Literal["collect", "search", "respond"]:
    """Route after validation"""
    if state.get("is_requirements_valid"):
        return "search"
    elif state.get("missing_requirements"):
        return "collect"
    else:
        return "respond"


def route_after_search(state: WorkflowState) -> Literal["analyze", "respond"]:
    """Route after vendor search"""
    if state.get("available_vendors"):
        return "analyze"
    else:
        return "respond"


def route_after_analysis(state: WorkflowState) -> Literal["rank", "respond"]:
    """Route after analysis"""
    vendor_analysis = state.get("vendor_analysis")
    if vendor_analysis and vendor_analysis.vendor_matches:
        return "rank"
    else:
        return "respond"


# ============================================================================
# WORKFLOW CREATION
# ============================================================================

def create_procurement_workflow() -> StateGraph:
    """
    Create the main procurement workflow using LangGraph

    Workflow:
    START → classify_intent → validate_requirements → search_vendors
          → analyze_products → rank_products → generate_response → END
    """

    # Create the graph
    workflow = StateGraph(WorkflowState)

    # Add nodes
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("validate_requirements", validate_requirements_node)
    workflow.add_node("collect_missing_info", collect_missing_info_node)
    workflow.add_node("search_vendors", search_vendors_node)
    workflow.add_node("analyze_products", analyze_products_node)
    workflow.add_node("rank_products", rank_products_node)
    workflow.add_node("generate_response", generate_response_node)

    # Set entry point
    workflow.set_entry_point("classify_intent")

    # Add conditional edges from classify_intent
    workflow.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        {
            "validate": "validate_requirements",
            "search": "search_vendors",
            "respond": "generate_response"
        }
    )

    # Add conditional edges from validate_requirements
    workflow.add_conditional_edges(
        "validate_requirements",
        route_after_validation,
        {
            "collect": "collect_missing_info",
            "search": "search_vendors",
            "respond": "generate_response"
        }
    )

    # Add edge from collect_missing_info
    workflow.add_edge("collect_missing_info", "generate_response")

    # Add conditional edges from search_vendors
    workflow.add_conditional_edges(
        "search_vendors",
        route_after_search,
        {
            "analyze": "analyze_products",
            "respond": "generate_response"
        }
    )

    # Add conditional edges from analyze_products
    workflow.add_conditional_edges(
        "analyze_products",
        route_after_analysis,
        {
            "rank": "rank_products",
            "respond": "generate_response"
        }
    )

    # Add edge from rank_products
    workflow.add_edge("rank_products", "generate_response")

    # Add edge to END
    workflow.add_edge("generate_response", END)

    return workflow


def create_instrument_identification_workflow() -> StateGraph:
    """
    Create a specialized workflow for instrument identification

    Workflow:
    START → identify_instruments → identify_accessories → generate_response → END
    """
    from tools.instrument_tools import identify_instruments_tool, identify_accessories_tool

    def identify_instruments_node(state: WorkflowState) -> WorkflowState:
        """Identify instruments from requirements"""
        result = identify_instruments_tool.invoke({
            "requirements": state["user_input"]
        })

        if result.get("success"):
            state["instruments"] = result.get("instruments", [])
            state["response_data"] = {
                "project_name": result.get("project_name"),
                "instruments": result.get("instruments", []),
                "summary": result.get("summary")
            }
        return state

    def identify_accessories_node(state: WorkflowState) -> WorkflowState:
        """Identify accessories for instruments"""
        if state.get("instruments"):
            result = identify_accessories_tool.invoke({
                "instruments": state["instruments"],
                "process_context": state["user_input"]
            })

            if result.get("success"):
                state["accessories"] = result.get("accessories", [])
                if state.get("response_data"):
                    state["response_data"]["accessories"] = result.get("accessories", [])
        return state

    def format_bom_response(state: WorkflowState) -> WorkflowState:
        """Format Bill of Materials response"""
        instruments = state.get("instruments", [])
        accessories = state.get("accessories", [])

        response_lines = ["**Bill of Materials:**\n"]

        if instruments:
            response_lines.append("**Instruments:**")
            for i, inst in enumerate(instruments, 1):
                response_lines.append(f"{i}. {inst.get('category', 'Unknown')} - {inst.get('product_name', '')}")

        if accessories:
            response_lines.append("\n**Accessories:**")
            for i, acc in enumerate(accessories, 1):
                response_lines.append(f"{i}. {acc.get('category', 'Unknown')} - {acc.get('accessory_name', '')}")

        state["response"] = "\n".join(response_lines)
        return state

    # Create workflow
    workflow = StateGraph(WorkflowState)

    workflow.add_node("identify_instruments", identify_instruments_node)
    workflow.add_node("identify_accessories", identify_accessories_node)
    workflow.add_node("format_response", format_bom_response)

    workflow.set_entry_point("identify_instruments")
    workflow.add_edge("identify_instruments", "identify_accessories")
    workflow.add_edge("identify_accessories", "format_response")
    workflow.add_edge("format_response", END)

    return workflow


# ============================================================================
# WORKFLOW EXECUTION
# ============================================================================

def run_workflow(
    user_input: str,
    session_id: str = "default",
    workflow_type: str = "procurement"
) -> Dict[str, Any]:
    """
    Run a workflow with the given input

    Args:
        user_input: User's input message
        session_id: Session identifier for state management
        workflow_type: Type of workflow ("procurement" or "instrument_identification")

    Returns:
        Workflow result with response and data
    """
    try:
        # Create initial state
        initial_state = create_initial_state(user_input, session_id)

        # Create workflow
        if workflow_type == "instrument_identification":
            workflow = create_instrument_identification_workflow()
        else:
            workflow = create_procurement_workflow()

        # Compile with memory
        memory = MemorySaver()
        app = workflow.compile(checkpointer=memory)

        # Run workflow
        config = {"configurable": {"thread_id": session_id}}
        final_state = app.invoke(initial_state, config)

        return {
            "success": True,
            "response": final_state.get("response"),
            "response_data": final_state.get("response_data"),
            "intent": final_state.get("intent").value if final_state.get("intent") else None,
            "product_type": final_state.get("product_type"),
            "requires_user_input": final_state.get("requires_user_input", False),
            "current_step": final_state.get("current_step").value if final_state.get("current_step") else None,
            "error": final_state.get("error")
        }

    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        return {
            "success": False,
            "response": "I encountered an error processing your request. Please try again.",
            "error": str(e)
        }
