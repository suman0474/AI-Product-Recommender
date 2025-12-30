# agentic/instrument_identifier_workflow.py
# Instrument Identifier Workflow - List Generator Only
# This workflow identifies instruments/accessories and generates a selection list.
# It does NOT perform product search - that's handled by the SOLUTION workflow.

import json
import logging
from typing import Dict, Any
from langgraph.graph import StateGraph, END

from .models import (
    InstrumentIdentifierState,
    create_instrument_identifier_state
)
from .checkpointing import compile_with_checkpointing

from tools.intent_tools import classify_intent_tool
from tools.instrument_tools import identify_instruments_tool, identify_accessories_tool

import os
from dotenv import load_dotenv
from llm_fallback import create_llm_with_fallback
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

Create a numbered list for user selection. Each item should have:
- number: Sequential number (1, 2, 3, ...)
- type: "instrument" or "accessory"
- name: Product name
- category: Product category
- quantity: How many needed
- key_specs: Brief specification summary (1 line)

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
    "message": "Please select an item number to search for products."
}}
"""


# ============================================================================
# WORKFLOW NODES
# ============================================================================

def classify_initial_intent_node(state: InstrumentIdentifierState) -> InstrumentIdentifierState:
    """
    Node 1: Initial Intent Classification.
    Determines if this is an instrument/accessory identification request.
    """
    logger.info("[IDENTIFIER] Node 1: Initial intent classification...")

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

        logger.info(f"[IDENTIFIER] Initial intent: {state['initial_intent']}")

    except Exception as e:
        logger.error(f"[IDENTIFIER] Initial intent classification failed: {e}")
        state["error"] = str(e)

    return state


def identify_instruments_and_accessories_node(state: InstrumentIdentifierState) -> InstrumentIdentifierState:
    """
    Node 2: Instrument/Accessory Identifier.
    Identifies ALL instruments and accessories from user requirements.
    Generates sample_input for each item.
    """
    logger.info("[IDENTIFIER] Node 2: Identifying instruments and accessories...")

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
            state["identified_instruments"] = [{
                "category": "Industrial Instrument",
                "product_name": "General Instrument",
                "quantity": 1,
                "specifications": {},
                "sample_input": "Industrial instrument based on requirements"
            }]

        # Build unified item list with sample_inputs
        all_items = []
        item_number = 1

        # Add instruments
        for instrument in state["identified_instruments"]:
            all_items.append({
                "number": item_number,
                "type": "instrument",
                "name": instrument.get("product_name", "Unknown Instrument"),
                "category": instrument.get("category", "Instrument"),
                "quantity": instrument.get("quantity", 1),
                "specifications": instrument.get("specifications", {}),
                "sample_input": instrument.get("sample_input", ""),  # KEY FIELD
                "strategy": instrument.get("strategy", "")
            })
            item_number += 1

        # Add accessories
        for accessory in state["identified_accessories"]:
            # Construct sample_input for accessories
            acc_sample_input = f"{accessory.get('category', 'Accessory')} for {accessory.get('related_instrument', 'instruments')}"

            all_items.append({
                "number": item_number,
                "type": "accessory",
                "name": accessory.get("accessory_name", "Unknown Accessory"),
                "category": accessory.get("category", "Accessory"),
                "quantity": accessory.get("quantity", 1),
                "sample_input": acc_sample_input,
                "related_instrument": accessory.get("related_instrument", "")
            })
            item_number += 1

        state["all_items"] = all_items
        state["total_items"] = len(all_items)

        state["current_step"] = "format_list"

        total_items = len(state["identified_instruments"]) + len(state["identified_accessories"])

        state["messages"] = state.get("messages", []) + [{
            "role": "system",
            "content": f"Identified {len(state['identified_instruments'])} instruments and {len(state['identified_accessories'])} accessories"
        }]

        logger.info(f"[IDENTIFIER] Found {total_items} items total")

    except Exception as e:
        logger.error(f"[IDENTIFIER] Identification failed: {e}")
        state["error"] = str(e)

    return state


def format_selection_list_node(state: InstrumentIdentifierState) -> InstrumentIdentifierState:
    """
    Node 3: Format Selection List.
    Formats identified items into user-friendly numbered list with sample_inputs.
    This is the FINAL node - workflow ends here and waits for user selection.
    """
    logger.info("[IDENTIFIER] Node 3: Formatting selection list...")

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
        response_lines = [
            f"**📋 {state.get('project_name', 'Project')} - Identified Items**\n",
            f"I've identified **{state['total_items']} items** for your project:\n"
        ]

        formatted_list = result.get("formatted_list", [])

        for item in formatted_list:
            emoji = "🔧" if item.get("type") == "instrument" else "🔩"
            response_lines.append(
                f"{item['number']}. {emoji} **{item.get('name', 'Unknown')}** ({item.get('category', '')})"
            )
            response_lines.append(
                f"   Quantity: {item.get('quantity', 1)}"
            )

            # Show sample_input preview (first 80 chars)
            # Find the actual item from all_items to get the full sample_input
            actual_item = next((i for i in state["all_items"] if i["number"] == item["number"]), None)
            if actual_item and actual_item.get("sample_input"):
                sample_preview = actual_item["sample_input"][:80] + "..." if len(actual_item["sample_input"]) > 80 else actual_item["sample_input"]
                response_lines.append(
                    f"   🔍 Search query: {sample_preview}"
                )

            response_lines.append("")

        response_lines.append(
            f"\n**📌 Next Steps:**\n"
            f"Reply with an item number (1-{state['total_items']}) to search for vendor products.\n"
            f"I'll then find specific product recommendations for your selected item."
        )

        state["response"] = "\n".join(response_lines)
        state["response_data"] = {
            "workflow": "instrument_identifier",
            "project_name": state["project_name"],
            "items": state["all_items"],  # Full items with sample_inputs
            "total_items": state["total_items"],
            "awaiting_selection": True,
            "instructions": f"Reply with item number (1-{state['total_items']}) to get product recommendations"
        }

        state["current_step"] = "complete"

        logger.info(f"[IDENTIFIER] Selection list formatted with {state['total_items']} items")

    except Exception as e:
        logger.error(f"[IDENTIFIER] List formatting failed: {e}")
        state["error"] = str(e)

        # Fallback: Create simple text list
        response_lines = [
            f"**📋 {state.get('project_name', 'Project')} - Identified Items**\n"
        ]

        for item in state["all_items"]:
            emoji = "🔧" if item.get("type") == "instrument" else "🔩"
            response_lines.append(
                f"{item['number']}. {emoji} {item.get('name', 'Unknown')} ({item.get('category', '')})"
            )

        response_lines.append(
            f"\nReply with an item number (1-{state['total_items']}) to search for products."
        )

        state["response"] = "\n".join(response_lines)
        state["response_data"] = {
            "workflow": "instrument_identifier",
            "items": state["all_items"],
            "total_items": state["total_items"],
            "awaiting_selection": True
        }

    return state


# ============================================================================
# WORKFLOW CREATION
# ============================================================================

def create_instrument_identifier_workflow() -> StateGraph:
    """
    Create the Instrument Identifier Workflow.

    This is a simplified 3-node workflow that ONLY identifies instruments/accessories
    and presents them for user selection. It does NOT perform product search.

    Flow:
    1. Initial Intent Classification
    2. Instrument/Accessory Identification (with sample_input generation)
    3. Format Selection List

    After this workflow completes, user selects an item, and the sample_input
    is routed to the SOLUTION workflow for product search.
    """

    workflow = StateGraph(InstrumentIdentifierState)

    # Add 3 nodes
    workflow.add_node("classify_intent", classify_initial_intent_node)
    workflow.add_node("identify_items", identify_instruments_and_accessories_node)
    workflow.add_node("format_list", format_selection_list_node)

    # Set entry point
    workflow.set_entry_point("classify_intent")

    # Add edges - linear flow, ends after formatting
    workflow.add_edge("classify_intent", "identify_items")
    workflow.add_edge("identify_items", "format_list")
    workflow.add_edge("format_list", END)  # WORKFLOW ENDS HERE - waits for user selection

    return workflow


# ============================================================================
# WORKFLOW EXECUTION
# ============================================================================

def run_instrument_identifier_workflow(
    user_input: str,
    session_id: str = "default",
    checkpointing_backend: str = "memory"
) -> Dict[str, Any]:
    """
    Run the instrument identifier workflow.

    This workflow identifies instruments/accessories and returns a selection list.
    It does NOT perform product search.

    Args:
        user_input: User's project requirements (e.g., "I need instruments for crude oil refinery")
        session_id: Session identifier
        checkpointing_backend: Backend for state persistence

    Returns:
        {
            "response": "Formatted selection list",
            "response_data": {
                "workflow": "instrument_identifier",
                "project_name": "...",
                "items": [
                    {
                        "number": 1,
                        "type": "instrument",
                        "name": "...",
                        "sample_input": "..."  # To be used for product search
                    },
                    ...
                ],
                "total_items": N,
                "awaiting_selection": True
            }
        }
    """
    try:
        logger.info(f"[IDENTIFIER] Starting workflow for session: {session_id}")
        logger.info(f"[IDENTIFIER] User input: {user_input[:100]}...")

        # Create initial state
        initial_state = create_instrument_identifier_state(user_input, session_id)

        # Create and compile workflow
        workflow = create_instrument_identifier_workflow()
        compiled = compile_with_checkpointing(workflow, checkpointing_backend)

        # Execute workflow
        result = compiled.invoke(
            initial_state,
            config={"configurable": {"thread_id": session_id}}
        )

        logger.info(f"[IDENTIFIER] Workflow completed successfully")
        logger.info(f"[IDENTIFIER] Generated {result.get('total_items', 0)} items for selection")

        return {
            "response": result.get("response", ""),
            "response_data": result.get("response_data", {})
        }

    except Exception as e:
        logger.error(f"[IDENTIFIER] Workflow execution failed: {e}")
        import traceback
        traceback.print_exc()

        return {
            "response": f"I encountered an error while identifying instruments: {str(e)}",
            "response_data": {
                "workflow": "instrument_identifier",
                "error": str(e),
                "awaiting_selection": False
            }
        }


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'InstrumentIdentifierState',
    'create_instrument_identifier_workflow',
    'run_instrument_identifier_workflow'
]
