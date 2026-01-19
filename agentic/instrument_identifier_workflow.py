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

# Standards RAG enrichment for grounded identification
from .standards_rag.standards_rag_enrichment import enrich_identified_items_with_standards

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
    
    OPTIMIZATION: Runs instrument and accessory identification in PARALLEL
    to reduce overall latency.
    """
    logger.info("[IDENTIFIER] Node 2: Identifying instruments and accessories (PARALLEL)...")

    # Initialize empty lists
    state["identified_instruments"] = []
    state["identified_accessories"] = []
    state["project_name"] = "Untitled Project"

    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def identify_instruments_task():
        """Task to identify instruments."""
        try:
            return identify_instruments_tool.invoke({
                "requirements": state["user_input"]
            })
        except Exception as e:
            logger.error(f"[IDENTIFIER] Instrument identification exception: {e}")
            return {"success": False, "error": str(e), "instruments": []}
    
    def identify_accessories_task(instruments_list):
        """Task to identify accessories based on instruments."""
        if not instruments_list:
            return {"success": False, "accessories": [], "error": "No instruments to base accessories on"}
        try:
            return identify_accessories_tool.invoke({
                "instruments": instruments_list,
                "process_context": state["user_input"]
            })
        except Exception as e:
            logger.error(f"[IDENTIFIER] Accessory identification exception: {e}")
            return {"success": False, "error": str(e), "accessories": []}

    # First, identify instruments (can't parallelize this with accessories since accessories depend on instruments)
    inst_result = identify_instruments_task()
    
    if inst_result.get("success"):
        state["identified_instruments"] = inst_result.get("instruments", [])
        state["project_name"] = inst_result.get("project_name", "Untitled Project")
        logger.info(f"[IDENTIFIER] Successfully identified {len(state['identified_instruments'])} instruments")
    else:
        logger.warning(f"[IDENTIFIER] Instrument identification failed: {inst_result.get('error', 'Unknown error')}")

    # Now identify accessories based on identified instruments
    if state["identified_instruments"]:
        acc_result = identify_accessories_task(state["identified_instruments"])
        
        if acc_result.get("success"):
            state["identified_accessories"] = acc_result.get("accessories", [])
            logger.info(f"[IDENTIFIER] Successfully identified {len(state['identified_accessories'])} accessories")
        else:
            logger.warning(f"[IDENTIFIER] Accessory identification failed: {acc_result.get('error', 'Unknown error')}")

    # ENHANCED FALLBACK: If still no items found, use rule-based extraction
    if not state["identified_instruments"] and not state["identified_accessories"]:
        logger.warning("[IDENTIFIER] LLM-based identification failed, using rule-based fallback")
        
        # Simple keyword-based extraction
        user_input_lower = state["user_input"].lower()
        
        # Common instrument keywords
        instrument_keywords = {
            "temperature": ("Temperature Sensor/Transmitter", "Temperature Measurement"),
            "pressure": ("Pressure Transmitter", "Pressure Measurement"),
            "flow": ("Flow Meter", "Flow Measurement"),
            "level": ("Level Transmitter", "Level Measurement"),
            "valve": ("Control Valve", "Valves & Actuators"),
            "analyzer": ("Analyzer", "Analytical Instruments"),
            "transmitter": ("Transmitter", "General Instruments"),
            "sensor": ("Sensor", "General Instruments"),
            "thermocouple": ("Thermocouple", "Temperature Measurement"),
            "rtd": ("RTD Sensor", "Temperature Measurement"),
        }
        
        # Extract instruments based on keywords
        detected_instruments = []
        for keyword, (product_name, category) in instrument_keywords.items():
            if keyword in user_input_lower:
                detected_instruments.append({
                    "category": category,
                    "product_name": product_name,
                    "quantity": 1,
                    "specifications": {},
                    "sample_input": f"{product_name} for {state['user_input'][:100]}",
                    "strategy": "keyword_extraction"
                })
        
        # If still nothing, create a generic item
        if not detected_instruments:
            detected_instruments = [{
                "category": "Industrial Instrument",
                "product_name": "General Instrument",
                "quantity": 1,
                "specifications": {},
                "sample_input": state["user_input"][:200] if len(state["user_input"]) > 200 else state["user_input"],
                "strategy": "generic_fallback"
            }]
            logger.warning("[IDENTIFIER] Created generic fallback instrument")
        else:
            logger.info(f"[IDENTIFIER] Keyword extraction found {len(detected_instruments)} instruments")
        
        state["identified_instruments"] = detected_instruments
        state["project_name"] = "Keyword-Based Identification"

    # Build unified item list with sample_inputs (runs for ALL cases)
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
        
        # Smart category extraction: If category is generic "Accessories" or "Accessory",
        # extract the product type from accessory_name (e.g., "Thermowell for X" -> "Thermowell")
        raw_category = accessory.get("category", "Accessory")
        accessory_name = accessory.get("accessory_name", "Unknown Accessory")
        
        if raw_category.lower() in ["accessories", "accessory", ""]:
            # Extract product type from accessory name (before " for ")
            if " for " in accessory_name:
                extracted_type = accessory_name.split(" for ")[0].strip()
                smart_category = extracted_type if extracted_type else raw_category
            else:
                smart_category = accessory_name  # Use full name if no " for " pattern
        else:
            smart_category = raw_category

        all_items.append({
            "number": item_number,
            "type": "accessory",
            "name": accessory_name,
            "category": smart_category,  # Use smart category instead of raw
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

    return state


def _extract_spec_value(spec_value) -> str:
    """
    Extract ONLY the value from any specification format.
    
    Handles:
    - dict with 'value' key: {'value': 'x', 'confidence': 0.9} → 'x'
    - nested category dict: {'temp_range': {'value': 'y'}} → 'y' (for single key)
    - plain string: 'plain value' → 'plain value'
    - list: [{'value': 'a'}, {'value': 'b'}] → 'a, b'
    """
    if spec_value is None:
        return ""
    
    if isinstance(spec_value, dict):
        # Case 1: Direct value dict {'value': 'x', 'confidence': ...}
        if 'value' in spec_value:
            val = spec_value['value']
            # Handle double-nested case
            if isinstance(val, dict) and 'value' in val:
                val = val['value']
            return str(val) if val and str(val).lower() not in ['null', 'none', 'n/a'] else ""
        
        # Case 2: Category wrapper dict - will be handled by _flatten_nested_specs
        return ""
    
    if isinstance(spec_value, list):
        values = []
        for v in spec_value:
            extracted = _extract_spec_value(v)
            if extracted:
                values.append(extracted)
        return ", ".join(values) if values else ""
    
    # Plain string
    return str(spec_value) if spec_value else ""


def _flatten_nested_specs(spec_dict: dict) -> dict:
    """
    Flatten nested specification dictionaries into simple key-value pairs.
    
    Input:  {'environmental_specs': {'temperature_range': {'value': '200°C', 'confidence': 0.9, 'note': '...'}}}
    Output: {'temperature_range': '200°C'}
    
    Input:  {'accuracy': {'value': '±0.1%', 'confidence': 0.8}}
    Output: {'accuracy': '±0.1%'}
    """
    flattened = {}
    
    for key, value in spec_dict.items():
        if isinstance(value, dict):
            # Check if this is a value dict
            if 'value' in value:
                extracted = _extract_spec_value(value)
                if extracted:
                    flattened[key] = extracted
            else:
                # This is a category wrapper - flatten its contents
                for inner_key, inner_value in value.items():
                    if isinstance(inner_value, dict) and 'value' in inner_value:
                        extracted = _extract_spec_value(inner_value)
                        if extracted:
                            flattened[inner_key] = extracted
                    elif inner_value and str(inner_value).lower() not in ['null', 'none', '']:
                        flattened[inner_key] = str(inner_value)
        elif value and str(value).lower() not in ['null', 'none', '']:
            flattened[key] = str(value)
    
    return flattened


def _merge_deep_agent_specs_for_display(items: list) -> list:
    """
    Merge Deep Agent specifications into the specifications field for UI display.
    Labels values based on source:
    - Standards -> [STANDARDS]
    - LLM/Inferred -> [INFERRED]
    - User -> (No label)
    """
    import re
    
    # Patterns that indicate raw JSON structure - EXACT matches only for these
    RAW_JSON_PATTERNS = [
        "{'value':",
        "'confidence':",
        "'note':",
        "[{",
        "{'",
    ]
    
    # Patterns that indicate the ENTIRE value is invalid (use startswith/full match)
    INVALID_START_PATTERNS = [
        "i don't have",
        "i do not have",
        "no information",
        "not found in",
        "cannot determine",
        "not available in",
    ]
    
    # Placeholder values to skip (exact matches only)
    PLACEHOLDER_VALUES = [
        "null", "none", "n/a", "na", "not applicable",
        "extracted value or null", "consolidated value or null",
        "value if applicable", "not specified", ""
    ]
    
    for item in items:
        # Get Deep Agent specs if available
        combined_specs = item.get("combined_specifications", {})
        standards_specs = item.get("standards_specifications", {})
        llm_specs = item.get("llm_generated_specs", {})
        
        # Create clean specifications for display
        display_specs = {}
        
        # Strategy:
        # 1. If combined_specs exists (Unified path), use its 'source' field
        # 2. If separate specs exist, use implicit labeling
        
        if combined_specs:
            flattened = _flatten_nested_specs(combined_specs)
            
            for key, val in flattened.items():
                if not val: continue
                
                # Check source in original dict if possible, or flattened might have lost it
                # We need to look up the key in combined_specs to find source
                source_label = ""
                if key in combined_specs and isinstance(combined_specs[key], dict):
                    src = combined_specs[key].get("source", "")
                    if src == "standards":
                        source_label = " [STANDARDS]"
                    elif src == "llm_generated":
                        source_label = " [INFERRED]"
                
                # Helper to check validity
                val_str = str(val).strip()
                # Clean up any pre-existing tags
                val_str = val_str.replace("[INFERRED]", "").replace("[STANDARDS]", "").strip()
                val_lower = val_str.lower()
                
                if any(p in val_lower for p in RAW_JSON_PATTERNS): continue
                if any(val_lower.startswith(p) for p in INVALID_START_PATTERNS): continue
                if val_lower in PLACEHOLDER_VALUES: continue
                
                display_specs[key] = f"{val_str}{source_label}"

        elif standards_specs:
            # Fallback: All are standards
            flattened = _flatten_nested_specs(standards_specs)
            for key, val in flattened.items():
                if not val: continue
                if str(val).lower() in PLACEHOLDER_VALUES: continue
                display_specs[key] = f"{str(val)} [STANDARDS]"
                
        elif llm_specs:
            # Fallback: All are inferred
            flattened = _flatten_nested_specs(llm_specs)
            for key, val in flattened.items():
                if not val: continue
                if str(val).lower() in PLACEHOLDER_VALUES: continue
                display_specs[key] = f"{str(val)} [INFERRED]"
        
        # Merge with any existing specs (User specs usually here)
        original_specs = item.get("specifications", {})
        if isinstance(original_specs, dict):
             # Original specs typically come from "Identify Instrument" prompt which are "User/Inferred"
             # But if they clash with display_specs (which came from enrichment), ENRICHMENT wins for display usually?
             # Actually, user specs should ALREADY be in combined_specs if enrichment ran.
             # So we only add original specs if they are NOT in display_specs.
             
             for key, val in original_specs.items():
                 # Key normalization for comparison
                 key_clean = key.lower().replace("_", "").replace(" ", "")
                 existing_keys = [k.lower().replace("_", "").replace(" ", "") for k in display_specs.keys()]
                 
                 if key_clean not in existing_keys:
                     val_str = str(val).strip()
                     # Clean up any pre-existing tags to avoid double labeling
                     val_str = val_str.replace("[INFERRED]", "").replace("[STANDARDS]", "").strip()
                     
                     if val_str and val_str.lower() not in PLACEHOLDER_VALUES:
                         display_specs[key] = val_str # No label for user/original specs
        
        # Update item
        if display_specs:
            item["specifications"] = display_specs
    
    return items




def format_selection_list_node(state: InstrumentIdentifierState) -> InstrumentIdentifierState:
    """
    Node 3: Format Selection List.
    Formats identified items into user-friendly numbered list with sample_inputs.
    This is the FINAL node - workflow ends here and waits for user selection.
    """
    logger.info("[IDENTIFIER] Node 3: Formatting selection list...")

    # Defensive check: ensure all_items exists
    if "all_items" not in state or not state["all_items"]:
        logger.warning("[IDENTIFIER] No items found, creating fallback response")
        state["response"] = "I couldn't identify any instruments or accessories. Please provide more details about your requirements."
        state["response_data"] = {
            "workflow": "instrument_identifier",
            "items": [],
            "total_items": 0,
            "awaiting_selection": False,
            "error": state.get("error", "No items identified")
        }
        state["current_step"] = "complete"
        return state

    # === DEBUG: Print specification source analysis ===
    try:
        from debug_spec_tracker import print_specification_report
        if state.get("all_items"):
            print_specification_report(state["all_items"])
    except Exception as debug_err:
        logger.debug(f"[IDENTIFIER] Debug tracker not available: {debug_err}")
    # === END DEBUG ===

    try:
        llm = create_llm_with_fallback(
            model="gemini-2.5-flash",
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
        
        # === MERGE DEEP AGENT SPECS FOR UI DISPLAY ===
        # This ensures Deep Agent specifications are shown in UI with [STANDARDS] labels
        _merge_deep_agent_specs_for_display(state["all_items"])
        logger.info(f"[IDENTIFIER] Merged Deep Agent specs for {len(state['all_items'])} items")
        # === END MERGE ===
        
        state["response_data"] = {
            "workflow": "instrument_identifier",
            "project_name": state["project_name"],
            "items": state["all_items"],  # Full items with merged specs
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
# STANDARDS RAG ENRICHMENT NODE (BATCH OPTIMIZED)
# ============================================================================

def enrich_with_standards_node(state: InstrumentIdentifierState) -> InstrumentIdentifierState:
    """
    Node 4: Parallel 3-Source Specification Enrichment.
    
    Runs 3 specification sources in PARALLEL:
    1. USER_SPECIFIED: Extract explicit specs from user input (MANDATORY)
    2. LLM_GENERATED: Generate specs for each product type
    3. STANDARDS_BASED: Extract from standards documents
    
    All results stored in Deep Agent Memory, deduplicated, and merged.
    
    SPECIFICATION PRIORITY:
    - User-specified specs are MANDATORY (never overwritten)
    - Standards specs take precedence over LLM for conflicts
    - Non-conflicting specs from all sources are included
    
    Adds to each item:
    - user_specified_specs: Explicit specs from user (MANDATORY)
    - llm_generated_specs: LLM-generated specs for product type
    - standards_specifications: Specs from standards documents
    - combined_specifications: Final merged specs with source metadata
    - specifications: Flattened values for backward compatibility
    """
    logger.info("[IDENTIFIER] Node 4: Running PARALLEL 3-Source Specification Enrichment...")
    
    all_items = state.get("all_items", [])
    user_input = state.get("user_input", "")
    
    if not all_items:
        logger.warning("[IDENTIFIER] No items to enrich")
        return state
    
    try:
        # Import OPTIMIZED Parallel Enrichment (uses shared LLM + true parallel products)
        from agentic.deep_agent.optimized_parallel_agent import run_optimized_parallel_enrichment
        
        logger.info(f"[IDENTIFIER] Starting OPTIMIZED parallel enrichment for {len(all_items)} items...")
        
        # =====================================================
        # OPTIMIZED PARALLEL ENRICHMENT
        # - Single shared LLM instance (no repeated test calls)
        # - All products processed in parallel
        # - Significant time savings vs sequential processing
        # =====================================================
        result = run_optimized_parallel_enrichment(
            items=all_items,
            user_input=user_input,
            session_id=state.get("session_id", "identifier-opt-parallel"),
            domain_context=None,
            safety_requirements=None,
            max_parallel_products=5  # Process up to 5 products simultaneously
        )
        
        if result.get("success"):
            enriched_items = result.get("items", all_items)
            metadata = result.get("metadata", {})
            
            processing_time = metadata.get("processing_time_ms", 0)
            thread_results = metadata.get("thread_results", {})
            
            logger.info(f"[IDENTIFIER] Parallel enrichment complete in {processing_time}ms")
            logger.info(f"[IDENTIFIER] Thread results: {thread_results}")
            
            # Extract applicable standards and certifications for each item
            for item in enriched_items:
                if item.get("standards_info", {}).get("enrichment_status") == "success":
                    # Add extracted standards and certifications
                    standards_specs = item.get("standards_specifications", {})
                    item["standards_info"]["applicable_standards"] = _extract_standards_from_specs(
                        {"specifications": standards_specs}
                    )
                    item["standards_info"]["certifications"] = _extract_certifications_from_specs(
                        {"specifications": standards_specs}
                    )
            
            state["all_items"] = enriched_items
            
            # Count spec sources
            total_user_specs = sum(len(item.get("user_specified_specs", {})) for item in enriched_items)
            total_llm_specs = sum(len(item.get("llm_generated_specs", {})) for item in enriched_items)
            total_std_specs = sum(len(item.get("standards_specifications", {})) for item in enriched_items)
            
            state["messages"] = state.get("messages", []) + [{
                "role": "system",
                "content": f"Parallel 3-source enrichment complete: {total_user_specs} user specs (MANDATORY), {total_llm_specs} LLM specs, {total_std_specs} standards specs in {processing_time}ms"
            }]
            
            logger.info(f"[IDENTIFIER] Spec breakdown: {total_user_specs} user, {total_llm_specs} LLM, {total_std_specs} standards")
            
        else:
            error = result.get("error", "Unknown error")
            logger.error(f"[IDENTIFIER] Parallel enrichment failed: {error}")
            state["all_items"] = result.get("items", all_items)
            
            state["messages"] = state.get("messages", []) + [{
                "role": "system",
                "content": f"Parallel enrichment failed (non-critical): {error}"
            }]
        
    except ImportError as ie:
        logger.error(f"[IDENTIFIER] Parallel enrichment import failed: {ie}")
        # Fallback to existing Standards RAG enrichment
        try:
            enriched_items = enrich_identified_items_with_standards(
                items=all_items,
                product_type=None,
                domain=None,
                top_k=3
            )
            state["all_items"] = enriched_items
            logger.info("[IDENTIFIER] Fallback to existing Standards RAG enrichment")
        except Exception as e2:
            logger.error(f"[IDENTIFIER] Fallback enrichment also failed: {e2}")
            
    except Exception as e:
        logger.error(f"[IDENTIFIER] Specification enrichment failed: {e}")
        import traceback
        traceback.print_exc()
        state["messages"] = state.get("messages", []) + [{
            "role": "system",
            "content": f"Specification enrichment failed (non-critical): {str(e)}"
        }]
    
    state["current_step"] = "format_list"
    return state


def _extract_standards_from_specs(final_specs: dict) -> list:
    """Extract applicable standards codes from Deep Agent results."""
    standards = []
    constraints = final_specs.get("constraints_applied", [])
    for c in constraints:
        if isinstance(c, dict):
            ref = c.get("standard_reference") or c.get("constraint", "")
            if ref:
                # Extract standard codes like IEC, ISO, API
                import re
                matches = re.findall(r'\b(IEC|ISO|API|ANSI|ISA|EN|NFPA|ASME|IEEE)\s*[\d.-]+', str(ref), re.IGNORECASE)
                standards.extend(matches)
    return list(set(standards))


def _extract_certifications_from_specs(final_specs: dict) -> list:
    """Extract certifications from Deep Agent results."""
    certs = []
    specs = final_specs.get("specifications", {})
    for key, value in specs.items():
        key_lower = key.lower()
        value_str = str(value).upper()
        if any(x in key_lower for x in ["cert", "sil", "atex", "rating", "approval"]):
            certs.append(f"{key}: {value}")
        # Also check for certification patterns in values
        import re
        cert_patterns = re.findall(r'\b(SIL\s*[1-4]|ATEX|IECEx|CE|UL|CSA|FM|IP\d{2})', value_str)
        certs.extend(cert_patterns)
    return list(set(certs))


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

    # Add 4 nodes (including Standards RAG enrichment)
    workflow.add_node("classify_intent", classify_initial_intent_node)
    workflow.add_node("identify_items", identify_instruments_and_accessories_node)
    workflow.add_node("enrich_with_standards", enrich_with_standards_node)  # NEW: Standards RAG
    workflow.add_node("format_list", format_selection_list_node)

    # Set entry point
    workflow.set_entry_point("classify_intent")

    # Add edges - linear flow with Standards RAG enrichment
    workflow.add_edge("classify_intent", "identify_items")
    workflow.add_edge("identify_items", "enrich_with_standards")  # NEW: Route to Standards RAG
    workflow.add_edge("enrich_with_standards", "format_list")  # NEW: Then to formatting
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
