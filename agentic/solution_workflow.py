# agentic/solution_workflow.py
# =============================================================================
# SOLUTION WORKFLOW - Instrument Identification for Solutions
# =============================================================================
#
# PURPOSE: Analyze solution descriptions and identify instruments/accessories.
# User describes a solution → Identify instruments/accessories → Generate 
# sample_input for each → User selects → Product Search Workflow
#
# This workflow works SIMILARLY to instrument_identifier_workflow.py but
# focuses on high-level solution descriptions rather than direct requirements.
#
# FLOW:
#   User: "I'm designing a crude oil distillation unit..."
#   ↓
#   1. Analyze Solution Context (extract domain, process, industry)
#   2. Identify Instruments & Accessories (with sample_input for each)
#   3. Format Selection List (await user selection)
#   ↓
#   User selects item → sample_input → Product Search Workflow
#
# =============================================================================

import json
import logging
from typing import Dict, Any, List, Optional, Callable
from langgraph.graph import StateGraph, END

from .models import (
    SolutionState,
    create_solution_state
)
from .checkpointing import compile_with_checkpointing

from tools.intent_tools import classify_intent_tool
from tools.instrument_tools import identify_instruments_tool, identify_accessories_tool

# Standards RAG enrichment for grounded identification
from .standards_rag.standards_rag_enrichment import (
    enrich_identified_items_with_standards,
    validate_items_against_domain_standards
)

import os
from dotenv import load_dotenv
from llm_fallback import create_llm_with_fallback
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Import lock utilities
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.workflow_sync import with_workflow_lock

load_dotenv()
logger = logging.getLogger(__name__)


# =============================================================================
# PROMPTS
# =============================================================================

SOLUTION_ANALYSIS_PROMPT = """
You are Engenie's Solution Analyzer. Analyze the user's solution description and extract:
1. Industry domain (Oil & Gas, Chemical, Pharma, etc.)
2. Process type (Distillation, Refining, Manufacturing, etc.)
3. Key operational parameters (temperature, pressure, flow rates, etc.)
4. Safety requirements (SIL levels, hazardous areas, etc.)
5. Environmental conditions

User's Solution Description:
{solution_description}

Return ONLY valid JSON:
{{
    "solution_name": "<descriptive name for the solution>",
    "industry": "<industry domain>",
    "process_type": "<type of process>",
    "key_parameters": {{
        "temperature_range": "<if mentioned>",
        "pressure_range": "<if mentioned>",
        "flow_rates": "<if mentioned>",
        "materials": ["<materials handled>"]
    }},
    "safety_requirements": {{
        "sil_level": "<SIL level if mentioned>",
        "atex_zone": "<ATEX zone if mentioned>",
        "hazardous_area": <true/false>
    }},
    "environmental": {{
        "location": "<indoor/outdoor>",
        "conditions": "<special conditions>"
    }},
    "context_for_instruments": "<summary context to help identify instruments>"
}}
"""

SOLUTION_INSTRUMENT_LIST_PROMPT = """
You are Engenie's Solution Instrument Presenter. Format the identified instruments and 
accessories for user selection, specifically tailored for the solution context.

Solution Analysis:
{solution_analysis}

Identified Instruments:
{instruments}

Identified Accessories:
{accessories}

Create a numbered list for user selection. Each item should have:
- number: Sequential number (1, 2, 3, ...)
- type: "instrument" or "accessory"
- name: Product name
- category: Product category  
- quantity: How many needed
- purpose: Why this is needed for the solution
- key_specs: Brief specification summary based on solution requirements

Return ONLY valid JSON:
{{
    "formatted_list": [
        {{
            "number": 1,
            "type": "instrument" | "accessory",
            "name": "<product name>",
            "category": "<category>",
            "quantity": <quantity>,
            "purpose": "<why needed for this solution>",
            "key_specs": "<specs derived from solution context>"
        }}
    ],
    "total_items": <count>,
    "solution_summary": "<brief summary of the solution and its instrument needs>"
}}
"""


# =============================================================================
# WORKFLOW NODES
# =============================================================================

def analyze_solution_node(state: SolutionState) -> SolutionState:
    """
    Node 1: Analyze Solution Context.
    Extracts domain, process type, parameters, and safety requirements
    from the user's solution description.
    """
    logger.info("[SOLUTION] Node 1: Analyzing solution context...")

    try:
        llm = create_llm_with_fallback(
            model="gemini-2.5-flash",
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        prompt = ChatPromptTemplate.from_template(SOLUTION_ANALYSIS_PROMPT)
        parser = JsonOutputParser()
        chain = prompt | llm | parser

        result = chain.invoke({
            "solution_description": state["user_input"]
        })

        # Store solution analysis
        state["solution_analysis"] = result
        state["solution_name"] = result.get("solution_name", "Industrial Solution")

        # Extract key context for instrument identification
        context = result.get("context_for_instruments", state["user_input"])
        state["instrument_context"] = context

        # Extract safety requirements
        safety = result.get("safety_requirements", {})
        if safety.get("sil_level"):
            state["provided_requirements"]["sil_level"] = safety["sil_level"]
        if safety.get("hazardous_area"):
            state["provided_requirements"]["hazardous_area"] = True

        state["current_step"] = "identify_instruments"

        state["messages"] = state.get("messages", []) + [{
            "role": "system",
            "content": f"Solution analyzed: {state['solution_name']} ({result.get('industry', 'Unknown')})"
        }]

        logger.info(f"[SOLUTION] Solution name: {state['solution_name']}")

    except Exception as e:
        logger.error(f"[SOLUTION] Solution analysis failed: {e}")
        state["error"] = str(e)
        state["solution_analysis"] = {}
        state["solution_name"] = "Solution"
        state["instrument_context"] = state["user_input"]

    return state


def identify_solution_instruments_node(state: SolutionState) -> SolutionState:
    """
    Node 2: Identify Instruments & Accessories for the Solution.
    Uses the solution context to identify all required instruments and accessories.
    Generates sample_input for each item.
    
    OPTIMIZATION: Uses helper functions with clean error handling for identification.
    """
    logger.info("[SOLUTION] Node 2: Identifying instruments and accessories...")

    def identify_instruments_task(context):
        """Task to identify instruments."""
        try:
            return identify_instruments_tool.invoke({
                "requirements": context
            })
        except Exception as e:
            logger.error(f"[SOLUTION] Instrument identification exception: {e}")
            return {"success": False, "error": str(e), "instruments": []}
    
    def identify_accessories_task(instruments_list, context):
        """Task to identify accessories based on instruments."""
        if not instruments_list:
            return {"success": False, "accessories": [], "error": "No instruments"}
        try:
            return identify_accessories_tool.invoke({
                "instruments": instruments_list,
                "process_context": context
            })
        except Exception as e:
            logger.error(f"[SOLUTION] Accessory identification exception: {e}")
            return {"success": False, "error": str(e), "accessories": []}

    try:
        # Use the enriched context from solution analysis
        context = state.get("instrument_context", state["user_input"])
        solution_analysis = state.get("solution_analysis", {})

        # REVERTED FIX #2 - Bug in parallel implementation broke instrument identification
        # Sequential approach is reliable and only costs 2-3 seconds
        logger.info("[SOLUTION] Node 2: Identifying instruments and accessories...")

        # Identify instruments
        inst_result = identify_instruments_task(context)

        if inst_result.get("success"):
            instruments = inst_result.get("instruments", [])

            # Enrich instruments with solution context
            for inst in instruments:
                # Add solution-specific details to sample_input
                sample_input = inst.get("sample_input", "")
                if solution_analysis.get("safety_requirements", {}).get("sil_level"):
                    sample_input += f" with {solution_analysis['safety_requirements']['sil_level']} rating"
                if solution_analysis.get("key_parameters", {}).get("temperature_range"):
                    sample_input += f" for {solution_analysis['key_parameters']['temperature_range']} temperature"
                inst["sample_input"] = sample_input
                inst["solution_purpose"] = f"Required for {state.get('solution_name', 'solution')}"

            state["identified_instruments"] = instruments
            state["project_name"] = state.get("solution_name", "Industrial Solution")
        else:
            state["identified_instruments"] = []

        # Identify accessories
        acc_result = identify_accessories_task(state["identified_instruments"], context)

        if acc_result.get("success"):
            accessories = acc_result.get("accessories", [])

            # Enrich accessories with solution context
            for acc in accessories:
                acc_sample_input = f"{acc.get('category', 'Accessory')} for {acc.get('related_instrument', 'instruments')}"
                if solution_analysis.get("safety_requirements", {}).get("hazardous_area"):
                    acc_sample_input += " (explosion-proof required)"
                acc["sample_input"] = acc_sample_input

            state["identified_accessories"] = accessories
        else:
            state["identified_accessories"] = []

        # If no items found, create defaults based on solution
        if not state["identified_instruments"] and not state["identified_accessories"]:
            industry = solution_analysis.get("industry", "Industrial")
            process_type = solution_analysis.get("process_type", "process")
            
            state["identified_instruments"] = [{
                "category": "Process Instrument",
                "product_name": f"{industry} Instrument",
                "quantity": 1,
                "specifications": {},
                "sample_input": f"Industrial instrument for {process_type} in {industry}",
                "solution_purpose": f"General instrumentation for {state.get('solution_name', 'solution')}"
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
                "purpose": instrument.get("solution_purpose", ""),
                "strategy": instrument.get("strategy", "")
            })
            item_number += 1

        # Add accessories
        for accessory in state["identified_accessories"]:
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
                "sample_input": accessory.get("sample_input", ""),
                "purpose": f"Supports {accessory.get('related_instrument', 'instruments')}",
                "related_instrument": accessory.get("related_instrument", "")
            })
            item_number += 1

        state["all_items"] = all_items
        state["total_items"] = len(all_items)

        state["current_step"] = "format_list"

        total_items = len(state["identified_instruments"]) + len(state["identified_accessories"])

        state["messages"] = state.get("messages", []) + [{
            "role": "system",
            "content": f"Identified {len(state['identified_instruments'])} instruments and {len(state['identified_accessories'])} accessories for solution"
        }]

        logger.info(f"[SOLUTION] Found {total_items} items for solution")

    except Exception as e:
        logger.error(f"[SOLUTION] Instrument identification failed: {e}")
        state["error"] = str(e)

    return state


# =============================================================================
# STANDARDS RAG ENRICHMENT NODE (BATCH OPTIMIZED)
# =============================================================================

def enrich_solution_with_standards_node(state: SolutionState) -> SolutionState:
    """
    Node 3: Parallel 3-Source Specification Enrichment.
    
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
    logger.info("[SOLUTION] Node 3: Running PARALLEL 3-Source Specification Enrichment...")
    
    all_items = state.get("all_items", [])
    solution_analysis = state.get("solution_analysis", {})
    user_input = state.get("user_input", "")
    
    if not all_items:
        logger.warning("[SOLUTION] No items to enrich")
        state["current_step"] = "format_list"
        return state
    
    try:
        # Import OPTIMIZED Parallel Enrichment (uses shared LLM + true parallel products)
        from agentic.deep_agent.optimized_parallel_agent import run_optimized_parallel_enrichment
        
        # Get domain and safety context from solution analysis
        domain = solution_analysis.get("industry", "")
        safety_requirements = solution_analysis.get("safety_requirements", {})
        
        logger.info(f"[SOLUTION] Starting OPTIMIZED parallel enrichment for {len(all_items)} items...")

        # =====================================================
        # 🔥 FIX #5: ITEM DEDUPLICATION (200+ seconds saved!)
        # =====================================================
        # Problem: Same items appear multiple times and are enriched separately
        # Solution: Group by category, enrich unique once, copy to duplicates

        # Group items by category to find duplicates
        items_by_category = {}
        item_to_group = {}  # Map each item number to its category group

        for item in all_items:
            # Use category+name as unique key to identify duplicate items
            item_key = (item.get("category", "unknown"), item.get("name", "unknown"))

            if item_key not in items_by_category:
                items_by_category[item_key] = []

            items_by_category[item_key].append(item)
            item_to_group[item.get("number")] = item_key

        # Check if we found duplicates
        unique_count = len(items_by_category)
        duplicate_count = len(all_items) - unique_count

        if duplicate_count > 0:
            logger.info(f"[SOLUTION] 🔥 FIX #5: Deduplication found {duplicate_count} duplicates")
            logger.info(f"[SOLUTION]    {len(all_items)} total items → {unique_count} unique types")

            # Create list of unique items (first from each group)
            unique_items = []
            unique_mapping = {}  # Maps category to first enriched item

            for item_key, items_in_group in items_by_category.items():
                first_item = items_in_group[0]
                unique_items.append(first_item)
                unique_mapping[item_key] = first_item

            logger.info(f"[SOLUTION]    Enriching only {unique_count} unique items...")

            # Enrich only unique items
            result = run_optimized_parallel_enrichment(
                items=unique_items,
                user_input=user_input,
                session_id=state.get("session_id", "solution-opt-parallel"),
                domain_context=domain,
                safety_requirements=safety_requirements,
                max_parallel_products=5  # Process up to 5 products simultaneously
            )

            # If enrichment succeeded, apply results to all items (including duplicates)
            if result.get("success"):
                enriched_unique = result.get("items", unique_items)

                # Map enriched results back by category
                enriched_by_category = {}
                for enriched_item in enriched_unique:
                    item_key = (enriched_item.get("category", "unknown"), enriched_item.get("name", "unknown"))
                    enriched_by_category[item_key] = enriched_item

                # Apply enrichment to ALL items (including duplicates)
                enriched_items = []
                for original_item in all_items:
                    item_key = item_to_group.get(original_item.get("number"))

                    if item_key and item_key in enriched_by_category:
                        # Copy enrichment from unique item
                        enriched_copy = enriched_by_category[item_key].copy()

                        # Preserve original item's metadata (number, original name, etc)
                        enriched_copy["number"] = original_item.get("number")
                        enriched_copy["name"] = original_item.get("name") or enriched_copy.get("name")

                        enriched_items.append(enriched_copy)
                    else:
                        # No enrichment for this item - use original
                        enriched_items.append(original_item)

                # Update result to use all enriched items (including duplicates)
                result["items"] = enriched_items
                logger.info(f"[SOLUTION]    ✅ Applied enrichment to all {len(enriched_items)} items (including {duplicate_count} duplicates)")
        else:
            # No duplicates - enrich all items normally
            logger.info(f"[SOLUTION]    No duplicates found - all {unique_count} items are unique")
            result = run_optimized_parallel_enrichment(
                items=all_items,
                user_input=user_input,
                session_id=state.get("session_id", "solution-opt-parallel"),
                domain_context=domain,
                safety_requirements=safety_requirements,
                max_parallel_products=5  # Process up to 5 products simultaneously
            )
        
        if result.get("success"):
            enriched_items = result.get("items", all_items)
            metadata = result.get("metadata", {})
            
            processing_time = metadata.get("processing_time_ms", 0)
            thread_results = metadata.get("thread_results", {})
            
            logger.info(f"[SOLUTION] Parallel enrichment complete in {processing_time}ms")
            logger.info(f"[SOLUTION] Thread results: {thread_results}")
            
            # Extract applicable standards and certifications for each item
            for item in enriched_items:
                if item.get("standards_info", {}).get("enrichment_status") == "success":
                    # Add extracted standards and certifications
                    standards_specs = item.get("standards_specifications", {})
                    item["standards_info"]["applicable_standards"] = _extract_standards_from_solution_specs(
                        {"specifications": standards_specs}
                    )
                    item["standards_info"]["certifications"] = _extract_certifications_from_solution_specs(
                        {"specifications": standards_specs}
                    )
                    item["standards_info"]["domain"] = domain
            
            state["all_items"] = enriched_items
            
            # Count spec sources
            total_user_specs = sum(len(item.get("user_specified_specs", {})) for item in enriched_items)
            total_llm_specs = sum(len(item.get("llm_generated_specs", {})) for item in enriched_items)
            total_std_specs = sum(len(item.get("standards_specifications", {})) for item in enriched_items)
            
            state["messages"] = state.get("messages", []) + [{
                "role": "system",
                "content": f"Parallel 3-source enrichment complete: {total_user_specs} user specs (MANDATORY), {total_llm_specs} LLM specs, {total_std_specs} standards specs in {processing_time}ms"
            }]
            
            logger.info(f"[SOLUTION] Spec breakdown: {total_user_specs} user, {total_llm_specs} LLM, {total_std_specs} standards")
            
        else:
            error = result.get("error", "Unknown error")
            logger.error(f"[SOLUTION] Parallel enrichment failed: {error}")
            state["all_items"] = result.get("items", all_items)
            
            state["messages"] = state.get("messages", []) + [{
                "role": "system",
                "content": f"Parallel enrichment failed (non-critical): {error}"
            }]
        
        # Optional: Validate against domain-specific standards
        if domain or safety_requirements:
            logger.info(f"[SOLUTION] Validating against domain standards: {domain}")
            try:
                validation_result = validate_items_against_domain_standards(
                    items=state["all_items"],
                    domain=domain,
                    safety_requirements=safety_requirements
                )
                state["standards_validation"] = validation_result
                
                if not validation_result.get("is_compliant"):
                    logger.warning(f"[SOLUTION] Compliance issues found: {len(validation_result.get('compliance_issues', []))}")
            except Exception as ve:
                logger.warning(f"[SOLUTION] Domain validation skipped: {ve}")
        
    except ImportError as ie:
        logger.error(f"[SOLUTION] Parallel enrichment import failed: {ie}")
        # Fallback to existing Standards RAG enrichment
        try:
            domain = solution_analysis.get("industry", "")
            enriched_items = enrich_identified_items_with_standards(
                items=all_items,
                product_type=None,
                domain=domain,
                top_k=3
            )
            state["all_items"] = enriched_items
            logger.info("[SOLUTION] Fallback to existing Standards RAG enrichment")
        except Exception as e2:
            logger.error(f"[SOLUTION] Fallback enrichment also failed: {e2}")
            
    except Exception as e:
        logger.error(f"[SOLUTION] Specification enrichment failed: {e}")
        import traceback
        traceback.print_exc()
        state["messages"] = state.get("messages", []) + [{
            "role": "system",
            "content": f"Specification enrichment failed (non-critical): {str(e)}"
        }]
    
    state["current_step"] = "format_list"
    return state


def _extract_standards_from_solution_specs(final_specs: dict) -> list:
    """Extract applicable standards codes from Deep Agent results."""
    standards = []
    constraints = final_specs.get("constraints_applied", [])
    for c in constraints:
        if isinstance(c, dict):
            ref = c.get("standard_reference") or c.get("constraint", "")
            if ref:
                import re
                matches = re.findall(r'\b(IEC|ISO|API|ANSI|ISA|EN|NFPA|ASME|IEEE)\s*[\d.-]+', str(ref), re.IGNORECASE)
                standards.extend(matches)
    return list(set(standards))


def _extract_certifications_from_solution_specs(final_specs: dict) -> list:
    """Extract certifications from Deep Agent results."""
    certs = []
    specs = final_specs.get("specifications", {})
    for key, value in specs.items():
        key_lower = key.lower()
        value_str = str(value).upper()
        if any(x in key_lower for x in ["cert", "sil", "atex", "rating", "approval"]):
            certs.append(f"{key}: {value}")
        import re
        cert_patterns = re.findall(r'\b(SIL\s*[1-4]|ATEX|IECEx|CE|UL|CSA|FM|IP\d{2})', value_str)
        certs.extend(cert_patterns)
    return list(set(certs))


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
                
                # Check source in original dict if possible
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
             for key, val in original_specs.items():
                 # Key normalization for comparison
                 key_clean = key.lower().replace("_", "").replace(" ", "")
                 existing_keys = [k.lower().replace("_", "").replace(" ", "") for k in display_specs.keys()]
                 
                 if key_clean not in existing_keys:
                     val_str = str(val).strip()
                     val_str = val_str.replace("[INFERRED]", "").replace("[STANDARDS]", "").strip()
                     
                     if val_str and val_str.lower() not in PLACEHOLDER_VALUES:
                         display_specs[key] = val_str
        
        # Update item
        if display_specs:
            item["specifications"] = display_specs
    
    return items




def format_solution_list_node(state: SolutionState) -> SolutionState:
    """
    Node 3: Format Selection List for Solution.
    Formats identified items into user-friendly numbered list with sample_inputs.
    This is the FINAL node - workflow ends here and waits for user selection.
    """
    logger.info("[SOLUTION] Node 3: Formatting selection list...")
    
    # === DEBUG: Print specification source analysis ===
    try:
        from debug_spec_tracker import print_specification_report
        all_items = state.get("all_items", [])
        if all_items:
            print_specification_report(all_items)
    except Exception as debug_err:
        logger.debug(f"[SOLUTION] Debug tracker not available: {debug_err}")
    # === END DEBUG ===

    try:
        llm = create_llm_with_fallback(
            model="gemini-2.5-flash",
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        prompt = ChatPromptTemplate.from_template(SOLUTION_INSTRUMENT_LIST_PROMPT)
        parser = JsonOutputParser()
        chain = prompt | llm | parser

        result = chain.invoke({
            "solution_analysis": json.dumps(state.get("solution_analysis", {}), indent=2),
            "instruments": json.dumps(state["identified_instruments"], indent=2),
            "accessories": json.dumps(state["identified_accessories"], indent=2)
        })

        solution_name = state.get("solution_name", "Solution")
        solution_analysis = state.get("solution_analysis", {})

        # Format response for user
        response_lines = [
            f"**🏗️ {solution_name} - Instrument Requirements**\n",
            f"*Industry: {solution_analysis.get('industry', 'Industrial')} | Process: {solution_analysis.get('process_type', 'General')}*\n",
            f"\nI've identified **{state['total_items']} items** required for your solution:\n"
        ]

        formatted_list = result.get("formatted_list", [])

        for item in formatted_list:
            emoji = "🔧" if item.get("type") == "instrument" else "🔩"
            response_lines.append(
                f"{item['number']}. {emoji} **{item.get('name', 'Unknown')}** ({item.get('category', '')})"
            )
            response_lines.append(
                f"   📊 Quantity: {item.get('quantity', 1)}"
            )
            
            # Show purpose
            purpose = item.get("purpose", "")
            if purpose:
                response_lines.append(f"   💡 Purpose: {purpose}")

            # Show sample_input preview
            actual_item = next((i for i in state["all_items"] if i["number"] == item["number"]), None)
            if actual_item and actual_item.get("sample_input"):
                sample_preview = actual_item["sample_input"][:100] + "..." if len(actual_item["sample_input"]) > 100 else actual_item["sample_input"]
                response_lines.append(
                    f"   🔍 Search query: {sample_preview}"
                )

            response_lines.append("")

        # Add solution summary if available
        solution_summary = result.get("solution_summary", "")
        if solution_summary:
            response_lines.append(f"**📋 Solution Summary:** {solution_summary}\n")

        response_lines.append(
            f"\n**📌 Next Steps:**\n"
            f"Reply with an item number (1-{state['total_items']}) to search for vendor products.\n"
            f"I'll then find specific product recommendations for your selected item."
        )

        state["response"] = "\n".join(response_lines)
        
        # === MERGE DEEP AGENT SPECS FOR UI DISPLAY ===
        # This ensures Deep Agent specifications are shown in UI with [STANDARDS] labels
        _merge_deep_agent_specs_for_display(state["all_items"])
        logger.info(f"[SOLUTION] Merged Deep Agent specs for {len(state['all_items'])} items")
        # === END MERGE ===
        
        state["response_data"] = {
            "workflow": "solution",
            "solution_name": solution_name,
            "solution_analysis": solution_analysis,
            "project_name": solution_name,  # For compatibility
            "items": state["all_items"],  # Full items with merged specs
            "total_items": state["total_items"],
            "awaiting_selection": True,
            "instructions": f"Reply with item number (1-{state['total_items']}) to get product recommendations"
        }

        state["current_step"] = "complete"

        logger.info(f"[SOLUTION] Selection list formatted with {state['total_items']} items")

    except Exception as e:
        logger.error(f"[SOLUTION] List formatting failed: {e}")
        state["error"] = str(e)

        # Fallback: Create simple text list
        response_lines = [
            f"**🏗️ {state.get('solution_name', 'Solution')} - Identified Items**\n"
        ]

        for item in state.get("all_items", []):
            emoji = "🔧" if item.get("type") == "instrument" else "🔩"
            response_lines.append(
                f"{item['number']}. {emoji} {item.get('name', 'Unknown')} ({item.get('category', '')})"
            )

        response_lines.append(
            f"\nReply with an item number (1-{state.get('total_items', 0)}) to search for products."
        )

        state["response"] = "\n".join(response_lines)
        state["response_data"] = {
            "workflow": "solution",
            "items": state.get("all_items", []),
            "total_items": state.get("total_items", 0),
            "awaiting_selection": True
        }

    return state


# =============================================================================
# WORKFLOW CREATION
# =============================================================================

def create_solution_workflow() -> StateGraph:
    """
    Create the Solution Workflow.

    This workflow analyzes solution descriptions and identifies required
    instruments/accessories, similar to instrument_identifier_workflow.py.

    Flow:
    1. Analyze Solution Context (domain, process, safety)
    2. Identify Instruments & Accessories (with sample_input generation)
    3. Format Selection List

    After this workflow completes, user selects an item, and the sample_input
    is routed to the PRODUCT SEARCH workflow.

    Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                   SOLUTION WORKFLOW                          │
    ├─────────────────────────────────────────────────────────────┤
    │  User: "I'm designing a crude oil distillation unit..."    │
    │                              ↓                               │
    │  ┌───────────────────────────────────────────────────────┐  │
    │  │           analyze_solution                             │  │
    │  │   (Extract domain, process, safety requirements)       │  │
    │  └──────────────────────┬────────────────────────────────┘  │
    │                         ↓                                    │
    │  ┌───────────────────────────────────────────────────────┐  │
    │  │         identify_solution_instruments                  │  │
    │  │   (Identify instruments + accessories + sample_input) │  │
    │  └──────────────────────┬────────────────────────────────┘  │
    │                         ↓                                    │
    │  ┌───────────────────────────────────────────────────────┐  │
    │  │            format_solution_list                        │  │
    │  │   (Generate numbered list for user selection)          │  │
    │  └──────────────────────┬────────────────────────────────┘  │
    │                         ↓                                    │
    │              [END - Awaiting User Selection]                 │
    │                                                              │
    │  When user selects item:                                     │
    │    → sample_input is sent to Product Search Workflow         │
    └─────────────────────────────────────────────────────────────┘
    """

    workflow = StateGraph(SolutionState)

    # Add 4 nodes (including Standards RAG enrichment)
    workflow.add_node("analyze_solution", analyze_solution_node)
    workflow.add_node("identify_instruments", identify_solution_instruments_node)
    workflow.add_node("enrich_with_standards", enrich_solution_with_standards_node)  # NEW: Standards RAG
    workflow.add_node("format_list", format_solution_list_node)

    # Set entry point
    workflow.set_entry_point("analyze_solution")

    # Add edges - linear flow with Standards RAG enrichment
    workflow.add_edge("analyze_solution", "identify_instruments")
    workflow.add_edge("identify_instruments", "enrich_with_standards")  # NEW: Route to Standards RAG
    workflow.add_edge("enrich_with_standards", "format_list")  # NEW: Then to formatting
    workflow.add_edge("format_list", END)  # WORKFLOW ENDS HERE - waits for user selection

    return workflow


# =============================================================================
# WORKFLOW EXECUTION
# =============================================================================

@with_workflow_lock(session_id_param="session_id", timeout=60.0)
def run_solution_workflow(
    user_input: str,
    session_id: str = "default",
    checkpointing_backend: str = "memory"
) -> Dict[str, Any]:
    """
    Run the solution workflow.

    This workflow analyzes solution descriptions and returns a selection list
    of identified instruments/accessories. It does NOT perform product search.

    Args:
        user_input: User's solution description (e.g., "I'm designing a crude oil distillation unit...")
        session_id: Session identifier
        checkpointing_backend: Backend for state persistence

    Returns:
        {
            "success": True,
            "response": "Formatted selection list",
            "response_data": {
                "workflow": "solution",
                "solution_name": "...",
                "solution_analysis": {...},
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
        logger.info(f"[SOLUTION] Starting workflow for session: {session_id}")
        logger.info(f"[SOLUTION] User input: {user_input[:100]}...")

        # Create initial state
        initial_state = create_solution_state(user_input, session_id)

        # Create and compile workflow
        workflow = create_solution_workflow()
        compiled = compile_with_checkpointing(workflow, checkpointing_backend)

        # Execute workflow
        result = compiled.invoke(
            initial_state,
            config={"configurable": {"thread_id": session_id}}
        )

        logger.info(f"[SOLUTION] Workflow completed successfully")
        logger.info(f"[SOLUTION] Generated {result.get('total_items', 0)} items for selection")

        return {
            "success": True,
            "response": result.get("response", ""),
            "response_data": result.get("response_data", {}),
            "error": result.get("error")
        }

    except TimeoutError as e:
        logger.error(f"[SOLUTION] Workflow lock timeout for session {session_id}: {e}")
        return {
            "success": False,
            "error": "Another workflow is currently running for this session. Please try again."
        }
    except Exception as e:
        logger.error(f"[SOLUTION] Workflow execution failed: {e}")
        import traceback
        traceback.print_exc()

        return {
            "success": False,
            "response": f"I encountered an error while analyzing your solution: {str(e)}",
            "response_data": {
                "workflow": "solution",
                "error": str(e),
                "awaiting_selection": False
            }
        }


def run_solution_workflow_stream(
    user_input: str,
    session_id: str = "default",
    checkpointing_backend: str = "memory",
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
) -> Dict[str, Any]:
    """
    Run the solution workflow with streaming progress updates.

    Args:
        user_input: User's solution description
        session_id: Session identifier
        checkpointing_backend: Backend for state persistence
        progress_callback: Callback function to emit progress updates

    Returns:
        Workflow result with selection list
    """
    from .streaming_utils import ProgressEmitter

    emitter = ProgressEmitter(progress_callback)

    try:
        # Step 1: Initialize (10%)
        emitter.emit(
            'initialize',
            'Analyzing your solution description...',
            10,
            data={'user_input_preview': user_input[:100]}
        )

        # Step 2: Analyze Solution (30%)
        emitter.emit(
            'analyze_solution',
            'Extracting solution context and requirements...',
            30
        )

        # Step 3: Identify Instruments (60%)
        emitter.emit(
            'identify_instruments',
            'Identifying required instruments and accessories...',
            60
        )

        # Step 4: Format List (85%)
        emitter.emit(
            'format_list',
            'Preparing instrument selection list...',
            85
        )

        # Execute the actual workflow
        logger.info(f"[SOLUTION-STREAM] Starting workflow for session {session_id}")
        result = run_solution_workflow(user_input, session_id, checkpointing_backend)

        # Step 5: Complete (100%)
        if result.get("success"):
            response_data = result.get("response_data", {})
            emitter.emit(
                'complete',
                'Solution analysis complete! Please select an item.',
                100,
                data={
                    'item_count': response_data.get('total_items', 0),
                    'solution_name': response_data.get('solution_name', ''),
                    'awaiting_selection': response_data.get('awaiting_selection', True)
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


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'SolutionState',
    'create_solution_workflow',
    'run_solution_workflow',
    'run_solution_workflow_stream'
]
