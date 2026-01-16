# agentic/deep_agent_workflow.py
# =============================================================================
# DEEP AGENT WORKFLOW WITH PLANNER
# =============================================================================
#
# Memory-centric Deep Agent that:
# 1. Loads ALL data upfront (user input, standards docs, identified items)
# 2. Analyzes and stores in Memory Bank
# 3. Uses hierarchical threading (main threads + sub-threads per product type)
# 4. Retrieves from memory to build Standard Specifications JSON
#
# =============================================================================

import logging
import time
import json
from typing import Dict, Any, List, Optional, TypedDict, Literal
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from langgraph.graph import StateGraph, END

from .memory import (
    DeepAgentMemory,
    StandardsDocumentAnalysis,
    UserContextMemory,
    IdentifiedItemMemory,
    ThreadInfo,
    ExecutionPlan,
    get_relevant_documents_for_product
)
from .document_loader import (
    STANDARDS_DIRECTORY,
    populate_memory_with_documents,
    populate_memory_with_items,
    analyze_user_input
)
from ..checkpointing import compile_with_checkpointing
from tools.instrument_tools import identify_instruments_tool, identify_accessories_tool
from llm_fallback import create_llm_with_fallback
from .schema_field_extractor import extract_standards_from_value
from .spec_output_normalizer import (
    normalize_full_item_specs,
    deduplicate_specs,
    normalize_key
)

import os
import re
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# =============================================================================
# FINAL VALUE VALIDATION
# =============================================================================

def validate_and_clean_spec_value(value: str, field_name: str = "") -> Optional[str]:
    """
    Final validation of a specification value before storage.

    Uses ValueNormalizer to ensure the value is a clean technical specification,
    not raw text or a description.

    Args:
        value: The specification value to validate
        field_name: Optional field name for logging

    Returns:
        Cleaned value or None if invalid
    """
    if not value:
        return None

    try:
        from .value_normalizer import ValueNormalizer
        normalizer = ValueNormalizer()

        result = normalizer.extract_and_validate(str(value))

        if result["is_valid"]:
            return result["normalized_value"]
        else:
            logger.debug(f"[FinalValidation] Rejected value for '{field_name}': {str(value)[:50]}...")
            return None

    except ImportError:
        # Fallback: basic validation if normalizer not available
        value_str = str(value).strip()

        # Reject obvious descriptions
        description_indicators = [
            "typically", "usually", "generally", "should be", "must be",
            "depends on", "according to", "the value is", "is defined as",
            "for applications", "in order to", "when used in"
        ]

        value_lower = value_str.lower()
        for indicator in description_indicators:
            if indicator in value_lower:
                logger.debug(f"[FinalValidation] Rejected descriptive value for '{field_name}'")
                return None

        # Reject too-long values (likely sentences)
        if len(value_str) > 150:
            logger.debug(f"[FinalValidation] Rejected long value for '{field_name}'")
            return None

        return value_str


def validate_spec_dict(spec_dict: Dict[str, Any], spec_type: str = "unknown") -> Dict[str, Any]:
    """
    Validate all values in a specification dictionary.

    Args:
        spec_dict: Dictionary of spec key -> spec data
        spec_type: Type name for logging (e.g., "mandatory", "optional")

    Returns:
        Validated dictionary with invalid values removed
    """
    validated = {}
    removed_count = 0

    for key, data in spec_dict.items():
        if isinstance(data, dict):
            raw_value = data.get("value", "")
            if raw_value:
                cleaned_value = validate_and_clean_spec_value(str(raw_value), key)
                if cleaned_value:
                    # Update with cleaned value
                    validated[key] = {
                        **data,
                        "value": cleaned_value
                    }
                else:
                    removed_count += 1
        elif isinstance(data, str) and data:
            cleaned_value = validate_and_clean_spec_value(data, key)
            if cleaned_value:
                validated[key] = cleaned_value
            else:
                removed_count += 1

    if removed_count > 0:
        logger.info(f"[FinalValidation] Removed {removed_count} invalid {spec_type} specs")

    return validated


def filter_options_by_user_input(value: str, user_input: str) -> str:
    """
    Filter multi-option values based on user input.
    
    If value contains multiple options (e.g., "3mm, 6mm, 8mm") and user specified
    one of them, return only the user-specified option.
    
    Args:
        value: The specification value (may contain multiple comma-separated options)
        user_input: The user's original input string
        
    Returns:
        Filtered value if user specified one option, otherwise original value
    """
    if not value or not user_input:
        return value
    
    # Detect multi-option patterns (comma-separated or "or" separated)
    if "," not in value and " or " not in value.lower():
        return value  # Not a multi-option value
    
    # Split options
    options = re.split(r',|\s+or\s+', value, flags=re.IGNORECASE)
    options = [opt.strip() for opt in options if opt.strip()]
    
    if len(options) <= 1:
        return value  # Not actually multi-option
    
    user_input_lower = user_input.lower()
    # Also normalize user input: remove hyphens for matching
    user_input_normalized = user_input_lower.replace("-", " ").replace("_", " ")
    
    # First, try to find exact word matches (more specific = better)
    for opt in options:
        opt_clean = opt.strip()
        opt_lower = opt_clean.lower()
        
        # Strategy 1: Exact word boundary match for the full option
        # Use word boundaries to match "ungrounded" but not "grounded" in "ungrounded"
        opt_pattern = r'\b' + re.escape(opt_lower) + r'\b'
        if re.search(opt_pattern, user_input_lower):
            logger.debug(f"[FilterOptions] User specified '{opt_clean}' from options (exact word)")
            return opt_clean
    
    # Second pass: try partial/pattern matches
    for opt in options:
        opt_clean = opt.strip()
        opt_lower = opt_clean.lower()
        opt_normalized = opt_lower.replace("-", " ")
        
        # Strategy 2: Extract numeric value for matching (e.g., "3mm" -> "3")
        numeric_match = re.search(r'([\d.]+)\s*(mm|inch|in|"|npt|bsp)', opt_clean, re.IGNORECASE)
        if numeric_match:
            numeric_val = numeric_match.group(1)
            unit = numeric_match.group(2).lower()
            # Check if user input contains this specific value with word boundary
            pattern = rf'\b{numeric_val}\s*{unit}\b'
            if re.search(pattern, user_input_lower):
                logger.debug(f"[FilterOptions] User specified '{opt_clean}' from options (numeric)")
                return opt_clean
        
        # Strategy 3: Check for wire configurations (2-wire, 3-wire, 4-wire)
        wire_match = re.search(r'(\d)\s*-?\s*wire', opt_lower)
        if wire_match:
            wire_num = wire_match.group(1)
            # Use word boundary for wire matching
            wire_pattern = rf'\b{wire_num}\s*-?\s*wire\b'
            if re.search(wire_pattern, user_input_normalized):
                logger.debug(f"[FilterOptions] User specified '{opt_clean}' from options (wire)")
                return opt_clean
        
        # Strategy 4: Check for protocol/keyword matches (HART, Profibus, etc.)
        # Extract first significant word - use word boundary
        first_word = opt_clean.split()[0] if opt_clean.split() else ""
        if len(first_word) >= 4:
            word_pattern = rf'\b{re.escape(first_word.lower())}\b'
            if re.search(word_pattern, user_input_lower):
                logger.debug(f"[FilterOptions] User specified '{opt_clean}' from options (keyword)")
                return opt_clean
    
    # User didn't specify any option - return all options
    return value


# =============================================================================
# WORKFLOW STATE
# =============================================================================

class DeepAgentState(TypedDict, total=False):
    """State for Deep Agent with Planner"""

    # Session
    session_id: str
    user_input: str

    # Identified Items (from identification tools)
    identified_instruments: List[Dict[str, Any]]
    identified_accessories: List[Dict[str, Any]]

    # Planner Output
    execution_plan: ExecutionPlan
    user_context: UserContextMemory

    # Thread Results
    thread_results: Dict[str, Dict[str, Any]]

    # Final Output
    standard_specifications_json: Dict[str, Any]

    # Workflow tracking
    current_step: str
    steps_completed: List[str]
    error: Optional[str]

    # Timing
    start_time: float
    processing_time_ms: int

    # Response
    response: str
    response_data: Dict[str, Any]


def create_deep_agent_state(
    user_input: str,
    session_id: Optional[str] = None,
    identified_instruments: Optional[List[Dict[str, Any]]] = None,
    identified_accessories: Optional[List[Dict[str, Any]]] = None
) -> DeepAgentState:
    """Create initial state for Deep Agent workflow"""

    return DeepAgentState(
        session_id=session_id or f"deep-agent-{int(time.time())}",
        user_input=user_input,
        identified_instruments=identified_instruments or [],
        identified_accessories=identified_accessories or [],
        execution_plan=None,
        user_context=None,
        thread_results={},
        standard_specifications_json={},
        current_step="init",
        steps_completed=[],
        error=None,
        start_time=time.time(),
        processing_time_ms=0,
        response="",
        response_data={}
    )


# =============================================================================
# MEMORY MANAGER (Global, outside state graph)
# =============================================================================

# Global memory storage - indexed by session_id
_memory_store: Dict[str, DeepAgentMemory] = {}
_memory_lock = threading.Lock()


def get_or_create_memory(session_id: str) -> DeepAgentMemory:
    """Get or create memory for a session."""
    with _memory_lock:
        if session_id not in _memory_store:
            _memory_store[session_id] = DeepAgentMemory()
        return _memory_store[session_id]


def get_memory(session_id: str) -> Optional[DeepAgentMemory]:
    """Get memory for a session."""
    with _memory_lock:
        return _memory_store.get(session_id)


def clear_memory(session_id: str) -> None:
    """Clear memory for a session."""
    with _memory_lock:
        if session_id in _memory_store:
            del _memory_store[session_id]


# =============================================================================
# NODE 1: LOAD AND ANALYZE DOCUMENTS
# =============================================================================

def load_documents_node(state: DeepAgentState) -> DeepAgentState:
    """
    Node 1: Load all standards documents and populate memory.

    This node:
    1. Loads all 12 standards documents from local storage
    2. Parses each document into sections
    3. Extracts standards codes, certifications, specs, guidelines
    4. Stores all analysis in Memory Bank
    """
    logger.info("=" * 70)
    logger.info("[Node 1] LOAD DOCUMENTS - Loading all standards documents")
    logger.info("=" * 70)

    state["current_step"] = "load_documents"

    try:
        # Get or create memory (stored globally, not in state)
        memory = get_or_create_memory(state["session_id"])

        # Populate memory with document analyses
        memory = populate_memory_with_documents(memory, STANDARDS_DIRECTORY)

        state["steps_completed"].append("load_documents")

        logger.info(
            f"[Node 1] Completed: {memory.documents_loaded} documents, "
            f"{memory.sections_analyzed} sections analyzed"
        )

    except Exception as e:
        logger.error(f"[Node 1] Error loading documents: {e}", exc_info=True)
        state["error"] = f"Document loading failed: {str(e)}"

    return state


# =============================================================================
# NODE 2: ANALYZE USER INPUT
# =============================================================================

def analyze_user_input_node(state: DeepAgentState) -> DeepAgentState:
    """
    Node 2: Analyze user input to extract context.

    Extracts:
    - Domain (Oil & Gas, Chemical, etc.)
    - Process type (Distillation, Refining, etc.)
    - Safety requirements (SIL, ATEX, etc.)
    - Key parameters (temperature, pressure ranges)
    """
    logger.info("=" * 70)
    logger.info("[Node 2] ANALYZE USER INPUT")
    logger.info("=" * 70)

    state["current_step"] = "analyze_user_input"

    try:
        user_input = state["user_input"]

        # Analyze user input
        user_context = analyze_user_input(user_input)
        state["user_context"] = user_context

        # Store in memory (get from global store)
        memory = get_or_create_memory(state["session_id"])
        memory.store_user_context(user_context)

        state["steps_completed"].append("analyze_user_input")

        logger.info(
            f"[Node 2] Context: domain={user_context.get('domain')}, "
            f"process={user_context.get('process_type')}, "
            f"safety={user_context.get('safety_requirements')}"
        )

    except Exception as e:
        logger.error(f"[Node 2] Error analyzing user input: {e}", exc_info=True)
        state["error"] = f"User input analysis failed: {str(e)}"

    return state


# =============================================================================
# NODE 3: IDENTIFY INSTRUMENTS AND ACCESSORIES
# =============================================================================

def identify_items_node(state: DeepAgentState) -> DeepAgentState:
    """
    Node 3: Identify instruments and accessories from user input.

    If items are already provided in state, skip identification.
    Otherwise, use LLM-based identification tools.
    """
    logger.info("=" * 70)
    logger.info("[Node 3] IDENTIFY INSTRUMENTS AND ACCESSORIES")
    logger.info("=" * 70)

    state["current_step"] = "identify_items"

    try:
        # Check if items already provided
        if state["identified_instruments"] and len(state["identified_instruments"]) > 0:
            logger.info(
                f"[Node 3] Using provided items: "
                f"{len(state['identified_instruments'])} instruments, "
                f"{len(state['identified_accessories'])} accessories"
            )
        else:
            # Use identification tools
            logger.info("[Node 3] Identifying instruments from user input...")

            # Identify instruments
            inst_result = identify_instruments_tool.invoke({
                "requirements": state["user_input"]
            })

            if inst_result.get("success"):
                state["identified_instruments"] = inst_result.get("instruments", [])
                logger.info(f"[Node 3] Identified {len(state['identified_instruments'])} instruments")

                # Identify accessories
                if state["identified_instruments"]:
                    acc_result = identify_accessories_tool.invoke({
                        "instruments": state["identified_instruments"],
                        "process_context": state["user_input"]
                    })

                    if acc_result.get("success"):
                        state["identified_accessories"] = acc_result.get("accessories", [])
                        logger.info(f"[Node 3] Identified {len(state['identified_accessories'])} accessories")
            else:
                logger.warning(f"[Node 3] Identification failed: {inst_result.get('error')}")

        # Populate memory with identified items (get from global store)
        memory = get_or_create_memory(state["session_id"])
        # Pass user_context for constraint-aware document selection
        user_context = state.get("user_context", {})
        populate_memory_with_items(
            memory,
            state["identified_instruments"],
            state["identified_accessories"],
            user_context=user_context
        )

        state["steps_completed"].append("identify_items")

    except Exception as e:
        logger.error(f"[Node 3] Error identifying items: {e}", exc_info=True)
        state["error"] = f"Item identification failed: {str(e)}"

    return state


# =============================================================================
# NODE 4: CREATE EXECUTION PLAN (PLANNER)
# =============================================================================

def create_execution_plan_node(state: DeepAgentState) -> DeepAgentState:
    """
    Node 4: Create execution plan with thread structure.

    Creates:
    - Main thread for instruments (with sub-threads for each product type)
    - Main thread for accessories (with sub-threads for each product type)
    """
    logger.info("=" * 70)
    logger.info("[Node 4] CREATE EXECUTION PLAN (PLANNER)")
    logger.info("=" * 70)

    state["current_step"] = "create_execution_plan"

    try:
        # Get memory from global store
        memory = get_or_create_memory(state["session_id"])

        # Build instrument threads
        instrument_threads = []
        for item in memory.identified_instruments:
            thread_info: ThreadInfo = {
                "thread_id": item["item_id"],
                "product_type": item["product_type"],
                "item_type": "instrument",
                "status": "pending",
                "relevant_docs": item["relevant_documents"],
                "result": None
            }
            instrument_threads.append(thread_info)

        # Build accessory threads
        accessory_threads = []
        for item in memory.identified_accessories:
            thread_info: ThreadInfo = {
                "thread_id": item["item_id"],
                "product_type": item["product_type"],
                "item_type": "accessory",
                "status": "pending",
                "relevant_docs": item["relevant_documents"],
                "result": None
            }
            accessory_threads.append(thread_info)

        # Create execution plan
        execution_plan: ExecutionPlan = {
            "main_threads": {
                "instruments": {
                    "count": len(instrument_threads),
                    "sub_threads": instrument_threads
                },
                "accessories": {
                    "count": len(accessory_threads),
                    "sub_threads": accessory_threads
                }
            },
            "total_threads": len(instrument_threads) + len(accessory_threads),
            "created_at": datetime.now().isoformat()
        }

        state["execution_plan"] = execution_plan
        memory.execution_plan = execution_plan

        state["steps_completed"].append("create_execution_plan")

        logger.info(
            f"[Node 4] Execution plan created: "
            f"{len(instrument_threads)} instrument threads, "
            f"{len(accessory_threads)} accessory threads"
        )

    except Exception as e:
        logger.error(f"[Node 4] Error creating execution plan: {e}", exc_info=True)
        state["error"] = f"Execution plan creation failed: {str(e)}"

    return state


# =============================================================================
# NODE 5: EXECUTE THREADS (PARALLEL PROCESSING)
# =============================================================================

def _process_single_thread(
    thread_info: ThreadInfo,
    session_id: str,
    user_context: UserContextMemory
) -> Dict[str, Any]:
    """
    Process a single product type thread.

    Uses Extractor and Critic sub-agents to:
    1. Extract JSON technical specifications from standards documents
    2. Validate the extraction is proper JSON format
    3. Retry with feedback if validation fails
    4. Build Standard Specifications JSON for the product type
    """
    thread_id = thread_info["thread_id"]
    product_type = thread_info["product_type"]
    item_type = thread_info["item_type"]

    logger.info(f"[Thread {thread_id}] Processing: {product_type}")

    try:
        # Get memory from global store
        memory = get_or_create_memory(session_id)

        # RETRIEVE specifications from memory (pre-analyzed data)
        specs_from_memory = memory.get_specifications_for_product_type(product_type)

        # Get the identified item details
        if item_type == "instrument":
            item_details = next(
                (i for i in memory.identified_instruments if i["item_id"] == thread_id),
                {}
            )
        else:
            item_details = next(
                (i for i in memory.identified_accessories if i["item_id"] == thread_id),
                {}
            )

        # ==========================================================
        # NEW: Use Extractor + Critic Sub-Agents for JSON extraction
        # ==========================================================
        llm_extracted_specs = {}
        
        try:
            from .sub_agents import extract_specifications_with_validation
            
            # Get relevant document content for this product
            # IMPROVED (v2): Increased content limits from 8K/5K to 20K/15K for better spec extraction
            relevant_docs = item_details.get("relevant_documents", [])
            combined_content = ""

            for doc_key in relevant_docs:
                analysis = memory.standards_analysis.get(doc_key)
                if analysis:
                    # Increased from 8000 to 20000 chars (+150% content visibility)
                    combined_content += analysis.get("full_content", "")[:20000] + "\n\n"

            # Fallback: use first few documents if no specific ones found
            if not combined_content and memory.standards_analysis:
                for doc_name, analysis in list(memory.standards_analysis.items())[:3]:
                    # Increased from 5000 to 15000 chars (+200% content visibility)
                    combined_content += analysis.get("full_content", "")[:15000] + "\n\n"
            
            if combined_content:
                logger.info(f"[Thread {thread_id}] Running Extractor+Critic for: {product_type}")
                
                # Run extraction with validation loop
                llm_extracted_specs = extract_specifications_with_validation(
                    product_type=product_type,
                    document_content=combined_content,
                    user_context=user_context,
                    max_retries=3
                )
                
                extraction_meta = llm_extracted_specs.get("_extraction_metadata", {})
                logger.info(
                    f"[Thread {thread_id}] Extraction complete: "
                    f"score={extraction_meta.get('quality_score', 0):.2f}, "
                    f"attempts={extraction_meta.get('attempts', 1)}"
                )
        
        except Exception as e:
            logger.warning(f"[Thread {thread_id}] LLM extraction failed, using fallback: {e}")
            llm_extracted_specs = {}

        # Build specification JSON (merge memory + LLM extraction)
        spec_json = build_product_specification_json(
            thread_id=thread_id,
            product_type=product_type,
            item_type=item_type,
            item_details=item_details,
            specs_from_memory=specs_from_memory,
            user_context=user_context,
            memory=memory,
            llm_extracted_specs=llm_extracted_specs  # Pass LLM-extracted specs
        )

        logger.info(f"[Thread {thread_id}] Completed: {product_type}")

        return {
            "thread_id": thread_id,
            "product_type": product_type,
            "status": "completed",
            "result": spec_json
        }

    except Exception as e:
        logger.error(f"[Thread {thread_id}] Error: {e}")
        return {
            "thread_id": thread_id,
            "product_type": product_type,
            "status": "error",
            "error": str(e),
            "result": None
        }


def build_product_specification_json(
    thread_id: str,
    product_type: str,
    item_type: str,
    item_details: Dict[str, Any],
    specs_from_memory: Dict[str, Any],
    user_context: UserContextMemory,
    memory: DeepAgentMemory,
    llm_extracted_specs: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Build Standard Specifications JSON for a product type.

    Merges specifications from:
    1. LLM Extraction (highest priority - intelligent extraction)
    2. Memory (pre-analyzed document data)
    3. User Input (user-provided specifications)
    """
    # Get relevant document analyses
    relevant_docs = item_details.get("relevant_documents", [])
    sources_used = []

    # Build mandatory specifications
    mandatory_specs = {}
    optional_specs = {}
    safety_specs = {}
    environmental_specs = {}

    # ==========================================================
    # PRIORITY 1: LLM-Extracted Specifications (Sub-Agent output)
    # ==========================================================
    if llm_extracted_specs:
        llm_specs = llm_extracted_specs.get("specifications", {})
        
        # Merge LLM mandatory specs
        llm_mandatory = llm_specs.get("mandatory", {})
        for key, value in llm_mandatory.items():
            if isinstance(value, dict) and value.get("value"):
                mandatory_specs[key] = value
            elif isinstance(value, str) and value:
                mandatory_specs[key] = {
                    "value": value,
                    "source": "standards_specifications",
                    "confidence": 0.85,
                    "standards_referenced": extract_standards_from_value(value)
                }
        
        # Merge LLM optional specs
        llm_optional = llm_specs.get("optional", {})
        for key, value in llm_optional.items():
            if isinstance(value, dict) and value.get("value"):
                optional_specs[key] = value
            elif isinstance(value, str) and value:
                optional_specs[key] = {
                    "value": value,
                    "source": "standards_specifications",
                    "confidence": 0.85,
                    "standards_referenced": extract_standards_from_value(value)
                }
        
        # Merge LLM safety specs
        llm_safety = llm_specs.get("safety", {})
        for key, value in llm_safety.items():
            if isinstance(value, dict) and value.get("value"):
                safety_specs[key] = value
            elif isinstance(value, str) and value:
                safety_specs[key] = {
                    "value": value,
                    "source": "standards_specifications",
                    "confidence": 0.85,
                    "standards_referenced": extract_standards_from_value(value)
                }
        
        logger.info(
            f"[BuildSpec] LLM extracted: {len(llm_mandatory)} mandatory, "
            f"{len(llm_optional)} optional, {len(llm_safety)} safety specs"
        )

    # ==========================================================
    # PRIORITY 2: Memory-based Specifications (regex extraction)
    # ==========================================================
    equipment_specs = specs_from_memory.get("equipment_specs", {})

    # Map common spec fields
    spec_field_mapping = {
        "accuracy": "accuracy",
        "repeatability": "repeatability",
        "pressure_range": "measurementRange",
        "temperature_range": "measurementRange",
        "flow_range": "measurementRange",
        "output_signal": "outputSignal",
        "power_supply": "powerSupply",
        "process_connection": "processConnection",
        "response_time": "responseTime",
    }

    for raw_field, json_field in spec_field_mapping.items():
        # Only add if not already from LLM
        if raw_field in equipment_specs and json_field not in mandatory_specs:
            mandatory_specs[json_field] = {
                "value": equipment_specs[raw_field],
                "source": "standards_specifications",
                "confidence": 0.8,
                "standards_referenced": extract_standards_from_value(str(equipment_specs[raw_field]))
            }

    # ==========================================================
    # PRIORITY 3: User-provided Specifications (ALWAYS OVERRIDE)
    # User input has highest priority - always overrides LLM/standards
    # ==========================================================
    user_specs = item_details.get("user_specifications", {})
    user_input_text = item_details.get("user_input", "") or item_details.get("sample_input", "")
    
    for key, value in user_specs.items():
        if value and value != "":
            # User specs ALWAYS override existing values
            mandatory_specs[key] = {
                "value": value,
                "source": "user_input",
                "confidence": 1.0,
                "standards_referenced": extract_standards_from_value(str(value))
            }
    
    # Also filter multi-option values based on user input
    if user_input_text:
        for key, spec_data in mandatory_specs.items():
            if isinstance(spec_data, dict) and spec_data.get("source") != "user_input":
                original_value = spec_data.get("value", "")
                if original_value:
                    filtered_value = filter_options_by_user_input(str(original_value), user_input_text)
                    if filtered_value != original_value:
                        mandatory_specs[key] = {
                            "value": filtered_value,
                            "source": "user_filtered_standards",
                            "confidence": 0.95,
                            "standards_referenced": spec_data.get("standards_referenced", [])
                        }

    # Safety specifications from memory
    safety_data = specs_from_memory.get("safety", {})

    # Apply user context safety requirements
    if user_context and user_context.get("safety_requirements"):
        user_safety = user_context["safety_requirements"]
        if user_safety.get("sil_level") and "silRating" not in safety_specs:
            safety_specs["silRating"] = {
                "value": user_safety["sil_level"],
                "source": "user_input",
                "confidence": 1.0,
                "standards_referenced": ["IEC 61508"]
            }
        if user_safety.get("hazardous_area") and "hazardousAreaRating" not in safety_specs:
            safety_specs["hazardousAreaRating"] = {
                "value": user_safety.get("atex_zone", "ATEX Zone 1"),
                "source": "user_input",
                "confidence": 1.0,
                "standards_referenced": ["IEC 60079"]
            }

    # Get certifications (merge LLM + memory)
    certifications = specs_from_memory.get("certifications", [])
    if llm_extracted_specs:
        llm_certs = llm_extracted_specs.get("specifications", {}).get("certifications", [])
        if llm_certs:
            certifications = list(set(certifications + llm_certs))

    # Environmental specifications
    if "ingress_protection" in equipment_specs and "ingressProtection" not in environmental_specs:
        environmental_specs["ingressProtection"] = {
            "value": equipment_specs["ingress_protection"],
            "source": "standards_specifications",
            "confidence": 0.8,
            "standards_referenced": extract_standards_from_value(str(equipment_specs["ingress_protection"]))
        }
    if "ambient_temperature" in equipment_specs and "ambientTemperature" not in environmental_specs:
        environmental_specs["ambientTemperature"] = {
            "value": equipment_specs["ambient_temperature"],
            "source": "standards_specifications",
            "confidence": 0.8,
            "standards_referenced": extract_standards_from_value(str(equipment_specs["ambient_temperature"]))
        }

    # Get applicable standards (merge LLM + memory)
    applicable_standards = []
    for code in specs_from_memory.get("applicable_standards", []):
        applicable_standards.append({
            "code": code,
            "source": "standards_documents"
        })
    
    # Add LLM-extracted standards
    if llm_extracted_specs:
        llm_standards = llm_extracted_specs.get("applicable_standards", [])
        existing_codes = {s["code"] for s in applicable_standards}
        for code in llm_standards:
            if isinstance(code, str) and code not in existing_codes:
                applicable_standards.append({
                    "code": code,
                    "source": "llm_extraction"
                })

    # Get installation guidelines (merge LLM + memory)
    guidelines = []
    for guideline in specs_from_memory.get("installation_guidelines", [])[:5]:
        guidelines.append({
            "guideline": guideline,
            "source": "standards_documents"
        })
    
    # Add LLM-extracted guidelines
    if llm_extracted_specs:
        llm_guidelines = llm_extracted_specs.get("guidelines", [])
        existing_guidelines = {g["guideline"] for g in guidelines}
        for guideline in llm_guidelines[:5]:
            if isinstance(guideline, str) and guideline not in existing_guidelines:
                guidelines.append({
                    "guideline": guideline,
                    "source": "llm_extraction"
                })

    # Get calibration info
    calibration = specs_from_memory.get("calibration", {})

    # Track sources
    sources_used = list(set(specs_from_memory.get("sources", [])))

    # ==========================================================
    # FINAL VALIDATION: Ensure all values are clean specs
    # This is the last line of defense against raw text leaking through
    # ==========================================================
    logger.info(f"[BuildSpec] Running final validation for {product_type}...")

    # Validate all spec dictionaries
    mandatory_specs = validate_spec_dict(mandatory_specs, "mandatory")
    optional_specs = validate_spec_dict(optional_specs, "optional")
    safety_specs = validate_spec_dict(safety_specs, "safety")
    environmental_specs = validate_spec_dict(environmental_specs, "environmental")

    # Build final JSON structure
    spec_json = {
        "thread_id": thread_id,
        "product_type": product_type,
        "item_type": item_type,
        "category": item_details.get("category", ""),
        "quantity": item_details.get("quantity", 1),

        "specifications": {
            "mandatory": mandatory_specs,
            "optional": optional_specs,
            "safety": safety_specs,
            "environmental": environmental_specs
        },

        "applicable_standards": applicable_standards,
        "certifications": certifications,
        "guidelines": guidelines,
        "calibration": calibration,
        "sources_used": sources_used
    }

    # Add extraction metadata if available
    if llm_extracted_specs and "_extraction_metadata" in llm_extracted_specs:
        spec_json["extraction_metadata"] = llm_extracted_specs["_extraction_metadata"]

    # Add related instrument for accessories
    if item_type == "accessory" and item_details.get("related_instrument"):
        spec_json["related_instrument"] = item_details["related_instrument"]

    # Log summary
    total_specs = len(mandatory_specs) + len(optional_specs) + len(safety_specs) + len(environmental_specs)
    logger.info(
        f"[BuildSpec] {product_type}: {total_specs} validated specs, "
        f"{len(applicable_standards)} standards, {len(certifications)} certs"
    )

    return spec_json


def execute_threads_node(state: DeepAgentState) -> DeepAgentState:
    """
    Node 5: Execute all threads in parallel.

    Processes:
    - All instrument sub-threads
    - All accessory sub-threads

    Each thread RETRIEVES from memory and builds specs JSON.
    """
    logger.info("=" * 70)
    logger.info("[Node 5] EXECUTE THREADS (PARALLEL PROCESSING)")
    logger.info("=" * 70)

    state["current_step"] = "execute_threads"

    try:
        session_id = state["session_id"]
        user_context = state["user_context"]
        execution_plan = state["execution_plan"]

        # Get memory from global store
        memory = get_or_create_memory(session_id)

        # Collect all threads
        all_threads = []

        # Instrument threads
        instrument_threads = execution_plan["main_threads"]["instruments"]["sub_threads"]
        all_threads.extend(instrument_threads)

        # Accessory threads
        accessory_threads = execution_plan["main_threads"]["accessories"]["sub_threads"]
        all_threads.extend(accessory_threads)

        logger.info(f"[Node 5] Executing {len(all_threads)} threads in parallel...")

        # Execute threads in parallel
        thread_results = {}
        max_workers = min(len(all_threads), 5)  # Limit parallel workers

        if len(all_threads) == 0:
            logger.warning("[Node 5] No threads to execute")
        elif len(all_threads) == 1:
            # Single thread - no parallelization needed
            result = _process_single_thread(all_threads[0], session_id, user_context)
            thread_results[result["thread_id"]] = result
        else:
            # Multiple threads - use parallel processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_thread = {
                    executor.submit(
                        _process_single_thread, thread_info, session_id, user_context
                    ): thread_info
                    for thread_info in all_threads
                }

                for future in as_completed(future_to_thread):
                    thread_info = future_to_thread[future]
                    try:
                        result = future.result()
                        thread_results[result["thread_id"]] = result

                        # Store in memory
                        memory.store_thread_result(result["thread_id"], result)

                    except Exception as e:
                        logger.error(f"[Node 5] Thread error: {e}")
                        thread_results[thread_info["thread_id"]] = {
                            "thread_id": thread_info["thread_id"],
                            "status": "error",
                            "error": str(e)
                        }

        state["thread_results"] = thread_results
        state["steps_completed"].append("execute_threads")

        completed_count = sum(1 for r in thread_results.values() if r.get("status") == "completed")
        logger.info(f"[Node 5] Completed: {completed_count}/{len(all_threads)} threads")

    except Exception as e:
        logger.error(f"[Node 5] Error executing threads: {e}", exc_info=True)
        state["error"] = f"Thread execution failed: {str(e)}"

    return state


# =============================================================================
# NODE 6: AGGREGATE RESULTS
# =============================================================================

def aggregate_results_node(state: DeepAgentState) -> DeepAgentState:
    """
    Node 6: Aggregate all thread results into final Standard Specifications JSON.
    """
    logger.info("=" * 70)
    logger.info("[Node 6] AGGREGATE RESULTS")
    logger.info("=" * 70)

    state["current_step"] = "aggregate_results"

    try:
        thread_results = state["thread_results"]
        user_context = state["user_context"]

        # Get memory from global store
        memory = get_or_create_memory(state["session_id"])

        # Separate instrument and accessory results
        instrument_specs = []
        accessory_specs = []

        for thread_id, result in thread_results.items():
            if result.get("status") != "completed":
                continue

            spec_json = result.get("result", {})
            
            # Normalize specifications to flat key-value format
            # This converts nested {value, confidence, note} to flat values
            # while preserving confidence as ghost values
            try:
                normalized_spec = normalize_full_item_specs(spec_json, preserve_ghost_values=True)
                # Deduplicate across sections
                if "specifications" in normalized_spec:
                    normalized_spec["specifications"] = deduplicate_specs(
                        normalized_spec["specifications"]
                    )
                spec_json = normalized_spec
                logger.debug(f"[Node 6] Normalized specs for {spec_json.get('product_type', 'unknown')}")
            except Exception as norm_err:
                logger.warning(f"[Node 6] Normalization warning: {norm_err}")
                # Continue with original spec_json if normalization fails

            if spec_json.get("item_type") == "instrument":
                instrument_specs.append(spec_json)
            else:
                accessory_specs.append(spec_json)

        # Build final Standard Specifications JSON
        standard_specifications_json = {
            "project_name": f"{user_context.get('process_type', 'Industrial')} Instrumentation",
            "domain": user_context.get("domain", "Industrial"),
            "process_type": user_context.get("process_type", "General Process"),
            "industry": user_context.get("industry", "Industrial"),
            "created_at": datetime.now().isoformat(),

            "user_requirements": {
                "raw_input": state["user_input"],
                "safety_requirements": user_context.get("safety_requirements", {}),
                "environmental_conditions": user_context.get("environmental_conditions", {}),
                "key_parameters": user_context.get("key_parameters", {})
            },

            "instruments": instrument_specs,
            "accessories": accessory_specs,

            "metadata": {
                "total_instruments": len(instrument_specs),
                "total_accessories": len(accessory_specs),
                "documents_analyzed": memory.documents_loaded,
                "sections_analyzed": memory.sections_analyzed,
                "threads_executed": len(thread_results),
                "threads_completed": sum(1 for r in thread_results.values() if r.get("status") == "completed"),
                "processing_time_ms": int((time.time() - state["start_time"]) * 1000),
                "llm_model": "gemini-2.5-flash",
                "session_id": state["session_id"]
            }
        }

        state["standard_specifications_json"] = standard_specifications_json
        state["steps_completed"].append("aggregate_results")

        logger.info(
            f"[Node 6] Aggregated: {len(instrument_specs)} instruments, "
            f"{len(accessory_specs)} accessories"
        )

    except Exception as e:
        logger.error(f"[Node 6] Error aggregating results: {e}", exc_info=True)
        state["error"] = f"Result aggregation failed: {str(e)}"

    return state


# =============================================================================
# NODE 7: FORMAT OUTPUT
# =============================================================================

def format_output_node(state: DeepAgentState) -> DeepAgentState:
    """
    Node 7: Format final output and response.
    """
    logger.info("=" * 70)
    logger.info("[Node 7] FORMAT OUTPUT")
    logger.info("=" * 70)

    state["current_step"] = "format_output"

    try:
        # Calculate processing time
        state["processing_time_ms"] = int((time.time() - state["start_time"]) * 1000)

        # Build response
        specs_json = state["standard_specifications_json"]

        response_parts = [
            f"**Standard Specifications Generated**",
            f"",
            f"**Project:** {specs_json.get('project_name', 'N/A')}",
            f"**Domain:** {specs_json.get('domain', 'N/A')}",
            f"**Process:** {specs_json.get('process_type', 'N/A')}",
            f"",
            f"**Instruments:** {len(specs_json.get('instruments', []))}",
            f"**Accessories:** {len(specs_json.get('accessories', []))}",
            f"",
            f"**Documents Analyzed:** {specs_json.get('metadata', {}).get('documents_analyzed', 0)}",
            f"**Processing Time:** {state['processing_time_ms']}ms",
        ]

        state["response"] = "\n".join(response_parts)

        state["response_data"] = {
            "success": True,
            "standard_specifications_json": specs_json,
            "processing_time_ms": state["processing_time_ms"],
            "session_id": state["session_id"]
        }

        state["steps_completed"].append("format_output")

        logger.info(f"[Node 7] Output formatted. Processing time: {state['processing_time_ms']}ms")

    except Exception as e:
        logger.error(f"[Node 7] Error formatting output: {e}", exc_info=True)
        state["error"] = f"Output formatting failed: {str(e)}"
        state["response_data"] = {
            "success": False,
            "error": str(e)
        }

    return state


# =============================================================================
# ERROR HANDLING NODE
# =============================================================================

def handle_error_node(state: DeepAgentState) -> DeepAgentState:
    """Handle errors in the workflow."""
    logger.error(f"[Error Handler] Error occurred: {state.get('error')}")

    state["processing_time_ms"] = int((time.time() - state["start_time"]) * 1000)

    state["response"] = f"Error: {state.get('error', 'Unknown error occurred')}"
    state["response_data"] = {
        "success": False,
        "error": state.get("error"),
        "steps_completed": state.get("steps_completed", []),
        "processing_time_ms": state["processing_time_ms"]
    }

    return state


# =============================================================================
# ROUTING FUNCTIONS
# =============================================================================

def should_continue(state: DeepAgentState) -> Literal["continue", "error"]:
    """Check if workflow should continue or handle error."""
    if state.get("error"):
        return "error"
    return "continue"


# =============================================================================
# WORKFLOW GRAPH CONSTRUCTION
# =============================================================================

def create_deep_agent_workflow():
    """
    Create the Deep Agent workflow graph.

    Graph structure:
    START → load_documents → analyze_user_input → identify_items
                                                       ↓
                                              create_execution_plan
                                                       ↓
                                              execute_threads
                                                       ↓
                                              aggregate_results
                                                       ↓
                                              format_output → END

    (Error handling at each step → handle_error → END)
    """
    logger.info("Creating Deep Agent workflow...")

    # Create graph
    workflow = StateGraph(DeepAgentState)

    # Add nodes
    workflow.add_node("load_documents", load_documents_node)
    workflow.add_node("analyze_user_input", analyze_user_input_node)
    workflow.add_node("identify_items", identify_items_node)
    workflow.add_node("create_execution_plan", create_execution_plan_node)
    workflow.add_node("execute_threads", execute_threads_node)
    workflow.add_node("aggregate_results", aggregate_results_node)
    workflow.add_node("format_output", format_output_node)
    workflow.add_node("handle_error", handle_error_node)

    # Set entry point
    workflow.set_entry_point("load_documents")

    # Add conditional edges with error checking
    workflow.add_conditional_edges(
        "load_documents",
        should_continue,
        {"continue": "analyze_user_input", "error": "handle_error"}
    )

    workflow.add_conditional_edges(
        "analyze_user_input",
        should_continue,
        {"continue": "identify_items", "error": "handle_error"}
    )

    workflow.add_conditional_edges(
        "identify_items",
        should_continue,
        {"continue": "create_execution_plan", "error": "handle_error"}
    )

    workflow.add_conditional_edges(
        "create_execution_plan",
        should_continue,
        {"continue": "execute_threads", "error": "handle_error"}
    )

    workflow.add_conditional_edges(
        "execute_threads",
        should_continue,
        {"continue": "aggregate_results", "error": "handle_error"}
    )

    workflow.add_conditional_edges(
        "aggregate_results",
        should_continue,
        {"continue": "format_output", "error": "handle_error"}
    )

    # Final edges
    workflow.add_edge("format_output", END)
    workflow.add_edge("handle_error", END)

    # Compile
    app = compile_with_checkpointing(workflow)

    logger.info("Deep Agent workflow created successfully")

    return app


# =============================================================================
# WORKFLOW EXECUTION
# =============================================================================

# Singleton workflow instance
_deep_agent_app = None


def get_deep_agent_workflow():
    """Get or create the Deep Agent workflow instance (singleton)."""
    global _deep_agent_app
    if _deep_agent_app is None:
        _deep_agent_app = create_deep_agent_workflow()
    return _deep_agent_app


def run_deep_agent_workflow(
    user_input: str,
    session_id: Optional[str] = None,
    identified_instruments: Optional[List[Dict[str, Any]]] = None,
    identified_accessories: Optional[List[Dict[str, Any]]] = None,
    cleanup_after: bool = True
) -> Dict[str, Any]:
    """
    Run the Deep Agent workflow.

    Args:
        user_input: User's requirements/input
        session_id: Optional session ID
        identified_instruments: Optional pre-identified instruments
        identified_accessories: Optional pre-identified accessories
        cleanup_after: Whether to clean up memory after completion

    Returns:
        Final result with standard_specifications_json
    """
    logger.info("=" * 80)
    logger.info("DEEP AGENT WORKFLOW - STARTING")
    logger.info("=" * 80)
    logger.info(f"User input: {user_input[:100]}...")

    # Create initial state
    final_session_id = session_id or f"deep-agent-{int(time.time())}"
    initial_state = create_deep_agent_state(
        user_input=user_input,
        session_id=final_session_id,
        identified_instruments=identified_instruments,
        identified_accessories=identified_accessories
    )

    # Get workflow
    app = get_deep_agent_workflow()

    # Execute workflow
    try:
        config = {"configurable": {"thread_id": final_session_id}}
        final_state = app.invoke(initial_state, config=config)

        logger.info("=" * 80)
        logger.info("DEEP AGENT WORKFLOW - COMPLETED")
        logger.info(f"Processing time: {final_state.get('processing_time_ms', 0)}ms")
        logger.info("=" * 80)

        return final_state

    except Exception as e:
        logger.error(f"Deep Agent workflow error: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "response_data": {
                "success": False,
                "error": str(e)
            }
        }
    finally:
        # Clean up memory
        if cleanup_after:
            clear_memory(final_session_id)
            logger.info(f"Memory cleaned up for session: {final_session_id}")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "DeepAgentState",
    "create_deep_agent_state",
    "create_deep_agent_workflow",
    "get_deep_agent_workflow",
    "run_deep_agent_workflow",
    "get_memory",
    "get_or_create_memory",
    "clear_memory",
]
