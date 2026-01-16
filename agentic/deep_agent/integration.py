# agentic/deep_agent_integration.py
# =============================================================================
# DEEP AGENT INTEGRATION MODULE
# =============================================================================
#
# PURPOSE: Integration layer for using Deep Agent with Solution and Instrument
# Identifier workflows. Provides helper functions to run Deep Agent and populate
# SCHEMA FIELD VALUES from standards specifications.
#
# KEY FUNCTIONALITY:
# 1. Run Deep Agent to analyze standards documents
# 2. Extract specifications from standards for each product type
# 3. POPULATE SCHEMA FIELD VALUES from the extracted specifications
# 4. Map specifications to schema fields (mandatory and optional)
#
# USAGE:
#   from .deep_agent_integration import integrate_deep_agent_specifications
#   enriched_items = integrate_deep_agent_specifications(
#       all_items=state["all_items"],
#       user_input=state["user_input"],
#       solution_context=state.get("solution_analysis")
#   )
#
# =============================================================================

import json
import logging
import re
from typing import Dict, Any, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


# =============================================================================
# SCHEMA FIELD MAPPING
# =============================================================================

# Map Deep Agent specification keys to schema field names
SPEC_TO_SCHEMA_MAPPING = {
    # Pressure Transmitter
    "accuracy": ["accuracy", "Accuracy", "measurementAccuracy"],
    "repeatability": ["repeatability", "Repeatability"],
    "pressure_range": ["pressureRange", "pressure_range", "measurementRange", "range"],
    "measurementRange": ["pressureRange", "measurementRange", "range"],
    "output": ["outputSignal", "output_signal", "output", "communicationProtocol"],
    "outputSignal": ["outputSignal", "output_signal", "output"],
    "processConnection": ["processConnection", "process_connection", "connection"],
    "wettedParts": ["wettedParts", "wetted_parts", "material", "wettedMaterial"],

    # Temperature Sensor
    "temperature_range": ["temperatureRange", "temperature_range", "measurementRange"],
    "temperatureRange": ["temperatureRange", "temperature_range"],
    "sensorType": ["sensorType", "sensor_type", "type"],
    "responseTime": ["responseTime", "response_time"],
    "sheathMaterial": ["sheathMaterial", "sheath_material", "probeMaterial"],

    # Flow Meter
    "flowType": ["flowType", "flow_type", "measurementType"],
    "flowRange": ["flowRange", "flow_range", "measurementRange"],
    "pipeSize": ["pipeSize", "pipe_size", "lineSize"],
    "fluidType": ["fluidType", "fluid_type", "medium"],

    # Level Transmitter
    "measurementType": ["measurementType", "measurement_type", "technologyType"],
    "levelRange": ["measurementRange", "levelRange", "range"],

    # Safety & Certifications
    "silRating": ["safetyRating", "silRating", "sil_rating", "SIL"],
    "hazardousAreaRating": ["hazardousAreaRating", "atexRating", "hazardous_area", "ATEX"],
    "ingressProtection": ["ingressProtection", "ipRating", "IP"],
    "certifications": ["certifications", "certification", "approvals"],

    # Environmental
    "ambientTemperature": ["ambientTemperature", "ambient_temperature", "operatingTemperature"],
    "storageTemperature": ["storageTemperature", "storage_temperature"],

    # Communication
    "protocol": ["protocol", "communicationProtocol", "digital_protocol"],
    "powerSupply": ["powerSupply", "power_supply", "supplyVoltage"],
}


def _normalize_field_name(field_name: str) -> str:
    """Normalize field name for matching."""
    return field_name.lower().replace("_", "").replace("-", "").replace(" ", "")


def _find_schema_field_match(spec_key: str, schema_fields: List[str]) -> Optional[str]:
    """
    Find the matching schema field for a specification key.

    Args:
        spec_key: The specification key from Deep Agent
        schema_fields: List of available schema field names

    Returns:
        Matching schema field name or None
    """
    # Try direct mapping first
    if spec_key in SPEC_TO_SCHEMA_MAPPING:
        for possible_match in SPEC_TO_SCHEMA_MAPPING[spec_key]:
            if possible_match in schema_fields:
                return possible_match
            # Try normalized match
            normalized_possible = _normalize_field_name(possible_match)
            for schema_field in schema_fields:
                if _normalize_field_name(schema_field) == normalized_possible:
                    return schema_field

    # Try normalized direct match
    normalized_spec = _normalize_field_name(spec_key)
    for schema_field in schema_fields:
        if _normalize_field_name(schema_field) == normalized_spec:
            return schema_field

    # Try partial match
    for schema_field in schema_fields:
        normalized_schema = _normalize_field_name(schema_field)
        if normalized_spec in normalized_schema or normalized_schema in normalized_spec:
            return schema_field

    return None


def _extract_all_schema_fields(schema: Dict[str, Any]) -> List[str]:
    """
    Recursively extract all field names from a schema.

    Args:
        schema: Schema dictionary

    Returns:
        List of all field names
    """
    fields = []

    def extract_from_dict(obj: Any):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key.startswith("_") or key in ["standards", "normalized_category"]:
                    continue
                fields.append(key)
                if isinstance(value, dict):
                    extract_from_dict(value)

    # Extract from all sections
    for section in ["mandatory", "mandatory_requirements", "optional", "optional_requirements",
                    "Compliance", "Electrical", "Mechanical", "Performance", "Environmental", "Features"]:
        if section in schema:
            extract_from_dict(schema[section])

    # Also extract from root level
    extract_from_dict(schema)

    # Deduplicate
    return list(set(fields))


def _get_spec_value(spec_data: Any) -> Dict[str, Any]:
    """
    Extract the specification data with full metadata from a specification entry.
    Returns a dictionary with value, source, confidence, and standards_referenced.
    """
    from .schema_field_extractor import extract_standards_from_value
    
    if isinstance(spec_data, dict):
        value = spec_data.get("value", "")
        if not value and not spec_data.get("confidence"):
            # This is just a nested dict, convert it to string
            value = str(spec_data)
        
        # Extract standards from value if not already present
        existing_refs = spec_data.get("standards_referenced", [])
        if not existing_refs and value:
            existing_refs = extract_standards_from_value(str(value))
        
        return {
            "value": value if value else str(spec_data),
            "source": spec_data.get("source", "standards_specifications"),
            "confidence": spec_data.get("confidence", 0.8),
            "standards_referenced": existing_refs
        }
    
    # For plain string values
    value_str = str(spec_data)
    standards_refs = extract_standards_from_value(value_str)
    return {
        "value": value_str,
        "source": "standards_specifications",
        "confidence": 0.8,
        "standards_referenced": standards_refs
    }


# =============================================================================
# SCHEMA LOADING
# =============================================================================

def load_schema_for_product(product_type: str) -> Dict[str, Any]:
    """
    Load the schema for a product type.

    Args:
        product_type: Product type name

    Returns:
        Schema dictionary with mandatory and optional fields
    """
    try:
        from tools.schema_tools import load_schema_tool

        result = load_schema_tool.invoke({
            "product_type": product_type,
            "enable_ppi": False  # Don't generate new schema, use existing
        })

        if result.get("success") and result.get("schema"):
            logger.info(f"[DEEP_AGENT_INTEGRATION] Loaded schema for: {product_type}")
            return result["schema"]
        else:
            logger.warning(f"[DEEP_AGENT_INTEGRATION] No schema found for: {product_type}")
            return {}

    except Exception as e:
        logger.error(f"[DEEP_AGENT_INTEGRATION] Failed to load schema for {product_type}: {e}")
        return {}


# =============================================================================
# SCHEMA POPULATION
# =============================================================================

def populate_schema_from_deep_agent(
    product_type: str,
    schema: Dict[str, Any],
    deep_agent_specs: Dict[str, Any],
    applicable_standards: List[Dict[str, Any]] = None,
    certifications: List[str] = None
) -> Dict[str, Any]:
    """
    Populate schema field VALUES from Deep Agent specifications.

    This is the KEY function that fills in schema fields with values
    extracted from standards documents by the Deep Agent.

    Args:
        product_type: Product type
        schema: Schema with empty or partial field values
        deep_agent_specs: Specifications extracted by Deep Agent
        applicable_standards: List of applicable standards codes
        certifications: List of certifications

    Returns:
        Schema with populated field values from Deep Agent specifications
    """
    logger.info(f"[DEEP_AGENT_INTEGRATION] Populating schema fields for: {product_type}")

    if not schema:
        logger.warning("[DEEP_AGENT_INTEGRATION] Empty schema provided")
        return schema

    # Create deep copy
    populated_schema = json.loads(json.dumps(schema))

    # Get all available schema fields
    schema_fields = _extract_all_schema_fields(schema)
    logger.info(f"[DEEP_AGENT_INTEGRATION] Found {len(schema_fields)} schema fields")

    # Track populated fields
    fields_populated = 0

    # Get the specifications from Deep Agent
    # Handle both nested (mandatory/safety) and flat structure
    all_specs = {}
    if isinstance(deep_agent_specs, dict):
        # Extract from nested structures
        for key, value in deep_agent_specs.items():
            if isinstance(value, dict):
                # Nested section (like "mandatory", "safety", "optional")
                for sub_key, sub_value in value.items():
                    all_specs[sub_key] = sub_value
            else:
                all_specs[key] = value

    logger.info(f"[DEEP_AGENT_INTEGRATION] Processing {len(all_specs)} specifications from Deep Agent")

    # Populate schema fields from Deep Agent specifications
    def update_schema_section(section: Dict[str, Any], section_name: str):
        nonlocal fields_populated

        for field_key, field_value in list(section.items()):
            if field_key.startswith("_"):
                continue

            if isinstance(field_value, dict):
                # Recurse into nested dict
                update_schema_section(field_value, f"{section_name}.{field_key}")
            elif isinstance(field_value, str):
                # This is a field that might need a value
                # Check if value is empty or just a description
                if not field_value.strip() or field_value.lower() == "not specified":
                    # Try to find matching spec from Deep Agent
                    for spec_key, spec_value in all_specs.items():
                        schema_match = _find_schema_field_match(spec_key, [field_key])
                        if schema_match:
                            section[field_key] = _get_spec_value(spec_value)
                            fields_populated += 1
                            logger.debug(f"[DEEP_AGENT_INTEGRATION] Populated {field_key} = {section[field_key]}")
                            break

    # Update mandatory requirements
    for section_key in ["mandatory", "mandatory_requirements"]:
        if section_key in populated_schema and isinstance(populated_schema[section_key], dict):
            update_schema_section(populated_schema[section_key], section_key)

    # Update optional requirements
    for section_key in ["optional", "optional_requirements"]:
        if section_key in populated_schema and isinstance(populated_schema[section_key], dict):
            update_schema_section(populated_schema[section_key], section_key)

    # Update other sections (Compliance, Electrical, etc.)
    for section_key in ["Compliance", "Electrical", "Mechanical", "Performance", "Environmental", 
                        "Features", "Integration", "MechanicalOptions", "ServiceAndSupport", "Certifications"]:
        if section_key in populated_schema and isinstance(populated_schema[section_key], dict):
            update_schema_section(populated_schema[section_key], section_key)

    # =========================================================================
    # SECONDARY PASS: Use Schema Field Extractor for remaining empty fields
    # This ensures all fields get populated with standards-based values
    # =========================================================================
    try:
        from .schema_field_extractor import extract_schema_field_values_from_standards
        
        logger.info(f"[DEEP_AGENT_INTEGRATION] Running secondary extraction for remaining fields...")
        populated_schema = extract_schema_field_values_from_standards(
            product_type=product_type,
            schema=populated_schema
        )
        
        # Get secondary pass stats
        secondary_stats = populated_schema.get("_schema_field_extraction", {})
        secondary_fields = secondary_stats.get("fields_populated", 0)
        fields_populated += secondary_fields
        
        if secondary_fields > 0:
            logger.info(f"[DEEP_AGENT_INTEGRATION] Secondary extraction populated {secondary_fields} additional fields")
    except Exception as e:
        logger.warning(f"[DEEP_AGENT_INTEGRATION] Secondary extraction failed: {e}")

    # Add standards section if not present
    if applicable_standards and "standards" not in populated_schema:
        populated_schema["standards"] = {
            "applicable_standards": [
                s.get("code", s) if isinstance(s, dict) else s
                for s in applicable_standards
            ],
            "source": "deep_agent"
        }

    # Add certifications to schema
    if certifications:
        # Find appropriate place for certifications
        for section_key in ["optional", "optional_requirements", "Compliance", "Certifications"]:
            if section_key in populated_schema:
                if isinstance(populated_schema[section_key], dict):
                    if "certifications" not in populated_schema[section_key]:
                        populated_schema[section_key]["certifications"] = ", ".join(certifications)
                    # Also update Suggested Values if present
                    if "Suggested Values" in populated_schema[section_key]:
                        populated_schema[section_key]["Suggested Values"] = ", ".join(certifications)
                break

    # Add population metadata
    populated_schema["_deep_agent_population"] = {
        "product_type": product_type,
        "fields_populated": fields_populated,
        "total_specs_from_deep_agent": len(all_specs),
        "standards_count": len(applicable_standards) if applicable_standards else 0,
        "certifications_count": len(certifications) if certifications else 0,
        "source": "deep_agent_standards"
    }

    logger.info(f"[DEEP_AGENT_INTEGRATION] Populated {fields_populated} schema fields (Deep Agent + Standards Defaults)")

    return populated_schema


# =============================================================================
# INPUT PREPARATION
# =============================================================================

def prepare_deep_agent_input(
    all_items: List[Dict[str, Any]],
    user_input: str,
    solution_context: Optional[Dict[str, Any]] = None,
    domain: Optional[str] = None
) -> Dict[str, Any]:
    """
    Prepare input data for Deep Agent from identified items and context.

    Args:
        all_items: List of identified instruments and accessories
        user_input: User's original input/requirements
        solution_context: Solution analysis context (optional)
        domain: Domain/industry context (optional)

    Returns:
        Deep Agent input dictionary
    """
    logger.info("[DEEP_AGENT_INTEGRATION] Preparing Deep Agent input")

    # Extract safety requirements from solution context
    safety_requirements = {}
    environmental_conditions = {}

    if solution_context:
        safety_req = solution_context.get("safety_requirements", {})
        safety_requirements = {
            "sil_level": safety_req.get("sil_level"),
            "hazardous_area": safety_req.get("hazardous_area"),
            "atex_zone": safety_req.get("atex_zone")
        }

        env_context = solution_context.get("environmental", {})
        environmental_conditions = {
            "location": env_context.get("location"),
            "conditions": env_context.get("conditions"),
            "temperature_range": solution_context.get("key_parameters", {}).get("temperature_range"),
            "pressure_range": solution_context.get("key_parameters", {}).get("pressure_range")
        }

    # Convert all_items to Deep Agent format
    identified_items = []
    for item in all_items:
        item_data = {
            "product_type": item.get("name") or item.get("product_name", "Unknown"),
            "category": item.get("category", "Instrument" if item.get("type") == "instrument" else "Accessory"),
            "type": item.get("type", "instrument"),
            "quantity": item.get("quantity", 1),
            "sample_input": item.get("sample_input", ""),
            "specifications": item.get("specifications", {}),
            "purpose": item.get("purpose", "")
        }
        identified_items.append(item_data)

    deep_agent_input = {
        "user_input": user_input,
        "domain": domain or (solution_context.get("industry") if solution_context else "General"),
        "identified_items": identified_items,
        "safety_requirements": safety_requirements,
        "environmental_conditions": environmental_conditions,
        "solution_context": solution_context or {}
    }

    logger.info(f"[DEEP_AGENT_INTEGRATION] Prepared input with {len(identified_items)} items")
    return deep_agent_input


# =============================================================================
# MAIN INTEGRATION FUNCTION
# =============================================================================

def integrate_deep_agent_specifications(
    all_items: List[Dict[str, Any]],
    user_input: str,
    solution_context: Optional[Dict[str, Any]] = None,
    domain: Optional[str] = None,
    enable_schema_population: bool = True
) -> List[Dict[str, Any]]:
    """
    Run Deep Agent and extract technical specifications from standards documents.

    This function:
    1. Runs Deep Agent to analyze standards documents
    2. Extracts technical specifications for each product type
    3. Optionally loads schema and populates field values (if enable_schema_population=True)
    4. Returns items with extracted specifications

    Args:
        all_items: List of identified instruments and accessories
        user_input: User's original input/requirements
        solution_context: Solution analysis context (optional)
        domain: Domain/industry context (optional)
        enable_schema_population: If True, load schemas and populate fields.
                                  If False, only extract raw specifications
                                  (faster, for Solution Workflow).

    Returns:
        Enriched items list with extracted specifications (and optionally populated schemas)
    """
    logger.info(f"[DEEP_AGENT_INTEGRATION] Starting Deep Agent specification integration")
    logger.info(f"[DEEP_AGENT_INTEGRATION] Schema population: {'ENABLED' if enable_schema_population else 'DISABLED (specs-only mode)'}")

    try:
        # Import here to avoid circular imports
        from .workflow import run_deep_agent_workflow

        # Generate session ID for this Deep Agent execution
        session_id = f"integration_{uuid4().hex[:12]}"

        # Prepare input for Deep Agent
        deep_agent_input = prepare_deep_agent_input(
            all_items=all_items,
            user_input=user_input,
            solution_context=solution_context,
            domain=domain
        )

        # Convert identified_items to separate instruments and accessories lists
        identified_instruments = []
        identified_accessories = []
        for item in deep_agent_input.get("identified_items", []):
            if item.get("type") == "accessory":
                identified_accessories.append({
                    "category": item.get("category", "Accessory"),
                    "accessory_name": item.get("product_type", "Unknown Accessory"),
                    "quantity": item.get("quantity", 1),
                    "specifications": item.get("specifications", {}),
                    "related_instrument": item.get("related_instrument", "")
                })
            else:
                identified_instruments.append({
                    "category": item.get("category", "Instrument"),
                    "product_name": item.get("product_type", "Unknown Instrument"),
                    "quantity": item.get("quantity", 1),
                    "specifications": item.get("specifications", {}),
                    "sample_input": item.get("sample_input", "")
                })

        # Run Deep Agent workflow synchronously
        logger.info(f"[DEEP_AGENT_INTEGRATION] Running Deep Agent for session {session_id}")
        logger.info(f"[DEEP_AGENT_INTEGRATION] Items: {len(identified_instruments)} instruments, {len(identified_accessories)} accessories")
        final_state = run_deep_agent_workflow(
            user_input=deep_agent_input["user_input"],
            session_id=session_id,
            identified_instruments=identified_instruments,
            identified_accessories=identified_accessories
        )

        # Extract specifications from Deep Agent output
        # The Deep Agent workflow returns standard_specifications_json with instruments and accessories
        standard_specs_json = final_state.get("standard_specifications_json", {})
        
        # Combine instruments and accessories into a single list for iteration
        generated_specs = []
        if standard_specs_json:
            instrument_specs = standard_specs_json.get("instruments", [])
            accessory_specs = standard_specs_json.get("accessories", [])
            generated_specs = instrument_specs + accessory_specs
            logger.info(f"[DEEP_AGENT_INTEGRATION] Extracted {len(instrument_specs)} instrument specs and {len(accessory_specs)} accessory specs")
        else:
            # Fallback: try legacy format
            generated_specs = final_state.get("generated_specifications", [])
            logger.info(f"[DEEP_AGENT_INTEGRATION] Using legacy format: {len(generated_specs)} specs")

        # Merge specifications with original items AND populate schema fields
        enriched_items = []
        for i, item in enumerate(all_items):
            enriched_item = item.copy()
            product_type = item.get("name") or item.get("product_name", "Unknown")

            # Find corresponding spec from Deep Agent
            if i < len(generated_specs):
                deep_agent_spec = generated_specs[i]

                # Get Deep Agent specifications
                deep_specs = deep_agent_spec.get("specifications", {})
                applicable_standards = deep_agent_spec.get("applicable_standards", [])
                certifications = deep_agent_spec.get("certifications", [])
                guidelines = deep_agent_spec.get("guidelines", [])
                sources_used = deep_agent_spec.get("sources_used", [])

                # ==========================================================
                # CONDITIONAL: Load and populate schema only if enabled
                # For Solution Workflow, we skip this to save time
                # ==========================================================

                if enable_schema_population:
                    # Load schema for this product type
                    product_schema = load_schema_for_product(product_type)

                    if product_schema:
                        # POPULATE SCHEMA FIELD VALUES from Deep Agent specifications
                        populated_schema = populate_schema_from_deep_agent(
                            product_type=product_type,
                            schema=product_schema,
                            deep_agent_specs=deep_specs,
                            applicable_standards=applicable_standards,
                            certifications=certifications
                        )

                        # Store populated schema in item
                        enriched_item["schema"] = populated_schema
                        enriched_item["schema_populated"] = True

                        logger.info(
                            f"[DEEP_AGENT_INTEGRATION] Populated schema for {product_type}: "
                            f"{populated_schema.get('_deep_agent_population', {}).get('fields_populated', 0)} fields"
                        )
                    else:
                        # No schema found - store specifications directly
                        enriched_item["schema"] = {
                            "specifications_from_standards": deep_specs,
                            "_note": "No predefined schema found, using raw specifications"
                        }
                        enriched_item["schema_populated"] = False
                else:
                    # Schema population disabled - just store extracted specifications
                    # This is much faster and suitable for Solution Workflow
                    enriched_item["schema_populated"] = False
                    logger.debug(f"[DEEP_AGENT_INTEGRATION] Skipping schema population for {product_type} (disabled)")

                # Always store raw Deep Agent specifications for reference
                enriched_item["deep_agent_specifications"] = deep_specs
                enriched_item["applicable_standards"] = applicable_standards
                enriched_item["certifications"] = certifications
                enriched_item["guidelines"] = guidelines
                enriched_item["sources_used"] = sources_used
                enriched_item["enrichment_status"] = "success"

            else:
                enriched_item["enrichment_status"] = "no_match"
                enriched_item["schema_populated"] = False
                logger.warning(f"[DEEP_AGENT_INTEGRATION] No spec found for item {i}: {item.get('name')}")

            enriched_items.append(enriched_item)

        # Count successful enrichments
        specs_extracted = sum(1 for item in enriched_items if item.get("enrichment_status") == "success")
        schemas_populated = sum(1 for item in enriched_items if item.get("schema_populated", False))

        logger.info(
            f"[DEEP_AGENT_INTEGRATION] Successfully enriched {len(enriched_items)} items: "
            f"{specs_extracted} specs extracted, {schemas_populated} schemas populated"
        )

        return enriched_items

    except Exception as e:
        logger.error(f"[DEEP_AGENT_INTEGRATION] Deep Agent integration failed: {e}")
        import traceback
        traceback.print_exc()

        # Return items with error status
        for item in all_items:
            item["enrichment_status"] = "failed"
            item["enrichment_error"] = str(e)
            item["schema_populated"] = False

        return all_items


# =============================================================================
# ASYNC VERSION
# =============================================================================

async def run_deep_agent_for_specifications(
    all_items: List[Dict[str, Any]],
    user_input: str,
    solution_context: Optional[Dict[str, Any]] = None,
    domain: Optional[str] = None
) -> Dict[str, Any]:
    """
    Async version: Run Deep Agent workflow to generate specifications.

    Args:
        all_items: List of identified instruments and accessories
        user_input: User's original input/requirements
        solution_context: Solution analysis context (optional)
        domain: Domain/industry context (optional)

    Returns:
        Dictionary with enriched items and Deep Agent results
    """
    logger.info("[DEEP_AGENT_INTEGRATION] Starting async Deep Agent workflow")

    try:
        from .workflow import get_deep_agent_workflow, create_deep_agent_state

        session_id = f"integration_{uuid4().hex[:12]}"

        deep_agent_input = prepare_deep_agent_input(
            all_items=all_items,
            user_input=user_input,
            solution_context=solution_context,
            domain=domain
        )

        # Convert identified_items to separate instruments and accessories lists
        identified_instruments = []
        identified_accessories = []
        for item in deep_agent_input.get("identified_items", []):
            if item.get("type") == "accessory":
                identified_accessories.append({
                    "category": item.get("category", "Accessory"),
                    "accessory_name": item.get("product_type", "Unknown Accessory"),
                    "quantity": item.get("quantity", 1),
                    "specifications": item.get("specifications", {}),
                    "related_instrument": item.get("related_instrument", "")
                })
            else:
                identified_instruments.append({
                    "category": item.get("category", "Instrument"),
                    "product_name": item.get("product_type", "Unknown Instrument"),
                    "quantity": item.get("quantity", 1),
                    "specifications": item.get("specifications", {}),
                    "sample_input": item.get("sample_input", "")
                })

        # Create initial state
        initial_state = create_deep_agent_state(
            user_input=deep_agent_input["user_input"],
            session_id=session_id,
            identified_instruments=identified_instruments,
            identified_accessories=identified_accessories
        )

        workflow = get_deep_agent_workflow()

        logger.info(f"[DEEP_AGENT_INTEGRATION] Running async Deep Agent for session {session_id}")
        final_state = await workflow.ainvoke(
            initial_state,
            {"recursion_limit": 100, "configurable": {"thread_id": session_id}}
        )

        # Extract specifications from the correct key
        standard_specs_json = final_state.get("standard_specifications_json", {})
        specifications = []
        if standard_specs_json:
            specifications = standard_specs_json.get("instruments", []) + standard_specs_json.get("accessories", [])

        return {
            "success": True,
            "session_id": session_id,
            "specifications": specifications,
            "aggregated_specs": standard_specs_json,
            "thread_results": final_state.get("thread_results", {}),
            "processing_time_ms": final_state.get("processing_time_ms", 0),
            "documents_analyzed": standard_specs_json.get("metadata", {}).get("documents_analyzed", 0)
        }

    except Exception as e:
        logger.error(f"[DEEP_AGENT_INTEGRATION] Async Deep Agent workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "specifications": [],
            "aggregated_specs": {}
        }


# =============================================================================
# DISPLAY FORMATTING
# =============================================================================

def format_deep_agent_specs_for_display(
    items_with_specs: List[Dict[str, Any]]
) -> str:
    """
    Format Deep Agent specifications for user display.

    Args:
        items_with_specs: Items enriched with Deep Agent specifications

    Returns:
        Formatted markdown string for display
    """
    lines = []
    lines.append("## Deep Agent Specifications Analysis\n")

    for i, item in enumerate(items_with_specs, 1):
        lines.append(f"### {i}. {item.get('name', 'Unknown')}")
        lines.append(f"   **Type:** {item.get('type', 'Unknown').capitalize()}")
        lines.append(f"   **Category:** {item.get('category', 'Unknown')}")
        lines.append(f"   **Quantity:** {item.get('quantity', 1)}")
        lines.append(f"   **Schema Populated:** {'Yes' if item.get('schema_populated') else 'No'}\n")

        # Show populated schema
        schema = item.get("schema", {})
        if schema and item.get("schema_populated"):
            population_info = schema.get("_deep_agent_population", {})
            lines.append(f"   **Schema Fields Populated:** {population_info.get('fields_populated', 0)}")

            # Show mandatory requirements
            mandatory = schema.get("mandatory_requirements", schema.get("mandatory", {}))
            if mandatory:
                lines.append("   **Mandatory Requirements (from standards):**")
                for key, value in mandatory.items():
                    if not key.startswith("_") and value:
                        lines.append(f"   - {key}: {value}")

            # Show optional requirements
            optional = schema.get("optional_requirements", schema.get("optional", {}))
            if optional:
                lines.append("   **Optional Requirements (from standards):**")
                for key, value in list(optional.items())[:5]:
                    if not key.startswith("_") and value:
                        lines.append(f"   - {key}: {value}")
            lines.append("")

        # Show applicable standards
        standards = item.get("applicable_standards", [])
        if standards:
            codes = [s.get("code", s) if isinstance(s, dict) else s for s in standards[:5]]
            lines.append(f"   **Applicable Standards:** {', '.join(codes)}")
            if len(standards) > 5:
                lines.append(f"   ... and {len(standards) - 5} more")
            lines.append("")

        # Show certifications
        certs = item.get("certifications", [])
        if certs:
            lines.append(f"   **Certifications:** {', '.join(certs[:5])}")
            lines.append("")

        # Show guidelines
        guidelines = item.get("guidelines", [])
        if guidelines:
            lines.append(f"   **Guidelines:** {len(guidelines)} items available")
            for j, guideline in enumerate(guidelines[:3], 1):
                guideline_str = str(guideline)[:80]
                lines.append(f"   {j}. {guideline_str}...")
            if len(guidelines) > 3:
                lines.append(f"   ... and {len(guidelines) - 3} more")
            lines.append("")

        lines.append("")

    return "\n".join(lines)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_schema_population_stats(enriched_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get statistics about schema field population.

    Args:
        enriched_items: Items enriched with Deep Agent specifications

    Returns:
        Dictionary with population statistics
    """
    total_items = len(enriched_items)
    schemas_populated = sum(1 for item in enriched_items if item.get("schema_populated", False))
    total_fields_populated = 0
    total_standards = 0
    total_certifications = 0

    for item in enriched_items:
        schema = item.get("schema", {})
        population_info = schema.get("_deep_agent_population", {})
        total_fields_populated += population_info.get("fields_populated", 0)
        total_standards += len(item.get("applicable_standards", []))
        total_certifications += len(item.get("certifications", []))

    return {
        "total_items": total_items,
        "schemas_populated": schemas_populated,
        "population_rate": (schemas_populated / total_items * 100) if total_items > 0 else 0,
        "total_fields_populated": total_fields_populated,
        "avg_fields_per_item": total_fields_populated / total_items if total_items > 0 else 0,
        "total_standards_codes": total_standards,
        "total_certifications": total_certifications
    }
