# agentic/deep_agent_schema_populator.py
# =============================================================================
# DEEP AGENT SCHEMA POPULATOR - Wrapper Module
# =============================================================================
#
# This module provides the missing `deep_agent_schema_populator` import that
# was causing [FIX5] Module not found errors.
#
# It wraps the SchemaGenerationDeepAgent to provide a simple interface for
# populating schema fields using the deep agent with failure memory.
#
# =============================================================================

import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# =============================================================================
# LEARNING ENGINE TOGGLE - Set to False to disable learning components
# =============================================================================
LEARNING_ENGINE_ENABLED = False  # DISABLED - Set to True to enable learning


def populate_schema_with_deep_agent(
    product_type: str,
    schema: Dict[str, Any],
    session_id: Optional[str] = None,
    use_memory: bool = True
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Populate schema fields using the Deep Agent with failure memory.

    This is the main entry point called by validation_tool.py.

    Args:
        product_type: Product type for schema population
        schema: Existing schema to populate
        session_id: Session identifier for tracking
        use_memory: Whether to use failure memory for learning

    Returns:
        Tuple of (populated_schema, population_stats)
    """
    logger.info(
        f"\n{'='*70}\n"
        f"[DEEP_AGENT_POPULATOR] Starting schema population\n"
        f"   Product: {product_type}\n"
        f"   Fields in schema: {_count_schema_fields(schema)}\n"
        f"{'='*70}\n"
    )

    try:
        from agentic.deep_agent.schema_generation_deep_agent import (
            get_schema_generation_deep_agent,
            generate_schema_with_deep_agent
        )

        # Get the deep agent
        agent = get_schema_generation_deep_agent()

        # Extract existing user specs from schema
        user_specs = _extract_existing_values(schema)

        # Generate/populate schema using deep agent
        result = agent.generate_schema(
            product_type=product_type,
            user_specs=user_specs,
            session_id=session_id,
            skip_cache=False  # Use cache if available
        )

        if result.success:
            # Merge generated specs into existing schema
            populated_schema = _merge_specs_into_schema(
                schema, result.specifications
            )

            stats = {
                "success": True,
                "fields_populated": result.spec_count,
                "total_fields": _count_schema_fields(populated_schema),
                "sources_used": result.sources_used,
                "source_contributions": result.source_contributions,
                "duration_ms": result.total_duration_ms,
                "from_cache": result.from_cache,
                "recovery_actions": result.recovery_actions
            }

            logger.info(
                f"\n{'='*70}\n"
                f"[DEEP_AGENT_POPULATOR] Schema population COMPLETED\n"
                f"   Fields populated: {result.spec_count}\n"
                f"   Sources: {', '.join(result.sources_used)}\n"
                f"   Duration: {result.total_duration_ms}ms\n"
                f"{'='*70}\n"
            )

            return populated_schema, stats

        else:
            logger.warning(
                f"[DEEP_AGENT_POPULATOR] Schema generation failed: {result.error}"
            )
            return schema, {
                "success": False,
                "error": result.error,
                "fields_populated": 0
            }

    except ImportError as e:
        logger.error(f"[DEEP_AGENT_POPULATOR] Import error: {e}")
        # Fallback to basic population
        return _fallback_populate(product_type, schema)

    except Exception as e:
        logger.error(f"[DEEP_AGENT_POPULATOR] Error: {e}")
        return schema, {
            "success": False,
            "error": str(e),
            "fields_populated": 0
        }


def populate_schema_fields_with_standards(
    product_type: str,
    schema: Dict[str, Any],
    fields: List[str],
    session_id: Optional[str] = None
) -> Tuple[Dict[str, Any], int]:
    """
    Populate specific schema fields using standards RAG.

    Args:
        product_type: Product type
        schema: Schema to populate
        fields: Specific fields to populate
        session_id: Session identifier

    Returns:
        Tuple of (populated_schema, fields_populated_count)
    """
    try:
        from agentic.deep_agent.adaptive_prompt_engine import get_adaptive_prompt_engine
        from agentic.standards_rag.standards_rag_workflow import run_standards_rag_workflow

        prompt_engine = get_adaptive_prompt_engine()

        # Get optimized prompt for these fields
        prompt, optimization = prompt_engine.get_schema_population_prompt(
            product_type=product_type,
            fields=fields
        )

        logger.info(
            f"[DEEP_AGENT_POPULATOR] Using {optimization.strategy.value} strategy "
            f"for {len(fields)} fields"
        )

        # Query standards RAG
        result = run_standards_rag_workflow(
            question=prompt,
            session_id=session_id or f"populate_{product_type}",
            memory=None
        )

        if result.get("success"):
            # Extract values from response
            values = _extract_values_from_response(result.get("answer", ""))

            # Update schema with extracted values
            populated_count = 0
            for field in fields:
                if field in values and values[field]:
                    _set_field_value(schema, field, values[field])
                    populated_count += 1

            # Report result to prompt engine for learning
            prompt_engine.report_prompt_result(
                product_type=product_type,
                prompt=prompt,
                success=True,
                fields_populated=populated_count,
                total_fields=len(fields)
            )

            return schema, populated_count
        else:
            return schema, 0

    except Exception as e:
        logger.error(f"[DEEP_AGENT_POPULATOR] Field population error: {e}")
        return schema, 0


def get_population_stats(product_type: str) -> Dict[str, Any]:
    """
    Get population statistics for a product type.

    Args:
        product_type: Product type to get stats for

    Returns:
        Statistics dictionary
    """
    try:
        from agentic.deep_agent.schema_failure_memory import get_schema_failure_memory

        memory = get_schema_failure_memory()
        return memory.get_product_type_summary(product_type)

    except Exception as e:
        logger.error(f"[DEEP_AGENT_POPULATOR] Stats error: {e}")
        return {"error": str(e)}


def predict_population_success(product_type: str) -> Dict[str, Any]:
    """
    Predict the likelihood of successful schema population.

    Args:
        product_type: Product type to predict for

    Returns:
        Prediction with risk level and recommendations
    """
    try:
        from agentic.deep_agent.schema_failure_memory import get_schema_failure_memory

        memory = get_schema_failure_memory()
        return memory.predict_failure_risk(product_type)

    except Exception as e:
        logger.error(f"[DEEP_AGENT_POPULATOR] Prediction error: {e}")
        return {"risk_level": "unknown", "error": str(e)}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _count_schema_fields(schema: Dict[str, Any]) -> int:
    """Count total fields in a schema."""
    count = 0

    for section in ["mandatory_requirements", "optional_requirements", "specifications"]:
        if section in schema and isinstance(schema[section], dict):
            count += len(schema[section])

    return count


def _extract_existing_values(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Extract existing non-empty values from schema."""
    values = {}

    for section in ["mandatory_requirements", "optional_requirements", "specifications"]:
        if section in schema and isinstance(schema[section], dict):
            for key, val in schema[section].items():
                if isinstance(val, dict):
                    value = val.get("value") or val.get("default")
                else:
                    value = val

                if value and str(value).strip() and str(value).lower() not in ["null", "none", "n/a"]:
                    values[key] = value

    return values


def _merge_specs_into_schema(
    schema: Dict[str, Any],
    specs: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge generated specs into existing schema structure."""
    # Initialize sections if not present
    if "mandatory_requirements" not in schema:
        schema["mandatory_requirements"] = {}
    if "optional_requirements" not in schema:
        schema["optional_requirements"] = {}
    if "specifications" not in schema:
        schema["specifications"] = {}

    # Common mandatory field patterns
    mandatory_patterns = [
        'accuracy', 'range', 'output', 'material', 'certification',
        'temperature', 'pressure', 'voltage', 'current', 'ip_rating'
    ]

    for key, value in specs.items():
        if not value or str(value).strip() == "":
            continue

        # Determine target section
        key_lower = key.lower().replace(' ', '_').replace('-', '_')
        is_mandatory = any(p in key_lower for p in mandatory_patterns)

        target_section = "mandatory_requirements" if is_mandatory else "optional_requirements"

        # Don't overwrite existing values
        if key in schema.get("mandatory_requirements", {}):
            existing = schema["mandatory_requirements"][key]
            if isinstance(existing, dict) and existing.get("value"):
                continue
        if key in schema.get("optional_requirements", {}):
            existing = schema["optional_requirements"][key]
            if isinstance(existing, dict) and existing.get("value"):
                continue

        # Add to schema
        schema[target_section][key] = {
            "value": value,
            "description": key.replace('_', ' ').title(),
            "source": "deep_agent"
        }

    # Also add to flat specifications
    schema["specifications"].update(specs)

    return schema


def _set_field_value(schema: Dict[str, Any], field: str, value: Any):
    """Set a field value in the schema."""
    for section in ["mandatory_requirements", "optional_requirements"]:
        if section in schema and field in schema[section]:
            if isinstance(schema[section][field], dict):
                schema[section][field]["value"] = value
            else:
                schema[section][field] = {"value": value}
            return

    # Add to optional if not found
    if "optional_requirements" not in schema:
        schema["optional_requirements"] = {}
    schema["optional_requirements"][field] = {"value": value}


def _extract_values_from_response(response: str) -> Dict[str, Any]:
    """Extract field values from LLM response."""
    import json
    import re

    values = {}

    # Try JSON parsing first
    try:
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            values = json.loads(json_match.group())
            return values
    except:
        pass

    # Fallback: extract key-value pairs
    patterns = [
        r'"?(\w+[\w\s]*?)"?\s*[:=]\s*"?([^"\n,]+)"?',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, response)
        for key, value in matches:
            key = key.strip().lower().replace(' ', '_')
            value = value.strip()
            if key and value and len(key) < 50:
                values[key] = value

    return values


def _fallback_populate(
    product_type: str,
    schema: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Fallback population using templates only."""
    try:
        from agentic.deep_agent.augmented_product_templates import AugmentedProductTemplates

        template = AugmentedProductTemplates.get_augmented_template(product_type)
        if template:
            specs = {}
            for key, spec in template.items():
                if hasattr(spec, 'default_value') and spec.default_value:
                    specs[key] = spec.default_value

            populated = _merge_specs_into_schema(schema, specs)
            return populated, {
                "success": True,
                "fields_populated": len(specs),
                "sources_used": ["template_fallback"]
            }

    except Exception as e:
        logger.error(f"[DEEP_AGENT_POPULATOR] Fallback error: {e}")

    return schema, {
        "success": False,
        "error": "Fallback failed",
        "fields_populated": 0
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "populate_schema_with_deep_agent",
    "populate_schema_fields_with_standards",
    "get_population_stats",
    "predict_population_success"
]
