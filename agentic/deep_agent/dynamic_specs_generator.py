# agentic/deep_agent/dynamic_specs_generator.py
# =============================================================================
# DYNAMIC SPECIFICATION GENERATOR WITH DEEP REASONING
# =============================================================================
#
# Uses a 2-phase deep reasoning approach:
# Phase 1: DISCOVER relevant specification keys for the product type
# Phase 2: GENERATE values for the discovered keys
#
# This eliminates the need for hardcoded PRODUCT_TYPE_SCHEMA_FIELDS and
# works for ANY product type dynamically.
#
# =============================================================================

import logging
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

from llm_fallback import create_llm_with_fallback

load_dotenv()
logger = logging.getLogger(__name__)

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Use Gemini 3 Pro for deep reasoning (better at discovery)
REASONING_MODEL = "gemini-2.5-pro"  # For Phase 1: Key Discovery
GENERATION_MODEL = "gemini-2.5-flash"  # For Phase 2: Value Generation

# Singleton instances
_reasoning_llm = None
_generation_llm = None
_llm_lock = threading.Lock()


def _get_reasoning_llm():
    """Get singleton reasoning LLM (Gemini Pro for deep thinking)."""
    global _reasoning_llm
    if _reasoning_llm is None:
        with _llm_lock:
            if _reasoning_llm is None:
                logger.info("[DYNAMIC_SPECS] Initializing reasoning LLM (Gemini 2.5 Pro)...")
                _reasoning_llm = create_llm_with_fallback(
                    model=REASONING_MODEL,
                    temperature=0.2,  # Low for precise reasoning
                    google_api_key=os.getenv("GOOGLE_API_KEY")
                )
    return _reasoning_llm


def _get_generation_llm():
    """Get singleton generation LLM (Gemini Flash for fast generation)."""
    global _generation_llm
    if _generation_llm is None:
        with _llm_lock:
            if _generation_llm is None:
                logger.info("[DYNAMIC_SPECS] Initializing generation LLM (Gemini 2.5 Flash)...")
                _generation_llm = create_llm_with_fallback(
                    model=GENERATION_MODEL,
                    temperature=0.3,
                    google_api_key=os.getenv("GOOGLE_API_KEY")
                )
    return _generation_llm


# =============================================================================
# PHASE 1: SPECIFICATION KEY DISCOVERY (Deep Reasoning)
# =============================================================================

DISCOVERY_PROMPT = """You are an industrial instrumentation expert with deep knowledge of technical specifications.

TASK: Generate a COMPREHENSIVE list of 60+ relevant technical specification keys for this product type.
The goal is to have as many specification options as possible to create rich product databases.

PRODUCT TYPE: {product_type}
CATEGORY: {category}
CONTEXT: {context}

=== MANDATORY MINIMUM: 60+ SPECIFICATION KEYS ===

You MUST generate AT LEAST 60 distinct specification keys. This is not optional.
- If you generate fewer than 60 keys, your response is incomplete.
- Aim for 70-100+ keys for maximum database richness.
- Include both common and specialized specs.

=== DEEP REASONING INSTRUCTIONS ===

Think step by step:
1. What IS this product? (Function, purpose, typical applications)
2. What technical parameters define this product's PERFORMANCE? (accuracy, range, response, etc.)
3. What safety/compliance specs are REQUIRED? (certifications, standards, approvals)
4. What physical/mechanical specs MATTER? (size, weight, materials, connections)
5. What electrical/signal specs are RELEVANT? (voltage, current, protocols, impedance)
6. What environmental specs APPLY? (temperature, humidity, vibration, pressure)
7. What material specs AFFECT performance? (wetted materials, seals, coatings, hardness)
8. What maintenance/calibration specs are NEEDED? (intervals, procedures, warranty)
9. What integration/interface specs are REQUIRED? (connections, adapters, compatibility)
10. What operational/functional specs are IMPORTANT? (speed, capacity, throughput, efficiency)
11. What testing/validation specs matter? (test conditions, calibration points, verification)
12. What special features/options should be captured?

=== COMPREHENSIVE SPECIFICATION CATEGORIES ===

PERFORMANCE SPECS (15-20 keys):
- accuracy, repeatability, linearity, hysteresis, precision, conformity
- measurement_range, span, rangeability, turndown_ratio, max_range, min_range
- resolution, sensitivity, drift, stability, response_time, settling_time
- bandwidth, frequency_response, phase_shift, damping

MEASUREMENT & RANGE SPECS (8-10 keys):
- measurement_units, measurement_type, measurement_principle
- full_scale_deflection, zero_offset, zero_adjustment_range
- span_adjustment, measuring_range_low, measuring_range_high

ELECTRICAL SPECS (15-20 keys):
- output_signal, output_type, signal_levels, signal_range
- supply_voltage, supply_voltage_range, voltage_tolerance, supply_current
- power_consumption, power_rating, max_power_draw, idle_power
- output_impedance, input_impedance, loop_resistance, cable_length_max
- communication_protocol, baudrate, data_format, transmission_speed
- isolation_voltage, isolation_type, grounding_requirements
- EMC_compliance, emc_immunity, emission_level

PHYSICAL & MECHANICAL SPECS (15-20 keys):
- process_connection, connection_type, thread_size, flange_type
- mounting_type, mounting_orientation, installation_method
- dimensions_length, dimensions_width, dimensions_height, overall_dimensions
- weight, weight_without_packaging, material_body, material_housing
- insertion_length, bore_size, face_seal_diameter
- vibration_resistance, shock_resistance, mechanical_strength
- exterior_finish, surface_treatment, corrosion_protection

MATERIAL SPECS (12-15 keys):
- material_wetted, material_housing, material_seal, material_gasket
- diaphragm_material, fill_fluid, fill_material, elastomer_type
- coating_type, coating_thickness, surface_hardness, corrosion_grade
- material_compatibility, chemical_resistance

ENVIRONMENTAL & OPERATIONAL SPECS (12-15 keys):
- temperature_range, ambient_temperature_range, process_temperature_range
- temperature_storage, temperature_operating_min, temperature_operating_max
- humidity_range, humidity_operating, humidity_storage, moisture_resistance
- protection_rating, degree_of_protection, ip_rating, enclosure_type
- vibration_frequency, vibration_amplitude, vibration_tolerance
- altitude_rating, atmospheric_pressure_range
- dust_resistance, splash_resistance, submersion_rating

SAFETY & COMPLIANCE SPECS (10-15 keys):
- hazardous_area_approval, atex_certification, iec_classification
- sil_rating, pl_rating, safety_integrity_level
- pressure_rating, burst_pressure, working_pressure, max_pressure_differential
- safety_factor, proof_pressure, maximum_safe_pressure
- certifications, standards_compliance, regulatory_approvals
- hazard_warning, safety_markings, accident_prevention_rating

MAINTENANCE & RELIABILITY SPECS (10-12 keys):
- calibration_interval, recalibration_frequency, calibration_method
- service_life, mtbf, mttf, mean_time_between_failures
- warranty_period, warranty_coverage, maintenance_requirements
- cleaning_method, storage_requirements, spare_parts_availability
- preventive_maintenance_interval, expected_lifespan

INTEGRATION & COMPATIBILITY SPECS (8-12 keys):
- system_compatibility, software_compatibility, driver_availability
- interface_standard, connector_type, connector_pin_count
- protocol_support, multi_protocol, backward_compatibility
- gateway_compatibility, network_integration, api_support
- third_party_integration, middleware_support

FEATURE & FUNCTIONALITY SPECS (8-12 keys):
- adjustable_range, field_adjustable, remote_adjustment
- display_type, local_display, remote_display, display_resolution
- memory_storage, data_logging, data_retention, buffering_capability
- alarm_functions, alert_thresholds, warning_levels
- self_diagnosis, self_test, health_monitoring, diagnostic_output
- software_features, firmware_version, upgrade_capability

REFERENCE & DOCUMENTATION SPECS (5-8 keys):
- reference_standard, industry_standard, manufacturing_standard
- test_conditions, reference_conditions, ambient_conditions_for_specs
- traceability, calibration_certificate, documentation_provided
- manual_availability, technical_support, documentation_language

SPECIAL FEATURES (5-10 keys depending on product):
- hazardous_location_rating, sil_loop_capable, intrinsic_safety
- modular_design, expandability, scalability, future_proof_rating
- redundancy_support, fail_safe_mode, degradation_mode
- energy_efficiency, power_saving_mode, low_power_option
- wireless_capability, remote_monitoring, predictive_maintenance_ready

=== CRITICAL RULES ===

1. MANDATORY: Generate AT LEAST 60 specification keys
2. ONLY include specs that are RELEVANT to this specific product type
3. A Junction Box doesn't need "accuracy" but DOES need "connector_pin_count", "cable_length_max", "enclosure_type"
4. A Power Supply needs "supply_voltage", "power_rating", "protection_rating", "efficiency" but NOT "measurement_range"
5. A Thermowell needs "bore_size", "insertion_length", "pressure_rating", "material_wetted", "process_connection"
6. Be comprehensive - think about what an engineer would need to specify EVERY detail
7. Categorize keys into mandatory, optional, and safety_critical based on product type
8. PRIORITIZE BREADTH: 60+ keys is the minimum, not the target

=== OUTPUT FORMAT ===

Return ONLY valid JSON. IMPORTANT: Ensure total key count (mandatory + optional + safety_critical) is AT LEAST 60:
{{
    "product_analysis": {{
        "product_function": "What this product does",
        "primary_purpose": "Main use case",
        "typical_applications": ["app1", "app2"]
    }},
    "specification_keys": {{
        "mandatory": [
            {{"key": "spec_key_name", "description": "What this spec measures", "typical_format": "e.g., ±0.1%, -40 to +85°C"}}
        ],
        "optional": [
            {{"key": "spec_key_name", "description": "What this spec measures", "typical_format": "example"}}
        ],
        "safety_critical": [
            {{"key": "spec_key_name", "description": "What this spec measures", "typical_format": "example"}}
        ]
    }},
    "total_keys_generated": 60+,
    "discovery_confidence": 0.0-1.0,
    "reasoning_notes": "Brief explanation of key selections and justification for achieving 60+ keys"
}}
"""


def discover_specification_keys(
    product_type: str,
    category: Optional[str] = None,
    context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Phase 1: Use deep reasoning to discover relevant specification keys.
    
    This replaces hardcoded PRODUCT_TYPE_SCHEMA_FIELDS with dynamic discovery.
    """
    logger.info(f"[PHASE1:DISCOVERY] Discovering specs for: {product_type}")
    start_time = time.time()
    
    try:
        llm = _get_reasoning_llm()
        prompt = ChatPromptTemplate.from_template(DISCOVERY_PROMPT)
        parser = JsonOutputParser()
        chain = prompt | llm | parser
        
        result = chain.invoke({
            "product_type": product_type,
            "category": category or "Industrial Instrumentation",
            "context": context or "General industrial application"
        })
        
        elapsed = time.time() - start_time
        logger.info(f"[PHASE1:DISCOVERY] Completed in {elapsed:.2f}s")
        
        # Extract discovered keys
        spec_keys = result.get("specification_keys", {})
        mandatory = [item.get("key") for item in spec_keys.get("mandatory", [])]
        optional = [item.get("key") for item in spec_keys.get("optional", [])]
        safety = [item.get("key") for item in spec_keys.get("safety_critical", [])]
        
        logger.info(f"[PHASE1:DISCOVERY] Found {len(mandatory)} mandatory, {len(optional)} optional, {len(safety)} safety keys")
        
        return {
            "success": True,
            "product_type": product_type,
            "product_analysis": result.get("product_analysis", {}),
            "mandatory_keys": mandatory,
            "optional_keys": optional,
            "safety_keys": safety,
            "all_keys": mandatory + optional + safety,
            "key_details": spec_keys,
            "discovery_confidence": result.get("discovery_confidence", 0.8),
            "reasoning_notes": result.get("reasoning_notes", ""),
            "discovery_time_ms": int(elapsed * 1000)
        }
        
    except Exception as e:
        logger.error(f"[PHASE1:DISCOVERY] Failed: {e}")
        # Fallback to minimal generic keys
        return {
            "success": False,
            "product_type": product_type,
            "mandatory_keys": ["temperature_range", "material_housing", "certifications"],
            "optional_keys": ["weight", "protection_rating"],
            "safety_keys": ["hazardous_area_approval"],
            "all_keys": ["temperature_range", "material_housing", "certifications", 
                        "weight", "protection_rating", "hazardous_area_approval"],
            "error": str(e),
            "discovery_time_ms": int((time.time() - start_time) * 1000)
        }


# =============================================================================
# PHASE 2: SPECIFICATION VALUE GENERATION
# =============================================================================

GENERATION_PROMPT = """You are an industrial instrumentation expert.

TASK: Generate technical specification VALUES for ALL discovered keys.

PRODUCT TYPE: {product_type}
CATEGORY: {category}
CONTEXT: {context}

=== SPECIFICATION KEYS TO GENERATE VALUES FOR ===

MANDATORY (must provide values for ALL):
{mandatory_keys}

OPTIONAL (provide values for ALL applicable ones):
{optional_keys}

SAFETY CRITICAL (must provide values for ALL applicable ones):
{safety_keys}

=== VALUE FORMAT RULES ===

Return ONLY clean technical values - NO descriptions, NO explanations.

CORRECT: "±0.1%", "IP67", "4-20mA HART", "-40 to +85°C", "SIL 2", "316L SS", "1/4 NPT", "0-10V DC"
WRONG: "typically ±0.1%", "IP67 for outdoor use", "depends on application", "various options"

FORBIDDEN words in values: "typically", "usually", "approximately", "depends on", "varies", "optional", "may", "can be"

=== MANDATORY GENERATION RULES ===

1. Generate values for EVERY SINGLE KEY in the lists above
2. Do NOT skip keys - if a key is listed, you MUST provide a value
3. For keys that don't apply to this product, use "N/A" ONLY as last resort
4. Be specific to the product type - generate realistic, product-appropriate values
5. Provide confidence scores: 0.9+ for definitive specs, 0.7-0.8 for typical specs, 0.5-0.7 for variable specs
6. Each value must be complete and technical (not descriptive)

=== VALUE GENERATION EXAMPLES BY CATEGORY ===

PERFORMANCE SPECS:
- accuracy: "±0.1% of full scale" or "±0.25% of reading"
- repeatability: "±0.05% of span"
- linearity: "±0.1%"
- hysteresis: "±0.05% of range"
- resolution: "0.01 units" or "12 bits"
- response_time: "500 ms" or "<1 second"
- settling_time: "2 seconds"
- bandwidth: "0-100 Hz"

RANGE & MEASUREMENT:
- measurement_range: "0 to 100 bar" or "0-500°C"
- span: "100 bar"
- rangeability: "1:100"
- resolution: "0.1"
- measuring_range_low: "0"
- measuring_range_high: "100"
- measurement_units: "bar, psi, MPa"

ELECTRICAL:
- output_signal: "4-20mA" or "0-10V DC"
- supply_voltage: "24 VDC"
- power_consumption: "2 W max"
- communication_protocol: "HART, Modbus RTU"
- baudrate: "19200 bps"
- cable_length_max: "3000 m"
- isolation_voltage: "1500 V"
- EMC_compliance: "EN 61000-6-2, EN 61000-6-4"

PHYSICAL:
- process_connection: "1/2 NPT"
- mounting_type: "Flush diaphragm"
- dimensions_length: "45 mm"
- weight: "150 g"
- material_housing: "Stainless steel 316L"
- material_wetted: "304 SS"
- bore_size: "3/4 inch"

ENVIRONMENTAL:
- temperature_range: "-20 to +80°C"
- humidity_range: "0 to 95% RH"
- protection_rating: "IP67"
- vibration_resistance: "5G at 20-2000 Hz"
- altitude_rating: "0 to 3000 m"

SAFETY & COMPLIANCE:
- sil_rating: "SIL 2"
- hazardous_area_approval: "ATEX II 2G Ex db ib"
- pressure_rating: "16 bar"
- certifications: "CE, UL, CSA"
- standards_compliance: "EN 61508, IEC 61511"

MAINTENANCE:
- calibration_interval: "12 months"
- warranty_period: "2 years"
- mtbf: "50000 hours"
- service_life: "10 years"
- maintenance_requirements: "Annual inspection"

=== OUTPUT FORMAT ===

Return ONLY valid JSON with specifications for ALL provided keys:
{{
    "specifications": {{
        "key_name": {{"value": "technical value", "confidence": 0.8}},
        "another_key": {{"value": "technical value", "confidence": 0.9}},
        ... (for ALL mandatory, optional, and safety_critical keys provided)
    }},
    "generation_notes": "Brief note on assumptions made and any special considerations",
    "total_values_generated": number_of_specs_with_values
}}

RULES:
1. Generate values for EVERY key in the lists - this is mandatory
2. Provide at least one value for each key unless it's truly impossible
3. Use "N/A" ONLY if the specification is completely irrelevant to this product type
4. Each value must be specific and technical - no generic descriptions
5. High confidence (0.9) for standard industrial specifications, lower (0.5-0.7) if specialized
"""


def generate_specification_values(
    product_type: str,
    discovered_keys: Dict[str, Any],
    category: Optional[str] = None,
    context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Phase 2: Generate values for the discovered specification keys.
    """
    logger.info(f"[PHASE2:GENERATION] Generating values for: {product_type}")
    start_time = time.time()
    
    mandatory_keys = discovered_keys.get("mandatory_keys", [])
    optional_keys = discovered_keys.get("optional_keys", [])
    safety_keys = discovered_keys.get("safety_keys", [])
    
    try:
        llm = _get_generation_llm()
        prompt = ChatPromptTemplate.from_template(GENERATION_PROMPT)
        parser = JsonOutputParser()
        chain = prompt | llm | parser
        
        result = chain.invoke({
            "product_type": product_type,
            "category": category or "Industrial Instrumentation",
            "context": context or "General industrial application",
            "mandatory_keys": ", ".join(mandatory_keys) if mandatory_keys else "None",
            "optional_keys": ", ".join(optional_keys) if optional_keys else "None",
            "safety_keys": ", ".join(safety_keys) if safety_keys else "None"
        })
        
        elapsed = time.time() - start_time
        logger.info(f"[PHASE2:GENERATION] Completed in {elapsed:.2f}s")
        
        raw_specs = result.get("specifications", {})
        
        # Flatten and clean
        clean_specs = {}
        for key, value_data in raw_specs.items():
            if isinstance(value_data, dict):
                val = value_data.get("value", "")
                conf = value_data.get("confidence", 0.7)
            else:
                val = str(value_data)
                conf = 0.7
            
            # Skip empty/null values
            if val and str(val).lower() not in ["null", "none", "n/a", ""]:
                clean_specs[key] = {
                    "value": val,
                    "confidence": conf,
                    "source": "llm_generated"
                }
        
        logger.info(f"[PHASE2:GENERATION] Generated {len(clean_specs)} specification values")
        
        return {
            "success": True,
            "specifications": clean_specs,
            "generation_notes": result.get("generation_notes", ""),
            "generation_time_ms": int(elapsed * 1000)
        }
        
    except Exception as e:
        logger.error(f"[PHASE2:GENERATION] Failed: {e}")
        return {
            "success": False,
            "specifications": {},
            "error": str(e),
            "generation_time_ms": int((time.time() - start_time) * 1000)
        }


# =============================================================================
# COMBINED: DYNAMIC SPECIFICATION GENERATION
# =============================================================================

def generate_dynamic_specs(
    product_type: str,
    category: Optional[str] = None,
    context: Optional[str] = None,
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Complete 2-phase dynamic specification generation.
    
    Phase 1: Deep reasoning to DISCOVER relevant keys for this product type
    Phase 2: Generate VALUES for the discovered keys
    
    This replaces the hardcoded PRODUCT_TYPE_SCHEMA_FIELDS approach.
    """
    logger.info(f"[DYNAMIC_SPECS] Starting 2-phase generation for: {product_type}")
    total_start = time.time()
    
    # Phase 1: Discover relevant keys
    discovered = discover_specification_keys(
        product_type=product_type,
        category=category,
        context=context
    )
    
    if not discovered.get("success"):
        logger.warning(f"[DYNAMIC_SPECS] Discovery failed, using fallback keys")
    
    # Phase 2: Generate values for discovered keys
    generated = generate_specification_values(
        product_type=product_type,
        discovered_keys=discovered,
        category=category,
        context=context
    )
    
    total_elapsed = time.time() - total_start
    
    return {
        "success": discovered.get("success", False) and generated.get("success", False),
        "product_type": product_type,
        "category": category,
        
        # Discovery results
        "product_analysis": discovered.get("product_analysis", {}),
        "discovered_keys": {
            "mandatory": discovered.get("mandatory_keys", []),
            "optional": discovered.get("optional_keys", []),
            "safety_critical": discovered.get("safety_keys", [])
        },
        "discovery_confidence": discovered.get("discovery_confidence", 0.0),
        "reasoning_notes": discovered.get("reasoning_notes", ""),
        
        # Generation results
        "specifications": generated.get("specifications", {}),
        "generation_notes": generated.get("generation_notes", ""),
        
        # Timing
        "discovery_time_ms": discovered.get("discovery_time_ms", 0),
        "generation_time_ms": generated.get("generation_time_ms", 0),
        "total_time_ms": int(total_elapsed * 1000),
        
        # Metadata
        "source": "dynamic_llm_generated",
        "reasoning_model": REASONING_MODEL,
        "generation_model": GENERATION_MODEL,
        "timestamp": datetime.now().isoformat()
    }


def generate_dynamic_specs_batch(
    items: List[Dict[str, Any]],
    max_workers: int = 3
) -> List[Dict[str, Any]]:
    """
    Generate dynamic specs for multiple items in parallel.
    
    Uses ThreadPoolExecutor for concurrent processing.
    """
    if not items:
        return []
    
    logger.info(f"[DYNAMIC_SPECS] Batch processing {len(items)} items...")
    
    def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
        product_type = item.get("name") or item.get("product_name", "Unknown")
        category = item.get("category", "Industrial Instrumentation")
        context = item.get("sample_input", "")
        
        result = generate_dynamic_specs(
            product_type=product_type,
            category=category,
            context=context
        )
        
        result["item_name"] = product_type
        result["item_type"] = item.get("type", "instrument")
        return result
    
    results = []
    
    with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as executor:
        future_to_item = {
            executor.submit(process_item, item): item
            for item in items
        }
        
        for future in as_completed(future_to_item):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                item = future_to_item[future]
                item_name = item.get("name", "Unknown")
                logger.error(f"[DYNAMIC_SPECS] Batch item failed: {item_name}: {e}")
                results.append({
                    "success": False,
                    "product_type": item_name,
                    "specifications": {},
                    "error": str(e)
                })
    
    logger.info(f"[DYNAMIC_SPECS] Batch complete: {len(results)} items processed")
    return results


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "discover_specification_keys",
    "generate_specification_values",
    "generate_dynamic_specs",
    "generate_dynamic_specs_batch"
]
