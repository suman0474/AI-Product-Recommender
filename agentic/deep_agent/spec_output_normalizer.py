# agentic/deep_agent/spec_output_normalizer.py
# =============================================================================
# SPECIFICATION OUTPUT NORMALIZER
# =============================================================================
#
# Centralized normalization layer for all specification outputs.
# Transforms nested/complex LLM outputs into clean, flat key-value pairs.
#
# =============================================================================

import logging
import re
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)


# =============================================================================
# KEY NORMALIZATION MAPPINGS
# =============================================================================

# Standard key mappings - normalize various formats to snake_case
STANDARD_KEY_MAPPINGS = {
    # Common variations
    "silRating": "sil_rating",
    "sil_rating": "sil_rating",
    "Sil rating": "sil_rating",
    "SIL Rating": "sil_rating",
    "SIL_Rating": "sil_rating",
    
    "hazardousAreaRating": "hazardous_area_approval",
    "hazardous_area_rating": "hazardous_area_approval",
    "Hazardous area approval": "hazardous_area_approval",
    "hazardousAreaApproval": "hazardous_area_approval",
    
    "materialWetted": "material_wetted",
    "Material wetted": "material_wetted",
    "Material_wetted": "material_wetted",
    
    "materialHousing": "material_housing",
    "Material housing": "material_housing",
    "Material_housing": "material_housing",
    
    "processConnection": "process_connection",
    "Process connection": "process_connection",
    "Process_connection": "process_connection",
    
    "outputSignal": "output_signal",
    "Output signal": "output_signal",
    "Output_signal": "output_signal",
    
    "supplyVoltage": "supply_voltage",
    "Supply voltage": "supply_voltage",
    "Supply_voltage": "supply_voltage",
    
    "protectionRating": "protection_rating",
    "Protection rating": "protection_rating",
    "Protection_rating": "protection_rating",
    
    "temperatureRange": "temperature_range",
    "Temperature range": "temperature_range",
    "Temperature_range": "temperature_range",
    
    "pressureRange": "pressure_range",
    "Pressure range": "pressure_range",
    "Pressure_range": "pressure_range",
    
    "ambientTemperature": "ambient_temperature",
    "Ambient temperature": "ambient_temperature",
    "Ambient_temperature": "ambient_temperature",
    
    "processTemperature": "process_temperature",
    "Process temperature": "process_temperature",
    "Process_temperature": "process_temperature",
    
    "communicationProtocol": "communication_protocol",
    "Communication protocol": "communication_protocol",
    "Communication_protocol": "communication_protocol",
    
    "responseTime": "response_time",
    "Response time": "response_time",
    "Response_time": "response_time",
    
    "calibrationInterval": "calibration_interval",
    "Calibration interval": "calibration_interval",
    "Calibration_interval": "calibration_interval",
    
    "powerConsumption": "power_consumption",
    "Power consumption": "power_consumption",
    "Power_consumption": "power_consumption",
    
    "humidityRange": "humidity_range",
    "Humidity range": "humidity_range",
    "Humidity_range": "humidity_range",
    
    "measurementRange": "measurement_range",
    "Measurement range": "measurement_range",
    "Measurement_range": "measurement_range",
    
    "wakeFrequencyCalculation": "wake_frequency_calculation",
    "Wake frequency calculation": "wake_frequency_calculation",
    
    "shankStyle": "shank_style",
    "Shank style": "shank_style",
    
    "pressureRating": "pressure_rating",
    "Pressure rating": "pressure_rating",
    
    "terminalBlockType": "terminal_block_type",
    "Terminal block type": "terminal_block_type",
    
    "cableEntry": "cable_entry",
    "Cable entry": "cable_entry",
    
    "conductorGauge": "conductor_gauge",
    "Conductor gauge": "conductor_gauge",
    
    "conductorMaterial": "conductor_material",
    "Conductor material": "conductor_material",
    
    "insulationResistance": "insulation_resistance",
    "Insulation resistance": "insulation_resistance",
    
    "colorCoding": "color_coding",
    "Color coding": "color_coding",
    
    "pinMaterial": "pin_material",
    "Pin material": "pin_material",
    
    "terminalType": "terminal_type",
    "Terminal type": "terminal_type",
    
    "thermocoupleTypeCompatibility": "thermocouple_type_compatibility",
    "Thermocouple type compatibility": "thermocouple_type_compatibility",
}


# =============================================================================
# VALUE EXTRACTION PATTERNS
# =============================================================================

# Patterns to detect and extract actual values from descriptive text
DESCRIPTIVE_PREFIXES = [
    r"^N/A\s*\(typically\s+",
    r"^N/A\s*\(usually\s+",
    r"^typically\s+",
    r"^usually\s+",
    r"^generally\s+",
    r"^commonly\s+",
]

# Patterns that indicate the text is fully N/A
NA_PATTERNS = [
    r"^N/A$",
    r"^n/a$",
    r"^NA$",
    r"^not\s+applicable$",
    r"^none$",
    r"^null$",
    r"^-$",
    r"^\s*$",
]


# =============================================================================
# CORE NORMALIZATION FUNCTIONS
# =============================================================================

def normalize_key(key: str) -> str:
    """
    Normalize a specification key to snake_case.
    
    Args:
        key: The original key in any format
        
    Returns:
        Normalized snake_case key
    """
    # Check direct mapping first
    if key in STANDARD_KEY_MAPPINGS:
        return STANDARD_KEY_MAPPINGS[key]
    
    # Convert camelCase to snake_case
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', key)
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
    
    # Replace spaces and multiple underscores
    result = re.sub(r'[\s]+', '_', s2)
    result = re.sub(r'_+', '_', result)
    
    return result.lower().strip('_')


def extract_value_from_nested(data: Any, preserve_ghost: bool = True) -> tuple:
    """
    Extract the actual value from a nested structure.
    
    Args:
        data: The data which may be nested or flat
        preserve_ghost: If True, return confidence as ghost value
        
    Returns:
        Tuple of (extracted_value, ghost_metadata)
    """
    ghost_metadata = {}
    
    if data is None:
        return "N/A", ghost_metadata
    
    if isinstance(data, str):
        return data, ghost_metadata
    
    if isinstance(data, (int, float, bool)):
        return str(data), ghost_metadata
    
    if isinstance(data, dict):
        # Extract confidence and notes as ghost values
        if preserve_ghost:
            if "confidence" in data:
                ghost_metadata["_confidence"] = data["confidence"]
            if "note" in data:
                ghost_metadata["_note"] = data["note"]
            if "source" in data:
                ghost_metadata["_source"] = data["source"]
        
        # Look for value field
        if "value" in data:
            value = data["value"]
            if isinstance(value, str):
                return value, ghost_metadata
            elif isinstance(value, dict):
                # Recursively extract
                nested_val, nested_ghost = extract_value_from_nested(value, preserve_ghost)
                ghost_metadata.update(nested_ghost)
                return nested_val, ghost_metadata
            else:
                return str(value) if value is not None else "N/A", ghost_metadata
        
        # No value field - might be the spec itself as a dict
        # Return as-is if it looks like a valid spec dict
        return data, ghost_metadata
    
    if isinstance(data, list):
        # Join list values
        if all(isinstance(item, str) for item in data):
            return ", ".join(data), ghost_metadata
        return str(data), ghost_metadata
    
    return str(data), ghost_metadata


def clean_value(value: str) -> str:
    """
    Clean a specification value by removing descriptive prefixes and normalizing.
    
    Args:
        value: The raw value string
        
    Returns:
        Cleaned value or "N/A"
    """
    if not isinstance(value, str):
        return str(value) if value is not None else "N/A"
    
    value = value.strip()
    
    # Check if it's explicitly N/A
    for pattern in NA_PATTERNS:
        if re.match(pattern, value, re.IGNORECASE):
            return "N/A"
    
    # Handle N/A with parenthetical explanation
    # e.g., "N/A (typically a temperature transmitter with HART...)"
    na_with_info = re.match(r'^N/A\s*\((.+)\)$', value, re.IGNORECASE | re.DOTALL)
    if na_with_info:
        inner = na_with_info.group(1).strip()
        
        # Check if the inner content contains actual values we can extract
        # Look for patterns like "HART, Foundation Fieldbus, Profibus PA"
        tech_values = extract_technical_values(inner)
        if tech_values:
            return tech_values
        
        # If it's pure explanation, just return N/A
        if is_descriptive_text(inner):
            return "N/A"
        
        return "N/A"
    
    # Remove document references like [51948133245220†L3910-L3925]
    value = re.sub(r'\[\d+[†\u2020]L\d+-L\d+\]', '', value)
    value = re.sub(r'【\d+[†\u2020]L\d+-L\d+】', '', value)
    
    # Clean up trailing punctuation
    value = value.rstrip('.').strip()
    
    # Normalize whitespace
    value = ' '.join(value.split())
    
    return value if value else "N/A"


def extract_technical_values(text: str) -> Optional[str]:
    """
    Extract technical specification values from descriptive text.
    
    Args:
        text: Text that may contain embedded technical values
        
    Returns:
        Extracted values or None if no values found
    """
    # Common patterns for technical values
    patterns = [
        # Protocol lists: "HART, Foundation Fieldbus, Profibus PA"
        r'((?:HART|Modbus|Profibus|Foundation Fieldbus|EtherNet/IP|DeviceNet|BACnet)(?:\s*,\s*(?:HART|Modbus|Profibus|Foundation Fieldbus|EtherNet/IP|DeviceNet|BACnet))*)',
        
        # Thermocouple types: "Type K, J, T, N, E, S, R, B"
        r'(Type\s+[A-Z](?:\s*,\s*[A-Z])+)',  # Multiple types like "Type K, J, T"
        r'(Type\s+[KJTNERSB])',  # Single type like "Type K"
        
        # Signal types: "4-20mA", "0-10V"
        r'(\d+-\d+\s*mA(?:\s*DC)?)',
        r'(\d+-\d+\s*V(?:\s*DC)?)',
        
        # Temperature ranges: "-40 to 85°C"
        r'(-?\d+\s*(?:to|-)\s*-?\d+\s*°?[CF])',
        
        # Pressure: "0-350 bar", "1500 psi"
        r'(\d+(?:\.\d+)?\s*(?:bar|psi|kPa|MPa))',
        
        # IP ratings: "IP66", "IP67"
        r'(IP\d{2}(?:/IP\d{2})?)',
        
        # NEMA ratings: "NEMA 4X"
        r'(NEMA\s*\d+[A-Z]?)',
        
        # SIL ratings: "SIL 2", "SIL 3"
        r'(SIL\s*\d)',
        
        # Accuracy: "±0.1%", "±2°C"
        r'([±]\s*\d+(?:\.\d+)?\s*%)',
        r'([±]\s*\d+(?:\.\d+)?\s*°?[CF])',
        
        # Materials: "316L Stainless Steel", "Hastelloy C-276"
        r'(316L?\s*(?:Stainless\s*Steel|SS))',
        r'(Hastelloy\s*[A-Z]-?\d+)',
        r'(Inconel\s*\d+)',
    ]
    
    found_values = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        found_values.extend(matches)
    
    if found_values:
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for v in found_values:
            v_clean = v.strip()
            if v_clean.lower() not in seen:
                seen.add(v_clean.lower())
                unique.append(v_clean)
        return ", ".join(unique)
    
    return None


def is_descriptive_text(text: str) -> bool:
    """
    Check if the text is purely descriptive (not a technical value).
    
    Args:
        text: Text to check
        
    Returns:
        True if text is descriptive, False if it contains technical values
    """
    if not text:
        return True
    
    # Descriptive patterns
    descriptive_patterns = [
        r'typically\s+a\s+',
        r'will\s+be\s+used',
        r'is\s+provided\s+via',
        r'needs\s+to\s+be',
        r'should\s+be',
        r'may\s+be',
        r'can\s+be',
        r'refer\s+to',
        r'consult\s+',
        r'depending\s+on',
        r'varies\s+with',
        r'application[\s-]specific',
        r'application[\s-]dependent',
        r'options\s+for',
    ]
    
    for pattern in descriptive_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    return False


# =============================================================================
# MAIN NORMALIZATION FUNCTION
# =============================================================================

def normalize_specification_output(
    raw_specs: Dict[str, Any],
    preserve_ghost_values: bool = True
) -> Dict[str, Any]:
    """
    Normalize a raw specification dictionary to clean, flat key-value pairs.
    
    Transforms nested structures like:
    {
        "accuracy": {"value": "±1%", "confidence": 0.9, "note": "..."}
    }
    
    Into:
    {
        "accuracy": "±1%",
        "_ghost": {"accuracy": {"_confidence": 0.9, "_note": "..."}}
    }
    
    Args:
        raw_specs: The raw specification dictionary from LLM
        preserve_ghost_values: If True, preserve confidence/notes as ghost values
        
    Returns:
        Normalized specification dictionary with ghost values if requested
    """
    normalized = {}
    ghost_values = {}
    
    for key, value in raw_specs.items():
        # Skip internal/meta fields
        if key.startswith('_'):
            continue
        
        # Normalize the key
        norm_key = normalize_key(key)
        
        # Extract value from nested structure
        extracted_value, ghost = extract_value_from_nested(value, preserve_ghost_values)
        
        # Handle dict values (nested specs)
        if isinstance(extracted_value, dict):
            # Recursively normalize nested dicts
            nested_result = normalize_specification_output(
                extracted_value, 
                preserve_ghost_values
            )
            # Merge nested results
            if "_ghost" in nested_result:
                for nk, nv in nested_result.pop("_ghost").items():
                    ghost_values[f"{norm_key}.{nk}"] = nv
            normalized[norm_key] = {
                normalize_key(k): v 
                for k, v in nested_result.items()
            }
        else:
            # Clean the value
            cleaned = clean_value(str(extracted_value))
            normalized[norm_key] = cleaned
            
            # Store ghost metadata if present
            if ghost and preserve_ghost_values:
                ghost_values[norm_key] = ghost
    
    # Add ghost values if any
    if ghost_values and preserve_ghost_values:
        normalized["_ghost"] = ghost_values
    
    return normalized


def normalize_section_specs(
    section_specs: Dict[str, Any],
    section_name: str = "",
    preserve_ghost_values: bool = True
) -> Dict[str, Any]:
    """
    Normalize specifications within a specific section (e.g., mandatory, safety).
    
    Args:
        section_specs: Specs within a section
        section_name: Name of the section for logging
        preserve_ghost_values: If True, preserve ghost values
        
    Returns:
        Normalized section specifications
    """
    result = normalize_specification_output(section_specs, preserve_ghost_values)
    
    # Log normalization summary
    original_count = len(section_specs)
    final_count = len([k for k in result.keys() if not k.startswith('_')])
    
    if original_count != final_count:
        logger.debug(f"[Normalizer] Section '{section_name}': {original_count} -> {final_count} specs")
    
    return result


def normalize_full_item_specs(
    item: Dict[str, Any],
    preserve_ghost_values: bool = True
) -> Dict[str, Any]:
    """
    Normalize all specifications within an instrument/accessory item.
    
    Args:
        item: Full item dictionary with specifications
        preserve_ghost_values: If True, preserve ghost values
        
    Returns:
        Item with normalized specifications
    """
    result = item.copy()
    
    if "specifications" in result:
        specs = result["specifications"]
        normalized_specs = {}
        
        for section_name, section_data in specs.items():
            if isinstance(section_data, dict):
                normalized_specs[section_name] = normalize_section_specs(
                    section_data, 
                    section_name,
                    preserve_ghost_values
                )
            else:
                normalized_specs[section_name] = section_data
        
        result["specifications"] = normalized_specs
    
    return result


def deduplicate_specs(specs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove duplicate specifications across sections.
    
    If the same key appears in multiple sections with identical values,
    keep only the first occurrence.
    
    Args:
        specs: Specifications dictionary with multiple sections
        
    Returns:
        Deduplicated specifications
    """
    if not isinstance(specs, dict):
        return specs
    
    seen_values = {}  # key -> (section, value)
    result = {}
    
    for section, section_data in specs.items():
        if not isinstance(section_data, dict):
            result[section] = section_data
            continue
        
        deduped_section = {}
        for key, value in section_data.items():
            if key.startswith('_'):
                deduped_section[key] = value
                continue
            
            # Check for duplicates
            if key in seen_values:
                prev_section, prev_value = seen_values[key]
                if prev_value == value:
                    logger.debug(f"[Normalizer] Removing duplicate: {key} (same value in {prev_section})")
                    continue
            
            seen_values[key] = (section, value)
            deduped_section[key] = value
        
        result[section] = deduped_section
    
    return result


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "normalize_key",
    "extract_value_from_nested",
    "clean_value",
    "extract_technical_values",
    "is_descriptive_text",
    "normalize_specification_output",
    "normalize_section_specs",
    "normalize_full_item_specs",
    "deduplicate_specs",
    "STANDARD_KEY_MAPPINGS",
]
