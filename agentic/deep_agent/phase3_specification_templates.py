# agentic/deep_agent/phase3_specification_templates.py
# =============================================================================
# SPECIFICATION TEMPLATES FOR ALL PRODUCT TYPES
# =============================================================================
#
# Purpose: Provide comprehensive 60+ specification templates for each product
# type. These templates ensure minimum 60+ specifications per product type
# regardless of user input or standards availability.
#
# Key Features:
# 1. 60+ specifications per product type
# 2. Hierarchical category organization
# 3. Default/typical values for gaps
# 4. Confidence scoring framework
# 5. Spec priority and importance levels
#
# =============================================================================

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class SpecImportance(Enum):
    """Specification importance levels"""
    CRITICAL = 1.0      # Must have (safety, core function)
    REQUIRED = 0.9      # Should have (standard parameters)
    IMPORTANT = 0.7     # Recommended (detailed specs)
    OPTIONAL = 0.5      # Nice to have (extra details)
    DERIVED = 0.6       # Computed from other specs


@dataclass
class SpecificationDefinition:
    """Single specification definition"""
    key: str
    category: str
    description: str
    unit: Optional[str] = None
    data_type: str = "string"  # string, number, boolean, list
    typical_value: Optional[Any] = None
    typical_range: Optional[tuple] = None
    options: Optional[List[str]] = None
    importance: SpecImportance = SpecImportance.OPTIONAL
    source_priority: int = 0  # Lower = higher priority


class TemperatureSensorTemplate:
    """
    Comprehensive specification template for Temperature Sensors
    60+ specifications covering all categories
    """

    SPECIFICATIONS: Dict[str, SpecificationDefinition] = {
        # A. MEASUREMENT SPECIFICATIONS (12)
        "measurement_range_min": SpecificationDefinition(
            key="measurement_range_min", category="Measurement", description="Minimum measurable temperature",
            unit="°C", data_type="number", typical_value="-50", importance=SpecImportance.CRITICAL
        ),
        "measurement_range_max": SpecificationDefinition(
            key="measurement_range_max", category="Measurement", description="Maximum measurable temperature",
            unit="°C", data_type="number", typical_value="500", importance=SpecImportance.CRITICAL
        ),
        "measurement_unit": SpecificationDefinition(
            key="measurement_unit", category="Measurement", description="Temperature unit",
            unit="", data_type="string", options=["Celsius", "Fahrenheit", "Kelvin"],
            typical_value="Celsius", importance=SpecImportance.CRITICAL
        ),
        "accuracy_absolute": SpecificationDefinition(
            key="accuracy_absolute", category="Measurement", description="Absolute accuracy",
            unit="°C", data_type="number", typical_value="±0.5", importance=SpecImportance.CRITICAL
        ),
        "accuracy_percent_reading": SpecificationDefinition(
            key="accuracy_percent_reading", category="Measurement", description="Accuracy as % of reading",
            unit="%", data_type="number", typical_value="0.5", importance=SpecImportance.CRITICAL
        ),
        "accuracy_percent_fs": SpecificationDefinition(
            key="accuracy_percent_fs", category="Measurement", description="Accuracy as % of full scale",
            unit="%", data_type="number", typical_value="1.0", importance=SpecImportance.IMPORTANT
        ),
        "repeatability": SpecificationDefinition(
            key="repeatability", category="Measurement", description="Measurement repeatability",
            unit="%", data_type="number", typical_value="0.1", importance=SpecImportance.IMPORTANT
        ),
        "hysteresis": SpecificationDefinition(
            key="hysteresis", category="Measurement", description="Hysteresis effect",
            unit="%", data_type="number", typical_value="0.2", importance=SpecImportance.IMPORTANT
        ),
        "linearity": SpecificationDefinition(
            key="linearity", category="Measurement", description="Linearity deviation",
            unit="%", data_type="number", typical_value="0.3", importance=SpecImportance.OPTIONAL
        ),
        "resolution": SpecificationDefinition(
            key="resolution", category="Measurement", description="Temperature resolution",
            unit="°C", data_type="number", typical_value="0.1", importance=SpecImportance.OPTIONAL
        ),
        "sensitivity": SpecificationDefinition(
            key="sensitivity", category="Measurement", description="Sensor sensitivity",
            unit="mV/°C or Ω/°C", data_type="number", typical_value="varies", importance=SpecImportance.OPTIONAL
        ),
        "zero_drift": SpecificationDefinition(
            key="zero_drift", category="Measurement", description="Zero point drift over time",
            unit="%/year", data_type="number", typical_value="0.5", importance=SpecImportance.OPTIONAL
        ),

        # B. OPERATING CONDITIONS (10)
        "operating_temperature_min": SpecificationDefinition(
            key="operating_temperature_min", category="Operating Conditions", description="Minimum operating temperature",
            unit="°C", data_type="number", typical_value="-40", importance=SpecImportance.CRITICAL
        ),
        "operating_temperature_max": SpecificationDefinition(
            key="operating_temperature_max", category="Operating Conditions", description="Maximum operating temperature",
            unit="°C", data_type="number", typical_value="85", importance=SpecImportance.CRITICAL
        ),
        "ambient_temperature_min": SpecificationDefinition(
            key="ambient_temperature_min", category="Operating Conditions", description="Minimum ambient temperature",
            unit="°C", data_type="number", typical_value="-40", importance=SpecImportance.IMPORTANT
        ),
        "ambient_temperature_max": SpecificationDefinition(
            key="ambient_temperature_max", category="Operating Conditions", description="Maximum ambient temperature",
            unit="°C", data_type="number", typical_value="70", importance=SpecImportance.IMPORTANT
        ),
        "humidity_range_min": SpecificationDefinition(
            key="humidity_range_min", category="Operating Conditions", description="Minimum operating humidity",
            unit="%", data_type="number", typical_value="5", importance=SpecImportance.IMPORTANT
        ),
        "humidity_range_max": SpecificationDefinition(
            key="humidity_range_max", category="Operating Conditions", description="Maximum operating humidity",
            unit="%", data_type="number", typical_value="95", importance=SpecImportance.IMPORTANT
        ),
        "humidity_non_condensing": SpecificationDefinition(
            key="humidity_non_condensing", category="Operating Conditions", description="Non-condensing humidity requirement",
            unit="", data_type="boolean", typical_value=True, importance=SpecImportance.IMPORTANT
        ),
        "altitude_max": SpecificationDefinition(
            key="altitude_max", category="Operating Conditions", description="Maximum operating altitude",
            unit="m", data_type="number", typical_value="3000", importance=SpecImportance.OPTIONAL
        ),
        "storage_temperature_min": SpecificationDefinition(
            key="storage_temperature_min", category="Operating Conditions", description="Minimum storage temperature",
            unit="°C", data_type="number", typical_value="-40", importance=SpecImportance.OPTIONAL
        ),
        "storage_temperature_max": SpecificationDefinition(
            key="storage_temperature_max", category="Operating Conditions", description="Maximum storage temperature",
            unit="°C", data_type="number", typical_value="70", importance=SpecImportance.OPTIONAL
        ),

        # C. RESPONSE CHARACTERISTICS (7)
        "response_time_ms": SpecificationDefinition(
            key="response_time_ms", category="Response", description="Response time (0-63% rise time)",
            unit="ms", data_type="number", typical_value="100", importance=SpecImportance.CRITICAL
        ),
        "time_constant_seconds": SpecificationDefinition(
            key="time_constant_seconds", category="Response", description="Time constant (tau)",
            unit="s", data_type="number", typical_value="5.0", importance=SpecImportance.CRITICAL
        ),
        "settling_time": SpecificationDefinition(
            key="settling_time", category="Response", description="Time to settle to ±2% of final value",
            unit="s", data_type="number", typical_value="10", importance=SpecImportance.IMPORTANT
        ),
        "thermal_lag_milliseconds": SpecificationDefinition(
            key="thermal_lag_milliseconds", category="Response", description="Thermal lag time",
            unit="ms", data_type="number", typical_value="500", importance=SpecImportance.IMPORTANT
        ),
        "frequency_response": SpecificationDefinition(
            key="frequency_response", category="Response", description="Frequency response range",
            unit="Hz", data_type="string", typical_value="DC to 10Hz", importance=SpecImportance.OPTIONAL
        ),
        "damping_type": SpecificationDefinition(
            key="damping_type", category="Response", description="Damping characteristic",
            unit="", data_type="string", options=["Critical", "Underdamped", "Overdamped"],
            typical_value="Slightly underdamped", importance=SpecImportance.OPTIONAL
        ),
        "overshoot_percentage": SpecificationDefinition(
            key="overshoot_percentage", category="Response", description="Overshoot percentage",
            unit="%", data_type="number", typical_value="5", importance=SpecImportance.OPTIONAL
        ),

        # D. SIGNAL & OUTPUT (8)
        "output_signal_type": SpecificationDefinition(
            key="output_signal_type", category="Signal Output", description="Type of output signal",
            unit="", data_type="string", options=["4-20mA", "0-10V", "1-5V", "RTD", "Digital"],
            typical_value="4-20mA", importance=SpecImportance.CRITICAL
        ),
        "output_signal_min": SpecificationDefinition(
            key="output_signal_min", category="Signal Output", description="Minimum output signal value",
            unit="mA or V", data_type="number", typical_value="4", importance=SpecImportance.CRITICAL
        ),
        "output_signal_max": SpecificationDefinition(
            key="output_signal_max", category="Signal Output", description="Maximum output signal value",
            unit="mA or V", data_type="number", typical_value="20", importance=SpecImportance.CRITICAL
        ),
        "load_resistance": SpecificationDefinition(
            key="load_resistance", category="Signal Output", description="Maximum load resistance",
            unit="Ω", data_type="number", typical_value="600", importance=SpecImportance.IMPORTANT
        ),
        "loop_power_consumption": SpecificationDefinition(
            key="loop_power_consumption", category="Signal Output", description="Loop power consumption",
            unit="mA", data_type="number", typical_value="4", importance=SpecImportance.IMPORTANT
        ),
        "signal_conditioning": SpecificationDefinition(
            key="signal_conditioning", category="Signal Output", description="Signal conditioning applied",
            unit="", data_type="string", typical_value="Yes", importance=SpecImportance.OPTIONAL
        ),
        "digital_output_type": SpecificationDefinition(
            key="digital_output_type", category="Signal Output", description="Type if digital output available",
            unit="", data_type="string", options=["Modbus RTU", "HART", "Profibus"], typical_value="Optional",
            importance=SpecImportance.OPTIONAL
        ),
        "communication_protocol": SpecificationDefinition(
            key="communication_protocol", category="Signal Output", description="Communication protocol",
            unit="", data_type="string", options=["HART", "Modbus", "Profibus", "4-20mA analog"],
            typical_value="HART (optional)", importance=SpecImportance.OPTIONAL
        ),

        # E. ELECTRICAL SPECIFICATIONS (10)
        "supply_voltage_min": SpecificationDefinition(
            key="supply_voltage_min", category="Electrical", description="Minimum supply voltage",
            unit="V DC", data_type="number", typical_value="10", importance=SpecImportance.CRITICAL
        ),
        "supply_voltage_max": SpecificationDefinition(
            key="supply_voltage_max", category="Electrical", description="Maximum supply voltage",
            unit="V DC", data_type="number", typical_value="32", importance=SpecImportance.CRITICAL
        ),
        "supply_voltage_nominal": SpecificationDefinition(
            key="supply_voltage_nominal", category="Electrical", description="Nominal supply voltage",
            unit="V DC", data_type="number", typical_value="24", importance=SpecImportance.IMPORTANT
        ),
        "voltage_stability": SpecificationDefinition(
            key="voltage_stability", category="Electrical", description="Voltage stability requirement",
            unit="%", data_type="number", typical_value="±10", importance=SpecImportance.OPTIONAL
        ),
        "power_consumption_max": SpecificationDefinition(
            key="power_consumption_max", category="Electrical", description="Maximum power consumption",
            unit="mW", data_type="number", typical_value="150", importance=SpecImportance.IMPORTANT
        ),
        "short_circuit_protection": SpecificationDefinition(
            key="short_circuit_protection", category="Electrical", description="Short circuit protection",
            unit="", data_type="string", typical_value="Yes", importance=SpecImportance.IMPORTANT
        ),
        "reverse_polarity_protection": SpecificationDefinition(
            key="reverse_polarity_protection", category="Electrical", description="Reverse polarity protection",
            unit="", data_type="string", typical_value="Yes", importance=SpecImportance.IMPORTANT
        ),
        "electromagnetic_immunity": SpecificationDefinition(
            key="electromagnetic_immunity", category="Electrical", description="EMI immunity class",
            unit="", data_type="string", options=["Class A", "Class B", "Class C"], typical_value="Class B",
            importance=SpecImportance.OPTIONAL
        ),
        "noise_immunity": SpecificationDefinition(
            key="noise_immunity", category="Electrical", description="Noise immunity",
            unit="mV", data_type="number", typical_value="100", importance=SpecImportance.OPTIONAL
        ),
        "insulation_resistance": SpecificationDefinition(
            key="insulation_resistance", category="Electrical", description="Insulation resistance",
            unit="MΩ", data_type="number", typical_value="100", importance=SpecImportance.IMPORTANT
        ),

        # F. MECHANICAL SPECIFICATIONS (12)
        "overall_length": SpecificationDefinition(
            key="overall_length", category="Mechanical", description="Overall length",
            unit="mm", data_type="number", typical_value="200", importance=SpecImportance.IMPORTANT
        ),
        "overall_diameter": SpecificationDefinition(
            key="overall_diameter", category="Mechanical", description="Overall diameter",
            unit="mm", data_type="number", typical_value="6", importance=SpecImportance.IMPORTANT
        ),
        "immersion_depth": SpecificationDefinition(
            key="immersion_depth", category="Mechanical", description="Recommended immersion depth",
            unit="mm", data_type="number", typical_value="100", importance=SpecImportance.CRITICAL
        ),
        "insertion_length": SpecificationDefinition(
            key="insertion_length", category="Mechanical", description="Insertion length into process",
            unit="mm", data_type="number", typical_value="75", importance=SpecImportance.CRITICAL
        ),
        "bare_probe_length": SpecificationDefinition(
            key="bare_probe_length", category="Mechanical", description="Bare probe length (no sheath)",
            unit="mm", data_type="number", typical_value="50", importance=SpecImportance.IMPORTANT
        ),
        "probe_diameter": SpecificationDefinition(
            key="probe_diameter", category="Mechanical", description="Probe diameter",
            unit="mm", data_type="number", typical_value="3", importance=SpecImportance.IMPORTANT
        ),
        "sheath_diameter": SpecificationDefinition(
            key="sheath_diameter", category="Mechanical", description="Sheath/thermowell diameter",
            unit="mm", data_type="number", typical_value="6", importance=SpecImportance.CRITICAL
        ),
        "sheath_length": SpecificationDefinition(
            key="sheath_length", category="Mechanical", description="Sheath length",
            unit="mm", data_type="number", typical_value="200", importance=SpecImportance.IMPORTANT
        ),
        "probe_tip_design": SpecificationDefinition(
            key="probe_tip_design", category="Mechanical", description="Probe tip design",
            unit="", data_type="string", options=["Hemispherical", "Conical", "Flat"],
            typical_value="Hemispherical", importance=SpecImportance.OPTIONAL
        ),
        "weight": SpecificationDefinition(
            key="weight", category="Mechanical", description="Sensor weight",
            unit="g", data_type="number", typical_value="50", importance=SpecImportance.OPTIONAL
        ),
        "junction_type": SpecificationDefinition(
            key="junction_type", category="Mechanical", description="Thermocouple junction type",
            unit="", data_type="string", options=["Grounded", "Ungrounded", "Exposed"],
            typical_value="Grounded", importance=SpecImportance.IMPORTANT
        ),
        "cold_junction_compensation": SpecificationDefinition(
            key="cold_junction_compensation", category="Mechanical", description="Cold junction compensation",
            unit="", data_type="string", typical_value="Internal", importance=SpecImportance.OPTIONAL
        ),

        # G. PRESSURE & INTEGRITY (10)
        "process_pressure_max": SpecificationDefinition(
            key="process_pressure_max", category="Pressure", description="Maximum process pressure",
            unit="bar", data_type="number", typical_value="100", importance=SpecImportance.CRITICAL
        ),
        "proof_pressure": SpecificationDefinition(
            key="proof_pressure", category="Pressure", description="Proof pressure (1.5x max)",
            unit="bar", data_type="number", typical_value="150", importance=SpecImportance.CRITICAL
        ),
        "burst_pressure": SpecificationDefinition(
            key="burst_pressure", category="Pressure", description="Burst pressure (3x max)",
            unit="bar", data_type="number", typical_value="300", importance=SpecImportance.CRITICAL
        ),
        "pressure_connection_type": SpecificationDefinition(
            key="pressure_connection_type", category="Pressure", description="Type of pressure connection",
            unit="", data_type="string", options=["NPT", "BSPP", "SAE", "Flange", "Compression"],
            typical_value="1/2\" NPT", importance=SpecImportance.CRITICAL
        ),
        "pressure_port_size": SpecificationDefinition(
            key="pressure_port_size", category="Pressure", description="Pressure port size",
            unit="inches or mm", data_type="string", typical_value="1/2\"", importance=SpecImportance.CRITICAL
        ),
        "backpressure_rating": SpecificationDefinition(
            key="backpressure_rating", category="Pressure", description="Maximum backpressure",
            unit="bar", data_type="number", typical_value="50", importance=SpecImportance.OPTIONAL
        ),
        "pressure_shock_resistance": SpecificationDefinition(
            key="pressure_shock_resistance", category="Pressure", description="Pressure shock resistance",
            unit="bar/s", data_type="number", typical_value="100", importance=SpecImportance.OPTIONAL
        ),
        "max_proof_pressure_percentage": SpecificationDefinition(
            key="max_proof_pressure_percentage", category="Pressure", description="Max proof pressure as % of rated",
            unit="%", data_type="number", typical_value="150", importance=SpecImportance.OPTIONAL
        ),
        "overpressure_protection": SpecificationDefinition(
            key="overpressure_protection", category="Pressure", description="Overpressure protection mechanism",
            unit="", data_type="string", typical_value="Integral relief", importance=SpecImportance.OPTIONAL
        ),
        "pressure_rating_type": SpecificationDefinition(
            key="pressure_rating_type", category="Pressure", description="Type of pressure rating",
            unit="", data_type="string", options=["Gauge", "Absolute", "Differential"],
            typical_value="Gauge", importance=SpecImportance.IMPORTANT
        ),

        # H. MATERIALS (10)
        "wetted_material": SpecificationDefinition(
            key="wetted_material", category="Materials", description="Wetted material",
            unit="", data_type="string", options=["Stainless 304", "Stainless 316L", "Hastelloy", "Invar"],
            typical_value="Stainless 316L", importance=SpecImportance.CRITICAL
        ),
        "wetted_material_grade": SpecificationDefinition(
            key="wetted_material_grade", category="Materials", description="Grade of wetted material",
            unit="", data_type="string", typical_value="1.4571 (EN spec)", importance=SpecImportance.IMPORTANT
        ),
        "probe_material": SpecificationDefinition(
            key="probe_material", category="Materials", description="Probe material",
            unit="", data_type="string", options=["Stainless", "Inconel", "Thermocouple wire"],
            typical_value="Stainless", importance=SpecImportance.CRITICAL
        ),
        "sheath_material": SpecificationDefinition(
            key="sheath_material", category="Materials", description="Sheath material",
            unit="", data_type="string", options=["Stainless 304", "Stainless 316L", "Inconel"],
            typical_value="Stainless 316L", importance=SpecImportance.CRITICAL
        ),
        "housing_material": SpecificationDefinition(
            key="housing_material", category="Materials", description="Housing material",
            unit="", data_type="string", options=["Stainless", "Aluminum", "Plastic", "Cast aluminum"],
            typical_value="Stainless", importance=SpecImportance.IMPORTANT
        ),
        "connector_material": SpecificationDefinition(
            key="connector_material", category="Materials", description="Connector material",
            unit="", data_type="string", options=["Stainless", "Brass", "Plastic"],
            typical_value="Brass", importance=SpecImportance.IMPORTANT
        ),
        "gasket_material": SpecificationDefinition(
            key="gasket_material", category="Materials", description="Gasket material",
            unit="", data_type="string", options=["PTFE", "Graphite", "Viton"],
            typical_value="PTFE", importance=SpecImportance.IMPORTANT
        ),
        "sealing_material": SpecificationDefinition(
            key="sealing_material", category="Materials", description="Sealing compound",
            unit="", data_type="string", typical_value="Stainless steel based", importance=SpecImportance.OPTIONAL
        ),
        "corrosion_resistance_class": SpecificationDefinition(
            key="corrosion_resistance_class", category="Materials", description="Corrosion resistance class",
            unit="", data_type="string", options=["C3-M", "C4-M", "C5-M"],
            typical_value="C3-M", importance=SpecImportance.IMPORTANT
        ),
        "material_certification": SpecificationDefinition(
            key="material_certification", category="Materials", description="Material certification",
            unit="", data_type="string", options=["ASME", "NIST", "EN 10204"],
            typical_value="ASME", importance=SpecImportance.IMPORTANT
        ),

        # I. CONNECTION & INSTALLATION (12)
        "connection_type": SpecificationDefinition(
            key="connection_type", category="Connection", description="Type of sensor head connection",
            unit="", data_type="string", options=["Head-mounted", "Transmitter module", "DIN connector", "M16"],
            typical_value="Head-mounted", importance=SpecImportance.CRITICAL
        ),
        "connection_thread_size": SpecificationDefinition(
            key="connection_thread_size", category="Connection", description="Thread size of connection",
            unit="", data_type="string", typical_value="1/2\" NPT", importance=SpecImportance.CRITICAL
        ),
        "cable_diameter": SpecificationDefinition(
            key="cable_diameter", category="Connection", description="Cable outer diameter",
            unit="mm", data_type="number", typical_value="6", importance=SpecImportance.IMPORTANT
        ),
        "cable_length_standard": SpecificationDefinition(
            key="cable_length_standard", category="Connection", description="Standard cable length",
            unit="m", data_type="number", typical_value="2", importance=SpecImportance.IMPORTANT
        ),
        "cable_length_max": SpecificationDefinition(
            key="cable_length_max", category="Connection", description="Maximum available cable length",
            unit="m", data_type="number", typical_value="20", importance=SpecImportance.OPTIONAL
        ),
        "connector_type": SpecificationDefinition(
            key="connector_type", category="Connection", description="Type of electrical connector",
            unit="", data_type="string", options=["M12", "M16", "Mini-DIN", "DIN 43650"],
            typical_value="M12", importance=SpecImportance.OPTIONAL
        ),
        "connector_pins": SpecificationDefinition(
            key="connector_pins", category="Connection", description="Number of connector pins",
            unit="pins", data_type="number", typical_value="2", importance=SpecImportance.OPTIONAL
        ),
        "connector_pin_arrangement": SpecificationDefinition(
            key="connector_pin_arrangement", category="Connection", description="Pin arrangement",
            unit="", data_type="string", typical_value="A-coded", importance=SpecImportance.OPTIONAL
        ),
        "mounting_orientation": SpecificationDefinition(
            key="mounting_orientation", category="Connection", description="Mounting orientation",
            unit="", data_type="string", options=["Vertical (immersion)", "Horizontal", "Any"],
            typical_value="Any", importance=SpecImportance.IMPORTANT
        ),
        "installation_direction": SpecificationDefinition(
            key="installation_direction", category="Connection", description="Installation direction",
            unit="", data_type="string", options=["Axial", "Radial", "Either"],
            typical_value="Either", importance=SpecImportance.OPTIONAL
        ),
        "installation_location": SpecificationDefinition(
            key="installation_location", category="Connection", description="Installation location",
            unit="", data_type="string", options=["Vessel wall", "Thermowell", "Direct immersion"],
            typical_value="Thermowell", importance=SpecImportance.IMPORTANT
        ),
        "installation_depth_recommendation": SpecificationDefinition(
            key="installation_depth_recommendation", category="Connection", description="Recommended insertion depth",
            unit="mm", data_type="number", typical_value="100", importance=SpecImportance.CRITICAL
        ),

        # J. THERMAL CHARACTERISTICS (8)
        "thermal_shock_resistance": SpecificationDefinition(
            key="thermal_shock_resistance", category="Thermal", description="Max thermal shock",
            unit="°C/s", data_type="number", typical_value="50", importance=SpecImportance.IMPORTANT
        ),
        "thermal_shock_delta": SpecificationDefinition(
            key="thermal_shock_delta", category="Thermal", description="Maximum temperature differential for shock",
            unit="°C", data_type="number", typical_value="100", importance=SpecImportance.IMPORTANT
        ),
        "thermal_conductivity": SpecificationDefinition(
            key="thermal_conductivity", category="Thermal", description="Required thermal conductivity",
            unit="W/m·K", data_type="number", typical_value="50", importance=SpecImportance.OPTIONAL
        ),
        "thermal_lag": SpecificationDefinition(
            key="thermal_lag", category="Thermal", description="Thermal lag characteristic",
            unit="s", data_type="number", typical_value="5", importance=SpecImportance.OPTIONAL
        ),
        "max_heating_rate": SpecificationDefinition(
            key="max_heating_rate", category="Thermal", description="Maximum heating rate",
            unit="°C/s", data_type="number", typical_value="20", importance=SpecImportance.OPTIONAL
        ),
        "heat_dissipation_capability": SpecificationDefinition(
            key="heat_dissipation_capability", category="Thermal", description="Heat dissipation capability",
            unit="W/m²K", data_type="number", typical_value="50", importance=SpecImportance.OPTIONAL
        ),
        "freeze_protection": SpecificationDefinition(
            key="freeze_protection", category="Thermal", description="Freeze protection required",
            unit="", data_type="boolean", typical_value=False, importance=SpecImportance.OPTIONAL
        ),
        "thermal_mass": SpecificationDefinition(
            key="thermal_mass", category="Thermal", description="Thermal mass effect",
            unit="", data_type="string", typical_value="Minimal", importance=SpecImportance.OPTIONAL
        ),

        # K. CALIBRATION (10)
        "calibration_accuracy_class": SpecificationDefinition(
            key="calibration_accuracy_class", category="Calibration", description="Calibration accuracy class",
            unit="", data_type="string", options=["A (IEC 751)", "B (IEC 751)", "AA"],
            typical_value="B", importance=SpecImportance.IMPORTANT
        ),
        "calibration_interval": SpecificationDefinition(
            key="calibration_interval", category="Calibration", description="Recommended calibration interval",
            unit="months", data_type="number", typical_value="12", importance=SpecImportance.IMPORTANT
        ),
        "calibration_type": SpecificationDefinition(
            key="calibration_type", category="Calibration", description="Type of calibration",
            unit="", data_type="string", options=["Primary", "Secondary", "Field"],
            typical_value="Secondary", importance=SpecImportance.OPTIONAL
        ),
        "calibration_standard": SpecificationDefinition(
            key="calibration_standard", category="Calibration", description="Calibration standard reference",
            unit="", data_type="string", options=["IEC 60751", "DIN 43760", "NIST"],
            typical_value="IEC 60751", importance=SpecImportance.IMPORTANT
        ),
        "traceability_requirement": SpecificationDefinition(
            key="traceability_requirement", category="Calibration", description="Traceability requirement",
            unit="", data_type="string", typical_value="NIST", importance=SpecImportance.OPTIONAL
        ),
        "calibration_uncertainty": SpecificationDefinition(
            key="calibration_uncertainty", category="Calibration", description="Calibration uncertainty",
            unit="°C", data_type="number", typical_value="0.1", importance=SpecImportance.IMPORTANT
        ),
        "calibration_temperature": SpecificationDefinition(
            key="calibration_temperature", category="Calibration", description="Reference calibration temperature",
            unit="°C", data_type="number", typical_value="20", importance=SpecImportance.OPTIONAL
        ),
        "zero_adjustment_capability": SpecificationDefinition(
            key="zero_adjustment_capability", category="Calibration", description="Zero adjustment available",
            unit="", data_type="boolean", typical_value=True, importance=SpecImportance.OPTIONAL
        ),
        "span_adjustment_capability": SpecificationDefinition(
            key="span_adjustment_capability", category="Calibration", description="Span adjustment available",
            unit="", data_type="boolean", typical_value=True, importance=SpecImportance.OPTIONAL
        ),
        "field_calibration_possible": SpecificationDefinition(
            key="field_calibration_possible", category="Calibration", description="Can be calibrated in field",
            unit="", data_type="boolean", typical_value=True, importance=SpecImportance.OPTIONAL
        ),

        # L. CERTIFICATION & COMPLIANCE (12)
        "atex_category": SpecificationDefinition(
            key="atex_category", category="Certification", description="ATEX category",
            unit="", data_type="string", options=["2G", "3G", "Not-rated"],
            typical_value="Not-rated", importance=SpecImportance.CRITICAL
        ),
        "atex_group": SpecificationDefinition(
            key="atex_group", category="Certification", description="ATEX group (if applicable)",
            unit="", data_type="string", options=["IIC", "IIB", "IIA"],
            typical_value="N/A", importance=SpecImportance.CRITICAL
        ),
        "iecex_certification": SpecificationDefinition(
            key="iecex_certification", category="Certification", description="IECEx certification",
            unit="", data_type="boolean", typical_value=False, importance=SpecImportance.IMPORTANT
        ),
        "iecex_code": SpecificationDefinition(
            key="iecex_code", category="Certification", description="IECEx certification code",
            unit="", data_type="string", typical_value="N/A", importance=SpecImportance.OPTIONAL
        ),
        "sil_rating": SpecificationDefinition(
            key="sil_rating", category="Certification", description="SIL rating",
            unit="", data_type="string", options=["SIL1", "SIL2", "SIL3", "Not-rated"],
            typical_value="Not-rated", importance=SpecImportance.CRITICAL
        ),
        "sil_certification_level": SpecificationDefinition(
            key="sil_certification_level", category="Certification", description="SIL certification level",
            unit="", data_type="string", typical_value="N/A", importance=SpecImportance.OPTIONAL
        ),
        "ce_marking": SpecificationDefinition(
            key="ce_marking", category="Certification", description="CE marking",
            unit="", data_type="boolean", typical_value=True, importance=SpecImportance.IMPORTANT
        ),
        "eu_directive_compliance": SpecificationDefinition(
            key="eu_directive_compliance", category="Certification", description="EU directive compliance",
            unit="", data_type="string", options=["2014/30/EU", "2014/35/EU", "2014/65/EU"],
            typical_value="2014/30/EU", importance=SpecImportance.OPTIONAL
        ),
        "industry_standards": SpecificationDefinition(
            key="industry_standards", category="Certification", description="Industry standards compliance",
            unit="", data_type="string", options=["IEC 60751", "DIN 43760", "API 571"],
            typical_value="IEC 60751", importance=SpecImportance.IMPORTANT
        ),
        "third_party_certification": SpecificationDefinition(
            key="third_party_certification", category="Certification", description="Third-party certification",
            unit="", data_type="string", options=["DNV", "ABS", "Lloyd's", "None"],
            typical_value="None", importance=SpecImportance.OPTIONAL
        ),
        "certification_body": SpecificationDefinition(
            key="certification_body", category="Certification", description="Certification body name",
            unit="", data_type="string", typical_value="N/A", importance=SpecImportance.OPTIONAL
        ),
        "certificate_reference": SpecificationDefinition(
            key="certificate_reference", category="Certification", description="Certificate reference number",
            unit="", data_type="string", typical_value="N/A", importance=SpecImportance.OPTIONAL
        ),

        # M. PROTECTION & SAFETY (10)
        "ip_rating": SpecificationDefinition(
            key="ip_rating", category="Protection", description="IP protection rating",
            unit="", data_type="string", options=["IP65", "IP67", "IP69K"],
            typical_value="IP67", importance=SpecImportance.IMPORTANT
        ),
        "nema_rating": SpecificationDefinition(
            key="nema_rating", category="Protection", description="NEMA protection rating",
            unit="", data_type="string", options=["NEMA 4X", "NEMA 6P"],
            typical_value="NEMA 4X", importance=SpecImportance.OPTIONAL
        ),
        "overpressure_protection_type": SpecificationDefinition(
            key="overpressure_protection_type", category="Protection", description="Type of overpressure protection",
            unit="", data_type="string", typical_value="Integral relief", importance=SpecImportance.IMPORTANT
        ),
        "overvoltage_protection": SpecificationDefinition(
            key="overvoltage_protection", category="Protection", description="Overvoltage protection mechanism",
            unit="", data_type="string", typical_value="Transient suppression", importance=SpecImportance.IMPORTANT
        ),
        "overcurrent_protection": SpecificationDefinition(
            key="overcurrent_protection", category="Protection", description="Overcurrent protection type",
            unit="", data_type="string", typical_value="Fused or electronic", importance=SpecImportance.OPTIONAL
        ),
        "safety_shutdown_mechanism": SpecificationDefinition(
            key="safety_shutdown_mechanism", category="Protection", description="Safety shutdown mechanism",
            unit="", data_type="string", typical_value="Yes", importance=SpecImportance.OPTIONAL
        ),
        "fail_safe_mode": SpecificationDefinition(
            key="fail_safe_mode", category="Protection", description="Fail-safe mode",
            unit="", data_type="string", options=["Safe-state", "Last-value", "Error"],
            typical_value="Last-value", importance=SpecImportance.IMPORTANT
        ),
        "hazardous_area_category": SpecificationDefinition(
            key="hazardous_area_category", category="Protection", description="Hazardous area category",
            unit="", data_type="string", options=["Zone 0", "Zone 1", "Zone 2", "Not-applicable"],
            typical_value="Not-applicable", importance=SpecImportance.CRITICAL
        ),
        "temperature_protection_class": SpecificationDefinition(
            key="temperature_protection_class", category="Protection", description="Temperature protection class",
            unit="", data_type="string", options=["T1", "T2", "T3", "T4", "N/A"],
            typical_value="N/A", importance=SpecImportance.OPTIONAL
        ),
        "ingress_protection_rating": SpecificationDefinition(
            key="ingress_protection_rating", category="Protection", description="Ingress protection rating",
            unit="", data_type="string", typical_value="IP67", importance=SpecImportance.IMPORTANT
        ),
    }

    @classmethod
    def get_all_specifications(cls) -> Dict[str, SpecificationDefinition]:
        """Get all 62+ specifications for temperature sensors"""
        return cls.SPECIFICATIONS.copy()

    @classmethod
    def get_specification_count(cls) -> int:
        """Get total count of specifications"""
        return len(cls.SPECIFICATIONS)

    @classmethod
    def get_specifications_by_category(cls) -> Dict[str, List[str]]:
        """Get specifications grouped by category"""
        categories: Dict[str, List[str]] = {}
        for key, spec in cls.SPECIFICATIONS.items():
            category = spec.category
            if category not in categories:
                categories[category] = []
            categories[category].append(key)
        return categories


# Product Type Templates (to be continued with other product types)
PRODUCT_TYPE_TEMPLATES = {
    "temperature_sensor": TemperatureSensorTemplate,
    # Similar templates for other types:
    # "pressure_transmitter": PressureTransmitterTemplate,
    # "flow_meter": FlowMeterTemplate,
    # "level_transmitter": LevelTransmitterTemplate,
    # "analyzer": AnalyzerTemplate,
    # "control_valve": ControlValveTemplate,
    # "thermowell": ThermowellTemplate,
    # ... etc
}


def get_template_for_product_type(product_type: str) -> Optional[type]:
    """Get the specification template for a product type"""
    normalized_type = product_type.lower().replace(" ", "_")
    return PRODUCT_TYPE_TEMPLATES.get(normalized_type)


def get_all_specs_for_product_type(product_type: str) -> Dict[str, SpecificationDefinition]:
    """Get all specifications for a product type"""
    template_class = get_template_for_product_type(product_type)
    if template_class:
        return template_class.get_all_specifications()
    return {}


def get_spec_count_for_product_type(product_type: str) -> int:
    """Get specification count for a product type"""
    specs = get_all_specs_for_product_type(product_type)
    return len(specs)


def export_template_as_dict(product_type: str) -> Dict[str, Any]:
    """Export template as dictionary"""
    template_class = get_template_for_product_type(product_type)
    if not template_class:
        return {}

    specs = template_class.get_all_specifications()
    categories = template_class.get_specifications_by_category()

    return {
        "product_type": product_type,
        "total_specifications": len(specs),
        "categories": categories,
        "specifications": {
            key: {
                "category": spec.category,
                "description": spec.description,
                "unit": spec.unit,
                "data_type": spec.data_type,
                "typical_value": spec.typical_value,
                "options": spec.options,
                "importance": spec.importance.name
            }
            for key, spec in specs.items()
        }
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "TemperatureSensorTemplate",
    "SpecificationDefinition",
    "SpecImportance",
    "get_template_for_product_type",
    "get_all_specs_for_product_type",
    "get_spec_count_for_product_type",
    "export_template_as_dict",
    "PRODUCT_TYPE_TEMPLATES"
]
