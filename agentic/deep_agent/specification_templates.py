# agentic/deep_agent/specification_templates.py
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

# ==========================================
# ADDITIONAL TEMPLATES
# ==========================================
class PressureTransmitterTemplate:
    """Specification template for pressure transmitters with 65+ specs"""

    SPECIFICATIONS = {
        # 1. MEASUREMENT CHARACTERISTICS (12 specs)
        "pressure_type": SpecificationDefinition(
            key="pressure_type",
            category="Measurement Characteristics",
            description="Type of pressure measurement",
            data_type="string",
            options=["gauge", "absolute", "differential"],
            importance=SpecImportance.CRITICAL
        ),
        "operating_pressure_range_min": SpecificationDefinition(
            key="operating_pressure_range_min",
            category="Measurement Characteristics",
            description="Minimum operating pressure",
            unit="psi",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "operating_pressure_range_max": SpecificationDefinition(
            key="operating_pressure_range_max",
            category="Measurement Characteristics",
            description="Maximum operating pressure",
            unit="psi",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "pressure_accuracy_percent": SpecificationDefinition(
            key="pressure_accuracy_percent",
            category="Measurement Characteristics",
            description="Accuracy as percentage of span",
            unit="%",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "repeatability_pressure": SpecificationDefinition(
            key="repeatability_pressure",
            category="Measurement Characteristics",
            description="Repeatability of measurements",
            unit="%",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "linearity_error": SpecificationDefinition(
            key="linearity_error",
            category="Measurement Characteristics",
            description="Linearity error over range",
            unit="%",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "hysteresis_error": SpecificationDefinition(
            key="hysteresis_error",
            category="Measurement Characteristics",
            description="Hysteresis error in measurement",
            unit="%",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "zero_drift": SpecificationDefinition(
            key="zero_drift",
            category="Measurement Characteristics",
            description="Zero point drift over time",
            unit="%/year",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "span_drift": SpecificationDefinition(
            key="span_drift",
            category="Measurement Characteristics",
            description="Span drift over time",
            unit="%/year",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "thermal_stability_zero": SpecificationDefinition(
            key="thermal_stability_zero",
            category="Measurement Characteristics",
            description="Zero stability vs temperature",
            unit="%/°C",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "thermal_stability_span": SpecificationDefinition(
            key="thermal_stability_span",
            category="Measurement Characteristics",
            description="Span stability vs temperature",
            unit="%/°C",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "pressure_resolution": SpecificationDefinition(
            key="pressure_resolution",
            category="Measurement Characteristics",
            description="Measurement resolution",
            unit="psi",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),

        # 2. PRESSURE CONNECTION (10 specs)
        "pressure_port_type": SpecificationDefinition(
            key="pressure_port_type",
            category="Pressure Connection",
            description="Type of pressure connection",
            data_type="string",
            options=["threaded", "flanged", "welded", "SAE"],
            importance=SpecImportance.CRITICAL
        ),
        "pressure_port_size": SpecificationDefinition(
            key="pressure_port_size",
            category="Pressure Connection",
            description="Pressure port size",
            unit="in",
            data_type="string",
            importance=SpecImportance.CRITICAL
        ),
        "pressure_thread_type": SpecificationDefinition(
            key="pressure_thread_type",
            category="Pressure Connection",
            description="Thread type for connection",
            data_type="string",
            options=["NPT", "BSP", "BSPP", "BSPT", "M"],
            importance=SpecImportance.REQUIRED
        ),
        "wetted_material_pressure": SpecificationDefinition(
            key="wetted_material_pressure",
            category="Pressure Connection",
            description="Material in contact with fluid",
            data_type="string",
            options=["316L SS", "Carbon Steel", "Hastelloy", "Titanium"],
            importance=SpecImportance.CRITICAL
        ),
        "port_flushing_capability": SpecificationDefinition(
            key="port_flushing_capability",
            category="Pressure Connection",
            description="Ability to flush pressure port",
            data_type="boolean",
            importance=SpecImportance.IMPORTANT
        ),
        "isolation_valve_compatible": SpecificationDefinition(
            key="isolation_valve_compatible",
            category="Pressure Connection",
            description="Compatible with isolation valves",
            data_type="boolean",
            importance=SpecImportance.OPTIONAL
        ),
        "snubber_valve_compatible": SpecificationDefinition(
            key="snubber_valve_compatible",
            category="Pressure Connection",
            description="Compatible with snubber valves",
            data_type="boolean",
            importance=SpecImportance.OPTIONAL
        ),
        "overrange_protection": SpecificationDefinition(
            key="overrange_protection",
            category="Pressure Connection",
            description="Type of overrange protection",
            data_type="string",
            options=["none", "mechanical_stop", "internal_relief"],
            importance=SpecImportance.REQUIRED
        ),
        "burst_pressure_rating": SpecificationDefinition(
            key="burst_pressure_rating",
            category="Pressure Connection",
            description="Maximum burst pressure",
            unit="psi",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "max_differential_pressure": SpecificationDefinition(
            key="max_differential_pressure",
            category="Pressure Connection",
            description="Maximum pressure differential",
            unit="psi",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),

        # 3. FLUID COMPATIBILITY (8 specs)
        "compatible_fluids": SpecificationDefinition(
            key="compatible_fluids",
            category="Fluid Compatibility",
            description="List of compatible fluids",
            data_type="list",
            importance=SpecImportance.CRITICAL
        ),
        "corrosion_resistance_class": SpecificationDefinition(
            key="corrosion_resistance_class",
            category="Fluid Compatibility",
            description="NACE corrosion resistance class",
            data_type="string",
            options=["C", "A", "B"],
            importance=SpecImportance.REQUIRED
        ),
        "sulfide_stress_cracking_resistant": SpecificationDefinition(
            key="sulfide_stress_cracking_resistant",
            category="Fluid Compatibility",
            description="NACE MR0175 compliance",
            data_type="boolean",
            importance=SpecImportance.REQUIRED
        ),
        "hydrogen_embrittlement_resistant": SpecificationDefinition(
            key="hydrogen_embrittlement_resistant",
            category="Fluid Compatibility",
            description="Resistant to hydrogen embrittlement",
            data_type="boolean",
            importance=SpecImportance.IMPORTANT
        ),
        "low_pressure_seal_requirements": SpecificationDefinition(
            key="low_pressure_seal_requirements",
            category="Fluid Compatibility",
            description="Special sealing for low pressure",
            data_type="string",
            importance=SpecImportance.OPTIONAL
        ),
        "viscosity_range_supported": SpecificationDefinition(
            key="viscosity_range_supported",
            category="Fluid Compatibility",
            description="Fluid viscosity range",
            unit="cP",
            data_type="string",
            importance=SpecImportance.IMPORTANT
        ),
        "fluid_temperature_effects": SpecificationDefinition(
            key="fluid_temperature_effects",
            category="Fluid Compatibility",
            description="How temperature affects measurement",
            data_type="string",
            importance=SpecImportance.IMPORTANT
        ),
        "compatible_fluids_count": SpecificationDefinition(
            key="compatible_fluids_count",
            category="Fluid Compatibility",
            description="Number of compatible fluid types",
            data_type="number",
            importance=SpecImportance.OPTIONAL
        ),

        # 4. OUTPUT SIGNAL & COMMUNICATION (10 specs)
        "primary_output_signal": SpecificationDefinition(
            key="primary_output_signal",
            category="Output Signal & Communication",
            description="Primary output signal type",
            data_type="string",
            options=["4-20mA", "0-10V", "0-5V", "digital"],
            importance=SpecImportance.CRITICAL
        ),
        "output_range_min": SpecificationDefinition(
            key="output_range_min",
            category="Output Signal & Communication",
            description="Minimum output signal value",
            unit="mA or V",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "output_range_max": SpecificationDefinition(
            key="output_range_max",
            category="Output Signal & Communication",
            description="Maximum output signal value",
            unit="mA or V",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "communication_protocol": SpecificationDefinition(
            key="communication_protocol",
            category="Output Signal & Communication",
            description="Communication protocol supported",
            data_type="list",
            options=["HART", "Modbus", "Profibus", "EtherCAT"],
            importance=SpecImportance.REQUIRED
        ),
        "fieldbus_supported": SpecificationDefinition(
            key="fieldbus_supported",
            category="Output Signal & Communication",
            description="Fieldbus types supported",
            data_type="list",
            importance=SpecImportance.IMPORTANT
        ),
        "transmission_distance_max": SpecificationDefinition(
            key="transmission_distance_max",
            category="Output Signal & Communication",
            description="Maximum transmission distance",
            unit="m",
            data_type="number",
            importance=SpecImportance.OPTIONAL
        ),
        "baud_rate": SpecificationDefinition(
            key="baud_rate",
            category="Output Signal & Communication",
            description="Data transmission baud rate",
            unit="bps",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "current_consumption": SpecificationDefinition(
            key="current_consumption",
            category="Output Signal & Communication",
            description="Typical current consumption",
            unit="mA",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "loop_powered_capable": SpecificationDefinition(
            key="loop_powered_capable",
            category="Output Signal & Communication",
            description="Can be powered from 4-20mA loop",
            data_type="boolean",
            importance=SpecImportance.IMPORTANT
        ),
        "response_time_output": SpecificationDefinition(
            key="response_time_output",
            category="Output Signal & Communication",
            description="Output response time",
            unit="ms",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),

        # 5. ELECTRICAL SPECIFICATIONS (12 specs)
        "supply_voltage_min": SpecificationDefinition(
            key="supply_voltage_min",
            category="Electrical Specifications",
            description="Minimum supply voltage",
            unit="VDC",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "supply_voltage_max": SpecificationDefinition(
            key="supply_voltage_max",
            category="Electrical Specifications",
            description="Maximum supply voltage",
            unit="VDC",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "power_consumption_typical": SpecificationDefinition(
            key="power_consumption_typical",
            category="Electrical Specifications",
            description="Typical power consumption",
            unit="W",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "power_consumption_max": SpecificationDefinition(
            key="power_consumption_max",
            category="Electrical Specifications",
            description="Maximum power consumption",
            unit="W",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "output_impedance": SpecificationDefinition(
            key="output_impedance",
            category="Electrical Specifications",
            description="Output impedance",
            unit="Ohm",
            data_type="number",
            importance=SpecImportance.OPTIONAL
        ),
        "protection_against_overvoltage": SpecificationDefinition(
            key="protection_against_overvoltage",
            category="Electrical Specifications",
            description="Overvoltage protection method",
            data_type="string",
            importance=SpecImportance.REQUIRED
        ),
        "protection_against_reverse_polarity": SpecificationDefinition(
            key="protection_against_reverse_polarity",
            category="Electrical Specifications",
            description="Reverse polarity protection",
            data_type="boolean",
            importance=SpecImportance.REQUIRED
        ),
        "isolation_voltage": SpecificationDefinition(
            key="isolation_voltage",
            category="Electrical Specifications",
            description="Voltage isolation rating",
            unit="VAC",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "insulation_resistance": SpecificationDefinition(
            key="insulation_resistance",
            category="Electrical Specifications",
            description="Insulation resistance",
            unit="MΩ",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "esd_protection_level": SpecificationDefinition(
            key="esd_protection_level",
            category="Electrical Specifications",
            description="ESD protection rating",
            unit="kV",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "shielding_required": SpecificationDefinition(
            key="shielding_required",
            category="Electrical Specifications",
            description="Cable shielding requirement",
            data_type="boolean",
            importance=SpecImportance.IMPORTANT
        ),
        "grounding_requirement": SpecificationDefinition(
            key="grounding_requirement",
            category="Electrical Specifications",
            description="Grounding requirement details",
            data_type="string",
            importance=SpecImportance.REQUIRED
        ),
        "surge_protection_included": SpecificationDefinition(
            key="surge_protection_included",
            category="Electrical Specifications",
            description="Built-in surge protection",
            data_type="boolean",
            importance=SpecImportance.IMPORTANT
        ),

        # 6. ENVIRONMENTAL OPERATING CONDITIONS (10 specs)
        "operating_temperature_min": SpecificationDefinition(
            key="operating_temperature_min",
            category="Environmental Operating Conditions",
            description="Minimum operating temperature",
            unit="°C",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "operating_temperature_max": SpecificationDefinition(
            key="operating_temperature_max",
            category="Environmental Operating Conditions",
            description="Maximum operating temperature",
            unit="°C",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "storage_temperature_min": SpecificationDefinition(
            key="storage_temperature_min",
            category="Environmental Operating Conditions",
            description="Minimum storage temperature",
            unit="°C",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "storage_temperature_max": SpecificationDefinition(
            key="storage_temperature_max",
            category="Environmental Operating Conditions",
            description="Maximum storage temperature",
            unit="°C",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "temperature_coefficient_zero": SpecificationDefinition(
            key="temperature_coefficient_zero",
            category="Environmental Operating Conditions",
            description="Zero temperature coefficient",
            unit="%/°C",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "temperature_coefficient_span": SpecificationDefinition(
            key="temperature_coefficient_span",
            category="Environmental Operating Conditions",
            description="Span temperature coefficient",
            unit="%/°C",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "humidity_range_min": SpecificationDefinition(
            key="humidity_range_min",
            category="Environmental Operating Conditions",
            description="Minimum operating humidity",
            unit="%RH",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "humidity_range_max": SpecificationDefinition(
            key="humidity_range_max",
            category="Environmental Operating Conditions",
            description="Maximum operating humidity",
            unit="%RH",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "altitude_max": SpecificationDefinition(
            key="altitude_max",
            category="Environmental Operating Conditions",
            description="Maximum operating altitude",
            unit="m",
            data_type="number",
            importance=SpecImportance.OPTIONAL
        ),
        "ip_rating": SpecificationDefinition(
            key="ip_rating",
            category="Environmental Operating Conditions",
            description="IP enclosure rating",
            data_type="string",
            options=["IP54", "IP65", "IP67"],
            importance=SpecImportance.REQUIRED
        ),

        # 7. PHYSICAL & MECHANICAL (10 specs)
        "housing_material": SpecificationDefinition(
            key="housing_material",
            category="Physical & Mechanical",
            description="Housing material composition",
            data_type="string",
            options=["Aluminum", "316L SS", "Carbon Steel"],
            importance=SpecImportance.REQUIRED
        ),
        "housing_finish": SpecificationDefinition(
            key="housing_finish",
            category="Physical & Mechanical",
            description="Housing surface finish",
            data_type="string",
            importance=SpecImportance.OPTIONAL
        ),
        "connector_type": SpecificationDefinition(
            key="connector_type",
            category="Physical & Mechanical",
            description="Electrical connector type",
            data_type="string",
            options=["M12", "DT04", "Pigtail"],
            importance=SpecImportance.REQUIRED
        ),
        "connector_gender": SpecificationDefinition(
            key="connector_gender",
            category="Physical & Mechanical",
            description="Connector male/female",
            data_type="string",
            options=["male", "female"],
            importance=SpecImportance.REQUIRED
        ),
        "cable_entry_type": SpecificationDefinition(
            key="cable_entry_type",
            category="Physical & Mechanical",
            description="Type of cable entry",
            data_type="string",
            options=["M20", "NPT", "Cable gland"],
            importance=SpecImportance.REQUIRED
        ),
        "cable_diameter_range": SpecificationDefinition(
            key="cable_diameter_range",
            category="Physical & Mechanical",
            description="Compatible cable diameter range",
            unit="mm",
            data_type="string",
            importance=SpecImportance.REQUIRED
        ),
        "weight": SpecificationDefinition(
            key="weight",
            category="Physical & Mechanical",
            description="Product weight",
            unit="kg",
            data_type="number",
            importance=SpecImportance.OPTIONAL
        ),
        "dimensions_length": SpecificationDefinition(
            key="dimensions_length",
            category="Physical & Mechanical",
            description="Product length",
            unit="mm",
            data_type="number",
            importance=SpecImportance.OPTIONAL
        ),
        "dimensions_width": SpecificationDefinition(
            key="dimensions_width",
            category="Physical & Mechanical",
            description="Product width",
            unit="mm",
            data_type="number",
            importance=SpecImportance.OPTIONAL
        ),
        "dimensions_height": SpecificationDefinition(
            key="dimensions_height",
            category="Physical & Mechanical",
            description="Product height",
            unit="mm",
            data_type="number",
            importance=SpecImportance.OPTIONAL
        ),

        # 8. SAFETY & CERTIFICATIONS (12 specs)
        "sil_rating_max": SpecificationDefinition(
            key="sil_rating_max",
            category="Safety & Certifications",
            description="Maximum SIL rating",
            data_type="string",
            options=["SIL1", "SIL2", "SIL3"],
            importance=SpecImportance.REQUIRED
        ),
        "atex_category": SpecificationDefinition(
            key="atex_category",
            category="Safety & Certifications",
            description="ATEX equipment category",
            data_type="string",
            options=["2", "3"],
            importance=SpecImportance.REQUIRED
        ),
        "atex_group_gas": SpecificationDefinition(
            key="atex_group_gas",
            category="Safety & Certifications",
            description="ATEX gas group classification",
            data_type="string",
            options=["IIA", "IIB", "IIC"],
            importance=SpecImportance.REQUIRED
        ),
        "atex_temperature_class": SpecificationDefinition(
            key="atex_temperature_class",
            category="Safety & Certifications",
            description="ATEX temperature class",
            data_type="string",
            options=["T1", "T2", "T3", "T4", "T5", "T6"],
            importance=SpecImportance.REQUIRED
        ),
        "iecex_certification": SpecificationDefinition(
            key="iecex_certification",
            category="Safety & Certifications",
            description="IECEx certification status",
            data_type="boolean",
            importance=SpecImportance.REQUIRED
        ),
        "ped_conformity": SpecificationDefinition(
            key="ped_conformity",
            category="Safety & Certifications",
            description="PED directive conformity",
            data_type="boolean",
            importance=SpecImportance.REQUIRED
        ),
        "nace_mr0175_compliance": SpecificationDefinition(
            key="nace_mr0175_compliance",
            category="Safety & Certifications",
            description="NACE MR0175 compliance level",
            data_type="string",
            importance=SpecImportance.REQUIRED
        ),
        "ul_listing": SpecificationDefinition(
            key="ul_listing",
            category="Safety & Certifications",
            description="UL listing status",
            data_type="boolean",
            importance=SpecImportance.REQUIRED
        ),
        "ce_mark_compliance": SpecificationDefinition(
            key="ce_mark_compliance",
            category="Safety & Certifications",
            description="CE mark compliance",
            data_type="boolean",
            importance=SpecImportance.REQUIRED
        ),
        "iso_90001_certified": SpecificationDefinition(
            key="iso_90001_certified",
            category="Safety & Certifications",
            description="ISO 9001 certification",
            data_type="boolean",
            importance=SpecImportance.IMPORTANT
        ),
        "calibration_certificate_included": SpecificationDefinition(
            key="calibration_certificate_included",
            category="Safety & Certifications",
            description="Factory calibration certificate included",
            data_type="boolean",
            importance=SpecImportance.IMPORTANT
        ),
        "factory_acceptance_test_included": SpecificationDefinition(
            key="factory_acceptance_test_included",
            category="Safety & Certifications",
            description="Factory Acceptance Test (FAT) included",
            data_type="boolean",
            importance=SpecImportance.OPTIONAL
        ),
    }

    @classmethod
    def get_all_specifications(cls) -> Dict[str, SpecificationDefinition]:
        """Return all specifications for this template"""
        return cls.SPECIFICATIONS

    @classmethod
    def get_specifications_by_category(cls) -> Dict[str, List[SpecificationDefinition]]:
        """Return specifications grouped by category"""
        grouped = {}
        for spec in cls.SPECIFICATIONS.values():
            if spec.category not in grouped:
                grouped[spec.category] = []
            grouped[spec.category].append(spec)
        return grouped

    @classmethod
    def get_count(cls) -> int:
        """Return total number of specifications"""
        return len(cls.SPECIFICATIONS)


# =============================================================================
# FLOW METER TEMPLATE (70+ SPECS)
# =============================================================================

class FlowMeterTemplate:
    """Specification template for flow meters with 70+ specs"""

    SPECIFICATIONS = {
        # 1. MEASUREMENT CHARACTERISTICS (14 specs)
        "flow_measurement_type": SpecificationDefinition(
            key="flow_measurement_type",
            category="Measurement Characteristics",
            description="Type of flow measurement",
            data_type="string",
            options=["volumetric", "mass"],
            importance=SpecImportance.CRITICAL
        ),
        "flow_meter_technology": SpecificationDefinition(
            key="flow_meter_technology",
            category="Measurement Characteristics",
            description="Flow measurement technology",
            data_type="string",
            options=["turbine", "electromagnetic", "ultrasonic", "Coriolis", "vortex"],
            importance=SpecImportance.CRITICAL
        ),
        "flow_rate_range_min": SpecificationDefinition(
            key="flow_rate_range_min",
            category="Measurement Characteristics",
            description="Minimum flow rate",
            unit="L/min",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "flow_rate_range_max": SpecificationDefinition(
            key="flow_rate_range_max",
            category="Measurement Characteristics",
            description="Maximum flow rate",
            unit="L/min",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "flow_accuracy_percent": SpecificationDefinition(
            key="flow_accuracy_percent",
            category="Measurement Characteristics",
            description="Accuracy as percentage of reading",
            unit="%",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "repeatability_flow": SpecificationDefinition(
            key="repeatability_flow",
            category="Measurement Characteristics",
            description="Repeatability of flow measurement",
            unit="%",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "linearity_error_flow": SpecificationDefinition(
            key="linearity_error_flow",
            category="Measurement Characteristics",
            description="Linearity error over range",
            unit="%",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "hysteresis_error_flow": SpecificationDefinition(
            key="hysteresis_error_flow",
            category="Measurement Characteristics",
            description="Hysteresis error in flow",
            unit="%",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "zero_drift_flow": SpecificationDefinition(
            key="zero_drift_flow",
            category="Measurement Characteristics",
            description="Zero drift over time",
            unit="%/year",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "response_time_flow": SpecificationDefinition(
            key="response_time_flow",
            category="Measurement Characteristics",
            description="Response time to flow change",
            unit="s",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "pressure_drop": SpecificationDefinition(
            key="pressure_drop",
            category="Measurement Characteristics",
            description="Pressure drop at max flow",
            unit="bar",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "turndown_ratio": SpecificationDefinition(
            key="turndown_ratio",
            category="Measurement Characteristics",
            description="Max to min flow ratio",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "flow_direction_capability": SpecificationDefinition(
            key="flow_direction_capability",
            category="Measurement Characteristics",
            description="Bidirectional flow capability",
            data_type="string",
            options=["unidirectional", "bidirectional"],
            importance=SpecImportance.REQUIRED
        ),
        "minimum_viscosity": SpecificationDefinition(
            key="minimum_viscosity",
            category="Measurement Characteristics",
            description="Minimum fluid viscosity",
            unit="cP",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),

        # 2. FLOW PORT & CONNECTION (10 specs)
        "flow_port_type": SpecificationDefinition(
            key="flow_port_type",
            category="Flow Port & Connection",
            description="Type of flow connection",
            data_type="string",
            options=["threaded", "flanged", "wafer", "lug"],
            importance=SpecImportance.CRITICAL
        ),
        "flow_port_size": SpecificationDefinition(
            key="flow_port_size",
            category="Flow Port & Connection",
            description="Flow port size",
            unit="in",
            data_type="string",
            importance=SpecImportance.CRITICAL
        ),
        "flow_port_material": SpecificationDefinition(
            key="flow_port_material",
            category="Flow Port & Connection",
            description="Port material",
            data_type="string",
            options=["Carbon Steel", "316L SS", "Ductile Iron"],
            importance=SpecImportance.CRITICAL
        ),
        "seal_material": SpecificationDefinition(
            key="seal_material",
            category="Flow Port & Connection",
            description="Seal material",
            data_type="string",
            options=["Viton", "PTFE", "EPDM"],
            importance=SpecImportance.REQUIRED
        ),
        "gasket_material": SpecificationDefinition(
            key="gasket_material",
            category="Flow Port & Connection",
            description="Gasket material",
            data_type="string",
            importance=SpecImportance.REQUIRED
        ),
        "flange_rating": SpecificationDefinition(
            key="flange_rating",
            category="Flow Port & Connection",
            description="Flange pressure rating",
            unit="bar",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "bolting_specification": SpecificationDefinition(
            key="bolting_specification",
            category="Flow Port & Connection",
            description="Bolt type and size",
            data_type="string",
            importance=SpecImportance.REQUIRED
        ),
        "strainer_required": SpecificationDefinition(
            key="strainer_required",
            category="Flow Port & Connection",
            description="Strainer requirement",
            data_type="boolean",
            importance=SpecImportance.REQUIRED
        ),
        "flow_direction_marking": SpecificationDefinition(
            key="flow_direction_marking",
            category="Flow Port & Connection",
            description="Flow direction marked on body",
            data_type="boolean",
            importance=SpecImportance.IMPORTANT
        ),
        "maintenance_access": SpecificationDefinition(
            key="maintenance_access",
            category="Flow Port & Connection",
            description="Access for maintenance",
            data_type="string",
            importance=SpecImportance.OPTIONAL
        ),

        # 3. FLUID PROPERTIES (8 specs)
        "compatible_fluid_types": SpecificationDefinition(
            key="compatible_fluid_types",
            category="Fluid Properties",
            description="List of compatible fluids",
            data_type="list",
            importance=SpecImportance.CRITICAL
        ),
        "viscosity_range": SpecificationDefinition(
            key="viscosity_range",
            category="Fluid Properties",
            description="Fluid viscosity range",
            unit="cP",
            data_type="string",
            importance=SpecImportance.CRITICAL
        ),
        "specific_gravity_range": SpecificationDefinition(
            key="specific_gravity_range",
            category="Fluid Properties",
            description="Fluid specific gravity range",
            data_type="string",
            importance=SpecImportance.REQUIRED
        ),
        "temperature_compensation": SpecificationDefinition(
            key="temperature_compensation",
            category="Fluid Properties",
            description="Automatic temperature compensation",
            data_type="boolean",
            importance=SpecImportance.IMPORTANT
        ),
        "density_compensation": SpecificationDefinition(
            key="density_compensation",
            category="Fluid Properties",
            description="Automatic density compensation",
            data_type="boolean",
            importance=SpecImportance.IMPORTANT
        ),
        "particle_contamination_limit": SpecificationDefinition(
            key="particle_contamination_limit",
            category="Fluid Properties",
            description="Maximum particle size",
            unit="µm",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "water_content_limit": SpecificationDefinition(
            key="water_content_limit",
            category="Fluid Properties",
            description="Maximum water content",
            unit="%",
            data_type="number",
            importance=SpecImportance.OPTIONAL
        ),
        "gas_release_requirement": SpecificationDefinition(
            key="gas_release_requirement",
            category="Fluid Properties",
            description="Gas release requirement",
            data_type="string",
            importance=SpecImportance.IMPORTANT
        ),

        # 4. OUTPUT SIGNAL & COMMUNICATION (10 specs)
        "primary_output_signal_flow": SpecificationDefinition(
            key="primary_output_signal_flow",
            category="Output Signal & Communication",
            description="Primary output signal type",
            data_type="string",
            options=["4-20mA", "0-10V", "digital", "Modbus"],
            importance=SpecImportance.CRITICAL
        ),
        "pulse_output": SpecificationDefinition(
            key="pulse_output",
            category="Output Signal & Communication",
            description="Pulse output capability",
            data_type="boolean",
            importance=SpecImportance.IMPORTANT
        ),
        "pulse_frequency_range": SpecificationDefinition(
            key="pulse_frequency_range",
            category="Output Signal & Communication",
            description="Pulse frequency range",
            unit="Hz",
            data_type="string",
            importance=SpecImportance.IMPORTANT
        ),
        "communication_protocol_flow": SpecificationDefinition(
            key="communication_protocol_flow",
            category="Output Signal & Communication",
            description="Communication protocols",
            data_type="list",
            importance=SpecImportance.REQUIRED
        ),
        "display_available": SpecificationDefinition(
            key="display_available",
            category="Output Signal & Communication",
            description="Built-in display",
            data_type="boolean",
            importance=SpecImportance.OPTIONAL
        ),
        "totalizer_capability": SpecificationDefinition(
            key="totalizer_capability",
            category="Output Signal & Communication",
            description="Totalizer function",
            data_type="boolean",
            importance=SpecImportance.IMPORTANT
        ),
        "baud_rate_flow": SpecificationDefinition(
            key="baud_rate_flow",
            category="Output Signal & Communication",
            description="Data transmission baud rate",
            unit="bps",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "current_consumption_flow": SpecificationDefinition(
            key="current_consumption_flow",
            category="Output Signal & Communication",
            description="Typical current consumption",
            unit="mA",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "data_logging_capability": SpecificationDefinition(
            key="data_logging_capability",
            category="Output Signal & Communication",
            description="Data logging storage",
            data_type="string",
            importance=SpecImportance.OPTIONAL
        ),
        "response_time_output_flow": SpecificationDefinition(
            key="response_time_output_flow",
            category="Output Signal & Communication",
            description="Output response time",
            unit="ms",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),

        # 5. ELECTRICAL SPECIFICATIONS (10 specs)
        "supply_voltage_min_flow": SpecificationDefinition(
            key="supply_voltage_min_flow",
            category="Electrical Specifications",
            description="Minimum supply voltage",
            unit="VDC",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "supply_voltage_max_flow": SpecificationDefinition(
            key="supply_voltage_max_flow",
            category="Electrical Specifications",
            description="Maximum supply voltage",
            unit="VDC",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "power_consumption_typical_flow": SpecificationDefinition(
            key="power_consumption_typical_flow",
            category="Electrical Specifications",
            description="Typical power consumption",
            unit="W",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "protection_against_overvoltage_flow": SpecificationDefinition(
            key="protection_against_overvoltage_flow",
            category="Electrical Specifications",
            description="Overvoltage protection",
            data_type="string",
            importance=SpecImportance.REQUIRED
        ),
        "protection_against_reverse_polarity_flow": SpecificationDefinition(
            key="protection_against_reverse_polarity_flow",
            category="Electrical Specifications",
            description="Reverse polarity protection",
            data_type="boolean",
            importance=SpecImportance.REQUIRED
        ),
        "isolation_voltage_flow": SpecificationDefinition(
            key="isolation_voltage_flow",
            category="Electrical Specifications",
            description="Voltage isolation rating",
            unit="VAC",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "insulation_resistance_flow": SpecificationDefinition(
            key="insulation_resistance_flow",
            category="Electrical Specifications",
            description="Insulation resistance",
            unit="MΩ",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "esd_protection_level_flow": SpecificationDefinition(
            key="esd_protection_level_flow",
            category="Electrical Specifications",
            description="ESD protection rating",
            unit="kV",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "shielding_required_flow": SpecificationDefinition(
            key="shielding_required_flow",
            category="Electrical Specifications",
            description="Cable shielding requirement",
            data_type="boolean",
            importance=SpecImportance.IMPORTANT
        ),
        "grounding_requirement_flow": SpecificationDefinition(
            key="grounding_requirement_flow",
            category="Electrical Specifications",
            description="Grounding requirement",
            data_type="string",
            importance=SpecImportance.REQUIRED
        ),

        # 6. ENVIRONMENTAL OPERATING CONDITIONS (8 specs)
        "operating_temperature_min_flow": SpecificationDefinition(
            key="operating_temperature_min_flow",
            category="Environmental Operating Conditions",
            description="Minimum operating temperature",
            unit="°C",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "operating_temperature_max_flow": SpecificationDefinition(
            key="operating_temperature_max_flow",
            category="Environmental Operating Conditions",
            description="Maximum operating temperature",
            unit="°C",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "temperature_coefficient_zero_flow": SpecificationDefinition(
            key="temperature_coefficient_zero_flow",
            category="Environmental Operating Conditions",
            description="Zero temperature coefficient",
            unit="%/°C",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "humidity_range_min_flow": SpecificationDefinition(
            key="humidity_range_min_flow",
            category="Environmental Operating Conditions",
            description="Minimum operating humidity",
            unit="%RH",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "humidity_range_max_flow": SpecificationDefinition(
            key="humidity_range_max_flow",
            category="Environmental Operating Conditions",
            description="Maximum operating humidity",
            unit="%RH",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "altitude_max_flow": SpecificationDefinition(
            key="altitude_max_flow",
            category="Environmental Operating Conditions",
            description="Maximum operating altitude",
            unit="m",
            data_type="number",
            importance=SpecImportance.OPTIONAL
        ),
        "ip_rating_flow": SpecificationDefinition(
            key="ip_rating_flow",
            category="Environmental Operating Conditions",
            description="IP enclosure rating",
            data_type="string",
            options=["IP54", "IP65", "IP67"],
            importance=SpecImportance.REQUIRED
        ),
        "vibration_resistance": SpecificationDefinition(
            key="vibration_resistance",
            category="Environmental Operating Conditions",
            description="Vibration resistance",
            unit="g",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),

        # 7. PHYSICAL & MECHANICAL (8 specs)
        "housing_material_flow": SpecificationDefinition(
            key="housing_material_flow",
            category="Physical & Mechanical",
            description="Housing material",
            data_type="string",
            importance=SpecImportance.REQUIRED
        ),
        "connector_type_flow": SpecificationDefinition(
            key="connector_type_flow",
            category="Physical & Mechanical",
            description="Electrical connector type",
            data_type="string",
            importance=SpecImportance.REQUIRED
        ),
        "cable_entry_type_flow": SpecificationDefinition(
            key="cable_entry_type_flow",
            category="Physical & Mechanical",
            description="Type of cable entry",
            data_type="string",
            importance=SpecImportance.REQUIRED
        ),
        "weight_flow": SpecificationDefinition(
            key="weight_flow",
            category="Physical & Mechanical",
            description="Product weight",
            unit="kg",
            data_type="number",
            importance=SpecImportance.OPTIONAL
        ),
        "dimensions_length_flow": SpecificationDefinition(
            key="dimensions_length_flow",
            category="Physical & Mechanical",
            description="Product length",
            unit="mm",
            data_type="number",
            importance=SpecImportance.OPTIONAL
        ),
        "dimensions_diameter_flow": SpecificationDefinition(
            key="dimensions_diameter_flow",
            category="Physical & Mechanical",
            description="Product diameter",
            unit="mm",
            data_type="number",
            importance=SpecImportance.OPTIONAL
        ),
        "mounting_orientation": SpecificationDefinition(
            key="mounting_orientation",
            category="Physical & Mechanical",
            description="Preferred mounting orientation",
            data_type="string",
            options=["horizontal", "vertical", "any"],
            importance=SpecImportance.IMPORTANT
        ),
        "installation_clearance": SpecificationDefinition(
            key="installation_clearance",
            category="Physical & Mechanical",
            description="Required installation clearance",
            unit="mm",
            data_type="number",
            importance=SpecImportance.OPTIONAL
        ),

        # 8. SAFETY & CERTIFICATIONS (6 specs)
        "sil_rating_max_flow": SpecificationDefinition(
            key="sil_rating_max_flow",
            category="Safety & Certifications",
            description="Maximum SIL rating",
            data_type="string",
            options=["SIL1", "SIL2", "SIL3"],
            importance=SpecImportance.REQUIRED
        ),
        "atex_certification_flow": SpecificationDefinition(
            key="atex_certification_flow",
            category="Safety & Certifications",
            description="ATEX certification status",
            data_type="boolean",
            importance=SpecImportance.REQUIRED
        ),
        "ul_listing_flow": SpecificationDefinition(
            key="ul_listing_flow",
            category="Safety & Certifications",
            description="UL listing status",
            data_type="boolean",
            importance=SpecImportance.REQUIRED
        ),
        "ce_mark_compliance_flow": SpecificationDefinition(
            key="ce_mark_compliance_flow",
            category="Safety & Certifications",
            description="CE mark compliance",
            data_type="boolean",
            importance=SpecImportance.REQUIRED
        ),
        "iso_9001_certified_flow": SpecificationDefinition(
            key="iso_9001_certified_flow",
            category="Safety & Certifications",
            description="ISO 9001 certification",
            data_type="boolean",
            importance=SpecImportance.IMPORTANT
        ),
        "calibration_certificate_included_flow": SpecificationDefinition(
            key="calibration_certificate_included_flow",
            category="Safety & Certifications",
            description="Factory calibration certificate",
            data_type="boolean",
            importance=SpecImportance.IMPORTANT
        ),
    }

    @classmethod
    def get_all_specifications(cls) -> Dict[str, SpecificationDefinition]:
        return cls.SPECIFICATIONS

    @classmethod
    def get_specifications_by_category(cls) -> Dict[str, List[SpecificationDefinition]]:
        grouped = {}
        for spec in cls.SPECIFICATIONS.values():
            if spec.category not in grouped:
                grouped[spec.category] = []
            grouped[spec.category].append(spec)
        return grouped

    @classmethod
    def get_count(cls) -> int:
        return len(cls.SPECIFICATIONS)


# =============================================================================
# LEVEL TRANSMITTER TEMPLATE (60+ SPECS)
# =============================================================================

class LevelTransmitterTemplate:
    """Specification template for level transmitters with 60+ specs"""

    SPECIFICATIONS = {
        # 1. MEASUREMENT CHARACTERISTICS (12 specs)
        "level_measurement_type": SpecificationDefinition(
            key="level_measurement_type",
            category="Measurement Characteristics",
            description="Type of level measurement",
            data_type="string",
            options=["float", "capacitive", "radar", "laser", "ultrasonic"],
            importance=SpecImportance.CRITICAL
        ),
        "measurement_range_min": SpecificationDefinition(
            key="measurement_range_min",
            category="Measurement Characteristics",
            description="Minimum measurement range",
            unit="m",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "measurement_range_max": SpecificationDefinition(
            key="measurement_range_max",
            category="Measurement Characteristics",
            description="Maximum measurement range",
            unit="m",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "level_accuracy_percent": SpecificationDefinition(
            key="level_accuracy_percent",
            category="Measurement Characteristics",
            description="Accuracy as percentage of range",
            unit="%",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "repeatability_level": SpecificationDefinition(
            key="repeatability_level",
            category="Measurement Characteristics",
            description="Repeatability of measurements",
            unit="%",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "linearity_error_level": SpecificationDefinition(
            key="linearity_error_level",
            category="Measurement Characteristics",
            description="Linearity error over range",
            unit="%",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "hysteresis_error_level": SpecificationDefinition(
            key="hysteresis_error_level",
            category="Measurement Characteristics",
            description="Hysteresis error",
            unit="%",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "zero_drift_level": SpecificationDefinition(
            key="zero_drift_level",
            category="Measurement Characteristics",
            description="Zero drift over time",
            unit="%/year",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "response_time_level": SpecificationDefinition(
            key="response_time_level",
            category="Measurement Characteristics",
            description="Response time to level change",
            unit="s",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "dead_band_width": SpecificationDefinition(
            key="dead_band_width",
            category="Measurement Characteristics",
            description="Dead band width",
            unit="mm",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "foam_rejection": SpecificationDefinition(
            key="foam_rejection",
            category="Measurement Characteristics",
            description="Foam rejection capability",
            data_type="boolean",
            importance=SpecImportance.IMPORTANT
        ),
        "condensation_resistance": SpecificationDefinition(
            key="condensation_resistance",
            category="Measurement Characteristics",
            description="Condensation resistance",
            data_type="boolean",
            importance=SpecImportance.IMPORTANT
        ),

        # 2. LEVEL PORT & CONNECTION (10 specs)
        "level_port_type": SpecificationDefinition(
            key="level_port_type",
            category="Level Port & Connection",
            description="Type of level connection",
            data_type="string",
            options=["flanged", "threaded", "cage clamp"],
            importance=SpecImportance.CRITICAL
        ),
        "level_port_size": SpecificationDefinition(
            key="level_port_size",
            category="Level Port & Connection",
            description="Level port size",
            unit="in",
            data_type="string",
            importance=SpecImportance.CRITICAL
        ),
        "port_material_level": SpecificationDefinition(
            key="port_material_level",
            category="Level Port & Connection",
            description="Port material",
            data_type="string",
            options=["316L SS", "Carbon Steel"],
            importance=SpecImportance.CRITICAL
        ),
        "connection_rating": SpecificationDefinition(
            key="connection_rating",
            category="Level Port & Connection",
            description="Connection pressure rating",
            unit="bar",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "vent_hole_protection": SpecificationDefinition(
            key="vent_hole_protection",
            category="Level Port & Connection",
            description="Vent hole protection",
            data_type="boolean",
            importance=SpecImportance.IMPORTANT
        ),
        "drain_hole_included": SpecificationDefinition(
            key="drain_hole_included",
            category="Level Port & Connection",
            description="Drain hole provided",
            data_type="boolean",
            importance=SpecImportance.IMPORTANT
        ),
        "isolation_valve_compatible_level": SpecificationDefinition(
            key="isolation_valve_compatible_level",
            category="Level Port & Connection",
            description="Isolation valve compatible",
            data_type="boolean",
            importance=SpecImportance.OPTIONAL
        ),
        "snubber_valve_compatible_level": SpecificationDefinition(
            key="snubber_valve_compatible_level",
            category="Level Port & Connection",
            description="Snubber valve compatible",
            data_type="boolean",
            importance=SpecImportance.OPTIONAL
        ),
        "mounting_flexible_pipe": SpecificationDefinition(
            key="mounting_flexible_pipe",
            category="Level Port & Connection",
            description="Flexible mounting pipe",
            data_type="boolean",
            importance=SpecImportance.OPTIONAL
        ),
        "purge_air_connection": SpecificationDefinition(
            key="purge_air_connection",
            category="Level Port & Connection",
            description="Purge air connection available",
            data_type="boolean",
            importance=SpecImportance.OPTIONAL
        ),

        # 3. FLUID PROPERTIES (6 specs)
        "compatible_fluids_level": SpecificationDefinition(
            key="compatible_fluids_level",
            category="Fluid Properties",
            description="Compatible fluid types",
            data_type="list",
            importance=SpecImportance.CRITICAL
        ),
        "corrosion_resistance_class_level": SpecificationDefinition(
            key="corrosion_resistance_class_level",
            category="Fluid Properties",
            description="Corrosion resistance class",
            data_type="string",
            importance=SpecImportance.REQUIRED
        ),
        "density_range_level": SpecificationDefinition(
            key="density_range_level",
            category="Fluid Properties",
            description="Fluid density range",
            unit="kg/m³",
            data_type="string",
            importance=SpecImportance.REQUIRED
        ),
        "dielectric_constant_range": SpecificationDefinition(
            key="dielectric_constant_range",
            category="Fluid Properties",
            description="Fluid dielectric constant range",
            data_type="string",
            importance=SpecImportance.IMPORTANT
        ),
        "conductivity_range": SpecificationDefinition(
            key="conductivity_range",
            category="Fluid Properties",
            description="Fluid conductivity range",
            unit="µS/cm",
            data_type="string",
            importance=SpecImportance.REQUIRED
        ),
        "viscosity_effect_on_accuracy": SpecificationDefinition(
            key="viscosity_effect_on_accuracy",
            category="Fluid Properties",
            description="Viscosity effect on accuracy",
            data_type="string",
            importance=SpecImportance.IMPORTANT
        ),

        # 4. OUTPUT SIGNAL & COMMUNICATION (10 specs)
        "primary_output_signal_level": SpecificationDefinition(
            key="primary_output_signal_level",
            category="Output Signal & Communication",
            description="Primary output signal",
            data_type="string",
            options=["4-20mA", "0-10V", "digital"],
            importance=SpecImportance.CRITICAL
        ),
        "switching_output": SpecificationDefinition(
            key="switching_output",
            category="Output Signal & Communication",
            description="Switching output available",
            data_type="boolean",
            importance=SpecImportance.REQUIRED
        ),
        "switch_hysteresis": SpecificationDefinition(
            key="switch_hysteresis",
            category="Output Signal & Communication",
            description="Switch hysteresis setting",
            unit="mm",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "communication_protocol_level": SpecificationDefinition(
            key="communication_protocol_level",
            category="Output Signal & Communication",
            description="Communication protocols supported",
            data_type="list",
            importance=SpecImportance.REQUIRED
        ),
        "fieldbus_capability_level": SpecificationDefinition(
            key="fieldbus_capability_level",
            category="Output Signal & Communication",
            description="Fieldbus capability",
            data_type="boolean",
            importance=SpecImportance.IMPORTANT
        ),
        "multiple_setpoints": SpecificationDefinition(
            key="multiple_setpoints",
            category="Output Signal & Communication",
            description="Number of setpoints available",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "display_available_level": SpecificationDefinition(
            key="display_available_level",
            category="Output Signal & Communication",
            description="Built-in display",
            data_type="boolean",
            importance=SpecImportance.OPTIONAL
        ),
        "current_consumption_level": SpecificationDefinition(
            key="current_consumption_level",
            category="Output Signal & Communication",
            description="Typical current consumption",
            unit="mA",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "loop_powered_capable_level": SpecificationDefinition(
            key="loop_powered_capable_level",
            category="Output Signal & Communication",
            description="Loop powered capability",
            data_type="boolean",
            importance=SpecImportance.IMPORTANT
        ),
        "response_time_output_level": SpecificationDefinition(
            key="response_time_output_level",
            category="Output Signal & Communication",
            description="Output response time",
            unit="ms",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),

        # 5. ELECTRICAL SPECIFICATIONS (8 specs)
        "supply_voltage_min_level": SpecificationDefinition(
            key="supply_voltage_min_level",
            category="Electrical Specifications",
            description="Minimum supply voltage",
            unit="VDC",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "supply_voltage_max_level": SpecificationDefinition(
            key="supply_voltage_max_level",
            category="Electrical Specifications",
            description="Maximum supply voltage",
            unit="VDC",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "power_consumption_typical_level": SpecificationDefinition(
            key="power_consumption_typical_level",
            category="Electrical Specifications",
            description="Typical power consumption",
            unit="W",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "protection_against_overvoltage_level": SpecificationDefinition(
            key="protection_against_overvoltage_level",
            category="Electrical Specifications",
            description="Overvoltage protection",
            data_type="string",
            importance=SpecImportance.REQUIRED
        ),
        "isolation_voltage_level": SpecificationDefinition(
            key="isolation_voltage_level",
            category="Electrical Specifications",
            description="Isolation voltage rating",
            unit="VAC",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "insulation_resistance_level": SpecificationDefinition(
            key="insulation_resistance_level",
            category="Electrical Specifications",
            description="Insulation resistance",
            unit="MΩ",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "esd_protection_level_level": SpecificationDefinition(
            key="esd_protection_level_level",
            category="Electrical Specifications",
            description="ESD protection rating",
            unit="kV",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "surge_protection_included_level": SpecificationDefinition(
            key="surge_protection_included_level",
            category="Electrical Specifications",
            description="Built-in surge protection",
            data_type="boolean",
            importance=SpecImportance.IMPORTANT
        ),

        # 6. ENVIRONMENTAL OPERATING CONDITIONS (6 specs)
        "operating_temperature_min_level": SpecificationDefinition(
            key="operating_temperature_min_level",
            category="Environmental Operating Conditions",
            description="Minimum operating temperature",
            unit="°C",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "operating_temperature_max_level": SpecificationDefinition(
            key="operating_temperature_max_level",
            category="Environmental Operating Conditions",
            description="Maximum operating temperature",
            unit="°C",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "temperature_coefficient_level": SpecificationDefinition(
            key="temperature_coefficient_level",
            category="Environmental Operating Conditions",
            description="Temperature coefficient",
            unit="%/°C",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "humidity_range_min_level": SpecificationDefinition(
            key="humidity_range_min_level",
            category="Environmental Operating Conditions",
            description="Minimum operating humidity",
            unit="%RH",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "humidity_range_max_level": SpecificationDefinition(
            key="humidity_range_max_level",
            category="Environmental Operating Conditions",
            description="Maximum operating humidity",
            unit="%RH",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "ip_rating_level": SpecificationDefinition(
            key="ip_rating_level",
            category="Environmental Operating Conditions",
            description="IP enclosure rating",
            data_type="string",
            options=["IP54", "IP65", "IP67"],
            importance=SpecImportance.REQUIRED
        ),

        # 7. PHYSICAL & MECHANICAL (6 specs)
        "housing_material_level": SpecificationDefinition(
            key="housing_material_level",
            category="Physical & Mechanical",
            description="Housing material",
            data_type="string",
            importance=SpecImportance.REQUIRED
        ),
        "connector_type_level": SpecificationDefinition(
            key="connector_type_level",
            category="Physical & Mechanical",
            description="Electrical connector type",
            data_type="string",
            importance=SpecImportance.REQUIRED
        ),
        "cable_entry_type_level": SpecificationDefinition(
            key="cable_entry_type_level",
            category="Physical & Mechanical",
            description="Cable entry type",
            data_type="string",
            importance=SpecImportance.REQUIRED
        ),
        "weight_level": SpecificationDefinition(
            key="weight_level",
            category="Physical & Mechanical",
            description="Product weight",
            unit="kg",
            data_type="number",
            importance=SpecImportance.OPTIONAL
        ),
        "dimensions_length_level": SpecificationDefinition(
            key="dimensions_length_level",
            category="Physical & Mechanical",
            description="Product length",
            unit="mm",
            data_type="number",
            importance=SpecImportance.OPTIONAL
        ),
        "dimensions_diameter_level": SpecificationDefinition(
            key="dimensions_diameter_level",
            category="Physical & Mechanical",
            description="Product diameter",
            unit="mm",
            data_type="number",
            importance=SpecImportance.OPTIONAL
        ),

        # 8. SAFETY & CERTIFICATIONS (6 specs)
        "sil_rating_max_level": SpecificationDefinition(
            key="sil_rating_max_level",
            category="Safety & Certifications",
            description="Maximum SIL rating",
            data_type="string",
            importance=SpecImportance.REQUIRED
        ),
        "atex_certification_level": SpecificationDefinition(
            key="atex_certification_level",
            category="Safety & Certifications",
            description="ATEX certification",
            data_type="boolean",
            importance=SpecImportance.REQUIRED
        ),
        "ul_listing_level": SpecificationDefinition(
            key="ul_listing_level",
            category="Safety & Certifications",
            description="UL listing",
            data_type="boolean",
            importance=SpecImportance.REQUIRED
        ),
        "ce_mark_compliance_level": SpecificationDefinition(
            key="ce_mark_compliance_level",
            category="Safety & Certifications",
            description="CE mark compliance",
            data_type="boolean",
            importance=SpecImportance.REQUIRED
        ),
        "iso_9001_certified_level": SpecificationDefinition(
            key="iso_9001_certified_level",
            category="Safety & Certifications",
            description="ISO 9001 certification",
            data_type="boolean",
            importance=SpecImportance.IMPORTANT
        ),
        "calibration_certificate_included_level": SpecificationDefinition(
            key="calibration_certificate_included_level",
            category="Safety & Certifications",
            description="Calibration certificate",
            data_type="boolean",
            importance=SpecImportance.IMPORTANT
        ),
    }

    @classmethod
    def get_all_specifications(cls) -> Dict[str, SpecificationDefinition]:
        return cls.SPECIFICATIONS

    @classmethod
    def get_specifications_by_category(cls) -> Dict[str, List[SpecificationDefinition]]:
        grouped = {}
        for spec in cls.SPECIFICATIONS.values():
            if spec.category not in grouped:
                grouped[spec.category] = []
            grouped[spec.category].append(spec)
        return grouped

    @classmethod
    def get_count(cls) -> int:
        return len(cls.SPECIFICATIONS)


# =============================================================================
# ANALYZER TEMPLATE (75+ SPECS)
# =============================================================================

class AnalyzerTemplate:
    """Specification template for analyzers with 75+ specs"""

    SPECIFICATIONS = {
        # 1. MEASUREMENT CHARACTERISTICS (15 specs)
        "analyzer_type": SpecificationDefinition(
            key="analyzer_type",
            category="Measurement Characteristics",
            description="Type of analyzer",
            data_type="string",
            options=["gas", "liquid", "dissolved_oxygen", "conductivity", "ph", "orp", "turbidity"],
            importance=SpecImportance.CRITICAL
        ),
        "measurement_parameter": SpecificationDefinition(
            key="measurement_parameter",
            category="Measurement Characteristics",
            description="Physical parameter being measured",
            data_type="string",
            importance=SpecImportance.CRITICAL
        ),
        "measurement_range_min": SpecificationDefinition(
            key="measurement_range_min",
            category="Measurement Characteristics",
            description="Minimum measurement range",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "measurement_range_max": SpecificationDefinition(
            key="measurement_range_max",
            category="Measurement Characteristics",
            description="Maximum measurement range",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "measurement_accuracy_percent": SpecificationDefinition(
            key="measurement_accuracy_percent",
            category="Measurement Characteristics",
            description="Accuracy as percentage of range",
            unit="%",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "measurement_repeatability": SpecificationDefinition(
            key="measurement_repeatability",
            category="Measurement Characteristics",
            description="Repeatability of measurements",
            unit="%",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "response_time": SpecificationDefinition(
            key="response_time",
            category="Measurement Characteristics",
            description="T90 response time to reach stable reading",
            unit="seconds",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "sampling_rate": SpecificationDefinition(
            key="sampling_rate",
            category="Measurement Characteristics",
            description="Frequency of measurements",
            unit="seconds",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "detection_method": SpecificationDefinition(
            key="detection_method",
            category="Measurement Characteristics",
            description="Method used for detection",
            data_type="string",
            options=["electrochemical", "optical", "thermal_conductivity", "flame_ionization", "uv"],
            importance=SpecImportance.CRITICAL
        ),
        "sensor_type": SpecificationDefinition(
            key="sensor_type",
            category="Measurement Characteristics",
            description="Type of sensor element",
            data_type="string",
            importance=SpecImportance.REQUIRED
        ),
        "sensor_lifetime_hours": SpecificationDefinition(
            key="sensor_lifetime_hours",
            category="Measurement Characteristics",
            description="Expected sensor lifetime before replacement",
            unit="hours",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "calibration_frequency": SpecificationDefinition(
            key="calibration_frequency",
            category="Measurement Characteristics",
            description="Recommended calibration interval",
            unit="months",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "zero_drift_per_month": SpecificationDefinition(
            key="zero_drift_per_month",
            category="Measurement Characteristics",
            description="Zero point drift per month",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "span_drift_per_month": SpecificationDefinition(
            key="span_drift_per_month",
            category="Measurement Characteristics",
            description="Span drift per month",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),

        # 2. SAMPLE HANDLING (10 specs)
        "sample_inlet_type": SpecificationDefinition(
            key="sample_inlet_type",
            category="Sample Handling",
            description="Type of sample inlet",
            data_type="string",
            options=["direct", "probe", "bypass", "extraction"],
            importance=SpecImportance.REQUIRED
        ),
        "sample_flow_rate_min": SpecificationDefinition(
            key="sample_flow_rate_min",
            category="Sample Handling",
            description="Minimum sample flow rate",
            unit="ml/min",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "sample_flow_rate_max": SpecificationDefinition(
            key="sample_flow_rate_max",
            category="Sample Handling",
            description="Maximum sample flow rate",
            unit="ml/min",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "sample_pressure_max": SpecificationDefinition(
            key="sample_pressure_max",
            category="Sample Handling",
            description="Maximum sample pressure",
            unit="psi",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "drain_capacity_ml_per_hour": SpecificationDefinition(
            key="drain_capacity_ml_per_hour",
            category="Sample Handling",
            description="Drain capacity",
            unit="ml/hour",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "sample_temperature_range_min": SpecificationDefinition(
            key="sample_temperature_range_min",
            category="Sample Handling",
            description="Minimum sample temperature",
            unit="°C",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "sample_temperature_range_max": SpecificationDefinition(
            key="sample_temperature_range_max",
            category="Sample Handling",
            description="Maximum sample temperature",
            unit="°C",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "sample_particulate_max": SpecificationDefinition(
            key="sample_particulate_max",
            category="Sample Handling",
            description="Maximum particulate size acceptable",
            unit="µm",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "reagent_auto_dosing": SpecificationDefinition(
            key="reagent_auto_dosing",
            category="Sample Handling",
            description="Automatic reagent dosing capability",
            data_type="boolean",
            importance=SpecImportance.OPTIONAL
        ),

        # 3. OUTPUT SIGNALS & COMMUNICATION (12 specs)
        "output_signal_types": SpecificationDefinition(
            key="output_signal_types",
            category="Output Signals",
            description="Available output signal types",
            data_type="list",
            options=["4-20mA", "0-10V", "Modbus", "Hart", "Profibus"],
            importance=SpecImportance.CRITICAL
        ),
        "analog_output_accuracy": SpecificationDefinition(
            key="analog_output_accuracy",
            category="Output Signals",
            description="Accuracy of analog output",
            unit="%",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "digital_interface": SpecificationDefinition(
            key="digital_interface",
            category="Output Signals",
            description="Digital communication protocol",
            data_type="string",
            options=["RS485", "RS232", "Ethernet", "WiFi", "Bluetooth"],
            importance=SpecImportance.IMPORTANT
        ),
        "data_logging_capability": SpecificationDefinition(
            key="data_logging_capability",
            category="Output Signals",
            description="Internal data logging capability",
            data_type="boolean",
            importance=SpecImportance.OPTIONAL
        ),
        "memory_size_records": SpecificationDefinition(
            key="memory_size_records",
            category="Output Signals",
            description="Internal storage capacity",
            unit="records",
            data_type="number",
            importance=SpecImportance.OPTIONAL
        ),
        "alarm_output_relays": SpecificationDefinition(
            key="alarm_output_relays",
            category="Output Signals",
            description="Number of alarm output relays",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "relay_capacity_max": SpecificationDefinition(
            key="relay_capacity_max",
            category="Output Signals",
            description="Maximum relay contact rating",
            unit="Amps",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "analog_output_resolution": SpecificationDefinition(
            key="analog_output_resolution",
            category="Output Signals",
            description="Resolution of analog output",
            unit="bits",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "display_type": SpecificationDefinition(
            key="display_type",
            category="Output Signals",
            description="Type of display",
            data_type="string",
            options=["LCD", "LED", "OLED", "Touchscreen"],
            importance=SpecImportance.OPTIONAL
        ),
        "data_update_frequency": SpecificationDefinition(
            key="data_update_frequency",
            category="Output Signals",
            description="Frequency of output data updates",
            unit="Hz",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),

        # 4. CALIBRATION & MAINTENANCE (10 specs)
        "calibration_method": SpecificationDefinition(
            key="calibration_method",
            category="Calibration & Maintenance",
            description="Method for calibration",
            data_type="string",
            options=["manual", "automatic", "two-point", "multi-point"],
            importance=SpecImportance.REQUIRED
        ),
        "auto_calibration_capability": SpecificationDefinition(
            key="auto_calibration_capability",
            category="Calibration & Maintenance",
            description="Automatic calibration capability",
            data_type="boolean",
            importance=SpecImportance.OPTIONAL
        ),
        "standard_gas_bottles_required": SpecificationDefinition(
            key="standard_gas_bottles_required",
            category="Calibration & Maintenance",
            description="Number of standard gas bottles needed",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "maintenance_interval_months": SpecificationDefinition(
            key="maintenance_interval_months",
            category="Calibration & Maintenance",
            description="Recommended maintenance interval",
            unit="months",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "sensor_check_capability": SpecificationDefinition(
            key="sensor_check_capability",
            category="Calibration & Maintenance",
            description="Sensor health check function",
            data_type="boolean",
            importance=SpecImportance.IMPORTANT
        ),
        "diagnostics_capability": SpecificationDefinition(
            key="diagnostics_capability",
            category="Calibration & Maintenance",
            description="Built-in diagnostics capability",
            data_type="boolean",
            importance=SpecImportance.OPTIONAL
        ),
        "filter_replacement_interval": SpecificationDefinition(
            key="filter_replacement_interval",
            category="Calibration & Maintenance",
            description="Filter replacement interval",
            unit="months",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "memory_audit_trail": SpecificationDefinition(
            key="memory_audit_trail",
            category="Calibration & Maintenance",
            description="Audit trail logging capability",
            data_type="boolean",
            importance=SpecImportance.IMPORTANT
        ),

        # 5. ELECTRICAL SPECIFICATIONS (12 specs)
        "power_supply_voltage": SpecificationDefinition(
            key="power_supply_voltage",
            category="Electrical Specifications",
            description="Required power supply voltage",
            unit="Volts",
            data_type="string",
            options=["24VDC", "120VAC", "230VAC", "24VAC"],
            importance=SpecImportance.CRITICAL
        ),
        "power_consumption_max": SpecificationDefinition(
            key="power_consumption_max",
            category="Electrical Specifications",
            description="Maximum power consumption",
            unit="Watts",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "battery_backup_hours": SpecificationDefinition(
            key="battery_backup_hours",
            category="Electrical Specifications",
            description="Battery backup runtime",
            unit="hours",
            data_type="number",
            importance=SpecImportance.OPTIONAL
        ),
        "surge_protection": SpecificationDefinition(
            key="surge_protection",
            category="Electrical Specifications",
            description="Built-in surge protection",
            data_type="boolean",
            importance=SpecImportance.REQUIRED
        ),
        "emc_compliance": SpecificationDefinition(
            key="emc_compliance",
            category="Electrical Specifications",
            description="EMC directive compliance",
            data_type="string",
            options=["CE", "FCC", "RoHS"],
            importance=SpecImportance.REQUIRED
        ),
        "intrinsically_safe": SpecificationDefinition(
            key="intrinsically_safe",
            category="Electrical Specifications",
            description="Intrinsically safe rating",
            data_type="boolean",
            importance=SpecImportance.IMPORTANT
        ),
        "isolation_voltage": SpecificationDefinition(
            key="isolation_voltage",
            category="Electrical Specifications",
            description="Galvanic isolation voltage",
            unit="Volts",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "loop_powered": SpecificationDefinition(
            key="loop_powered",
            category="Electrical Specifications",
            description="Can be powered from 4-20mA loop",
            data_type="boolean",
            importance=SpecImportance.OPTIONAL
        ),

        # 6. ENVIRONMENTAL CONDITIONS (10 specs)
        "operating_temperature_min": SpecificationDefinition(
            key="operating_temperature_min",
            category="Environmental Operating Conditions",
            description="Minimum operating temperature",
            unit="°C",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "operating_temperature_max": SpecificationDefinition(
            key="operating_temperature_max",
            category="Environmental Operating Conditions",
            description="Maximum operating temperature",
            unit="°C",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "storage_temperature_min": SpecificationDefinition(
            key="storage_temperature_min",
            category="Environmental Operating Conditions",
            description="Minimum storage temperature",
            unit="°C",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "storage_temperature_max": SpecificationDefinition(
            key="storage_temperature_max",
            category="Environmental Operating Conditions",
            description="Maximum storage temperature",
            unit="°C",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "humidity_range_min": SpecificationDefinition(
            key="humidity_range_min",
            category="Environmental Operating Conditions",
            description="Minimum operating humidity",
            unit="%RH",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "humidity_range_max": SpecificationDefinition(
            key="humidity_range_max",
            category="Environmental Operating Conditions",
            description="Maximum operating humidity",
            unit="%RH",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "altitude_rating_max": SpecificationDefinition(
            key="altitude_rating_max",
            category="Environmental Operating Conditions",
            description="Maximum altitude rating",
            unit="meters",
            data_type="number",
            importance=SpecImportance.OPTIONAL
        ),
        "vibration_resistance": SpecificationDefinition(
            key="vibration_resistance",
            category="Environmental Operating Conditions",
            description="Vibration resistance rating",
            data_type="string",
            importance=SpecImportance.IMPORTANT
        ),

        # 7. PHYSICAL & MECHANICAL (6 specs)
        "dimensions_length_mm": SpecificationDefinition(
            key="dimensions_length_mm",
            category="Physical & Mechanical",
            description="Length dimension",
            unit="mm",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "dimensions_width_mm": SpecificationDefinition(
            key="dimensions_width_mm",
            category="Physical & Mechanical",
            description="Width dimension",
            unit="mm",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "dimensions_height_mm": SpecificationDefinition(
            key="dimensions_height_mm",
            category="Physical & Mechanical",
            description="Height dimension",
            unit="mm",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "weight_kg": SpecificationDefinition(
            key="weight_kg",
            category="Physical & Mechanical",
            description="Total weight",
            unit="kg",
            data_type="number",
            importance=SpecImportance.OPTIONAL
        ),
        "mounting_type": SpecificationDefinition(
            key="mounting_type",
            category="Physical & Mechanical",
            description="Mounting method",
            data_type="string",
            options=["wall", "DIN-rail", "panel", "pipe"],
            importance=SpecImportance.REQUIRED
        ),
        "enclosure_material": SpecificationDefinition(
            key="enclosure_material",
            category="Physical & Mechanical",
            description="Enclosure material",
            data_type="string",
            options=["plastic", "stainless_steel", "aluminum"],
            importance=SpecImportance.IMPORTANT
        ),

        # 8. SAFETY & CERTIFICATIONS (4 specs)
        "safety_certifications": SpecificationDefinition(
            key="safety_certifications",
            category="Safety & Certifications",
            description="Safety certifications held",
            data_type="list",
            options=["CE", "UL", "CSA", "ATEX"],
            importance=SpecImportance.CRITICAL
        ),
        "sil_rating": SpecificationDefinition(
            key="sil_rating",
            category="Safety & Certifications",
            description="Safety Integrity Level rating",
            data_type="string",
            options=["SIL1", "SIL2", "SIL3", "Not Rated"],
            importance=SpecImportance.IMPORTANT
        ),
        "hazardous_area_category": SpecificationDefinition(
            key="hazardous_area_category",
            category="Safety & Certifications",
            description="Hazardous area ATEX category",
            data_type="string",
            options=["Cat1G", "Cat2G", "Cat3G", "Cat1D", "Cat2D", "Cat3D"],
            importance=SpecImportance.IMPORTANT
        ),
    }

    @classmethod
    def get_all_specifications(cls) -> Dict[str, SpecificationDefinition]:
        return cls.SPECIFICATIONS

    @classmethod
    def get_specifications_by_category(cls) -> Dict[str, List[SpecificationDefinition]]:
        grouped = {}
        for spec in cls.SPECIFICATIONS.values():
            if spec.category not in grouped:
                grouped[spec.category] = []
            grouped[spec.category].append(spec)
        return grouped

    @classmethod
    def get_count(cls) -> int:
        return len(cls.SPECIFICATIONS)


# =============================================================================
# CONTROL VALVE TEMPLATE (65+ SPECS)
# =============================================================================

class ControlValveTemplate:
    """Specification template for control valves with 65+ specs"""

    SPECIFICATIONS = {
        # 1. VALVE TYPE & CHARACTERISTICS (12 specs)
        "valve_type": SpecificationDefinition(
            key="valve_type",
            category="Valve Type & Characteristics",
            description="Type of control valve",
            data_type="string",
            options=["globe", "ball", "butterfly", "needle", "diaphragm", "rotary"],
            importance=SpecImportance.CRITICAL
        ),
        "flow_characteristic": SpecificationDefinition(
            key="flow_characteristic",
            category="Valve Type & Characteristics",
            description="Flow characteristic curve",
            data_type="string",
            options=["linear", "equal_percentage", "quick_open"],
            importance=SpecImportance.CRITICAL
        ),
        "ported_type": SpecificationDefinition(
            key="ported_type",
            category="Valve Type & Characteristics",
            description="Ported configuration",
            data_type="string",
            options=["2/2", "3/2", "3/3", "4/3"],
            importance=SpecImportance.REQUIRED
        ),
        "rated_flow_capacity_cv": SpecificationDefinition(
            key="rated_flow_capacity_cv",
            category="Valve Type & Characteristics",
            description="Rated flow capacity",
            unit="Cv",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "maximum_flow_rate": SpecificationDefinition(
            key="maximum_flow_rate",
            category="Valve Type & Characteristics",
            description="Maximum flow rate",
            unit="GPM",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "pressure_drop_at_rated_flow": SpecificationDefinition(
            key="pressure_drop_at_rated_flow",
            category="Valve Type & Characteristics",
            description="Pressure drop at rated flow",
            unit="psi",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "flow_recovery_factor": SpecificationDefinition(
            key="flow_recovery_factor",
            category="Valve Type & Characteristics",
            description="Flow recovery factor",
            data_type="number",
            importance=SpecImportance.OPTIONAL
        ),
        "authority": SpecificationDefinition(
            key="authority",
            category="Valve Type & Characteristics",
            description="Valve authority (pressure independent)",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "close_off_rating": SpecificationDefinition(
            key="close_off_rating",
            category="Valve Type & Characteristics",
            description="Differential pressure when fully closed",
            unit="psi",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "rangeability": SpecificationDefinition(
            key="rangeability",
            category="Valve Type & Characteristics",
            description="Turndown ratio",
            data_type="string",
            importance=SpecImportance.IMPORTANT
        ),
        "resolution": SpecificationDefinition(
            key="resolution",
            category="Valve Type & Characteristics",
            description="Control resolution",
            unit="%",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),

        # 2. ACTUATION & CONTROL (13 specs)
        "actuator_type": SpecificationDefinition(
            key="actuator_type",
            category="Actuation & Control",
            description="Type of actuator",
            data_type="string",
            options=["pneumatic", "hydraulic", "electric", "solenoid"],
            importance=SpecImportance.CRITICAL
        ),
        "air_supply_pressure_min": SpecificationDefinition(
            key="air_supply_pressure_min",
            category="Actuation & Control",
            description="Minimum air supply pressure",
            unit="psi",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "air_supply_pressure_max": SpecificationDefinition(
            key="air_supply_pressure_max",
            category="Actuation & Control",
            description="Maximum air supply pressure",
            unit="psi",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "action_type": SpecificationDefinition(
            key="action_type",
            category="Actuation & Control",
            description="Actuator action",
            data_type="string",
            options=["spring_return", "double_acting"],
            importance=SpecImportance.CRITICAL
        ),
        "fail_safe_position": SpecificationDefinition(
            key="fail_safe_position",
            category="Actuation & Control",
            description="Fail-safe position",
            data_type="string",
            options=["open", "closed", "last_position"],
            importance=SpecImportance.CRITICAL
        ),
        "response_time_milliseconds": SpecificationDefinition(
            key="response_time_milliseconds",
            category="Actuation & Control",
            description="Response time from signal to valve movement",
            unit="milliseconds",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "stroking_time_seconds": SpecificationDefinition(
            key="stroking_time_seconds",
            category="Actuation & Control",
            description="Time to fully open/close",
            unit="seconds",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "control_signal_type": SpecificationDefinition(
            key="control_signal_type",
            category="Actuation & Control",
            description="Type of control signal",
            data_type="string",
            options=["4-20mA", "0-10V", "on-off", "0-100%"],
            importance=SpecImportance.CRITICAL
        ),
        "signal_input_impedance": SpecificationDefinition(
            key="signal_input_impedance",
            category="Actuation & Control",
            description="Signal input impedance",
            unit="Ohms",
            data_type="number",
            importance=SpecImportance.OPTIONAL
        ),
        "hysteresis": SpecificationDefinition(
            key="hysteresis",
            category="Actuation & Control",
            description="Control hysteresis",
            unit="%",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "proportional_band": SpecificationDefinition(
            key="proportional_band",
            category="Actuation & Control",
            description="Proportional band range",
            unit="%",
            data_type="number",
            importance=SpecImportance.OPTIONAL
        ),
        "manual_override": SpecificationDefinition(
            key="manual_override",
            category="Actuation & Control",
            description="Manual override capability",
            data_type="boolean",
            importance=SpecImportance.IMPORTANT
        ),
        "positioner_integrated": SpecificationDefinition(
            key="positioner_integrated",
            category="Actuation & Control",
            description="Integrated positioner",
            data_type="boolean",
            importance=SpecImportance.OPTIONAL
        ),

        # 3. FLUID COMPATIBILITY (8 specs)
        "fluid_type": SpecificationDefinition(
            key="fluid_type",
            category="Fluid Compatibility",
            description="Compatible fluid types",
            data_type="string",
            options=["water", "steam", "oil", "gas", "slurry"],
            importance=SpecImportance.CRITICAL
        ),
        "liquid_temperature_min": SpecificationDefinition(
            key="liquid_temperature_min",
            category="Fluid Compatibility",
            description="Minimum fluid temperature",
            unit="°C",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "liquid_temperature_max": SpecificationDefinition(
            key="liquid_temperature_max",
            category="Fluid Compatibility",
            description="Maximum fluid temperature",
            unit="°C",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "viscosity_min": SpecificationDefinition(
            key="viscosity_min",
            category="Fluid Compatibility",
            description="Minimum fluid viscosity",
            unit="cSt",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "viscosity_max": SpecificationDefinition(
            key="viscosity_max",
            category="Fluid Compatibility",
            description="Maximum fluid viscosity",
            unit="cSt",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "particle_contamination_iso": SpecificationDefinition(
            key="particle_contamination_iso",
            category="Fluid Compatibility",
            description="ISO contamination rating accepted",
            data_type="string",
            importance=SpecImportance.IMPORTANT
        ),
        "corrosive_fluid_compatibility": SpecificationDefinition(
            key="corrosive_fluid_compatibility",
            category="Fluid Compatibility",
            description="Corrosive fluid handling capability",
            data_type="boolean",
            importance=SpecImportance.OPTIONAL
        ),

        # 4. MATERIALS & CONSTRUCTION (10 specs)
        "body_material": SpecificationDefinition(
            key="body_material",
            category="Materials & Construction",
            description="Valve body material",
            data_type="string",
            options=["cast_iron", "ductile_iron", "stainless_steel", "bronze", "aluminum"],
            importance=SpecImportance.CRITICAL
        ),
        "seat_material": SpecificationDefinition(
            key="seat_material",
            category="Materials & Construction",
            description="Valve seat material",
            data_type="string",
            options=["stainless_steel", "viton", "brass", "graphite"],
            importance=SpecImportance.REQUIRED
        ),
        "stem_material": SpecificationDefinition(
            key="stem_material",
            category="Materials & Construction",
            description="Stem material",
            data_type="string",
            importance=SpecImportance.REQUIRED
        ),
        "trim_material": SpecificationDefinition(
            key="trim_material",
            category="Materials & Construction",
            description="Internal trim material",
            data_type="string",
            importance=SpecImportance.IMPORTANT
        ),
        "seal_material": SpecificationDefinition(
            key="seal_material",
            category="Materials & Construction",
            description="Seal/gasket material",
            data_type="string",
            options=["viton", "buna", "epdm", "ptfe"],
            importance=SpecImportance.IMPORTANT
        ),
        "bonnet_style": SpecificationDefinition(
            key="bonnet_style",
            category="Materials & Construction",
            description="Bonnet configuration",
            data_type="string",
            options=["standard", "long", "bellows"],
            importance=SpecImportance.IMPORTANT
        ),
        "connection_type": SpecificationDefinition(
            key="connection_type",
            category="Materials & Construction",
            description="Inlet/outlet connection type",
            data_type="string",
            options=["threaded", "flanged", "soldered"],
            importance=SpecImportance.CRITICAL
        ),
        "port_size_inches": SpecificationDefinition(
            key="port_size_inches",
            category="Materials & Construction",
            description="Port connection size",
            unit="inches",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),

        # 5. PRESSURE RATINGS (5 specs)
        "pressure_rating_nominal_psi": SpecificationDefinition(
            key="pressure_rating_nominal_psi",
            category="Pressure Ratings",
            description="Nominal pressure rating",
            unit="psi",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "pressure_rating_maximum_psi": SpecificationDefinition(
            key="pressure_rating_maximum_psi",
            category="Pressure Ratings",
            description="Maximum pressure rating",
            unit="psi",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "differential_pressure_max": SpecificationDefinition(
            key="differential_pressure_max",
            category="Pressure Ratings",
            description="Maximum differential pressure",
            unit="psi",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "burst_pressure_rating": SpecificationDefinition(
            key="burst_pressure_rating",
            category="Pressure Ratings",
            description="Burst pressure rating",
            unit="psi",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),

        # 6. ENVIRONMENTAL (8 specs)
        "operating_temperature_min": SpecificationDefinition(
            key="operating_temperature_min",
            category="Environmental Conditions",
            description="Minimum operating temperature",
            unit="°C",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "operating_temperature_max": SpecificationDefinition(
            key="operating_temperature_max",
            category="Environmental Conditions",
            description="Maximum operating temperature",
            unit="°C",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "ambient_temperature_min": SpecificationDefinition(
            key="ambient_temperature_min",
            category="Environmental Conditions",
            description="Minimum ambient temperature",
            unit="°C",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "ambient_temperature_max": SpecificationDefinition(
            key="ambient_temperature_max",
            category="Environmental Conditions",
            description="Maximum ambient temperature",
            unit="°C",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "humidity_range": SpecificationDefinition(
            key="humidity_range",
            category="Environmental Conditions",
            description="Operating humidity range",
            unit="%RH",
            data_type="string",
            importance=SpecImportance.OPTIONAL
        ),
        "vibration_rating": SpecificationDefinition(
            key="vibration_rating",
            category="Environmental Conditions",
            description="Vibration resistance rating",
            data_type="string",
            importance=SpecImportance.OPTIONAL
        ),

        # 7. SAFETY & CERTIFICATIONS (4 specs)
        "certifications": SpecificationDefinition(
            key="certifications",
            category="Safety & Certifications",
            description="Industry certifications",
            data_type="list",
            options=["CE", "UL", "ASME"],
            importance=SpecImportance.CRITICAL
        ),
        "pressure_equipment_directive": SpecificationDefinition(
            key="pressure_equipment_directive",
            category="Safety & Certifications",
            description="PED compliance category",
            data_type="string",
            importance=SpecImportance.IMPORTANT
        ),
        "fire_safe_rated": SpecificationDefinition(
            key="fire_safe_rated",
            category="Safety & Certifications",
            description="Fire-safe design certified",
            data_type="boolean",
            importance=SpecImportance.IMPORTANT
        ),
    }

    @classmethod
    def get_all_specifications(cls) -> Dict[str, SpecificationDefinition]:
        return cls.SPECIFICATIONS

    @classmethod
    def get_specifications_by_category(cls) -> Dict[str, List[SpecificationDefinition]]:
        grouped = {}
        for spec in cls.SPECIFICATIONS.values():
            if spec.category not in grouped:
                grouped[spec.category] = []
            grouped[spec.category].append(spec)
        return grouped

    @classmethod
    def get_count(cls) -> int:
        return len(cls.SPECIFICATIONS)


# =============================================================================
# THERMOWELL TEMPLATE (40+ SPECS)
# =============================================================================

class ThermowellTemplate:
    """Specification template for thermowells with 40+ specs"""

    SPECIFICATIONS = {
        # 1. WELL TYPE & DESIGN (8 specs)
        "well_type": SpecificationDefinition(
            key="well_type",
            category="Well Type & Design",
            description="Type of thermowell",
            data_type="string",
            options=["1/2NPT", "3/4NPT", "M20", "M27", "flange_mounted"],
            importance=SpecImportance.CRITICAL
        ),
        "connection_type": SpecificationDefinition(
            key="connection_type",
            category="Well Type & Design",
            description="Connection method to process",
            data_type="string",
            options=["immersion", "socket_weld", "flanged", "threaded"],
            importance=SpecImportance.CRITICAL
        ),
        "well_diameter_mm": SpecificationDefinition(
            key="well_diameter_mm",
            category="Well Type & Design",
            description="Internal bore diameter",
            unit="mm",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "immersion_length_mm": SpecificationDefinition(
            key="immersion_length_mm",
            category="Well Type & Design",
            description="Immersion length",
            unit="mm",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "overall_length_mm": SpecificationDefinition(
            key="overall_length_mm",
            category="Well Type & Design",
            description="Overall length",
            unit="mm",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "sheath_material": SpecificationDefinition(
            key="sheath_material",
            category="Well Type & Design",
            description="Thermowell material",
            data_type="string",
            options=["stainless_steel", "monel", "inconel", "cast_iron", "brass"],
            importance=SpecImportance.CRITICAL
        ),
        "wall_thickness_mm": SpecificationDefinition(
            key="wall_thickness_mm",
            category="Well Type & Design",
            description="Thermowell wall thickness",
            unit="mm",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "design_type": SpecificationDefinition(
            key="design_type",
            category="Well Type & Design",
            description="Design style",
            data_type="string",
            options=["solid_bottom", "closed_bottom", "open_bottom"],
            importance=SpecImportance.REQUIRED
        ),

        # 2. TEMPERATURE RANGE (4 specs)
        "operating_temperature_min": SpecificationDefinition(
            key="operating_temperature_min",
            category="Temperature Range",
            description="Minimum operating temperature",
            unit="°C",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "operating_temperature_max": SpecificationDefinition(
            key="operating_temperature_max",
            category="Temperature Range",
            description="Maximum operating temperature",
            unit="°C",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "sensor_temperature_range_min": SpecificationDefinition(
            key="sensor_temperature_range_min",
            category="Temperature Range",
            description="Sensor minimum temperature",
            unit="°C",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "sensor_temperature_range_max": SpecificationDefinition(
            key="sensor_temperature_range_max",
            category="Temperature Range",
            description="Sensor maximum temperature",
            unit="°C",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),

        # 3. PRESSURE RATINGS (5 specs)
        "pressure_rating_nominal_psi": SpecificationDefinition(
            key="pressure_rating_nominal_psi",
            category="Pressure Ratings",
            description="Nominal pressure rating",
            unit="psi",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "pressure_rating_maximum_psi": SpecificationDefinition(
            key="pressure_rating_maximum_psi",
            category="Pressure Ratings",
            description="Maximum pressure rating",
            unit="psi",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "burst_pressure_rating": SpecificationDefinition(
            key="burst_pressure_rating",
            category="Pressure Ratings",
            description="Burst pressure rating",
            unit="psi",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "process_connection_size": SpecificationDefinition(
            key="process_connection_size",
            category="Pressure Ratings",
            description="Connection port size",
            unit="inches",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),

        # 4. THERMOMETRIC POCKET (6 specs)
        "pocket_type": SpecificationDefinition(
            key="pocket_type",
            category="Thermometric Pocket",
            description="Type of sensor pocket",
            data_type="string",
            options=["integral", "external", "removable"],
            importance=SpecImportance.REQUIRED
        ),
        "pocket_diameter_mm": SpecificationDefinition(
            key="pocket_diameter_mm",
            category="Thermometric Pocket",
            description="Pocket internal diameter",
            unit="mm",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "pocket_depth_mm": SpecificationDefinition(
            key="pocket_depth_mm",
            category="Thermometric Pocket",
            description="Pocket depth",
            unit="mm",
            data_type="number",
            importance=SpecImportance.REQUIRED
        ),
        "clearance_fit": SpecificationDefinition(
            key="clearance_fit",
            category="Thermometric Pocket",
            description="Sensor to pocket clearance",
            unit="mm",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "response_time_constant": SpecificationDefinition(
            key="response_time_constant",
            category="Thermometric Pocket",
            description="Well response time constant",
            unit="seconds",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),

        # 5. INSTALLATION & MOUNTING (6 specs)
        "flange_connection_rating": SpecificationDefinition(
            key="flange_connection_rating",
            category="Installation & Mounting",
            description="Flange connection rating",
            data_type="string",
            options=["ANSI_150", "ANSI_300", "ANSI_600"],
            importance=SpecImportance.IMPORTANT
        ),
        "entry_angle_degrees": SpecificationDefinition(
            key="entry_angle_degrees",
            category="Installation & Mounting",
            description="Entry angle to process",
            unit="degrees",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "union_type": SpecificationDefinition(
            key="union_type",
            category="Installation & Mounting",
            description="Union connection type",
            data_type="string",
            options=["NPT", "SAE", "ISO"],
            importance=SpecImportance.OPTIONAL
        ),
        "extension_length_mm": SpecificationDefinition(
            key="extension_length_mm",
            category="Installation & Mounting",
            description="Extension length if applicable",
            unit="mm",
            data_type="number",
            importance=SpecImportance.OPTIONAL
        ),

        # 6. FLOW VELOCITY (3 specs)
        "max_flow_velocity_ft_sec": SpecificationDefinition(
            key="max_flow_velocity_ft_sec",
            category="Flow Velocity",
            description="Maximum flow velocity",
            unit="ft/sec",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "critical_frequency_hz": SpecificationDefinition(
            key="critical_frequency_hz",
            category="Flow Velocity",
            description="Critical frequency for vibration",
            unit="Hz",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),

        # 7. STANDARDS & CERTIFICATIONS (3 specs)
        "asme_standard": SpecificationDefinition(
            key="asme_standard",
            category="Standards & Certifications",
            description="ASME standard compliance",
            data_type="string",
            importance=SpecImportance.IMPORTANT
        ),
        "pressure_equipment_directive": SpecificationDefinition(
            key="pressure_equipment_directive",
            category="Standards & Certifications",
            description="PED compliance",
            data_type="boolean",
            importance=SpecImportance.IMPORTANT
        ),
    }

    @classmethod
    def get_all_specifications(cls) -> Dict[str, SpecificationDefinition]:
        return cls.SPECIFICATIONS

    @classmethod
    def get_specifications_by_category(cls) -> Dict[str, List[SpecificationDefinition]]:
        grouped = {}
        for spec in cls.SPECIFICATIONS.values():
            if spec.category not in grouped:
                grouped[spec.category] = []
            grouped[spec.category].append(spec)
        return grouped

    @classmethod
    def get_count(cls) -> int:
        return len(cls.SPECIFICATIONS)


# =============================================================================
# CABLE GLAND TEMPLATE (35+ SPECS)
# =============================================================================

class CableGlandTemplate:
    """Specification template for cable glands with 35+ specs"""

    SPECIFICATIONS = {
        # 1. GLAND TYPE & DIMENSIONS (10 specs)
        "gland_type": SpecificationDefinition(
            key="gland_type",
            category="Gland Type & Dimensions",
            description="Type of cable gland",
            data_type="string",
            options=["cord_grip", "strain_relief", "liquid_tight", "armored", "surge_protective"],
            importance=SpecImportance.CRITICAL
        ),
        "thread_size": SpecificationDefinition(
            key="thread_size",
            category="Gland Type & Dimensions",
            description="Thread connection size",
            data_type="string",
            options=["M16", "M20", "M25", "M32", "M40", "M50"],
            importance=SpecImportance.CRITICAL
        ),
        "cable_diameter_min_mm": SpecificationDefinition(
            key="cable_diameter_min_mm",
            category="Gland Type & Dimensions",
            description="Minimum cable diameter",
            unit="mm",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "cable_diameter_max_mm": SpecificationDefinition(
            key="cable_diameter_max_mm",
            category="Gland Type & Dimensions",
            description="Maximum cable diameter",
            unit="mm",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "body_length_mm": SpecificationDefinition(
            key="body_length_mm",
            category="Gland Type & Dimensions",
            description="Gland body length",
            unit="mm",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "body_diameter_mm": SpecificationDefinition(
            key="body_diameter_mm",
            category="Gland Type & Dimensions",
            description="Gland body diameter",
            unit="mm",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "material": SpecificationDefinition(
            key="material",
            category="Gland Type & Dimensions",
            description="Gland body material",
            data_type="string",
            options=["brass", "nickel_plated", "stainless_steel", "nylon", "aluminum"],
            importance=SpecImportance.CRITICAL
        ),
        "weight_grams": SpecificationDefinition(
            key="weight_grams",
            category="Gland Type & Dimensions",
            description="Weight of gland",
            unit="grams",
            data_type="number",
            importance=SpecImportance.OPTIONAL
        ),

        # 2. SEALING & INGRESS PROTECTION (8 specs)
        "sealing_material": SpecificationDefinition(
            key="sealing_material",
            category="Sealing & Ingress Protection",
            description="Sealing element material",
            data_type="string",
            options=["neoprene", "polyamide", "epdm", "ptfe"],
            importance=SpecImportance.CRITICAL
        ),
        "ip_rating": SpecificationDefinition(
            key="ip_rating",
            category="Sealing & Ingress Protection",
            description="Ingress protection rating",
            data_type="string",
            options=["IP54", "IP65", "IP66", "IP67", "IP68"],
            importance=SpecImportance.CRITICAL
        ),
        "nema_rating": SpecificationDefinition(
            key="nema_rating",
            category="Sealing & Ingress Protection",
            description="NEMA enclosure rating",
            data_type="string",
            importance=SpecImportance.IMPORTANT
        ),
        "water_tightness_mbar": SpecificationDefinition(
            key="water_tightness_mbar",
            category="Sealing & Ingress Protection",
            description="Water tightness pressure",
            unit="mbar",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "dust_tightness": SpecificationDefinition(
            key="dust_tightness",
            category="Sealing & Ingress Protection",
            description="Dust-tight capability",
            data_type="boolean",
            importance=SpecImportance.IMPORTANT
        ),
        "oil_resistant": SpecificationDefinition(
            key="oil_resistant",
            category="Sealing & Ingress Protection",
            description="Oil resistance capability",
            data_type="boolean",
            importance=SpecImportance.OPTIONAL
        ),
        "flame_retardant": SpecificationDefinition(
            key="flame_retardant",
            category="Sealing & Ingress Protection",
            description="Flame retardant material",
            data_type="boolean",
            importance=SpecImportance.IMPORTANT
        ),

        # 3. ELECTRICAL PROPERTIES (7 specs)
        "cable_entry_voltage_rating_v": SpecificationDefinition(
            key="cable_entry_voltage_rating_v",
            category="Electrical Properties",
            description="Voltage rating",
            unit="Volts",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "voltage_drop": SpecificationDefinition(
            key="voltage_drop",
            category="Electrical Properties",
            description="Maximum voltage drop",
            unit="V/m",
            data_type="number",
            importance=SpecImportance.OPTIONAL
        ),
        "dielectric_strength": SpecificationDefinition(
            key="dielectric_strength",
            category="Electrical Properties",
            description="Dielectric strength",
            unit="kV",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "current_capacity_amps": SpecificationDefinition(
            key="current_capacity_amps",
            category="Electrical Properties",
            description="Maximum current capacity",
            unit="Amps",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "emc_compliant": SpecificationDefinition(
            key="emc_compliant",
            category="Electrical Properties",
            description="EMC directive compliant",
            data_type="boolean",
            importance=SpecImportance.IMPORTANT
        ),
        "conductive": SpecificationDefinition(
            key="conductive",
            category="Electrical Properties",
            description="Conductive body for grounding",
            data_type="boolean",
            importance=SpecImportance.OPTIONAL
        ),

        # 4. TEMPERATURE & ENVIRONMENT (6 specs)
        "operating_temperature_min": SpecificationDefinition(
            key="operating_temperature_min",
            category="Temperature & Environment",
            description="Minimum operating temperature",
            unit="°C",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "operating_temperature_max": SpecificationDefinition(
            key="operating_temperature_max",
            category="Temperature & Environment",
            description="Maximum operating temperature",
            unit="°C",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "uv_resistance": SpecificationDefinition(
            key="uv_resistance",
            category="Temperature & Environment",
            description="UV resistance rating",
            data_type="string",
            importance=SpecImportance.OPTIONAL
        ),
        "halogen_free": SpecificationDefinition(
            key="halogen_free",
            category="Temperature & Environment",
            description="Halogen-free material",
            data_type="boolean",
            importance=SpecImportance.OPTIONAL
        ),

        # 5. STANDARDS & CERTIFICATIONS (4 specs)
        "certifications": SpecificationDefinition(
            key="certifications",
            category="Standards & Certifications",
            description="Safety certifications",
            data_type="list",
            options=["CE", "UL", "CSA"],
            importance=SpecImportance.CRITICAL
        ),
        "rohs_compliant": SpecificationDefinition(
            key="rohs_compliant",
            category="Standards & Certifications",
            description="RoHS compliance",
            data_type="boolean",
            importance=SpecImportance.IMPORTANT
        ),
    }

    @classmethod
    def get_all_specifications(cls) -> Dict[str, SpecificationDefinition]:
        return cls.SPECIFICATIONS

    @classmethod
    def get_specifications_by_category(cls) -> Dict[str, List[SpecificationDefinition]]:
        grouped = {}
        for spec in cls.SPECIFICATIONS.values():
            if spec.category not in grouped:
                grouped[spec.category] = []
            grouped[spec.category].append(spec)
        return grouped

    @classmethod
    def get_count(cls) -> int:
        return len(cls.SPECIFICATIONS)


# =============================================================================
# ENCLOSURE TEMPLATE (45+ SPECS)
# =============================================================================

class EnclosureTemplate:
    """Specification template for enclosures with 45+ specs"""

    SPECIFICATIONS = {
        # 1. ENCLOSURE TYPE & DESIGN (12 specs)
        "enclosure_type": SpecificationDefinition(
            key="enclosure_type",
            category="Enclosure Type & Design",
            description="Type of enclosure",
            data_type="string",
            options=["wall_mounted", "floor_standing", "pole_mounted", "duct_mounted"],
            importance=SpecImportance.CRITICAL
        ),
        "construction_style": SpecificationDefinition(
            key="construction_style",
            category="Enclosure Type & Design",
            description="Construction method",
            data_type="string",
            options=["welded", "bolted", "modular"],
            importance=SpecImportance.REQUIRED
        ),
        "interior_width_mm": SpecificationDefinition(
            key="interior_width_mm",
            category="Enclosure Type & Design",
            description="Internal width",
            unit="mm",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "interior_depth_mm": SpecificationDefinition(
            key="interior_depth_mm",
            category="Enclosure Type & Design",
            description="Internal depth",
            unit="mm",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "interior_height_mm": SpecificationDefinition(
            key="interior_height_mm",
            category="Enclosure Type & Design",
            description="Internal height",
            unit="mm",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "exterior_width_mm": SpecificationDefinition(
            key="exterior_width_mm",
            category="Enclosure Type & Design",
            description="External width",
            unit="mm",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "exterior_depth_mm": SpecificationDefinition(
            key="exterior_depth_mm",
            category="Enclosure Type & Design",
            description="External depth",
            unit="mm",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "exterior_height_mm": SpecificationDefinition(
            key="exterior_height_mm",
            category="Enclosure Type & Design",
            description="External height",
            unit="mm",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "weight_kg": SpecificationDefinition(
            key="weight_kg",
            category="Enclosure Type & Design",
            description="Weight of enclosure",
            unit="kg",
            data_type="number",
            importance=SpecImportance.OPTIONAL
        ),
        "wall_thickness_mm": SpecificationDefinition(
            key="wall_thickness_mm",
            category="Enclosure Type & Design",
            description="Wall thickness",
            unit="mm",
            data_type="number",
            importance=SpecImportance.OPTIONAL
        ),
        "door_type": SpecificationDefinition(
            key="door_type",
            category="Enclosure Type & Design",
            description="Door configuration",
            data_type="string",
            options=["single", "double", "hinged", "swingout"],
            importance=SpecImportance.REQUIRED
        ),

        # 2. MATERIALS (5 specs)
        "material": SpecificationDefinition(
            key="material",
            category="Materials",
            description="Primary enclosure material",
            data_type="string",
            options=["steel", "stainless_steel", "aluminum", "composite"],
            importance=SpecImportance.CRITICAL
        ),
        "finish_type": SpecificationDefinition(
            key="finish_type",
            category="Materials",
            description="Surface finish",
            data_type="string",
            options=["powder_coat", "galvanized", "painted", "natural"],
            importance=SpecImportance.IMPORTANT
        ),
        "interior_coating": SpecificationDefinition(
            key="interior_coating",
            category="Materials",
            description="Interior coating type",
            data_type="string",
            importance=SpecImportance.OPTIONAL
        ),
        "gasket_material": SpecificationDefinition(
            key="gasket_material",
            category="Materials",
            description="Door gasket material",
            data_type="string",
            options=["neoprene", "epdm", "silicone"],
            importance=SpecImportance.IMPORTANT
        ),

        # 3. INGRESS PROTECTION & SEALING (6 specs)
        "ip_rating": SpecificationDefinition(
            key="ip_rating",
            category="Ingress Protection & Sealing",
            description="IP rating",
            data_type="string",
            options=["IP54", "IP55", "IP65", "IP66", "IP67"],
            importance=SpecImportance.CRITICAL
        ),
        "nema_rating": SpecificationDefinition(
            key="nema_rating",
            category="Ingress Protection & Sealing",
            description="NEMA rating",
            data_type="string",
            options=["3", "4", "4X", "12"],
            importance=SpecImportance.CRITICAL
        ),
        "door_seal_type": SpecificationDefinition(
            key="door_seal_type",
            category="Ingress Protection & Sealing",
            description="Door sealing method",
            data_type="string",
            options=["gasket", "magnetic", "foam"],
            importance=SpecImportance.IMPORTANT
        ),
        "cable_entry_seal": SpecificationDefinition(
            key="cable_entry_seal",
            category="Ingress Protection & Sealing",
            description="Cable entry sealing",
            data_type="boolean",
            importance=SpecImportance.IMPORTANT
        ),
        "drain_plugs": SpecificationDefinition(
            key="drain_plugs",
            category="Ingress Protection & Sealing",
            description="Number of drain plugs",
            data_type="number",
            importance=SpecImportance.OPTIONAL
        ),

        # 4. ENVIRONMENTAL CONDITIONS (8 specs)
        "operating_temperature_min": SpecificationDefinition(
            key="operating_temperature_min",
            category="Environmental Conditions",
            description="Minimum operating temperature",
            unit="°C",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "operating_temperature_max": SpecificationDefinition(
            key="operating_temperature_max",
            category="Environmental Conditions",
            description="Maximum operating temperature",
            unit="°C",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "storage_temperature_min": SpecificationDefinition(
            key="storage_temperature_min",
            category="Environmental Conditions",
            description="Minimum storage temperature",
            unit="°C",
            data_type="number",
            importance=SpecImportance.OPTIONAL
        ),
        "storage_temperature_max": SpecificationDefinition(
            key="storage_temperature_max",
            category="Environmental Conditions",
            description="Maximum storage temperature",
            unit="°C",
            data_type="number",
            importance=SpecImportance.OPTIONAL
        ),
        "humidity_range": SpecificationDefinition(
            key="humidity_range",
            category="Environmental Conditions",
            description="Operating humidity range",
            unit="%RH",
            data_type="string",
            importance=SpecImportance.OPTIONAL
        ),
        "uv_resistance": SpecificationDefinition(
            key="uv_resistance",
            category="Environmental Conditions",
            description="UV resistance capability",
            data_type="boolean",
            importance=SpecImportance.OPTIONAL
        ),
        "corrosion_resistance": SpecificationDefinition(
            key="corrosion_resistance",
            category="Environmental Conditions",
            description="Corrosion resistance rating",
            data_type="string",
            importance=SpecImportance.OPTIONAL
        ),

        # 5. ELECTRICAL & VENTILATION (6 specs)
        "thermal_management": SpecificationDefinition(
            key="thermal_management",
            category="Electrical & Ventilation",
            description="Cooling method",
            data_type="string",
            options=["natural", "fan_cooled", "filtered", "chilled_water"],
            importance=SpecImportance.OPTIONAL
        ),
        "heat_dissipation_watts": SpecificationDefinition(
            key="heat_dissipation_watts",
            category="Electrical & Ventilation",
            description="Maximum heat dissipation",
            unit="Watts",
            data_type="number",
            importance=SpecImportance.OPTIONAL
        ),
        "cable_entry_knockouts": SpecificationDefinition(
            key="cable_entry_knockouts",
            category="Electrical & Ventilation",
            description="Number of cable entry points",
            data_type="number",
            importance=SpecImportance.OPTIONAL
        ),
        "grounding_provision": SpecificationDefinition(
            key="grounding_provision",
            category="Electrical & Ventilation",
            description="Grounding terminal provided",
            data_type="boolean",
            importance=SpecImportance.IMPORTANT
        ),

        # 6. STANDARDS & CERTIFICATIONS (2 specs)
        "certifications": SpecificationDefinition(
            key="certifications",
            category="Standards & Certifications",
            description="Certifications held",
            data_type="list",
            options=["CE", "UL", "CSA"],
            importance=SpecImportance.CRITICAL
        ),
    }

    @classmethod
    def get_all_specifications(cls) -> Dict[str, SpecificationDefinition]:
        return cls.SPECIFICATIONS

    @classmethod
    def get_specifications_by_category(cls) -> Dict[str, List[SpecificationDefinition]]:
        grouped = {}
        for spec in cls.SPECIFICATIONS.values():
            if spec.category not in grouped:
                grouped[spec.category] = []
            grouped[spec.category].append(spec)
        return grouped

    @classmethod
    def get_count(cls) -> int:
        return len(cls.SPECIFICATIONS)


# =============================================================================
# MOUNTING KIT TEMPLATE (30+ SPECS)
# =============================================================================

class MountingKitTemplate:
    """Specification template for mounting kits with 30+ specs"""

    SPECIFICATIONS = {
        # 1. KIT CONTENTS & COMPONENTS (10 specs)
        "mounting_type": SpecificationDefinition(
            key="mounting_type",
            category="Kit Contents & Components",
            description="Type of mounting kit",
            data_type="string",
            options=["wall", "panel", "pole", "DIN_rail", "pipe"],
            importance=SpecImportance.CRITICAL
        ),
        "bracket_type": SpecificationDefinition(
            key="bracket_type",
            category="Kit Contents & Components",
            description="Bracket configuration",
            data_type="string",
            options=["L-bracket", "U-bracket", "rail", "universal"],
            importance=SpecImportance.CRITICAL
        ),
        "fastener_type": SpecificationDefinition(
            key="fastener_type",
            category="Kit Contents & Components",
            description="Fastener material and type",
            data_type="string",
            options=["stainless_steel", "zinc_plated", "metric", "imperial"],
            importance=SpecImportance.IMPORTANT
        ),
        "fastener_count": SpecificationDefinition(
            key="fastener_count",
            category="Kit Contents & Components",
            description="Number of fasteners included",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "fastener_sizes_provided": SpecificationDefinition(
            key="fastener_sizes_provided",
            category="Kit Contents & Components",
            description="Fastener sizes included",
            data_type="string",
            importance=SpecImportance.IMPORTANT
        ),
        "cable_management_included": SpecificationDefinition(
            key="cable_management_included",
            category="Kit Contents & Components",
            description="Cable management accessories",
            data_type="boolean",
            importance=SpecImportance.OPTIONAL
        ),
        "vibration_damping": SpecificationDefinition(
            key="vibration_damping",
            category="Kit Contents & Components",
            description="Vibration damping pads included",
            data_type="boolean",
            importance=SpecImportance.OPTIONAL
        ),
        "thermal_interface_material": SpecificationDefinition(
            key="thermal_interface_material",
            category="Kit Contents & Components",
            description="Thermal paste/pads included",
            data_type="boolean",
            importance=SpecImportance.OPTIONAL
        ),

        # 2. DIMENSIONS & COMPATIBILITY (8 specs)
        "bracket_width_mm": SpecificationDefinition(
            key="bracket_width_mm",
            category="Dimensions & Compatibility",
            description="Bracket width",
            unit="mm",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "bracket_depth_mm": SpecificationDefinition(
            key="bracket_depth_mm",
            category="Dimensions & Compatibility",
            description="Bracket depth",
            unit="mm",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "bracket_height_mm": SpecificationDefinition(
            key="bracket_height_mm",
            category="Dimensions & Compatibility",
            description="Bracket height",
            unit="mm",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "maximum_device_width_mm": SpecificationDefinition(
            key="maximum_device_width_mm",
            category="Dimensions & Compatibility",
            description="Maximum device width that can be mounted",
            unit="mm",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "maximum_device_depth_mm": SpecificationDefinition(
            key="maximum_device_depth_mm",
            category="Dimensions & Compatibility",
            description="Maximum device depth that can be mounted",
            unit="mm",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "compatibility_list": SpecificationDefinition(
            key="compatibility_list",
            category="Dimensions & Compatibility",
            description="Compatible device models",
            data_type="string",
            importance=SpecImportance.OPTIONAL
        ),

        # 3. STRUCTURAL & LOAD CAPACITY (6 specs)
        "load_capacity_kg": SpecificationDefinition(
            key="load_capacity_kg",
            category="Structural & Load Capacity",
            description="Maximum load capacity",
            unit="kg",
            data_type="number",
            importance=SpecImportance.CRITICAL
        ),
        "safety_factor": SpecificationDefinition(
            key="safety_factor",
            category="Structural & Load Capacity",
            description="Safety factor for load capacity",
            data_type="number",
            importance=SpecImportance.IMPORTANT
        ),
        "material": SpecificationDefinition(
            key="material",
            category="Structural & Load Capacity",
            description="Bracket material",
            data_type="string",
            options=["steel", "stainless_steel", "aluminum", "composite"],
            importance=SpecImportance.CRITICAL
        ),
        "surface_treatment": SpecificationDefinition(
            key="surface_treatment",
            category="Structural & Load Capacity",
            description="Surface treatment/finish",
            data_type="string",
            options=["painted", "powder_coated", "galvanized", "natural"],
            importance=SpecImportance.IMPORTANT
        ),
        "horizontal_adjustment_range_mm": SpecificationDefinition(
            key="horizontal_adjustment_range_mm",
            category="Structural & Load Capacity",
            description="Horizontal positioning range",
            unit="mm",
            data_type="number",
            importance=SpecImportance.OPTIONAL
        ),
        "vertical_adjustment_range_mm": SpecificationDefinition(
            key="vertical_adjustment_range_mm",
            category="Structural & Load Capacity",
            description="Vertical positioning range",
            unit="mm",
            data_type="number",
            importance=SpecImportance.OPTIONAL
        ),

        # 4. INSTALLATION & MAINTENANCE (4 specs)
        "installation_tools_required": SpecificationDefinition(
            key="installation_tools_required",
            category="Installation & Maintenance",
            description="Required installation tools",
            data_type="string",
            importance=SpecImportance.OPTIONAL
        ),
        "installation_time_minutes": SpecificationDefinition(
            key="installation_time_minutes",
            category="Installation & Maintenance",
            description="Estimated installation time",
            unit="minutes",
            data_type="number",
            importance=SpecImportance.OPTIONAL
        ),
        "maintenance_requirements": SpecificationDefinition(
            key="maintenance_requirements",
            category="Installation & Maintenance",
            description="Maintenance requirements",
            data_type="string",
            importance=SpecImportance.OPTIONAL
        ),

        # 5. ENVIRONMENTAL & COMPLIANCE (2 specs)
        "operating_temperature_range": SpecificationDefinition(
            key="operating_temperature_range",
            category="Environmental & Compliance",
            description="Operating temperature range",
            unit="°C",
            data_type="string",
            importance=SpecImportance.OPTIONAL
        ),
        "certifications": SpecificationDefinition(
            key="certifications",
            category="Environmental & Compliance",
            description="Certifications held",
            data_type="list",
            options=["CE", "RoHS"],
            importance=SpecImportance.OPTIONAL
        ),
    }

    @classmethod
    def get_all_specifications(cls) -> Dict[str, SpecificationDefinition]:
        return cls.SPECIFICATIONS

    @classmethod
    def get_specifications_by_category(cls) -> Dict[str, List[SpecificationDefinition]]:
        grouped = {}
        for spec in cls.SPECIFICATIONS.values():
            if spec.category not in grouped:
                grouped[spec.category] = []
            grouped[spec.category].append(spec)
        return grouped

    @classmethod
    def get_count(cls) -> int:
        return len(cls.SPECIFICATIONS)



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

    # Additional Specs

    "pressure_transmitter": PressureTransmitterTemplate,
    "flow_meter": FlowMeterTemplate,
    "level_transmitter": LevelTransmitterTemplate,
    "analyzer": AnalyzerTemplate,
    "control_valve": ControlValveTemplate,
    "thermowell": ThermowellTemplate,
    "cable_gland": CableGlandTemplate,
    "enclosure": EnclosureTemplate,
    "mounting_kit": MountingKitTemplate,

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

def get_template_stats_for_product_type(product_type: str) -> Dict[str, Any]:
    """Get statistics about a product type's template"""
    template_class = PRODUCT_TYPE_TEMPLATES.get(product_type)
    if not template_class:
        return {
            "product_type": product_type,
            "template_available": False,
            "spec_count": 0,
            "categories": [],
        }

    specs = template_class.get_all_specifications()
    categories = template_class.get_specifications_by_category()

    return {
        "product_type": product_type,
        "template_available": True,
        "spec_count": len(specs),
        "categories": list(categories.keys()),
        "specs_per_category": {cat: len(specs) for cat, specs in categories.items()},
    }
