"""
Schema Field Extractor for Deep Agent

This module extracts specific schema field values from standards documents
using LLM-powered RAG queries. It queries the Standards RAG for each schema
field and extracts relevant technical specifications.
"""

import logging
import json
import re
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# =============================================================================
# FIELD CATEGORIES MAPPING
# Maps schema section names to the type of standards query to perform
# =============================================================================

SECTION_TO_QUERY_TYPE = {
    "Performance": "performance specifications technical requirements",
    "Electrical": "electrical specifications power output signal",
    "Mechanical": "mechanical specifications materials construction",
    "Environmental": "environmental operating conditions temperature humidity",
    "Compliance": "certifications standards compliance safety",
    "Features": "features capabilities options",
    "Integration": "communication protocols fieldbus integration",
    "MechanicalOptions": "mounting installation mechanical options",
    "ServiceAndSupport": "calibration maintenance warranty service",
    "Certifications": "certifications approvals ratings",
}

# =============================================================================
# COMPREHENSIVE FIELD VALUE MAPPING
# Maps common schema field names to likely values from standards
# =============================================================================

THERMOCOUPLE_DEFAULTS = {
    # Performance
    "Accuracy": "±0.75% or ±2.5°C (whichever is greater) per IEC 60584",
    "Junction Type": "Grounded, Ungrounded, or Exposed per application",
    "Measurement Type": "Temperature",
    "Thermocouple Type": "Type K (Chromel-Alumel) per IEC 60584-1, Type J/T/N available",
    "Temperature Range": "-200°C to +1260°C per IEC 60584-1",
    "Response Time": "T90 < 10 seconds per application",
    "Repeatability": "±0.1°C or ±0.1% of reading per IEC 60584",
    "Number Of Measuring Points": "1 to 24 points per assembly configuration",
    "Point Spacing Junction Locations": "As per customer specification, typical 50-500mm spacing",

    # Electrical
    "Output Signal": "mV (direct) or 4-20mA with transmitter",
    "Power Supply": "10.5-42.5 VDC loop powered (with transmitter)",
    "Insulation Resistance": ">100MΩ at 500VDC per IEC 60584",
    "Lead Wire Type": "Thermocouple extension wire, Type KX",
    "Wiring Configuration": "2-wire or extension cable",
    "Digital Communication Protocol": "HART 7, Foundation Fieldbus H1, Profibus PA",
    "Cable Entry": "M20x1.5 or 1/2\" NPT cable gland per IEC 60079",
    "Transmitter Integration": "Head-mounted transmitter, 4-20mA HART output",

    # Mechanical
    "Sheath Diameter": "3mm, 4.5mm, 6mm, 8mm per application requirements",
    "Insertion Length": "50mm to 2000mm per process requirements",
    "Process Connection": "1/2\" NPT, 1/2\" BSP, M20x1.5, compression fitting, thermowell per ISO 1179",
    "Sheath Material": "Inconel 600, 316 SS, Hastelloy per process compatibility",
    "Connection Head Material": "Aluminum, 316 SS, or Cast Iron per IEC 60079",
    "Connection Head Type": "KNE, DIN B, explosion-proof weatherproof per IEC 60079",
    "Sensor Element Configuration": "Single or Dual element per application",

    # Environmental
    "Ambient Temperature Range": "-40°C to +85°C",
    "Hazardous Area Rating": "ATEX II 1/2 G Ex ia/d, IECEx, Class I Div 1/2 per IEC 60079",
    "Ingress Protection": "IP66/IP67 per IEC 60529",
    "Storage Temperature": "-40°C to +100°C",
    "Process Pressure Rating": "Up to 400 bar with appropriate thermowell",
    "Vibration Resistance": "4g at 10-500Hz per IEC 60068-2-6",

    # Features
    "Calibration Options": "NIST traceable, IEC 60584-1 compliant, Accredited per ISO/IEC 17025",
    "Cold Junction Compensation": "Internal or External CJC",
    "Data Logging": "Optional with transmitter",
    "Diagnostics": "Sensor break detection, loop diagnostics, corrosion monitoring per NE107",
    "Sensor Failure Indication": "Burnout detection per NAMUR NE43",
    "Transmitter Type": "Head-mounted or Remote transmitter",
    "Local Configuration Interface": "LCD display with pushbuttons",

    # Integration
    "Communication Protocol": "HART, 4-20mA, Foundation Fieldbus",
    "Fieldbus Compatibility": "HART 7, FF H1, Profibus PA",
    "Device Integration Files": "EDDL, FDT/DTM available",
    "Wireless Communication Option": "WirelessHART per IEC 62591",

    # Mechanical Options
    "Adjustable Immersion Length": "100mm to 1000mm typical",
    "Enclosure Type": "Explosion-proof, weatherproof per ATEX/IECEx",
    "Housing Material": "316 SS, Aluminum, or Cast Iron",
    "Mounting Type": "Thermowell-mounted, direct insertion, flange-mounted per application",
    "Termination Type": "Terminal block, flying leads, M12 connector",
    "Thermowell Material": "316 SS, Inconel 600, Hastelloy C-276 per ASME PTC 19.3",
    "Probe Tip Design": "Grounded, Ungrounded, or Exposed",
    "Integral Display Option": "LCD display, -40 to +85°C operation",
    "Extension Lead Wire Length": "1m to 50m per customer specification",
    "Extension Lead Wire Material": "Type KX thermocouple extension wire, PVC/Teflon insulation",
    "Head Enclosure Material": "Aluminum, 316 SS, or Cast Iron per IEC 60079",
    "Internal Insulation Material": "Magnesium oxide (MgO) compacted per IEC 60584",
    "Removable Insert Feature": "Available for easy maintenance and replacement",
    "Spring Loaded Design": "Available for improved thermal contact",
    "Special Sheath Treatments": "Ground finish, passivation, electropolish available",

    # Service and Support
    "Calibration Frequency": "Annual per ISO 9001 / IEC 17025",
    "Material Certificates": "EN 10204 3.1 available",
    "Warranty": "2-5 years manufacturer warranty",

    # Certifications / Compliance
    "Certifications": "ATEX, IECEx, CE, SIL2 capable",
    "Suggested Values": "FM, CSA, ATEX, IECEx certifications available",
    "Safety Integrity Level": "SIL 2 capable per IEC 61508/IEC 61511",
    "PED Compliance": "PED-compliant per Directive 2014/68/EU",
    "Marine Classifications": "DNV, ABS, Lloyd's Register available",
}

PRESSURE_TRANSMITTER_DEFAULTS = {
    "Accuracy": "±0.075% of span per IEC 60770",
    "Measurement Type": "Gauge, Absolute, or Differential",
    "Pressure Measurement Range": "0-10 bar to 0-700 bar per application",
    "Output Signal": "4-20mA with HART per NAMUR NE43",
    "Power Supply": "10.5-42.5 VDC loop powered",
    "Process Connection": "1/2\" NPT per ANSI B1.20.1",
    "Wetted Parts Material": "316L SS, Hastelloy C-276 per NACE MR0175",
    "Housing Material": "Aluminum or 316 SS per IEC 60079",
    "Safety Integrity Level": "SIL 2/3 per IEC 61508/IEC 61511",
    "Hazardous Area Rating": "ATEX II 1/2 G Ex ia/d per IEC 60079",
    "Ingress Protection": "IP66/IP67 per IEC 60529",
    "Ambient Temperature Range": "-40°C to +85°C",
    "Process Temperature Range": "-40°C to +120°C (with remote seals: -70°C to +400°C)",
    "Communication Protocol": "HART 7, Foundation Fieldbus H1, Profibus PA, WirelessHART",
    "Response Time": "100ms typical per IEC 60770",
    "Long Term Stability": "±0.1% URL per 5 years",
    "Turndown Ratio": "100:1",
    "Repeatability": "±0.05% of span",
}

TEMPERATURE_SENSOR_DEFAULTS = {
    # Performance
    "Accuracy": "Class A: ±(0.15 + 0.002×|t|)°C per IEC 60751",
    "Accuracy Class": "Class A: ±(0.15 + 0.002×|t|)°C per IEC 60751",
    "Sensor Type": "Pt100/Pt1000 RTD or Type K/J/T Thermocouple",
    "Temperature Range": "-200°C to +850°C (RTD) / -200°C to +1260°C (T/C)",
    "Measurement Range": "-200°C to +1260°C per IEC 60584/IEC 60751",
    "Response Time": "T90 < 5 seconds per IEC 60751",
    "Response Rate": "T90 < 5 seconds per IEC 60751",

    # Electrical
    "Output Signal": "4-20mA with HART, or resistance output",
    "Output Signal Type": "4-20mA DC with HART per NAMUR NE43",
    "Power Supply": "10.5-42.5 VDC loop powered",
    "Power Supply Range": "10.5-42.5 VDC loop powered per NAMUR NE21",
    "Digital Communication Protocol": "HART 7, Foundation Fieldbus H1, Profibus PA",
    "Wiring Configuration": "2-wire, 3-wire, or 4-wire per IEC 60751",

    # Mechanical
    "Sheath Material": "316 SS, Inconel 600, Hastelloy per process compatibility",
    "Wetted Parts Material": "316L SS, Inconel 600, Hastelloy C-276 per NACE MR0175",
    "Probe Diameter": "3mm, 6mm, 8mm per application",
    "Probe Length": "50mm to 2000mm per application",
    "Insertion Length Range": "50mm to 2000mm per application",
    "Process Connection": "1/2\" NPT, compression fitting, thermowell",
    "Process Connection Method": "1/2\" NPT, 3/4\" NPT, flanged, compression fitting per ASME B1.20.1",
    "Connection Head Material": "Aluminum, 316 SS, or Cast Iron per IEC 60079",
    "Sensor Element Configuration": "Single or Dual element per application",

    # Environmental
    "Insulation Resistance": ">100MΩ at 500VDC per IEC 60751",
    "Ingress Protection": "IP66/IP67 per IEC 60529",
    "Enclosure Protection Rating": "IP66/IP67 per IEC 60529",
    "Ambient Operating Temperature Range": "-40°C to +85°C",
    "Storage Temperature Range": "-40°C to +100°C",
    "Process Pressure Rating": "Up to 400 bar with appropriate thermowell",
    "Vibration Resistance Standards": "IEC 60068-2-6: 4g, 10-500Hz",

    # Compliance
    "Hazardous Area Certifications": "ATEX II 1/2 G Ex ia/d, IECEx, Class I Div 1/2 per IEC 60079",
    "Safety Integrity Level": "SIL 2 capable per IEC 61508/IEC 61511",

    # Features
    "Calibration Options": "NIST traceable, IEC 60751 compliant",
    "Cold Junction Compensation": "Internal or External CJC",
    "Data Logging": "Optional with transmitter",
    "Data Logging Capability": "Optional with smart transmitter",
    "Diagnostics": "Sensor break detection, loop diagnostics",
    "Advanced Diagnostic Functions": "Sensor integrity, loop diagnostics, device health per NE107",
    "Sensor Failure Indication": "Burnout detection per NAMUR NE43",
    "Transmitter Type": "Head-mounted or Remote transmitter",
    "Local Configuration Interface": "LCD display with pushbuttons",

    # Integration
    "Communication Protocol": "HART, 4-20mA, Foundation Fieldbus",
    "Fieldbus Compatibility": "HART 7, FF H1, Profibus PA",
    "Device Integration Files": "EDDL, FDT/DTM available",
    "Wireless Communication Option": "WirelessHART per IEC 62591",

    # Mechanical Options
    "Adjustable Immersion Length": "100mm to 1000mm typical",
    "Enclosure Type": "Explosion-proof, weatherproof per ATEX/IECEx",
    "Housing Material": "316 SS, Aluminum, or Cast Iron",
    "Mounting Options": "Thermowell, direct insertion, flanged",
    "Termination Type": "Terminal block, flying leads, M12 connector",
    "Connection Head Design": "KNE, DIN B, explosion-proof per IEC 60079",
    "Thermowell Type And Material": "Barstock 316 SS, flanged or threaded per ASME PTC 19.3",
    "Probe Tip Design": "Grounded, Ungrounded, or Exposed",
    "Cable Entry Type And Size": "M20x1.5 or 1/2\" NPT cable gland",
    "Integral Display Option": "LCD display, -40 to +85°C operation",

    # Service and Support
    "Calibration Frequency": "Annual per ISO 9001 / IEC 17025",
    "Calibration Certificate Options": "Works, NIST traceable, or Accredited per ISO/IEC 17025",
    "Material Certificates": "EN 10204 3.1 available",
    "Warranty": "Standard manufacturer warranty",
    "Warranty Period": "2-5 years manufacturer warranty",

    # Certifications
    "Certifications": "ATEX, IECEx, CE, FM, CSA, SIL2 capable",
    "Suggested Values": "ATEX, IECEx, FM, CSA, UL certifications",
    "PED Compliance": "PED-compliant per Directive 2014/68/EU",
    "Marine Classifications": "DNV, ABS, Lloyd's Register available",
}

# =============================================================================
# CONNECTOR / CABLE DEFAULTS
# =============================================================================
CONNECTOR_DEFAULTS = {
    # Electrical
    "Contact Resistance": "< 10 mΩ per IEC 60512",
    "Current Rating": "4-20mA per NAMUR NE43",
    "Voltage Rating": "30-250 V per IEC 61076",
    "Dielectric Withstanding Voltage": "500-1000 VAC per IEC 60512",
    "Insulation Resistance": "> 5000 MΩ-km per IEC 60512",
    "Numberof Contacts": "2, 3, 4, 5, 8 pins per application",
    "Shielding": "Shielded or Unshielded per EMC requirements",
    "Surge Protection": "6 kV per IEC 61000-4-5",

    # Mechanical
    "Connector Type": "M12, M8 circular per IEC 61076",
    "Connector Style": "Straight, Right-angle per application",
    "Contact Material": "Brass, Gold Plated per IEC 60512",
    "Housing Material": "Brass, Stainless Steel, Plastic (PA/PBT)",
    "Wire Gauge Size": "24-16 AWG per IEC 60228",
    "Mounting Type": "Panel Mount, Cable Mount per application",
    "Durability Mating Cycles": "> 500 cycles per IEC 60512",
    "Cable Retention Force": "> 80 N per IEC 60512",

    # Environmental
    "Operating Temperature Range": "-40°C to +85°C",
    "Storage Temperature Range": "-40°C to +85°C",
    "Ingress Protection": "IP67 per IEC 60529",
    "Vibration Resistance": "10-2000 Hz at 1.5mm, 20g per IEC 60068-2-6",
    "Shock Resistance": "50g per IEC 60068-2-27",
    "Flammability Rating": "UL94 V-0",

    # Features
    "Integrated Cable Assembly": "Available per application",
    "Tooling Options": "Crimping, IDC, Screw terminal",
    "Locking Feature": "Screw locking, Push-pull per IEC 61076",
    "Keying": "A-coded, B-coded, D-coded per IEC 61076",
    "Polarization": "Keyed for proper orientation",

    # Materials
    "Contact Plating": "Gold over Nickel per IEC 60512",
    "Plating Thickness": "0.76 µm Au minimum",

    # Compliance
    "Certifications": "RoHS, REACH, UL, CE",
    "REACHCompliance": "Yes",
    "Ro HSCompliance": "Yes",
}

# =============================================================================
# JUNCTION BOX / ENCLOSURE DEFAULTS
# =============================================================================
JUNCTION_BOX_DEFAULTS = {
    # Mechanical
    "Material Of Construction": "Stainless steel 316L, Aluminum, GRP/Fiberglass",
    "Enclosure Material": "Stainless Steel 316L, Aluminum, Fiberglass per NEMA 250",
    "Enclosure Type": "Explosion-proof, Weatherproof per IEC 60079",
    "Enclosure Size": "Various sizes per application",
    "Mounting Type": "Wall, Pole, Panel mount",
    "Mounting Method": "Wall bracket, Pole clamp, Panel cutout",
    "Dimensions": "Varies by model",
    "Weight": "Varies by model and material",
    "Door Type": "Hinged, Screw cover per application",
    "Hinge Type": "Continuous, Piano hinge per application",
    "Material Thickness": "1.5-3.0 mm per NEMA 250",
    "Finish": "Electro-polished, Powder coated",

    # Environmental
    "Operating Temperature Range": "-40°C to +80°C",
    "Storage Temperature Range": "-40°C to +70°C",
    "Ingress Protection": "IP66/IP67 per IEC 60529",
    "IP Rating": "IP66/IP67 per IEC 60529",
    "NEMA Rating": "NEMA 4X per NEMA 250",
    "Corrosion Resistance": "ASTM B117 Salt Spray 500+ hours",
    "Salt Spray Resistance": "720 hours per ASTM B117",
    "Humidity Resistance": "95% RH non-condensing",
    "UV Resistance": "UV stabilized per ASTM G154",
    "Vibration Resistance": "10-55Hz, 1.5mm per IEC 60068-2-6",
    "Shock Resistance": "30g per IEC 60068-2-27",

    # Electrical
    "Grounding Connection": "Internal ground stud provided",
    "Grounding Provisions": "Yes, per NEC requirements",
    "Cable Entry Type": "M20, M25, 1/2\" NPT, 3/4\" NPT per application",
    "Number Cable Entries": "2-12 per model",

    # Features
    "Locking Mechanism": "Padlock hasp, Key lock per application",
    "Internal Mounting Panel Material": "Galvanized steel, Aluminum per application",
    "Access Restrictions": "Lockable per security requirements",
    "Integrated Damping": "Available per application",

    # Certifications
    "Certifications": "ATEX, IECEx, cULus, CE, RoHS",
    "UL Listing": "UL 508A, UL 50E per application",
    "Hazardous Area Rating": "ATEX Zone 1/2, Class I Div 1/2 per IEC 60079",
    "Safety Standards": "IEC 61010-1",

    # Service
    "Maintenance Requirements": "Regular inspection per manufacturer guidelines",
    "Warranty Period": "1-2 years manufacturer warranty",
}

# =============================================================================
# MOUNTING BRACKET DEFAULTS
# =============================================================================
MOUNTING_BRACKET_DEFAULTS = {
    # Mechanical
    "Material Of Construction": "Stainless steel 316L, Carbon steel, Aluminum",
    "Mounting Type": "Wall, Pipe, Panel, DIN rail",
    "Load Capacity": "25-100 kg per design",
    "Finish": "Electro-polished, Hot-dip galvanized, Powder coated",
    "Hardware Material": "Stainless Steel 304/316 per application",
    "Dimensions": "Varies by model",
    "Weight": "Varies by model",

    # Environmental
    "Operating Temperature Range": "-40°C to +85°C",
    "Corrosion Resistance": "ASTM B117 per material",
    "UV Resistance": "UV stabilized coating available",

    # Features
    "Adjustability": "±10 degrees tilt adjustment",
    "Compatibility": "Universal or instrument-specific",
    "Adapter Plate Compatibility": "VESA 75/100 mm where applicable",
    "Cable Management Features": "Integrated cable routing",
    "Tagging Options": "Engraved or stamped tags available",
    "Custom Drilling": "Available per customer specification",

    # Coating
    "Coating Type": "Powder Coat, Epoxy, Hot-dip galvanized",
    "Surface Finish": "Ra < 3.2 µm for stainless steel",
    "Color": "Silver, Gray, Custom colors available",

    # Certifications
    "Certifications": "CE, RoHS, REACH",

    # Service
    "Maintenance Requirements": "Annual inspection of fasteners",
    "Warranty Period": "1-2 years manufacturer warranty",
}

# =============================================================================
# FLOW METER / FLOW TRANSMITTER DEFAULTS
# =============================================================================
FLOW_METER_DEFAULTS = {
    # Performance
    "Accuracy": "±0.5% of reading per ISO 5167 / API MPMS",
    "Repeatability": "±0.1% of reading",
    "Turndown Ratio": "100:1 or higher",
    "Flow Range": "0.01 to 10,000 m³/h depending on size",
    "Measurement Type": "Volumetric or Mass flow",
    "Response Time": "<1 second per IEC 61298",

    # Electrical
    "Output Signal": "4-20mA with HART per NAMUR NE43",
    "Power Supply": "10.5-42.5 VDC loop powered or 85-265 VAC",
    "Communication Protocol": "HART 7, Foundation Fieldbus H1, Profibus PA, Modbus",
    "Digital Communication Protocol": "HART 7, Foundation Fieldbus, Modbus RTU/TCP",

    # Mechanical
    "Process Connection": "Flanged (ANSI, DIN, JIS), wafer, threaded per ASME B16.5",
    "Wetted Parts Material": "316L SS, Hastelloy C-276, Tantalum per NACE MR0175",
    "Housing Material": "Aluminum or 316 SS per IEC 60079",
    "Line Size": "DN15 to DN3000 / 0.5\" to 120\"",

    # Environmental
    "Process Temperature Range": "-200°C to +400°C",
    "Process Pressure Rating": "Up to 420 bar / 6000 psi per ASME B16.5",
    "Ambient Temperature Range": "-40°C to +85°C",
    "Ingress Protection": "IP66/IP67 per IEC 60529",
    "Hazardous Area Rating": "ATEX II 1/2 G Ex ia/d, IECEx, Class I Div 1/2 per IEC 60079",

    # Features
    "Diagnostics": "Advanced diagnostics per NE107, process noise, electrode coating detection",
    "Local Display": "Backlit LCD with configuration buttons",
    "Calibration Options": "NIST traceable, ISO 17025 accredited",

    # Compliance
    "Safety Integrity Level": "SIL 2/3 per IEC 61508/IEC 61511",
    "Custody Transfer": "MID, OIML R117, API MPMS compliant",
    "Certifications": "ATEX, IECEx, FM, CSA, CE, MID",
}

# =============================================================================
# LEVEL TRANSMITTER DEFAULTS
# =============================================================================
LEVEL_TRANSMITTER_DEFAULTS = {
    # Performance
    "Accuracy": "±0.1% of span per IEC 61298",
    "Repeatability": "±0.05% of span",
    "Measurement Range": "0.3m to 100m depending on technology",
    "Resolution": "1 mm or better",
    "Response Time": "<0.5 seconds per IEC 61298",
    "Measurement Type": "Continuous level, point level, or interface",

    # Electrical
    "Output Signal": "4-20mA with HART per NAMUR NE43",
    "Power Supply": "10.5-42.5 VDC loop powered",
    "Communication Protocol": "HART 7, Foundation Fieldbus H1, Profibus PA, WirelessHART",

    # Mechanical
    "Process Connection": "Flanged, threaded (NPT/BSP), sanitary (Tri-clamp) per ASME B16.5",
    "Wetted Parts Material": "316L SS, PTFE, Hastelloy, ceramic per NACE MR0175",
    "Housing Material": "Aluminum or 316 SS per IEC 60079",
    "Antenna Type": "Horn, rod, parabolic, drop antenna per application",

    # Environmental
    "Process Temperature Range": "-196°C to +450°C",
    "Process Pressure Rating": "Up to 400 bar / 5800 psi per ASME B16.5",
    "Ambient Temperature Range": "-40°C to +80°C",
    "Ingress Protection": "IP66/IP67/IP68 per IEC 60529",
    "Hazardous Area Rating": "ATEX II 1/2 G Ex ia/d, IECEx, Class I Div 1/2 per IEC 60079",

    # Features
    "Diagnostics": "Echo curve analysis, blocking distance alarm per NE107",
    "Local Display": "Backlit LCD with pushbuttons",
    "Empty Pipe Detection": "Available for radar and ultrasonic",
    "Foam Compensation": "Available for radar technology",

    # Compliance
    "Safety Integrity Level": "SIL 2/3 per IEC 61508/IEC 61511",
    "Certifications": "ATEX, IECEx, FM, CSA, CE, 3A sanitary",
    "Marine Classifications": "DNV, ABS, Lloyd's Register available",
}

# =============================================================================
# CONTROL VALVE DEFAULTS
# =============================================================================
CONTROL_VALVE_DEFAULTS = {
    # Performance
    "Rangeability": "50:1 minimum per IEC 60534",
    "Leakage Class": "Class IV, V, or VI per ANSI/FCI 70-2",
    "Cv Range": "0.01 to 10,000+ per valve size",
    "Flow Characteristic": "Linear, Equal Percentage, Quick Opening per IEC 60534",
    "Trim Type": "Parabolic, V-notch, cage-guided per application",

    # Mechanical
    "Body Material": "Carbon steel, 316 SS, Alloy 20, Hastelloy per ASME B16.34",
    "Trim Material": "316 SS, 17-4PH, Stellite, Tungsten Carbide per service",
    "Packing Material": "PTFE, Graphite, low-emission per EPA Method 21",
    "End Connections": "Flanged RF/RTJ, butt-weld, socket-weld per ASME B16.5",
    "Pressure Class": "ANSI 150, 300, 600, 900, 1500, 2500 per ASME B16.34",
    "Body Size": "1/2\" to 36\" per application",

    # Actuator
    "Actuator Type": "Pneumatic (spring-return/double-acting), Electric, Hydraulic",
    "Air Supply": "3-15 psi or 6-30 psi per ISA S7.0.01",
    "Fail-Safe Action": "Fail-Open, Fail-Close, Fail-Last per application",

    # Environmental
    "Process Temperature Range": "-196°C to +815°C per material selection",
    "Ambient Temperature Range": "-40°C to +85°C",
    "Hazardous Area Rating": "ATEX II 2 G/D, IECEx, Class I Div 1/2 per IEC 60079",

    # Positioner
    "Positioner Type": "Digital/Smart positioner with diagnostics",
    "Communication Protocol": "HART, Foundation Fieldbus, Profibus PA",
    "Input Signal": "4-20mA or digital fieldbus",

    # Features
    "Diagnostics": "Valve signature analysis, travel deviation, friction monitoring",
    "Position Feedback": "4-20mA, HART, limit switches",
    "Noise Reduction": "Low-noise trim, diffuser plates available",
    "Cavitation Control": "Anti-cavitation trim available per IEC 60534-8",

    # Compliance
    "Safety Integrity Level": "SIL 2/3 per IEC 61508/IEC 61511",
    "Fugitive Emissions": "ISO 15848-1 certified, EPA Method 21 compliant",
    "Certifications": "ATEX, IECEx, SIL, PED, NACE MR0175",
    "Fire-Safe": "API 607 / ISO 10497 fire-tested",
}

# =============================================================================
# ANALYZER DEFAULTS (pH, Conductivity, Dissolved Oxygen, etc.)
# =============================================================================
ANALYZER_DEFAULTS = {
    # Performance - pH
    "pH Measurement Range": "0 to 14 pH",
    "pH Accuracy": "±0.01 pH per IEC 746-2",
    "pH Resolution": "0.001 pH",

    # Performance - Conductivity
    "Conductivity Range": "0.001 µS/cm to 2000 mS/cm",
    "Conductivity Accuracy": "±0.5% of reading per IEC 746-3",

    # Performance - Dissolved Oxygen
    "DO Range": "0 to 20 mg/L (ppm) or 0 to 200% saturation",
    "DO Accuracy": "±1% of reading or ±0.1 mg/L",

    # Performance - General
    "Response Time": "T90 < 30 seconds per application",
    "Sensor Type": "pH glass, ion-selective, amperometric, optical per application",

    # Electrical
    "Output Signal": "4-20mA with HART, RS-485 Modbus",
    "Power Supply": "24 VDC or 85-265 VAC",
    "Communication Protocol": "HART, Modbus RTU/TCP, Profibus PA",

    # Mechanical
    "Process Connection": "1\" NPT, 1-1/2\" NPT, DN25 flange, sanitary per application",
    "Wetted Parts Material": "316L SS, PEEK, PTFE, Hastelloy per media compatibility",
    "Housing Material": "316 SS, Polycarbonate, Aluminum per IEC 60079",
    "Sensor Mounting": "Immersion, retractable, flow-through per application",

    # Environmental
    "Process Temperature Range": "-10°C to +130°C",
    "Process Pressure Rating": "Up to 10 bar",
    "Ambient Temperature Range": "-20°C to +60°C",
    "Ingress Protection": "IP66/IP67 per IEC 60529",
    "Hazardous Area Rating": "ATEX II 1/2 G Ex ia/d, IECEx per IEC 60079",

    # Features
    "Diagnostics": "Sensor health, calibration due, electrode impedance",
    "Local Display": "Backlit LCD touchscreen or keypad",
    "Calibration": "1-point, 2-point, or grab sample calibration",
    "Auto-Cleaning": "Automatic cleaning cycles available",
    "Temperature Compensation": "Automatic temperature compensation (ATC)",

    # Compliance
    "Certifications": "ATEX, IECEx, CE, 3A, EHEDG (sanitary)",
    "Water Quality": "EPA, drinking water, USP purified water compliant",
}

# =============================================================================
# SIGNAL CONDITIONER / ISOLATOR DEFAULTS
# =============================================================================
SIGNAL_CONDITIONER_DEFAULTS = {
    # Performance
    "Accuracy": "±0.05% of span or better",
    "Linearity": "±0.01% of span",
    "Temperature Drift": "<50 ppm/°C",
    "Response Time": "<100 ms",
    "Bandwidth": "DC to 10 kHz typical",

    # Input
    "Input Types": "4-20mA, 0-10V, RTD, TC, mV, frequency, pulse",
    "Input Impedance": ">10 MΩ for voltage, <250Ω for current",
    "Thermocouple Types": "K, J, T, N, R, S, B per IEC 60584",
    "RTD Types": "Pt100, Pt1000, 2/3/4-wire per IEC 60751",

    # Output
    "Output Signal": "4-20mA, 0-10V, relay, frequency",
    "Output Load": ">500Ω at 24V for current output",
    "Isolation Voltage": "2500 Vrms or 3000 VDC per IEC 61010",

    # Electrical
    "Power Supply": "24 VDC (18-32V) or loop powered",
    "Power Consumption": "<1.5W typical",
    "Loop Power": "24 VDC at up to 22mA for 2-wire transmitters",

    # Mechanical
    "Mounting Type": "DIN rail (35mm), panel mount",
    "Housing Material": "Fire-retardant plastic UL94 V-0",
    "Dimensions": "Compact 6mm, 12.5mm, or 22.5mm width",

    # Environmental
    "Operating Temperature Range": "-20°C to +60°C",
    "Storage Temperature Range": "-40°C to +85°C",
    "Humidity": "10-95% RH non-condensing",

    # Features
    "Galvanic Isolation": "3-way isolation (input/output/power)",
    "Configuration": "DIP switches, potentiometers, or software programmable",
    "LED Indicators": "Power, fault, status LEDs",
    "Surge Protection": "Built-in transient protection",

    # Compliance
    "Safety": "IEC 61010-1, UL/cUL listed",
    "EMC": "CE marked per EN 61326-1",
    "SIL Capability": "SIL 2 per IEC 61508 (selected models)",
    "Certifications": "ATEX Zone 2, Class I Div 2 (selected models)",
}

# =============================================================================
# CALIBRATOR DEFAULTS
# =============================================================================
CALIBRATOR_DEFAULTS = {
    # Performance
    "Accuracy": "±0.01% to ±0.05% of reading per calibration function",
    "Resolution": "5 to 6 digits",
    "Stability": "<25 ppm/year for reference standards",

    # Measurement & Source
    "Voltage Range": "0-100 VDC measure/source",
    "Current Range": "0-24 mA or 0-50 mA measure/source",
    "Resistance Range": "0-4000Ω for RTD simulation",
    "Temperature Simulation": "TC/RTD simulation per IEC 60584/60751",
    "Pressure Range": "-1 to 700 bar with external modules",
    "Frequency Range": "0.01 Hz to 50 kHz",

    # Electrical
    "Power Supply": "Rechargeable battery, 4-8 hours operation",
    "Loop Power": "24 VDC at 24mA for transmitter testing",
    "Communication": "USB, Bluetooth, RS-232 for documentation",

    # Features
    "HART Communication": "HART communicator function built-in",
    "Data Logging": "Internal memory for calibration records",
    "Auto-Step": "Automated calibration routines",
    "Diagnostics": "Loop diagnostics, 24V power for transmitters",
    "Documentation": "Automatic calibration certificates",

    # Environmental
    "Operating Temperature Range": "-10°C to +50°C",
    "Storage Temperature Range": "-20°C to +60°C",
    "Ingress Protection": "IP54 minimum for field use",
    "Drop Resistance": "1m drop test per IEC 61010",

    # Mechanical
    "Housing Material": "Rugged ABS with rubber overmold",
    "Display": "Backlit LCD, 5.5 digits or graphical",
    "Weight": "Portable handheld design <1 kg",

    # Compliance
    "Safety": "IEC 61010-1 CAT II 300V",
    "Calibration": "NIST traceable, ISO 17025 accredited",
    "EMC": "CE marked per EN 61326-1",
}

# Map product types to their defaults - EXTENDED for better matching
PRODUCT_TYPE_DEFAULTS = {
    # Thermocouples - all variants
    "thermocouple": THERMOCOUPLE_DEFAULTS,
    "type k thermocouple": THERMOCOUPLE_DEFAULTS,
    "type j thermocouple": THERMOCOUPLE_DEFAULTS,
    "type t thermocouple": THERMOCOUPLE_DEFAULTS,
    "type n thermocouple": THERMOCOUPLE_DEFAULTS,
    "type r thermocouple": THERMOCOUPLE_DEFAULTS,
    "type s thermocouple": THERMOCOUPLE_DEFAULTS,
    "multipoint thermocouple": THERMOCOUPLE_DEFAULTS,
    "thermocouple assembly": THERMOCOUPLE_DEFAULTS,
    "thermocouple sensor": THERMOCOUPLE_DEFAULTS,
    "tc sensor": THERMOCOUPLE_DEFAULTS,

    # Temperature sensors - all variants
    "temperature sensor": TEMPERATURE_SENSOR_DEFAULTS,
    "temperature transmitter": TEMPERATURE_SENSOR_DEFAULTS,
    "temp sensor": TEMPERATURE_SENSOR_DEFAULTS,
    "temp transmitter": TEMPERATURE_SENSOR_DEFAULTS,
    "temperature measurement": TEMPERATURE_SENSOR_DEFAULTS,
    "rtd": TEMPERATURE_SENSOR_DEFAULTS,
    "pt100": TEMPERATURE_SENSOR_DEFAULTS,
    "pt1000": TEMPERATURE_SENSOR_DEFAULTS,
    "resistance temperature": TEMPERATURE_SENSOR_DEFAULTS,
    "rtd sensor": TEMPERATURE_SENSOR_DEFAULTS,
    "rtd assembly": TEMPERATURE_SENSOR_DEFAULTS,

    # Pressure - all variants
    "pressure transmitter": PRESSURE_TRANSMITTER_DEFAULTS,
    "pressure sensor": PRESSURE_TRANSMITTER_DEFAULTS,
    "pressure gauge": PRESSURE_TRANSMITTER_DEFAULTS,
    "pressure transducer": PRESSURE_TRANSMITTER_DEFAULTS,
    "differential pressure": PRESSURE_TRANSMITTER_DEFAULTS,
    "absolute pressure": PRESSURE_TRANSMITTER_DEFAULTS,
    "gauge pressure": PRESSURE_TRANSMITTER_DEFAULTS,
    "pressure measurement": PRESSURE_TRANSMITTER_DEFAULTS,

    # Flow meters
    "flow meter": FLOW_METER_DEFAULTS,
    "flow transmitter": FLOW_METER_DEFAULTS,
    "flowmeter": FLOW_METER_DEFAULTS,
    "flow sensor": FLOW_METER_DEFAULTS,
    "magnetic flow meter": FLOW_METER_DEFAULTS,
    "mag meter": FLOW_METER_DEFAULTS,
    "coriolis meter": FLOW_METER_DEFAULTS,
    "coriolis flow meter": FLOW_METER_DEFAULTS,
    "mass flow meter": FLOW_METER_DEFAULTS,
    "ultrasonic flow meter": FLOW_METER_DEFAULTS,
    "vortex flow meter": FLOW_METER_DEFAULTS,
    "turbine flow meter": FLOW_METER_DEFAULTS,
    "dp flow meter": FLOW_METER_DEFAULTS,
    "orifice plate": FLOW_METER_DEFAULTS,
    "venturi": FLOW_METER_DEFAULTS,
    "flow measurement": FLOW_METER_DEFAULTS,

    # Level transmitters
    "level transmitter": LEVEL_TRANSMITTER_DEFAULTS,
    "level sensor": LEVEL_TRANSMITTER_DEFAULTS,
    "level meter": LEVEL_TRANSMITTER_DEFAULTS,
    "level gauge": LEVEL_TRANSMITTER_DEFAULTS,
    "radar level": LEVEL_TRANSMITTER_DEFAULTS,
    "guided wave radar": LEVEL_TRANSMITTER_DEFAULTS,
    "gwr": LEVEL_TRANSMITTER_DEFAULTS,
    "ultrasonic level": LEVEL_TRANSMITTER_DEFAULTS,
    "capacitance level": LEVEL_TRANSMITTER_DEFAULTS,
    "dp level": LEVEL_TRANSMITTER_DEFAULTS,
    "hydrostatic level": LEVEL_TRANSMITTER_DEFAULTS,
    "float level": LEVEL_TRANSMITTER_DEFAULTS,
    "level switch": LEVEL_TRANSMITTER_DEFAULTS,
    "level measurement": LEVEL_TRANSMITTER_DEFAULTS,

    # Control valves
    "control valve": CONTROL_VALVE_DEFAULTS,
    "control valves": CONTROL_VALVE_DEFAULTS,
    "valve": CONTROL_VALVE_DEFAULTS,
    "globe valve": CONTROL_VALVE_DEFAULTS,
    "ball valve": CONTROL_VALVE_DEFAULTS,
    "butterfly valve": CONTROL_VALVE_DEFAULTS,
    "rotary valve": CONTROL_VALVE_DEFAULTS,
    "diaphragm valve": CONTROL_VALVE_DEFAULTS,
    "actuated valve": CONTROL_VALVE_DEFAULTS,
    "pneumatic valve": CONTROL_VALVE_DEFAULTS,
    "electric valve": CONTROL_VALVE_DEFAULTS,
    "positioner": CONTROL_VALVE_DEFAULTS,
    "valve positioner": CONTROL_VALVE_DEFAULTS,

    # Analyzers
    "analyzer": ANALYZER_DEFAULTS,
    "ph analyzer": ANALYZER_DEFAULTS,
    "ph meter": ANALYZER_DEFAULTS,
    "ph sensor": ANALYZER_DEFAULTS,
    "conductivity analyzer": ANALYZER_DEFAULTS,
    "conductivity meter": ANALYZER_DEFAULTS,
    "dissolved oxygen": ANALYZER_DEFAULTS,
    "do analyzer": ANALYZER_DEFAULTS,
    "turbidity analyzer": ANALYZER_DEFAULTS,
    "chlorine analyzer": ANALYZER_DEFAULTS,
    "orp analyzer": ANALYZER_DEFAULTS,
    "water quality analyzer": ANALYZER_DEFAULTS,
    "gas analyzer": ANALYZER_DEFAULTS,
    "oxygen analyzer": ANALYZER_DEFAULTS,
    "process analyzer": ANALYZER_DEFAULTS,

    # Signal conditioners
    "signal conditioner": SIGNAL_CONDITIONER_DEFAULTS,
    "signal conditioners": SIGNAL_CONDITIONER_DEFAULTS,
    "isolator": SIGNAL_CONDITIONER_DEFAULTS,
    "signal isolator": SIGNAL_CONDITIONER_DEFAULTS,
    "loop isolator": SIGNAL_CONDITIONER_DEFAULTS,
    "transmitter isolator": SIGNAL_CONDITIONER_DEFAULTS,
    "galvanic isolator": SIGNAL_CONDITIONER_DEFAULTS,
    "i/p converter": SIGNAL_CONDITIONER_DEFAULTS,
    "current converter": SIGNAL_CONDITIONER_DEFAULTS,
    "rtd converter": SIGNAL_CONDITIONER_DEFAULTS,
    "temperature converter": SIGNAL_CONDITIONER_DEFAULTS,
    "signal converter": SIGNAL_CONDITIONER_DEFAULTS,
    "din rail module": SIGNAL_CONDITIONER_DEFAULTS,

    # Calibrators
    "calibrator": CALIBRATOR_DEFAULTS,
    "calibrators": CALIBRATOR_DEFAULTS,
    "process calibrator": CALIBRATOR_DEFAULTS,
    "multifunction calibrator": CALIBRATOR_DEFAULTS,
    "pressure calibrator": CALIBRATOR_DEFAULTS,
    "temperature calibrator": CALIBRATOR_DEFAULTS,
    "loop calibrator": CALIBRATOR_DEFAULTS,
    "hart communicator": CALIBRATOR_DEFAULTS,
    "documenting calibrator": CALIBRATOR_DEFAULTS,

    # Connectors / Cables
    "connector": CONNECTOR_DEFAULTS,
    "cable connector": CONNECTOR_DEFAULTS,
    "cable/connector": CONNECTOR_DEFAULTS,
    "cable/connectors": CONNECTOR_DEFAULTS,
    "m12 connector": CONNECTOR_DEFAULTS,
    "m8 connector": CONNECTOR_DEFAULTS,
    "circular connector": CONNECTOR_DEFAULTS,
    "cable assembly": CONNECTOR_DEFAULTS,
    "cable": CONNECTOR_DEFAULTS,
    "extension cable": CONNECTOR_DEFAULTS,
    "instrumentation cable": CONNECTOR_DEFAULTS,

    # Junction Boxes / Enclosures
    "junction box": JUNCTION_BOX_DEFAULTS,
    "junction boxes": JUNCTION_BOX_DEFAULTS,
    "junction boxes and enclosures": JUNCTION_BOX_DEFAULTS,
    "enclosure": JUNCTION_BOX_DEFAULTS,
    "terminal box": JUNCTION_BOX_DEFAULTS,
    "connection box": JUNCTION_BOX_DEFAULTS,
    "field enclosure": JUNCTION_BOX_DEFAULTS,
    "instrument enclosure": JUNCTION_BOX_DEFAULTS,

    # Mounting Brackets
    "mounting bracket": MOUNTING_BRACKET_DEFAULTS,
    "mounting brackets": MOUNTING_BRACKET_DEFAULTS,
    "bracket": MOUNTING_BRACKET_DEFAULTS,
    "pipe mount": MOUNTING_BRACKET_DEFAULTS,
    "wall mount": MOUNTING_BRACKET_DEFAULTS,
    "panel mount": MOUNTING_BRACKET_DEFAULTS,
    "mounting hardware": MOUNTING_BRACKET_DEFAULTS,
    "instrument mount": MOUNTING_BRACKET_DEFAULTS,
}


# Regex patterns for extracting standard codes from value strings
STANDARDS_PATTERNS = [
    r'\bIEC\s*\d+[-\d]*',           # IEC 60584, IEC 60584-1, IEC 60751
    r'\bISO\s*\d+[-\d]*',           # ISO 9001, ISO 1179
    r'\bASME\s*[A-Z]*\s*[\d.]+',    # ASME B1.20.1, ASME PTC 19.3
    r'\bANSI\s*[A-Z]*\s*[\d.]+',    # ANSI B1.20.1
    r'\bATEX\s*[IVX]*\s*\d*',       # ATEX, ATEX II
    r'\bIECEx',                      # IECEx
    r'\bNAMUR\s*NE\s*\d+',          # NAMUR NE43, NAMUR NE21
    r'\bNACE\s*MR\s*\d+',           # NACE MR0175
    r'\bAPI\s*\d+',                  # API standards
    r'\bEN\s*\d+[-\d]*',            # EN 10204
    r'\bSIL\s*\d+',                  # SIL 2, SIL 3
    r'\bPED\s*[-\w]*',              # PED compliance
    r'\bHART\s*\d*',                 # HART, HART 7
    r'\bNIST',                       # NIST traceable
]


def extract_standards_from_value(value: str) -> List[str]:
    """
    Extract standard codes referenced in a value string.

    Args:
        value: The field value string (e.g., "±0.75% or ±2.5°C per IEC 60584")

    Returns:
        List of standard codes found (e.g., ["IEC 60584"])
    """
    if not value:
        return []

    standards = []
    for pattern in STANDARDS_PATTERNS:
        matches = re.findall(pattern, value, re.IGNORECASE)
        for match in matches:
            # Clean up the match
            cleaned = match.strip()
            if cleaned and cleaned not in standards:
                standards.append(cleaned)

    return standards


def _camel_to_words(name: str) -> str:
    """Convert camelCase or PascalCase to space-separated words."""
    # Insert space before uppercase letters that follow lowercase letters
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', name)
    # Insert space before uppercase letters that are followed by lowercase
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1 \2', s1)
    return s2.lower()


def _normalize_field_name(field_name: str) -> str:
    """Normalize a field name for matching."""
    # Convert camelCase to words
    words = _camel_to_words(field_name)
    # Also replace underscores and hyphens
    words = words.replace("_", " ").replace("-", " ")
    # Remove extra spaces
    return " ".join(words.split())


def get_default_value_for_field(product_type: str, field_name: str) -> Optional[str]:
    """
    Get a default value for a schema field based on product type.

    Uses multi-strategy matching:
    1. Exact product type match
    2. Substring match (both directions)
    3. Keyword-based match

    Args:
        product_type: The product type (e.g., "Type K Thermocouple", "Multipoint Thermocouple Assembly")
        field_name: The schema field name (e.g., "Accuracy" or "accuracyToleranceClass")

    Returns:
        Default value from standards or None
    """
    product_lower = product_type.lower().strip()

    # === STRATEGY 1: Find matching product type ===
    defaults = None

    # First try: exact key match
    if product_lower in PRODUCT_TYPE_DEFAULTS:
        defaults = PRODUCT_TYPE_DEFAULTS[product_lower]
        logger.debug(f"[SCHEMA_FIELD] Exact match for product: {product_type}")

    # Second try: key is substring of product (e.g., "thermocouple" in "multipoint thermocouple assembly")
    if not defaults:
        for key, values in PRODUCT_TYPE_DEFAULTS.items():
            if key in product_lower:
                defaults = values
                logger.debug(f"[SCHEMA_FIELD] Substring match: '{key}' in '{product_lower}'")
                break

    # Third try: product is substring of key (e.g., "rtd" matches "rtd sensor")
    if not defaults:
        for key, values in PRODUCT_TYPE_DEFAULTS.items():
            if product_lower in key:
                defaults = values
                logger.debug(f"[SCHEMA_FIELD] Reverse substring match: '{product_lower}' in '{key}'")
                break

    # Fourth try: keyword overlap
    if not defaults:
        product_words = set(product_lower.split())
        for key, values in PRODUCT_TYPE_DEFAULTS.items():
            key_words = set(key.split())
            # If any significant word matches
            common = product_words & key_words
            if common and any(len(w) > 3 for w in common):  # Ignore short words
                defaults = values
                logger.debug(f"[SCHEMA_FIELD] Keyword match: {common} for '{product_type}'")
                break

    if not defaults:
        logger.debug(f"[SCHEMA_FIELD] No defaults found for product type: {product_type}")
        return None

    # === STRATEGY 2: Match field name ===
    field_normalized = _normalize_field_name(field_name)
    field_words = set(field_normalized.split())

    # Build normalized lookup
    defaults_normalized = {
        _normalize_field_name(k): v for k, v in defaults.items()
    }

    # Try 1: Exact match
    if field_normalized in defaults_normalized:
        logger.debug(f"[SCHEMA_FIELD] Exact field match: {field_name}")
        return defaults_normalized[field_normalized]

    # Try 2: Contains match (either direction)
    for default_norm, value in defaults_normalized.items():
        if field_normalized in default_norm or default_norm in field_normalized:
            logger.debug(f"[SCHEMA_FIELD] Contains match: '{field_name}' ~ '{default_norm}'")
            return value

    # Try 3: Keyword overlap with relaxed threshold
    best_match = None
    best_score = 0
    best_key = None

    for default_norm, value in defaults_normalized.items():
        default_words = set(default_norm.split())
        common = field_words & default_words

        if common:
            # Score: common words / total unique words (Jaccard-like)
            score = len(common) / len(field_words | default_words)
            if score > best_score:
                best_score = score
                best_match = value
                best_key = default_norm

    # Lowered threshold from 0.5 to 0.25 for better matching
    if best_score >= 0.25:
        logger.debug(f"[SCHEMA_FIELD] Fuzzy match: '{field_name}' -> '{best_key}' (score={best_score:.2f})")
        return best_match

    logger.debug(f"[SCHEMA_FIELD] No field match for: {field_name} (normalized: {field_normalized})")
    return None


def extract_schema_field_values_from_standards(
    product_type: str,
    schema: Dict[str, Any],
    standards_rag_results: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Extract values for all schema fields from standards specifications.

    This function:
    1. Iterates through all schema fields
    2. Queries Standards RAG for relevant information (if available)
    3. Uses default technical values from standards for each field

    Args:
        product_type: The product type being processed
        schema: The schema with fields to populate
        standards_rag_results: Pre-fetched Standards RAG results (optional)

    Returns:
        Schema with populated field values
    """
    logger.info(f"[SCHEMA_FIELD_EXTRACTOR] Extracting field values for: {product_type}")

    if not schema:
        return schema

    # Deep copy the schema
    populated_schema = json.loads(json.dumps(schema))

    fields_populated = 0
    fields_total = 0

    def populate_section(section: Dict[str, Any], section_path: str = ""):
        """Recursively populate a section of the schema."""
        nonlocal fields_populated, fields_total

        for field_key, field_value in list(section.items()):
            if field_key.startswith("_"):
                continue

            current_path = f"{section_path}.{field_key}" if section_path else field_key

            if isinstance(field_value, dict):
                # Check if this is a nested section or a field with metadata
                if "value" in field_value or "suggested_values" in field_value:
                    # This is a field with metadata
                    fields_total += 1
                    if not field_value.get("value") or field_value.get("value") == "":
                        default_value = get_default_value_for_field(product_type, field_key)
                        if default_value:
                            # Extract standards referenced from the value string
                            standards_refs = extract_standards_from_value(default_value)

                            field_value["value"] = default_value
                            field_value["source"] = "standards_specifications"
                            field_value["confidence"] = 0.9  # High confidence for standards-based defaults
                            field_value["standards_referenced"] = standards_refs if standards_refs else []

                            fields_populated += 1
                            logger.debug(f"[SCHEMA_FIELD_EXTRACTOR] Populated {current_path} with standards: {standards_refs}")
                else:
                    # Recurse into nested section
                    populate_section(field_value, current_path)
            elif isinstance(field_value, str):
                fields_total += 1
                # Empty string or "Not specified"
                if not field_value.strip() or field_value.lower() in ["not specified", ""]:
                    default_value = get_default_value_for_field(product_type, field_key)
                    if default_value:
                        # Extract standards referenced from the value string
                        standards_refs = extract_standards_from_value(default_value)

                        # Convert string field to full metadata dict
                        section[field_key] = {
                            "value": default_value,
                            "source": "standards_specifications",
                            "confidence": 0.9,  # High confidence for standards-based defaults
                            "standards_referenced": standards_refs if standards_refs else []
                        }

                        fields_populated += 1
                        logger.debug(f"[SCHEMA_FIELD_EXTRACTOR] Populated {current_path} = {default_value[:50]}... with standards: {standards_refs}")

    # Process each section of the schema
    sections_to_process = [
        "mandatory_requirements", "mandatory",
        "optional_requirements", "optional",
        "Compliance", "Electrical", "Mechanical", "Performance",
        "Environmental", "Features", "Integration", "MechanicalOptions",
        "ServiceAndSupport", "Certifications"
    ]

    for section_name in sections_to_process:
        if section_name in populated_schema:
            populate_section(populated_schema[section_name], section_name)

    # Also process root-level fields
    populate_section(populated_schema, "root")

    # Add population metadata
    populated_schema["_schema_field_extraction"] = {
        "product_type": product_type,
        "fields_total": fields_total,
        "fields_populated": fields_populated,
        "source": "standards_specifications",
        "method": "default_values_from_standards"
    }

    logger.info(
        f"[SCHEMA_FIELD_EXTRACTOR] Populated {fields_populated}/{fields_total} fields "
        f"for {product_type}"
    )

    return populated_schema


def query_standards_for_field(
    product_type: str,
    field_name: str,
    field_context: str = ""
) -> Optional[str]:
    """
    Query Standards RAG for a specific field value.

    Args:
        product_type: Product type being processed
        field_name: The schema field name
        field_context: Additional context about the field

    Returns:
        Extracted value or None
    """
    try:
        from agentic.standards_rag.standards_rag_workflow import run_standards_rag_workflow

        query = f"What is the standard {field_name} specification for {product_type}?"
        if field_context:
            query += f" Context: {field_context}"

        result = run_standards_rag_workflow(
            question=query,
            top_k=3
        )

        if result.get("status") == "success":
            answer = result.get("final_response", {}).get("answer", "")
            if answer and len(answer) > 10:
                return answer

        return None

    except Exception as e:
        logger.warning(f"[SCHEMA_FIELD_EXTRACTOR] RAG query failed for {field_name}: {e}")
        return None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "extract_schema_field_values_from_standards",
    "extract_standards_from_value",
    "get_default_value_for_field",
    "query_standards_for_field",
    "THERMOCOUPLE_DEFAULTS",
    "PRESSURE_TRANSMITTER_DEFAULTS",
    "TEMPERATURE_SENSOR_DEFAULTS",
    "CONNECTOR_DEFAULTS",
    "JUNCTION_BOX_DEFAULTS",
    "MOUNTING_BRACKET_DEFAULTS",
    "FLOW_METER_DEFAULTS",
    "LEVEL_TRANSMITTER_DEFAULTS",
    "CONTROL_VALVE_DEFAULTS",
    "ANALYZER_DEFAULTS",
    "SIGNAL_CONDITIONER_DEFAULTS",
    "CALIBRATOR_DEFAULTS",
]
