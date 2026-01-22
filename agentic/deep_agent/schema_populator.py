"""
Deep Agent Schema Populator
============================

This module integrates Deep Agent with schema population to fill all schema fields
with technical specifications extracted from standards documents.

Architecture:
1. Schema Section Segregation: Analyze schema fields and group by sections
2. Standards Document Mapping: Map standards documents to relevant schema sections
3. Parallel Value Extraction: Use threads to fetch values from standards per section
4. Result Aggregation: Build key-value pairs for each schema field

Flow:
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DEEP AGENT SCHEMA POPULATOR                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌────────────────────────────────────────────────┐ │
│  │   Input Schema  │───▶│  PHASE 1: Schema Section Segregation          │ │
│  │  (from PPI/DB)  │    │  • Extract all fields from schema              │ │
│  └─────────────────┘    │  • Group fields by section type                │ │
│                         │  • Store in Deep Agent Memory                  │ │
│                         └────────────────────────────────────────────────┘ │
│                                          │                                  │
│                                          ▼                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  PHASE 2: Standards Document Analysis                               │   │
│  │  • Load all 12 DOCX standards documents                            │   │
│  │  • Parse sections from each document                               │   │
│  │  • Map document sections to schema sections                        │   │
│  │  • Store mappings in Deep Agent Memory                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                          │                                  │
│                                          ▼                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  PHASE 3: Parallel Section Processing (ThreadPoolExecutor)         │   │
│  │                                                                     │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │   │
│  │  │Performance│ │Electrical│ │Mechanical│ │Environmt│ │Compliance│  │   │
│  │  │  Thread  │ │  Thread  │ │  Thread  │ │  Thread │ │  Thread  │  │   │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬────┘ └────┬─────┘  │   │
│  │       │            │            │            │           │         │   │
│  │       ▼            ▼            ▼            ▼           ▼         │   │
│  │  ┌──────────────────────────────────────────────────────────────┐  │   │
│  │  │  For each field in section:                                  │  │   │
│  │  │  • Query relevant standards documents                        │  │   │
│  │  │  • Extract technical specification value                     │  │   │
│  │  │  • Record source document reference                          │  │   │
│  │  │  • Store {field_name: value, source: document}              │  │   │
│  │  └──────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                          │                                  │
│                                          ▼                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  PHASE 4: Result Aggregation                                        │   │
│  │  • Merge all thread results                                        │   │
│  │  • Build populated schema with sections                            │   │
│  │  • Add source tracing metadata                                     │   │
│  │  • Format for UI display                                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                          │                                  │
│                                          ▼                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  OUTPUT: Populated Schema with Section-Based Organization          │   │
│  │  {                                                                  │   │
│  │    "Performance": {"accuracy": {"value": "±0.1%", "source": ...}}, │   │
│  │    "Electrical": {"outputSignal": {"value": "4-20mA", "source":}}, │   │
│  │    "Mechanical": {"processConnection": {"value": "1/2 NPT", ...}}, │   │
│  │    ...                                                              │   │
│  │    "_metadata": {sections_populated, fields_filled, sources_used}  │   │
│  │  }                                                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
"""

import logging
import json
import time
import threading
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# VALUE SANITIZATION
# =============================================================================

def sanitize_extracted_value(value: str) -> Optional[str]:
    """
    Sanitize LLM-extracted values by filtering invalid patterns and normalizing.

    This function uses the ValueNormalizer to:
    1. Filter out 'I don't have specific information' type messages
    2. Remove descriptive/sentence-like content
    3. Strip leading/trailing phrases
    4. Validate the value contains technical indicators
    5. Returns None for invalid values (will display as 'Not available' in UI)

    Args:
        value: Raw extracted value from LLM

    Returns:
        Cleaned value or None if invalid
    """
    if not value:
        return None

    value = str(value).strip()

    # Use the centralized ValueNormalizer for consistent sanitization
    try:
        from .value_normalizer import ValueNormalizer
        normalizer = ValueNormalizer()

        result = normalizer.extract_and_validate(value)

        if result["is_valid"]:
            return result["normalized_value"]
        else:
            logger.debug(f"[SANITIZE] Value failed validation: {value[:50]}...")
            return None

    except ImportError:
        # Fallback to basic sanitization if normalizer not available
        logger.warning("[SANITIZE] ValueNormalizer not available, using basic sanitization")
        return _basic_sanitize(value)


def _basic_sanitize(value: str) -> Optional[str]:
    """
    Basic sanitization fallback when ValueNormalizer is not available.

    Args:
        value: Raw extracted value

    Returns:
        Cleaned value or None if invalid
    """
    if not value:
        return None

    value = str(value).strip()

    # Invalid patterns - if found, return None (will show "Not available")
    INVALID_PATTERNS = [
        "i don't have specific information",
        "don't have specific information",
        "not in the available standards",
        "not explicitly specified",
        "not explicitly mentioned",
        "no specific information",
        "information is not available",
        "not found in the documents",
        "is not explicitly",
        "documented on the data sheet",
        "not available in the",
        "cannot determine",
        "not provided in",
        "refer to manufacturer",
        "consult the datasheet",
        "varies by model",
        "varies by application",
        # Additional patterns for descriptions
        "typically", "usually", "generally",
        "should be", "must be", "can be",
        "depends on", "based on", "according to",
        "the value is", "the accuracy is",
        "for applications", "in order to",
    ]

    value_lower = value.lower()
    for pattern in INVALID_PATTERNS:
        if pattern in value_lower:
            logger.debug(f"[SANITIZE] Filtered invalid pattern '{pattern}' in value")
            return None

    # Remove [Source: filename.docx] citations
    value = re.sub(r'\[Source:\s*[^\]]+\]', '', value).strip()

    # Remove leading ** prefix (often used in LLM responses)
    if value.startswith("**"):
        value = value[2:].strip()

    # Remove trailing ** suffix
    if value.endswith("**"):
        value = value[:-2].strip()

    # After cleanup, check if value is still meaningful
    if not value or value.lower() in ["", "n/a", "none", "not specified", "null", "-"]:
        return None

    # Filter values that are too short to be meaningful (single char or just punctuation)
    if len(value) < 2 or value in ["*", "**", "--", "..."]:
        return None

    # Reject sentence-like values (too long and contains multiple words)
    if len(value) > 100 and value.count(' ') > 10:
        logger.debug(f"[SANITIZE] Rejected sentence-like value: {value[:50]}...")
        return None

    return value


# =============================================================================
# SECTION DEFINITIONS
# =============================================================================

# Standard schema sections and their related keywords
SCHEMA_SECTIONS = {
    "Performance": [
        "accuracy", "repeatability", "response", "stability", "resolution",
        "linearity", "hysteresis", "drift", "range", "span", "turndown",
        "measurement", "sensor", "sensitivity"
    ],
    "Electrical": [
        "output", "signal", "power", "supply", "voltage", "current", "loop",
        "communication", "protocol", "hart", "fieldbus", "profibus", "modbus",
        "insulation", "resistance", "wiring", "wire", "cable", "terminal"
    ],
    "Mechanical": [
        "process", "connection", "probe", "diameter", "length", "sheath",
        "material", "wetted", "housing", "mounting", "flange", "thread",
        "insertion", "immersion", "thermowell", "head", "enclosure"
    ],
    "Environmental": [
        "temperature", "ambient", "storage", "humidity", "pressure", "vibration",
        "ip", "ingress", "protection", "nema", "weather", "corrosion"
    ],
    "Compliance": [
        "certification", "standard", "atex", "iecex", "sil", "safety",
        "hazardous", "zone", "class", "division", "ce", "ul", "fm", "csa",
        "ped", "marine", "dnv", "approval"
    ],
    "Features": [
        "display", "diagnostic", "calibration", "configuration", "logging",
        "data", "alarm", "option", "function", "transmitter", "indicator"
    ],
    "Integration": [
        "fieldbus", "wireless", "device", "integration", "asset", "ams",
        "eddl", "fdt", "dtm", "pactware"
    ],
    "ServiceAndSupport": [
        "warranty", "calibration", "certificate", "maintenance", "service",
        "traceability", "spare", "support"
    ],
    # General section - maps to Performance for defaults lookup
    "General": [
        "accuracy", "certification", "ip", "rating", "material", "output",
        "range", "product", "type"
    ],
    # Additional sections from specs_bug.md
    "FunctionalParameters": [
        "max", "process", "pressure", "temperature", "medium", "compatibility"
    ],
    "InstallationRequirements": [
        "mounting", "type", "installation"
    ],
    "MechanicalIntegration": [
        "bore", "damping", "design", "filling", "insertion", "connection",
        "tip", "wetted", "parts"
    ],
    "DocumentationAndServices": [
        "calibration", "pre", "assembly", "tagging", "traceability", "warranty"
    ]
}

# =============================================================================
# SECTION NAME NORMALIZATION
# =============================================================================

# Map variant section names to their canonical names
SECTION_NAME_MAPPINGS = {
    # Space-separated to camelCase
    "Service And Support": "ServiceAndSupport",
    "service and support": "ServiceAndSupport",
    "Mechanical Options": "MechanicalOptions",
    "mechanical options": "MechanicalOptions",
    "Compliance And Safety": "Compliance",
    "compliance and safety": "Compliance",
    "Environmental And Safety": "Environmental",
    "environmental and safety": "Environmental",
    "Documentation And Services": "DocumentationAndServices",
    "documentation and services": "DocumentationAndServices",
    "Functional Parameters": "FunctionalParameters",
    "functional parameters": "FunctionalParameters",
    "Installation Requirements": "InstallationRequirements",
    "installation requirements": "InstallationRequirements",
    "Mechanical Integration": "MechanicalIntegration",
    "mechanical integration": "MechanicalIntegration",
    "Advanced Mechanical Options": "MechanicalOptions",
    "advanced mechanical options": "MechanicalOptions",
    "Specific Features": "Features",
    "specific features": "Features",
    # General maps to Performance for defaults lookup (common fields)
    "General": "General",
    "general": "General",
}


def normalize_section_name(name: str) -> str:
    """
    Normalize section names for consistent matching.

    Handles various naming conventions:
    - "Service And Support" -> "ServiceAndSupport"
    - "Mechanical Options" -> "MechanicalOptions"
    - "General" -> "General" (kept as-is, but can look up Performance defaults)

    Args:
        name: Original section name

    Returns:
        Normalized section name
    """
    if not name:
        return name

    # Check direct mapping first
    if name in SECTION_NAME_MAPPINGS:
        return SECTION_NAME_MAPPINGS[name]

    # Try lowercase version
    lower_name = name.lower()
    if lower_name in SECTION_NAME_MAPPINGS:
        return SECTION_NAME_MAPPINGS[lower_name]

    # If no mapping found, return original
    return name


def get_section_for_defaults_lookup(section_name: str) -> str:
    """
    Get the section name to use for defaults lookup.

    Some sections like 'General' should look up defaults from 'Performance'.

    Args:
        section_name: The section name

    Returns:
        Section name to use for defaults lookup
    """
    # Normalize first
    normalized = normalize_section_name(section_name)

    # General section uses Performance defaults for common fields
    # like Accuracy, Range, etc.
    if normalized == "General":
        return "Performance"

    return normalized

# Map section names to relevant standards document categories
SECTION_TO_STANDARDS_DOCS = {
    "Performance": [
        "instrumentation_pressure_standards.docx",
        "instrumentation_temperature_standards.docx",
        "instrumentation_flow_standards.docx",
        "instrumentation_level_standards.docx",
        "instrumentation_analytical_standards.docx"
    ],
    "Electrical": [
        "instrumentation_comm_signal_standards.docx",
        "instrumentation_control_systems_standards.docx"
    ],
    "Mechanical": [
        "instrumentation_valves_actuators_standards.docx",
        "instrumentation_accessories_calibration_standards.docx"
    ],
    "Environmental": [
        "instrumentation_condition_monitoring_standards.docx",
        "instrumentation_safety_standards.docx"
    ],
    "Compliance": [
        "instrumentation_safety_standards.docx",
        "instrumentation_control_systems_standards.docx"
    ],
    "Features": [
        "instrumentation_control_systems_standards.docx",
        "instrumentation_comm_signal_standards.docx"
    ],
    "Integration": [
        "instrumentation_comm_signal_standards.docx",
        "instrumentation_control_systems_standards.docx"
    ],
    "ServiceAndSupport": [
        "instrumentation_calibration_maintenance_standards.docx",
        "instrumentation_accessories_calibration_standards.docx"
    ]
}


# =============================================================================
# DEEP AGENT SCHEMA MEMORY
# =============================================================================

@dataclass
class SchemaPopulatorMemory:
    """Memory structure for schema population process."""

    # Schema analysis
    schema_sections: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    total_fields: int = 0

    # Standards document analysis
    standards_analysis: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    section_to_docs_mapping: Dict[str, List[str]] = field(default_factory=dict)

    # Thread results
    section_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Metadata
    product_type: str = ""
    start_time: float = 0.0
    fields_populated: int = 0
    sources_used: List[str] = field(default_factory=list)

    # Thread safety
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def store_section_result(self, section_name: str, result: Dict[str, Any]):
        """Thread-safe storage of section results."""
        with self._lock:
            self.section_results[section_name] = result
            self.sources_used.extend(result.get("sources_used", []))
            self.fields_populated += result.get("fields_populated", 0)


# =============================================================================
# PHASE 1: SCHEMA SECTION SEGREGATION
# =============================================================================

def segregate_schema_by_sections(schema: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Analyze schema fields and segregate them by section type.

    Args:
        schema: The input schema to analyze

    Returns:
        Dictionary mapping section names to lists of fields
    """
    logger.info("[SchemaPopulator] Phase 1: Segregating schema by sections")

    sections: Dict[str, List[Dict[str, Any]]] = {
        section: [] for section in SCHEMA_SECTIONS.keys()
    }
    sections["Other"] = []  # For fields that don't match any section

    def classify_field(field_name: str, field_value: Any, path: str) -> str:
        """Classify a field into a section based on its name."""
        field_lower = field_name.lower().replace("_", " ").replace("-", " ")

        for section_name, keywords in SCHEMA_SECTIONS.items():
            for keyword in keywords:
                if keyword in field_lower:
                    return section_name

        return "Other"

    # Track processed field paths to avoid duplicates
    processed_paths = set()

    def extract_fields_dedup(obj: Any, path: str = ""):
        """Recursively extract fields from schema, avoiding duplicates."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                # Skip metadata keys
                if key.startswith("_"):
                    continue

                current_path = f"{path}.{key}" if path else key

                # Skip if already processed
                if current_path in processed_paths:
                    continue

                if isinstance(value, dict):
                    # Check if this is a leaf field or a nested section
                    if any(k in value for k in ["value", "description", "suggested_values"]):
                        # This is a field with metadata
                        processed_paths.add(current_path)
                        section = classify_field(key, value, current_path)
                        sections[section].append({
                            "name": key,
                            "path": current_path,
                            "current_value": value.get("value", ""),
                            "description": value.get("description", ""),
                            "suggested_values": value.get("suggested_values", [])
                        })
                    else:
                        # Recurse into nested section
                        extract_fields_dedup(value, current_path)
                elif isinstance(value, str):
                    # Simple string field
                    processed_paths.add(current_path)
                    section = classify_field(key, value, current_path)
                    sections[section].append({
                        "name": key,
                        "path": current_path,
                        "current_value": value,
                        "description": "",
                        "suggested_values": []
                    })

    # Process all main sections of the schema
    # Include both camelCase and space-separated versions
    main_sections = [
        "mandatory", "mandatory_requirements",
        "optional", "optional_requirements",
        "Performance", "Electrical", "Mechanical",
        "Environmental", "Compliance", "Features",
        "Integration", "ServiceAndSupport", "MechanicalOptions",
        # General section (common in inferred specs)
        "General",
        # Space-separated variants
        "Service And Support", "Mechanical Options", "Compliance And Safety",
        "Environmental And Safety", "Documentation And Services",
        # Additional sections from specs_bug.md
        "Functional Parameters", "Installation Requirements",
        "Mechanical Integration", "Advanced Mechanical Options",
        "Specific Features", "Certifications"
    ]

    for section_name in main_sections:
        if section_name in schema:
            # Normalize section name for consistent classification
            normalized_section = normalize_section_name(section_name)
            extract_fields_dedup(schema[section_name], normalized_section)

    # Also extract from root level (skip already processed)
    extract_fields_dedup(schema)

    # Log results
    total_fields = sum(len(fields) for fields in sections.values())
    logger.info(f"[SchemaPopulator] Segregated {total_fields} fields into sections:")
    for section, fields in sections.items():
        if fields:
            logger.info(f"  {section}: {len(fields)} fields")

    return sections


# =============================================================================
# PHASE 2: STANDARDS DOCUMENT ANALYSIS
# =============================================================================

def analyze_standards_for_sections(
    product_type: str,
    schema_sections: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze standards documents and map them to schema sections.

    Args:
        product_type: The product type being processed
        schema_sections: Segregated schema sections

    Returns:
        Dictionary mapping sections to relevant standards content
    """
    logger.info("[SchemaPopulator] Phase 2: Analyzing standards documents")

    try:
        from .document_loader import analyze_all_documents

        # Get all document analyses
        all_analyses = analyze_all_documents()

        # Map sections to relevant document content
        section_standards: Dict[str, Dict[str, Any]] = {}

        for section_name in schema_sections.keys():
            if not schema_sections[section_name]:
                continue

            # Normalize section name for consistent lookup
            normalized_section = normalize_section_name(section_name)
            lookup_section = get_section_for_defaults_lookup(normalized_section)

            # Get relevant document names for this section
            relevant_docs = SECTION_TO_STANDARDS_DOCS.get(lookup_section, [])
            # Also try original section name if no match
            if not relevant_docs:
                relevant_docs = SECTION_TO_STANDARDS_DOCS.get(section_name, [])

            # Collect content from relevant documents
            section_content = {
                "documents": [],
                "standards_codes": [],
                "certifications": [],
                "content_snippets": []
            }

            for doc_name, analysis in all_analyses.items():
                # Check if this document is relevant to the section
                is_relevant = any(
                    rel_doc.lower() in doc_name.lower()
                    for rel_doc in relevant_docs
                )

                # Also check if the document is relevant to the product type
                product_keywords = product_type.lower().split()
                doc_keywords = doc_name.lower().replace("_", " ").replace(".docx", "")
                is_product_relevant = any(
                    kw in doc_keywords for kw in product_keywords
                )

                if is_relevant or is_product_relevant:
                    section_content["documents"].append(doc_name)
                    section_content["standards_codes"].extend(
                        analysis.get("standard_codes", [])
                    )
                    section_content["certifications"].extend(
                        analysis.get("certifications", [])
                    )

                    # Extract relevant content snippets
                    for section_type, section_data in analysis.get("sections", {}).items():
                        section_content["content_snippets"].append({
                            "source": doc_name,
                            "section": section_type,
                            "content": section_data.get("content", "")[:2000]
                        })

            # Deduplicate
            section_content["standards_codes"] = list(set(section_content["standards_codes"]))
            section_content["certifications"] = list(set(section_content["certifications"]))

            section_standards[section_name] = section_content

            logger.info(
                f"[SchemaPopulator] Section '{section_name}': "
                f"{len(section_content['documents'])} docs, "
                f"{len(section_content['standards_codes'])} standards"
            )

        return section_standards

    except Exception as e:
        logger.error(f"[SchemaPopulator] Error analyzing standards: {e}")
        return {}


# =============================================================================
# PHASE 3: PARALLEL SECTION PROCESSING
# =============================================================================

def extract_field_value_from_standards(
    field_name: str,
    field_info: Dict[str, Any],
    product_type: str,
    standards_content: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Extract a value for a schema field from standards content.

    Uses a multi-strategy approach:
    1. First tries to get default value from schema_field_extractor
    2. Falls back to LLM extraction from standards documents

    Args:
        field_name: The field name to populate
        field_info: Field metadata
        product_type: The product type
        standards_content: Relevant standards content

    Returns:
        Dictionary with extracted value and source
    """
    try:
        # Strategy 1: Try to get default value from schema_field_extractor
        from .schema_field_extractor import get_default_value_for_field

        default_value = get_default_value_for_field(product_type, field_name)
        if default_value:
            sources = standards_content.get("documents", [])
            return {
                "value": default_value,
                "source": "standards_specifications",
                "standards_referenced": standards_content.get("standards_codes", [])[:3],
                "confidence": 0.9
            }

        # Strategy 2: Try LLM extraction if we have content
        context_parts = []
        for snippet in standards_content.get("content_snippets", [])[:5]:
            context_parts.append(
                f"[Source: {snippet['source']}]\n{snippet['content']}"
            )
        context = "\n\n".join(context_parts)

        if context:
            try:
                from llm_fallback import create_llm_with_fallback
                import os

                # Create extraction prompt
                prompt = f"""Extract the technical specification value for "{field_name}"
for a {product_type} based on these standards:

{context[:3000]}

Provide ONLY the value (e.g., "4-20mA", "0.1%", "IP66"). No explanation."""

                llm = create_llm_with_fallback(
                    model="gemini-2.5-flash",
                    temperature=0.1,
                    google_api_key=os.getenv("GOOGLE_API_KEY")
                )

                response = llm.invoke(prompt)
                # Sanitize the extracted value to filter invalid patterns
                extracted_value = sanitize_extracted_value(response.content)

                if extracted_value:
                    sources = standards_content.get("documents", [])
                    return {
                        "value": extracted_value,
                        "source": sources[0] if sources else "llm_extraction",
                        "standards_referenced": standards_content.get("standards_codes", [])[:3],
                        "confidence": 0.7
                    }
            except Exception as llm_err:
                logger.debug(f"[SchemaPopulator] LLM extraction failed for {field_name}: {llm_err}")

        # FALLBACK: Always try default values from schema_field_extractor
        try:
            from .schema_field_extractor import get_default_value_for_field
            default_val = get_default_value_for_field(product_type, field_name)
            if default_val:
                logger.info(f"[SchemaPopulator] Using default for {field_name}: {default_val[:50]}...")
                return {
                    "value": default_val,
                    "source": "standards_specifications_default",
                    "standards_referenced": [],
                    "confidence": 0.85
                }
        except Exception as fallback_err:
            logger.debug(f"[SchemaPopulator] Default fallback failed for {field_name}: {fallback_err}")

        return {"value": "", "source": None, "confidence": 0.0}

    except Exception as e:
        logger.debug(f"[SchemaPopulator] Field extraction error for {field_name}: {e}")
        # Last resort: try defaults even on error
        try:
            from .schema_field_extractor import get_default_value_for_field
            default_val = get_default_value_for_field(product_type, field_name)
            if default_val:
                return {
                    "value": default_val,
                    "source": "standards_specifications_default",
                    "standards_referenced": [],
                    "confidence": 0.80
                }
        except:
            pass
        return {"value": "", "source": None, "confidence": 0.0}


def process_section_thread(
    section_name: str,
    fields: List[Dict[str, Any]],
    product_type: str,
    standards_content: Dict[str, Any],
    memory: SchemaPopulatorMemory
) -> Dict[str, Any]:
    """
    Process all fields in a section (runs in thread).

    Args:
        section_name: Name of the section being processed
        fields: List of fields in this section
        product_type: Product type being processed
        standards_content: Relevant standards for this section
        memory: Shared memory object

    Returns:
        Dictionary with section results
    """
    logger.info(f"[SchemaPopulator] Thread started for section: {section_name}")

    section_result = {
        "section_name": section_name,
        "fields": {},
        "fields_total": len(fields),
        "fields_populated": 0,
        "sources_used": [],
        "status": "processing"
    }

    try:
        for field_info in fields:
            field_name = field_info["name"]

            # Skip if already has a value
            current_value = field_info.get("current_value", "")
            if current_value and current_value.lower() not in ["", "not specified"]:
                section_result["fields"][field_name] = {
                    "value": current_value,
                    "source": "existing",
                    "confidence": 1.0
                }
                continue

            # Extract value from standards
            extracted = extract_field_value_from_standards(
                field_name=field_name,
                field_info=field_info,
                product_type=product_type,
                standards_content=standards_content
            )

            section_result["fields"][field_name] = extracted

            if extracted.get("value"):
                section_result["fields_populated"] += 1
                if extracted.get("source"):
                    section_result["sources_used"].append(extracted["source"])

        section_result["status"] = "completed"
        section_result["sources_used"] = list(set(section_result["sources_used"]))

        # Store in memory
        memory.store_section_result(section_name, section_result)

        logger.info(
            f"[SchemaPopulator] Thread completed for {section_name}: "
            f"{section_result['fields_populated']}/{section_result['fields_total']} fields"
        )

    except Exception as e:
        section_result["status"] = "error"
        section_result["error"] = str(e)
        logger.error(f"[SchemaPopulator] Thread error for {section_name}: {e}")

    return section_result


def run_parallel_section_extraction(
    schema_sections: Dict[str, List[Dict[str, Any]]],
    section_standards: Dict[str, Dict[str, Any]],
    product_type: str,
    memory: SchemaPopulatorMemory,
    max_workers: int = 4
) -> Dict[str, Dict[str, Any]]:
    """
    Run parallel extraction for all sections.

    Args:
        schema_sections: Segregated schema sections
        section_standards: Standards content mapped to sections
        product_type: Product type being processed
        memory: Shared memory object
        max_workers: Maximum parallel threads

    Returns:
        Dictionary with all section results
    """
    logger.info(f"[SchemaPopulator] Phase 3: Parallel extraction with {max_workers} workers")

    all_results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        for section_name, fields in schema_sections.items():
            if not fields:
                continue

            standards_content = section_standards.get(section_name, {})

            future = executor.submit(
                process_section_thread,
                section_name=section_name,
                fields=fields,
                product_type=product_type,
                standards_content=standards_content,
                memory=memory
            )
            futures[future] = section_name

        # Collect results
        for future in as_completed(futures):
            section_name = futures[future]
            try:
                result = future.result(timeout=60)
                all_results[section_name] = result
            except Exception as e:
                logger.error(f"[SchemaPopulator] Future error for {section_name}: {e}")
                all_results[section_name] = {
                    "section_name": section_name,
                    "status": "error",
                    "error": str(e)
                }

    return all_results


# =============================================================================
# PHASE 4: RESULT AGGREGATION
# =============================================================================

def build_populated_schema(
    original_schema: Dict[str, Any],
    section_results: Dict[str, Dict[str, Any]],
    memory: SchemaPopulatorMemory
) -> Dict[str, Any]:
    """
    Build the final populated schema from section results.

    Args:
        original_schema: The original input schema
        section_results: Results from parallel section processing
        memory: Memory with aggregated data

    Returns:
        Fully populated schema with section-based organization
    """
    logger.info("[SchemaPopulator] Phase 4: Building populated schema")

    # Deep copy original schema
    populated_schema = json.loads(json.dumps(original_schema))

    # Track statistics
    total_populated = 0
    total_fields = 0

    def update_field_in_schema(schema_obj: Dict, field_name: str, field_data: Dict):
        """Recursively find and update a field in the schema."""
        nonlocal total_populated

        for key, value in list(schema_obj.items()):
            if key == field_name:
                new_value = field_data.get("value", "")
                if new_value:
                    # Update with rich metadata
                    schema_obj[key] = {
                        "value": new_value,
                        "source": field_data.get("source", "deep_agent"),
                        "standards_referenced": field_data.get("standards_referenced", []),
                        "confidence": field_data.get("confidence", 0.8)
                    }
                    total_populated += 1
                return True
            elif isinstance(value, dict):
                if update_field_in_schema(value, field_name, field_data):
                    return True
        return False

    # Apply section results to schema
    for section_name, section_data in section_results.items():
        if section_data.get("status") != "completed":
            continue

        for field_name, field_data in section_data.get("fields", {}).items():
            total_fields += 1
            raw_value = field_data.get("value")
            # Final sanitization check before storing
            sanitized_value = sanitize_extracted_value(raw_value) if raw_value else None
            if sanitized_value:
                # Create sanitized field data
                sanitized_field_data = {
                    "value": sanitized_value,
                    "source": field_data.get("source"),
                    "standards_referenced": field_data.get("standards_referenced", []),
                    "confidence": field_data.get("confidence", 0.8)
                }
                # Try to update in various schema locations
                updated = False
                for section_key in ["mandatory_requirements", "optional_requirements",
                                   "mandatory", "optional", section_name]:
                    if section_key in populated_schema:
                        if update_field_in_schema(populated_schema[section_key], field_name, sanitized_field_data):
                            updated = True
                            break

                if not updated:
                    # Update at root level
                    update_field_in_schema(populated_schema, field_name, sanitized_field_data)

    # Add section-based organization for UI display
    populated_schema["_deep_agent_sections"] = {}
    for section_name, section_data in section_results.items():
        if section_data.get("status") == "completed":
            populated_schema["_deep_agent_sections"][section_name] = {
                "fields": section_data.get("fields", {}),
                "fields_total": section_data.get("fields_total", 0),
                "fields_populated": section_data.get("fields_populated", 0),
                "sources_used": section_data.get("sources_used", [])
            }

    # Add metadata
    processing_time = time.time() - memory.start_time
    populated_schema["_deep_agent_population"] = {
        "product_type": memory.product_type,
        "total_fields": total_fields,
        "fields_populated": total_populated,
        "sections_processed": len([s for s in section_results.values() if s.get("status") == "completed"]),
        "sources_used": list(set(memory.sources_used)),
        "processing_time_ms": int(processing_time * 1000),
        "method": "deep_agent_schema_populator"
    }

    logger.info(
        f"[SchemaPopulator] Completed: {total_populated}/{total_fields} fields populated "
        f"in {processing_time:.2f}s"
    )

    return populated_schema


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def populate_schema_with_deep_agent(
    product_type: str,
    schema: Dict[str, Any],
    max_workers: int = 4
) -> Dict[str, Any]:
    """
    Main function to populate a schema using Deep Agent.

    This function:
    1. Segregates schema fields by sections
    2. Analyzes standards documents and maps to sections
    3. Uses parallel threads to extract values for each section
    4. Aggregates results into a fully populated schema

    Args:
        product_type: The product type being processed
        schema: The schema to populate
        max_workers: Maximum parallel threads (default: 4)

    Returns:
        Populated schema with section-based organization
    """
    logger.info(f"[SchemaPopulator] Starting Deep Agent schema population for: {product_type}")
    print(f"\n{'='*80}")
    print(f"[DEEP AGENT] SCHEMA POPULATOR")
    print(f"   Product Type: {product_type}")
    print(f"{'='*80}\n")

    # Initialize memory
    memory = SchemaPopulatorMemory()
    memory.product_type = product_type
    memory.start_time = time.time()

    try:
        # Phase 1: Segregate schema by sections
        print("[PHASE] Phase 1: Analyzing schema structure...")
        schema_sections = segregate_schema_by_sections(schema)
        memory.schema_sections = schema_sections
        memory.total_fields = sum(len(fields) for fields in schema_sections.values())

        # Phase 2: Analyze standards documents
        print("[PHASE] Phase 2: Analyzing standards documents...")
        section_standards = analyze_standards_for_sections(product_type, schema_sections)
        memory.section_to_docs_mapping = {
            section: content.get("documents", [])
            for section, content in section_standards.items()
        }

        # Phase 3: Parallel extraction
        print(f"[PHASE] Phase 3: Parallel extraction with {max_workers} threads...")
        section_results = run_parallel_section_extraction(
            schema_sections=schema_sections,
            section_standards=section_standards,
            product_type=product_type,
            memory=memory,
            max_workers=max_workers
        )

        # Phase 4: Build populated schema
        print("[PHASE] Phase 4: Building populated schema...")
        populated_schema = build_populated_schema(
            original_schema=schema,
            section_results=section_results,
            memory=memory
        )

        # Summary
        pop_info = populated_schema.get("_deep_agent_population", {})
        print(f"\n{'='*80}")
        print(f"[OK] DEEP AGENT SCHEMA POPULATION COMPLETED")
        print(f"   Fields Populated: {pop_info.get('fields_populated', 0)}/{pop_info.get('total_fields', 0)}")
        print(f"   Sections Processed: {pop_info.get('sections_processed', 0)}")
        print(f"   Sources Used: {len(pop_info.get('sources_used', []))}")
        print(f"   Processing Time: {pop_info.get('processing_time_ms', 0)}ms")
        print(f"{'='*80}\n")

        return populated_schema

    except Exception as e:
        logger.error(f"[SchemaPopulator] Error: {e}", exc_info=True)
        print(f"\n[ERROR] ERROR: {e}\n")

        # Return original schema with error metadata
        schema["_deep_agent_population"] = {
            "product_type": product_type,
            "status": "error",
            "error": str(e),
            "processing_time_ms": int((time.time() - memory.start_time) * 1000)
        }
        return schema


# =============================================================================
# INTEGRATION HELPER
# =============================================================================

def integrate_deep_agent_with_validation(
    product_type: str,
    schema: Dict[str, Any],
    enable_deep_agent: bool = True
) -> Dict[str, Any]:
    """
    Helper function to integrate Deep Agent population with validation workflow.

    This can be called from ValidationTool to enrich schemas.

    Args:
        product_type: Product type
        schema: Schema from Azure/PPI
        enable_deep_agent: Whether to use Deep Agent (default: True)

    Returns:
        Enriched schema
    """
    if not enable_deep_agent:
        return schema

    try:
        return populate_schema_with_deep_agent(product_type, schema)
    except Exception as e:
        logger.warning(f"[Integration] Deep Agent population failed: {e}")
        return schema


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "populate_schema_with_deep_agent",
    "integrate_deep_agent_with_validation",
    "segregate_schema_by_sections",
    "SchemaPopulatorMemory",
    "SCHEMA_SECTIONS",
    "SECTION_TO_STANDARDS_DOCS",
    # New normalization functions
    "normalize_section_name",
    "get_section_for_defaults_lookup",
    "SECTION_NAME_MAPPINGS",
]


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("\n" + "="*80)
    print("DEEP AGENT SCHEMA POPULATOR - TEST")
    print("="*80)

    # Test schema
    test_schema = {
        "mandatory_requirements": {
            "Performance": {
                "accuracy": "",
                "measurementRange": "",
                "responseTime": ""
            },
            "Electrical": {
                "outputSignal": "",
                "powerSupply": ""
            }
        },
        "optional_requirements": {
            "Mechanical": {
                "processConnection": "",
                "sheathMaterial": ""
            },
            "Environmental": {
                "ambientTemperatureRange": "",
                "ingressProtection": ""
            },
            "Compliance": {
                "hazardousAreaRating": "",
                "safetyIntegrityLevel": ""
            }
        }
    }

    # Run population
    result = populate_schema_with_deep_agent(
        product_type="pressure transmitter",
        schema=test_schema,
        max_workers=3
    )

    # Print results
    print("\n" + "="*80)
    print("POPULATED SCHEMA SECTIONS:")
    print("="*80)

    sections = result.get("_deep_agent_sections", {})
    for section_name, section_data in sections.items():
        print(f"\n[{section_name}]")
        for field_name, field_data in section_data.get("fields", {}).items():
            value = field_data.get("value", "N/A")[:50]
            source = field_data.get("source", "N/A")
            print(f"  {field_name}: {value}... (source: {source})")

    print("\n" + "="*80)
    print("TEST COMPLETED")
    print("="*80)
