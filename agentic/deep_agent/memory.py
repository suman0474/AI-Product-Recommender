# agentic/deep_agent_memory.py
# =============================================================================
# DEEP AGENT MEMORY STRUCTURES
# =============================================================================
#
# Memory Bank for Deep Agent - Stores all analyzed data for quick retrieval.
# Memory is populated ONCE during Planner phase, then retrieved during thread execution.
#
# =============================================================================

import logging
from typing import Dict, Any, List, Optional, TypedDict
from dataclasses import dataclass, field
from datetime import datetime
import threading

logger = logging.getLogger(__name__)


# =============================================================================
# SECTION ANALYSIS STRUCTURES
# =============================================================================

class SectionAnalysis(TypedDict, total=False):
    """Analysis of a single section within a document"""
    section_name: str
    section_type: str  # "applicable_standards", "equipment_types", etc.
    raw_content: str
    extracted_data: Dict[str, Any]
    extracted_codes: List[str]  # Standard codes like IEC 61508
    extracted_specs: Dict[str, str]  # Specification key-value pairs
    key_points: List[str]
    summary: str


class StandardsDocumentAnalysis(TypedDict, total=False):
    """Analysis of a single standards document"""
    filename: str
    document_type: str  # "pressure", "temperature", "flow", etc.
    full_content: str

    # Section Analyses
    sections: Dict[str, SectionAnalysis]

    # Quick Access Extractions (aggregated from sections)
    all_standard_codes: List[str]
    all_certifications: List[str]
    all_equipment_types: List[str]
    all_specifications: Dict[str, str]
    all_guidelines: List[str]

    # Metadata
    total_sections: int
    analysis_timestamp: str


class UserContextMemory(TypedDict, total=False):
    """Memory of user context extracted from input"""
    raw_input: str
    domain: str  # "Oil & Gas", "Chemical", etc.
    process_type: str  # "Distillation", "Refining", etc.
    industry: str
    safety_requirements: Dict[str, Any]
    environmental_conditions: Dict[str, Any]
    constraint_keywords: List[str]  # NEW: Keywords that drive standards lookup (e.g., "hydrogen", "vibration")
    key_parameters: Dict[str, Any]  # temperature, pressure, flow ranges
    extracted_keywords: List[str]


class IdentifiedItemMemory(TypedDict, total=False):
    """Memory of a single identified instrument or accessory"""
    item_id: str
    item_type: str  # "instrument" or "accessory"
    product_type: str  # "Pressure Transmitter", "Thermowell", etc.
    category: str
    quantity: int
    user_specifications: Dict[str, Any]
    relevant_documents: List[str]  # Document keys that apply
    related_instrument: Optional[str]  # For accessories


class ThreadInfo(TypedDict):
    """Information about a sub-thread"""
    thread_id: str
    product_type: str
    item_type: str  # "instrument" or "accessory"
    status: str  # "pending", "in_progress", "completed", "error"
    relevant_docs: List[str]
    result: Optional[Dict[str, Any]]


class ExecutionPlan(TypedDict):
    """Execution plan created by Planner"""
    main_threads: Dict[str, Dict[str, Any]]
    total_threads: int
    created_at: str


# =============================================================================
# PARALLEL 3-SOURCE ENRICHMENT STRUCTURES
# =============================================================================

class SpecificationSource(TypedDict, total=False):
    """Metadata for a single specification value from any source"""
    value: Any                          # The actual specification value
    source: str                         # "user_specified" | "llm_generated" | "standards"
    confidence: float                   # Confidence score 0.0-1.0
    standard_reference: Optional[str]   # Reference like "IEC 61511" if from standards
    timestamp: str                      # When this spec was extracted


class ParallelEnrichmentResult(TypedDict, total=False):
    """Result from parallel 3-source specification enrichment for a single item"""
    item_id: str
    item_name: str
    item_type: str                      # "instrument" or "accessory"
    
    # Three parallel source results
    user_specified_specs: Dict[str, Any]    # MANDATORY - never override these
    llm_generated_specs: Dict[str, Any]     # LLM-generated specs for product type
    standards_specs: Dict[str, Any]         # Extracted from standards documents
    
    # Final merged result with full metadata
    merged_specs: Dict[str, SpecificationSource]
    
    # Enrichment metadata
    enrichment_metadata: Dict[str, Any]     # Processing stats, sources used, etc.


# =============================================================================
# DEEP AGENT MEMORY CLASS
# =============================================================================

@dataclass
class DeepAgentMemory:
    """
    Memory Bank for Deep Agent.

    Stores all analyzed data for quick retrieval during thread execution.
    Thread-safe for concurrent access.
    """

    # Standards Analysis Memory - Pre-analyzed content from all documents
    standards_analysis: Dict[str, StandardsDocumentAnalysis] = field(default_factory=dict)

    # User Context Memory - Parsed from user input
    user_context: Optional[UserContextMemory] = None

    # Product-Type Index - Maps product types to relevant standards docs
    product_type_index: Dict[str, List[str]] = field(default_factory=dict)

    # Section Index - Cross-document section lookup
    section_index: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Identified Items Memory
    identified_instruments: List[IdentifiedItemMemory] = field(default_factory=list)
    identified_accessories: List[IdentifiedItemMemory] = field(default_factory=list)

    # Execution Plan
    execution_plan: Optional[ExecutionPlan] = None

    # Thread Results
    thread_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Parallel 3-Source Enrichment Results
    parallel_enrichment_results: Dict[str, ParallelEnrichmentResult] = field(default_factory=dict)

    # Thread lock for concurrent access
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    documents_loaded: int = 0
    sections_analyzed: int = 0

    def store_standards_analysis(self, filename: str, analysis: StandardsDocumentAnalysis) -> None:
        """Store analysis of a standards document"""
        with self._lock:
            self.standards_analysis[filename] = analysis
            self.documents_loaded += 1
            self.sections_analyzed += analysis.get("total_sections", 0)
            logger.info(f"[Memory] Stored analysis for: {filename}")

    def get_standards_analysis(self, filename: str) -> Optional[StandardsDocumentAnalysis]:
        """Retrieve analysis of a standards document"""
        return self.standards_analysis.get(filename)

    def store_user_context(self, context: UserContextMemory) -> None:
        """Store user context"""
        with self._lock:
            self.user_context = context
            logger.info(f"[Memory] Stored user context: domain={context.get('domain')}")

    def get_user_context(self) -> Optional[UserContextMemory]:
        """Retrieve user context"""
        return self.user_context

    def store_product_type_mapping(self, product_type: str, relevant_docs: List[str]) -> None:
        """Store mapping of product type to relevant documents"""
        with self._lock:
            self.product_type_index[product_type] = relevant_docs

    def get_relevant_documents(self, product_type: str) -> List[str]:
        """Get relevant documents for a product type"""
        return self.product_type_index.get(product_type, [])

    def store_section_index(self, section_type: str, data: Dict[str, Any]) -> None:
        """Store cross-document section index"""
        with self._lock:
            self.section_index[section_type] = data

    def get_section_across_documents(self, section_type: str) -> Dict[str, Any]:
        """Get section data across all documents"""
        return self.section_index.get(section_type, {})

    def store_identified_item(self, item: IdentifiedItemMemory) -> None:
        """Store an identified instrument or accessory"""
        with self._lock:
            if item.get("item_type") == "instrument":
                self.identified_instruments.append(item)
            else:
                self.identified_accessories.append(item)

    def store_thread_result(self, thread_id: str, result: Dict[str, Any]) -> None:
        """Store result from a thread execution"""
        with self._lock:
            self.thread_results[thread_id] = result
            logger.info(f"[Memory] Stored thread result: {thread_id}")

    def get_thread_result(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve result from a thread"""
        return self.thread_results.get(thread_id)

    def get_all_thread_results(self) -> Dict[str, Dict[str, Any]]:
        """Get all thread results"""
        return self.thread_results.copy()

    def get_all_standard_codes(self) -> List[str]:
        """Get all standard codes across all documents"""
        all_codes = []
        for analysis in self.standards_analysis.values():
            all_codes.extend(analysis.get("all_standard_codes", []))
        return list(set(all_codes))

    def get_all_certifications(self) -> List[str]:
        """Get all certifications across all documents"""
        all_certs = []
        for analysis in self.standards_analysis.values():
            all_certs.extend(analysis.get("all_certifications", []))
        return list(set(all_certs))

    def get_specifications_for_product_type(self, product_type: str) -> Dict[str, Any]:
        """
        Get all specifications relevant to a product type.
        Retrieves from memory (pre-analyzed data).
        """
        relevant_docs = self.get_relevant_documents(product_type)

        merged_specs = {
            "applicable_standards": [],
            "certifications": [],
            "equipment_specs": {},
            "safety": {},
            "installation_guidelines": [],
            "calibration": {},
            "sources": []
        }

        for doc_key in relevant_docs:
            analysis = self.standards_analysis.get(doc_key)
            if not analysis:
                continue

            # Merge standard codes
            merged_specs["applicable_standards"].extend(
                analysis.get("all_standard_codes", [])
            )

            # Merge certifications
            merged_specs["certifications"].extend(
                analysis.get("all_certifications", [])
            )

            # Merge specifications
            merged_specs["equipment_specs"].update(
                analysis.get("all_specifications", {})
            )

            # Merge guidelines
            merged_specs["installation_guidelines"].extend(
                analysis.get("all_guidelines", [])
            )

            # Track sources
            merged_specs["sources"].append(analysis.get("filename", doc_key))

            # Get section-specific data
            sections = analysis.get("sections", {})

            # Safety section
            if "safety_considerations" in sections:
                safety_section = sections["safety_considerations"]
                merged_specs["safety"].update(
                    safety_section.get("extracted_data", {})
                )

            # Calibration section
            if "calibration" in sections or "calibration_and_testing" in sections:
                cal_section = sections.get("calibration") or sections.get("calibration_and_testing", {})
                merged_specs["calibration"].update(
                    cal_section.get("extracted_data", {})
                )

        # Deduplicate lists
        merged_specs["applicable_standards"] = list(set(merged_specs["applicable_standards"]))
        merged_specs["certifications"] = list(set(merged_specs["certifications"]))
        merged_specs["sources"] = list(set(merged_specs["sources"]))

        return merged_specs

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        return {
            "documents_loaded": self.documents_loaded,
            "sections_analyzed": self.sections_analyzed,
            "instruments_count": len(self.identified_instruments),
            "accessories_count": len(self.identified_accessories),
            "thread_results_count": len(self.thread_results),
            "parallel_enrichments_count": len(self.parallel_enrichment_results),
            "product_types_indexed": len(self.product_type_index),
            "created_at": self.created_at
        }

    # =========================================================================
    # PARALLEL 3-SOURCE ENRICHMENT METHODS
    # =========================================================================

    def store_parallel_enrichment_result(self, item_id: str, result: ParallelEnrichmentResult) -> None:
        """
        Store result from parallel 3-source enrichment for an item.
        
        Args:
            item_id: Unique identifier for the item
            result: ParallelEnrichmentResult containing all 3 sources
        """
        with self._lock:
            self.parallel_enrichment_results[item_id] = result
            logger.info(f"[Memory] Stored parallel enrichment for: {result.get('item_name', item_id)}")

    def get_parallel_enrichment_result(self, item_id: str) -> Optional[ParallelEnrichmentResult]:
        """Retrieve parallel enrichment result for an item"""
        return self.parallel_enrichment_results.get(item_id)

    def get_all_parallel_enrichment_results(self) -> Dict[str, ParallelEnrichmentResult]:
        """Get all parallel enrichment results"""
        return self.parallel_enrichment_results.copy()

    def get_merged_specifications_for_item(self, item_id: str) -> Dict[str, SpecificationSource]:
        """
        Get the final merged specifications for an item.
        
        Returns the deduplicated, merged specs with source metadata.
        """
        result = self.parallel_enrichment_results.get(item_id)
        if result:
            return result.get("merged_specs", {})
        return {}


# =============================================================================
# PRODUCT TYPE TO DOCUMENT MAPPING
# =============================================================================

# Static mapping of product types to relevant standards documents
PRODUCT_TYPE_DOCUMENT_MAP = {
    # Instruments
    "pressure transmitter": ["instrumentation_pressure_standards.docx", "instrumentation_safety_standards.docx", "instrumentation_comm_signal_standards.docx"],
    "pressure gauge": ["instrumentation_pressure_standards.docx", "instrumentation_accessories_calibration_standards.docx"],
    "pressure switch": ["instrumentation_pressure_standards.docx", "instrumentation_safety_standards.docx"],
    "differential pressure": ["instrumentation_pressure_standards.docx", "instrumentation_flow_standards.docx"],

    "temperature transmitter": ["instrumentation_temperature_standards.docx", "instrumentation_safety_standards.docx", "instrumentation_comm_signal_standards.docx"],
    "temperature sensor": ["instrumentation_temperature_standards.docx", "instrumentation_safety_standards.docx"],
    "thermocouple": ["instrumentation_temperature_standards.docx", "instrumentation_calibration_maintenance_standards.docx"],
    "rtd": ["instrumentation_temperature_standards.docx", "instrumentation_calibration_maintenance_standards.docx"],
    "thermistor": ["instrumentation_temperature_standards.docx"],

    "flow meter": ["instrumentation_flow_standards.docx", "instrumentation_safety_standards.docx", "instrumentation_comm_signal_standards.docx"],
    "flow transmitter": ["instrumentation_flow_standards.docx", "instrumentation_safety_standards.docx"],
    "flow switch": ["instrumentation_flow_standards.docx", "instrumentation_safety_standards.docx"],
    "coriolis": ["instrumentation_flow_standards.docx", "instrumentation_calibration_maintenance_standards.docx"],
    "ultrasonic flow": ["instrumentation_flow_standards.docx"],
    "magnetic flow": ["instrumentation_flow_standards.docx"],
    "vortex flow": ["instrumentation_flow_standards.docx"],

    "level transmitter": ["instrumentation_level_standards.docx", "instrumentation_safety_standards.docx", "instrumentation_comm_signal_standards.docx"],
    "level sensor": ["instrumentation_level_standards.docx", "instrumentation_safety_standards.docx"],
    "level switch": ["instrumentation_level_standards.docx", "instrumentation_safety_standards.docx"],
    "radar level": ["instrumentation_level_standards.docx"],
    "ultrasonic level": ["instrumentation_level_standards.docx"],

    "control valve": ["instrumentation_valves_actuators_standards.docx", "instrumentation_safety_standards.docx"],
    "actuator": ["instrumentation_valves_actuators_standards.docx"],
    "positioner": ["instrumentation_valves_actuators_standards.docx", "instrumentation_comm_signal_standards.docx"],
    "isolation valve": ["instrumentation_valves_actuators_standards.docx"],

    "analyzer": ["instrumentation_analytical_standards.docx", "instrumentation_safety_standards.docx"],
    "ph sensor": ["instrumentation_analytical_standards.docx", "instrumentation_calibration_maintenance_standards.docx"],
    "conductivity sensor": ["instrumentation_analytical_standards.docx"],
    "oxygen analyzer": ["instrumentation_analytical_standards.docx", "instrumentation_safety_standards.docx"],
    "gas detector": ["instrumentation_safety_standards.docx", "instrumentation_analytical_standards.docx"],

    "plc": ["instrumentation_control_systems_standards.docx", "instrumentation_safety_standards.docx"],
    "dcs": ["instrumentation_control_systems_standards.docx", "instrumentation_safety_standards.docx"],
    "controller": ["instrumentation_control_systems_standards.docx"],

    "vibration sensor": ["instrumentation_condition_monitoring_standards.docx"],
    "vibration transmitter": ["instrumentation_condition_monitoring_standards.docx"],

    # Accessories
    "manifold": ["instrumentation_accessories_calibration_standards.docx", "instrumentation_pressure_standards.docx"],
    "thermowell": ["instrumentation_accessories_calibration_standards.docx", "instrumentation_temperature_standards.docx"],
    "cable": ["instrumentation_comm_signal_standards.docx", "instrumentation_accessories_calibration_standards.docx"],
    "junction box": ["instrumentation_accessories_calibration_standards.docx", "instrumentation_safety_standards.docx"],
    "enclosure": ["instrumentation_accessories_calibration_standards.docx", "instrumentation_safety_standards.docx"],
    "mounting bracket": ["instrumentation_accessories_calibration_standards.docx"],
    "impulse line": ["instrumentation_accessories_calibration_standards.docx", "instrumentation_pressure_standards.docx"],
    "tubing": ["instrumentation_accessories_calibration_standards.docx"],
    "fitting": ["instrumentation_accessories_calibration_standards.docx"],
    "valve manifold": ["instrumentation_accessories_calibration_standards.docx", "instrumentation_pressure_standards.docx"],
    "protection head": ["instrumentation_accessories_calibration_standards.docx", "instrumentation_temperature_standards.docx"],
    "transmitter housing": ["instrumentation_accessories_calibration_standards.docx"],
    "calibrator": ["instrumentation_calibration_maintenance_standards.docx"],
    "test equipment": ["instrumentation_calibration_maintenance_standards.docx"],
}


def get_relevant_documents_for_product(
    product_type: str, 
    constraint_keywords: List[str] = None
) -> List[str]:
    """
    Get list of relevant standards documents for a product type.
    Uses fuzzy matching for flexibility.
    
    Args:
        product_type: Product type string
        constraint_keywords: Optional list of constraint keywords that drive additional 
                           standards lookup (e.g., ["vibration", "hydrogen"])
    """
    product_lower = product_type.lower().strip()
    constraint_keywords = constraint_keywords or []

    # Direct match
    if product_lower in PRODUCT_TYPE_DOCUMENT_MAP:
        base_docs = list(PRODUCT_TYPE_DOCUMENT_MAP[product_lower])
    else:
        base_docs = []
        
        # Partial match
        for key, docs in PRODUCT_TYPE_DOCUMENT_MAP.items():
            if key in product_lower or product_lower in key:
                base_docs = list(docs)
                break

    # Keyword-based matching (if no direct match)
    if not base_docs:
        keywords_to_docs = {
            "pressure": ["instrumentation_pressure_standards.docx"],
            "temperature": ["instrumentation_temperature_standards.docx"],
            "temp": ["instrumentation_temperature_standards.docx"],
            "flow": ["instrumentation_flow_standards.docx"],
            "level": ["instrumentation_level_standards.docx"],
            "valve": ["instrumentation_valves_actuators_standards.docx"],
            "actuator": ["instrumentation_valves_actuators_standards.docx"],
            "analyzer": ["instrumentation_analytical_standards.docx"],
            "analytical": ["instrumentation_analytical_standards.docx"],
            "safety": ["instrumentation_safety_standards.docx"],
            "control": ["instrumentation_control_systems_standards.docx"],
            "communication": ["instrumentation_comm_signal_standards.docx"],
            "signal": ["instrumentation_comm_signal_standards.docx"],
            "calibration": ["instrumentation_calibration_maintenance_standards.docx"],
            "accessory": ["instrumentation_accessories_calibration_standards.docx"],
            "mount": ["instrumentation_accessories_calibration_standards.docx"],
            "cable": ["instrumentation_comm_signal_standards.docx"],
            "vibration": ["instrumentation_condition_monitoring_standards.docx"],
        }

        for keyword, docs in keywords_to_docs.items():
            if keyword in product_lower:
                base_docs.extend(docs)

    # ==========================================================
    # CONSTRAINT-BASED ADDITIONAL DOCUMENTS
    # Add specialized standards based on constraint keywords
    # ==========================================================
    constraint_doc_mapping = {
        # Mechanical/vibration constraints
        "vibration": ["instrumentation_condition_monitoring_standards.docx"],
        "pulsation": ["instrumentation_pressure_standards.docx", "instrumentation_flow_standards.docx"],
        
        # Chemical service constraints (material compatibility)
        "hydrogen": ["instrumentation_safety_standards.docx", "instrumentation_accessories_calibration_standards.docx"],
        "sour_gas": ["instrumentation_safety_standards.docx", "instrumentation_pressure_standards.docx"],
        "cryogenic": ["instrumentation_temperature_standards.docx", "instrumentation_safety_standards.docx"],
        "seawater": ["instrumentation_accessories_calibration_standards.docx"],
        "acid": ["instrumentation_analytical_standards.docx", "instrumentation_accessories_calibration_standards.docx"],
        "caustic": ["instrumentation_analytical_standards.docx"],
        
        # Material requirements
        "stainless": ["instrumentation_accessories_calibration_standards.docx"],
        "duplex": ["instrumentation_accessories_calibration_standards.docx"],
        "inconel": ["instrumentation_accessories_calibration_standards.docx"],
        "hastelloy": ["instrumentation_accessories_calibration_standards.docx"],
        
        # Physical conditions
        "abrasive": ["instrumentation_flow_standards.docx", "instrumentation_level_standards.docx"],
        "viscous": ["instrumentation_flow_standards.docx"],
    }
    
    for constraint in constraint_keywords:
        if constraint in constraint_doc_mapping:
            base_docs.extend(constraint_doc_mapping[constraint])
            logger.debug(f"[DocMapping] Added docs for constraint '{constraint}'")

    # Default fallback if still empty
    if not base_docs:
        base_docs = [
            "instrumentation_accessories_calibration_standards.docx",
            "instrumentation_safety_standards.docx"
        ]
    else:
        # Always add safety standards for instruments
        if "instrumentation_safety_standards.docx" not in base_docs:
            base_docs.append("instrumentation_safety_standards.docx")
    
    return list(set(base_docs))


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "SectionAnalysis",
    "StandardsDocumentAnalysis",
    "UserContextMemory",
    "IdentifiedItemMemory",
    "ThreadInfo",
    "ExecutionPlan",
    "SpecificationSource",
    "ParallelEnrichmentResult",
    "DeepAgentMemory",
    "PRODUCT_TYPE_DOCUMENT_MAP",
    "get_relevant_documents_for_product",
]

