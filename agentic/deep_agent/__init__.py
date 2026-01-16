# agentic/deep_agent/__init__.py
# =============================================================================
# DEEP AGENT PACKAGE
# =============================================================================
#
# Unified package containing all Deep Agent components for standards-based
# specification extraction and schema population.
#
# =============================================================================

from .memory import (
    DeepAgentMemory,
    StandardsDocumentAnalysis,
    SectionAnalysis,
    UserContextMemory,
    IdentifiedItemMemory,
    ThreadInfo,
    ExecutionPlan,
    SpecificationSource,
    ParallelEnrichmentResult,
    PRODUCT_TYPE_DOCUMENT_MAP,
    get_relevant_documents_for_product,
)

from .document_loader import (
    STANDARDS_DIRECTORY,
    load_all_standards_documents,
    populate_memory_with_documents,
    populate_memory_with_items,
    analyze_user_input,
    analyze_all_documents,
    analyze_standards_document,
    extract_standard_codes,
    extract_certifications,
    extract_specifications,
    extract_guidelines,
)

from .workflow import (
    DeepAgentState,
    create_deep_agent_state,
    run_deep_agent_workflow,
    get_deep_agent_workflow,
    get_or_create_memory,
    get_memory,
    clear_memory,
)

from .integration import (
    integrate_deep_agent_specifications,
    prepare_deep_agent_input,
    populate_schema_from_deep_agent,
    load_schema_for_product,
    format_deep_agent_specs_for_display,
    run_deep_agent_for_specifications,
)

from .schema_populator import (
    populate_schema_with_deep_agent,
    extract_field_value_from_standards,
    SchemaPopulatorMemory,
    segregate_schema_by_sections,
    analyze_standards_for_sections,
    run_parallel_section_extraction,
    build_populated_schema,
)

from .sub_agents import (
    ExtractorAgent,
    CriticAgent,
    extract_specifications_with_validation,
    extract_specifications_for_products,
    get_schema_fields_for_product,
    PRODUCT_TYPE_SCHEMA_FIELDS,
)

from .api import deep_agent_bp

from .standards_deep_agent import (
    StandardsDeepAgentState,
    ConsolidatedSpecs,
    WorkerResult,
    StandardConstraint,
    run_standards_deep_agent,
    run_standards_deep_agent_batch,
    get_standards_deep_agent_workflow,
    STANDARD_DOMAINS,
    STANDARD_FILES,
)

from .spec_verifier import (
    SpecVerifierAgent,
    DescriptionExtractorAgent,
    verify_and_reextract_specs,
)

# NEW: Moved from agentic/ root
from .user_specs_extractor import (
    extract_user_specified_specs,
    extract_user_specs_batch,
    _get_user_specs_llm,  # Singleton accessor for performance
)

from .llm_specs_generator import (
    generate_llm_specs,
    generate_llm_specs_batch,
    generate_llm_specs_true_batch,  # Phase 2: Single LLM call for all items
    clear_llm_specs_cache,          # Phase 2: Cache management
    get_cache_stats,                # Phase 2: Cache statistics
    _get_llm_specs_llm,  # Singleton accessor for performance
)

from .parallel_specs_enrichment import (
    run_parallel_3_source_enrichment,
    deduplicate_and_merge_specifications,
)

from .schema_field_extractor import (
    extract_schema_field_values_from_standards,
    extract_standards_from_value,
    get_default_value_for_field,
    query_standards_for_field,
    SECTION_TO_QUERY_TYPE,
    PRODUCT_TYPE_DEFAULTS,
)

from .spec_output_normalizer import (
    normalize_specification_output,
    normalize_full_item_specs,
    normalize_key,
    deduplicate_specs,
    clean_value,
    extract_technical_values,
    STANDARD_KEY_MAPPINGS,
)

__all__ = [
    # Memory
    "DeepAgentMemory",
    "StandardsDocumentAnalysis",
    "SectionAnalysis",
    "UserContextMemory",
    "IdentifiedItemMemory",
    "ThreadInfo",
    "ExecutionPlan",
    "SpecificationSource",
    "ParallelEnrichmentResult",
    "PRODUCT_TYPE_DOCUMENT_MAP",
    "get_relevant_documents_for_product",
    
    # Document Loader
    "STANDARDS_DIRECTORY",
    "load_all_standards_documents",
    "populate_memory_with_documents",
    "populate_memory_with_items",
    "analyze_user_input",
    "analyze_all_documents",
    "analyze_standards_document",
    "extract_standard_codes",
    "extract_certifications",
    "extract_specifications",
    "extract_guidelines",
    
    # Workflow
    "DeepAgentState",
    "create_deep_agent_state",
    "run_deep_agent_workflow",
    "get_deep_agent_workflow",
    "get_or_create_memory",
    "get_memory",
    "clear_memory",
    
    # Integration
    "integrate_deep_agent_specifications",
    "prepare_deep_agent_input",
    "populate_schema_from_deep_agent",
    "load_schema_for_product",
    "format_deep_agent_specs_for_display",
    "run_deep_agent_for_specifications",
    
    # Schema Populator
    "populate_schema_with_deep_agent",
    "extract_field_value_from_standards",
    "SchemaPopulatorMemory",
    "segregate_schema_by_sections",
    "analyze_standards_for_sections",
    "run_parallel_section_extraction",
    "build_populated_schema",
    
    # Sub Agents
    "ExtractorAgent",
    "CriticAgent",
    "extract_specifications_with_validation",
    "extract_specifications_for_products",
    "get_schema_fields_for_product",
    "PRODUCT_TYPE_SCHEMA_FIELDS",
    
    # API
    "deep_agent_bp",
    
    # Standards Deep Agent
    "StandardsDeepAgentState",
    "ConsolidatedSpecs",
    "WorkerResult",
    "StandardConstraint",
    "run_standards_deep_agent",
    "run_standards_deep_agent_batch",
    "get_standards_deep_agent_workflow",
    "STANDARD_DOMAINS",
    "STANDARD_FILES",
    
    # Spec Verifier (Dual-Step Verification)
    "SpecVerifierAgent",
    "DescriptionExtractorAgent",
    "verify_and_reextract_specs",

    # User Specs Extractor (moved from agentic/)
    "extract_user_specified_specs",
    "extract_user_specs_batch",
    "_get_user_specs_llm",  # Singleton accessor

    # LLM Specs Generator (moved from agentic/)
    "generate_llm_specs",
    "generate_llm_specs_batch",
    "generate_llm_specs_true_batch",  # Phase 2: Single LLM call
    "clear_llm_specs_cache",          # Phase 2: Cache management
    "get_cache_stats",                # Phase 2: Cache statistics
    "_get_llm_specs_llm",  # Singleton accessor

    # Parallel Specs Enrichment (moved from agentic/)
    "run_parallel_3_source_enrichment",
    "deduplicate_and_merge_specifications",

    # Schema Field Extractor (moved from agentic/)
    "extract_schema_field_values_from_standards",
    "extract_standards_from_value",
    "get_default_value_for_field",
    "query_standards_for_field",
    "SECTION_TO_QUERY_TYPE",
    "PRODUCT_TYPE_DEFAULTS",

    # Spec Output Normalizer (NEW)
    "normalize_specification_output",
    "normalize_full_item_specs",
    "normalize_key",
    "deduplicate_specs",
    "clean_value",
    "extract_technical_values",
    "STANDARD_KEY_MAPPINGS",
]
