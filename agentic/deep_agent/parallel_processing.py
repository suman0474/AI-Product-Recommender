# agentic/deep_agent/parallel_processing.py
# =============================================================================
# UNIFIED PARALLEL PROCESSING MODULE
# =============================================================================
#
# Consolidated from:
# - parallel_specs_enrichment.py: 3-source parallel enrichment
# - optimized_parallel_agent.py: Optimized parallel with shared LLM
# - parallel_enrichment_engine.py: Parallel enrichment engine class
# - parallel_schema_generator.py: Parallel schema generation
# - async_schema_generator.py: Async schema generation
#
# =============================================================================

# Re-export from parallel_specs_enrichment
from .parallel_specs_enrichment import (
    run_parallel_3_source_enrichment,
    deduplicate_and_merge_specifications,
)

# Re-export from optimized_parallel_agent
from .optimized_parallel_agent import (
    run_optimized_parallel_enrichment,
    get_shared_llm,
    reset_shared_llm,
    extract_user_specs_with_shared_llm,
    generate_llm_specs_with_shared_llm,
)

# Re-export from parallel_enrichment_engine
from .parallel_enrichment_engine import (
    ParallelEnrichmentEngine,
)

# Re-export from parallel_schema_generator
from .parallel_schema_generator import (
    ParallelSchemaGenerator,
    generate_schemas_parallel,
)

# Re-export from async_schema_generator
from .async_schema_generator import (
    AsyncSchemaGenerator,
)


# =============================================================================
# EXPORTS - All parallel processing in one place
# =============================================================================

__all__ = [
    # 3-Source Parallel Enrichment
    "run_parallel_3_source_enrichment",
    "deduplicate_and_merge_specifications",
    
    # Optimized Parallel Agent
    "run_optimized_parallel_enrichment",
    "get_shared_llm",
    "reset_shared_llm",
    "extract_user_specs_with_shared_llm",
    "generate_llm_specs_with_shared_llm",
    
    # Parallel Enrichment Engine
    "ParallelEnrichmentEngine",
    
    # Parallel Schema Generator
    "ParallelSchemaGenerator",
    "generate_schemas_parallel",
    
    # Async Schema Generator
    "AsyncSchemaGenerator",
]
