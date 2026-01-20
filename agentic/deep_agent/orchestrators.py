# agentic/deep_agent/orchestrators.py
# =============================================================================
# UNIFIED ORCHESTRATORS MODULE
# =============================================================================
#
# Consolidated from:
# - spec_generation_orchestrator.py: Single-item spec generation orchestration
# - batch_spec_orchestrator.py: Batch item spec generation orchestration
#
# =============================================================================

# Re-export from spec_generation_orchestrator
from .spec_generation_orchestrator import (
    run_orchestrated_spec_generation,
    get_spec_generation_orchestrator,
    OrchestratorState,
    WorkerResult,
)

# Re-export from batch_spec_orchestrator
from .batch_spec_orchestrator import (
    run_batch_orchestrated_enrichment,
    BatchItemState,
    batch_coordinator,
)


# =============================================================================
# EXPORTS - All orchestration in one place
# =============================================================================

__all__ = [
    # Single-Item Orchestration
    "run_orchestrated_spec_generation",
    "get_spec_generation_orchestrator",
    "OrchestratorState",
    "WorkerResult",
    
    # Batch Orchestration
    "run_batch_orchestrated_enrichment",
    "BatchItemState",
    "batch_coordinator",
]
