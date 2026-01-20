# agentic/deep_agent/learning_engine.py
# =============================================================================
# UNIFIED LEARNING ENGINE MODULE
# =============================================================================
#
# Consolidated from:
# - schema_failure_memory.py: Failure tracking, pattern learning, recovery suggestions
# - adaptive_prompt_engine.py: Prompt optimization based on success/failure history
#
# =============================================================================

# Re-export from schema_failure_memory
from .schema_failure_memory import (
    SchemaFailureMemory,
    FailureEntry,
    SuccessEntry,
    FailureType,
    RecoveryAction,
    FailurePattern,
    get_schema_failure_memory,
    reset_failure_memory,
)

# Re-export from adaptive_prompt_engine
from .adaptive_prompt_engine import (
    AdaptivePromptEngine,
    PromptStrategy,
    PromptOptimization,
    get_adaptive_prompt_engine,
)


# =============================================================================
# EXPORTS - All learning/adaptation in one place
# =============================================================================

__all__ = [
    # Schema Failure Memory
    "SchemaFailureMemory",
    "FailureEntry",
    "SuccessEntry",
    "FailureType",
    "RecoveryAction",
    "FailurePattern",
    "get_schema_failure_memory",
    "reset_failure_memory",
    
    # Adaptive Prompt Engine
    "AdaptivePromptEngine",
    "PromptStrategy",
    "PromptOptimization",
    "get_adaptive_prompt_engine",
]
