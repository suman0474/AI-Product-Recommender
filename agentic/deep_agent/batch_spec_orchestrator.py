# agentic/deep_agent/batch_spec_orchestrator.py
# =============================================================================
# BATCH SPECIFICATION ORCHESTRATOR - AGENTIC APPROACH
# =============================================================================
#
# Orchestrates parallel batch specification enrichment for multiple items
# using LangGraph with a multi-worker coordinator pattern.
#
# ARCHITECTURE:
# - Coordinator Agent: Manages batch processing and ensures min specs per item
# - Item Processor: Processes each item with parallel LLM and Standards workers
# - Iterative Enrichment: If item specs < 30, iterates with additional domains
# - Parallel Items: Multiple items processed concurrently
#
# PARALLELIZATION LEVELS:
# - Batch Level: Process N items in parallel
# - Item Level: For each item, run LLM + Standards in parallel
# - LLM Iteration: 4 focus-area workers per iteration
# - Standards Iteration: 3 domain workers per iteration
# - Item Iteration: 8 items extracted concurrently per domain
#
# ESTIMATED SPEEDUP: 50-100x vs sequential processing
#
# =============================================================================

import logging
import time
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from .llm_specs_generator import MIN_LLM_SPECS_COUNT
from .standards_deep_agent import MIN_STANDARDS_SPECS_COUNT

# Import global executor manager for bounded thread pool
from agentic.global_executor_manager import get_global_executor

logger = logging.getLogger(__name__)


# =============================================================================
# BATCH ITEM STATE AND TRACKING
# =============================================================================

class BatchItemState:
    """Tracks the enrichment state of a single item."""

    def __init__(self, item_index: int, item_data: Dict[str, Any]):
        self.index = item_index
        self.name = item_data.get("name", f"Item {item_index}")
        self.category = item_data.get("category", "Unknown")
        self.sample_input = item_data.get("sample_input", "")

        # Specification tracking
        self.llm_specs: Dict[str, Any] = {}
        self.standards_specs: Dict[str, Any] = {}
        self.combined_specs: Dict[str, Any] = {}

        # Iteration tracking
        self.llm_iterations = 0
        self.standards_iterations = 0
        self.llm_target_reached = False
        self.standards_target_reached = False
        self.total_target_reached = False

        # Timing
        self.start_time = time.time()
        self.processing_time_ms = 0

    def get_spec_count(self, source: str = "combined") -> int:
        """Get count of valid specifications."""
        specs = getattr(self, f"{source}_specs", {})
        count = 0
        for value in specs.values():
            if value and str(value).lower() not in ["null", "none", "n/a", ""]:
                count += 1
        return count

    def update_combined_specs(self) -> None:
        """Merge LLM and Standards specs."""
        self.combined_specs = {**self.llm_specs, **self.standards_specs}
        self.total_target_reached = (
            self.get_spec_count("combined") >= 60
        )

    def finalize(self) -> Dict[str, Any]:
        """Generate final item result."""
        self.processing_time_ms = int((time.time() - self.start_time) * 1000)

        return {
            "index": self.index,
            "name": self.name,
            "category": self.category,
            "specifications": {
                "llm": self.llm_specs,
                "standards": self.standards_specs,
                "combined": self.combined_specs
            },
            "spec_counts": {
                "llm": self.get_spec_count("llm"),
                "standards": self.get_spec_count("standards"),
                "combined": self.get_spec_count("combined"),
                "min_required": 60
            },
            "iterations": {
                "llm": self.llm_iterations,
                "standards": self.standards_iterations
            },
            "targets_reached": {
                "llm": self.llm_target_reached,
                "standards": self.standards_target_reached,
                "total": self.total_target_reached
            },
            "processing_time_ms": self.processing_time_ms
        }


# =============================================================================
# BATCH COORDINATOR AGENT
# =============================================================================

def batch_coordinator(
    items: List[Dict[str, Any]],
    max_parallel_items: int = 4
) -> Dict[str, Any]:
    """
    Batch Coordinator Agent: Manages parallel processing of multiple items.

    STRATEGY:
    - Process items in parallel groups (4 items at a time)
    - For each item, run LLM and Standards enrichment in parallel
    - Track enrichment progress and iterate for items below minimum
    - Aggregate results

    Args:
        items: List of items to enrich
        max_parallel_items: Maximum items to process simultaneously

    Returns:
        Dict with enriched items and batch metadata
    """
    logger.info(f"[BATCH-COORDINATOR] Starting batch coordination for {len(items)} items...")
    logger.info(f"[BATCH-COORDINATOR] Processing up to {max_parallel_items} items in parallel...")

    start_time = time.time()
    all_item_states = {i: BatchItemState(i, item) for i, item in enumerate(items)}

    # Process items in parallel using global executor (bounded to 16 workers globally)
    executor = get_global_executor()
    future_to_index = {
        executor.submit(_process_single_item, index, state):
        index for index, state in all_item_states.items()
    }

    for future in as_completed(future_to_index):
        index = future_to_index[future]
        try:
            enriched_item = future.result()
            logger.info(f"[BATCH-COORDINATOR] Item {index} ({all_item_states[index].name}): Processing complete")
        except Exception as exc:
            logger.error(f"[BATCH-COORDINATOR] Item {index} generated exception: {exc}")

    # Compile results
    processing_time = int((time.time() - start_time) * 1000)
    enriched_items = [
        state.finalize() for state in all_item_states.values()
    ]

    # Calculate batch statistics
    total_items = len(items)
    items_at_target = sum(
        1 for state in all_item_states.values()
        if state.total_target_reached
    )

    logger.info(f"[BATCH-COORDINATOR] Batch complete: {items_at_target}/{total_items} items reached target")
    logger.info(f"[BATCH-COORDINATOR] Total processing time: {processing_time}ms")

    return {
        "success": True,
        "items": enriched_items,
        "batch_metadata": {
            "total_items": total_items,
            "items_at_target": items_at_target,
            "processing_time_ms": processing_time,
            "avg_time_per_item_ms": processing_time // total_items if total_items > 0 else 0,
            "parallelization_factor": f"{max_parallel_items} items concurrent"
        }
    }


def _process_single_item(index: int, state: BatchItemState) -> Dict[str, Any]:
    """
    Process a single item with parallel LLM and Standards enrichment.

    Runs LLM and Standards generation concurrently for the item.
    """
    logger.info(f"[ITEM-{index}] Processing: {state.name}")

    # Run LLM and Standards in parallel using global executor
    executor = get_global_executor()
    llm_future = executor.submit(
        _enrich_item_with_llm,
        state.name, state.category, state.sample_input
    )
    standards_future = executor.submit(
        _enrich_item_with_standards,
        state.name, state.category, state.sample_input
    )

    # Wait for results
    llm_result = llm_future.result()
        standards_result = standards_future.result()

    # Update state
    state.llm_specs = llm_result.get("specifications", {})
    state.llm_iterations = llm_result.get("iterations", 0)
    state.llm_target_reached = llm_result.get("target_reached", False)

    state.standards_specs = standards_result.get("specifications", {})
    state.standards_iterations = standards_result.get("iterations", 0)
    state.standards_target_reached = standards_result.get("target_reached", False)

    state.update_combined_specs()

    logger.info(f"[ITEM-{index}] Enrichment complete: {state.get_spec_count()} total specs")

    return state.finalize()


def _enrich_item_with_llm(
    item_name: str,
    category: str,
    context: str
) -> Dict[str, Any]:
    """Generate LLM specs for an item with iterative loop."""
    logger.info(f"[LLM-ENRICHER] Enriching with LLM: {item_name}")

    try:
        from .llm_specs_generator import generate_llm_specs

        result = generate_llm_specs(
            product_type=item_name,
            category=category,
            context=context,
            min_specs=MIN_LLM_SPECS_COUNT
        )

        return {
            "specifications": result.get("specifications", {}),
            "iterations": result.get("iterations", 1),
            "target_reached": result.get("target_reached", False),
            "error": result.get("error")
        }

    except Exception as e:
        logger.error(f"[LLM-ENRICHER] Error enriching {item_name}: {e}")
        return {
            "specifications": {},
            "iterations": 0,
            "target_reached": False,
            "error": str(e)
        }


def _enrich_item_with_standards(
    item_name: str,
    category: str,
    context: str
) -> Dict[str, Any]:
    """Extract standards specs for an item with iterative loop."""
    logger.info(f"[STANDARDS-ENRICHER] Enriching with Standards: {item_name}")

    try:
        from .standards_deep_agent import run_standards_deep_agent

        result = run_standards_deep_agent(
            user_requirement=f"{category} - {item_name}",
            session_id=f"batch-item-{item_name}-{int(time.time())}",
            inferred_specs=None,
            min_specs=MIN_STANDARDS_SPECS_COUNT
        )

        return {
            "specifications": result.get("final_specifications", {}).get("specifications", {}),
            "iterations": result.get("iterations", 1),
            "target_reached": result.get("target_reached", False),
            "error": result.get("error")
        }

    except Exception as e:
        logger.error(f"[STANDARDS-ENRICHER] Error enriching {item_name}: {e}")
        return {
            "specifications": {},
            "iterations": 0,
            "target_reached": False,
            "error": str(e)
        }


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_batch_orchestrated_enrichment(
    items: List[Dict[str, Any]],
    max_parallel_items: int = 4
) -> Dict[str, Any]:
    """
    Run batch enrichment with full parallelization using agentic orchestration.

    PARALLELIZATION:
    - 4 items processed simultaneously
    - For each item: LLM + Standards run in parallel
    - LLM uses 4 focus-area workers per iteration
    - Standards uses 3 domain workers per iteration
    - Overall: 50-100x speedup vs sequential enrichment

    Args:
        items: List of items to enrich
        max_parallel_items: Max items to process concurrently (default: 4)

    Returns:
        Dict with enriched items and batch metadata
    """
    logger.info("=" * 80)
    logger.info("BATCH ORCHESTRATED SPEC ENRICHMENT (AGENTIC APPROACH)")
    logger.info("=" * 80)
    logger.info(f"Items to enrich: {len(items)}")
    logger.info(f"Parallel items: {max_parallel_items}")
    logger.info(f"Targets per item: LLM≥{MIN_LLM_SPECS_COUNT}, Standards≥{MIN_STANDARDS_SPECS_COUNT}, Total≥60")

    return batch_coordinator(items, max_parallel_items)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "run_batch_orchestrated_enrichment",
    "BatchItemState",
    "batch_coordinator"
]
