# agentic/deep_agent/phase3_memory_manager.py
# =============================================================================
# DEEP AGENT MEMORY MANAGER FOR PHASE 3
# =============================================================================
#
# Purpose: Manage hierarchical memory with mandatory spec locking throughout
# the multi-level parallel specification enrichment pipeline
#
# Key Features:
# 1. Immediate locking of mandatory user specs (Level 3A)
# 2. Prevent overwriting of locked specs through levels 3B, 3C, 4
# 3. Per-item isolation (no cross-contamination)
# 4. Comprehensive audit trail
# 5. Memory consolidation before Level 5 merge
#
# =============================================================================

import logging
from typing import Dict, Set, List, Any, Optional, TypedDict
from datetime import datetime
from dataclasses import dataclass, field, asdict
import json
import threading

logger = logging.getLogger(__name__)


@dataclass
class SpecificationEntry:
    """Single specification with source metadata"""
    key: str
    value: Any
    source: str  # "user_specified", "llm_generated", "standards"
    confidence: float = 0.7
    standard_reference: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    locked: bool = False


@dataclass
class ItemSpecMemory:
    """Memory store for a single item's specifications"""
    item_id: str
    item_name: str
    item_type: str  # "instrument" or "accessory"
    product_type: str  # "temperature_sensor", "thermowell", etc.

    # Specifications by source
    mandatory_specs: Dict[str, SpecificationEntry] = field(default_factory=dict)
    user_specified_specs: Dict[str, SpecificationEntry] = field(default_factory=dict)
    llm_generated_specs: Dict[str, SpecificationEntry] = field(default_factory=dict)
    standards_specs: Dict[str, SpecificationEntry] = field(default_factory=dict)

    # Merged result
    merged_specs: Dict[str, SpecificationEntry] = field(default_factory=dict)

    # Tracking
    locked_keys: Set[str] = field(default_factory=set)
    processing_log: List[Dict] = field(default_factory=list)

    # Locks for thread safety
    lock = threading.RLock()


class Phase3MemoryManager:
    """
    Hierarchical memory manager for Phase 3 processing.

    Maintains mandatory spec locking from Level 3A through Level 5.
    Ensures user-specified specs CANNOT be overwritten by subsequent sources.
    """

    def __init__(self, session_id: str, max_items: int = 1000):
        """
        Initialize Phase 3 memory manager.

        Args:
            session_id: Session identifier
            max_items: Maximum items to store (prevent memory explosion)
        """
        self.session_id = session_id
        self.max_items = max_items

        # Per-item memory stores
        self.item_memories: Dict[str, ItemSpecMemory] = {}

        # Global stats
        self.stats = {
            "items_processed": 0,
            "total_mandatory_specs": 0,
            "total_user_specs": 0,
            "total_llm_specs": 0,
            "total_standards_specs": 0,
            "conflict_prevented": 0,
            "lock_violations_prevented": 0
        }

        # Thread safety
        self.lock = threading.RLock()

        logger.info(f"[MEMORY] Phase3MemoryManager initialized for session {session_id}")

    # =========================================================================
    # LEVEL 3A: MANDATORY SPEC LOCKING
    # =========================================================================

    def initialize_item(
        self,
        item_id: str,
        item_name: str,
        item_type: str,
        product_type: str
    ) -> ItemSpecMemory:
        """
        Initialize memory for a new item.

        Args:
            item_id: Unique item identifier
            item_name: Display name
            item_type: "instrument" or "accessory"
            product_type: Specific product type (e.g., "temperature_sensor")

        Returns:
            ItemSpecMemory for this item
        """
        with self.lock:
            if len(self.item_memories) >= self.max_items:
                logger.warning(f"[MEMORY] Max items ({self.max_items}) reached, clearing oldest")
                # Remove oldest item (FIFO)
                oldest_id = next(iter(self.item_memories))
                del self.item_memories[oldest_id]

            memory = ItemSpecMemory(
                item_id=item_id,
                item_name=item_name,
                item_type=item_type,
                product_type=product_type
            )

            self.item_memories[item_id] = memory
            self.stats["items_processed"] += 1

            logger.debug(f"[MEMORY] Initialized memory for {item_id} ({product_type})")

            return memory

    def lock_mandatory_specs(
        self,
        item_id: str,
        mandatory_specs: Dict[str, Any]
    ) -> Set[str]:
        """
        CRITICAL: Lock user-specified specs immediately.

        These specs CANNOT be overwritten by subsequent enrichment sources.

        Args:
            item_id: Item identifier
            mandatory_specs: User-specified specifications

        Returns:
            Set of locked keys
        """
        with self.lock:
            if item_id not in self.item_memories:
                logger.error(f"[MEMORY] Item {item_id} not initialized")
                return set()

            memory = self.item_memories[item_id]

            locked_keys = set()

            for key, value in mandatory_specs.items():
                if not value or str(value).lower() in ["null", "none", "n/a", ""]:
                    continue

                spec_entry = SpecificationEntry(
                    key=key,
                    value=value,
                    source="user_specified",
                    confidence=1.0,
                    locked=True
                )

                memory.mandatory_specs[key] = spec_entry
                memory.locked_keys.add(key)
                locked_keys.add(key)

            memory.processing_log.append({
                "timestamp": datetime.now().isoformat(),
                "action": "lock_mandatory_specs",
                "spec_count": len(locked_keys),
                "locked_keys": list(locked_keys)
            })

            self.stats["total_mandatory_specs"] += len(locked_keys)

            logger.info(
                f"[MEMORY] Locked {len(locked_keys)} mandatory specs for {item_id}: {list(locked_keys)}"
            )

            return locked_keys

    # =========================================================================
    # LEVEL 3B: USER-SPECIFIED SPECS STORAGE
    # =========================================================================

    def store_user_specified_specs(
        self,
        item_id: str,
        user_specs: Dict[str, Any]
    ) -> int:
        """
        Store user-specified specs (from explicit user input).

        Args:
            item_id: Item identifier
            user_specs: User-specified specifications

        Returns:
            Count of stored specs
        """
        with self.lock:
            if item_id not in self.item_memories:
                logger.error(f"[MEMORY] Item {item_id} not initialized")
                return 0

            memory = self.item_memories[item_id]
            stored_count = 0

            for key, value in user_specs.items():
                # Skip if already locked as mandatory
                if key in memory.locked_keys:
                    continue

                if not value or str(value).lower() in ["null", "none", "n/a", ""]:
                    continue

                spec_entry = SpecificationEntry(
                    key=key,
                    value=value,
                    source="user_specified",
                    confidence=1.0
                )

                memory.user_specified_specs[key] = spec_entry
                stored_count += 1

            memory.processing_log.append({
                "timestamp": datetime.now().isoformat(),
                "action": "store_user_specified",
                "spec_count": stored_count
            })

            self.stats["total_user_specs"] += stored_count

            logger.debug(f"[MEMORY] Stored {stored_count} user specs for {item_id}")

            return stored_count

    # =========================================================================
    # LEVEL 3B/3C: LLM & STANDARDS SPECS STORAGE (WITH LOCKING)
    # =========================================================================

    def store_enrichment_specs(
        self,
        item_id: str,
        source: str,  # "llm_generated" or "standards"
        specs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Store enrichment specs, respecting locked mandatory specs.

        Args:
            item_id: Item identifier
            source: "llm_generated" or "standards"
            specs: Specifications to store

        Returns:
            Filtered specs (excluding locked keys)
        """
        with self.lock:
            if item_id not in self.item_memories:
                logger.error(f"[MEMORY] Item {item_id} not initialized")
                return {}

            memory = self.item_memories[item_id]
            filtered_specs = {}
            conflicts_prevented = 0

            for key, value in specs.items():
                # CRITICAL: Skip if key is locked (mandatory)
                if key in memory.locked_keys:
                    self.stats["lock_violations_prevented"] += 1
                    conflicts_prevented += 1
                    logger.debug(
                        f"[MEMORY] PREVENTED overwrite of locked spec {key} in {item_id} "
                        f"(current value: {memory.mandatory_specs[key].value}, "
                        f"attempted: {value})"
                    )
                    continue

                if not value or str(value).lower() in ["null", "none", "n/a", ""]:
                    continue

                # Check for existing value conflict
                existing_spec = None
                if source == "llm_generated":
                    existing_spec = memory.llm_generated_specs.get(key)
                elif source == "standards":
                    existing_spec = memory.standards_specs.get(key)

                if existing_spec and existing_spec.value != value:
                    self.stats["conflict_prevented"] += 1
                    logger.debug(f"[MEMORY] Spec conflict for {key}: {existing_spec.value} vs {value}")

                spec_entry = SpecificationEntry(
                    key=key,
                    value=value,
                    source=source,
                    confidence=0.9 if source == "standards" else 0.7
                )

                if source == "llm_generated":
                    memory.llm_generated_specs[key] = spec_entry
                elif source == "standards":
                    memory.standards_specs[key] = spec_entry

                filtered_specs[key] = value

            memory.processing_log.append({
                "timestamp": datetime.now().isoformat(),
                "action": f"store_{source}",
                "spec_count": len(filtered_specs),
                "conflicts_prevented": conflicts_prevented
            })

            if source == "llm_generated":
                self.stats["total_llm_specs"] += len(filtered_specs)
            elif source == "standards":
                self.stats["total_standards_specs"] += len(filtered_specs)

            logger.debug(
                f"[MEMORY] Stored {len(filtered_specs)} {source} specs for {item_id} "
                f"({conflicts_prevented} conflicts prevented)"
            )

            return filtered_specs

    # =========================================================================
    # LEVEL 4: MERGE & CONSOLIDATION
    # =========================================================================

    def merge_all_specs(self, item_id: str) -> Dict[str, SpecificationEntry]:
        """
        Merge all specs for an item with priority.

        Priority: user_specified > standards > llm_generated

        Args:
            item_id: Item identifier

        Returns:
            Merged specifications dictionary
        """
        with self.lock:
            if item_id not in self.item_memories:
                logger.error(f"[MEMORY] Item {item_id} not initialized")
                return {}

            memory = self.item_memories[item_id]
            merged: Dict[str, SpecificationEntry] = {}

            # 1. Add mandatory specs (always included)
            merged.update(memory.mandatory_specs)

            # 2. Add user-specified specs (don't overwrite)
            for key, spec in memory.user_specified_specs.items():
                if key not in merged:
                    merged[key] = spec

            # 3. Add standards specs (higher priority than LLM)
            for key, spec in memory.standards_specs.items():
                if key not in merged:
                    merged[key] = spec

            # 4. Add LLM specs (lowest priority)
            for key, spec in memory.llm_generated_specs.items():
                if key not in merged:
                    merged[key] = spec

            memory.merged_specs = merged

            memory.processing_log.append({
                "timestamp": datetime.now().isoformat(),
                "action": "merge_all_specs",
                "total_merged": len(merged),
                "mandatory": len(memory.mandatory_specs),
                "user_specified": len(memory.user_specified_specs),
                "standards": len(memory.standards_specs),
                "llm_generated": len(memory.llm_generated_specs)
            })

            logger.info(
                f"[MEMORY] Merged {len(merged)} specs for {item_id} "
                f"(mandatory: {len(memory.mandatory_specs)}, "
                f"user: {len(memory.user_specified_specs)}, "
                f"standards: {len(memory.standards_specs)}, "
                f"llm: {len(memory.llm_generated_specs)})"
            )

            return merged

    # =========================================================================
    # RETRIEVAL & VALIDATION
    # =========================================================================

    def get_locked_keys(self, item_id: str) -> Set[str]:
        """Get all locked (mandatory) keys for an item"""
        with self.lock:
            if item_id not in self.item_memories:
                return set()
            return self.item_memories[item_id].locked_keys.copy()

    def get_merged_specs(self, item_id: str) -> Dict[str, Any]:
        """Get merged specifications for an item"""
        with self.lock:
            if item_id not in self.item_memories:
                return {}

            memory = self.item_memories[item_id]
            return {
                key: spec.value
                for key, spec in memory.merged_specs.items()
            }

    def get_merged_specs_with_metadata(self, item_id: str) -> Dict[str, SpecificationEntry]:
        """Get merged specifications with full metadata"""
        with self.lock:
            if item_id not in self.item_memories:
                return {}
            return self.item_memories[item_id].merged_specs.copy()

    def validate_no_mandatory_overwrites(self, item_id: str) -> bool:
        """Validate that no mandatory specs were overwritten"""
        with self.lock:
            if item_id not in self.item_memories:
                return True

            memory = self.item_memories[item_id]

            # Check if all mandatory keys are still in merged
            for key in memory.locked_keys:
                if key not in memory.merged_specs:
                    logger.error(f"[MEMORY] VALIDATION FAILED: Mandatory spec {key} missing for {item_id}")
                    return False

                merged_value = memory.merged_specs[key].value
                mandatory_value = memory.mandatory_specs[key].value

                if merged_value != mandatory_value:
                    logger.error(
                        f"[MEMORY] VALIDATION FAILED: Mandatory spec {key} overwritten "
                        f"(was {mandatory_value}, now {merged_value})"
                    )
                    return False

            logger.debug(f"[MEMORY] Validation PASSED for {item_id}")
            return True

    # =========================================================================
    # MEMORY STATISTICS & REPORTING
    # =========================================================================

    def get_item_memory(self, item_id: str) -> Optional[ItemSpecMemory]:
        """Get full memory for an item"""
        with self.lock:
            return self.item_memories.get(item_id)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        with self.lock:
            return {
                "session_id": self.session_id,
                "items_in_memory": len(self.item_memories),
                "total_items_processed": self.stats["items_processed"],
                "total_mandatory_specs": self.stats["total_mandatory_specs"],
                "total_user_specs": self.stats["total_user_specs"],
                "total_llm_specs": self.stats["total_llm_specs"],
                "total_standards_specs": self.stats["total_standards_specs"],
                "conflicts_prevented": self.stats["conflict_prevented"],
                "lock_violations_prevented": self.stats["lock_violations_prevented"]
            }

    def get_all_item_stats(self) -> List[Dict[str, Any]]:
        """Get stats for all items"""
        with self.lock:
            stats = []
            for item_id, memory in self.item_memories.items():
                stats.append({
                    "item_id": item_id,
                    "item_name": memory.item_name,
                    "product_type": memory.product_type,
                    "locked_keys": len(memory.locked_keys),
                    "mandatory_specs": len(memory.mandatory_specs),
                    "user_specs": len(memory.user_specified_specs),
                    "llm_specs": len(memory.llm_generated_specs),
                    "standards_specs": len(memory.standards_specs),
                    "merged_specs": len(memory.merged_specs),
                    "processing_events": len(memory.processing_log)
                })
            return stats

    def get_processing_log(self, item_id: str) -> List[Dict]:
        """Get detailed processing log for an item"""
        with self.lock:
            if item_id not in self.item_memories:
                return []
            return self.item_memories[item_id].processing_log.copy()

    # =========================================================================
    # CLEANUP & RESET
    # =========================================================================

    def clear_item(self, item_id: str):
        """Remove item from memory"""
        with self.lock:
            if item_id in self.item_memories:
                del self.item_memories[item_id]
                logger.debug(f"[MEMORY] Cleared memory for {item_id}")

    def clear_all(self):
        """Clear all memory (use with caution)"""
        with self.lock:
            self.item_memories.clear()
            logger.warning(f"[MEMORY] All memory cleared for session {self.session_id}")

    def export_memory_snapshot(self) -> Dict[str, Any]:
        """Export full memory state as JSON-serializable dict"""
        with self.lock:
            snapshot = {
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "stats": self.stats.copy(),
                "items": {}
            }

            for item_id, memory in self.item_memories.items():
                snapshot["items"][item_id] = {
                    "item_name": memory.item_name,
                    "item_type": memory.item_type,
                    "product_type": memory.product_type,
                    "locked_keys": list(memory.locked_keys),
                    "spec_counts": {
                        "mandatory": len(memory.mandatory_specs),
                        "user_specified": len(memory.user_specified_specs),
                        "llm_generated": len(memory.llm_generated_specs),
                        "standards": len(memory.standards_specs),
                        "merged": len(memory.merged_specs)
                    },
                    "processing_events": len(memory.processing_log)
                }

            return snapshot


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "Phase3MemoryManager",
    "ItemSpecMemory",
    "SpecificationEntry"
]
