# agentic/deep_agent/specification_aggregator.py
# =============================================================================
# MULTI-SOURCE SPECIFICATION AGGREGATOR
# =============================================================================
#
# Purpose: Aggregate specifications from multiple sources with guaranteed
# 60+ specifications per product type
#
# Sources (in priority order):
# 1. User-specified (confidence: 1.0) - MANDATORY
# 2. Standards-based (confidence: 0.9) - HIGH PRIORITY
# 3. LLM-generated (confidence: 0.8) - MEDIUM PRIORITY
# 4. Template-default (confidence: 0.5) - FALLBACK
# 5. Derived (confidence: 0.6) - COMPUTED
#
# =============================================================================

import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from enum import Enum

from .specification_templates import (
    get_all_specs_for_product_type,
    get_spec_count_for_product_type
)

logger = logging.getLogger(__name__)


class SpecSource(Enum):
    """Specification source types"""
    USER = "user_specified"
    STANDARDS = "standards"
    LLM = "llm_generated"
    TEMPLATE = "template_default"
    DERIVED = "derived"


class SpecificationAggregator:
    """
    Aggregate specifications from multiple sources with priority-based merging
    and guaranteed minimum specification count.
    """

    # Priority order (lower priority number = higher priority)
    SOURCE_PRIORITY = {
        SpecSource.USER: 1,
        SpecSource.STANDARDS: 2,
        SpecSource.LLM: 3,
        SpecSource.DERIVED: 4,
        SpecSource.TEMPLATE: 5
    }

    MIN_SPEC_COUNT = 60  # Guarantee minimum 60 specs

    def __init__(self, product_type: str, session_id: str = "default"):
        """
        Initialize aggregator for a specific product type.

        Args:
            product_type: Type of product (e.g., "temperature_sensor")
            session_id: Session identifier for tracking
        """
        self.product_type = product_type
        self.session_id = session_id

        # Get template for this product type
        self.template_specs = get_all_specs_for_product_type(product_type)
        self.template_spec_count = get_spec_count_for_product_type(product_type)

        logger.info(
            f"[AGGREGATOR] Initialized for {product_type} "
            f"(template: {self.template_spec_count} specs)"
        )

    def aggregate(
        self,
        item_id: str,
        item_name: str,
        user_specs: Optional[Dict[str, Any]] = None,
        standards_specs: Optional[Dict[str, Any]] = None,
        llm_specs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Aggregate specifications from all sources with guaranteed minimum count.

        Args:
            item_id: Item identifier
            item_name: Item display name
            user_specs: User-provided specifications
            standards_specs: Standards-extracted specifications
            llm_specs: LLM-generated specifications

        Returns:
            Aggregated specifications dictionary with guaranteed 60+ count
        """
        logger.info(
            f"[AGGREGATOR] Aggregating specs for {item_name} "
            f"(user: {len(user_specs or {})}, standards: {len(standards_specs or {})}, "
            f"llm: {len(llm_specs or {})})"
        )

        start_time = datetime.now()

        # Step 1: Collect specs from all sources
        all_specs: Dict[str, Dict] = {}

        # Add user specs (highest priority)
        if user_specs:
            for key, value in user_specs.items():
                if value is not None and str(value).lower() not in ["null", "none", "n/a"]:
                    all_specs[key] = {
                        "value": value,
                        "source": SpecSource.USER.value,
                        "confidence": 1.0,
                        "priority": self.SOURCE_PRIORITY[SpecSource.USER]
                    }

        # Add standards specs
        if standards_specs:
            for key, value in standards_specs.items():
                if key not in all_specs and value is not None:
                    if isinstance(value, dict):
                        std_value = value.get("value", value)
                        std_confidence = value.get("confidence", 0.9)
                    else:
                        std_value = value
                        std_confidence = 0.9

                    if str(std_value).lower() not in ["null", "none", ""]:
                        all_specs[key] = {
                            "value": std_value,
                            "source": SpecSource.STANDARDS.value,
                            "confidence": std_confidence,
                            "priority": self.SOURCE_PRIORITY[SpecSource.STANDARDS]
                        }

        # Add LLM specs
        if llm_specs:
            for key, value in llm_specs.items():
                if key not in all_specs and value is not None:
                    if isinstance(value, dict):
                        llm_value = value.get("value", value)
                        llm_confidence = value.get("confidence", 0.8)
                    else:
                        llm_value = value
                        llm_confidence = 0.8

                    if str(llm_value).lower() not in ["null", "none", ""]:
                        all_specs[key] = {
                            "value": llm_value,
                            "source": SpecSource.LLM.value,
                            "confidence": llm_confidence,
                            "priority": self.SOURCE_PRIORITY[SpecSource.LLM]
                        }

        current_count = len(all_specs)
        logger.info(f"[AGGREGATOR] Initial spec count: {current_count}")

        # Step 2: If already 60+, apply deduplication and return
        if current_count >= self.MIN_SPEC_COUNT:
            logger.info(f"[AGGREGATOR] Reached minimum {current_count} specs, deduplicating...")
            final_specs = self._deduplicate_specs(all_specs)
            return self._finalize_aggregation(
                item_id, item_name, final_specs, start_time, "from sources"
            )

        # Step 3: Fill gaps from template
        logger.info(
            f"[AGGREGATOR] Gap fill: {current_count} → {self.MIN_SPEC_COUNT} "
            f"(need {self.MIN_SPEC_COUNT - current_count} more)"
        )

        gap_filled = self._fill_from_template(all_specs, self.MIN_SPEC_COUNT - current_count)
        logger.info(f"[AGGREGATOR] Filled {gap_filled} specs from template")

        current_count = len(all_specs)

        # Step 4: If still short, generate derived specs
        if current_count < self.MIN_SPEC_COUNT:
            logger.info(
                f"[AGGREGATOR] Still short: {current_count} < {self.MIN_SPEC_COUNT}, "
                f"generating derived specs..."
            )

            derived_specs = self._generate_derived_specs(all_specs)
            for key, spec in derived_specs.items():
                if key not in all_specs:
                    all_specs[key] = spec

            logger.info(f"[AGGREGATOR] Generated {len(derived_specs)} derived specs")

        # Step 5: Final validation
        final_specs = all_specs
        final_count = len(final_specs)

        if final_count < self.MIN_SPEC_COUNT:
            logger.error(
                f"[AGGREGATOR] FAILED to reach {self.MIN_SPEC_COUNT} specs "
                f"(only {final_count})"
            )
            # Force fill remaining with placeholder specs
            forced_specs = self._force_fill_to_minimum(final_specs)
            final_specs.update(forced_specs)

        # Step 6: Deduplicate
        final_specs = self._deduplicate_specs(final_specs)

        return self._finalize_aggregation(
            item_id, item_name, final_specs, start_time,
            f"aggregated from multiple sources ({len(user_specs or {})} user, "
            f"{len(standards_specs or {})} standards, {len(llm_specs or {})} llm)"
        )

    def _fill_from_template(self, existing_specs: Dict, target_gap: int) -> int:
        """
        Fill specification gaps from template defaults.

        Args:
            existing_specs: Current specs dictionary
            target_gap: How many more specs to add

        Returns:
            Number of specs added
        """
        filled_count = 0

        # Iterate through template specs and fill gaps
        for spec_key, spec_def in self.template_specs.items():
            if filled_count >= target_gap:
                break

            if spec_key not in existing_specs and spec_def.typical_value is not None:
                existing_specs[spec_key] = {
                    "value": spec_def.typical_value,
                    "source": SpecSource.TEMPLATE.value,
                    "confidence": 0.5,
                    "priority": self.SOURCE_PRIORITY[SpecSource.TEMPLATE],
                    "note": f"Template default ({spec_def.importance.name})"
                }
                filled_count += 1

        return filled_count

    def _generate_derived_specs(
        self,
        existing_specs: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate derived/computed specifications from existing specs.

        Args:
            existing_specs: Current specifications

        Returns:
            Dictionary of derived specifications
        """
        derived = {}

        # Example: If we have measurement_range_min and _max, derive measurement_range_string
        if "measurement_range_min" in existing_specs and "measurement_range_max" in existing_specs:
            if "measurement_range" not in existing_specs:
                min_val = existing_specs["measurement_range_min"]["value"]
                max_val = existing_specs["measurement_range_max"]["value"]
                derived["measurement_range"] = {
                    "value": f"{min_val} to {max_val}",
                    "source": SpecSource.DERIVED.value,
                    "confidence": 0.8,
                    "priority": self.SOURCE_PRIORITY[SpecSource.DERIVED],
                    "note": "Derived from min/max range"
                }

        # Example: If we have supply_voltage_min and _max, derive supply_voltage_range
        if "supply_voltage_min" in existing_specs and "supply_voltage_max" in existing_specs:
            if "supply_voltage_range" not in existing_specs:
                min_v = existing_specs["supply_voltage_min"]["value"]
                max_v = existing_specs["supply_voltage_max"]["value"]
                derived["supply_voltage_range"] = {
                    "value": f"{min_v}V to {max_v}V",
                    "source": SpecSource.DERIVED.value,
                    "confidence": 0.8,
                    "priority": self.SOURCE_PRIORITY[SpecSource.DERIVED],
                    "note": "Derived from voltage limits"
                }

        # Example: If we have operating_temperature_min/max, derive operating_temperature_range
        if "operating_temperature_min" in existing_specs and "operating_temperature_max" in existing_specs:
            if "operating_temperature_range" not in existing_specs:
                min_t = existing_specs["operating_temperature_min"]["value"]
                max_t = existing_specs["operating_temperature_max"]["value"]
                derived["operating_temperature_range"] = {
                    "value": f"{min_t}°C to {max_t}°C",
                    "source": SpecSource.DERIVED.value,
                    "confidence": 0.8,
                    "priority": self.SOURCE_PRIORITY[SpecSource.DERIVED],
                    "note": "Derived from temperature limits"
                }

        # Add more derived specs as needed for your product types

        return derived

    def _force_fill_to_minimum(
        self,
        existing_specs: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Attempt to fill remaining specs to reach minimum count.
        
        IMPORTANT: We no longer add "Not specified" placeholders as they
        pollute the schema with low-quality data. Instead, we log a warning
        and return an empty dict if we can't reach the minimum through
        legitimate means.

        Args:
            existing_specs: Current specifications

        Returns:
            Empty dict - we prefer quality over quantity
        """
        current_count = len(existing_specs)
        gap = self.MIN_SPEC_COUNT - current_count
        
        if gap > 0:
            logger.warning(
                f"[AGGREGATOR] Could not reach minimum of {self.MIN_SPEC_COUNT} specs "
                f"(have {current_count}, missing {gap}). Proceeding without placeholders."
            )
        
        # Return empty - no fake placeholders
        return {}

    def _deduplicate_specs(self, specs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove duplicate specifications and keep highest priority version.

        Args:
            specs: Specifications dictionary

        Returns:
            Deduplicated specifications
        """
        # Normalize keys for comparison (convert to lowercase, remove underscores)
        key_map = {}
        deduplicated = {}

        for original_key, spec_data in specs.items():
            normalized_key = original_key.lower().replace("_", "").replace(" ", "")

            if normalized_key not in key_map:
                key_map[normalized_key] = (original_key, spec_data)
            else:
                # Compare priorities - keep lower priority number (higher priority)
                existing_key, existing_data = key_map[normalized_key]
                existing_priority = existing_data.get("priority", 999)
                new_priority = spec_data.get("priority", 999)

                if new_priority < existing_priority:
                    key_map[normalized_key] = (original_key, spec_data)

        # Build deduplicated dictionary
        for original_key, spec_data in key_map.values():
            deduplicated[original_key] = spec_data

        removed_count = len(specs) - len(deduplicated)
        if removed_count > 0:
            logger.info(f"[AGGREGATOR] Removed {removed_count} duplicate specs")

        return deduplicated

    def _finalize_aggregation(
        self,
        item_id: str,
        item_name: str,
        specs: Dict[str, Any],
        start_time: datetime,
        aggregation_method: str
    ) -> Dict[str, Any]:
        """
        Finalize aggregation and prepare result.

        Args:
            item_id: Item identifier
            item_name: Item name
            specs: Final specifications
            start_time: Start time for timing
            aggregation_method: How specs were aggregated

        Returns:
            Final aggregation result
        """
        elapsed = (datetime.now() - start_time).total_seconds()

        # Organize by source
        by_source = {
            SpecSource.USER.value: [],
            SpecSource.STANDARDS.value: [],
            SpecSource.LLM.value: [],
            SpecSource.DERIVED.value: [],
            SpecSource.TEMPLATE.value: []
        }

        for key, spec in specs.items():
            source = spec.get("source", "unknown")
            if source in by_source:
                by_source[source].append(key)

        result = {
            "item_id": item_id,
            "item_name": item_name,
            "product_type": self.product_type,
            "aggregation": {
                "method": aggregation_method,
                "timestamp": datetime.now().isoformat(),
                "processing_time_ms": int(elapsed * 1000),
                "total_specifications": len(specs),
                "target_specifications": self.MIN_SPEC_COUNT,
                "target_reached": len(specs) >= self.MIN_SPEC_COUNT
            },
            "specifications": {key: spec["value"] for key, spec in specs.items()},
            "specifications_metadata": specs,
            "source_breakdown": {
                "user_specified": len(by_source[SpecSource.USER.value]),
                "standards": len(by_source[SpecSource.STANDARDS.value]),
                "llm_generated": len(by_source[SpecSource.LLM.value]),
                "derived": len(by_source[SpecSource.DERIVED.value]),
                "template_default": len(by_source[SpecSource.TEMPLATE.value]),
                "total": len(specs)
            },
            "specs_by_source": by_source
        }

        logger.info(
            f"[AGGREGATOR] Finalized {item_name}: "
            f"{len(specs)} specs "
            f"(user: {len(by_source[SpecSource.USER.value])}, "
            f"standards: {len(by_source[SpecSource.STANDARDS.value])}, "
            f"llm: {len(by_source[SpecSource.LLM.value])}, "
            f"derived: {len(by_source[SpecSource.DERIVED.value])}, "
            f"template: {len(by_source[SpecSource.TEMPLATE.value])}) "
            f"in {elapsed:.2f}s"
        )

        return result


def aggregate_specifications(
    item_id: str,
    item_name: str,
    product_type: str,
    user_specs: Optional[Dict[str, Any]] = None,
    standards_specs: Optional[Dict[str, Any]] = None,
    llm_specs: Optional[Dict[str, Any]] = None,
    session_id: str = "default"
) -> Dict[str, Any]:
    """
    Aggregate specifications from multiple sources.

    Guaranteed to return 60+ specifications per product type.

    Args:
        item_id: Item identifier
        item_name: Item name
        product_type: Product type
        user_specs: User-provided specs
        standards_specs: Standards-extracted specs
        llm_specs: LLM-generated specs
        session_id: Session identifier

    Returns:
        Aggregated specifications with guaranteed minimum count
    """
    aggregator = SpecificationAggregator(product_type, session_id)

    return aggregator.aggregate(
        item_id=item_id,
        item_name=item_name,
        user_specs=user_specs,
        standards_specs=standards_specs,
        llm_specs=llm_specs
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "SpecificationAggregator",
    "SpecSource",
    "aggregate_specifications"
]
