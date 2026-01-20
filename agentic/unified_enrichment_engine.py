# agentic/unified_enrichment_engine.py
# =============================================================================
# UNIFIED SPECIFICATION ENRICHMENT ENGINE
# =============================================================================
#
# Provides a unified framework for generating 60+ specifications across:
# - PPI Workflow (Product Index schema generation)
# - Product Search Workflow (validation and enrichment)
# - Solution Workflow (user-centric specification extraction)
#
# This module bridges the gap between different workflows to ensure
# consistent, high-quality 60+ specifications everywhere.
#
# =============================================================================

import logging
import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from llm_fallback import create_llm_with_fallback

logger = logging.getLogger(__name__)


class UnifiedEnrichmentEngine:
    """
    Unified engine for 60+ specification generation from multiple sources.

    Sources (in priority order):
    1. User-specified (confidence: 1.0) - HIGHEST PRIORITY
    2. Standards-based (confidence: 0.9)
    3. Vendor data (confidence: 0.7-0.8)
    4. LLM-generated (confidence: 0.8)
    5. Template defaults (confidence: 0.5) - FALLBACK
    """

    def __init__(self, product_type: str, session_id: Optional[str] = None):
        """
        Initialize the enrichment engine.

        Args:
            product_type: Type of product (e.g., "Temperature Sensor")
            session_id: Optional session ID for tracking
        """
        self.product_type = product_type
        self.session_id = session_id or f"enrichment_{datetime.now().isoformat()}"
        logger.info(f"[ENRICHMENT_ENGINE] Initialized for {product_type}")

    def enrich_from_multiple_sources(
        self,
        user_input: Optional[str] = None,
        user_specs: Optional[Dict[str, Any]] = None,
        vendor_data: Optional[List[Dict[str, Any]]] = None,
        standards_context: Optional[str] = None,
        target_spec_count: int = 60
    ) -> Dict[str, Any]:
        """
        Enrich specifications from multiple sources to reach target count.

        Args:
            user_input: Raw user description/requirements
            user_specs: Structured user specifications
            vendor_data: Extracted vendor data from PDFs
            standards_context: Standards-relevant context
            target_spec_count: Minimum target (default: 60)

        Returns:
            {
                "specifications": {...},  # Dict of all 60+ specs
                "aggregation": {
                    "total_specifications": 60,
                    "spec_sources": {
                        "user": 5,
                        "standards": 12,
                        "vendor": 15,
                        "llm": 20,
                        "template": 8
                    },
                    "confidence_scores": {...}
                },
                "metadata": {
                    "product_type": "...",
                    "generation_time": "...",
                    "session_id": "..."
                }
            }
        """
        logger.info(f"[ENRICHMENT_ENGINE] Starting enrichment with target {target_spec_count} specs")

        all_specs = {}
        spec_sources = {}

        # Step 1: Extract user specifications (highest priority)
        user_extracted_specs = {}
        if user_input or user_specs:
            logger.info("[ENRICHMENT_ENGINE] Step 1: Extracting user specifications...")
            user_extracted_specs = self._extract_user_specs(user_input, user_specs)
            all_specs.update(user_extracted_specs)
            spec_sources["user"] = len(user_extracted_specs)
            logger.info(f"[ENRICHMENT_ENGINE] ✓ Extracted {len(user_extracted_specs)} user specs")

        # Step 2: Extract standards specifications
        logger.info("[ENRICHMENT_ENGINE] Step 2: Extracting standards specifications...")
        standards_specs = self._extract_standards_specs(standards_context)
        # Only add standards specs that don't conflict with user specs
        for key, val in standards_specs.items():
            if key not in all_specs:
                all_specs[key] = val
        spec_sources["standards"] = len(standards_specs)
        logger.info(f"[ENRICHMENT_ENGINE] ✓ Extracted {len(standards_specs)} standards specs")

        # Step 3: Extract vendor data specifications
        vendor_specs = {}
        if vendor_data:
            logger.info("[ENRICHMENT_ENGINE] Step 3: Extracting vendor specifications...")
            vendor_specs = self._extract_vendor_specs(vendor_data)
            for key, val in vendor_specs.items():
                if key not in all_specs:
                    all_specs[key] = val
            spec_sources["vendor"] = len(vendor_specs)
            logger.info(f"[ENRICHMENT_ENGINE] ✓ Extracted {len(vendor_specs)} vendor specs")

        # Step 4: Generate LLM specifications (if still below target)
        current_count = len(all_specs)
        if current_count < target_spec_count:
            gap = target_spec_count - current_count
            logger.info(f"[ENRICHMENT_ENGINE] Step 4: Gap {gap} specs, generating with LLM...")

            llm_specs = self._generate_llm_specs(
                product_type=self.product_type,
                existing_specs=all_specs,
                target_gap=gap
            )

            for key, val in llm_specs.items():
                if key not in all_specs:
                    all_specs[key] = val
            spec_sources["llm"] = len(llm_specs)
            logger.info(f"[ENRICHMENT_ENGINE] ✓ Generated {len(llm_specs)} LLM specs")

        # Step 5: Fill remaining gaps from templates
        current_count = len(all_specs)
        if current_count < target_spec_count:
            gap = target_spec_count - current_count
            logger.info(f"[ENRICHMENT_ENGINE] Step 5: Still gap {gap} specs, filling from template...")

            template_specs = self._get_template_specs(gap)
            for key, val in template_specs.items():
                if key not in all_specs:
                    all_specs[key] = val
            spec_sources["template"] = len(template_specs)
            logger.info(f"[ENRICHMENT_ENGINE] ✓ Filled {len(template_specs)} template specs")

        # Final validation
        final_count = len(all_specs)
        if final_count < target_spec_count:
            logger.warning(f"[ENRICHMENT_ENGINE] ⚠️ Final count {final_count} < target {target_spec_count}")
            # This shouldn't happen with proper aggregator, but warn anyway
        else:
            logger.info(f"[ENRICHMENT_ENGINE] ✅ Reached target: {final_count} specs >= {target_spec_count}")

        return {
            "specifications": all_specs,
            "aggregation": {
                "total_specifications": len(all_specs),
                "target_specifications": target_spec_count,
                "spec_sources": spec_sources,
                "gap_filled": len(all_specs) >= target_spec_count
            },
            "metadata": {
                "product_type": self.product_type,
                "generation_time": datetime.now().isoformat(),
                "session_id": self.session_id
            }
        }

    def _extract_user_specs(
        self,
        user_input: Optional[str],
        user_specs: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract specifications provided directly by user."""
        extracted = {}

        if user_specs:
            # User specs are already structured, just add with high confidence
            for key, val in user_specs.items():
                extracted[key] = {
                    "value": val,
                    "source": "user_specified",
                    "confidence": 1.0
                }

        if user_input and not user_specs:
            # Parse user input text to extract specs
            try:
                from agentic.deep_agent.llm_specs_generator import extract_user_specified_specs
                extracted_from_text = extract_user_specified_specs(user_input, self.product_type)
                for key, val in extracted_from_text.items():
                    extracted[key] = {
                        "value": val,
                        "source": "user_input_extracted",
                        "confidence": 0.9
                    }
            except Exception as e:
                logger.warning(f"[ENRICHMENT_ENGINE] Failed to extract user specs from text: {e}")

        return extracted

    def _extract_standards_specs(self, standards_context: Optional[str]) -> Dict[str, Any]:
        """Extract specifications from standards documents."""
        extracted = {}

        if not standards_context:
            return extracted

        try:
            from agentic.standards_rag.standards_rag_enrichment import StandardsEnrichment

            enricher = StandardsEnrichment()
            standards_result = enricher.enrich_with_standards(
                product_type=self.product_type,
                context=standards_context
            )

            if standards_result and standards_result.get("specifications"):
                for key, val in standards_result["specifications"].items():
                    extracted[key] = {
                        "value": val,
                        "source": "standards_rag",
                        "confidence": 0.9
                    }
        except Exception as e:
            logger.warning(f"[ENRICHMENT_ENGINE] Failed to extract standards specs: {e}")

        return extracted

    def _extract_vendor_specs(self, vendor_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract specifications from vendor data (PDFs)."""
        extracted = {}

        if not vendor_data:
            return extracted

        try:
            from test import aggregate_results

            aggregated = aggregate_results(vendor_data, self.product_type)

            # Extract specifications from aggregated vendor data
            for model in aggregated.get("models", []):
                for key, val in model.get("specifications", {}).items():
                    if key not in extracted:  # Keep first (best) match
                        extracted[key] = {
                            "value": val,
                            "source": "vendor_data",
                            "confidence": 0.7
                        }
        except Exception as e:
            logger.warning(f"[ENRICHMENT_ENGINE] Failed to extract vendor specs: {e}")

        return extracted

    def _generate_llm_specs(
        self,
        product_type: str,
        existing_specs: Dict[str, Any],
        target_gap: int
    ) -> Dict[str, Any]:
        """Generate specifications using LLM to fill gaps."""
        generated = {}

        try:
            from agentic.deep_agent.llm_specs_generator import generate_llm_specs

            existing_keys = set(existing_specs.keys())
            context = f"Already have {len(existing_keys)} specs: {', '.join(list(existing_keys)[:10])}"

            llm_result = generate_llm_specs(
                product_type=product_type,
                category=product_type,
                context=context
            )

            raw_specs = llm_result.get("specifications", {})

            # Only add new specs (not already present)
            for key, spec_data in raw_specs.items():
                if key not in existing_specs:
                    generated[key] = {
                        "value": spec_data.get("value", ""),
                        "source": "llm_generated",
                        "confidence": spec_data.get("confidence", 0.8)
                    }

                    if len(generated) >= target_gap:
                        break
        except Exception as e:
            logger.warning(f"[ENRICHMENT_ENGINE] Failed to generate LLM specs: {e}")

        return generated

    def _get_template_specs(self, target_count: int) -> Dict[str, Any]:
        """Get template specifications to fill remaining gaps."""
        specs = {}

        try:
            from agentic.deep_agent.specification_templates import (
                TemperatureSensorTemplate,
                PressureTransmitterTemplate,
            )

            # Map product type to template
            templates = {
                "temperature_sensor": TemperatureSensorTemplate,
                "pressure_transmitter": PressureTransmitterTemplate,
                # Add more as needed
            }

            template_class = templates.get(self.product_type.lower().replace(" ", "_"))
            if template_class:
                template_specs = template_class.get_all_specifications()

                # Take up to target_count specs
                for i, (key, spec) in enumerate(template_specs.items()):
                    if i >= target_count:
                        break
                    specs[key] = {
                        "value": spec.get("typical_value", "Not specified"),
                        "source": "template_default",
                        "confidence": 0.5
                    }
        except Exception as e:
            logger.warning(f"[ENRICHMENT_ENGINE] Failed to get template specs: {e}")

        return specs


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def enrich_product_with_60_plus_specs(
    product_type: str,
    user_input: Optional[str] = None,
    user_specs: Optional[Dict[str, Any]] = None,
    vendor_data: Optional[List[Dict[str, Any]]] = None,
    standards_context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to enrich a product with 60+ specifications.

    Usage:
        result = enrich_product_with_60_plus_specs(
            product_type="Pressure Transmitter",
            user_input="SIL2, 4-20mA output, stainless steel",
            vendor_data=[...]
        )
        print(f"Generated {len(result['specifications'])} specs")
    """
    engine = UnifiedEnrichmentEngine(product_type)
    return engine.enrich_from_multiple_sources(
        user_input=user_input,
        user_specs=user_specs,
        vendor_data=vendor_data,
        standards_context=standards_context
    )


# =============================================================================
# INTEGRATION WITH EXISTING WORKFLOWS
# =============================================================================

def integrate_with_ppi_workflow():
    """
    Integration points for PPI Workflow:

    In ppi_workflow.py generate_schema_node():

    from agentic.unified_enrichment_engine import enrich_product_with_60_plus_specs

    # After collecting vendor data
    enrichment_result = enrich_product_with_60_plus_specs(
        product_type=product_type,
        vendor_data=extracted_data,
        standards_context=f"Industrial instruments for {product_type}"
    )

    schema["aggregated_specifications"] = enrichment_result["specifications"]
    """
    pass


def integrate_with_product_search_workflow():
    """
    Integration points for Product Search Workflow:

    In validation_tool.py validate():

    from agentic.unified_enrichment_engine import enrich_product_with_60_plus_specs

    # After loading schema
    enrichment_result = enrich_product_with_60_plus_specs(
        product_type=product_type,
        user_input=user_input,
        standards_context=product_type
    )

    schema["specifications"] = enrichment_result["specifications"]
    """
    pass


def integrate_with_solution_workflow():
    """
    Integration points for Solution Workflow:

    In solution_workflow.py enrich_solution_with_standards_node():

    from agentic.unified_enrichment_engine import enrich_product_with_60_plus_specs

    # For each item identified
    for item in items:
        enrichment_result = enrich_product_with_60_plus_specs(
            product_type=item["type"],
            user_input=solution_context,
            standards_context=solution_domain
        )

        item["specifications"] = enrichment_result["specifications"]
    """
    pass


if __name__ == "__main__":
    # Example usage
    result = enrich_product_with_60_plus_specs(
        product_type="Temperature Sensor",
        user_specs={
            "accuracy": "±0.1%",
            "range": "-50 to +500°C",
            "output": "4-20mA"
        }
    )

    print(f"\n✅ Generated {result['aggregation']['total_specifications']} specifications")
    print(f"Sources: {result['aggregation']['spec_sources']}")
