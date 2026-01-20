# =============================================================================
# AUGMENTED PRODUCT TEMPLATES WITH STANDARDS INTEGRATION
# =============================================================================
#
# Phase 3C - Specification Templates Augmented with Standards RAG
#
# This module provides augmented specification templates that combine:
# 1. Base template specifications (30-75 specs per product)
# 2. Standards-derived specifications (15-25 specs per product)
#
# Expected total: 60-90+ specifications per product type
# Expected compliance improvement: 60-70% (vs <20% without standards)
#
# =============================================================================

import logging
from typing import Dict, List, Any, Optional
from agentic.deep_agent.specification_templates import (
    PRODUCT_TYPE_TEMPLATES,
    SpecificationDefinition,
)
from agentic.deep_agent.standards_specs_integration import (
    merge_standards_into_template,
    get_standards_coverage_stats,
)

logger = logging.getLogger(__name__)


# =============================================================================
# AUGMENTED TEMPLATES (Base + Standards)
# =============================================================================

class AugmentedProductTemplates:
    """
    Product templates augmented with standards-derived specifications.

    Each template includes:
    - Base specifications from domain knowledge
    - Standards-derived specifications from ISO/IEC/ANSI standards
    - Proper organization by category and importance
    """

    # Fallback mapping for product types not in templates
    # Maps unknown product types to similar existing templates
    PRODUCT_TYPE_FALLBACK_MAP = {
        # Motor/Drive related -> use control_valve as base (similar electrical/mechanical specs)
        "pump_motor": "control_valve",
        "motor": "control_valve",
        "vfd": "control_valve",
        "motor_vfd": "control_valve",
        "electric_motor": "control_valve",
        # Pump related -> use control_valve (similar flow/pressure specs)
        "rotary_pd_pump": "control_valve",
        "pump": "control_valve",
        "centrifugal_pump": "control_valve",
        "positive_displacement_pump": "control_valve",
        # Transmitter variants -> use pressure_transmitter
        "process_pressure_transmitter": "pressure_transmitter",
        "temperature_transmitter": "pressure_transmitter",
        "flow_transmitter": "flow_meter",
        "differential_pressure_transmitter": "pressure_transmitter",
        # Level related
        "level_sensor": "level_transmitter",
        "level_switch": "level_transmitter",
        # Valve variants
        "relief_valve": "control_valve",
        "pressure_relief_valve": "control_valve",
        "check_valve": "control_valve",
        "isolation_valve": "control_valve",
        # Accessory mappings
        "junction_box": "enclosure",
        "terminal_box": "enclosure",
    }

    @staticmethod
    def _normalize_product_type(product_type: str) -> str:
        """Normalize product type string for template lookup"""
        return product_type.lower().replace(" ", "_").replace("-", "_")

    @staticmethod
    def get_augmented_template(product_type: str) -> Optional[Dict[str, SpecificationDefinition]]:
        """
        Get augmented template (base + standards) for a product type.

        Args:
            product_type: Type of product

        Returns:
            Merged specification dictionary, or None if product type not found
        """
        # Normalize the product type
        normalized_type = AugmentedProductTemplates._normalize_product_type(product_type)
        
        # Get base template
        template_class = PRODUCT_TYPE_TEMPLATES.get(normalized_type)
        
        # Try fallback mapping if not found
        if not template_class:
            fallback_type = AugmentedProductTemplates.PRODUCT_TYPE_FALLBACK_MAP.get(normalized_type)
            if fallback_type:
                template_class = PRODUCT_TYPE_TEMPLATES.get(fallback_type)
                if template_class:
                    logger.info(f"[AugmentedTemplates] Using fallback template: {product_type} -> {fallback_type}")
        
        if not template_class:
            logger.warning(f"[AugmentedTemplates] Product type not found: {product_type} (normalized: {normalized_type})")
            return None

        # Get base specs from template
        base_specs = template_class.get_all_specifications()

        # Merge in standards specs
        augmented = merge_standards_into_template(base_specs, product_type)

        logger.info(f"[AugmentedTemplates] Created augmented template for {product_type}")
        logger.info(f"  Base specs: {len(base_specs)}")
        logger.info(f"  Added standards: {len(augmented) - len(base_specs)}")
        logger.info(f"  Total: {len(augmented)}")

        return augmented

    @staticmethod
    def get_augmented_by_category(product_type: str) -> Optional[Dict[str, List[SpecificationDefinition]]]:
        """
        Get augmented template organized by category.

        Args:
            product_type: Type of product

        Returns:
            Dictionary mapping category names to lists of specifications
        """
        augmented_specs = AugmentedProductTemplates.get_augmented_template(product_type)
        if not augmented_specs:
            return None

        grouped = {}
        for spec in augmented_specs.values():
            if spec.category not in grouped:
                grouped[spec.category] = []
            grouped[spec.category].append(spec)

        return grouped

    @staticmethod
    def get_augmented_by_importance(product_type: str) -> Optional[Dict[str, List[SpecificationDefinition]]]:
        """
        Get augmented template organized by importance level.

        Args:
            product_type: Type of product

        Returns:
            Dictionary mapping importance levels to lists of specifications
        """
        augmented_specs = AugmentedProductTemplates.get_augmented_template(product_type)
        if not augmented_specs:
            return None

        grouped = {}
        for spec in augmented_specs.values():
            imp = spec.importance.value  # Get string value of Enum
            if imp not in grouped:
                grouped[imp] = []
            grouped[imp].append(spec)

        return grouped

    @staticmethod
    def get_augmented_count(product_type: str) -> int:
        """Get total specification count for augmented template"""
        specs = AugmentedProductTemplates.get_augmented_template(product_type)
        return len(specs) if specs else 0

    @staticmethod
    def get_all_augmented_templates() -> Dict[str, Dict[str, SpecificationDefinition]]:
        """Get augmented templates for all 9 product types"""
        templates = {}
        for product_type in PRODUCT_TYPE_TEMPLATES.keys():
            augmented = AugmentedProductTemplates.get_augmented_template(product_type)
            if augmented:
                templates[product_type] = augmented
        return templates


# =============================================================================
# SPECIFICATION VALIDATION & FILTERING
# =============================================================================

def filter_specs_by_importance(
    specs: Dict[str, SpecificationDefinition],
    importance_levels: List[str]
) -> Dict[str, SpecificationDefinition]:
    """
    Filter specifications by importance level.

    Args:
        specs: Specifications to filter
        importance_levels: List of importance levels to include (e.g., ["CRITICAL", "REQUIRED"])

    Returns:
        Filtered specifications dictionary
    """
    filtered = {}
    for key, spec in specs.items():
        if spec.importance.value in importance_levels:
            filtered[key] = spec
    return filtered


def filter_specs_by_category(
    specs: Dict[str, SpecificationDefinition],
    categories: List[str]
) -> Dict[str, SpecificationDefinition]:
    """
    Filter specifications by category.

    Args:
        specs: Specifications to filter
        categories: List of categories to include

    Returns:
        Filtered specifications dictionary
    """
    filtered = {}
    for key, spec in specs.items():
        if spec.category in categories:
            filtered[key] = spec
    return filtered


# =============================================================================
# TEMPLATE STATISTICS & REPORTING
# =============================================================================

def get_template_coverage_report(product_type: str) -> Dict[str, Any]:
    """
    Get comprehensive coverage report for a product type template.

    Includes base template coverage, standards coverage, and merged coverage.

    Args:
        product_type: Type of product

    Returns:
        Dictionary with coverage statistics
    """
    # Get base template
    template_class = PRODUCT_TYPE_TEMPLATES.get(product_type)
    if not template_class:
        return {}

    base_specs = template_class.get_all_specifications()
    augmented_specs = AugmentedProductTemplates.get_augmented_template(product_type)

    if not augmented_specs:
        return {}

    # Count by importance
    base_importance = {}
    for spec in base_specs.values():
        imp = spec.importance.value
        base_importance[imp] = base_importance.get(imp, 0) + 1

    augmented_importance = {}
    for spec in augmented_specs.values():
        imp = spec.importance.value
        augmented_importance[imp] = augmented_importance.get(imp, 0) + 1

    # Count by category
    base_categories = set(spec.category for spec in base_specs.values())
    augmented_categories = set(spec.category for spec in augmented_specs.values())

    return {
        "product_type": product_type,
        "base_template": {
            "total_specs": len(base_specs),
            "importance_distribution": base_importance,
            "categories": len(base_categories),
        },
        "augmented_template": {
            "total_specs": len(augmented_specs),
            "importance_distribution": augmented_importance,
            "categories": len(augmented_categories),
        },
        "added_by_standards": {
            "total_specs": len(augmented_specs) - len(base_specs),
            "new_categories": len(augmented_categories - base_categories),
        },
    }


def print_template_comparison_report():
    """Print side-by-side comparison of base vs augmented templates"""
    print("\n" + "=" * 100)
    print("PRODUCT TEMPLATE COVERAGE REPORT: BASE VS AUGMENTED (WITH STANDARDS)")
    print("=" * 100 + "\n")

    print(f"{'Product Type':<25} {'Base':<12} {'Augmented':<12} {'Added':<12} {'Improvement':<15}")
    print("-" * 100)

    total_base = 0
    total_augmented = 0

    for product_type in sorted(PRODUCT_TYPE_TEMPLATES.keys()):
        report = get_template_coverage_report(product_type)
        if report:
            base_count = report["base_template"]["total_specs"]
            aug_count = report["augmented_template"]["total_specs"]
            added = report["added_by_standards"]["total_specs"]
            improvement = f"+{(added/base_count)*100:.0f}%" if base_count > 0 else "N/A"

            total_base += base_count
            total_augmented += aug_count

            print(
                f"{product_type:<25} {base_count:<12} {aug_count:<12} {added:<12} {improvement:<15}"
            )

    print("-" * 100)
    total_added = total_augmented - total_base
    total_improvement = f"+{(total_added/total_base)*100:.0f}%" if total_base > 0 else "N/A"
    print(
        f"{'TOTAL':<25} {total_base:<12} {total_augmented:<12} {total_added:<12} {total_improvement:<15}"
    )
    print("\n" + "=" * 100 + "\n")


def print_detailed_template_report(product_type: str):
    """Print detailed report for a specific product type"""
    report = get_template_coverage_report(product_type)
    if not report:
        print(f"Product type '{product_type}' not found")
        return

    print(f"\n{'='*80}")
    print(f"DETAILED TEMPLATE REPORT: {product_type.upper()}")
    print(f"{'='*80}\n")

    # Base template
    base = report["base_template"]
    print(f"BASE TEMPLATE:")
    print(f"  Total Specifications: {base['total_specs']}")
    print(f"  Categories: {base['categories']}")
    print(f"  Importance Distribution: {base['importance_distribution']}\n")

    # Augmented template
    aug = report["augmented_template"]
    print(f"AUGMENTED TEMPLATE (WITH STANDARDS):")
    print(f"  Total Specifications: {aug['total_specs']}")
    print(f"  Categories: {aug['categories']}")
    print(f"  Importance Distribution: {aug['importance_distribution']}\n")

    # Comparison
    added = report["added_by_standards"]
    print(f"STANDARDS CONTRIBUTION:")
    print(f"  New Specifications: {added['total_specs']}")
    print(f"  New Categories: {added['new_categories']}")
    improvement = (added['total_specs'] / base['total_specs'] * 100) if base['total_specs'] > 0 else 0
    print(f"  Coverage Improvement: +{improvement:.1f}%\n")

    # Get actual specs to show standards additions
    base_template_class = PRODUCT_TYPE_TEMPLATES.get(product_type)
    if base_template_class:
        base_specs = base_template_class.get_all_specifications()
        augmented_specs = AugmentedProductTemplates.get_augmented_template(product_type)

        if augmented_specs:
            standards_added = set(augmented_specs.keys()) - set(base_specs.keys())
            print(f"STANDARDS-DERIVED SPECIFICATIONS ({len(standards_added)} added):")
            for spec_key in sorted(standards_added):
                spec = augmented_specs[spec_key]
                print(f"  ✓ {spec_key}")
                print(f"    Category: {spec.category}")
                print(f"    Importance: {spec.importance.value}")
                if hasattr(spec, 'source_priority'):
                    print(f"    Source: STANDARDS_RAG (Priority: {spec.source_priority})")

    print(f"\n{'='*80}\n")


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

def get_critical_and_required_specs(product_type: str) -> Dict[str, SpecificationDefinition]:
    """Get only CRITICAL and REQUIRED specifications"""
    specs = AugmentedProductTemplates.get_augmented_template(product_type)
    if not specs:
        return {}
    return filter_specs_by_importance(specs, ["CRITICAL", "REQUIRED"])


def get_mandatory_specifications_count(product_type: str) -> int:
    """Get count of CRITICAL + REQUIRED specifications"""
    return len(get_critical_and_required_specs(product_type))


def get_optional_specifications(product_type: str) -> Dict[str, SpecificationDefinition]:
    """Get OPTIONAL and IMPORTANT specifications"""
    specs = AugmentedProductTemplates.get_augmented_template(product_type)
    if not specs:
        return {}
    return filter_specs_by_importance(specs, ["OPTIONAL", "IMPORTANT"])


# =============================================================================
# INITIALIZATION & CACHING
# =============================================================================

_augmented_template_cache = {}


def get_cached_augmented_template(product_type: str) -> Optional[Dict[str, SpecificationDefinition]]:
    """Get augmented template with caching"""
    if product_type not in _augmented_template_cache:
        template = AugmentedProductTemplates.get_augmented_template(product_type)
        if template:
            _augmented_template_cache[product_type] = template
    return _augmented_template_cache.get(product_type)


def clear_augmented_template_cache():
    """Clear the augmented template cache"""
    global _augmented_template_cache
    _augmented_template_cache.clear()
    logger.info("[AugmentedTemplates] Cache cleared")


def preload_all_augmented_templates():
    """Pre-load all augmented templates into cache"""
    for product_type in PRODUCT_TYPE_TEMPLATES.keys():
        get_cached_augmented_template(product_type)
    logger.info(f"[AugmentedTemplates] Pre-loaded {len(_augmented_template_cache)} augmented templates")
