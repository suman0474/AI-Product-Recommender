"""
Advanced Parameters Discovery Tool for Product Search Workflow
===============================================================

Step 2 of Product Search Workflow:
- Discovers latest advanced specifications from top vendors
- Includes series numbers and model information
- Filters against existing schema (no duplicates)
- Returns structured list of unique specifications

This tool uses the existing advanced_parameters.py implementation
and wraps it for agentic workflow usage.
"""

import logging
from typing import Dict, Any, List, Optional

# Configure logging
logger = logging.getLogger(__name__)


class AdvancedParametersTool:
    """
    Advanced Parameters Discovery Tool - Step 2 of Product Search Workflow

    Responsibilities:
    1. Discover latest advanced specifications from vendor PDFs
    2. Include series numbers and model information
    3. Filter against existing schema to avoid duplicates
    4. Return structured, deduplicated list of specifications

    Integration:
    - Uses advanced_parameters.py for core discovery logic
    - Caches results in MongoDB for performance
    - Supports multi-vendor parallel discovery
    """

    def __init__(self):
        """Initialize the advanced parameters discovery tool."""
        logger.info("[AdvancedParametersTool] Initialized")

    def discover(
        self,
        product_type: str,
        session_id: Optional[str] = None,
        existing_schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Discover advanced parameters for a product type.

        Args:
            product_type: Product type to discover parameters for
            session_id: Session identifier (for logging/tracking)
            existing_schema: Optional existing schema to filter against

        Returns:
            Discovery result with:
            {
                "success": bool,
                "product_type": str,
                "unique_specifications": [
                    {
                        "key": "parameter_key",
                        "name": "Human Readable Name",
                        "description": "Parameter description",
                        "vendor": "Vendor name",
                        "series": "Series/Model info"
                    }
                ],
                "total_unique_specifications": int,
                "existing_specifications_filtered": int,
                "vendors_searched": list,
                "session_id": str
            }
        """
        logger.info("[AdvancedParametersTool] Starting discovery")
        logger.info("[AdvancedParametersTool] Product Type: %s", product_type)
        logger.info("[AdvancedParametersTool] Session: %s", session_id or "N/A")

        result = {
            "success": False,
            "product_type": product_type,
            "session_id": session_id
        }

        try:
            # Import the core discovery function
            from advanced_parameters import discover_advanced_parameters

            # =================================================================
            # DISCOVER ADVANCED PARAMETERS
            # =================================================================
            logger.info("[AdvancedParametersTool] Invoking core discovery function")

            discovery_result = discover_advanced_parameters(
                product_type=product_type.strip()
            )

            # Extract results
            unique_specs = discovery_result.get('unique_specifications', [])
            existing_filtered = discovery_result.get('existing_specifications_filtered', 0)
            vendors_searched = discovery_result.get('vendors_searched', [])

            specs_count = len(unique_specs)

            # Log results
            if specs_count > 0:
                logger.info("[AdvancedParametersTool] ✓ Discovered %d unique specifications",
                           specs_count)
                logger.info("[AdvancedParametersTool] ℹ Filtered %d existing specifications",
                           existing_filtered)
                logger.info("[AdvancedParametersTool] ℹ Searched %d vendors",
                           len(vendors_searched))

                # Log first few specifications
                for i, spec in enumerate(unique_specs[:3], 1):
                    logger.info("[AdvancedParametersTool]   %d. %s",
                               i, spec.get('name', spec.get('key')))

                if specs_count > 3:
                    logger.info("[AdvancedParametersTool]   ... and %d more",
                               specs_count - 3)
            else:
                logger.info("[AdvancedParametersTool] ℹ No new specifications discovered")

            # =================================================================
            # BUILD RESULT
            # =================================================================
            result.update({
                "success": True,
                "unique_specifications": unique_specs,
                "total_unique_specifications": specs_count,
                "existing_specifications_filtered": existing_filtered,
                "vendors_searched": vendors_searched,
                "discovery_successful": specs_count > 0
            })

            logger.info("[AdvancedParametersTool] ✓ Discovery completed successfully")
            return result

        except Exception as e:
            logger.error("[AdvancedParametersTool] ✗ Discovery failed: %s",
                        e, exc_info=True)
            result.update({
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "unique_specifications": [],
                "total_unique_specifications": 0
            })
            return result

    def discover_with_filtering(
        self,
        product_type: str,
        existing_schema: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Discover parameters with explicit schema-based filtering.

        Args:
            product_type: Product type to discover parameters for
            existing_schema: Schema to filter against
            session_id: Session identifier

        Returns:
            Discovery result with filtered specifications
        """
        logger.info("[AdvancedParametersTool] Discovery with explicit filtering")

        # Perform standard discovery
        result = self.discover(
            product_type=product_type,
            session_id=session_id
        )

        if not result['success']:
            return result

        # Additional filtering against provided schema
        unique_specs = result.get('unique_specifications', [])
        schema_keys = set()

        # Extract all keys from schema
        if existing_schema:
            schema_keys.update(existing_schema.get('mandatory', {}).keys())
            schema_keys.update(existing_schema.get('optional', {}).keys())
            schema_keys.update(existing_schema.get('mandatory_requirements', {}).keys())
            schema_keys.update(existing_schema.get('optional_requirements', {}).keys())

        # Filter specifications
        filtered_specs = []
        for spec in unique_specs:
            spec_key = spec.get('key', '').lower()
            if spec_key not in schema_keys:
                filtered_specs.append(spec)

        additional_filtered = len(unique_specs) - len(filtered_specs)

        if additional_filtered > 0:
            logger.info("[AdvancedParametersTool] ℹ Filtered %d additional specs against provided schema",
                       additional_filtered)

        result['unique_specifications'] = filtered_specs
        result['total_unique_specifications'] = len(filtered_specs)
        result['existing_specifications_filtered'] += additional_filtered

        return result

    def format_for_display(
        self,
        specifications: List[Dict[str, Any]],
        max_items: Optional[int] = None
    ) -> str:
        """
        Format specifications for user-friendly display.

        Args:
            specifications: List of specification dictionaries
            max_items: Maximum items to include (None for all)

        Returns:
            Formatted string for display
        """
        if not specifications:
            return "No specifications available."

        specs_to_show = specifications[:max_items] if max_items else specifications
        remaining = len(specifications) - len(specs_to_show)

        lines = []
        for i, spec in enumerate(specs_to_show, 1):
            name = spec.get('name', spec.get('key', 'Unknown'))
            description = spec.get('description', '')
            vendor = spec.get('vendor', '')

            line = f"{i}. {name}"
            if vendor:
                line += f" ({vendor})"
            if description:
                line += f" - {description}"

            lines.append(line)

        if remaining > 0:
            lines.append(f"... and {remaining} more specifications")

        return "\n".join(lines)

    def get_specification_keys(
        self,
        specifications: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Extract specification keys from discovery result.

        Args:
            specifications: List of specification dictionaries

        Returns:
            List of specification keys
        """
        return [spec.get('key', '') for spec in specifications if spec.get('key')]

    def get_specifications_by_vendor(
        self,
        specifications: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group specifications by vendor.

        Args:
            specifications: List of specification dictionaries

        Returns:
            Dictionary mapping vendor names to their specifications
        """
        by_vendor = {}

        for spec in specifications:
            vendor = spec.get('vendor', 'Unknown')
            if vendor not in by_vendor:
                by_vendor[vendor] = []
            by_vendor[vendor].append(spec)

        return by_vendor


# ============================================================================
# STANDALONE USAGE EXAMPLE
# ============================================================================

def example_usage():
    """Example usage of AdvancedParametersTool"""
    print("\n" + "="*70)
    print("ADVANCED PARAMETERS TOOL - STANDALONE EXAMPLE")
    print("="*70)

    # Initialize tool
    tool = AdvancedParametersTool()

    # Example 1: Discover parameters
    print("\n[Example 1] Discover advanced parameters:")
    result = tool.discover(
        product_type="pressure transmitter",
        session_id="test_session_001"
    )

    print(f"✓ Success: {result['success']}")
    print(f"✓ Product Type: {result.get('product_type')}")
    print(f"✓ Unique Specifications: {result.get('total_unique_specifications')}")
    print(f"✓ Existing Filtered: {result.get('existing_specifications_filtered')}")
    print(f"✓ Vendors Searched: {len(result.get('vendors_searched', []))}")

    # Example 2: Format for display
    if result['success'] and result.get('unique_specifications'):
        print("\n[Example 2] Format specifications:")
        specs = result['unique_specifications']
        formatted = tool.format_for_display(specs, max_items=5)
        print(formatted)

        # Example 3: Group by vendor
        print("\n[Example 3] Group by vendor:")
        by_vendor = tool.get_specifications_by_vendor(specs)
        for vendor, vendor_specs in by_vendor.items():
            print(f"  {vendor}: {len(vendor_specs)} specifications")


if __name__ == "__main__":
    # Configure logging for standalone testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    example_usage()
