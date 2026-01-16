"""
Agentic Advanced Parameters Tool

This tool is a replica of the /api/advanced_parameters Flask endpoint,
designed for use in agentic workflows.

Purpose:
    Discovers latest advanced specifications with series numbers from top vendors
    for a product type. This runs after validation to enhance requirements with
    cutting-edge specifications.

Integration:
    - Used as Step 2 in Product Search Workflow (after Validation, before Sales Agent)
    - Automatically discovers and suggests modern specs not in the base schema
    - Helps users keep up with latest technology trends

Usage:
    tool = AdvancedParametersTool()
    result = tool.discover(product_type="Pressure Transmitter")
"""

import logging
from typing import Any, Dict, List, Optional

# Import the core discovery function
from advanced_parameters import discover_advanced_parameters

logger = logging.getLogger(__name__)


class AdvancedParametersTool:
    """
    Agentic tool for discovering advanced parameters.

    This tool wraps the advanced_parameters module functionality for use
    in agentic workflows, providing a clean interface for parameter discovery.
    """

    def __init__(self):
        """Initialize the advanced parameters tool."""
        logger.info("[AdvancedParametersTool] Tool initialized")

    def discover(
        self,
        product_type: str,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Discover latest advanced specifications for a product type.

        This method:
        1. Checks cache (in-memory and MongoDB)
        2. Gets existing parameters from schema
        3. Uses LLM to discover new advanced parameters
        4. Filters out duplicates
        5. Returns structured list with key/name pairs
        6. Caches results for future use

        Args:
            product_type: The product type to discover parameters for
            session_id: Optional session identifier for logging

        Returns:
            Dictionary containing:
                - product_type: The product type
                - unique_specifications: List of [{"key": str, "name": str}, ...]
                - total_unique_specifications: Count of new specs found
                - existing_specifications_filtered: Count of specs filtered
                - vendor_specifications: Empty list (for compatibility)

        Example:
            >>> tool = AdvancedParametersTool()
            >>> result = tool.discover("Pressure Transmitter")
            >>> print(result['unique_specifications'])
            [
                {
                    "key": "series_5000_hart_protocol",
                    "name": "Series 5000 HART Protocol"
                },
                {
                    "key": "wireless_diagnostics",
                    "name": "Wireless Diagnostics"
                }
            ]
        """
        try:
            log_prefix = f"[Session: {session_id}]" if session_id else ""
            logger.info(
                f"{log_prefix} [AdvancedParametersTool] Starting discovery for: {product_type}"
            )

            # Validate input
            if not product_type or not product_type.strip():
                logger.warning(
                    f"{log_prefix} [AdvancedParametersTool] Missing product_type"
                )
                return {
                    "product_type": "",
                    "unique_specifications": [],
                    "total_unique_specifications": 0,
                    "existing_specifications_filtered": 0,
                    "vendor_specifications": [],
                    "error": "Missing product_type parameter"
                }

            # Call the core discovery function
            result = discover_advanced_parameters(product_type.strip())

            # Log results
            unique_count = len(result.get('unique_specifications', []))
            filtered_count = result.get('existing_specifications_filtered', 0)

            logger.info(
                f"{log_prefix} [AdvancedParametersTool] Discovery complete: "
                f"{unique_count} new specifications found, "
                f"{filtered_count} existing specifications filtered"
            )

            # Add metadata for workflow tracking
            result['session_id'] = session_id
            result['tool_name'] = 'advanced_parameters'
            result['success'] = True

            return result

        except Exception as e:
            logger.error(
                f"[AdvancedParametersTool] Discovery failed for {product_type}: {e}",
                exc_info=True
            )
            return {
                "product_type": product_type,
                "unique_specifications": [],
                "total_unique_specifications": 0,
                "existing_specifications_filtered": 0,
                "vendor_specifications": [],
                "session_id": session_id,
                "tool_name": 'advanced_parameters',
                "success": False,
                "error": str(e)
            }

    def format_for_display(
        self,
        specifications: List[Dict[str, str]]
    ) -> str:
        """
        Format discovered specifications for user-friendly display.

        Args:
            specifications: List of spec dictionaries with 'key' and 'name'

        Returns:
            Formatted string for display to user
        """
        if not specifications:
            return "No new advanced specifications found."

        formatted_specs = []
        for i, spec in enumerate(specifications, 1):
            name = spec.get('name', spec.get('key', 'Unknown'))
            formatted_specs.append(f"{i}. {name}")

        return "\n".join([
            "Discovered the following advanced specifications:",
            "",
            *formatted_specs,
            "",
            "Would you like to add any of these to your requirements?"
        ])

    def invoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        LangChain-compatible invoke method.

        Args:
            input_dict: Dictionary with 'product_type' and optional 'session_id'

        Returns:
            Discovery results dictionary
        """
        product_type = input_dict.get('product_type', '')
        session_id = input_dict.get('session_id')

        return self.discover(product_type=product_type, session_id=session_id)


# Create a singleton instance for easy import
advanced_parameters_tool = AdvancedParametersTool()


# Example usage and testing
if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Test the tool
    tool = AdvancedParametersTool()

    # Test 1: Pressure Transmitter
    print("\n" + "="*70)
    print("TEST 1: Pressure Transmitter")
    print("="*70)
    result = tool.discover("Pressure Transmitter")
    print(f"Success: {result.get('success')}")
    print(f"Found {result.get('total_unique_specifications')} specifications")
    print(f"Filtered {result.get('existing_specifications_filtered')} existing specs")
    print("\nSpecifications:")
    for spec in result.get('unique_specifications', [])[:5]:
        print(f"  - {spec.get('name')} ({spec.get('key')})")

    # Test 2: Format for display
    print("\n" + "="*70)
    print("TEST 2: Format for Display")
    print("="*70)
    formatted = tool.format_for_display(result.get('unique_specifications', []))
    print(formatted)

    # Test 3: LangChain-compatible invoke
    print("\n" + "="*70)
    print("TEST 3: LangChain Invoke")
    print("="*70)
    result = tool.invoke({
        "product_type": "Temperature Transmitter",
        "session_id": "test_session_123"
    })
    print(f"Success: {result.get('success')}")
    print(f"Session ID: {result.get('session_id')}")
    print(f"Found {result.get('total_unique_specifications')} specifications")
