"""
Agentic Validation Tool
-----------------------
Replica of the /validate API endpoint for use in agentic workflows.

This tool:
1. Detects product type from user input
2. Retrieves schema from database
3. Maps user input to schema structure
4. Validates mandatory fields
5. Returns structured validation result
"""

import json
import logging
import copy
import re
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Import required utilities
from chaining import setup_ai_components, parse_json_response
from models import RequirementValidation
from loading import load_requirements_schema, build_requirements_schema_from_web
import prompts

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationTool:
    """
    Validation tool for agentic workflows

    This tool replicates the functionality of the /validate Flask endpoint
    for use in autonomous agent workflows.
    """

    def __init__(self):
        """Initialize the validation tool with AI components"""
        logger.info("[ValidationTool] Initializing AI components...")
        self.components = setup_ai_components()
        logger.info("[ValidationTool] Validation tool ready")

    def clean_empty_values(self, data: Any) -> Any:
        """
        Recursively replaces 'Not specified', 'Not requested', etc., with empty strings.

        Args:
            data: Dictionary, list, or string to clean

        Returns:
            Cleaned data structure
        """
        if isinstance(data, dict):
            return {k: self.clean_empty_values(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.clean_empty_values(item) for item in data]
        elif isinstance(data, str) and data.lower().strip() in [
            "not specified", "not requested", "none specified", "n/a", "na"
        ]:
            return ""
        return data

    def map_provided_to_schema(self, detected_schema: dict, provided: dict) -> dict:
        """
        Always maps providedRequirements (flat or nested) into the schema structure.
        Works dynamically for any product type.

        Args:
            detected_schema: The detected/loaded schema structure
            provided: The provided requirements from user input

        Returns:
            Mapped requirements following schema structure
        """
        mapped = copy.deepcopy(detected_schema)

        # Case 1: Provided already structured (mandatory/optional) → overlay values
        if "mandatoryRequirements" in provided or "optionalRequirements" in provided:
            for section in ["mandatoryRequirements", "optionalRequirements"]:
                if section in provided and section in mapped:
                    for key, value in provided[section].items():
                        if key in mapped[section]:
                            mapped[section][key] = value
            return mapped

        # Case 2: Provided is flat dict → distribute into schema
        for key, value in provided.items():
            if key in mapped.get("mandatoryRequirements", {}):
                mapped["mandatoryRequirements"][key] = value
            elif key in mapped.get("optionalRequirements", {}):
                mapped["optionalRequirements"][key] = value

        return mapped

    def get_missing_mandatory_fields(self, provided: dict, schema: dict) -> List[str]:
        """
        Get list of missing mandatory fields by traversing the schema.

        Args:
            provided: The provided requirements
            schema: The schema structure

        Returns:
            List of missing mandatory field names
        """
        missing = []
        mandatory_schema = schema.get("mandatoryRequirements", {})
        provided_mandatory = provided.get("mandatoryRequirements", {})

        def traverse_and_check(schema_node, provided_node):
            for key, schema_value in schema_node.items():
                if isinstance(schema_value, dict):
                    traverse_and_check(
                        schema_value,
                        provided_node.get(key, {}) if isinstance(provided_node, dict) else {}
                    )
                else:
                    provided_value = provided_node.get(key) if isinstance(provided_node, dict) else None
                    if provided_value is None or str(provided_value).strip() in ["", ","]:
                        missing.append(key)

        traverse_and_check(mandatory_schema, provided_mandatory)
        return missing

    def friendly_field_name(self, field: str) -> str:
        """
        Convert camelCase field name to friendly label.

        Args:
            field: Field name in camelCase

        Returns:
            Friendly formatted field name
        """
        s1 = re.sub('([a-z0-9])([A-Z])', r'\1 \2', field)
        return s1.replace("_", " ").title()

    def convert_keys_to_camel_case(self, obj: Any) -> Any:
        """
        Recursively converts dictionary keys from snake_case to camelCase.

        Args:
            obj: Dictionary, list, or other object to convert

        Returns:
            Object with camelCase keys
        """
        if isinstance(obj, dict):
            new_dict = {}
            for key, value in obj.items():
                camel_key = re.sub(r'_([a-z])', lambda m: m.group(1).upper(), key)
                new_dict[camel_key] = self.convert_keys_to_camel_case(value)
            return new_dict
        elif isinstance(obj, list):
            return [self.convert_keys_to_camel_case(item) for item in obj]
        return obj

    def validate(
        self,
        user_input: str,
        search_session_id: str = "default",
        reset: bool = False,
        is_repeat: bool = False
    ) -> Dict[str, Any]:
        """
        Main validation function - replicates /validate endpoint functionality.

        Args:
            user_input: User's input text describing product requirements
            search_session_id: Session identifier for tracking (optional)
            reset: Whether to reset previous detection state (optional)
            is_repeat: Whether this is a repeat validation attempt (optional)

        Returns:
            Dictionary containing:
                - productType: Detected product type
                - detectedSchema: Schema structure for the product type
                - providedRequirements: User's requirements mapped to schema
                - validationAlert: Alert if mandatory fields are missing (optional)

        Raises:
            ValueError: If user_input is empty
            Exception: If validation fails
        """
        try:
            # Validate input
            if not user_input or not user_input.strip():
                raise ValueError("user_input is required and cannot be empty")

            logger.info(f"[ValidationTool] Starting validation for session: {search_session_id}")
            logger.info(f"[ValidationTool] User input: {user_input[:100]}...")

            # Step 1: Load initial schema (generic schema for product type detection)
            initial_schema = load_requirements_schema()
            logger.info(f"[ValidationTool] Loaded initial schema")

            # Step 2: First validation pass - detect product type
            # Add session context to prevent cross-contamination
            session_isolated_input = f"[Session: {search_session_id}] - This is a fresh, independent validation request. User input: {user_input}"

            # Prepare validation prompt
            validation_prompt = prompts.validation_prompt.format(
                user_input=session_isolated_input,
                schema=json.dumps(initial_schema, indent=2),
                format_instructions=self.components['validation_format_instructions']
            )

            # Invoke LLM for initial validation
            logger.info(f"[ValidationTool] Invoking LLM for product type detection...")
            llm_response = self.components['llm_flash'].invoke(validation_prompt)
            temp_validation_result = parse_json_response(llm_response, RequirementValidation)

            detected_type = temp_validation_result.get('product_type', 'UnknownProduct')
            logger.info(f"[ValidationTool] Detected product type: {detected_type}")

            # Step 3: Load specific schema for detected product type
            specific_schema = load_requirements_schema(detected_type)

            # If schema doesn't exist, build it from web
            if not specific_schema or not specific_schema.get("mandatory_requirements"):
                logger.info(f"[ValidationTool] No schema found for '{detected_type}', building from web...")
                specific_schema = build_requirements_schema_from_web(detected_type)

            logger.info(f"[ValidationTool] Loaded specific schema for '{detected_type}'")

            # Step 4: Second validation pass with specific schema
            validation_prompt_specific = prompts.validation_prompt.format(
                user_input=session_isolated_input,
                schema=json.dumps(specific_schema, indent=2),
                format_instructions=self.components['validation_format_instructions']
            )

            logger.info(f"[ValidationTool] Invoking LLM for detailed validation...")
            llm_response_specific = self.components['llm_flash'].invoke(validation_prompt_specific)
            validation_result = parse_json_response(llm_response_specific, RequirementValidation)

            # Step 5: Clean empty values from provided requirements
            cleaned_provided_reqs = self.clean_empty_values(
                validation_result.get("provided_requirements", {})
            )

            # Step 6: Map provided requirements to schema structure
            mapped_provided_reqs = self.map_provided_to_schema(
                self.convert_keys_to_camel_case(specific_schema),
                self.convert_keys_to_camel_case(cleaned_provided_reqs)
            )

            # Step 7: Build response data
            response_data = {
                "productType": validation_result.get("product_type", detected_type),
                "detectedSchema": self.convert_keys_to_camel_case(specific_schema),
                "providedRequirements": mapped_provided_reqs
            }

            # Step 8: Check for missing mandatory fields
            missing_mandatory_fields = self.get_missing_mandatory_fields(
                mapped_provided_reqs,
                response_data["detectedSchema"]
            )

            # Step 9: Generate validation alert if mandatory fields are missing
            if missing_mandatory_fields:
                logger.info(f"[ValidationTool] Missing mandatory fields: {missing_mandatory_fields}")

                # Convert missing fields to friendly labels
                missing_fields_friendly = [
                    self.friendly_field_name(f) for f in missing_mandatory_fields
                ]
                missing_fields_str = ", ".join(missing_fields_friendly)

                # Select appropriate prompt based on whether this is a repeat
                if not is_repeat:
                    alert_prompt = prompts.validation_alert_initial_prompt
                else:
                    alert_prompt = prompts.validation_alert_repeat_prompt

                # Generate friendly alert message using LLM
                alert_prompt_text = alert_prompt.format(
                    product_type=response_data["productType"],
                    missing_fields=missing_fields_str
                )

                logger.info(f"[ValidationTool] Generating validation alert message...")
                alert_response = self.components['llm'].invoke(alert_prompt_text)

                response_data["validationAlert"] = {
                    "message": alert_response,
                    "canContinue": True,
                    "missingFields": missing_mandatory_fields
                }

            logger.info(f"[ValidationTool] Validation completed successfully")
            return response_data

        except ValueError as ve:
            logger.error(f"[ValidationTool] Validation error: {ve}")
            raise
        except Exception as e:
            logger.error(f"[ValidationTool] Validation failed: {e}", exc_info=True)
            raise Exception(f"Validation failed: {str(e)}")


# Convenience function for single-use validation
def validate_user_input(
    user_input: str,
    search_session_id: str = "default",
    reset: bool = False,
    is_repeat: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to validate user input without instantiating the tool.

    Args:
        user_input: User's input text describing product requirements
        search_session_id: Session identifier for tracking (optional)
        reset: Whether to reset previous detection state (optional)
        is_repeat: Whether this is a repeat validation attempt (optional)

    Returns:
        Validation result dictionary

    Example:
        >>> result = validate_user_input("I need a pressure transmitter with HART protocol")
        >>> print(result['productType'])
        'Pressure Transmitter'
        >>> print(result['providedRequirements'])
        {...}
    """
    tool = ValidationTool()
    return tool.validate(user_input, search_session_id, reset, is_repeat)


# Example usage for testing
if __name__ == "__main__":
    # Initialize the validation tool
    validator = ValidationTool()

    # Example 1: Basic validation
    print("\n=== Example 1: Basic Validation ===")
    test_input = "I need a pressure transmitter with 4-20mA output and HART protocol"

    try:
        result = validator.validate(test_input)
        print(f"Product Type: {result['productType']}")
        print(f"Provided Requirements: {json.dumps(result['providedRequirements'], indent=2)}")

        if 'validationAlert' in result:
            print(f"\nValidation Alert: {result['validationAlert']['message']}")
            print(f"Missing Fields: {result['validationAlert']['missingFields']}")
    except Exception as e:
        print(f"Error: {e}")

    # Example 2: Using convenience function
    print("\n=== Example 2: Convenience Function ===")
    test_input_2 = "Looking for a butterfly valve DN100 PN16"

    try:
        result_2 = validate_user_input(test_input_2, search_session_id="session_123")
        print(f"Product Type: {result_2['productType']}")
        print(f"Schema Categories: {list(result_2['detectedSchema'].get('mandatoryRequirements', {}).keys())}")
    except Exception as e:
        print(f"Error: {e}")
