"""
Input Validation Utilities for Tools

Provides decorators and validators to ensure tool inputs are valid
before processing, preventing runtime errors.
"""

import functools
import logging
from typing import Any, Callable, Dict, List, Optional, Union
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class ValidationException(Exception):
    """Raised when input validation fails"""
    pass


def validate_not_empty(value: Any, field_name: str) -> None:
    """
    Validate that a value is not None or empty.

    Args:
        value: Value to validate
        field_name: Name of the field for error messages

    Raises:
        ValidationException: If value is None or empty
    """
    if value is None:
        raise ValidationException(f"{field_name} cannot be None")

    if isinstance(value, (str, list, dict)) and not value:
        raise ValidationException(f"{field_name} cannot be empty")


def validate_string(value: Any, field_name: str, min_length: int = 1, max_length: Optional[int] = None) -> None:
    """
    Validate string input.

    Args:
        value: Value to validate
        field_name: Name of the field
        min_length: Minimum length
        max_length: Maximum length (optional)

    Raises:
        ValidationException: If validation fails
    """
    if not isinstance(value, str):
        raise ValidationException(f"{field_name} must be a string, got {type(value).__name__}")

    if len(value) < min_length:
        raise ValidationException(f"{field_name} must be at least {min_length} characters")

    if max_length and len(value) > max_length:
        raise ValidationException(f"{field_name} must be at most {max_length} characters")


def validate_list(value: Any, field_name: str, min_items: int = 0, max_items: Optional[int] = None,
                   item_type: Optional[type] = None) -> None:
    """
    Validate list input.

    Args:
        value: Value to validate
        field_name: Name of the field
        min_items: Minimum number of items
        max_items: Maximum number of items (optional)
        item_type: Expected type of items (optional)

    Raises:
        ValidationException: If validation fails
    """
    if not isinstance(value, list):
        raise ValidationException(f"{field_name} must be a list, got {type(value).__name__}")

    if len(value) < min_items:
        raise ValidationException(f"{field_name} must have at least {min_items} items")

    if max_items and len(value) > max_items:
        raise ValidationException(f"{field_name} must have at most {max_items} items")

    if item_type:
        for i, item in enumerate(value):
            if not isinstance(item, item_type):
                raise ValidationException(
                    f"{field_name}[{i}] must be {item_type.__name__}, got {type(item).__name__}"
                )


def validate_dict(value: Any, field_name: str, required_keys: Optional[List[str]] = None) -> None:
    """
    Validate dictionary input.

    Args:
        value: Value to validate
        field_name: Name of the field
        required_keys: List of required keys (optional)

    Raises:
        ValidationException: If validation fails
    """
    if not isinstance(value, dict):
        raise ValidationException(f"{field_name} must be a dict, got {type(value).__name__}")

    if required_keys:
        missing_keys = [key for key in required_keys if key not in value]
        if missing_keys:
            raise ValidationException(f"{field_name} missing required keys: {missing_keys}")


def validate_numeric(value: Any, field_name: str, min_value: Optional[float] = None,
                      max_value: Optional[float] = None) -> None:
    """
    Validate numeric input.

    Args:
        value: Value to validate
        field_name: Name of the field
        min_value: Minimum value (optional)
        max_value: Maximum value (optional)

    Raises:
        ValidationException: If validation fails
    """
    if not isinstance(value, (int, float)):
        raise ValidationException(f"{field_name} must be numeric, got {type(value).__name__}")

    if min_value is not None and value < min_value:
        raise ValidationException(f"{field_name} must be at least {min_value}")

    if max_value is not None and value > max_value:
        raise ValidationException(f"{field_name} must be at most {max_value}")


def validate_tool_input(schema: BaseModel):
    """
    Decorator to validate tool input against Pydantic schema.

    Args:
        schema: Pydantic BaseModel class

    Usage:
        @validate_tool_input(MyToolInput)
        @tool
        def my_tool(arg1: str, arg2: int):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Validate kwargs against schema
                if kwargs:
                    validated = schema(**kwargs)
                    kwargs = validated.dict()
                return func(*args, **kwargs)
            except ValidationError as e:
                logger.error(f"Input validation failed for {func.__name__}: {e}")
                raise ValidationException(f"Invalid input: {str(e)}")
        return wrapper
    return decorator


def safe_get(data: dict, key: str, default: Any = None, required: bool = False) -> Any:
    """
    Safely get value from dictionary with validation.

    Args:
        data: Dictionary to get value from
        key: Key to retrieve
        default: Default value if key missing
        required: If True, raise exception if key missing

    Returns:
        Value from dictionary or default

    Raises:
        ValidationException: If required=True and key missing
    """
    if key not in data:
        if required:
            raise ValidationException(f"Required key '{key}' missing from input")
        return default

    return data[key]


def validate_vendor_list(vendors: List[str], field_name: str = "vendors") -> None:
    """
    Validate vendor list input.

    Args:
        vendors: List of vendor names
        field_name: Name of the field

    Raises:
        ValidationException: If validation fails
    """
    validate_list(vendors, field_name, min_items=1, item_type=str)

    for vendor in vendors:
        if not vendor.strip():
            raise ValidationException(f"{field_name} contains empty vendor name")


def validate_product_type(product_type: str) -> None:
    """
    Validate product type input.

    Args:
        product_type: Product type string

    Raises:
        ValidationException: If validation fails
    """
    validate_string(product_type, "product_type", min_length=2, max_length=100)

    # Product type should not contain special characters that could cause issues
    invalid_chars = ['<', '>', ';', '&', '|', '$']
    for char in invalid_chars:
        if char in product_type:
            raise ValidationException(f"product_type contains invalid character: {char}")


def validate_requirements(requirements: Union[str, dict], field_name: str = "requirements") -> None:
    """
    Validate requirements input.

    Args:
        requirements: Requirements as string or dict
        field_name: Name of the field

    Raises:
        ValidationException: If validation fails
    """
    if isinstance(requirements, str):
        validate_string(requirements, field_name, min_length=5)
    elif isinstance(requirements, dict):
        validate_dict(requirements, field_name)
        if not requirements:
            raise ValidationException(f"{field_name} dictionary cannot be empty")
    else:
        raise ValidationException(
            f"{field_name} must be string or dict, got {type(requirements).__name__}"
        )


__all__ = [
    'ValidationException',
    'validate_not_empty',
    'validate_string',
    'validate_list',
    'validate_dict',
    'validate_numeric',
    'validate_tool_input',
    'safe_get',
    'validate_vendor_list',
    'validate_product_type',
    'validate_requirements',
]
