# agentic/exceptions.py
"""
Custom Exception Classes for Agentic Workflows

This module defines custom exception classes for better error handling
and debugging across all agentic workflows.
"""


class AgenticException(Exception):
    """Base exception class for all agentic workflow errors"""
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ConfigurationError(AgenticException):
    """Raised when configuration is missing or invalid"""
    pass


class APIKeyError(ConfigurationError):
    """Raised when API key is missing or invalid"""
    pass


class WorkflowError(AgenticException):
    """Base exception for workflow execution errors"""
    pass


class WorkflowTimeoutError(WorkflowError):
    """Raised when a workflow operation times out"""
    pass


class WorkflowValidationError(WorkflowError):
    """Raised when workflow input validation fails"""
    pass


class LLMError(AgenticException):
    """Base exception for LLM-related errors"""
    pass


class LLMTimeoutError(LLMError):
    """Raised when LLM request times out"""
    pass


class LLMRateLimitError(LLMError):
    """Raised when LLM rate limit is exceeded"""
    pass


class VendorAnalysisError(WorkflowError):
    """Raised when vendor analysis fails"""
    pass


class VendorAnalysisTimeoutError(VendorAnalysisError, WorkflowTimeoutError):
    """Raised when vendor analysis times out"""
    pass


class RAGError(AgenticException):
    """Base exception for RAG-related errors"""
    pass


class RAGRetrievalError(RAGError):
    """Raised when RAG retrieval fails"""
    pass


class RAGGenerationError(RAGError):
    """Raised when RAG generation fails"""
    pass


class ChromaDBError(AgenticException):
    """Base exception for ChromaDB-related errors"""
    pass


class ChromaDBConnectionError(ChromaDBError):
    """Raised when ChromaDB connection fails"""
    pass


class ChromaDBTimeoutError(ChromaDBError, WorkflowTimeoutError):
    """Raised when ChromaDB operation times out"""
    pass


class SchemaError(AgenticException):
    """Base exception for schema-related errors"""
    pass


class SchemaNotFoundError(SchemaError):
    """Raised when product schema is not found"""
    pass


class SchemaValidationError(SchemaError):
    """Raised when schema validation fails"""
    pass


class ToolExecutionError(AgenticException):
    """Raised when tool execution fails"""
    pass


class InstrumentIdentificationError(WorkflowError):
    """Raised when instrument identification fails"""
    pass


class RoutingError(WorkflowError):
    """Raised when workflow routing fails"""
    pass


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_error_message(error: Exception, context: dict = None) -> dict:
    """
    Format exception for consistent error responses.

    Args:
        error: The exception to format
        context: Optional context dictionary

    Returns:
        Formatted error dictionary
    """
    error_dict = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "success": False
    }

    if isinstance(error, AgenticException) and error.details:
        error_dict["details"] = error.details

    if context:
        error_dict["context"] = context

    return error_dict


def handle_workflow_error(error: Exception, workflow_name: str) -> dict:
    """
    Handle workflow errors with proper logging and formatting.

    Args:
        error: The exception that occurred
        workflow_name: Name of the workflow

    Returns:
        Formatted error response
    """
    import logging
    logger = logging.getLogger(__name__)

    context = {"workflow": workflow_name}

    # Log based on error type
    if isinstance(error, (WorkflowTimeoutError, LLMTimeoutError, ChromaDBTimeoutError)):
        logger.warning(f"[{workflow_name}] Timeout error: {error}")
    elif isinstance(error, ConfigurationError):
        logger.error(f"[{workflow_name}] Configuration error: {error}")
    elif isinstance(error, LLMRateLimitError):
        logger.warning(f"[{workflow_name}] Rate limit exceeded: {error}")
    else:
        logger.error(f"[{workflow_name}] Unexpected error: {error}", exc_info=True)

    return format_error_message(error, context)


__all__ = [
    # Base exceptions
    "AgenticException",
    "ConfigurationError",
    "APIKeyError",

    # Workflow exceptions
    "WorkflowError",
    "WorkflowTimeoutError",
    "WorkflowValidationError",

    # LLM exceptions
    "LLMError",
    "LLMTimeoutError",
    "LLMRateLimitError",

    # Vendor analysis exceptions
    "VendorAnalysisError",
    "VendorAnalysisTimeoutError",

    # RAG exceptions
    "RAGError",
    "RAGRetrievalError",
    "RAGGenerationError",

    # ChromaDB exceptions
    "ChromaDBError",
    "ChromaDBConnectionError",
    "ChromaDBTimeoutError",

    # Schema exceptions
    "SchemaError",
    "SchemaNotFoundError",
    "SchemaValidationError",

    # Tool exceptions
    "ToolExecutionError",
    "InstrumentIdentificationError",
    "RoutingError",

    # Utility functions
    "format_error_message",
    "handle_workflow_error"
]
