"""
Centralized Timeout Configuration

Defines timeout values for various operations to prevent hanging requests.
All values in seconds.
"""

# HTTP Request Timeouts
HTTP_REQUEST_TIMEOUT = 30  # External API calls
HTTP_SEARCH_TIMEOUT = 15  # Search API calls (faster expected)
HTTP_PDF_DOWNLOAD_TIMEOUT = 60  # PDF downloads (larger files)

# LLM Call Timeouts
LLM_QUICK_TIMEOUT = 30  # Simple classification, validation
LLM_STANDARD_TIMEOUT = 60  # Standard analysis, generation
LLM_LONG_TIMEOUT = 120  # Complex analysis, multiple vendors

# Workflow Timeouts
WORKFLOW_NODE_TIMEOUT = 180  # Maximum time for single workflow node
WORKFLOW_TOTAL_TIMEOUT = 600  # Maximum time for entire workflow (10 min)

# Tool Timeouts
TOOL_DEFAULT_TIMEOUT = 45  # Default for tool execution
TOOL_SEARCH_TIMEOUT = 20  # Search tools
TOOL_ANALYSIS_TIMEOUT = 90  # Analysis tools

# Database Timeouts
MONGODB_QUERY_TIMEOUT = 10  # MongoDB queries
MONGODB_CONNECTION_TIMEOUT = 5  # Connection establishment

# Parallel Execution
THREAD_POOL_TIMEOUT = 120  # Max wait for ThreadPoolExecutor

__all__ = [
    'HTTP_REQUEST_TIMEOUT',
    'HTTP_SEARCH_TIMEOUT',
    'HTTP_PDF_DOWNLOAD_TIMEOUT',
    'LLM_QUICK_TIMEOUT',
    'LLM_STANDARD_TIMEOUT',
    'LLM_LONG_TIMEOUT',
    'WORKFLOW_NODE_TIMEOUT',
    'WORKFLOW_TOTAL_TIMEOUT',
    'TOOL_DEFAULT_TIMEOUT',
    'TOOL_SEARCH_TIMEOUT',
    'TOOL_ANALYSIS_TIMEOUT',
    'MONGODB_QUERY_TIMEOUT',
    'MONGODB_CONNECTION_TIMEOUT',
    'THREAD_POOL_TIMEOUT',
]
