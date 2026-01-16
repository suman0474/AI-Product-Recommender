# tools/__init__.py
# LangChain Tools for Agentic AI Workflow

from .intent_tools import (
    classify_intent_tool,
    extract_requirements_tool
)

from .schema_tools import (
    load_schema_tool,
    validate_requirements_tool,
    get_missing_fields_tool
)

from .analysis_tools import (
    analyze_vendor_match_tool,
    calculate_match_score_tool,
    extract_specifications_tool
)

from .search_tools import (
    search_product_images_tool,
    search_pdf_datasheets_tool,
    web_search_tool,
    search_vendors_tool,
    get_vendor_products_tool,
    fuzzy_match_vendors_tool
)

from .ranking_tools import (
    rank_products_tool,
    judge_analysis_tool
)

from .instrument_tools import (
    identify_instruments_tool,
    identify_accessories_tool
)

from .sales_workflow_tools import (
    # Enums
    SalesWorkflowStep,
    SalesWorkflowIntent,
    # Tools
    classify_sales_intent_tool,
    generate_sales_greeting_tool,
    extract_product_requirements_tool,
    merge_sales_requirements_tool,
    select_advanced_parameters_tool,
    generate_sales_summary_tool,
    generate_sales_response_tool,
    detect_user_confirmation_tool,
    get_workflow_step_info_tool,
    # Helpers
    detect_yes_no_response,
    format_parameters_display,
    format_requirements_summary
)

# Index RAG Tools
from .metadata_filter import (
    filter_by_hierarchy,
    extract_product_types,
    extract_vendors,
    extract_models,
    extract_metadata_with_llm
)

from .parallel_indexer import (
    run_parallel_indexing,
    index_database,
    index_web_search
)

from .specs_filter import (
    apply_specs_and_strategy_filter,
    apply_standards_filter,
    apply_strategy_filter
)

__all__ = [
    # Intent Tools
    'classify_intent_tool',
    'extract_requirements_tool',
    # Schema Tools
    'load_schema_tool',
    'validate_requirements_tool',
    'get_missing_fields_tool',
    # Vendor Tools
    'search_vendors_tool',
    'get_vendor_products_tool',
    'fuzzy_match_vendors_tool',
    # Analysis Tools
    'analyze_vendor_match_tool',
    'calculate_match_score_tool',
    'extract_specifications_tool',
    # Search Tools
    'search_product_images_tool',
    'search_pdf_datasheets_tool',
    'web_search_tool',
    # Ranking Tools
    'rank_products_tool',
    'judge_analysis_tool',
    # Instrument Tools
    'identify_instruments_tool',
    'identify_accessories_tool',
    # Sales Workflow Tools
    'SalesWorkflowStep',
    'SalesWorkflowIntent',
    'classify_sales_intent_tool',
    'generate_sales_greeting_tool',
    'extract_product_requirements_tool',
    'merge_sales_requirements_tool',
    'select_advanced_parameters_tool',
    'generate_sales_summary_tool',
    'generate_sales_response_tool',
    'detect_user_confirmation_tool',
    'get_workflow_step_info_tool',
    'detect_yes_no_response',
    'format_parameters_display',
    'format_requirements_summary',
    # Index RAG Tools
    'filter_by_hierarchy',
    'extract_product_types',
    'extract_vendors',
    'extract_models',
    'extract_metadata_with_llm',
    'run_parallel_indexing',
    'index_database',
    'index_web_search',
    'apply_specs_and_strategy_filter',
    'apply_standards_filter',
    'apply_strategy_filter'
]
