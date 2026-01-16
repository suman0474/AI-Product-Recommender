# agentic/strategy_rag/__init__.py
# Strategy RAG Package - Filter and prioritize vendors based on procurement strategy

from .strategy_rag_workflow import (
    StrategyRAGState,
    create_strategy_rag_state,
    create_strategy_rag_workflow,
    run_strategy_rag_workflow,
    get_strategy_for_product,
    get_strategy_rag_workflow
)

from .strategy_chat_agent import (
    StrategyChatAgent,
    create_strategy_chat_agent
)

from .strategy_csv_filter import (
    StrategyCSVFilter,
    filter_vendors_by_strategy,
    get_vendor_strategy_info,
    get_strategy_filter
)

from .strategy_rag_enrichment import (
    enrich_with_strategy_rag,
    enrich_schema_with_strategy,
    filter_vendors_by_strategy_data
)

# Memory (NEW)
from .strategy_rag_memory import (
    StrategyRAGMemory,
    strategy_rag_memory,
    get_strategy_rag_memory,
    add_to_strategy_memory,
    clear_strategy_memory
)

__all__ = [
    # Workflow (LangGraph)
    'StrategyRAGState',
    'create_strategy_rag_state',
    'create_strategy_rag_workflow',
    'run_strategy_rag_workflow',
    'get_strategy_for_product',
    'get_strategy_rag_workflow',
    
    # Chat Agent
    'StrategyChatAgent',
    'create_strategy_chat_agent',
    
    # CSV Filter
    'StrategyCSVFilter',
    'filter_vendors_by_strategy',
    'get_vendor_strategy_info',
    'get_strategy_filter',
    
    # Enrichment
    'enrich_with_strategy_rag',
    'enrich_schema_with_strategy',
    'filter_vendors_by_strategy_data',
    
    # Memory (NEW)
    'StrategyRAGMemory',
    'strategy_rag_memory',
    'get_strategy_rag_memory',
    'add_to_strategy_memory',
    'clear_strategy_memory'
]
