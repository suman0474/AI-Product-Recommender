# Agentic AI Module
# LangChain Tools, Agents, and LangGraph Workflow Implementation

# ============================================================================
# MODELS
# ============================================================================
from .models import (
    # Original Models
    WorkflowState,
    IntentType,
    WorkflowStep,
    IntentClassification,
    RequirementValidation,
    VendorMatch,
    VendorAnalysis,
    ProductRanking,
    OverallRanking,
    InstrumentIdentification,
    create_initial_state,
    
    # Enhanced Models
    RequestMode,
    ConstraintContext,
    VendorModelCandidate,
    ScoringBreakdown,
    RankedComparisonProduct,
    ComparisonMatrix,
    RAGQueryResult,
    
    # SpecObject for UI comparison
    SpecObject,
    ComparisonType,
    ComparisonInput,
    
    # Enhanced States
    ComparisonState,
    SolutionState,
    ComparisonState,
    SolutionState,
    InstrumentIdentifierState,  # NEW: Identifier workflow state
    PotentialProductIndexState,

    # State Factory Functions
    create_comparison_state,
    create_solution_state,
    create_solution_state,
    create_instrument_identifier_state,  # NEW: Identifier state factory
    create_potential_product_index_state
)


# ============================================================================
# RAG COMPONENTS
# ============================================================================
from .rag_components import (
    RAGAggregator,
    StrategyFilter,
    create_rag_aggregator,
    create_strategy_filter
)

# Strategy RAG (TRUE RAG with vector store)
from .strategy_rag.strategy_chat_agent import (
    StrategyChatAgent,
    create_strategy_chat_agent,
    ask_strategy_question,
    get_strategy_for_product as get_strategy_for_product_agent
)

from .strategy_rag.strategy_rag_workflow import (
    StrategyRAGState,
    create_strategy_rag_state,
    run_strategy_rag_workflow,
    get_strategy_for_product
)

# CSV-based strategy filter
from .strategy_rag.strategy_csv_filter import (
    StrategyCSVFilter,
    get_strategy_filter,
    filter_vendors_by_strategy
)

# Index RAG (LLM-powered product indexing)
from .index_rag.index_rag_agent import (
    IndexRAGAgent,
    create_index_rag_agent,
    run_index_rag
)

from .index_rag.index_rag_workflow import (
    IndexRAGState,
    create_index_rag_state,
    create_index_rag_workflow,
    run_index_rag_workflow
)

# ============================================================================
# CHECKPOINTING
# ============================================================================
from .checkpointing import (
    get_checkpointer,
    compile_with_checkpointing,
    WorkflowExecutor
)

# ============================================================================
# ORIGINAL WORKFLOWS
# ============================================================================
from .workflow import (
    create_procurement_workflow,
    create_instrument_identification_workflow,
    run_workflow
)

# ============================================================================
# ENHANCED WORKFLOWS
# ============================================================================
from .solution_workflow import (
    create_solution_workflow,
    run_solution_workflow
)




from .instrument_identifier_workflow import (
    create_instrument_identifier_workflow,
    run_instrument_identifier_workflow
)



from .potential_product_index import (
    create_potential_product_index_workflow,
    run_potential_product_index_workflow
)

# ============================================================================
# DEEP AGENT (Memory-centric workflow with planner)
# ============================================================================
from .deep_agent import (
    # Memory
    DeepAgentMemory,
    StandardsDocumentAnalysis,
    UserContextMemory,
    IdentifiedItemMemory,
    ThreadInfo,
    ExecutionPlan,
    SectionAnalysis,
    PRODUCT_TYPE_DOCUMENT_MAP,
    get_relevant_documents_for_product,
    
    # Document Loader
    populate_memory_with_documents,
    populate_memory_with_items,
    STANDARDS_DIRECTORY,
    
    # Workflow
    DeepAgentState,
    create_deep_agent_state,
    get_deep_agent_workflow,
    run_deep_agent_workflow,
    get_memory,
    get_or_create_memory,
    clear_memory,
    
    # Integration
    prepare_deep_agent_input,
    run_deep_agent_for_specifications,
    integrate_deep_agent_specifications,
    format_deep_agent_specs_for_display,
    load_schema_for_product,
    populate_schema_from_deep_agent,
    
    # Sub Agents
    ExtractorAgent,
    CriticAgent,
    extract_specifications_with_validation,
    extract_specifications_for_products,
    get_schema_fields_for_product,
    PRODUCT_TYPE_SCHEMA_FIELDS,
    
    # API
    deep_agent_bp,
    
    # Standards Deep Agent
    StandardsDeepAgentState,
    run_standards_deep_agent,
    run_standards_deep_agent_batch,
)

# Alias for backward compatibility
analyze_user_input_deep_agent = None  # deprecated
try:
    from .deep_agent.document_loader import analyze_user_input as analyze_user_input_deep_agent
except ImportError:
    pass

# Create alias for create_deep_agent_workflow to maintain backward compatibility
def create_deep_agent_workflow():
    """Backward compatibility alias for get_deep_agent_workflow"""
    return get_deep_agent_workflow()

# ============================================================================
# API
# ============================================================================
from .api import agentic_bp

# ============================================================================
# SESSION & INSTANCE ORCHESTRATION (NEW)
# ============================================================================
from .session_orchestrator import (
    SessionOrchestrator,
    SessionContext,
    WorkflowContext,
    get_session_orchestrator
)

from .instance_manager import (
    WorkflowInstanceManager,
    InstanceMetadata,
    InstancePool,
    InstanceStatus,
    InstancePriority,
    get_instance_manager
)

from .orchestrator_utils import (
    validate_main_thread_id,
    validate_workflow_thread_id,
    validate_item_thread_id,
    validate_thread_id,
    extract_user_from_main_thread_id,
    get_thread_type,
    verify_session_for_request,
    get_or_create_instance,
    complete_instance_with_result,
    start_background_cleanup,
    stop_background_cleanup,
    get_orchestrator_stats,
    reset_all_orchestrators
)

from .session_api import (
    session_bp,
    instances_bp,
    initialize_session_api,
    register_session_blueprints
)


__all__ = [
    # ==================== MODELS ====================
    # Original Models
    'WorkflowState',
    'IntentType',
    'WorkflowStep',
    'IntentClassification',
    'RequirementValidation',
    'VendorMatch',
    'VendorAnalysis',
    'ProductRanking',
    'OverallRanking',
    'InstrumentIdentification',
    'create_initial_state',
    
    # Enhanced Models
    'RequestMode',
    'ConstraintContext',
    'VendorModelCandidate',
    'ScoringBreakdown',
    'RankedComparisonProduct',
    'ComparisonMatrix',
    'RAGQueryResult',
    
    # Enhanced States
    'ComparisonState',
    'SolutionState',
    'PotentialProductIndexState',
    
    # State Factories
    'create_comparison_state',
    'create_solution_state',
    'create_potential_product_index_state',
    
    
    # ==================== RAG ====================
    'RAGAggregator',
    'StrategyFilter',
    'create_rag_aggregator',
    'create_strategy_filter',
    
    # Strategy RAG (TRUE RAG with vector store)
    'StrategyChatAgent',
    'create_strategy_chat_agent',
    'ask_strategy_question',
    'get_strategy_for_product',
    'StrategyRAGState',
    'create_strategy_rag_state',
    'create_strategy_rag_workflow',
    'run_strategy_rag_workflow',
    
    # Index RAG (LLM-powered product indexing)
    'IndexRAGAgent',
    'create_index_rag_agent',
    'run_index_rag',
    'IndexRAGState',
    'create_index_rag_state',
    'create_index_rag_workflow',
    'run_index_rag_workflow',
    
    # ==================== CHECKPOINTING ====================
    'get_checkpointer',
    'compile_with_checkpointing',
    'WorkflowExecutor',
    
    # ==================== ORIGINAL WORKFLOWS ====================
    'create_procurement_workflow',
    'create_instrument_identification_workflow',
    'run_workflow',
    
    # ==================== ENHANCED WORKFLOWS ====================
    'create_solution_workflow',
    'run_solution_workflow',
    'create_solution_workflow',
    'run_solution_workflow',
    'build_spec_object_from_state',

    'create_potential_product_index_workflow',
    'run_potential_product_index_workflow',

    # ==================== SPEC OBJECT MODELS ====================
    'SpecObject',
    'ComparisonType',
    'ComparisonInput',
    
    # ==================== CHAT AGENTS ====================
    'ChatAgent',
    'ResponseValidatorAgent',
    'SessionManagerAgent',

    # ==================== DEEP AGENT ====================
    'DeepAgentMemory',
    'StandardsDocumentAnalysis',
    'UserContextMemory',
    'IdentifiedItemMemory',
    'ThreadInfo',
    'ExecutionPlan',
    'SectionAnalysis',
    'PRODUCT_TYPE_DOCUMENT_MAP',
    'get_relevant_documents_for_product',
    'populate_memory_with_documents',
    'populate_memory_with_items',
    'analyze_user_input_deep_agent',
    'STANDARDS_DIRECTORY',
    'DeepAgentState',
    'create_deep_agent_state',
    'create_deep_agent_workflow',
    'get_deep_agent_workflow',
    'run_deep_agent_workflow',
    'get_memory',
    'get_or_create_memory',
    'clear_memory',

    # ==================== DEEP AGENT INTEGRATION ====================
    'prepare_deep_agent_input',
    'run_deep_agent_for_specifications',
    'integrate_deep_agent_specifications',
    'format_deep_agent_specs_for_display',
    # Schema Population Functions
    'load_schema_for_product',
    'populate_schema_from_deep_agent',
    'get_schema_population_stats',

    # ==================== API ====================
    'agentic_bp',

    # ==================== SESSION & INSTANCE ORCHESTRATION (NEW) ====================
    # Session Orchestrator
    'SessionOrchestrator',
    'SessionContext',
    'WorkflowContext',
    'get_session_orchestrator',

    # Instance Manager
    'WorkflowInstanceManager',
    'InstanceMetadata',
    'InstancePool',
    'InstanceStatus',
    'InstancePriority',
    'get_instance_manager',

    # Orchestrator Utilities
    'validate_main_thread_id',
    'validate_workflow_thread_id',
    'validate_item_thread_id',
    'validate_thread_id',
    'extract_user_from_main_thread_id',
    'get_thread_type',
    'verify_session_for_request',
    'get_or_create_instance',
    'complete_instance_with_result',
    'start_background_cleanup',
    'stop_background_cleanup',
    'get_orchestrator_stats',
    'reset_all_orchestrators',

    # Session API Blueprints
    'session_bp',
    'instances_bp',
    'initialize_session_api',
    'register_session_blueprints'
]

