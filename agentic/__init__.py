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
    InstrumentDetailState,
    InstrumentIdentifierState,  # NEW: Identifier workflow state
    PotentialProductIndexState,

    # State Factory Functions
    create_comparison_state,
    create_solution_state,
    create_instrument_detail_state,
    create_instrument_identifier_state,  # NEW: Identifier state factory
    create_potential_product_index_state
)

# ============================================================================
# AGENTS
# ============================================================================
from .agents import (
    BaseAgent,
    IntentClassifierAgent,
    ValidationAgent,
    VendorSearchAgent,
    ProductAnalysisAgent,
    RankingAgent,
    SalesAgent,
    InstrumentIdentifierAgent,
    ImageSearchAgent,
    PDFSearchAgent,
    AgentFactory
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

from .comparison_workflow import (
    create_comparison_workflow,
    run_comparison_workflow,
    run_comparison_from_spec,
    discover_candidates_from_ppi
)

from .instrument_detail_workflow import (
    create_instrument_detail_workflow,
    run_instrument_detail_workflow,
    run_instrument_detail_with_comparison,
    build_spec_object_from_state
)

from .instrument_identifier_workflow import (
    create_instrument_identifier_workflow,
    run_instrument_identifier_workflow
)

from .potential_product_index import (
    create_potential_product_index_workflow,
    run_potential_product_index_workflow
)

from .grounded_chat_workflow import (
    create_grounded_chat_workflow,
    run_grounded_chat_workflow,
    GroundedChatState,
    create_grounded_chat_state
)

# ============================================================================
# CHAT AGENTS
# ============================================================================
from .chat_agents import (
    ChatAgent,
    ResponseValidatorAgent,
    SessionManagerAgent
)

# ============================================================================
# API
# ============================================================================
from .api import agentic_bp


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
    'InstrumentDetailState',
    'PotentialProductIndexState',
    
    # State Factories
    'create_comparison_state',
    'create_solution_state',
    'create_instrument_detail_state',
    'create_potential_product_index_state',
    
    # ==================== AGENTS ====================
    'BaseAgent',
    'IntentClassifierAgent',
    'ValidationAgent',
    'VendorSearchAgent',
    'ProductAnalysisAgent',
    'RankingAgent',
    'SalesAgent',
    'InstrumentIdentifierAgent',
    'ImageSearchAgent',
    'PDFSearchAgent',
    'AgentFactory',
    
    # ==================== RAG ====================
    'RAGAggregator',
    'StrategyFilter',
    'create_rag_aggregator',
    'create_strategy_filter',
    
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
    'create_comparison_workflow',
    'run_comparison_workflow',
    'run_comparison_from_spec',
    'discover_candidates_from_ppi',
    'create_instrument_detail_workflow',
    'run_instrument_detail_workflow',
    'run_instrument_detail_with_comparison',
    'build_spec_object_from_state',
    'create_potential_product_index_workflow',
    'run_potential_product_index_workflow',
    'create_grounded_chat_workflow',
    'run_grounded_chat_workflow',
    'GroundedChatState',
    'create_grounded_chat_state',
    
    # ==================== SPEC OBJECT MODELS ====================
    'SpecObject',
    'ComparisonType',
    'ComparisonInput',
    
    # ==================== CHAT AGENTS ====================
    'ChatAgent',
    'ResponseValidatorAgent',
    'SessionManagerAgent',
    
    # ==================== API ====================
    'agentic_bp'
]

