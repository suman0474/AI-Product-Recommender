# agentic/models.py
# Centralized Models, States, and Type Definitions
#
# This module contains all TypedDict states, Pydantic models, Enums,
# and factory functions used across agentic workflows.

from typing import Dict, Any, List, Optional, TypedDict, Annotated, Literal
from enum import Enum
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage


# ============================================================================
# ENUMS
# ============================================================================

class IntentType(str, Enum):
    """User intent types for workflow routing"""
    GREETING = "greeting"
    REQUIREMENTS = "requirements"
    QUESTION = "question"
    CONFIRM = "confirm"
    REJECT = "reject"
    ADDITIONAL_SPECS = "additionalSpecs"
    UNRELATED = "unrelated"
    SOLUTION = "solution"
    PRODUCT_SEARCH = "productSearch"
    PRODUCT_INFO = "productInfo"
    CHAT = "chat"
    INVALID = "invalid"


class WorkflowType(str, Enum):
    """Available workflow types"""
    SOLUTION = "solution"
    INSTRUMENT_IDENTIFIER = "instrument_identifier"
    PRODUCT_INFO = "product_info"
    INVALID = "invalid"
    PRODUCT_SEARCH = "product_search"
    COMPARISON = "comparison"
    PPI = "ppi"


class WorkflowStep(str, Enum):
    """Workflow execution steps"""
    INTENT_CLASSIFICATION = "intentClassification"
    VALIDATION = "validation"
    VENDOR_SEARCH = "vendorSearch"
    PRODUCT_ANALYSIS = "productAnalysis"
    RANKING = "ranking"
    RESPONSE = "response"
    COLLECT_MISSING_INFO = "collectMissingInfo"
    INSTRUMENT_IDENTIFICATION = "instrumentIdentification"
    ACCESSORY_IDENTIFICATION = "accessoryIdentification"
    BOM_FORMATTING = "bomFormatting"
    COMPLETED = "completed"
    ERROR = "error"


class ComparisonType(str, Enum):
    """Types of product comparison"""
    COMPETITIVE = "competitive"  # Compare competing products
    REPLACEMENT = "replacement"  # Find replacement for existing
    UPGRADE = "upgrade"          # Upgrade analysis


class RequestMode(str, Enum):
    """Solution workflow request modes"""
    DESIGN = "design"            # Design new system
    SEARCH = "search"            # Find existing product
    COMPARISON = "comparison"    # Compare multiple options


class SalesAgentStep(str, Enum):
    """Sales Agent workflow steps (FSM states)"""
    GREETING = "greeting"
    INITIAL_INPUT = "initialInput"
    AWAIT_MISSING_INFO = "awaitMissingInfo"
    AWAIT_ADDITIONAL_SPECS = "awaitAdditionalSpecs"
    AWAIT_ADVANCED_SPECS = "awaitAdvancedSpecs"
    SHOW_SUMMARY = "showSummary"
    FINAL_ANALYSIS = "finalAnalysis"
    END = "end"
    ERROR = "error"


class SalesAgentIntent(str, Enum):
    """Sales Agent intent types"""
    GREETING = "greeting"
    PRODUCT_REQUIREMENTS = "productRequirements"
    KNOWLEDGE_QUESTION = "knowledgeQuestion"
    WORKFLOW = "workflow"
    CONFIRM = "confirm"
    REJECT = "reject"
    CHITCHAT = "chitchat"
    OTHER = "other"


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class IntentClassification(BaseModel):
    """Intent classification result"""
    intent: str
    confidence: float
    reasoning: Optional[str] = None


class RequirementValidation(BaseModel):
    """Requirement validation result"""
    is_valid: bool
    product_type: Optional[str] = None
    provided_requirements: Dict[str, Any] = Field(default_factory=dict)
    missing_fields: List[str] = Field(default_factory=list)
    confidence: float = 0.0


class VendorMatch(BaseModel):
    """Single vendor match result"""
    vendor: str
    relevance_score: float
    reasoning: str
    products_found: int = 0


class VendorAnalysis(BaseModel):
    """Detailed vendor analysis"""
    vendor: str
    product_name: str
    model_family: Optional[str] = None
    match_score: float
    key_strengths: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    reasoning: str
    specifications: Dict[str, Any] = Field(default_factory=dict)


class ProductRanking(BaseModel):
    """Single product ranking"""
    vendor: str
    product_name: str
    rank: int
    overall_score: float
    technical_score: float
    compliance_score: float
    recommendation: Optional[str] = None


class OverallRanking(BaseModel):
    """Complete ranking results"""
    ranked_products: List[ProductRanking]
    summary: str
    top_recommendation: Optional[str] = None


class InstrumentIdentification(BaseModel):
    """Instrument identification result"""
    project_name: str
    instruments: List[Dict[str, Any]] = Field(default_factory=list)
    accessories: List[Dict[str, Any]] = Field(default_factory=list)
    summary: str


class ConstraintContext(BaseModel):
    """Constraint context from RAG"""
    preferred_vendors: List[str] = Field(default_factory=list)
    forbidden_vendors: List[str] = Field(default_factory=list)
    required_standards: List[str] = Field(default_factory=list)
    required_certifications: List[str] = Field(default_factory=list)
    installed_series: List[str] = Field(default_factory=list)


class VendorModelCandidate(BaseModel):
    """Vendor model candidate for comparison"""
    vendor: str
    model_family: str
    product_name: str
    specifications: Dict[str, Any] = Field(default_factory=dict)


class ScoringBreakdown(BaseModel):
    """Detailed scoring breakdown"""
    mandatory_match: float = 0.0
    optional_match: float = 0.0
    advanced_params: float = 0.0
    industry_fit: float = 0.0
    total: float = 0.0


class RankedComparisonProduct(BaseModel):
    """Product with comparison scoring"""
    vendor: str
    product_name: str
    model_family: Optional[str] = None
    rank: int
    overall_score: float
    scoring_breakdown: ScoringBreakdown
    key_differentiators: List[str] = Field(default_factory=list)
    reasoning: str


class ComparisonMatrix(BaseModel):
    """Comparison matrix result"""
    products: List[RankedComparisonProduct]
    comparison_type: ComparisonType
    winner: Optional[str] = None
    summary: str


class RAGQueryResult(BaseModel):
    """RAG query result"""
    query: str
    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: float = 0.0


class SpecObject(BaseModel):
    """Specification object for UI comparison"""
    field_name: str
    user_value: Any
    comparison_type: str  # "exact", "range", "fuzzy", etc.
    importance: str = "mandatory"  # "mandatory", "optional", "advanced"


class ComparisonInput(BaseModel):
    """Input for comparison workflow"""
    product_type: str
    user_requirements: Dict[str, Any]
    spec_objects: List[SpecObject]
    comparison_type: ComparisonType = ComparisonType.COMPETITIVE



# ============================================================================
# THREAD TREE TYPEDDICT MODELS
# ============================================================================

class ThreadNode(TypedDict, total=False):
    """
    Thread node in the hierarchical thread tree.

    Represents a single node in the tree structure:
    - Main threads (root)
    - Workflow threads (children of main)
    - Item threads (children of workflow)
    """
    thread_id: str
    thread_type: str  # "main", "workflow", "item"
    parent_thread_id: Optional[str]
    children_thread_ids: List[str]
    zone: str
    user_id: str
    workflow_type: Optional[str]
    item_type: Optional[str]  # "instrument" or "accessory"
    item_number: Optional[int]
    created_at: str
    updated_at: Optional[str]
    metadata: Dict[str, Any]


class ItemState(TypedDict, total=False):
    """
    Full persistent state for each item sub-thread.

    This represents the complete state of an identified instrument or accessory
    that is stored in Azure Blob Storage for persistence.
    """
    # Thread identification
    thread_id: str
    parent_workflow_thread_id: str
    main_thread_id: Optional[str]
    zone: str

    # Item identification
    item_number: int
    item_type: str  # "instrument" or "accessory"
    item_name: str
    category: str
    quantity: int

    # Specifications
    specifications: Dict[str, Any]
    sample_input: str
    user_specified_specs: Dict[str, Any]  # Specs explicitly from user (MANDATORY)
    llm_generated_specs: Dict[str, Any]  # LLM-inferred specs
    standards_specifications: Dict[str, Any]  # Specs from standards documents
    combined_specifications: Dict[str, Any]  # Merged specs with source metadata

    # Enrichment data
    enrichment_data: Dict[str, Any]
    standards_info: Dict[str, Any]
    applicable_standards: List[str]
    certifications: List[str]

    # Status tracking
    status: str  # "identified", "selected", "searched", "completed"
    selection_timestamp: Optional[str]
    search_timestamp: Optional[str]
    completion_timestamp: Optional[str]

    # Search results (populated when item is selected and searched)
    search_results: Optional[List[Dict[str, Any]]]
    selected_product: Optional[Dict[str, Any]]

    # Timestamps
    created_at: str
    updated_at: str


class ThreadTreeInfo(TypedDict, total=False):
    """
    Information about a complete thread tree.

    Returned by API endpoints to provide full tree structure.
    """
    main_thread_id: str
    zone: str
    user_id: str
    created_at: str
    workflow_threads: List[ThreadNode]
    item_count: int
    total_threads: int


# ============================================================================
# TYPEDDICT STATES
# ============================================================================

class WorkflowState(TypedDict, total=False):
    """Base workflow state (original/legacy)"""
    # Session & Context
    session_id: str
    user_id: Optional[str]
    user_input: str
    messages: List[BaseMessage]

    # Intent & Classification
    intent: str
    intent_confidence: float
    current_step: str

    # Product & Requirements
    product_type: Optional[str]
    schema: Optional[Dict[str, Any]]
    provided_requirements: Dict[str, Any]
    missing_fields: List[str]

    # Search & Analysis
    filtered_vendors: List[str]
    vendor_matches: List[Dict[str, Any]]
    analysis_results: List[Dict[str, Any]]
    ranked_results: List[Dict[str, Any]]

    # Response
    response: str
    error: Optional[str]


class ProductSearchState(TypedDict, total=False):
    """Product Search Workflow State"""
    # Session & Context
    session_id: str
    user_id: Optional[str]
    search_session_id: str
    thread_id: Optional[str]

    # User Input & Intent
    user_input: str
    messages: List[BaseMessage]
    current_intent: str
    current_step: str
    previous_step: Optional[str]

    # Product & Schema
    product_type: Optional[str]
    schema: Optional[Dict[str, Any]]
    ppi_needed: bool
    ppi_result: Optional[Dict[str, Any]]

    # Requirements
    provided_requirements: Dict[str, Any]
    missing_requirements: List[str]
    requirements_summary: Optional[str]

    # Advanced Parameters
    available_advanced_params: List[str]
    advanced_params_selected: Dict[str, Any]
    advanced_params_discovery_result: Optional[Dict[str, Any]]

    # RAG Context
    rag_context: Dict[str, Any]
    strategy_data: Optional[Dict[str, Any]]
    standards_data: Optional[Dict[str, Any]]

    # Vendor Search & Analysis
    filtered_vendors: List[str]
    parallel_analysis_results: List[Dict[str, Any]]
    ranked_results: List[Dict[str, Any]]

    # Sales Agent Integration
    sales_agent_response: Optional[str]
    current_sales_step: Optional[str]
    awaiting_user_input: bool
    user_confirmed: bool

    # UI Response
    response: str
    response_data: Optional[Dict[str, Any]]
    ui_phase: Optional[str]
    premium_result_cards: Optional[List[Dict[str, Any]]]

    # Error Handling
    error: Optional[str]
    error_recovery_count: int

    # Internal Flags
    _advanced_params_presented: bool
    _missing_info_presented: bool


class SolutionState(TypedDict, total=False):
    """Solution Workflow State (Design & Search)"""
    # Session
    session_id: str
    user_id: Optional[str]

    # Thread Tree Fields (NEW - Hierarchical Thread System)
    main_thread_id: Optional[str]
    workflow_thread_id: Optional[str]
    zone: Optional[str]
    item_threads: Dict[int, str]  # Maps item number to item thread ID

    # Input
    user_input: str
    messages: List[Dict[str, Any]]  # Changed from BaseMessage to Dict for flexibility

    # Intent & Mode
    intent: str
    intent_confidence: float
    request_mode: str  # "design", "search", "comparison"
    comparison_mode: bool

    # Product & Schema
    product_type: Optional[str]
    schema: Optional[Dict[str, Any]]
    provided_requirements: Dict[str, Any]
    missing_fields: List[str]

    # Solution Analysis (from analyze_solution_node)
    solution_analysis: Dict[str, Any]
    solution_name: str
    instrument_context: str
    project_name: Optional[str]

    # Identified Items (from identify_solution_instruments_node)
    identified_instruments: List[Dict[str, Any]]
    identified_accessories: List[Dict[str, Any]]
    all_items: List[Dict[str, Any]]
    total_items: int

    # RAG Context
    rag_strategy: Optional[Dict[str, Any]]
    rag_standards: Optional[Dict[str, Any]]
    rag_inventory: Optional[Dict[str, Any]]
    constraint_context: Optional[ConstraintContext]

    # Search & Analysis
    filtered_vendors: List[str]
    vendor_matches: List[Dict[str, Any]]
    analysis_results: List[Dict[str, Any]]
    judge_results: Optional[Dict[str, Any]]
    ranked_products: List[Dict[str, Any]]

    # Comparison Mode
    comparison_output: Optional[Dict[str, Any]]
    comparison_matrix: Optional[ComparisonMatrix]

    # Response
    response: str
    response_data: Optional[Dict[str, Any]]
    current_step: str
    error: Optional[str]

    # Standards Validation (from enrichment)
    standards_validation: Optional[Dict[str, Any]]

    # Thread Tree Metadata (for API response)
    thread_info: Optional[Dict[str, Any]]


class ComparisonState(TypedDict, total=False):
    """Comparison Workflow State"""
    # Session
    session_id: str
    user_id: Optional[str]

    # Input
    user_input: str
    product_type: str
    comparison_type: ComparisonType

    # Requirements
    user_requirements: Dict[str, Any]
    spec_objects: List[SpecObject]

    # Candidates
    candidates: List[VendorModelCandidate]
    candidate_count: int

    # Analysis
    analysis_results: List[Dict[str, Any]]
    comparison_matrix: Optional[ComparisonMatrix]
    ranked_products: List[RankedComparisonProduct]

    # Response
    response: str
    winner: Optional[str]
    error: Optional[str]




class InstrumentIdentifierState(TypedDict, total=False):
    """Instrument Identifier Workflow State"""
    # Session
    session_id: str
    user_id: Optional[str]

    # Thread Tree Fields (NEW - Hierarchical Thread System)
    main_thread_id: Optional[str]
    workflow_thread_id: Optional[str]
    zone: Optional[str]
    item_threads: Dict[int, str]  # Maps item number to item thread ID

    # Input
    user_input: str
    messages: List[BaseMessage]

    # Classification
    input_type: str  # "requirements", "greeting", "question", "unrelated"
    initial_intent: Optional[str]  # Added for classify_initial_intent_node
    classification_confidence: str
    classification_reasoning: str

    # Identification
    project_name: Optional[str]
    identified_instruments: List[Dict[str, Any]]
    identified_accessories: List[Dict[str, Any]]
    all_items: List[Dict[str, Any]]  # Combined list
    total_items: int

    # Response
    response: str
    response_data: Optional[Dict[str, Any]]  # Detailed UI data
    instrument_list: Optional[str]
    current_step: str
    error: Optional[str]

    # Thread Tree Metadata (for API response)
    thread_info: Optional[Dict[str, Any]]


class PotentialProductIndexState(TypedDict, total=False):
    """Potential Product Index (PPI) Workflow State"""
    # Session
    session_id: str
    parent_workflow: str  # Parent workflow that triggered PPI
    rag_context: Dict[str, Any]  # Context from parent workflow

    # Input
    product_type: str
    user_requirements: Optional[Dict[str, Any]]

    # Messages
    messages: List[Dict[str, Any]]  # Workflow messages for tracking
    current_phase: str  # Current workflow phase

    # Vendor Discovery
    potential_vendors: List[str]  # Legacy field
    discovered_vendors: List[str]  # Active field used by workflow
    vendor_model_families: Dict[str, List[str]]
    vendor_processing_status: Dict[str, str]  # Track vendor processing status
    failed_vendors: List[str]  # Vendors that failed processing

    # PDF Search & Processing
    pdf_search_results: Dict[str, List[Dict[str, Any]]]  # Changed to Dict for per-vendor results
    downloaded_pdfs: List[Dict[str, Any]]
    processed_pdfs: List[Dict[str, Any]]
    pdf_download_status: Dict[str, str]  # Track download status per vendor
    pdf_success_rate: float
    extracted_content: str  # JSON string of extracted content

    # RAG Indexing
    indexed_content: List[Dict[str, Any]]
    indexed_chunks: List[Dict[str, Any]]  # Indexed content chunks
    vector_store_ids: List[str]  # Vector store IDs
    index_success: bool

    # Schema Generation
    generated_schema: Optional[Dict[str, Any]]
    schema_version: int
    schema_validated: bool
    schema_saved: bool  # Flag indicating schema was saved

    # Output
    ppi_success: bool
    retry_count: int
    max_retries: int
    error: Optional[str]
    current_step: str


class GroundedChatState(TypedDict, total=False):
    """Grounded Knowledge Chat Workflow State"""
    # Question & Session
    user_question: str
    session_id: str
    user_id: Optional[str]
    question_type: Optional[str]

    # Product Context
    product_type: Optional[str]
    entities: List[str]
    is_valid_question: bool

    # PPI Integration
    ppi_needed: bool
    ppi_result: Optional[Dict[str, Any]]
    schema_available: bool

    # RAG Context
    rag_context: Dict[str, Any]
    constraint_context: Optional[ConstraintContext]
    preferred_vendors: List[str]
    required_standards: List[str]
    installed_series: List[str]

    # Response Generation
    generated_answer: Optional[str]
    citations: List[str]
    rag_sources_used: List[str]
    confidence: float

    # Validation
    is_valid: bool
    validation_score: float
    validation_issues: List[str]
    hallucination_detected: bool
    retry_count: int
    max_retries: int
    validation_feedback: Optional[str]

    # Session Management
    total_interactions: int
    conversation_context: List[Dict[str, Any]]
    final_response: Optional[str]

    # Workflow
    current_node: str
    messages: List[BaseMessage]
    error: Optional[str]
    completed: bool


class SalesAgentState(TypedDict, total=False):
    """Sales Agent Conversational Workflow State"""
    # Session Management
    session_id: str
    user_id: Optional[str]
    search_session_id: str

    # User Input & Intent
    user_input: str
    current_intent: str
    current_step: str
    previous_step: Optional[str]

    # Product & Schema
    product_type: Optional[str]
    schema: Optional[Dict[str, Any]]

    # Requirements Collection
    provided_requirements: Dict[str, Any]
    missing_requirements: List[str]

    # Additional Parameters
    available_additional_params: List[str]
    selected_additional_params: Dict[str, Any]

    # Advanced Parameters
    available_advanced_params: List[str]
    selected_advanced_params: Dict[str, Any]

    # Search Results
    filtered_vendors: List[str]
    analysis_results: List[Dict[str, Any]]
    ranked_results: List[Dict[str, Any]]

    # Sales Agent Response
    sales_agent_response: Optional[str]
    response_data: Optional[Dict[str, Any]]

    # Conversation
    messages: List[BaseMessage]

    # Error Handling
    error: Optional[str]
    error_recovery_count: int


class StandardsRAGState(TypedDict, total=False):
    """Standards RAG Workflow State"""
    # Input
    question: str
    session_id: str
    top_k: int

    # Validation
    question_valid: bool
    key_terms: List[str]

    # Retrieval
    retrieved_docs: List[Dict[str, Any]]
    context: str
    source_metadata: List[Dict[str, Any]]

    # Generation
    answer: Optional[str]
    citations: List[str]
    confidence: float
    sources_used: List[str]
    generation_count: int

    # Validation & Retry
    validation_result: Optional[Dict[str, Any]]
    is_valid: bool
    retry_count: int
    max_retries: int
    validation_feedback: Optional[str]

    # Output
    final_response: Optional[str]
    status: str
    error: Optional[str]

    # Metrics
    start_time: Optional[float]
    processing_time_ms: Optional[float]


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_initial_state(
    user_input: str,
    session_id: str = "default",
    user_id: Optional[str] = None
) -> WorkflowState:
    """Create initial workflow state"""
    return WorkflowState(
        session_id=session_id,
        user_id=user_id,
        user_input=user_input,
        messages=[],
        intent="",
        intent_confidence=0.0,
        current_step="intent_classification",
        product_type=None,
        schema=None,
        provided_requirements={},
        missing_fields=[],
        filtered_vendors=[],
        vendor_matches=[],
        analysis_results=[],
        ranked_results=[],
        response="",
        error=None
    )


def create_product_search_state(
    user_input: str,
    session_id: str = "default",
    user_id: Optional[str] = None,
    search_session_id: Optional[str] = None
) -> ProductSearchState:
    """Create product search workflow state"""
    return ProductSearchState(
        session_id=session_id,
        user_id=user_id,
        search_session_id=search_session_id or session_id,
        thread_id=None,
        user_input=user_input,
        messages=[],
        current_intent="",
        current_step="intent_classification",
        previous_step=None,
        product_type=None,
        schema=None,
        ppi_needed=False,
        ppi_result=None,
        provided_requirements={},
        missing_requirements=[],
        requirements_summary=None,
        available_advanced_params=[],
        advanced_params_selected={},
        advanced_params_discovery_result=None,
        rag_context={},
        strategy_data=None,
        standards_data=None,
        filtered_vendors=[],
        parallel_analysis_results=[],
        ranked_results=[],
        sales_agent_response=None,
        current_sales_step=None,
        awaiting_user_input=False,
        user_confirmed=False,
        response="",
        response_data=None,
        ui_phase=None,
        premium_result_cards=None,
        error=None,
        error_recovery_count=0,
        _advanced_params_presented=False,
        _missing_info_presented=False
    )


def create_solution_state(
    user_input: str,
    session_id: str = "default",
    user_id: Optional[str] = None
) -> SolutionState:
    """Create solution workflow state"""
    return SolutionState(
        session_id=session_id,
        user_id=user_id,
        user_input=user_input,
        messages=[],
        intent="",
        intent_confidence=0.0,
        request_mode="search",
        comparison_mode=False,
        product_type=None,
        schema=None,
        provided_requirements={},
        missing_fields=[],
        # Solution Analysis fields
        solution_analysis={},
        solution_name="",
        instrument_context="",
        project_name=None,
        # Identified Items fields
        identified_instruments=[],
        identified_accessories=[],
        all_items=[],
        total_items=0,
        # RAG Context
        rag_strategy=None,
        rag_standards=None,
        rag_inventory=None,
        constraint_context=None,
        filtered_vendors=[],
        vendor_matches=[],
        analysis_results=[],
        judge_results=None,
        ranked_products=[],
        comparison_output=None,
        comparison_matrix=None,
        response="",
        response_data=None,
        current_step="analyze_solution",
        error=None
    )


def create_comparison_state(
    user_input: str,
    product_type: str,
    user_requirements: Dict[str, Any],
    spec_objects: List[SpecObject],
    session_id: str = "default",
    user_id: Optional[str] = None,
    comparison_type: ComparisonType = ComparisonType.COMPETITIVE
) -> ComparisonState:
    """Create comparison workflow state"""
    return ComparisonState(
        session_id=session_id,
        user_id=user_id,
        user_input=user_input,
        product_type=product_type,
        comparison_type=comparison_type,
        user_requirements=user_requirements,
        spec_objects=spec_objects,
        candidates=[],
        candidate_count=0,
        analysis_results=[],
        comparison_matrix=None,
        ranked_products=[],
        response="",
        winner=None,
        error=None
    )




def create_instrument_identifier_state(
    user_input: str,
    session_id: str = "default",
    user_id: Optional[str] = None
) -> InstrumentIdentifierState:
    """Create instrument identifier workflow state"""
    return InstrumentIdentifierState(
        session_id=session_id,
        user_id=user_id,
        user_input=user_input,
        messages=[],
        input_type="",
        classification_confidence="",
        classification_reasoning="",
        project_name=None,
        identified_instruments=[],
        identified_accessories=[],
        response="",
        instrument_list=None,
        current_step="classify_input",
        error=None
    )


def create_potential_product_index_state(
    product_type: str,
    session_id: str = "default",
    user_requirements: Optional[Dict[str, Any]] = None,
    parent_workflow: str = "solution",
    rag_context: Optional[Dict[str, Any]] = None
) -> PotentialProductIndexState:
    """Create PPI workflow state"""
    return PotentialProductIndexState(
        session_id=session_id,
        parent_workflow=parent_workflow,
        rag_context=rag_context or {},
        product_type=product_type,
        user_requirements=user_requirements or {},
        messages=[],
        current_phase="discover_vendors",
        potential_vendors=[],
        discovered_vendors=[],
        vendor_model_families={},
        vendor_processing_status={},
        failed_vendors=[],
        pdf_search_results={},
        downloaded_pdfs=[],
        processed_pdfs=[],
        pdf_download_status={},
        pdf_success_rate=0.0,
        extracted_content="",
        indexed_content=[],
        indexed_chunks=[],
        vector_store_ids=[],
        index_success=False,
        generated_schema=None,
        schema_version=1,
        schema_validated=False,
        schema_saved=False,
        ppi_success=False,
        retry_count=0,
        max_retries=2,
        error=None,
        current_step="discover_vendors"
    )


def create_grounded_chat_state(
    user_question: str,
    session_id: str = "default",
    user_id: Optional[str] = None,
    product_type: Optional[str] = None
) -> GroundedChatState:
    """Create grounded chat workflow state"""
    return GroundedChatState(
        user_question=user_question,
        session_id=session_id,
        user_id=user_id,
        question_type=None,
        product_type=product_type,
        entities=[],
        is_valid_question=True,
        ppi_needed=False,
        ppi_result=None,
        schema_available=False,
        rag_context={},
        constraint_context=None,
        preferred_vendors=[],
        required_standards=[],
        installed_series=[],
        generated_answer=None,
        citations=[],
        rag_sources_used=[],
        confidence=0.0,
        is_valid=True,
        validation_score=0.0,
        validation_issues=[],
        hallucination_detected=False,
        retry_count=0,
        max_retries=2,
        validation_feedback=None,
        total_interactions=0,
        conversation_context=[],
        final_response=None,
        current_node="classify_question",
        messages=[],
        error=None,
        completed=False
    )


def create_sales_agent_state(
    user_input: str,
    session_id: str = "default",
    user_id: Optional[str] = None,
    search_session_id: Optional[str] = None
) -> SalesAgentState:
    """Create sales agent workflow state"""
    return SalesAgentState(
        session_id=session_id,
        user_id=user_id,
        search_session_id=search_session_id or session_id,
        user_input=user_input,
        current_intent="",
        current_step=SalesAgentStep.GREETING.value,
        previous_step=None,
        product_type=None,
        schema=None,
        provided_requirements={},
        missing_requirements=[],
        available_additional_params=[],
        selected_additional_params={},
        available_advanced_params=[],
        selected_advanced_params={},
        filtered_vendors=[],
        analysis_results=[],
        ranked_results=[],
        sales_agent_response=None,
        response_data=None,
        messages=[],
        error=None,
        error_recovery_count=0
    )


def create_standards_rag_state(
    question: str,
    session_id: str = "default",
    top_k: int = 5
) -> StandardsRAGState:
    """Create standards RAG workflow state"""
    return StandardsRAGState(
        question=question,
        session_id=session_id,
        top_k=top_k,
        question_valid=True,
        key_terms=[],
        retrieved_docs=[],
        context="",
        source_metadata=[],
        answer=None,
        citations=[],
        confidence=0.0,
        sources_used=[],
        generation_count=0,
        validation_result=None,
        is_valid=True,
        retry_count=0,
        max_retries=2,
        validation_feedback=None,
        final_response=None,
        status="started",
        error=None,
        start_time=None,
        processing_time_ms=None
    )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "IntentType",
    "WorkflowType",
    "WorkflowStep",
    "AmbiguityLevel",
    "ComparisonType",
    "RequestMode",
    "SalesAgentStep",
    "SalesAgentIntent",

    # Pydantic Models
    "IntentClassification",
    "RequirementValidation",
    "VendorMatch",
    "VendorAnalysis",
    "ProductRanking",
    "OverallRanking",
    "InstrumentIdentification",
    "ConstraintContext",
    "VendorModelCandidate",
    "ScoringBreakdown",
    "RankedComparisonProduct",
    "ComparisonMatrix",
    "RAGQueryResult",
    "SpecObject",
    "ComparisonInput",

    # Thread Tree TypedDict Models (NEW - Hierarchical Thread System)
    "ThreadNode",
    "ItemState",
    "ThreadTreeInfo",

    # TypedDict States
    "WorkflowState",
    "ProductSearchState",
    "SolutionState",
    "ComparisonState",
    "InstrumentIdentifierState",
    "PotentialProductIndexState",
    "GroundedChatState",
    "SalesAgentState",
    "StandardsRAGState",

    # Factory Functions
    "create_initial_state",
    "create_product_search_state",
    "create_solution_state",
    "create_comparison_state",
    "create_instrument_identifier_state",
    "create_potential_product_index_state",
    "create_grounded_chat_state",
    "create_sales_agent_state",
    "create_standards_rag_state",
]
