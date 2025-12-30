# agentic/models.py
# Pydantic Models and State Definitions for LangGraph Workflow

from typing import Dict, List, Optional, Any, TypedDict, Annotated
from pydantic import BaseModel, Field
from enum import Enum
import operator


# ============================================================================
# ENUMS
# ============================================================================

class IntentType(str, Enum):
    """User intent classification types"""
    GREETING = "greeting"
    REQUIREMENTS = "requirements"
    QUESTION = "question"
    ADDITIONAL_SPECS = "additional_specs"
    CONFIRM = "confirm"
    REJECT = "reject"
    CHITCHAT = "chitchat"
    UNRELATED = "unrelated"


class WorkflowStep(str, Enum):
    """Workflow step identifiers"""
    START = "start"
    CLASSIFY_INTENT = "classify_intent"
    VALIDATE_REQUIREMENTS = "validate_requirements"
    COLLECT_MISSING_INFO = "collect_missing_info"
    SEARCH_VENDORS = "search_vendors"
    ANALYZE_PRODUCTS = "analyze_products"
    JUDGE_RESULTS = "judge_results"
    RANK_PRODUCTS = "rank_products"
    GENERATE_RESPONSE = "generate_response"
    END = "end"


class WorkflowType(str, Enum):
    """Workflow types for routing - 5 workflows (3 main + invalid + chat)"""
    SOLUTION = "solution"
    INSTRUMENT_DETAIL = "instrument_detail"
    GROUNDED_CHAT = "grounded_chat"
    CHAT = "chat"  # Generic conversational intents (greetings, acknowledgments)
    INVALID = "invalid"  # Out-of-domain queries


class AmbiguityLevel(str, Enum):
    """Ambiguity level for routing decisions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ============================================================================
# PYDANTIC MODELS - Tool Inputs/Outputs
# ============================================================================

class IntentClassification(BaseModel):
    """Output of intent classification"""
    intent: IntentType = Field(description="Classified user intent")
    confidence: float = Field(ge=0, le=1, description="Confidence score 0-1")
    next_step: Optional[WorkflowStep] = Field(default=None, description="Suggested next workflow step")
    extracted_info: Optional[Dict[str, Any]] = Field(default=None, description="Extracted information from input")


class RequirementValidation(BaseModel):
    """Validation result for user requirements"""
    is_valid: bool = Field(description="Whether requirements are valid")
    product_type: Optional[str] = Field(default=None, description="Detected product type")
    provided_requirements: Dict[str, Any] = Field(default_factory=dict, description="Requirements provided by user")
    missing_fields: List[str] = Field(default_factory=list, description="Missing required fields")
    optional_fields: List[str] = Field(default_factory=list, description="Available optional fields")
    validation_messages: List[str] = Field(default_factory=list, description="Validation messages")


class VendorMatch(BaseModel):
    """Single vendor match result"""
    vendor: str = Field(description="Vendor name")
    product_name: str = Field(description="Specific product name/model")
    model_family: str = Field(description="Product model family/series")
    match_score: float = Field(ge=0, le=100, description="Match score 0-100")
    requirements_match: bool = Field(description="Whether all mandatory requirements are met")
    matched_requirements: Dict[str, str] = Field(default_factory=dict, description="Matched requirements with values")
    unmatched_requirements: List[str] = Field(default_factory=list, description="Unmatched requirements")
    reasoning: str = Field(description="Reasoning for the match")
    limitations: Optional[str] = Field(default=None, description="Product limitations")
    pdf_source: Optional[str] = Field(default=None, description="PDF datasheet source")
    image_url: Optional[str] = Field(default=None, description="Product image URL")


class VendorAnalysis(BaseModel):
    """Complete vendor analysis result"""
    vendor_matches: List[VendorMatch] = Field(default_factory=list, description="List of vendor matches")
    analysis_summary: Optional[str] = Field(default=None, description="Summary of analysis")
    total_vendors_analyzed: int = Field(default=0, description="Number of vendors analyzed")


class ProductRanking(BaseModel):
    """Product ranking result"""
    rank: int = Field(ge=1, description="Product rank")
    vendor: str = Field(description="Vendor name")
    product_name: str = Field(description="Product name")
    model_family: str = Field(description="Model family")
    overall_score: float = Field(ge=0, le=100, description="Overall score 0-100")
    key_strengths: List[str] = Field(default_factory=list, description="Key product strengths")
    concerns: List[str] = Field(default_factory=list, description="Product concerns")
    recommendation: Optional[str] = Field(default=None, description="Recommendation text")


class OverallRanking(BaseModel):
    """Overall ranking result"""
    ranked_products: List[ProductRanking] = Field(default_factory=list, description="Ranked list of products")
    ranking_summary: Optional[str] = Field(default=None, description="Summary of ranking")


class RoutingDecision(BaseModel):
    """Routing decision from WorkflowRouter"""
    workflow: WorkflowType = Field(description="Selected workflow type")
    confidence: float = Field(ge=0, le=1, description="Confidence score 0-1")
    reasoning: str = Field(description="Reasoning for the routing decision")
    ambiguity_level: AmbiguityLevel = Field(description="Level of ambiguity in the decision")
    rule_match: Optional[str] = Field(default=None, description="Which rule matched (if any)")
    llm_used: bool = Field(default=False, description="Whether LLM was used for classification")
    alternatives: List[WorkflowType] = Field(default_factory=list, description="Alternative workflow suggestions")
    timestamp: Optional[str] = Field(default=None, description="Timestamp of routing decision")


class InstrumentSpecification(BaseModel):
    """Instrument specification"""
    category: str = Field(description="Instrument category")
    product_name: str = Field(description="Generic product name")
    quantity: int = Field(ge=1, description="Required quantity")
    specifications: Dict[str, Any] = Field(default_factory=dict, description="Technical specifications")
    strategy: Optional[str] = Field(default=None, description="Procurement strategy")
    sample_input: Optional[str] = Field(default=None, description="Sample input for analysis")


class AccessorySpecification(BaseModel):
    """Accessory specification"""
    category: str = Field(description="Accessory category")
    accessory_name: str = Field(description="Accessory name")
    quantity: int = Field(ge=1, description="Required quantity")
    specifications: Dict[str, Any] = Field(default_factory=dict, description="Specifications")
    for_instrument: Optional[str] = Field(default=None, description="Related instrument")


class InstrumentIdentification(BaseModel):
    """Instrument identification result"""
    project_name: str = Field(description="Project name")
    instruments: List[InstrumentSpecification] = Field(default_factory=list, description="Identified instruments")
    accessories: List[AccessorySpecification] = Field(default_factory=list, description="Identified accessories")
    summary: Optional[str] = Field(default=None, description="Summary of identification")


class AgentResponse(BaseModel):
    """Generic agent response"""
    success: bool = Field(description="Whether the agent succeeded")
    message: str = Field(description="Response message")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Response data")
    next_step: Optional[WorkflowStep] = Field(default=None, description="Suggested next step")
    requires_user_input: bool = Field(default=False, description="Whether user input is needed")


# ============================================================================
# LANGGRAPH STATE DEFINITION
# ============================================================================

class WorkflowState(TypedDict):
    """
    LangGraph Workflow State
    This is the state that flows through the entire workflow graph
    """
    # User Input
    user_input: str
    session_id: str

    # Intent Classification
    intent: Optional[IntentType]
    intent_confidence: float

    # Product Information
    product_type: Optional[str]
    schema: Optional[Dict[str, Any]]

    # Requirements
    provided_requirements: Dict[str, Any]
    missing_requirements: List[str]
    is_requirements_valid: bool

    # Vendor Analysis
    available_vendors: List[str]
    filtered_vendors: List[str]
    vendor_analysis: Optional[VendorAnalysis]

    # PDF and Product Data
    pdf_content: Dict[str, str]
    products_data: List[Dict[str, Any]]

    # Ranking
    ranking: Optional[OverallRanking]

    # Instrument Identification
    instruments: List[InstrumentSpecification]
    accessories: List[AccessorySpecification]

    # Workflow Control
    current_step: WorkflowStep
    next_step: Optional[WorkflowStep]
    requires_user_input: bool

    # Messages (for agent communication)
    messages: Annotated[List[Dict[str, Any]], operator.add]

    # Response
    response: Optional[str]
    response_data: Optional[Dict[str, Any]]

    # Error Handling
    error: Optional[str]
    retry_count: int


def create_initial_state(user_input: str, session_id: str) -> WorkflowState:
    """Create initial workflow state"""
    return WorkflowState(
        user_input=user_input,
        session_id=session_id,
        intent=None,
        intent_confidence=0.0,
        product_type=None,
        schema=None,
        provided_requirements={},
        missing_requirements=[],
        is_requirements_valid=False,
        available_vendors=[],
        filtered_vendors=[],
        vendor_analysis=None,
        pdf_content={},
        products_data=[],
        ranking=None,
        instruments=[],
        accessories=[],
        current_step=WorkflowStep.START,
        next_step=WorkflowStep.CLASSIFY_INTENT,
        requires_user_input=False,
        messages=[],
        response=None,
        response_data=None,
        error=None,
        retry_count=0
    )


# ============================================================================
# TOOL INPUT SCHEMAS
# ============================================================================

class ClassifyIntentInput(BaseModel):
    """Input schema for intent classification tool"""
    user_input: str = Field(description="User's input message to classify")
    context: Optional[str] = Field(default=None, description="Conversation context")


class ValidateRequirementsInput(BaseModel):
    """Input schema for requirements validation tool"""
    user_input: str = Field(description="User's requirements input")
    product_type: Optional[str] = Field(default=None, description="Detected product type")
    product_schema: Optional[Dict[str, Any]] = Field(default=None, description="Product schema")


class SearchVendorsInput(BaseModel):
    """Input schema for vendor search tool"""
    product_type: str = Field(description="Product type to search for")
    requirements: Dict[str, Any] = Field(description="User requirements")
    vendor_filter: Optional[List[str]] = Field(default=None, description="List of vendors to filter by")


class AnalyzeProductInput(BaseModel):
    """Input schema for product analysis tool"""
    vendor: str = Field(description="Vendor name")
    requirements: Dict[str, Any] = Field(description="User requirements")
    pdf_content: Optional[str] = Field(default=None, description="PDF datasheet content")
    product_data: Optional[Dict[str, Any]] = Field(default=None, description="Product JSON data")


class RankProductsInput(BaseModel):
    """Input schema for product ranking tool"""
    vendor_matches: List[Dict[str, Any]] = Field(description="List of vendor match results")
    requirements: Dict[str, Any] = Field(description="Original requirements")


class IdentifyInstrumentsInput(BaseModel):
    """Input schema for instrument identification tool"""
    requirements: str = Field(description="Process requirements description")


class SearchImagesInput(BaseModel):
    """Input schema for image search tool"""
    vendor: str = Field(description="Vendor name")
    product_name: str = Field(description="Product name")
    product_type: str = Field(description="Product type")


class SearchPDFsInput(BaseModel):
    """Input schema for PDF search tool"""
    vendor: str = Field(description="Vendor name")
    product_type: str = Field(description="Product type")
    model_family: Optional[str] = Field(default=None, description="Model family")


# ============================================================================
# ENHANCED MODELS FOR COMPARISON WORKFLOW
# ============================================================================

class RequestMode(str, Enum):
    """Request mode classification"""
    SINGLE_LOOKUP = "single"
    COMPARISON = "comparison"


class ConstraintContext(BaseModel):
    """Unified constraint set from all RAG sources"""
    
    # Strategy Constraints
    preferred_vendors: List[str] = Field(default_factory=list, description="Preferred vendors from strategy")
    forbidden_vendors: List[str] = Field(default_factory=list, description="Forbidden vendors from strategy")
    neutral_vendors: List[str] = Field(default_factory=list, description="Neutral vendors")
    procurement_priorities: Dict[str, int] = Field(default_factory=dict, description="Vendor priority scores")
    
    # Standards Constraints
    required_sil_rating: Optional[str] = Field(default=None, description="Required SIL rating (SIL1, SIL2, SIL3)")
    atex_zone: Optional[str] = Field(default=None, description="Required ATEX zone (Zone 0, 1, 2)")
    required_certifications: List[str] = Field(default_factory=list, description="Required certifications")
    plant_codes: List[str] = Field(default_factory=list, description="Plant-specific codes")
    
    # Inventory Constraints
    installed_series: Dict[str, List[str]] = Field(default_factory=dict, description="Installed series per vendor")
    series_restrictions: List[str] = Field(default_factory=list, description="Series restrictions")
    available_spare_parts: Dict[str, List[str]] = Field(default_factory=dict, description="Available spare parts per model")
    standardized_vendor: Optional[str] = Field(default=None, description="Plant standardized vendor")
    
    # Computed
    excluded_models: List[str] = Field(default_factory=list, description="Models to exclude")
    boosted_models: List[str] = Field(default_factory=list, description="Models to boost")


class SpecObject(BaseModel):
    """
    Finalized instrument/accessory specification from Detail Capture workflow.
    This is the structured input for comparison workflow.
    """
    product_type: str = Field(description="Product type (e.g., 'pressure transmitter')")
    category: str = Field(default="", description="Product category")
    subcategory: Optional[str] = Field(default=None, description="Product subcategory")
    
    # Technical specifications
    specifications: Dict[str, Any] = Field(default_factory=dict, description="Technical specs (range, accuracy, etc.)")
    
    # Certifications and standards
    required_certifications: List[str] = Field(default_factory=list, description="Required certifications (SIL2, ATEX, etc.)")
    sil_rating: Optional[str] = Field(default=None, description="Required SIL rating")
    atex_zone: Optional[str] = Field(default=None, description="Required ATEX zone")
    
    # Environment
    environment: Optional[str] = Field(default=None, description="Operating environment")
    temperature_range: Optional[str] = Field(default=None, description="Temperature range")
    
    # Source workflow
    source_workflow: str = Field(default="manual", description="Source workflow (solution, instrument_detail, manual)")
    session_id: Optional[str] = Field(default=None, description="Session ID from source workflow")


class ComparisonType(str, Enum):
    """Types of comparison analysis"""
    VENDOR = "vendor"       # Compare across vendors
    SERIES = "series"       # Compare series within vendor
    MODEL = "model"         # Compare models within series
    FULL = "full"           # Full multi-level comparison


class ComparisonInput(BaseModel):
    """
    Input for comparison workflow from UI [COMPARE VENDORS] button.
    """
    spec_object: SpecObject = Field(description="Finalized specification object")
    comparison_type: ComparisonType = Field(default=ComparisonType.FULL, description="Type of comparison")
    session_id: str = Field(default="", description="Session identifier")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    
    # Optional filters
    vendor_filter: Optional[List[str]] = Field(default=None, description="Specific vendors to compare")
    max_candidates: int = Field(default=10, description="Maximum candidates per level")


class VendorModelCandidate(BaseModel):
    """Single vendor/model in candidate pool"""
    vendor: str = Field(description="Vendor name")
    series: str = Field(description="Product series")
    model: str = Field(description="Specific model")
    is_preferred: bool = Field(default=False, description="Is preferred vendor")
    priority_boost: int = Field(default=0, description="Priority boost points")
    meets_standards: bool = Field(default=True, description="Meets standards requirements")
    matches_installed_base: bool = Field(default=False, description="Matches installed base")


class ScoringBreakdown(BaseModel):
    """Detailed scoring for a product"""
    strategy_priority: int = Field(ge=0, le=25, description="Strategy priority score /25")
    technical_fit: int = Field(ge=0, le=25, description="Technical fit score /25")
    asset_alignment: int = Field(ge=0, le=20, description="Asset alignment score /20")
    standards_compliance: int = Field(ge=0, le=15, description="Standards compliance score /15")
    data_completeness: int = Field(ge=0, le=15, description="Data completeness score /15")
    
    @property
    def overall_score(self) -> int:
        return (self.strategy_priority + self.technical_fit + 
                self.asset_alignment + self.standards_compliance + 
                self.data_completeness)


class RankedComparisonProduct(BaseModel):
    """Single ranked product in comparison output"""
    rank: int = Field(ge=1, description="Product rank")
    vendor: str = Field(description="Vendor name")
    model: str = Field(description="Model name")
    overall_score: int = Field(ge=0, le=100, description="Overall score /100")
    scoring_breakdown: ScoringBreakdown = Field(description="Detailed scoring breakdown")
    constraints_met: Dict[str, bool] = Field(default_factory=dict, description="Constraints met status")
    key_advantages: List[str] = Field(default_factory=list, description="Key advantages")


class ComparisonMatrix(BaseModel):
    """Unified comparison results"""
    candidates: List[VendorModelCandidate] = Field(default_factory=list, description="Candidate products")
    within_vendor_comparisons: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict, description="Within-vendor comparisons")
    cross_vendor_comparisons: List[Dict[str, Any]] = Field(default_factory=list, description="Cross-vendor comparisons")
    spec_match_scores: Dict[str, float] = Field(default_factory=dict, description="Spec match scores per model")


class RAGQueryResult(BaseModel):
    """Result from RAG query"""
    source: str = Field(description="RAG source (strategy, standards, inventory)")
    relevant_chunks: List[str] = Field(default_factory=list, description="Relevant text chunks")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Result metadata")
    confidence: float = Field(ge=0, le=1, default=0.0, description="Query confidence")


# ============================================================================
# ENHANCED WORKFLOW STATES
# ============================================================================

class ComparisonState(TypedDict):
    """
    State for Comparative Analysis Workflow
    """
    # User Input
    user_input: str
    session_id: str
    
    # Request Classification
    request_mode: Optional[str]  # "single" or "comparison"
    mode_confidence: float
    
    # Instrument Identification
    instrument_type: Optional[str]
    instrument_category: Optional[str]
    critical_specs: Dict[str, Any]
    
    # RAG Constraints
    constraint_context: Optional[Dict[str, Any]]  # Serialized ConstraintContext
    rag_results: Dict[str, Any]  # Results from all 3 RAGs
    
    # Candidate Filtering
    all_vendors: List[str]
    filtered_candidates: List[Dict[str, Any]]  # Serialized VendorModelCandidate list
    exclusion_reasons: List[Dict[str, str]]
    
    # Parallel Analysis
    vendor_analysis_results: List[Dict[str, Any]]
    within_vendor_comparisons: Dict[str, List[Dict[str, Any]]]
    cross_vendor_comparisons: List[Dict[str, Any]]
    
    # Validation
    validated_results: List[Dict[str, Any]]
    flagged_results: List[Dict[str, Any]]
    removed_results: List[Dict[str, Any]]
    
    # Comparison Matrix
    comparison_matrix: Optional[Dict[str, Any]]
    
    # Ranked Output
    ranked_products: List[Dict[str, Any]]  # Serialized RankedComparisonProduct list
    
    # Workflow Control
    current_phase: str
    messages: Annotated[List[Dict[str, Any]], operator.add]
    
    # Response
    response: Optional[str]
    response_data: Optional[Dict[str, Any]]
    formatted_output: Optional[str]
    
    # Error Handling
    error: Optional[str]


class SolutionState(TypedDict):
    """
    State for Solution-Based Workflow (with Comparison Mode support)
    """
    # User Input
    user_input: str
    session_id: str

    # Intent Classification
    intent: Optional[str]
    intent_confidence: float

    # Validation
    product_type: Optional[str]
    schema: Optional[Dict[str, Any]]
    schema_source: Optional[str]  # "mongodb" or "generated"
    provided_requirements: Dict[str, Any]
    missing_requirements: List[str]
    is_requirements_valid: bool

    # RAG Data
    rag_context: Dict[str, Any]
    strategy_present: bool
    allowed_vendors: List[str]

    # Vendor Analysis
    available_vendors: List[str]
    filtered_vendors: List[str]
    parallel_analysis_results: List[Dict[str, Any]]
    summarized_results: List[Dict[str, Any]]

    # Validation & Ranking
    judge_validation: Dict[str, Any]
    ranked_results: List[Dict[str, Any]]

    # Comparison Mode (NEW - integrated from comparison_workflow)
    comparison_mode: bool  # True if this is a comparison request ("compare", "vs", etc.)
    mode_confidence: float  # Confidence in comparison detection (0.0-1.0)
    comparison_output: Optional[Dict[str, Any]]  # Formatted comparison result with winner/trade-offs

    # Workflow Control
    current_step: str
    next_step: Optional[str]
    requires_user_input: bool
    messages: Annotated[List[Dict[str, Any]], operator.add]

    # Response
    response: Optional[str]
    response_data: Optional[Dict[str, Any]]

    # Error Handling
    error: Optional[str]
    retry_count: int


class InstrumentDetailState(TypedDict):
    """
    State for Instrument/Accessory Detail Capture Workflow
    """
    # User Input
    user_input: str
    session_id: str
    
    # Intent Classification (First)
    initial_intent: Optional[str]
    
    # Instrument Identification
    identified_instruments: List[Dict[str, Any]]
    identified_accessories: List[Dict[str, Any]]
    project_name: Optional[str]
    
    # User Selection
    selected_item: Optional[Dict[str, Any]]
    selected_type: Optional[str]  # "instrument" or "accessory"
    
    # Intent Classification (Second - for detail)
    detail_intent: Optional[str]
    
    # Validation
    product_type: Optional[str]
    schema: Optional[Dict[str, Any]]
    provided_requirements: Dict[str, Any]
    missing_requirements: List[str]
    is_requirements_valid: bool
    
    # RAG Data
    rag_context: Dict[str, Any]
    strategy_present: bool
    allowed_vendors: List[str]
    
    # Vendor Analysis
    available_vendors: List[str]
    filtered_vendors: List[str]
    parallel_analysis_results: List[Dict[str, Any]]
    summarized_results: List[Dict[str, Any]]
    
    # Validation & Ranking
    judge_validation: Dict[str, Any]
    ranked_results: List[Dict[str, Any]]
    
    # Workflow Control
    current_step: str
    next_step: Optional[str]
    requires_user_input: bool
    messages: Annotated[List[Dict[str, Any]], operator.add]
    
    # Response
    response: Optional[str]
    response_data: Optional[Dict[str, Any]]
    
    # Error Handling
    error: Optional[str]


class InstrumentIdentifierState(TypedDict):
    """
    State for Instrument Identifier Workflow (List Generator Only)

    This workflow identifies instruments/accessories from project requirements
    and generates a selection list with sample_input strings.
    It does NOT perform product search - that's handled by the SOLUTION workflow.
    """
    # User Input
    user_input: str
    session_id: str

    # Intent Classification
    initial_intent: Optional[str]

    # Identification Results
    identified_instruments: List[Dict[str, Any]]
    identified_accessories: List[Dict[str, Any]]
    project_name: str

    # Unified Item List (with sample_input for each)
    all_items: List[Dict[str, Any]]  # Each item has: number, type, name, category, sample_input
    total_items: int

    # Workflow Control
    current_step: str
    messages: Annotated[List[Dict[str, Any]], operator.add]

    # Output
    response: Optional[str]
    response_data: Optional[Dict[str, Any]]

    # Error Handling
    error: Optional[str]


class PotentialProductIndexState(TypedDict):
    """
    State for Potential Product Index Sub-Workflow
    """
    # Input from Parent Workflow
    product_type: str
    session_id: str
    parent_workflow: str  # "solution" or "instrument_detail"
    
    # RAG Context
    rag_context: Dict[str, Any]
    
    # Vendor Discovery
    discovered_vendors: List[str]
    vendor_model_families: Dict[str, List[str]]
    
    # Parallel Processing
    vendor_processing_status: Dict[str, str]  # vendor -> status
    pdf_search_results: Dict[str, List[Dict[str, Any]]]
    pdf_download_status: Dict[str, str]
    extracted_content: Dict[str, str]
    
    # Azure Blob Storage
    blob_urls: Dict[str, str]  # vendor -> blob URL
    
    # RAG Indexing
    indexed_chunks: List[Dict[str, Any]]
    vector_store_ids: List[str]
    
    # Schema Generation
    generated_schema: Optional[Dict[str, Any]]
    schema_saved: bool
    
    # Workflow Control
    current_phase: str
    messages: Annotated[List[Dict[str, Any]], operator.add]
    
    # Error Handling
    error: Optional[str]
    failed_vendors: List[str]


# ============================================================================
# STATE FACTORY FUNCTIONS
# ============================================================================

def create_comparison_state(user_input: str, session_id: str) -> ComparisonState:
    """Create initial comparison workflow state"""
    return ComparisonState(
        user_input=user_input,
        session_id=session_id,
        request_mode=None,
        mode_confidence=0.0,
        instrument_type=None,
        instrument_category=None,
        critical_specs={},
        constraint_context=None,
        rag_results={},
        all_vendors=[],
        filtered_candidates=[],
        exclusion_reasons=[],
        vendor_analysis_results=[],
        within_vendor_comparisons={},
        cross_vendor_comparisons=[],
        validated_results=[],
        flagged_results=[],
        removed_results=[],
        comparison_matrix=None,
        ranked_products=[],
        current_phase="classify_request",
        messages=[],
        response=None,
        response_data=None,
        formatted_output=None,
        error=None
    )


def create_solution_state(user_input: str, session_id: str) -> SolutionState:
    """Create initial solution workflow state (with comparison mode support)"""
    return SolutionState(
        user_input=user_input,
        session_id=session_id,
        intent=None,
        intent_confidence=0.0,
        product_type=None,
        schema=None,
        schema_source=None,
        provided_requirements={},
        missing_requirements=[],
        is_requirements_valid=False,
        rag_context={},
        strategy_present=False,
        allowed_vendors=[],
        available_vendors=[],
        filtered_vendors=[],
        parallel_analysis_results=[],
        summarized_results=[],
        judge_validation={},
        ranked_results=[],
        comparison_mode=False,  # NEW: Default to regular search mode
        mode_confidence=0.0,  # NEW: Will be set by comparison detection
        comparison_output=None,  # NEW: Will be populated if comparison mode
        current_step="classify_intent",
        next_step=None,
        requires_user_input=False,
        messages=[],
        response=None,
        response_data=None,
        error=None,
        retry_count=0
    )


def create_instrument_detail_state(user_input: str, session_id: str) -> InstrumentDetailState:
    """Create initial instrument detail workflow state"""
    return InstrumentDetailState(
        user_input=user_input,
        session_id=session_id,
        initial_intent=None,
        identified_instruments=[],
        identified_accessories=[],
        project_name=None,
        selected_item=None,
        selected_type=None,
        detail_intent=None,
        product_type=None,
        schema=None,
        provided_requirements={},
        missing_requirements=[],
        is_requirements_valid=False,
        rag_context={},
        strategy_present=False,
        allowed_vendors=[],
        available_vendors=[],
        filtered_vendors=[],
        parallel_analysis_results=[],
        summarized_results=[],
        judge_validation={},
        ranked_results=[],
        current_step="classify_intent",
        next_step=None,
        requires_user_input=False,
        messages=[],
        response=None,
        response_data=None,
        error=None
    )


def create_instrument_identifier_state(user_input: str, session_id: str) -> InstrumentIdentifierState:
    """Create initial instrument identifier workflow state"""
    return InstrumentIdentifierState(
        user_input=user_input,
        session_id=session_id,
        initial_intent=None,
        identified_instruments=[],
        identified_accessories=[],
        project_name="Project",
        all_items=[],
        total_items=0,
        current_step="classify_intent",
        messages=[],
        response=None,
        response_data=None,
        error=None
    )


def create_potential_product_index_state(
    product_type: str, 
    session_id: str, 
    parent_workflow: str,
    rag_context: Dict[str, Any] = None
) -> PotentialProductIndexState:
    """Create initial potential product index state"""
    return PotentialProductIndexState(
        product_type=product_type,
        session_id=session_id,
        parent_workflow=parent_workflow,
        rag_context=rag_context or {},
        discovered_vendors=[],
        vendor_model_families={},
        vendor_processing_status={},
        pdf_search_results={},
        pdf_download_status={},
        extracted_content={},
        blob_urls={},
        indexed_chunks=[],
        vector_store_ids=[],
        generated_schema=None,
        schema_saved=False,
        current_phase="discover_vendors",
        messages=[],
        error=None,
        failed_vendors=[]
    )

