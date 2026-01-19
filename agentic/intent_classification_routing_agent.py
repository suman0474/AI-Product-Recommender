"""
Intent Classification Routing Agent

Routes user input from the UI textarea to the appropriate agentic workflow:
1. Solution Workflow - Complex engineering challenges requiring multiple instruments
2. Instrument Identifier Workflow - Single product requirements
3. Product Info Workflow - Questions about products, standards, vendors

Also rejects out-of-domain queries (unrelated to industrial automation).

Usage:
    agent = IntentClassificationRoutingAgent()
    result = agent.classify(query="I need a pressure transmitter 0-100 PSI", session_id="abc123")
    # Returns: WorkflowRoutingResult with target workflow and reasoning
"""

import logging
from typing import Dict, Optional, Any
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from threading import Lock

logger = logging.getLogger(__name__)


# =============================================================================
# WORKFLOW STATE MEMORY (Session-based memory for workflow locking)
# =============================================================================

class WorkflowStateMemory:
    """
    Singleton memory class to track workflow state per session.
    
    This is the SINGLE SOURCE OF TRUTH for workflow state - stored entirely
    in the backend, not dependent on frontend state.
    
    Usage:
        memory = WorkflowStateMemory()
        memory.set_workflow("session_123", "engenie_chat")
        current = memory.get_workflow("session_123")  # Returns "engenie_chat"
        memory.clear_workflow("session_123")
    """
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._workflow_states: Dict[str, str] = {}
        self._workflow_timestamps: Dict[str, datetime] = {}
        self._state_lock = Lock()
        logger.info("[WorkflowStateMemory] Initialized - Backend workflow state tracking enabled")
    
    def get_workflow(self, session_id: str) -> Optional[str]:
        """Get current workflow for a session."""
        with self._state_lock:
            workflow = self._workflow_states.get(session_id)
            if workflow:
                logger.debug(f"[WorkflowStateMemory] Session {session_id[:8]}...: current workflow = {workflow}")
            return workflow
    
    def set_workflow(self, session_id: str, workflow: str) -> None:
        """Set workflow for a session."""
        with self._state_lock:
            old_workflow = self._workflow_states.get(session_id)
            self._workflow_states[session_id] = workflow
            self._workflow_timestamps[session_id] = datetime.now()
            logger.info(f"[WorkflowStateMemory] Session {session_id[:8]}...: workflow changed {old_workflow} -> {workflow}")
    
    def clear_workflow(self, session_id: str) -> None:
        """Clear workflow for a session (allows re-classification)."""
        with self._state_lock:
            old_workflow = self._workflow_states.pop(session_id, None)
            self._workflow_timestamps.pop(session_id, None)
            if old_workflow:
                logger.info(f"[WorkflowStateMemory] Session {session_id[:8]}...: workflow cleared (was {old_workflow})")
    
    def is_locked(self, session_id: str) -> bool:
        """Check if session has an active workflow."""
        return self.get_workflow(session_id) is not None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        with self._state_lock:
            return {
                "active_sessions": len(self._workflow_states),
                "workflows": dict(self._workflow_states)
            }

# Global singleton instance
_workflow_memory = WorkflowStateMemory()

def get_workflow_memory() -> WorkflowStateMemory:
    """Get the global workflow state memory instance."""
    return _workflow_memory


# =============================================================================
# EXIT DETECTION
# =============================================================================

# Phrases that indicate user wants to exit current workflow
EXIT_PHRASES = [
    "start over", "new search", "reset", "clear", "begin again", 
    "start new", "exit", "quit", "cancel", "back to start"
]

# Greetings that indicate new conversation
GREETING_PHRASES = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]

def should_exit_workflow(user_input: str) -> bool:
    """Check if user wants to exit current workflow."""
    lower_input = user_input.lower().strip()
    
    # Check for exit phrases
    if any(phrase in lower_input for phrase in EXIT_PHRASES):
        return True
    
    # Check for pure greetings (new conversation)
    if lower_input in GREETING_PHRASES:
        return True
    
    return False


# =============================================================================
# WORKFLOW TARGETS
# =============================================================================

class WorkflowTarget(Enum):
    """Available workflow routing targets."""
    SOLUTION_WORKFLOW = "solution"              # Complex systems, multiple instruments
    INSTRUMENT_IDENTIFIER = "instrument_identifier"  # Single product requirements
    PRODUCT_INFO = "product_info"               # Questions, greetings, confirmations
    OUT_OF_DOMAIN = "out_of_domain"             # Unrelated queries


# =============================================================================
# WORKFLOW ROUTING RESULT
# =============================================================================

@dataclass
class WorkflowRoutingResult:
    """Result of workflow routing classification."""
    query: str                          # Original query
    target_workflow: WorkflowTarget     # Which workflow to route to
    intent: str                         # Raw intent from classify_intent_tool
    confidence: float                   # Confidence (0.0-1.0)
    reasoning: str                      # Explanation for routing decision
    is_solution: bool                   # Whether this is a solution-type request
    solution_indicators: list           # Indicators that triggered solution detection
    extracted_info: Dict                # Any extracted information
    classification_time_ms: float       # Time taken to classify
    timestamp: str                      # ISO timestamp
    reject_message: Optional[str]       # Message for out-of-domain queries

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "query": self.query,
            "target_workflow": self.target_workflow.value,
            "intent": self.intent,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "is_solution": self.is_solution,
            "solution_indicators": self.solution_indicators,
            "extracted_info": self.extracted_info,
            "classification_time_ms": self.classification_time_ms,
            "timestamp": self.timestamp,
            "reject_message": self.reject_message
        }


# =============================================================================
# OUT OF DOMAIN RESPONSE
# =============================================================================

OUT_OF_DOMAIN_MESSAGE = """
I'm EnGenie, your industrial automation and procurement assistant.

I can help you with:
• **Instrument Identification** - Finding sensors, transmitters, valves, and accessories
• **Solution Design** - Complete instrumentation systems for your process needs
• **Product Information** - Specifications, datasheets, and comparisons
• **Standards & Compliance** - IEC, ISO, SIL, ATEX, and other certifications

Please ask a question related to industrial automation or procurement.
"""


# =============================================================================
# INTENT TO WORKFLOW MAPPING
# =============================================================================

INTENT_TO_WORKFLOW_MAP = {
    # Solution Workflow - Complex systems
    "solution": WorkflowTarget.SOLUTION_WORKFLOW,

    # Instrument Identifier Workflow - Single products
    "requirements": WorkflowTarget.INSTRUMENT_IDENTIFIER,
    "additional_specs": WorkflowTarget.INSTRUMENT_IDENTIFIER,

    # Product Info Workflow - Questions, greetings, workflow control
    "question": WorkflowTarget.PRODUCT_INFO,
    "productInfo": WorkflowTarget.PRODUCT_INFO,
    "greeting": WorkflowTarget.PRODUCT_INFO,
    "confirm": WorkflowTarget.PRODUCT_INFO,   # Continue current workflow
    "reject": WorkflowTarget.PRODUCT_INFO,    # Cancel current workflow

    # Out of Domain - Reject
    "chitchat": WorkflowTarget.OUT_OF_DOMAIN,
    "unrelated": WorkflowTarget.OUT_OF_DOMAIN,
}


# =============================================================================
# INTENT CLASSIFICATION ROUTING AGENT
# =============================================================================

class IntentClassificationRoutingAgent:
    """
    Agent that classifies user queries and routes to appropriate workflows.

    Uses classify_intent_tool as the core classifier and maps intents to
    workflow targets.
    
    WORKFLOW STATE LOCKING:
    - Tracks which workflow each session is in via WorkflowStateMemory
    - Once a session enters a workflow, subsequent queries stay in that workflow
    - User can exit workflow by saying "start over", "reset", etc.

    Workflow Routing:
    - solution → Solution Workflow (complex systems)
    - requirements, additional_specs → Instrument Identifier Workflow
    - question, productInfo, greeting, confirm, reject → Product Info Workflow
    - chitchat, unrelated → OUT_OF_DOMAIN (reject)
    """

    def __init__(self, name: str = "WorkflowRouter"):
        """Initialize the agent."""
        self.name = name
        self.classification_count = 0
        self.last_classification_time_ms = 0.0
        self._memory = get_workflow_memory()  # Use singleton workflow memory
        logger.info(f"[{self.name}] Initialized - Workflow Routing Agent with state memory")

    def classify(
        self, 
        query: str, 
        session_id: str = "default",
        context: Optional[Dict] = None
    ) -> WorkflowRoutingResult:
        """
        Classify a query and determine which workflow to route to.
        
        WORKFLOW LOCKING: If session is already in a workflow, returns that
        workflow without re-classification (unless user wants to exit).

        Args:
            query: User query string from UI textarea
            session_id: Session ID for workflow state tracking
            context: Optional context (current workflow step, conversation history)

        Returns:
            WorkflowRoutingResult with target workflow and details
        """
        start_time = datetime.now()
        
        logger.info(f"[{self.name}] Classifying: '{query[:80]}...' (session: {session_id[:8]}...)")

        # =====================================================================
        # STEP 1: CHECK FOR EXIT REQUEST
        # =====================================================================
        if should_exit_workflow(query):
            self._memory.clear_workflow(session_id)
            logger.info(f"[{self.name}] Exit detected - clearing workflow state")
        
        # =====================================================================
        # STEP 2: CHECK WORKFLOW LOCK
        # =====================================================================
        current_workflow = self._memory.get_workflow(session_id)
        
        if current_workflow and not should_exit_workflow(query):
            # Session is LOCKED in a workflow - return that workflow
            logger.info(f"[{self.name}] WORKFLOW LOCKED: Session in '{current_workflow}' - skipping classification")
            
            # Map workflow to target
            workflow_map = {
                "engenie_chat": WorkflowTarget.PRODUCT_INFO,
                "product_info": WorkflowTarget.PRODUCT_INFO,
                "instrument_identifier": WorkflowTarget.INSTRUMENT_IDENTIFIER,
                "solution": WorkflowTarget.SOLUTION_WORKFLOW
            }
            
            target_workflow = workflow_map.get(current_workflow, WorkflowTarget.PRODUCT_INFO)
            
            # Calculate time
            end_time = datetime.now()
            classification_time_ms = (end_time - start_time).total_seconds() * 1000
            
            return WorkflowRoutingResult(
                query=query,
                target_workflow=target_workflow,
                intent="workflow_locked",
                confidence=1.0,
                reasoning=f"Session locked in {current_workflow} workflow",
                is_solution=(current_workflow == "solution"),
                solution_indicators=[],
                extracted_info={"workflow_locked": True, "current_workflow": current_workflow},
                classification_time_ms=classification_time_ms,
                timestamp=datetime.now().isoformat(),
                reject_message=None
            )

        # =====================================================================
        # STEP 3: NORMAL CLASSIFICATION (no lock or exit requested)
        # =====================================================================
        
        # Import classify_intent_tool here to avoid circular imports
        try:
            from tools.intent_tools import classify_intent_tool
        except ImportError:
            logger.error("Could not import classify_intent_tool")
            return self._create_error_result(query, start_time, "Import error")

        # Get context values
        current_step = context.get("current_step") if context else None
        context_str = context.get("context") if context else None

        # Call the core classifier
        try:
            intent_result = classify_intent_tool.invoke({
                "user_input": query,
                "current_step": current_step,
                "context": context_str
            })
        except Exception as e:
            logger.error(f"[{self.name}] classify_intent_tool failed: {e}")
            return self._create_error_result(query, start_time, str(e))

        # Extract intent details
        intent = intent_result.get("intent", "unrelated")
        confidence = intent_result.get("confidence", 0.5)
        is_solution = intent_result.get("is_solution", False)
        solution_indicators = intent_result.get("solution_indicators", [])
        extracted_info = intent_result.get("extracted_info", {})

        # Map intent to workflow
        target_workflow = INTENT_TO_WORKFLOW_MAP.get(intent, WorkflowTarget.OUT_OF_DOMAIN)
        
        # Override: if is_solution flag is set, force Solution Workflow
        if is_solution and target_workflow != WorkflowTarget.SOLUTION_WORKFLOW:
            target_workflow = WorkflowTarget.SOLUTION_WORKFLOW
            logger.info(f"[{self.name}] Overriding to SOLUTION due to is_solution=True")

        # =====================================================================
        # STEP 4: SET WORKFLOW STATE FOR SESSION
        # =====================================================================
        workflow_name = {
            WorkflowTarget.PRODUCT_INFO: "engenie_chat",
            WorkflowTarget.INSTRUMENT_IDENTIFIER: "instrument_identifier",
            WorkflowTarget.SOLUTION_WORKFLOW: "solution"
        }.get(target_workflow)
        
        if workflow_name:
            self._memory.set_workflow(session_id, workflow_name)
            logger.info(f"[{self.name}] Workflow state set: {workflow_name}")

        # Build reasoning
        reasoning = self._build_reasoning(intent, target_workflow, is_solution, solution_indicators)
        
        # Prepare reject message for out-of-domain
        reject_message = OUT_OF_DOMAIN_MESSAGE if target_workflow == WorkflowTarget.OUT_OF_DOMAIN else None

        # Calculate classification time
        end_time = datetime.now()
        classification_time_ms = (end_time - start_time).total_seconds() * 1000
        self.last_classification_time_ms = classification_time_ms
        self.classification_count += 1

        result = WorkflowRoutingResult(
            query=query,
            target_workflow=target_workflow,
            intent=intent,
            confidence=confidence,
            reasoning=reasoning,
            is_solution=is_solution,
            solution_indicators=solution_indicators,
            extracted_info=extracted_info,
            classification_time_ms=classification_time_ms,
            timestamp=datetime.now().isoformat(),
            reject_message=reject_message
        )

        logger.info(f"[{self.name}] Result: {target_workflow.value} (intent={intent}, conf={confidence:.2f}) in {classification_time_ms:.1f}ms")

        return result

    def _build_reasoning(
        self,
        intent: str,
        target_workflow: WorkflowTarget,
        is_solution: bool,
        solution_indicators: list
    ) -> str:
        """Build human-readable reasoning for the routing decision."""

        if target_workflow == WorkflowTarget.SOLUTION_WORKFLOW:
            if solution_indicators:
                return f"Solution detected: {', '.join(solution_indicators[:3])}"
            return "Complex system requiring multiple instruments detected"

        elif target_workflow == WorkflowTarget.INSTRUMENT_IDENTIFIER:
            return "Single product requirements detected"

        elif target_workflow == WorkflowTarget.PRODUCT_INFO:
            if intent == "greeting":
                return "Greeting detected"
            elif intent == "confirm":
                return "User confirmation detected"
            elif intent == "reject":
                return "User rejection/cancellation detected"
            return "Product/standards question detected"

        elif target_workflow == WorkflowTarget.OUT_OF_DOMAIN:
            return f"Out of domain: '{intent}' is not related to industrial automation"

        return f"Classified as '{intent}'"

    def _create_error_result(self, query: str, start_time: datetime, error: str) -> WorkflowRoutingResult:
        """Create an error result."""
        end_time = datetime.now()
        classification_time_ms = (end_time - start_time).total_seconds() * 1000
        
        return WorkflowRoutingResult(
            query=query,
            target_workflow=WorkflowTarget.OUT_OF_DOMAIN,
            intent="error",
            confidence=0.0,
            reasoning=f"Classification error: {error}",
            is_solution=False,
            solution_indicators=[],
            extracted_info={},
            classification_time_ms=classification_time_ms,
            timestamp=datetime.now().isoformat(),
            reject_message=OUT_OF_DOMAIN_MESSAGE
        )

    def get_stats(self) -> Dict:
        """Get agent statistics."""
        return {
            "name": self.name,
            "classification_count": self.classification_count,
            "last_classification_time_ms": self.last_classification_time_ms
        }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def route_to_workflow(query: str, context: Optional[Dict] = None) -> WorkflowRoutingResult:
    """
    Convenience function for quick workflow routing.

    Args:
        query: User query string
        context: Optional context dict

    Returns:
        WorkflowRoutingResult
    """
    agent = IntentClassificationRoutingAgent()
    return agent.classify(query, context)


def get_workflow_target(query: str) -> str:
    """
    Get just the workflow target name.

    Args:
        query: User query string

    Returns:
        Workflow target value (e.g., "solution", "instrument_identifier")
    """
    result = route_to_workflow(query)
    return result.target_workflow.value


def is_valid_domain_query(query: str) -> bool:
    """
    Check if a query is within the valid domain.

    Args:
        query: User query string

    Returns:
        True if valid, False if out-of-domain
    """
    result = route_to_workflow(query)
    return result.target_workflow != WorkflowTarget.OUT_OF_DOMAIN


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'WorkflowTarget',
    'WorkflowRoutingResult',
    'IntentClassificationRoutingAgent',
    'WorkflowStateMemory',
    'get_workflow_memory',
    'should_exit_workflow',
    'route_to_workflow',
    'get_workflow_target',
    'is_valid_domain_query',
    'OUT_OF_DOMAIN_MESSAGE',
    'EXIT_PHRASES',
    'GREETING_PHRASES'
]
