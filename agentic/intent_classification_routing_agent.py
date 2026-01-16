"""
Intent Classification Routing Agent

Routes user input from the UI textarea to the appropriate agentic workflow:
1. Solution Workflow - Complex engineering challenges requiring multiple instruments
2. Instrument Identifier Workflow - Single product requirements
3. Product Info Workflow - Questions about products, standards, vendors

Also rejects out-of-domain queries (unrelated to industrial automation).

Usage:
    agent = IntentClassificationRoutingAgent()
    result = agent.classify(query="I need a pressure transmitter 0-100 PSI")
    # Returns: WorkflowRoutingResult with target workflow and reasoning
"""

import logging
from typing import Dict, Optional, Any
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


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
        logger.info(f"[{self.name}] Initialized - Workflow Routing Agent")

    def classify(self, query: str, context: Optional[Dict] = None) -> WorkflowRoutingResult:
        """
        Classify a query and determine which workflow to route to.

        Args:
            query: User query string from UI textarea
            context: Optional context (current workflow step, conversation history)

        Returns:
            WorkflowRoutingResult with target workflow and details
        """
        start_time = datetime.now()
        
        logger.info(f"[{self.name}] Classifying: '{query[:80]}...'")

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
    'route_to_workflow',
    'get_workflow_target',
    'is_valid_domain_query',
    'OUT_OF_DOMAIN_MESSAGE'
]
