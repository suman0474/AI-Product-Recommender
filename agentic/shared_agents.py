"""
Shared Agents Module

Reusable agents for validation, verification, and generation.
Used by Product Info, Standards RAG, Grounded Chat, and other workflows.

Migrated from engenie_workflow.py to enable reuse across workflows
without creating circular dependencies.
"""

import json
import logging
import re
import os
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

from llm_fallback import create_llm_with_fallback

# Import Intent Classification Agent for convenience
# (Avoids circular dependency as it doesn't import from shared_agents)
from .intent_classification_routing_agent import (
    IntentClassificationRoutingAgent,
    WorkflowRoutingResult,
    WorkflowTarget
)

# Import DataSource from product_info_intent_agent for convenience
from .product_info_intent_agent import DataSource as IntentDataSource

load_dotenv()

logger = logging.getLogger(__name__)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_json_from_response(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from LLM response, handling various formats.
    
    Handles:
    - Pure JSON responses
    - JSON wrapped in markdown code blocks (```json ... ```)
    - JSON with leading/trailing text
    
    Args:
        text: Raw LLM response text
        
    Returns:
        Parsed JSON dictionary or None if extraction fails
    """
    if not text:
        return None
    
    # Try direct JSON parsing first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    
    # Try extracting from markdown code blocks
    patterns = [
        r'```json\s*([\s\S]*?)\s*```',  # ```json ... ```
        r'```\s*([\s\S]*?)\s*```',       # ``` ... ```
        r'\{[\s\S]*\}',                   # Find any JSON object
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.MULTILINE)
        for match in matches:
            try:
                cleaned = match.strip() if isinstance(match, str) else match
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
    
    # Last resort: try to find JSON-like structure
    try:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_str = text[start:end + 1]
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    logger.warning(f"Failed to extract JSON from response: {text[:200]}...")
    return None


# ============================================================================
# OUTPUT MODELS
# ============================================================================

class ChatResponse(BaseModel):
    """Output from ChatAgent"""
    answer: str = Field(description="The grounded answer to the user's question")
    citations: List[Dict[str, str]] = Field(default_factory=list, description="Source citations")
    rag_sources_used: List[str] = Field(default_factory=list, description="RAG sources queried")
    confidence: float = Field(default=0.0, description="Confidence score 0-1")


class ValidationResult(BaseModel):
    """Output from ResponseValidatorAgent"""
    is_valid: bool = Field(description="Whether the response passes validation")
    overall_score: float = Field(default=0.0, description="Overall validation score 0-1")
    relevance_score: float = Field(default=0.0, description="Relevance to question 0-1")
    accuracy_score: float = Field(default=0.0, description="Factual accuracy 0-1")
    grounding_score: float = Field(default=0.0, description="Grounded in context 0-1")
    citation_score: float = Field(default=0.0, description="Proper citations 0-1")
    hallucination_detected: bool = Field(default=False, description="Hallucination detected")
    issues_found: List[str] = Field(default_factory=list, description="Issues to fix")
    suggestions: str = Field(default="", description="Suggestions for improvement")


class WebVerificationResult(BaseModel):
    """Output from WebSearchVerifierAgent"""
    fact_check_score: float = Field(default=0.5, description="Fact-check score 0-1")
    credibility_score: float = Field(default=0.5, description="Source credibility 0-1")
    hallucination_risk: str = Field(default="medium", description="low/medium/high")
    cross_reference_status: str = Field(default="partial", description="verified/partial/unverified")
    reliable_sources: List[str] = Field(default_factory=list, description="Trustworthy URLs")
    warnings: List[str] = Field(default_factory=list, description="Risk indicators")


# ============================================================================
# PROMPTS
# ============================================================================

CHAT_AGENT_PROMPT = """
You are Engenie's Industrial Instrumentation Expert. Answer user questions using ONLY the provided context.

USER QUESTION: {question}

PRODUCT TYPE: {product_type}

RAG CONTEXT (Company-specific knowledge):
{rag_context}

COMPANY PREFERENCES:
- Preferred Vendors: {preferred_vendors}
- Required Standards: {required_standards}
- Installed Series: {installed_series}

INSTRUCTIONS:
1. Answer ONLY based on the provided context
2. If information is not in context, say "I don't have specific information about that"
3. Cite your sources using [Source: <source_name>] format
4. Be conversational but precise
5. Mention specific product series, models, or standards when relevant
6. NO HALLUCINATION - Do not make up information

Return ONLY valid JSON:
{{
    "answer": "<your grounded answer with [Source: ...] citations>",
    "citations": [
        {{"source": "<source_name>", "content": "<relevant quote>"}}
    ],
    "rag_sources_used": ["Strategy RAG", "Standards RAG", "Inventory RAG"],
    "confidence": <0.0-1.0 based on context relevance>
}}
"""

VALIDATOR_PROMPT = """
You are a Response Quality Validator. Evaluate this response for grounding and accuracy.

USER QUESTION: {question}

GENERATED RESPONSE: {response}

AVAILABLE CONTEXT: {context}

Perform these 5 validation checks:

1. RELEVANCE (0-1): Does the response directly address the user's question?
2. ACCURACY (0-1): Is the information factually correct based on context?
3. GROUNDING (0-1): Is the response grounded in the provided context (no external info)?
4. CITATIONS (0-1): Are sources properly cited with [Source: ...] format?
5. HALLUCINATION: Is there any fabricated/made-up information NOT in context?

Return ONLY valid JSON:
{{
    "is_valid": <true if all scores >= 0.6 and no hallucination>,
    "overall_score": <average of 4 scores>,
    "relevance_score": <0.0-1.0>,
    "accuracy_score": <0.0-1.0>,
    "grounding_score": <0.0-1.0>,
    "citation_score": <0.0-1.0>,
    "hallucination_detected": <true/false>,
    "issues_found": ["<issue 1>", "<issue 2>"],
    "suggestions": "<how to fix issues>"
}}
"""

WEB_VERIFIER_PROMPT = """
You are a Web Search Results Verifier for industrial instrumentation questions.

USER QUESTION: {question}

WEB RESULTS:
{web_results}

EXTRACTED ENTITIES: {entities}

Perform 4-dimensional verification:

1. FACT_CHECK (0-1): Are the facts in these results verifiable and consistent across sources?
2. CREDIBILITY (0-1): Are these sources authoritative? (vendor sites, technical docs, journals > forums, blogs)
3. HALLUCINATION_RISK: low/medium/high - Risk of using unverified information
4. CROSS_REFERENCE: verified/partial/unverified - Do sources corroborate each other?

Return ONLY valid JSON:
{{
    "fact_check_score": <0.0-1.0>,
    "credibility_score": <0.0-1.0>,
    "hallucination_risk": "low" | "medium" | "high",
    "cross_reference_status": "verified" | "partial" | "unverified",
    "reliable_sources": ["<source1>", "<source2>"],
    "warnings": ["<warning if any>"]
}}
"""


# ============================================================================
# RESPONSE VALIDATOR AGENT
# ============================================================================

class ResponseValidatorAgent:
    """
    Validates responses for grounding, accuracy, and hallucination.

    5-dimensional validation:
    1. Relevance - Does response address the question?
    2. Accuracy - Is information factually correct?
    3. Grounding - Is response grounded in context?
    4. Citations - Are sources properly cited?
    5. Hallucination - Any fabricated information?
    """

    def __init__(self, llm=None, components: Dict = None):
        self.components = components or {}
        self.llm = llm or create_llm_with_fallback(
            model="gemini-2.5-flash",
            temperature=0.1,  # Low temperature for objective validation
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        logger.info("ResponseValidatorAgent initialized")

    def run(
        self,
        response: str,
        user_question: str,
        context: str = ""
    ) -> Dict[str, Any]:
        """Validate a generated response.

        Args:
            response: The generated response to validate
            user_question: The original user question
            context: Available context from sources

        Returns:
            Validation result with scores and issues
        """
        logger.info("ResponseValidatorAgent validating response...")

        try:
            prompt = ChatPromptTemplate.from_template(VALIDATOR_PROMPT)
            parser = JsonOutputParser()
            chain = prompt | self.llm | parser
            
            result = None
            
            # Try using the chain with JsonOutputParser first
            try:
                result = chain.invoke({
                    "question": user_question,
                    "response": response,
                    "context": context or "No context provided"
                })
            except (OutputParserException, json.JSONDecodeError) as parse_error:
                # Fallback: invoke LLM directly without parser
                logger.warning(f"Validation parsing failed, attempting fallback: {parse_error}")
                
                raw_chain = prompt | self.llm
                raw_response = raw_chain.invoke({
                    "question": user_question,
                    "response": response,
                    "context": context or "No context provided"
                })
                
                # Extract text content
                if hasattr(raw_response, 'content'):
                    raw_text = raw_response.content
                elif hasattr(raw_response, 'text'):
                    raw_text = raw_response.text
                else:
                    raw_text = str(raw_response)
                
                # Use helper function to extract JSON
                result = extract_json_from_response(raw_text)
                
                if result:
                    logger.info("Successfully extracted validation JSON using fallback")
                else:
                    # Default to assuming valid if we can't parse
                    logger.warning("Could not parse validation result, defaulting to valid=True")
                    result = {
                        "is_valid": True,
                        "overall_score": 0.7,
                        "parsing_fallback": True
                    }

            if result is None:
                result = {}

            is_valid = result.get("is_valid", False)
            overall_score = result.get("overall_score", 0.0)

            logger.info(f"Validation complete: valid={is_valid}, score={overall_score:.2f}")

            return {
                "success": True,
                "is_valid": is_valid,
                "overall_score": overall_score,
                "relevance_score": result.get("relevance_score", 0.0),
                "accuracy_score": result.get("accuracy_score", 0.0),
                "grounding_score": result.get("grounding_score", 0.0),
                "citation_score": result.get("citation_score", 0.0),
                "hallucination_detected": result.get("hallucination_detected", False),
                "issues_found": result.get("issues_found", []),
                "suggestions": result.get("suggestions", "")
            }

        except Exception as e:
            logger.error(f"ResponseValidatorAgent failed: {e}")
            return {
                "success": False,
                "is_valid": True,  # Default to valid to avoid blocking the workflow
                "overall_score": 0.5,
                "issues_found": [f"Validation error: {str(e)}"],
                "error": str(e)
            }


# ============================================================================
# WEB SEARCH VERIFIER AGENT
# ============================================================================

class WebSearchVerifierAgent:
    """
    Verifies web search results for credibility and accuracy.

    Performs 4-dimensional verification:
    1. Fact-checking across sources
    2. Source credibility assessment
    3. Hallucination risk evaluation
    4. Cross-reference status
    """

    def __init__(self, llm=None, components: Dict = None):
        self.components = components or {}
        self.llm = llm or create_llm_with_fallback(
            model="gemini-2.5-flash",
            temperature=0.1,  # Low temperature for objective verification
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        logger.info("WebSearchVerifierAgent initialized")

    def run(
        self,
        user_question: str,
        web_results: List[Dict[str, Any]],
        entities: List[str] = None
    ) -> Dict[str, Any]:
        """Verify web search results.

        Args:
            user_question: The user's question
            web_results: List of web search results
            entities: Extracted entities from the question

        Returns:
            Verification result with scores and warnings
        """
        logger.info(f"WebSearchVerifierAgent verifying {len(web_results)} results...")

        try:
            # Format web results for prompt
            results_text = ""
            for i, result in enumerate(web_results[:5], 1):
                title = result.get("title", "No title")
                snippet = result.get("snippet", result.get("description", "No snippet"))
                link = result.get("link", result.get("url", "No link"))
                results_text += f"\n{i}. [{title}]({link})\n   {snippet}\n"

            prompt = ChatPromptTemplate.from_template(WEB_VERIFIER_PROMPT)
            parser = JsonOutputParser()
            chain = prompt | self.llm | parser

            result = chain.invoke({
                "question": user_question,
                "web_results": results_text,
                "entities": ", ".join(entities) if entities else "None specified"
            })

            logger.info(f"Verification: credibility={result.get('credibility_score', 0):.2f}, "
                       f"risk={result.get('hallucination_risk', 'unknown')}")

            return {
                "success": True,
                "fact_check_score": result.get("fact_check_score", 0.5),
                "credibility_score": result.get("credibility_score", 0.5),
                "hallucination_risk": result.get("hallucination_risk", "medium"),
                "cross_reference_status": result.get("cross_reference_status", "partial"),
                "reliable_sources": result.get("reliable_sources", []),
                "warnings": result.get("warnings", [])
            }

        except Exception as e:
            logger.error(f"WebSearchVerifierAgent failed: {e}")
            return {
                "success": False,
                "fact_check_score": 0.5,
                "credibility_score": 0.5,
                "hallucination_risk": "medium",
                "cross_reference_status": "partial",
                "reliable_sources": [],
                "warnings": [f"Verification error: {str(e)}"],
                "error": str(e)
            }


# ============================================================================
# CHAT AGENT (Optional - for generating grounded responses)
# ============================================================================

class ChatAgent:
    """
    Generates grounded responses with citations using RAG context.
    """

    def __init__(self, llm=None, components: Dict = None):
        self.components = components or {}
        self.llm = llm or create_llm_with_fallback(
            model="gemini-2.5-flash",
            temperature=0.3,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        logger.info("ChatAgent initialized")

    def run(
        self,
        user_question: str,
        product_type: str = "",
        rag_context: Dict[str, Any] = None,
        specifications: Dict[str, Any] = None,
        user_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Generate a grounded answer to the user's question.

        Args:
            user_question: The user's question
            product_type: Type of product being queried
            rag_context: RAG context from various sources
            specifications: Company specifications and preferences
            user_context: Additional user context

        Returns:
            Generated response with citations
        """
        logger.info(f"ChatAgent processing: {user_question[:50]}...")

        try:
            rag_context = rag_context or {}
            specifications = specifications or {}

            rag_context_str = json.dumps(rag_context, indent=2) if isinstance(rag_context, dict) else str(rag_context)

            preferred_vendors = specifications.get("preferred_vendors", [])
            required_standards = specifications.get("required_standards", [])
            installed_series = specifications.get("installed_series", [])

            prompt = ChatPromptTemplate.from_template(CHAT_AGENT_PROMPT)
            parser = JsonOutputParser()
            chain = prompt | self.llm | parser

            result = chain.invoke({
                "question": user_question,
                "product_type": product_type or "general",
                "rag_context": rag_context_str,
                "preferred_vendors": ", ".join(preferred_vendors) if preferred_vendors else "Not specified",
                "required_standards": ", ".join(required_standards) if required_standards else "Not specified",
                "installed_series": ", ".join(installed_series) if installed_series else "Not specified"
            })

            return {
                "success": True,
                "answer": result.get("answer", ""),
                "citations": result.get("citations", []),
                "rag_sources_used": result.get("rag_sources_used", []),
                "confidence": result.get("confidence", 0.0)
            }

        except Exception as e:
            logger.error(f"ChatAgent failed: {e}")
            return {
                "success": False,
                "answer": "I apologize, but I encountered an error generating an answer.",
                "citations": [],
                "rag_sources_used": [],
                "confidence": 0.0,
                "error": str(e)
            }


# ============================================================================
# SESSION MANAGER AGENT
# ============================================================================

class SessionManagerAgent:
    """
    Manages multi-turn conversation memory.
    """

    def __init__(self, llm=None, components: Dict = None):
        self.components = components or {}
        self.sessions: Dict[str, Dict] = {}
        logger.info("SessionManagerAgent initialized")

    def run(
        self,
        session_id: str,
        question: str = None,
        answer: str = None,
        validation_score: float = 0.0,
        citations: List[Dict] = None,
        operation: str = "update"
    ) -> Dict[str, Any]:
        """Manage session state.

        Args:
            session_id: Session identifier
            question: User question (for update)
            answer: Generated answer (for update)
            validation_score: Validation score (for update)
            citations: Source citations (for update)
            operation: Operation type - "update", "get", or "clear"

        Returns:
            Session state information
        """
        from datetime import datetime
        logger.info(f"SessionManagerAgent: {operation} for session {session_id}")

        try:
            if operation == "clear":
                if session_id in self.sessions:
                    del self.sessions[session_id]
                return {
                    "success": True,
                    "session_id": session_id,
                    "total_interactions": 0,
                    "updated": True,
                    "message": "Session cleared"
                }

            if operation == "get":
                session = self.sessions.get(session_id, {})
                return {
                    "success": True,
                    "session_id": session_id,
                    "total_interactions": len(session.get("interactions", [])),
                    "interactions": session.get("interactions", []),
                    "updated": False
                }

            # Update operation
            if session_id not in self.sessions:
                self.sessions[session_id] = {
                    "created_at": datetime.utcnow().isoformat(),
                    "interactions": []
                }

            session = self.sessions[session_id]

            if question and answer:
                interaction = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "question": question,
                    "answer": answer,
                    "validation_score": validation_score,
                    "citations": citations or []
                }
                session["interactions"].append(interaction)
                session["last_updated"] = datetime.utcnow().isoformat()

            total_interactions = len(session["interactions"])

            logger.info(f"Session {session_id}: {total_interactions} interactions")

            return {
                "success": True,
                "session_id": session_id,
                "total_interactions": total_interactions,
                "updated": True
            }

        except Exception as e:
            logger.error(f"SessionManagerAgent failed: {e}")
            return {
                "success": False,
                "session_id": session_id,
                "error": str(e)
            }

    def get_conversation_context(self, session_id: str, max_turns: int = 5) -> str:
        """Get formatted conversation history for context.

        Args:
            session_id: Session identifier
            max_turns: Maximum number of turns to include

        Returns:
            Formatted conversation context string
        """
        session = self.sessions.get(session_id, {})
        interactions = session.get("interactions", [])

        if not interactions:
            return ""

        recent = interactions[-max_turns:]

        context_parts = ["Previous conversation:"]
        for i, interaction in enumerate(recent, 1):
            context_parts.append(f"Q{i}: {interaction['question']}")
            context_parts.append(f"A{i}: {interaction['answer'][:200]}...")

        return "\n".join(context_parts)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def optimize_web_query(
    user_question: str,
    entities: List[str]
) -> str:
    """
    Optimize user question for web search.

    - Remove question words
    - Add key entities
    - Add industrial context if needed

    Args:
        user_question: Original user question
        entities: Extracted entities from the question

    Returns:
        Optimized search query
    """
    query = user_question.lower()

    # Remove common question words
    question_words = ["what is", "how does", "why", "explain", "tell me about", "can you"]
    for qw in question_words:
        query = query.replace(qw, "")

    # Add entities
    if entities:
        query = f"{query} {' '.join(entities[:3])}"

    # Clean up
    query = query.strip()

    # Add industrial context if not already present
    industrial_keywords = ["industrial", "instrumentation", "transmitter", "sensor", "valve", "automation"]
    if not any(kw in query.lower() for kw in industrial_keywords):
        query = f"{query} industrial instrumentation"

    return query


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Models
    "ChatResponse",
    "ValidationResult",
    "WebVerificationResult",
    "WorkflowRoutingResult",
    # Agents
    "ResponseValidatorAgent",
    "WebSearchVerifierAgent",
    "ChatAgent",
    "SessionManagerAgent",
    "IntentClassificationRoutingAgent",
    # Prompts
    "CHAT_AGENT_PROMPT",
    "VALIDATOR_PROMPT",
    "WEB_VERIFIER_PROMPT",
    # Functions
    "optimize_web_query",
    "extract_json_from_response",
    # Intent Classification (re-exported for convenience)
    "IntentDataSource"  # Aliased from DataSource to avoid naming conflicts
]
