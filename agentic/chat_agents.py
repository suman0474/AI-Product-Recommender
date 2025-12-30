# agentic/chat_agents.py
# Chat-specific agents for Grounded Knowledge Q&A Workflow

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

import os
from dotenv import load_dotenv
from llm_fallback import create_llm_with_fallback

load_dotenv()
logger = logging.getLogger(__name__)


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


class SessionInfo(BaseModel):
    """Output from SessionManagerAgent"""
    session_id: str = Field(description="Session identifier")
    total_interactions: int = Field(default=0, description="Total Q&A pairs in session")
    updated: bool = Field(default=False, description="Whether session was updated")


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


# ============================================================================
# CHAT AGENT
# ============================================================================

class ChatAgent:
    """
    Generates grounded responses with citations using RAG context.
    
    Responsibilities:
    - Synthesize information from RAG context
    - Generate conversational, accurate responses
    - Include proper source citations
    - Track confidence based on context relevance
    """
    
    def __init__(self, llm=None, components: Dict = None):
        self.components = components or {}
        self.llm = llm or create_llm_with_fallback(
            model="gemini-2.0-flash-exp",
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
        """
        Generate a grounded answer to the user's question.
        
        Args:
            user_question: The user's question
            product_type: Detected product type (if any)
            rag_context: Retrieved RAG context
            specifications: Additional specifications/constraints
            user_context: User session context
        
        Returns:
            ChatResponse as dict with answer, citations, confidence
        """
        logger.info(f"ChatAgent processing: {user_question[:50]}...")
        
        try:
            # Build context strings
            rag_context = rag_context or {}
            specifications = specifications or {}
            
            rag_context_str = json.dumps(rag_context, indent=2) if rag_context else "No context available"
            
            preferred_vendors = specifications.get("preferred_vendors", [])
            required_standards = specifications.get("required_standards", [])
            installed_series = specifications.get("installed_series", [])
            
            # Create prompt and chain
            prompt = ChatPromptTemplate.from_template(CHAT_AGENT_PROMPT)
            parser = JsonOutputParser()
            chain = prompt | self.llm | parser
            
            # Invoke
            result = chain.invoke({
                "question": user_question,
                "product_type": product_type or "general",
                "rag_context": rag_context_str,
                "preferred_vendors": ", ".join(preferred_vendors) if preferred_vendors else "Not specified",
                "required_standards": ", ".join(required_standards) if required_standards else "Not specified",
                "installed_series": ", ".join(installed_series) if installed_series else "Not specified"
            })
            
            logger.info(f"ChatAgent generated response with confidence: {result.get('confidence', 0)}")
            
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
# RESPONSE VALIDATOR AGENT
# ============================================================================

class ResponseValidatorAgent:
    """
    Validates responses for grounding, accuracy, and hallucination.
    
    Performs 5 validation checks:
    1. RELEVANCE - Does answer address the question?
    2. ACCURACY - Is information factually correct?
    3. GROUNDING - Is answer grounded in context?
    4. CITATIONS - Are sources properly cited?
    5. HALLUCINATION - Detect fabricated information
    """
    
    def __init__(self, llm=None, components: Dict = None):
        self.components = components or {}
        self.llm = llm or create_llm_with_fallback(
            model="gemini-2.0-flash-exp",
            temperature=0.1,  # Low temperature for consistent validation
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        logger.info("ResponseValidatorAgent initialized")
    
    def run(
        self,
        response: str,
        user_question: str,
        context: str = ""
    ) -> Dict[str, Any]:
        """
        Validate a generated response.
        
        Args:
            response: The generated response to validate
            user_question: Original user question
            context: Available RAG context
        
        Returns:
            ValidationResult as dict
        """
        logger.info("ResponseValidatorAgent validating response...")
        
        try:
            prompt = ChatPromptTemplate.from_template(VALIDATOR_PROMPT)
            parser = JsonOutputParser()
            chain = prompt | self.llm | parser
            
            result = chain.invoke({
                "question": user_question,
                "response": response,
                "context": context or "No context provided"
            })
            
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
                "is_valid": False,
                "overall_score": 0.0,
                "issues_found": [f"Validation error: {str(e)}"],
                "error": str(e)
            }


# ============================================================================
# SESSION MANAGER AGENT
# ============================================================================

class SessionManagerAgent:
    """
    Manages multi-turn conversation memory.
    
    Responsibilities:
    - Store Q&A pairs in session
    - Track conversation context
    - Enable follow-up questions
    - Prepare context for next turn
    """
    
    def __init__(self, llm=None, components: Dict = None):
        self.components = components or {}
        self.sessions: Dict[str, Dict] = {}  # In-memory session store
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
        """
        Manage session state.
        
        Args:
            session_id: Unique session identifier
            question: User question (for update)
            answer: Generated answer (for update)
            validation_score: Validation score
            citations: Source citations
            operation: "update", "get", or "clear"
        
        Returns:
            SessionInfo as dict
        """
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
        """
        Get formatted conversation history for context.
        
        Args:
            session_id: Session to get context from
            max_turns: Maximum conversation turns to include
        
        Returns:
            Formatted conversation history string
        """
        session = self.sessions.get(session_id, {})
        interactions = session.get("interactions", [])
        
        if not interactions:
            return ""
        
        # Get last N interactions
        recent = interactions[-max_turns:]
        
        context_parts = ["Previous conversation:"]
        for i, interaction in enumerate(recent, 1):
            context_parts.append(f"Q{i}: {interaction['question']}")
            context_parts.append(f"A{i}: {interaction['answer'][:200]}...")
        
        return "\n".join(context_parts)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "ChatAgent",
    "ResponseValidatorAgent", 
    "SessionManagerAgent",
    "ChatResponse",
    "ValidationResult",
    "SessionInfo"
]
