# agentic/standards_rag/standards_rag_memory.py
# Conversation Memory for Standards RAG
# Provides session-based memory for follow-up query resolution

import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class StandardsRAGMemory:
    """
    Conversation memory for Standards RAG.
    
    Features:
    - Store Q&A history per session
    - Resolve follow-up queries ("What about SIL-2?")
    - Session-based (per user requirement, not persistent)
    - Automatic cleanup of expired sessions
    """
    
    SESSION_TIMEOUT_SECONDS = 3600  # 1 hour
    
    def __init__(self):
        """Initialize conversation memory."""
        self.sessions: Dict[str, Dict] = {}
        logger.info("StandardsRAGMemory initialized")
    
    def add_to_memory(
        self, 
        session_id: str, 
        question: str, 
        answer: str = None,
        citations: List[Dict] = None,
        key_terms: List[str] = None
    ):
        """
        Add a conversation turn to memory.
        
        Args:
            session_id: Session identifier
            question: User's question
            answer: Generated answer
            citations: Source citations
            key_terms: Extracted key terms
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "created_at": time.time(),
                "last_updated": time.time(),
                "history": [],
                "context": {
                    "standards_mentioned": [],
                    "topics_discussed": []
                }
            }
        
        session = self.sessions[session_id]
        session["last_updated"] = time.time()
        
        # Add to history
        session["history"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "question": question,
            "answer": answer or "",
            "citations": citations or [],
            "key_terms": key_terms or []
        })
        
        # Update context with standards and topics
        if key_terms:
            for term in key_terms:
                term_upper = term.upper()
                # Extract standard codes
                if any(code in term_upper for code in ["IEC", "ISO", "API", "ASME", "SIL", "ATEX"]):
                    if term not in session["context"]["standards_mentioned"]:
                        session["context"]["standards_mentioned"].append(term)
                else:
                    if term not in session["context"]["topics_discussed"]:
                        session["context"]["topics_discussed"].append(term)
        
        logger.debug(f"Added to memory: session={session_id}, history_len={len(session['history'])}")
    
    def get_history(self, session_id: str) -> List[Dict]:
        """Get conversation history for a session."""
        session = self.sessions.get(session_id, {})
        return session.get("history", [])
    
    def get_last_context(self, session_id: str) -> Dict[str, Any]:
        """Get the last conversation context for a session."""
        session = self.sessions.get(session_id, {})
        history = session.get("history", [])
        
        if not history:
            return {}
        
        last = history[-1]
        return {
            "last_question": last.get("question", ""),
            "last_answer": last.get("answer", ""),
            "key_terms": last.get("key_terms", []),
            "standards_mentioned": session.get("context", {}).get("standards_mentioned", []),
            "topics_discussed": session.get("context", {}).get("topics_discussed", [])
        }
    
    def is_follow_up_query(self, question: str) -> bool:
        """
        Check if query looks like a follow-up.
        
        Follow-up indicators:
        - "What about X?"
        - "How about X?"
        - "And X?"
        - "What's the difference..."
        - Short queries (< 5 words)
        - Pronouns referring to previous context
        """
        question_lower = question.lower().strip()
        
        # Follow-up patterns
        follow_up_starts = [
            "what about", "how about", "and ", "also ", 
            "what's the difference", "what is the difference",
            "compare", "versus", "vs", "or ",
            "can you explain more", "tell me more",
            "what if", "how does that"
        ]
        
        for pattern in follow_up_starts:
            if question_lower.startswith(pattern):
                return True
        
        # Short queries often are follow-ups
        words = question_lower.split()
        if len(words) < 5:
            return True
        
        # Pronoun references
        pronouns = ["it", "they", "this", "that", "these", "those", "the standard", "the requirement"]
        for pronoun in pronouns:
            if pronoun in question_lower:
                return True
        
        return False
    
    def resolve_follow_up(self, session_id: str, question: str) -> str:
        """
        Resolve follow-up queries using conversation history.
        
        Example:
        - Previous: "What is SIL-2 certification?"
        - Current: "What about SIL-3?"
        - Resolved: "What is SIL-3 certification?"
        
        Args:
            session_id: Session identifier
            question: User's current question
            
        Returns:
            Resolved query (unchanged if not a follow-up)
        """
        if not self.is_follow_up_query(question):
            return question
        
        context = self.get_last_context(session_id)
        if not context:
            return question
        
        last_question = context.get("last_question", "")
        standards = context.get("standards_mentioned", [])
        topics = context.get("topics_discussed", [])
        
        question_lower = question.lower().strip()
        
        # Handle "What about X?" pattern
        if question_lower.startswith("what about ") or question_lower.startswith("how about "):
            prefix = "what about " if question_lower.startswith("what about ") else "how about "
            new_subject = question[len(prefix):].strip().rstrip("?")
            
            # If last question exists, substitute the subject
            if last_question:
                # Try to find and replace standards or topics
                resolved = last_question
                for standard in standards:
                    if standard.lower() in last_question.lower():
                        resolved = last_question.replace(standard, new_subject)
                        break
                
                if resolved != last_question:
                    logger.info(f"Resolved follow-up: '{question}' -> '{resolved}'")
                    return resolved
            
            # Fallback: Create a question about the new subject
            return f"What is {new_subject}?"
        
        # Handle "And X?" pattern
        if question_lower.startswith("and "):
            new_subject = question[4:].strip().rstrip("?")
            if last_question:
                return f"{last_question.rstrip('?')} and {new_subject}?"
        
        # Handle comparison patterns
        if any(p in question_lower for p in ["versus", "vs", "compare"]):
            if standards:
                # Add context from previous standards
                return f"{question} (in context of {', '.join(standards[:2])})"
        
        # If we have context, append it for clarity
        if standards or topics:
            context_hint = standards[:2] if standards else topics[:2]
            return f"{question} (regarding {', '.join(context_hint)})"
        
        return question
    
    def clear_session(self, session_id: str):
        """Clear conversation memory for a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Cleared session: {session_id}")
    
    def cleanup_expired_sessions(self) -> int:
        """
        Remove expired sessions.
        
        Returns:
            Number of sessions removed
        """
        current_time = time.time()
        expired = []
        
        for session_id, session in self.sessions.items():
            last_updated = session.get("last_updated", 0)
            if current_time - last_updated > self.SESSION_TIMEOUT_SECONDS:
                expired.append(session_id)
        
        for session_id in expired:
            del self.sessions[session_id]
        
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")
        
        return len(expired)


# Global instance
standards_rag_memory = StandardsRAGMemory()


def get_standards_rag_memory() -> StandardsRAGMemory:
    """Get the global StandardsRAGMemory instance."""
    return standards_rag_memory


def resolve_standards_follow_up(session_id: str, question: str) -> str:
    """Convenience function for follow-up resolution."""
    return standards_rag_memory.resolve_follow_up(session_id, question)


def add_to_standards_memory(
    session_id: str,
    question: str,
    answer: str = None,
    citations: List[Dict] = None,
    key_terms: List[str] = None
):
    """Convenience function for adding to memory."""
    standards_rag_memory.add_to_memory(
        session_id=session_id,
        question=question,
        answer=answer,
        citations=citations,
        key_terms=key_terms
    )


def clear_standards_memory(session_id: str):
    """Convenience function for clearing memory."""
    standards_rag_memory.clear_session(session_id)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'StandardsRAGMemory',
    'standards_rag_memory',
    'get_standards_rag_memory',
    'resolve_standards_follow_up',
    'add_to_standards_memory',
    'clear_standards_memory'
]
