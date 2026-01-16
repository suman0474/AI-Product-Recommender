# agentic/index_rag_memory.py
# Conversation Memory for Index RAG
# Provides session-based memory for follow-up query resolution

import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class IndexRAGMemory:
    """
    Conversation memory for Index RAG.
    
    Features:
    - Store query history per session
    - Resolve follow-up queries ("What about Emerson?")
    - Cache query results
    """
    
    def __init__(self):
        """Initialize conversation memory."""
        # Session -> conversation history
        self._conversation_memory: Dict[str, List[Dict[str, Any]]] = {}
        
        # Query hash -> result cache
        self._result_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_max_size = 100
        
        # Session TTL (1 hour)
        self._session_ttl_seconds = 3600
        
        logger.info("[IndexRAGMemory] Initialized")
    
    def add_to_memory(
        self, 
        session_id: str, 
        query: str, 
        intent: str = None,
        product_type: str = None,
        vendors: List[str] = None,
        models: List[str] = None
    ):
        """
        Add a conversation turn to memory.
        
        Args:
            session_id: Session identifier
            query: User's query
            intent: Classified intent
            product_type: Detected product type
            vendors: Detected vendors
            models: Detected models
        """
        if session_id not in self._conversation_memory:
            self._conversation_memory[session_id] = []
        
        turn = {
            "query": query,
            "intent": intent,
            "product_type": product_type,
            "vendors": vendors or [],
            "models": models or [],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self._conversation_memory[session_id].append(turn)
        
        # Keep last 10 turns per session
        if len(self._conversation_memory[session_id]) > 10:
            self._conversation_memory[session_id] = self._conversation_memory[session_id][-10:]
        
        logger.debug(f"[IndexRAGMemory] Added turn for {session_id}: {len(self._conversation_memory[session_id])} turns")
    
    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a session."""
        return self._conversation_memory.get(session_id, [])
    
    def get_last_context(self, session_id: str) -> Dict[str, Any]:
        """Get the last conversation context for a session."""
        history = self.get_history(session_id)
        if history:
            return history[-1]
        return {}
    
    def is_follow_up_query(self, query: str) -> bool:
        """
        Check if query looks like a follow-up.
        
        Follow-up indicators:
        - "What about X?"
        - "How about X?"
        - "And X?"
        - Short queries (< 5 words)
        - Pronouns referring to previous context
        """
        query_lower = query.lower().strip()
        
        # Explicit follow-up phrases
        follow_up_indicators = [
            "what about", "how about", "and for", "same for",
            "what's the", "how's the", "tell me about",
            "compare with", "versus", "vs", "and",
            "what if", "how does", "show me"
        ]
        
        if any(indicator in query_lower for indicator in follow_up_indicators):
            return True
        
        # Very short queries are likely follow-ups
        if len(query.split()) <= 4:
            return True
        
        # Pronouns indicating reference to previous context
        pronouns = ["it", "them", "those", "these", "that", "this"]
        words = query_lower.split()
        if words and words[0] in pronouns:
            return True
        
        return False
    
    def resolve_follow_up(self, session_id: str, query: str) -> str:
        """
        Resolve follow-up queries using conversation history.
        
        Example:
        - Previous: "Honeywell pressure transmitters"
        - Current: "What about Emerson?"
        - Resolved: "Emerson pressure transmitters"
        
        Args:
            session_id: Session identifier
            query: User's current query
            
        Returns:
            Resolved query (unchanged if not a follow-up)
        """
        # Check if this is a follow-up
        if not self.is_follow_up_query(query):
            return query
        
        # Get previous context
        last_context = self.get_last_context(session_id)
        if not last_context:
            return query  # No history, can't resolve
        
        last_query = last_context.get('query', '')
        last_product_type = last_context.get('product_type', '')
        last_vendors = last_context.get('vendors', [])
        
        if not last_product_type and not last_vendors:
            return query  # No useful context
        
        # Use LLM to resolve the follow-up
        try:
            from llm_fallback import create_llm_with_fallback
            
            llm = create_llm_with_fallback(
                model="gemini-2.5-flash",
                temperature=0.1,
                max_tokens=200
            )
            
            prompt = ChatPromptTemplate.from_template("""Resolve this follow-up question using the conversation context.

PREVIOUS QUESTION: {last_query}
PREVIOUS CONTEXT:
- Product type: {product_type}
- Vendors discussed: {vendors}

CURRENT FOLLOW-UP: {current_query}

INSTRUCTIONS:
- If the user is asking about a different vendor/product in the same context, expand the query
- Example: Previous "Honeywell pressure transmitters", Current "What about Emerson?" -> "Emerson pressure transmitters"
- If it's not clearly a follow-up, return the original query unchanged
- Return ONLY the resolved query text, nothing else

RESOLVED QUERY:""")
            
            chain = prompt | llm | StrOutputParser()
            
            resolved = chain.invoke({
                "last_query": last_query,
                "product_type": last_product_type or "not specified",
                "vendors": ", ".join(last_vendors) if last_vendors else "none",
                "current_query": query
            })
            
            resolved_query = resolved.strip()
            
            if resolved_query and resolved_query != query:
                logger.info(f"[IndexRAGMemory] Resolved follow-up: '{query}' -> '{resolved_query}'")
                return resolved_query
            
            return query
            
        except Exception as e:
            logger.warning(f"[IndexRAGMemory] Follow-up resolution failed: {e}")
            return query
    
    def clear_session(self, session_id: str):
        """Clear conversation memory for a session."""
        if session_id in self._conversation_memory:
            del self._conversation_memory[session_id]
            logger.info(f"[IndexRAGMemory] Cleared session: {session_id}")
    
    def cleanup_expired_sessions(self) -> int:
        """
        Remove expired sessions.
        
        Returns:
            Number of sessions removed
        """
        now = time.time()
        expired = []
        
        for session_id, history in self._conversation_memory.items():
            if history:
                # Check last turn timestamp
                last_turn = history[-1]
                last_time_str = last_turn.get('timestamp', '')
                if last_time_str:
                    try:
                        last_dt = datetime.fromisoformat(last_time_str)
                        age_seconds = (datetime.utcnow() - last_dt).total_seconds()
                        if age_seconds > self._session_ttl_seconds:
                            expired.append(session_id)
                    except Exception:
                        pass
        
        for session_id in expired:
            del self._conversation_memory[session_id]
        
        if expired:
            logger.info(f"[IndexRAGMemory] Cleaned up {len(expired)} expired sessions")
        
        return len(expired)
    
    def cache_result(self, query_hash: str, result: Dict[str, Any]):
        """Cache a query result."""
        if len(self._result_cache) >= self._cache_max_size:
            # Remove oldest entry
            oldest = next(iter(self._result_cache))
            del self._result_cache[oldest]
        
        self._result_cache[query_hash] = {
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_cached_result(self, query_hash: str) -> Optional[Dict[str, Any]]:
        """Get a cached result if available."""
        cached = self._result_cache.get(query_hash)
        if cached:
            return cached.get("result")
        return None


# Global instance
index_rag_memory = IndexRAGMemory()


def get_index_rag_memory() -> IndexRAGMemory:
    """Get the global IndexRAGMemory instance."""
    return index_rag_memory


def resolve_follow_up_query(session_id: str, query: str) -> str:
    """Convenience function for follow-up resolution."""
    return index_rag_memory.resolve_follow_up(session_id, query)


def add_to_conversation_memory(
    session_id: str,
    query: str,
    intent: str = None,
    product_type: str = None,
    vendors: List[str] = None,
    models: List[str] = None
):
    """Convenience function for adding to memory."""
    index_rag_memory.add_to_memory(
        session_id=session_id,
        query=query,
        intent=intent,
        product_type=product_type,
        vendors=vendors,
        models=models
    )


def clear_conversation_memory(session_id: str):
    """Convenience function for clearing memory."""
    index_rag_memory.clear_session(session_id)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'IndexRAGMemory',
    'index_rag_memory',
    'get_index_rag_memory',
    'resolve_follow_up_query',
    'add_to_conversation_memory',
    'clear_conversation_memory'
]
