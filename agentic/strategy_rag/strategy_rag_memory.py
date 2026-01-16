# agentic/strategy_rag/strategy_rag_memory.py
# Conversation Memory for Strategy RAG
# Provides session-based memory for follow-up query resolution

import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class StrategyRAGMemory:
    """
    Conversation memory for Strategy RAG.
    
    Features:
    - Store query history per session
    - Resolve follow-up queries ("What about Honeywell?")
    - Session-based (not persistent)
    - Automatic cleanup of expired sessions
    """
    
    SESSION_TIMEOUT_SECONDS = 3600  # 1 hour
    
    def __init__(self):
        """Initialize conversation memory."""
        self.sessions: Dict[str, Dict] = {}
        logger.info("StrategyRAGMemory initialized")
    
    def add_to_memory(
        self, 
        session_id: str, 
        product_type: str,
        preferred_vendors: List[str] = None,
        strategy_notes: str = None
    ):
        """
        Add a strategy query to memory.
        
        Args:
            session_id: Session identifier
            product_type: Product type searched
            preferred_vendors: Vendors found
            strategy_notes: Strategy notes returned
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "created_at": time.time(),
                "last_updated": time.time(),
                "history": [],
                "context": {
                    "product_types_searched": [],
                    "vendors_mentioned": []
                }
            }
        
        session = self.sessions[session_id]
        session["last_updated"] = time.time()
        
        # Add to history
        session["history"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "product_type": product_type,
            "preferred_vendors": preferred_vendors or [],
            "strategy_notes": strategy_notes or ""
        })
        
        # Update context
        if product_type and product_type not in session["context"]["product_types_searched"]:
            session["context"]["product_types_searched"].append(product_type)
        
        if preferred_vendors:
            for vendor in preferred_vendors:
                if vendor not in session["context"]["vendors_mentioned"]:
                    session["context"]["vendors_mentioned"].append(vendor)
        
        logger.debug(f"Added to strategy memory: session={session_id}")
    
    def get_history(self, session_id: str) -> List[Dict]:
        """Get query history for a session."""
        session = self.sessions.get(session_id, {})
        return session.get("history", [])
    
    def get_last_context(self, session_id: str) -> Dict[str, Any]:
        """Get the last query context for a session."""
        session = self.sessions.get(session_id, {})
        history = session.get("history", [])
        
        if not history:
            return {}
        
        last = history[-1]
        return {
            "last_product_type": last.get("product_type", ""),
            "last_vendors": last.get("preferred_vendors", []),
            "product_types_searched": session.get("context", {}).get("product_types_searched", []),
            "vendors_mentioned": session.get("context", {}).get("vendors_mentioned", [])
        }
    
    def is_follow_up_query(self, query: str) -> bool:
        """Check if query looks like a follow-up."""
        query_lower = query.lower().strip()
        
        follow_up_starts = [
            "what about", "how about", "and ", "also ",
            "compare with", "versus", "vs",
            "show me", "what's the strategy for"
        ]
        
        for pattern in follow_up_starts:
            if query_lower.startswith(pattern):
                return True
        
        # Short queries
        words = query_lower.split()
        if len(words) < 4:
            return True
        
        return False
    
    def resolve_follow_up(self, session_id: str, query: str) -> Dict[str, Any]:
        """
        Resolve follow-up queries using session history.
        
        Returns:
            Dict with resolved product_type and any context
        """
        if not self.is_follow_up_query(query):
            return {"product_type": query, "is_follow_up": False}
        
        context = self.get_last_context(session_id)
        if not context:
            return {"product_type": query, "is_follow_up": False}
        
        query_lower = query.lower().strip()
        
        # Handle "What about [vendor]?" - return same product type, filter for vendor
        if query_lower.startswith("what about ") or query_lower.startswith("how about "):
            prefix = "what about " if query_lower.startswith("what about ") else "how about "
            subject = query[len(prefix):].strip().rstrip("?")
            
            # Check if it's asking about a vendor
            last_product_type = context.get("last_product_type", "")
            if last_product_type:
                return {
                    "product_type": last_product_type,
                    "vendor_filter": subject,
                    "is_follow_up": True,
                    "original_query": query
                }
        
        # Return the query as product type with follow-up flag
        return {
            "product_type": query,
            "is_follow_up": True,
            "context": context
        }
    
    def clear_session(self, session_id: str):
        """Clear memory for a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Cleared strategy session: {session_id}")
    
    def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions."""
        current_time = time.time()
        expired = []
        
        for session_id, session in self.sessions.items():
            last_updated = session.get("last_updated", 0)
            if current_time - last_updated > self.SESSION_TIMEOUT_SECONDS:
                expired.append(session_id)
        
        for session_id in expired:
            del self.sessions[session_id]
        
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired strategy sessions")
        
        return len(expired)


# Global instance
strategy_rag_memory = StrategyRAGMemory()


def get_strategy_rag_memory() -> StrategyRAGMemory:
    """Get the global StrategyRAGMemory instance."""
    return strategy_rag_memory


def add_to_strategy_memory(
    session_id: str,
    product_type: str,
    preferred_vendors: List[str] = None,
    strategy_notes: str = None
):
    """Convenience function for adding to memory."""
    strategy_rag_memory.add_to_memory(
        session_id=session_id,
        product_type=product_type,
        preferred_vendors=preferred_vendors,
        strategy_notes=strategy_notes
    )


def clear_strategy_memory(session_id: str):
    """Convenience function for clearing memory."""
    strategy_rag_memory.clear_session(session_id)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'StrategyRAGMemory',
    'strategy_rag_memory',
    'get_strategy_rag_memory',
    'add_to_strategy_memory',
    'clear_strategy_memory'
]
