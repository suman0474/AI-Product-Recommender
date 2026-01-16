# agentic/standards_chat_agent.py
# Specialized Chat Agent for Standards Documentation Q&A

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

import os
from dotenv import load_dotenv

# Import existing infrastructure
from agentic.vector_store import get_vector_store
from llm_fallback import create_llm_with_fallback

load_dotenv()
logger = logging.getLogger(__name__)


# ============================================================================
# PROMPT TEMPLATE
# ============================================================================

STANDARDS_CHAT_PROMPT = """
You are a Standards Documentation Expert for industrial instrumentation.

Your role is to answer technical questions based ONLY on the provided standards documents.
You have access to instrumentation standards covering:
- Safety and protection (SIL ratings, ATEX zones, certifications)
- Pressure measurement
- Temperature measurement
- Flow measurement
- Level measurement
- Control systems
- Valves and actuators
- Calibration and maintenance
- Communication and signals
- Condition monitoring
- Analytical instrumentation
- Accessories and calibration

USER QUESTION:
{question}

RETRIEVED STANDARDS DOCUMENTS:
{retrieved_context}

INSTRUCTIONS:
1. Answer ONLY based on the provided standards documents above
2. If the information is not in the documents, clearly state: "I don't have specific information about that in the available standards documents."
3. Cite sources using [Source: filename] format inline in your answer
4. Reference specific standard codes (IEC, ISO, API, ANSI, ISA, etc.) when they appear in the documents
5. Be precise and technical - this is for engineering professionals
6. Include relevant details like specifications, ratings, requirements, or procedures
7. NO HALLUCINATION - Do not invent or assume information not present in the documents

Return ONLY valid JSON in this exact format:
{{
    "answer": "<your detailed answer with inline [Source: filename] citations>",
    "citations": [
        {{
            "source": "<document filename>",
            "content": "<relevant quote from document>",
            "relevance": <0.0-1.0 relevance score from retrieval>
        }}
    ],
    "confidence": <0.0-1.0 confidence score>,
    "sources_used": ["<filename1.docx>", "<filename2.docx>"]
}}

IMPORTANT: Your response must be valid JSON only, no additional text.
"""


# ============================================================================
# STANDARDS CHAT AGENT
# ============================================================================

class StandardsChatAgent:
    """
    Specialized agent for answering questions using standards documents from Pinecone.

    This agent retrieves relevant standards documents from the vector store,
    constructs context, and generates grounded answers using Google Generative AI.
    Includes production-ready retry logic and error handling.
    """

    def __init__(self, llm=None, temperature: float = 0.1):
        """
        Initialize the Standards Chat Agent.

        Args:
            llm: Language model instance (if None, creates default Gemini)
            temperature: Temperature for generation (default 0.1 for factual responses)
        """
        # Initialize LLM
        if llm is None:
            self.llm = create_llm_with_fallback(
                model="gemini-2.5-flash",
                temperature=temperature,
                max_tokens=2000
            )
        else:
            self.llm = llm

        # Get vector store
        self.vector_store = get_vector_store()

        # Create prompt template
        self.prompt = ChatPromptTemplate.from_template(STANDARDS_CHAT_PROMPT)

        # Create output parser
        self.parser = JsonOutputParser()

        # Create chain
        self.chain = self.prompt | self.llm | self.parser

        logger.info("StandardsChatAgent initialized with Google Generative AI")

    def retrieve_documents(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Retrieve relevant standards documents from vector store.

        Args:
            question: User's question
            top_k: Number of top documents to retrieve

        Returns:
            Dictionary with retrieval results
        """
        try:
            # Search vector store
            search_results = self.vector_store.search(
                collection_type="standards",
                query=question,
                top_k=top_k
            )

            logger.info(f"Retrieved {search_results.get('result_count', 0)} documents for question")

            return search_results

        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return {
                "results": [],
                "result_count": 0,
                "error": str(e)
            }

    def build_context(self, search_results: Dict[str, Any]) -> str:
        """
        Build formatted context from search results.

        Args:
            search_results: Results from vector store search

        Returns:
            Formatted context string
        """
        results = search_results.get('results', [])

        if not results:
            return "No relevant standards documents found."

        context_parts = []

        for i, result in enumerate(results, 1):
            # Extract metadata
            metadata = result.get('metadata', {})
            source = metadata.get('filename', 'unknown')
            standard_type = metadata.get('standard_type', 'general')
            standards_refs = metadata.get('standards_references', [])

            # Get content and relevance
            content = result.get('content', '')
            relevance = result.get('relevance_score', 0.0)

            # Build document section
            doc_section = f"[Document {i}: {source}]\n"
            doc_section += f"Type: {standard_type}\n"
            if standards_refs:
                doc_section += f"Referenced Standards: {', '.join(standards_refs[:5])}\n"
            doc_section += f"Relevance: {relevance:.3f}\n"
            doc_section += f"\nContent:\n{content}"

            context_parts.append(doc_section)

        return "\n\n" + "="*80 + "\n\n".join(context_parts)

    def extract_citations(self, search_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract citation information from search results.

        Args:
            search_results: Results from vector store search

        Returns:
            List of citations
        """
        citations = []

        for result in search_results.get('results', []):
            metadata = result.get('metadata', {})
            citation = {
                'source': metadata.get('filename', 'unknown'),
                'content': result.get('content', '')[:200] + "...",  # First 200 chars
                'relevance': result.get('relevance_score', 0.0),
                'standard_type': metadata.get('standard_type', 'general'),
                'standards_references': metadata.get('standards_references', [])
            }
            citations.append(citation)

        return citations

    def run(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Answer question using standards documents.

        Args:
            question: User's question
            top_k: Number of documents to retrieve

        Returns:
            Dictionary with answer, citations, confidence, and sources
        """
        try:
            logger.info(f"Processing question: {question[:100]}...")

            # Step 1: Retrieve documents
            search_results = self.retrieve_documents(question, top_k)

            if search_results.get('result_count', 0) == 0:
                return {
                    'answer': "I don't have any relevant standards documents to answer this question.",
                    'citations': [],
                    'confidence': 0.0,
                    'sources_used': [],
                    'error': 'No documents found'
                }

            # Step 2: Build context
            context = self.build_context(search_results)

            # Step 3: Generate answer
            logger.info("Generating answer with LLM...")

            response = self.chain.invoke({
                'question': question,
                'retrieved_context': context
            })

            # Step 4: Add metadata
            result = {
                'answer': response.get('answer', ''),
                'citations': response.get('citations', []),
                'confidence': response.get('confidence', 0.0),
                'sources_used': response.get('sources_used', [])
            }

            # Calculate average retrieval relevance as fallback confidence
            if 'results' in search_results and search_results['results']:
                avg_relevance = sum(r.get('relevance_score', 0) for r in search_results['results']) / len(search_results['results'])
                # Use the higher of LLM confidence or retrieval relevance
                result['confidence'] = max(result['confidence'], avg_relevance)

            logger.info(f"Answer generated with confidence: {result['confidence']:.2f}")

            return result

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            return {
                'answer': "Error: Invalid response format from LLM",
                'citations': [],
                'confidence': 0.0,
                'sources_used': [],
                'error': f'JSON parsing error: {str(e)}'
            }

        except Exception as e:
            logger.error(f"Error in StandardsChatAgent.run: {e}", exc_info=True)
            return {
                'answer': f"An error occurred while processing your question: {str(e)}",
                'citations': [],
                'confidence': 0.0,
                'sources_used': [],
                'error': str(e)
            }


# ============================================================================
# SINGLETON INSTANCE (Performance Optimization)
# ============================================================================

_standards_chat_agent_instance = None
_standards_chat_agent_lock = None

def _get_agent_lock():
    """Get thread lock for singleton (lazy initialization)."""
    global _standards_chat_agent_lock
    if _standards_chat_agent_lock is None:
        import threading
        _standards_chat_agent_lock = threading.Lock()
    return _standards_chat_agent_lock


def get_standards_chat_agent(temperature: float = 0.1) -> StandardsChatAgent:
    """
    Get or create the singleton StandardsChatAgent instance.
    
    This avoids expensive re-initialization of LLM and vector store connections
    on every call, significantly improving performance for batch operations.
    
    Args:
        temperature: Temperature for generation (only used on first call)
        
    Returns:
        Cached StandardsChatAgent instance
    """
    global _standards_chat_agent_instance
    
    if _standards_chat_agent_instance is None:
        with _get_agent_lock():
            # Double-check after acquiring lock
            if _standards_chat_agent_instance is None:
                logger.info("[StandardsChatAgent] Creating singleton instance (first call)")
                _standards_chat_agent_instance = StandardsChatAgent(temperature=temperature)
    
    return _standards_chat_agent_instance


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_standards_chat_agent(temperature: float = 0.1) -> StandardsChatAgent:
    """
    Create a StandardsChatAgent instance with default settings.
    
    NOTE: For better performance, use get_standards_chat_agent() instead,
    which returns a cached singleton instance.

    Args:
        temperature: Temperature for generation

    Returns:
        StandardsChatAgent instance (uses singleton for efficiency)
    """
    # Use singleton for better performance
    return get_standards_chat_agent(temperature=temperature)


def ask_standards_question(question: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Quick function to ask a question to the standards system.

    Args:
        question: Question to ask
        top_k: Number of documents to retrieve

    Returns:
        Answer dictionary
    """
    agent = create_standards_chat_agent()
    return agent.run(question, top_k)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    # Test the agent
    logger.info("="*80)
    logger.info("TESTING STANDARDS CHAT AGENT")
    logger.info("="*80)

    test_questions = [
        "What are the SIL2 requirements for pressure transmitters?",
        "What calibration standards apply to temperature sensors?",
        "What are the ATEX requirements for flow measurement devices?"
    ]

    agent = create_standards_chat_agent()

    for i, question in enumerate(test_questions, 1):
        logger.info(f"\n[Test {i}] Question: {question}")
        logger.info("-"*80)

        result = agent.run(question, top_k=3)

        logger.info(f"\nAnswer: {result['answer'][:300]}...")
        logger.info(f"\nConfidence: {result['confidence']:.2f}")
        logger.info(f"Sources: {', '.join(result['sources_used'])}")

        if result.get('citations'):
            logger.info(f"\nCitations:")
            for j, cite in enumerate(result['citations'][:2], 1):
                logger.info(f"  [{j}] {cite['source']} (relevance: {cite.get('relevance', 0):.2f})")

        logger.info("="*80)
