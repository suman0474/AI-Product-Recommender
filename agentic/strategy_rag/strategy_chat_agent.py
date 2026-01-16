# agentic/strategy_chat_agent.py
# Specialized Chat Agent for Strategy Documentation Q&A
# TRUE RAG Implementation for Procurement Strategy Knowledge Base

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

STRATEGY_CHAT_PROMPT = """
You are a Procurement Strategy Expert for industrial instrumentation.

Your role is to answer questions about procurement strategy based ONLY on the provided strategy documents.
You have access to strategy documents covering:
- Preferred vendors and strategic partnerships
- Forbidden vendors and reasons for exclusion
- Procurement policies and guidelines
- Vendor evaluation criteria
- Cost optimization strategies
- Lifecycle cost considerations
- Sustainability requirements
- Single-source vs multi-source policies
- Regional vendor preferences
- Contract terms and conditions
- Lead time requirements
- Quality certification requirements

USER QUESTION:
{question}

PRODUCT CONTEXT:
- Product Type: {product_type}
- Requirements: {requirements}

RETRIEVED STRATEGY DOCUMENTS:
{retrieved_context}

INSTRUCTIONS:
1. Answer ONLY based on the provided strategy documents above
2. If the information is not in the documents, clearly state: "I don't have specific strategy information for that in the available documents."
3. Cite sources using [Source: filename] format inline in your answer
4. Identify specific vendor names, policies, or guidelines mentioned in the documents
5. Be precise and actionable - this is for procurement professionals
6. Include relevant details like vendor ratings, contract terms, or policy requirements
7. NO HALLUCINATION - Do not invent or assume information not present in the documents

Return ONLY valid JSON in this exact format:
{{
    "answer": "<your detailed answer with inline [Source: filename] citations>",
    "preferred_vendors": ["<list of preferred vendors mentioned>"],
    "forbidden_vendors": ["<list of forbidden/excluded vendors mentioned>"],
    "neutral_vendors": ["<list of neutral/acceptable vendors mentioned>"],
    "procurement_priorities": {{
        "<vendor_name>": <priority_score 1-10 based on document content>
    }},
    "strategy_notes": "<key strategy insights from the documents>",
    "citations": [
        {{
            "source": "<document filename>",
            "content": "<relevant quote from document>",
            "relevance": <0.0-1.0 relevance score from retrieval>
        }}
    ],
    "confidence": <0.0-1.0 confidence score>,
    "sources_used": ["<filename1>", "<filename2>"]
}}

IMPORTANT: Your response must be valid JSON only, no additional text.
"""


# ============================================================================
# STRATEGY CHAT AGENT
# ============================================================================

class StrategyChatAgent:
    """
    Specialized agent for answering questions using strategy documents from Pinecone.

    This agent retrieves relevant strategy documents from the vector store,
    constructs context, and generates grounded answers using Google Generative AI.
    Includes production-ready retry logic and error handling.
    
    This is a TRUE RAG implementation - answers are grounded in retrieved documents.
    """

    def __init__(self, llm=None, temperature: float = 0.1):
        """
        Initialize the Strategy Chat Agent.

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
        self.prompt = ChatPromptTemplate.from_template(STRATEGY_CHAT_PROMPT)

        # Create output parser
        self.parser = JsonOutputParser()

        # Create chain
        self.chain = self.prompt | self.llm | self.parser

        logger.info("StrategyChatAgent initialized with Google Generative AI (TRUE RAG)")

    def retrieve_documents(self, question: str, product_type: str = "", top_k: int = 7) -> Dict[str, Any]:
        """
        Retrieve relevant strategy documents from vector store.

        Args:
            question: User's question or product type query
            product_type: Product type for context
            top_k: Number of top documents to retrieve

        Returns:
            Dictionary with retrieval results
        """
        try:
            # Build search query
            search_query = question
            if product_type:
                search_query = f"{product_type} {question}"
            
            # Search vector store - using "strategy" collection type
            search_results = self.vector_store.search(
                collection_type="strategy",  # Maps to "strategy_documents" namespace
                query=search_query,
                top_k=top_k
            )

            logger.info(f"Retrieved {search_results.get('result_count', 0)} strategy documents")

            return search_results

        except Exception as e:
            logger.error(f"Error retrieving strategy documents: {e}")
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
            return "No relevant strategy documents found."

        context_parts = []

        for i, result in enumerate(results, 1):
            # Extract metadata
            metadata = result.get('metadata', {})
            source = metadata.get('filename', 'unknown')
            doc_type = metadata.get('document_type', 'strategy')
            vendor_refs = metadata.get('vendor_references', [])
            policy_refs = metadata.get('policy_references', [])

            # Get content and relevance
            content = result.get('content', '')
            relevance = result.get('relevance_score', 0.0)

            # Build document section
            doc_section = f"[Document {i}: {source}]\n"
            doc_section += f"Type: {doc_type}\n"
            if vendor_refs:
                doc_section += f"Vendors: {', '.join(vendor_refs[:5])}\n"
            if policy_refs:
                doc_section += f"Policies: {', '.join(policy_refs[:3])}\n"
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
                'content': result.get('content', '')[:200] + "...",
                'relevance': result.get('relevance_score', 0.0),
                'document_type': metadata.get('document_type', 'strategy'),
                'vendor_references': metadata.get('vendor_references', []),
                'policy_references': metadata.get('policy_references', [])
            }
            citations.append(citation)

        return citations

    def run(
        self, 
        question: str = "",
        product_type: str = "",
        requirements: Dict[str, Any] = None,
        top_k: int = 7
    ) -> Dict[str, Any]:
        """
        Answer question using strategy documents.

        Args:
            question: User's question (optional if product_type provided)
            product_type: Product type for strategy lookup
            requirements: Additional requirements context
            top_k: Number of documents to retrieve

        Returns:
            Dictionary with answer, strategy data, citations, and sources
        """
        try:
            # Build question from product_type if not provided
            if not question and product_type:
                question = f"What is the procurement strategy for {product_type}?"
            
            logger.info(f"Processing strategy question: {question[:100]}...")

            # Step 1: Retrieve documents
            search_results = self.retrieve_documents(
                question=question,
                product_type=product_type,
                top_k=top_k
            )

            if search_results.get('result_count', 0) == 0:
                logger.warning("No strategy documents found in vector store")
                return {
                    'success': False,
                    'answer': "I don't have any strategy documents to answer this question. The strategy knowledge base may be empty.",
                    'preferred_vendors': [],
                    'forbidden_vendors': [],
                    'neutral_vendors': [],
                    'procurement_priorities': {},
                    'strategy_notes': 'No strategy documents available',
                    'citations': [],
                    'confidence': 0.0,
                    'sources_used': [],
                    'error': 'No documents found'
                }

            # Step 2: Build context
            context = self.build_context(search_results)

            # Step 3: Generate answer
            logger.info("Generating strategy answer with LLM...")

            response = self.chain.invoke({
                'question': question,
                'product_type': product_type or 'general',
                'requirements': json.dumps(requirements or {}, indent=2),
                'retrieved_context': context
            })

            # Step 4: Build result
            result = {
                'success': True,
                'answer': response.get('answer', ''),
                'preferred_vendors': response.get('preferred_vendors', []),
                'forbidden_vendors': response.get('forbidden_vendors', []),
                'neutral_vendors': response.get('neutral_vendors', []),
                'procurement_priorities': response.get('procurement_priorities', {}),
                'strategy_notes': response.get('strategy_notes', ''),
                'citations': response.get('citations', []),
                'confidence': response.get('confidence', 0.0),
                'sources_used': response.get('sources_used', [])
            }

            # Calculate average retrieval relevance as fallback confidence
            if 'results' in search_results and search_results['results']:
                avg_relevance = sum(r.get('relevance_score', 0) for r in search_results['results']) / len(search_results['results'])
                result['confidence'] = max(result['confidence'], avg_relevance)

            logger.info(f"Strategy answer generated with confidence: {result['confidence']:.2f}")
            logger.info(f"Preferred vendors: {result['preferred_vendors']}")
            logger.info(f"Forbidden vendors: {result['forbidden_vendors']}")

            return result

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            return {
                'success': False,
                'answer': "Error: Invalid response format from LLM",
                'preferred_vendors': [],
                'forbidden_vendors': [],
                'neutral_vendors': [],
                'procurement_priorities': {},
                'strategy_notes': '',
                'citations': [],
                'confidence': 0.0,
                'sources_used': [],
                'error': f'JSON parsing error: {str(e)}'
            }

        except Exception as e:
            logger.error(f"Error in StrategyChatAgent.run: {e}", exc_info=True)
            return {
                'success': False,
                'answer': f"An error occurred while processing your question: {str(e)}",
                'preferred_vendors': [],
                'forbidden_vendors': [],
                'neutral_vendors': [],
                'procurement_priorities': {},
                'strategy_notes': '',
                'citations': [],
                'confidence': 0.0,
                'sources_used': [],
                'error': str(e)
            }


# ============================================================================
# SINGLETON INSTANCE (Performance Optimization)
# ============================================================================

_strategy_chat_agent_instance = None
_strategy_chat_agent_lock = None

def _get_strategy_agent_lock():
    """Get thread lock for singleton (lazy initialization)."""
    global _strategy_chat_agent_lock
    if _strategy_chat_agent_lock is None:
        import threading
        _strategy_chat_agent_lock = threading.Lock()
    return _strategy_chat_agent_lock


def get_strategy_chat_agent(temperature: float = 0.1) -> StrategyChatAgent:
    """
    Get or create the singleton StrategyChatAgent instance.
    
    This avoids expensive re-initialization of LLM and vector store connections
    on every call, significantly improving performance for batch operations.
    
    Args:
        temperature: Temperature for generation (only used on first call)
        
    Returns:
        Cached StrategyChatAgent instance
    """
    global _strategy_chat_agent_instance
    
    if _strategy_chat_agent_instance is None:
        with _get_strategy_agent_lock():
            # Double-check after acquiring lock
            if _strategy_chat_agent_instance is None:
                logger.info("[StrategyChatAgent] Creating singleton instance (first call)")
                _strategy_chat_agent_instance = StrategyChatAgent(temperature=temperature)
    
    return _strategy_chat_agent_instance


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_strategy_chat_agent(temperature: float = 0.1) -> StrategyChatAgent:
    """
    Create a StrategyChatAgent instance with default settings.
    
    NOTE: For better performance, use get_strategy_chat_agent() instead,
    which returns a cached singleton instance.

    Args:
        temperature: Temperature for generation

    Returns:
        StrategyChatAgent instance (uses singleton for efficiency)
    """
    # Use singleton for better performance
    return get_strategy_chat_agent(temperature=temperature)


def ask_strategy_question(
    question: str = "",
    product_type: str = "",
    requirements: Dict[str, Any] = None,
    top_k: int = 7
) -> Dict[str, Any]:
    """
    Quick function to ask a question to the strategy system.

    Args:
        question: Question to ask
        product_type: Product type context
        requirements: Additional requirements
        top_k: Number of documents to retrieve

    Returns:
        Strategy answer dictionary
    """
    agent = create_strategy_chat_agent()
    return agent.run(
        question=question,
        product_type=product_type,
        requirements=requirements,
        top_k=top_k
    )


def get_strategy_for_product(
    product_type: str,
    requirements: Dict[str, Any] = None,
    top_k: int = 7
) -> Dict[str, Any]:
    """
    Get procurement strategy for a specific product type.

    This is the main entry point for the product search workflow.

    Args:
        product_type: Product type to get strategy for
        requirements: Additional requirements context
        top_k: Number of documents to retrieve

    Returns:
        Strategy data with preferred/forbidden vendors and priorities
    """
    agent = create_strategy_chat_agent()
    return agent.run(
        question=f"What is the procurement strategy for {product_type}?",
        product_type=product_type,
        requirements=requirements,
        top_k=top_k
    )


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the agent
    logger.info("="*80)
    logger.info("TESTING STRATEGY CHAT AGENT (TRUE RAG)")
    logger.info("="*80)

    test_queries = [
        {"product_type": "pressure transmitter", "question": ""},
        {"product_type": "flow meter", "question": "What vendors should we prefer?"},
        {"product_type": "temperature sensor", "question": "Are there any forbidden vendors?"}
    ]

    agent = create_strategy_chat_agent()

    for i, query in enumerate(test_queries, 1):
        logger.info(f"\n[Test {i}] Product: {query['product_type']}, Question: {query['question'] or 'Default'}")
        logger.info("-"*80)

        result = agent.run(
            question=query['question'],
            product_type=query['product_type'],
            top_k=3
        )

        if result['success']:
            logger.info(f"\nAnswer: {result['answer'][:300]}...")
            logger.info(f"\nPreferred Vendors: {result['preferred_vendors']}")
            logger.info(f"Forbidden Vendors: {result['forbidden_vendors']}")
            logger.info(f"Confidence: {result['confidence']:.2f}")
            logger.info(f"Sources: {', '.join(result['sources_used'])}")
        else:
            logger.error(f"\n[ERROR] {result.get('error', 'Unknown error')}")

        logger.info("="*80)
