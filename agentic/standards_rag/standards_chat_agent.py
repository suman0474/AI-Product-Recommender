# agentic/standards_chat_agent.py
# Specialized Chat Agent for Standards Documentation Q&A

import json
import logging
import re
from typing import Dict, Any, List, Optional
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, field_validator

import os
from dotenv import load_dotenv

# Import existing infrastructure
from agentic.vector_store import get_vector_store
from llm_fallback import create_llm_with_fallback

load_dotenv()
logger = logging.getLogger(__name__)


# ============================================================================
# RESPONSE VALIDATION MODELS (Pydantic)
# ============================================================================

class CitationModel(BaseModel):
    """Citation from a standards document."""
    source: str
    content: str
    relevance: float

    @field_validator('relevance')
    @classmethod
    def validate_relevance(cls, v):
        """Ensure relevance is between 0.0 and 1.0."""
        if not isinstance(v, (int, float)):
            raise ValueError('relevance must be numeric')
        if not (0.0 <= v <= 1.0):
            raise ValueError(f'relevance must be between 0.0 and 1.0, got {v}')
        return float(v)


class StandardsRAGResponse(BaseModel):
    """Standard format for Standards RAG responses."""
    answer: str
    citations: List[CitationModel] = []
    confidence: float
    sources_used: List[str] = []

    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v):
        """Ensure confidence is between 0.0 and 1.0."""
        if not isinstance(v, (int, float)):
            raise ValueError('confidence must be numeric')
        if not (0.0 <= v <= 1.0):
            raise ValueError(f'confidence must be between 0.0 and 1.0, got {v}')
        return float(v)


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
    - Malformed JSON with unescaped newlines/quotes
    - Invalid control characters

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
                # Clean up the match
                cleaned = match.strip() if isinstance(match, str) else match

                # Try sanitizing before parsing
                sanitized = _sanitize_json_string(cleaned)
                parsed = json.loads(sanitized)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue

    # Last resort: try to find JSON-like structure and parse it
    try:
        # Find the first { and last }
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_str = text[start:end + 1]

            # Try sanitizing before parsing
            sanitized = _sanitize_json_string(json_str)
            return json.loads(sanitized)
    except json.JSONDecodeError:
        pass

    logger.warning(f"Failed to extract JSON from response: {text[:200]}...")
    return None


def _sanitize_json_string(text: str) -> str:
    """
    Sanitize JSON string by fixing common formatting issues.

    Fixes:
    - Unescaped newlines in string values
    - Unescaped carriage returns
    - Invalid control characters
    - Mismatched quotes

    Args:
        text: Raw JSON string

    Returns:
        Sanitized JSON string
    """
    # Remove invalid control characters (except whitespace)
    text = ''.join(ch for ch in text if ord(ch) >= 32 or ch in '\n\r\t')

    # Replace literal newlines/carriage returns with escaped versions
    # But be careful not to replace them inside string values
    # This is a heuristic approach: replace \n with \\n outside of quotes

    result = []
    in_string = False
    escape_next = False

    for i, char in enumerate(text):
        if escape_next:
            result.append(char)
            escape_next = False
            continue

        if char == '\\':
            result.append(char)
            escape_next = True
            continue

        if char == '"' and (i == 0 or text[i-1] != '\\'):
            in_string = not in_string
            result.append(char)
        elif char == '\n' and not in_string:
            # Literal newline outside string - skip it
            continue
        elif char == '\r' and not in_string:
            # Literal carriage return outside string - skip it
            continue
        else:
            result.append(char)

    return ''.join(result)


def normalize_response_dict(response_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    FIX #2: Normalize LLM response dictionary to match StandardsRAGResponse schema.

    Handles:
    - answer field being a dict instead of string → converts to string
    - Missing required fields → adds defaults
    - Invalid data types → coerces to correct types
    - Malformed citations → reconstructs valid list

    Args:
        response_dict: Raw parsed response from LLM

    Returns:
        Normalized dictionary matching StandardsRAGResponse schema
    """
    logger.info("[FIX2] Normalizing response dictionary for schema compliance")

    # Ensure answer is a string
    answer = response_dict.get('answer', '')
    if isinstance(answer, dict):
        logger.warning("[FIX2] answer field is dict, converting to string")
        # Try to extract text from dict
        answer = answer.get('text', str(answer))
    answer = str(answer).strip() if answer else "Unable to generate answer."

    # Ensure confidence is a float between 0-1
    confidence = response_dict.get('confidence', 0.5)
    try:
        confidence = float(confidence)
        confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
    except (ValueError, TypeError):
        logger.warning("[FIX2] Invalid confidence value, using default")
        confidence = 0.5

    # Handle citations
    citations = response_dict.get('citations', [])
    if not isinstance(citations, list):
        citations = []

    # Validate and reconstruct citations
    valid_citations = []
    for cite in citations:
        if not isinstance(cite, dict):
            continue
        try:
            valid_cite = {
                'source': str(cite.get('source', 'unknown')),
                'content': str(cite.get('content', '')),
                'relevance': float(cite.get('relevance', 0.5))
            }
            valid_cite['relevance'] = max(0.0, min(1.0, valid_cite['relevance']))
            valid_citations.append(valid_cite)
        except (ValueError, TypeError):
            continue

    # Handle sources_used
    sources_used = response_dict.get('sources_used', [])
    if not isinstance(sources_used, list):
        sources_used = []
    sources_used = [str(s) for s in sources_used if s]

    normalized = {
        'answer': answer,
        'citations': valid_citations,
        'confidence': confidence,
        'sources_used': sources_used
    }

    logger.info("[FIX2] Response normalized successfully")
    return normalized


# ============================================================================
# PROMPT TEMPLATE
# ============================================================================

STANDARDS_CHAT_PROMPT = """You are a Standards Documentation Expert for industrial instrumentation.

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

CRITICAL INSTRUCTIONS FOR JSON OUTPUT:
1. You MUST respond with ONLY valid JSON - nothing else
2. Do NOT include markdown formatting (no ```json ... ``` code blocks)
3. Do NOT include any text before or after the JSON
4. The JSON must be parseable by Python's json.loads() function
5. All string values must use double quotes and proper escaping
6. No trailing commas in arrays or objects
7. All special characters in strings must be escaped

CONTENT INSTRUCTIONS:
1. Answer ONLY based on the provided standards documents above
2. If the information is not in the documents, state: "I don't have specific information about that in the available standards documents."
3. Cite sources using [Source: filename] format inline in your answer
4. Reference specific standard codes (IEC, ISO, API, ANSI, ISA, etc.)
5. Be precise and technical - this is for engineering professionals
6. Include relevant details like specifications, ratings, requirements, or procedures
7. NO HALLUCINATION - Do not invent information not present in the documents

JSON STRUCTURE (REQUIRED - must be valid JSON):
{{
    "answer": "Your detailed answer with inline [Source: filename] citations",
    "citations": [
        {{
            "source": "document_filename.docx",
            "content": "Relevant quote from document",
            "relevance": 0.85
        }}
    ],
    "confidence": 0.90,
    "sources_used": ["filename1.docx", "filename2.docx"]
}}

CRITICAL: Your entire response must be ONLY the JSON object above - no markdown, no code blocks, no explanatory text."""


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
        search_results = None
        context = ""
        
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

            # Step 3: Generate answer with strict JSON parsing
            logger.info("Generating answer with LLM...")

            response = None
            raw_text = None

            try:
                # Invoke LLM directly (without the parser in the chain)
                raw_chain = self.prompt | self.llm
                raw_response = raw_chain.invoke({
                    'question': question,
                    'retrieved_context': context
                })

                # Extract text content from response
                if hasattr(raw_response, 'content'):
                    raw_text = raw_response.content
                elif hasattr(raw_response, 'text'):
                    raw_text = raw_response.text
                else:
                    raw_text = str(raw_response)

                logger.debug(f"Raw LLM response: {raw_text[:200]}...")

                # Parse JSON with strict validation
                try:
                    # First, try direct JSON parsing
                    response_dict = json.loads(raw_text)
                    logger.debug("JSON parsed successfully (direct)")
                except json.JSONDecodeError as json_error:
                    logger.warning(
                        f"Direct JSON parse failed: {json_error}. "
                        f"Attempting extraction from response: {raw_text[:100]}..."
                    )
                    # Try extracting JSON from markdown or other formats
                    response_dict = extract_json_from_response(raw_text)

                    if response_dict is None:
                        logger.error(
                            f"Failed to extract JSON from response. Raw: {raw_text[:300]}"
                        )
                        raise ValueError(
                            "LLM returned invalid JSON format. "
                            "Response must be valid JSON only."
                        )
                    logger.info("JSON extracted successfully (from response)")

                # FIX #2: Normalize response dictionary for schema compliance
                logger.debug("Normalizing response for schema compliance...")
                response_dict = normalize_response_dict(response_dict)

                # Validate against StandardsRAGResponse schema
                try:
                    response = StandardsRAGResponse(**response_dict)
                    logger.debug(f"Response validated: confidence={response.confidence}")
                except Exception as validation_error:
                    logger.error(f"Response validation failed: {validation_error}")
                    logger.error(f"Response dict: {response_dict}")
                    raise ValueError(
                        f"LLM response structure invalid: {validation_error}"
                    )

            except Exception as e:
                logger.error(f"Error generating answer: {e}", exc_info=True)
                # Create fallback response with retrieved sources
                sources_used = list(set(
                    r.get('metadata', {}).get('filename', 'unknown')
                    for r in search_results.get('results', [])
                ))
                response = StandardsRAGResponse(
                    answer=f"I found relevant standards documents but encountered a processing error. Sources: {', '.join(sources_used[:3])}",
                    citations=self.extract_citations(search_results),
                    confidence=0.3,
                    sources_used=sources_used
                )

            # Convert Pydantic model to dict for consistency with rest of code
            if isinstance(response, StandardsRAGResponse):
                response = response.model_dump()
            elif response is None:
                logger.error("Response is None after all attempts")
                response = {}

            # Step 4: Add metadata
            result = {
                'answer': response.get('answer', ''),
                'citations': response.get('citations', []),
                'confidence': response.get('confidence', 0.0),
                'sources_used': response.get('sources_used', [])
            }

            # Calculate average retrieval relevance as fallback confidence
            if search_results and 'results' in search_results and search_results['results']:
                avg_relevance = sum(r.get('relevance_score', 0) for r in search_results['results']) / len(search_results['results'])
                # Use the higher of LLM confidence or retrieval relevance
                result['confidence'] = max(result['confidence'], avg_relevance)
                
                # If sources_used is empty but we have results, populate from search results
                if not result['sources_used']:
                    result['sources_used'] = list(set(
                        r.get('metadata', {}).get('filename', 'unknown')
                        for r in search_results['results']
                    ))

            logger.info(f"Answer generated with confidence: {result['confidence']:.2f}")
            logger.info(f"Sources used: {result['sources_used']}")

            return result

        except Exception as e:
            logger.error(f"Error in StandardsChatAgent.run: {e}", exc_info=True)
            
            # Even on error, try to return useful info if we have search results
            sources_used = []
            if search_results and 'results' in search_results:
                sources_used = list(set(
                    r.get('metadata', {}).get('filename', 'unknown')
                    for r in search_results['results']
                ))
            
            return {
                'answer': f"An error occurred while processing your question: {str(e)}",
                'citations': [],
                'confidence': 0.0,
                'sources_used': sources_used,
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
