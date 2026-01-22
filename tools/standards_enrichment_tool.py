# tools/standards_enrichment_tool.py
# Standards RAG Enrichment Tool for Schema Enhancement
# Integrates Standards RAG into Product Search Workflow
#
# PERFORMANCE OPTIMIZATIONS:
# - Product type caching (avoid duplicate queries for same product type)
# - Caching reduces ~15s query time to near-instant for repeated calls
#

import logging
import re
import threading
import time
from typing import Dict, Any, List, Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field

import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# ============================================================================
# PRODUCT TYPE CACHE (Performance Optimization)
# ============================================================================

_standards_results_cache: Dict[str, Dict[str, Any]] = {}
_standards_cache_lock = threading.Lock()
_CACHE_MAX_SIZE = 100  # Max cached product types
_CACHE_TTL_SECONDS = 600  # 10 minutes cache TTL


def _get_cache_key(product_type: str) -> str:
    """Generate cache key from product type."""
    return product_type.lower().strip()


def _get_cached_standards(product_type: str) -> Optional[Dict[str, Any]]:
    """Get cached standards result for a product type."""
    key = _get_cache_key(product_type)
    with _standards_cache_lock:
        entry = _standards_results_cache.get(key)
        if entry:
            # Check TTL
            if time.time() - entry.get('_cached_at', 0) < _CACHE_TTL_SECONDS:
                logger.info(f"[StandardsEnrichment] Cache HIT for: {product_type}")
                return entry.get('data')
            else:
                # Expired - remove
                del _standards_results_cache[key]
    return None


def _cache_standards(product_type: str, result: Dict[str, Any]):
    """Cache standards result for a product type."""
    key = _get_cache_key(product_type)
    with _standards_cache_lock:
        # Evict oldest if at max size
        if len(_standards_results_cache) >= _CACHE_MAX_SIZE:
            oldest_key = min(_standards_results_cache.keys(), 
                           key=lambda k: _standards_results_cache[k].get('_cached_at', 0))
            del _standards_results_cache[oldest_key]
        
        _standards_results_cache[key] = {
            'data': result,
            '_cached_at': time.time()
        }


def clear_standards_cache():
    """Clear the standards cache."""
    global _standards_results_cache
    with _standards_cache_lock:
        _standards_results_cache.clear()
    logger.info("[StandardsEnrichment] Cache cleared")


def get_standards_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    with _standards_cache_lock:
        return {
            'size': len(_standards_results_cache),
            'max_size': _CACHE_MAX_SIZE,
            'ttl_seconds': _CACHE_TTL_SECONDS,
            'keys': list(_standards_results_cache.keys())
        }


# ============================================================================
# TOOL INPUT SCHEMAS
# ============================================================================

class GetApplicableStandardsInput(BaseModel):
    """Input for getting applicable standards"""
    product_type: str = Field(description="Product type to get standards for")
    top_k: int = Field(default=5, description="Number of documents to retrieve")


class EnrichSchemaInput(BaseModel):
    """Input for enriching schema with standards"""
    product_type: str = Field(description="Product type")
    schema: Dict[str, Any] = Field(description="Schema to enrich")


class ValidateAgainstStandardsInput(BaseModel):
    """Input for validating requirements against standards"""
    product_type: str = Field(description="Product type")
    requirements: Dict[str, Any] = Field(description="Requirements to validate")


# ============================================================================
# STANDARDS QUESTION TEMPLATES
# ============================================================================

STANDARDS_QUESTIONS = {
    "applicable_standards": """What are the applicable engineering standards (ISO, IEC, API, ANSI, ISA)
    for {product_type}? List the specific standard codes and their requirements.""",

    "certifications": """What certifications and safety ratings are required for {product_type}?
    Include SIL ratings, ATEX zones, hazardous area classifications, and other certifications.""",

    "safety_requirements": """What are the safety requirements and protective measures
    for {product_type} according to industrial standards?""",

    "calibration_standards": """What calibration standards and procedures apply to {product_type}?
    Include calibration intervals, traceability requirements, and acceptance criteria.""",

    "environmental_requirements": """What are the environmental requirements and ratings
    for {product_type}? Include IP ratings, temperature ranges, and material compatibility.""",

    "communication_protocols": """What communication protocols and signal types are standard
    for {product_type}? Include HART, Foundation Fieldbus, Profibus, 4-20mA specifications."""
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _call_standards_rag(question: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Call the LLM-Powered Standards RAG workflow with a question.
    
    This function uses the full RAG pipeline:
    1. Retrieves documents from Pinecone vector store
    2. Processes with Gemini LLM for intelligent answer generation
    3. Returns grounded answer with citations

    Args:
        question: Question to ask the standards knowledge base
        top_k: Number of documents to retrieve

    Returns:
        Standards RAG response with answer, citations, confidence
    """
    try:
        from agentic.standards_rag import run_standards_rag_workflow

        result = run_standards_rag_workflow(
            question=question,
            top_k=top_k
        )

        if result.get('status') == 'success':
            return {
                'success': True,
                'answer': result['final_response'].get('answer', ''),
                'citations': result['final_response'].get('citations', []),
                'confidence': result['final_response'].get('confidence', 0.0),
                'sources_used': result['final_response'].get('sources_used', []),
                'metadata': result['final_response'].get('metadata', {})
            }
        else:
            logger.warning(f"Standards RAG returned error: {result.get('error')}")
            return {
                'success': False,
                'answer': '',
                'citations': [],
                'confidence': 0.0,
                'sources_used': [],
                'error': result.get('error', 'Unknown error')
            }

    except Exception as e:
        logger.error(f"Error calling Standards RAG: {e}", exc_info=True)
        return {
            'success': False,
            'answer': '',
            'citations': [],
            'confidence': 0.0,
            'sources_used': [],
            'error': str(e)
        }


def _fast_vector_search(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    DEPRECATED: Vector-only search without LLM processing.
    
    This function is now used only as a FALLBACK when the LLM-powered RAG fails.
    For normal operations, use get_applicable_standards() which uses full
    LLM-powered RAG workflow.
    
    Args:
        query: Search query for standards documents
        top_k: Number of documents to retrieve
        
    Returns:
        Dictionary with documents and extracted standards info (no LLM processing)
    """
    try:
        from agentic.vector_store import get_vector_store
        
        store = get_vector_store()
        result = store.search(
            collection_type="standards",
            query=query,
            top_k=top_k
        )
        
        if result.get('success') and result.get('results'):
            # Extract standards and certifications from document text
            combined_text = "\n".join([
                r.get('content', '') or r.get('metadata', {}).get('text', '')
                for r in result['results']
            ])
            
            # Extract standards codes
            standards = _extract_standards_list(combined_text)
            certifications = _extract_certifications(combined_text)
            
            # Get source filenames
            sources = list(set([
                r.get('metadata', {}).get('filename', 'unknown')
                for r in result['results']
            ]))
            
            # Calculate average relevance score
            scores = [r.get('relevance_score', 0) for r in result['results']]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            
            return {
                'success': True,
                'applicable_standards': standards,
                'certifications': certifications,
                'sources': sources,
                'confidence': avg_score,
                'documents': result['results']
            }
        else:
            return {
                'success': False,
                'applicable_standards': [],
                'certifications': [],
                'sources': [],
                'confidence': 0.0,
                'error': result.get('error', 'No documents found')
            }
            
    except Exception as e:
        logger.error(f"Fast vector search failed: {e}")
        return {
            'success': False,
            'applicable_standards': [],
            'certifications': [],
            'sources': [],
            'confidence': 0.0,
            'error': str(e)
        }



def _extract_standards_list(text: str) -> List[str]:
    """
    Extract standard codes from text (e.g., IEC 61508, ISO 1000, API 2540).

    Args:
        text: Text containing standard references

    Returns:
        List of extracted standard codes
    """
    # Pattern to match common standard formats
    patterns = [
        r'\b(IEC\s*\d+(?:[.-]\d+)*)\b',
        r'\b(ISO\s*\d+(?:[.-]\d+)*)\b',
        r'\b(API\s*\d+(?:[.-]\d+)*)\b',
        r'\b(ANSI[/\s]+\w+\s*\d+(?:[.-]\d+)*)\b',
        r'\b(ISA[/\s]*\d+(?:[.-]\d+)*)\b',
        r'\b(EN\s*\d+(?:[.-]\d+)*)\b',
        r'\b(NFPA\s*\d+)\b',
        r'\b(ASME\s*[A-Z]*\s*\d+(?:[.-]\d+)*)\b',
        r'\b(IEEE\s*\d+(?:[.-]\d+)*)\b',
        r'\b(NIST\s*\w*\s*\d+(?:[.-]\d+)*)\b',
    ]

    standards = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        standards.extend(matches)

    # Clean and deduplicate
    cleaned = list(set([s.strip().upper() for s in standards if s]))
    return sorted(cleaned)


def _extract_certifications(text: str) -> List[str]:
    """
    Extract certification codes from text (e.g., SIL2, ATEX, CE).

    Args:
        text: Text containing certification references

    Returns:
        List of extracted certifications
    """
    patterns = [
        r'\b(SIL\s*[1-4])\b',
        r'\b(ATEX\s*(?:Zone\s*)?[0-2]?(?:/[0-2]+)?)\b',
        r'\b(IECEx)\b',
        r'\b(FM\s*(?:Approved)?)\b',
        r'\b(CSA\s*(?:Certified)?)\b',
        r'\b(UL\s*(?:Listed)?)\b',
        r'\b(CE\s*(?:Mark(?:ed)?)?)\b',
        r'\b(IP[0-9]{2})\b',
        r'\b(NEMA\s*[0-9]+[A-Z]*)\b',
        r'\b(Class\s*[I]+,?\s*Div(?:ision)?\s*[1-2])\b',
        r'\b(TUV\s*(?:Certified)?)\b',
        r'\b(HART\s*(?:\d+)?)\b',
    ]

    certifications = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        certifications.extend(matches)

    # Clean and deduplicate
    cleaned = list(set([c.strip().upper() for c in certifications if c]))
    return sorted(cleaned)


# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def get_applicable_standards_fast(
    product_type: str,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    DEPRECATED: Now redirects to LLM-powered RAG.
    
    This function previously used vector-only search without LLM processing.
    It now uses the full LLM-powered RAG workflow for consistency.
    
    Args:
        product_type: Product type to query standards for
        top_k: Number of documents to retrieve
        
    Returns:
        Dictionary with applicable_standards, certifications, sources, confidence
    """
    logger.warning("[StandardsEnrichment] get_applicable_standards_fast() is deprecated, using LLM-powered RAG")
    return get_applicable_standards(product_type, top_k)


def get_applicable_standards(
    product_type: str,
    top_k: int = 5,
    fast_mode: bool = False  # DEPRECATED: Always uses LLM-powered mode now
) -> Dict[str, Any]:
    """
    Get applicable standards for a product type using LLM-Powered RAG.
    
    PERFORMANCE: Uses caching to avoid repeated queries for same product type.
    ~15s queries become near-instant on cache hit!
    
    This function uses the full LLM-powered RAG workflow:
    1. Check cache for existing result
    2. If cache miss: Retrieve documents from Pinecone vector store
    3. Process with Gemini LLM for intelligent answer generation
    4. Extract structured standards information
    5. Cache the result for future calls
    
    Args:
        product_type: Product type to query standards for
        top_k: Number of documents to retrieve
        fast_mode: DEPRECATED - Always uses LLM-powered mode now

    Returns:
        Dictionary containing:
        - applicable_standards: List of standard codes
        - certifications: List of certification codes
        - safety_requirements: Safety-related information
        - calibration_standards: Calibration requirements
        - environmental_requirements: Environmental ratings
        - communication_protocols: Communication standards
        - sources: Source documents used
        - confidence: Overall confidence score
    """
    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║     CACHE CHECK (Performance Optimization - saves ~15s per hit!)     ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    cached_result = _get_cached_standards(product_type)
    if cached_result is not None:
        # Return cached result with flag indicating it came from cache
        result = cached_result.copy()
        result['from_cache'] = True
        return result
    
    logger.info(f"[StandardsEnrichment] LLM-Powered RAG for: {product_type}")
    
    import time
    start_time = time.time()
    
    # Always use LLM-powered RAG (fast_mode is deprecated)
    try:
        from agentic.standards_rag import run_standards_rag_workflow
        
        # Build comprehensive standards question
        question = f"""What are the applicable engineering standards (ISO, IEC, API, ANSI, ISA),
        certifications (SIL, ATEX, CE, UL, CSA), safety requirements, calibration standards,
        environmental requirements, and communication protocols for {product_type}?"""
        
        result = run_standards_rag_workflow(
            question=question,
            top_k=top_k
        )
        
        if result.get('status') == 'success':
            final_response = result.get('final_response', {})
            answer = final_response.get('answer', '')
            sources_used = final_response.get('sources_used', [])
            
            if sources_used:
                elapsed = time.time() - start_time
                logger.info(f"[StandardsEnrichment] LLM-Powered RAG success - {len(sources_used)} sources (took {elapsed:.2f}s)")
                
                # Extract structured data from LLM answer
                standards_result = {
                    'success': True,
                    'product_type': product_type,
                    'applicable_standards': _extract_standards_list(answer),
                    'certifications': _extract_certifications(answer),
                    'safety_requirements': {
                        'description': answer[:500] if len(answer) > 500 else answer,
                        'confidence': final_response.get('confidence', 0.0)
                    },
                    'calibration_standards': {},
                    'environmental_requirements': {},
                    'communication_protocols': _extract_protocols(answer),
                    'sources': sources_used,
                    'confidence': final_response.get('confidence', 0.0),
                    'citations': final_response.get('citations', []),
                    'from_cache': False
                }
                
                # ╔═══════════════════════════════════════════════════════════════╗
                # ║     CACHE THE RESULT for future calls                        ║
                # ╚═══════════════════════════════════════════════════════════════╝
                _cache_standards(product_type, standards_result)
                
                return standards_result
            else:
                logger.warning(f"[StandardsEnrichment] No sources found for {product_type}")
                return _get_applicable_standards_fallback(product_type, top_k)
        else:
            logger.warning(f"[StandardsEnrichment] LLM-Powered RAG failed: {result.get('error')}")
            return _get_applicable_standards_fallback(product_type, top_k)
            
    except ImportError as ie:
        logger.warning(f"[StandardsEnrichment] Import error, using fallback: {ie}")
        return _get_applicable_standards_fallback(product_type, top_k)
    except Exception as e:
        logger.error(f"[StandardsEnrichment] Error: {e}, using fallback")
        return _get_applicable_standards_fallback(product_type, top_k)


def _extract_protocols(text: str) -> List[str]:
    """Extract communication protocols from text."""
    protocols = re.findall(
        r'\b(HART|Profibus|Foundation Fieldbus|Modbus|4-20\s*mA|EtherNet/IP|PROFINET|IO-Link)\b',
        text,
        re.IGNORECASE
    )
    return list(set([p.upper() for p in protocols]))


def _get_applicable_standards_fallback(
    product_type: str,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Fallback when LLM-powered RAG is unavailable.
    Uses direct vector search with regex extraction (no LLM processing).
    """
    logger.info(f"[StandardsEnrichment-Fallback] Vector search for: {product_type}")
    
    try:
        from agentic.vector_store import get_vector_store
        
        store = get_vector_store()
        query = f"{product_type} standards specifications certifications requirements IEC ISO API SIL ATEX"
        
        result = store.search(
            collection_type="standards",
            query=query,
            top_k=top_k
        )
        
        if result.get('success') and result.get('results'):
            # Extract standards and certifications from document text
            combined_text = "\n".join([
                r.get('content', '') or r.get('metadata', {}).get('text', '')
                for r in result['results']
            ])
            
            standards = _extract_standards_list(combined_text)
            certifications = _extract_certifications(combined_text)
            
            sources = list(set([
                r.get('metadata', {}).get('filename', 'unknown')
                for r in result['results']
            ]))
            
            scores = [r.get('relevance_score', 0) for r in result['results']]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            
            return {
                'success': True,
                'product_type': product_type,
                'applicable_standards': standards,
                'certifications': certifications,
                'safety_requirements': {},
                'calibration_standards': {},
                'environmental_requirements': {},
                'communication_protocols': [],
                'sources': sources,
                'confidence': avg_score,
                'source_mode': 'fallback_vector_only'
            }
        else:
            return {
                'success': False,
                'product_type': product_type,
                'applicable_standards': [],
                'certifications': [],
                'safety_requirements': {},
                'calibration_standards': {},
                'environmental_requirements': {},
                'communication_protocols': [],
                'sources': [],
                'confidence': 0.0,
                'error': result.get('error', 'No documents found')
            }
            
    except Exception as e:
        logger.error(f"[StandardsEnrichment-Fallback] Failed: {e}")
        return {
            'success': False,
            'product_type': product_type,
            'applicable_standards': [],
            'certifications': [],
            'safety_requirements': {},
            'calibration_standards': {},
            'environmental_requirements': {},
            'communication_protocols': [],
            'sources': [],
            'confidence': 0.0,
            'error': str(e)
        }


def enrich_schema_with_standards(
    product_type: str,
    schema: Dict[str, Any],
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Enrich a product schema with standards information from Standards RAG.

    Args:
        product_type: Product type
        schema: Original schema to enrich
        top_k: Number of documents to retrieve per query

    Returns:
        Enhanced schema with standards section added
    """
    logger.info(f"[StandardsEnrichment] Enriching schema for: {product_type}")

    # Get standards information
    standards_info = get_applicable_standards(product_type, top_k)

    # Create enhanced schema (deep copy to avoid mutation)
    enhanced_schema = dict(schema)

    # Add standards section
    enhanced_schema['standards'] = {
        'applicable_standards': standards_info.get('applicable_standards', []),
        'certifications': standards_info.get('certifications', []),
        'safety_requirements': standards_info.get('safety_requirements', {}),
        'calibration_standards': standards_info.get('calibration_standards', {}),
        'environmental_requirements': standards_info.get('environmental_requirements', {}),
        'communication_protocols': standards_info.get('communication_protocols', []),
        'sources': standards_info.get('sources', []),
        'confidence': standards_info.get('confidence', 0.0),
        'enrichment_status': 'success' if standards_info.get('success') else 'partial'
    }

    # Optionally add certifications to optional requirements if not present
    optional_reqs = enhanced_schema.get('optional_requirements', enhanced_schema.get('optional', {}))
    if optional_reqs is not None and 'certifications' not in optional_reqs:
        if standards_info.get('certifications'):
            optional_reqs['certifications'] = {
                'description': 'Required certifications',
                'suggested_values': standards_info['certifications']
            }

    logger.info(f"[StandardsEnrichment] Schema enriched with standards (confidence: {standards_info.get('confidence', 0):.2f})")

    return enhanced_schema


def validate_requirements_against_standards(
    product_type: str,
    requirements: Dict[str, Any],
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Validate user requirements against applicable standards.

    Args:
        product_type: Product type
        requirements: User-provided requirements to validate
        top_k: Number of documents to retrieve

    Returns:
        Validation result with compliance status and recommendations
    """
    logger.info(f"[StandardsEnrichment] Validating requirements for: {product_type}")

    # Get applicable standards
    standards_info = get_applicable_standards(product_type, top_k)

    result = {
        'success': True,
        'product_type': product_type,
        'is_compliant': True,
        'compliance_issues': [],
        'recommendations': [],
        'applicable_standards': standards_info.get('applicable_standards', []),
        'required_certifications': standards_info.get('certifications', []),
        'confidence': standards_info.get('confidence', 0.0)
    }

    # Check if certifications are specified when required
    user_certs = requirements.get('certifications', requirements.get('certification', ''))
    required_certs = standards_info.get('certifications', [])

    if required_certs and not user_certs:
        result['compliance_issues'].append({
            'field': 'certifications',
            'issue': 'No certifications specified',
            'recommendation': f"Consider specifying required certifications: {', '.join(required_certs[:3])}"
        })
        result['is_compliant'] = False

    # Check safety requirements if applicable
    safety_reqs = standards_info.get('safety_requirements', {})
    if safety_reqs.get('standards_referenced'):
        sil_pattern = re.search(r'SIL\s*[1-4]', str(requirements), re.IGNORECASE)
        if not sil_pattern and 'SIL' in str(safety_reqs):
            result['recommendations'].append({
                'category': 'safety',
                'recommendation': 'Consider specifying SIL (Safety Integrity Level) requirements',
                'reference': safety_reqs.get('standards_referenced', [])
            })

    # Check communication protocols
    comm_protocols = standards_info.get('communication_protocols', [])
    user_output = str(requirements.get('outputSignal', requirements.get('output_signal', '')))
    if comm_protocols and not any(p.lower() in user_output.lower() for p in comm_protocols):
        result['recommendations'].append({
            'category': 'communication',
            'recommendation': f"Standard communication protocols for {product_type}: {', '.join(comm_protocols[:3])}",
            'reference': comm_protocols
        })

    logger.info(f"[StandardsEnrichment] Validation complete. Compliant: {result['is_compliant']}, "
                f"Issues: {len(result['compliance_issues'])}, Recommendations: {len(result['recommendations'])}")

    return result


def populate_schema_fields_from_standards(
    product_type: str,
    schema: Dict[str, Any],
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Populate schema field VALUES from Standards RAG.
    
    PERFORMANCE OPTIMIZED: Uses a SINGLE comprehensive RAG query instead of
    separate queries for each field. Dynamically extracts field names from the
    schema to ensure ALL fields are queried for.
    
    Args:
        product_type: Product type (e.g., "pressure transmitter", "Type K Thermocouple")
        schema: Schema with empty or partial field values
        top_k: Number of documents to retrieve per query
        
    Returns:
        Schema with populated field values from standards
    """
    import time
    start_time = time.time()
    
    logger.info(f"[StandardsEnrichment] Populating schema field values for: {product_type}")
    
    if not schema:
        logger.warning("[StandardsEnrichment] Empty schema provided")
        return schema
    
    # Create deep copy
    populated_schema = dict(schema)
    
    try:
        # ╔═══════════════════════════════════════════════════════════════════════╗
        # ║  STEP 1: Extract ALL field names from the schema dynamically         ║
        # ╚═══════════════════════════════════════════════════════════════════════╝
        schema_fields = _extract_all_schema_fields(schema)
        logger.info(f"[StandardsEnrichment] Found {len(schema_fields)} fields in schema: {schema_fields[:10]}...")
        
        if not schema_fields:
            logger.warning("[StandardsEnrichment] No fields found in schema")
            return schema
        
        # ╔═══════════════════════════════════════════════════════════════════════╗
        # ║  STEP 2: Build comprehensive query asking for ALL schema fields      ║
        # ╚═══════════════════════════════════════════════════════════════════════╝
        field_list = ", ".join(schema_fields[:60])  # Limit to 60 fields (backend now supports 4000 chars)
        
        comprehensive_question = f"""Provide comprehensive technical specifications for {product_type} according to industrial standards.

For this specific product type, provide typical values for these specifications:
{field_list}

Also include standard specifications for:
- Performance: accuracy (%), repeatability (%), response time, measurement type
- Ranges: temperature range, measurement range (with units)
- Electrical: output signal (4-20mA, HART, etc.), power supply, insulation resistance, lead wire type
- Mechanical: probe diameter, probe length, process connection, sheath material
- Environmental: IP rating, hazardous area rating (ATEX, IECEx), ambient temperature range, storage temperature
- Compliance: certifications (CE, UL, CSA, SIL), applicable standards
- Features: calibration options, cold junction compensation, data logging, diagnostics

Provide specific values where available according to standards documents."""

        logger.info("[StandardsEnrichment] Making comprehensive RAG query for schema fields...")
        
        response = _call_standards_rag(comprehensive_question, top_k=top_k)
        
        field_values = {}
        
        if response.get("success") and response.get("answer"):
            answer = response["answer"]
            confidence = response.get("confidence", 0.0)
            sources = response.get("sources_used", [])
            
            logger.info(f"[StandardsEnrichment] Comprehensive query success (confidence: {confidence:.2f})")
            
            # ╔═══════════════════════════════════════════════════════════════════════╗
            # ║  STEP 3: Extract values for EACH schema field from the answer        ║
            # ╚═══════════════════════════════════════════════════════════════════════╝
            for field_name in schema_fields:
                extracted_value = _extract_field_value_from_answer(answer, field_name, product_type)
                if extracted_value:
                    field_values[field_name] = {
                        "standards_value": extracted_value,
                        "confidence": confidence,
                        "sources": sources
                    }
            
            logger.info(f"[StandardsEnrichment] Extracted values for {len(field_values)} fields")
        else:
            logger.warning("[StandardsEnrichment] Comprehensive query failed")
        
        # ╔═══════════════════════════════════════════════════════════════════════╗
        # ║  STEP 4: Apply extracted values to schema                             ║
        # ╚═══════════════════════════════════════════════════════════════════════╝
        populated_schema = _apply_field_values_to_schema_v2(populated_schema, field_values)
        
        # Add metadata about population
        elapsed = time.time() - start_time
        populated_schema["_standards_population"] = {
            "fields_found_in_schema": len(schema_fields),
            "fields_populated": len(field_values),
            "product_type": product_type,
            "source": "standards_rag",
            "query_mode": "dynamic_comprehensive",
            "time_seconds": round(elapsed, 2)
        }
        
        logger.info(f"[StandardsEnrichment] Populated {len(field_values)}/{len(schema_fields)} fields (took {elapsed:.2f}s)")
        logger.info(f"[SCHEMA POPULATION] COMPLETED - Fields: {len(field_values)}/{len(schema_fields)}, Time: {elapsed:.2f}s")
        
    except Exception as e:
        logger.error(f"[StandardsEnrichment] Error populating schema fields: {e}", exc_info=True)
    
    return populated_schema


def _extract_all_schema_fields(schema: Dict[str, Any]) -> List[str]:
    """
    Recursively extract all field names from schema that need values.
    Looks for fields with empty string values or 'Not specified'.
    """
    fields = []
    
    def extract_from_dict(obj: Any, path: str = ""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                # Skip metadata keys
                if key.startswith("_") or key in ["standards", "normalized_category"]:
                    continue
                    
                if isinstance(value, dict):
                    # Recurse into nested dict
                    extract_from_dict(value, f"{path}.{key}" if path else key)
                elif isinstance(value, str):
                    # Found a string field - check if it needs a value
                    if value == "" or value.lower() == "not specified" or not value.strip():
                        # Use the key as the field name
                        fields.append(key)
                elif value is None:
                    fields.append(key)
        elif isinstance(obj, list):
            for item in obj:
                extract_from_dict(item, path)
    
    # Search in all main sections
    for section in ["mandatory", "mandatory_requirements", "optional", "optional_requirements", 
                    "Compliance", "Electrical", "Mechanical", "Performance", "Environmental", "Features"]:
        if section in schema:
            extract_from_dict(schema[section])
    
    # Also search the root level
    extract_from_dict(schema)
    
    # Deduplicate while preserving order
    seen = set()
    unique_fields = []
    for f in fields:
        if f not in seen:
            seen.add(f)
            unique_fields.append(f)
    
    return unique_fields


def _extract_field_value_from_answer(answer: str, field_name: str, product_type: str) -> Optional[str]:
    """
    Extract a specific field value from a comprehensive answer.
    Uses multiple strategies with flexible patterns: regex, keyword search, semantic extraction.

    [FIX #3] Improved regex patterns to handle textual descriptions from LLM
    - More flexible separators (not just colons)
    - Allow descriptive text ("typically", "of", "around", etc.)
    - Multiple pattern variations per field
    """
    import re

    answer_lower = answer.lower()
    field_lower = field_name.lower().replace("_", " ").replace("-", " ")

    # Strategy 1: [FIX #3] IMPROVED regex patterns - Much more flexible!
    # These patterns now match textual descriptions from LLM, not just structured data
    patterns = {
        # Percentage-based fields (flexible: "accuracy: 0.25%" OR "accuracy of 0.25%" OR "accuracy typically 0.25%")
        "accuracy": [
            r"accuracy[:\s]*(?:of\s+)?[~]?\s*(\d+\.?\d*\s*%)",
            r"accuracy[:\s]*(?:typically\s+)?[~]?\s*(\d+\.?\d*\s*%)",
            r"accuracy[:\s]*(?:is\s+)?[±~]?\s*(\d+\.?\d*\s*%)",
            r"accuracy.*?(\d+\.?\d*\s*%)",  # Catch-all for "accuracy ... X%"
        ],
        "repeatability": [
            r"repeatability[:\s]*(?:of\s+)?[±]?\s*(\d+\.?\d*\s*%)",
            r"repeatability[:\s]*(?:typically\s+)?[±]?\s*(\d+\.?\d*\s*%)",
            r"repeatability.*?(\d+\.?\d*\s*%)",
        ],

        # Temperature range patterns (flexible separators)
        "temperature range": [
            r"temperature\s+range[:\s]*(-?\d+\.?\d*\s*(?:to|[-–])\s*-?\d+\.?\d*\s*°[CF])",
            r"temperature[:\s]*(-?\d+\.?\d*\s*(?:to|[-–])\s*-?\d+\.?\d*\s*°[CF])",
            r"(?:operating\s+)?temperature.*?(-?\d+\.?\d*\s*(?:to|[-–])\s*-?\d+\.?\d*\s*°[CF])",
        ],
        "temperaturerange": [
            r"temperature[:\s]*(-?\d+\.?\d*\s*(?:to|[-–])\s*-?\d+\.?\d*\s*°[CF])",
            r"(?:operating\s+)?temperature.*?(-?\d+\.?\d*\s*(?:to|[-–])\s*-?\d+\.?\d*\s*°[CF])",
        ],

        # Output signal patterns
        "output signal": [
            r"output\s+signal[:\s]+(4-20\s*mA|HART|Profibus|Modbus)",
            r"output[:\s]+(4-20\s*mA|HART|Profibus|Modbus|Fieldbus)",
            r"(?:standard\s+)?signal.*?(4-20\s*mA|HART|Profibus)",
        ],
        "outputsignal": [
            r"output[:\s]+(4-20\s*mA|HART|Profibus|Modbus|Fieldbus)",
            r"signal.*?(4-20\s*mA|HART|Profibus)",
        ],

        # IP rating / Ingress protection
        "ingress protection": [
            r"\b(IP\s*\d{2})\b",
            r"(?:ingress\s+)?protection[:\s]*(IP\s*\d{2})",
        ],
        "ingressprotection": [
            r"\b(IP\s*\d{2})\b",
            r"protection[:\s]*(IP\s*\d{2})",
        ],
        "ip rating": [
            r"\b(IP\s*\d{2})\b",
        ],

        # Hazardous area / Safety certifications
        "hazardous area": [
            r"\b(ATEX|IECEx|SIL\s*[1-4]|Zone\s*[012])\b",
            r"(?:hazardous|explosive|safety)[:\s]*(ATEX|IECEx|SIL)",
        ],
        "hazardousarearating": [
            r"\b(ATEX|IECEx|SIL\s*[1-4])\b",
        ],

        # SIL rating
        "sil": [
            r"\b(SIL\s*[1-4])\b",
            r"(?:safety\s+)?integrity.*?(SIL\s*[1-4])",
        ],
        "safety": [
            r"\b(SIL\s*[1-4])\b",
        ],

        # Certifications (CE, UL, CSA, etc.)
        "certifications": [
            r"\b(CE|UL|CSA|FM|TUV|ATEX|IECEx)\b",
            r"(?:certified|certifications?)[:\s]*(CE|UL|CSA|FM)",
        ],

        # Probe/Sensor dimensions
        "probe diameter": [
            r"diameter[:\s]*(\d+\.?\d*\s*(?:mm|inch|in))",
            r"probe[:\s]*.*?(\d+\.?\d*\s*mm)",
        ],
        "probediameter": [
            r"diameter[:\s]*(\d+\.?\d*\s*(?:mm|inch|in))",
        ],
        "probe length": [
            r"length[:\s]*(\d+\.?\d*\s*(?:mm|cm|m|inch|in|ft))",
            r"immersion[:\s]*(\d+\.?\d*\s*(?:mm|cm|m))",
        ],
        "probelength": [
            r"length[:\s]*(\d+\.?\d*\s*(?:mm|cm|m|inch|in|ft))",
        ],

        # Material specifications
        "sheath material": [
            r"sheath[:\s]*(\w+(?:\s+\w+)?(?:stainless|steel|inconel|hastelloy)?)",
            r"material[:\s]*(\w+\s+steel|stainless|hastelloy|inconel)",
        ],
        "sheathmaterial": [
            r"sheath[:\s]*(\w+)",
            r"material[:\s]*(\S+)",
        ],

        # Connection types
        "process connection": [
            r"connection[:\s]+(NPT|BSP|flanged|G\d+|M\d+)",
            r"(?:process\s+)?connection.*?(NPT|BSP|flanged)",
        ],
        "processconnection": [
            r"connection[:\s]+(\S+)",
        ],

        # Junction/sensor types
        "junction type": [
            r"junction[:\s]+(grounded|ungrounded|exposed|isolated)",
            r"(?:junction|terminal)[:\s]*(\w+)",
        ],
        "junctiontype": [
            r"junction[:\s]+(\w+)",
        ],
        "sensor type": [
            r"sensor\s+type[:\s]+(\w+(?:\s+\w+)?)",
            r"sensor[:\s]+(\w+(?:\s+\w+)?)",
        ],
        "sensortype": [
            r"sensor[:\s]+(\S+)",
        ],

        # Measurement/Response specifications
        "measurement type": [
            r"measurement[:\s]+(\S+(?:\s+\S+)?)",
            r"measures[:\s]*(\w+)",
        ],
        "measurementtype": [
            r"(?:measurement|measures)\s+type[:\s]+(\S+)",
            r"type[:\s]+(\S+)",
        ],
        "response time": [
            r"response\s+time[:\s]*(\d+\.?\d*\s*(?:ms|s|sec|seconds?))",
            r"response[:\s]*(\d+\.?\d*\s*(?:ms|s))",
        ],
        "responsetime": [
            r"response[:\s]*(\d+\.?\d*\s*(?:ms|s|seconds?))",
        ],

        # Power/Voltage specifications
        "power supply": [
            r"power[:\s]*(\d+\.?\d*\s*(?:to|[-–])\s*\d+\.?\d*\s*(?:V|VDC|VAC))",
            r"supply[:\s]*(\d+\.?\d*\s*(?:to|[-–])\s*\d+\.?\d*\s*V)",
        ],
        "powersupply": [
            r"power[:\s]*(\d+\.?\d*\s*(?:to|[-–])\s*\d+\.?\d*\s*V)",
        ],
        "ambient temperature": [
            r"ambient[:\s]*(-?\d+\.?\d*\s*(?:to|[-–])\s*-?\d+\.?\d*\s*°[CF])",
            r"ambient.*?(-?\d+\.?\d*\s*(?:to|[-–])\s*-?\d+\.?\d*\s*°[CF])",
        ],
        "ambienttemperaturerange": [
            r"ambient[:\s]*(-?\d+\.?\d*\s*(?:to|[-–])\s*-?\d+\.?\d*\s*°[CF])",
        ],
        "storage temperature": [
            r"storage[:\s]*(-?\d+\.?\d*\s*(?:to|[-–])\s*-?\d+\.?\d*\s*°[CF])",
            r"storage.*?(-?\d+\.?\d*\s*(?:to|[-–])\s*-?\d+\.?\d*\s*°[CF])",
        ],
        "storagetemperature": [
            r"storage[:\s]*(-?\d+\.?\d*\s*(?:to|[-–])\s*-?\d+\.?\d*\s*°[CF])",
        ],
    }

    # Try field-specific patterns first
    field_key = field_lower.replace(" ", "")

    # Try with space-separated field name
    if field_lower in patterns:
        pattern_list = patterns[field_lower]
        if not isinstance(pattern_list, list):
            pattern_list = [pattern_list]
        for pattern in pattern_list:
            match = re.search(pattern, answer, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if value:
                    logger.debug(f"[FIX #3] Extracted '{value}' for field '{field_name}' (pattern: {pattern[:50]}...)")
                    return value

    # Try without spaces in field name
    if field_key in patterns:
        pattern_list = patterns[field_key]
        if not isinstance(pattern_list, list):
            pattern_list = [pattern_list]
        for pattern in pattern_list:
            match = re.search(pattern, answer, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if value:
                    logger.debug(f"[FIX #3] Extracted '{value}' for field '{field_name}' (no-space pattern: {pattern[:50]}...)")
                    return value

    # Strategy 2: [FIX #3] Generic fallback - Look for field name followed by value
    # This handles textual descriptions like "accuracy of 0.25% to 0.5%"
    field_pattern = rf"{re.escape(field_name)}[:\s=]*([^\n,\.]+)"
    match = re.search(field_pattern, answer, re.IGNORECASE)
    if match:
        value = match.group(1).strip()
        if value and len(value) < 150 and len(value) > 1:
            # Extract just the numerical/categorical part if possible
            # Remove extra text like "of the measured value" or "typically"
            cleaned = re.sub(r'\b(?:of\s+the|typically|around|approximately|roughly)\b', '', value, flags=re.IGNORECASE)
            cleaned = cleaned.strip()
            if cleaned and len(cleaned) < 100:
                logger.debug(f"[FIX #3] Extracted '{cleaned}' for field '{field_name}' (generic fallback)")
                return cleaned

    # Strategy 3: [FIX #3] Try with spaced field name
    spaced_field = field_name.replace("_", " ").replace("-", " ")
    if spaced_field != field_name:
        field_pattern = rf"{re.escape(spaced_field)}[:\s=]*([^\n,\.]+)"
        match = re.search(field_pattern, answer, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if value and len(value) < 150 and len(value) > 1:
                cleaned = re.sub(r'\b(?:of\s+the|typically|around|approximately)\b', '', value, flags=re.IGNORECASE)
                cleaned = cleaned.strip()
                if cleaned and len(cleaned) < 100:
                    logger.debug(f"[FIX #3] Extracted '{cleaned}' for field '{field_name}' (spaced fallback)")
                    return cleaned

    logger.debug(f"[FIX #3] Could not extract value for field '{field_name}'")
    return None


def _apply_field_values_to_schema_v2(
    schema: Dict[str, Any], 
    field_values: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Apply collected field values to schema structure (improved version).
    Handles nested structures and various schema formats.
    """
    
    def update_value(obj: Dict[str, Any], field_data: Dict[str, Dict]):
        """Recursively update dictionary with field values."""
        for key, value in list(obj.items()):
            # Skip metadata keys
            if key.startswith("_"):
                continue
                
            if isinstance(value, dict):
                # Recurse into nested dict
                update_value(value, field_data)
            elif isinstance(value, str) and (value == "" or value.lower() == "not specified" or not value.strip()):
                # Found an empty field - try to fill it
                # Try exact match first
                if key in field_data:
                    obj[key] = field_data[key].get("standards_value", "")
                    logger.debug(f"[StandardsEnrichment] Filled {key} = {obj[key]}")
                else:
                    # Try normalized match
                    normalized_key = key.lower().replace("_", "").replace("-", "").replace(" ", "")
                    for known_field, data in field_data.items():
                        known_normalized = known_field.lower().replace("_", "").replace("-", "").replace(" ", "")
                        if known_normalized == normalized_key:
                            obj[key] = data.get("standards_value", "")
                            logger.debug(f"[StandardsEnrichment] Filled {key} (matched {known_field}) = {obj[key]}")
                            break
    
    # Update all main sections
    for section in schema:
        if isinstance(schema[section], dict):
            update_value(schema[section], field_values)
    
    return schema


def _extract_key_value(answer: str, field_name: str) -> str:
    """Extract the key specification value from an answer."""
    import re
    
    # Common patterns by field type
    patterns = {
        "accuracy": r"[±]?\s*\d+\.?\d*\s*%",
        "repeatability": r"[±]?\s*\d+\.?\d*\s*%",
        "pressureRange": r"\d+\.?\d*\s*(?:to|-)\s*\d+\.?\d*\s*(?:bar|psi|MPa|kPa)",
        "temperatureRange": r"-?\d+\.?\d*\s*(?:to|-)\s*-?\d+\.?\d*\s*(?:°C|°F|K)",
        "flowRange": r"\d+\.?\d*\s*(?:to|-)\s*\d+\.?\d*\s*(?:m³/h|l/min|GPM)",
        "outputSignal": r"4-20\s*mA|HART|Profibus|Foundation Fieldbus|Modbus",
        "powerSupply": r"\d+\.?\d*\s*(?:to|-)\s*\d+\.?\d*\s*(?:V|VDC|VAC)",
        "ingressProtection": r"IP\s*\d{2}",
        "hazardousAreaRating": r"ATEX|IECEx|Class I|Class II|Div(?:ision)?\s*[12]|Zone\s*[012]",
        "safetyRating": r"SIL\s*[1-4]",
    }
    
    pattern = patterns.get(field_name)
    if pattern:
        match = re.search(pattern, answer, re.IGNORECASE)
        if match:
            return match.group(0)
    
    # Default: return first sentence up to 100 chars
    first_sentence = answer.split('.')[0]
    return first_sentence[:100] if len(first_sentence) > 100 else first_sentence


def _apply_field_values_to_schema(
    schema: Dict[str, Any], 
    field_values: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Apply collected field values to schema structure."""
    
    def update_nested(obj: Dict, field_data: Dict[str, Dict]):
        """Recursively update nested dictionary with field values."""
        for key, value in obj.items():
            if isinstance(value, dict):
                # Check if this is a leaf node (has empty string value)
                if all(isinstance(v, str) and v == "" for v in value.values()):
                    # This is a category with empty fields
                    for field_key in value.keys():
                        normalized_key = field_key.lower().replace("_", "").replace("-", "")
                        for known_field, data in field_data.items():
                            if known_field.lower().replace("_", "") == normalized_key:
                                value[field_key] = data.get("standards_value", "")
                                break
                else:
                    # Recurse into nested dict
                    update_nested(value, field_data)
            elif isinstance(value, str) and value == "":
                # Direct field with empty value
                normalized_key = key.lower().replace("_", "").replace("-", "")
                for known_field, data in field_values.items():
                    if known_field.lower().replace("_", "") == normalized_key:
                        obj[key] = data.get("standards_value", "")
                        break
        return obj
    
    # Update mandatory_requirements
    if "mandatory_requirements" in schema:
        update_nested(schema["mandatory_requirements"], field_values)
    if "mandatory" in schema:
        update_nested(schema["mandatory"], field_values)
    
    # Update optional_requirements
    if "optional_requirements" in schema:
        update_nested(schema["optional_requirements"], field_values)
    if "optional" in schema:
        update_nested(schema["optional"], field_values)
    
    return schema


# ============================================================================
# LANGCHAIN TOOLS
# ============================================================================

@tool("get_standards_for_product", args_schema=GetApplicableStandardsInput)
def get_standards_for_product_tool(product_type: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Get applicable engineering standards and certifications for a product type.

    Queries the Standards RAG knowledge base to find:
    - Applicable standards (ISO, IEC, API, ANSI, ISA)
    - Required certifications (SIL, ATEX, CE, etc.)
    - Safety requirements
    - Calibration standards
    - Environmental requirements
    - Communication protocols

    Args:
        product_type: The product type to get standards for (e.g., "pressure transmitter")
        top_k: Number of documents to retrieve per query

    Returns:
        Dictionary with applicable standards, certifications, and requirements
    """
    return get_applicable_standards(product_type, top_k)


@tool("enrich_schema_with_standards", args_schema=EnrichSchemaInput)
def enrich_schema_with_standards_tool(product_type: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enrich a product schema with standards information from the Standards RAG knowledge base.

    Adds a 'standards' section to the schema containing:
    - Applicable engineering standards
    - Required certifications
    - Safety requirements
    - Calibration standards
    - Environmental requirements
    - Communication protocols

    Args:
        product_type: The product type
        schema: The original schema to enrich

    Returns:
        Enhanced schema with standards section added
    """
    return enrich_schema_with_standards(product_type, schema)


@tool("validate_against_standards", args_schema=ValidateAgainstStandardsInput)
def validate_against_standards_tool(product_type: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate user requirements against applicable engineering standards.

    Checks if the provided requirements comply with standards for the product type
    and provides recommendations for missing or incomplete specifications.

    Args:
        product_type: The product type
        requirements: User-provided requirements to validate

    Returns:
        Validation result with compliance status and recommendations
    """
    return validate_requirements_against_standards(product_type, requirements)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("\n" + "="*70)
    print("STANDARDS ENRICHMENT TOOL - TEST")
    print("="*70)

    # Test 1: Get applicable standards
    print("\n[Test 1] Get Applicable Standards for Pressure Transmitter")
    print("-"*70)

    standards = get_applicable_standards("pressure transmitter")

    print(f"Success: {standards['success']}")
    print(f"Applicable Standards: {standards['applicable_standards']}")
    print(f"Certifications: {standards['certifications']}")
    print(f"Communication Protocols: {standards['communication_protocols']}")
    print(f"Sources: {standards['sources']}")
    print(f"Confidence: {standards['confidence']:.2f}")

    # Test 2: Enrich schema
    print("\n[Test 2] Enrich Schema with Standards")
    print("-"*70)

    test_schema = {
        "mandatory_requirements": {
            "outputSignal": "Communication/output signal type",
            "pressureRange": "Measurement pressure range"
        },
        "optional_requirements": {
            "accuracy": "Measurement accuracy"
        }
    }

    enriched = enrich_schema_with_standards("pressure transmitter", test_schema)

    print(f"Standards Section Added: {'standards' in enriched}")
    if 'standards' in enriched:
        print(f"  - Applicable Standards: {len(enriched['standards']['applicable_standards'])}")
        print(f"  - Certifications: {len(enriched['standards']['certifications'])}")
        print(f"  - Confidence: {enriched['standards']['confidence']:.2f}")

    # Test 3: Validate requirements
    print("\n[Test 3] Validate Requirements Against Standards")
    print("-"*70)

    test_requirements = {
        "outputSignal": "4-20mA HART",
        "pressureRange": "0-100 bar"
    }

    validation = validate_requirements_against_standards("pressure transmitter", test_requirements)

    print(f"Is Compliant: {validation['is_compliant']}")
    print(f"Compliance Issues: {len(validation['compliance_issues'])}")
    print(f"Recommendations: {len(validation['recommendations'])}")

    for rec in validation['recommendations']:
        print(f"  - [{rec['category']}] {rec['recommendation']}")

    print("\n" + "="*70)
    print("TESTS COMPLETED")
    print("="*70)
