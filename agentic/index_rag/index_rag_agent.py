# agentic/index_rag_agent.py
# Index RAG Agent - LLM-Powered Product Indexing and Retrieval
# Uses Flash Lite for intent classification and output structuring

import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Import prompt loader
from prompts_library import load_prompt


# ============================================================================
# PROMPTS - Loaded from prompts_library
# ============================================================================

INTENT_CLASSIFICATION_PROMPT = load_prompt("index_rag_intent_classification_prompt")

OUTPUT_STRUCTURING_PROMPT = load_prompt("index_rag_output_structuring_prompt")


# ============================================================================
# INDEX RAG AGENT CLASS
# ============================================================================

class IndexRAGAgent:
    """
    LLM-Powered Index RAG Agent for product search.
    
    Features:
    - Flash Lite intent classification
    - Hierarchical metadata filtering
    - Parallel database + web indexing
    - Standards + Strategy filtering
    - Structured output generation
    """
    
    def __init__(self, temperature: float = 0.1):
        """
        Initialize the Index RAG Agent.
        
        Args:
            temperature: LLM temperature for generation
        """
        from llm_fallback import create_llm_with_fallback
        
        # Use Flash for fast classification and structuring
        self.llm = create_llm_with_fallback(
            model="gemini-2.5-flash",
            temperature=temperature,
            max_tokens=2000
        )
        
        self.parser = JsonOutputParser()
        
        # Create chains
        self.intent_chain = (
            ChatPromptTemplate.from_template(INTENT_CLASSIFICATION_PROMPT) 
            | self.llm 
            | self.parser
        )
        
        self.output_chain = (
            ChatPromptTemplate.from_template(OUTPUT_STRUCTURING_PROMPT)
            | self.llm
            | self.parser
        )
        
        logger.info("[IndexRAGAgent] Initialized with Gemini Flash")
    
    def classify_intent(self, query: str) -> Dict[str, Any]:
        """
        Classify user intent using LLM.
        
        Args:
            query: User's search query
            
        Returns:
            Classified intent with extracted metadata
        """
        logger.info(f"[IndexRAGAgent] Classifying intent for: {query[:100]}...")
        
        try:
            result = self.intent_chain.invoke({"query": query})
            
            logger.info(f"[IndexRAGAgent] Intent: {result.get('intent')}, "
                       f"Product: {result.get('product_type')}, "
                       f"Confidence: {result.get('confidence', 0):.2f}")
            
            return {
                "success": True,
                **result
            }
            
        except Exception as e:
            logger.error(f"[IndexRAGAgent] Intent classification failed: {e}")
            
            # Fallback to regex extraction
            from tools.metadata_filter import (
                extract_product_types,
                extract_vendors,
                extract_models
            )
            
            return {
                "success": False,
                "intent": "search",
                "product_type": extract_product_types(query)[0] if extract_product_types(query) else None,
                "vendors": extract_vendors(query),
                "models": extract_models(query),
                "specifications": {},
                "filters": {},
                "confidence": 0.5,
                "error": str(e)
            }
    
    def structure_output(
        self,
        query: str,
        search_results: List[Dict],
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Structure search results using LLM.
        
        Args:
            query: Original user query
            search_results: List of filtered product results
            metadata: Additional context (filters applied, etc.)
            
        Returns:
            Structured output for frontend
        """
        logger.info(f"[IndexRAGAgent] Structuring {len(search_results)} results...")
        
        try:
            # Format results for LLM
            results_text = json.dumps(search_results[:10], indent=2, default=str)  # Limit for context
            
            result = self.output_chain.invoke({
                "query": query,
                "result_count": len(search_results),
                "search_results": results_text
            })
            
            logger.info(f"[IndexRAGAgent] Structured output: {len(result.get('recommended_products', []))} products")
            
            return {
                "success": True,
                **result
            }
            
        except Exception as e:
            logger.error(f"[IndexRAGAgent] Output structuring failed: {e}")
            
            # Fallback to simple formatting
            return {
                "success": False,
                "summary": f"Found {len(search_results)} products",
                "recommended_products": search_results[:10],
                "total_found": len(search_results),
                "error": str(e)
            }
    
    def run(
        self,
        query: str,
        requirements: Dict[str, Any] = None,
        top_k: int = 7,
        enable_web_search: bool = True
    ) -> Dict[str, Any]:
        """
        Full Index RAG pipeline.
        
        Steps:
        1. Classify intent with LLM
        2. Apply hierarchical metadata filter
        3. Run parallel indexing (database + web)
        4. Apply specs filter (standards + strategy)
        5. Structure output with LLM
        
        Args:
            query: User's search query
            requirements: Optional structured requirements
            top_k: Max results to return
            enable_web_search: Enable web search thread
            
        Returns:
            Structured search results
        """
        logger.info(f"[IndexRAGAgent] Starting Index RAG for: {query[:100]}...")
        start_time = datetime.utcnow()
        
        result = {
            "success": False,
            "query": query,
            "stages": {}
        }
        
        try:
            # Stage 1: Intent Classification
            logger.info("[IndexRAGAgent] Stage 1: Intent Classification")
            intent_result = self.classify_intent(query)
            result["stages"]["intent"] = intent_result
            
            product_type = intent_result.get("product_type")
            explicit_vendors = intent_result.get("vendors", [])
            
            # Stage 2: Hierarchical Metadata Filter
            logger.info("[IndexRAGAgent] Stage 2: Metadata Filter")
            from tools.metadata_filter import filter_by_hierarchy
            
            metadata_result = filter_by_hierarchy(
                query=query,
                top_k=top_k,
                explicit_product_type=product_type,
                explicit_vendors=explicit_vendors if explicit_vendors else None
            )
            result["stages"]["metadata_filter"] = {
                "product_types": metadata_result.get("product_types"),
                "vendors_count": len(metadata_result.get("vendors", [])),
                "filter_applied": metadata_result.get("filter_applied")
            }
            
            # Stage 3: Parallel Indexing
            logger.info("[IndexRAGAgent] Stage 3: Parallel Indexing")
            from tools.parallel_indexer import run_parallel_indexing
            
            # Extract product types and vendors from metadata filter result
            detected_product_types = metadata_result.get("product_types", [])
            filtered_vendors = metadata_result.get("vendors", [])
            
            index_result = run_parallel_indexing(
                query=query,
                product_type=detected_product_types[0] if detected_product_types else product_type,
                vendors=filtered_vendors if filtered_vendors else None,
                top_k=top_k,
                enable_web_search=enable_web_search
            )
            result["stages"]["parallel_indexing"] = {
                "database_count": len(index_result.get("database", {}).get("results", [])),
                "web_count": len(index_result.get("web_search", {}).get("results", [])) if enable_web_search else 0,
                "merged_count": index_result.get("total_count", 0)
            }
            
            merged_products = index_result.get("merged_results", [])
            
            # Stage 4: Specs + Strategy Filter
            logger.info("[IndexRAGAgent] Stage 4: Specs & Strategy Filter")
            from tools.specs_filter import apply_specs_and_strategy_filter
            
            filter_result = apply_specs_and_strategy_filter(
                products=merged_products,
                product_type=product_type,
                requirements=requirements,
                top_k=top_k
            )
            result["stages"]["specs_filter"] = {
                "standards_applied": filter_result.get("standards", {}).get("applied", False),
                "strategy_applied": filter_result.get("strategy", {}).get("applied", False),
                "excluded_count": filter_result.get("strategy", {}).get("excluded_count", 0),
                "final_count": filter_result.get("filter_count", 0)
            }
            
            filtered_products = filter_result.get("filtered_products", merged_products)
            
            # Stage 5: Structure Output
            logger.info("[IndexRAGAgent] Stage 5: Output Structuring")
            structured_result = self.structure_output(
                query=query,
                search_results=filtered_products,
                metadata={
                    "intent": intent_result,
                    "filters": filter_result
                }
            )
            result["stages"]["output"] = {
                "structured": structured_result.get("success", False),
                "product_count": len(structured_result.get("recommended_products", []))
            }
            
            # Build final result
            result["success"] = True
            result["intent"] = intent_result.get("intent")
            result["product_type"] = product_type
            result["output"] = structured_result
            result["raw_results"] = filtered_products
            result["processing_time_ms"] = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            logger.info(f"[IndexRAGAgent] Pipeline complete in {result['processing_time_ms']}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"[IndexRAGAgent] Pipeline failed: {e}", exc_info=True)
            result["success"] = False
            result["error"] = str(e)
            return result


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_index_rag_agent(temperature: float = 0.1) -> IndexRAGAgent:
    """Create an IndexRAGAgent instance."""
    return IndexRAGAgent(temperature=temperature)


def run_index_rag(
    query: str,
    requirements: Dict[str, Any] = None,
    top_k: int = 7,
    enable_web_search: bool = True
) -> Dict[str, Any]:
    """
    Run the Index RAG pipeline.
    
    Args:
        query: User's search query
        requirements: Optional requirements
        top_k: Max results
        enable_web_search: Enable web search
        
    Returns:
        Structured search results
    """
    agent = create_index_rag_agent()
    return agent.run(
        query=query,
        requirements=requirements,
        top_k=top_k,
        enable_web_search=enable_web_search
    )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'IndexRAGAgent',
    'create_index_rag_agent',
    'run_index_rag'
]


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("TESTING INDEX RAG AGENT")
    print("=" * 80)
    
    test_queries = [
        "I need a pressure transmitter from Yokogawa for hazardous area",
        "Compare Emerson and Rosemount flow meters",
        "EJA110E specifications"
    ]
    
    agent = create_index_rag_agent()
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[Test {i}] Query: {query}")
        print("-" * 80)
        
        result = agent.run(query=query, top_k=3, enable_web_search=False)
        
        if result["success"]:
            print(f"Intent: {result.get('intent')}")
            print(f"Product Type: {result.get('product_type')}")
            print(f"Processing Time: {result.get('processing_time_ms')}ms")
            
            output = result.get("output", {})
            print(f"Summary: {output.get('summary', 'N/A')}")
            print(f"Products Found: {output.get('total_found', 0)}")
        else:
            print(f"Error: {result.get('error')}")
        
        print("=" * 80)
