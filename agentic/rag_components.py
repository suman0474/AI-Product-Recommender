# agentic/rag_components.py
# RAG System Components for Strategy, Standards, and Inventory

import json
import logging
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os
from dotenv import load_dotenv
from llm_fallback import create_llm_with_fallback

load_dotenv()
logger = logging.getLogger(__name__)


# ============================================================================
# RAG QUERY PROMPTS
# ============================================================================

STRATEGY_RAG_PROMPT = """
You are Engenie's Strategy RAG component. Extract procurement strategy information relevant to the query.

Query Context:
- Product Type: {product_type}
- User Requirements: {requirements}

STRATEGY CATEGORIES TO IDENTIFY:
1. Preferred Vendors - Vendors with strategic partnerships or preferred status
2. Forbidden Vendors - Vendors to avoid due to quality, compliance, or policy
3. Neutral Vendors - All other acceptable vendors
4. Procurement Priorities - Cost optimization, lifecycle cost, sustainability, etc.

Based on your knowledge of industrial procurement strategies, provide:

Return ONLY valid JSON:
{{
    "preferred_vendors": ["<list of preferred vendors for this product type>"],
    "forbidden_vendors": ["<list of vendors to avoid>"],
    "neutral_vendors": ["<list of neutral vendors>"],
    "procurement_priorities": {{
        "<vendor>": <priority_score 1-10>
    }},
    "strategy_notes": "<any relevant strategy notes>",
    "confidence": <0.0-1.0>
}}
"""

STANDARDS_RAG_PROMPT = """
You are Engenie's Standards RAG component. Extract standards and certification requirements.

Query Context:
- Product Type: {product_type}
- User Requirements: {requirements}

STANDARDS TO CHECK:
1. SIL Ratings - Safety Integrity Level requirements (SIL1, SIL2, SIL3)
2. ATEX Zones - Explosive atmosphere zones (Zone 0, 1, 2)
3. API Standards - American Petroleum Institute standards
4. Plant Codes - Facility-specific codes
5. Certifications - ISO, CE, NACE, etc.

Based on your knowledge of industrial standards, provide:

Return ONLY valid JSON:
{{
    "required_sil_rating": "<SIL1|SIL2|SIL3 or null>",
    "atex_zone": "<Zone 0|Zone 1|Zone 2 or null>",
    "required_certifications": ["<list of required certifications>"],
    "api_standards": ["<list of applicable API standards>"],
    "plant_codes": ["<list of plant codes>"],
    "standards_notes": "<any relevant standards notes>",
    "confidence": <0.0-1.0>
}}
"""

INVENTORY_RAG_PROMPT = """
You are Engenie's Inventory RAG component. Extract installed base and inventory constraints.

Query Context:
- Product Type: {product_type}
- User Requirements: {requirements}

INVENTORY ASPECTS TO IDENTIFY:
1. Installed Series - What product series are already installed?
2. Series Restrictions - Are there restrictions like "800 series only"?
3. Spare Parts - Which spare parts are available?
4. Standardized Vendor - Is there a plant-standardized vendor?
5. Compatibility - What compatibility requirements exist?

Based on your knowledge of industrial installations, provide:

Return ONLY valid JSON:
{{
    "installed_series": {{
        "<vendor>": ["<list of installed series>"]
    }},
    "series_restrictions": ["<list of series restrictions>"],
    "available_spare_parts": {{
        "<model>": ["<list of spare parts>"]
    }},
    "standardized_vendor": "<vendor name or null>",
    "compatibility_notes": "<any compatibility requirements>",
    "confidence": <0.0-1.0>
}}
"""


# ============================================================================
# RAG AGGREGATOR CLASS
# ============================================================================

class RAGAggregator:
    """
    Aggregates data from Strategy, Standards, and Inventory RAG sources.
    Supports parallel queries and constraint merging.
    """
    
    def __init__(self, model_name: str = "gemini-2.5-flash", temperature: float = 0.1):
        self.llm = create_llm_with_fallback(
            model=model_name,
            temperature=temperature,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.parser = JsonOutputParser()
        logger.info("RAGAggregator initialized")
    
    def query_strategy_rag(
        self, 
        product_type: str, 
        requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Query the Strategy RAG for procurement strategy data.
        
        This method now uses TRUE RAG with vector store retrieval from Pinecone.
        Falls back to LLM-based inference if the strategy knowledge base is empty.
        """
        try:
            # Try TRUE RAG first (vector store based)
            from .strategy_rag_workflow import get_strategy_for_product
            
            logger.info(f"[Strategy RAG] Querying TRUE RAG for {product_type}...")
            
            result = get_strategy_for_product(
                product_type=product_type,
                requirements=requirements,
                top_k=7
            )
            
            # Check if we got results from vector store
            if result.get('sources_used') and len(result.get('sources_used', [])) > 0:
                logger.info(f"[Strategy RAG] TRUE RAG success - {len(result['sources_used'])} sources used")
                return {
                    "success": True,
                    "source": "strategy_rag",  # Indicates TRUE RAG was used
                    "data": {
                        "preferred_vendors": result.get("preferred_vendors", []),
                        "forbidden_vendors": result.get("forbidden_vendors", []),
                        "neutral_vendors": result.get("neutral_vendors", []),
                        "procurement_priorities": result.get("procurement_priorities", {}),
                        "strategy_notes": result.get("strategy_notes", ""),
                        "confidence": result.get("confidence", 0.0),
                        "sources_used": result.get("sources_used", []),
                        "citations": result.get("citations", [])
                    }
                }
            else:
                logger.warning("[Strategy RAG] No documents in vector store, falling back to LLM inference")
                # Fall back to LLM-based inference
                return self._query_strategy_rag_fallback(product_type, requirements)
                
        except ImportError as ie:
            logger.warning(f"[Strategy RAG] Import error, using fallback: {ie}")
            return self._query_strategy_rag_fallback(product_type, requirements)
        except Exception as e:
            logger.error(f"[Strategy RAG] Error: {e}, using fallback")
            return self._query_strategy_rag_fallback(product_type, requirements)
    
    def _query_strategy_rag_fallback(
        self,
        product_type: str,
        requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Fallback to LLM-based strategy inference when vector store is empty.
        This is the ORIGINAL implementation (LLM inference, not TRUE RAG).
        """
        try:
            prompt = ChatPromptTemplate.from_template(STRATEGY_RAG_PROMPT)
            chain = prompt | self.llm | self.parser
            
            result = chain.invoke({
                "product_type": product_type,
                "requirements": json.dumps(requirements, indent=2)
            })
            
            logger.info(f"[Strategy RAG Fallback] LLM inference completed with confidence: {result.get('confidence', 0)}")
            return {
                "success": True,
                "source": "strategy_llm_inference",  # Indicates fallback was used
                "data": result
            }
            
        except Exception as e:
            logger.error(f"[Strategy RAG Fallback] Failed: {e}")
            return {
                "success": False,
                "source": "strategy",
                "error": str(e),
                "data": {
                    "preferred_vendors": [],
                    "forbidden_vendors": [],
                    "neutral_vendors": [],
                    "procurement_priorities": {}
                }
            }
    
    def query_standards_rag(
        self, 
        product_type: str, 
        requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Query the Standards RAG for certification and compliance data.
        
        Uses the LLM-Powered Standards RAG Agent:
        1. Retrieves top 5 documents from Pinecone vector store
        2. Processes with Gemini LLM to generate grounded answer
        3. Validates response for hallucination
        4. Returns structured data with citations
        """
        try:
            # Try TRUE LLM-Powered RAG first (vector retrieval + LLM processing)
            from .standards_rag_workflow import run_standards_rag_workflow
            
            logger.info(f"[Standards RAG] Querying LLM-Powered RAG for {product_type}...")
            
            # Build comprehensive question for the LLM-powered agent
            question = f"""What are the standards, certifications, SIL ratings, ATEX zones, 
            API standards, plant codes, and compliance requirements for {product_type}?
            
            Context from user requirements:
            {json.dumps(requirements, indent=2)}
            """
            
            # Call LLM-Powered Standards RAG Agent (retrieval + LLM + validation)
            result = run_standards_rag_workflow(
                question=question,
                top_k=5  # Retrieve 5 closest documents
            )
            
            if result.get('status') == 'success':
                final_response = result.get('final_response', {})
                sources_used = final_response.get('sources_used', [])
                
                # Check if we got results from vector store
                if sources_used and len(sources_used) > 0:
                    logger.info(f"[Standards RAG] LLM-Powered RAG success - {len(sources_used)} sources used")
                    
                    # Parse the LLM's answer to extract structured data
                    answer = final_response.get('answer', '')
                    
                    return {
                        "success": True,
                        "source": "standards_rag_llm_powered",  # Indicates TRUE LLM-powered RAG
                        "data": {
                            "required_sil_rating": self._extract_sil_from_answer(answer),
                            "atex_zone": self._extract_atex_from_answer(answer),
                            "required_certifications": self._extract_certifications_from_answer(answer),
                            "api_standards": self._extract_api_standards_from_answer(answer),
                            "plant_codes": [],
                            "standards_notes": answer[:500] if answer else "",
                            "confidence": final_response.get('confidence', 0.0),
                            "sources_used": sources_used,
                            "citations": final_response.get('citations', [])
                        }
                    }
                else:
                    logger.warning("[Standards RAG] No documents in vector store, using fallback")
                    return self._query_standards_rag_fallback(product_type, requirements)
            else:
                logger.warning(f"[Standards RAG] LLM-Powered RAG failed: {result.get('error')}")
                return self._query_standards_rag_fallback(product_type, requirements)
                
        except ImportError as ie:
            logger.warning(f"[Standards RAG] Import error, using fallback: {ie}")
            return self._query_standards_rag_fallback(product_type, requirements)
        except Exception as e:
            logger.error(f"[Standards RAG] Error: {e}, using fallback")
            return self._query_standards_rag_fallback(product_type, requirements)
    
    def _query_standards_rag_fallback(
        self,
        product_type: str,
        requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Fallback to LLM-based standards inference when vector store is empty.
        This uses the ORIGINAL implementation (LLM inference without vector retrieval).
        """
        try:
            prompt = ChatPromptTemplate.from_template(STANDARDS_RAG_PROMPT)
            chain = prompt | self.llm | self.parser
            
            result = chain.invoke({
                "product_type": product_type,
                "requirements": json.dumps(requirements, indent=2)
            })
            
            logger.info(f"[Standards RAG Fallback] LLM inference completed with confidence: {result.get('confidence', 0)}")
            return {
                "success": True,
                "source": "standards_llm_inference",  # Indicates fallback was used
                "data": result
            }
            
        except Exception as e:
            logger.error(f"[Standards RAG Fallback] Failed: {e}")
            return {
                "success": False,
                "source": "standards",
                "error": str(e),
                "data": {
                    "required_sil_rating": None,
                    "atex_zone": None,
                    "required_certifications": [],
                    "api_standards": [],
                    "plant_codes": []
                }
            }
    
    def _extract_sil_from_answer(self, answer: str) -> Optional[str]:
        """Extract SIL rating from LLM answer."""
        import re
        match = re.search(r'SIL\s*([1-4])', answer, re.IGNORECASE)
        return f"SIL{match.group(1)}" if match else None

    def _extract_atex_from_answer(self, answer: str) -> Optional[str]:
        """Extract ATEX zone from LLM answer."""
        import re
        match = re.search(r'(?:ATEX\s*)?Zone\s*([0-2])', answer, re.IGNORECASE)
        return f"Zone {match.group(1)}" if match else None

    def _extract_certifications_from_answer(self, answer: str) -> List[str]:
        """Extract certifications from LLM answer."""
        import re
        patterns = ['SIL[1-4]', 'ATEX', 'IECEx', 'CE', 'UL', 'CSA', 'FM', 'TUV', 'IP[0-9]{2}', 'NEMA']
        certs = []
        for pattern in patterns:
            matches = re.findall(pattern, answer, re.IGNORECASE)
            certs.extend([m.upper() for m in matches])
        return list(set(certs))

    def _extract_api_standards_from_answer(self, answer: str) -> List[str]:
        """Extract API standards from LLM answer."""
        import re
        matches = re.findall(r'API\s*\d+(?:\.\d+)?', answer, re.IGNORECASE)
        return list(set([m.upper().replace(' ', '') for m in matches]))
    
    def query_inventory_rag(
        self, 
        product_type: str, 
        requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Query the Inventory RAG for installed base data"""
        try:
            prompt = ChatPromptTemplate.from_template(INVENTORY_RAG_PROMPT)
            chain = prompt | self.llm | self.parser
            
            result = chain.invoke({
                "product_type": product_type,
                "requirements": json.dumps(requirements, indent=2)
            })
            
            logger.info(f"Inventory RAG query completed with confidence: {result.get('confidence', 0)}")
            return {
                "success": True,
                "source": "inventory",
                "data": result
            }
            
        except Exception as e:
            logger.error(f"Inventory RAG query failed: {e}")
            return {
                "success": False,
                "source": "inventory", 
                "error": str(e),
                "data": {
                    "installed_series": {},
                    "series_restrictions": [],
                    "available_spare_parts": {},
                    "standardized_vendor": None
                }
            }
    
    def query_all_parallel(
        self, 
        product_type: str, 
        requirements: Dict[str, Any],
        max_workers: int = 3
    ) -> Dict[str, Any]:
        """
        Query all three RAG sources in parallel.
        Returns merged results from Strategy, Standards, and Inventory.
        """
        logger.info(f"Starting parallel RAG queries for {product_type}")
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.query_strategy_rag, product_type, requirements): "strategy",
                executor.submit(self.query_standards_rag, product_type, requirements): "standards",
                executor.submit(self.query_inventory_rag, product_type, requirements): "inventory"
            }
            
            for future in as_completed(futures):
                source = futures[future]
                try:
                    result = future.result()
                    results[source] = result
                except Exception as e:
                    logger.error(f"RAG query failed for {source}: {e}")
                    results[source] = {
                        "success": False,
                        "source": source,
                        "error": str(e)
                    }
        
        logger.info(f"Parallel RAG queries completed. Success: {all(r.get('success', False) for r in results.values())}")
        return results
    
    def merge_to_constraint_context(
        self, 
        rag_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge RAG results into a unified ConstraintContext.
        """
        strategy_data = rag_results.get("strategy", {}).get("data", {})
        standards_data = rag_results.get("standards", {}).get("data", {})
        inventory_data = rag_results.get("inventory", {}).get("data", {})
        
        constraint_context = {
            # Strategy Constraints
            "preferred_vendors": strategy_data.get("preferred_vendors", []),
            "forbidden_vendors": strategy_data.get("forbidden_vendors", []),
            "neutral_vendors": strategy_data.get("neutral_vendors", []),
            "procurement_priorities": strategy_data.get("procurement_priorities", {}),
            
            # Standards Constraints
            "required_sil_rating": standards_data.get("required_sil_rating"),
            "atex_zone": standards_data.get("atex_zone"),
            "required_certifications": standards_data.get("required_certifications", []),
            "plant_codes": standards_data.get("plant_codes", []),
            
            # Inventory Constraints
            "installed_series": inventory_data.get("installed_series", {}),
            "series_restrictions": inventory_data.get("series_restrictions", []),
            "available_spare_parts": inventory_data.get("available_spare_parts", {}),
            "standardized_vendor": inventory_data.get("standardized_vendor"),
            
            # Computed fields (to be populated by Strategy Filter)
            "excluded_models": [],
            "boosted_models": []
        }
        
        logger.info("Constraint context merged successfully")
        return constraint_context


# ============================================================================
# STRATEGY FILTER
# ============================================================================

class StrategyFilter:
    """
    Applies constraint rules to filter and prioritize vendor candidates.
    """
    
    def __init__(self):
        logger.info("StrategyFilter initialized")
    
    def apply_strategy_rules(
        self, 
        vendors: List[str],
        constraint_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply strategy rules to vendors.
        
        Rules:
        - Forbidden vendors → EXCLUDE
        - Preferred vendors → INCLUDE with priority boost
        - Neutral vendors → INCLUDE normally
        """
        preferred = constraint_context.get("preferred_vendors", [])
        forbidden = constraint_context.get("forbidden_vendors", [])
        priorities = constraint_context.get("procurement_priorities", {})
        
        filtered = []
        excluded = []
        
        for vendor in vendors:
            vendor_lower = vendor.lower()
            
            # Check if forbidden
            if any(f.lower() in vendor_lower or vendor_lower in f.lower() for f in forbidden):
                excluded.append({
                    "vendor": vendor,
                    "reason": "Forbidden by strategy",
                    "rule": "STRATEGY_FORBIDDEN"
                })
                continue
            
            # Determine preference status
            is_preferred = any(p.lower() in vendor_lower or vendor_lower in p.lower() for p in preferred)
            priority_boost = 7 if is_preferred else 0
            priority_boost += priorities.get(vendor, 0)
            
            filtered.append({
                "vendor": vendor,
                "is_preferred": is_preferred,
                "priority_boost": priority_boost,
                "status": "preferred" if is_preferred else "neutral"
            })
        
        # Sort by priority boost (descending)
        filtered.sort(key=lambda x: x["priority_boost"], reverse=True)
        
        logger.info(f"Strategy filter: {len(filtered)} passed, {len(excluded)} excluded")
        return {
            "filtered_vendors": filtered,
            "excluded_vendors": excluded
        }
    
    def apply_installed_base_rules(
        self, 
        candidates: List[Dict[str, Any]],
        constraint_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply installed-base rules.
        
        Rules:
        - Series restrictions → EXCLUDE non-matching series
        - Standardized vendor → BOOST
        - Spare parts available → PREFER
        """
        series_restrictions = constraint_context.get("series_restrictions", [])
        standardized_vendor = constraint_context.get("standardized_vendor")
        spare_parts = constraint_context.get("available_spare_parts", {})
        
        filtered = []
        excluded = []
        
        for candidate in candidates:
            vendor = candidate.get("vendor", "")
            series = candidate.get("series", "")
            model = candidate.get("model", "")
            
            # Check series restrictions
            excluded_by_series = False
            for restriction in series_restrictions:
                # Parse restrictions like "800 series only"
                if "only" in restriction.lower():
                    required_series = restriction.lower().replace("series", "").replace("only", "").strip()
                    if required_series and required_series not in series.lower():
                        excluded_by_series = True
                        break
            
            if excluded_by_series:
                excluded.append({
                    "vendor": vendor,
                    "model": model,
                    "reason": f"Does not match series restriction: {restriction}",
                    "rule": "INSTALLED_BASE_SERIES"
                })
                continue
            
            # Apply boosts
            boost = candidate.get("priority_boost", 0)
            
            # Standardized vendor boost
            if standardized_vendor and standardized_vendor.lower() in vendor.lower():
                boost += 5
                candidate["matches_standardized"] = True
            
            # Spare parts boost
            if model in spare_parts and spare_parts[model]:
                boost += 3
                candidate["has_spare_parts"] = True
            
            candidate["priority_boost"] = boost
            filtered.append(candidate)
        
        logger.info(f"Installed-base filter: {len(filtered)} passed, {len(excluded)} excluded")
        return {
            "filtered_candidates": filtered,
            "excluded_candidates": excluded
        }
    
    def apply_standards_rules(
        self, 
        candidates: List[Dict[str, Any]],
        constraint_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply standards compliance rules.
        
        Rules:
        - SIL rating required → EXCLUDE non-compliant
        - ATEX zone required → EXCLUDE non-rated
        - Certifications required → FILTER
        """
        required_sil = constraint_context.get("required_sil_rating")
        required_atex = constraint_context.get("atex_zone")
        required_certs = constraint_context.get("required_certifications", [])
        
        # Note: In a real implementation, this would check against actual product data
        # For now, we pass through all candidates with a placeholder for standards check
        
        filtered = []
        excluded = []
        
        for candidate in candidates:
            # Mark for standards verification
            candidate["standards_check_required"] = {
                "sil_rating": required_sil,
                "atex_zone": required_atex,
                "certifications": required_certs
            }
            candidate["meets_standards"] = True  # Placeholder - actual check in vendor analysis
            filtered.append(candidate)
        
        logger.info(f"Standards filter: {len(filtered)} candidates marked for verification")
        return {
            "filtered_candidates": filtered,
            "excluded_candidates": excluded,
            "verification_required": bool(required_sil or required_atex or required_certs)
        }
    
    def apply_all_rules(
        self, 
        vendors: List[str],
        constraint_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply all filtering rules in sequence.
        """
        all_excluded = []
        
        # Step 1: Strategy rules
        strategy_result = self.apply_strategy_rules(vendors, constraint_context)
        all_excluded.extend(strategy_result["excluded_vendors"])
        
        # Convert to candidate format
        candidates = []
        for v in strategy_result["filtered_vendors"]:
            candidates.append({
                "vendor": v["vendor"],
                "series": "",  # To be populated later
                "model": "",   # To be populated later
                "is_preferred": v["is_preferred"],
                "priority_boost": v["priority_boost"]
            })
        
        # Step 2: Installed-base rules (if candidates have series/model info)
        # This step is more meaningful after vendor analysis provides model data
        
        # Step 3: Standards rules
        standards_result = self.apply_standards_rules(candidates, constraint_context)
        all_excluded.extend(standards_result.get("excluded_candidates", []))
        
        return {
            "filtered_candidates": standards_result["filtered_candidates"],
            "excluded": all_excluded,
            "total_filtered": len(standards_result["filtered_candidates"]),
            "total_excluded": len(all_excluded)
        }


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_rag_aggregator() -> RAGAggregator:
    """Create a new RAG aggregator instance"""
    return RAGAggregator()


def create_strategy_filter() -> StrategyFilter:
    """Create a new strategy filter instance"""
    return StrategyFilter()
