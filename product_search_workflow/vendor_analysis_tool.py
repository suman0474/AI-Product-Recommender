"""
Vendor Analysis Tool for Product Search Workflow
=================================================

Step 4 of Product Search Workflow:
- Loads vendors matching product type
- **APPLIES STRATEGY RAG** to filter/prioritize vendors before analysis
- Retrieves PDF datasheets and JSON product catalogs for approved vendors
- Runs parallel vendor analysis using get_vendor_prompt
- Returns matched products with detailed analysis + strategy context

STRATEGY RAG INTEGRATION (Step 2.5):
- Retrieves strategic context from Pinecone vector store (TRUE RAG)
- Falls back to LLM inference if vector store is empty
- Filters out FORBIDDEN vendors defined in strategy documents
- Prioritizes PREFERRED vendors for analysis
- Enriches final matches with strategy_priority scores

Strategic context includes:
- Cost optimization priorities
- Sustainability requirements
- Compliance alignment
- Supplier reliability metrics
- Long-term partnership alignment

This tool integrates:
- Vendor loading from MongoDB/Azure
- Strategy RAG for vendor filtering (NEW)
- PDF content extraction
- JSON product catalog loading
- LLM-powered vendor analysis
"""

import json
import logging
import os
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Configure logging
logger = logging.getLogger(__name__)


class VendorAnalysisTool:
    """
    Vendor Analysis Tool - Step 4 of Product Search Workflow

    Responsibilities:
    1. Load vendors matching the detected product type
    2. Retrieve vendor documentation (PDFs and JSON)
    3. Run parallel vendor analysis
    4. Return matched products with detailed analysis
    """

    def __init__(self, max_workers: int = 5, max_retries: int = 3):
        """
        Initialize the vendor analysis tool.

        Args:
            max_workers: Maximum parallel workers for vendor analysis
            max_retries: Maximum retries for rate-limited requests
        """
        self.max_workers = max_workers
        self.max_retries = max_retries
        logger.info("[VendorAnalysisTool] Initialized with max_workers=%d", max_workers)

    def analyze(
        self,
        structured_requirements: Dict[str, Any],
        product_type: str,
        session_id: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze vendors for matching products with Strategy RAG filtering.
        
        WORKFLOW:
        1. Load vendors for product type
        2. Apply Strategy RAG to filter/prioritize vendors based on:
           - Cost optimization priorities
           - Sustainability requirements
           - Compliance alignment
           - Supplier reliability
           - Long-term partnership alignment
        3. Load product data for strategically-approved vendors only
        4. Run parallel vendor analysis
        5. Return matches enriched with strategy context

        Args:
            structured_requirements: Collected user requirements
            product_type: Detected product type
            session_id: Session tracking ID
            schema: Optional product schema

        Returns:
            {
                'success': bool,
                'product_type': str,
                'vendor_matches': list,  # Enriched with strategy_priority and is_preferred_vendor
                'vendor_run_details': list,
                'total_matches': int,
                'vendors_analyzed': int,
                'original_vendor_count': int,  # Before strategy filtering
                'filtered_vendor_count': int,  # After strategy filtering
                'excluded_by_strategy': int,   # Count of excluded vendors
                'strategy_context': {          # Strategy RAG results
                    'applied': bool,
                    'rag_type': str,           # 'true_rag' or 'llm_inference'
                    'preferred_vendors': list,
                    'forbidden_vendors': list,
                    'excluded_vendors': list,
                    'vendor_priorities': dict,
                    'confidence': float,
                    'strategy_notes': str,
                    'sources_used': list
                },
                'analysis_summary': str
            }
        """
        logger.info("[VendorAnalysisTool] Starting vendor analysis")
        logger.info("[VendorAnalysisTool] Product type: %s", product_type)
        logger.info("[VendorAnalysisTool] Session: %s", session_id or "N/A")

        result = {
            "success": False,
            "product_type": product_type,
            "session_id": session_id
        }

        try:
            # Import required modules
            from chaining import (
                setup_langchain_components,
                invoke_vendor_chain,
                to_dict_if_pydantic
            )
            from azure_blob_utils import (
                get_vendors_for_product_type,
                get_products_for_vendors
            )
            import json

            # Step 1: Setup LLM components
            logger.info("[VendorAnalysisTool] Step 1: Setting up LLM components")
            components = setup_langchain_components()

            # Step 2: Load vendors directly from Azure Blob Storage
            logger.info("[VendorAnalysisTool] Step 2: Loading vendors for product type: '%s'", product_type)
            
            vendors = get_vendors_for_product_type(product_type) if product_type else []
            
            logger.info("[VendorAnalysisTool] ===== DIAGNOSTIC: VENDOR LOADING =====")
            logger.info("[VendorAnalysisTool] Product type: '%s'", product_type)
            logger.info("[VendorAnalysisTool] Found vendors: %d - %s", len(vendors), vendors[:5] if vendors else [])

            if not vendors:
                logger.warning("[VendorAnalysisTool] NO VENDORS FOUND - Returning empty result")
                logger.warning("[VendorAnalysisTool] This could mean: 1) No data in Azure Blob, 2) Product type doesn't match metadata")
                result['success'] = True
                result['vendor_matches'] = []
                result['vendor_run_details'] = []
                result['total_matches'] = 0
                result['vendors_analyzed'] = 0
                result['analysis_summary'] = "No vendors available for analysis"
                return result

            # =================================================================
            # STEP 2.5: STRATEGY RAG NODE - Filter/Prioritize Vendors
            # Apply strategic context BEFORE analysis:
            # - Cost optimization priorities
            # - Sustainability requirements
            # - Compliance alignment
            # - Supplier reliability
            # - Long-term partnership alignment
            # =================================================================
            
            # ╔══════════════════════════════════════════════════════════════╗
            # ║           STRATEGY RAG INVOCATION STARTING             ║
            # ╚══════════════════════════════════════════════════════════════╝
            import datetime
            strategy_rag_invocation_time = datetime.datetime.now().isoformat()
            strategy_rag_invoked = False
            
            logger.info("="*70)
            logger.info("STRATEGY RAG INVOKED")
            logger.info(f"   Timestamp: {strategy_rag_invocation_time}")
            logger.info(f"   Product Type: {product_type}")
            logger.info(f"   Vendors to Filter: {len(vendors)}")
            logger.info(f"   Session: {session_id}")
            logger.info("="*70)
            logger.info("="*70)
            logger.info("[STRATEGY RAG] INVOCATION STARTED")
            logger.info(f"   Time: {strategy_rag_invocation_time}")
            logger.info(f"   Product: {product_type}")
            logger.info(f"   Vendors: {len(vendors)}")
            logger.info("="*70)
            
            logger.info("[VendorAnalysisTool] Step 2.5: Applying Strategy RAG to filter/prioritize vendors")
            
            strategy_context = None
            filtered_vendors = vendors.copy()
            excluded_vendors = []
            vendor_priorities = {}
            
            try:
                strategy_rag_invoked = True
                # FIX: Import from correct path (strategy_rag subdirectory, not stub)
                from agentic.strategy_rag.strategy_rag_enrichment import (
                    get_strategy_with_auto_fallback,
                    filter_vendors_by_strategy
                )
                
                # Get strategy data from TRUE RAG (or fallback to LLM)
                strategy_context = get_strategy_with_auto_fallback(
                    product_type=product_type,
                    requirements=structured_requirements,
                    top_k=7
                )
                
                if strategy_context.get('success'):
                    rag_type = strategy_context.get('rag_type', 'unknown')
                    preferred = strategy_context.get('preferred_vendors', [])
                    forbidden = strategy_context.get('forbidden_vendors', [])
                    priorities = strategy_context.get('procurement_priorities', {})
                    strategy_notes = strategy_context.get('strategy_notes', '')
                    confidence = strategy_context.get('confidence', 0.0)
                    
                    logger.info("="*70)
                    logger.info(f"STRATEGY RAG APPLIED SUCCESSFULLY ({rag_type})")
                    logger.info(f"   Preferred vendors: {preferred}")
                    logger.info(f"   Forbidden vendors: {forbidden}")
                    logger.info(f"   Confidence: {confidence:.2f}")
                    logger.info("="*70)
                    logger.info("="*70)
                    logger.info(f"[STRATEGY RAG] COMPLETED - Type: {rag_type}")
                    logger.info(f"   Preferred: {preferred}")
                    logger.info(f"   Forbidden: {forbidden}")
                    logger.info(f"   Confidence: {confidence:.2f}")
                    logger.info("="*70)
                    
                    logger.info(f"[VendorAnalysisTool] Priorities: {priorities}")
                    logger.info(f"[VendorAnalysisTool] Strategy notes: {strategy_notes[:200] if strategy_notes else 'None'}...")
                    
                    # Apply vendor filtering
                    filter_result = filter_vendors_by_strategy(vendors, strategy_context)
                    
                    # Extract filtered vendors (prioritized order)
                    accepted = filter_result.get('accepted_vendors', [])
                    excluded_vendors = filter_result.get('excluded_vendors', [])
                    
                    if accepted:
                        # Use filtered, prioritized vendor list
                        filtered_vendors = [v['vendor'] for v in accepted]
                        vendor_priorities = {v['vendor']: v.get('priority_score', 0) for v in accepted}
                        
                        logger.info(f"[VendorAnalysisTool] Strategy filtering: {len(filtered_vendors)} accepted, {len(excluded_vendors)} excluded")
                        logger.info(f"[VendorAnalysisTool] Prioritized vendor order: {filtered_vendors[:5]}...")
                        
                        # Log excluded vendors
                        for ex in excluded_vendors:
                            logger.info(f"[VendorAnalysisTool] Excluded: {ex['vendor']} - {ex['reason']}")
                    else:
                        logger.warning("[VendorAnalysisTool] No vendors passed strategy filter - using original list")
                        logger.warning("="*70)
                        logger.warning("[STRATEGY RAG] No vendors passed filter")
                        logger.warning("="*70)
                        filtered_vendors = vendors
                else:
                    logger.warning(f"[VendorAnalysisTool] Strategy RAG returned no results: {strategy_context.get('error', 'Unknown')}")
                    logger.warning("="*70)
                    logger.warning(f"[STRATEGY RAG] NO RESULTS: {strategy_context.get('error', 'Unknown')}")
                    logger.warning("="*70)
                    
            except Exception as strategy_error:
                logger.warning(f"[VendorAnalysisTool] ⚠ Strategy RAG failed (proceeding without): {strategy_error}")
                logger.error("="*70)
                logger.error(f"[STRATEGY RAG] ERROR: {strategy_error}")
                logger.error("="*70)
                filtered_vendors = vendors
            
            # Store strategy context in result for downstream use
            # ══════════════════════════════════════════════════════════════
            # RAG INVOCATION TRACKING - Visible in browser Network tab
            # ══════════════════════════════════════════════════════════════
            result['rag_invocations'] = {
                'strategy_rag': {
                    'invoked': strategy_rag_invoked,
                    'invocation_time': strategy_rag_invocation_time,
                    'success': strategy_context is not None and strategy_context.get('success', False),
                    'rag_type': strategy_context.get('rag_type') if strategy_context else None,
                    'product_type': product_type,
                    'vendors_before_filter': len(vendors),
                    'vendors_after_filter': len(filtered_vendors),
                    'excluded_count': len(excluded_vendors)
                },
                'standards_rag': {
                    'invoked': False,
                    'note': 'Standards RAG is applied in validation_tool.py, not during vendor analysis'
                }
            }
            
            result['strategy_context'] = {
                'applied': strategy_context is not None and strategy_context.get('success', False),
                'rag_type': strategy_context.get('rag_type') if strategy_context else None,
                'preferred_vendors': strategy_context.get('preferred_vendors', []) if strategy_context else [],
                'forbidden_vendors': strategy_context.get('forbidden_vendors', []) if strategy_context else [],
                'excluded_vendors': excluded_vendors,
                'vendor_priorities': vendor_priorities,
                'confidence': strategy_context.get('confidence', 0.0) if strategy_context else 0.0,
                'strategy_notes': strategy_context.get('strategy_notes', '') if strategy_context else '',
                'sources_used': strategy_context.get('sources_used', []) if strategy_context else []
            }
            
            # Use filtered vendors for product loading
            vendors_for_analysis = filtered_vendors
            logger.info(f"[VendorAnalysisTool] Proceeding with {len(vendors_for_analysis)} strategically-filtered vendors")


            # Step 3: Load product data for STRATEGICALLY-FILTERED vendors
            logger.info("[VendorAnalysisTool] Step 3: Loading product data from Azure Blob (filtered vendors only)")
            
            # Use vendors_for_analysis which has been filtered by Strategy RAG
            products_data = get_products_for_vendors(vendors_for_analysis, product_type)
            
            logger.info("[VendorAnalysisTool] ===== DIAGNOSTIC: PRODUCT DATA (STRATEGY-FILTERED) =====")
            logger.info("[VendorAnalysisTool] Products loaded for %d strategically-approved vendors", len(products_data))
            for v, products in list(products_data.items())[:3]:
                product_count = len(products) if products else 0
                priority = vendor_priorities.get(v, 0)
                logger.info("[VendorAnalysisTool]   - %s: %d product entries (priority: %d)", v, product_count, priority)

            if not products_data:
                logger.warning("[VendorAnalysisTool] NO PRODUCT DATA LOADED")
                result['success'] = True
                result['vendor_matches'] = []
                result['vendor_run_details'] = []
                result['total_matches'] = 0
                result['vendors_analyzed'] = 0
                result['analysis_summary'] = "No product data available for analysis"
                return result

            # Step 4: Prepare structured requirements string
            logger.info("[VendorAnalysisTool] Step 4: Preparing requirements")
            requirements_str = self._format_requirements(structured_requirements)
            logger.info("[VendorAnalysisTool] Requirements: %s...", requirements_str[:200] if requirements_str else "EMPTY")

            # Step 5: Prepare vendor payloads (product JSON as content, no PDF required)
            logger.info("[VendorAnalysisTool] Step 5: Preparing vendor payloads")
            vendor_payloads = {}
            
            for vendor_name, products in products_data.items():
                if products:
                    # Convert products to JSON string for the prompt
                    products_json = json.dumps(products, indent=2, ensure_ascii=False)
                    vendor_payloads[vendor_name] = {
                        "products": products,
                        "pdf_text": products_json  # Use product JSON as the analysis content
                    }
                    
            logger.info("[VendorAnalysisTool] ===== DIAGNOSTIC: VENDOR PAYLOADS =====")
            logger.info("[VendorAnalysisTool] Total payloads prepared: %d", len(vendor_payloads))
            for v, data in list(vendor_payloads.items())[:5]:
                content_len = len(data.get('pdf_text', '')) if data.get('pdf_text') else 0
                products_list = data.get('products', [])
                products_count = len(products_list)
                
                # Detailed structure analysis
                models_count = 0
                submodels_count = 0
                specs_count = 0
                
                if products_list and isinstance(products_list[0], dict):
                    first_product = products_list[0]
                    if 'models' in first_product:
                        models = first_product['models']
                        models_count = len(models) if models else 0
                        for model in (models or []):
                            if 'sub_models' in model:
                                submodels_count += len(model['sub_models'])
                                for sub in model['sub_models']:
                                    if 'specifications' in sub:
                                        specs_count += len(sub.get('specifications', {}))
                
                logger.info("[VendorAnalysisTool]   - %s: Content=%d chars, Products=%d, Models=%d, Submodels=%d, Specs=%d", 
                           v, content_len, products_count, models_count, submodels_count, specs_count)

            if not vendor_payloads:
                logger.warning("[VendorAnalysisTool] NO VALID VENDOR PAYLOADS")
                result['success'] = True
                result['vendor_matches'] = []
                result['vendor_run_details'] = []
                result['total_matches'] = 0
                result['vendors_analyzed'] = 0
                result['analysis_summary'] = "No vendor data available for analysis"
                return result

            # Step 6: Run parallel vendor analysis
            logger.info("[VendorAnalysisTool] Step 6: Running parallel vendor analysis")
            vendor_matches = []
            run_details = []

            # Run parallel analysis
            actual_workers = min(len(vendor_payloads), self.max_workers)
            logger.info("[VendorAnalysisTool] Using %d workers for %d vendors",
                       actual_workers, len(vendor_payloads))

            with ThreadPoolExecutor(max_workers=actual_workers) as executor:
                futures = {}
                for i, (vendor, data) in enumerate(vendor_payloads.items()):
                    # Stagger submissions to avoid rate limiting
                    if i > 0:
                        time.sleep(5)

                    future = executor.submit(
                        self._analyze_vendor,
                        components,
                        requirements_str,
                        vendor,
                        data
                    )
                    futures[future] = vendor

                for future in as_completed(futures):
                    vendor = futures[future]
                    try:
                        vendor_result, error = future.result()

                        if vendor_result and isinstance(vendor_result.get("vendor_matches"), list):
                            for match in vendor_result["vendor_matches"]:
                                match_dict = to_dict_if_pydantic(match)
                                # Normalize to camelCase for frontend compatibility
                                normalized_match = {
                                    'productName': match_dict.get('product_name', match_dict.get('productName', '')),
                                    'vendor': vendor,
                                    'modelFamily': match_dict.get('model_family', match_dict.get('modelFamily', '')),
                                    'productType': match_dict.get('product_type', match_dict.get('productType', '')),
                                    'matchScore': match_dict.get('match_score', match_dict.get('matchScore', 0)),
                                    'requirementsMatch': match_dict.get('requirements_match', match_dict.get('requirementsMatch', False)),
                                    'reasoning': match_dict.get('reasoning', ''),
                                    'limitations': match_dict.get('limitations', '')
                                }
                                vendor_matches.append(normalized_match)
                                logger.info(f"[VendorAnalysisTool] DEBUG: Normalized match for {vendor}: {normalized_match}")
                            run_details.append({"vendor": vendor, "status": "success"})
                            logger.info("[VendorAnalysisTool] Vendor '%s' returned %d matches",
                                       vendor, len(vendor_result["vendor_matches"]))
                        else:
                            run_details.append({
                                "vendor": vendor,
                                "status": "failed" if error else "empty",
                                "error": error
                            })
                            logger.warning("[VendorAnalysisTool] Vendor '%s' failed: %s", vendor, error)

                    except Exception as e:
                        logger.error("[VendorAnalysisTool] Vendor '%s' exception: %s", vendor, str(e))
                        run_details.append({
                            "vendor": vendor,
                            "status": "error",
                            "error": str(e)
                        })

            # Build result with strategy context
            result['success'] = True
            result['vendor_matches'] = vendor_matches
            result['vendor_run_details'] = run_details
            result['total_matches'] = len(vendor_matches)
            result['vendors_analyzed'] = len(vendor_payloads)
            result['original_vendor_count'] = len(vendors)  # Before filtering
            result['filtered_vendor_count'] = len(vendors_for_analysis)  # After strategy filtering
            result['excluded_by_strategy'] = len(excluded_vendors)  # Vendors excluded by strategy

            # Enrich matches with strategy priority scores
            for match in vendor_matches:
                vendor_name = match.get('vendor', '')
                match['strategy_priority'] = vendor_priorities.get(vendor_name, 0)
                match['is_preferred_vendor'] = vendor_name.lower() in [
                    p.lower() for p in (strategy_context.get('preferred_vendors', []) if strategy_context else [])
                ]
                
                # Ensure requirementsMatch is set (Frontend relies on this)
                if 'requirementsMatch' not in match:
                    score = match.get('matchScore', 0)
                    match['requirementsMatch'] = score >= 80


            # Generate enhanced summary with strategy info
            successful = sum(1 for d in run_details if d['status'] == 'success')
            strategy_source = result.get('strategy_context', {}).get('rag_type', 'none')
            
            result['analysis_summary'] = (
                f"Strategy RAG ({strategy_source}): Filtered {len(vendors)} → {len(vendors_for_analysis)} vendors | "
                f"Analyzed {len(vendor_payloads)} vendors, {successful} successful | "
                f"Found {len(vendor_matches)} matching products"
            )

            logger.info("[VendorAnalysisTool] ===== ANALYSIS COMPLETE =====")
            logger.info("[VendorAnalysisTool] %s", result['analysis_summary'])
            if excluded_vendors:
                logger.info("[VendorAnalysisTool] Excluded by strategy: %s", [e['vendor'] for e in excluded_vendors])

            return result

        except Exception as e:
            logger.error("[VendorAnalysisTool] Analysis failed: %s", str(e), exc_info=True)
            result['success'] = False
            result['error'] = str(e)
            result['error_type'] = type(e).__name__
            return result

    def _format_requirements(self, requirements: Dict[str, Any]) -> str:
        """Format requirements dictionary into structured string."""
        lines = []

        # Handle nested structure
        if 'mandatoryRequirements' in requirements or 'mandatory' in requirements:
            mandatory = requirements.get('mandatoryRequirements') or requirements.get('mandatory', {})
            if mandatory:
                lines.append("## Mandatory Requirements")
                for key, value in mandatory.items():
                    if value:
                        lines.append(f"- {self._format_field_name(key)}: {value}")

        if 'optionalRequirements' in requirements or 'optional' in requirements:
            optional = requirements.get('optionalRequirements') or requirements.get('optional', {})
            if optional:
                lines.append("\n## Optional Requirements")
                for key, value in optional.items():
                    if value:
                        lines.append(f"- {self._format_field_name(key)}: {value}")

        if 'selectedAdvancedParams' in requirements or 'advancedSpecs' in requirements:
            advanced = requirements.get('selectedAdvancedParams') or requirements.get('advancedSpecs', {})
            if advanced:
                lines.append("\n## Advanced Specifications")
                for key, value in advanced.items():
                    if value:
                        lines.append(f"- {key}: {value}")

        # FIX: Return structured guidance that instructs LLM to still return JSON
        if not lines:
            return """## Requirements Summary
No specific mandatory or optional requirements have been provided for this product search.

## Analysis Instruction
Analyze available products and return JSON with general recommendations based on:
- Standard industrial specifications and certifications
- Product feature completeness and quality
- Typical use case suitability for this product type
- Provide match_score based on product quality (use 85-95 range for well-documented, certified products)"""
        
        return "\n".join(lines)

    def _format_field_name(self, field: str) -> str:
        """Convert camelCase or snake_case to Title Case."""
        # Replace underscores and split on capital letters
        import re
        words = re.sub(r'([a-z])([A-Z])', r'\1 \2', field)
        words = words.replace('_', ' ')
        return words.title()

    def _prepare_payloads(
        self,
        vendors: List[str],
        pdf_content: Dict[str, str],
        products_json: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Prepare vendor payloads for analysis."""
        payloads = {}

        for vendor in vendors:
            pdf_text = pdf_content.get(vendor, "")
            products = products_json.get(vendor, [])

            # Only include vendors with some data
            if pdf_text or products:
                payloads[vendor] = {
                    "pdf_text": pdf_text,
                    "products": products
                }
                logger.debug("[VendorAnalysisTool] Prepared payload for '%s': PDF=%s, Products=%d",
                           vendor, bool(pdf_text), len(products) if products else 0)

        return payloads

    def _analyze_vendor(
        self,
        components: Dict[str, Any],
        requirements_str: str,
        vendor: str,
        vendor_data: Dict[str, Any]
    ) -> tuple:
        """Analyze a single vendor."""
        error = None
        result = None
        base_retry_delay = 15

        logger.info("[VendorAnalysisTool] START analysis for vendor: %s", vendor)

        for attempt in range(self.max_retries):
            try:
                from chaining import invoke_vendor_chain

                pdf_text = vendor_data.get("pdf_text", "")
                products = vendor_data.get("products", [])

                pdf_payload = json.dumps({vendor: pdf_text}, ensure_ascii=False) if pdf_text else "{}"
                products_payload = json.dumps(products, ensure_ascii=False)

                result = invoke_vendor_chain(
                    components,
                    requirements_str,
                    products_payload,
                    pdf_payload,
                    components['vendor_format_instructions']
                )

                # Convert to dict if needed and parse for robust handling
                from chaining import to_dict_if_pydantic, parse_vendor_analysis_response
                result = to_dict_if_pydantic(result)
                
                # FIX: Handle malformed responses (missing vendor_matches wrapper)
                result = parse_vendor_analysis_response(result, vendor)

                logger.info("[VendorAnalysisTool] END analysis for vendor: %s (success)", vendor)
                return result, None

            except Exception as e:
                error_msg = str(e)
                is_rate_limit = any(x in error_msg.lower() for x in ['429', 'resource has been exhausted', 'quota', '503', 'overloaded'])

                if is_rate_limit and attempt < self.max_retries - 1:
                    wait_time = base_retry_delay * (2 ** attempt)
                    logger.warning("[VendorAnalysisTool] Rate limit for %s, retry %d/%d after %ds",
                                  vendor, attempt + 1, self.max_retries, wait_time)
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error("[VendorAnalysisTool] Analysis failed for %s: %s", vendor, error_msg)
                    error = error_msg
                    break

        logger.info("[VendorAnalysisTool] END analysis for vendor: %s (error: %s)", vendor, error)
        return None, error


# Convenience function
def analyze_vendors(
    structured_requirements: Dict[str, Any],
    product_type: str,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to run vendor analysis.

    Args:
        structured_requirements: Collected user requirements
        product_type: Detected product type
        session_id: Session tracking ID

    Returns:
        Vendor analysis result
    """
    tool = VendorAnalysisTool()
    return tool.analyze(
        structured_requirements=structured_requirements,
        product_type=product_type,
        session_id=session_id
    )
