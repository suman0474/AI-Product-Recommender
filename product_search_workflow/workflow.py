"""
Product Search Workflow Orchestrator
=====================================

Complete orchestration of the Product Search Workflow integrating:
- Step 1: Validation Tool (product type detection & schema generation)
- Step 2: Advanced Parameters Discovery Tool (latest specifications from vendors)
- Step 3: Sales Agent Tool (step-by-step requirements collection)
- Step 4: Vendor Analysis Tool (parallel vendor matching & product analysis)
- Step 5: Ranking Tool (final product ranking with scores)

Accepts input from:
- Instruments Identifier Workflow (standalone)
- Solution Workflow (standalone)
- Direct user input

UI Integration:
- LeftSidebar: Displays schema (mandatory/optional requirements) with field descriptions
- RightPanel: Displays ranked products with images, pricing, feedback section

API Integration:
- /api/agentic/validate - Validation Tool
- /api/agentic/advanced-parameters - Advanced Parameters Tool
- /api/agentic/sales-agent - Sales Agent Tool
- /api/sales-agent - Main workflow API (main.py)
- /api/feedback - Feedback collection

Version: 2.0.0
"""

import json
import uuid
import logging
from typing import Dict, Any, List, Optional

from .validation_tool import ValidationTool
from .advanced_parameters_tool import AdvancedParametersTool
from .sales_agent_tool import SalesAgentTool, WorkflowStep
from .vendor_analysis_tool import VendorAnalysisTool
from .ranking_tool import RankingTool

# Configure logging
logger = logging.getLogger(__name__)


class ProductSearchWorkflow:
    """
    Complete Product Search Agentic Workflow

    Orchestrates the complete product search process from requirements
    validation through final product ranking.

    Input Sources:
    - Instruments Identifier Workflow (standalone)
    - Solution Workflow (standalone)
    - Direct user input

    Workflow Steps (Complete Pipeline):
    1. Validation Tool - Product type detection & schema generation (with PPI)
    2. Advanced Parameters Tool - Discover latest specifications from vendors
    3. Sales Agent Tool - Step-by-step requirements collection
    4. Vendor Analysis Tool - Parallel vendor matching & product analysis
    5. Ranking Tool - Final product ranking with scores

    UI Data Flow:
    - LeftSidebar receives: schema (mandatoryRequirements, optionalRequirements)
    - RightPanel receives: analysisResult (vendorAnalysis, overallRanking, pricing, images)
    """

    def __init__(
        self,
        enable_ppi_workflow: bool = True,
        auto_mode: bool = False,
        max_vendor_workers: int = 5,
        llm=None
    ):
        """
        Initialize the product search workflow.

        Args:
            enable_ppi_workflow: Enable PPI workflow for schema generation
            auto_mode: Auto-respond to sales agent prompts (for automation)
            max_vendor_workers: Maximum parallel workers for vendor analysis
            llm: LLM instance for sales agent (optional)
        """
        logger.info("[ProductSearchWorkflow] Initializing workflow v2.0...")

        self.enable_ppi_workflow = enable_ppi_workflow
        self.auto_mode = auto_mode
        self.max_vendor_workers = max_vendor_workers

        # Initialize all tools
        self.validation_tool = ValidationTool(enable_ppi=enable_ppi_workflow)
        self.advanced_params_tool = AdvancedParametersTool()
        self.sales_agent_tool = SalesAgentTool(llm=llm)
        self.vendor_analysis_tool = VendorAnalysisTool(max_workers=max_vendor_workers)
        self.ranking_tool = RankingTool()

        logger.info("[ProductSearchWorkflow] Workflow ready with all tools")
        logger.info("[ProductSearchWorkflow]   - PPI Workflow: %s",
                   "enabled" if enable_ppi_workflow else "disabled")
        logger.info("[ProductSearchWorkflow]   - Auto Mode: %s",
                   "enabled" if auto_mode else "disabled")
        logger.info("[ProductSearchWorkflow]   - Vendor Workers: %d",
                   max_vendor_workers)

    def run_single_product_workflow(
        self,
        user_input: str,
        expected_product_type: Optional[str] = None,
        session_id: Optional[str] = None,
        skip_advanced_params: bool = False
    ) -> Dict[str, Any]:
        """
        Run complete workflow for a single product.

        Args:
            user_input: User's requirement description
            expected_product_type: Expected product type (optional)
            session_id: Session identifier (auto-generated if not provided)
            skip_advanced_params: Skip advanced parameters discovery

        Returns:
            Complete workflow result with all collected data
        """
        # Generate session ID if not provided
        if not session_id:
            session_id = f"workflow_{uuid.uuid4().hex[:8]}"

        logger.info(f"\n{'='*70}")
        logger.info(f"[ProductSearchWorkflow] Starting workflow")
        logger.info(f"[ProductSearchWorkflow] Session: {session_id}")
        logger.info(f"[ProductSearchWorkflow] Input: {user_input[:100]}...")
        logger.info(f"{'='*70}")

        result = {
            "session_id": session_id,
            "success": False,
            "steps_completed": []
        }

        try:
            # =================================================================
            # STEP 1: VALIDATION TOOL
            # - Detects product type
            # - Loads or generates schema (with PPI workflow if needed)
            # - Validates requirements against schema
            # =================================================================
            logger.info(f"\n[STEP 1/5] VALIDATION TOOL")
            logger.info("-" * 70)

            validation_result = self.validation_tool.validate(
                user_input=user_input,
                expected_product_type=expected_product_type,
                session_id=session_id
            )

            if not validation_result['success']:
                logger.error("[ProductSearchWorkflow] ✗ Validation failed")
                result['error'] = validation_result.get('error', 'Validation failed')
                return result

            product_type = validation_result['product_type']
            schema = validation_result['schema']
            is_valid = validation_result['is_valid']
            missing_fields = validation_result['missing_fields']
            ppi_used = validation_result['ppi_workflow_used']

            logger.info(f"✓ Product Type: {product_type}")
            logger.info(f"✓ Schema: {'Generated via PPI' if ppi_used else 'Loaded from DB'}")
            logger.info(f"✓ Validation: {'All fields provided' if is_valid else f'{len(missing_fields)} fields missing'}")

            result['steps_completed'].append('validation')
            result['validation_result'] = {
                "productType": product_type,
                "detectedSchema": schema,
                "providedRequirements": validation_result['provided_requirements'],
                "ppiWorkflowUsed": ppi_used,
                "schemaSource": validation_result['schema_source'],
                "isValid": is_valid,
                "missingFields": missing_fields
            }

            # =================================================================
            # STEP 2: ADVANCED PARAMETERS DISCOVERY
            # Discovers latest advanced specifications from top vendors
            # =================================================================
            if not skip_advanced_params:
                logger.info(f"\n[STEP 2/5] ADVANCED PARAMETERS DISCOVERY")
                logger.info("-" * 70)

                advanced_params_result = self.advanced_params_tool.discover(
                    product_type=product_type,
                    session_id=session_id,
                    existing_schema=schema
                )

                unique_specs = advanced_params_result.get('unique_specifications', [])
                specs_count = len(unique_specs)

                if specs_count > 0:
                    logger.info(f"✓ Discovered {specs_count} new advanced specifications")
                    for spec in unique_specs[:5]:  # Log first 5
                        logger.info(f"  - {spec.get('name', spec.get('key'))}")
                    if specs_count > 5:
                        logger.info(f"  ... and {specs_count - 5} more")
                else:
                    logger.info("ℹ No new advanced specifications found")

                result['steps_completed'].append('advanced_parameters')
                result['advanced_parameters_result'] = {
                    "discovered_specifications": unique_specs,
                    "total_discovered": specs_count,
                    "existing_filtered": advanced_params_result.get('existing_specifications_filtered', 0),
                    "vendors_searched": advanced_params_result.get('vendors_searched', []),
                    "success": advanced_params_result.get('success', True)
                }
            else:
                logger.info(f"\n[STEP 2/5] ADVANCED PARAMETERS DISCOVERY - SKIPPED")
                result['advanced_parameters_result'] = {
                    "discovered_specifications": [],
                    "total_discovered": 0,
                    "skipped": True
                }

            # =================================================================
            # STEP 3: SALES AGENT (simplified for auto mode)
            # Prepares structured requirements for vendor analysis
            # =================================================================
            logger.info(f"\n[STEP 3/5] SALES AGENT TOOL")
            logger.info("-" * 70)

            unique_specs = result.get('advanced_parameters_result', {}).get('discovered_specifications', [])
            provided_reqs = validation_result['provided_requirements']

            # Build structured requirements for vendor analysis
            structured_requirements = {
                "productType": product_type,
                "mandatoryRequirements": provided_reqs.get('mandatory', {}),
                "optionalRequirements": provided_reqs.get('optional', {}),
                "selectedAdvancedParams": {
                    spec.get('key'): "" for spec in unique_specs
                } if unique_specs else {}
            }

            if self.auto_mode:
                logger.info("[SalesAgent] Auto mode - using validated requirements")
            else:
                logger.info("[SalesAgent] Requirements prepared for interactive collection")

            result['steps_completed'].append('sales_agent')
            result['sales_agent_result'] = {
                "mode": "auto" if self.auto_mode else "interactive",
                "structured_requirements": structured_requirements,
                "final_step": "complete"
            }

            # =================================================================
            # STEP 4: VENDOR ANALYSIS
            # Parallel analysis of vendors with PDF/JSON matching
            # =================================================================
            logger.info(f"\n[STEP 4/5] VENDOR ANALYSIS TOOL")
            logger.info("-" * 70)

            vendor_analysis_result = self.vendor_analysis_tool.analyze(
                structured_requirements=structured_requirements,
                product_type=product_type,
                session_id=session_id,
                schema=schema
            )

            if not vendor_analysis_result.get('success'):
                logger.warning("[ProductSearchWorkflow] Vendor analysis failed but continuing")
                vendor_analysis_result = {
                    'success': True,
                    'vendor_matches': [],
                    'total_matches': 0,
                    'analysis_summary': 'No vendor analysis available'
                }

            vendor_matches = vendor_analysis_result.get('vendor_matches', [])
            logger.info(f"✓ Vendors Analyzed: {vendor_analysis_result.get('vendors_analyzed', 0)}")
            logger.info(f"✓ Products Matched: {len(vendor_matches)}")

            result['steps_completed'].append('vendor_analysis')
            result['vendor_analysis_result'] = {
                "vendorMatches": vendor_matches,
                "vendorRunDetails": vendor_analysis_result.get('vendor_run_details', []),
                "totalMatches": len(vendor_matches),
                "vendorsAnalyzed": vendor_analysis_result.get('vendors_analyzed', 0),
                "analysisSummary": vendor_analysis_result.get('analysis_summary', '')
            }

            # =================================================================
            # STEP 5: RANKING
            # Final product ranking with scores and recommendations
            # =================================================================
            logger.info(f"\n[STEP 5/5] RANKING TOOL")
            logger.info("-" * 70)

            ranking_result = self.ranking_tool.rank(
                vendor_analysis=vendor_analysis_result,
                session_id=session_id
            )

            if not ranking_result.get('success'):
                logger.warning("[ProductSearchWorkflow] Ranking failed but continuing")
                ranking_result = {
                    'success': True,
                    'overall_ranking': [],
                    'top_product': None,
                    'total_ranked': 0,
                    'ranking_summary': 'No ranking available'
                }

            overall_ranking = ranking_result.get('overall_ranking', [])
            top_product = ranking_result.get('top_product')

            logger.info(f"✓ Products Ranked: {len(overall_ranking)}")
            if top_product:
                top_name = top_product.get('product_name', top_product.get('productName', 'Unknown'))
                top_vendor = top_product.get('vendor', 'Unknown')
                logger.info(f"✓ Top Recommendation: {top_name} by {top_vendor}")

            result['steps_completed'].append('ranking')
            result['ranking_result'] = {
                "overallRanking": {
                    "rankedProducts": overall_ranking
                },
                "topProduct": top_product,
                "totalRanked": len(overall_ranking),
                "rankingSummary": ranking_result.get('ranking_summary', '')
            }

            # =================================================================
            # WORKFLOW COMPLETE - Build Final Result
            # =================================================================
            logger.info(f"\n{'='*70}")
            logger.info(f"[ProductSearchWorkflow] ✓ WORKFLOW COMPLETE")
            logger.info(f"[ProductSearchWorkflow] Product Type: {product_type}")
            logger.info(f"[ProductSearchWorkflow] Steps: {', '.join(result['steps_completed'])}")
            logger.info(f"[ProductSearchWorkflow] Products Found: {len(overall_ranking)}")
            logger.info(f"{'='*70}")

            # Build final result for UI
            result['success'] = True
            result['product_type'] = product_type

            # Final requirements structure (for LeftSidebar schema display)
            result['final_requirements'] = structured_requirements

            # Available advanced parameters (for frontend selection)
            result['available_advanced_params'] = unique_specs

            # Analysis result structure (for RightPanel display)
            # This matches the expected format by RightPanel.tsx
            result['analysis_result'] = {
                "productType": product_type,
                "vendorAnalysis": {
                    "vendorMatches": vendor_matches,
                    "vendorRunDetails": vendor_analysis_result.get('vendor_run_details', [])
                },
                "overallRanking": {
                    "rankedProducts": overall_ranking
                },
                "topRecommendation": top_product,
                "totalMatches": len(vendor_matches),
                "exactMatchCount": sum(1 for p in overall_ranking if p.get('requirementsMatch')),
                "approximateMatchCount": sum(1 for p in overall_ranking if not p.get('requirementsMatch'))
            }

            return result

        except Exception as e:
            logger.error(f"[ProductSearchWorkflow] Workflow failed: {e}", exc_info=True)
            result['success'] = False
            result['error'] = str(e)
            result['error_type'] = type(e).__name__
            return result

    def run_validation_only(
        self,
        user_input: str,
        expected_product_type: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run only the validation step (skip advanced parameters).

        Args:
            user_input: User's requirement description
            expected_product_type: Expected product type (optional)
            session_id: Session identifier

        Returns:
            Validation result
        """
        logger.info("[ProductSearchWorkflow] Running validation only")

        return self.validation_tool.validate(
            user_input=user_input,
            expected_product_type=expected_product_type,
            session_id=session_id
        )

    def run_advanced_params_only(
        self,
        product_type: str,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run only the advanced parameters discovery step.

        Args:
            product_type: Product type to discover parameters for
            session_id: Session identifier

        Returns:
            Discovery result
        """
        logger.info("[ProductSearchWorkflow] Running advanced parameters discovery only")

        return self.advanced_params_tool.discover(
            product_type=product_type,
            session_id=session_id
        )

    def run_analysis_only(
        self,
        structured_requirements: Dict[str, Any],
        product_type: str,
        schema: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run only the analysis phase (Steps 4-5: Vendor Analysis + Ranking).

        This is called after the user has confirmed their requirements through
        the interactive workflow. The frontend collects requirements via
        sales-agent and calls this when user clicks "Proceed" in showSummary.

        Args:
            structured_requirements: Complete user requirements
                {
                    "mandatoryRequirements": {...},
                    "optionalRequirements": {...},
                    "selectedAdvancedParams": {...}
                }
            product_type: Detected product type
            schema: Optional schema for validation
            session_id: Session identifier

        Returns:
            Analysis result with vendor matches and ranked products
            {
                "success": bool,
                "vendorAnalysis": {...},
                "overallRanking": {...},
                "topRecommendation": {...}
            }
        """
        if not session_id:
            session_id = f"analysis_{uuid.uuid4().hex[:8]}"

        logger.info(f"\n{'='*70}")
        logger.info(f"[ProductSearchWorkflow] Running ANALYSIS ONLY")
        logger.info(f"[ProductSearchWorkflow] Session: {session_id}")
        logger.info(f"[ProductSearchWorkflow] Product Type: {product_type}")
        logger.info(f"{'='*70}")

        result = {
            "session_id": session_id,
            "product_type": product_type,
            "success": False
        }

        try:
            # STEP 4: VENDOR ANALYSIS
            logger.info(f"\n[STEP 1/2] VENDOR ANALYSIS")
            logger.info("-" * 70)

            vendor_analysis_result = self.vendor_analysis_tool.analyze(
                structured_requirements=structured_requirements,
                product_type=product_type,
                session_id=session_id,
                schema=schema
            )

            if not vendor_analysis_result.get('success'):
                logger.error("[ProductSearchWorkflow] Vendor analysis failed")
                result['error'] = vendor_analysis_result.get('error', 'Vendor analysis failed')
                return result

            vendor_matches = vendor_analysis_result.get('vendor_matches', [])
            logger.info(f"✓ Vendors Analyzed: {vendor_analysis_result.get('vendors_analyzed', 0)}")
            logger.info(f"✓ Products Matched: {len(vendor_matches)}")

            result['vendorAnalysis'] = {
                "vendorMatches": vendor_matches,
                "vendorRunDetails": vendor_analysis_result.get('vendor_run_details', []),
                "totalMatches": len(vendor_matches),
                "vendorsAnalyzed": vendor_analysis_result.get('vendors_analyzed', 0),
                "analysisSummary": vendor_analysis_result.get('analysis_summary', '')
            }

            # STEP 5: RANKING
            logger.info(f"\n[STEP 2/2] RANKING")
            logger.info("-" * 70)

            ranking_result = self.ranking_tool.rank(
                vendor_analysis=vendor_analysis_result,
                session_id=session_id
            )

            if not ranking_result.get('success'):
                logger.warning("[ProductSearchWorkflow] Ranking failed")
                # Continue with empty ranking
                ranking_result = {
                    'success': True,
                    'overall_ranking': [],
                    'top_product': None,
                    'ranking_summary': 'Ranking not available'
                }

            overall_ranking = ranking_result.get('overall_ranking', [])
            top_product = ranking_result.get('top_product')

            logger.info(f"✓ Products Ranked: {len(overall_ranking)}")
            if top_product:
                top_name = top_product.get('product_name', top_product.get('productName', 'Unknown'))
                logger.info(f"✓ Top Recommendation: {top_name}")

            result['overallRanking'] = {
                "rankedProducts": overall_ranking
            }
            result['topRecommendation'] = top_product
            result['totalRanked'] = len(overall_ranking)
            result['exactMatchCount'] = sum(1 for p in overall_ranking if p.get('requirementsMatch'))
            result['approximateMatchCount'] = sum(1 for p in overall_ranking if not p.get('requirementsMatch'))

            # Build analysis_result structure for RightPanel compatibility
            result['analysisResult'] = {
                "productType": product_type,
                "vendorAnalysis": result['vendorAnalysis'],
                "overallRanking": result['overallRanking']
            }

            # Debug log for final result structure
            logger.info(f"[ProductSearchWorkflow] DEBUG: Final analysisResult keys: {list(result['analysisResult'].keys())}")
            if result['analysisResult']['overallRanking']['rankedProducts']:
                logger.info(f"[ProductSearchWorkflow] DEBUG: First ranked product: {result['analysisResult']['overallRanking']['rankedProducts'][0]}")

            logger.info(f"\n{'='*70}")
            logger.info(f"[ProductSearchWorkflow] ✓ ANALYSIS COMPLETE")
            logger.info(f"{'='*70}")
            
            # Log Strategy RAG summary at completion for visibility
            strategy_context = vendor_analysis_result.get('strategy_context', {})
            if strategy_context.get('applied'):
                logger.info("")
                logger.info("🟠 STRATEGY RAG SUMMARY 🟠")
                logger.info(f"   Type: {strategy_context.get('rag_type', 'unknown')}")
                logger.info(f"   Preferred Vendors: {strategy_context.get('preferred_vendors', [])}")
                logger.info(f"   Forbidden Vendors: {strategy_context.get('forbidden_vendors', [])}")
                logger.info(f"   Vendors Analyzed: {vendor_analysis_result.get('vendors_analyzed', 0)}")
                logger.info(f"   Confidence: {strategy_context.get('confidence', 0):.2f}")
                if strategy_context.get('strategy_notes'):
                    logger.info(f"   Strategy Notes: {strategy_context.get('strategy_notes', '')[:100]}...")
                logger.info("")

            result['success'] = True
            return result

        except Exception as e:
            logger.error(f"[ProductSearchWorkflow] Analysis failed: {e}", exc_info=True)
            result['success'] = False
            result['error'] = str(e)
            result['error_type'] = type(e).__name__
            return result

    def process_from_instrument_identifier(
        self,
        identifier_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process input from Instruments Identifier Workflow.

        Args:
            identifier_output: Output from instrument identifier
                {
                    "identified_instruments": [
                        {
                            "product_type": "Pressure Transmitter",
                            "quantity": 2,
                            "basic_requirements": {...}
                        }
                    ]
                }

        Returns:
            Results for each identified instrument
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"[ProductSearchWorkflow] Processing from INSTRUMENTS IDENTIFIER")
        logger.info(f"{'='*70}")

        identified_instruments = identifier_output.get('identified_instruments', [])
        results = []

        for i, instrument in enumerate(identified_instruments, 1):
            product_type = instrument.get('product_type')
            basic_requirements = instrument.get('basic_requirements', {})
            quantity = instrument.get('quantity', 1)

            logger.info(f"\n[Instrument {i}/{len(identified_instruments)}] {product_type} (Qty: {quantity})")

            # Build user input string
            user_input = f"I need {quantity} {product_type}"
            if basic_requirements:
                specs = ", ".join([f"{k}: {v}" for k, v in basic_requirements.items()])
                user_input += f" with {specs}"

            # Run workflow
            result = self.run_single_product_workflow(
                user_input=user_input,
                expected_product_type=product_type
            )

            results.append({
                "product_type": product_type,
                "quantity": quantity,
                "workflow_result": result
            })

        return {
            "source": "instruments_identifier",
            "results": results,
            "total_products": len(results),
            "successful": sum(1 for r in results if r['workflow_result']['success'])
        }

    def process_from_solution_workflow(
        self,
        solution_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process input from Solution Workflow.

        Args:
            solution_output: Output from solution workflow
                {
                    "solution_name": "Water Treatment Plant",
                    "required_products": [
                        {
                            "product_type": "pH Meter",
                            "application": "Wastewater monitoring",
                            "requirements": {...}
                        }
                    ]
                }

        Returns:
            Results for the complete solution
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"[ProductSearchWorkflow] Processing from SOLUTION WORKFLOW")
        logger.info(f"{'='*70}")

        solution_name = solution_output.get('solution_name', 'Unknown Solution')
        required_products = solution_output.get('required_products', [])

        logger.info(f"[Solution] {solution_name}")
        logger.info(f"[Products] {len(required_products)} products required")

        results = []

        for i, product in enumerate(required_products, 1):
            product_type = product.get('product_type')
            requirements = product.get('requirements', {})
            application = product.get('application', '')

            logger.info(f"\n[Product {i}/{len(required_products)}] {product_type} - {application}")

            # Build user input string
            user_input = f"I need a {product_type}"
            if application:
                user_input += f" for {application}"
            if requirements:
                specs = ", ".join([f"{k}: {v}" for k, v in requirements.items()])
                user_input += f" with {specs}"

            # Run workflow
            result = self.run_single_product_workflow(
                user_input=user_input,
                expected_product_type=product_type
            )

            results.append({
                "product_type": product_type,
                "application": application,
                "workflow_result": result
            })

        return {
            "source": "solution_workflow",
            "solution_name": solution_name,
            "results": results,
            "total_products": len(results),
            "successful": sum(1 for r in results if r['workflow_result']['success'])
        }


# ============================================================================
# CONVENIENCE FUNCTION FOR DIRECT ACCESS
# ============================================================================

def product_search_workflow(
    user_input: str,
    expected_product_type: Optional[str] = None,
    enable_ppi: bool = True,
    skip_advanced_params: bool = False,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function for running the product search workflow.

    Args:
        user_input: User's requirement description
        expected_product_type: Expected product type (optional)
        enable_ppi: Enable PPI workflow for schema generation
        skip_advanced_params: Skip advanced parameters discovery
        session_id: Session identifier

    Returns:
        Complete workflow result
    """
    workflow = ProductSearchWorkflow(enable_ppi_workflow=enable_ppi, auto_mode=True)

    return workflow.run_single_product_workflow(
        user_input=user_input,
        expected_product_type=expected_product_type,
        session_id=session_id,
        skip_advanced_params=skip_advanced_params
    )


# ============================================================================
# EXAMPLE USAGE & TESTING
# ============================================================================

def example_direct_input():
    """Example: Direct user input"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Direct User Input")
    print("="*70)

    workflow = ProductSearchWorkflow(auto_mode=True)

    result = workflow.run_single_product_workflow(
        user_input="I need a pressure transmitter with 4-20mA output and HART protocol"
    )

    print(f"\n✓ Success: {result['success']}")
    print(f"✓ Product Type: {result.get('product_type')}")
    print(f"✓ Steps Completed: {', '.join(result['steps_completed'])}")
    print(f"✓ Ready for Vendor Search: {result.get('ready_for_vendor_search')}")
    print(f"✓ Advanced Specs Discovered: {len(result.get('available_advanced_params', []))}")


def example_validation_only():
    """Example: Validation only"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Validation Only")
    print("="*70)

    workflow = ProductSearchWorkflow()

    result = workflow.run_validation_only(
        user_input="I need a flow meter for water, 0-100 GPM"
    )

    print(f"\n✓ Success: {result['success']}")
    print(f"✓ Product Type: {result.get('product_type')}")
    print(f"✓ Valid: {result.get('is_valid')}")
    print(f"✓ Schema Source: {result.get('schema_source')}")


def example_from_instrument_identifier():
    """Example: Input from Instruments Identifier Workflow"""
    print("\n" + "="*70)
    print("EXAMPLE 3: From Instruments Identifier Workflow")
    print("="*70)

    workflow = ProductSearchWorkflow(auto_mode=True)

    identifier_output = {
        "identified_instruments": [
            {
                "product_type": "Pressure Transmitter",
                "quantity": 2,
                "basic_requirements": {
                    "measurementRange": "0-100 bar",
                    "outputSignal": "4-20mA"
                }
            }
        ]
    }

    result = workflow.process_from_instrument_identifier(identifier_output)

    print(f"\n✓ Source: {result['source']}")
    print(f"✓ Total Products: {result['total_products']}")
    print(f"✓ Successful: {result['successful']}/{result['total_products']}")


if __name__ == "__main__":
    # Configure logging for standalone testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("\n" + "="*70)
    print("PRODUCT SEARCH WORKFLOW - COMPLETE EXAMPLES")
    print("="*70)

    # Run examples
    example_direct_input()
    example_validation_only()
    example_from_instrument_identifier()

    print("\n" + "="*70)
    print("✓ ALL EXAMPLES COMPLETED")
    print("="*70)
