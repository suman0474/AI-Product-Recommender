"""
Validation Tool for Product Search Workflow
============================================

Step 1 of Product Search Workflow:
- Detects product type from user input
- Loads or generates schema (with PPI workflow if needed)
- Validates requirements against schema
- Returns structured validation result

This tool integrates:
- Intent extraction (product type detection)
- Schema loading/generation (with PPI workflow)
- Requirements validation
"""

import logging
from typing import Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)


class ValidationTool:
    """
    Validation Tool - Step 1 of Product Search Workflow

    Responsibilities:
    1. Extract product type from user input
    2. Load or generate product schema (PPI workflow if needed)
    3. Validate user requirements against schema
    4. Return structured validation result
    """

    def __init__(self, enable_ppi: bool = True):
        """
        Initialize the validation tool.

        Args:
            enable_ppi: Enable PPI workflow for schema generation
        """
        self.enable_ppi = enable_ppi
        logger.info("[ValidationTool] Initialized with PPI workflow: %s",
                   "enabled" if enable_ppi else "disabled")

    def validate(
        self,
        user_input: str,
        expected_product_type: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate user input and requirements.

        Args:
            user_input: User's requirement description
            expected_product_type: Expected product type (optional, for validation)
            session_id: Session identifier (for logging/tracking)

        Returns:
            Validation result with:
            {
                "success": bool,
                "product_type": str,
                "schema": dict,
                "provided_requirements": dict,
                "missing_fields": list,
                "is_valid": bool,
                "ppi_workflow_used": bool,
                "schema_source": str,  # "azure", "ppi_workflow", or "default_fallback"
                "session_id": str
            }
        """
        logger.info("[ValidationTool] Starting validation")
        logger.info("[ValidationTool] Session: %s", session_id or "N/A")
        logger.info("[ValidationTool] Input: %s", user_input[:100] + "..." if len(user_input) > 100 else user_input)

        result = {
            "success": False,
            "session_id": session_id
        }

        try:
            # Import required tools
            from tools.schema_tools import load_schema_tool, validate_requirements_tool
            from tools.intent_tools import extract_requirements_tool

            # =================================================================
            # STEP 1.1: EXTRACT PRODUCT TYPE
            # =================================================================
            logger.info("[ValidationTool] Step 1.1: Extracting product type")

            extract_result = extract_requirements_tool.invoke({
                "user_input": user_input
            })

            product_type = extract_result.get("product_type", expected_product_type or "")
            logger.info("[ValidationTool] ✓ Detected product type: %s", product_type)

            # Validate against expected type if provided
            if expected_product_type and product_type.lower() != expected_product_type.lower():
                logger.warning(
                    "[ValidationTool] ⚠ Product type mismatch - Expected: %s, Detected: %s",
                    expected_product_type,
                    product_type
                )

            # =================================================================
            # STEP 1.2: LOAD OR GENERATE SCHEMA
            # =================================================================
            logger.info("[ValidationTool] Step 1.2: Loading/generating schema")

            schema_result = load_schema_tool.invoke({
                "product_type": product_type,
                "enable_ppi": self.enable_ppi
            })

            schema = schema_result.get("schema", {})
            schema_source = schema_result.get("source", "unknown")
            ppi_used = schema_result.get("ppi_used", False)
            from_database = schema_result.get("from_database", False)

            # Log schema source
            if from_database:
                logger.info("[ValidationTool] ✓ Schema loaded from Azure Blob Storage")
            elif ppi_used:
                logger.info("[ValidationTool] ✓ Schema generated via PPI workflow")
            else:
                logger.warning("[ValidationTool] ⚠ Using default schema (fallback)")

            # =================================================================
            # STEP 1.2.1: ENRICH SCHEMA WITH STANDARDS RAG (FIELD VALUES + STANDARDS SECTION)
            # Uses both tools.standards_enrichment_tool and agentic.standards_rag_enrichment
            # =================================================================
            standards_info = None
            enrichment_result = None
            standards_rag_invoked = False
            standards_rag_invocation_time = None
            try:
                # ╔══════════════════════════════════════════════════════════════╗
                # ║           🔵 STANDARDS RAG INVOCATION STARTING 🔵            ║
                # ╚══════════════════════════════════════════════════════════════╝
                import datetime
                standards_rag_invocation_time = datetime.datetime.now().isoformat()
                logger.info("="*70)
                logger.info("🔵 STANDARDS RAG INVOKED 🔵")
                logger.info(f"   Timestamp: {standards_rag_invocation_time}")
                logger.info(f"   Product Type: {product_type}")
                logger.info(f"   Session: {session_id}")
                logger.info("="*70)
                print("\n" + "="*70)
                print("🔵 [STANDARDS RAG] INVOCATION STARTED")
                print(f"   Time: {standards_rag_invocation_time}")
                print(f"   Product: {product_type}")
                print("="*70 + "\n")
                
                standards_rag_invoked = True
                logger.info("[ValidationTool] Step 1.2.1: Enriching schema with Standards RAG")
                
                # Import from both modules for comprehensive enrichment
                from tools.standards_enrichment_tool import (
                    get_applicable_standards,
                    populate_schema_fields_from_standards
                )
                from agentic.standards_rag_enrichment import (
                    enrich_identified_items_with_standards,
                    is_standards_related_question
                )
                from agentic.deep_agent_schema_populator import (
                    populate_schema_with_deep_agent,
                    integrate_deep_agent_with_validation
                )

                # Step 1.2.1a: POPULATE field values from Standards RAG (using tools module)
                if not schema.get("_standards_population"):
                    logger.info("[ValidationTool] Step 1.2.1a: Populating schema field values from standards")
                    schema = populate_schema_fields_from_standards(product_type, schema)
                    fields_populated = schema.get("_standards_population", {}).get("fields_populated", 0)
                    logger.info(f"[ValidationTool] ✓ Populated {fields_populated} fields with standards values")
                else:
                    logger.info("[ValidationTool] ✓ Schema already has standards-populated field values")

                # Step 1.2.1b: GET applicable standards for standards section (using tools module)
                standards_info = get_applicable_standards(product_type, top_k=5)

                if standards_info.get('success'):
                    # Add standards to schema if not already present
                    if 'standards' not in schema:
                        schema['standards'] = {
                            'applicable_standards': standards_info.get('applicable_standards', []),
                            'certifications': standards_info.get('certifications', []),
                            'safety_requirements': standards_info.get('safety_requirements', {}),
                            'calibration_standards': standards_info.get('calibration_standards', {}),
                            'environmental_requirements': standards_info.get('environmental_requirements', {}),
                            'communication_protocols': standards_info.get('communication_protocols', []),
                            'sources': standards_info.get('sources', []),
                            'confidence': standards_info.get('confidence', 0.0)
                        }

                    num_standards = len(standards_info.get('applicable_standards', []))
                    num_certs = len(standards_info.get('certifications', []))
                    logger.info(f"[ValidationTool] ✓ Standards enriched: {num_standards} standards, {num_certs} certifications")
                    
                    # Log success indicator
                    logger.info("="*70)
                    logger.info("🔵 STANDARDS RAG COMPLETED SUCCESSFULLY 🔵")
                    logger.info(f"   Standards Found: {num_standards}")
                    logger.info(f"   Certifications Found: {num_certs}")
                    logger.info("="*70)
                    print("\n" + "="*70)
                    print("🔵 [STANDARDS RAG] COMPLETED SUCCESSFULLY")
                    print(f"   Standards: {num_standards}, Certs: {num_certs}")
                    print("="*70 + "\n")
                else:
                    logger.warning(f"[ValidationTool] ⚠ Standards RAG returned no results: {standards_info.get('error', 'Unknown')}")
                    print("\n" + "="*70)
                    print("🔵 [STANDARDS RAG] NO RESULTS RETURNED")
                    print("="*70 + "\n")

                # Step 1.2.1c: ENRICH with normalized category using agentic module
                # This provides structured standards_info with normalized_category for the product type
                try:
                    # Create a mock item representing this product type for enrichment
                    product_item = [{
                        "name": product_type,
                        "category": product_type,
                        "specifications": schema.get("mandatory", {})
                    }]
                    
                    enriched_items = enrich_identified_items_with_standards(
                        items=product_item,
                        product_type=product_type,
                        top_k=3
                    )
                    
                    if enriched_items and len(enriched_items) > 0:
                        enrichment_result = enriched_items[0].get("standards_info", {})
                        
                        # Add normalized category to schema if available
                        if enriched_items[0].get("normalized_category"):
                            schema["normalized_category"] = enriched_items[0]["normalized_category"]
                            logger.info(f"[ValidationTool] ✓ Normalized category: {schema['normalized_category']}")
                        
                        # Merge additional enrichment info into standards section
                        if enrichment_result.get("enrichment_status") == "success":
                            if "standards" in schema:
                                # Merge communication protocols if new ones found
                                existing_protocols = set(schema["standards"].get("communication_protocols", []))
                                new_protocols = set(enrichment_result.get("communication_protocols", []))
                                schema["standards"]["communication_protocols"] = list(existing_protocols | new_protocols)
                                
                                # Merge certifications if new ones found
                                existing_certs = set(schema["standards"].get("certifications", []))
                                new_certs = set(enrichment_result.get("certifications", []))
                                schema["standards"]["certifications"] = list(existing_certs | new_certs)
                                
                            logger.info("[ValidationTool] ✓ Additional enrichment merged from standards_rag_enrichment")
                            
                except Exception as enrich_err:
                    logger.debug(f"[ValidationTool] Additional enrichment skipped: {enrich_err}")
                    # Non-critical - continue without additional enrichment

                # ╔══════════════════════════════════════════════════════════════╗
                # ║   🔷 DEEP AGENT SCHEMA POPULATION - COMPREHENSIVE EXTRACTION 🔷║
                # ╚══════════════════════════════════════════════════════════════╝
                # Step 1.2.1d: Use Deep Agent to populate ALL schema fields from standards docs
                try:
                    if not schema.get("_deep_agent_population"):
                        logger.info("[ValidationTool] Step 1.2.1d: Deep Agent schema population starting")
                        print("\n" + "="*70)
                        print("🔷 [DEEP AGENT] SCHEMA POPULATION STARTING")
                        print(f"   Product: {product_type}")
                        print("="*70 + "\n")

                        # Use Deep Agent to populate all schema fields
                        schema = populate_schema_with_deep_agent(
                            product_type=product_type,
                            schema=schema,
                            max_workers=4
                        )

                        pop_info = schema.get("_deep_agent_population", {})
                        fields_populated = pop_info.get("fields_populated", 0)
                        total_fields = pop_info.get("total_fields", 0)

                        logger.info(f"[ValidationTool] ✓ Deep Agent populated {fields_populated}/{total_fields} fields")
                        print("\n" + "="*70)
                        print("🔷 [DEEP AGENT] SCHEMA POPULATION COMPLETED")
                        print(f"   Fields Populated: {fields_populated}/{total_fields}")
                        print(f"   Sources Used: {len(pop_info.get('sources_used', []))}")
                        print("="*70 + "\n")
                    else:
                        logger.info("[ValidationTool] ✓ Schema already has Deep Agent population")

                except Exception as deep_agent_err:
                    logger.warning(f"[ValidationTool] ⚠ Deep Agent population failed (non-critical): {deep_agent_err}")
                    print("\n" + "="*70)
                    print(f"🔷 [DEEP AGENT] WARNING: {deep_agent_err}")
                    print("="*70 + "\n")
                    # Non-critical - continue without Deep Agent population

            except Exception as standards_error:
                logger.warning(f"[ValidationTool] ⚠ Standards enrichment failed (non-critical): {standards_error}")
                print("\n" + "="*70)
                print(f"🔵 [STANDARDS RAG] ERROR: {standards_error}")
                print("="*70 + "\n")
                # Continue without standards - this is non-critical

            # NOTE: Strategy RAG is NOT applied during initial validation.
            # Strategy-based vendor filtering is applied during Final Vendor Analysis
            # in vendor_analysis_tool.py to filter/prioritize vendors before analysis.
            strategy_info = None  # Placeholder for result compatibility


            # =================================================================
            # STEP 1.3: VALIDATE REQUIREMENTS
            # =================================================================
            logger.info("[ValidationTool] Step 1.3: Validating requirements")

            validation_result = validate_requirements_tool.invoke({
                "user_input": user_input,
                "product_type": product_type,
                "schema": schema
            })

            provided_requirements = validation_result.get("provided_requirements", {})
            missing_fields = validation_result.get("missing_fields", [])
            is_valid = validation_result.get("is_valid", False)

            # Log validation results
            if is_valid:
                logger.info("[ValidationTool] ✓ All mandatory fields provided")
            else:
                logger.info("[ValidationTool] ⚠ Missing mandatory fields: %s", missing_fields)

            # =================================================================
            # BUILD RESULT
            # =================================================================
            result.update({
                "success": True,
                "product_type": product_type,
                "normalized_category": schema.get("normalized_category"),  # From standards_rag_enrichment
                "schema": schema,
                "provided_requirements": provided_requirements,
                "missing_fields": missing_fields,
                "optional_fields": validation_result.get("optional_fields", []),
                "is_valid": is_valid,
                "ppi_workflow_used": ppi_used,
                "schema_source": schema_source,
                "from_database": from_database,
                # ══════════════════════════════════════════════════════════════
                # RAG INVOCATION TRACKING - Visible in browser Network tab
                # ══════════════════════════════════════════════════════════════
                "rag_invocations": {
                    "standards_rag": {
                        "invoked": standards_rag_invoked,
                        "invocation_time": standards_rag_invocation_time,
                        "success": standards_info.get('success', False) if standards_info else False,
                        "product_type": product_type,
                        "results_count": len(standards_info.get('applicable_standards', [])) if standards_info else 0
                    },
                    "strategy_rag": {
                        "invoked": False,
                        "note": "Strategy RAG is applied in vendor_analysis_tool.py, not during validation"
                    }
                },
                # Standards RAG enrichment results (combined from tools and agentic modules)
                "standards_info": {
                    "applicable_standards": standards_info.get('applicable_standards', []) if standards_info else [],
                    "certifications": standards_info.get('certifications', []) if standards_info else [],
                    "communication_protocols": standards_info.get('communication_protocols', []) if standards_info else [],
                    "safety_requirements": standards_info.get('safety_requirements', {}) if standards_info else {},
                    "environmental_requirements": standards_info.get('environmental_requirements', {}) if standards_info else {},
                    "confidence": standards_info.get('confidence', 0.0) if standards_info else 0.0,
                    "sources": standards_info.get('sources', []) if standards_info else [],
                    "enrichment_success": standards_info.get('success', False) if standards_info else False,
                    # Additional data from standards_rag_enrichment module
                    "additional_enrichment": {
                        "status": enrichment_result.get("enrichment_status") if enrichment_result else "not_performed",
                        "merged_protocols": enrichment_result.get("communication_protocols", []) if enrichment_result else [],
                        "merged_certifications": enrichment_result.get("certifications", []) if enrichment_result else []
                    } if enrichment_result else None
                },
                # Strategy RAG enrichment results (TRUE RAG with vector store)
                "strategy_info": {
                    "preferred_vendors": strategy_info.get('preferred_vendors', []) if strategy_info else [],
                    "forbidden_vendors": strategy_info.get('forbidden_vendors', []) if strategy_info else [],
                    "neutral_vendors": strategy_info.get('neutral_vendors', []) if strategy_info else [],
                    "procurement_priorities": strategy_info.get('procurement_priorities', {}) if strategy_info else {},
                    "strategy_notes": strategy_info.get('strategy_notes', '') if strategy_info else '',
                    "confidence": strategy_info.get('confidence', 0.0) if strategy_info else 0.0,
                    "rag_type": strategy_info.get('rag_type', 'unknown') if strategy_info else 'not_performed',  # 'true_rag' or 'llm_inference'
                    "sources_used": strategy_info.get('sources_used', []) if strategy_info else [],
                    "enrichment_success": strategy_info.get('success', False) if strategy_info else False
                },
                # ══════════════════════════════════════════════════════════════
                # DEEP AGENT SCHEMA POPULATION - Section-based specifications
                # ══════════════════════════════════════════════════════════════
                "deep_agent_info": {
                    "population_performed": schema.get("_deep_agent_population") is not None,
                    "fields_populated": schema.get("_deep_agent_population", {}).get("fields_populated", 0),
                    "total_fields": schema.get("_deep_agent_population", {}).get("total_fields", 0),
                    "sections_processed": schema.get("_deep_agent_population", {}).get("sections_processed", 0),
                    "sources_used": schema.get("_deep_agent_population", {}).get("sources_used", []),
                    "processing_time_ms": schema.get("_deep_agent_population", {}).get("processing_time_ms", 0),
                    # Section-based field values for UI display
                    "sections": schema.get("_deep_agent_sections", {})
                }
            })

            logger.info("[ValidationTool] ✓ Validation completed successfully")
            return result

        except Exception as e:
            logger.error("[ValidationTool] ✗ Validation failed: %s", e, exc_info=True)
            result.update({
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            })
            return result

    def get_schema_only(
        self,
        product_type: str,
        enable_ppi: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Load or generate schema without validation.

        Args:
            product_type: Product type to get schema for
            enable_ppi: Override PPI setting (uses instance setting if None)

        Returns:
            Schema result with source information
        """
        logger.info("[ValidationTool] Loading schema for: %s", product_type)

        try:
            from tools.schema_tools import load_schema_tool

            schema_result = load_schema_tool.invoke({
                "product_type": product_type,
                "enable_ppi": enable_ppi if enable_ppi is not None else self.enable_ppi
            })

            logger.info("[ValidationTool] ✓ Schema loaded from: %s",
                       schema_result.get("source", "unknown"))

            return schema_result

        except Exception as e:
            logger.error("[ValidationTool] ✗ Schema loading failed: %s", e, exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "schema": {}
            }

    def validate_with_schema(
        self,
        user_input: str,
        product_type: str,
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate user input against a provided schema.

        Args:
            user_input: User's requirement description
            product_type: Product type
            schema: Pre-loaded schema

        Returns:
            Validation result
        """
        logger.info("[ValidationTool] Validating with provided schema")

        try:
            from tools.schema_tools import validate_requirements_tool

            validation_result = validate_requirements_tool.invoke({
                "user_input": user_input,
                "product_type": product_type,
                "schema": schema
            })

            return {
                "success": True,
                "product_type": product_type,
                "provided_requirements": validation_result.get("provided_requirements", {}),
                "missing_fields": validation_result.get("missing_fields", []),
                "is_valid": validation_result.get("is_valid", False)
            }

        except Exception as e:
            logger.error("[ValidationTool] ✗ Validation failed: %s", e, exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }


# ============================================================================
# STANDALONE USAGE EXAMPLE
# ============================================================================

def example_usage():
    """Example usage of ValidationTool"""
    print("\n" + "="*70)
    print("VALIDATION TOOL - STANDALONE EXAMPLE")
    print("="*70)

    # Initialize tool
    tool = ValidationTool(enable_ppi=True)

    # Example 1: Validate user input
    print("\n[Example 1] Validate user input:")
    result = tool.validate(
        user_input="I need a pressure transmitter with 4-20mA output, 0-100 PSI range",
        session_id="test_session_001"
    )

    print(f"✓ Success: {result['success']}")
    print(f"✓ Product Type: {result.get('product_type')}")
    print(f"✓ Valid: {result.get('is_valid')}")
    print(f"✓ Schema Source: {result.get('schema_source')}")
    print(f"✓ PPI Used: {result.get('ppi_workflow_used')}")
    print(f"✓ Missing Fields: {result.get('missing_fields', [])}")

    # Example 2: Get schema only
    print("\n[Example 2] Get schema only:")
    schema_result = tool.get_schema_only("flow meter")
    print(f"✓ Schema Source: {schema_result.get('source')}")
    print(f"✓ Has Mandatory Fields: {bool(schema_result.get('schema', {}).get('mandatory'))}")


if __name__ == "__main__":
    # Configure logging for standalone testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    example_usage()
