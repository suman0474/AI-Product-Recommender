# tools/schema_tools.py
# Schema Management and Requirements Validation Tools

import json
import logging
from typing import Dict, Any, List, Optional
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
from llm_fallback import create_llm_with_fallback
from prompts_library import load_prompt

load_dotenv()
logger = logging.getLogger(__name__)


# ============================================================================
# TOOL INPUT SCHEMAS
# ============================================================================

class LoadSchemaInput(BaseModel):
    """Input for loading product schema"""
    product_type: str = Field(description="Product type to load schema for")
    enable_ppi: bool = Field(
        default=True,
        description="Enable PPI (Product-Price-Information) workflow if schema not found in database"
    )


class ValidateRequirementsInput(BaseModel):
    """Input for requirements validation"""
    user_input: str = Field(description="User's requirements input")
    product_type: str = Field(description="Detected product type")
    product_schema: Optional[Dict[str, Any]] = Field(default=None, description="Product schema")


class GetMissingFieldsInput(BaseModel):
    """Input for getting missing fields"""
    provided_requirements: Dict[str, Any] = Field(description="Requirements already provided")
    product_schema: Dict[str, Any] = Field(description="Product schema with mandatory/optional fields")


# ============================================================================
# PROMPTS - Loaded from prompts_library
# ============================================================================

VALIDATION_PROMPT = load_prompt("schema_validation_prompt")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_schema_from_azure(product_type: str) -> Optional[Dict[str, Any]]:
    """Load schema from Azure Blob Storage"""
    try:
        from azure_blob_utils import azure_blob_file_manager
        schema = azure_blob_file_manager.get_schema_from_azure(product_type)
        return schema
    except Exception as e:
        logger.warning(f"Failed to load schema from Azure Blob: {e}")
        return None


def get_default_schema(product_type: str) -> Dict[str, Any]:
    """Get default schema for common product types"""
    default_schemas = {
        "pressure transmitter": {
            "mandatory": {
                "outputSignal": "Communication/output signal type",
                "pressureRange": "Measurement pressure range",
                "processConnection": "Process connection type"
            },
            "optional": {
                "accuracy": "Measurement accuracy",
                "wettedParts": "Wetted parts material",
                "displayType": "Display type",
                "certifications": "Required certifications",
                "ambientTemperature": "Operating temperature range",
                "protocol": "Communication protocol"
            }
        },
        "flow meter": {
            "mandatory": {
                "flowType": "Type of flow measurement",
                "flowRange": "Flow measurement range",
                "pipeSize": "Pipe/line size"
            },
            "optional": {
                "accuracy": "Measurement accuracy",
                "fluidType": "Type of fluid",
                "outputSignal": "Output signal type",
                "processConnection": "Process connection",
                "material": "Construction material"
            }
        },
        "temperature sensor": {
            "mandatory": {
                "sensorType": "Type of sensor (RTD, thermocouple)",
                "temperatureRange": "Temperature measurement range"
            },
            "optional": {
                "accuracy": "Measurement accuracy",
                "responseTime": "Response time",
                "sheathMaterial": "Sheath material",
                "connectionType": "Connection type"
            }
        },
        "level transmitter": {
            "mandatory": {
                "measurementType": "Measurement type (radar, ultrasonic, etc.)",
                "measurementRange": "Measurement range"
            },
            "optional": {
                "accuracy": "Measurement accuracy",
                "processConnection": "Process connection",
                "outputSignal": "Output signal",
                "material": "Construction material"
            }
        }
    }

    # Normalize product type
    product_type_lower = product_type.lower().strip()

    for key, schema in default_schemas.items():
        if key in product_type_lower or product_type_lower in key:
            return schema

    # Return generic schema
    return {
        "mandatory": {
            "productType": "Product type specification"
        },
        "optional": {
            "accuracy": "Measurement accuracy",
            "outputSignal": "Output signal type",
            "material": "Construction material"
        }
    }


# ============================================================================
# TOOLS
# ============================================================================

@tool("load_schema", args_schema=LoadSchemaInput)
def load_schema_tool(product_type: str, enable_ppi: bool = True) -> Dict[str, Any]:
    """
    Load the requirements schema for a specific product type.

    Priority order:
    1. Try Azure Blob Storage (and enrich with Standards RAG field values)
    2. If not found and enable_ppi=True, invoke LangGraph PPI workflow to generate schema
    3. Fall back to default schemas only as last resort

    All schemas are enriched with Standards RAG to fill field values with:
    - Allowed values/ranges from standards
    - Standard units and typical specifications
    - References to specific standards for each field

    Args:
        product_type: Product type to load schema for
        enable_ppi: Enable PPI (Product-Price-Information) workflow if schema not found

    Returns:
        Dict with success, schema, and metadata about source
    """
    try:
        # Step 1: Try Azure Blob first
        schema = get_schema_from_azure(product_type)

        if schema and schema.get("mandatory_requirements") and schema.get("optional_requirements"):
            logger.info(f"✓ Loaded schema from Azure Blob for {product_type}")
            
            # Step 1.1: Enrich loaded schema with Standards RAG field values
            try:
                from tools.standards_enrichment_tool import (
                    populate_schema_fields_from_standards,
                    enrich_schema_with_standards
                )
                
                # Check if schema already has standards population
                if not schema.get("_standards_population"):
                    logger.info(f"  ⚙ Enriching Azure schema with Standards RAG field values")
                    enriched_schema = populate_schema_fields_from_standards(product_type, schema)
                    
                    # Also add standards section if not present
                    if "standards" not in enriched_schema:
                        enriched_schema = enrich_schema_with_standards(product_type, enriched_schema)
                    
                    fields_populated = enriched_schema.get("_standards_population", {}).get("fields_populated", 0)
                    logger.info(f"  ✓ Enriched schema with {fields_populated} standards-based field values")
                    schema = enriched_schema
                else:
                    logger.info(f"  ✓ Schema already enriched with Standards RAG")
                    
            except Exception as enrich_error:
                logger.warning(f"  ⚠ Standards enrichment failed (using original): {enrich_error}")
            
            return {
                "success": True,
                "product_type": product_type,
                "schema": schema,
                "from_database": True,
                "ppi_used": False,
                "source": "azure",
                "standards_enriched": schema.get("_standards_population") is not None
            }

        # Step 2: Schema not found in Azure - invoke LangGraph PPI workflow if enabled
        logger.info(f"Schema not found in Azure for {product_type}")

        if enable_ppi:
            logger.info(f"⚙ Invoking LangGraph PPI workflow to generate schema for {product_type}")
            try:
                # Import the LangGraph PPI workflow
                from product_search_workflow.ppi_workflow import run_ppi_workflow

                # Run the PPI workflow
                ppi_result = run_ppi_workflow(product_type)

                if ppi_result.get("success") and ppi_result.get("schema"):
                    generated_schema = ppi_result["schema"]
                    logger.info(f"✓ Successfully generated schema via LangGraph PPI for {product_type}")
                    logger.info(f"  Vendors processed: {ppi_result.get('vendors_processed', 0)}")
                    logger.info(f"  PDFs processed: {ppi_result.get('pdfs_processed', 0)}")
                    
                    return {
                        "success": True,
                        "product_type": product_type,
                        "schema": generated_schema,
                        "from_database": False,
                        "ppi_used": True,
                        "source": "ppi_langgraph_workflow",
                        "ppi_details": {
                            "vendors_processed": ppi_result.get("vendors_processed", 0),
                            "pdfs_processed": ppi_result.get("pdfs_processed", 0)
                        }
                    }
                else:
                    logger.warning(f"⚠ LangGraph PPI workflow returned no schema for {product_type}")
                    logger.warning(f"  Error: {ppi_result.get('error', 'Unknown')}")

            except Exception as ppi_error:
                logger.error(f"✗ LangGraph PPI workflow failed for {product_type}: {ppi_error}", exc_info=True)
        else:
            logger.info(f"PPI workflow disabled for {product_type}")

        # Step 3: Fall back to default schema only as last resort
        logger.warning(f"⚠ Using default schema for {product_type} (Azure: not found, PPI: {'disabled' if not enable_ppi else 'failed'})")
        default_schema = get_default_schema(product_type)

        return {
            "success": True,
            "product_type": product_type,
            "schema": default_schema,
            "from_database": False,
            "ppi_used": False,
            "source": "default_fallback"
        }

    except Exception as e:
        logger.error(f"✗ Failed to load schema for {product_type}: {e}", exc_info=True)
        return {
            "success": False,
            "product_type": product_type,
            "schema": get_default_schema(product_type),
            "from_database": False,
            "ppi_used": False,
            "error": str(e),
            "source": "error_fallback"
        }


@tool("validate_requirements", args_schema=ValidateRequirementsInput)
def validate_requirements_tool(
    user_input: str,
    product_type: str,
    product_schema: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Validate user requirements against the product schema.
    Returns validation status, provided requirements, and missing fields.
    """
    try:
        # Load schema if not provided
        if not product_schema:
            schema_result = load_schema_tool.invoke({"product_type": product_type})
            product_schema = schema_result.get("schema", {})

        llm = create_llm_with_fallback(
            model="gemini-2.5-flash",
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        prompt = ChatPromptTemplate.from_template(VALIDATION_PROMPT)
        parser = JsonOutputParser()

        chain = prompt | llm | parser

        result = chain.invoke({
            "user_input": user_input,
            "product_type": product_type,
            "schema": json.dumps(product_schema, indent=2)
        })

        return {
            "success": True,
            "is_valid": result.get("is_valid", False),
            "product_type": result.get("product_type", product_type),
            "provided_requirements": result.get("provided_requirements", {}),
            "missing_fields": result.get("missing_fields", []),
            "optional_fields": result.get("optional_fields", []),
            "validation_messages": result.get("validation_messages", [])
        }

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return {
            "success": False,
            "is_valid": False,
            "product_type": product_type,
            "error": str(e)
        }


@tool("get_missing_fields", args_schema=GetMissingFieldsInput)
def get_missing_fields_tool(
    provided_requirements: Dict[str, Any],
    product_schema: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Identify missing mandatory and optional fields based on schema.
    """
    try:
        mandatory_fields = product_schema.get("mandatory", {})
        optional_fields = product_schema.get("optional", {})

        # Find missing mandatory fields
        missing_mandatory = []
        for field, description in mandatory_fields.items():
            if field not in provided_requirements or not provided_requirements[field]:
                missing_mandatory.append({
                    "field": field,
                    "description": description
                })

        # Find available optional fields
        available_optional = []
        for field, description in optional_fields.items():
            if field not in provided_requirements or not provided_requirements[field]:
                available_optional.append({
                    "field": field,
                    "description": description
                })

        return {
            "success": True,
            "missing_mandatory": missing_mandatory,
            "available_optional": available_optional,
            "is_complete": len(missing_mandatory) == 0,
            "provided_count": len(provided_requirements),
            "total_mandatory": len(mandatory_fields)
        }

    except Exception as e:
        logger.error(f"Failed to get missing fields: {e}")
        return {
            "success": False,
            "missing_mandatory": [],
            "available_optional": [],
            "error": str(e)
        }
