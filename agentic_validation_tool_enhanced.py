"""
Enhanced Agentic Validation Tool with PPI Workflow
---------------------------------------------------
Integrates Product-Price-Information (PPI) workflow for schema generation
when schema is not found in database.

Enhanced Features:
1. Auto-discovers top 5 vendors when schema missing
2. Uses tier-based PDF search (Tier 1, 2, 3 fallback)
3. Downloads and extracts PDF data
4. Generates schema from extracted data
5. Stores schema and vendor data in database
"""

import json
import logging
import copy
import re
import io
import requests
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Import base validation tool
from agentic_validation_tool import ValidationTool

# Import required utilities
from chaining import setup_ai_components, parse_json_response
from models import RequirementValidation
from loading import (
    load_requirements_schema,
    discover_top_vendors,
    schema_cache,
    _save_schema_to_specs,
    create_schema_from_vendor_data
)
import prompts

# Import PDF search tools
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools'))
from search_tools import search_pdf_datasheets_tool

# Import PDF extraction
from test import extract_data_from_pdf, send_to_language_model, aggregate_results

# Import Azure/MongoDB utilities
from azure_blob_utils import azure_blob_file_manager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PPIWorkflowEngine:
    """
    Product-Price-Information (PPI) Workflow Engine

    Orchestrates the process of:
    1. Discovering vendors
    2. Searching PDFs with tier fallback
    3. Extracting data from PDFs
    4. Generating schema
    5. Storing in database
    """

    def __init__(self, llm_components=None):
        """Initialize PPI workflow engine"""
        self.components = llm_components or setup_ai_components()
        logger.info("[PPI] Workflow engine initialized")

    def download_pdf(self, pdf_url: str) -> Optional[bytes]:
        """
        Download PDF from URL

        Args:
            pdf_url: URL to PDF file

        Returns:
            PDF bytes or None if failed
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            logger.info(f"[PPI] Downloading PDF: {pdf_url[:80]}...")
            response = requests.get(pdf_url, timeout=30, headers=headers, stream=True)
            response.raise_for_status()

            # Collect PDF data
            pdf_data = b''
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    pdf_data += chunk

            # Validate PDF
            if len(pdf_data) < 1024:
                logger.warning(f"[PPI] PDF too small: {len(pdf_data)} bytes")
                return None

            if not pdf_data.startswith(b'%PDF'):
                logger.warning(f"[PPI] Invalid PDF format")
                return None

            logger.info(f"[PPI] PDF downloaded: {len(pdf_data)} bytes")
            return pdf_data

        except Exception as e:
            logger.error(f"[PPI] PDF download failed: {e}")
            return None

    def extract_pdf_data(self, pdf_bytes: bytes, vendor: str, product_type: str) -> List[Dict[str, Any]]:
        """
        Extract structured data from PDF

        Args:
            pdf_bytes: PDF file bytes
            vendor: Vendor name
            product_type: Product type

        Returns:
            List of extracted product data
        """
        try:
            logger.info(f"[PPI] Extracting data from PDF for {vendor}")

            # Convert to BytesIO
            pdf_file = io.BytesIO(pdf_bytes)

            # Extract text chunks
            text_chunks = extract_data_from_pdf(pdf_file)

            if not text_chunks or len(text_chunks) == 0:
                logger.warning(f"[PPI] No text extracted from PDF")
                return []

            logger.info(f"[PPI] Extracted {len(text_chunks)} text chunks")

            # Use LLM to extract structured data
            logger.info(f"[PPI] Sending {len(text_chunks)} chunks to LLM")
            extracted_data = send_to_language_model(text_chunks)

            # Flatten results
            flattened = []
            for item in extracted_data:
                if isinstance(item, list):
                    flattened.extend(item)
                else:
                    flattened.append(item)

            logger.info(f"[PPI] Extracted {len(flattened)} product entries")
            return flattened

        except Exception as e:
            logger.error(f"[PPI] PDF extraction failed: {e}")
            return []

    def store_pdf_in_database(self, pdf_bytes: bytes, metadata: Dict[str, Any]) -> Optional[str]:
        """
        Store PDF in database (MongoDB/Azure)

        Args:
            pdf_bytes: PDF file bytes
            metadata: PDF metadata

        Returns:
            File ID or None if failed
        """
        try:
            logger.info(f"[PPI] Storing PDF in database: {metadata.get('filename', 'unknown')}")

            # Prepare metadata
            upload_metadata = {
                'collection_type': 'documents',
                'file_type': 'pdf',
                'file_size': len(pdf_bytes),
                **metadata
            }

            # Upload to Azure/MongoDB
            file_id = azure_blob_file_manager.upload_to_azure(pdf_bytes, upload_metadata)

            logger.info(f"[PPI] PDF stored with ID: {file_id}")
            return file_id

        except Exception as e:
            logger.error(f"[PPI] Failed to store PDF: {e}")
            return None

    def store_vendor_data(self, vendor_data: Dict[str, Any], product_type: str) -> bool:
        """
        Store vendor product data in database

        Args:
            vendor_data: Vendor product data
            product_type: Product type

        Returns:
            True if successful
        """
        try:
            vendor_name = vendor_data.get('vendor', 'Unknown')
            logger.info(f"[PPI] Storing vendor data: {vendor_name}")

            # Prepare metadata
            metadata = {
                'collection_type': 'vendors',
                'product_type': product_type,
                'vendor_name': vendor_name,
                'filename': f"{vendor_name.lower().replace(' ', '_')}_{product_type.lower().replace(' ', '_')}.json",
                'file_type': 'json'
            }

            # Upload to database
            doc_id = azure_blob_file_manager.upload_json_data(vendor_data, metadata)

            logger.info(f"[PPI] Vendor data stored with ID: {doc_id}")
            return True

        except Exception as e:
            logger.error(f"[PPI] Failed to store vendor data: {e}")
            return False

    def search_pdfs_with_fallback(self, vendor: str, product_type: str, model_family: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for PDFs using tier-based fallback

        Uses search_pdf_datasheets_tool which implements:
        - Tier 1: Serper API
        - Tier 2: SerpAPI
        - Tier 3: Google CSE

        Args:
            vendor: Vendor name
            product_type: Product type
            model_family: Optional model family

        Returns:
            List of PDF search results
        """
        try:
            logger.info(f"[PPI] Searching PDFs for {vendor} - {product_type}")

            # Use the tier-based PDF search tool
            result = search_pdf_datasheets_tool(
                vendor=vendor,
                product_type=product_type,
                model_family=model_family
            )

            if result.get('success'):
                pdfs = result.get('pdfs', [])
                logger.info(f"[PPI] Found {len(pdfs)} PDFs using tier fallback")
                return pdfs
            else:
                logger.warning(f"[PPI] PDF search failed: {result.get('error')}")
                return []

        except Exception as e:
            logger.error(f"[PPI] PDF search with fallback failed: {e}")
            return []

    def process_vendor(self, vendor_info: Dict[str, Any], product_type: str) -> Dict[str, Any]:
        """
        Process a single vendor: search PDFs, download, extract, store

        Args:
            vendor_info: Vendor information (name, model_families)
            product_type: Product type

        Returns:
            Processing result with extracted data
        """
        vendor_name = vendor_info.get('vendor', 'Unknown')
        model_families = vendor_info.get('model_families', [])

        logger.info(f"[PPI] Processing vendor: {vendor_name}")

        all_extracted_data = []
        pdfs_processed = 0

        # Process each model family (or just product type if no families)
        families_to_process = model_families[:3] if model_families else [None]  # Limit to 3 families

        for model_family in families_to_process:
            # Search for PDFs with tier fallback
            pdfs = self.search_pdfs_with_fallback(vendor_name, product_type, model_family)

            # Process top 2 PDFs per model family
            for pdf_info in pdfs[:2]:
                pdf_url = pdf_info.get('url')
                if not pdf_url:
                    continue

                # Download PDF
                pdf_bytes = self.download_pdf(pdf_url)
                if not pdf_bytes:
                    continue

                # Store PDF in database
                pdf_metadata = {
                    'product_type': product_type,
                    'vendor_name': vendor_name,
                    'model_family': model_family,
                    'filename': os.path.basename(pdf_url.split('?')[0]) or f"{vendor_name}.pdf",
                    'source_url': pdf_url,
                    'pdf_title': pdf_info.get('title', ''),
                    'source': pdf_info.get('source', 'unknown')
                }

                file_id = self.store_pdf_in_database(pdf_bytes, pdf_metadata)

                # Extract data from PDF
                extracted = self.extract_pdf_data(pdf_bytes, vendor_name, product_type)

                if extracted:
                    all_extracted_data.extend(extracted)
                    pdfs_processed += 1
                    logger.info(f"[PPI] Extracted {len(extracted)} entries from {vendor_name} PDF")

        return {
            'vendor': vendor_name,
            'product_type': product_type,
            'extracted_data': all_extracted_data,
            'pdfs_processed': pdfs_processed,
            'success': pdfs_processed > 0
        }

    def generate_schema_from_vendors(
        self,
        vendor_results: List[Dict[str, Any]],
        product_type: str
    ) -> Optional[Dict[str, Any]]:
        """
        Generate schema from vendor data using LLM

        Args:
            vendor_results: List of vendor processing results
            product_type: Product type

        Returns:
            Generated schema or None
        """
        try:
            logger.info(f"[PPI] Generating schema from {len(vendor_results)} vendors")

            # Prepare vendor info for schema generation
            vendors_with_families = []
            for result in vendor_results:
                if result.get('success'):
                    vendors_with_families.append({
                        'vendor': result['vendor'],
                        'model_families': []  # Model families extracted from data
                    })

            if not vendors_with_families:
                logger.warning(f"[PPI] No successful vendor results for schema generation")
                return None

            # Use LLM to generate schema
            schema = create_schema_from_vendor_data(
                product_type,
                vendors_with_families,
                self.components['llm']
            )

            if schema and schema.get('mandatory_requirements'):
                logger.info(f"[PPI] Schema generated successfully")
                return schema
            else:
                logger.warning(f"[PPI] Generated schema is empty")
                return None

        except Exception as e:
            logger.error(f"[PPI] Schema generation failed: {e}")
            return None

    def run_ppi_workflow(self, product_type: str) -> Optional[Dict[str, Any]]:
        """
        Run complete PPI workflow:
        1. Discover top 5 vendors
        2. Search PDFs with tier fallback
        3. Download and extract data
        4. Generate schema
        5. Store in database

        Args:
            product_type: Product type to generate schema for

        Returns:
            Generated schema or None
        """
        logger.info(f"[PPI] Starting PPI workflow for: {product_type}")

        # Step 1: Discover top 5 vendors
        logger.info(f"[PPI] Step 1/5: Discovering top 5 vendors")
        vendors = discover_top_vendors(product_type, self.components['llm'])

        if not vendors:
            logger.error(f"[PPI] No vendors discovered")
            return None

        logger.info(f"[PPI] Discovered {len(vendors)} vendors")

        # Step 2-4: Process each vendor (search PDFs, download, extract)
        logger.info(f"[PPI] Step 2-4/5: Processing vendors (PDF search, download, extract)")
        vendor_results = []

        for vendor_info in vendors[:5]:  # Top 5 vendors
            result = self.process_vendor(vendor_info, product_type)
            vendor_results.append(result)

            # Store vendor data if successful
            if result.get('success') and result.get('extracted_data'):
                # Aggregate extracted data
                aggregated = aggregate_results(result['extracted_data'], product_type)

                vendor_data = {
                    'vendor': result['vendor'],
                    'product_type': product_type,
                    'models': aggregated.get('models', [])
                }

                self.store_vendor_data(vendor_data, product_type)

        # Step 5: Generate and store schema
        logger.info(f"[PPI] Step 5/5: Generating schema from vendor data")
        schema = self.generate_schema_from_vendors(vendor_results, product_type)

        if schema:
            # Store schema in database
            try:
                schema_path = _save_schema_to_specs(product_type, schema)
                logger.info(f"[PPI] Schema saved to database: {schema_path}")

                # Cache schema
                schema_cache.set(product_type, schema)
                logger.info(f"[PPI] Schema cached in memory")

            except Exception as e:
                logger.error(f"[PPI] Failed to save schema: {e}")

        logger.info(f"[PPI] PPI workflow completed")
        return schema


class EnhancedValidationTool(ValidationTool):
    """
    Enhanced Validation Tool with PPI Workflow Integration

    When schema is not found in database:
    1. Invokes PPI workflow
    2. Discovers top 5 vendors
    3. Searches PDFs with tier fallback
    4. Extracts data from PDFs
    5. Generates schema
    6. Stores in database
    """

    def __init__(self):
        """Initialize enhanced validation tool"""
        super().__init__()
        self.ppi_engine = PPIWorkflowEngine(self.components)
        logger.info("[EnhancedValidationTool] Initialized with PPI workflow support")

    def validate(
        self,
        user_input: str,
        search_session_id: str = "default",
        reset: bool = False,
        is_repeat: bool = False,
        enable_ppi_workflow: bool = True
    ) -> Dict[str, Any]:
        """
        Enhanced validation with PPI workflow integration

        Args:
            user_input: User's input text
            search_session_id: Session identifier
            reset: Reset previous state
            is_repeat: Is repeat validation
            enable_ppi_workflow: Enable PPI workflow if schema not found

        Returns:
            Validation result with product type, schema, requirements
        """
        try:
            # Validate input
            if not user_input or not user_input.strip():
                raise ValueError("user_input is required and cannot be empty")

            logger.info(f"[EnhancedValidation] Starting validation for session: {search_session_id}")

            # Step 1: Load initial schema
            initial_schema = load_requirements_schema()

            # Step 2: Detect product type
            session_isolated_input = f"[Session: {search_session_id}] - {user_input}"

            validation_prompt = prompts.validation_prompt.format(
                user_input=session_isolated_input,
                schema=json.dumps(initial_schema, indent=2),
                format_instructions=self.components['validation_format_instructions']
            )

            logger.info(f"[EnhancedValidation] Detecting product type...")
            llm_response = self.components['llm_flash'].invoke(validation_prompt)
            temp_validation_result = parse_json_response(llm_response, RequirementValidation)

            detected_type = temp_validation_result.get('product_type', 'UnknownProduct')
            logger.info(f"[EnhancedValidation] Detected product type: {detected_type}")

            # Step 3: Load specific schema (with PPI workflow if not found)
            specific_schema = load_requirements_schema(detected_type)

            # Check if schema is valid
            schema_exists = (
                specific_schema and
                specific_schema.get("mandatory_requirements") and
                specific_schema.get("optional_requirements")
            )

            if not schema_exists and enable_ppi_workflow:
                logger.info(f"[EnhancedValidation] Schema not found - Invoking PPI workflow")

                # Run PPI workflow to generate schema
                ppi_schema = self.ppi_engine.run_ppi_workflow(detected_type)

                if ppi_schema:
                    specific_schema = ppi_schema
                    logger.info(f"[EnhancedValidation] Schema generated via PPI workflow")
                else:
                    logger.warning(f"[EnhancedValidation] PPI workflow failed - using empty schema")
                    specific_schema = {
                        "product_type": detected_type,
                        "mandatory_requirements": {},
                        "optional_requirements": {}
                    }

            # Continue with standard validation flow
            logger.info(f"[EnhancedValidation] Running detailed validation...")

            validation_prompt_specific = prompts.validation_prompt.format(
                user_input=session_isolated_input,
                schema=json.dumps(specific_schema, indent=2),
                format_instructions=self.components['validation_format_instructions']
            )

            llm_response_specific = self.components['llm_flash'].invoke(validation_prompt_specific)
            validation_result = parse_json_response(llm_response_specific, RequirementValidation)

            # Clean and map requirements
            cleaned_provided_reqs = self.clean_empty_values(
                validation_result.get("provided_requirements", {})
            )

            mapped_provided_reqs = self.map_provided_to_schema(
                self.convert_keys_to_camel_case(specific_schema),
                self.convert_keys_to_camel_case(cleaned_provided_reqs)
            )

            # Build response
            response_data = {
                "productType": validation_result.get("product_type", detected_type),
                "detectedSchema": self.convert_keys_to_camel_case(specific_schema),
                "providedRequirements": mapped_provided_reqs,
                "ppiWorkflowUsed": not schema_exists and enable_ppi_workflow
            }

            # Check for missing mandatory fields
            missing_mandatory_fields = self.get_missing_mandatory_fields(
                mapped_provided_reqs,
                response_data["detectedSchema"]
            )

            if missing_mandatory_fields:
                missing_fields_friendly = [
                    self.friendly_field_name(f) for f in missing_mandatory_fields
                ]
                missing_fields_str = ", ".join(missing_fields_friendly)

                alert_prompt = prompts.validation_alert_initial_prompt if not is_repeat else prompts.validation_alert_repeat_prompt
                alert_prompt_text = alert_prompt.format(
                    product_type=response_data["productType"],
                    missing_fields=missing_fields_str
                )

                alert_response = self.components['llm'].invoke(alert_prompt_text)

                response_data["validationAlert"] = {
                    "message": alert_response,
                    "canContinue": True,
                    "missingFields": missing_mandatory_fields
                }

            logger.info(f"[EnhancedValidation] Validation completed")
            return response_data

        except Exception as e:
            logger.error(f"[EnhancedValidation] Validation failed: {e}", exc_info=True)
            raise


# Convenience function
def validate_with_ppi(
    user_input: str,
    search_session_id: str = "default",
    reset: bool = False,
    is_repeat: bool = False,
    enable_ppi_workflow: bool = True
) -> Dict[str, Any]:
    """
    Convenience function for enhanced validation with PPI workflow

    Args:
        user_input: User input
        search_session_id: Session ID
        reset: Reset state
        is_repeat: Is repeat
        enable_ppi_workflow: Enable PPI workflow

    Returns:
        Validation result
    """
    tool = EnhancedValidationTool()
    return tool.validate(user_input, search_session_id, reset, is_repeat, enable_ppi_workflow)


# Example usage
if __name__ == "__main__":
    print("\n" + "="*70)
    print("ENHANCED VALIDATION TOOL WITH PPI WORKFLOW")
    print("="*70)

    # Example: Validate with PPI workflow
    test_input = "I need a humidity transmitter with 4-20mA output"

    print(f"\nInput: {test_input}")
    print("\nRunning enhanced validation with PPI workflow...")

    try:
        result = validate_with_ppi(test_input, enable_ppi_workflow=True)

        print(f"\n✓ Product Type: {result['productType']}")
        print(f"✓ PPI Workflow Used: {result.get('ppiWorkflowUsed', False)}")

        if 'validationAlert' in result:
            print(f"\n⚠ Validation Alert:")
            print(f"  {result['validationAlert']['message']}")

        print(f"\n✓ Schema loaded with {len(result['detectedSchema'].get('mandatoryRequirements', {}))} mandatory categories")

    except Exception as e:
        print(f"\n✗ Error: {e}")
