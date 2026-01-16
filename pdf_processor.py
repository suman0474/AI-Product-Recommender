# pdf_processor.py
"""
PDF Processing and Data Extraction Module

This module provides functions for:
- Extracting text from PDF documents
- Sending text to LLM for structured data extraction
- Aggregating and organizing extraction results
- Splitting results by product types

Note: This file was previously named 'test.py' but was renamed to avoid 
conflicts with Python's built-in 'test' module.
"""

import fitz  # PyMuPDF
import json
import logging
import os
import re
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# Import the LLM fallback for robust LLM usage
try:
    from llm_fallback import create_llm_with_fallback
except ImportError:
    create_llm_with_fallback = None

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global LLM instance with fallback
_llm = None


def _get_llm():
    """Get or create the LLM instance with fallback support."""
    global _llm
    if _llm is None:
        try:
            if create_llm_with_fallback:
                _llm = create_llm_with_fallback(
                    model="gemini-2.5-flash",
                    temperature=0,
                    google_api_key=os.getenv("GOOGLE_API_KEY")
                )
            else:
                _llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    temperature=0,
                    google_api_key=os.getenv("GOOGLE_API_KEY")
                )
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    return _llm


# Prompt template for extracting product specifications from PDF text
EXTRACTION_PROMPT = ChatPromptTemplate.from_template("""
You are an expert at extracting structured product specification data from industrial instrument datasheets.

Analyze the following text from a PDF datasheet and extract the product information in JSON format.

**Required Output Format:**
{{
    "product_type": "string (e.g., 'Pressure Transmitter', 'Temperature Sensor')",
    "vendor": "string (manufacturer name)",
    "model_series": "string (product series/family name)",
    "specifications": {{
        "key": "value pairs of all technical specifications found"
    }},
    "features": ["list of key features"],
    "applications": ["list of applications if mentioned"],
    "certifications": ["list of certifications if mentioned"],
    "materials": {{"component": "material"}},
    "ranges": {{"parameter": "range"}}
}}

**Important Guidelines:**
1. Extract ALL technical specifications you can find
2. Use standard units (convert if necessary)
3. If information is not found, omit the field (don't include null)
4. For numerical ranges, use format "min to max unit"
5. Extract vendor/manufacturer name accurately
6. Identify the correct product type category

**Text to analyze:**
{text}

Return ONLY valid JSON. Do not include any explanatory text before or after the JSON.
""")


def extract_data_from_pdf(pdf_bytes_or_path: Union[str, BytesIO, bytes]) -> List[str]:
    """
    Extract text content from a PDF file in chunks.
    
    Args:
        pdf_bytes_or_path: Either a file path string, BytesIO object, or raw bytes
        
    Returns:
        List of text strings, one per page or logical section
    """
    text_chunks = []
    doc = None
    
    try:
        # Handle different input types
        if isinstance(pdf_bytes_or_path, str):
            # It's a file path
            doc = fitz.open(pdf_bytes_or_path)
        elif isinstance(pdf_bytes_or_path, bytes):
            # It's raw bytes
            doc = fitz.open(stream=pdf_bytes_or_path, filetype="pdf")
        elif isinstance(pdf_bytes_or_path, BytesIO):
            # It's a BytesIO object
            pdf_bytes_or_path.seek(0)
            doc = fitz.open(stream=pdf_bytes_or_path.read(), filetype="pdf")
        else:
            raise ValueError(f"Unsupported input type: {type(pdf_bytes_or_path)}")
        
        logger.info(f"[PDF_EXTRACT] Processing PDF with {len(doc)} pages")
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text("text")
            
            # Skip pages with minimal content
            if page_text and len(page_text.strip()) > 50:
                # Clean up the text
                cleaned_text = re.sub(r'\s+', ' ', page_text).strip()
                text_chunks.append(cleaned_text)
                logger.info(f"[PDF_EXTRACT] Page {page_num + 1}: Extracted {len(cleaned_text)} characters")
            else:
                logger.debug(f"[PDF_EXTRACT] Page {page_num + 1}: Skipped (insufficient content)")
        
        logger.info(f"[PDF_EXTRACT] Total chunks extracted: {len(text_chunks)}")
        
    except Exception as e:
        logger.error(f"[PDF_EXTRACT] Failed to extract text from PDF: {e}")
        raise
    finally:
        if doc:
            doc.close()
    
    return text_chunks


def send_to_language_model(text_chunks: List[str]) -> List[Dict[str, Any]]:
    """
    Send text chunks to the LLM for structured data extraction.
    
    Args:
        text_chunks: List of text strings from PDF extraction
        
    Returns:
        List of extracted data dictionaries
    """
    results = []
    llm = _get_llm()
    
    # Create the extraction chain
    extraction_chain = EXTRACTION_PROMPT | llm | StrOutputParser()
    
    # Process chunks - combine small chunks for efficiency
    combined_text = ""
    CHUNK_SIZE_LIMIT = 15000  # Characters per LLM call
    
    for i, chunk in enumerate(text_chunks):
        combined_text += f"\n\n--- Page {i + 1} ---\n{chunk}"
        
        # If we've accumulated enough text or this is the last chunk
        if len(combined_text) >= CHUNK_SIZE_LIMIT or i == len(text_chunks) - 1:
            try:
                logger.info(f"[LLM_EXTRACT] Sending {len(combined_text)} characters to LLM")
                response = extraction_chain.invoke({"text": combined_text[:CHUNK_SIZE_LIMIT]})
                
                # Parse the JSON response
                try:
                    # Clean up the response - remove markdown code blocks if present
                    cleaned_response = response.strip()
                    if cleaned_response.startswith("```"):
                        cleaned_response = re.sub(r'^```[a-zA-Z]*\n?', '', cleaned_response)
                        cleaned_response = re.sub(r'\n?```$', '', cleaned_response)
                    
                    parsed_result = json.loads(cleaned_response)
                    results.append(parsed_result)
                    logger.info(f"[LLM_EXTRACT] Successfully parsed LLM response")
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"[LLM_EXTRACT] Failed to parse LLM response as JSON: {e}")
                    # Try to extract JSON from the response
                    json_match = re.search(r'\{[\s\S]*\}', response)
                    if json_match:
                        try:
                            parsed_result = json.loads(json_match.group())
                            results.append(parsed_result)
                            logger.info(f"[LLM_EXTRACT] Extracted JSON from response")
                        except json.JSONDecodeError:
                            logger.error(f"[LLM_EXTRACT] Could not extract valid JSON from response")
                    
            except Exception as e:
                logger.error(f"[LLM_EXTRACT] LLM call failed: {e}")
            
            combined_text = ""  # Reset for next batch
    
    logger.info(f"[LLM_EXTRACT] Total results extracted: {len(results)}")
    return results


def aggregate_results(results: List[Dict[str, Any]], product_type: str = "") -> Dict[str, Any]:
    """
    Aggregate multiple extraction results into a single structured result.
    
    Args:
        results: List of extracted data dictionaries
        product_type: Optional product type hint
        
    Returns:
        Aggregated result dictionary
    """
    if not results:
        return {
            "product_type": product_type or "Unknown",
            "vendor": "Unknown",
            "models": []
        }
    
    # Find the most common vendor
    vendors = [r.get("vendor", "") for r in results if r.get("vendor")]
    vendor = max(set(vendors), key=vendors.count) if vendors else "Unknown"
    
    # Find the most common product type
    product_types = [r.get("product_type", "") for r in results if r.get("product_type")]
    detected_product_type = max(set(product_types), key=product_types.count) if product_types else product_type
    
    # Aggregate models
    models = []
    for result in results:
        model_series = result.get("model_series", "")
        if not model_series:
            continue
            
        model_entry = {
            "model_series": model_series,
            "specifications": result.get("specifications", {}),
            "features": result.get("features", []),
            "applications": result.get("applications", []),
            "certifications": result.get("certifications", []),
            "materials": result.get("materials", {}),
            "ranges": result.get("ranges", {})
        }
        
        # Check if this model already exists
        existing = next((m for m in models if m["model_series"] == model_series), None)
        if existing:
            # Merge specifications
            existing["specifications"].update(model_entry["specifications"])
            existing["features"] = list(set(existing.get("features", []) + model_entry.get("features", [])))
            existing["applications"] = list(set(existing.get("applications", []) + model_entry.get("applications", [])))
            existing["certifications"] = list(set(existing.get("certifications", []) + model_entry.get("certifications", [])))
        else:
            models.append(model_entry)
    
    aggregated = {
        "product_type": detected_product_type or product_type or "Unknown",
        "vendor": vendor,
        "models": models
    }
    
    logger.info(f"[AGGREGATE] Aggregated {len(results)} results into {len(models)} models for vendor '{vendor}'")
    return aggregated


def split_product_types(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Split aggregated results by product type.
    
    Args:
        results: List of aggregated result dictionaries
        
    Returns:
        List of results split by product type
    """
    split_results = []
    
    for result in results:
        product_type = result.get("product_type", "Unknown")
        vendor = result.get("vendor", "Unknown")
        models = result.get("models", [])
        
        # If there's only one product type, just return as-is
        if models:
            split_results.append({
                "product_type": product_type,
                "vendor": vendor,
                "models": models
            })
    
    logger.info(f"[SPLIT] Split into {len(split_results)} product type groups")
    return split_results


def save_json(data: Dict[str, Any], path: str) -> str:
    """
    Save data as JSON to the specified path.
    
    Args:
        data: Dictionary to save
        path: File path to save to
        
    Returns:
        The path where the file was saved
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[SAVE_JSON] Saved JSON to: {path}")
        return path
        
    except Exception as e:
        logger.error(f"[SAVE_JSON] Failed to save JSON to {path}: {e}")
        raise


def generate_dynamic_path(base_dir: str, vendor: str, product_type: str, 
                          model_series: str, extension: str = ".json") -> str:
    """
    Generate a dynamic file path based on vendor, product type, and model.
    
    Args:
        base_dir: Base directory for the path
        vendor: Vendor name
        product_type: Product type
        model_series: Model series name
        extension: File extension (default: .json)
        
    Returns:
        Generated file path
    """
    # Sanitize names for file system
    def sanitize(name: str) -> str:
        # Replace problematic characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', name)
        sanitized = sanitized.strip().replace(' ', '_')
        return sanitized or "Unknown"
    
    safe_vendor = sanitize(vendor)
    safe_product_type = sanitize(product_type)
    safe_model = sanitize(model_series)
    
    path = os.path.join(base_dir, safe_vendor, safe_product_type, f"{safe_model}{extension}")
    
    logger.debug(f"[DYNAMIC_PATH] Generated path: {path}")
    return path


# Additional utility functions that might be needed

def extract_tables_from_pdf(pdf_bytes_or_path: Union[str, BytesIO, bytes]) -> List[Dict]:
    """
    Extract tables from a PDF using PyMuPDF's table detection.
    
    Args:
        pdf_bytes_or_path: PDF file path, bytes, or BytesIO object
        
    Returns:
        List of table dictionaries with headers and rows
    """
    tables = []
    doc = None
    
    try:
        if isinstance(pdf_bytes_or_path, str):
            doc = fitz.open(pdf_bytes_or_path)
        elif isinstance(pdf_bytes_or_path, bytes):
            doc = fitz.open(stream=pdf_bytes_or_path, filetype="pdf")
        elif isinstance(pdf_bytes_or_path, BytesIO):
            pdf_bytes_or_path.seek(0)
            doc = fitz.open(stream=pdf_bytes_or_path.read(), filetype="pdf")
        
        for page_num, page in enumerate(doc):
            page_tables = page.find_tables()
            if page_tables:
                for table in page_tables:
                    table_data = table.extract()
                    if table_data and len(table_data) > 1:
                        tables.append({
                            "page": page_num + 1,
                            "headers": table_data[0],
                            "rows": table_data[1:]
                        })
        
        logger.info(f"[TABLE_EXTRACT] Found {len(tables)} tables in PDF")
        
    except Exception as e:
        logger.error(f"[TABLE_EXTRACT] Failed to extract tables: {e}")
    finally:
        if doc:
            doc.close()
    
    return tables
