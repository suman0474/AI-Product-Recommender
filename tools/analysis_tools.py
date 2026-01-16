# tools/analysis_tools.py
# Product Analysis and Matching Tools

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

load_dotenv()
logger = logging.getLogger(__name__)


# ============================================================================
# TOOL INPUT SCHEMAS
# ============================================================================

class AnalyzeVendorMatchInput(BaseModel):
    """Input for vendor match analysis"""
    vendor: str = Field(description="Vendor name")
    requirements: Dict[str, Any] = Field(description="User requirements")
    pdf_content: Optional[str] = Field(default=None, description="PDF datasheet content")
    product_data: Optional[Dict[str, Any]] = Field(default=None, description="Product JSON data")
    matching_tier: Optional[str] = Field(default="auto", description="Matching tier preference: 'auto', 'pdf_only', 'json_only'")


class CalculateMatchScoreInput(BaseModel):
    """Input for match score calculation"""
    requirements: Dict[str, Any] = Field(description="User requirements")
    product_specs: Dict[str, Any] = Field(description="Product specifications")


class ExtractSpecificationsInput(BaseModel):
    """Input for specifications extraction"""
    pdf_content: str = Field(description="PDF datasheet content")
    product_type: str = Field(description="Product type to extract specs for")


# ============================================================================
# PROMPTS
# ============================================================================

VENDOR_ANALYSIS_PROMPT = """
You are Engenie - a meticulous procurement and technical matching expert.
Your task is to analyze user requirements against vendor product documentation.

**IMPORTANT: Think step-by-step through your analysis process.**

Before providing your final matching results:
1. First, identify and list all mandatory and optional requirements from the user input
2. Then, systematically check each requirement against the available documentation
3. Explain your reasoning for each match or mismatch with specific references
4. Finally, calculate an overall match score based on your findings

**CRITICAL: Model Family vs Product Name**

You MUST provide BOTH fields for every product match:

**1. product_name** = The EXACT model you are recommending (e.g., "STD850", "3051CD", "EJA118A")

**2. model_family** = The BASE series without variant suffixes (e.g., "STD800", "3051C", "EJA110")

**Simple Extraction Rules:**

**Rule 1: Remove the last 1-2 digits/letters from the model number**
- STD850 → STD800 (remove "50")
- STT850 → STT800 (remove "50")
- 3051CD → 3051C (remove "D")
- EJA118A → EJA110 (remove "8A")

**Rule 2: For compound names, keep only the main identifier**
- "SITRANS P DS III" → "SITRANS P" (remove variant "DS III")
- "Rosemount 3051CD" → "Rosemount 3051C" (remove variant "D")

**If unsure:** Use the product documentation's "series" or "family" name.

---

**Vendor:** {vendor}

**User Requirements:**
{requirements}

**Product Documentation (PDF Content):**
{pdf_content}

**Product Data (JSON):**
{product_data}

---

## Matching System

**Tier 1: Mandatory & Optional Specifications (From PDF)**
This is the primary tier and relies on the presence of a PDF document.

*Scenario A: Model Selection Guide EXISTS*
- Source: The main PDF document
- Logic:
  - Any specification listed within the Model Selection Guide is considered CRITICAL
  - All other specifications found in the rest of the PDF are considered OPTIONAL/SECONDARY
- Analysis Output: Parameter-by-parameter breakdown

*Scenario B: Model Selection Guide is MISSING*
- Source: The main PDF document
- Logic: Intelligently identify which specifications are mandatory/critical versus optional
  - Specifications essential for basic operation, safety, or core functionality are CRITICAL
  - Specifications that enhance performance or provide customization are OPTIONAL
  - Look for contextual clues: required vs available options, standard vs optional features
- Analysis Output: Parameter-by-parameter breakdown with reasoning

**Tier 2: Optional Specifications (Other Document Specs)**
Only applicable under Tier 1's "Scenario A" (when Model Selection Guide exists).
- Source: Specs in PDF not in Model Selection Guide
- Logic: All requirements matching specifications are automatically OPTIONAL/SECONDARY

**Tier 3: Fallback to JSON Product Summary**
Used only when no PDF document can be found.
- Source: The JSON product summary file
- Logic: Match against all available JSON data
  - CRITICAL if parameter is fundamental to product's core function
  - Otherwise OPTIONAL
- Analysis Output: Same parameter-by-parameter analysis as with PDFs

## Parameter-Level Analysis Requirements

For each parameter (mandatory or optional):
1. Show **Parameter Name** (User Requirement)
2. Show **Product Specification with source reference (PDF field or JSON field)**
3. Provide a **single holistic explanation paragraph** that:
  - Explains why the user requirement **matches** the product specification
  - Justifies the match using **datasheet or JSON evidence**
  - Describes how this requirement contributes to **overall suitability**
  - Mentions interactions with **other parameters** if relevant
  - Avoid breaking into subpoints; keep it integrated

**IMPORTANT - Empty Data Handling:**
- If BOTH PDF and JSON sources are empty or contain no valid product data, respond with:
  {{"vendor_matches": [], "error": "No valid product data available for analysis"}}

Return ONLY valid JSON:
{{
    "vendor": "{vendor}",
    "product_name": "<specific product model>",
    "model_family": "<product series/family>",
    "match_score": <0-100>,
    "requirements_match": <true if all mandatory requirements met>,
    "matched_requirements": {{
        "<requirement>": {{
            "user_value": "<what user requested>",
            "product_value": "<what product offers>",
            "status": "MATCHED|PARTIAL|UNMATCHED",
            "is_critical": <true|false>,
            "explanation": "<detailed holistic explanation>",
            "source_reference": "<PDF page/section or JSON field>"
        }}
    }},
    "unmatched_requirements": ["<list of unmatched requirements>"],
    "reasoning": "<overall reasoning for the match>",
    "limitations": "<any limitations or concerns>",
    "key_strengths": ["<list of key strengths>"]
}}
"""

SPEC_EXTRACTION_PROMPT = """
You are Engenie - an expert in industrial instrumentation datasheets.
Extract all technical specifications from this PDF content.

Product Type: {product_type}

PDF Content:
{pdf_content}

Extract:
- Model numbers and variants
- Measurement ranges
- Accuracy specifications
- Output signals and protocols
- Materials (wetted parts, housing)
- Process connections
- Environmental ratings
- Certifications

Return ONLY valid JSON:
{{
    "product_name": "<main product model>",
    "model_family": "<product series>",
    "specifications": {{
        "<spec_name>": "<spec_value>"
    }},
    "available_options": {{
        "<option_category>": ["<available options>"]
    }},
    "certifications": ["<list of certifications>"],
    "key_features": ["<list of key features>"]
}}
"""


# ============================================================================
# TOOLS
# ============================================================================

@tool("analyze_vendor_match", args_schema=AnalyzeVendorMatchInput)
def analyze_vendor_match_tool(
    vendor: str,
    requirements: Dict[str, Any],
    pdf_content: Optional[str] = None,
    product_data: Optional[Dict[str, Any]] = None,
    matching_tier: Optional[str] = "auto"
) -> Dict[str, Any]:
    """
    Analyze how well a vendor's product matches user requirements.
    Uses PDF datasheets and/or JSON product data for analysis with 3-tier matching system.

    Tier 1: PDF with Model Selection Guide (CRITICAL vs OPTIONAL)
    Tier 2: PDF without Model Selection Guide (Intelligent classification)
    Tier 3: JSON fallback (when no PDF available)

    Returns detailed parameter-by-parameter analysis with match score.
    Includes model family extraction (e.g., STD850 → STD800, 3051CD → 3051C).
    """
    try:
        # Validate inputs
        if not pdf_content and not product_data:
            return {
                "success": False,
                "vendor": vendor,
                "error": "No product data available for analysis"
            }

        llm = create_llm_with_fallback(
            model="gemini-2.5-flash",
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        prompt = ChatPromptTemplate.from_template(VENDOR_ANALYSIS_PROMPT)
        parser = JsonOutputParser()

        chain = prompt | llm | parser

        result = chain.invoke({
            "vendor": vendor,
            "requirements": json.dumps(requirements, indent=2),
            "pdf_content": pdf_content or "No PDF content available",
            "product_data": json.dumps(product_data, indent=2) if product_data else "No JSON data available"
        })

        return {
            "success": True,
            "vendor": vendor,
            "product_name": result.get("product_name"),
            "model_family": result.get("model_family"),
            "match_score": result.get("match_score", 0),
            "requirements_match": result.get("requirements_match", False),
            "matched_requirements": result.get("matched_requirements", {}),
            "unmatched_requirements": result.get("unmatched_requirements", []),
            "reasoning": result.get("reasoning"),
            "limitations": result.get("limitations"),
            "key_strengths": result.get("key_strengths", [])
        }

    except Exception as e:
        logger.error(f"Vendor analysis failed for {vendor}: {e}")
        return {
            "success": False,
            "vendor": vendor,
            "match_score": 0,
            "error": str(e)
        }


@tool("calculate_match_score", args_schema=CalculateMatchScoreInput)
def calculate_match_score_tool(
    requirements: Dict[str, Any],
    product_specs: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calculate a match score between requirements and product specifications.
    Uses weighted scoring based on requirement criticality.
    """
    try:
        total_weight = 0
        matched_weight = 0
        matches = {}
        mismatches = []

        for req_key, req_value in requirements.items():
            # Determine weight (critical requirements have higher weight)
            is_critical = req_key in ["outputSignal", "pressureRange", "processConnection"]
            weight = 2 if is_critical else 1
            total_weight += weight

            # Check if requirement is in product specs
            if req_key in product_specs:
                # Simple string matching (could be enhanced with semantic matching)
                product_value = str(product_specs[req_key]).lower()
                req_value_lower = str(req_value).lower()

                if req_value_lower in product_value or product_value in req_value_lower:
                    matched_weight += weight
                    matches[req_key] = {
                        "required": req_value,
                        "provided": product_specs[req_key],
                        "matched": True
                    }
                else:
                    matches[req_key] = {
                        "required": req_value,
                        "provided": product_specs[req_key],
                        "matched": False
                    }
                    mismatches.append(req_key)
            else:
                mismatches.append(req_key)

        # Calculate score
        score = (matched_weight / total_weight * 100) if total_weight > 0 else 0

        return {
            "success": True,
            "match_score": round(score, 2),
            "total_requirements": len(requirements),
            "matched_count": len(requirements) - len(mismatches),
            "matches": matches,
            "mismatches": mismatches,
            "all_critical_matched": all(
                req not in mismatches
                for req in ["outputSignal", "pressureRange", "processConnection"]
                if req in requirements
            )
        }

    except Exception as e:
        logger.error(f"Score calculation failed: {e}")
        return {
            "success": False,
            "match_score": 0,
            "error": str(e)
        }


@tool("extract_specifications", args_schema=ExtractSpecificationsInput)
def extract_specifications_tool(
    pdf_content: str,
    product_type: str
) -> Dict[str, Any]:
    """
    Extract technical specifications from PDF datasheet content.
    Returns structured specification data.
    """
    try:
        llm = create_llm_with_fallback(
            model="gemini-2.5-flash",
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        prompt = ChatPromptTemplate.from_template(SPEC_EXTRACTION_PROMPT)
        parser = JsonOutputParser()

        chain = prompt | llm | parser

        result = chain.invoke({
            "pdf_content": pdf_content[:50000],  # Limit content length
            "product_type": product_type
        })

        return {
            "success": True,
            "product_name": result.get("product_name"),
            "model_family": result.get("model_family"),
            "specifications": result.get("specifications", {}),
            "available_options": result.get("available_options", {}),
            "certifications": result.get("certifications", []),
            "key_features": result.get("key_features", [])
        }

    except Exception as e:
        logger.error(f"Specification extraction failed: {e}")
        return {
            "success": False,
            "specifications": {},
            "error": str(e)
        }
