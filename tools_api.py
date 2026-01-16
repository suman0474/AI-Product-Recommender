# tools_api.py
# Flask Blueprint for LangChain Tool Wrapper APIs
#
# Exposes individual LangChain tools as REST endpoints for:
# - Direct frontend access
# - Testing and debugging
# - Consistent API architecture
#
# All endpoints require login via @login_required decorator.

import json
import logging
from typing import Dict, Any
from flask import Blueprint, request, jsonify, session
from functools import wraps

logger = logging.getLogger(__name__)

# Create Blueprint
tools_bp = Blueprint('tools', __name__, url_prefix='/api/tools')


# =============================================================================
# AUTHENTICATION DECORATOR
# =============================================================================

def login_required(func):
    """Require user to be logged in."""
    @wraps(func)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({"error": "Unauthorized: Please log in"}), 401
        return func(*args, **kwargs)
    return decorated_function


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def api_response(success: bool, data: Any = None, error: str = None, status_code: int = 200):
    """Create standardized API response."""
    response = {
        "success": success,
        "data": data,
        "error": error
    }
    return jsonify(response), status_code


def handle_errors(f):
    """Decorator for error handling."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Tool API Error: {e}", exc_info=True)
            return api_response(False, error=str(e), status_code=500)
    return decorated_function


# =============================================================================
# SCHEMA TOOLS
# =============================================================================

@tools_bp.route('/schema/load', methods=['POST'])
@login_required
@handle_errors
def load_schema():
    """
    Load Product Schema
    ---
    tags:
      - LangChain Tools
    summary: Load schema for a product type
    description: |
      Loads the requirements schema for a specific product type from Azure/MongoDB.
      Falls back to web-based schema generation if not found.
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - product_type
          properties:
            product_type:
              type: string
              example: "pressure transmitter"
            enable_ppi:
              type: boolean
              default: true
              description: Enable PPI workflow if schema not found
    responses:
      200:
        description: Schema loaded successfully
    """
    from tools.schema_tools import load_schema_tool
    
    data = request.get_json()
    product_type = data.get('product_type', '').strip()
    enable_ppi = data.get('enable_ppi', True)
    
    if not product_type:
        return api_response(False, error="product_type is required", status_code=400)
    
    logger.info(f"[TOOLS_API] Loading schema for: {product_type}")
    result = load_schema_tool.invoke({
        "product_type": product_type,
        "enable_ppi": enable_ppi
    })
    
    return api_response(True, data=result)


@tools_bp.route('/schema/validate', methods=['POST'])
@login_required
@handle_errors
def validate_requirements():
    """
    Validate User Requirements
    ---
    tags:
      - LangChain Tools
    summary: Validate requirements against schema
    description: Extracts and validates user requirements against the product schema.
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - user_input
            - product_type
          properties:
            user_input:
              type: string
              example: "I need a pressure transmitter with 0-100 bar range"
            product_type:
              type: string
              example: "pressure transmitter"
            product_schema:
              type: object
              description: Optional schema to validate against
    responses:
      200:
        description: Validation result
    """
    from tools.schema_tools import validate_requirements_tool
    
    data = request.get_json()
    user_input = data.get('user_input', '').strip()
    product_type = data.get('product_type', '').strip()
    product_schema = data.get('product_schema')
    
    if not user_input or not product_type:
        return api_response(False, error="user_input and product_type are required", status_code=400)
    
    logger.info(f"[TOOLS_API] Validating requirements for: {product_type}")
    result = validate_requirements_tool.invoke({
        "user_input": user_input,
        "product_type": product_type,
        "product_schema": product_schema
    })
    
    return api_response(True, data=result)


@tools_bp.route('/schema/missing-fields', methods=['POST'])
@login_required
@handle_errors
def get_missing_fields():
    """
    Get Missing Schema Fields
    ---
    tags:
      - LangChain Tools
    summary: Identify missing mandatory/optional fields
    description: Compares provided requirements against schema to find missing fields.
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - provided_requirements
            - product_schema
          properties:
            provided_requirements:
              type: object
            product_schema:
              type: object
    responses:
      200:
        description: Missing fields list
    """
    from tools.schema_tools import get_missing_fields_tool
    
    data = request.get_json()
    provided_requirements = data.get('provided_requirements', {})
    product_schema = data.get('product_schema', {})
    
    if not product_schema:
        return api_response(False, error="product_schema is required", status_code=400)
    
    result = get_missing_fields_tool.invoke({
        "provided_requirements": provided_requirements,
        "product_schema": product_schema
    })
    
    return api_response(True, data=result)


# =============================================================================
# ANALYSIS TOOLS
# =============================================================================

@tools_bp.route('/analysis/match', methods=['POST'])
@login_required
@handle_errors
def analyze_vendor_match():
    """
    Analyze Vendor Match
    ---
    tags:
      - LangChain Tools
    summary: Analyze vendor product match against requirements
    description: |
      Analyzes how well a vendor's product matches user requirements.
      Uses 3-tier matching: PDF with Model Selection Guide, PDF Basic, JSON.
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - vendor
            - requirements
          properties:
            vendor:
              type: string
              example: "Yokogawa"
            requirements:
              type: object
            pdf_content:
              type: string
              description: Optional PDF datasheet content
            product_data:
              type: object
              description: Optional JSON product data
            matching_tier:
              type: string
              enum: [auto, pdf_only, json_only]
              default: auto
    responses:
      200:
        description: Vendor match analysis
    """
    from tools.analysis_tools import analyze_vendor_match_tool
    
    data = request.get_json()
    vendor = data.get('vendor', '').strip()
    requirements = data.get('requirements', {})
    pdf_content = data.get('pdf_content')
    product_data = data.get('product_data')
    matching_tier = data.get('matching_tier', 'auto')
    
    if not vendor or not requirements:
        return api_response(False, error="vendor and requirements are required", status_code=400)
    
    logger.info(f"[TOOLS_API] Analyzing vendor match for: {vendor}")
    result = analyze_vendor_match_tool.invoke({
        "vendor": vendor,
        "requirements": requirements,
        "pdf_content": pdf_content,
        "product_data": product_data,
        "matching_tier": matching_tier
    })
    
    return api_response(True, data=result)


@tools_bp.route('/analysis/score', methods=['POST'])
@login_required
@handle_errors
def calculate_match_score():
    """
    Calculate Match Score
    ---
    tags:
      - LangChain Tools
    summary: Calculate match score between requirements and specs
    description: Uses weighted scoring based on requirement criticality.
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - requirements
            - product_specs
          properties:
            requirements:
              type: object
            product_specs:
              type: object
    responses:
      200:
        description: Match score result
    """
    from tools.analysis_tools import calculate_match_score_tool
    
    data = request.get_json()
    requirements = data.get('requirements', {})
    product_specs = data.get('product_specs', {})
    
    if not requirements or not product_specs:
        return api_response(False, error="requirements and product_specs are required", status_code=400)
    
    result = calculate_match_score_tool.invoke({
        "requirements": requirements,
        "product_specs": product_specs
    })
    
    return api_response(True, data=result)


@tools_bp.route('/analysis/extract-specs', methods=['POST'])
@login_required
@handle_errors
def extract_specifications():
    """
    Extract Specifications from PDF
    ---
    tags:
      - LangChain Tools
    summary: Extract specs from PDF datasheet content
    description: Returns structured specification data from PDF text.
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - pdf_content
            - product_type
          properties:
            pdf_content:
              type: string
            product_type:
              type: string
    responses:
      200:
        description: Extracted specifications
    """
    from tools.analysis_tools import extract_specifications_tool
    
    data = request.get_json()
    pdf_content = data.get('pdf_content', '').strip()
    product_type = data.get('product_type', '').strip()
    
    if not pdf_content or not product_type:
        return api_response(False, error="pdf_content and product_type are required", status_code=400)
    
    result = extract_specifications_tool.invoke({
        "pdf_content": pdf_content,
        "product_type": product_type
    })
    
    return api_response(True, data=result)


# =============================================================================
# RANKING TOOLS
# =============================================================================

@tools_bp.route('/ranking/rank', methods=['POST'])
@login_required
@handle_errors
def rank_products():
    """
    Rank Products
    ---
    tags:
      - LangChain Tools
    summary: Rank products based on vendor analysis
    description: Returns ordered list with scores and recommendations.
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - vendor_matches
            - requirements
          properties:
            vendor_matches:
              type: array
              items:
                type: object
            requirements:
              type: object
    responses:
      200:
        description: Ranked products
    """
    from tools.ranking_tools import rank_products_tool
    
    data = request.get_json()
    vendor_matches = data.get('vendor_matches', [])
    requirements = data.get('requirements', {})
    
    if not vendor_matches or not requirements:
        return api_response(False, error="vendor_matches and requirements are required", status_code=400)
    
    logger.info(f"[TOOLS_API] Ranking {len(vendor_matches)} vendor matches")
    result = rank_products_tool.invoke({
        "vendor_matches": vendor_matches,
        "requirements": requirements
    })
    
    return api_response(True, data=result)


@tools_bp.route('/ranking/judge', methods=['POST'])
@login_required
@handle_errors
def judge_analysis():
    """
    Judge Analysis
    ---
    tags:
      - LangChain Tools
    summary: Validate and judge vendor analysis results
    description: Checks for consistency, accuracy, and strategy compliance.
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - original_requirements
            - vendor_analysis
          properties:
            original_requirements:
              type: object
            vendor_analysis:
              type: object
            strategy_rules:
              type: object
              description: Optional procurement strategy rules
    responses:
      200:
        description: Judging result
    """
    from tools.ranking_tools import judge_analysis_tool
    
    data = request.get_json()
    original_requirements = data.get('original_requirements', {})
    vendor_analysis = data.get('vendor_analysis', {})
    strategy_rules = data.get('strategy_rules')
    
    if not original_requirements or not vendor_analysis:
        return api_response(False, error="original_requirements and vendor_analysis are required", status_code=400)
    
    result = judge_analysis_tool.invoke({
        "original_requirements": original_requirements,
        "vendor_analysis": vendor_analysis,
        "strategy_rules": strategy_rules
    })
    
    return api_response(True, data=result)


# =============================================================================
# SEARCH TOOLS
# =============================================================================

@tools_bp.route('/search/images', methods=['POST'])
@login_required
@handle_errors
def search_product_images():
    """
    Search Product Images
    ---
    tags:
      - LangChain Tools
    summary: Search for product images
    description: Uses multi-tier fallback (Google CSE, Serper, SerpAPI).
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - vendor
            - product_name
            - product_type
          properties:
            vendor:
              type: string
            product_name:
              type: string
            product_type:
              type: string
            model_family:
              type: string
    responses:
      200:
        description: Product images
    """
    from tools.search_tools import search_product_images_tool
    
    data = request.get_json()
    vendor = data.get('vendor', '').strip()
    product_name = data.get('product_name', '').strip()
    product_type = data.get('product_type', '').strip()
    model_family = data.get('model_family')
    
    if not vendor or not product_name or not product_type:
        return api_response(False, error="vendor, product_name, and product_type are required", status_code=400)
    
    result = search_product_images_tool.invoke({
        "vendor": vendor,
        "product_name": product_name,
        "product_type": product_type,
        "model_family": model_family
    })
    
    return api_response(True, data=result)


@tools_bp.route('/search/pdfs', methods=['POST'])
@login_required
@handle_errors
def search_pdf_datasheets():
    """
    Search PDF Datasheets
    ---
    tags:
      - LangChain Tools
    summary: Search for PDF datasheets
    description: Returns PDF URLs with metadata.
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - vendor
            - product_type
          properties:
            vendor:
              type: string
            product_type:
              type: string
            model_family:
              type: string
    responses:
      200:
        description: PDF datasheet URLs
    """
    from tools.search_tools import search_pdf_datasheets_tool
    
    data = request.get_json()
    vendor = data.get('vendor', '').strip()
    product_type = data.get('product_type', '').strip()
    model_family = data.get('model_family')
    
    if not vendor or not product_type:
        return api_response(False, error="vendor and product_type are required", status_code=400)
    
    result = search_pdf_datasheets_tool.invoke({
        "vendor": vendor,
        "product_type": product_type,
        "model_family": model_family
    })
    
    return api_response(True, data=result)


@tools_bp.route('/search/web', methods=['POST'])
@login_required
@handle_errors
def web_search():
    """
    Web Search
    ---
    tags:
      - LangChain Tools
    summary: General web search
    description: Uses multi-tier fallback for web search.
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - query
          properties:
            query:
              type: string
            num_results:
              type: integer
              default: 5
    responses:
      200:
        description: Web search results
    """
    from tools.search_tools import web_search_tool
    
    data = request.get_json()
    query = data.get('query', '').strip()
    num_results = data.get('num_results', 5)
    
    if not query:
        return api_response(False, error="query is required", status_code=400)
    
    result = web_search_tool.invoke({
        "query": query,
        "num_results": num_results
    })
    
    return api_response(True, data=result)


@tools_bp.route('/search/vendors', methods=['POST'])
@login_required
@handle_errors
def search_vendors():
    """
    Search Vendors
    ---
    tags:
      - LangChain Tools
    summary: Search for vendors by product type
    description: Returns list of vendors with available product families.
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - product_type
          properties:
            product_type:
              type: string
            requirements:
              type: object
    responses:
      200:
        description: Vendor list
    """
    from tools.search_tools import search_vendors_tool
    
    data = request.get_json()
    product_type = data.get('product_type', '').strip()
    requirements = data.get('requirements')
    
    if not product_type:
        return api_response(False, error="product_type is required", status_code=400)
    
    result = search_vendors_tool.invoke({
        "product_type": product_type,
        "requirements": requirements
    })
    
    return api_response(True, data=result)


@tools_bp.route('/search/vendor-products', methods=['POST'])
@login_required
@handle_errors
def get_vendor_products():
    """
    Get Vendor Products
    ---
    tags:
      - LangChain Tools
    summary: Get products from a specific vendor
    description: Returns product list with specifications.
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - vendor
            - product_type
          properties:
            vendor:
              type: string
            product_type:
              type: string
    responses:
      200:
        description: Vendor products
    """
    from tools.search_tools import get_vendor_products_tool
    
    data = request.get_json()
    vendor = data.get('vendor', '').strip()
    product_type = data.get('product_type', '').strip()
    
    if not vendor or not product_type:
        return api_response(False, error="vendor and product_type are required", status_code=400)
    
    result = get_vendor_products_tool.invoke({
        "vendor": vendor,
        "product_type": product_type
    })
    
    return api_response(True, data=result)


# =============================================================================
# INSTRUMENT TOOLS
# =============================================================================

@tools_bp.route('/instruments/identify', methods=['POST'])
@login_required
@handle_errors
def identify_instruments():
    """
    Identify Instruments
    ---
    tags:
      - LangChain Tools
    summary: Identify instruments from process requirements
    description: Extracts product types, specifications, and creates Bill of Materials.
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - requirements
          properties:
            requirements:
              type: string
              description: Process requirements or problem statement
    responses:
      200:
        description: Identified instruments
    """
    from tools.instrument_tools import identify_instruments_tool
    
    data = request.get_json()
    requirements = data.get('requirements', '').strip()
    
    if not requirements:
        return api_response(False, error="requirements is required", status_code=400)
    
    logger.info(f"[TOOLS_API] Identifying instruments from requirements")
    result = identify_instruments_tool.invoke({
        "requirements": requirements
    })
    
    return api_response(True, data=result)


@tools_bp.route('/instruments/accessories', methods=['POST'])
@login_required
@handle_errors
def identify_accessories():
    """
    Identify Accessories
    ---
    tags:
      - LangChain Tools
    summary: Identify accessories for instruments
    description: Returns list of accessories needed for the identified instruments.
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - instruments
          properties:
            instruments:
              type: array
              items:
                type: object
            process_context:
              type: string
    responses:
      200:
        description: Identified accessories
    """
    from tools.instrument_tools import identify_accessories_tool
    
    data = request.get_json()
    instruments = data.get('instruments', [])
    process_context = data.get('process_context')
    
    if not instruments:
        return api_response(False, error="instruments is required", status_code=400)
    
    result = identify_accessories_tool.invoke({
        "instruments": instruments,
        "process_context": process_context
    })
    
    return api_response(True, data=result)


# =============================================================================
# INTENT TOOLS
# =============================================================================

@tools_bp.route('/intent/extract-requirements', methods=['POST'])
@login_required
@handle_errors
def extract_requirements():
    """
    Extract Requirements
    ---
    tags:
      - LangChain Tools
    summary: Extract structured requirements from user input
    description: Uses LLM to extract technical requirements from natural language.
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - user_input
          properties:
            user_input:
              type: string
            product_type:
              type: string
    responses:
      200:
        description: Extracted requirements
    """
    from tools.intent_tools import extract_requirements_tool
    
    data = request.get_json()
    user_input = data.get('user_input', '').strip()
    product_type = data.get('product_type', '').strip()
    
    if not user_input:
        return api_response(False, error="user_input is required", status_code=400)
    
    result = extract_requirements_tool.invoke({
        "user_input": user_input,
        "product_type": product_type
    })
    
    return api_response(True, data=result)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = ['tools_bp']
