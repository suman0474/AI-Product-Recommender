# agentic/api.py
# Flask API Endpoints for Agentic Workflow
#
# ARCHITECTURE PRINCIPLE:
# - All workflow execution MUST go through API endpoints
# - Direct workflow function calls ONLY allowed within endpoint view functions
# - Orchestration code (router, chainers) MUST use internal_api.api_client
# - This ensures complete decoupling between workflows
#
# This module exposes LangGraph workflows as REST API endpoints that can be
# called by the UI or internally by other workflows through the api_client.


import json
import logging
import uuid
import threading
import time as time_module
from typing import Dict, Any, List
from flask import Blueprint, request, jsonify, session
from functools import wraps

# Import rate limiting
from rate_limiter import get_limiter
from rate_limit_config import RateLimitConfig

# Define login_required decorator locally to avoid circular import
def login_required(func):
    @wraps(func)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({"error": "Unauthorized: Please log in"}), 401
        return func(*args, **kwargs)
    return decorated_function


# Import tags module for response tagging
from tags import classify_response, ResponseTags

from .workflow import run_workflow
from .solution_workflow import run_solution_workflow
# Comparison logic consolidated into product_search_workflow
from .instrument_identifier_workflow import run_instrument_identifier_workflow
from .potential_product_index import run_potential_product_index_workflow

# Streaming endpoints are in api_streaming.py

# Import internal API client for workflow orchestration
from .internal_api import api_client
from .models import create_initial_state, IntentType, WorkflowType

logger = logging.getLogger(__name__)

# Create Blueprint
agentic_bp = Blueprint('agentic', __name__, url_prefix='/api/agentic')


# ============================================================================
# SERVER-SIDE WORKFLOW STATE STORAGE
# Replaces Flask session to fix concurrent tab issues (cookie overwrite)
# ============================================================================
_workflow_states: Dict[str, Dict[str, Any]] = {}
_workflow_states_lock = threading.Lock()
_WORKFLOW_STATE_TTL = 3600  # 1 hour


def get_workflow_state(thread_id: str) -> Dict[str, Any]:
    """Get workflow state for a thread (thread-safe)."""
    with _workflow_states_lock:
        state = _workflow_states.get(thread_id, {})
        if state:
            logger.debug(f"[WORKFLOW_STATE] Retrieved state for {thread_id}: phase={state.get('phase')}")
        return state.copy() if state else {}


def set_workflow_state(thread_id: str, state: Dict[str, Any]) -> None:
    """Save workflow state for a thread (thread-safe)."""
    with _workflow_states_lock:
        state['_last_updated'] = time_module.time()
        _workflow_states[thread_id] = state.copy()
        logger.debug(f"[WORKFLOW_STATE] Saved state for {thread_id}: phase={state.get('phase')}")


def cleanup_expired_workflow_states() -> int:
    """Remove states older than TTL. Returns count removed."""
    with _workflow_states_lock:
        now = time_module.time()
        expired = [k for k, v in _workflow_states.items() 
                   if now - v.get('_last_updated', 0) > _WORKFLOW_STATE_TTL]
        for k in expired:
            del _workflow_states[k]
        if expired:
            logger.info(f"[WORKFLOW_STATE] Cleaned up {len(expired)} expired states")
        return len(expired)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_session_id() -> str:
    """Get or create session ID"""
    if 'agentic_session_id' not in session:
        session['agentic_session_id'] = str(uuid.uuid4())
    return session['agentic_session_id']


def api_response(success: bool, data: Any = None, error: str = None, status_code: int = 200, tags: ResponseTags = None):
    """
    Create standardized API response with optional tags.

    Args:
        success: Whether the request was successful
        data: Response data
        error: Error message (if any)
        status_code: HTTP status code
        tags: Optional ResponseTags object for frontend routing/UI hints

    Returns:
        JSON response with optional tags field
    """
    response = {
        "success": success,
        "data": data,
        "error": error
    }

    # Add tags if provided (backward compatible - only adds if tags exist)
    if tags is not None:
        response["tags"] = tags.dict()

    return jsonify(response), status_code


def handle_errors(f):
    """Decorator for error handling"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"API Error: {e}")
            return api_response(False, error=str(e), status_code=500)
    return decorated_function


# Rate limit decorator helpers
def get_rate_limit_decorator(limit_type):
    """
    Get a rate limit decorator for the specified type.

    Args:
        limit_type: Type of limit ('agentic_workflow', 'agentic_tool', etc.)

    Returns:
        Decorator function or no-op if limiter not available
    """
    limiter = get_limiter()
    if not limiter:
        return lambda f: f  # No-op if limiter not available

    limits = RateLimitConfig.LIMITS.get(limit_type, RateLimitConfig.DEFAULT_LIMITS)
    return limiter.limit(limits)


# Convenience decorator functions
workflow_limited = lambda f: get_rate_limit_decorator('agentic_workflow')(f) if get_limiter() else f
tool_limited = lambda f: get_rate_limit_decorator('agentic_tool')(f) if get_limiter() else f
session_limited = lambda f: get_rate_limit_decorator('session_management')(f) if get_limiter() else f
health_limited = lambda f: get_rate_limit_decorator('health')(f) if get_limiter() else f


# ============================================================================
# ROUTER ENDPOINTS
# ============================================================================

@agentic_bp.route('/classify-route', methods=['POST'])
@login_required
@handle_errors
def classify_route():
    """
    Classify Query and Route to Workflow
    ---
    tags:
      - LangChain Agents
    summary: Classify user input and route to appropriate workflow
    description: |
      Uses IntentClassificationRoutingAgent to classify user queries from the UI textarea
      and determine which workflow to route to:
      - solution: Complex systems requiring multiple instruments
      - instrument_identifier: Single product requirements
      - product_info: Questions about products/standards
      - out_of_domain: Unrelated queries (rejected with helpful message)
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
              description: User query from UI textarea
              example: "I need a pressure transmitter 0-100 PSI"
            context:
              type: object
              description: Optional context (current_step, conversation history)
    responses:
      200:
        description: Workflow routing decision
        schema:
          type: object
          properties:
            success:
              type: boolean
            data:
              type: object
              properties:
                target_workflow:
                  type: string
                  enum: [solution, instrument_identifier, product_info, out_of_domain]
                intent:
                  type: string
                  description: Raw intent from classify_intent_tool
                confidence:
                  type: number
                reasoning:
                  type: string
                is_solution:
                  type: boolean
                reject_message:
                  type: string
                  description: Message for out-of-domain queries
    """
    from .intent_classification_routing_agent import IntentClassificationRoutingAgent
    
    data = request.get_json()
    query = data.get('query', '').strip()
    context = data.get('context')
    
    if not query:
        return api_response(False, error="query is required", status_code=400)
    
    logger.info(f"[CLASSIFY_ROUTE] Classifying query: {query[:100]}...")
    
    agent = IntentClassificationRoutingAgent()
    result = agent.classify(query, context)
    
    return api_response(True, data=result.to_dict())


@agentic_bp.route('/product-info-decision', methods=['POST'])
@login_required
@handle_errors
def product_info_decision():
    """
    Get Product Info Page Routing Decision
    ---
    tags:
      - LangChain Agents
    summary: Determine if query should route to Product Info page
    description: |
      Uses ProductInfoIntentAgent to make detailed routing decisions for
      the Product Info page in the frontend.
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
              description: User query to analyze
              example: "Show me Yokogawa pressure transmitter models"
    responses:
      200:
        description: Routing decision
        schema:
          type: object
          properties:
            success:
              type: boolean
            data:
              type: object
              properties:
                should_route:
                  type: boolean
                confidence:
                  type: number
                data_source:
                  type: string
                sources:
                  type: array
                  items:
                    type: string
                reasoning:
                  type: string
    """
    from .product_info_intent_agent import get_product_info_route_decision
    
    data = request.get_json()
    query = data.get('query', '').strip()
    
    if not query:
        return api_response(False, error="query is required", status_code=400)
    
    logger.info(f"[PRODUCT_INFO_DECISION] Analyzing query: {query[:100]}...")
    
    result = get_product_info_route_decision(query)
    
    return api_response(True, data=result)





@agentic_bp.route('/validate-product-input', methods=['POST'])
@login_required
@handle_errors
def validate_product_input():
    """
    Validation API - Step 1: Detect Product Type from User Input

    Matches main.py /validate-product-type implementation.
    Uses session management and same helper functions as main.py.

    Request:
        {
            "user_input": "I need a pressure transmitter with 0-100 bar range",
            "search_session_id": "optional_session_id"
        }

    Response:
        {
            "success": true,
            "data": {
                "productType": "pressure transmitter",
                "confidence": 0.9,
                "reasoning": "Detected from user input analysis",
                "normalizedInput": "...",
                "sessionId": "session_id"
            }
        }
    """
    data = request.get_json()

    logger.info("=" * 60)
    logger.info("[VALIDATE] Product Type Detection API Called")

    user_input = data.get('user_input') or data.get('message')

    if not user_input:
        logger.error("[VALIDATE] No user_input provided")
        return api_response(False, error="user_input is required", status_code=400)

    search_session_id = data.get('search_session_id', get_session_id())

    logger.info(f"[VALIDATE] Session {search_session_id}: Detecting product type")
    logger.info(f"[VALIDATE] User input: {user_input[:100]}...")

    try:
        # Import from main.py's loading module (same as main.py uses)
        from loading import load_requirements_schema

        # Load initial generic schema for product type detection
        initial_schema = load_requirements_schema()

        # Get the validation components from main app if available
        # This ensures we use the same LLM chain as main.py
        try:
            from main import components
            if not components:
                raise Exception("Backend components not ready")
        except:
            # Fallback: use our own LLM if main components not available
            from llm_fallback import create_llm_with_fallback
            from langchain_core.output_parsers import JsonOutputParser
            from langchain_core.prompts import ChatPromptTemplate
            import os
            import json

            detection_prompt = ChatPromptTemplate.from_template("""
You are an expert validator. Extract the product type from user input.

User Input: {user_input}

Return JSON with 'product_type' field containing the detected product type in lowercase.
""")

            llm = create_llm_with_fallback(model="gemini-2.5-flash", temperature=0.1)
            parser = JsonOutputParser()
            chain = detection_prompt | llm | parser

            detection_result = chain.invoke({"user_input": user_input})
            components = {'validation_chain': chain, 'validation_format_instructions': ''}

        # Add session context to prevent cross-contamination (same as main.py)
        session_isolated_input = f"[Session: {search_session_id}] - Product type detection. User input: {user_input}"

        # Use validation chain to detect product type (same as main.py)
        if hasattr(components.get('validation_chain'), 'invoke'):
            detection_result = components['validation_chain'].invoke({
                "user_input": session_isolated_input,
                "schema": json.dumps(initial_schema, indent=2),
                "format_instructions": components.get('validation_format_instructions', '')
            })
        else:
            # Fallback
            detection_result = {'product_type': 'unknown'}

        detected_type = detection_result.get('product_type', 'UnknownProduct')

        logger.info(f"[VALIDATE] Detected product type: {detected_type}")

        # Store in session for later use (same as main.py)
        session[f'product_type_{search_session_id}'] = detected_type
        session[f'log_user_query_{search_session_id}'] = user_input

        response_data = {
            "productType": detected_type,
            "confidence": 0.9,
            "reasoning": "Detected from user input analysis",
            "normalizedInput": user_input,
            "sessionId": search_session_id
        }

        return api_response(True, data=response_data)

    except Exception as e:
        logger.error(f"[VALIDATE] Product type detection failed: {e}")
        import traceback
        logger.error(f"[VALIDATE] Traceback: {traceback.format_exc()}")
        return api_response(False, error=str(e), status_code=500)


@agentic_bp.route('/get-product-schema', methods=['POST'])
@login_required
@handle_errors
def get_product_schema():
    """
    Schema Get API - Step 2: Get Schema and Map with User Input

    Matches main.py /validate implementation pattern.
    Uses same helper functions: convert_keys_to_camel_case, map_provided_to_schema, clean_empty_values.

    Request:
        {
            "product_type": "pressure transmitter",
            "user_input": "I need a pressure transmitter with 0-100 bar range",
            "search_session_id": "optional_session_id"
        }

    Response:
        {
            "success": true,
            "data": {
                "productType": "pressure transmitter",
                "detectedSchema": {
                    "mandatoryRequirements": {...},
                    "optionalRequirements": {...}
                },
                "providedRequirements": {...},
                "missingMandatory": ["outputSignal"],
                "validationAlert": {
                    "message": "...",
                    "canContinue": true,
                    "missingFields": [...]
                }
            }
        }
    """
    data = request.get_json()

    logger.info("=" * 60)
    logger.info("[SCHEMA_GET] Schema Retrieval and Mapping API Called")

    product_type = data.get('product_type')
    user_input = data.get('user_input') or data.get('message')
    search_session_id = data.get('search_session_id', get_session_id())

    if not product_type:
        logger.error("[SCHEMA_GET] No product_type provided")
        return api_response(False, error="product_type is required", status_code=400)

    if not user_input:
        logger.error("[SCHEMA_GET] No user_input provided")
        return api_response(False, error="user_input is required", status_code=400)

    logger.info(f"[SCHEMA_GET] Product type: {product_type}")
    logger.info(f"[SCHEMA_GET] User input: {user_input[:100]}...")

    try:
        # Import helper functions from main.py
        import sys
        import re
        import copy
        import json

        # Helper functions (same as main.py)
        def convert_keys_to_camel_case(obj):
            """Recursively converts dictionary keys from snake_case to camelCase."""
            if isinstance(obj, dict):
                new_dict = {}
                for key, value in obj.items():
                    camel_key = re.sub(r'_([a-z])', lambda m: m.group(1).upper(), key)
                    new_dict[camel_key] = convert_keys_to_camel_case(value)
                return new_dict
            elif isinstance(obj, list):
                return [convert_keys_to_camel_case(item) for item in obj]
            return obj

        def clean_empty_values(data):
            """Recursively replaces 'Not specified', etc., with empty strings."""
            if isinstance(data, dict):
                return {k: clean_empty_values(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [clean_empty_values(item) for item in data]
            elif isinstance(data, str) and data.lower().strip() in ["not specified", "not requested", "none specified", "n/a", "na"]:
                return ""
            return data

        def map_provided_to_schema(detected_schema: dict, provided: dict) -> dict:
            """Maps providedRequirements into the schema structure."""
            mapped = copy.deepcopy(detected_schema)
            if "mandatoryRequirements" in provided or "optionalRequirements" in provided:
                for section in ["mandatoryRequirements", "optionalRequirements"]:
                    if section in provided and section in mapped:
                        for key, value in provided[section].items():
                            if key in mapped[section]:
                                mapped[section][key] = value
                return mapped
            for key, value in provided.items():
                if key in mapped.get("mandatoryRequirements", {}):
                    mapped["mandatoryRequirements"][key] = value
                elif key in mapped.get("optionalRequirements", {}):
                    mapped["optionalRequirements"][key] = value
            return mapped

        # Load schema (same as main.py)
        from loading import load_requirements_schema, build_requirements_schema_from_web

        logger.info("[SCHEMA_GET] Loading schema...")
        specific_schema = load_requirements_schema(product_type)

        if not specific_schema or (not specific_schema.get("mandatory_requirements") and not specific_schema.get("optional_requirements")):
            logger.warning(f"[SCHEMA_GET] Schema not found, building from web for {product_type}")
            try:
                specific_schema = build_requirements_schema_from_web(product_type)
            except Exception as build_error:
                logger.error(f"[SCHEMA_GET] Web schema build failed: {build_error}")
                specific_schema = {
                    "mandatory_requirements": {},
                    "optional_requirements": {}
                }

        # Get validation components (same pattern as main.py)
        try:
            from main import components
            if not components:
                raise Exception("Components not ready")
        except:
            # Fallback if main components not available
            components = None

        # Add session context (same as main.py)
        session_isolated_input = f"[Session: {search_session_id}] - Schema validation. User input: {user_input}"

        # Validate using chain if available
        if components and hasattr(components.get('validation_chain'), 'invoke'):
            validation_result = components['validation_chain'].invoke({
                "user_input": session_isolated_input,
                "schema": json.dumps(specific_schema, indent=2),
                "format_instructions": components.get('validation_format_instructions', '')
            })
        else:
            # Fallback: basic extraction
            validation_result = {
                "product_type": product_type,
                "provided_requirements": {}
            }

        # Clean and map (same as main.py)
        cleaned_provided_reqs = clean_empty_values(validation_result.get("provided_requirements", {}))

        mapped_provided_reqs = map_provided_to_schema(
            convert_keys_to_camel_case(specific_schema),
            convert_keys_to_camel_case(cleaned_provided_reqs)
        )

        # Build response (same structure as main.py)
        response_data = {
            "productType": validation_result.get("product_type", product_type),
            "detectedSchema": convert_keys_to_camel_case(specific_schema),
            "providedRequirements": mapped_provided_reqs
        }

        # Get missing mandatory fields (same logic as main.py)
        def get_missing_mandatory_fields(provided: dict, schema: dict) -> list:
            missing = []
            mandatory_schema = schema.get("mandatoryRequirements", {})
            provided_mandatory = provided.get("mandatoryRequirements", {})

            def traverse_and_check(schema_node, provided_node):
                for key, schema_value in schema_node.items():
                    if isinstance(schema_value, dict):
                        traverse_and_check(schema_value, provided_node.get(key, {}) if isinstance(provided_node, dict) else {})
                    else:
                        provided_value = provided_node.get(key) if isinstance(provided_node, dict) else None
                        if provided_value is None or str(provided_value).strip() in ["", ","]:
                            missing.append(key)

            traverse_and_check(mandatory_schema, provided_mandatory)
            return missing

        missing_mandatory_fields = get_missing_mandatory_fields(
            mapped_provided_reqs, response_data["detectedSchema"]
        )

        # Add validation alert if missing fields (same as main.py)
        if missing_mandatory_fields:
            def friendly_field_name(field):
                s1 = re.sub('([a-z0-9])([A-Z])', r'\1 \2', field)
                return s1.replace("_", " ").title()

            missing_fields_friendly = [friendly_field_name(f) for f in missing_mandatory_fields]
            missing_fields_str = ", ".join(missing_fields_friendly)

            response_data["validationAlert"] = {
                "message": f"Please provide the following required information: {missing_fields_str}",
                "canContinue": True,
                "missingFields": missing_mandatory_fields
            }
            response_data["missingMandatory"] = missing_mandatory_fields

        logger.info(f"[SCHEMA_GET] Extracted {len(cleaned_provided_reqs)} requirements")
        logger.info(f"[SCHEMA_GET] Missing mandatory: {len(missing_mandatory_fields)}")

        # Store in session (same as main.py)
        session[f'product_type_{search_session_id}'] = response_data["productType"]

        return api_response(True, data=response_data)

    except Exception as e:
        logger.error(f"[SCHEMA_GET] Schema retrieval and mapping failed: {e}")
        import traceback
        logger.error(f"[SCHEMA_GET] Traceback: {traceback.format_exc()}")
        return api_response(False, error=str(e), status_code=500)


# ============================================================================
# WORKFLOW ENDPOINTS
# ============================================================================

@agentic_bp.route('/chat', methods=['POST'])
@login_required
@handle_errors
def chat():
    """
    Main Chat Endpoint for Agentic Workflow
    ---
    tags:
      - Agentic Workflows
    summary: Process user message through agentic workflow
    description: |
      Main entry point for conversational AI workflows.
      Supports multiple workflow types including procurement and instrument identification.
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - message
          properties:
            message:
              type: string
              description: User message to process
              example: "I need pressure transmitters for a crude oil refinery"
            session_id:
              type: string
              description: Optional session ID for conversation continuity
              example: "abc123-session"
            workflow_type:
              type: string
              enum: [procurement, instrument_identification]
              default: procurement
              description: Type of workflow to run
    responses:
      200:
        description: Successful response from workflow
        schema:
          type: object
          properties:
            success:
              type: boolean
              example: true
            data:
              type: object
              properties:
                response:
                  type: string
                  description: Agent response text
                intent:
                  type: string
                  description: Classified intent
                product_type:
                  type: string
                  description: Detected product type
                requires_user_input:
                  type: boolean
                current_step:
                  type: string
      400:
        description: Bad request - missing required fields
    """
    data = request.get_json()

    if not data or 'message' not in data:
        return api_response(False, error="Message is required", status_code=400)

    message = data['message']
    session_id = data.get('session_id') or get_session_id()
    workflow_type = data.get('workflow_type', 'procurement')

    # Run workflow
    result = run_workflow(
        user_input=message,
        session_id=session_id,
        workflow_type=workflow_type
    )

    # Classify response and generate tags
    tags = classify_response(
        user_input=message,
        response_data=result,
        workflow_type=workflow_type
    )

    return api_response(True, data=result, tags=tags)


@agentic_bp.route('/identify', methods=['POST'])
@login_required
@handle_errors
def identify_instruments():
    """
    Identify instruments from process requirements

    Request Body:
    {
        "requirements": "process description or requirements text"
    }

    Response:
    {
        "success": true,
        "data": {
            "project_name": "...",
            "instruments": [...],
            "accessories": [...],
            "summary": "..."
        }
    }
    """
    data = request.get_json()

    if not data or 'requirements' not in data:
        return api_response(False, error="Requirements are required", status_code=400)

    requirements = data['requirements']
    session_id = data.get('session_id') or get_session_id()

    # Run instrument identification workflow
    result = run_workflow(
        user_input=requirements,
        session_id=session_id,
        workflow_type='instrument_identification'
    )

    # Classify response and generate tags
    tags = classify_response(
        user_input=requirements,
        response_data=result,
        workflow_type='instrument_identification'
    )

    return api_response(True, data=result, tags=tags)


@agentic_bp.route('/analyze', methods=['POST'])
@login_required
@handle_errors
def analyze_requirements():
    """
    Analyze requirements and run full procurement workflow

    Request Body:
    {
        "requirements": "technical requirements",
        "vendor_filter": ["optional", "vendor", "list"]
    }

    Response:
    {
        "success": true,
        "data": {
            "response": "...",
            "ranked_products": [...],
            "vendor_analysis": {...}
        }
    }
    """
    data = request.get_json()

    if not data or 'requirements' not in data:
        return api_response(False, error="Requirements are required", status_code=400)

    requirements = data['requirements']
    vendor_filter = data.get('vendor_filter')
    session_id = data.get('session_id') or get_session_id()

    # Store vendor filter in session if provided
    if vendor_filter:
        session['csv_vendor_filter'] = {
            'vendor_names': vendor_filter
        }

    # Run procurement workflow
    result = run_workflow(
        user_input=requirements,
        session_id=session_id,
        workflow_type='procurement'
    )

    # Classify response and generate tags
    tags = classify_response(
        user_input=requirements,
        response_data=result,
        workflow_type='procurement'
    )

    return api_response(True, data=result, tags=tags)



@agentic_bp.route('/run-analysis', methods=['POST'])
@login_required
@handle_errors
def run_analysis_endpoint():
    """
    Run Final Product Analysis (Steps 4-5)
    ---
    tags:
      - Product Search Workflow
    summary: Execute vendor analysis and ranking
    description: |
      Runs the analysis phase of the Product Search Workflow.
      Called after requirements collection is complete.
      Performs:
      1. Vendor Analysis (parallel matching)
      2. Product Ranking (scoring and recommendation)
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - structured_requirements
            - product_type
          properties:
            structured_requirements:
              type: object
              description: Complete collected requirements
            product_type:
              type: string
              description: Detected product type
            schema:
              type: object
              description: Optional product schema
            session_id:
              type: string
    responses:
      200:
        description: Analysis and ranking results
        schema:
          type: object
          properties:
            success:
              type: boolean
            data:
              type: object
    """
    data = request.get_json()

    if not data or 'structured_requirements' not in data or 'product_type' not in data:
        return api_response(False, error="structured_requirements and product_type are required", status_code=400)

    structured_requirements = data['structured_requirements']
    product_type = data['product_type']
    schema = data.get('schema')
    session_id = data.get('session_id') or get_session_id()
    
    logger.info(f"[RUN_ANALYSIS] Session: {session_id}")
    logger.info(f"[RUN_ANALYSIS] Product Type: {product_type}")

    try:
        from product_search_workflow.workflow import ProductSearchWorkflow
        
        # Initialize workflow
        workflow = ProductSearchWorkflow(enable_ppi_workflow=True, auto_mode=True)
        
        # Run analysis only
        result = workflow.run_analysis_only(
            structured_requirements=structured_requirements,
            product_type=product_type,
            schema=schema,
            session_id=session_id
        )
        
        if not result.get('success'):
            return api_response(False, error=result.get('error', 'Analysis failed'), status_code=500)
            
        # Classify response and generate tags
        tags = classify_response(
            user_input=f"Analyze {product_type} requirements",
            response_data=result,
            workflow_type='product_search'
        )
        
        return api_response(True, data=result, tags=tags)

    except ImportError:
        logger.error("Could not import ProductSearchWorkflow make sure product_search_workflow module is available")
        return api_response(False, error="Backend configuration error", status_code=500)
    except Exception as e:
        logger.error(f"[RUN_ANALYSIS] Failed: {e}", exc_info=True)
        return api_response(False, error=str(e), status_code=500)


# ============================================================================
# ENHANCED WORKFLOW ENDPOINTS
# ============================================================================

@agentic_bp.route('/solution', methods=['POST'])
@login_required
@handle_errors
def solution_workflow():
    """
    Solution-Based Workflow
    ---
    tags:
      - Enhanced Workflows
    summary: Run the Solution-Based workflow for design requests
    description: |
      Takes user requirements and produces ranked product recommendations.
      Uses RAG for strategy, standards, and inventory constraints.
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - message
          properties:
            message:
              type: string
              description: Product requirements
              example: "Need pressure transmitters for crude oil unit, SIL2 rated"
            session_id:
              type: string
              description: Optional session ID
    responses:
      200:
        description: Ranked product recommendations
        schema:
          type: object
          properties:
            success:
              type: boolean
            data:
              type: object
              properties:
                response:
                  type: string
                ranked_results:
                  type: array
                  items:
                    type: object
                product_type:
                  type: string
                strategy_present:
                  type: boolean
    """
    data = request.get_json()

    if not data or 'message' not in data:
        return api_response(False, error="Message is required", status_code=400)

    message = data['message']
    session_id = data.get('session_id') or get_session_id()

    result = run_solution_workflow(
        user_input=message,
        session_id=session_id
    )

    # Classify response and generate tags
    tags = classify_response(
        user_input=message,
        response_data=result,
        workflow_type="solution"
    )

    logger.info(f"[SOLUTION] Tags: intent={getattr(tags.intent_type, 'value', tags.intent_type)}, status={getattr(tags.response_status, 'value', tags.response_status)}")

    return api_response(True, data=result, tags=tags)


# ============================================================================
# INDEX RAG ENDPOINT
# ============================================================================

@agentic_bp.route('/index-rag', methods=['POST'])
@login_required
@handle_errors
def index_rag_search():
    """
    Index RAG Product Search
    ---
    tags:
      - Index RAG
    summary: Search products using Index RAG with parallel indexing
    description: |
      Runs the Index RAG workflow which:
      1. Classifies user intent with Flash LLM
      2. Applies hierarchical metadata filter (Product → Vendor → Model)
      3. Runs parallel indexing (Database + LLM Web Search)
      4. Structures output with LLM
      
      This is a 4-node workflow with embedded metadata filtering.
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
              description: Product search query
              example: "I need a pressure transmitter from Yokogawa"
            product_type:
              type: string
              description: Optional explicit product type
              example: "pressure_transmitter"
            vendors:
              type: array
              items:
                type: string
              description: Optional vendor filter
              example: ["yokogawa", "emerson"]
            top_k:
              type: integer
              description: Max results per source (default 7)
              example: 7
            enable_web_search:
              type: boolean
              description: Enable LLM web search thread (default true)
              example: true
            session_id:
              type: string
              description: Optional session ID
    responses:
      200:
        description: Index RAG search results
        schema:
          type: object
          properties:
            success:
              type: boolean
            data:
              type: object
              properties:
                output:
                  type: object
                  properties:
                    summary:
                      type: string
                    recommended_products:
                      type: array
                    total_found:
                      type: integer
                stats:
                  type: object
                  properties:
                    database_results:
                      type: integer
                    web_results:
                      type: integer
                    merged_results:
                      type: integer
                filters:
                  type: object
                metadata:
                  type: object
      400:
        description: Bad request - missing query
    """
    data = request.get_json()

    if not data or 'query' not in data:
        return api_response(False, error="query is required", status_code=400)

    query = data['query']
    product_type = data.get('product_type')
    vendors = data.get('vendors')
    top_k = data.get('top_k', 7)
    enable_web_search = data.get('enable_web_search', True)
    session_id = data.get('session_id') or get_session_id()

    logger.info("=" * 60)
    logger.info("[INDEX_RAG] Index RAG Search API Called")
    logger.info(f"[INDEX_RAG] Query: {query[:100]}...")
    logger.info(f"[INDEX_RAG] Product Type: {product_type}")
    logger.info(f"[INDEX_RAG] Vendors: {vendors}")
    logger.info(f"[INDEX_RAG] top_k: {top_k}, web_search: {enable_web_search}")

    try:
        from .index_rag_workflow import run_index_rag_workflow

        result = run_index_rag_workflow(
            query=query,
            requirements={
                "product_type": product_type,
                "vendors": vendors
            } if product_type or vendors else None,
            session_id=session_id,
            top_k=top_k,
            enable_web_search=enable_web_search
        )

        if not result.get('success'):
            logger.error(f"[INDEX_RAG] Workflow failed: {result.get('error')}")
            return api_response(False, error=result.get('error', 'Index RAG failed'), status_code=500)

        stats = result.get('stats', {})
        logger.info(f"[INDEX_RAG] Success: {stats.get('filtered_results', 0)} results "
                   f"(JSON: {stats.get('json_count', 0)}, PDF: {stats.get('pdf_count', 0)}, Web: {stats.get('web_results', 0)})")
        logger.info(f"[INDEX_RAG] Processing time: {result.get('metadata', {}).get('processing_time_ms')}ms")
        
        if result.get('is_follow_up'):
            logger.info(f"[INDEX_RAG] Follow-up resolved: '{query}' -> '{result.get('resolved_query')}'")


        # Classify response and generate tags
        tags = classify_response(
            user_input=query,
            response_data=result,
            workflow_type="index_rag"
        )

        return api_response(True, data=result, tags=tags)

    except ImportError as ie:
        logger.error(f"[INDEX_RAG] Import error: {ie}")
        return api_response(False, error=f"Index RAG module not available: {ie}", status_code=500)
    except Exception as e:
        logger.error(f"[INDEX_RAG] Failed: {e}", exc_info=True)
        return api_response(False, error=str(e), status_code=500)


@agentic_bp.route('/compare', methods=['POST'])
@login_required
@handle_errors
def comparison_workflow():
    """
    Comparative Analysis Workflow
    ---
    tags:
      - Enhanced Workflows
    summary: Run vendor/product comparison workflow
    description: |
      Compares vendors and products based on text request.
      Returns ranked products with scoring breakdown.
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - message
          properties:
            message:
              type: string
              description: Comparison request
              example: "Compare Honeywell ST800 vs Emerson 3051S for pressure measurement"
            session_id:
              type: string
    responses:
      200:
        description: Ranked comparison results
        schema:
          type: object
          properties:
            success:
              type: boolean
            data:
              type: object
              properties:
                ranked_products:
                  type: array
                formatted_output:
                  type: string
    """
    data = request.get_json()

    if not data or 'message' not in data:
        return api_response(False, error="Message is required", status_code=400)

    message = data['message']
    session_id = data.get('session_id') or get_session_id()

    from .product_search_workflow import run_product_search_with_comparison
    result = run_product_search_with_comparison(
        sample_input=message,
        product_type="instrument",
        session_id=session_id,
        auto_compare=True
    )

    # Classify response and generate tags
    tags = classify_response(
        user_input=message,
        response_data=result,
        workflow_type="comparison"
    )

    logger.info(f"[COMPARISON] Tags: intent={getattr(tags.intent_type, 'value', tags.intent_type)}, status={getattr(tags.response_status, 'value', tags.response_status)}")

    return api_response(True, data=result, tags=tags)


@agentic_bp.route('/compare-from-spec', methods=['POST'])
@login_required
@handle_errors
def compare_from_spec():
    """
    Compare Vendors from SpecObject
    ---
    tags:
      - Enhanced Workflows
    summary: Run comparison from finalized specification
    description: |
      Triggered by [COMPARE VENDORS] button in UI.
      Takes a SpecObject from instrument detail capture and runs multi-level comparison.
      Supports vendor, series, and model level comparisons.
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - spec_object
          properties:
            spec_object:
              type: object
              required:
                - product_type
              properties:
                product_type:
                  type: string
                  example: "pressure transmitter"
                category:
                  type: string
                  example: "Process Instrumentation"
                specifications:
                  type: object
                  example: {"range": "0-500 psi", "accuracy": "0.04%"}
                required_certifications:
                  type: array
                  items:
                    type: string
                  example: ["SIL2", "ATEX"]
                source_workflow:
                  type: string
                  example: "instrument_detail"
            comparison_type:
              type: string
              enum: [vendor, series, model, full]
              default: full
            session_id:
              type: string
    responses:
      200:
        description: Multi-level comparison results
        schema:
          type: object
          properties:
            success:
              type: boolean
            data:
              type: object
              properties:
                vendor_ranking:
                  type: array
                series_comparisons:
                  type: object
                top_recommendation:
                  type: object
                  properties:
                    vendor:
                      type: string
                    series:
                      type: string
                    model:
                      type: string
                    overall_score:
                      type: integer
                formatted_output:
                  type: string
    """
    data = request.get_json()

    if not data or 'spec_object' not in data:
        return api_response(False, error="spec_object is required", status_code=400)

    spec_object = data['spec_object']
    comparison_type = data.get('comparison_type', 'full')
    session_id = data.get('session_id') or get_session_id()
    user_id = data.get('user_id')

    from .product_search_workflow import trigger_comparison_from_product_search
    # Wrap spec_object as search_result for consolidated function
    result = trigger_comparison_from_product_search(
        search_result={"spec_object": spec_object, "ranked_results": spec_object.get("candidates", [])},
        session_id=session_id
    )

    # Classify response and generate tags
    # For spec-based comparison, use product_type from spec_object as user input context
    user_input_context = f"Compare {spec_object.get('product_type', 'products')} from specification"
    tags = classify_response(
        user_input=user_input_context,
        response_data=result,
        workflow_type='comparison'
    )

    return api_response(True, data=result, tags=tags)


@agentic_bp.route('/instrument-identifier', methods=['POST'])
@login_required
@handle_errors
def instrument_identifier():
    """
    Instrument Identifier Endpoint (List Generator)
    Identifies instruments and accessories from requirements and returns a selection list.
    """
    data = request.get_json()
    if not data:
         return api_response(False, error="No data provided", status_code=400)
    
    # Frontend sends 'message', check for it, fallback to 'requirements'
    message = data.get('message') or data.get('requirements')
    if not message:
        return api_response(False, error="Message or requirements is required", status_code=400)

    session_id = data.get('session_id') or get_session_id()
    
    # Run the workflow from instrument_identifier_workflow.py
    result = run_instrument_identifier_workflow(
        user_input=message,
        session_id=session_id
    )

    # Classify response and generate tags
    tags = classify_response(
        user_input=message,
        response_data=result,
        workflow_type='instrument_identifier'
    )

    return api_response(True, data=result, tags=tags)


@agentic_bp.route('/solution', methods=['POST'])
@login_required
@handle_errors
def solution_workflow_endpoint():
    """
    Solution Workflow Endpoint (Complex Engineering Challenges)
    
    This endpoint handles complex engineering challenges that require multiple
    instruments/accessories as a complete solution (e.g., reactor instrumentation,
    distillation column setup, process unit measurement systems).
    
    The solution workflow:
    1. Analyzes the solution context (industry, process type, parameters)
    2. Identifies ALL required instruments and accessories
    3. Generates sample_input for each item for subsequent product search
    
    Request:
        {
            "message": "Design a temperature measurement system for a chemical reactor...",
            "session_id": "optional_session_id"
        }
    
    Response:
        {
            "success": true,
            "data": {
                "response": "I've analyzed your engineering challenge...",
                "response_data": {
                    "workflow": "solution",
                    "solution_name": "Chemical Reactor Temperature System",
                    "items": [...],
                    "total_items": N,
                    "awaiting_selection": true
                }
            }
        }
    """
    data = request.get_json()
    if not data:
        return api_response(False, error="No data provided", status_code=400)
    
    # Get the message/requirements
    message = data.get('message') or data.get('requirements') or data.get('user_input')
    if not message:
        return api_response(False, error="Message or requirements is required", status_code=400)

    session_id = data.get('session_id') or get_session_id()
    
    # Log solution workflow invocation
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"[SOLUTION_API] Invoking solution workflow for session: {session_id}")
    logger.info(f"[SOLUTION_API] Input preview: {message[:100]}...")
    
    # Run the solution workflow
    result = run_solution_workflow(
        user_input=message,
        session_id=session_id
    )

    # Classify response and generate tags
    tags = classify_response(
        user_input=message,
        response_data=result,
        workflow_type='solution'
    )
    
    logger.info(f"[SOLUTION_API] Solution workflow complete, items: {result.get('response_data', {}).get('total_items', 0)}")

    return api_response(True, data=result, tags=tags)



@agentic_bp.route('/potential-product-index', methods=['POST'])
@login_required
@handle_errors
def potential_product_index():
    """
    Potential Product Index Workflow
    ---
    tags:
      - Enhanced Workflows
    summary: Discover and index new product types
    description: |
      Triggered when no schema exists for a product type.
      Discovers vendors/models via LLM and generates schema.
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
              description: Product type to index
              example: "differential pressure transmitter"
            session_id:
              type: string
    responses:
      200:
        description: Discovered vendors and generated schema
        schema:
          type: object
          properties:
            success:
              type: boolean
            data:
              type: object
              properties:
                product_type:
                  type: string
                discovered_vendors:
                  type: array
                vendor_model_families:
                  type: object
                generated_schema:
                  type: object
                schema_saved:
                  type: boolean
    """
    data = request.get_json()

    if not data or 'product_type' not in data:
        return api_response(False, error="product_type is required", status_code=400)

    product_type = data['product_type']
    session_id = data.get('session_id') or get_session_id()

    result = run_potential_product_index_workflow(
        product_type=product_type,
        session_id=session_id
    )

    # Classify response and generate tags
    # This workflow discovers new product types, so use product_type as context
    user_input_context = f"Index {product_type}"
    tags = classify_response(
        user_input=user_input_context,
        response_data=result,
        workflow_type='product_search'  # This is essentially a product discovery workflow
    )

    return api_response(True, data=result, tags=tags)



# ============================================================================
# TOOL ENDPOINTS
# ============================================================================

@agentic_bp.route('/tools/classify-intent', methods=['POST'])
@login_required
@handle_errors
def classify_intent_endpoint():
    """
    Test Intent Classification Tool
    ---
    tags:
      - LangChain Tools
    summary: Test classify_intent_tool directly
    description: |
      Directly invoke the LangChain classify_intent_tool.
      This tool classifies user input into intent categories for workflow routing.
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
              description: User message to classify
              example: "I need pressure transmitters"
            current_step:
              type: string
              description: Current workflow step
              example: "start"
            context:
              type: string
              description: Conversation context
              example: "New conversation"
    responses:
      200:
        description: Intent classification result
        schema:
          type: object
          properties:
            success:
              type: boolean
            data:
              type: object
              properties:
                intent:
                  type: string
                  example: "requirements"
                confidence:
                  type: number
                  example: 0.95
                next_step:
                  type: string
    """
    from tools.intent_tools import classify_intent_tool

    data = request.get_json()
    if not data or 'user_input' not in data:
        return api_response(False, error="user_input is required", status_code=400)

    result = classify_intent_tool.invoke({
        "user_input": data['user_input'],
        "current_step": data.get('current_step'),
        "context": data.get('context')
    })

    return api_response(True, data=result)


@agentic_bp.route('/tools/validate-requirements', methods=['POST'])
@login_required
@handle_errors
def validate_requirements_endpoint():
    """
    Test Requirements Validation Tool
    ---
    tags:
      - LangChain Tools
    summary: Test validate_requirements_tool directly
    description: |
      Directly invoke the LangChain validate_requirements_tool.
      Validates user requirements against product schema.
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
              description: Requirements text
              example: "Need 4-20mA pressure transmitter, 0-500 psi range"
            product_type:
              type: string
              description: Product type
              example: "pressure transmitter"
    responses:
      200:
        description: Validation result
        schema:
          type: object
          properties:
            success:
              type: boolean
            data:
              type: object
              properties:
                is_valid:
                  type: boolean
                missing_fields:
                  type: array
                  items:
                    type: string
    """
    from tools.schema_tools import validate_requirements_tool, load_schema_tool

    data = request.get_json()
    if not data or 'user_input' not in data:
        return api_response(False, error="user_input is required", status_code=400)

    product_type = data.get('product_type', '')

    # Load schema
    schema_result = load_schema_tool.invoke({"product_type": product_type})
    schema = schema_result.get("schema", {})

    # Validate
    result = validate_requirements_tool.invoke({
        "user_input": data['user_input'],
        "product_type": product_type,
        "schema": schema
    })

    return api_response(True, data=result)


@agentic_bp.route('/tools/search-vendors', methods=['POST'])
@login_required
@handle_errors
def search_vendors_endpoint():
    """
    Test Vendor Search Tool
    ---
    tags:
      - LangChain Tools
    summary: Test search_vendors_tool directly
    description: |
      Directly invoke the LangChain search_vendors_tool.
      Searches MongoDB for vendors offering specific products.
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
              description: Product type to search for
              example: "pressure transmitter"
            requirements:
              type: object
              description: Optional requirements filter
    responses:
      200:
        description: Vendor search results
        schema:
          type: object
          properties:
            success:
              type: boolean
            data:
              type: object
              properties:
                vendors:
                  type: array
                  items:
                    type: string
                vendor_count:
                  type: integer
    """
    from tools.search_tools import search_vendors_tool

    data = request.get_json()
    if not data or 'product_type' not in data:
        return api_response(False, error="product_type is required", status_code=400)

    result = search_vendors_tool.invoke({
        "product_type": data['product_type'],
        "requirements": data.get('requirements', {})
    })

    return api_response(True, data=result)


@agentic_bp.route('/tools/analyze-match', methods=['POST'])
@login_required
@handle_errors
def analyze_match_endpoint():
    """
    Test Vendor Match Analysis Tool
    ---
    tags:
      - LangChain Tools
    summary: Test analyze_vendor_match_tool directly
    description: |
      Directly invoke the LangChain analyze_vendor_match_tool.
      Performs detailed parameter-by-parameter analysis of vendor products.
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
              description: Vendor name
              example: "Honeywell"
            requirements:
              type: object
              description: User requirements
              example: {"outputSignal": "4-20mA", "range": "0-500 psi"}
            pdf_content:
              type: string
              description: Optional PDF datasheet content
            product_data:
              type: object
              description: Optional product JSON data
    responses:
      200:
        description: Analysis result
        schema:
          type: object
          properties:
            success:
              type: boolean
            data:
              type: object
              properties:
                match_score:
                  type: integer
                  example: 85
                matched_requirements:
                  type: object
    """
    from tools.analysis_tools import analyze_vendor_match_tool

    data = request.get_json()
    if not data or 'vendor' not in data or 'requirements' not in data:
        return api_response(False, error="vendor and requirements are required", status_code=400)

    result = analyze_vendor_match_tool.invoke({
        "vendor": data['vendor'],
        "requirements": data['requirements'],
        "pdf_content": data.get('pdf_content'),
        "product_data": data.get('product_data')
    })

    return api_response(True, data=result)


@agentic_bp.route('/tools/rank-products', methods=['POST'])
@login_required
@handle_errors
def rank_products_endpoint():
    """
    Test Product Ranking Tool
    ---
    tags:
      - LangChain Tools
    summary: Test rank_products_tool directly
    description: |
      Directly invoke the LangChain rank_products_tool.
      Ranks products based on analysis results using weighted criteria.
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - vendor_matches
          properties:
            vendor_matches:
              type: array
              description: Array of vendor analysis results
              items:
                type: object
            requirements:
              type: object
              description: Original requirements
    responses:
      200:
        description: Ranked products
        schema:
          type: object
          properties:
            success:
              type: boolean
            data:
              type: object
              properties:
                ranked_products:
                  type: array
                  items:
                    type: object
                top_pick:
                  type: object
    """
    from tools.ranking_tools import rank_products_tool

    data = request.get_json()
    if not data or 'vendor_matches' not in data:
        return api_response(False, error="vendor_matches is required", status_code=400)

    result = rank_products_tool.invoke({
        "vendor_matches": data['vendor_matches'],
        "requirements": data.get('requirements', {})
    })

    return api_response(True, data=result)


@agentic_bp.route('/tools/search-images', methods=['POST'])
@login_required
@handle_errors
def search_images_endpoint():
    """
    Test Product Image Search Tool
    ---
    tags:
      - LangChain Tools
    summary: Test search_product_images_tool directly
    description: |
      Directly invoke the LangChain search_product_images_tool.
      Searches for product images using multi-tier fallback (Google CSE → Serper → SerpAPI).
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
              description: Vendor name
              example: "Honeywell"
            product_name:
              type: string
              description: Product model name
              example: "ST800"
            product_type:
              type: string
              description: Product type
              example: "pressure transmitter"
            model_family:
              type: string
              description: Optional model family
    responses:
      200:
        description: Image search results
        schema:
          type: object
          properties:
            success:
              type: boolean
            data:
              type: object
              properties:
                images:
                  type: array
                  items:
                    type: object
    """
    from tools.search_tools import search_product_images_tool

    data = request.get_json()
    required = ['vendor', 'product_name', 'product_type']
    if not data or not all(k in data for k in required):
        return api_response(False, error=f"{required} are required", status_code=400)

    result = search_product_images_tool.invoke({
        "vendor": data['vendor'],
        "product_name": data['product_name'],
        "product_type": data['product_type'],
        "model_family": data.get('model_family')
    })

    return api_response(True, data=result)


@agentic_bp.route('/tools/search-pdfs', methods=['POST'])
@login_required
@handle_errors
def search_pdfs_endpoint():
    """
    Test PDF Datasheet Search Tool
    ---
    tags:
      - LangChain Tools
    summary: Test search_pdf_datasheets_tool directly
    description: |
      Directly invoke the LangChain search_pdf_datasheets_tool.
      Searches for PDF datasheets using multi-tier fallback.
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
              description: Vendor name
              example: "Emerson"
            product_type:
              type: string
              description: Product type
              example: "pressure transmitter"
            model_family:
              type: string
              description: Optional model family
              example: "3051S"
    responses:
      200:
        description: PDF search results
        schema:
          type: object
          properties:
            success:
              type: boolean
            data:
              type: object
              properties:
                pdfs:
                  type: array
                  items:
                    type: object
                    properties:
                      url:
                        type: string
                      title:
                        type: string
    """
    from tools.search_tools import search_pdf_datasheets_tool

    data = request.get_json()
    if not data or 'vendor' not in data or 'product_type' not in data:
        return api_response(False, error="vendor and product_type are required", status_code=400)

    result = search_pdf_datasheets_tool.invoke({
        "vendor": data['vendor'],
        "product_type": data['product_type'],
        "model_family": data.get('model_family')
    })

    return api_response(True, data=result)


@agentic_bp.route('/tools/sales-interact', methods=['POST'])
@login_required
@handle_errors
def sales_interact_endpoint():
    """
    Test Sales Agent Interaction Tool
    ---
    tags:
      - LangChain Tools
    summary: Test sales_agent_interact_tool directly
    description: |
      Directly invoke the LangChain sales_agent_interact_tool.
      Handles conversational state and user interaction for the product search workflow.
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - step
            - user_message
          properties:
            step:
              type: string
              description: Current workflow step
              example: "initialInput"
            user_message:
              type: string
              description: User message
              example: "I need 4-20mA pressure transmitters"
            product_type:
              type: string
              description: Detected product type
            data_context:
              type: object
              description: Context data for the step
    responses:
      200:
        description: Sales interaction result
    """
    from tools.sales_agent_tools import sales_agent_interact_tool

    data = request.get_json()
    if not data or 'step' not in data or 'user_message' not in data:
        return api_response(False, error="step and user_message are required", status_code=400)

    result = sales_agent_interact_tool.invoke({
        "step": data['step'],
        "user_message": data['user_message'],
        "product_type": data.get('product_type'),
        "data_context": data.get('data_context'),
        "intent": data.get('intent'),
        "session_id": data.get('session_id') or get_session_id()
    })

    return api_response(True, data=result)


@agentic_bp.route('/tools/identify-instruments', methods=['POST'])
@login_required
@handle_errors
def identify_instruments_endpoint():
    """
    Test Instrument Identification Tool
    ---
    tags:
      - LangChain Tools
    summary: Test identify_instruments_tool directly
    description: |
      Directly invoke the LangChain identify_instruments_tool.
      Identifies instruments needed from process requirements.
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
              description: Process requirements
              example: "I need to measure pressure and flow in my water treatment plant"
    responses:
      200:
        description: Instrument identification result
    """
    from tools.instrument_tools import identify_instruments_tool

    data = request.get_json()
    if not data or 'requirements' not in data:
        return api_response(False, error="requirements is required", status_code=400)

    result = identify_instruments_tool.invoke({
        "requirements": data['requirements']
    })

    return api_response(True, data=result)


# ============================================================================
# SESSION ENDPOINTS
# ============================================================================

@agentic_bp.route('/session', methods=['GET'])
@login_required
@handle_errors
def get_session():
    """Get current session info"""
    return api_response(True, data={
        "session_id": get_session_id(),
        "vendor_filter": session.get('csv_vendor_filter')
    })


@agentic_bp.route('/session', methods=['DELETE'])
@login_required
@handle_errors
def clear_session():
    """Clear current session"""
    session.pop('agentic_session_id', None)
    session.pop('csv_vendor_filter', None)
    return api_response(True, data={"message": "Session cleared"})


@agentic_bp.route('/session/vendor-filter', methods=['POST'])
@login_required
@handle_errors
def set_vendor_filter():
    """
    Set vendor filter from CSV upload

    Request Body:
    {
        "vendor_names": ["Vendor1", "Vendor2"]
    }
    """
    data = request.get_json()
    if not data or 'vendor_names' not in data:
        return api_response(False, error="vendor_names is required", status_code=400)

    session['csv_vendor_filter'] = {
        'vendor_names': data['vendor_names']
    }

    return api_response(True, data={
        "message": "Vendor filter set",
        "vendor_count": len(data['vendor_names'])
    })




# ============================================================================
# VALIDATION TOOL WRAPPER
# ============================================================================

@agentic_bp.route('/validate', methods=['POST'])
@handle_errors
@login_required
def agentic_validate():
    """
    Validation Tool Wrapper API

    Standalone validation endpoint for agentic workflows.
    Detects product type, loads/generates schema, and validates requirements.

    Request Body:
        {
            "user_input": str,      # Required: User's requirements description
            "message": str,         # Alternative to user_input
            "product_type": str,    # Optional: Expected product type
            "session_id": str,      # Optional: Session tracking ID
            "enable_ppi": bool      # Optional: Enable PPI workflow (default: True)
        }

    Returns:
        {
            "success": bool,
            "data": {
                "product_type": str,
                "detected_schema": dict,
                "provided_requirements": dict,
                "ppi_workflow_used": bool,
                "is_valid": bool,
                "missing_fields": list
            }
        }
    """
    try:
        from tools.schema_tools import load_schema_tool, validate_requirements_tool
        from tools.intent_tools import extract_requirements_tool

        data = request.get_json()

        # Accept both 'user_input' and 'message'
        user_input = data.get('user_input') or data.get('message')
        if not user_input:
            return api_response(False, error="user_input or message is required", status_code=400)

        expected_product_type = data.get('product_type')
        session_id = data.get('session_id', 'default')
        enable_ppi = data.get('enable_ppi', True)

        logger.info(f"[VALIDATION_TOOL] Starting validation for session: {session_id}")
        logger.info(f"[VALIDATION_TOOL] User input: {user_input[:100]}...")

        # Step 1: Extract product type and requirements
        extract_result = extract_requirements_tool.invoke({
            "user_input": user_input
        })

        product_type = extract_result.get("product_type", expected_product_type or "")
        logger.info(f"[VALIDATION_TOOL] Detected Product Type: {product_type}")

        # Step 2: Load or generate schema
        schema_result = load_schema_tool.invoke({
            "product_type": product_type,
            "enable_ppi": enable_ppi
        })

        schema = schema_result.get("schema", {})
        ppi_used = not schema_result.get("from_database", True)

        logger.info(f"[VALIDATION_TOOL] Schema: {'Generated via PPI' if ppi_used else 'Loaded from DB'}")

        # Step 3: Validate requirements against schema
        validation_result = validate_requirements_tool.invoke({
            "user_input": user_input,
            "product_type": product_type,
            "schema": schema
        })

        requirements = validation_result.get("provided_requirements", {})
        missing_fields = validation_result.get("missing_fields", [])
        is_valid = validation_result.get("is_valid", False)

        if missing_fields:
            logger.info(f"[VALIDATION_TOOL] Missing Fields: {missing_fields}")
        else:
            logger.info(f"[VALIDATION_TOOL] All mandatory fields provided")

        # =====================================================================
        # FIELD DESCRIPTIONS FOR ON-HOVER TOOLTIPS
        # Extracts descriptions from template specifications for frontend display
        # =====================================================================
        field_descriptions = {}
        try:
            # Try to get template specifications (60+ specs per product type)
            try:
                from agentic.deep_agent.phase3_specification_templates import (
                    get_all_specs_for_product_type
                )
                template_specs = get_all_specs_for_product_type(product_type)
                if template_specs:
                    for spec_key, spec_def in template_specs.items():
                        field_descriptions[spec_key] = spec_def.description
                    logger.info(f"[VALIDATION_TOOL] Loaded {len(field_descriptions)} field descriptions from templates")
            except ImportError:
                pass

            # Generate descriptions for schema fields not in templates
            # Extract all leaf field keys from schema
            def extract_keys(obj, prefix=""):
                keys = []
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        full_key = f"{prefix}.{key}" if prefix else key
                        if isinstance(value, dict) and "value" not in value:
                            keys.extend(extract_keys(value, full_key))
                        else:
                            keys.append(full_key)
                return keys
            
            def prettify_field_name(field_name: str) -> str:
                """Convert field_name or fieldName to human-readable description"""
                import re
                # Handle camelCase: split on capital letters
                words = re.sub(r'([a-z])([A-Z])', r'\1 \2', field_name)
                # Handle snake_case: replace underscores with spaces
                words = words.replace('_', ' ').replace('-', ' ')
                # Capitalize first letter of each word
                words = ' '.join(word.capitalize() for word in words.split())
                return f"Specification for {words}"
            
            all_keys = extract_keys(schema.get("mandatoryRequirements", {}))
            all_keys.extend(extract_keys(schema.get("optionalRequirements", {})))
            
            # For fields not in templates, create a description from the field name
            for field_key in all_keys:
                field_name = field_key.split(".")[-1]
                # Check if we already have a template description (match by field name)
                if field_name not in field_descriptions and field_key not in field_descriptions:
                    # Generate a readable description from the field name
                    field_descriptions[field_key] = prettify_field_name(field_name)
                elif field_name in field_descriptions:
                    # Copy template description to full key path
                    field_descriptions[field_key] = field_descriptions[field_name]
                
        except Exception as desc_err:
            logger.warning(f"[VALIDATION_TOOL] Field description extraction failed: {desc_err}")

        result = {
            "product_type": product_type,
            "detected_schema": schema,
            "provided_requirements": requirements,
            "ppi_workflow_used": ppi_used,
            "is_valid": is_valid,
            "missing_fields": missing_fields,
            "session_id": session_id,
            # Field descriptions for on-hover tooltips
            "field_descriptions": field_descriptions,
            "field_descriptions_count": len(field_descriptions)
        }

        logger.info(f"[VALIDATION_TOOL] Validation complete with {len(field_descriptions)} field descriptions")

        return api_response(True, data=result)

    except Exception as e:
        logger.error(f"[VALIDATION_TOOL] Validation failed: {e}", exc_info=True)
        return api_response(False, error=str(e), status_code=500)


# ============================================================================
# ADVANCED PARAMETERS TOOL WRAPPER
# ============================================================================

@agentic_bp.route('/advanced-parameters', methods=['POST'])
@handle_errors
@login_required
def agentic_advanced_parameters():
    """
    Advanced Parameters Discovery Tool Wrapper API

    Standalone advanced parameters discovery endpoint for agentic workflows.
    Discovers latest advanced specifications with series numbers from top vendors.

    Request Body:
        {
            "product_type": str,    # Required: Product type to discover parameters for
            "session_id": str       # Optional: Session tracking ID
        }

    Returns:
        {
            "success": bool,
            "data": {
                "product_type": str,
                "unique_specifications": [
                    {
                        "key": str,
                        "name": str
                    }
                ],
                "total_unique_specifications": int,
                "existing_specifications_filtered": int,
                "vendor_specifications": list
            }
        }
    """
    try:
        from agentic_advanced_parameters_tool import AdvancedParametersTool

        data = request.get_json()

        # Validate input
        product_type = data.get('product_type', '').strip()
        if not product_type:
            return api_response(False, error="product_type is required", status_code=400)

        session_id = data.get('session_id', 'default')

        logger.info(f"[ADVANCED_PARAMS_TOOL] Starting discovery for: {product_type}")
        logger.info(f"[ADVANCED_PARAMS_TOOL] Session: {session_id}")

        # Initialize and run the tool
        tool = AdvancedParametersTool()
        result = tool.discover(
            product_type=product_type,
            session_id=session_id
        )

        # Log results
        unique_count = len(result.get('unique_specifications', []))
        filtered_count = result.get('existing_specifications_filtered', 0)

        logger.info(
            f"[ADVANCED_PARAMS_TOOL] Discovery complete: "
            f"{unique_count} new specifications, "
            f"{filtered_count} existing filtered"
        )

        return api_response(True, data=result)

    except Exception as e:
        logger.error(f"[ADVANCED_PARAMS_TOOL] Discovery failed: {e}", exc_info=True)
        return api_response(False, error=str(e), status_code=500)


# ============================================================================
# PRODUCT SEARCH WORKFLOW
# ============================================================================


@agentic_bp.route('/product-search', methods=['POST'])
@handle_errors
@login_required
def product_search():
    """
    Product Search Deep Agentic Workflow

    This endpoint implements a STATEFUL workflow with Deep Agent capabilities:

    Flow (handled by DeepAgenticWorkflowOrchestrator):
    1. VALIDATION - Product type detection & schema generation (with failure memory)
    2. AWAIT_MISSING_FIELDS - User decision on missing fields
    3. ADVANCED_DISCOVERY - Discover specifications from vendors
    4. VENDOR_ANALYSIS - Parallel vendor matching
    5. RANKING - Final product ranking

    Features:
    - Session state management with persistence
    - User decision parsing and handling
    - Failure memory and learning from past errors
    - Adaptive prompt optimization
    - Automatic phase progression

    Request Body:
        {
            "user_input": str,          # Required on first call
            "message": str,             # Alternative to user_input
            "thread_id": str,           # Required for resuming workflow
            "user_decision": str,       # User's choice: "add_fields", "continue", "yes", "no"
            "user_provided_fields": {},  # Fields provided by user
            "product_type": str,        # Optional: Product type hint
            "session_id": str           # Session tracking ID
        }

    Returns:
        {
            "success": bool,
            "data": {
                "thread_id": str,               # For resuming conversation
                "session_id": str,              # Session identifier
                "awaiting_user_input": bool,    # True if workflow is paused
                "current_phase": str,           # Current workflow phase
                "sales_agent_response": str,    # Message to display to user
                "schema": dict,                 # Schema for left sidebar display
                "missing_fields": list,         # Missing mandatory fields
                "validation_result": dict,
                "available_advanced_params": list,
                "completed": bool               # True if workflow is complete
            }
        }
    """
    try:
        # Import the Deep Agentic Workflow Orchestrator
        from .deep_agent.deep_agentic_workflow import get_deep_agentic_orchestrator

        data = request.get_json()
        if not data:
            return api_response(False, error="Request body is required", status_code=400)

        # Extract parameters from request
        user_input = data.get('user_input') or data.get('message', '')
        thread_id = data.get('thread_id')
        user_decision = data.get('user_decision')
        user_provided_fields = data.get('user_provided_fields', {})
        product_type_hint = data.get('product_type')
        session_id = data.get('session_id') or data.get('search_session_id')

        logger.info(
            f"\n{'='*60}\n"
            f"[PRODUCT_SEARCH] Deep Agentic Workflow Request\n"
            f"   Session: {session_id}\n"
            f"   Thread: {thread_id}\n"
            f"   Decision: {user_decision}\n"
            f"   Input: {user_input[:50] if user_input else 'None'}...\n"
            f"{'='*60}"
        )

        # Get the orchestrator and process the request
        orchestrator = get_deep_agentic_orchestrator()

        result = orchestrator.process_request(
            user_input=user_input,
            session_id=session_id,
            thread_id=thread_id,
            user_decision=user_decision,
            user_provided_fields=user_provided_fields,
            product_type_hint=product_type_hint
        )

        # Check if orchestrator returned an error
        if not result.get('success', True):
            return api_response(
                False,
                error=result.get('error', 'Workflow failed'),
                status_code=500
            )

        # Add available_advanced_params for backward compatibility
        if 'discovered_specs' in result:
            result['available_advanced_params'] = result.get('discovered_specs', [])

        # Classify response and generate tags for frontend
        tags = classify_response(
            user_input=user_input or '',
            response_data=result,
            workflow_type='product_search'
        )

        return api_response(True, data=result, tags=tags)

    except ImportError as e:
        logger.error(f"[PRODUCT_SEARCH] Import error - Deep Agentic Workflow not available: {e}")
        return api_response(
            False,
            error="Deep Agentic Workflow module not available. Please check installation.",
            status_code=500
        )

    except Exception as e:
        logger.error(f"[PRODUCT_SEARCH] Workflow failed: {e}", exc_info=True)
        return api_response(False, error=str(e), status_code=500)

# DEPRECATED: Old manual phase handling code removed
# Now using DeepAgenticWorkflowOrchestrator for all workflow management


@agentic_bp.route('/run-analysis', methods=['POST'])
@handle_errors
@login_required
def run_product_analysis():
    """
    Run Final Product Analysis (Steps 4-5: Vendor Analysis + Ranking)

    This endpoint executes the actual product search after requirements are collected.
    It calls workflow.run_analysis_only() to:
    - Step 4: Run vendor analysis (parallel PDF/JSON matching)
    - Step 5: Rank products with scores

    Request Body:
        {
            "structured_requirements": {
                "productType": str,
                "mandatoryRequirements": dict,
                "optionalRequirements": dict,
                "selectedAdvancedParams": dict  # Optional
            },
            "product_type": str,
            "schema": dict,  # Optional
            "session_id": str  # Optional
        }

    Returns:
        {
            "success": bool,
            "data": {
                "vendorAnalysis": {
                    "vendorMatches": [...],
                    "totalMatches": int
                },
                "overallRanking": {
                    "rankedProducts": [...]
                },
                "topRecommendation": {...},
                "analysisResult": {...}  # Complete result for RightPanel
            }
        }
    """
    try:
        from product_search_workflow.workflow import ProductSearchWorkflow

        data = request.get_json()
        if not data:
            return api_response(False, error="Request body is required", status_code=400)

        # Extract parameters
        structured_requirements = data.get('structured_requirements')
        product_type = data.get('product_type')
        schema = data.get('schema')
        session_id = data.get('session_id') or data.get('search_session_id') or f"analysis_{uuid.uuid4().hex[:8]}"

        # Validate required fields
        if not structured_requirements:
            return api_response(False, error="structured_requirements is required", status_code=400)

        if not product_type:
            return api_response(False, error="product_type is required", status_code=400)

        logger.info(f"[RUN_ANALYSIS] Starting final analysis")
        logger.info(f"[RUN_ANALYSIS] Product Type: {product_type}")
        logger.info(f"[RUN_ANALYSIS] Session: {session_id}")

        # Initialize workflow
        workflow = ProductSearchWorkflow(
            enable_ppi_workflow=False,  # Schema already determined
            auto_mode=True,
            max_vendor_workers=5
        )

        # Run analysis only (Steps 4-5)
        analysis_result = workflow.run_analysis_only(
            structured_requirements=structured_requirements,
            product_type=product_type,
            schema=schema,
            session_id=session_id
        )

        if not analysis_result.get('success'):
            logger.error(f"[RUN_ANALYSIS] Analysis failed: {analysis_result.get('error')}")
            return api_response(False, error=analysis_result.get('error', 'Analysis failed'), status_code=500)

        logger.info(f"[RUN_ANALYSIS] Analysis complete")
        logger.info(f"[RUN_ANALYSIS] Products ranked: {analysis_result.get('totalRanked', 0)}")
        logger.info(f"[RUN_ANALYSIS] Exact matches: {analysis_result.get('exactMatchCount', 0)}")
        logger.info(f"[RUN_ANALYSIS] Approximate matches: {analysis_result.get('approximateMatchCount', 0)}")

        # =====================================================================
        # PRIMARY IMAGE FETCHING - Fetch generic images for ranked products
        # This is the primary layer; frontend will handle fallback for missing
        # =====================================================================
        try:
            from generic_image_utils import fetch_generic_product_image
            
            ranked_products = analysis_result.get('overallRanking', {}).get('rankedProducts', [])
            images_fetched = 0
            
            # Fetch images for top 20 products to avoid rate limiting
            for product in ranked_products[:20]:
                # Skip if product already has an image
                if product.get('top_image') or product.get('topImage'):
                    continue
                
                # Use product_type for generic image lookup
                pt = product.get('productType') or product.get('product_type') or product_type
                if not pt:
                    continue
                
                try:
                    image_result = fetch_generic_product_image(pt)
                    if image_result and image_result.get('url'):
                        product['top_image'] = {
                            'url': image_result['url'],
                            'source': image_result.get('source', 'generic_cache'),
                            'product_type': pt
                        }
                        images_fetched += 1
                        logger.debug(f"[RUN_ANALYSIS] Image fetched for {product.get('productName', 'Unknown')}")
                except Exception as img_err:
                    logger.warning(f"[RUN_ANALYSIS] Image fetch failed for {pt}: {img_err}")
            
            logger.info(f"[RUN_ANALYSIS] Images fetched: {images_fetched}/{min(20, len(ranked_products))}")
            
        except ImportError as e:
            logger.warning(f"[RUN_ANALYSIS] Could not import generic_image_utils: {e}")
        except Exception as e:
            logger.warning(f"[RUN_ANALYSIS] Image fetching failed: {e}")

        # Return analysis result
        return api_response(True, data=analysis_result)

    except Exception as e:
        logger.error(f"[RUN_ANALYSIS] Failed: {e}", exc_info=True)
        return api_response(False, error=str(e), status_code=500)


# ============================================================================
# SALES AGENT TOOL WRAPPER API
# ============================================================================


@agentic_bp.route('/sales-agent', methods=['POST'])
@handle_errors
@login_required
def agentic_sales_agent():
    """
    Sales Agent Tool Wrapper API

    Provides conversational AI interface for product requirements collection.
    Handles step-by-step workflow with LLM-powered responses.

    Request Body:
        {
            "step": "initialInput",  # Current workflow step
            "user_message": "I need a pressure transmitter",
            "data_context": {  # Context data for current step
                "productType": "Pressure Transmitter",
                "availableParameters": [...],
                "selectedParameters": {...}
            },
            "session_id": "session_123",  # Session identifier
            "intent": "workflow",  # "workflow" or "knowledgeQuestion"
            "save_immediately": false  # Skip greeting if true
        }

    Response:
        {
            "success": true,
            "data": {
                "content": "AI-generated response message",
                "nextStep": "awaitAdditionalAndLatestSpecs",
                "maintainWorkflow": true,
                "dataContext": {...},  # Updated context
                "discoveredParameters": [...]  # Optional
            }
        }

    Workflow Steps:
        - greeting: Welcome message
        - initialInput: Initial product requirements
        - awaitMissingInfo: Collect missing mandatory fields
        - awaitAdditionalAndLatestSpecs: Additional specifications
        - awaitAdvancedSpecs: Advanced parameter specifications
        - showSummary: Display requirements summary
        - finalAnalysis: Complete analysis
    """
    try:
        from product_search_workflow import SalesAgentTool

        data = request.get_json()

        # Validate required fields
        if not data:
            return api_response(False, error="Request body is required", status_code=400)

        step = data.get('step')
        if not step:
            return api_response(False, error="'step' field is required", status_code=400)

        user_message = data.get('user_message', data.get('userMessage', ''))
        data_context = data.get('data_context', data.get('dataContext', {}))
        session_id = data.get('session_id', data.get('search_session_id', 'default'))
        intent = data.get('intent', 'workflow')
        save_immediately = data.get('save_immediately', data.get('saveImmediately', False))

        logger.info(f"[SALES_AGENT] Session {session_id}: Step={step}, Intent={intent}")
        logger.info(f"[SALES_AGENT] User message: {user_message[:100] if user_message else '(empty)'}...")

        # Initialize Sales Agent Tool
        # Note: LLM instance can be passed here if available
        sales_agent = SalesAgentTool(llm=None)  # TODO: Integrate with LLM

        # Process the workflow step
        result = sales_agent.process_step(
            step=step,
            user_message=user_message,
            data_context=data_context,
            session_id=session_id,
            intent=intent,
            save_immediately=save_immediately
        )

        # Check if advanced parameters discovery should be triggered
        if result.get('triggerDiscovery'):
            product_type = result.get('productType')
            if product_type:
                try:
                    # Import and use AdvancedParametersTool
                    from product_search_workflow import AdvancedParametersTool

                    logger.info(f"[SALES_AGENT] Triggering parameter discovery for: {product_type}")

                    params_tool = AdvancedParametersTool()
                    params_result = params_tool.discover(
                        product_type=product_type,
                        session_id=session_id
                    )

                    if params_result['success']:
                        discovered_specs = params_result.get('unique_specifications', [])
                        logger.info(f"[SALES_AGENT] Discovered {len(discovered_specs)} parameters")

                        # Update data context with discovered parameters
                        updated_context = data_context.copy()
                        updated_context['availableParameters'] = discovered_specs

                        # Re-process the step with discovered parameters
                        result = sales_agent.process_step(
                            step=step,
                            user_message="",  # Empty message to display parameters
                            data_context=updated_context,
                            session_id=session_id,
                            intent=intent
                        )

                        # Add discovery info to result
                        result['discoveredParameters'] = discovered_specs
                        result['dataContext'] = updated_context
                    else:
                        logger.warning(f"[SALES_AGENT] Parameter discovery failed")
                        result['discoveryError'] = True

                except Exception as disc_error:
                    logger.error(f"[SALES_AGENT] Discovery error: {disc_error}", exc_info=True)
                    result['discoveryError'] = True

        logger.info(f"[SALES_AGENT] Response generated, next step: {result.get('nextStep')}")

        return api_response(True, data=result)

    except Exception as e:
        logger.error(f"[SALES_AGENT] Error: {e}", exc_info=True)
        return api_response(False, error=str(e), status_code=500)


# ============================================================================
# STANDARDS RAG API
# ============================================================================


@agentic_bp.route('/standards-query', methods=['POST'])
@login_required
@handle_errors
def standards_query():
    """
    Standards RAG Query API - Query engineering standards documentation.

    Queries the Standards RAG knowledge base for information about:
    - Applicable standards (ISO, IEC, API, ANSI, ISA)
    - Required certifications (SIL, ATEX, CE, etc.)
    - Safety requirements
    - Calibration standards
    - Environmental requirements
    - Communication protocols

    Request:
        {
            "question": "What are the SIL requirements for pressure transmitters?",
            "top_k": 5,
            "session_id": "optional_session_id"
        }

    Response:
        {
            "success": true,
            "data": {
                "answer": "According to IEC 61508...",
                "citations": [...],
                "confidence": 0.85,
                "sources_used": ["standards_doc.docx", ...],
                "metadata": {
                    "processing_time_ms": 1234,
                    "documents_retrieved": 5
                }
            }
        }
    """
    data = request.get_json()

    logger.info("=" * 60)
    logger.info("[STANDARDS-RAG] Standards Query API Called")

    question = data.get('question')
    if not question:
        logger.error("[STANDARDS-RAG] No question provided")
        return api_response(False, error="question is required", status_code=400)

    top_k = data.get('top_k', 5)
    session_id = data.get('session_id')

    logger.info(f"[STANDARDS-RAG] Question: {question[:100]}...")
    logger.info(f"[STANDARDS-RAG] top_k: {top_k}")

    try:
        from agentic.standards_rag import run_standards_rag_workflow

        # Run the Standards RAG workflow
        result = run_standards_rag_workflow(
            question=question,
            session_id=session_id,
            top_k=top_k
        )

        if result.get('status') == 'success':
            logger.info("[STANDARDS-RAG] Query successful")
            return api_response(True, data={
                "answer": result['final_response'].get('answer', ''),
                "citations": result['final_response'].get('citations', []),
                "confidence": result['final_response'].get('confidence', 0.0),
                "sources_used": result['final_response'].get('sources_used', []),
                "metadata": result['final_response'].get('metadata', {}),
                "sessionId": session_id
            })
        else:
            logger.warning(f"[STANDARDS-RAG] Query failed: {result.get('error')}")
            return api_response(False, error=result.get('error', 'Standards query failed'))

    except Exception as e:
        logger.error(f"[STANDARDS-RAG] Error: {e}", exc_info=True)
        return api_response(False, error=str(e), status_code=500)


@agentic_bp.route('/standards-enrich', methods=['POST'])
@login_required
@handle_errors
def standards_enrich():
    """
    Standards Enrichment API - Enrich a product schema with standards information.

    Takes a product type and optionally a schema, and returns the schema
    enriched with applicable standards from the Standards RAG knowledge base.

    Request:
        {
            "product_type": "pressure transmitter",
            "schema": { ... }  // Optional - will use default if not provided
        }

    Response:
        {
            "success": true,
            "data": {
                "product_type": "pressure transmitter",
                "applicable_standards": ["IEC 61508", "ISO 10849", ...],
                "certifications": ["SIL2", "ATEX Zone 1", ...],
                "safety_requirements": { ... },
                "calibration_standards": { ... },
                "environmental_requirements": { ... },
                "communication_protocols": ["HART", "4-20MA", ...],
                "confidence": 0.85,
                "sources": ["standards_doc.docx", ...]
            }
        }
    """
    data = request.get_json()

    logger.info("=" * 60)
    logger.info("[STANDARDS-ENRICH] Standards Enrichment API Called")

    product_type = data.get('product_type')
    if not product_type:
        logger.error("[STANDARDS-ENRICH] No product_type provided")
        return api_response(False, error="product_type is required", status_code=400)

    schema = data.get('schema', {})

    logger.info(f"[STANDARDS-ENRICH] Product type: {product_type}")

    try:
        from tools.standards_enrichment_tool import get_applicable_standards, enrich_schema_with_standards

        if schema:
            # Enrich provided schema
            logger.info("[STANDARDS-ENRICH] Enriching provided schema")
            enriched_schema = enrich_schema_with_standards(product_type, schema)
            return api_response(True, data={
                "product_type": product_type,
                "enriched_schema": enriched_schema,
                "standards_added": 'standards' in enriched_schema
            })
        else:
            # Just get applicable standards
            logger.info("[STANDARDS-ENRICH] Getting applicable standards")
            standards_info = get_applicable_standards(product_type)
            return api_response(True, data={
                "product_type": product_type,
                "applicable_standards": standards_info.get('applicable_standards', []),
                "certifications": standards_info.get('certifications', []),
                "safety_requirements": standards_info.get('safety_requirements', {}),
                "calibration_standards": standards_info.get('calibration_standards', {}),
                "environmental_requirements": standards_info.get('environmental_requirements', {}),
                "communication_protocols": standards_info.get('communication_protocols', []),
                "confidence": standards_info.get('confidence', 0.0),
                "sources": standards_info.get('sources', [])
            })

    except Exception as e:
        logger.error(f"[STANDARDS-ENRICH] Error: {e}", exc_info=True)
        return api_response(False, error=str(e), status_code=500)


@agentic_bp.route('/standards-validate', methods=['POST'])
@login_required
@handle_errors
def standards_validate():
    """
    Standards Validation API - Validate requirements against applicable standards.

    Checks if user requirements comply with engineering standards and provides
    recommendations for missing or incomplete specifications.

    Request:
        {
            "product_type": "pressure transmitter",
            "requirements": {
                "outputSignal": "4-20mA HART",
                "pressureRange": "0-100 bar"
            }
        }

    Response:
        {
            "success": true,
            "data": {
                "is_compliant": false,
                "compliance_issues": [...],
                "recommendations": [...],
                "applicable_standards": [...],
                "required_certifications": [...]
            }
        }
    """
    data = request.get_json()

    logger.info("=" * 60)
    logger.info("[STANDARDS-VALIDATE] Standards Validation API Called")

    product_type = data.get('product_type')
    requirements = data.get('requirements')

    if not product_type:
        return api_response(False, error="product_type is required", status_code=400)
    if not requirements:
        return api_response(False, error="requirements is required", status_code=400)

    logger.info(f"[STANDARDS-VALIDATE] Product type: {product_type}")

    try:
        from tools.standards_enrichment_tool import validate_requirements_against_standards

        validation_result = validate_requirements_against_standards(product_type, requirements)

        return api_response(True, data={
            "product_type": product_type,
            "is_compliant": validation_result.get('is_compliant', False),
            "compliance_issues": validation_result.get('compliance_issues', []),
            "recommendations": validation_result.get('recommendations', []),
            "applicable_standards": validation_result.get('applicable_standards', []),
            "required_certifications": validation_result.get('required_certifications', []),
            "confidence": validation_result.get('confidence', 0.0)
        })

    except Exception as e:
        logger.error(f"[STANDARDS-VALIDATE] Error: {e}", exc_info=True)
        return api_response(False, error=str(e), status_code=500)



# ============================================================================
# DEEP AGENT TEST ENDPOINT
# ============================================================================


@agentic_bp.route('/test-deep-agent', methods=['POST'])
@handle_errors
def test_deep_agent_schema_population():
    """
    Test Deep Agent Schema Population
    
    This endpoint tests the Deep Agent integration by running the 
    instrument identifier workflow with verbose logging.
    
    Request:
        {
            "user_input": "I need a pressure transmitter for crude oil storage",
            "run_full_workflow": false  // if true, runs instrument_identifier_workflow
        }
    
    Response:
        {
            "success": true,
            "data": {
                "items_enriched": 2,
                "schemas_populated": 2,
                "total_fields_populated": 15,
                "items": [...]
            }
        }
    """
    data = request.get_json() or {}
    
    print("\n" + "=" * 80)
    print("[TEST] DEEP AGENT SCHEMA POPULATION TEST")
    print("=" * 80)
    
    user_input = data.get('user_input', 'I need a pressure transmitter for crude oil storage with SIL2 requirements')
    run_full_workflow = data.get('run_full_workflow', False)
    
    print(f"[TEST] User input: {user_input[:80]}...")
    print(f"[TEST] Run full workflow: {run_full_workflow}")
    
    try:
        if run_full_workflow:
            # Run the full instrument identifier workflow
            from agentic.instrument_identifier_workflow import run_instrument_identifier_workflow
            
            print("\n[TEST] Running FULL Instrument Identifier Workflow...")
            result = run_instrument_identifier_workflow(
                user_input=user_input,
                session_id=f"test_{uuid.uuid4().hex[:8]}"
            )
            
            response_data = result.get('response_data', {})
            items = response_data.get('items', [])
            
            # Analyze enrichment results
            schemas_populated = sum(1 for item in items if item.get('schema_populated', False))
            total_fields = 0
            for item in items:
                schema = item.get('schema', {})
                pop_info = schema.get('_deep_agent_population', {})
                total_fields += pop_info.get('fields_populated', 0)
            
            return api_response(True, data={
                "test_type": "full_workflow",
                "workflow": "instrument_identifier",
                "items_total": len(items),
                "items_enriched": sum(1 for item in items if item.get('enrichment_status') == 'success'),
                "schemas_populated": schemas_populated,
                "total_fields_populated": total_fields,
                "response": result.get('response', ''),
                "items": items
            })
        
        else:
            # Run only the Deep Agent integration directly
            from agentic.deep_agent_integration import integrate_deep_agent_specifications
            
            # Create test items
            test_items = [
                {
                    "number": 1,
                    "type": "instrument",
                    "name": "Pressure Transmitter",
                    "category": "Pressure Measurement",
                    "quantity": 1,
                    "sample_input": user_input
                }
            ]
            
            # Check if user mentions temperature
            if 'temperature' in user_input.lower():
                test_items.append({
                    "number": 2,
                    "type": "instrument",
                    "name": "Temperature Sensor",
                    "category": "Temperature Measurement",
                    "quantity": 1,
                    "sample_input": user_input
                })
            
            print(f"\n[TEST] Test items: {len(test_items)}")
            for item in test_items:
                print(f"  - {item['name']} ({item['type']})")
            
            print("\n[TEST] Running Deep Agent integration (specs-only mode)...")
            
            # Run Deep Agent with schema population DISABLED
            # This matches the production workflow behavior - just extract specs
            enriched = integrate_deep_agent_specifications(
                all_items=test_items,
                user_input=user_input,
                solution_context=None,
                domain=None,
                enable_schema_population=False  # Match production behavior
            )
            
            # Analyze results
            results_summary = []
            total_fields = 0
            schemas_populated = 0
            
            print("\n[TEST] RESULTS:")
            print("-" * 60)
            
            for item in enriched:
                status = item.get('enrichment_status', 'unknown')
                schema_pop = item.get('schema_populated', False)
                if schema_pop:
                    schemas_populated += 1
                
                schema = item.get('schema', {})
                pop_info = schema.get('_deep_agent_population', {})
                fields = pop_info.get('fields_populated', 0)
                total_fields += fields
                
                standards = item.get('applicable_standards', [])
                certs = item.get('certifications', [])
                
                item_info = {
                    "name": item.get('name'),
                    "enrichment_status": status,
                    "schema_populated": schema_pop,
                    "fields_populated": fields,
                    "standards_count": len(standards),
                    "certifications_count": len(certs),
                    "standards": [s.get('code', s) if isinstance(s, dict) else s for s in standards[:5]],
                    "certifications": certs[:5]
                }
                results_summary.append(item_info)
                
                print(f"\n  📋 {item.get('name')}")
                print(f"     Status: {status}")
                print(f"     Schema Populated: {'✅' if schema_pop else '❌'}")
                print(f"     Fields: {fields}, Standards: {len(standards)}, Certs: {len(certs)}")
            
            print("\n" + "-" * 60)
            print(f"[TEST] SUMMARY:")
            print(f"  Items: {len(enriched)}")
            print(f"  Schemas populated: {schemas_populated}")
            print(f"  Total fields: {total_fields}")
            print("=" * 80 + "\n")
            
            return api_response(True, data={
                "test_type": "direct_integration",
                "user_input": user_input,
                "items_total": len(enriched),
                "items_enriched": sum(1 for item in enriched if item.get('enrichment_status') == 'success'),
                "schemas_populated": schemas_populated,
                "total_fields_populated": total_fields,
                "items_summary": results_summary,
                "full_items": enriched
            })
    
    except Exception as e:
        print(f"\n[TEST] ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return api_response(False, error=str(e), status_code=500)


# ============================================================================
# HEALTH CHECK
# ============================================================================


@agentic_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return api_response(True, data={
        "status": "healthy",
        "service": "agentic-workflow"
    })
