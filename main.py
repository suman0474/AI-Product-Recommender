import asyncio
from datetime import datetime
from flask import Flask, request, jsonify, session, send_file
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import json
import logging
import re
import os
import urllib.parse
from werkzeug.utils import secure_filename
import requests
from io import BytesIO
from serpapi import GoogleSearch
import threading
import csv
from fuzzywuzzy import fuzz, process

from functools import wraps
from dotenv import load_dotenv

# Suppress noisy Azure SDK logs
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logging.getLogger("azure.identity").setLevel(logging.WARNING)

# --- NEW IMPORTS FOR SEARCH FUNCTIONALITY ---
from googleapiclient.discovery import build

# --- NEW IMPORTS FOR AUTHENTICATION ---
from auth_models import db, User, Log
from auth_utils import hash_password, check_password

# --- MongoDB Project Management ---
from mongo_project_manager import mongo_project_manager

# --- LLM CHAINING IMPORTS ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from chaining import setup_langchain_components, create_analysis_chain
import prompts  # Using compatibility shim (prompts.py) - TODO: Refactor to use sales_agent_tools.py
from loading import load_requirements_schema, build_requirements_schema_from_web
from flask import Flask, session
from flask_session import Session

# Import latest advanced specifications functionality
from advanced_parameters import discover_advanced_parameters

# Import Azure Blob utilities (MongoDB API compatible)
from azure_blob_utils import get_schema_from_mongodb, get_json_from_mongodb, MongoDBFileManager, mongodb_file_manager
from azure_blob_config import get_azure_blob_connection as get_mongodb_connection

# Import Swagger/Flasgger for API documentation
from flasgger import Swagger, swag_from

# Swagger/OpenAPI Configuration
swagger_template = {
    "swagger": "2.0",
    "info": {
        "title": "AI Product Recommender API",
        "description": "A comprehensive API for AI-powered product recommendations, vendor analysis, and intelligent procurement assistance. This API provides endpoints for product requirement analysis, vendor comparison, and workflow-based product selection.",
        "version": "1.0.0",
        "contact": {
            "name": "AI Product Recommender Team",
            "email": "support@aiproductrecommender.com"
        },
        "license": {
            "name": "Proprietary",
            "url": "https://aiproductrecommender.com/license"
        }
    },
    "host": "",  # Will be set dynamically
    "basePath": "/",
    "schemes": ["http", "https"],
    "securityDefinitions": {
        "session": {
            "type": "apiKey",
            "name": "session",
            "in": "cookie",
            "description": "Session-based authentication. Login via /login endpoint first."
        }
    },
    "security": [{"session": []}],
    "tags": [
        {"name": "Authentication", "description": "User authentication and session management"},
        {"name": "Health", "description": "API health check endpoints"},
        {"name": "Workflow", "description": "AI-powered product selection workflow"},
        {"name": "Products", "description": "Product analysis and recommendations"},
        {"name": "Vendors", "description": "Vendor search and analysis"},
        {"name": "Projects", "description": "Project management and persistence"},
        {"name": "Files", "description": "File upload and document processing"},
        {"name": "Images", "description": "Image serving and product images"},
        {"name": "Admin", "description": "Administrative operations"},
        {"name": "Standardization", "description": "Data standardization utilities"},
        {"name": "LangChain Tools", "description": "Direct testing of individual LangChain tools (16 tools)"},
        {"name": "LangChain Agents", "description": "Direct access to 9 LangChain agents with LangGraph ReAct pattern"},
        {"name": "Agentic Workflows", "description": "Base agentic AI workflows including chat and identification"},
        {"name": "Enhanced Workflows", "description": "Advanced multi-step workflows: Solution, Comparison, Detail Capture, PPI"},
        {"name": "Chat & Q&A", "description": "Grounded knowledge chat and Q&A endpoints"}
    ]
}

swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": "apispec",
            "route": "/apispec.json",
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/apidocs/"
}
from mongodb_projects import (
    save_project_to_mongodb,
    get_user_projects_from_mongodb,
    get_project_details_from_mongodb,
    delete_project_from_mongodb
)

# Load environment variables
load_dotenv()

# =========================================================================
# === FLASK APP CONFIGURATION ===
# =========================================================================
app = Flask(__name__, static_folder="static")

# Manual CORS handling

# A list of allowed origins for CORS
allowed_origins = [
    "https://ai-product-recommender-ui.vercel.app",  # Your production frontend
    "https://en-genie.vercel.app",                   # Your new frontend domain
    "http://localhost:8080",                         # Add your specific local dev port
    "http://localhost:5173",
    "http://localhost:3000"
]

# Dynamically add Vercel preview URLs to allowed_origins
if os.environ.get("VERCEL") == "1":
    # Add the production URL
    prod_url = os.environ.get("VERCEL_URL")
    if prod_url:
        allowed_origins.append(f"https://{prod_url}")
    # Add the preview URL for the current branch
    branch_url = os.environ.get("VERCEL_BRANCH_URL")
    if branch_url and branch_url != prod_url:
        allowed_origins.append(f"https://{branch_url}")

# Replace your old CORS line with this one
CORS(app, origins=allowed_origins, supports_credentials=True)
logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.INFO)

if os.getenv('FLASK_ENV') == 'production' or os.getenv('RAILWAY_ENVIRONMENT'):
    # Production session settings
    app.config["SESSION_PERMANENT"] = True
    app.config["SESSION_TYPE"] = "filesystem"
    app.config["SESSION_FILE_DIR"] = "/tmp/flask_session"
    app.config["SESSION_COOKIE_SECURE"] = True
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "None"
else:
    # Development session settings
    app.config["SESSION_PERMANENT"] = False
    app.config["SESSION_TYPE"] = "filesystem" 

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = os.getenv('SECRET_KEY', 'fallback-secret-key-for-development')
Session(app)
db.init_app(app)

# Initialize Swagger/Flasgger for API documentation
swagger = Swagger(app, template=swagger_template, config=swagger_config)

# --- Initialize Rate Limiting ---
from rate_limiter import init_limiter
limiter = init_limiter(app)
logging.info("Rate limiting initialized successfully")

# --- Import and Register Agentic Workflow Blueprint ---
from agentic.api import agentic_bp
app.register_blueprint(agentic_bp)
logging.info("Agentic workflow blueprint registered at /api/agentic")

# --- Import and Register Deep Agent Blueprint ---
from agentic.deep_agent.api import deep_agent_bp
app.register_blueprint(deep_agent_bp)
logging.info("Deep Agent blueprint registered at /deep-agent")

# --- Import and Register EnGenie Chat API Blueprint ---
from agentic.engenie_chat.engenie_chat_api import engenie_chat_bp
app.register_blueprint(engenie_chat_bp)
logging.info("EnGenie Chat API blueprint registered at /api/engenie-chat")

# --- Import and Register Tools API Blueprint ---
# --- Import and Register Tools API Blueprint ---
from tools_api import tools_bp
app.register_blueprint(tools_bp)
logging.info("Tools API blueprint registered at /api/tools")


# --- Import and Register Session API Blueprints ---
from agentic.session_api import register_session_blueprints
register_session_blueprints(app)
logging.info("Session and Instance blueprints registered")


# =========================================================================
# === HELPER FUNCTIONS AND UTILITIES ===
# =========================================================================
# --- Import and Register Resource Monitoring Blueprint ---
try:
    from agentic.resource_monitoring_api import resource_bp
    app.register_blueprint(resource_bp)
    logging.info("Resource Monitoring blueprint registered at /api/resources")
except ImportError:
    logging.warning("Resource Monitoring API not available")
from test import (
    extract_data_from_pdf,
    send_to_language_model,
    aggregate_results,
    generate_dynamic_path,
    split_product_types,
    save_json
    # Removed: identify_and_save_product_image - no longer needed with API-based images
)

# Import standardization utilities
from standardization_utils import (
    standardize_vendor_analysis_result,
    standardize_ranking_result,
    enhance_submodel_mapping,
    standardize_product_image_mapping,
    create_standardization_report,
    update_existing_vendor_files_with_standardization,
    standardize_vendor_name
)

# Initialize LangChain components
try:
    components = setup_langchain_components()
    analysis_chain = create_analysis_chain(components)  # Uses MongoDB, no local paths needed
    
    logging.info("LangChain components initialized.")
except Exception as e:
    logging.error(f"Initialization failed: {e}")
    components = None
    analysis_chain = None

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

def apply_standardization_to_response(data):
    """
    Apply appropriate standardization to response data based on its content.
    This function detects the type of data and applies the correct standardization.
    Only applies LLM-based standardization for critical analysis endpoints.
    """
    if not isinstance(data, dict):
        return data
    
    try:
        # Only apply LLM-based standardization for analysis results
        # Skip for basic endpoints like /vendors to prevent connection issues
        
        # For nested dictionaries, apply standardization recursively to relevant parts
        standardized = data.copy()
        for key, value in data.items():
            if key in ["vendor_analysis", "vendorAnalysis"] and isinstance(value, dict):
                try:
                    standardized[key] = standardize_vendor_analysis_result(value)
                except Exception as e:
                    logging.warning(f"Failed to standardize vendor analysis: {e}")
                    standardized[key] = value
            elif key in ["overall_ranking", "overallRanking", "ranking"] and isinstance(value, dict):
                try:
                    standardized[key] = standardize_ranking_result(value)
                except Exception as e:
                    logging.warning(f"Failed to standardize ranking: {e}")
                    standardized[key] = value
            elif key in ["vendors"] and isinstance(value, list):
                # Apply basic vendor name standardization only
                standardized_vendors = []
                for vendor in value:
                    if isinstance(vendor, dict) and "name" in vendor:
                        vendor_copy = vendor.copy()
                        try:
                            vendor_copy["name"] = standardize_vendor_name(vendor.get("name", ""))
                        except Exception as e:
                            logging.warning(f"Failed to standardize vendor name: {e}")
                            vendor_copy["name"] = vendor.get("name", "")
                        standardized_vendors.append(vendor_copy)
                    else:
                        standardized_vendors.append(vendor)
                standardized[key] = standardized_vendors
        return standardized
    
    except Exception as e:
        logging.warning(f"Standardization failed for response data: {e}")
        return data
    
    return data

def standardized_jsonify(data, status_code=200):
    """
    Enhanced jsonify function that applies standardization before converting to camelCase.
    Use this instead of jsonify() for responses containing vendor/product data.
    """
    try:
        # Apply standardization first (with timeout protection)
        standardized_data = apply_standardization_to_response(data)
        
        # Then convert to camelCase for frontend compatibility
        camel_case_data = convert_keys_to_camel_case(standardized_data)
        
        return jsonify(camel_case_data), status_code
    except Exception as e:
        logging.error(f"Failed to apply standardization in jsonify wrapper: {e}")
        # Fallback to regular conversion if standardization fails
        try:
            camel_case_data = convert_keys_to_camel_case(data)
            return jsonify(camel_case_data), status_code
        except Exception as fallback_error:
            logging.error(f"Even fallback conversion failed: {fallback_error}")
            # Last resort: return data as-is
            return jsonify(data), status_code

def prettify_req(req):
    return req.replace('_', ' ').replace('-', ' ').title()

def flatten_schema(schema_dict):
    flat = {}
    for k, v in schema_dict.items():
        if isinstance(v, dict):
            for subk, subv in v.items():
                flat[subk] = subv
        else:
            flat[k] = v
    return flat

def login_required(func):
    @wraps(func)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({"error": "Unauthorized: Please log in"}), 401
        return func(*args, **kwargs)
    return decorated_function

ALLOWED_EXTENSIONS = {"pdf"}

def allowed_file(filename: str):
    """Check if the uploaded filename has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# =============================================================================
# OLD /api/intent ENDPOINT (COMMENTED OUT)
# This endpoint used an independent LLM chain for intent classification.
# Now replaced with a wrapper that calls classify_intent_tool from intent_tools.py
# for consistent intent classification across the system.
# =============================================================================
# @app.route("/api/intent", methods=["POST"])
# @login_required
# def api_intent_old():
#     """
#     [DEPRECATED] Old intent classification endpoint.
#     Kept for reference - used independent LLM chain instead of intent_tools.
#     """
#     pass
# =============================================================================

# Import the intent tool for the new wrapper API
from tools.intent_tools import classify_intent_tool

@app.route("/api/intent", methods=["POST"])
@login_required
def api_intent():
    """
    Classify user intent and determine workflow step (Wrapper for classify_intent_tool)
    ---
    tags:
      - Workflow
    summary: Classify user intent
    description: |
      Classifies user intent and determines the next workflow step based on input and current session state.
      This endpoint wraps the classify_intent_tool from intent_tools.py for consistent classification.
      
      **Intent Types:**
      - `greeting` - Initial greeting
      - `solution` - Complex engineering challenge requiring multiple instruments
      - `requirements` - Single product specifications
      - `question` - Asking about industrial topics
      - `productRequirements` - User is providing product specifications
      - `knowledgeQuestion` - User is asking a question
      - `workflow` - Continuing an existing workflow
      - `confirm` / `reject` - User confirmations/rejections
      - `chitchat` - Casual conversation
      - `other` - Unrecognized intent
    consumes:
      - application/json
    produces:
      - application/json
    parameters:
      - in: body
        name: body
        description: User input for intent classification
        required: true
        schema:
          type: object
          required:
            - userInput
          properties:
            userInput:
              type: string
              description: The user's message or input to classify
              example: "I need a pressure transmitter with HART protocol"
            search_session_id:
              type: string
              description: Session ID for workflow isolation
              example: "session_12345"
    responses:
      200:
        description: Intent classification result
        schema:
          type: object
          properties:
            intent:
              type: string
              enum: [greeting, solution, requirements, question, productRequirements, knowledgeQuestion, workflow, confirm, reject, chitchat, other]
              description: Classified intent type
            nextStep:
              type: string
              description: Next workflow step to navigate to
              example: "initialInput"
            resumeWorkflow:
              type: boolean
              description: Whether to resume the current workflow
            confidence:
              type: number
              description: Confidence score (0.0-1.0)
            isSolution:
              type: boolean
              description: Whether this is a complex engineering solution request
      400:
        description: Bad request - missing userInput
        schema:
          type: object
          properties:
            error:
              type: string
              example: "userInput is required"
      401:
        description: Unauthorized - login required
      500:
        description: Intent classification failed
    """
    data = request.get_json(force=True)
    user_input = data.get("userInput", "").strip()
    if not user_input:
        return jsonify({"error": "userInput is required"}), 400

    # Get search session ID if provided (for session isolation)
    search_session_id = data.get("search_session_id", "default")

    # Get current workflow state from session (session-isolated)
    current_step_key = f'current_step_{search_session_id}'
    current_intent_key = f'current_intent_{search_session_id}'
    
    current_step = session.get(current_step_key, None)
    current_intent = session.get(current_intent_key, None)
    
    # --- Handle skip for missing mandatory fields ---
    # Accept both legacy and frontend step names when user wants to skip missing mandatory fields
    if current_step in ("awaitMandatory", "awaitMissingInfo") and user_input.lower() in ["yes", "skip", "y"]:
        session[f'current_step_{search_session_id}'] = "awaitAdditionalAndLatestSpecs"
        response = {
            "intent": "workflow",
            "nextStep": "awaitAdditionalAndLatestSpecs",
            "resumeWorkflow": True,
            "message": "Skipping missing mandatory fields. Additional and latest specifications are available."
        }
        return jsonify(response), 200
    
    # =========================================================================
    # USE WorkflowStateMemory FROM IntentClassificationRoutingAgent
    # This is the SINGLE SOURCE OF TRUTH for workflow state (stored in backend)
    # =========================================================================
    from agentic.intent_classification_routing_agent import (
        get_workflow_memory, 
        should_exit_workflow,
        is_knowledge_question,  # NEW: Detect knowledge questions
        EXIT_PHRASES,
        GREETING_PHRASES
    )
    
    workflow_memory = get_workflow_memory()
    
    # Check if user wants to exit workflow
    wants_to_exit = should_exit_workflow(user_input)
    
    # NEW: Check if user is asking a knowledge question (should break workflow lock)
    is_knowledge_q = is_knowledge_question(user_input)
    
    # If user wants to exit OR is asking a knowledge question, clear workflow state
    if wants_to_exit or is_knowledge_q:
        workflow_memory.clear_workflow(search_session_id)
        if is_knowledge_q:
            logging.info(f"[INTENT_API] Clearing workflow state - detected KNOWLEDGE QUESTION: {user_input[:50]}...")
        else:
            logging.info(f"[INTENT_API] Clearing workflow state - user requested exit")
    
    # Check WORKFLOW LOCK from backend memory (single source of truth)
    current_workflow = workflow_memory.get_workflow(search_session_id)
    
    if current_workflow and not wants_to_exit and not is_knowledge_q:
        logging.info(f"[INTENT_API] WORKFLOW LOCK: User is in '{current_workflow}' workflow - staying in workflow")
        
        # Map workflow to intent
        workflow_to_intent = {
            "engenie_chat": "knowledgeQuestion",
            "productInfo": "knowledgeQuestion", 
            "index_rag": "knowledgeQuestion",
            "instrument_identifier": "productRequirements",
            "solution": "solution"
        }
        
        locked_intent = workflow_to_intent.get(current_workflow, current_workflow)
        
        result_json = {
            "intent": locked_intent,
            "nextStep": None,
            "resumeWorkflow": True,
            "confidence": 1.0,
            "isSolution": current_workflow == "solution",
            "extractedInfo": {},
            "solutionIndicators": [],
            "workflowLocked": True,
            "currentWorkflow": current_workflow
        }
        
        logging.info(f"[INTENT_API] Workflow locked response: {result_json}")
        return jsonify(result_json), 200

    try:
        # =====================================================================
        # CALL classify_intent_tool from intent_tools.py
        # This is the centralized intent classification tool
        # =====================================================================
        logging.info(f"[INTENT_API] Calling classify_intent_tool for input: {user_input[:100]}...")
        
        # Build context from session state
        context = f"Current step: {current_step or 'None'}, Current intent: {current_intent or 'None'}, Current workflow: {current_workflow or 'None'}"
        
        # Invoke the classify_intent_tool
        tool_result = classify_intent_tool.invoke({
            "user_input": user_input,
            "current_step": current_step,
            "context": context
        })
        
        logging.info(f"[INTENT_API] classify_intent_tool result: {tool_result}")
        
        if not tool_result.get("success"):
            # Tool failed, return error
            return jsonify({
                "error": tool_result.get("error", "Intent classification failed"),
                "intent": "other",
                "nextStep": None,
                "resumeWorkflow": False
            }), 500
        
        # Map tool result to API response format
        intent = tool_result.get("intent", "other")
        is_solution = tool_result.get("is_solution", False)
        confidence = tool_result.get("confidence", 0.5)
        next_step = tool_result.get("next_step")
        
        # Map intent types to frontend expected values
        intent_mapping = {
            "greeting": "greeting",
            "solution": "solution",
            "requirements": "productRequirements",
            "question": "knowledgeQuestion",
            "additional_specs": "workflow",
            "confirm": "workflow",
            "reject": "workflow",
            "chitchat": "chitchat",
            "unrelated": "other"
        }
        
        mapped_intent = intent_mapping.get(intent, intent)
        
        # Determine next step and workflow based on intent
        suggest_workflow = None  # For suggesting workflows without auto-routing
        
        if mapped_intent == "greeting":
            next_step = "greeting"
            resume_workflow = False
            new_workflow = None  # Clear workflow on greeting
        elif mapped_intent == "solution" or is_solution:
            mapped_intent = "solution"
            next_step = "solutionWorkflow"
            resume_workflow = False
            is_solution = True
            new_workflow = "solution"
        elif mapped_intent == "productRequirements":
            next_step = "initialInput"
            resume_workflow = False
            new_workflow = "instrument_identifier"
        elif mapped_intent == "knowledgeQuestion":
            # DON'T auto-route to EnGenie Chat - suggest it instead
            next_step = None
            resume_workflow = False
            new_workflow = None  # Don't set workflow yet
            suggest_workflow = {
                "name": "EnGenie Chat",
                "workflow_id": "engenie_chat",
                "description": "Get answers about products, standards, and industrial topics",
                "action": "openEnGenieChat"
            }
            logging.info(f"[INTENT_API] Suggesting EnGenie Chat workflow instead of auto-routing")
        elif mapped_intent == "workflow":
            resume_workflow = True
            new_workflow = current_workflow  # Keep current workflow
        else:
            resume_workflow = False
            new_workflow = None  # Don't auto-set workflow
            suggest_workflow = {
                "name": "EnGenie Chat",
                "workflow_id": "engenie_chat", 
                "description": "Get answers about products, standards, and industrial topics",
                "action": "openEnGenieChat"
            }
        
        # Build response
        result_json = {
            "intent": mapped_intent,
            "nextStep": next_step,
            "resumeWorkflow": resume_workflow,
            "confidence": confidence,
            "isSolution": is_solution,
            "extractedInfo": tool_result.get("extracted_info", {}),
            "solutionIndicators": tool_result.get("solution_indicators", []),
            "currentWorkflow": new_workflow,
            "suggestWorkflow": suggest_workflow  # Suggestion for UI to display
        }
        
        # Update session based on classification
        # Use WorkflowStateMemory for workflow state (single source of truth)
        if mapped_intent == "greeting":
            session[f'current_step_{search_session_id}'] = 'greeting'
            session[f'current_intent_{search_session_id}'] = 'greeting'
            workflow_memory.clear_workflow(search_session_id)  # Clear workflow on greeting
        elif mapped_intent == "solution" or is_solution:
            session[f'current_step_{search_session_id}'] = 'solutionWorkflow'
            session[f'current_intent_{search_session_id}'] = 'solution'
            workflow_memory.set_workflow(search_session_id, 'solution')
            logging.info("[SOLUTION_ROUTING] Detected solution/engineering challenge - routing to solution workflow")
        elif mapped_intent == "productRequirements":
            session[f'current_step_{search_session_id}'] = 'initialInput'
            session[f'current_intent_{search_session_id}'] = 'productRequirements'
            workflow_memory.set_workflow(search_session_id, 'instrument_identifier')
        elif mapped_intent == "knowledgeQuestion":
            session[f'current_intent_{search_session_id}'] = 'knowledgeQuestion'
            workflow_memory.set_workflow(search_session_id, 'engenie_chat')
        elif mapped_intent == "workflow" and next_step:
            session[f'current_step_{search_session_id}'] = next_step
            session[f'current_intent_{search_session_id}'] = 'workflow'
        
        logging.info(f"[INTENT_API] Final response: {result_json}")
        logging.info(f"[INTENT_API] Workflow set to: {workflow_memory.get_workflow(search_session_id)}")
        return jsonify(result_json), 200

    except Exception as e:
        error_msg = str(e)
        logging.exception("[INTENT_API] Intent classification failed.")
        
        # Return 503 for rate limit errors (temporary condition), 500 for other errors
        is_rate_limit = any(x in error_msg for x in ['429', 'Resource exhausted', 'RESOURCE_EXHAUSTED', 'quota'])
        status_code = 503 if is_rate_limit else 500
        
        return jsonify({
            "error": error_msg, 
            "intent": "other", 
            "nextStep": None, 
            "resumeWorkflow": False,
            "retryable": is_rate_limit
        }), status_code

@app.route('/health', methods=['GET'])
@login_required
def health_check():
    """
    API Health Check
    ---
    tags:
      - Health
    summary: Check API health status
    description: Returns the current health status of the API and its components.
    produces:
      - application/json
    responses:
      200:
        description: API is healthy
        schema:
          type: object
          properties:
            status:
              type: string
              example: "healthy"
            workflow_initialized:
              type: boolean
              description: Whether workflow engine is initialized
            langsmith_enabled:
              type: boolean
              description: Whether LangSmith monitoring is enabled
      401:
        description: Unauthorized - login required
    """
    return {
        "status": "healthy",
        "workflow_initialized": False,
        "langsmith_enabled": False
    }, 200

# =========================================================================
# === IMAGE SERVING ENDPOINT (Azure Blob Storage Primary)
# =========================================================================
@app.route('/api/images/<path:file_id>', methods=['GET'])
def serve_image(file_id):
    """
    Serve images from Azure Blob Storage
    ---
    tags:
      - Images
    summary: Serve image from Azure Blob Storage
    description: Retrieves and serves an image stored in Azure Blob Storage by its file ID (UUID).
    produces:
      - image/jpeg
      - image/png
      - image/gif
      - image/webp
      - application/json
    parameters:
      - name: file_id
        in: path
        type: string
        required: true
        description: Azure Blob file ID (UUID format)
        example: "083015c2-300c-44f3-be9a-3fdafebc39b0"
    responses:
      200:
        description: Image data returned successfully
        headers:
          Cache-Control:
            type: string
            description: Caching directive
            example: "public, max-age=2592000"
      404:
        description: Image not found
        schema:
          type: object
          properties:
            error:
              type: string
              example: "Image not found"
      500:
        description: Internal server error
    """
    try:
        from azure_blob_config import azure_blob_manager, is_azure_blob_available
        import os
        
        # Check if Azure Blob Storage is available
        if not is_azure_blob_available():
            logging.error(f"Azure Blob Storage is not available for serving image: {file_id}")
            return jsonify({"error": "Image storage service unavailable"}), 503
        
        container_client = azure_blob_manager.container_client
        base_path = azure_blob_manager.base_path
        
        # Search paths: files (UUIDs), images (cached), generic_images (generated)
        search_paths = [
            f"{base_path}/files",
            f"{base_path}/images",
            f"{base_path}/generic_images"
        ]
        
        # Search for the blob with matching file_id
        image_blob = None
        blob_name = None
        content_type = 'image/png'  # Default content type
        
        # Optimization: Check if file_id is a full blob path (e.g., "generic_images/filename.png")
        # Try direct access first before searching all blobs
        if '/' in file_id:
            try:
                # file_id might be a relative path like "generic_images/viscousliquidflowtransmitter.png"
                full_blob_path = f"{base_path}/{file_id}"
                blob_client = container_client.get_blob_client(full_blob_path)
                blob_properties = blob_client.get_blob_properties()
                if blob_properties:
                    blob_name = full_blob_path
                    if blob_properties.content_settings and blob_properties.content_settings.content_type:
                        content_type = blob_properties.content_settings.content_type
                    logging.info(f"[SERVE_IMAGE] Direct access successful for blob path: {file_id}")
            except Exception as direct_err:
                logging.debug(f"[SERVE_IMAGE] Direct access failed for {file_id}: {direct_err}, falling back to search")
        
        # Only search if direct access didn't find the blob
        if blob_name is None:
            try:
                for path in search_paths:
                    if image_blob: 
                        break
                        
                    # List blobs and find the one with matching file_id in the name or metadata
                    blobs = container_client.list_blobs(
                        name_starts_with=path,
                        include=['metadata']
                    )
                    
                    for blob in blobs:
                        # Check if file_id is in the blob name (UUID match or simple filename match)
                        # Note: We check if the passed file_id is contained in the blob name
                        # This handles UUIDs "uuid_filename.png" and simple names "vendor_model.jpg"
                        if file_id in blob.name:
                            image_blob = blob
                            blob_name = blob.name
                            if blob.content_settings and blob.content_settings.content_type:
                                content_type = blob.content_settings.content_type
                            break
                        
                        # Also check metadata for file_id
                        if blob.metadata and blob.metadata.get('file_id') == file_id:
                            image_blob = blob
                            blob_name = blob.name
                            if blob.content_settings and blob.content_settings.content_type:
                                content_type = blob.content_settings.content_type
                            break
            
            except Exception as e:
                logging.warning(f"Error searching for image blob {file_id}: {e}")
        
        if blob_name is None:
            logging.error(f"Image not found in Azure Blob Storage: {file_id}")
            return jsonify({"error": "Image not found"}), 404
        
        # Download the blob content
        try:
            blob_client = container_client.get_blob_client(blob_name)
            download_stream = blob_client.download_blob()
            image_data = download_stream.readall()
        except Exception as e:
            logging.error(f"Failed to download image blob {file_id}: {e}")
            return jsonify({"error": "Failed to retrieve image"}), 500
        
        # Extract filename from blob name
        filename = os.path.basename(blob_name)
        
        # Create response with proper headers
        response = send_file(
            BytesIO(image_data),
            mimetype=content_type,
            as_attachment=False,
            download_name=filename
        )
        
        # Add caching headers (cache for 30 days)
        response.headers['Cache-Control'] = 'public, max-age=2592000'
        
        logging.info(f"Served image from Azure Blob Storage: {file_id} ({len(image_data)} bytes)")
        return response
        
    except Exception as e:
        logging.exception(f"Error serving image {file_id}: {e}")
        return jsonify({"error": "Internal server error"}), 500

# Fallback route for images requested without /api/images/ prefix
@app.route('/<string:file_id>', methods=['GET'])
def serve_image_root(file_id):
    """
    Fallback for images requested at root (e.g. legacy or malformed URLs)
    Only handles UUID format or long hashes to avoid conflict with other routes
    """
    # Check if it's a UUID format (36 chars with hyphens: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)
    import re
    uuid_pattern = r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
    if re.match(uuid_pattern, file_id):
        return serve_image(file_id)
    
    # Also handle old formats: hash (64 chars) or ObjectId (24 chars)
    if len(file_id) in [24, 64] and all(c in '0123456789abcdefABCDEF' for c in file_id):
        return serve_image(file_id)
    
    # Otherwise let it 404 (or be handled by frontend router if applicable)
    return jsonify({"error": "Not found"}), 404

@app.route('/api/generic_image/<product_type>', methods=['GET'])
@login_required
def get_generic_image(product_type):
    """
    Fetch generic product type image with MongoDB caching
    
    Strategy:
    1. Check MongoDB cache first
    2. If not found, search external APIs with "generic <product_type>"
    3. Cache the result
    4. Return image URL or GridFS file ID
    
    Args:
        product_type: Product type name (e.g., "Pressure Transmitter")
    """
    try:
        from generic_image_utils import fetch_generic_product_image
        
        # Decode URL-encoded product type
        import urllib.parse
        decoded_product_type = urllib.parse.unquote(product_type)
        
        logging.info(f"[API] ===== Generic Image Request START =====")
        logging.info(f"[API] Product Type (raw): {product_type}")
        logging.info(f"[API] Product Type (decoded): {decoded_product_type}")
        
        # Fetch image (checks cache first, then external APIs)
        image_result = fetch_generic_product_image(decoded_product_type)
        
        if image_result:
            logging.info(f"[API] ✓ Image found! Source: {image_result.get('source')}, Cached: {image_result.get('cached')}")
            logging.info(f"[API] ===== Generic Image Request END (SUCCESS) =====")
            return jsonify({
                "success": True,
                "image": image_result,
                "product_type": decoded_product_type
            }), 200
        else:
            logging.warning(f"[API] ✗ No image found for: {decoded_product_type}")
            logging.info(f"[API] ===== Generic Image Request END (NOT FOUND) =====")
            return jsonify({
                "success": False,
                "error": "No image found",
                "product_type": decoded_product_type
            }), 404
            
    except Exception as e:
        logging.exception(f"[API] ✗ ERROR fetching generic image for {product_type}: {e}")
        logging.info(f"[API] ===== Generic Image Request END (ERROR) =====")
        return jsonify({
            "success": False,
            "error": str(e),
            "product_type": product_type
        }), 500


@app.route('/api/generic_image_fast/<product_type>', methods=['GET'])
@login_required
def get_generic_image_fast(product_type):
    """
    Fetch generic product type image with FAST-FAIL behavior.
    
    Unlike the regular endpoint, this returns IMMEDIATELY if:
    - Cache is empty AND LLM is rate-limited
    
    Returns a 'use_placeholder' flag so the frontend can show a placeholder.
    
    Response:
        {
            "success": true/false,
            "image": { "url": "...", ... } or null,
            "use_placeholder": true/false,
            "reason": "cached" | "generated" | "rate_limited"
        }
    """
    try:
        from generic_image_utils import fetch_generic_product_image_fast
        import urllib.parse
        
        decoded_product_type = urllib.parse.unquote(product_type)
        logging.info(f"[API_FAST] Fast image request: {decoded_product_type}")
        
        result = fetch_generic_product_image_fast(decoded_product_type)
        
        if result.get('success'):
            logging.info(f"[API_FAST] ✓ Image found: {result.get('reason')}")
            return jsonify({
                "success": True,
                "image": result,
                "product_type": decoded_product_type,
                "use_placeholder": False,
                "reason": result.get('reason', 'cached')
            }), 200
        else:
            logging.warning(f"[API_FAST] ✗ Using placeholder: {result.get('reason')}")
            return jsonify({
                "success": False,
                "image": None,
                "product_type": decoded_product_type,
                "use_placeholder": True,
                "reason": result.get('reason', 'rate_limited')
            }), 200  # Return 200 with use_placeholder=True
            
    except Exception as e:
        logging.exception(f"[API_FAST] Error: {e}")
        return jsonify({
            "success": False,
            "image": None,
            "product_type": product_type,
            "use_placeholder": True,
            "reason": "error",
            "error": str(e)
        }), 200  # Return 200 so frontend doesn't show error


@app.route('/api/generic_images/batch', methods=['POST'])
@login_required
def get_generic_images_batch():
    """
    Fetch generic product type images for MULTIPLE product types IN PARALLEL.
    
    PARALLELIZATION STRATEGY:
    - Phase 1: Check Azure cache for ALL product types simultaneously (fast)
    - Phase 2: For cache misses, generate with LLM (sequential to respect rate limits)
    
    Request Body:
        {
            "product_types": ["Pressure Transmitter", "Flow Meter", ...]
        }
    
    Returns:
        {
            "success": true,
            "images": {
                "Pressure Transmitter": {...image_result...},
                "Flow Meter": {...image_result...}
            },
            "cache_hits": 5,
            "cache_misses": 2,
            "processing_time_ms": 1234
        }
    """
    try:
        from generic_image_utils import fetch_generic_images_batch
        import time
        
        start_time = time.time()
        
        data = request.get_json()
        if not data or 'product_types' not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'product_types' in request body"
            }), 400
        
        product_types = data.get('product_types', [])
        if not isinstance(product_types, list):
            return jsonify({
                "success": False,
                "error": "'product_types' must be a list"
            }), 400
        
        # Deduplicate and clean
        product_types = list(set([pt.strip() for pt in product_types if pt and isinstance(pt, str)]))
        
        if not product_types:
            return jsonify({
                "success": True,
                "images": {},
                "cache_hits": 0,
                "cache_misses": 0,
                "processing_time_ms": 0
            }), 200
        
        logging.info(f"[API_BATCH] Starting batch fetch for {len(product_types)} product types...")
        
        # Fetch all images in parallel
        results = fetch_generic_images_batch(product_types)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        # Count hits/misses
        cache_hits = sum(1 for r in results.values() if r and r.get('cached'))
        cache_misses = len(product_types) - cache_hits
        
        logging.info(f"[API_BATCH] Completed: {cache_hits} hits, {cache_misses} misses in {processing_time}ms")
        
        return jsonify({
            "success": True,
            "images": results,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "total_requested": len(product_types),
            "processing_time_ms": processing_time
        }), 200
        
    except Exception as e:
        logging.exception(f"[API_BATCH] Error in batch image fetch: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# =========================================================================
# === FILE UPLOAD AND TEXT EXTRACTION ENDPOINT ===
# =========================================================================
@app.route('/api/upload-requirements', methods=['POST'])
@login_required
def upload_requirements_file():
    """
    Upload file (PDF, DOCX, TXT, Images) and extract text as requirements
    
    Accepts: multipart/form-data with 'file' field
    Returns: Extracted text from the file
    """
    try:
        from file_extraction_utils import extract_text_from_file
        
        logging.info("[API] ===== File Upload Request START =====")
        
        # Check if file is present
        if 'file' not in request.files:
            logging.warning("[API] No file provided in request")
            return jsonify({
                "success": False,
                "error": "No file provided"
            }), 400
        
        file = request.files['file']
        
        # Check if filename is empty
        if file.filename == '':
            logging.warning("[API] Empty filename")
            return jsonify({
                "success": False,
                "error": "No file selected"
            }), 400
        
        # Read file bytes
        file_bytes = file.read()
        filename = file.filename
        
        logging.info(f"[API] Processing file: {filename} ({len(file_bytes)} bytes)")
        
        # Extract text from file
        extraction_result = extract_text_from_file(file_bytes, filename)
        
        if not extraction_result['success']:
            logging.warning(f"[API] Failed to extract text from {filename}")
            return jsonify({
                "success": False,
                "error": f"Could not extract text from {extraction_result['file_type']} file",
                "file_type": extraction_result['file_type']
            }), 400
        
        logging.info(f"[API] ✓ Successfully extracted {extraction_result['character_count']} characters from {filename}")
        logging.info("[API] ===== File Upload Request END (SUCCESS) =====")
        
        return jsonify({
            "success": True,
            "extracted_text": extraction_result['extracted_text'],
            "filename": filename,
            "file_type": extraction_result['file_type'],
            "character_count": extraction_result['character_count']
        }), 200
        
    except Exception as e:
        logging.exception(f"[API] ✗ ERROR processing file upload: {e}")
        logging.info("[API] ===== File Upload Request END (ERROR) =====")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# =========================================================================
# === CORE LLM-BASED SALES AGENT ENDPOINT ===
# =========================================================================
@app.route("/api/sales-agent", methods=["POST"])
@login_required
def api_sales_agent():
    """
    AI Sales Agent - Workflow Engine
    ---
    tags:
      - Workflow
    summary: Process sales agent workflow step
    description: |
      Core AI-powered endpoint that handles the step-based product selection workflow.
      Supports session tracking, knowledge question handling, and maintains conversation flow.
      
      **Workflow Steps:**
      - `initialInput` - Initial product requirements
      - `awaitMissingInfo` - Waiting for missing mandatory fields
      - `awaitAdditionalAndLatestSpecs` - Additional specifications
      - `awaitAdvancedSpecs` - Advanced parameter specifications
      - `showSummary` - Display requirements summary
    consumes:
      - application/json
    produces:
      - application/json
    parameters:
      - in: body
        name: body
        description: Workflow step data
        required: true
        schema:
          type: object
          properties:
            step:
              type: string
              description: Current workflow step
              example: "initialInput"
            userMessage:
              type: string
              description: User's message or input
              example: "I need a pressure transmitter"
            intent:
              type: string
              description: Classified intent from /api/intent
            dataContext:
              type: object
              description: Context data for the current step
            search_session_id:
              type: string
              description: Session ID for workflow isolation
    responses:
      200:
        description: Workflow response
        schema:
          type: object
          properties:
            content:
              type: string
              description: AI-generated response message
            nextStep:
              type: string
              description: Next workflow step
            maintainWorkflow:
              type: boolean
              description: Whether to maintain current workflow
      401:
        description: Unauthorized - login required
      503:
        description: Service unavailable - LLM not ready
    """
    if not components or not components.get('llm'):
        return jsonify({"error": "LLM component is not ready."}), 503

    try:
        # NOTE: request.get_json(force=True) is used for debugging/non-standard headers
        data = request.get_json(force=True)
        # Debug log incoming product_type information to trace saving issues (sales agent)
        incoming_pt = data.get('product_type') if isinstance(data, dict) else None
        incoming_detected = data.get('detected_product_type') if isinstance(data, dict) else None
        logging.info(f"[SALES_AGENT_INCOMING] Incoming product_type='{incoming_pt}' detected_product_type='{incoming_detected}' project_name='{data.get('project_name') if isinstance(data, dict) else None}'")
        

        
        # Continue with normal sales-agent workflow if no CSV vendors
        step = data.get("step")
        data_context = data.get("dataContext", {})
        user_message = data.get("userMessage", "")
        intent = data.get("intent", "")
        search_session_id = data.get("search_session_id", "default")

        # Log session-specific request for debugging
        logging.info(f"[SALES_AGENT] Session {search_session_id}: Step={step}, Intent={intent}")

        # Get session state for workflow continuity (session-isolated)
        current_step_key = f'current_step_{search_session_id}'
        current_intent_key = f'current_intent_{search_session_id}'
        current_step = session.get(current_step_key)
        current_intent = session.get(current_intent_key)
        
        # --- Helper function for formatting advanced parameters ---
        def format_available_parameters(params):
            """
            Return parameter keys one by one, each on a separate line.
            Each item in `params` can be:
            - A dict with 'key' field: {"key": "parameter_key_name", ...}
            - A string: "parameter_key_name"
            Replaces underscores with spaces and formats for display.
            """
            formatted = []
            for param in params:
                # Extract the key from dict or use string directly
                if isinstance(param, dict):
                    # Priority: prefer human-friendly 'name' field, then 'key',
                    # then fall back to the first dict key if present.
                    name = param.get('name') or param.get('key') or (list(param.keys())[0] if param else '')
                else:
                    # Parameter keys are typically strings like "parameter_key_name"
                    name = str(param).strip()

                # Replace underscores with spaces
                name = name.replace('_', ' ')
                # Remove any parenthetical or bracketed content by taking text
                # before the first bracket. Example: "X (Y)" -> "X"
                name = re.split(r'[\(\[\{]', name, 1)[0].strip()
                # Normalize spacing (remove extra spaces)
                name = " ".join(name.split())
                # Title case for display (first letter of each word capitalized)
                name = name.title()

                # Prefix with a bullet so the list appears as points one per line
                formatted.append(f"- {name}")
            return "\n".join(formatted)

        # Treat short affirmative/negative replies (e.g., 'yes', 'no') as
        # workflow input regardless of the classifier. The frontend's
        # intent classifier can sometimes label brief confirmations as
        # knowledge questions; we prefer to route them into the sales-agent
        # workflow so steps like awaitAdditionalAndLatestSpecs and
        # awaitAdvancedSpecs behave deterministically.
        try:
            short_yesno_re = re.compile(r"^\s*(?:yes|y|yeah|yep|sure|ok|okay|no|n|nope|skip)\b[\.\!\?\s]*$", re.IGNORECASE)
        except Exception:
            short_yesno_re = None

        if isinstance(user_message, str) and short_yesno_re and short_yesno_re.match(user_message):
            matched = short_yesno_re.match(user_message).group(0).strip()
            if intent == 'knowledgeQuestion':
                logging.info(f"[SALES_AGENT] Overriding intent 'knowledgeQuestion' for short reply: '{user_message}' (matched='{matched}')")
            intent = 'workflow'
            logging.info(f"[SALES_AGENT] Routed short reply to workflow branch: '{user_message}' (matched='{matched}', step={step})")

        # Handle knowledge questions - answer and resume workflow
        if intent == "knowledgeQuestion":
            # Determine context-aware response based on current workflow step
            if step == "awaitMissingInfo":
                context_hint = "Once you have the information you need, please provide the missing details so we can continue with your product selection or Would you like to continue anyway?"
            elif step == "awaitAdditionalAndLatestSpecs":
                context_hint = "Now, let's continue - would you like to add additional and latest specifications?"
            elif step == "awaitAdvancedSpecs":
                context_hint = "Now, let's continue with advanced specifications."
            elif step == "showSummary":
                context_hint = "Now, let's proceed with your product analysis."
            else:
                context_hint = "Now, let's continue with your product selection."
            
            # Build and execute LLM chain
            response_chain = prompts.sales_agent_knowledge_question_prompt | components['llm'] | StrOutputParser()
            llm_response = response_chain.invoke({"user_message": user_message, "context_hint": context_hint})
            
            # Return response without changing workflow step - use the step sent by frontend
            return jsonify({
                "content": llm_response,
                "nextStep": step,  # Resume the exact step where user was interrupted
                "maintainWorkflow": True
            }), 200

        # === Step-Based Workflow Responses (Preserving Original Prompts) ===
        
        # --- Original Prompt Selection Based on Step ---
        if step == 'initialInput':
            product_type = data_context.get('productType', 'a product')
            # If frontend indicated this request should save immediately (e.g., initial full-spec submit),
            # bypass the greeting prompt and persist the detected product type in session.
            save_flag = False
            if isinstance(data, dict):
                # Accept both `saveImmediately` boolean and `action: 'save'` patterns
                save_flag = bool(data.get('saveImmediately')) or data.get('action') == 'save'
            if save_flag:
                session[f'product_type_{search_session_id}'] = product_type
                session[f'current_step_{search_session_id}'] = 'initialInput'
                session.modified = True
                # Return quick confirmation to frontend so it can proceed without receiving the greeting prompt
                return jsonify({
                    "content": f"Saved product type: {product_type}",
                    "nextStep": "awaitAdditionalAndLatestSpecs"
                }), 200
            
            
            current_prompt_template = prompts.sales_agent_initial_input_prompt
            next_step = "awaitAdditionalAndLatestSpecs"
            next_step = "awaitAdditionalAndLatestSpecs"
            
        elif step == 'awaitAdditionalAndLatestSpecs':
            # Handle the combined "Additional and Latest Specs" step
            user_lower = user_message.lower().strip()
            
            # Define yes/no keywords
            affirmative_keywords = ['yes', 'y', 'yeah', 'yep', 'sure', 'ok', 'okay']
            negative_keywords = ['no', 'n', 'nope', 'skip']
            
            # Check if we're waiting for yes/no or collecting specs input
            # Track state in session to know if we've asked the question
            asking_state_key = f'awaiting_additional_specs_yesno_{search_session_id}'
            is_awaiting_yesno = session.get(asking_state_key, True)
            
            # Check if user response is valid yes/no
            is_yes = any(keyword in user_lower for keyword in affirmative_keywords)
            is_no = any(keyword in user_lower for keyword in negative_keywords)
            
            if is_awaiting_yesno:
                # First interaction - asking yes/no question
                if is_no:
                    # User says NO -> skip directly to showSummary
                    session[asking_state_key] = False
                    # Use LLM to produce the fixed summary-intro sentence
                    current_prompt_template = prompts.sales_agent_no_additional_specs_prompt
                    next_step = "showSummary"
                    next_step = "showSummary"
                elif is_yes:
                    # User says YES -> ask them to enter their additional/latest specifications
                    session[asking_state_key] = False  # Now we're collecting input
                    # Use LLM to remind them of the latest advanced specifications, then ask for input
                    available_parameters = data_context.get('availableParameters', [])
                    if available_parameters:
                        params_display = format_available_parameters(available_parameters)
                        current_prompt_template = prompts.sales_agent_yes_additional_specs_prompt
                    # else:
                    #     prompt_template = "Great! Please enter your additional or latest specifications."

                    next_step = "awaitAdditionalAndLatestSpecs"  # Stay in step to collect input
                else:
                    # Invalid response - ask for yes/no
                    llm_response = "Please respond with yes or no. Additional and latest specifications are available. Would you like to add them?"
                    current_prompt_template = None
                    next_step = "awaitAdditionalAndLatestSpecs"  # Stay in step until valid answer
            else:
                # We're collecting the actual specifications input
                # User has provided their additional/latest specs - process and move to advanced parameters
                product_type = data_context.get('productType') or session.get(f'product_type_{search_session_id}')
                
                # Process the additional requirements using additional_requirements endpoint logic
                # Store the user input for processing - frontend will call /additional_requirements to extract and merge
                session[f'additional_specs_input_{search_session_id}'] = user_message
                session[asking_state_key] = True  # Reset for next time
                
                current_prompt_template = prompts.sales_agent_acknowledge_additional_specs_prompt
                next_step = "awaitAdvancedSpecs"
                next_step = "awaitAdvancedSpecs"
        
        elif step == 'awaitAdvancedSpecs':
            # Handle advanced parameters step
            user_lower = user_message.lower().strip()

            # Get context data (session-isolated)
            product_type = data_context.get('productType') or session.get(f'product_type_{search_session_id}')
            # NOTE: available_parameters is expected to be a list of strings or dicts
            available_parameters = data_context.get('availableParameters', [])
            selected_parameters = data_context.get('selectedParameters', {})
            total_selected = data_context.get('totalSelected', 0)
            
            # Define trigger keywords
            display_keywords = ['show', 'display', 'list', 'see', 'view', 'what are', 'remind']
            affirmative_keywords = ['yes', 'y', 'yeah', 'yep', 'sure', 'proceed', 'continue', 'ok', 'okay', 'go ahead']
            negative_keywords = ['no', 'n', 'skip', 'none', 'not needed', 'done', 'not interested']

            # Debug logging
            logging.info(f"awaitAdvancedSpecs - product_type: {product_type}")
            logging.info(f"awaitAdvancedSpecs - available_parameters count: {len(available_parameters)}")
            logging.info(f"awaitAdvancedSpecs - user_message: {user_message}")
            
            # Check if this is first time (no parameters discovered yet)
            if not available_parameters or len(available_parameters) == 0:
                
                # --- Handling retry/skip when discovery yielded 0 results or after an error ---
                parameter_error = data_context.get('parameterError', False)
                no_params_found = data_context.get('no_params_found', False)
                
                if parameter_error and user_lower in affirmative_keywords:
                    # User confirms they want to skip after an error
                    current_prompt_template = prompts.sales_agent_default_prompt
                    next_step = "showSummary"
                elif parameter_error and user_lower in negative_keywords:
                    # User wants to retry (or says 'no' to skipping) - fall through to discovery
                    prompt_template = "" # Clears the error state prompt
                    pass
                elif user_lower in affirmative_keywords or user_lower == "":
                    # User agreed to proceed with discovery (or sent empty message) - run discovery
                    prompt_template = "" # Clears any prior prompt
                    pass
                elif user_lower in negative_keywords:
                    # User says 'no' or 'skip' - they want to skip advanced parameters entirely
                    # Proceed directly to showSummary without discovering parameters
                    llm_response = "No problem! Proceeding without advanced specifications."
                    current_prompt_template = None
                    next_step = "showSummary"
                else:
                    # Default action for unexpected input when no params are known
                    llm_response = "I'm not sure how to proceed. Would you like me to try discovering the parameters, or shall we skip to the summary?"
                    current_prompt_template = None
                    next_step = "awaitAdvancedSpecs"
                    
                # --- Initial Discovery Block (Runs on first entry or retry) ---
                # Only run discovery if no specific prompt_template has been set above
                if not prompt_template.strip() and 'llm_response' not in locals():
                    logging.info(f"Attempting discovery for product_type: {product_type}")
                    try:
                        if product_type:
                            # Discover advanced parameters (works for both MongoDB cache and LLM discovery)
                            parameters_result = discover_advanced_parameters(product_type)
                            # Handle both 'unique_parameters' and 'unique_specifications' keys (MongoDB uses 'unique_specifications')
                            discovered_params = parameters_result.get('unique_parameters') or parameters_result.get('unique_specifications', [])
                            discovered_params = discovered_params[:15] if discovered_params else []
                            filtered_count = parameters_result.get('existing_parameters_filtered', 0) or parameters_result.get('existing_specifications_filtered', 0)

                            # Store discovered parameters in session for future use
                            data_context['availableParameters'] = discovered_params
                            # Track whether discovery returned zero parameters
                            data_context['no_params_found'] = len(discovered_params) == 0
                            session['data'] = data_context
                            session.modified = True

                            if len(discovered_params) == 0:
                                # CASE 2 — No advanced parameters found: ask if user wants to proceed to summary
                                filter_info = (
                                    f" All {filtered_count} potential advanced parameters were already covered in your mandatory/optional requirements."
                                    if filtered_count > 0
                                    else " No new advanced parameters were found for this product type."
                                )

                                # Direct deterministic response (no extra LLM prompting)
                                llm_response = (
                                    f"No advanced parameters were found.{filter_info}\n\n"
                                    "No advanced parameters were found. Do you want to proceed to summary?"
                                )
                                prompt_template = ""
                            else:
                                # CASE 1 — Advanced parameters found: show list and ask if user wants to add them
                                params_display = format_available_parameters(discovered_params)

                                # Direct deterministic response listing parameters
                                llm_response = (
                                    "These advanced parameters were identified:\n\n"
                                    f"{params_display}\n\n"
                                    "Do you want to add these advanced parameters?"
                                )
                                # Set prompt_template to empty so LLM is not called
                                prompt_template = ""
                                current_prompt_template = None
                        else:
                            # No product type found
                            data_context['parameterError'] = True
                            session['data'] = data_context
                            session.modified = True
                            llm_response = "I'm having trouble accessing advanced parameters because the product type isn't clear. Would you like to skip this step?"
                            current_prompt_template = None

                    except Exception as e:
                        # General error case
                        logging.error(f"Error during parameter discovery: {e}", exc_info=True)
                        data_context['parameterError'] = True
                        session['data'] = data_context
                        session.modified = True
                        prompt_template = "I encountered an issue discovering advanced parameters. Would you like to skip this step?"
                
                # Now interpret user reply when no available_parameters exist (only if we didn't run discovery or discovery yielded 0)
                if data_context.get('no_params_found', False) or parameter_error:
                    # CASE 2 — Follow-up after "No advanced parameters were found. Do you want to proceed to summary?"
                    if user_lower in affirmative_keywords:
                        # User said YES -> go directly to summary
                        llm_response = "Okay, I'll proceed to the summary without advanced parameters."
                        prompt_template = ""
                        next_step = "showSummary"
                    elif user_lower in negative_keywords:
                        # User said NO -> retry discovery (loop will run discovery again)
                        data_context['parameterError'] = False
                        data_context['no_params_found'] = False  # Force a retry
                        session['data'] = data_context
                        session.modified = True
                        llm_response = "No problem, I'll try discovering advanced parameters again."
                        prompt_template = ""
                        next_step = "awaitAdvancedSpecs"
                    elif parameter_error and not user_message.strip():
                        llm_response = "I encountered an issue discovering advanced parameters. Would you like to skip this step?"
                        current_prompt_template = None
                        next_step = "awaitAdvancedSpecs"
                    elif not user_message.strip():
                        # Empty message while in no_params_found state - repeat the question
                        llm_response = "No advanced parameters were found. Do you want to proceed to summary?"
                        prompt_template = ""
                        next_step = "awaitAdvancedSpecs"
                    else:
                        # If user gave something else, ask the clarifying question again
                        llm_response = "Please answer with yes or no. Do you want to proceed to summary without advanced parameters?"
                        prompt_template = ""
                        next_step = "awaitAdvancedSpecs"
                else:
                    # Default: stay in awaitAdvancedSpecs and attempt discovery based on affirmative/empty
                    next_step = "awaitAdvancedSpecs"
            else:
                # --- Parameters already discovered - handle user response ---
                parameter_error = data_context.get('parameterError', False)
                
                wants_display = any(keyword in user_lower for keyword in display_keywords)
                user_affirmed = any(keyword in user_lower for keyword in affirmative_keywords)
                user_denied = any(keyword in user_lower for keyword in negative_keywords)

                # CASE 1: User says 'yes' to adding parameters - show keys and ask them to provide values
                # Check this FIRST before CASE 2, so "yes" doesn't get caught by the display condition
                if user_affirmed:
                    # CASE 1 — User said YES: use LLM to ask for values, listing the parameters
                        current_prompt_template = prompts.sales_agent_advanced_specs_yes_prompt
                        next_step = "awaitAdvancedSpecs"  # Stay in step to collect values
                    
                # CASE 2: User says 'no' to adding parameters (normal flow)
                elif user_denied:
                    # User explicitly declined adding advanced parameters -> go directly to SUMMARY
                    # Use the same summary-intro sentence as in awaitAdditionalAndLatestSpecs
                    current_prompt_template = prompts.sales_agent_advanced_specs_no_prompt
                    next_step = "showSummary"
                    
                # CASE 3: Force the list to display if the user explicitly asks to see it, or empty message
                elif wants_display or (not user_message.strip() and not total_selected > 0):
                    params_display = format_available_parameters(available_parameters)

                    current_prompt_template = prompts.sales_agent_advanced_specs_display_prompt
                    next_step = "awaitAdvancedSpecs"
                    
                # CASE 4: User provided parameter selections/values
                elif total_selected > 0 or user_message.strip():
                    # User provided values for parameters
                    selected_names = [param.replace('_', ' ').title() for param in selected_parameters.keys()] if selected_parameters else []
                    if selected_names:
                        selected_display = ", ".join(selected_names)
                        # Use a direct response for this common feedback
                        llm_response = f"**Added Advanced Parameters:** {selected_display}\n\nProceeding to the summary now."
                        current_prompt_template = None
                    else:
                        llm_response = "Thank you for providing the advanced specifications. Proceeding to the summary now."
                        current_prompt_template = None
                    next_step = "showSummary"
                    
                # CASE 5: No parameters matched or user provided other input (Default fallback)
                else:
                    llm_response = "Please respond with yes or no. These additional advanced parameters were identified. Would you like to add them?"
                    current_prompt_template = None
                    next_step = "awaitAdvancedSpecs"
            
        elif step == 'confirmAfterMissingInfo':
            # Discover advanced parameters to show in the response
            product_type = data_context.get('productType') or session.get(f'product_type_{search_session_id}')
            
            # Initial prompt is set as a fallback
            current_prompt_template = prompts.sales_agent_confirm_after_missing_info_prompt
            
            if product_type:
                try:
                    # Discover advanced parameters
                    parameters_result = discover_advanced_parameters(product_type)
                    discovered_params = parameters_result.get('unique_parameters', []) or parameters_result.get('unique_specifications', [])
                    discovered_params = discovered_params[:15] if discovered_params else []
                    
                    if len(discovered_params) > 0:
                        # Format parameters for display
                        params_display = format_available_parameters(discovered_params)
                        
                        # Store discovered parameters in session for later use
                        data_context['availableParameters'] = discovered_params
                        session['data'] = data_context
                        session.modified = True

                        # Use LLM with a strict prompt so it returns the exact desired message
                        current_prompt_template = prompts.sales_agent_confirm_after_missing_info_with_params_prompt
                    # Else: prompt_template remains the one set before the try block
                except Exception as e:
                    logging.error(f"Error discovering parameters in confirmAfterMissingInfo: {e}", exc_info=True)
                    # Fallback prompt is already set
            
            next_step = "awaitAdditionalAndLatestSpecs"
            
        elif step == 'showSummary':
            # Check if user is confirming to proceed with analysis
            user_lower = user_message.lower().strip()
            if user_lower in ['yes', 'y', 'proceed', 'continue', 'run', 'analyze', 'ok', 'okay']:
                current_prompt_template = prompts.sales_agent_show_summary_proceed_prompt
                next_step = "finalAnalysis"
            else:
                # First time showing summary - trigger summary generation
                # The frontend will call handleShowSummaryAndProceed which generates the summary
                current_prompt_template = prompts.sales_agent_show_summary_intro_prompt
                next_step = "showSummary"  # Stay in showSummary to trigger summary display
            
        elif step == 'finalAnalysis':
            ranked_products = data_context.get('analysisResult', {}).get('overallRanking', {}).get('rankedProducts', [])
            # NOTE: Logic to determine 'matching_products' based on 'requirementsMatch' is in the original code.
            # Assuming 'requirementsMatch' is a boolean key in each product dict.
            matching_products = [p for p in ranked_products if p.get('requirementsMatch') is True] 
            count = len(matching_products)
            current_prompt_template = prompts.sales_agent_final_analysis_prompt
            next_step = None  # End of workflow
            
        elif step == 'analysisError':
            current_prompt_template = prompts.sales_agent_analysis_error_prompt
            next_step = "showSummary"  # Allow retry from summary
            
        elif step == 'default':
            current_prompt_template = prompts.sales_agent_default_prompt
            next_step = current_step or None
            
        # === NEW WORKFLOW STEPS (Added for enhanced functionality) ===
        elif step == 'greeting':
            current_prompt_template = prompts.sales_agent_greeting_prompt
            next_step = "initialInput"
            
        else:
            # Default fallback for unrecognized steps
            current_prompt_template = prompts.sales_agent_default_prompt
            next_step = current_step or "greeting"

        # --- Build Chain and Generate Response ---
        # Initialize llm_response if not already set (for direct responses without LLM)
        if 'llm_response' not in locals():
            llm_response = ""
            
        if current_prompt_template:
            if isinstance(current_prompt_template, str):
                full_prompt = ChatPromptTemplate.from_template(current_prompt_template)
            else:
                full_prompt = current_prompt_template
                
            response_chain = full_prompt | components['llm'] | StrOutputParser()
            llm_response = response_chain.invoke({
                "user_input": user_message, 
                "product_type": data_context.get('productType'), 
                "count": count if 'count' in locals() else 0, 
                "params_display": params_display if 'params_display' in locals() else 'No parameters.', 
                "search_session_id": search_session_id
            })

        # Update session with new step (session-isolated)
        if next_step:
            session[f'current_step_{search_session_id}'] = next_step
            session[f'current_intent_{search_session_id}'] = 'workflow'

        # Prepare response
        response_data = {
            "content": llm_response,
            "nextStep": next_step
        }

        # Store the sales agent response as system response for logging

        return jsonify(response_data), 200

    except Exception as e:
        logging.exception("Sales agent response generation failed.")
        # Retrieve current step for fallback, defaulting to 'initialInput' if not found
        fallback_step = session.get(f'current_step_{search_session_id}', 'initialInput')
        return jsonify({
            "error": "Failed to generate response: " + str(e),
            "content": "I apologize, but I'm having technical difficulties. Please try again.",
            "nextStep": fallback_step
        }), 500


# =========================================================================
# === NEW FEEDBACK ENDPOINT ===
# =========================================================================
@app.route("/api/feedback", methods=["POST"])
@login_required
def handle_feedback():
    """
    Handles user feedback and saves a complete log entry to the database.
    """
    if not components or not components.get('llm'):
        return jsonify({"error": "LLM component is not ready."}), 503

    try:
        data = request.get_json(force=True)
        feedback_type = data.get("feedbackType")
        comment = data.get("comment", "")

        # --- DATABASE LOGGING LOGIC STARTS HERE ---
        
        # 1. Retrieve the stored data from the session
        user_query = session.get('log_user_query', 'No query found - user may have provided feedback without validation')
        system_response = session.get('log_system_response', {})

        # 2. Format the feedback for the database
        feedback_log_entry = feedback_type
        if comment:
            feedback_log_entry += f" ({comment})"

        # 3. Get the current user's information to log their username
        current_user = db.session.get(User, session['user_id'])
        if not current_user:
            logging.error(f"Could not find user with ID {session['user_id']} to create log entry.")
            return jsonify({"error": "Authenticated user not found for logging."}), 404
        
        username = current_user.username

        # 4. Persist feedback to MongoDB only (do not store in SQL)
        try:
            project_id_for_feedback = session.get('current_project_id') or data.get('projectId')

            feedback_entry = {
                'timestamp': datetime.utcnow(),
                'user_id': str(session.get('user_id')) if session.get('user_id') else None,
                'user_name': username,
                'feedback_type': feedback_type,
                'comment': comment,
                'user_query': user_query,
                'system_response': system_response
            }

            from mongo_project_manager import mongo_project_manager
            # If we have a project id, append to that project's feedback_entries array
            if project_id_for_feedback:
                try:
                    mongo_project_manager.append_feedback_to_project(project_id_for_feedback, str(session.get('user_id')), feedback_entry)
                    logging.info(f"Appended feedback to MongoDB project {project_id_for_feedback}")
                except Exception as me:
                    logging.warning(f"Failed to append feedback to MongoDB project {project_id_for_feedback}: {me}")
                    # On failure to append to project, fall back to inserting into a top-level feedback collection
                    try:
                        mongo_project_manager.file_manager.db['feedback'].insert_one({**feedback_entry, 'project_id': project_id_for_feedback})
                    except Exception as e:
                        logging.error(f"Failed to save feedback to feedback collection: {e}")
            else:
                # No project id: save feedback as standalone document for later linking
                try:
                    mongo_project_manager.file_manager.db['feedback'].insert_one({**feedback_entry, 'project_id': None})
                    logging.info("Saved feedback to MongoDB 'feedback' collection (no project id)")
                except Exception as e:
                    logging.error(f"Failed to save feedback to MongoDB feedback collection: {e}")

        except Exception as e:
            logging.exception(f"Failed to persist feedback to MongoDB: {e}")

        # Clean up the session logging keys
        session.pop('log_user_query', None)
        session.pop('log_system_response', None)
        
        # --- LOGGING LOGIC ENDS ---

        if not feedback_type and not comment:
            return jsonify({"error": "No feedback provided."}), 400
        
        # --- LLM RESPONSE GENERATION ---
        if feedback_type == 'positive':
            feedback_chain = prompts.feedback_positive_prompt | components['llm'] | StrOutputParser()
        elif feedback_type == 'negative':
            feedback_chain = prompts.feedback_negative_prompt | components['llm'] | StrOutputParser()
        else:  # This handles the case where only a comment is provided
            feedback_chain = prompts.feedback_comment_prompt | components['llm'] | StrOutputParser()

        llm_response = feedback_chain.invoke({"comment": comment})

        return jsonify({"response": llm_response}), 200

    except Exception as e:
        logging.exception("Feedback handling or MongoDB storage failed.")
        return jsonify({"error": "Failed to process feedback: " + str(e)}), 500

# =========================================================================
# === INSTRUMENT IDENTIFICATION ENDPOINT ===
# =========================================================================
@app.route("/api/identify-instruments", methods=["POST"])
@login_required
def identify_instruments():
    """
    Handles user input in project page with three cases:
    1. Greeting - Returns friendly greeting response
    2. Requirements - Returns identified instruments and accessories
    3. Industrial Question - Returns answer or redirect if not related
    """
    if not components or not components.get('llm_pro'):
        return jsonify({"error": "LLM component is not ready."}), 503

    try:
        data = request.get_json(force=True)
        requirements = data.get("requirements", "").strip()
        search_session_id = data.get("search_session_id", "default")
        
        if not requirements:
            return jsonify({"error": "Requirements text is required"}), 400

        # Pre-classification heuristics to catch obvious cases
        requirements_lower = requirements.lower()
        
        # Strong indicators of unrelated content (emails, job offers, etc.)
        unrelated_indicators = [
            'from:', 'to:', 'subject:', 'date:',  # Email headers
            'congratulations for the selection', 'job offer', 'recruitment',
            'campus placement', 'hr department', 'hiring', 'interview process',
            'provisionally selected', 'offer letter', 'employment application',
            'dear sir', 'dear madam', 'forwarded message',
            'training and placement officer', 'campus recruitment'
        ]
        
        # Check if content has strong unrelated indicators
        unrelated_count = sum(1 for indicator in unrelated_indicators if indicator in requirements_lower)
        
        # If 2+ strong indicators, skip LLM and classify as unrelated immediately
        if unrelated_count >= 2:
            logging.info(f"[CLASSIFY] Pre-classification: UNRELATED (found {unrelated_count} indicators)")
            input_type = "unrelated"
            confidence = "high"
            reasoning = f"Contains {unrelated_count} strong indicators of non-industrial content (email headers, job/recruitment terms)"
        else:
            # Step 1: Classify the input type using LLM
            classification_chain = prompts.identify_classification_prompt | components['llm'] | StrOutputParser()
            classification_response = classification_chain.invoke({"user_input": requirements})
            
            # Clean and parse classification
            cleaned_classification = classification_response.strip()
            if cleaned_classification.startswith("```json"):
                cleaned_classification = cleaned_classification[7:]
            elif cleaned_classification.startswith("```"):
                cleaned_classification = cleaned_classification[3:]
            if cleaned_classification.endswith("```"):
                cleaned_classification = cleaned_classification[:-3]
            cleaned_classification = cleaned_classification.strip()
            
            try:
                classification = json.loads(cleaned_classification)
                input_type = classification.get("type", "requirements").lower()
                confidence = classification.get("confidence", "medium").lower()
                reasoning = classification.get("reasoning", "")
                
                # Log classification for debugging
                logging.info(f"[CLASSIFY] LLM classified as '{input_type}' (confidence: {confidence})")
                logging.info(f"[CLASSIFY] Reasoning: {reasoning}")
                
            except Exception as e:
                # Default to requirements if classification fails
                input_type = "requirements"
                confidence = "low"
                reasoning = "Classification parsing failed"
                logging.warning(f"Failed to parse classification, defaulting to requirements. Response: {classification_response}")
                logging.exception(e)

        # CASE 1: Greeting
        if input_type == "greeting":
            greeting_chain = prompts.identify_greeting_prompt | components['llm'] | StrOutputParser()
            greeting_response = greeting_chain.invoke({"user_input": requirements})
            
            from testing_utils import standardized_jsonify
            return standardized_jsonify({
                "response_type": "greeting",
                "message": greeting_response.strip(),
                "instruments": [],
                "accessories": []
            }, 200)

        # CASE 2: Requirements - Identify instruments and accessories
        elif input_type == "requirements":
            session_isolated_requirements = f"[Session: {search_session_id}] - This is an independent instrument identification request. Requirements: {requirements}"
            
            response_chain = prompts.identify_instrument_prompt | components['llm'] | StrOutputParser()
            llm_response = response_chain.invoke({"requirements": session_isolated_requirements})

            # Clean the LLM response
            cleaned_response = llm_response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]  
            elif cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]  
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]  
            cleaned_response = cleaned_response.strip()

            try:
                result = json.loads(cleaned_response)
                
                # Validate the response structure
                if "instruments" not in result or not isinstance(result["instruments"], list):
                    raise ValueError("Invalid response structure from LLM")
                
                # Ensure all instruments have required fields
                for instrument in result["instruments"]:
                    if not all(key in instrument for key in ["category", "product_name", "specifications", "sample_input"]):
                        raise ValueError("Missing required fields in instrument data")
                
                # Validate accessories if present
                if "accessories" in result:
                    if not isinstance(result["accessories"], list):
                        raise ValueError("'accessories' must be a list if provided")
                    for accessory in result["accessories"]:
                        expected_acc_keys = ["category", "accessory_name", "specifications", "sample_input"]
                        if not all(key in accessory for key in expected_acc_keys):
                            raise ValueError("Missing required fields in accessory data")
                
                # Ensure strategy field exists for all instruments and accessories
                for instrument in result.get("instruments", []):
                    if "strategy" not in instrument:
                        instrument["strategy"] = ""

                for accessory in result.get("accessories", []):
                    if "strategy" not in accessory:
                        accessory["strategy"] = ""
                
                # Add response type
                result["response_type"] = "requirements"
                return standardized_jsonify(result, 200)
                
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse LLM response as JSON: {e}")
                logging.error(f"LLM Response: {llm_response}")
                
                return jsonify({
                    "response_type": "error",
                    "error": "Failed to parse instrument identification",
                    "instruments": [],
                    "accessories": [],
                    "summary": "Unable to identify instruments from the provided requirements"
                }), 500

        # CASE 3: Unrelated content - Politely redirect
        elif input_type == "unrelated":
            unrelated_chain = prompts.identify_unrelated_prompt | components['llm'] | StrOutputParser()
            unrelated_response = unrelated_chain.invoke({"reasoning": reasoning})
            
            from testing_utils import standardized_jsonify
            return standardized_jsonify({
                "response_type": "question",  # Use "question" type for frontend compatibility
                "is_industrial": False,
                "message": unrelated_response.strip(),
                "instruments": [],
                "accessories": []
            }, 200)

        # CASE 4: Question - Answer industrial questions or redirect
        elif input_type == "question":
            question_chain = prompts.identify_question_prompt | components['llm'] | StrOutputParser()
            question_response = question_chain.invoke({"user_input": requirements})
            
            # Clean and parse question response
            cleaned_question = question_response.strip()
            if cleaned_question.startswith("```json"):
                cleaned_question = cleaned_question[7:]
            elif cleaned_question.startswith("```"):
                cleaned_question = cleaned_question[3:]
            if cleaned_question.endswith("```"):
                cleaned_question = cleaned_question[:-3]
            cleaned_question = cleaned_question.strip()
            
            try:
                question_result = json.loads(cleaned_question)
                return standardized_jsonify({
                    "response_type": "question",
                    "is_industrial": question_result.get("is_industrial", True),
                    "message": question_result.get("answer", ""),
                    "instruments": [],
                    "accessories": []
                }, 200)
            except:
                # Fallback if parsing fails - generate response from LLM without JSON constraint
                fallback_chain = prompts.identify_fallback_prompt | components['llm'] | StrOutputParser()
                fallback_message = fallback_chain.invoke({"requirements": requirements})
                
                return standardized_jsonify({
                    "response_type": "question",
                    "is_industrial": False,
                    "message": fallback_message.strip(),
                    "instruments": [],
                    "accessories": []
                }, 200)
        
        # CASE 5: Unexpected classification type - Default fallback
        else:
            logging.warning(f"[CLASSIFY] Unexpected classification type: {input_type}")
            # Generate dynamic response from LLM
            unexpected_chain = prompts.identify_unexpected_prompt | components['llm'] | StrOutputParser()
            unexpected_message = unexpected_chain.invoke({"user_input": requirements})
            
            return standardized_jsonify({
                "response_type": "question",
                "is_industrial": False,
                "message": unexpected_message.strip(),
                "instruments": [],
                "accessories": []
            }, 200)

    except Exception as e:
        logging.exception("Instrument identification failed.")
        return jsonify({
            "response_type": "error",
            "error": "Failed to process request: " + str(e),
            "instruments": [],
            "accessories": [],
            "summary": ""
        }), 500

@app.route("/api/search-vendors", methods=["POST"])
@login_required
def search_vendors():
    """
    Search for vendors
    ---
    tags:
      - Vendors
    summary: Search vendors by product criteria
    description: |
      Search for vendors based on selected instrument/accessory details.
      Maps category, product name, and strategy to CSV data and returns filtered vendor list.
      
      Uses fuzzy matching to find the best matching vendors based on:
      - Product category
      - Product/accessory name
      - Procurement strategy
    consumes:
      - application/json
    produces:
      - application/json
    parameters:
      - in: body
        name: body
        description: Search criteria
        required: true
        schema:
          type: object
          required:
            - category
            - product_name
          properties:
            category:
              type: string
              description: Product category
              example: "Pressure Transmitters"
            product_name:
              type: string
              description: Product or instrument name
              example: "Rosemount 3051"
            accessory_name:
              type: string
              description: Alternative to product_name for accessories
            strategy:
              type: string
              description: Procurement strategy
              example: "Critical"
    responses:
      200:
        description: List of matching vendors
        schema:
          type: object
          properties:
            vendors:
              type: array
              items:
                type: object
                properties:
                  name:
                    type: string
                  category:
                    type: string
                  strategy:
                    type: string
            total_count:
              type: integer
            matching_criteria:
              type: object
      400:
        description: Missing required fields
      401:
        description: Unauthorized - login required
      500:
        description: Vendor data not available
    """
    try:
        data = request.get_json(force=True)
        category = data.get("category", "").strip()
        product_name = data.get("product_name", "").strip() or data.get("accessory_name", "").strip()
        strategy = data.get("strategy", "").strip()
        
        print(f"[VENDOR_SEARCH] Received request: category='{category}', product='{product_name}', strategy='{strategy}'")
        
        if not category or not product_name:
            return jsonify({"error": "Category and product_name/accessory_name are required"}), 400
        
        # Load CSV data
        csv_path = os.path.join(os.path.dirname(__file__), 'instrumentation_procurement_strategy.csv')
        
        if not os.path.exists(csv_path):
            return jsonify({"error": "Vendor data not available"}), 500
        
        vendors = []
        matched_vendors = []
        
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            vendors = list(reader)
        
        if not vendors:
            return jsonify({"vendors": [], "total_count": 0, "matching_criteria": {}}), 200
        
        # Get unique categories and subcategories for fuzzy matching
        csv_categories = list(set([v.get('category', '').strip() for v in vendors if v.get('category')]))
        csv_subcategories = list(set([v.get('subcategory', '').strip() for v in vendors if v.get('subcategory')]))
        csv_strategies = list(set([v.get('strategy', '').strip() for v in vendors if v.get('strategy')]))
        
        # Step 1: Match Category using dynamic fuzzy matching
        matched_category = None
        category_match_type = None
        
        # Exact match first
        for csv_cat in csv_categories:
            if csv_cat.lower() == category.lower():
                matched_category = csv_cat
                category_match_type = "exact"
                break
        
        # Fuzzy match if no exact match
        if not matched_category:
            fuzzy_result = process.extractOne(category, csv_categories, scorer=fuzz.ratio)
            if fuzzy_result and fuzzy_result[1] >= 50:  # Flexible threshold for better matching
                matched_category = fuzzy_result[0]
                category_match_type = "fuzzy"
        
        print(f"[VENDOR_SEARCH] Category matching result: '{category}' -> '{matched_category}' (type: {category_match_type})")
        
        if not matched_category:
            print(f"[VENDOR_SEARCH] No category match found. Available categories: {csv_categories}")
            return jsonify({
                "vendors": [],
                "total_count": 0,
                "matching_criteria": {
                    "category_match": None,
                    "subcategory_match": None,
                    "strategy_match": None,
                    "message": f"No matching category found for '{category}'. Available: {csv_categories}"
                }
            }), 200
        
        # Step 2: Match Product Name to Subcategory (exact first, then fuzzy)
        matched_subcategory = None
        subcategory_match_type = None
        
        # Exact match
        for csv_subcat in csv_subcategories:
            if csv_subcat.lower() == product_name.lower():
                matched_subcategory = csv_subcat
                subcategory_match_type = "exact"
                break
        
        # Fuzzy match if no exact match
        if not matched_subcategory:
            fuzzy_result = process.extractOne(product_name, csv_subcategories, scorer=fuzz.ratio)
            if fuzzy_result and fuzzy_result[1] >= 70:  # 70% similarity threshold
                matched_subcategory = fuzzy_result[0]
                subcategory_match_type = "fuzzy"
        
        # Step 3: Match Strategy (optional field, exact first, then fuzzy)
        matched_strategy = None
        strategy_match_type = None
        
        if strategy:  # Only if strategy is provided
            # Exact match
            for csv_strategy in csv_strategies:
                if csv_strategy.lower() == strategy.lower():
                    matched_strategy = csv_strategy
                    strategy_match_type = "exact"
                    break
            
            # Fuzzy match if no exact match
            if not matched_strategy:
                fuzzy_result = process.extractOne(strategy, csv_strategies, scorer=fuzz.ratio)
                if fuzzy_result and fuzzy_result[1] >= 70:  # 70% similarity threshold
                    matched_strategy = fuzzy_result[0]
                    strategy_match_type = "fuzzy"
        
        # Step 4: Filter vendors based on matches
        filtered_vendors = []
        
        for vendor in vendors:
            vendor_category = vendor.get('category', '').strip()
            vendor_subcategory = vendor.get('subcategory', '').strip()
            vendor_strategy = vendor.get('strategy', '').strip()
            
            # Category must match
            if vendor_category != matched_category:
                continue
            
            # Subcategory should match if we found a match
            if matched_subcategory and vendor_subcategory != matched_subcategory:
                continue
            
            # Strategy should match if provided and we found a match
            if strategy and matched_strategy and vendor_strategy != matched_strategy:
                continue
            
            # Add vendor to results
            filtered_vendors.append({
                "vendor_id": vendor.get('vendor ID', ''),
                "vendor_name": vendor.get('vendor name', ''),
                "category": vendor_category,
                "subcategory": vendor_subcategory,
                "strategy": vendor_strategy,
                "refinery": vendor.get('refinery', ''),
                "additional_comments": vendor.get('additional comments', ''),
                "owner_name": vendor.get('owner name', '')
            })
        
        # Prepare response
        matching_criteria = {
            "category_match": {
                "input": category,
                "matched": matched_category,
                "match_type": category_match_type
            },
            "subcategory_match": {
                "input": product_name,
                "matched": matched_subcategory,
                "match_type": subcategory_match_type
            } if matched_subcategory else None,
            "strategy_match": {
                "input": strategy,
                "matched": matched_strategy,
                "match_type": strategy_match_type
            } if strategy and matched_strategy else None
        }
        
        # Store CSV vendor filter in session for analysis chain
        if filtered_vendors:
            session['csv_vendor_filter'] = {
                'vendor_names': [v.get('vendor_name', '').strip() for v in filtered_vendors],
                'csv_data': filtered_vendors,
                'product_type': category,
                'detected_product': product_name,
                'matching_criteria': matching_criteria
            }
            print(f"[VENDOR_SEARCH] Stored {len(filtered_vendors)} vendors in session for analysis filtering")
        else:
            # Clear any existing filter if no vendors found
            session.pop('csv_vendor_filter', None)
            print("[VENDOR_SEARCH] No vendors found, cleared session filter")
        
        # Return only vendor names list for frontend
        vendor_names_only = [v.get('vendor_name', '').strip() for v in filtered_vendors]
        
        return standardized_jsonify({
            "vendors": vendor_names_only,
            "total_count": len(filtered_vendors),
            "matching_criteria": matching_criteria
        }, 200)
        
    except Exception as e:
        logging.exception("Vendor search failed.")
        return jsonify({
            "error": "Failed to search vendors: " + str(e),
            "vendors": [],
            "total_count": 0
        }), 500

# =========================================================================
# === API ENDPOINTS ===
# =========================================================================
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
GOOGLE_API_KEY1 = os.getenv("GOOGLE_API_KEY1")
GOOGLE_CSE_ID = "066b7345f94f64897"

# Image search API configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY1")  # Use GOOGLE_API_KEY1 for custom search
GOOGLE_CX = os.getenv("GOOGLE_CX", GOOGLE_CSE_ID)  # Custom search engine ID
SERPER_API_KEY_IMAGES = os.getenv("SERPER_API_KEY", SERPER_API_KEY)  # Reuse existing key

if not SERPER_API_KEY:
    logging.warning("SERPER_API_KEY environment variable not set! Will fallback to SERP API.")

if not SERPAPI_KEY:
    logging.warning("SERPAPI_KEY environment variable not set! Will use Google Custom Search as fallback.")

if not GOOGLE_API_KEY1:
    logging.warning("GOOGLE_API_KEY1 environment variable not set! Custom search fallback not available.")

if not GOOGLE_API_KEY:
    logging.warning("GOOGLE_API_KEY environment variable not set! Image search functionality limited.")

if not GOOGLE_CX:
    logging.warning("GOOGLE_CX environment variable not set! Custom image search not available.")


def serper_search_pdfs(query):
    """Perform a PDF search with Serper API."""
    if not SERPER_API_KEY:
        return []
    
    try:
        url = "https://google.serper.dev/search"
        payload = {
            "q": f"{query} filetype:pdf",
            "num": 10
        }
        headers = {
            "X-API-KEY": SERPER_API_KEY,
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        results = data.get("organic", [])
        
        pdf_results = []
        for item in results:
            pdf_results.append({
                "title": item.get("title"),
                "url": item.get("link"),
                "snippet": item.get("snippet", ""),
                "source": "serper"
            })
        return pdf_results
    except Exception as e:
        print(f"[WARNING] Serper API search failed: {e}")
        return []


def serpapi_search_pdfs(query):
    """Perform a PDF search with SerpApi."""
    if not SERPAPI_KEY:
        return []
    
    try:
        search = GoogleSearch({
            "q": f"{query} filetype:pdf",
            "api_key": SERPAPI_KEY,
            "num": 10
        })
        results = search.get_dict()
        items = results.get("organic_results", [])
        pdf_results = []
        for item in items:
            pdf_results.append({
                "title": item.get("title"),
                "url": item.get("link"),
                "snippet": item.get("snippet", ""),
                "source": "serpapi"
            })
        return pdf_results
    except Exception as e:
        print(f"[WARNING] SerpAPI search failed: {e}")
        return []


def google_custom_search_pdfs(query):
    """Perform a PDF search with Google Custom Search API as fallback."""
    if not GOOGLE_API_KEY1 or not GOOGLE_CSE_ID:
        return []
    
    try:
        import threading
        from googleapiclient.discovery import build
        
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY1)
        
        result_container = [None]
        exception_container = [None]
        
        def google_request():
            try:
                result = service.cse().list(
                    q=f"{query} filetype:pdf",
                    cx=GOOGLE_CSE_ID,
                    num=10,
                    fileType='pdf'
                ).execute()
                result_container[0] = result.get('items', [])
            except Exception as e:
                exception_container[0] = e
        
        thread = threading.Thread(target=google_request)
        thread.daemon = True
        thread.start()
        thread.join(timeout=60)
        
        if thread.is_alive() or exception_container[0]:
            return []
        
        items = result_container[0] or []
        
        pdf_results = []
        for item in items:
            title = item.get('title', '')
            link = item.get('link', '')
            snippet = item.get('snippet', '')
            
            if link:
                pdf_results.append({
                    "title": title.strip(),
                    "url": link,
                    "snippet": snippet,
                    "source": "google_custom"
                })
        
        return pdf_results
        
    except Exception as e:
        print(f"[WARNING] Google Custom Search failed: {e}")
        return []


def search_pdfs_with_fallback(query):
    """
    Search for PDFs using Serper API first, then SERP API, then Google Custom Search as final fallback.
    Returns combined results with source information.
    """
    # First, try Serper API
    serper_results = serper_search_pdfs(query)
    
    # If Serper API returns results, use them
    if serper_results:
        return serper_results
    
    # If Serper API fails or returns no results, try SERP API
    serpapi_results = serpapi_search_pdfs(query)
    
    # If SERP API returns results, use them
    if serpapi_results:
        return serpapi_results
    
    # If both Serper and SERP API fail or return no results, try Google Custom Search
    google_results = google_custom_search_pdfs(query)
    
    if google_results:
        return google_results
    
    # If all three fail, return empty list
    return []

@app.route("/api/search_pdfs", methods=["GET"])
@login_required
def search_pdfs():
    """
    Performs a PDF search using Serper API first, then SERP API, then Google Custom Search as final fallback.
    """
    query = request.args.get("query")
    if not query:
        return jsonify({"error": "Missing search query"}), 400

    try:
        # Use the new fallback search function
        results = search_pdfs_with_fallback(query)
        
        # Add metadata about which search engine was used
        response_data = {
            "results": results,
            "total_results": len(results),
            "query": query
        }
        
        # Add source information if results exist
        if results:
            sources_used = list(set(result.get("source", "unknown") for result in results))
            response_data["sources_used"] = sources_used
            
            # Add fallback indicator based on the new three-tier system
            if "serper" in sources_used:
                response_data["fallback_used"] = False
                response_data["message"] = "Results from Serper API"
            elif "serpapi" in sources_used:
                response_data["fallback_used"] = True
                response_data["message"] = "Results from SERP API (Serper fallback)"
            elif "google_custom" in sources_used:
                response_data["fallback_used"] = True
                response_data["message"] = "Results from Google Custom Search (final fallback)"
            else:
                response_data["fallback_used"] = False
        else:
            response_data["sources_used"] = []
            response_data["fallback_used"] = True
            response_data["message"] = "No results found from any search engine"
        
        return jsonify(response_data), 200

    except Exception as e:
        logging.exception("PDF search failed.")
        return jsonify({"error": "Failed to perform PDF search: " + str(e)}), 500


@app.route("/api/view_pdf", methods=["GET"])
@login_required
def view_pdf():
    """
    Fetches a PDF from a URL and serves it for viewing.
    """
    pdf_url = request.args.get("url")
    if not pdf_url:
        return jsonify({"error": "Missing PDF URL"}), 400
    
    try:
        response = requests.get(pdf_url, stream=True)
        response.raise_for_status()
        
        pdf_stream = BytesIO(response.content)
        
        return send_file(
            pdf_stream,
            mimetype='application/pdf',
            as_attachment=False,
            download_name=os.path.basename(pdf_url)
        )
    except requests.exceptions.RequestException as e:
        return jsonify({"error": "Failed to fetch PDF from URL: " + str(e)}), 500
    except Exception as e:
        return jsonify({"error": "An error occurred while viewing the PDF: " + str(e)}), 500


def fetch_price_and_reviews_serpapi(product_name: str):
    """Use SerpApi to fetch price and review info for a product."""
    if not SERPAPI_KEY:
        return []
    
    try:
        search = GoogleSearch({
            "q": f"{product_name} price review",
            "api_key": SERPAPI_KEY,
            "num": 10
        })
        res = search.get_dict()
        results = []

        for item in res.get("organic_results", []):
            snippet = item.get("snippet", "")
            price = None
            reviews = None
            source = item.get("source")
            link = item.get("link")

            # Try to pull price from structured extensions
            ext = (
                item.get("rich_snippet", {})
                    .get("bottom", {})
                    .get("detected_extensions", {})
            )
            if "price" in ext:
                price = f"${ext['price']}"
            elif "price_from" in ext and "price_to" in ext:
                price = f"${ext['price_from']} to ${ext['price_to']}"
            else:
                # Fallback: regex on snippet
                price_match = re.search(r"([$₹€£¥])\s?\d+(?:[.,]\d+)?", snippet)
                if price_match:
                    price = price_match.group(0)

            # Extract reviews (look in snippet)
            review_match = re.search(r"(\d(?:\.\d)?)\s?out of 5", snippet)
            if review_match:
                reviews = float(review_match.group(1))

            if price or reviews or source or link:
                results.append({
                    "price": price,
                    "reviews": reviews,
                    "source": source,
                    "link": link
                })

        return results

    except Exception as e:
        print(f"[WARNING] SerpAPI price/review search failed for {product_name}: {e}")
        return []


def fetch_price_and_reviews_serper(product_name: str):
    """Use Serper API to fetch price and review info for a product."""
    if not SERPER_API_KEY:
        return []
    
    try:
        url = "https://google.serper.dev/search"
        payload = {
            "q": f"{product_name} price review",
            "num": 10
        }
        headers = {
            "X-API-KEY": SERPER_API_KEY,
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        results = []

        for item in data.get("organic", []):
            snippet = item.get("snippet", "")
            price = None
            reviews = None
            source = item.get("displayLink")
            link = item.get("link")

            # Extract price from snippet using regex
            price_match = re.search(r"([$₹€£¥])\s?\d+(?:[.,]\d+)?", snippet)
            if price_match:
                price = price_match.group(0)

            # Extract reviews (look in snippet)
            review_match = re.search(r"(\d(?:\.\d)?)\s?out of 5", snippet)
            if review_match:
                reviews = float(review_match.group(1))

            if price or reviews or source or link:
                results.append({
                    "price": price,
                    "reviews": reviews,
                    "source": source,
                    "link": link
                })

        return results

    except Exception as e:
        print(f"[WARNING] Serper API price/review search failed for {product_name}: {e}")
        return []


def fetch_price_and_reviews_google_custom(product_name: str):
    """Use Google Custom Search to fetch price and review info for a product as fallback."""
    if not GOOGLE_API_KEY1 or not GOOGLE_CSE_ID:
        return []
    
    try:
        import threading
        from googleapiclient.discovery import build
        
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY1)
        
        result_container = [None]
        exception_container = [None]
        
        def google_request():
            try:
                result = service.cse().list(
                    q=f"{product_name} price review",
                    cx=GOOGLE_CSE_ID,
                    num=10
                ).execute()
                result_container[0] = result.get('items', [])
            except Exception as e:
                exception_container[0] = e
        
        thread = threading.Thread(target=google_request)
        thread.daemon = True
        thread.start()
        thread.join(timeout=60)
        
        if thread.is_alive() or exception_container[0]:
            return []
        
        items = result_container[0] or []
        results = []

        for item in items:
            snippet = item.get("snippet", "")
            price = None
            reviews = None
            source = item.get("displayLink")
            link = item.get("link")

            # Extract price using regex
            price_match = re.search(r"([$₹€£¥])\s?\d+(?:[.,]\d+)?", snippet)
            if price_match:
                price = price_match.group(0)

            # Extract reviews
            review_match = re.search(r"(\d(?:\.\d)?)\s?out of 5", snippet)
            if review_match:
                reviews = float(review_match.group(1))

            if price or reviews or source or link:
                results.append({
                    "price": price,
                    "reviews": reviews,
                    "source": source,
                    "link": link
                })

        return results

    except Exception as e:
        print(f"[WARNING] Google Custom Search price/review search failed for {product_name}: {e}")
        return []


def fetch_price_and_reviews(product_name: str):
    """
    Fetch price and review info using SERP API first, then Serper API, then Google Custom Search as final fallback.
    Special order for pricing: SERP API → Serper → Google Custom Search
    Returns a structured response with results and metadata.
    """
    # First, try SERP API (special order for pricing)
    serpapi_results = fetch_price_and_reviews_serpapi(product_name)
    
    # If SERP API returns results, use them
    if serpapi_results:
        return {
            "productName": product_name, 
            "results": serpapi_results,
            "source_used": "serpapi",
            "fallback_used": False
        }
    
    # If SERP API fails or returns no results, try Serper API
    serper_results = fetch_price_and_reviews_serper(product_name)
    
    if serper_results:
        return {
            "productName": product_name, 
            "results": serper_results,
            "source_used": "serper",
            "fallback_used": True
        }
    
    # If both SERP API and Serper fail or return no results, try Google Custom Search
    google_results = fetch_price_and_reviews_google_custom(product_name)
    
    if google_results:
        return {
            "productName": product_name, 
            "results": google_results,
            "source_used": "google_custom",
            "fallback_used": True
        }
    
    # If all three fail, return empty results
    return {
        "productName": product_name, 
        "results": [],
        "source_used": "none",
        "fallback_used": True
    }


# =========================================================================
# === IMAGE SEARCH FUNCTIONS ===
# =========================================================================

def get_manufacturer_domains_from_llm(vendor_name: str) -> list:
    """
    Use LLM to dynamically generate manufacturer domain names based on vendor name
    """
    if not components or not components.get('llm'):
        # Fallback to common domains if LLM is not available
        return [
            "emerson.com", "yokogawa.com", "siemens.com", "abb.com", "honeywell.com",
            "schneider-electric.com", "ge.com", "rockwellautomation.com", "endress.com",
            "fluke.com", "krohne.com", "rosemount.com", "fisher.com", "metso.com"
        ]
    
    try:
        
        chain = prompts.manufacturer_domain_prompt | components['llm'] | StrOutputParser()
        
        response = chain.invoke({"vendor_name": vendor_name})
        
        # Parse the response to extract domain names
        domains = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line and '.' in line:
                # Clean up the line - remove any prefixes, bullets, numbers
                domain = line.split()[-1] if ' ' in line else line
                domain = domain.replace('www.', '').replace('http://', '').replace('https://', '')
                domain = domain.strip('.,()[]{}')
                
                if '.' in domain and len(domain) > 3:
                    domains.append(domain)
        
        # Ensure we have at least some domains
        if not domains:
            # Fallback: generate based on vendor name
            vendor_clean = vendor_name.lower().replace(' ', '').replace('&', '').replace('+', '')
            domains = [f"{vendor_clean}.com", f"{vendor_clean}.de", f"{vendor_clean}group.com"]
        
        logging.info(f"LLM generated {len(domains)} domains for {vendor_name}: {domains[:5]}...")
        return domains[:15]  # Limit to 15 domains
        
    except Exception as e:
        logging.warning(f"Failed to generate domains via LLM for {vendor_name}: {e}")
        # Fallback: generate based on vendor name
        vendor_clean = vendor_name.lower().replace(' ', '').replace('&', '').replace('+', '')
        return [f"{vendor_clean}.com", f"{vendor_clean}.de", f"{vendor_clean}group.com"]

def fetch_images_google_cse_sync(vendor_name: str, product_name: str = None, manufacturer_domains: list = None, model_family: str = None, product_type: str = None):
    """
    Synchronous version: Google Custom Search API for images from manufacturer domains
    
    Args:
        vendor_name: Name of the vendor/manufacturer
        product_name: (Optional) Specific product name/model
        manufacturer_domains: (Optional) List of manufacturer domains to search within
        model_family: (Optional) Model family/series to include in search
        product_type: (Optional) Type of product to help refine search
    """
    if not GOOGLE_API_KEY or not GOOGLE_CX:
        logging.warning("Google CSE credentials not available for image search")
        return []
    
    try:
        # Build the search query in format <vendor_name><modelfamily><product type>
        query = vendor_name
        if model_family:
            query += f" {model_family}"  # Add space for better tokenization
        if product_type:
            query += f" {product_type}"  # Add space for better tokenization
            
        query += " product image"
            
        # We intentionally do NOT include raw product_name in the search query
        # to focus searches on model_family and product_type only.
        
        # Build site restriction for manufacturer domains using LLM (or reuse if provided)
        if manufacturer_domains is None:
            manufacturer_domains = get_manufacturer_domains_from_llm(vendor_name)
        domain_filter = " OR ".join([f"site:{domain}" for domain in manufacturer_domains])
        search_query = f"{query} ({domain_filter}) filetype:jpg OR filetype:png"
        
        # Use Google Custom Search API
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        result = service.cse().list(
            q=search_query,
            cx=GOOGLE_CX,
            searchType="image",
            num=8,
            safe="medium",
            imgSize="MEDIUM"
        ).execute()
        
        images = []
        unsupported_schemes = ['x-raw-image://', 'data:', 'blob:', 'chrome://', 'about:']
        
        for item in result.get("items", []):
            url = item.get("link")
            
            # Skip images with unsupported URL schemes
            if not url or any(url.startswith(scheme) for scheme in unsupported_schemes):
                logging.debug(f"Skipping image with unsupported URL scheme: {url}")
                continue
            
            # Only include http/https URLs
            if not url.startswith(('http://', 'https://')):
                logging.debug(f"Skipping non-HTTP URL: {url}")
                continue
                
            images.append({
                "url": url,
                "title": item.get("title", ""),
                "source": "google_cse",
                "thumbnail": item.get("image", {}).get("thumbnailLink", ""),
                "domain": item.get("displayLink", "")
            })
        
        if images:
            logging.info(f"Google CSE found {len(images)} valid images for {vendor_name}")
        return images
        
    except Exception as e:
        logging.warning(f"Google CSE image search failed for {vendor_name}: {e}")
        return []

def fetch_images_serpapi_sync(vendor_name: str, product_name: str = None, model_family: str = None, product_type: str = None):
    """
    Synchronous version: SerpAPI fallback for Google Images
    
    Args:
        vendor_name: Name of the vendor/manufacturer
        product_name: (Optional) Specific product name/model
        model_family: (Optional) Model family/series to include in search
        product_type: (Optional) Type of product to help refine search
    """
    if not SERPAPI_KEY:
        logging.warning("SerpAPI key not available for image search")
        return []
    
    try:
        # Build the base query in format <vendor_name> <model_family?> <product_type?>
        base_query = vendor_name
        if model_family:
            base_query += f" {model_family}"
        if product_type:
            base_query += f" {product_type}"

        # Do not include raw product_name here — rely on model_family/product_type

        # Add helpful positive/negative tokens used previously
        base_query += " product image OR product OR datasheet OR specification -used -refurbished -ebay -amazon -alibaba -walmart -etsy -pinterest -youtube -video -pdf -doc -xls -ppt -docx -xlsx -pptx"

        # Build manufacturer domain filter using LLM domains when available
        try:
            manufacturer_domains = get_manufacturer_domains_from_llm(vendor_name)
        except Exception:
            manufacturer_domains = []

        domain_filter = " OR ".join([f"site:{domain}" for domain in manufacturer_domains]) if manufacturer_domains else ""
        if domain_filter:
            search_query = f"{base_query} ({domain_filter}) filetype:jpg OR filetype:png"
        else:
            search_query = f"{base_query} filetype:jpg OR filetype:png"

        search = GoogleSearch({
            "q": search_query,
            "engine": "google_images",
            "api_key": SERPAPI_KEY,
            "num": 8,
            "safe": "medium",
            "ijn": 0
        })
        
        results = search.get_dict()
        images = []
        
        for item in results.get("images_results", []):
            images.append({
                "url": item.get("original"),
                "title": item.get("title", ""),
                "source": "serpapi",
                "thumbnail": item.get("thumbnail", ""),
                "domain": item.get("source", "")
            })
        
        if images:
            logging.info(f"SerpAPI found {len(images)} images for {vendor_name}")
        return images
        
    except Exception as e:
        logging.warning(f"SerpAPI image search failed for {vendor_name}: {e}")
        return []

def fetch_images_serper_sync(vendor_name: str, product_name: str = None, model_family: str = None, product_type: str = None):
    """
    Synchronous version: Serper.dev fallback for images
    
    Args:
        vendor_name: Name of the vendor/manufacturer
        product_name: (Optional) Specific product name/model
        model_family: (Optional) Model family/series to include in search
        product_type: (Optional) Type of product to help refine search
    """
    if not SERPER_API_KEY_IMAGES:
        logging.warning("Serper API key not available for image search")
        return []
    
    try:
        # Build the base query in format <vendor_name> <model_family?> <product_type?>
        base_query = vendor_name
        if model_family:
            base_query += f" {model_family}"
        if product_type:
            base_query += f" {product_type}"

        # Do not include raw product_name here — rely on model_family/product_type

        base_query += " product image OR product OR datasheet OR specification -used -refurbished -ebay -amazon -alibaba -walmart -etsy -pinterest -youtube -video -pdf -doc -xls -ppt -docx -xlsx -pptx"

        try:
            manufacturer_domains = get_manufacturer_domains_from_llm(vendor_name)
        except Exception:
            manufacturer_domains = []

        domain_filter = " OR ".join([f"site:{domain}" for domain in manufacturer_domains]) if manufacturer_domains else ""
        if domain_filter:
            search_query = f"{base_query} ({domain_filter}) filetype:jpg OR filetype:png"
        else:
            search_query = f"{base_query} filetype:jpg OR filetype:png"

        url = "https://google.serper.dev/images"
        payload = {
            "q": search_query,
            "num": 8
        }
        headers = {
            "X-API-KEY": SERPER_API_KEY_IMAGES,
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        images = []
        
        for item in data.get("images", []):
            images.append({
                "url": item.get("imageUrl"),
                "title": item.get("title", ""),
                "source": "serper",
                "thumbnail": item.get("imageUrl"),  # Serper doesn't provide separate thumbnail
                "domain": item.get("link", "")
            })
        
        if images:
            logging.info(f"Serper found {len(images)} images for {vendor_name}")
        return images
        
    except Exception as e:
        logging.warning(f"Serper image search failed for {vendor_name}: {e}")
        return []

def fetch_product_images_with_fallback_sync(vendor_name: str, product_name: str = None, manufacturer_domains: list = None, model_family: str = None, product_type: str = None):
    """
    Synchronous 3-level image search fallback system with MongoDB caching
    0. Check MongoDB cache first
    1. Google Custom Search API (manufacturer domains)
    2. SerpAPI Google Images
    3. Serper.dev images
    
    Args:
        vendor_name: Name of the vendor/manufacturer
        product_name: (Optional) Specific product name/model
        manufacturer_domains: (Optional) List of manufacturer domains to search within
        model_family: (Optional) Model family/series to include in search
        product_type: (Optional) Type of product to help refine search
    """
    logging.info(f"Starting image search for vendor: {vendor_name}, product: {product_name}, model family: {model_family}, type: {product_type}")
    
    # Step 0: Check MongoDB cache first (if model_family is provided)
    if vendor_name and model_family:
        from azure_blob_utils import get_cached_image, cache_image
        
        cached_image = get_cached_image(vendor_name, model_family)
        if cached_image:
            logging.info(f"Using cached image from GridFS for {vendor_name} - {model_family}")
            # Convert GridFS file_id to backend URL
            gridfs_file_id = cached_image.get('gridfs_file_id')
            backend_url = f"/api/images/{gridfs_file_id}"
            
            # Return image with backend URL
            image_response = {
                'url': backend_url,
                'title': cached_image.get('title', ''),
                'source': 'mongodb_gridfs',
                'thumbnail': backend_url,  # Same URL for thumbnail
                'domain': 'local',
                'cached': True,
                'gridfs_file_id': gridfs_file_id
            }
            return [image_response], "mongodb_gridfs"
    
    # Step 1: Try Google Custom Search API
    images = fetch_images_google_cse_sync(
        vendor_name=vendor_name,
        product_name=product_name,
        manufacturer_domains=manufacturer_domains,
        model_family=model_family,
        product_type=product_type
    )
    if images:
        logging.info(f"Using Google CSE images for {vendor_name}")
        # Cache the top image if model_family is provided
        if vendor_name and model_family and len(images) > 0:
            from azure_blob_utils import cache_image
            cache_image(vendor_name, model_family, images[0])
        return images, "google_cse"
    
    # Step 2: Try SerpAPI
    images = fetch_images_serpapi_sync(
        vendor_name=vendor_name,
        product_name=product_name,
        model_family=model_family,
        product_type=product_type
    )
    if images:
        logging.info(f"Using SerpAPI images for {vendor_name}")
        # Cache the top image if model_family is provided
        if vendor_name and model_family and len(images) > 0:
            from azure_blob_utils import cache_image
            cache_image(vendor_name, model_family, images[0])
        return images, "serpapi"
    
    # Step 3: Try Serper.dev
    images = fetch_images_serper_sync(
        vendor_name=vendor_name,
        product_name=product_name,
        model_family=model_family,
        product_type=product_type
    )
    if images:
        logging.info(f"Using Serper images for {vendor_name}")
        # Cache the top image if model_family is provided
        if vendor_name and model_family and len(images) > 0:
            from mongodb_utils import cache_image
            cache_image(vendor_name, model_family, images[0])
        return images, "serper"
    
    # All failed
    logging.warning(f"All image search APIs failed for {vendor_name}")
    return [], "none"

def fetch_vendor_logo_sync(vendor_name: str, manufacturer_domains: list = None):
    """
    Specialized function to fetch vendor logo with MongoDB caching
    """
    logging.info(f"Fetching logo for vendor: {vendor_name}")
    
    # Step 0: Check MongoDB cache first
    try:
        from azure_blob_utils import mongodb_file_manager, download_image_from_url
        
        logos_collection = mongodb_file_manager.collections.get('vendor_logos')
        if logos_collection is not None:
            normalized_vendor = vendor_name.strip().lower()
            
            cached_logo = logos_collection.find_one({
                'vendor_name_normalized': normalized_vendor
            })
            
            if cached_logo and cached_logo.get('gridfs_file_id'):
                logging.info(f"Using cached logo from GridFS for {vendor_name}")
                gridfs_file_id = cached_logo.get('gridfs_file_id')
                backend_url = f"/api/images/{gridfs_file_id}"
                
                return {
                    'url': backend_url,
                    'thumbnail': backend_url,
                    'source': 'mongodb_gridfs',
                    'title': cached_logo.get('title', f"{vendor_name} Logo"),
                    'domain': 'local',
                    'cached': True,
                    'gridfs_file_id': str(gridfs_file_id)
                }
    except Exception as e:
        logging.warning(f"Failed to check logo cache for {vendor_name}: {e}")
    
    # Step 1: Cache miss - fetch from web
    logo_result = None
    
    # Try different logo-specific searches
    logo_queries = [
        f"{vendor_name} logo",
        f"{vendor_name} company logo", 
        f"{vendor_name} brand",
        f"{vendor_name}"
    ]
    
    for query in logo_queries:
        try:
            # Use Google CSE first for official logos
            if GOOGLE_API_KEY and GOOGLE_CX:
                # Build site restriction for manufacturer domains using LLM (or reuse if provided)
                if manufacturer_domains is None:
                    manufacturer_domains = get_manufacturer_domains_from_llm(vendor_name)
                domain_filter = " OR ".join([f"site:{domain}" for domain in manufacturer_domains])
                search_query = f"{query} ({domain_filter}) filetype:jpg OR filetype:png OR filetype:svg"
                
                service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
                result = service.cse().list(
                    q=search_query,
                    cx=GOOGLE_CX,
                    searchType="image",
                    num=3,  # Only need a few logo options
                    safe="medium",
                    imgSize="MEDIUM"
                ).execute()
                
                for item in result.get("items", []):
                    logo_url = item.get("link")
                    title = item.get("title", "").lower()
                    
                    # Check if this looks like a logo
                    if any(keyword in title for keyword in ["logo", "brand", "company"]):
                        logo_result = {
                            "url": logo_url,
                            "thumbnail": item.get("image", {}).get("thumbnailLink", logo_url),
                            "source": "google_cse_logo",
                            "title": item.get("title", ""),
                            "domain": item.get("displayLink", "")
                        }
                        break
                
                # If no specific logo found, use first result from official domain
                if not logo_result and result.get("items"):
                    item = result["items"][0]
                    logo_result = {
                        "url": item.get("link"),
                        "thumbnail": item.get("image", {}).get("thumbnailLink", item.get("link")),
                        "source": "google_cse_general",
                        "title": item.get("title", ""),
                        "domain": item.get("displayLink", "")
                    }
                
                if logo_result:
                    break
                    
        except Exception as e:
            logging.warning(f"Logo search failed for query '{query}': {e}")
            continue
    
    # Fallback: use general vendor search
    if not logo_result:
        try:
            images, source = fetch_product_images_with_fallback_sync(vendor_name, "")
            if images:
                # Return first image as logo
                logo_result = images[0].copy()
                logo_result["source"] = f"{source}_fallback"
        except Exception as e:
            logging.warning(f"Fallback logo search failed for {vendor_name}: {e}")
    
    # Step 2: Cache the logo in MongoDB if found
    if logo_result:
        try:
            from mongodb_utils import mongodb_file_manager, download_image_from_url
            
            logo_url = logo_result.get('url')
            if logo_url and not logo_url.startswith('/api/images/'):  # Don't re-cache GridFS URLs
                # Download the logo
                download_result = download_image_from_url(logo_url)
                if download_result:
                    image_bytes, content_type, file_size = download_result
                    
                    gridfs = mongodb_file_manager.gridfs
                    logos_collection = mongodb_file_manager.collections.get('vendor_logos')
                    
                    if logos_collection is not None:
                        normalized_vendor = vendor_name.strip().lower()
                        file_extension = content_type.split('/')[-1] if '/' in content_type else 'png'
                        filename = f"logo_{normalized_vendor}.{file_extension}"
                        
                        # Store in GridFS
                        gridfs_file_id = gridfs.put(
                            image_bytes,
                            filename=filename,
                            content_type=content_type,
                            vendor_name=vendor_name,
                            original_url=logo_url,
                            logo_type='vendor_logo'
                        )
                        
                        logging.info(f"Stored vendor logo in GridFS: {filename} (ID: {gridfs_file_id})")
                        
                        # Store metadata
                        logo_doc = {
                            'vendor_name': vendor_name,
                            'vendor_name_normalized': normalized_vendor,
                            'gridfs_file_id': gridfs_file_id,
                            'original_url': logo_url,
                            'title': logo_result.get('title', f"{vendor_name} Logo"),
                            'source': logo_result.get('source', ''),
                            'domain': logo_result.get('domain', ''),
                            'content_type': content_type,
                            'file_size': file_size,
                            'filename': filename,
                            'created_at': datetime.utcnow()
                        }
                        
                        logos_collection.update_one(
                            {'vendor_name_normalized': normalized_vendor},
                            {'$set': logo_doc},
                            upsert=True
                        )
                        
                        logging.info(f"Successfully cached vendor logo for {vendor_name}")
                        
                        # Return cached version
                        backend_url = f"/api/images/{gridfs_file_id}"
                        return {
                            'url': backend_url,
                            'thumbnail': backend_url,
                            'source': 'mongodb_gridfs',
                            'title': logo_doc['title'],
                            'domain': 'local',
                            'cached': True,
                            'gridfs_file_id': str(gridfs_file_id)
                        }
        except Exception as e:
            logging.warning(f"Failed to cache vendor logo for {vendor_name}: {e}")
    
    return logo_result

async def fetch_images_google_cse(vendor_name: str, model_family: str = None, product_type: str = None):
    """
    Step 1: Google Custom Search API for images from manufacturer domains
    """
    if not GOOGLE_API_KEY or not GOOGLE_CX:
        logging.warning("Google CSE credentials not available for image search")
        return []
    
    try:
        query = f"{vendor_name}"
        if model_family:
            query += f" {model_family}"
        if product_type:
            query += f" {product_type}"
        
        # Build site restriction for manufacturer domains using LLM
        manufacturer_domains = get_manufacturer_domains_from_llm(vendor_name)
        domain_filter = " OR ".join([f"site:{domain}" for domain in manufacturer_domains])
        search_query = f"{query} ({domain_filter}) filetype:jpg OR filetype:png"
        
        # Use Google Custom Search API
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        result = service.cse().list(
            q=search_query,
            cx=GOOGLE_CX,
            searchType="image",
            num=8,
            safe="medium",
            imgSize="MEDIUM"
        ).execute()
        
        images = []
        unsupported_schemes = ['x-raw-image://', 'data:', 'blob:', 'chrome://', 'about:']
        
        for item in result.get("items", []):
            url = item.get("link")
            
            # Skip images with unsupported URL schemes
            if not url or any(url.startswith(scheme) for scheme in unsupported_schemes):
                logging.debug(f"Skipping image with unsupported URL scheme: {url}")
                continue
            
            # Only include http/https URLs
            if not url.startswith(('http://', 'https://')):
                logging.debug(f"Skipping non-HTTP URL: {url}")
                continue
                
            images.append({
                "url": url,
                "title": item.get("title", ""),
                "source": "google_cse",
                "thumbnail": item.get("image", {}).get("thumbnailLink", ""),
                "domain": item.get("displayLink", "")
            })
        
        if images:
            logging.info(f"Google CSE found {len(images)} valid images for {vendor_name}")
        return images
        
    except Exception as e:
        logging.warning(f"Google CSE image search failed for {vendor_name}: {e}")
        return []

async def fetch_images_serpapi(vendor_name: str, model_family: str = None, product_type: str = None):
    """
    Step 2: SerpAPI fallback for Google Images
    """
    if not SERPAPI_KEY:
        logging.warning("SerpAPI key not available for image search")
        return []
    
    try:
        base_query = vendor_name
        if model_family:
            base_query += f" {model_family}"
        if product_type:
            base_query += f" {product_type}"
        base_query += " product OR datasheet OR specification -used -refurbished -ebay -amazon -alibaba -walmart -etsy -pinterest -youtube -video -pdf -doc -xls -ppt -docx -xlsx -pptx"

        try:
            manufacturer_domains = get_manufacturer_domains_from_llm(vendor_name)
        except Exception:
            manufacturer_domains = []

        domain_filter = " OR ".join([f"site:{domain}" for domain in manufacturer_domains]) if manufacturer_domains else ""
        if domain_filter:
            search_query = f"{base_query} ({domain_filter}) filetype:jpg OR filetype:png"
        else:
            search_query = f"{base_query} filetype:jpg OR filetype:png"

        search = GoogleSearch({
            "q": search_query,
            "engine": "google_images",
            "api_key": SERPAPI_KEY,
            "num": 8,
            "safe": "medium",
            "ijn": 0
        })
        
        results = search.get_dict()
        images = []
        
        for item in results.get("images_results", []):
            images.append({
                "url": item.get("original"),
                "title": item.get("title", ""),
                "source": "serpapi",
                "thumbnail": item.get("thumbnail", ""),
                "domain": item.get("source", "")
            })
        
        if images:
            logging.info(f"SerpAPI found {len(images)} images for {vendor_name}")
        return images
        
    except Exception as e:
        logging.warning(f"SerpAPI image search failed for {vendor_name}: {e}")
        return []

async def fetch_images_serper(vendor_name: str, model_family: str = None, product_type: str = None):
    """
    Step 3: Serper.dev fallback for images
    """
    if not SERPER_API_KEY_IMAGES:
        logging.warning("Serper API key not available for image search")
        return []
    
    try:
        base_query = vendor_name
        if model_family:
            base_query += f" {model_family}"
        if product_type:
            base_query += f" {product_type}"
        base_query += " product OR datasheet OR specification -used -refurbished -ebay -amazon -alibaba -walmart -etsy -pinterest -youtube -video -pdf -doc -xls -ppt -docx -xlsx -pptx"

        try:
            manufacturer_domains = get_manufacturer_domains_from_llm(vendor_name)
        except Exception:
            manufacturer_domains = []

        domain_filter = " OR ".join([f"site:{domain}" for domain in manufacturer_domains]) if manufacturer_domains else ""
        if domain_filter:
            search_query = f"{base_query} ({domain_filter}) filetype:jpg OR filetype:png"
        else:
            search_query = f"{base_query} filetype:jpg OR filetype:png"

        url = "https://google.serper.dev/images"
        payload = {
            "q": search_query,
            "num": 8
        }
        headers = {
            "X-API-KEY": SERPER_API_KEY_IMAGES,
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        images = []
        
        for item in data.get("images", []):
            images.append({
                "url": item.get("imageUrl"),
                "title": item.get("title", ""),
                "source": "serper",
                "thumbnail": item.get("imageUrl"),  # Serper doesn't provide separate thumbnail
                "domain": item.get("link", "")
            })
        
        if images:
            logging.info(f"Serper found {len(images)} images for {vendor_name}")
        return images
        
    except Exception as e:
        logging.warning(f"Serper image search failed for {vendor_name}: {e}")
        return []

async def fetch_product_images_with_fallback(vendor_name: str, product_name: str = None, model_family: str = None, product_type: str = None):
    """
    3-level image search fallback system
    1. Google Custom Search API (manufacturer domains)
    2. SerpAPI Google Images
    3. Serper.dev images
    """
    logging.info(f"Starting image search for vendor: {vendor_name}, product: {product_name}")
    
    # Step 1: Try Google Custom Search API (pass model_family/product_type, avoid raw product_name)
    images = await fetch_images_google_cse(vendor_name, model_family if model_family else None)
    if images:
        logging.info(f"Using Google CSE images for {vendor_name}")
        return images, "google_cse"
    
    # Step 2: Try SerpAPI
    images = await fetch_images_serpapi(vendor_name, model_family if model_family else None)
    if images:
        logging.info(f"Using SerpAPI images for {vendor_name}")
        return images, "serpapi"
    
    # Step 3: Try Serper.dev
    images = await fetch_images_serper(vendor_name, model_family if model_family else None)
    if images:
        logging.info(f"Using Serper images for {vendor_name}")
        return images, "serper"
    
    # All failed
    logging.warning(f"All image search APIs failed for {vendor_name}")
    return [], "none"


@app.route("/api/test_image_search", methods=["GET"])
@login_required
def test_image_search():
    """
    Test endpoint for the image search functionality
    """
    vendor_name = request.args.get("vendor", "Emerson")
    product_name = request.args.get("product", "")
    
    try:
        # Use synchronous version for reliability; pass model_family instead of product_name
        model_family = None
        # If user provided product as a family list or string, prefer it
        if product_name and ',' in product_name:
            # accept '3051C,3051S' style input from quick tests
            model_family = product_name.split(',')[0].strip()
        elif product_name:
            model_family = product_name.strip()

        images, source_used = fetch_product_images_with_fallback_sync(
            vendor_name,
            product_name=None,
            manufacturer_domains=None,
            model_family=model_family,
            product_type=None
        )
        
        # Also test domain generation
        generated_domains = get_manufacturer_domains_from_llm(vendor_name)
        
        return jsonify({
            "vendor": vendor_name,
            "product": product_name,
            "images": images,
            "source_used": source_used,
            "count": len(images),
            "generated_domains": generated_domains
        })
        
    except Exception as e:
        logging.error(f"Image search test failed: {e}")
        return jsonify({
            "error": str(e),
            "vendor": vendor_name,
            "product": product_name,
            "images": [],
            "source_used": "error",
            "count": 0,
            "generated_domains": []
        }), 500


@app.route("/api/get_analysis_product_images", methods=["POST"])
@login_required
def get_analysis_product_images():
    """
    Get images for specific products from analysis results.
    Expected input:
    {
        "vendor": "Emerson",
        "product_type": "Flow Transmitter", 
        "product_name": "Rosemount 3051",
        "model_families": ["3051C", "3051S", "3051T"]
    }
    """
    try:
        data = request.get_json()

        vendor = data.get("vendor", "")
        product_type = data.get("product_type", "")
        product_name = data.get("product_name", "")
        model_families = data.get("model_families", [])

        if not vendor:
            return jsonify({"error": "Vendor name is required"}), 400

        # Removed requirements_match check - fetch images for ALL products (exact and approximate matches)
        # This supports the fallback display of approximate matches when no exact matches are found
        logging.info(f"Fetching images for analysis result: {vendor} {product_type} {product_name}")

        # Generate manufacturer domains once per request for this vendor
        manufacturer_domains = get_manufacturer_domains_from_llm(vendor)

        # Search for images with different combinations
        all_images = []
        search_combinations = []

        # Prefer model family for search if available (e.g., STD800 instead of submodel STD830)
        primary_family = None
        if isinstance(model_families, list) and model_families:
            primary_family = str(model_families[0]).strip()

        # Build a base name for search: model family if present, otherwise product_name
        # Example: "STD800" instead of "STD830 Pressure Transmitter"
        base_name_for_search = primary_family or product_name

        # 1. Most specific: vendor + base_name_for_search + product_type
        if base_name_for_search and product_type:
            search_query = f"{vendor} {base_name_for_search} {product_type}"
            search_combinations.append({
                "query": search_query,
                "type": "family_with_type",
                "priority": 1
            })

        # 2. Medium specific: vendor + base_name_for_search
        if base_name_for_search:
            search_query = f"{vendor} {base_name_for_search}"
            search_combinations.append({
                "query": search_query,
                "type": "family_or_name",
                "priority": 2
            })

        # 3. General: vendor + product_type
        if product_type:
            search_query = f"{vendor} {product_type}"
            search_combinations.append({
                "query": search_query,
                "type": "type_general",
                "priority": 3
            })
        
        # Execute searches and collect results
        for search_info in search_combinations:
            try:
                # Pass model_family and product_type to the fetcher and avoid using raw product_name
                images, source_used = fetch_product_images_with_fallback_sync(
                    vendor_name=vendor,
                    product_name=None,
                    manufacturer_domains=manufacturer_domains,
                    model_family=base_name_for_search if base_name_for_search else None,
                    product_type=product_type if product_type else None,
                )
                
                # Add metadata to images
                for img in images:
                    img["search_type"] = search_info["type"]
                    img["search_priority"] = search_info["priority"]
                    img["search_query"] = search_info["query"]
                
                all_images.extend(images)
                
                # If we get good results from high-priority search, we can stop early
                if len(images) >= 5 and search_info["priority"] <= 2:
                    logging.info(f"Got {len(images)} images from high-priority search: {search_info['type']}")
                    break
                    
            except Exception as e:
                logging.warning(f"Search failed for query '{search_info['query']}': {e}")
                continue
        
        # Remove duplicates based on URL
        unique_images = []
        seen_urls = set()
        for img in all_images:
            url = img.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_images.append(img)
        
        # Sort by priority and quality
        def image_quality_score(img):
            score = 0
            
            # Priority weight (lower priority number = higher score)
            score += (5 - img.get("search_priority", 5)) * 10
            
            # Domain quality (official domains get higher score)
            domain = img.get("domain", "").lower()
            if any(mfg_domain in domain for mfg_domain in manufacturer_domains):
                score += 15
            
            # Source quality
            source = img.get("source", "")
            if source == "google_cse":
                score += 10
            elif source == "serpapi":
                score += 5
            
            # Title relevance (contains product name or model family)
            title = img.get("title", "").lower()
            if product_name.lower() in title:
                score += 8
            for model in model_families:
                if model.lower() in title:
                    score += 6
                    break
            
            return score
        
        # Sort by quality score (highest first)
        unique_images.sort(key=image_quality_score, reverse=True)
        
        # Select best images - top 1 for main display, top 10 for "view more"
        top_image = unique_images[0] if unique_images else None
        best_images = unique_images[:10]
        
        # Get vendor logo using specialized logo search
        vendor_logo = None
        try:
            vendor_logo = fetch_vendor_logo_sync(vendor, manufacturer_domains=manufacturer_domains)
        except Exception as e:
            logging.warning(f"Failed to fetch vendor logo for {vendor}: {e}")
        
        # Prepare response
        response_data = {
            "vendor": vendor,
            "product_type": product_type,
            "product_name": product_name,
            "model_families": model_families,
            "top_image": top_image,  # Single best image for main display
            "vendor_logo": vendor_logo,  # Vendor logo
            "all_images": best_images,  # All images for "view more"
            # Compatibility fields: many frontends expect `images` or `image`
            "images": best_images,
            "image": top_image,
            "total_found": len(all_images),
            "unique_count": len(unique_images),
            "best_count": len(best_images),
            "search_summary": {
                "searches_performed": len(search_combinations),
                "search_types": list(set(img.get("search_type") for img in best_images)),
                "sources_used": list(set(img.get("source") for img in best_images))
            }
        }
        
        logging.info(f"Analysis image search completed: {len(best_images)} best images selected from {len(all_images)} total")
        return jsonify(response_data)
        
    except Exception as e:
        logging.error(f"Analysis product image search failed: {e}")
        return jsonify({
            "error": f"Failed to fetch analysis product images: {str(e)}",
            "vendor": data.get("vendor", ""),
            "product_type": data.get("product_type", ""),
            "product_name": data.get("product_name", ""),
            "model_families": data.get("model_families", []),
            "top_image": None,
            "vendor_logo": None,
            "all_images": [],
            "total_found": 0,
            "unique_count": 0,
            "best_count": 0
        }), 500


@app.route("/api/upload_pdf_from_url", methods=["POST"])
@login_required
def upload_pdf_from_url():
    data = request.get_json(force=True)
    pdf_url = data.get("url")
    if not pdf_url:
        return jsonify({"error": "Missing 'url' parameter"}), 400

    try:
        # --- 1. Download PDF ---
        print(f"[DOWNLOAD] Fetching PDF: {pdf_url}")
        response = requests.get(pdf_url, stream=True, timeout=30)
        response.raise_for_status()

        filename = os.path.basename(urllib.parse.urlparse(pdf_url).path) or "document.pdf"
        pdf_bytes = response.content  # keep PDF in memory

        # --- 2. Extract data from PDF ---
        text_chunks = extract_data_from_pdf(BytesIO(pdf_bytes))
        raw_results = send_to_language_model(text_chunks)

        # Flatten results
        def flatten_results(results):
            flat = []
            for r in results:
                if isinstance(r, list):
                    flat.extend(r)
                else:
                    flat.append(r)
            return flat

        all_results = flatten_results(raw_results)
        final_result = aggregate_results(all_results, filename)
        
        # Apply standardization to the final result before splitting
        try:
            standardized_final_result = standardize_vendor_analysis_result(final_result)
            logging.info("Applied standardization to PDF from URL analysis")
        except Exception as e:
            logging.warning(f"Failed to standardize PDF from URL result: {e}")
            standardized_final_result = final_result

        # --- 3. Split by product types ---
        split_results = split_product_types([standardized_final_result])

        saved_json_paths = []
        saved_pdf_paths = []

        for result in split_results:
            # --- 4. Save JSON result to MongoDB ---
            vendor = (result.get("vendor") or "UnknownVendor").replace(" ", "_")
            product_type = (result.get("product_type") or "UnknownProduct").replace(" ", "_")
            model_series = (
                result.get("models", [{}])[0].get("model_series") or "UnknownModel"
            ).replace(" ", " ")
            
            try:
                # Upload product JSON to MongoDB
                # Structure: vendors/{vendor}/{product_type}/{model}.json
                product_metadata = {
                    'vendor_name': vendor,
                    'product_type': product_type,
                    'model_series': model_series,
                    'file_type': 'json',
                    'collection_type': 'products',
                    'path': f'vendors/{vendor}/{product_type}/{model_series}.json'
                }
                mongodb_file_manager.upload_json_data(result, product_metadata)
                saved_json_paths.append(f"MongoDB:products:{vendor}:{product_type}:{model_series}")
                print(f"[INFO] Stored product JSON to MongoDB: {vendor} - {product_type}")
            except Exception as e:
                logging.error(f"Failed to save product JSON to MongoDB: {e}")

            # --- 5. Save PDF to MongoDB GridFS ---
            try:
                pdf_metadata = {
                    'vendor_name': vendor,
                    'product_type': product_type,
                    'model_series': model_series,
                    'file_type': 'pdf',
                    'collection_type': 'documents',
                    'filename': filename,
                    'path': f'documents/{vendor}/{product_type}/{filename}'
                }
                file_id = mongodb_file_manager.upload_to_mongodb(pdf_bytes, pdf_metadata)
                saved_pdf_paths.append(f"MongoDB:GridFS:{file_id}")
                print(f"[INFO] Stored PDF to MongoDB GridFS: {filename} (ID: {file_id})")
            except Exception as e:
                logging.error(f"Failed to save PDF to MongoDB: {e}")

            # --- Note: Product image extraction removed - now using API-based image search ---

        return jsonify({
            "data": split_results,
            "pdfFiles": saved_pdf_paths,
            "jsonFiles": saved_json_paths
        }), 200

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to fetch PDF from URL: {str(e)}"}), 500
    except Exception as e:
        logging.exception("PDF analysis from URL failed.")
        return jsonify({"error": f"Failed to analyze PDF from URL: {str(e)}"}), 500

    
@app.route("/register", methods=["POST"])
@limiter.limit("5 per minute;20 per hour;100 per day")
def register():
    """
    Register new user
    ---
    tags:
      - Authentication
    summary: Register a new user account
    description: |
      Creates a new user account with pending status. The account must be approved
      by an administrator before the user can log in.
    consumes:
      - application/json
    produces:
      - application/json
    parameters:
      - in: body
        name: body
        description: User registration data
        required: true
        schema:
          type: object
          required:
            - username
            - email
            - password
          properties:
            username:
              type: string
              description: Unique username
              example: "johndoe"
            email:
              type: string
              format: email
              description: User email address
              example: "john.doe@example.com"
            password:
              type: string
              format: password
              description: User password
              example: "SecurePass123!"
            first_name:
              type: string
              description: User's first name
              example: "John"
            last_name:
              type: string
              description: User's last name
              example: "Doe"
    responses:
      201:
        description: Registration successful
        schema:
          type: object
          properties:
            message:
              type: string
              example: "User registration submitted. Awaiting admin approval."
      400:
        description: Missing required fields
      409:
        description: Username or email already exists
    """
    data = request.get_json()
    username = data.get("username")
    email = data.get("email")
    password = data.get("password")
    first_name = data.get("first_name")
    last_name = data.get("last_name")

    if not username or not email or not password:
        return jsonify({"error": "Missing username, email, or password"}), 400

    if User.query.filter_by(username=username).first():
        return jsonify({"error": "Username already exists"}), 409
    if User.query.filter_by(email=email).first():
        return jsonify({"error": "Email already registered"}), 409

    hashed_pw = hash_password(password)
    new_user = User(
        username=username,
        email=email,
        password_hash=hashed_pw,
        first_name=first_name,
        last_name=last_name,
        status='pending',
        role='user'
    )
    db.session.add(new_user)
    db.session.commit()

    return jsonify({"message": "User registration submitted. Awaiting admin approval."}), 201

@app.route("/login", methods=["POST"])
@limiter.limit("5 per minute;20 per hour;100 per day")
def login():
    """
    User login
    ---
    tags:
      - Authentication
    summary: Authenticate user and create session
    description: |
      Authenticates user credentials and creates a session. The session cookie
      is automatically set and used for subsequent authenticated requests.
    consumes:
      - application/json
    produces:
      - application/json
    parameters:
      - in: body
        name: body
        description: Login credentials
        required: true
        schema:
          type: object
          required:
            - username
            - password
          properties:
            username:
              type: string
              description: Username
              example: "johndoe"
            password:
              type: string
              format: password
              description: User password
              example: "SecurePass123!"
    responses:
      200:
        description: Login successful
        schema:
          type: object
          properties:
            message:
              type: string
              example: "Login successful"
            user:
              type: object
              properties:
                username:
                  type: string
                name:
                  type: string
                first_name:
                  type: string
                last_name:
                  type: string
                email:
                  type: string
                role:
                  type: string
                  enum: [user, admin]
      401:
        description: Invalid credentials
      403:
        description: Account not active
    """
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    user = User.query.filter_by(username=username).first()
    if user and check_password(user.password_hash, password):
        if user.status != 'active':
            return jsonify({"error": f"Account not active. Current status: {user.status}."}), 403
        session['user_id'] = user.id
        # Construct full name from first_name and last_name
        full_name = ""
        if user.first_name and user.last_name:
            full_name = f"{user.first_name} {user.last_name}"
        elif user.first_name:
            full_name = user.first_name
        elif user.last_name:
            full_name = user.last_name
        else:
            full_name = user.username
        
        return jsonify({
            "message": "Login successful",
            "user": {
                "username": user.username,
                "name": full_name,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "email": user.email,
                "role": user.role
            }
        }), 200

    return jsonify({"error": "Invalid username or password"}), 401

@app.route("/logout", methods=["POST"])
def logout():
    """
    User logout
    ---
    tags:
      - Authentication
    summary: End user session
    description: Clears the user session and logs out the user.
    produces:
      - application/json
    responses:
      200:
        description: Logout successful
        schema:
          type: object
          properties:
            message:
              type: string
              example: "Logout successful"
    """
    session.pop('user_id', None)
    return jsonify({"message": "Logout successful"}), 200

@app.route("/user", methods=["GET"])
@login_required
def get_current_user():
    """
    Get current user
    ---
    tags:
      - Authentication
    summary: Get current authenticated user
    description: Returns the profile information of the currently logged-in user.
    produces:
      - application/json
    responses:
      200:
        description: User profile data
        schema:
          type: object
          properties:
            user:
              type: object
              properties:
                username:
                  type: string
                name:
                  type: string
                first_name:
                  type: string
                last_name:
                  type: string
                email:
                  type: string
                role:
                  type: string
      401:
        description: Unauthorized - login required
      404:
        description: User not found
    """
    user = db.session.get(User, session['user_id'])
    if not user:
        return jsonify({"error": "User not found"}), 404
    # Construct full name from first_name and last_name
    full_name = ""
    if user.first_name and user.last_name:
        full_name = f"{user.first_name} {user.last_name}"
    elif user.first_name:
        full_name = user.first_name
    elif user.last_name:
        full_name = user.last_name
    else:
        full_name = user.username
    
    return jsonify({
        "user": {
            "username": user.username,
            "name": full_name,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "email": user.email,
            "role": user.role
        }
    }), 200

def clean_empty_values(data):
    """Recursively replaces 'Not specified', 'Not requested', etc., with empty strings."""
    if isinstance(data, dict):
        return {k: clean_empty_values(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_empty_values(item) for item in data]
    elif isinstance(data, str) and data.lower().strip() in ["not specified", "not requested", "none specified", "n/a", "na"]:
        return ""
    return data
import copy

def map_provided_to_schema(detected_schema: dict, provided: dict) -> dict:
    """
    Always maps providedRequirements (flat or nested) into the schema structure.
    Works dynamically for any product type (Humidity, Pressure, etc.).
    """
    mapped = copy.deepcopy(detected_schema)

    # Case 1: Provided already structured (mandatory/optional) → overlay values
    if "mandatoryRequirements" in provided or "optionalRequirements" in provided:
        for section in ["mandatoryRequirements", "optionalRequirements"]:
            if section in provided and section in mapped:
                for key, value in provided[section].items():
                    if key in mapped[section]:
                        mapped[section][key] = value
        return mapped

    # Case 2: Provided is flat dict → distribute into schema
    for key, value in provided.items():
        if key in mapped.get("mandatoryRequirements", {}):
            mapped["mandatoryRequirements"][key] = value
        elif key in mapped.get("optionalRequirements", {}):
            mapped["optionalRequirements"][key] = value

    return mapped

# =========================================================================
# === PROGRESS TRACKING ENDPOINT ===
# =========================================================================

# Global progress tracker for long-running operations
current_operation_progress = None

@app.route("/api/progress", methods=["GET"])
@login_required
def get_operation_progress():
    """Get progress of current long-running operation"""
    global current_operation_progress
    
    if current_operation_progress is None:
        return jsonify({
            "status": "no_active_operation",
            "message": "No active operation in progress"
        }), 200
    
    try:
        progress_data = current_operation_progress.get_progress()
        return jsonify({
            "status": "in_progress",
            "progress": progress_data
        }), 200
    except Exception as e:
        logging.error(f"Failed to get progress: {e}")
        return jsonify({
            "status": "error",
            "message": "Failed to retrieve progress information"
        }), 500

# =========================================================================
# === VALIDATION ENDPOINT ===
# =========================================================================

@app.route("/debug-session/<session_id>", methods=["GET"])
@login_required
def debug_session_state(session_id):
    """Debug endpoint to check session state for a specific search session"""
    current_step_key = f'current_step_{session_id}'
    current_intent_key = f'current_intent_{session_id}'
    product_type_key = f'product_type_{session_id}'
    
    session_data = {
        'session_id': session_id,
        'current_step': session.get(current_step_key, 'None'),
        'current_intent': session.get(current_intent_key, 'None'),
        'product_type': session.get(product_type_key, 'None'),
        'all_session_keys': [k for k in session.keys() if session_id in k],
        'all_keys': list(session.keys()),  # Show all keys for debugging
        'session_size': len(session.keys())
    }
    
    return jsonify(session_data), 200

@app.route("/debug-session-clear/<session_id>", methods=["POST"])
@login_required  
def clear_session_state(session_id):
    """Debug endpoint to manually clear session state for testing"""
    keys_to_remove = [k for k in session.keys() if session_id in k]
    
    for key in keys_to_remove:
        del session[key]
    
    return jsonify({
        'session_id': session_id,
        'cleared_keys': keys_to_remove,
        'status': 'cleared'
    }), 200



@app.route("/validate", methods=["POST"])
@login_required
def api_validate():
    if not components:
        return jsonify({"error": "Backend is not ready. LangChain failed."}), 503
    try:
        data = request.get_json(force=True)
        user_input = data.get("user_input", "").strip()
        if not user_input:
            return jsonify({"error": "Missing user_input"}), 400

        # Get search session ID if provided (for multiple search tabs)
        search_session_id = data.get("search_session_id", "default")
        
        # By default preserve any previously-detected product type and workflow state for this
        # search session. Only clear them when the client explicitly requests a reset
        # (for example when initializing a brand-new independent search tab).
        session_key = f'product_type_{search_session_id}'
        step_key = f'current_step_{search_session_id}'
        intent_key = f'current_intent_{search_session_id}'

        if data.get('reset', False):
            if session_key in session:
                logging.info(f"[VALIDATE] Session {search_session_id}: Clearing previous product type due to reset request: {session[session_key]}")
                del session[session_key]
            if step_key in session:
                logging.info(f"[VALIDATE] Session {search_session_id}: Clearing step state due to reset request: {session[step_key]}")
                del session[step_key]
            if intent_key in session:
                logging.info(f"[VALIDATE] Session {search_session_id}: Clearing intent state due to reset request: {session[intent_key]}")
                del session[intent_key]
        else:
            logging.info(f"[VALIDATE] Session {search_session_id}: Preserving existing product_type and workflow state if present.")
        
        # Store original user input for logging (session-isolated)
        session[f'log_user_query_{search_session_id}'] = user_input

        initial_schema = load_requirements_schema()
        
        # Add session context to LLM validation to prevent cross-contamination
        session_isolated_input = f"[Session: {search_session_id}] - This is a fresh, independent validation request. User input: {user_input}"
        
        temp_validation_result = components['validation_chain'].invoke({
            "user_input": session_isolated_input,
            "schema": json.dumps(initial_schema, indent=2),
            "format_instructions": components['validation_format_instructions']
        })
        detected_type = temp_validation_result.get('product_type', 'UnknownProduct')
        
        specific_schema = load_requirements_schema(detected_type)
        if not specific_schema:
            global current_operation_progress
            try:
                # Set up progress tracking for web schema building
                from loading import ProgressTracker
                current_operation_progress = ProgressTracker(4, f"Building Schema for {detected_type}")
                specific_schema = build_requirements_schema_from_web(detected_type)
            finally:
                # Clear progress tracker when done
                current_operation_progress = None

        # Add session context to detailed validation as well
        session_isolated_input = f"[Session: {search_session_id}] - This is a fresh, independent validation request. User input: {user_input}"
        
        validation_result = components['validation_chain'].invoke({
            "user_input": session_isolated_input,
            "schema": json.dumps(specific_schema, indent=2),
            "format_instructions": components['validation_format_instructions']
        })

        cleaned_provided_reqs = clean_empty_values(validation_result.get("provided_requirements", {}))

        mapped_provided_reqs = map_provided_to_schema(
            convert_keys_to_camel_case(specific_schema),
            convert_keys_to_camel_case(cleaned_provided_reqs)
        )

        response_data = {
            "productType": validation_result.get("product_type", detected_type),
            "detectedSchema": convert_keys_to_camel_case(specific_schema),
            "providedRequirements": mapped_provided_reqs
        }

        # ---------------- Helpers for missing mandatory fields ----------------
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

        # ---------------- Helper: Convert camelCase to friendly label ----------------
        def friendly_field_name(field):
            import re
            s1 = re.sub('([a-z0-9])([A-Z])', r'\1 \2', field)
            return s1.replace("_", " ").title()

        # ---------------- Prompt user if any mandatory fields are missing ----------------
        if missing_mandatory_fields:
            # Convert missing fields to friendly labels
            missing_fields_friendly = [friendly_field_name(f) for f in missing_mandatory_fields]
            missing_fields_str = ", ".join(missing_fields_friendly)
            is_repeat = data.get("is_repeat", False)

            if not is_repeat:
                alert_prompt = prompts.validation_alert_initial_prompt
            else:
                alert_prompt = prompts.validation_alert_repeat_prompt

            alert_chain = alert_prompt | components['llm'] | StrOutputParser()
            agent_message = alert_chain.invoke({
                "product_type": response_data["productType"],
                "missing_fields": missing_fields_str
            })

            response_data["validationAlert"] = {
                "message": agent_message,
                "canContinue": True,
                "missingFields": missing_mandatory_fields
            }

        # Store product_type in session for later use in advanced parameters (session-isolated)
        session[f'product_type_{search_session_id}'] = response_data["productType"]

        return jsonify(response_data), 200

    except Exception as e:
        logging.exception("Validation failed.")
        return jsonify({"error": str(e)}), 500


@app.route("/new-search", methods=["POST"])
@login_required
def api_new_search():
    """Initialize a new search session, clearing any previous state"""
    try:
        data = request.get_json(force=True) if request.is_json else {}
        search_session_id = data.get("search_session_id", "default")
        
        # Clear all session data related to this search session
        keys_to_clear = [k for k in session.keys() if search_session_id in k or k.startswith('product_type')]
        for key in keys_to_clear:
            del session[key]
        
        # Clear general workflow state for new search
        workflow_keys = ['current_step', 'current_intent', 'data']
        for key in workflow_keys:
            if key in session:
                del session[key]
        
        logging.info(f"[NEW_SEARCH] Initialized new search session: {search_session_id}")
        logging.info(f"[NEW_SEARCH] Cleared session keys: {keys_to_clear}")
        
        return jsonify({
            "success": True,
            "search_session_id": search_session_id,
            "message": "New search session initialized"
        }), 200
        
    except Exception as e:
        logging.exception("Failed to initialize new search session.")
        return jsonify({"error": str(e)}), 500


@app.route("/schema", methods=["GET"])
@login_required
def api_schema():
    if not components:
        return jsonify({"error": "Backend is not ready. LangChain failed."}), 503
    try:
        product_type = request.args.get("product_type", "").strip()
        
        if product_type:
            try:
                # Try to load from MongoDB with timeout protection
                schema_data = load_requirements_schema(product_type)
                
                # Check if schema is valid (not empty)
                if schema_data and (schema_data.get("mandatory_requirements") or schema_data.get("optional_requirements")):
                    logging.info(f"[SCHEMA] Successfully loaded schema for '{product_type}'")
                    return jsonify(convert_keys_to_camel_case(schema_data)), 200
                else:
                    logging.warning(f"[SCHEMA] Empty schema returned for '{product_type}', building from web")
                    # Fallback to web discovery if schema is empty
                    schema_data = build_requirements_schema_from_web(product_type)
                    return jsonify(convert_keys_to_camel_case(schema_data)), 200
                    
            except Exception as mongo_error:
                # MongoDB timeout or connection error - fallback to web-based schema
                logging.error(f"[SCHEMA] MongoDB error for '{product_type}': {str(mongo_error)}")
                logging.info(f"[SCHEMA] Falling back to web-based schema generation for '{product_type}'")
                
                try:
                    schema_data = build_requirements_schema_from_web(product_type)
                    return jsonify(convert_keys_to_camel_case(schema_data)), 200
                except Exception as web_error:
                    logging.error(f"[SCHEMA] Web-based schema generation also failed: {str(web_error)}")
                    # Return minimal schema to prevent complete failure
                    return jsonify({
                        "productType": product_type,
                        "mandatoryRequirements": {},
                        "optionalRequirements": {},
                        "error": f"Failed to load schema: {str(mongo_error)}"
                    }), 200  # Return 200 with error message instead of 500
        else:
            # No product type - return generic schema
            schema_data = load_requirements_schema()
            return jsonify(convert_keys_to_camel_case(schema_data)), 200
            
    except Exception as e:
        logging.exception("Schema fetch failed.")
        # Return minimal schema with error instead of failing completely
        return jsonify({
            "productType": product_type if 'product_type' in locals() else "",
            "mandatoryRequirements": {},
            "optionalRequirements": {},
            "error": str(e)
        }), 200  # Return 200 to prevent frontend from breaking

@app.route("/additional_requirements", methods=["POST"])
@login_required
def api_additional_requirements():
    if not components:
        return jsonify({"error": "Backend is not ready. LangChain failed."}), 503
    try:
        data = request.get_json(force=True)
        product_type = data.get("product_type", "").strip()
        user_input = data.get("user_input", "").strip()
        search_session_id = data.get("search_session_id", "default")

        
        if not product_type:
            return jsonify({"error": "Missing product_type"}), 400

        specific_schema = load_requirements_schema(product_type)
        
        # Add session isolation to prevent cross-contamination
        session_isolated_input = f"[Session: {search_session_id}] - This is an independent additional requirements request. User input: {user_input}"
        
        validation_result = components['additional_requirements_chain'].invoke({
            "user_input": session_isolated_input,
            "product_type": product_type,
            "schema": json.dumps(specific_schema, indent=2),
            "format_instructions": components['additional_requirements_format_instructions']
        })

        new_requirements = validation_result.get("provided_requirements", {})
        combined_reqs = new_requirements

        if combined_reqs:
            reqs_for_llm = '\n'.join([
                f"- {prettify_req(key)}: {value}" for key, value in combined_reqs.items()
            ])
            llm_chain = prompts.requirement_explanation_prompt | components['llm'] | StrOutputParser()
            explanation = llm_chain.invoke({
                "product_type": prettify_req(product_type),
                "requirements": reqs_for_llm
            })
        else:
            explanation = "I could not identify any specific requirements from your input."

        provided_requirements = new_requirements
        if new_requirements.get('mandatoryRequirements'):
            provided_requirements = {
                **new_requirements.get('mandatoryRequirements', {}),
                **new_requirements.get('optionalRequirements', {})
            }

        response_data = {
            "explanation": explanation,
            "providedRequirements": convert_keys_to_camel_case(provided_requirements),
        }


        return jsonify(response_data), 200

    except Exception as e:
        logging.exception("Additional requirements handling failed.")
        return jsonify({"error": str(e)}), 500

@app.route("/structure_requirements", methods=["POST"])
@login_required
def api_structure_requirements():
    if not components:
        return jsonify({"error": "Backend is not ready. LangChain failed."}), 503
    try:
        data = request.get_json(force=True)
        full_input = data.get("full_input", "")
        if not full_input:
            return jsonify({"error": "Missing full_input"}), 400

        structured_req = components['requirements_chain'].invoke({"user_input": full_input})
        return jsonify({"structured_requirements": structured_req}), 200
    except Exception as e:
        logging.exception("Requirement structuring failed.")
        return jsonify({"error": str(e)}), 500

@app.route("/api/advanced_parameters", methods=["POST"])
@login_required
def api_advanced_parameters():
    """
    Discovers latest advanced specifications with series numbers from top vendors for a product type
    """
    try:
        data = request.get_json(force=True)
        product_type = data.get("product_type", "").strip()
        search_session_id = data.get("search_session_id", "default")
        
        if not product_type:
            return jsonify({"error": "Missing 'product_type' parameter"}), 400

        # Store for logging (session-isolated)
        session[f'log_user_query_{search_session_id}'] = f"Latest advanced specifications for {product_type}"
        
        # Discover advanced specifications with series numbers
        logging.info(f"Starting latest advanced specifications discovery for: {product_type}")
        result = discover_advanced_parameters(product_type)
        
        # Log detailed information about filtering
        unique_count = len(result.get('unique_specifications', result.get('unique_parameters', [])))
        filtered_count = result.get('existing_specifications_filtered', result.get('existing_parameters_filtered', 0))
        total_found = unique_count + filtered_count
        
        logging.info(f"Advanced specifications discovery complete: {total_found} total specifications found, {filtered_count} filtered out (already in schema), {unique_count} new specifications returned")
        
        # Store result for logging
        session['log_system_response'] = result
        
        # Convert to camelCase for frontend
        camel_case_result = convert_keys_to_camel_case(result)
        
        logging.info(f"Advanced specifications discovery complete: {len(result.get('unique_specifications', result.get('unique_parameters', [])))} new specifications found (filtered out {result.get('existing_specifications_filtered', result.get('existing_parameters_filtered', 0))} existing specifications)")
        
        return jsonify(camel_case_result), 200

    except Exception as e:
        logging.exception("Advanced specifications discovery failed.")
        return jsonify({"error": str(e)}), 500

@app.route("/api/add_advanced_parameters", methods=["POST"])
@login_required
def api_add_advanced_parameters():
    """
    Processes user input for latest advanced specifications selection with series numbers
    """
    if not components:
        return jsonify({"error": "Backend is not ready. LangChain failed."}), 503
    try:
        data = request.get_json(force=True)
        product_type = data.get("product_type", "").strip()
        user_input = data.get("user_input", "").strip()
        available_parameters = data.get("available_parameters", [])

        if not product_type:
            return jsonify({"error": "Missing product_type"}), 400

        if not user_input:
            return jsonify({"error": "Missing user_input"}), 400

        # Use LLM to extract selected specifications from user input
        prompt = prompts.advanced_parameter_selection_prompt

        try:
            chain = prompt | components['llm'] | StrOutputParser()
            llm_response = chain.invoke({
                "product_type": product_type,
                "available_parameters": json.dumps(available_parameters),
                "user_input": user_input
            })

            # Parse the LLM response
            result = json.loads(llm_response)
            selected_parameters = result.get("selected_parameters", {})
            explanation = result.get("explanation", "Latest specifications selected successfully.")

        except (json.JSONDecodeError, Exception) as e:
            logging.warning(f"LLM parsing failed, using fallback: {e}")
            # Fallback: simple keyword matching
            selected_parameters = {}
            user_lower = user_input.lower()
            
            if "all" in user_lower or "everything" in user_lower:
                # Handle both dict format (new) and string format (old)
                for param in available_parameters:
                    param_key = param.get('key', param) if isinstance(param, dict) else param
                    selected_parameters[param_key] = ""
                explanation = "All available latest specifications have been selected."
            else:
                # Look for specification names in user input
                for param in available_parameters:
                    # Handle both dict format (new) and string format (old)
                    if isinstance(param, dict):
                        param_key = param.get('key', '')
                        param_name = param.get('name', '').lower()
                        if param_key.lower() in user_lower or param_name in user_lower:
                            selected_parameters[param_key] = ""
                    else:
                        if param.lower() in user_lower or param.replace('_', ' ').lower() in user_lower:
                            selected_parameters[param] = ""
                
                explanation = f"Selected {len(selected_parameters)} latest specifications based on your input."

        def wants_parameter_display(text: str) -> bool:
            lowered = text.lower()
            display_keywords = ["show", "display", "list", "see", "view", "what are"]
            spec_keywords = ["spec", "parameter", "latest"]
            return any(keyword in lowered for keyword in display_keywords) and any(
                key in lowered for key in spec_keywords
            )

        normalized_input = user_input.strip().lower()
        
        # Generate friendly response
        if selected_parameters:
            param_list = ", ".join([param.replace('_', ' ').title() for param in selected_parameters.keys()])
            friendly_response = f"Great! I've added these latest advanced specifications: {param_list}. Would you like to add any more advanced specifications?"
        else:
            if wants_parameter_display(normalized_input) and available_parameters:
                formatted_available = ", ".join(
                    [
                        (param.get("name") or param.get("key", "")).strip()
                        if isinstance(param, dict)
                        else str(param)
                        for param in available_parameters
                    ]
                )
                friendly_response = (
                    "Here are the latest advanced specifications you can add: "
                    f"{formatted_available}. Let me know the names you want to include or say 'no' to skip."
                )
            else:
                friendly_response = "I didn't find any matching specifications in your input. Could you please specify which latest specifications you'd like to add?"

        response_data = {
            "selectedParameters": convert_keys_to_camel_case(selected_parameters),
            "explanation": explanation,
            "friendlyResponse": friendly_response,
            "totalSelected": len(selected_parameters)
        }

        return jsonify(response_data), 200

    except Exception as e:
        logging.exception("Latest advanced specifications addition failed.")
        return jsonify({"error": str(e)}), 500

@app.route("/analyze", methods=["POST"])
@login_required
def api_analyze():
    try:
        # Check if analysis chain is initialized
        if analysis_chain is None:
            logging.error("[ANALYZE] Analysis chain not initialized")
            return jsonify({"error": "Analysis service not available. Please try again later."}), 503

        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Check if CSV vendors are provided for targeted analysis
        csv_vendors = data.get("csv_vendors", [])
        requirements = data.get("requirements", "")
        product_type = data.get("product_type", "")
        detected_product = data.get("detected_product", "")
        user_input = data.get("user_input")
        
        # Handle CSV vendor filtering (optional - can be combined with user_input)
        if csv_vendors and len(csv_vendors) > 0:
            logging.info(f"[CSV_VENDOR_FILTER] Applying CSV vendor filter with {len(csv_vendors)} vendors")
            
            # Standardize CSV vendor names for filtering
            csv_vendor_names = []
            for csv_vendor in csv_vendors:
                try:
                    original_name = csv_vendor.get("vendor_name", "")
                    standardized_name = standardize_vendor_name(original_name)
                    csv_vendor_names.append(standardized_name.lower())
                except Exception as e:
                    logging.warning(f"Failed to standardize CSV vendor {csv_vendor.get('vendor_name', '')}: {e}")
                    csv_vendor_names.append(csv_vendor.get("vendor_name", "").lower())
            
            # Store CSV filter in session for analysis chain to use
            session[f'csv_vendor_filter_{session.get("user_id", "default")}'] = {
                'vendor_names': csv_vendor_names,
                'csv_vendors': csv_vendors,
                'product_type': product_type,
                'detected_product': detected_product
            }
            
            logging.info(f"[CSV_VENDOR_FILTER] Applied filter for vendors: {csv_vendor_names}")
        
        # Now handle the analysis (user_input is REQUIRED)
        if user_input is not None:
            # user_input can be a string or dict - handle both cases
            # The analysis chain expects the raw input string for LLM processing
            if isinstance(user_input, dict):
                # If it's already a dict, extract the raw input or convert to string
                if "raw_input" in user_input:
                    user_input_str = user_input["raw_input"]
                else:
                    # Convert dict to a formatted string for the LLM
                    user_input_str = json.dumps(user_input, indent=2)
            elif isinstance(user_input, str):
                # Try to parse as JSON first (might be a JSON string)
                try:
                    parsed = json.loads(user_input)
                    if isinstance(parsed, dict):
                        if "raw_input" in parsed:
                            user_input_str = parsed["raw_input"]
                        else:
                            user_input_str = json.dumps(parsed, indent=2)
                    else:
                        user_input_str = user_input
                except json.JSONDecodeError:
                    # It's a plain string - use as is
                    user_input_str = user_input
            else:
                return jsonify({"error": "user_input must be a string or dict"}), 400

            # Pass user_input as string to the analysis chain
            logging.info(f"[ANALYZE] Processing input: {user_input_str[:200]}...")
            analysis_result = analysis_chain({"user_input": user_input_str})
        
        else:
            return jsonify({"error": "Missing 'user_input' parameter or CSV vendor data"}), 400
        
        # Apply standardization to the analysis result
        try:
            # Standardize vendor analysis if it exists
            if "vendor_analysis" in analysis_result:
                analysis_result["vendor_analysis"] = standardize_vendor_analysis_result(analysis_result["vendor_analysis"])
            
            # Standardize overall ranking if it exists
            if "overall_ranking" in analysis_result:
                analysis_result["overall_ranking"] = standardize_ranking_result(analysis_result["overall_ranking"])
                
            logging.info("Applied standardization to analysis result")
        except Exception as e:
            logging.warning(f"Standardization failed, proceeding with original result: {e}")

        camel_case_result = convert_keys_to_camel_case(analysis_result)

        # Store the analysis result as system response for logging
        session['log_system_response'] = analysis_result

        return jsonify(camel_case_result)

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logging.error(f"Analysis failed: {str(e)}")
        logging.error(f"Traceback:\n{error_traceback}")
        return jsonify({
            "error": str(e),
            "details": error_traceback.split('\n')[-3] if error_traceback else None
        }), 500



def match_user_with_pdf(user_input, pdf_data):
    """
    Matches user input fields with PDF data.
    Accepts user_input as a dict or JSON string.
    """
    # Ensure user_input is a dict
    if isinstance(user_input, str):
        try:
            user_input = json.loads(user_input)
        except json.JSONDecodeError:
            logging.warning("user_input is a string that cannot be parsed; wrapping in dict.")
            user_input = {"raw_input": user_input}

    if not isinstance(user_input, dict):
        raise ValueError("user_input must be a dict after parsing.")

    matched_results = {}
    for field, requirement in user_input.items():
        # Example matching logic; replace with your actual logic
        matched_results[field] = pdf_data.get(field, None)

    return matched_results

@app.route("/get_field_description", methods=["POST"])
@login_required
def api_get_field_description():
    """
    Get field description and value from standards documents using Deep Agent.
    
    This endpoint uses the Deep Agent's field extraction functionality to:
    1. Query relevant standards documents for the field
    2. Extract specification value and description
    3. Return standards-based information for UI display
    
    Request Body:
        {
            "field": "Compliance.hazardousAreaRating.value",
            "product_type": "Multi-Point Thermocouple"
        }
    
    Response:
        {
            "success": true,
            "description": "ATEX II 1/2 G Ex ia/d per IEC 60079",
            "field": "hazardousAreaRating",
            "product_type": "Multi-Point Thermocouple",
            "source": "instrumentation_safety_standards.docx",
            "standards_referenced": ["IEC 60079", "ATEX"],
            "confidence": 0.85
        }
    """
    try:
        data = request.get_json(force=True)
        field_path = data.get("field", "").strip()
        product_type = data.get("product_type", "general").strip()

        if not field_path:
            return jsonify({"error": "Missing 'field' parameter.", "success": False}), 400

        # Parse field path (e.g., "Compliance.hazardousAreaRating.value" -> "hazardousAreaRating")
        field_parts = field_path.split(".")
        field_name = field_parts[-2] if len(field_parts) >= 2 and field_parts[-1] in ["value", "source", "confidence", "standards_referenced"] else field_parts[-1]
        section_name = field_parts[0] if len(field_parts) > 1 else "Other"
        
        logging.info(f"[FieldDescription] Extracting value for field: {field_name}, product: {product_type}, section: {section_name}")
        
        # =====================================================
        # STRATEGY 1: Direct Default Lookup (FAST PATH)
        # Uses comprehensive standards specifications from schema_field_extractor
        # =====================================================
        try:
            from agentic.deep_agent.schema_field_extractor import get_default_value_for_field
            
            default_value = get_default_value_for_field(product_type, field_name)
            
            if default_value:
                logging.info(f"[FieldDescription] ✓ Found default value for {field_name}: {default_value[:50]}...")
                return jsonify({
                    "success": True,
                    "description": default_value,
                    "field": field_name,
                    "field_path": field_path,
                    "section": section_name,
                    "product_type": product_type,
                    "source": "standards_specifications",
                    "standards_referenced": [],  # Could be enhanced to include IEC/ISO codes
                    "confidence": 0.9
                }), 200
            else:
                logging.debug(f"[FieldDescription] No default value found for {field_name}")
                
        except Exception as default_err:
            logging.warning(f"[FieldDescription] Default lookup failed: {default_err}")
        
        # =====================================================
        # STRATEGY 2: LLM-based Description (Fallback)
        # Used when no predefined default exists
        # =====================================================
        try:
            from llm_fallback import create_llm_with_fallback
            import os
            
            llm = create_llm_with_fallback(
                model="gemini-2.5-flash",
                temperature=0.3,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            
            # Create a focused prompt for technical specifications
            prompt = f"""Provide the typical technical specification value for "{field_name}" 
for a {product_type} in industrial instrumentation.

Guidelines:
- Provide a specific value or range (e.g., "±0.1%", "4-20mA", "IP66")
- Include relevant standards references (IEC, ISO, NAMUR, etc.) if applicable
- Keep response under 50 words
- Focus on the specification VALUE, not a description

Example format: "±0.75% or ±2.5°C (whichever is greater) per IEC 60584"
"""
            
            response = llm.invoke(prompt)
            description = response.content.strip()
            
            if description and description.lower() not in ["not specified", "n/a", "unknown", ""]:
                logging.info(f"[FieldDescription] ✓ LLM generated value for {field_name}")
                return jsonify({
                    "success": True,
                    "description": description,
                    "field": field_name,
                    "field_path": field_path,
                    "section": section_name,
                    "product_type": product_type,
                    "source": "llm_inference",
                    "standards_referenced": [],
                    "confidence": 0.7
                }), 200
                
        except Exception as llm_err:
            logging.warning(f"[FieldDescription] LLM fallback failed: {llm_err}")
        
        # =====================================================
        # STRATEGY 3: Generic Fallback
        # Last resort when all else fails
        # =====================================================
        field_display = field_name.replace("_", " ").replace("-", " ")
        # Convert camelCase to Title Case
        import re
        field_display = re.sub('([a-z])([A-Z])', r'\1 \2', field_display).title()
        
        return jsonify({
            "success": True,
            "description": f"Specification for {field_display}",
            "field": field_name,
            "field_path": field_path,
            "section": section_name,
            "product_type": product_type,
            "source": "generic",
            "standards_referenced": [],
            "confidence": 0.3
        }), 200

    except Exception as e:
        logging.exception("Failed to get field description.")
        return jsonify({
            "success": False,
            "error": "Failed to get field description: " + str(e),
            "description": ""
        }), 500

@app.route("/get_all_field_descriptions", methods=["POST"])
@login_required
def api_get_all_field_descriptions():
    """
    Get ALL field descriptions and values for a schema at once (BATCH API).
    
    This avoids multiple individual API calls by fetching all values in one request.
    
    Request Body:
        {
            "product_type": "Thermocouple",
            "fields": ["Performance.accuracy", "Electrical.outputSignal", ...]
        }
    
    Response:
        {
            "success": true,
            "product_type": "Thermocouple",
            "field_values": {
                "Performance.accuracy": {"value": "±0.75%...", "source": "standards"},
                "Electrical.outputSignal": {"value": "4-20mA...", "source": "standards"},
                ...
            },
            "total_fields": 30,
            "fields_populated": 28
        }
    """
    try:
        data = request.get_json(force=True)
        product_type = data.get("product_type", "general").strip()
        fields = data.get("fields", [])
        
        if not fields:
            return jsonify({"error": "Missing 'fields' parameter.", "success": False}), 400
        
        logging.info(f"[BatchFieldDescription] Processing {len(fields)} fields for {product_type}")
        
        # Import template specifications for actual descriptions
        template_descriptions = {}
        try:
            from agentic.deep_agent.specification_templates import get_all_specs_for_product_type
            template_specs = get_all_specs_for_product_type(product_type)
            if template_specs:
                for spec_key, spec_def in template_specs.items():
                    template_descriptions[spec_key] = spec_def.description
                logging.info(f"[BatchFieldDescription] Loaded {len(template_descriptions)} template descriptions")
        except ImportError as e:
            logging.warning(f"[BatchFieldDescription] Could not import templates: {e}")
        
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
        
        field_values = {}
        fields_populated = 0
        
        for field_path in fields:
            # Parse field path
            field_parts = field_path.split(".")
            field_name = field_parts[-2] if len(field_parts) >= 2 and field_parts[-1] in ["value", "source", "confidence", "standards_referenced"] else field_parts[-1]
            
            # Try to get description from templates first
            description = None
            source = "not_found"
            
            # Check template descriptions by field name
            if field_name in template_descriptions:
                description = template_descriptions[field_name]
                source = "template_specifications"
                fields_populated += 1
            elif field_path in template_descriptions:
                description = template_descriptions[field_path]
                source = "template_specifications"
                fields_populated += 1
            else:
                # Generate a human-readable description from the field name
                description = prettify_field_name(field_name)
                source = "generated"
                fields_populated += 1
            
            field_values[field_path] = {
                "value": description or "",
                "source": source,
                "field_name": field_name,
                "confidence": 0.9 if source == "template_specifications" else 0.5,
                "standards_referenced": []
            }
        
        logging.info(f"[BatchFieldDescription] Completed: {fields_populated}/{len(fields)} fields populated")
        
        return jsonify({
            "success": True,
            "product_type": product_type,
            "field_values": field_values,
            "total_fields": len(fields),
            "fields_populated": fields_populated
        }), 200
        
    except Exception as e:
        logging.exception("Failed to get batch field descriptions.")
        return jsonify({
            "success": False,
            "error": "Failed to get batch field descriptions: " + str(e),
            "field_values": {}
        }), 500

def get_submodel_to_model_series_mapping():
    """
    Creates a mapping from submodel names to their parent model series
    by scanning all vendor JSON files.
    """
    """Load submodel mapping from MongoDB instead of local files"""
    submodel_to_series = {}
    
    try:
        # Query MongoDB for all product data
        products_collection = mongodb_file_manager.collections.get('products')
        
        if not products_collection:
            logging.warning("Products collection not found in MongoDB")
            return submodel_to_series
        
        # Get all products from MongoDB
        cursor = products_collection.find({})
        
        for doc in cursor:
            try:
                # Extract product data
                if 'data' in doc:
                    data = doc['data']
                else:
                    data = {k: v for k, v in doc.items() if k not in ['_id', 'metadata']}
                
                # Process models and submodels
                models = data.get('models', [])
                for model in models:
                    model_series = model.get('model_series', '')
                    submodels = model.get('sub_models', [])
                    
                    for submodel in submodels:
                        submodel_name = submodel.get('name', '')
                        if submodel_name and model_series:
                            submodel_to_series[submodel_name] = model_series
                            
            except Exception as e:
                logging.warning(f"Failed to process MongoDB document: {e}")
                continue
                
    except Exception as e:
        logging.error(f"Failed to load submodel mapping from MongoDB: {e}")
        return submodel_to_series
                        
    logging.info(f"Generated submodel mapping with {len(submodel_to_series)} entries")
    return submodel_to_series

@app.route("/vendors", methods=["GET"])
@login_required
def get_vendors():
    """
    Get vendors with product images - ONLY for vendors in analysis results.
    Optimized to avoid unnecessary API calls.
    """
    try:
        # Get vendor list from query parameter (sent by frontend with analysis results)
        vendors_param = request.args.get('vendors', '')
        
        if vendors_param:
            # Use vendors from analysis results
            vendor_list = [v.strip() for v in vendors_param.split(',') if v.strip()]
            logging.info(f"Fetching images for {len(vendor_list)} vendors from analysis results: {vendor_list}")
        else:
            # Fallback: return empty list if no vendors specified
            logging.warning("No vendors specified in request, returning empty list")
            return jsonify({
                "vendors": [],
                "summary": {
                    "total_vendors": 0,
                    "total_images": 0,
                    "sources_used": {}
                }
            }), 200
        
        vendors = []
        
        def process_vendor(vendor_name):
            """Process a single vendor synchronously for better reliability"""
            try:
                # Fetch product images using the 3-level fallback system (sync version)
                images, source_used = fetch_product_images_with_fallback_sync(vendor_name)
                
                # Convert to expected format
                formatted_images = []
                for img in images:
                    # Create a normalized product key for frontend matching
                    title = img.get("title", "")
                    norm_key = re.sub(r"[\s_]+", "", title).replace("+", "").lower()
                    
                    formatted_images.append({
                        "fileName": title,
                        "url": img.get("url", ""),
                        "productKey": norm_key,
                        "thumbnail": img.get("thumbnail", ""),
                        "source": img.get("source", source_used),
                        "domain": img.get("domain", "")
                    })
                
                # Try to get logo from the first image or a specific logo search
                logo_url = None
                if formatted_images:
                    # Use first image as logo or search specifically for logo
                    logo_url = formatted_images[0].get("thumbnail") or formatted_images[0].get("url")
                
                vendor_data = {
                    "name": vendor_name,
                    "logoUrl": logo_url,
                    "images": formatted_images,
                    "source_used": source_used,
                    "image_count": len(formatted_images)
                }
                
                # Apply basic standardization to vendor data
                try:
                    vendor_data["name"] = standardize_vendor_name(vendor_data["name"])
                except Exception as e:
                    logging.warning(f"Failed to standardize vendor name {vendor_name}: {e}")
                    # Keep original name if standardization fails
                    vendor_data["name"] = vendor_name
                
                logging.info(f"Processed vendor {vendor_name}: {len(formatted_images)} images from {source_used}")
                return vendor_data
                
            except Exception as e:
                logging.warning(f"Failed to process vendor {vendor_name}: {e}")
                # Return minimal vendor data on failure
                return {
                    "name": vendor_name,
                    "logoUrl": None,
                    "images": [],
                    "source_used": "error",
                    "image_count": 0,
                    "error": str(e)
                }
        
        # Process only the vendors from analysis results
        for vendor_name in vendor_list:
            vendor_data = process_vendor(vendor_name)
            if vendor_data:
                vendors.append(vendor_data)
        
        # Filter out any None results and add summary info
        vendors = [v for v in vendors if v is not None]
        
        # Add summary statistics
        total_images = sum(v.get("image_count", 0) for v in vendors)
        sources_used = {}
        for v in vendors:
            source = v.get("source_used", "unknown")
            sources_used[source] = sources_used.get(source, 0) + 1
        
        response_data = {
            "vendors": vendors,
            "summary": {
                "total_vendors": len(vendors),
                "total_images": total_images,
                "sources_used": sources_used
            }
        }
        
        logging.info(f"Successfully processed {len(vendors)} vendors with {total_images} total images")
        return jsonify(response_data)
        
    except Exception as e:
        logging.error(f"Critical error in get_vendors: {e}")
        return jsonify({
            "error": "Failed to fetch vendors",
            "vendors": [],
            "summary": {
                "total_vendors": 0,
                "total_images": 0,
                "sources_used": {}
            }
        }), 500

@app.route("/submodel-mapping", methods=["GET"])
@login_required
def get_submodel_mapping():
    """
    Returns the mapping from submodel names to model series names.
    This helps the frontend map analysis results (submodel names) to images (model series names).
    """
    try:
        mapping = get_submodel_to_model_series_mapping()
        
        # Skip LLM-based standardization for this endpoint to prevent connection issues
        # Basic mapping is sufficient for frontend functionality
        logging.info(f"Retrieved {len(mapping)} submodel mappings")
        
        return jsonify({"mapping": mapping})
    except Exception as e:
        logging.error(f"Error getting submodel mapping: {e}")
        return jsonify({"error": "Failed to get submodel mapping", "mapping": {}}), 500

@app.route("/admin/approve_user", methods=["POST"])
@login_required
def approve_user():
    admin_user = db.session.get(User, session['user_id'])
    if admin_user.role != "admin":
        return jsonify({"error": "Forbidden: Admins only"}), 403

    data = request.get_json()
    user_id = data.get("user_id")
    action = data.get("action", "approve")
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404
    if action == "approve":
        user.status = "active"
    elif action == "reject":
        user.status = "rejected"
    else:
        return jsonify({"error": "Invalid action"}), 400

    db.session.commit()
    return jsonify({"message": f"User {user.username} status updated to {user.status}."}), 200

@app.route("/admin/pending_users", methods=["GET"])
@login_required
def pending_users():
    admin_user = db.session.get(User, session['user_id'])
    if admin_user.role != "admin":
        return jsonify({"error": "Forbidden: Admins only"}), 403

    pending = User.query.filter_by(status="pending").all()
    result = [{
        "id": u.id,
        "username": u.username,
        "email": u.email,
        "first_name": u.first_name,
        "last_name": u.last_name
    } for u in pending]
    return jsonify({"pending_users": result}), 200

# Duplicated ALLOWED_EXTENSIONS and allowed_file, can be removed.
# ALLOWED_EXTENSIONS = {"pdf"}
# def allowed_file(filename: str):
#    ...

@app.route("/api/get-price-review", methods=["GET"])
@login_required
def api_get_price_review():
    product_name = request.args.get("productName")
    if not product_name:
        return jsonify({"error": "Missing productName parameter"}), 400

    results = fetch_price_and_reviews(product_name)

    return jsonify(results), 200

@app.route("/upload", methods=["POST"])
@login_required
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request. Expected field name 'file'."}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only PDF files are allowed."}), 400

    filename = secure_filename(file.filename)

    try:
        # Read file into a BytesIO stream so it can be reused
        file_stream = BytesIO(file.read())

        # Extract text chunks from PDF
        text_chunks = extract_data_from_pdf(file_stream)
        raw_results = send_to_language_model(text_chunks)
        
        def flatten_results(results):
            flat = []
            for r in results:
                if isinstance(r, list): flat.extend(r)
                else: flat.append(r)
            return flat

        all_results = flatten_results(raw_results)
        final_result = aggregate_results(all_results, filename)
        
        # Apply standardization to the final result before splitting
        try:
            standardized_final_result = standardize_vendor_analysis_result(final_result)
            logging.info("Applied standardization to uploaded file analysis")
        except Exception as e:
            logging.warning(f"Failed to standardize uploaded file result: {e}")
            standardized_final_result = final_result
        
        split_results = split_product_types([standardized_final_result])

        saved_paths = []
        for result in split_results:
            # Save to MongoDB instead of local files
            vendor = (result.get("vendor") or "UnknownVendor").replace(" ", "_")
            product_type = (result.get("product_type") or "UnknownProduct").replace(" ", "_")
            model_series = (
                result.get("models", [{}])[0].get("model_series") or "UnknownModel"
            ).replace(" ", " ")
            
            try:
                product_metadata = {
                    'vendor_name': vendor,
                    'product_type': product_type,
                    'model_series': model_series,
                    'file_type': 'json',
                    'collection_type': 'products',
                    'path': f'vendors/{vendor}/{product_type}/{model_series}.json'
                }
                mongodb_file_manager.upload_json_data(result, product_metadata)
                saved_paths.append(f"MongoDB:vendors/{vendor}/{product_type}/{model_series}.json")
                print(f"[INFO] Stored product JSON to MongoDB: {vendor} - {product_type}")
            except Exception as e:
                logging.error(f"Failed to save product JSON to MongoDB: {e}")
            
            # Note: Product image extraction removed - now using API-based image search

        return jsonify({
            "data": split_results,
            "savedFiles": saved_paths
        })
    except Exception as e:
        logging.exception("File upload processing failed.")
        return jsonify({"error": str(e)}), 500

# =========================================================================
# === STANDARDIZATION ENDPOINTS ===
# === 
# === Integrated standardization functionality:
# === - /analyze endpoint: Standardizes vendor analysis and ranking results
# === - /vendors endpoint: Standardizes vendor names and product image mappings 
# === - /submodel-mapping endpoint: Enhances submodel mappings with standardization
# === - /upload endpoint: Standardizes analysis results from PDF uploads
# === - /api/upload_pdf_from_url endpoint: Standardizes analysis results from URL uploads
# === 
# === New standardization endpoints:
# === - GET /standardization/report: Generate comprehensive standardization report
# === - POST /standardization/update-files: Update existing files with standardization (admin only)
# === - POST /standardization/vendor-analysis: Standardize vendor analysis data
# === - POST /standardization/ranking: Standardize ranking data  
# === - POST /standardization/submodel-mapping: Enhance submodel mapping data
# =========================================================================

@app.route("/standardization/report", methods=["GET"])
@login_required
def get_standardization_report():
    """
    Generate and return a comprehensive standardization report
    """
    try:
        report = create_standardization_report()
        return jsonify(report), 200
    except Exception as e:
        logging.error(f"Failed to generate standardization report: {e}")
        return jsonify({"error": "Failed to generate standardization report"}), 500

@app.route("/standardization/update-files", methods=["POST"])
@login_required
def update_files_with_standardization():
    """
    Update existing vendor files with standardized naming
    """
    try:
        admin_user = db.session.get(User, session['user_id'])
        if admin_user.role != "admin":
            return jsonify({"error": "Forbidden: Admins only"}), 403
            
        updated_files = update_existing_vendor_files_with_standardization()
        return jsonify({
            "message": f"Successfully updated {len(updated_files)} files with standardization",
            "updated_files": updated_files
        }), 200
    except Exception as e:
        logging.error(f"Failed to update files with standardization: {e}")
        return jsonify({"error": "Failed to update files with standardization"}), 500

@app.route("/standardization/vendor-analysis", methods=["POST"])
@login_required
def standardize_vendor_analysis():
    """
    Standardize a vendor analysis result
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        analysis_result = data.get("analysis_result")
        if not analysis_result:
            return jsonify({"error": "Missing analysis_result parameter"}), 400
            
        standardized_result = standardize_vendor_analysis_result(analysis_result)
        return jsonify(standardized_result), 200
    except Exception as e:
        logging.error(f"Failed to standardize vendor analysis: {e}")
        return jsonify({"error": "Failed to standardize vendor analysis"}), 500

@app.route("/standardization/ranking", methods=["POST"])
@login_required
def standardize_ranking():
    """
    Standardize a ranking result
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        ranking_result = data.get("ranking_result")
        if not ranking_result:
            return jsonify({"error": "Missing ranking_result parameter"}), 400
            
        standardized_result = standardize_ranking_result(ranking_result)
        return jsonify(standardized_result), 200
    except Exception as e:
        logging.error(f"Failed to standardize ranking: {e}")
        return jsonify({"error": "Failed to standardize ranking"}), 500

@app.route("/standardization/submodel-mapping", methods=["POST"])
@login_required
def enhance_submodel_mapping_endpoint():
    """
    Enhance submodel to model series mapping with standardization
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        submodel_data = data.get("submodel_data")
        if not submodel_data:
            return jsonify({"error": "Missing submodel_data parameter"}), 400
            
        enhanced_result = enhance_submodel_mapping(submodel_data)
        return jsonify(enhanced_result), 200
    except Exception as e:
        logging.error(f"Failed to enhance submodel mapping: {e}")
        return jsonify({"error": "Failed to enhance submodel mapping"}), 500
    

# =========================================================================
# === PROJECT MANAGEMENT ENDPOINTS ===
# =========================================================================

@app.route("/api/projects/save", methods=["POST"])
@login_required
def save_project():
    """
    Save or update a project with all current state data using MongoDB
    """
    try:
        data = request.get_json(force=True)
        # Debug log incoming product_type information to trace saving issues (project save)
        try:
            incoming_pt = data.get('product_type') if isinstance(data, dict) else None
            incoming_detected = data.get('detected_product_type') if isinstance(data, dict) else None
            logging.info(f"[SAVE_PROJECT] Incoming product_type='{incoming_pt}' detected_product_type='{incoming_detected}' project_name='{data.get('project_name') if isinstance(data, dict) else None}' user_id={session.get('user_id')}")
        except Exception:
            logging.exception("Failed to log incoming project save payload")
        
        # Get current user ID
        user_id = str(session['user_id'])
        
        # Extract project data
        project_id = data.get("project_id")  # If updating existing project
        project_name = data.get("project_name", "").strip()
        
        if not project_name:
            return jsonify({"error": "Project name is required"}), 400
        
        # Check if initial_requirements is provided
        # Allow saving if project has instruments/accessories (already analyzed) even without requirements text
        has_requirements = bool(data.get("initial_requirements", "").strip())
        has_instruments = bool(data.get("identified_instruments") and len(data.get("identified_instruments", [])) > 0)
        has_accessories = bool(data.get("identified_accessories") and len(data.get("identified_accessories", [])) > 0)
        
        if not has_requirements and not has_instruments and not has_accessories:
            return jsonify({"error": "Initial requirements are required"}), 400
        
        # Save project to MongoDB using project manager
        # If the frontend provided a displayed_media_map, persist those images into GridFS
        try:
            displayed_media = data.get('displayed_media_map', {}) if isinstance(data, dict) else {}
            if displayed_media:
                from mongodb_utils import upload_to_mongodb
                # For each displayed media entry, fetch the URL and store bytes in GridFS
                for key, entry in displayed_media.items():
                    try:
                        top = entry.get('top_image') if isinstance(entry, dict) else None
                        vlogo = entry.get('vendor_logo') if isinstance(entry, dict) else None

                        def process_media(obj, subtype):
                            if not obj:
                                return None
                            url = obj.get('url') if isinstance(obj, dict) else (obj if isinstance(obj, str) else None)
                            if not url:
                                return None
                            # If url already references our API, skip re-upload
                            if url.startswith('/api/projects/file/'):
                                return url
                            # If it's a data URL, decode
                            if url.startswith('data:'):
                                import base64, re
                                m = re.match(r'data:(.*?);base64,(.*)', url)
                                if m:
                                    content_type = m.group(1)
                                    b = base64.b64decode(m.group(2))
                                    metadata = {'collection_type': 'documents', 'original_url': '', 'content_type': content_type}
                                    fid = upload_to_mongodb(b, metadata)
                                    return f"/api/projects/file/{fid}"
                                return None
                            # Otherwise attempt to download the URL
                            try:
                                resp = requests.get(url, timeout=8)
                                resp.raise_for_status()
                                content_type = resp.headers.get('Content-Type', 'application/octet-stream')
                                b = resp.content
                                metadata = {'collection_type': 'documents', 'original_url': url, 'content_type': content_type}
                                fid = upload_to_mongodb(b, metadata)
                                return f"/api/projects/file/{fid}"
                            except Exception as e:
                                logging.warning(f"Failed to fetch/displayed media URL {url}: {e}")
                                return None

                        new_top = process_media(top, 'top_image')
                        new_logo = process_media(vlogo, 'vendor_logo')

                        # Inject back into data so that stored project contains references to GridFS-served URLs
                        if new_top or new_logo:
                            # attempt to find product entries in data and replace matching keys
                            # The frontend sends a map keyed by `${vendor}-${productName}`; we'll store this map as `embedded_media`
                            if 'embedded_media' not in data:
                                data['embedded_media'] = {}
                            data['embedded_media'][key] = {}
                            if new_top:
                                data['embedded_media'][key]['top_image'] = {'url': new_top}
                            if new_logo:
                                data['embedded_media'][key]['vendor_logo'] = {'url': new_logo}
                    except Exception as e:
                        logging.warning(f"Error processing displayed_media_map entry {key}: {e}")
        except Exception as e:
            logging.warning(f"Failed to persist displayed_media_map: {e}")

        # Ensure pricing and feedback are passed through from frontend payload
        # If frontend uses `pricing` or `feedback_entries` include them in the saved document
        try:
            # If frontend supplied feedback, normalize to `feedback_entries`
            if 'feedback' in data and 'feedback_entries' not in data:
                data['feedback_entries'] = data.get('feedback')
        except Exception:
            logging.warning('Failed to normalize incoming feedback payload')

        saved_project = mongo_project_manager.save_project(user_id, data)

        # Store the saved project id in the session so future feedback posts can attach to it
        try:
            session['current_project_id'] = saved_project.get('project_id')
        except Exception:
            logging.warning('Failed to set current_project_id in session')
        
        # Return the saved project data
        return jsonify({
            "message": "Project saved successfully",
            "project": saved_project
        }), 200
        
    except ValueError as e:
        logging.warning(f"Project save validation error: {e}")
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logging.exception("Project save failed.")
        return jsonify({"error": "Failed to save project: " + str(e)}), 500


@app.route("/api/projects/preview-save", methods=["POST"])
@login_required
def preview_save_project():
    """
    Debug helper: compute resolved product_type (prefers detected_product_type)
    and return it without saving. Useful for quick verification.
    """
    try:
        data = request.get_json(force=True)
        project_name = (data.get('project_name') or '').strip()
        detected = data.get('detected_product_type')
        incoming = (data.get('product_type') or '').strip()

        if detected:
            resolved = detected.strip()
        else:
            if incoming and project_name and incoming.lower() == project_name.lower():
                resolved = ''
            else:
                resolved = incoming

        return jsonify({
            'resolved_product_type': resolved,
            'detected_product_type': detected,
            'incoming_product_type': incoming,
            'project_name': project_name
        }), 200
    except Exception as e:
        logging.exception('Preview save failed')
        return jsonify({'error': str(e)}), 500

@app.route("/api/projects", methods=["GET"])
@login_required
def get_user_projects():
    """
    Get all projects for the current user from MongoDB
    """
    try:
        user_id = str(session['user_id'])
        
        # Get all active projects for the user from MongoDB
        projects = mongo_project_manager.get_user_projects(user_id)
        
        return standardized_jsonify({
            "projects": projects,
            "total_count": len(projects)
        }, 200)
        
    except Exception as e:
        logging.exception("Failed to retrieve user projects.")
        return jsonify({"error": "Failed to retrieve projects: " + str(e)}), 500

@app.route("/api/projects/<project_id>", methods=["GET"])
@login_required
def get_project_details(project_id):
    """
    Get full project details for loading from MongoDB
    """
    try:
        user_id = str(session['user_id'])
        
        # Get project details from MongoDB
        project_details = mongo_project_manager.get_project_details(project_id, user_id)
        
        return standardized_jsonify({"project": project_details}, 200)
        
    except ValueError as e:
        logging.warning(f"Project access denied: {e}")
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logging.exception(f"Failed to retrieve project {project_id}.")
        return jsonify({"error": "Failed to retrieve project: " + str(e)}), 500


@app.route('/api/projects/file/<file_id>', methods=['GET'])
@login_required
def serve_project_file(file_id):
    """
    Serve a file stored in GridFS by its file ID.
    Returns raw bytes with appropriate content-type if available.
    """
    try:
        from bson import ObjectId
        from mongodb_utils import mongodb_file_manager

        try:
            oid = ObjectId(file_id)
        except Exception:
            return jsonify({'error': 'Invalid file id'}), 400

        grid_file = mongodb_file_manager.gridfs.get(oid)
        data = grid_file.read()
        content_type = grid_file.content_type if hasattr(grid_file, 'content_type') else grid_file.metadata.get('content_type') if getattr(grid_file, 'metadata', None) else 'application/octet-stream'

        return (data, 200, {
            'Content-Type': content_type,
            'Content-Length': str(len(data)),
            'Cache-Control': 'public, max-age=31536000'
        })
    except Exception as e:
        logging.exception('Failed to serve project file')
        return jsonify({'error': str(e)}), 500

@app.route("/api/projects/<project_id>", methods=["DELETE"])
@login_required
def delete_project(project_id):
    """
    Permanently delete a project from MongoDB
    """
    try:
        user_id = str(session['user_id'])
        
        # Delete project from MongoDB
        mongo_project_manager.delete_project(project_id, user_id)
        
        return standardized_jsonify({"message": "Project deleted successfully"}, 200)
        
    except ValueError as e:
        logging.warning(f"Project delete access denied: {e}")
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logging.exception(f"Failed to delete project {project_id}.")
        return jsonify({"error": "Failed to delete project: " + str(e)}), 500


def create_db():
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(role='admin').first():
            hashed_pw = hash_password("Daman@123")
            admin = User(
                username="Daman", 
                email="reddydaman04@gmail.com", 
                password_hash=hashed_pw, 
                first_name="Daman",
                last_name="Reddy",
                status='active', 
                role='admin'
            )
            db.session.add(admin)
            db.session.commit()
            print("Admin user created with username 'Daman' and password 'Daman@123'.")
if __name__ == "__main__":
    create_db()
    import os

    # Initialize automatic checkpoint cleanup (Phase 1 improvement)
    # Prevents unbounded memory growth from checkpoint accumulation
    try:
        from agentic.checkpointing import CheckpointManager, start_auto_checkpoint_cleanup

        # Create checkpoint manager and start automatic cleanup
        checkpoint_manager = CheckpointManager("sqlite")
        start_auto_checkpoint_cleanup(
            checkpoint_manager,
            cleanup_interval=300,  # Run cleanup every 5 minutes
            ttl=72  # Remove checkpoints older than 72 hours
        )
        logging.info("Automatic checkpoint cleanup initialized successfully")
    except Exception as e:
        logging.warning(f"Failed to initialize checkpoint cleanup: {e}")

    # Initialize automatic session cleanup (Phase 2 improvement)
    # Prevents memory leaks from accumulated session files
    session_cleanup_manager = None
    try:
        from agentic.session_cleanup_manager import SessionCleanupManager

        session_dir = app.config.get("SESSION_FILE_DIR", "/tmp/flask_session")
        session_cleanup_manager = SessionCleanupManager(
            session_dir=session_dir,
            cleanup_interval=600,  # Run cleanup every 10 minutes
            max_age_hours=24  # Remove sessions older than 24 hours
        )
        session_cleanup_manager.start()
        logging.info("Automatic session cleanup initialized successfully with proper lifecycle")
    except Exception as e:
        logging.warning(f"Failed to initialize session cleanup: {e}")

    # Initialize bounded workflow state management (Phase 4 improvement)
    # Prevents OOM crashes from unbounded state accumulation
    try:
        from agentic.workflow_state_manager import stop_workflow_state_manager

        logging.info("Workflow state manager initialized with bounded memory and auto-cleanup")
    except Exception as e:
        logging.warning(f"Failed to initialize workflow state manager: {e}")

    # Register shutdown handler for graceful cleanup (Phase 3-4 improvements)
    def shutdown_cleanup():
        """Graceful shutdown handler for all background managers."""
        # Stop session cleanup (Phase 3)
        if session_cleanup_manager:
            try:
                logging.info("[SHUTDOWN] Stopping session cleanup manager...")
                session_cleanup_manager.stop()
                logging.info("[SHUTDOWN] Session cleanup manager stopped successfully")
            except Exception as e:
                logging.error(f"[SHUTDOWN] Error stopping session cleanup: {e}")

        # Stop workflow state manager (Phase 4)
        try:
            logging.info("[SHUTDOWN] Stopping workflow state manager...")
            stop_workflow_state_manager()
            logging.info("[SHUTDOWN] Workflow state manager stopped successfully")
        except Exception as e:
            logging.error(f"[SHUTDOWN] Error stopping workflow state manager: {e}")

    app.teardown_appcontext(lambda exc: shutdown_cleanup() if exc is None else None)

    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True, use_reloader=False)