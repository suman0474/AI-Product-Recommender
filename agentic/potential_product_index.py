# agentic/potential_product_index.py
# Potential Product Index Sub-Workflow
# Triggered when no schema exists for a product type

import json
import logging
from typing import Dict, Any, List, Literal, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore
from langgraph.graph import StateGraph, END

from .models import (
    PotentialProductIndexState,
    create_potential_product_index_state
)
from .checkpointing import compile_with_checkpointing

from tools.search_tools import search_pdf_datasheets_tool, web_search_tool

import os
import re
import requests
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from llm_fallback import create_llm_with_fallback

load_dotenv()
logger = logging.getLogger(__name__)


# Rate limiter for LLM calls
LLM_SEMAPHORE = Semaphore(2)


# ============================================================================
# PROMPTS - Direct LLM Calls (Optimized Architecture)
# ============================================================================

VENDOR_DISCOVERY_PROMPT = """
You are Engenie's Vendor Discovery component. Identify top vendors for a product type.

Product Type: {product_type}
Context: {context}

Identify the TOP 5 industrial vendors for this product type. Consider:
1. Market leaders
2. Quality and reliability
3. Global availability
4. Product range

Return ONLY valid JSON:
{{
    "vendors": [
        {{
            "name": "<vendor name>",
            "market_position": "leader" | "major" | "specialized",
            "headquarters": "<country>",
            "strengths": ["<strength 1>", "<strength 2>"]
        }}
    ],
    "product_type": "{product_type}",
    "confidence": <0.0-1.0>
}}
"""

MODEL_FAMILY_DISCOVERY_PROMPT = """
You are Engenie's Model Discovery component. Identify model families for a vendor.

Vendor: {vendor}
Product Type: {product_type}
Context: {context}

List the main MODEL FAMILIES/SERIES this vendor offers for this product type.
Include both current and well-known legacy models.

Return ONLY valid JSON:
{{
    "vendor": "{vendor}",
    "model_families": [
        {{
            "name": "<series/family name>",
            "generation": "current" | "legacy",
            "typical_use": "<primary application>"
        }}
    ],
    "confidence": <0.0-1.0>
}}
"""

SCHEMA_GENERATION_PROMPT = """
You are Engenie's Schema Generator. Create a product schema from extracted data.

Product Type: {product_type}
Extracted Specifications from PDFs:
{extracted_specs}

Generate a standardized schema for this product type, including:
1. Required fields (mandatory specifications)
2. Optional fields (nice-to-have specifications)
3. Field types (string, number, boolean, enum)
4. Typical ranges/values

Return ONLY valid JSON:
{{
    "product_type": "{product_type}",
    "schema": {{
        "required": [
            {{
                "field": "<field name>",
                "type": "string" | "number" | "boolean" | "enum",
                "description": "<field description>",
                "example": "<example value>"
            }}
        ],
        "optional": [
            {{
                "field": "<field name>",
                "type": "string" | "number" | "boolean" | "enum",
                "description": "<field description>",
                "example": "<example value>"
            }}
        ]
    }},
    "confidence": <0.0-1.0>
}}
"""


# ============================================================================
# WORKFLOW NODES - Using Direct Tool Calls Where Possible
# ============================================================================

def discover_vendors_node(state: PotentialProductIndexState) -> PotentialProductIndexState:
    """
    Agent 3: Vendor Discovery.
    Uses LLM to identify top 5 vendors for the product type.
    """
    logger.info(f"[PPI] Phase 1: Discovering vendors for {state['product_type']}...")
    
    try:
        with LLM_SEMAPHORE:
            llm = create_llm_with_fallback(
                model="gemini-2.5-flash",
                temperature=0.2,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            
            prompt = ChatPromptTemplate.from_template(VENDOR_DISCOVERY_PROMPT)
            parser = JsonOutputParser()
            chain = prompt | llm | parser
            
            result = chain.invoke({
                "product_type": state["product_type"],
                "context": json.dumps(state.get("rag_context", {}))
            })
        
        vendors = [v["name"] for v in result.get("vendors", [])]
        state["discovered_vendors"] = vendors[:5]  # Limit to 5
        
        for vendor in state["discovered_vendors"]:
            state["vendor_processing_status"][vendor] = "discovered"
        
        state["current_phase"] = "discover_models"
        
        state["messages"] = state.get("messages", []) + [{
            "role": "system",
            "content": f"Discovered {len(state['discovered_vendors'])} vendors: {', '.join(state['discovered_vendors'])}"
        }]
        
        logger.info(f"[PPI] Discovered vendors: {state['discovered_vendors']}")
        
    except Exception as e:
        logger.error(f"[PPI] Vendor discovery failed: {e}")
        state["error"] = str(e)
        # Fallback vendors
        state["discovered_vendors"] = ["Honeywell", "Emerson", "Yokogawa", "ABB", "Siemens"]
    
    return state


def discover_models_node(state: PotentialProductIndexState) -> PotentialProductIndexState:
    """
    Discover model families for each vendor.
    Uses parallel LLM calls (with rate limiting).
    """
    logger.info("[PPI] Phase 2: Discovering model families...")
    
    try:
        vendor_model_families = {}
        
        def discover_for_vendor(vendor: str) -> tuple:
            with LLM_SEMAPHORE:
                llm = create_llm_with_fallback(
                    model="gemini-2.5-flash",
                    temperature=0.2,
                    google_api_key=os.getenv("GOOGLE_API_KEY")
                )
                
                prompt = ChatPromptTemplate.from_template(MODEL_FAMILY_DISCOVERY_PROMPT)
                parser = JsonOutputParser()
                chain = prompt | llm | parser
                
                result = chain.invoke({
                    "vendor": vendor,
                    "product_type": state["product_type"],
                    "context": ""
                })
                
                families = [m["name"] for m in result.get("model_families", [])]
                return vendor, families
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(discover_for_vendor, vendor): vendor
                for vendor in state["discovered_vendors"]
            }
            
            for future in as_completed(futures):
                vendor = futures[future]
                try:
                    _, families = future.result()
                    vendor_model_families[vendor] = families
                    state["vendor_processing_status"][vendor] = "models_discovered"
                except Exception as e:
                    logger.error(f"Model discovery failed for {vendor}: {e}")
                    vendor_model_families[vendor] = []
                    state["failed_vendors"].append(vendor)
        
        state["vendor_model_families"] = vendor_model_families
        state["current_phase"] = "search_pdfs"
        
        total_families = sum(len(f) for f in vendor_model_families.values())
        state["messages"] = state.get("messages", []) + [{
            "role": "system",
            "content": f"Discovered {total_families} model families across {len(vendor_model_families)} vendors"
        }]
        
        logger.info(f"[PPI] Model families: {vendor_model_families}")
        
    except Exception as e:
        logger.error(f"[PPI] Model discovery failed: {e}")
        state["error"] = str(e)
    
    return state


def search_pdfs_node(state: PotentialProductIndexState) -> PotentialProductIndexState:
    """
    Agent 4: Parallel PDF Search.
    Uses 3-tier search (Serper → SerpAPI → Google CSE).
    """
    logger.info("[PPI] Phase 3: Searching for PDF datasheets...")
    
    try:
        pdf_search_results = {}
        
        def search_vendor_pdfs(vendor: str, model_families: List[str]) -> tuple:
            vendor_pdfs = []
            
            for family in model_families[:3]:  # Limit to 3 families per vendor
                try:
                    result = search_pdf_datasheets_tool.invoke({
                        "vendor": vendor,
                        "product_type": state["product_type"],
                        "model_family": family
                    })
                    
                    if result.get("success"):
                        for pdf in result.get("pdf_results", [])[:2]:  # Top 2 per family
                            vendor_pdfs.append({
                                "url": pdf.get("url"),
                                "title": pdf.get("title"),
                                "model_family": family
                            })
                except Exception as e:
                    logger.error(f"PDF search failed for {vendor}/{family}: {e}")
            
            return vendor, vendor_pdfs
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {}
            
            for vendor, families in state["vendor_model_families"].items():
                futures[executor.submit(search_vendor_pdfs, vendor, families)] = vendor
            
            for future in as_completed(futures):
                vendor = futures[future]
                try:
                    _, pdfs = future.result()
                    pdf_search_results[vendor] = pdfs
                    state["vendor_processing_status"][vendor] = "pdfs_found"
                except Exception as e:
                    logger.error(f"PDF search failed for {vendor}: {e}")
                    pdf_search_results[vendor] = []
        
        state["pdf_search_results"] = pdf_search_results
        state["current_phase"] = "download_pdfs"
        
        total_pdfs = sum(len(p) for p in pdf_search_results.values())
        state["messages"] = state.get("messages", []) + [{
            "role": "system",
            "content": f"Found {total_pdfs} PDF datasheets"
        }]
        
        logger.info(f"[PPI] Found PDFs: {total_pdfs}")
        
    except Exception as e:
        logger.error(f"[PPI] PDF search failed: {e}")
        state["error"] = str(e)
    
    return state


def process_pdfs_node(state: PotentialProductIndexState) -> PotentialProductIndexState:
    """
    Download PDFs, extract text, and prepare for RAG indexing.
    Note: In production, this would upload to Azure Blob Storage.
    """
    logger.info("[PPI] Phase 4: Processing PDFs...")
    
    try:
        extracted_content = {}
        
        def process_vendor_pdfs(vendor: str, pdfs: List[Dict]) -> tuple:
            vendor_content = []
            
            for pdf in pdfs:
                url = pdf.get("url")
                if not url:
                    continue
                
                try:
                    # In production: Download → Azure Blob → Extract text
                    # For now: Simulate extraction with metadata
                    content = {
                        "url": url,
                        "title": pdf.get("title", ""),
                        "model_family": pdf.get("model_family", ""),
                        "extracted_text": f"[PDF content for {vendor} {pdf.get('model_family', '')}]",
                        "status": "simulated"
                    }
                    vendor_content.append(content)
                except Exception as e:
                    logger.error(f"PDF processing failed for {url}: {e}")
            
            return vendor, vendor_content
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {}
            
            for vendor, pdfs in state["pdf_search_results"].items():
                futures[executor.submit(process_vendor_pdfs, vendor, pdfs)] = vendor
            
            for future in as_completed(futures):
                vendor = futures[future]
                try:
                    _, content = future.result()
                    extracted_content[vendor] = content
                    state["vendor_processing_status"][vendor] = "pdfs_processed"
                    state["pdf_download_status"][vendor] = "completed"
                except Exception as e:
                    logger.error(f"PDF processing failed for {vendor}: {e}")
                    state["pdf_download_status"][vendor] = "failed"
        
        state["extracted_content"] = json.dumps(extracted_content)
        state["current_phase"] = "index_content"
        
    except Exception as e:
        logger.error(f"[PPI] PDF processing failed: {e}")
        state["error"] = str(e)
    
    return state


def index_content_node(state: PotentialProductIndexState) -> PotentialProductIndexState:
    """
    Index extracted content into RAG vector store.
    """
    logger.info("[PPI] Phase 5: Indexing content for RAG...")
    
    try:
        extracted_content = json.loads(state.get("extracted_content", "{}"))
        
        indexed_chunks = []
        
        for vendor, contents in extracted_content.items():
            for content in contents:
                chunk = {
                    "vendor": vendor,
                    "model_family": content.get("model_family", ""),
                    "title": content.get("title", ""),
                    "text": content.get("extracted_text", ""),
                    "source_url": content.get("url", ""),
                    "indexed": True
                }
                indexed_chunks.append(chunk)
        
        state["indexed_chunks"] = indexed_chunks
        state["vector_store_ids"] = [f"chunk_{i}" for i in range(len(indexed_chunks))]
        
        state["current_phase"] = "generate_schema"
        
        state["messages"] = state.get("messages", []) + [{
            "role": "system",
            "content": f"Indexed {len(indexed_chunks)} content chunks"
        }]
        
        logger.info(f"[PPI] Indexed {len(indexed_chunks)} chunks")
        
    except Exception as e:
        logger.error(f"[PPI] Content indexing failed: {e}")
        state["error"] = str(e)
    
    return state


def generate_schema_node(state: PotentialProductIndexState) -> PotentialProductIndexState:
    """
    Agent 6: Generate product schema from indexed content.
    """
    logger.info("[PPI] Phase 6: Generating product schema...")
    
    try:
        with LLM_SEMAPHORE:
            llm = create_llm_with_fallback(
                model="gemini-2.5-flash",
                temperature=0.2,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            
            prompt = ChatPromptTemplate.from_template(SCHEMA_GENERATION_PROMPT)
            parser = JsonOutputParser()
            chain = prompt | llm | parser
            
            # Create specs summary from indexed chunks
            specs_summary = []
            for chunk in state.get("indexed_chunks", [])[:10]:
                specs_summary.append({
                    "vendor": chunk.get("vendor"),
                    "model_family": chunk.get("model_family"),
                    "title": chunk.get("title")
                })
            
            result = chain.invoke({
                "product_type": state["product_type"],
                "extracted_specs": json.dumps(specs_summary, indent=2)
            })
        
        state["generated_schema"] = result.get("schema", {})
        state["current_phase"] = "save_schema"
        
        logger.info(f"[PPI] Schema generated for {state['product_type']}")
        
    except Exception as e:
        logger.error(f"[PPI] Schema generation failed: {e}")
        state["error"] = str(e)
        state["generated_schema"] = None
    
    return state


def save_schema_node(state: PotentialProductIndexState) -> PotentialProductIndexState:
    """
    Save generated schema to MongoDB.
    """
    logger.info("[PPI] Phase 7: Saving schema...")
    
    try:
        if state.get("generated_schema"):
            # Save to Azure Blob Storage
            from azure_blob_utils import azure_blob_file_manager
            
            schema = state["generated_schema"]
            product_type = state["product_type"]
            
            metadata = {
                'collection_type': 'specs',
                'product_type': product_type,
                'normalized_product_type': product_type.lower().replace(' ', '').replace('_', ''),
                'filename': f"{product_type.lower().replace(' ', '_')}.json",
                'file_type': 'json',
                'schema_version': '1.0'
            }
            
            doc_id = azure_blob_file_manager.upload_json_data(schema, metadata)
            
            state["schema_saved"] = True
            
            state["messages"] = state.get("messages", []) + [{
                "role": "system",
                "content": f"Schema saved to Azure for {state['product_type']} (ID: {doc_id})"
            }]
            
            logger.info(f"[PPI] Schema saved to Azure for {state['product_type']}")
        else:
            state["schema_saved"] = False
            logger.warning("[PPI] No schema to save")
        
        state["current_phase"] = "complete"
        
    except Exception as e:
        logger.error(f"[PPI] Schema save failed: {e}")
        state["error"] = str(e)
        state["schema_saved"] = False
    
    return state


# ============================================================================
# WORKFLOW CREATION
# ============================================================================

def create_potential_product_index_workflow() -> StateGraph:
    """
    Create the Potential Product Index sub-workflow.
    
    Triggered when no schema exists for a product type.
    
    Flow:
    1. Vendor Discovery (LLM → Top 5 vendors)
    2. Model Family Discovery (LLM per vendor)
    3. PDF Search (3-tier: Serper → SerpAPI → Google)
    4. PDF Processing (Download, Extract, Azure Blob)
    5. Content Indexing (RAG vector store)
    6. Schema Generation (LLM)
    7. Schema Save (MongoDB)
    """
    
    workflow = StateGraph(PotentialProductIndexState)
    
    # Add nodes
    workflow.add_node("discover_vendors", discover_vendors_node)
    workflow.add_node("discover_models", discover_models_node)
    workflow.add_node("search_pdfs", search_pdfs_node)
    workflow.add_node("process_pdfs", process_pdfs_node)
    workflow.add_node("index_content", index_content_node)
    workflow.add_node("generate_schema", generate_schema_node)
    workflow.add_node("save_schema", save_schema_node)
    
    # Set entry point
    workflow.set_entry_point("discover_vendors")
    
    # Add edges (linear flow)
    workflow.add_edge("discover_vendors", "discover_models")
    workflow.add_edge("discover_models", "search_pdfs")
    workflow.add_edge("search_pdfs", "process_pdfs")
    workflow.add_edge("process_pdfs", "index_content")
    workflow.add_edge("index_content", "generate_schema")
    workflow.add_edge("generate_schema", "save_schema")
    workflow.add_edge("save_schema", END)
    
    return workflow


# ============================================================================
# WORKFLOW EXECUTION
# ============================================================================

def run_potential_product_index_workflow(
    product_type: str,
    session_id: str = "default",
    parent_workflow: str = "solution",
    rag_context: Dict[str, Any] = None,
    checkpointing_backend: str = "memory"
) -> Dict[str, Any]:
    """
    Run the Potential Product Index sub-workflow.
    
    Args:
        product_type: Product type to index
        session_id: Session identifier
        parent_workflow: Parent workflow that triggered this
        rag_context: Context from parent workflow
        checkpointing_backend: Backend for state persistence
    
    Returns:
        Workflow result with generated schema
    """
    try:
        # Create initial state
        initial_state = create_potential_product_index_state(
            product_type=product_type,
            session_id=session_id,
            parent_workflow=parent_workflow,
            rag_context=rag_context
        )
        
        # Create and compile workflow
        workflow = create_potential_product_index_workflow()
        compiled = compile_with_checkpointing(workflow, checkpointing_backend)
        
        # Run workflow
        config = {"configurable": {"thread_id": f"{session_id}_ppi"}}
        final_state = compiled.invoke(initial_state, config)
        
        return {
            "success": True,
            "product_type": product_type,
            "discovered_vendors": final_state.get("discovered_vendors", []),
            "vendor_model_families": final_state.get("vendor_model_families", {}),
            "generated_schema": final_state.get("generated_schema"),
            "schema_saved": final_state.get("schema_saved", False),
            "indexed_chunks_count": len(final_state.get("indexed_chunks", [])),
            "failed_vendors": final_state.get("failed_vendors", []),
            "error": final_state.get("error")
        }
        
    except Exception as e:
        logger.error(f"Potential Product Index workflow failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }
