# advanced_parameters.py
# Advanced parameter discovery using pure LLM-based approach.
# Focuses on latest parameters added in the past 6 months for the detected product type.

import json
import logging
import os
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pymongo.errors import PyMongoError

from azure_blob_config import get_azure_blob_connection as get_mongodb_connection
from llm_fallback import create_llm_with_fallback

MONGO_TTL_DAYS = 30
IN_MEMORY_CACHE_TTL_MINUTES = 10
SCHEMA_CACHE_TTL_MINUTES = 30
MONGO_COLLECTION_CACHE = None
IN_MEMORY_ADVANCED_SPEC_CACHE: Dict[str, Dict[str, Any]] = {}
SCHEMA_PARAM_CACHE: Dict[str, Dict[str, Any]] = {}

# Load environment variables
load_dotenv()

def _normalize_product_type(product_type: str) -> str:
    return re.sub(r"[^a-z0-9]", "", product_type.lower()) if product_type else ""


def _get_in_memory_specs(product_type: str) -> Optional[List[Dict[str, Any]]]:
    if not product_type:
        return None

    normalized = _normalize_product_type(product_type)
    entry = IN_MEMORY_ADVANCED_SPEC_CACHE.get(normalized)
    if not entry:
        return None

    if entry["expires_at"] > datetime.utcnow():
        return entry["specs"]

    IN_MEMORY_ADVANCED_SPEC_CACHE.pop(normalized, None)
    return None


def _set_in_memory_specs(product_type: str, specs: List[Dict[str, Any]]) -> None:
    if not product_type:
        return

    normalized = _normalize_product_type(product_type)
    IN_MEMORY_ADVANCED_SPEC_CACHE[normalized] = {
        "specs": specs,
        "expires_at": datetime.utcnow() + timedelta(minutes=IN_MEMORY_CACHE_TTL_MINUTES),
    }


class AdvancedParametersDiscovery:
    """Discovers generic advanced specifications for fallback scenarios with OpenAI fallback."""

    def __init__(self):
        self.llm = create_llm_with_fallback(
            model="gemini-2.5-flash",
            temperature=0.4,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

    def get_generic_specifications(self, product_type: str) -> List[str]:
        """Get generic advanced specifications with series numbers for product type."""
        prompt = ChatPromptTemplate.from_template("""
List 15 latest advanced specifications for {product_type} (past 6 months).
Focus on: IoT, Industry 4.0, AI, protocols, cybersecurity, cloud, series numbers, model specifications.
Include specific series numbers and generation information.
Return JSON array in snake_case: ["series_5000_hart_protocol", "gen3_wireless_diagnostics", "model_pro_cybersecurity"]
""")
        try:
            response = (prompt | self.llm | StrOutputParser()).invoke({"product_type": product_type})
            params = json.loads(response.strip().replace('```json', '').replace('```', ''))
            return [str(p) for p in params][:15] if isinstance(params, list) else []
        except:
            return []

def get_existing_parameters(product_type: str) -> set:
    """Get existing parameters from schema with in-memory caching to speed up LLM requests."""
    if not product_type:
        return set()

    normalized = _normalize_product_type(product_type)
    now = datetime.utcnow()

    cached = SCHEMA_PARAM_CACHE.get(normalized)
    if cached and cached["expires_at"] > now:
        return cached["params"]

    try:
        from loading import load_requirements_schema

        schema = load_requirements_schema(product_type)
        if not schema:
            return set()

        params = set()
        for req_type in ["mandatory_requirements", "optional_requirements"]:
            for fields in schema.get(req_type, {}).values():
                if isinstance(fields, dict):
                    params.update(fields.keys())

        SCHEMA_PARAM_CACHE[normalized] = {
            "params": params,
            "expires_at": now + timedelta(minutes=SCHEMA_CACHE_TTL_MINUTES),
        }
        return params
    except Exception as exc:
        logging.warning(f"Failed to load schema for {product_type}: {exc}")
        return set()

def convert_specifications_to_human_readable(specs: List[str]) -> Dict[str, str]:
    """Convert specification keys to readable names without extra LLM calls."""
    readable: Dict[str, str] = {}
    for spec in specs:
        readable[spec] = spec.replace("_", " ").title()
    return readable


def _get_advanced_param_collection():
    global MONGO_COLLECTION_CACHE
    if MONGO_COLLECTION_CACHE is not None:
        return MONGO_COLLECTION_CACHE

    try:
        conn = get_mongodb_connection()
        MONGO_COLLECTION_CACHE = conn["collections"].get("advanced_parameters")
    except Exception as exc:
        logging.warning(f"MongoDB advanced parameters collection unavailable: {exc}")
        MONGO_COLLECTION_CACHE = None

    return MONGO_COLLECTION_CACHE


def _load_cached_specifications(product_type: str) -> Optional[List[Dict[str, Any]]]:
    in_memory = _get_in_memory_specs(product_type)
    if in_memory is not None:
        return in_memory

    collection = _get_advanced_param_collection()
    if collection is None or not product_type:
        return None

    cutoff = datetime.utcnow() - timedelta(days=MONGO_TTL_DAYS)
    normalized = _normalize_product_type(product_type)

    try:
        doc = collection.find_one(
            {
                "normalized_product_type": normalized,
                "created_at": {"$gte": cutoff},
            }
        )
        if doc:
            specs = doc.get("unique_specifications")
            if isinstance(specs, list):
                logging.info(f"Advanced parameters cache hit for {product_type}")
                _set_in_memory_specs(product_type, specs)
                return specs
    except PyMongoError as exc:
        logging.warning(f"Unable to read advanced parameters cache for {product_type}: {exc}")

    return None


def _persist_specifications(
    product_type: str,
    unique_specifications: List[Dict[str, Any]],
    existing_snapshot: List[str],
) -> None:
    collection = _get_advanced_param_collection()
    if collection is None or not product_type:
        return

    normalized = _normalize_product_type(product_type)
    now = datetime.utcnow()

    document = {
        "product_type": product_type,
        "normalized_product_type": normalized,
        "unique_specifications": unique_specifications,
        "existing_parameters_snapshot": existing_snapshot,
        "created_at": now,
        "updated_at": now,
    }

    try:
        collection.update_one(
            {"normalized_product_type": normalized},
            {"$set": document},
            upsert=True,
        )
        logging.info(
            f"Cached advanced parameters in MongoDB for {product_type} ({len(unique_specifications)} specs)"
        )
        _set_in_memory_specs(product_type, unique_specifications)
    except PyMongoError as exc:
        logging.warning(f"Failed to cache advanced parameters for {product_type}: {exc}")
        # Still keep in-memory cache even if Mongo upsert fails so current request benefits
        _set_in_memory_specs(product_type, unique_specifications)


def _build_response(product_type: str, unique_specifications: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "product_type": product_type,
        "vendor_specifications": [],
        "vendor_parameters": [],
        "unique_specifications": unique_specifications,
        "unique_parameters": unique_specifications,
        "total_vendors_searched": 0,
        "total_unique_specifications": len(unique_specifications),
        "total_unique_parameters": len(unique_specifications),
        "existing_specifications_filtered": 0,
    }


def _extract_json_object(raw_response: str) -> Dict[str, Any]:
    """Extract strict JSON even if the LLM emits conversational text."""
    if not raw_response:
        return {}
    # Remove common markdown fences and surrounding text
    cleaned = raw_response.strip()
    cleaned = cleaned.replace('```json', '').replace('```', '').strip()

    # Try to parse the whole cleaned text first (covers well-formed responses)
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Fallback: find the first JSON object in the text
    obj_start = cleaned.find('{')
    obj_end = cleaned.rfind('}')
    if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
        snippet = cleaned[obj_start:obj_end + 1]
        try:
            return json.loads(snippet)
        except Exception:
            logging.warning('Failed to decode JSON object from LLM response')

    # Fallback: try to find a JSON array (some responses return an array directly)
    arr_start = cleaned.find('[')
    arr_end = cleaned.rfind(']')
    if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
        snippet = cleaned[arr_start:arr_end + 1]
        try:
            arr = json.loads(snippet)
            # Wrap array into object for backward compatibility
            return {"parameters": arr} if isinstance(arr, list) else {}
        except Exception:
            logging.warning('Failed to decode JSON array from LLM response')

    logging.warning('LLM response missing usable JSON')

    # Last-resort attempt: try to extract a simple newline-separated list of items
    lines = [l.strip(' -*.\t') for l in cleaned.splitlines() if l.strip()]
    # Filter out lines that look like non-content
    candidate_items = []
    for line in lines:
        # Skip very short lines or lines that look like instructions
        if len(line) < 3:
            continue
        if line.lower().startswith(('return', 'task', 'rules', 'rules:')):
            continue
        # Remove numbering like '1.' or '- ' prefixes
        candidate_items.append(line.strip())

    if candidate_items:
        logging.info('Parsed %d candidate items from LLM free-text response', len(candidate_items))
        return {"parameters": candidate_items}

    # If nothing usable, log the full raw response for diagnostics and return empty
    logging.debug('Raw LLM response (no JSON found): %s', raw_response)
    return {}

def discover_advanced_parameters(product_type: str) -> Dict[str, Any]:
    """Discover latest advanced parameters for product type using a single LLM call.

    The LLM is responsible for:
    - Looking at the detected product type.
    - Considering latest advanced specifications from top vendors.
    - Comparing against the existing schema parameters.
    - Returning only **new** advanced specifications in a readable format.

    The public return structure stays compatible with existing callers:
    - "unique_specifications": list of {"key", "name"[, "description"]}
    - "existing_specifications_filtered": kept for logging (approximate)
    - "vendor_specifications": returned as an empty list (no per-vendor breakdown now)
    """

    try:
        if not product_type:
            return {
                "product_type": product_type,
                "vendor_specifications": [],
                "unique_specifications": [],
                "total_vendors_searched": 0,
                "total_unique_specifications": 0,
                "existing_specifications_filtered": 0,
            }

        # We always run the LLM for latest results but attempt to use cache for speed
        cached = _load_cached_specifications(product_type)
        if cached is not None:
            # return cached immediately to avoid extra LLM calls in high-load scenarios
            # but still proceed to refresh the cache in background could be implemented later
            return _build_response(product_type, cached)

        # Load existing parameters from schema so we can ask the LLM for *only new* ones
        existing = get_existing_parameters(product_type)
        existing_list = sorted(list(existing))

        logging.info(f"Starting single-call advanced specification discovery for: {product_type}")

        llm = create_llm_with_fallback(
            model="gemini-2.5-flash",
            temperature=0.4,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )


        # Request only human-readable parameter names from the LLM.
        # We'll deterministically create a snake_case key from each returned name.
        prompt = ChatPromptTemplate.from_template(
            """
You are an expert in industrial instrumentation and vendor product catalogs.

Task:
- Provide up to 5 human-friendly names of the latest advanced parameters or features for {product_type} from the past 6 months.
- Include series numbers and model/generation hints inside the human-readable name where applicable (for example: "Series 5000 HART Diagnostics").
- DO NOT return keys or code-friendly names — return only an array of human-readable strings.

Product type: {product_type}
Existing schema parameter keys (snake_case): {existing_parameters}

**CRITICAL - Avoid Duplicates:**
Do NOT suggest parameters that are semantically similar to existing ones.

Examples of what to AVOID:
- If "output_signal" exists, DON'T suggest "signal_output", "output_type", or "signal_type"
- If "pressure_range" exists, DON'T suggest "range_pressure", "pressure_limits", or "measurement_range"
- If "communication_protocol" exists, DON'T suggest "protocol_communication" or "comm_protocol"
- If "temperature_range" exists, DON'T suggest "temp_range" or "operating_temperature"

Examples of what IS acceptable (truly new parameters):
- If "output_signal" exists, you CAN suggest "wireless_diagnostics" (different concept)
- If "pressure_range" exists, you CAN suggest "predictive_maintenance" (different concept)

Return strict JSON only in this exact format and nothing else:
{{
  "parameters": [
    "Human Readable Parameter Name 1",
    "Another Parameter Name (Series X)",
  ]
}}

Rules:
- Do NOT include any parameter whose snake_case key (derived from the human name) would match an existing schema key.
- Prefer no more than 5 parameters.
- Do NOT include any extra commentary or markdown.
"""
        )

        chain = prompt | llm | StrOutputParser()
        raw_response = chain.invoke({"product_type": product_type, "existing_parameters": json.dumps(existing_list)})

        payload = _extract_json_object(raw_response)
        raw_params = payload.get("parameters", []) if isinstance(payload, dict) else []

        # Normalize existing keys for comparison
        existing_norm = {p.lower().replace("_", "") for p in existing}
        seen_norm = set()
        unique_specifications: List[Dict[str, Any]] = []

        for name in raw_params:
            if not isinstance(name, str):
                continue
            human_name = name.strip()
            if not human_name:
                continue

            # Generate a deterministic snake_case key from the human name
            key_candidate = human_name.lower()
            key_candidate = re.sub(r"[^a-z0-9 ]", "", key_candidate)
            key_candidate = re.sub(r"\s+", "_", key_candidate).strip("_")
            norm = key_candidate.replace("_", "")

            if not key_candidate or norm in existing_norm or norm in seen_norm:
                continue

            unique_specifications.append({"key": key_candidate, "name": human_name})
            seen_norm.add(norm)

        logging.info(
            f"Advanced specifications (single-call) discovery complete: {len(unique_specifications)} new specifications returned"
        )

        _persist_specifications(product_type, unique_specifications, existing_list)

        return _build_response(product_type, unique_specifications)
    except Exception as e:
        logging.error(f"Discovery failed (single-call): {e}")
        # Fallback to generic specs path to avoid breaking the UI completely
        discovery = AdvancedParametersDiscovery()
        generic_specs = discovery.get_generic_specifications(product_type)
        
        human_readable_map = convert_specifications_to_human_readable(generic_specs)
        
        fallback_specs = [
            {"key": p, "name": human_readable_map.get(p, p.replace('_', ' ').title())}
            for p in generic_specs
        ]
        fallback_response = _build_response(product_type, fallback_specs)
        fallback_response["fallback"] = True
        return fallback_response