# chaining.py
# Contains AI components setup and analysis chain creation (LangChain removed)
import json
import logging
import os
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from models import VendorAnalysis, OverallRanking, RequirementValidation
from loading import load_requirements_schema, load_products_runnable, load_pdf_content_runnable

# Create the actual load_products_data function from the runnable
_load_products_runnable_instance = load_products_runnable()
# Access the inner function from the RunnableLambda
load_products_data = _load_products_runnable_instance.func

# Create load_pdf_content function similarly
_load_pdf_content_runnable_instance = load_pdf_content_runnable()
load_pdf_content = _load_pdf_content_runnable_instance.func
from azure_blob_utils import get_available_vendors_from_mongodb, get_vendors_for_product_type
from dotenv import load_dotenv
import google.generativeai as genai

# Import standardization utilities
from standardization_utils import standardize_ranking_result

# Import LLM fallback utilities
from llm_fallback import FallbackLLMClient


# Load environment variables
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class GeminiClient:
    """Wrapper for Google Generative AI client with OpenAI fallback"""
    def __init__(self, model_name, temperature=0.1, api_key=None, openai_api_key=None):
        self.model_name = model_name
        self.temperature = temperature

        # Use FallbackLLMClient for automatic fallback
        self.client = FallbackLLMClient(
            model_name=model_name,
            temperature=temperature,
            api_key=api_key,
            openai_api_key=openai_api_key
        )

    def invoke(self, prompt):
        """Invoke the model with a prompt (with automatic fallback)"""
        try:
            return self.client.invoke(prompt)
        except Exception as e:
            logging.error(f"Error invoking LLM (both Gemini and OpenAI failed): {e}")
            raise


def parse_json_response(text, pydantic_model=None):
    """Parse JSON response from LLM, with optional Pydantic validation"""
    try:
        # Clean the response
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        # Parse JSON
        parsed = json.loads(text)

        # Validate with Pydantic if model provided
        if pydantic_model:
            validated = pydantic_model(**parsed)
            return validated.dict() if hasattr(validated, 'dict') else validated

        return parsed
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON: {e}")
        logging.error(f"Raw text: {text[:500]}")
        raise
    except Exception as e:
        logging.error(f"Error parsing response: {e}")
        raise


def get_format_instructions(pydantic_model):
    """Generate format instructions from Pydantic model"""
    if not pydantic_model:
        return "Return a valid JSON object."

    schema = pydantic_model.schema()
    properties = schema.get('properties', {})

    instructions = f"Return a JSON object with the following structure:\n{{\n"
    for key, value in properties.items():
        field_type = value.get('type', 'any')
        instructions += f'  "{key}": <{field_type}>,\n'
    instructions += "}"

    return instructions


def setup_ai_components():
    """Setup AI components with OpenAI fallback support"""

    # Get OpenAI API key for fallback
    openai_key = os.getenv("OPENAI_API_KEY")

    # Create different models for different purposes with fallback
    llm_flash = GeminiClient(
        model_name="gemini-2.0-flash-exp",
        temperature=0.1,
        api_key=os.getenv("GOOGLE_API_KEY"),
        openai_api_key=openai_key
    )

    llm_flash_lite = GeminiClient(
        model_name="gemini-2.0-flash-exp",
        temperature=0.1,
        api_key=os.getenv("GOOGLE_API_KEY"),
        openai_api_key=openai_key
    )

    llm_pro = GeminiClient(
        model_name="gemini-2.0-flash-exp",
        temperature=0.1,
        api_key=os.getenv("GOOGLE_API_KEY1"),
        openai_api_key=openai_key
    )

    # Get format instructions for different tasks
    validation_format_instructions = get_format_instructions(RequirementValidation)
    vendor_format_instructions = get_format_instructions(VendorAnalysis)
    ranking_format_instructions = get_format_instructions(OverallRanking)
    additional_requirements_format_instructions = get_format_instructions(RequirementValidation)

    return {
        'llm_flash': llm_flash,
        'llm_flash_lite': llm_flash_lite,
        'llm_pro': llm_pro,
        'validation_format_instructions': validation_format_instructions,
        'vendor_format_instructions': vendor_format_instructions,
        'ranking_format_instructions': ranking_format_instructions,
        'additional_requirements_format_instructions': additional_requirements_format_instructions,
    }


# Maintain backward compatibility
def setup_langchain_components():
    """Backward compatibility wrapper"""
    return setup_ai_components()


def invoke_validation_chain(components, user_input, schema, format_instructions):
    """Invoke validation chain"""
    from prompts import get_validation_prompt

    prompt = get_validation_prompt(user_input, schema, format_instructions)
    llm = components['llm_flash']
    response = llm.invoke(prompt)

    return parse_json_response(response, RequirementValidation)


def invoke_requirements_chain(components, user_input):
    """Invoke requirements chain"""
    from prompts import get_requirements_prompt

    prompt = get_requirements_prompt(user_input)
    llm = components['llm_flash']
    response = llm.invoke(prompt)

    return response


def invoke_vendor_chain(components, structured_requirements, products_json, pdf_content_json, format_instructions):
    """Invoke vendor analysis chain"""
    from prompts import get_vendor_prompt

    prompt = get_vendor_prompt(structured_requirements, products_json, pdf_content_json, format_instructions)
    llm = components['llm_pro']
    response = llm.invoke(prompt)

    return parse_json_response(response, VendorAnalysis)


def invoke_ranking_chain(components, vendor_analysis, format_instructions):
    """Invoke ranking chain"""
    from prompts import get_ranking_prompt

    prompt = get_ranking_prompt(vendor_analysis, format_instructions)
    llm = components['llm_pro']
    response = llm.invoke(prompt)

    return parse_json_response(response, OverallRanking)


def invoke_additional_requirements_chain(components, user_input, product_type, schema, format_instructions):
    """Invoke additional requirements chain"""
    from prompts import get_additional_requirements_prompt

    prompt = get_additional_requirements_prompt(user_input, product_type, schema, format_instructions)
    llm = components['llm_flash']
    response = llm.invoke(prompt)

    return parse_json_response(response, RequirementValidation)


def invoke_schema_description_chain(components, field_name, product_type):
    """Invoke schema description chain"""
    from prompts import get_schema_description_prompt

    prompt = get_schema_description_prompt(field_name, product_type)
    llm = components['llm_flash_lite']
    response = llm.invoke(prompt)

    return response


def get_final_ranking(vendor_analysis_dict):
    """
    Processes a dictionary of vendor analysis to create a final ranked list.
    Takes a dictionary, not a Pydantic object.
    """
    products = []
    # Use .get() for safe access in case vendor_matches is missing
    if not vendor_analysis_dict or not vendor_analysis_dict.get('vendor_matches'):
        return {'ranked_products': []}

    for product in vendor_analysis_dict['vendor_matches']:
        product_score = product.get('match_score', 0)
        product_name = product.get('product_name', '')

        # Extract model_family from the vendor analysis
        model_family = product.get('model_family', '')

        # Fallback: if model_family is empty, use product_name
        if not model_family:
            model_family = product_name

        # Ensure requirements_match is a boolean
        products.append({
            'product_name': product_name,
            'vendor': product.get('vendor', ''),
            'model_family': model_family,
            'match_score': product_score,
            'requirements_match': product.get('requirements_match', False),
            'reasoning': product.get('reasoning', ''),
            'limitations': product.get('limitations', '')
        })

    products_sorted = sorted(products, key=lambda x: x['match_score'], reverse=True)

    final_ranking = []
    rank = 1
    for product in products_sorted:
        final_ranking.append({
            'rank': rank,
            'product_name': product['product_name'],
            'vendor': product['vendor'],
            'model_family': product['model_family'],
            'overall_score': product['match_score'],
            'requirements_match': product['requirements_match'],
            'key_strengths': product['reasoning'],
            'concerns': product['limitations']
        })
        rank += 1

    # Apply standardization to the ranking result
    ranking_result = {'ranked_products': final_ranking}
    standardized_ranking = standardize_ranking_result(ranking_result)
    return standardized_ranking


def to_dict_if_pydantic(obj):
    """Helper function to safely convert Pydantic object to dict."""
    if hasattr(obj, 'dict'):
        return obj.dict()
    return obj


def _prepare_vendor_payloads(products_json_str, pdf_content_json_str):
    """Split combined vendor data into per-vendor payloads."""
    try:
        products = json.loads(products_json_str) if products_json_str else []
    except (json.JSONDecodeError, TypeError):
        logging.warning("Failed to parse products_json; defaulting to empty list.")
        products = []

    try:
        pdf_content = json.loads(pdf_content_json_str) if pdf_content_json_str else {}
    except (json.JSONDecodeError, TypeError):
        logging.warning("Failed to parse pdf_content_json; defaulting to empty dict.")
        pdf_content = {}

    logging.info(f"[VENDOR_PAYLOADS] Processing {len(pdf_content)} vendors from filtered PDF content")

    vendor_payloads = {}

    # First, seed payloads from PDF content
    for vendor_name, text in pdf_content.items():
        vendor_payloads[vendor_name] = {"products": [], "pdf_text": text}

    # Then, attach products to the corresponding PDF vendors
    for product in products:
        if not isinstance(product, dict):
            continue
        vendor_name = product.get("vendor") or product.get("vendor_name") or "Unknown Vendor"
        if vendor_name in vendor_payloads:
            vendor_payloads[vendor_name]["products"].append(product)

    # Final filtering: keep only vendors with non-empty PDF text
    vendor_payloads = {
        vendor: data
        for vendor, data in vendor_payloads.items()
        if data.get("pdf_text") and str(data["pdf_text"]).strip()
    }

    return vendor_payloads


def _invoke_vendor_chain_for_payload(
    components,
    format_instructions,
    structured_requirements,
    vendor_name,
    vendor_data,
):
    """Invoke vendor chain for a single vendor."""
    pdf_text = vendor_data.get("pdf_text")
    pdf_payload = json.dumps({vendor_name: pdf_text}, ensure_ascii=False) if pdf_text else json.dumps({})
    products_payload = json.dumps(vendor_data.get("products", []), ensure_ascii=False)

    return invoke_vendor_chain(
        components,
        structured_requirements,
        products_payload,
        pdf_payload,
        format_instructions
    )


def _run_parallel_vendor_analysis(
    structured_requirements,
    products_json_str,
    pdf_content_json_str,
    components,
    format_instructions,
):
    """Fan out vendor analysis so each vendor hits the LLM independently."""
    payloads = _prepare_vendor_payloads(products_json_str, pdf_content_json_str)

    if not payloads:
        logging.warning("[VENDOR_ANALYSIS] No vendor payloads available for analysis.")
        return {"vendor_matches": [], "vendor_run_details": []}

    logging.info("[VENDOR_ANALYSIS] Preparing analysis for vendors: %s", list(payloads.keys()))

    max_workers_env = os.getenv("VENDOR_ANALYSIS_MAX_WORKERS")
    try:
        max_workers = int(max_workers_env) if max_workers_env else 5
    except ValueError:
        max_workers = 5

    max_workers = max(1, min(len(payloads), max_workers))

    logging.info(
        "[VENDOR_ANALYSIS] Using max_workers=%s for %s vendors",
        max_workers,
        len(payloads),
    )

    vendor_matches = []
    run_details = []

    def _worker(vendor, data):
        result_dict = None
        error = None
        max_retries = 3
        base_retry_delay = 15

        logging.info("[VENDOR_ANALYSIS] START vendor=%s", vendor)

        for attempt in range(max_retries):
            try:
                result = _invoke_vendor_chain_for_payload(
                    components,
                    format_instructions,
                    structured_requirements,
                    vendor,
                    data,
                )
                result_dict = to_dict_if_pydantic(result)
                break

            except Exception as exc:
                error_msg = str(exc)
                is_rate_limit = "429" in error_msg or "Resource has been exhausted" in error_msg or "quota" in error_msg.lower()

                if is_rate_limit and attempt < max_retries - 1:
                    wait_time = base_retry_delay * (2 ** attempt)
                    logging.warning(f"[VENDOR_ANALYSIS] Rate limit hit for {vendor}, retry {attempt + 1}/{max_retries} after {wait_time}s")
                    time.sleep(wait_time)
                    continue
                else:
                    logging.error(f"Vendor analysis failed for {vendor}: {error_msg}")
                    error = error_msg
                    break

        logging.info("[VENDOR_ANALYSIS] END   vendor=%s error=%s", vendor, error)
        return vendor, result_dict, error

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {}
        for i, (vendor, data) in enumerate(payloads.items()):
            if i > 0:
                time.sleep(10)
                logging.info("[VENDOR_ANALYSIS] Submitting thread %d for vendor: %s (after 10s delay)", i+1, vendor)
            else:
                logging.info("[VENDOR_ANALYSIS] Submitting thread %d for vendor: %s (no delay)", i+1, vendor)
            future = executor.submit(_worker, vendor, data)
            future_map[future] = vendor

        for future in as_completed(future_map):
            vendor = future_map[future]
            vendor_result = None
            error = None
            try:
                vendor, vendor_result, error = future.result()
            except Exception as exc:
                error = str(exc)
                logging.error(f"Unexpected failure joining future for {vendor}: {exc}")

            if vendor_result and isinstance(vendor_result.get("vendor_matches"), list):
                for match in vendor_result["vendor_matches"]:
                    vendor_matches.append(to_dict_if_pydantic(match))
                run_details.append({"vendor": vendor, "status": "success"})
            else:
                entry = {
                    "vendor": vendor,
                    "status": "failed" if error else "empty",
                }
                if error:
                    entry["error"] = error
                run_details.append(entry)

    return {"vendor_matches": vendor_matches, "vendor_run_details": run_details}


def load_vendors_and_filter(input_dict):
    """Load vendors from database and apply CSV filtering after product type detection"""
    # Get detected product type from previous step
    detected_product_type = input_dict.get("detected_product_type", None)
    logging.info(f"[VENDOR_LOADING] Using detected product type: {detected_product_type}")

    # Get vendors that match the detected product type
    if detected_product_type:
        all_vendors = get_vendors_for_product_type(detected_product_type)
        logging.info(f"[VENDOR_LOADING] Found {len(all_vendors)} vendors for product type '{detected_product_type}': {all_vendors}")
    else:
        all_vendors = get_available_vendors_from_mongodb()
        logging.info(f"[VENDOR_LOADING] No product type detected, found {len(all_vendors)} vendors in database: {all_vendors}")

    # Check for CSV vendor filter in session
    from flask import session
    csv_vendor_filter = session.get('csv_vendor_filter', {})
    allowed_vendors = csv_vendor_filter.get('vendor_names', [])

    if allowed_vendors:
        logging.info(f"[CSV_FILTER] Restricting analysis to CSV vendors: {allowed_vendors}")

        from fuzzywuzzy import process
        from standardization_utils import standardize_vendor_name

        filtered_vendors = []
        logging.info(f"[CSV_FILTER] Starting fuzzy matching with {len(all_vendors)} DB vendors against {len(allowed_vendors)} CSV vendors")

        try:
            for i, db_vendor in enumerate(all_vendors):
                logging.info(f"[CSV_FILTER] Processing DB vendor {i+1}/{len(all_vendors)}: '{db_vendor}'")

                fuzzy_result = process.extractOne(db_vendor, allowed_vendors)

                if fuzzy_result and fuzzy_result[1] >= 70:
                    best_match = fuzzy_result[0]
                    best_score = fuzzy_result[1]
                    filtered_vendors.append(db_vendor)
                    logging.info(f"[CSV_FILTER] ✓ Matched DB vendor '{db_vendor}' with CSV vendor '{best_match}' (Score: {best_score}%)")
                else:
                    logging.info(f"[CSV_FILTER] ✗ No match found for DB vendor '{db_vendor}' (best score: {fuzzy_result[1] if fuzzy_result else 0}%)")

        except Exception as e:
            logging.error(f"[CSV_FILTER] Error in fuzzy matching: {e}")
            filtered_vendors = all_vendors
            logging.warning(f"[CSV_FILTER] Falling back to all {len(filtered_vendors)} vendors due to error")

        logging.info(f"[CSV_FILTER] After fuzzy filtering, will analyze vendors: {filtered_vendors}")

        if not filtered_vendors:
            logging.warning("[FALLBACK] CSV filtering resulted in 0 vendors, falling back to all database vendors")
            filtered_vendors = all_vendors
            logging.info(f"[FALLBACK] Using all {len(filtered_vendors)} database vendors for analysis: {filtered_vendors}")
    else:
        filtered_vendors = all_vendors
        logging.info("[CSV_FILTER] No CSV vendor filter found, analyzing all vendors")

    if not filtered_vendors:
        logging.warning("[SAFETY_FALLBACK] No vendors available after all filtering")
        if all_vendors:
            filtered_vendors = all_vendors
            logging.info(f"[SAFETY_FALLBACK] Using {len(filtered_vendors)} product-specific vendors: {filtered_vendors}")
        else:
            logging.error("[SAFETY_FALLBACK] No vendors found for this product type - analysis cannot proceed")

    enriched = dict(input_dict)
    enriched["available_vendors"] = all_vendors
    enriched["filtered_vendors"] = filtered_vendors
    return enriched


def load_filtered_pdf_content(input_dict):
    """Load PDF content only for filtered vendors"""
    filtered_vendors = input_dict.get("filtered_vendors", [])

    if not filtered_vendors:
        logging.warning("[PDF_LOADING] No filtered vendors available for PDF loading")
        enriched = dict(input_dict)
        enriched["pdf_content_json"] = json.dumps({})
        return enriched

    logging.info(f"[PDF_LOADING] Loading PDFs for {len(filtered_vendors)} vendors: {filtered_vendors}")

    # Use the module-level load_pdf_content function (already imported from runnable)
    pdf_result = load_pdf_content(input_dict)
    all_pdf_content = json.loads(pdf_result.get("pdf_content_json", "{}"))

    # Filter PDF content to only include filtered vendors
    filtered_pdf_content = {}
    for vendor in filtered_vendors:
        if vendor in all_pdf_content:
            filtered_pdf_content[vendor] = all_pdf_content[vendor]
            logging.info(f"[PDF_LOADING] Loaded PDF for vendor: {vendor}")
        else:
            # Try fuzzy matching
            matched_key = None
            vendor_lower = vendor.lower()

            for pdf_vendor_key in all_pdf_content.keys():
                pdf_vendor_lower = pdf_vendor_key.lower()

                if vendor_lower == pdf_vendor_lower:
                    matched_key = pdf_vendor_key
                    break

                from fuzzywuzzy import fuzz
                similarity = fuzz.ratio(vendor_lower, pdf_vendor_lower)
                if similarity >= 85:
                    matched_key = pdf_vendor_key
                    break

            if matched_key:
                filtered_pdf_content[vendor] = all_pdf_content[matched_key]
                logging.info(f"[PDF_LOADING] Loaded PDF for vendor: {vendor} (matched with: {matched_key})")
            else:
                logging.warning(f"[PDF_LOADING] No PDF found for vendor: {vendor}")

    logging.info(f"[PDF_LOADING] Successfully loaded PDFs for {len(filtered_pdf_content)} vendors")

    enriched = dict(pdf_result)
    enriched["pdf_content_json"] = json.dumps(filtered_pdf_content)
    enriched["filtered_vendors"] = filtered_vendors
    return enriched


def create_analysis_chain(components, vendors_base_path=None):
    """Create analysis chain without LangChain"""

    def run_analysis(input_dict):
        """Run the analysis pipeline"""
        user_input = input_dict.get("user_input", "")

        # Step 1: Generate structured requirements
        logging.info("[PIPELINE] Step 1: Generating structured requirements")
        structured_requirements = invoke_requirements_chain(components, user_input)
        input_dict["structured_requirements"] = structured_requirements

        # Step 2: Detect product type
        logging.info("[PIPELINE] Step 2: Detecting product type")
        schema = load_requirements_schema()
        validation_result = invoke_validation_chain(
            components,
            user_input,
            json.dumps(schema, indent=2),
            components['validation_format_instructions']
        )
        input_dict["detected_product_type"] = validation_result.get('product_type', None)

        # Step 3: Load product data
        logging.info("[PIPELINE] Step 3: Loading product data")
        input_dict = load_products_data(input_dict)

        # Step 4: Load and filter vendors
        logging.info("[PIPELINE] Step 4: Loading and filtering vendors")
        input_dict = load_vendors_and_filter(input_dict)

        # Step 5: Load filtered PDF content
        logging.info("[PIPELINE] Step 5: Loading filtered PDF content")
        input_dict = load_filtered_pdf_content(input_dict)

        # Step 6: Run parallel vendor analysis
        logging.info("[PIPELINE] Step 6: Running parallel vendor analysis")
        input_dict["vendor_analysis"] = _run_parallel_vendor_analysis(
            structured_requirements=input_dict.get("structured_requirements"),
            products_json_str=input_dict.get("products_json"),
            pdf_content_json_str=input_dict.get("pdf_content_json"),
            components=components,
            format_instructions=components['vendor_format_instructions']
        )

        # Step 7: Generate final ranking
        logging.info("[PIPELINE] Step 7: Generating final ranking")
        input_dict["overall_ranking"] = get_final_ranking(
            to_dict_if_pydantic(input_dict.get("vendor_analysis", {}))
        )

        return input_dict

    return run_analysis
