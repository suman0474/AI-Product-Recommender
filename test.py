from typing import IO, List, Dict, Any
from dotenv import load_dotenv
import os
import json
import re
import fitz  # PyMuPDF

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate

# REMOVED: MongoDB imports (upload_to_mongodb)
# REMOVED: LLM standardization import (llm_standardization)

print("1. Loading .env file...")
load_dotenv()


### REMOVED: identify_and_save_product_image function ###

# ---

### Extract text and tables from PDF using PyMuPDF ###
def extract_data_from_pdf(pdf_stream: IO[bytes]) -> List[str]:
    """
    Extracts text from PDF pages and converts tables heuristically into key-value lines.
    Each chunk includes the page number for context.
    """
    print("2. Extracting text from PDF using PyMuPDF...")
    page_chunks = []
    try:
        pdf_stream.seek(0)
        doc = fitz.open(stream=pdf_stream, filetype="pdf")

        for page_number, page in enumerate(doc, start=1):
            # Extract raw text
            page_text = page.get_text("text") or ""

            # Extract table-like blocks heuristically
            table_lines = []
            blocks = page.get_text("blocks") or []
            for block in blocks:
                lines = block[4].splitlines()
                for line in lines:
                    # Capture lines that look like key-value pairs (e.g., 'Key: Value')
                    if ":" in line:
                        table_lines.append(line.strip())

            # Combine text and table lines, and include page number
            combined_text = f"--- Page {page_number} ---\n{page_text}\n" + "\n".join(table_lines)
            combined_text = preprocess_specifications_text(combined_text)

            page_chunks.append(combined_text)

        print("2.1 PDF extraction into chunks successful.")
        return page_chunks

    except Exception as e:
        print(f"Error during PDF extraction: {e}")
        raise


def split_text(text: str, chunk_size: int = 3000) -> List[str]:
    """Splits text into chunks of a specified size."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def preprocess_specifications_text(text: str) -> str:
    """
    Convert lines like 'Key: Value' into 'Spec: key = value' for better LLM recognition.
    """
    lines = text.splitlines()
    processed_lines = []
    for line in lines:
        # Regex to match "Key: Value" or "Key - Value" for technical specs
        match = re.match(r'^([\w \-/\(\)]+):\s*(.+)$', line.strip())
        if match:
            key = match.group(1).strip()
            val = match.group(2).strip()
            processed_lines.append(f"Spec: {key} = {val}")
        else:
            processed_lines.append(line)
    return "\n".join(processed_lines)


### Send chunks to the LLM for structured JSON extraction ###
def send_to_language_model(chunks: List[str]) -> List[Dict[str, Any]]:
    print("3. Sending concatenated text to the language model...")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Please set your GOOGLE_API_KEY environment variable.")

    # Initialize LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

    prompt_template = PromptTemplate(
        input_variables=["full_text"],
        template="""
Extract structured technical data from the following text and return ONLY valid JSON according to this schema:
{{
  "product_type": "",
  "vendor": "",
  "models": [
    {{
      "model_series": "",
      "sub_models": [
        {{
          "name": "",
          "specifications": {{}}
        }}
      ]
    }}
  ]
}}

### Rules:
1. Output ONLY JSON. No explanations, comments, or extra text.
2. If any field is missing, use an empty string "" (do not omit fields).
3. Always include the keys exactly as shown in the schema.
4. Normalize specification keys:
    - lowercase
    - use underscores instead of spaces
5. Merge duplicate "model_series" entries into a single object with combined "sub_models".
6. Each sub-model includes:
    - "name": exact model name
    - "specifications": key-value pairs of all available specs
7. Flatten grouped specs into a single object.
8. If multiple model series exist, create separate JSON outputs for each model series.
9. For the product_type field: - If the product is a sub-category, return it under its parent category.
10. If a vendor is a sub-company or brand, return the vendor as the parent company.
11. Always include the original sub-model details even when changing the top-level product_type.
12. Do not include duplicate keys at different levels.
13. If no data is found for a field, leave it as "".
14. Return only valid JSON, with no extra characters or formatting.
15. Any "key: value" pair found in the text or tables MUST go into the "specifications" object of the corresponding sub_model.
16. If unsure which sub_model a specification belongs to, still include it under that sub_model's "specifications".


Text:
{full_text}
"""
    )

    full_text = "\n\n".join(chunks)
    prompt = prompt_template.format(full_text=full_text)

    # Use LLM to invoke the extraction
    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content.strip()

    # Clean up markdown code blocks around JSON
    if content.startswith("```json"):
        content = content[7:].strip()
    elif content.startswith("```"):
        content = content[3:].strip()
    if content.endswith("```"):
        content = content[:-3].strip()

    # Robust JSON parsing with multiple fallback strategies
    def try_parse_json(text: str) -> Any:
        """Try to parse JSON with multiple strategies."""
        
        # Strategy 1: Direct parsing
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Extract JSON from within text using regex (handles text before/after JSON)
        json_patterns = [
            r'(\{[\s\S]*\})',  # Match outermost curly braces
            r'(\[[\s\S]*\])',  # Match outermost square brackets
        ]
        for pattern in json_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        
        # Strategy 3: Handle truncated JSON by attempting to close brackets
        # Use a stack to track open brackets and close them in correct order
        open_braces = text.count('{') - text.count('}')
        open_brackets = text.count('[') - text.count(']')
        
        if open_braces > 0 or open_brackets > 0:
            # Try to repair by closing unclosed structures
            repaired = text.rstrip()
            
            # Remove trailing comma if present
            while repaired.endswith(',') or repaired.endswith(' '):
                repaired = repaired.rstrip(', ')
            
            # Close any open strings (look for unbalanced quotes)
            quote_count = repaired.count('"') - repaired.count('\\"')
            if quote_count % 2 == 1:
                repaired += '"'
            
            # Build a stack of unclosed brackets to determine correct closing order
            bracket_stack = []
            in_string = False
            escape_next = False
            
            for char in repaired:
                if escape_next:
                    escape_next = False
                    continue
                if char == '\\':
                    escape_next = True
                    continue
                if char == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if char == '{':
                    bracket_stack.append('}')
                elif char == '[':
                    bracket_stack.append(']')
                elif char in '}]' and bracket_stack and bracket_stack[-1] == char:
                    bracket_stack.pop()
            
            # Close brackets in reverse order (inner first)
            closing = ''.join(reversed(bracket_stack))
            repaired += closing
            
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                # Try alternate repairs
                # Sometimes we need to close with empty values
                alt_repairs = [
                    repaired,  # Already tried
                    text.rstrip().rstrip(',') + '"}' * open_braces + ']' * open_brackets,
                    text.rstrip().rstrip(',') + '"]}' * min(open_braces, open_brackets) + '}' * max(0, open_braces - open_brackets) + ']' * max(0, open_brackets - open_braces),
                ]
                for alt in alt_repairs[1:]:
                    try:
                        return json.loads(alt)
                    except json.JSONDecodeError:
                        continue
        
        # Strategy 4: Try to repair common JSON issues
        repaired_text = text
        # Fix trailing commas before closing brackets
        repaired_text = re.sub(r',(\s*[}\]])', r'\1', repaired_text)
        # Fix missing quotes around keys
        repaired_text = re.sub(r'(\{|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', repaired_text)
        
        try:
            return json.loads(repaired_text)
        except json.JSONDecodeError:
            pass
        
        # Strategy 5: Try parsing line by line for array of objects
        lines = text.strip().split('\n')
        if lines and lines[0].strip().startswith('{'):
            objects = []
            current_obj = []
            depth = 0
            for line in lines:
                current_obj.append(line)
                depth += line.count('{') - line.count('}')
                if depth == 0 and current_obj:
                    try:
                        obj_text = '\n'.join(current_obj)
                        obj = json.loads(obj_text)
                        objects.append(obj)
                        current_obj = []
                    except json.JSONDecodeError:
                        current_obj = []
                        depth = 0
            if objects:
                return objects if len(objects) > 1 else objects[0]
        
        return None
    
    # Try robust parsing
    data = try_parse_json(content)
    
    if data is not None:
        # Ensure the result is always a list, even if the LLM returns a single object
        return data if isinstance(data, list) else [data]
    else:
        print("Warning: Could not decode JSON from LLM response after all parsing strategies.")
        print("LLM Output (first 500 chars):", content[:500])
        
        # Last resort: Try to extract any model information using regex
        fallback_result = []
        
        # Try to extract product_type and vendor at minimum
        product_type_match = re.search(r'"product_type"\s*:\s*"([^"]*)"', content)
        vendor_match = re.search(r'"vendor"\s*:\s*"([^"]*)"', content)
        
        if product_type_match or vendor_match:
            fallback_result.append({
                "product_type": product_type_match.group(1) if product_type_match else "",
                "vendor": vendor_match.group(1) if vendor_match else "",
                "models": [],
                "_parsing_note": "Extracted via regex fallback due to JSON parsing failure"
            })
            print(f"Fallback extraction: product_type={product_type_match.group(1) if product_type_match else 'N/A'}, vendor={vendor_match.group(1) if vendor_match else 'N/A'}")
        
        return fallback_result


### Helpers for aggregation, normalization, and JSON saving ###
def normalize_series_name(series: str) -> str:
    """Creates a simple, normalized key from a model series name."""
    if not series:
        return ""
    # Simplified normalization: lowercase, replace spaces with underscores
    return series.lower().replace(" ", "_")


def split_product_types(results: List[Dict]) -> List[Dict]:
    """Splits results if 'product_type' contains multiple values (e.g., 'Type A / Type B')."""
    split_results = []
    for item in results:
        product_type = item.get("product_type", "").strip()
        vendor = item.get("vendor", "")
        models = item.get("models", [])
        # Split by common delimiters and filter empty strings
        types = [t.strip() for t in re.split(r'/|,', product_type) if t.strip()] if "/" in product_type or "," in product_type else [product_type]
        for pt in types:
            split_results.append({"product_type": pt, "vendor": vendor, "models": models})
    return split_results


def aggregate_results(results: List[Dict], file_name: str = "") -> Dict:
    """
    Aggregate and normalize extracted LLM results into a structured JSON format.
    """
    print("5. Aggregating and cleaning results...")

    def is_meaningful_spec(specs: Dict[str, Any]) -> bool:
        """Check if the specification dictionary has any meaningful values."""
        return bool(specs and any(v and str(v).strip() != "" for v in specs.values()))

    normalized_models = {}
    vendor = ""
    product_type = ""

    for item in results:
        if isinstance(item, list):
            continue  # skip nested lists

        # Try to capture vendor/product_type from the first meaningful result
        vendor = vendor or item.get("vendor", "").strip()
        product_type = product_type or item.get("product_type", "").strip()

        for model in item.get("models", []):
            series = model.get("model_series", "").strip()
            if not series:
                continue

            key = normalize_series_name(series)
            if key not in normalized_models:
                normalized_models[key] = {"model_series": series, "sub_models": []}

            for sub_model in model.get("sub_models", []):
                name = sub_model.get("name", "").strip()
                specs = sub_model.get("specifications") or {}
                # normalize keys as per LLM rule 4
                specs = {k.lower().replace(" ", "_"): v for k, v in specs.items() if v}

                if not name and not is_meaningful_spec(specs):
                    continue

                # Merge with existing sub_model if present (Rule 5 indirectly addressed via name match)
                existing = next((sm for sm in normalized_models[key]["sub_models"] if sm.get("name") == name), None)
                if existing:
                    for k, v in specs.items():
                        if k not in existing["specifications"]:
                            existing["specifications"][k] = v
                else:
                    # Add new sub_model
                    if not is_meaningful_spec(specs):
                        specs = {"_raw_text": "No structured specs extracted"}
                    normalized_models[key]["sub_models"].append({"name": name, "specifications": specs})

    # Filter out models with no meaningful sub-models
    filtered_models = []
    for model in normalized_models.values():
        meaningful_subs = [
            sm for sm in model["sub_models"]
            if sm.get("name") or is_meaningful_spec(sm.get("specifications", {}))
        ]
        if meaningful_subs:
            model["sub_models"] = meaningful_subs
            filtered_models.append(model)

    return {
        "product_type": product_type or "",
        "vendor": vendor or "",
        "models": filtered_models
    }


def generate_dynamic_path(final_result: Dict[str, Any]) -> str:
    """Generate MongoDB identifier instead of file path"""
    print("6. Generating MongoDB identifier...")
    vendor_name = re.sub(r'[<>:"/\\|?*]', '', final_result.get("vendor") or "UnknownVendor").strip()
    product_type = re.sub(r'[<>:"/\\|?*]', '', (final_result.get("product_type") or "UnknownProductType").lower()).strip()
    model_series_names = [m.get("model_series") for m in final_result.get("models", []) if m.get("model_series")]
    model_series = "_".join(model_series_names) if model_series_names else "unknown"
    
    # Return MongoDB identifier instead of file path
    return f"MongoDB:products:{vendor_name}:{product_type}:{model_series}"


def save_json(final_result: Dict[str, Any], file_path: str = None):
    """Save JSON to MongoDB instead of local file"""
    print(f"7. Saving JSON output to MongoDB...")
    
    from mongodb_utils import mongodb_file_manager
    
    vendor_name = final_result.get("vendor", "UnknownVendor").replace(" ", "_")
    product_type = final_result.get("product_type", "UnknownProduct").replace(" ", "_")
    model_series = final_result.get("models", [{}])[0].get("model_series", "unknown")
    
    try:
        # Structure: vendors/{vendor}/{product_type}/{model}.json
        metadata = {
            'vendor_name': vendor_name,
            'product_type': product_type,
            'model_series': model_series,
            'file_type': 'json',
            'collection_type': 'products',
            'path': f'vendors/{vendor_name}/{product_type}/{model_series}.json'
        }
        mongodb_file_manager.upload_json_data(final_result, metadata)
        print(f"7.1 JSON saved successfully to MongoDB: vendors/{vendor_name}/{product_type}/{model_series}.json")
    except Exception as e:
        print(f"7.1 Failed to save to MongoDB: {e}")
        raise


### Main function ###
def main(pdf_path: str):
    """
    Process a local PDF file: extract text and tables, generate structured JSON,
    and save the final aggregated data to MongoDB.
    and save the final aggregated data locally.

    Args:
        pdf_path (str): Path to the local PDF file

    Returns:
        List[Dict]: List of processed product results
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Open PDF as binary stream
    with open(pdf_path, "rb") as file_stream:
        file_name = os.path.basename(pdf_path)

        # 1. Extract text chunks
        text_chunks = extract_data_from_pdf(file_stream)

        # 2. Generate structured JSON
        all_results = send_to_language_model(text_chunks)
        # Flattens the list of results (handles cases where LLM returns a list of lists/objects)
        all_results = [item for r in all_results for item in (r if isinstance(r, list) else [r])]

        # 3. Aggregate and normalize
        final_result = aggregate_results(all_results, file_name)
        split_results = split_product_types([final_result])

        # 4. Save JSON
        for result in split_results:
            output_path = generate_dynamic_path(result)
            save_json(result, output_path)

        # REMOVED: Image identification and saving logic

    return split_results
