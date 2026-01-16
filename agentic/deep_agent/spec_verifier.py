# agentic/deep_agent/spec_verifier.py
# =============================================================================
# SPECIFICATION VERIFIER - Dual-Step Verification for Deep Agent
# =============================================================================
#
# This module verifies if extracted values are technical specifications
# vs descriptive text that needs re-extraction.
#
# Architecture:
# - SpecVerifierAgent: Classifies values as specs vs descriptions
# - DescriptionExtractorAgent: Extracts actual values from descriptions
# - verify_and_reextract_specs: Orchestrates the dual-step flow
#
# FIXED: Now uses content-based validation instead of length-based heuristics
# FIXED: Integrated with ValueNormalizer for consistent post-processing
#
# =============================================================================

import re
import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Import LLM with fallback
try:
    from llm_fallback import create_llm_with_fallback
except ImportError:
    create_llm_with_fallback = None

# Import value normalizer
try:
    from .value_normalizer import normalize_spec_value, is_valid_spec_value, extract_and_validate_spec
except ImportError:
    normalize_spec_value = None
    is_valid_spec_value = None
    extract_and_validate_spec = None


# =============================================================================
# SPECIFICATION VERIFIER AGENT
# =============================================================================

class SpecVerifierAgent:
    """
    Verifies if extracted values are technical specifications
    vs descriptive text that needs re-extraction.

    Technical specs: "±0.1%", "4-20mA", "IP67", "-40 to 85°C"
    Descriptions: "depends on sensor type", "varies by application"

    FIXED: Uses content-based validation instead of length-based heuristics
    """

    # Patterns indicating a value IS a TECHNICAL SPEC (keep as-is)
    TECHNICAL_PATTERNS = [
        # Accuracy/tolerance patterns
        r'[±\+\-]\s*\d+\.?\d*\s*%',                    # ±0.1%, +0.5%
        r'[±\+\-]\s*\d+\.?\d*\s*(°[CF]|K|bar|psi|Pa)', # ±1°C, ±0.5bar

        # Range patterns
        r'-?\d+\.?\d*\s*(?:to|[-–])\s*[+\-]?\d+\.?\d*\s*(°[CF]|K|bar|psi|Pa|kPa|MPa|mA|V|Hz|mm|m|kg)',
        r'-?\d+\s*to\s*[+\-]?\d+',                     # -40 to +85

        # Electrical patterns
        r'\d+\s*(VDC|VAC|V\s*DC|V\s*AC)',              # 24 VDC
        r'\d+[-–]\d+\s*mA',                            # 4-20mA
        r'\d+\s*mA(?:\s+HART)?',                       # 20mA, 4-20mA HART
        r'\d+[-–]\d+\s*V',                             # 10-30V

        # Protection ratings
        r'IP\s*\d{2}[A-Z]?',                           # IP67, IP65K
        r'NEMA\s*\d+[A-Z]*',                           # NEMA 4X

        # Safety ratings
        r'SIL\s*[1-4]',                                # SIL 2, SIL 3
        r'ATEX(?:\s+Zone\s*[0-2])?',                   # ATEX, ATEX Zone 1
        r'IECEx',                                      # IECEx
        r'Zone\s*[0-2](?:/[0-2]+)?',                   # Zone 1, Zone 0/1
        r'Class\s*[I]+\s*(?:,\s*)?Div(?:ision)?\s*[1-2]', # Class I Div 1

        # Ratio patterns
        r'\d+\s*:\s*\d+',                              # 100:1 (turn-down ratio)

        # Time patterns
        r'[<>≤≥]?\s*\d+\.?\d*\s*(ms|s|sec|min|hours?)', # <250ms, 1.5s
        r'T\d+\s*[<>≤≥]?\s*\d+\s*(s|ms)',              # T90 < 5s

        # Material specifications
        r'SS\s*\d+L?',                                 # SS316L
        r'316L?(?:\s*SS)?',                            # 316L, 316 SS
        r'304(?:\s*SS)?',                              # 304
        r'Hastelloy\s*[A-Z]?(?:-?\d+)?',               # Hastelloy C-276
        r'Monel(?:\s*\d+)?',                           # Monel 400
        r'Inconel(?:\s*\d+)?',                         # Inconel 625
        r'Titanium',                                   # Titanium
        r'PTFE|PFA|FEP|PEEK',                          # Fluoropolymers
        r'Viton|Kalrez|EPDM|NBR|FKM',                  # Elastomers

        # Standards references
        r'IEC\s*\d+',                                  # IEC 61508
        r'ISO\s*\d+',                                  # ISO 17025
        r'API\s*\d+',                                  # API 551
        r'ASME\s*[A-Z]+\s*\d*',                        # ASME B16
        r'ANSI[/\s]*\w+\s*\d+',                        # ANSI/ISA 82

        # Connection types
        r'\d+/\d+\s*(?:NPT|BSP|BSPT|BSPP)',            # 1/2 NPT
        r'DN\s*\d+',                                   # DN50
        r'G\s*\d+/\d+',                                # G1/2
        r'Tri-?[Cc]lamp',                              # Tri-Clamp
        r'(?:RF|RTJ|FF)\s*[Ff]lange',                  # RF Flange
        r'\d+"\s*(?:ANSI|PN)\s*\d+',                   # 2" ANSI 150

        # Communication protocols
        r'HART(?:\s*\d+)?',                            # HART, HART 7
        r'Modbus(?:\s*RTU|TCP)?',                      # Modbus RTU
        r'Profibus(?:\s*PA|DP)?',                      # Profibus PA
        r'Foundation\s*Fieldbus',                      # Foundation Fieldbus
        r'PROFINET',                                   # PROFINET
        r'EtherNet/IP',                                # EtherNet/IP
        r'4[-–]20\s*mA(?:\s+HART)?',                   # 4-20mA signal

        # Specific value formats
        r'\d+\.\d+\s*%(?:\s*(?:FS|span|URL))?',        # 0.075% FS
        r'\d+\s*°[CF]',                                # 85°C

        # Wire configurations
        r'[234]\s*-?\s*[Ww]ire',                       # 2-wire, 4 wire

        # Sensor types
        r'(?:Type\s*)?[JKTRSNEB](?:\s+thermocouple)?', # Type K
        r'Pt\s*\d+',                                   # Pt100
        r'RTD',                                        # RTD
    ]

    # Patterns indicating a value IS a DESCRIPTION (needs re-extraction)
    # EXPANDED: Added many more patterns to catch descriptive text
    DESCRIPTION_PATTERNS = [
        # Uncertainty phrases
        r'\b(depends?\s+(?:on|upon))',
        r'\b(varies?\s+(?:by|with|depending|based))',
        r'\b(according\s+to)',
        r'\b(subject\s+to)',
        r'\b(based\s+on(?:\s+the)?)',

        # Reference/instruction phrases
        r'\b(refer\s+to)',
        r'\b(see\s+(?:the|page|section|datasheet|manual|table))',
        r'\b(check\s+(?:with|the))',
        r'\b(contact\s+(?:the\s+)?(?:manufacturer|supplier|vendor))',
        r'\b(consult\s+(?:the|your))',
        r'\b(please\s+(?:refer|see|check|contact))',
        r'\b(for\s+more\s+(?:information|details))',
        r'\b(visit\s+(?:the|our))',

        # Uncertainty modifiers
        r'\b(typically|usually|generally|normally|often|sometimes)\b',
        r'\b(approximately|about|around|roughly)\s+(?![±\+\-]?\d)',  # "about 5" but not "about ±5%"
        r'\b(may|might|could|can)\s+(?:be|vary|differ|range)',
        r'\b(should\s+be)',
        r'\b(must\s+be)',
        r'\b(is\s+(?:determined|selected|specified)\s+by)',

        # Placeholder/unknown
        r'\b(TBD|TBC|T\.B\.D\.|T\.B\.C\.)\b',
        r'\b(N/?A|n/?a)\b',
        r'\b(not\s+(?:applicable|available|specified|defined|provided|found|mentioned))',
        r'\b(to\s+be\s+(?:determined|confirmed|specified))',
        r'\b(upon\s+request)',
        r'\b(available\s+(?:on|upon)\s+request)',
        r'\b(as\s+per\s+(?:customer|project|order))',
        r'\b(customer\s+specific)',
        r'\b(project\s+specific)',
        r'\b(application\s+specific)',
        r'\b(model\s+dependent)',

        # Options/configuration phrases
        r'\b(available\s+options?\s+(?:are|include))',
        r'\b(can\s+be\s+(?:configured|selected|specified|ordered))',
        r'\b(options?\s+(?:are|include))',
        r'\b(standard\s+options?\s+(?:are|include))',

        # Sentence starters (likely descriptions, not values)
        r'^(the\s+\w+\s+(?:is|are|has|was|were)\s+)',
        r'^(it\s+(?:is|has|can|should|was)\s+)',
        r'^(this\s+(?:is|has|should|was)\s+)',
        r'^(that\s+(?:is|has|should|was)\s+)',
        r'^(there\s+(?:is|are)\s+)',
        r'^(value\s+(?:is|can|should)\s+)',

        # Field name prefixes in value (LLM artifact)
        r'^(accuracy\s+(?:is|:)\s*)',
        r'^(range\s+(?:is|:)\s*)',
        r'^(output\s+(?:is|signal\s+is|:)\s*)',
        r'^(temperature\s+(?:is|range\s+is|:)\s*)',
        r'^(pressure\s+(?:is|range\s+is|:)\s*)',

        # LLM template artifacts
        r'extracted\s+value',
        r'consolidated\s+value',
        r'value\s+if\s+applicable',
        r'from\s+the\s+document',
        r'based\s+on\s+(?:the\s+)?(?:document|standard)',
        r'no\s+(?:specific\s+)?(?:information|data|value)',
        r"(?:i\s+)?(?:don't|do\s+not)\s+have",
        r'not\s+(?:explicitly\s+)?(?:mentioned|specified|found|stated)',
        r'information\s+(?:is\s+)?not\s+(?:available|provided)',
        r'i\s+couldn\'?t\s+find',
        r'unable\s+to\s+(?:find|determine|extract)',

        # JSON/structure artifacts
        r'\{[\'"]?value[\'"]?\s*:',
        r'[\'"]confidence[\'"]?\s*:',
        r'\[source:',

        # Sentence ending (values shouldn't end with period unless abbreviation)
        r'(?<![A-Z])\.\s*$',  # Period at end, but not after abbreviation like "etc."
    ]

    # Patterns that indicate SENTENCE STRUCTURE (not a value)
    SENTENCE_STRUCTURE_PATTERNS = [
        r'^(the|a|an|this|that|it|there)\s+\w+\s+(is|are|has|have|was|were|can|will|should)',
        r'\b(is|are|has|have|was|were)\s+(a|an|the|very|quite|rather)\s+',
        r'^(i|we|you|they)\s+',
        r'^(what|how|when|where|why|which|who)\b',  # Question words
    ]

    def __init__(self):
        """Initialize the Spec Verifier Agent."""
        # Compile patterns for efficiency
        self.tech_patterns = [re.compile(p, re.IGNORECASE) for p in self.TECHNICAL_PATTERNS]
        self.desc_patterns = [re.compile(p, re.IGNORECASE) for p in self.DESCRIPTION_PATTERNS]
        self.sentence_patterns = [re.compile(p, re.IGNORECASE) for p in self.SENTENCE_STRUCTURE_PATTERNS]
        logger.info("[SpecVerifierAgent] Initialized with enhanced pattern matching")

    def verify_value(self, key: str, value: Any) -> Tuple[bool, str]:
        """
        Verify if a value is a technical specification or description.

        FIXED: Uses content-based validation instead of length-based heuristics.

        Args:
            key: The specification field name
            value: The extracted value

        Returns:
            Tuple of (is_technical_spec, classification)
            - (True, "technical_spec") -> keep value as-is
            - (False, "description") -> needs re-extraction
            - (False, "empty") -> no value
            - (False, "placeholder") -> LLM placeholder text
        """
        # Handle None/empty
        if value is None:
            return False, "empty"

        # Get string value
        if isinstance(value, dict):
            value_str = str(value.get("value", ""))
        else:
            value_str = str(value)

        value_str = value_str.strip()

        # Handle empty strings
        if not value_str or value_str.lower() in ["", "null", "none", "n/a", "-", "—"]:
            return False, "empty"

        # Check for description patterns FIRST (higher priority)
        for pattern in self.desc_patterns:
            if pattern.search(value_str):
                logger.debug(f"[SpecVerifier] Description pattern found in '{key}': {value_str[:50]}")
                return False, "description"

        # Check for sentence structure (subject + verb = likely description)
        for pattern in self.sentence_patterns:
            if pattern.search(value_str):
                logger.debug(f"[SpecVerifier] Sentence structure in '{key}': {value_str[:50]}")
                return False, "description"

        # Check for technical spec patterns
        has_technical_pattern = False
        for pattern in self.tech_patterns:
            if pattern.search(value_str):
                has_technical_pattern = True
                logger.debug(f"[SpecVerifier] Tech pattern found in '{key}': {value_str[:50]}")
                break

        if has_technical_pattern:
            # Additional check: if it has technical pattern but is very long, might still be description
            if len(value_str) > 100:
                # Check word count - technical values usually have few words
                word_count = len(value_str.split())
                if word_count > 15:
                    logger.debug(f"[SpecVerifier] Long text despite tech pattern in '{key}'")
                    return False, "description"
            return True, "technical_spec"

        # CONTENT-BASED HEURISTICS (replaced length-based)

        # Check if it contains instructional words
        instructional_words = ['please', 'contact', 'consult', 'refer', 'check', 'see', 'visit', 'review']
        if any(word in value_str.lower() for word in instructional_words):
            return False, "description"

        # Check if it has too many words (technical values are concise)
        word_count = len(value_str.split())
        if word_count > 8:
            logger.debug(f"[SpecVerifier] Too many words ({word_count}) in '{key}'")
            return False, "description"

        # Check for common verb patterns that indicate description
        verb_patterns = r'\b(is|are|has|have|was|were|can|could|should|would|will|may|might)\b'
        if re.search(verb_patterns, value_str):
            # Has verb but no technical pattern - likely description
            if not has_technical_pattern:
                logger.debug(f"[SpecVerifier] Has verb without tech pattern in '{key}'")
                return False, "description"

        # Short values with numbers might be technical (e.g., "2-wire", "Type K")
        if re.search(r'\d', value_str) and word_count <= 4:
            # Has number and is short - likely a spec
            return True, "technical_spec"

        # Very short values without numbers - might be codes (e.g., "HART", "NPT")
        if len(value_str) <= 15 and word_count <= 2:
            # Could be a code or abbreviation
            if value_str.isupper() or re.match(r'^[A-Z][a-z]+$', value_str):
                return True, "technical_spec"

        # Default: if doesn't match any clear pattern, treat as description
        # This is more conservative - better to re-extract than keep bad data
        logger.debug(f"[SpecVerifier] No clear pattern for '{key}', treating as description")
        return False, "description"

    def verify_all_specs(self, specs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Verify all specifications in a dictionary.

        Args:
            specs: Dictionary of specifications (can be nested sections)

        Returns:
            Dictionary with verification results for each field
        """
        results = {}

        for section_name, section_data in specs.items():
            if isinstance(section_data, dict):
                for key, value in section_data.items():
                    is_spec, classification = self.verify_value(key, value)

                    results[f"{section_name}.{key}"] = {
                        "original_value": value,
                        "is_spec": is_spec,
                        "classification": classification
                    }

        # Log summary
        total = len(results)
        specs_count = sum(1 for r in results.values() if r["is_spec"])
        desc_count = sum(1 for r in results.values() if r["classification"] == "description")

        logger.info(f"[SpecVerifier] Verified {total} fields: {specs_count} specs, {desc_count} descriptions")

        return results


# =============================================================================
# DESCRIPTION EXTRACTOR AGENT
# =============================================================================

class DescriptionExtractorAgent:
    """
    Specialized agent for extracting actual technical values from description text.
    Uses focused LLM prompt to find specific value within description context.

    FIXED: Enhanced prompt to enforce VALUE-ONLY output format.
    """

    def __init__(self, llm=None):
        """Initialize the Description Extractor Agent."""
        if llm is None and create_llm_with_fallback:
            self.llm = create_llm_with_fallback(
                temperature=0.1,
                primary_timeout=30.0,
                fallback_timeout=45.0
            )
        else:
            self.llm = llm

        logger.info("[DescriptionExtractorAgent] Initialized")

    def extract_value_from_description(
        self,
        field_name: str,
        description: str,
        document_context: str,
        product_type: str
    ) -> Optional[str]:
        """
        Given a description, extract the actual technical value.

        FIXED: Enhanced prompt with strict VALUE-ONLY format and post-processing.

        Args:
            field_name: The specification field name (e.g., "accuracy")
            description: The descriptive text to extract from
            document_context: Relevant standards document content
            product_type: The product type for context

        Returns:
            Extracted technical value string, or None if not found
        """
        if not self.llm:
            logger.warning("[DescriptionExtractor] No LLM available")
            return None

        # FIXED: Enhanced prompt with strict VALUE-ONLY format
        prompt = f"""You are a Technical Value Extractor for industrial instrumentation.

TASK: Extract the EXACT technical value for "{field_name}" from the document.

PRODUCT TYPE: {product_type}

CRITICAL RULES:
1. Return ONLY the technical value - no descriptions, no explanations
2. NEVER include words like: "is", "are", "the", "typically", "usually", "approximately"
3. NEVER include the field name in your response
4. Strip any prefixes like "The accuracy is", "Output signal:", etc.

CORRECT OUTPUT FORMATS:
- Accuracy: "±0.1%" or "±0.5% FS" (NOT "The accuracy is ±0.1%")
- Range: "-40 to 85°C" or "0-400 bar" (NOT "Temperature range is -40 to 85°C")
- Rating: "IP67", "SIL 2", "ATEX Zone 1" (NOT "Protection rating is IP67")
- Signal: "4-20mA HART" (NOT "Output signal is 4-20mA with HART")
- Material: "316L SS" (NOT "The wetted parts are made of 316L SS")

CURRENT DESCRIPTION (this is NOT a proper value):
"{description}"

DOCUMENT CONTENT:
{document_context[:6000]}

If no specific value can be found, respond with exactly: NOT_FOUND

EXTRACTED VALUE (ONLY the technical value, nothing else):"""

        try:
            response = self.llm.invoke(prompt)

            if hasattr(response, 'content'):
                result = response.content.strip()
            else:
                result = str(response).strip()

            # Clean up result
            result = result.replace('"', '').replace("'", "").strip()

            # Remove any markdown
            result = re.sub(r'\*+', '', result)
            result = re.sub(r'`+', '', result)
            result = result.strip()

            # Check for NOT_FOUND
            if result.upper() == "NOT_FOUND" or "not_found" in result.lower() or "not found" in result.lower():
                logger.debug(f"[DescriptionExtractor] No value found for: {field_name}")
                return None

            # FIXED: Use normalizer for post-processing if available
            if normalize_spec_value and is_valid_spec_value:
                normalized, refs = normalize_spec_value(result, field_name)

                if normalized and is_valid_spec_value(normalized):
                    logger.info(f"[DescriptionExtractor] Extracted '{field_name}': {normalized}")
                    return normalized
                else:
                    logger.debug(f"[DescriptionExtractor] Normalized value invalid: {normalized[:50] if normalized else 'empty'}")
                    return None
            else:
                # Fallback: use SpecVerifierAgent
                verifier = SpecVerifierAgent()
                is_spec, classification = verifier.verify_value(field_name, result)

                if is_spec:
                    logger.info(f"[DescriptionExtractor] Extracted '{field_name}': {result}")
                    return result
                else:
                    logger.debug(f"[DescriptionExtractor] Extraction not a spec: {result[:50]}")
                    return None

        except Exception as e:
            logger.error(f"[DescriptionExtractor] Error: {e}")
            return None


# =============================================================================
# DUAL-STEP VERIFICATION ORCHESTRATION
# =============================================================================

def verify_and_reextract_specs(
    extracted_specs: Dict[str, Any],
    document_content: str,
    product_type: str,
    llm=None
) -> Dict[str, Any]:
    """
    Dual-step verification and re-extraction for specifications.

    FIXED: Integrated with ValueNormalizer for consistent post-processing.

    For each extracted spec:
    1. Normalize the value using ValueNormalizer
    2. Verify if it's a technical spec or description
    3. If description, attempt re-extraction with focused LLM prompt
    4. If re-extraction fails, remove the field (don't include invalid specs)

    Args:
        extracted_specs: The extracted specifications dictionary
        document_content: The standards document content for re-extraction
        product_type: The product type
        llm: Optional LLM instance

    Returns:
        Clean specifications with only verified technical values
    """
    logger.info(f"[DualStepVerify] Starting verification for: {product_type}")

    verifier = SpecVerifierAgent()
    description_extractor = DescriptionExtractorAgent(llm=llm)

    specs = extracted_specs.get("specifications", {})
    clean_specs = {}

    stats = {"kept": 0, "normalized": 0, "reextracted": 0, "removed": 0}

    for section in ["mandatory", "optional", "safety", "environmental"]:
        section_data = specs.get(section, {})
        if not isinstance(section_data, dict):
            continue

        clean_section = {}

        for key, value_data in section_data.items():
            # Get the value string
            if isinstance(value_data, dict):
                value = value_data.get("value", "")
                original_data = value_data
            else:
                value = value_data
                original_data = {"value": value_data}

            # Step 1: Normalize the value if normalizer is available
            normalized_value = str(value)
            standards_refs = []

            if normalize_spec_value:
                normalized_value, standards_refs = normalize_spec_value(str(value), key)

            # Step 2: Verify (use normalized value)
            is_spec, classification = verifier.verify_value(key, normalized_value)

            # Step 3: Additional validation with normalizer
            is_valid = True
            if is_valid_spec_value:
                is_valid = is_valid_spec_value(normalized_value) if normalized_value else False

            if is_spec and is_valid and normalized_value:
                # Technical spec - keep (with normalization applied)
                if normalized_value != str(value):
                    # Value was normalized
                    clean_section[key] = {
                        "value": normalized_value,
                        "source": original_data.get("source", "standards_specifications"),
                        "confidence": original_data.get("confidence", 0.85),
                        "standards_referenced": standards_refs or original_data.get("standards_referenced", [])
                    }
                    stats["normalized"] += 1
                else:
                    # Value was already clean
                    clean_section[key] = original_data
                    stats["kept"] += 1

            elif classification == "description" or not is_valid:
                # Description or invalid - attempt re-extraction
                new_value = description_extractor.extract_value_from_description(
                    field_name=key,
                    description=str(value),
                    document_context=document_content,
                    product_type=product_type
                )

                if new_value:
                    # Normalize re-extracted value
                    new_normalized = new_value
                    new_refs = []

                    if normalize_spec_value:
                        new_normalized, new_refs = normalize_spec_value(new_value, key)

                    # Validate re-extracted value
                    is_new_valid = True
                    if is_valid_spec_value:
                        is_new_valid = is_valid_spec_value(new_normalized) if new_normalized else False

                    if is_new_valid and new_normalized:
                        # Successful re-extraction
                        clean_section[key] = {
                            "value": new_normalized,
                            "source": "re-extracted_from_standards",
                            "confidence": 0.75,
                            "standards_referenced": new_refs,
                            "original_description": str(value)[:100]
                        }
                        stats["reextracted"] += 1
                    else:
                        # Re-extraction produced invalid value
                        stats["removed"] += 1
                        logger.debug(f"[DualStepVerify] Re-extracted value invalid: {key}")
                else:
                    # Re-extraction failed - remove field
                    stats["removed"] += 1
                    logger.debug(f"[DualStepVerify] Removed invalid field: {key}")

            else:
                # Empty or other - skip
                stats["removed"] += 1

        if clean_section:
            clean_specs[section] = clean_section

    logger.info(
        f"[DualStepVerify] Complete: "
        f"{stats['kept']} kept, {stats['normalized']} normalized, "
        f"{stats['reextracted']} re-extracted, {stats['removed']} removed"
    )

    # Return updated specs
    result = extracted_specs.copy()
    result["specifications"] = clean_specs
    result["_verification_stats"] = stats

    return result


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "SpecVerifierAgent",
    "DescriptionExtractorAgent",
    "verify_and_reextract_specs",
]
