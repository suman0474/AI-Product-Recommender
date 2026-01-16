"""
Deep Agent Sub-Agents: Extractor and Critic

This module implements the LLM-powered sub-agents for extracting and validating
technical specifications from standards documents.

Architecture:
- ExtractorAgent: Extracts JSON technical specifications from standards documents
- CriticAgent: Validates extracted data is proper JSON format
- Extraction loop: If Critic rejects, Extractor retries with feedback

"""

import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

# Import LLM with fallback
try:
    from llm_fallback import create_llm_with_fallback
except ImportError:
    create_llm_with_fallback = None


# =============================================================================
# SCHEMA FIELD DEFINITIONS
# Maps product types to their expected schema fields for extraction
# =============================================================================

PRODUCT_TYPE_SCHEMA_FIELDS = {
    "pressure transmitter": {
        "mandatory": [
            "accuracy", "measurementRange", "pressureType", "outputSignal",
            "powerSupply", "processConnection", "wettedPartsMaterial",
            "housingMaterial", "ingressProtection", "hazardousAreaRating",
            "ambientTemperatureRange", "processTemperatureRange"
        ],
        "optional": [
            "repeatability", "longTermStability", "responseTime", "turndownRatio",
            "communicationProtocol", "displayOptions", "remoteSealOptions",
            "calibrationCertificate", "warrantyPeriod", "diagnostics"
        ]
    },
    "temperature sensor": {
        "mandatory": [
            "accuracy", "accuracyClass", "sensorType", "measurementRange",
            "outputSignalType", "powerSupplyRange", "processConnectionMethod",
            "wettedPartsMaterial", "connectionHeadMaterial", "probeDiameter",
            "insertionLengthRange", "wiringConfiguration"
        ],
        "optional": [
            "responseRate", "sensorElementConfiguration", "thermalResponseTime",
            "insulationResistance", "vibrationResistance", "thermowellType",
            "calibrationCertificateOptions", "diagnosticFunctions"
        ]
    },
    "thermocouple": {
        "mandatory": [
            "accuracy", "junctionType", "sensorType", "temperatureRange",
            "insulationResistance", "leadWireType", "outputSignal",
            "probeDiameter", "probeLength", "processConnection", "sheathMaterial"
        ],
        "optional": [
            "coldJunctionCompensation", "responseTime", "transmitterType",
            "enclosureType", "housingMaterial", "mountingOptions",
            "calibrationFrequency", "certifications"
        ]
    },
    "flow meter": {
        "mandatory": [
            "accuracy", "measurementType", "flowRange", "outputSignal",
            "powerSupply", "processConnection", "lineSize", "wettedMaterials",
            "fluidTypes", "pressureRating", "temperatureRating"
        ],
        "optional": [
            "repeatability", "turndownRatio", "pulsedOutput", "displayOptions",
            "communicationProtocol", "diagnostics", "batchControl"
        ]
    },
    "level transmitter": {
        "mandatory": [
            "accuracy", "measurementType", "levelRange", "outputSignal",
            "powerSupply", "processConnection", "wettedMaterials",
            "operatingTemperature", "operatingPressure", "ingressProtection"
        ],
        "optional": [
            "emptyTankDetection", "foamDetection", "interfaceDetection",
            "displayOptions", "calibrationMethod", "diagnostics"
        ]
    },
    "control valve": {
        "mandatory": [
            "valveType", "sizeRange", "pressureClass", "bodyMaterial",
            "trimMaterial", "packingType", "flowCharacteristic",
            "actuatorType", "actionType", "signalInput"
        ],
        "optional": [
            "positioner", "feedbackType", "handwheel", "limitSwitches",
            "fireRating", "lowEmissionPacking"
        ]
    },
    # IMPROVED (v2): Added 10+ new product types for better coverage
    "differential pressure transmitter": {
        "mandatory": [
            "accuracy", "measurementRange", "pressureType", "outputSignal",
            "powerSupply", "processConnection", "wettedPartsMaterial",
            "housingMaterial", "ingressProtection", "hazardousAreaRating",
            "ambientTemperatureRange", "processTemperatureRange", "isolationMethod"
        ],
        "optional": [
            "repeatability", "responseTime", "communicationProtocol",
            "snubbingOptions", "diagnostics", "remoteCalibration", "coatingOptions"
        ]
    },
    "mass flow meter": {
        "mandatory": [
            "accuracy", "measurementType", "flowRange", "outputSignal",
            "powerSupply", "processConnection", "wettedMaterials",
            "fluidTypes", "operatingTemperature", "operatingPressure"
        ],
        "optional": [
            "repeatability", "responseTime", "densityMeasurement", "temperatureCompensation",
            "communicationProtocol", "displayOptions", "diagnostics"
        ]
    },
    "ultrasonic flow meter": {
        "mandatory": [
            "accuracy", "measurementType", "flowRange", "outputSignal",
            "powerSupply", "pipeSize", "pipelineType", "fluidType",
            "operatingTemperature", "operatingPressure"
        ],
        "optional": [
            "repeatability", "transducerType", "signalFrequency", "communicationProtocol",
            "dataLogging", "batteryLife", "diagnostics", "bypassProvisioning"
        ]
    },
    "electromagnetic flow meter": {
        "mandatory": [
            "accuracy", "lineSize", "flowRange", "outputSignal",
            "powerSupply", "pipelineType", "electrodeType",
            "liningMaterial", "operatingTemperature", "operatingPressure"
        ],
        "optional": [
            "repeatability", "conductivityRange", "communicationProtocol",
            "batteryOptions", "displayOptions", "diagnostics", "wiringOptions"
        ]
    },
    "rotameter (variable area)": {
        "mandatory": [
            "accuracy", "measurementRange", "floatMaterial", "tubeMaterial",
            "fluidTypes", "outputSignal", "pressureRating", "temperatureRating",
            "connectionType", "scaleOptions"
        ],
        "optional": [
            "stopcock", "snubbingValve", "integralValve", "illumination",
            "remoteTransmitterOptions", "protectionTubes", "certifications"
        ]
    },
    "turbine flow meter": {
        "mandatory": [
            "accuracy", "flowRange", "rotorType", "bearingType", "outputSignal",
            "powerSupply", "wettedMaterials", "fluidTypes",
            "operatingTemperature", "operatingPressure"
        ],
        "optional": [
            "repeatability", "responseTime", "communicationProtocol",
            "counterType", "displayOptions", "diagnostics", "filterRequirements"
        ]
    },
    "capacitive level transmitter": {
        "mandatory": [
            "accuracy", "measurementType", "levelRange", "outputSignal",
            "powerSupply", "processConnection", "wettedMaterials",
            "probeLength", "operatingTemperature", "operatingPressure"
        ],
        "optional": [
            "foamDetection", "interfaceDetection", "emptyTankDetection",
            "autoTuning", "communicationProtocol", "displayOptions", "diagnostics"
        ]
    },
    "radar level transmitter": {
        "mandatory": [
            "accuracy", "measurementType", "levelRange", "outputSignal",
            "powerSupply", "antennaType", "antennaLength", "mountingType",
            "operatingTemperature", "operatingPressure"
        ],
        "optional": [
            "frequencyBand", "foamCompensation", "multipleProducts", "communicationProtocol",
            "temperatureCompensation", "diagnostics", "displayOptions"
        ]
    },
    "ultrasonic level transmitter": {
        "mandatory": [
            "accuracy", "measurementType", "levelRange", "outputSignal",
            "powerSupply", "operatingTemperature", "operatingPressure",
            "sensorFrequency", "deadBand", "processConnection"
        ],
        "optional": [
            "temperatureCompensation", "foamDetection", "communicationProtocol",
            "displayOptions", "diagnostics", "mountingOptions"
        ]
    },
    "vibrating level switch": {
        "mandatory": [
            "switchType", "outputSignal", "powerSupply", "sensorFrequency",
            "wettedMaterials", "operatingTemperature", "operatingPressure",
            "processConnection", "dampeningAdjustment"
        ],
        "optional": [
            "repeatability", "responseTime", "ledIndicators", "certificationsAvailable",
            "mountingOptions", "wiringOptions"
        ]
    },
    "analyzer (pH/ORP)": {
        "mandatory": [
            "accuracy", "measurementRange", "sensorType", "outputSignal",
            "powerSupply", "processConnection", "flowRate", "temperature",
            "responseTime", "maintenanceInterval"
        ],
        "optional": [
            "bufferSets", "calibrationMethod", "displayOptions", "alarmContacts",
            "communicationProtocol", "diagnostics", "temperatureCompensation"
        ]
    },
    "conductivity transmitter": {
        "mandatory": [
            "accuracy", "measurementRange", "cellConstant", "outputSignal",
            "powerSupply", "processConnection", "electrodeType",
            "operatingTemperature", "operatingPressure", "responseTime"
        ],
        "optional": [
            "temperatureCompensation", "cellMaterial", "communicationProtocol",
            "displayOptions", "diagnostics", "calibrationMethod"
        ]
    },
    "dissolved oxygen transmitter": {
        "mandatory": [
            "accuracy", "measurementRange", "sensorType", "outputSignal",
            "powerSupply", "processConnection", "operatingTemperature",
            "operatingPressure", "responseTime", "maintenanceFrequency"
        ],
        "optional": [
            "membraneType", "electrolyteMaterial", "pressureCompensation",
            "temperatureCompensation", "communicationProtocol", "diagnostics"
        ]
    }
}

# Default schema fields for unknown product types
# IMPROVED (v2): Expanded from 9 to 14 mandatory fields for better unknown product coverage
DEFAULT_SCHEMA_FIELDS = {
    "mandatory": [
        "accuracy", "measurementRange", "outputSignal", "powerSupply",
        "processConnection", "materials", "ingressProtection",
        "temperatureRange", "hazardousAreaRating",
        # New additions for better coverage
        "operatingPressure", "wettedMaterials", "responseTime", "sensorType", "displayOptions"
    ],
    "optional": [
        "repeatability", "responseTime", "communicationProtocol",
        "displayOptions", "diagnostics", "calibration", "certifications",
        # New additions
        "housingMaterial", "ambientTemperature", "shockAndVibration",
        "installationRequirements", "maintenanceInterval", "warrantyPeriod",
        "approvals", "testCertificates"
    ]
}


def get_schema_fields_for_product(product_type: str) -> Dict[str, List[str]]:
    """Get schema fields for a product type."""
    product_lower = product_type.lower()
    
    for key, fields in PRODUCT_TYPE_SCHEMA_FIELDS.items():
        if key in product_lower:
            return fields
    
    # Partial match
    for key, fields in PRODUCT_TYPE_SCHEMA_FIELDS.items():
        key_words = key.split()
        if any(word in product_lower for word in key_words):
            return fields
    
    return DEFAULT_SCHEMA_FIELDS


# =============================================================================
# EXTRACTOR SUB-AGENT
# =============================================================================

class ExtractorAgent:
    """
    LLM-powered agent that extracts technical specifications from standards documents
    and formats them as JSON key-value pairs.
    """
    
    def __init__(self, llm=None):
        """Initialize the Extractor Agent."""
        if llm is None and create_llm_with_fallback:
            self.llm = create_llm_with_fallback(
                temperature=0.1,  # Low temperature for precise extraction
                primary_timeout=60.0,
                fallback_timeout=90.0
            )
        else:
            self.llm = llm
        
        logger.info("[ExtractorAgent] Initialized")
    
    def build_extraction_prompt(
        self,
        product_type: str,
        document_content: str,
        schema_fields: Dict[str, List[str]],
        user_context: Optional[Dict[str, Any]] = None,
        critic_feedback: Optional[str] = None
    ) -> str:
        """Build the extraction prompt for the LLM."""
        
        mandatory_fields = schema_fields.get("mandatory", [])
        optional_fields = schema_fields.get("optional", [])
        
        # Build user context section
        context_section = ""
        if user_context:
            context_parts = []
            if user_context.get("domain"):
                context_parts.append(f"Domain: {user_context['domain']}")
            if user_context.get("safety_requirements"):
                safety = user_context["safety_requirements"]
                if safety.get("sil_level"):
                    context_parts.append(f"Safety Level: {safety['sil_level']}")
                if safety.get("atex_zone"):
                    context_parts.append(f"ATEX Zone: {safety['atex_zone']}")
            if context_parts:
                context_section = "User Requirements Context:\n" + "\n".join(f"- {p}" for p in context_parts)
        
        # Build critic feedback section
        feedback_section = ""
        if critic_feedback:
            feedback_section = f"""
IMPORTANT: Previous extraction was rejected. Fix the following issues:
{critic_feedback}

Ensure your output is VALID JSON with no syntax errors.
"""
        
        prompt = f"""You are a Technical Specifications Extractor for industrial instrumentation.

TASK: Extract technical specifications for "{product_type}" from the standards document content below.

OUTPUT FORMAT: You MUST output ONLY a valid JSON object with this exact structure:
{{
    "product_type": "{product_type}",
    "specifications": {{
        "mandatory": {{
            // Each field should have this structure:
            // "fieldName": {{
            //     "value": "actual value",
            //     "source": "standards_specifications",
            //     "confidence": 0.9,
            //     "standards_referenced": ["IEC 60584", "ATEX"]
            // }}
        }},
        "optional": {{
            // Same structure as mandatory
        }},
        "safety": {{
            // Safety-related specifications with same structure
        }},
        "certifications": [
            // List of applicable certifications
        ]
    }},
    "applicable_standards": [
        // List of relevant standard codes (e.g., "IEC 61508", "API 551")
    ],
    "guidelines": [
        // Key installation/usage guidelines from the document
    ],
    "extraction_confidence": 0.0-1.0
}}

MANDATORY FIELDS TO EXTRACT (find values in the document):
{', '.join(mandatory_fields)}

OPTIONAL FIELDS TO EXTRACT (if found in document):
{', '.join(optional_fields)}

{context_section}
{feedback_section}

=== CRITICAL RULES FOR VALUES ===
Return ONLY the technical value - NO descriptions, NO explanations, NO sentences.

FORBIDDEN PATTERNS IN VALUES (NEVER include these):
- Words like: "typically", "usually", "generally", "approximately", "should be", "must be"
- Phrases like: "depends on", "varies by", "according to", "based on", "is defined as"
- Sentence starters: "The X is", "It has", "This should be", "According to", "Per standard"
- Explanatory text: "which means", "for applications", "in order to", "when used in"
- Vague terms: "appropriate", "suitable", "adequate", "sufficient", "as needed"

CORRECT VALUE EXAMPLES:
- accuracy: "±0.075%" (NOT "The accuracy is typically ±0.075% as defined by...")
- measurementRange: "-40°C to +85°C" (NOT "Temperature range varies from -40°C to +85°C depending on...")
- outputSignal: "4-20 mA HART" (NOT "The output signal should be 4-20 mA with HART protocol support")
- pressureRating: "PN40" (NOT "Pressure rating must be PN40 or higher for this application")
- ingressProtection: "IP67" (NOT "IP67 protection is required for outdoor installations")

WRONG VALUE EXAMPLES (NEVER OUTPUT THESE):
- "typically ±0.04% with digital communication" -> WRONG (contains "typically")
- "The pressure range should be 0-400 bar" -> WRONG (sentence format)
- "Class A per IEC 60770 standard requirements" -> WRONG (contains explanatory text)
- "4-20mA with optional HART, suitable for process control" -> WRONG (contains description)

RULES:
1. Extract ONLY the technical value - strip ALL surrounding context
2. Standards references go in "standards_referenced" array, NOT in the value
3. For ranges, use format like "-40°C to +85°C" or "0-400 bar"
4. For accuracy, use format like "±0.075%" or "Class A"
5. If a field is not found, do NOT include it (don't use "Not specified")
6. Output ONLY the JSON object, no markdown, no explanation

STANDARDS DOCUMENT CONTENT:
{document_content[:40000]}
"""
        return prompt
    
    def extract(
        self,
        product_type: str,
        document_content: str,
        user_context: Optional[Dict[str, Any]] = None,
        critic_feedback: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract technical specifications from document content.
        
        Args:
            product_type: The product type to extract specs for
            document_content: The standards document text
            user_context: Optional user context (domain, safety requirements)
            critic_feedback: Optional feedback from Critic for retry
        
        Returns:
            Dictionary with extracted specifications
        """
        logger.info(f"[ExtractorAgent] Extracting specs for: {product_type}")
        
        # Get schema fields for this product type
        schema_fields = get_schema_fields_for_product(product_type)
        
        # Build prompt
        prompt = self.build_extraction_prompt(
            product_type=product_type,
            document_content=document_content,
            schema_fields=schema_fields,
            user_context=user_context,
            critic_feedback=critic_feedback
        )
        
        try:
            if self.llm is None:
                logger.warning("[ExtractorAgent] No LLM available, using fallback extraction")
                return self._fallback_extraction(product_type, document_content)
            
            # Call LLM
            response = self.llm.invoke(prompt)
            
            # Extract content from response
            if hasattr(response, 'content'):
                llm_output = response.content
            else:
                llm_output = str(response)
            
            # Parse JSON from response
            result = self._parse_json_response(llm_output)
            
            if result:
                logger.info(f"[ExtractorAgent] Successfully extracted specs for {product_type}")
                return result
            else:
                logger.warning(f"[ExtractorAgent] Failed to parse JSON, using fallback")
                return self._fallback_extraction(product_type, document_content)
        
        except Exception as e:
            logger.error(f"[ExtractorAgent] Extraction failed: {e}")
            return self._fallback_extraction(product_type, document_content)
    
    def _parse_json_response(self, llm_output: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from LLM response."""
        # Try direct JSON parse
        try:
            return json.loads(llm_output)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code blocks
        json_patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
            r'\{[\s\S]*\}'
        ]
        
        for pattern in json_patterns:
            match = re.search(pattern, llm_output)
            if match:
                try:
                    json_str = match.group(1) if '```' in pattern else match.group(0)
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    continue
        
        return None
    
    def _fallback_extraction(
        self,
        product_type: str,
        document_content: str
    ) -> Dict[str, Any]:
        """Fallback regex-based extraction when LLM fails."""
        from .document_loader import (
            extract_standard_codes,
            extract_certifications,
            extract_specifications,
            extract_guidelines
        )
        from .schema_field_extractor import extract_standards_from_value
        from .value_normalizer import ValueNormalizer

        normalizer = ValueNormalizer()

        specs = extract_specifications(document_content)
        standards = extract_standard_codes(document_content)
        certifications = extract_certifications(document_content)
        guidelines = extract_guidelines(document_content)

        # Build specifications with proper metadata structure
        # Apply normalization to each extracted value
        mandatory_specs = {}
        for k, v in specs.items():
            if not v:
                continue

            # Extract and validate the value using normalizer
            result = normalizer.extract_and_validate(str(v))

            if result["is_valid"]:
                # Use normalized value
                mandatory_specs[k] = {
                    "value": result["normalized_value"],
                    "source": "standards_specifications",
                    "confidence": 0.7,  # Lower confidence for regex fallback
                    "standards_referenced": result["standards_extracted"]
                }
            else:
                # Value failed validation - still include but with lower confidence
                logger.debug(f"[ExtractorAgent] Fallback value for '{k}' failed validation: {v[:50]}")
                # Try to extract just the technical part
                normalized = normalizer.normalize(str(v))
                if normalizer.is_valid_spec_value(normalized):
                    standards_refs = extract_standards_from_value(str(v)) if v else []
                    mandatory_specs[k] = {
                        "value": normalized,
                        "source": "standards_specifications",
                        "confidence": 0.5,  # Even lower confidence
                        "standards_referenced": standards_refs
                    }

        return {
            "product_type": product_type,
            "specifications": {
                "mandatory": mandatory_specs,
                "optional": {},
                "safety": {},
                "certifications": certifications[:10]
            },
            "applicable_standards": standards[:15],
            "guidelines": guidelines[:10],
            "extraction_confidence": 0.5,
            "extraction_method": "regex_fallback"
        }


# =============================================================================
# CRITIC SUB-AGENT
# =============================================================================

class CriticAgent:
    """
    Agent that validates extracted specifications are in proper JSON format
    and contain required fields.
    """
    
    def __init__(self):
        """Initialize the Critic Agent."""
        self.required_top_level_keys = ["product_type", "specifications"]
        self.required_spec_sections = ["mandatory"]
        logger.info("[CriticAgent] Initialized")
    
    def validate(
        self,
        extracted_data: Dict[str, Any],
        product_type: str
    ) -> Tuple[bool, str]:
        """
        Validate extracted data is proper JSON with required structure.
        
        Args:
            extracted_data: The extracted specifications dictionary
            product_type: The product type being extracted
        
        Returns:
            Tuple of (is_valid, feedback_message)
        """
        logger.info(f"[CriticAgent] Validating extraction for: {product_type}")
        
        issues = []
        
        # Check if it's a dictionary
        if not isinstance(extracted_data, dict):
            return False, "Output is not a valid JSON object/dictionary"
        
        # Check required top-level keys
        for key in self.required_top_level_keys:
            if key not in extracted_data:
                issues.append(f"Missing required key: '{key}'")
        
        # Check specifications structure
        specs = extracted_data.get("specifications", {})
        if not isinstance(specs, dict):
            issues.append("'specifications' must be a JSON object")
        else:
            # Check for at least one spec section
            has_specs = False
            for section in ["mandatory", "optional", "safety"]:
                if section in specs and isinstance(specs[section], dict) and specs[section]:
                    has_specs = True
                    break
            
            if not has_specs:
                issues.append("No specifications found in mandatory/optional/safety sections")
        
        # Check product type matches
        if extracted_data.get("product_type", "").lower() != product_type.lower():
            extracted_pt = extracted_data.get("product_type", "None")
            issues.append(f"Product type mismatch: expected '{product_type}', got '{extracted_pt}'")
        
        # Check applicable_standards is a list
        standards = extracted_data.get("applicable_standards")
        if standards is not None and not isinstance(standards, list):
            issues.append("'applicable_standards' must be a list")
        
        # Check individual spec values format
        for section_name, section_data in specs.items():
            if isinstance(section_data, dict):
                for field_name, field_value in section_data.items():
                    if isinstance(field_value, dict):
                        if "value" not in field_value:
                            issues.append(f"Field '{section_name}.{field_name}' missing 'value' key")
                    # String values are also acceptable
        
        # Build result
        if issues:
            feedback = "Validation failed:\n" + "\n".join(f"- {issue}" for issue in issues)
            logger.warning(f"[CriticAgent] Validation failed: {len(issues)} issues")
            return False, feedback
        
        logger.info(f"[CriticAgent] Validation passed")
        return True, "Validation passed"
    
    def get_extraction_quality_score(self, extracted_data: Dict[str, Any]) -> float:
        """Calculate quality score for extracted data."""
        score = 0.0
        
        # Base score for being valid JSON
        if isinstance(extracted_data, dict):
            score += 0.2
        
        # Score for having specifications
        specs = extracted_data.get("specifications", {})
        if specs:
            mandatory = specs.get("mandatory", {})
            optional = specs.get("optional", {})
            
            # Score based on number of fields extracted
            total_fields = len(mandatory) + len(optional)
            if total_fields >= 10:
                score += 0.3
            elif total_fields >= 5:
                score += 0.2
            elif total_fields >= 1:
                score += 0.1
        
        # Score for having standards
        standards = extracted_data.get("applicable_standards", [])
        if len(standards) >= 5:
            score += 0.2
        elif len(standards) >= 1:
            score += 0.1
        
        # Score for having guidelines
        guidelines = extracted_data.get("guidelines", [])
        if len(guidelines) >= 3:
            score += 0.15
        elif len(guidelines) >= 1:
            score += 0.1
        
        # Score for confidence
        confidence = extracted_data.get("extraction_confidence", 0)
        score += confidence * 0.15
        
        return min(score, 1.0)


# =============================================================================
# EXTRACTION LOOP
# =============================================================================

def extract_specifications_with_validation(
    product_type: str,
    document_content: str,
    user_context: Optional[Dict[str, Any]] = None,
    max_retries: int = 3,
    llm=None
) -> Dict[str, Any]:
    """
    Extract specifications with Extractor-Critic loop and dual-step verification.
    
    DUAL-STEP VERIFICATION:
    1. Extractor + Critic loop: Extract JSON specs and validate structure
    2. SpecVerifier: Verify each value is a technical spec vs description
    3. Re-extraction: For descriptions, attempt focused extraction
    
    Args:
        product_type: Product type to extract specs for
        document_content: Standards document content
        user_context: Optional user context
        max_retries: Maximum extraction attempts
        llm: Optional LLM instance
    
    Returns:
        Validated and verified specifications dictionary
    """
    logger.info(f"[ExtractionLoop] Starting extraction for: {product_type}")
    
    extractor = ExtractorAgent(llm=llm)
    critic = CriticAgent()
    
    critic_feedback = None
    best_result = None
    best_score = 0.0
    
    # ===== STEP 1: Extractor-Critic Loop =====
    for attempt in range(max_retries):
        logger.info(f"[ExtractionLoop] Attempt {attempt + 1}/{max_retries}")
        
        # Extract
        extracted = extractor.extract(
            product_type=product_type,
            document_content=document_content,
            user_context=user_context,
            critic_feedback=critic_feedback
        )
        
        # Validate
        is_valid, feedback = critic.validate(extracted, product_type)
        
        # Calculate quality score
        quality_score = critic.get_extraction_quality_score(extracted)
        
        # Keep best result
        if quality_score > best_score:
            best_score = quality_score
            best_result = extracted
        
        if is_valid:
            logger.info(f"[ExtractionLoop] Success on attempt {attempt + 1}, score: {quality_score:.2f}")
            extracted["_extraction_metadata"] = {
                "attempts": attempt + 1,
                "quality_score": quality_score,
                "validated": True,
                "timestamp": datetime.now().isoformat()
            }
            best_result = extracted
            break
        
        # Set feedback for next attempt
        critic_feedback = feedback
        logger.info(f"[ExtractionLoop] Attempt {attempt + 1} failed, retrying...")
    
    # Use best result or fallback
    if best_result is None:
        best_result = extractor._fallback_extraction(product_type, document_content)
    
    if "_extraction_metadata" not in best_result:
        best_result["_extraction_metadata"] = {
            "attempts": max_retries,
            "quality_score": best_score,
            "validated": False,
            "timestamp": datetime.now().isoformat()
        }
    
    # ===== STEP 2: Dual-Step Verification =====
    try:
        from .spec_verifier import verify_and_reextract_specs
        
        logger.info(f"[ExtractionLoop] Running dual-step verification...")
        
        verified_result = verify_and_reextract_specs(
            extracted_specs=best_result,
            document_content=document_content,
            product_type=product_type,
            llm=llm
        )
        
        # Merge verification stats into metadata
        if "_verification_stats" in verified_result:
            best_result["_extraction_metadata"]["verification_stats"] = verified_result.pop("_verification_stats")
        
        best_result["specifications"] = verified_result.get("specifications", {})
        
        logger.info(f"[ExtractionLoop] Dual-step verification complete")
    
    except ImportError:
        logger.warning("[ExtractionLoop] spec_verifier not available, skipping dual-step verification")
    except Exception as e:
        logger.error(f"[ExtractionLoop] Dual-step verification error: {e}")
    
    return best_result


# =============================================================================
# BATCH EXTRACTION FOR MULTIPLE PRODUCTS
# =============================================================================

def extract_specifications_for_products(
    products: List[Dict[str, Any]],
    document_analyses: Dict[str, Any],
    user_context: Optional[Dict[str, Any]] = None,
    llm=None
) -> List[Dict[str, Any]]:
    """
    Extract specifications for multiple products.
    
    Args:
        products: List of products with product_type
        document_analyses: Dictionary of document analyses from memory
        user_context: Optional user context
        llm: Optional LLM instance
    
    Returns:
        List of products with extracted specifications
    """
    logger.info(f"[BatchExtraction] Processing {len(products)} products")
    
    results = []
    
    for product in products:
        product_type = product.get("product_type") or product.get("name", "Unknown")
        
        # Get relevant document content
        relevant_docs = product.get("relevant_documents", [])
        
        # Combine content from relevant documents
        combined_content = ""
        for doc_name in relevant_docs:
            if doc_name in document_analyses:
                analysis = document_analyses[doc_name]
                combined_content += analysis.get("full_content", "")[:5000] + "\n\n"
        
        # If no relevant docs, use all documents
        if not combined_content:
            for analysis in list(document_analyses.values())[:3]:
                combined_content += analysis.get("full_content", "")[:5000] + "\n\n"
        
        # Extract with validation
        extracted = extract_specifications_with_validation(
            product_type=product_type,
            document_content=combined_content,
            user_context=user_context,
            llm=llm
        )
        
        # Merge with original product
        enriched_product = product.copy()
        enriched_product["extracted_specifications"] = extracted.get("specifications", {})
        enriched_product["applicable_standards"] = extracted.get("applicable_standards", [])
        enriched_product["guidelines"] = extracted.get("guidelines", [])
        enriched_product["extraction_metadata"] = extracted.get("_extraction_metadata", {})
        
        results.append(enriched_product)
    
    logger.info(f"[BatchExtraction] Completed {len(results)} products")
    return results


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ExtractorAgent",
    "CriticAgent",
    "extract_specifications_with_validation",
    "extract_specifications_for_products",
    "get_schema_fields_for_product",
    "PRODUCT_TYPE_SCHEMA_FIELDS",
]
