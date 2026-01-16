# agentic/deep_agent/value_normalizer.py
# =============================================================================
# VALUE NORMALIZER - Centralized Specification Value Cleaning
# =============================================================================
#
# This module provides post-processing for LLM-extracted specification values
# to ensure they are clean technical specifications, not descriptive text.
#
# The normalizer:
# 1. Strips leading phrases like "The accuracy is", "Output signal:"
# 2. Strips trailing phrases like "per IEC 60770", "calibrated at 25°C"
# 3. Extracts standards references before stripping
# 4. Validates values contain technical indicators
# 5. Rejects descriptive text, instructions, and placeholders
#
# =============================================================================

import re
import logging
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)


class ValueNormalizer:
    """
    Post-processor that normalizes extracted specification values.

    Strips leading phrases, removes LLM artifacts, and ensures
    values are in proper technical specification format.
    """

    # Phrases to strip from the beginning of values
    LEADING_PHRASES = [
        # Common sentence starters
        r'^the\s+\w+\s+(is|are|has|was|were)\s+',
        r'^it\s+(is|has|can\s+be|should\s+be|was)\s+',
        r'^this\s+(is|has|should|was)\s+',
        r'^that\s+(is|has|should|was)\s+',
        r'^there\s+(is|are)\s+',

        # Field name prefixes
        r'^output\s+(signal\s+)?(is|:)\s*',
        r'^accuracy\s+(is|:)\s*',
        r'^range\s+(is|:)\s*',
        r'^temperature\s+(range\s+)?(is|:)\s*',
        r'^pressure\s+(range\s+)?(is|:)\s*',
        r'^measurement\s+(range\s+)?(is|:)\s*',
        r'^operating\s+(range\s+)?(is|:)\s*',
        r'^value\s+(is|:)\s*',
        r'^specification\s+(is|:)\s*',
        r'^signal\s+(is|:)\s*',
        r'^power\s+(supply\s+)?(is|:)\s*',
        r'^voltage\s+(is|:)\s*',
        r'^current\s+(is|:)\s*',
        r'^rating\s+(is|:)\s*',
        r'^protection\s+(is|:)\s*',
        r'^material\s+(is|:)\s*',

        # Modifiers
        r'^rated\s+(at|for)\s+',
        r'^typically\s+',
        r'^usually\s+',
        r'^generally\s+',
        r'^normally\s+',
        r'^approximately\s+',
        r'^about\s+',
        r'^around\s+',
        r'^up\s+to\s+',
        r'^at\s+least\s+',
        r'^minimum\s+(of\s+)?',
        r'^maximum\s+(of\s+)?',
        r'^standard\s+(is\s+)?',
        r'^default\s+(is\s+)?',
        r'^nominal\s+',

        # LLM artifacts
        r'^\*+\s*',
        r'^-\s+',
        r'^•\s+',
    ]

    # Phrases to strip from the end of values
    TRAILING_PHRASES = [
        # Standards references (will be extracted first)
        r'\s*(?:per|as\s+per|according\s+to|ref\.?|reference)\s+(?:IEC|ISO|API|ANSI|ISA|EN|NFPA|ASME|IEEE|ASTM|BS|DIN)[\s\d\-\.]+.*$',
        r'\s*\((?:per|ref\.?|see)\s+.*\)$',

        # Contextual suffixes
        r'\s*,?\s*(?:calibrated|measured|tested)\s+(?:at|in).*$',
        r'\s*,?\s*at\s+\d+\s*°[CF].*$',
        r'\s*,?\s*at\s+ambient.*$',
        r'\s*,?\s*under\s+(?:standard|normal|reference).*$',
        r'\s*,?\s*for\s+(?:outdoor|indoor|hazardous).*$',
        r'\s*,?\s*when\s+.*$',
        r'\s*,?\s*if\s+.*$',

        # Parenthetical notes
        r'\s*\((?:see|refer|check|note|optional|standard|typical).*\)$',
        r'\s*\[source:.*\]$',
        r'\s*\[ref:.*\]$',

        # LLM artifacts
        r'\s*\*+$',
        r'\s*\.+$',
        r'\s*;+$',
    ]

    # Patterns for invalid/description values
    INVALID_PATTERNS = [
        # Uncertainty phrases
        r'\b(depends?\s+(?:on|upon))',
        r'\b(varies?\s+(?:by|with|depending|based))',
        r'\b(according\s+to)',
        r'\b(subject\s+to)',
        r'\b(based\s+on)',

        # Reference/instruction phrases
        r'\b(refer\s+to)',
        r'\b(see\s+(?:the|page|section|datasheet|manual))',
        r'\b(check\s+(?:with|the))',
        r'\b(contact\s+(?:the\s+)?(?:manufacturer|supplier|vendor))',
        r'\b(consult\s+(?:the|your))',
        r'\b(please\s+(?:refer|see|check|contact))',
        r'\b(for\s+more\s+(?:information|details))',
        r'\b(visit\s+(?:the|our))',

        # Uncertainty modifiers
        r'\b(typically|usually|generally|normally|often|sometimes)\b',
        r'\b(approximately|about|around|roughly)\s+(?!\d)',  # Allow "around 5%" but not "around the"
        r'\b(may|might|could|can)\s+(?:be|vary|differ)',
        r'\b(should\s+be)',
        r'\b(must\s+be)',

        # Placeholder/unknown
        r'\b(TBD|TBC|T\.B\.D\.|T\.B\.C\.)\b',
        r'\b(N/?A|n/?a)\b',
        r'\b(not\s+(?:applicable|available|specified|defined|provided))',
        r'\b(to\s+be\s+(?:determined|confirmed|specified))',
        r'\b(upon\s+request)',
        r'\b(available\s+(?:on|upon)\s+request)',
        r'\b(as\s+per\s+(?:customer|project|order))',
        r'\b(customer\s+specific)',
        r'\b(project\s+specific)',
        r'\b(application\s+specific)',
        r'\b(model\s+dependent)',

        # Descriptive/instructional
        r'\b(available\s+options?\s+(?:are|include))',
        r'\b(can\s+be\s+(?:configured|selected|specified))',
        r'\b(options?\s+(?:are|include))',
        r'\b(standard\s+options?\s+(?:are|include))',
        r'\b(is\s+(?:determined|selected|specified)\s+by)',

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

        # Sentence indicators
        r'\.\s*$',  # Ends with period (likely a sentence)
    ]

    # Patterns that indicate a valid technical specification
    TECHNICAL_PATTERNS = [
        # Accuracy/tolerance
        r'[±\+\-]\s*\d+\.?\d*\s*%',              # ±0.1%, +0.5%
        r'[±\+\-]\s*\d+\.?\d*\s*(?:°[CF]|K)',    # ±1°C, ±0.5K
        r'[±\+\-]\s*\d+\.?\d*\s*(?:bar|psi|Pa|kPa|MPa)',  # ±0.5bar

        # Ranges
        r'-?\d+\.?\d*\s*(?:to|[-–])\s*[+\-]?\d+\.?\d*\s*(?:°[CF]|K)',  # -40 to 85°C
        r'\d+\.?\d*\s*(?:to|[-–])\s*\d+\.?\d*\s*(?:bar|psi|Pa|kPa|MPa)',  # 0-400 bar
        r'\d+\.?\d*\s*(?:to|[-–])\s*\d+\.?\d*\s*(?:mA|V|Hz)',  # 4-20mA, 0-10V
        r'\d+\.?\d*\s*(?:to|[-–])\s*\d+\.?\d*\s*(?:mm|m|cm|inch|in)',  # Length ranges

        # Electrical
        r'\d+\s*(?:VDC|VAC|V\s*DC|V\s*AC)',      # 24 VDC
        r'\d+[-–]\d+\s*mA',                       # 4-20mA
        r'\d+\s*mA(?:\s+HART)?',                  # 20mA, 4-20mA HART
        r'\d+[-–]\d+\s*V(?:DC|AC)?',              # 10-30VDC

        # Protection/safety ratings
        r'IP\s*\d{2}[A-Z]?',                      # IP67, IP65K
        r'NEMA\s*\d+[A-Z]*',                      # NEMA 4X
        r'SIL\s*[1-4]',                           # SIL 2
        r'ATEX(?:\s+Zone\s*[0-2])?',              # ATEX, ATEX Zone 1
        r'IECEx',                                  # IECEx
        r'Class\s*[I]+\s*(?:,\s*)?Div(?:ision)?\s*[1-2]',  # Class I Div 1
        r'Zone\s*[0-2](?:/[0-2]+)?',              # Zone 1, Zone 0/1

        # Materials
        r'SS\s*\d+L?',                            # SS316L
        r'316L?(?:\s*SS)?',                       # 316L, 316 SS
        r'304(?:\s*SS)?',                         # 304, 304 SS
        r'Hastelloy\s*[A-Z]?(?:-?\d+)?',          # Hastelloy C-276
        r'Inconel\s*\d*',                         # Inconel 625
        r'Monel\s*\d*',                           # Monel 400
        r'Titanium(?:\s+Grade\s*\d+)?',           # Titanium Grade 2
        r'PTFE|PFA|FEP|PEEK',                     # Fluoropolymers
        r'Viton|Kalrez|EPDM|NBR|FKM',             # Elastomers

        # Connections
        r'\d+/\d+\s*(?:NPT|BSP|BSPT|BSPP)',       # 1/2 NPT
        r'DN\s*\d+',                              # DN50
        r'G\s*\d+/\d+',                           # G1/2
        r'Tri-?[Cc]lamp',                         # Tri-Clamp
        r'(?:RF|RTJ|FF)\s*[Ff]lange',             # RF Flange
        r'\d+"\s*(?:ANSI|PN)\s*\d+',              # 2" ANSI 150

        # Communication protocols
        r'HART(?:\s*\d+)?',                       # HART, HART 7
        r'Modbus\s*(?:RTU|TCP)?',                 # Modbus RTU
        r'Profibus(?:\s*(?:PA|DP))?',             # Profibus PA
        r'Foundation\s*Fieldbus',                 # Foundation Fieldbus
        r'EtherNet/IP',                           # EtherNet/IP
        r'PROFINET',                              # PROFINET

        # Time/response
        r'[<>≤≥]?\s*\d+\.?\d*\s*(?:ms|s|sec|min)',  # <250ms, 1.5s
        r'T\d+\s*[<>≤≥]?\s*\d+\s*(?:s|ms)',      # T90 < 5s

        # Ratios
        r'\d+\s*:\s*\d+',                         # 100:1 turndown

        # Percentages with context
        r'\d+\.?\d*\s*%\s*(?:FS|span|URL|range)?',  # 0.1% FS

        # Specific units
        r'\d+\.?\d*\s*(?:Ω|ohm|kΩ|MΩ)',          # Resistance
        r'\d+\.?\d*\s*(?:pF|nF|µF|mF)',          # Capacitance
        r'\d+\.?\d*\s*(?:g|kg|lb)',              # Weight
    ]

    # Pattern to extract standards references before stripping
    STANDARDS_PATTERN = r'\b((?:IEC|ISO|API|ANSI|ISA|EN|NFPA|ASME|IEEE|ASTM|BS|DIN|NIST|OIML|MIL)\s*[\d\-\.]+(?:[A-Z])?)\b'

    def __init__(self):
        """Initialize the ValueNormalizer with compiled patterns."""
        self.leading_patterns = [re.compile(p, re.IGNORECASE) for p in self.LEADING_PHRASES]
        self.trailing_patterns = [re.compile(p, re.IGNORECASE) for p in self.TRAILING_PHRASES]
        self.invalid_patterns = [re.compile(p, re.IGNORECASE) for p in self.INVALID_PATTERNS]
        self.technical_patterns = [re.compile(p, re.IGNORECASE) for p in self.TECHNICAL_PATTERNS]
        self.standards_pattern = re.compile(self.STANDARDS_PATTERN, re.IGNORECASE)
        logger.info("[ValueNormalizer] Initialized with pattern matching")

    def normalize(self, value: str, field_name: str = "") -> Tuple[str, List[str]]:
        """
        Normalize a specification value.

        Args:
            value: The raw extracted value
            field_name: Optional field name for context-aware processing

        Returns:
            Tuple of (normalized_value, extracted_standards_references)
        """
        if not value:
            return "", []

        original = str(value).strip()
        normalized = original
        standards_refs = []

        # Step 1: Extract standards references before stripping
        standards_matches = self.standards_pattern.findall(normalized)
        if standards_matches:
            standards_refs = [s.upper().replace("  ", " ").strip() for s in standards_matches]
            standards_refs = list(set(standards_refs))

        # Step 2: Strip leading phrases
        for pattern in self.leading_patterns:
            normalized = pattern.sub('', normalized, count=1)

        # Step 3: Strip trailing phrases
        for pattern in self.trailing_patterns:
            normalized = pattern.sub('', normalized)

        # Step 4: Clean up whitespace and punctuation
        normalized = normalized.strip()
        normalized = re.sub(r'\s+', ' ', normalized)  # Normalize whitespace
        normalized = re.sub(r'^[,;:\s\-•*]+', '', normalized)  # Strip leading punctuation
        normalized = re.sub(r'[,;:\s]+$', '', normalized)  # Strip trailing punctuation

        # Step 5: Remove markdown/formatting artifacts
        normalized = re.sub(r'\*+', '', normalized)
        normalized = re.sub(r'`+', '', normalized)
        normalized = re.sub(r'#+\s*', '', normalized)

        # Step 6: Clean up orphaned parentheses/brackets
        normalized = re.sub(r'\(\s*\)', '', normalized)
        normalized = re.sub(r'\[\s*\]', '', normalized)

        # Step 7: Final trim
        normalized = normalized.strip()

        if normalized != original:
            logger.debug(f"[ValueNormalizer] Normalized '{field_name}': '{original[:50]}...' -> '{normalized[:50]}...'")

        return normalized, standards_refs

    def is_valid_spec_value(self, value: str) -> bool:
        """
        Check if a value is a valid technical specification.

        Returns False for descriptions, instructions, placeholder text,
        or values that don't contain technical indicators.

        Args:
            value: The value to validate

        Returns:
            True if the value is a valid technical specification
        """
        if not value or len(value.strip()) < 2:
            return False

        value_str = str(value).strip()

        # Check for invalid/description patterns
        for pattern in self.invalid_patterns:
            if pattern.search(value_str):
                logger.debug(f"[ValueNormalizer] Invalid pattern matched: {value_str[:50]}")
                return False

        # Check for sentence structure (subject + verb = likely description)
        sentence_indicators = [
            r'^(the|a|an|this|that|it|there)\s+\w+\s+(is|are|has|have|was|were|can|will|should)',
            r'\b(is|are|has|have|was|were)\s+(a|an|the)\s+',
            r'^(i|we|you|they)\s+',
        ]
        for pattern in sentence_indicators:
            if re.search(pattern, value_str, re.IGNORECASE):
                logger.debug(f"[ValueNormalizer] Sentence structure detected: {value_str[:50]}")
                return False

        # Check for question structure
        if re.search(r'^(what|how|when|where|why|which|who)\b', value_str, re.IGNORECASE):
            return False

        # Must contain at least one technical indicator
        has_technical = any(pattern.search(value_str) for pattern in self.technical_patterns)

        if not has_technical:
            # Additional check: short alphanumeric codes may be valid (e.g., "2-wire", "Type K")
            if len(value_str) < 30 and re.search(r'\d', value_str):
                # Short value with a number - might be valid
                # But reject if it contains too many words
                word_count = len(value_str.split())
                if word_count <= 4:
                    return True

            logger.debug(f"[ValueNormalizer] No technical indicator: {value_str[:50]}")
            return False

        return True

    def get_value_confidence_score(self, value: str) -> float:
        """
        Calculate a confidence score (0-1) for a value instead of binary pass/fail.

        IMPROVED (v2): Weighted scoring instead of hard rejection allows acceptance
        of borderline specifications with lower confidence scores.

        Args:
            value: The value to score

        Returns:
            float: Confidence score 0-1 where:
            - 0.9-1.0: High confidence (clean technical spec)
            - 0.7-0.9: Good confidence (minor issues but acceptable)
            - 0.5-0.7: Moderate confidence (some uncertainties or soft qualifiers)
            - 0.3-0.5: Low confidence (multiple issues but still possibly valid)
            - <0.3: Very low confidence (reject)
        """
        if not value or len(str(value).strip()) < 2:
            return 0.0

        value_str = str(value).strip()
        score = 0.5  # Base score: neutral

        # Check for SOFT indicators (deduct but don't reject entirely)
        soft_indicators = [
            (r'\b(typically|usually|generally|normally|often|sometimes)\b', -0.15),
            (r'\b(approximately|about|around)\s+(?!\d)', -0.10),
            (r'\b(may|might|could|can)\s+(?:be|vary|differ)', -0.10),
            (r'\b(should\s+be|must\s+be)\b', -0.10),
            (r'\b(depends?\s+(?:on|upon))\b', -0.15),
            (r'\b(varies?\s+(?:by|with|depending|based))\b', -0.15),
        ]

        for pattern, penalty in soft_indicators:
            if re.search(pattern, value_str, re.IGNORECASE):
                score += penalty

        # Check for HARD rejection patterns (immediate low score)
        hard_patterns = [
            (r'\b(TBD|TBC|T\.B\.D\.|T\.B\.C\.)\b', -0.5),
            (r'\b(not\s+(?:applicable|available|specified|defined|provided))\b', -0.5),
            (r'\b(N/?A|n/?a)\b', -0.5),
        ]

        for pattern, penalty in hard_patterns:
            if re.search(pattern, value_str, re.IGNORECASE):
                score += penalty
                if score < 0.2:  # Hard rejections
                    return 0.0

        # Check for sentence structure (deduct for likely description)
        sentence_indicators = [
            r'^(the|a|an|this|that|it|there)\s+\w+\s+(is|are|has|have|was|were|can|will|should)',
            r'\b(is|are|has|have|was|were)\s+(a|an|the)\s+',
        ]
        for pattern in sentence_indicators:
            if re.search(pattern, value_str, re.IGNORECASE):
                score -= 0.20

        # BONUS for technical indicators (increase confidence)
        technical_bonus = [
            (r'[±\+\-]\s*\d+\.?\d*\s*%', 0.25),              # Accuracy like ±0.1%
            (r'[±\+\-]\s*\d+\.?\d*\s*(?:°[CF]|K)', 0.25),    # Temperature ±1°C
            (r'\d+\.?\d*\s*(?:to|[-–])\s*\d+\.?\d*', 0.20),  # Range like 0-400
            (r'\d+\s*(?:VDC|VAC|mA|V|Hz|bar|psi)', 0.20),   # Units
            (r'(?:IP|NEMA|SIL|ATEX)\s*\d+', 0.25),          # Safety ratings
            (r'(?:SS|316|304|Hastelloy|Inconel)\s*\d*', 0.20),  # Materials
            (r'(?:NPT|BSP|BSPT|DN|Tri-Clamp)', 0.15),       # Connections
            (r'(?:HART|Modbus|Profibus|4-20mA)', 0.20),     # Protocols
        ]

        for pattern, bonus in technical_bonus:
            if re.search(pattern, value_str):
                score += bonus

        # Ensure score stays in valid range
        return max(0.0, min(score, 1.0))

    def extract_and_validate(self, value: str, field_name: str = "") -> Tuple[Optional[str], List[str]]:
        """
        Normalize a value and validate it in one step.

        Args:
            value: The raw extracted value
            field_name: Optional field name for context

        Returns:
            Tuple of (normalized_value or None if invalid, standards_references)
        """
        normalized, refs = self.normalize(value, field_name)

        if normalized and self.is_valid_spec_value(normalized):
            return normalized, refs

        return None, refs

    def extract_and_validate_with_confidence(
        self, value: str, field_name: str = "", min_confidence: float = 0.3
    ) -> Tuple[Optional[str], List[str], float]:
        """
        IMPROVED (v2): Normalize a value and return with confidence score instead of binary rejection.

        This variant allows acceptance of borderline values with lower confidence scores,
        useful for maximizing specification extraction from standards documents.

        Args:
            value: The raw extracted value
            field_name: Optional field name for context
            min_confidence: Minimum confidence threshold to accept (default 0.3 = 30%)

        Returns:
            Tuple of (normalized_value or None, standards_references, confidence_score 0-1)
        """
        normalized, refs = self.normalize(value, field_name)

        if not normalized:
            return None, refs, 0.0

        confidence = self.get_value_confidence_score(normalized)

        # Accept if confidence meets minimum threshold
        if confidence >= min_confidence:
            return normalized, refs, confidence

        return None, refs, confidence


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_normalizer: Optional[ValueNormalizer] = None


def get_value_normalizer() -> ValueNormalizer:
    """Get the singleton ValueNormalizer instance."""
    global _normalizer
    if _normalizer is None:
        _normalizer = ValueNormalizer()
    return _normalizer


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def normalize_spec_value(value: str, field_name: str = "") -> Tuple[str, List[str]]:
    """
    Convenience function to normalize a specification value.

    Args:
        value: The raw extracted value
        field_name: Optional field name for context

    Returns:
        Tuple of (normalized_value, standards_references)
    """
    return get_value_normalizer().normalize(value, field_name)


def is_valid_spec_value(value: str) -> bool:
    """
    Convenience function to check if a value is a valid technical specification.

    Args:
        value: The value to validate

    Returns:
        True if the value is a valid technical specification
    """
    return get_value_normalizer().is_valid_spec_value(value)


def extract_and_validate_spec(value: str, field_name: str = "") -> Tuple[Optional[str], List[str]]:
    """
    Convenience function to normalize and validate in one step.

    Args:
        value: The raw extracted value
        field_name: Optional field name for context

    Returns:
        Tuple of (normalized_value or None if invalid, standards_references)
    """
    return get_value_normalizer().extract_and_validate(value, field_name)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ValueNormalizer",
    "get_value_normalizer",
    "normalize_spec_value",
    "is_valid_spec_value",
    "extract_and_validate_spec",
]
