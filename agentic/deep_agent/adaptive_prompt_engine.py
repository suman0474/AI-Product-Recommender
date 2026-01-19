# agentic/deep_agent/adaptive_prompt_engine.py
# =============================================================================
# ADAPTIVE PROMPT ENGINE - Learn from Failures to Optimize Prompts
# =============================================================================
#
# Purpose: Generate optimized prompts based on historical failure/success data.
# Uses the SchemaFailureMemory to:
# 1. Add JSON format instructions for products with parse errors
# 2. Simplify prompts for products with validation failures
# 3. Batch fields for products with extraction failures
# 4. Use cached successful prompts when available
#
# =============================================================================

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .schema_failure_memory import (
    SchemaFailureMemory,
    FailureType,
    RecoveryAction,
    get_schema_failure_memory
)

logger = logging.getLogger(__name__)


class PromptStrategy(Enum):
    """Strategies for prompt generation"""
    STANDARD = "standard"
    STRICT_JSON = "strict_json"
    SIMPLIFIED = "simplified"
    BATCHED = "batched"
    MINIMAL = "minimal"
    CACHED = "cached"


@dataclass
class PromptOptimization:
    """Details about how a prompt was optimized"""
    strategy: PromptStrategy
    modifications: List[str]
    risk_level: str
    confidence: float
    source: str  # "learned", "default", "cached"


# =============================================================================
# JSON FORMAT INSTRUCTIONS
# =============================================================================

STRICT_JSON_PREAMBLE = """
CRITICAL: Your response MUST be valid JSON only. Follow these rules exactly:
1. Output ONLY a JSON object, no markdown, no explanations
2. Do NOT include ```json or ``` markers
3. Use double quotes for all strings
4. Escape special characters: newlines as \\n, quotes as \\"
5. Do NOT include trailing commas
6. Do NOT include comments in JSON
7. Ensure all brackets and braces are properly closed

"""

STRICT_JSON_SUFFIX = """

Remember: Return ONLY valid JSON. No text before or after the JSON object.
"""

FIELD_EXTRACTION_INSTRUCTIONS = """
For each field, extract ONLY the value without description or explanation.
Format: "fieldName": "value" (not "fieldName": {"value": "...", "description": "..."})

Example:
CORRECT: "accuracy": "0.5%"
WRONG: "accuracy": {"value": "0.5%", "description": "measurement accuracy"}
"""

SIMPLIFIED_PROMPT_TEMPLATE = """
Extract specifications for {product_type}.

Return JSON with these fields (use null if unknown):
{fields_list}

Output ONLY valid JSON, no explanations.
"""

MINIMAL_PROMPT_TEMPLATE = """
List specs for {product_type} as JSON.
Fields: {fields_csv}
JSON only:
"""


class AdaptivePromptEngine:
    """
    Engine that generates optimized prompts based on failure history.

    Uses machine learning-like approach:
    1. Analyze historical failures for product type
    2. Select appropriate prompt strategy
    3. Apply modifications to base prompt
    4. Track results to improve future prompts
    """

    def __init__(self, failure_memory: Optional[SchemaFailureMemory] = None):
        """
        Initialize the adaptive prompt engine.

        Args:
            failure_memory: Failure memory instance (uses global if None)
        """
        self.memory = failure_memory or get_schema_failure_memory()

        # Strategy selection thresholds
        self.json_error_threshold = 2  # Switch to strict JSON after 2 parse errors
        self.validation_error_threshold = 3  # Simplify after 3 validation errors
        self.extraction_error_threshold = 2  # Batch fields after 2 extraction errors

        # Track modifications applied this session
        self.session_modifications: Dict[str, List[str]] = {}

        logger.info("[ADAPTIVE_PROMPT] AdaptivePromptEngine initialized")

    def optimize_prompt(self,
                        product_type: str,
                        base_prompt: str,
                        fields: Optional[List[str]] = None,
                        context: Optional[Dict] = None) -> Tuple[str, PromptOptimization]:
        """
        Optimize a prompt based on failure history.

        Args:
            product_type: Product type for schema generation
            base_prompt: Original prompt to optimize
            fields: List of field names to extract (optional)
            context: Additional context (optional)

        Returns:
            Tuple of (optimized_prompt, optimization_details)
        """
        # Check for cached successful prompt first
        cached_prompt = self.memory.get_optimized_prompt(product_type)
        if cached_prompt:
            logger.info(f"[ADAPTIVE_PROMPT] Using cached successful prompt for {product_type}")
            return cached_prompt, PromptOptimization(
                strategy=PromptStrategy.CACHED,
                modifications=["Using cached successful prompt"],
                risk_level="low",
                confidence=0.9,
                source="cached"
            )

        # Analyze failure risk
        risk_analysis = self.memory.predict_failure_risk(product_type)

        # Select strategy based on risk
        strategy, modifications = self._select_strategy(
            product_type, risk_analysis, fields
        )

        # Apply strategy to prompt
        optimized_prompt = self._apply_strategy(
            base_prompt, strategy, product_type, fields, context
        )

        # Track modifications
        self.session_modifications[product_type] = modifications

        optimization = PromptOptimization(
            strategy=strategy,
            modifications=modifications,
            risk_level=risk_analysis.get("risk_level", "unknown"),
            confidence=self._calculate_confidence(strategy, risk_analysis),
            source="learned" if risk_analysis.get("has_history") else "default"
        )

        logger.info(
            f"[ADAPTIVE_PROMPT] Optimized prompt for {product_type}: "
            f"strategy={strategy.value}, modifications={len(modifications)}, "
            f"confidence={optimization.confidence:.2f}"
        )

        return optimized_prompt, optimization

    def _select_strategy(self,
                         product_type: str,
                         risk_analysis: Dict,
                         fields: Optional[List[str]]) -> Tuple[PromptStrategy, List[str]]:
        """
        Select the best prompt strategy based on failure analysis.

        Returns:
            Tuple of (strategy, list of modifications to apply)
        """
        modifications = []
        strategy = PromptStrategy.STANDARD

        if not risk_analysis.get("has_history"):
            return strategy, ["No history - using standard prompt"]

        likely_failures = risk_analysis.get("likely_failure_types", [])
        failure_history = self.memory.get_failure_history(product_type, limit=20)

        # Count failure types
        json_errors = sum(1 for f in failure_history if f.get("failure_type") == FailureType.JSON_PARSE_ERROR.value)
        validation_errors = sum(1 for f in failure_history if f.get("failure_type") == FailureType.VALIDATION_FAILED.value)
        extraction_errors = sum(1 for f in failure_history if f.get("failure_type") == FailureType.FIELD_EXTRACTION_FAILED.value)
        timeout_errors = sum(1 for f in failure_history if f.get("failure_type") == FailureType.TIMEOUT.value)

        # Priority 1: JSON parsing issues
        if json_errors >= self.json_error_threshold:
            strategy = PromptStrategy.STRICT_JSON
            modifications.append(f"Added strict JSON instructions ({json_errors} parse errors)")

        # Priority 2: Validation failures (use simplified)
        if validation_errors >= self.validation_error_threshold:
            if strategy == PromptStrategy.STRICT_JSON:
                # Combine strategies
                modifications.append(f"Simplified prompt structure ({validation_errors} validation errors)")
            else:
                strategy = PromptStrategy.SIMPLIFIED
                modifications.append(f"Using simplified prompt ({validation_errors} validation errors)")

        # Priority 3: Extraction failures (batch fields)
        if extraction_errors >= self.extraction_error_threshold and fields:
            if len(fields) > 15:
                strategy = PromptStrategy.BATCHED
                modifications.append(f"Batching fields ({extraction_errors} extraction errors, {len(fields)} fields)")

        # Priority 4: Timeout issues (use minimal)
        if timeout_errors >= 2:
            strategy = PromptStrategy.MINIMAL
            modifications.append(f"Using minimal prompt ({timeout_errors} timeouts)")

        # High risk override
        if risk_analysis.get("risk_level") == "high" and not modifications:
            strategy = PromptStrategy.STRICT_JSON
            modifications.append("High risk - applying strict JSON format")

        if not modifications:
            modifications.append("No critical issues - using standard prompt")

        return strategy, modifications

    def _apply_strategy(self,
                        base_prompt: str,
                        strategy: PromptStrategy,
                        product_type: str,
                        fields: Optional[List[str]],
                        context: Optional[Dict]) -> str:
        """
        Apply the selected strategy to the base prompt.

        Returns:
            Modified prompt string
        """
        if strategy == PromptStrategy.STANDARD:
            return base_prompt

        if strategy == PromptStrategy.STRICT_JSON:
            return self._apply_strict_json(base_prompt)

        if strategy == PromptStrategy.SIMPLIFIED:
            return self._apply_simplified(base_prompt, product_type, fields)

        if strategy == PromptStrategy.BATCHED:
            return self._apply_batched(base_prompt, product_type, fields)

        if strategy == PromptStrategy.MINIMAL:
            return self._apply_minimal(product_type, fields)

        return base_prompt

    def _apply_strict_json(self, base_prompt: str) -> str:
        """Add strict JSON formatting instructions."""
        # Remove any existing JSON instructions to avoid duplication
        prompt = re.sub(r'(respond|return|output)\s+(only\s+)?(valid\s+)?json', '', base_prompt, flags=re.IGNORECASE)

        return f"{STRICT_JSON_PREAMBLE}\n{prompt}\n{STRICT_JSON_SUFFIX}"

    def _apply_simplified(self,
                          base_prompt: str,
                          product_type: str,
                          fields: Optional[List[str]]) -> str:
        """Create a simplified version of the prompt."""
        if fields:
            fields_list = "\n".join(f"- {field}" for field in fields[:20])  # Limit to 20
            return SIMPLIFIED_PROMPT_TEMPLATE.format(
                product_type=product_type,
                fields_list=fields_list
            ) + STRICT_JSON_PREAMBLE

        # If no fields specified, simplify the existing prompt
        # Remove verbose instructions, keep core request
        lines = base_prompt.split('\n')
        simplified_lines = [
            line for line in lines
            if len(line.strip()) > 10 and not line.strip().startswith('#')
        ][:10]  # Keep max 10 lines

        return '\n'.join(simplified_lines) + "\n\nReturn ONLY valid JSON."

    def _apply_batched(self,
                       base_prompt: str,
                       product_type: str,
                       fields: Optional[List[str]]) -> str:
        """Split fields into batches for separate processing."""
        if not fields or len(fields) <= 15:
            return base_prompt

        # Create prompt for first batch (most important fields)
        batch_fields = fields[:15]
        fields_csv = ", ".join(batch_fields)

        batched_prompt = f"""
{STRICT_JSON_PREAMBLE}
Extract these specifications for {product_type}:
{fields_csv}

For each field, provide ONLY the value as a string.
Return as JSON object with field names as keys.
{FIELD_EXTRACTION_INSTRUCTIONS}
"""
        return batched_prompt

    def _apply_minimal(self, product_type: str, fields: Optional[List[str]]) -> str:
        """Create minimal prompt for timeout-prone requests."""
        if fields:
            fields_csv = ", ".join(fields[:10])  # Max 10 fields
        else:
            fields_csv = "accuracy, range, output, material, certification"

        return MINIMAL_PROMPT_TEMPLATE.format(
            product_type=product_type,
            fields_csv=fields_csv
        )

    def _calculate_confidence(self,
                              strategy: PromptStrategy,
                              risk_analysis: Dict) -> float:
        """Calculate confidence score for the optimization."""
        base_confidence = {
            PromptStrategy.STANDARD: 0.7,
            PromptStrategy.STRICT_JSON: 0.8,
            PromptStrategy.SIMPLIFIED: 0.75,
            PromptStrategy.BATCHED: 0.7,
            PromptStrategy.MINIMAL: 0.6,
            PromptStrategy.CACHED: 0.9
        }

        confidence = base_confidence.get(strategy, 0.7)

        # Adjust based on success rate
        success_count = risk_analysis.get("success_count", 0)
        total = risk_analysis.get("total_attempts", 1)
        if total > 0:
            success_rate = success_count / total
            confidence = confidence * (0.5 + 0.5 * success_rate)

        return min(confidence, 1.0)

    # =========================================================================
    # SPECIALIZED PROMPT GENERATORS
    # =========================================================================

    def get_schema_population_prompt(self,
                                     product_type: str,
                                     fields: List[str],
                                     context: Optional[str] = None) -> Tuple[str, PromptOptimization]:
        """
        Generate optimized prompt for schema field population.

        Args:
            product_type: Product type
            fields: Fields to populate
            context: Additional context from standards documents

        Returns:
            Tuple of (optimized_prompt, optimization_details)
        """
        base_prompt = f"""
Provide technical specifications for {product_type} according to industrial standards.

For each of the following fields, provide a specific value or range:
{chr(10).join(f'- {field}' for field in fields)}

{"Context from standards documents:" + chr(10) + context if context else ""}

Return as JSON object mapping field names to their values.
"""
        return self.optimize_prompt(product_type, base_prompt, fields)

    def get_standards_enrichment_prompt(self,
                                        product_type: str,
                                        existing_specs: Dict[str, Any]) -> Tuple[str, PromptOptimization]:
        """
        Generate optimized prompt for standards enrichment.

        Args:
            product_type: Product type
            existing_specs: Existing specifications to enrich

        Returns:
            Tuple of (optimized_prompt, optimization_details)
        """
        spec_summary = ", ".join(f"{k}={v}" for k, v in list(existing_specs.items())[:10])

        base_prompt = f"""
What are the applicable engineering standards (ISO, IEC, API, ANSI, ISA),
certifications (SIL, ATEX, CE), and safety requirements for {product_type}?

Current specifications: {spec_summary}

Return JSON with:
- "applicable_standards": list of standard codes
- "certifications": list of required certifications
- "safety_requirements": list of safety requirements
- "calibration_standards": list of calibration requirements
"""
        return self.optimize_prompt(product_type, base_prompt)

    def get_llm_specs_prompt(self,
                             product_type: str,
                             category: str,
                             target_count: int = 60) -> Tuple[str, PromptOptimization]:
        """
        Generate optimized prompt for LLM specifications generation.

        Args:
            product_type: Product type
            category: Product category
            target_count: Target number of specifications

        Returns:
            Tuple of (optimized_prompt, optimization_details)
        """
        base_prompt = f"""
Generate {target_count} technical specifications for {product_type} (category: {category}).

Include:
- Physical specifications (dimensions, weight, materials)
- Performance specifications (accuracy, range, resolution)
- Environmental specifications (temperature, humidity, IP rating)
- Electrical specifications (voltage, current, power)
- Communication specifications (protocols, outputs)
- Compliance specifications (standards, certifications)

Return JSON object with specification names as keys and values as strings.
Each value should be a typical/standard value for this product type.
"""
        return self.optimize_prompt(
            product_type, base_prompt,
            fields=["physical", "performance", "environmental", "electrical", "communication", "compliance"]
        )

    # =========================================================================
    # FEEDBACK & LEARNING
    # =========================================================================

    def report_prompt_result(self,
                             product_type: str,
                             prompt: str,
                             success: bool,
                             fields_populated: int = 0,
                             total_fields: int = 0,
                             error_type: Optional[FailureType] = None,
                             error_message: str = "",
                             duration_ms: int = 0):
        """
        Report the result of using an optimized prompt.

        This feeds back into the failure memory for future learning.

        Args:
            product_type: Product type
            prompt: The prompt that was used
            success: Whether the prompt succeeded
            fields_populated: Number of fields populated
            total_fields: Total fields attempted
            error_type: Type of error if failed
            error_message: Error message if failed
            duration_ms: Duration of the operation
        """
        if success:
            self.memory.record_success(
                product_type=product_type,
                prompt=prompt,
                fields_populated=fields_populated,
                total_fields=total_fields,
                confidence_score=fields_populated / max(total_fields, 1),
                sources_used=["adaptive_prompt"],
                duration_ms=duration_ms
            )
        elif error_type:
            self.memory.record_failure(
                product_type=product_type,
                failure_type=error_type,
                error_message=error_message,
                prompt=prompt,
                context={
                    "fields_attempted": total_fields,
                    "modifications": self.session_modifications.get(product_type, [])
                },
                duration_ms=duration_ms
            )

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get statistics about prompt optimizations."""
        return {
            "session_modifications": dict(self.session_modifications),
            "memory_stats": self.memory.get_stats()
        }


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_global_prompt_engine: Optional[AdaptivePromptEngine] = None


def get_adaptive_prompt_engine() -> AdaptivePromptEngine:
    """Get or create the global adaptive prompt engine."""
    global _global_prompt_engine

    if _global_prompt_engine is None:
        _global_prompt_engine = AdaptivePromptEngine()
    return _global_prompt_engine


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "AdaptivePromptEngine",
    "PromptStrategy",
    "PromptOptimization",
    "get_adaptive_prompt_engine",
    "STRICT_JSON_PREAMBLE",
    "STRICT_JSON_SUFFIX"
]
