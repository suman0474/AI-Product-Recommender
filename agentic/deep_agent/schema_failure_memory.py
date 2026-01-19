# agentic/deep_agent/schema_failure_memory.py
# =============================================================================
# SCHEMA FAILURE MEMORY - Learn from Schema Generation Failures
# =============================================================================
#
# Purpose: Store, analyze, and learn from schema generation failures to
# improve future schema generation success rates and speed.
#
# Key Features:
# 1. Persistent failure tracking by product type
# 2. Pattern recognition for common failure modes
# 3. Success pattern caching for optimized prompts
# 4. Adaptive prompt generation based on historical data
#
# =============================================================================

import logging
import json
import threading
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of schema generation failures"""
    JSON_PARSE_ERROR = "json_parse_error"
    VALIDATION_FAILED = "validation_failed"
    TIMEOUT = "timeout"
    EMPTY_RESPONSE = "empty_response"
    LOW_CONFIDENCE = "low_confidence"
    RATE_LIMITED = "rate_limited"
    NETWORK_ERROR = "network_error"
    LLM_REFUSAL = "llm_refusal"
    FIELD_EXTRACTION_FAILED = "field_extraction_failed"
    SCHEMA_MISMATCH = "schema_mismatch"


class RecoveryAction(Enum):
    """Actions that can recover from failures"""
    RETRY_WITH_SIMPLER_PROMPT = "retry_simpler_prompt"
    USE_TEMPLATE_FALLBACK = "template_fallback"
    USE_LLM_SPECS = "llm_specs"
    SPLIT_INTO_BATCHES = "split_batches"
    ADD_JSON_INSTRUCTIONS = "add_json_instructions"
    INCREASE_TIMEOUT = "increase_timeout"
    USE_CACHED_RESPONSE = "use_cached"
    SKIP_VALIDATION = "skip_validation"
    REDUCE_FIELD_COUNT = "reduce_fields"


@dataclass
class FailureEntry:
    """Single failure record"""
    failure_id: str
    product_type: str
    failure_type: FailureType
    error_message: str
    prompt_hash: str  # Hash of the prompt that failed
    prompt_preview: str  # First 200 chars of prompt
    timestamp: str
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_attempted: Optional[RecoveryAction] = None
    recovery_successful: bool = False
    duration_ms: int = 0

    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            'failure_type': self.failure_type.value,
            'recovery_attempted': self.recovery_attempted.value if self.recovery_attempted else None
        }


@dataclass
class SuccessEntry:
    """Record of successful schema generation"""
    success_id: str
    product_type: str
    prompt_hash: str
    prompt_preview: str
    fields_populated: int
    total_fields: int
    confidence_score: float
    timestamp: str
    duration_ms: int
    sources_used: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class FailurePattern:
    """Recognized pattern of failures"""
    pattern_id: str
    failure_type: FailureType
    product_type_pattern: str  # Regex or exact match
    occurrence_count: int
    recommended_action: RecoveryAction
    success_rate_after_action: float
    last_seen: str


class SchemaFailureMemory:
    """
    Persistent memory system for schema generation failures and successes.

    Learns from past failures to:
    1. Predict likely failures before they happen
    2. Recommend recovery actions
    3. Optimize prompts for specific product types
    4. Cache successful configurations
    """

    def __init__(self,
                 persistence_path: Optional[str] = None,
                 max_failures_per_type: int = 100,
                 max_successes_per_type: int = 50):
        """
        Initialize failure memory.

        Args:
            persistence_path: Path to persist memory (None for in-memory only)
            max_failures_per_type: Maximum failures to store per product type
            max_successes_per_type: Maximum successes to store per product type
        """
        self.persistence_path = persistence_path
        self.max_failures_per_type = max_failures_per_type
        self.max_successes_per_type = max_successes_per_type

        # Core storage
        self.failures: Dict[str, List[FailureEntry]] = {}  # By product type
        self.successes: Dict[str, List[SuccessEntry]] = {}  # By product type
        self.patterns: Dict[str, FailurePattern] = {}  # By pattern_id

        # Optimized prompt cache
        self.optimized_prompts: Dict[str, str] = {}  # product_type -> optimized prompt

        # Statistics
        self.stats = {
            "total_failures": 0,
            "total_successes": 0,
            "patterns_detected": 0,
            "recoveries_attempted": 0,
            "recoveries_successful": 0
        }

        # Thread safety
        self._lock = threading.RLock()

        # Load persisted data if available
        if persistence_path:
            self._load_from_disk()

        logger.info("[FAILURE_MEMORY] SchemaFailureMemory initialized")

    # =========================================================================
    # FAILURE RECORDING
    # =========================================================================

    def record_failure(self,
                       product_type: str,
                       failure_type: FailureType,
                       error_message: str,
                       prompt: str,
                       context: Optional[Dict] = None,
                       duration_ms: int = 0) -> FailureEntry:
        """
        Record a schema generation failure.

        Args:
            product_type: Product type that failed
            failure_type: Type of failure
            error_message: Error message/details
            prompt: The prompt that was used
            context: Additional context (field count, sources, etc.)
            duration_ms: How long the operation took

        Returns:
            FailureEntry record
        """
        with self._lock:
            prompt_hash = self._hash_prompt(prompt)
            failure_id = f"fail_{product_type}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

            entry = FailureEntry(
                failure_id=failure_id,
                product_type=product_type,
                failure_type=failure_type,
                error_message=error_message[:500],  # Truncate long errors
                prompt_hash=prompt_hash,
                prompt_preview=prompt[:200],
                timestamp=datetime.now().isoformat(),
                context=context or {},
                duration_ms=duration_ms
            )

            # Store failure
            if product_type not in self.failures:
                self.failures[product_type] = []

            self.failures[product_type].append(entry)

            # Enforce max limit (FIFO)
            if len(self.failures[product_type]) > self.max_failures_per_type:
                self.failures[product_type] = self.failures[product_type][-self.max_failures_per_type:]

            self.stats["total_failures"] += 1

            # Detect patterns
            self._detect_patterns(product_type, failure_type)

            logger.warning(
                f"[FAILURE_MEMORY] Recorded {failure_type.value} for {product_type}: "
                f"{error_message[:100]}..."
            )

            # Persist if configured
            if self.persistence_path:
                self._save_to_disk()

            return entry

    def record_success(self,
                       product_type: str,
                       prompt: str,
                       fields_populated: int,
                       total_fields: int,
                       confidence_score: float,
                       sources_used: List[str],
                       duration_ms: int = 0) -> SuccessEntry:
        """
        Record a successful schema generation.

        Args:
            product_type: Product type that succeeded
            prompt: The prompt that was used
            fields_populated: Number of fields successfully populated
            total_fields: Total fields in schema
            confidence_score: Overall confidence
            sources_used: List of sources that contributed
            duration_ms: How long the operation took

        Returns:
            SuccessEntry record
        """
        with self._lock:
            prompt_hash = self._hash_prompt(prompt)
            success_id = f"succ_{product_type}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

            entry = SuccessEntry(
                success_id=success_id,
                product_type=product_type,
                prompt_hash=prompt_hash,
                prompt_preview=prompt[:200],
                fields_populated=fields_populated,
                total_fields=total_fields,
                confidence_score=confidence_score,
                timestamp=datetime.now().isoformat(),
                duration_ms=duration_ms,
                sources_used=sources_used
            )

            # Store success
            if product_type not in self.successes:
                self.successes[product_type] = []

            self.successes[product_type].append(entry)

            # Enforce max limit
            if len(self.successes[product_type]) > self.max_successes_per_type:
                self.successes[product_type] = self.successes[product_type][-self.max_successes_per_type:]

            self.stats["total_successes"] += 1

            # Cache the successful prompt configuration
            if fields_populated >= total_fields * 0.7:  # 70%+ success rate
                self.optimized_prompts[product_type] = prompt

            logger.info(
                f"[FAILURE_MEMORY] Recorded SUCCESS for {product_type}: "
                f"{fields_populated}/{total_fields} fields, confidence={confidence_score:.2f}"
            )

            # Persist if configured
            if self.persistence_path:
                self._save_to_disk()

            return entry

    def record_recovery_result(self,
                               failure_id: str,
                               recovery_action: RecoveryAction,
                               successful: bool):
        """
        Record whether a recovery action succeeded.

        Args:
            failure_id: The failure that was being recovered
            recovery_action: Action that was attempted
            successful: Whether recovery succeeded
        """
        with self._lock:
            # Find and update the failure entry
            for failures in self.failures.values():
                for entry in failures:
                    if entry.failure_id == failure_id:
                        entry.recovery_attempted = recovery_action
                        entry.recovery_successful = successful
                        break

            self.stats["recoveries_attempted"] += 1
            if successful:
                self.stats["recoveries_successful"] += 1

            logger.info(
                f"[FAILURE_MEMORY] Recovery {recovery_action.value} for {failure_id}: "
                f"{'SUCCESS' if successful else 'FAILED'}"
            )

    # =========================================================================
    # FAILURE ANALYSIS & PREDICTION
    # =========================================================================

    def predict_failure_risk(self, product_type: str) -> Dict[str, Any]:
        """
        Predict the risk of failure for a product type based on history.

        Args:
            product_type: Product type to analyze

        Returns:
            Risk assessment with recommendations
        """
        with self._lock:
            failures = self.failures.get(product_type, [])
            successes = self.successes.get(product_type, [])

            total_attempts = len(failures) + len(successes)
            if total_attempts == 0:
                return {
                    "risk_level": "unknown",
                    "failure_rate": 0.0,
                    "recommendations": [],
                    "likely_failure_types": [],
                    "has_history": False
                }

            failure_rate = len(failures) / total_attempts

            # Analyze failure types
            failure_type_counts = {}
            for f in failures:
                ft = f.failure_type.value
                failure_type_counts[ft] = failure_type_counts.get(ft, 0) + 1

            likely_failures = sorted(
                failure_type_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]

            # Determine risk level
            if failure_rate >= 0.7:
                risk_level = "high"
            elif failure_rate >= 0.4:
                risk_level = "medium"
            else:
                risk_level = "low"

            # Generate recommendations
            recommendations = self._generate_recommendations(
                product_type, failures, likely_failures
            )

            return {
                "risk_level": risk_level,
                "failure_rate": failure_rate,
                "total_attempts": total_attempts,
                "failure_count": len(failures),
                "success_count": len(successes),
                "recommendations": recommendations,
                "likely_failure_types": [f[0] for f in likely_failures],
                "has_history": True
            }

    def get_recommended_recovery(self,
                                  product_type: str,
                                  failure_type: FailureType) -> Optional[RecoveryAction]:
        """
        Get the best recovery action for a specific failure type.

        Args:
            product_type: Product type
            failure_type: Type of failure that occurred

        Returns:
            Recommended recovery action or None
        """
        with self._lock:
            # Check if we have a pattern for this
            pattern_id = f"{product_type}_{failure_type.value}"
            if pattern_id in self.patterns:
                return self.patterns[pattern_id].recommended_action

            # Default recommendations by failure type
            default_actions = {
                FailureType.JSON_PARSE_ERROR: RecoveryAction.ADD_JSON_INSTRUCTIONS,
                FailureType.VALIDATION_FAILED: RecoveryAction.RETRY_WITH_SIMPLER_PROMPT,
                FailureType.TIMEOUT: RecoveryAction.INCREASE_TIMEOUT,
                FailureType.EMPTY_RESPONSE: RecoveryAction.USE_TEMPLATE_FALLBACK,
                FailureType.LOW_CONFIDENCE: RecoveryAction.USE_LLM_SPECS,
                FailureType.RATE_LIMITED: RecoveryAction.USE_CACHED_RESPONSE,
                FailureType.FIELD_EXTRACTION_FAILED: RecoveryAction.REDUCE_FIELD_COUNT,
            }

            return default_actions.get(failure_type)

    def get_optimized_prompt(self, product_type: str) -> Optional[str]:
        """
        Get a historically successful prompt for a product type.

        Args:
            product_type: Product type

        Returns:
            Optimized prompt or None if no history
        """
        with self._lock:
            return self.optimized_prompts.get(product_type)

    def get_failure_history(self,
                            product_type: str,
                            limit: int = 10) -> List[Dict]:
        """
        Get recent failure history for a product type.

        Args:
            product_type: Product type
            limit: Maximum entries to return

        Returns:
            List of failure records
        """
        with self._lock:
            failures = self.failures.get(product_type, [])
            return [f.to_dict() for f in failures[-limit:]]

    def get_success_history(self,
                            product_type: str,
                            limit: int = 10) -> List[Dict]:
        """
        Get recent success history for a product type.

        Args:
            product_type: Product type
            limit: Maximum entries to return

        Returns:
            List of success records
        """
        with self._lock:
            successes = self.successes.get(product_type, [])
            return [s.to_dict() for s in successes[-limit:]]

    # =========================================================================
    # PATTERN DETECTION
    # =========================================================================

    def _detect_patterns(self, product_type: str, failure_type: FailureType):
        """Detect failure patterns and update recommendations."""
        failures = self.failures.get(product_type, [])

        # Count recent failures of this type
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_same_type = [
            f for f in failures
            if f.failure_type == failure_type and
            datetime.fromisoformat(f.timestamp) > recent_cutoff
        ]

        if len(recent_same_type) >= 3:  # Pattern threshold
            pattern_id = f"{product_type}_{failure_type.value}"

            # Determine best recovery action based on history
            successful_recoveries = [
                f for f in recent_same_type
                if f.recovery_successful and f.recovery_attempted
            ]

            if successful_recoveries:
                # Use the most successful recovery action
                recovery_counts = {}
                for f in successful_recoveries:
                    ra = f.recovery_attempted.value
                    recovery_counts[ra] = recovery_counts.get(ra, 0) + 1

                best_action = max(recovery_counts.items(), key=lambda x: x[1])
                recommended = RecoveryAction(best_action[0])
                success_rate = best_action[1] / len(recent_same_type)
            else:
                # Use default recommendation
                recommended = self.get_recommended_recovery(product_type, failure_type)
                success_rate = 0.0

            self.patterns[pattern_id] = FailurePattern(
                pattern_id=pattern_id,
                failure_type=failure_type,
                product_type_pattern=product_type,
                occurrence_count=len(recent_same_type),
                recommended_action=recommended,
                success_rate_after_action=success_rate,
                last_seen=datetime.now().isoformat()
            )

            self.stats["patterns_detected"] = len(self.patterns)

            logger.info(
                f"[FAILURE_MEMORY] Pattern detected: {pattern_id} "
                f"({len(recent_same_type)} occurrences, recommend: {recommended.value})"
            )

    def _generate_recommendations(self,
                                   product_type: str,
                                   failures: List[FailureEntry],
                                   likely_failures: List[Tuple[str, int]]) -> List[str]:
        """Generate actionable recommendations based on failure analysis."""
        recommendations = []

        # Check for JSON parsing issues
        json_failures = sum(1 for f in failures if f.failure_type == FailureType.JSON_PARSE_ERROR)
        if json_failures > 0:
            recommendations.append(
                f"Add strict JSON format instructions - {json_failures} JSON parse failures detected"
            )

        # Check for validation failures
        validation_failures = sum(1 for f in failures if f.failure_type == FailureType.VALIDATION_FAILED)
        if validation_failures > 2:
            recommendations.append(
                f"Use simpler prompts - {validation_failures} validation failures detected"
            )

        # Check for timeouts
        timeout_failures = sum(1 for f in failures if f.failure_type == FailureType.TIMEOUT)
        if timeout_failures > 0:
            recommendations.append(
                f"Increase timeout or use parallel processing - {timeout_failures} timeouts"
            )

        # Check for low field extraction
        extraction_failures = sum(1 for f in failures if f.failure_type == FailureType.FIELD_EXTRACTION_FAILED)
        if extraction_failures > 1:
            recommendations.append(
                f"Use template fallback for field population - {extraction_failures} extraction failures"
            )

        # Check for cached success
        if product_type in self.optimized_prompts:
            recommendations.append(
                f"Use cached optimized prompt from previous success"
            )

        return recommendations

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    def _hash_prompt(self, prompt: str) -> str:
        """Create a hash of a prompt for comparison."""
        return hashlib.md5(prompt.encode()).hexdigest()[:16]

    def _save_to_disk(self):
        """Persist memory to disk."""
        if not self.persistence_path:
            return

        try:
            path = Path(self.persistence_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "failures": {
                    k: [f.to_dict() for f in v]
                    for k, v in self.failures.items()
                },
                "successes": {
                    k: [s.to_dict() for s in v]
                    for k, v in self.successes.items()
                },
                "optimized_prompts": self.optimized_prompts,
                "stats": self.stats,
                "saved_at": datetime.now().isoformat()
            }

            with open(path, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"[FAILURE_MEMORY] Failed to save to disk: {e}")

    def _load_from_disk(self):
        """Load memory from disk."""
        if not self.persistence_path:
            return

        try:
            path = Path(self.persistence_path)
            if not path.exists():
                return

            with open(path, 'r') as f:
                data = json.load(f)

            # Reconstruct failure entries
            for product_type, failures in data.get("failures", {}).items():
                self.failures[product_type] = [
                    FailureEntry(
                        failure_id=f["failure_id"],
                        product_type=f["product_type"],
                        failure_type=FailureType(f["failure_type"]),
                        error_message=f["error_message"],
                        prompt_hash=f["prompt_hash"],
                        prompt_preview=f["prompt_preview"],
                        timestamp=f["timestamp"],
                        context=f.get("context", {}),
                        recovery_attempted=RecoveryAction(f["recovery_attempted"]) if f.get("recovery_attempted") else None,
                        recovery_successful=f.get("recovery_successful", False),
                        duration_ms=f.get("duration_ms", 0)
                    )
                    for f in failures
                ]

            # Reconstruct success entries
            for product_type, successes in data.get("successes", {}).items():
                self.successes[product_type] = [
                    SuccessEntry(
                        success_id=s["success_id"],
                        product_type=s["product_type"],
                        prompt_hash=s["prompt_hash"],
                        prompt_preview=s["prompt_preview"],
                        fields_populated=s["fields_populated"],
                        total_fields=s["total_fields"],
                        confidence_score=s["confidence_score"],
                        timestamp=s["timestamp"],
                        duration_ms=s.get("duration_ms", 0),
                        sources_used=s.get("sources_used", [])
                    )
                    for s in successes
                ]

            self.optimized_prompts = data.get("optimized_prompts", {})
            self.stats = data.get("stats", self.stats)

            logger.info(
                f"[FAILURE_MEMORY] Loaded from disk: "
                f"{sum(len(v) for v in self.failures.values())} failures, "
                f"{sum(len(v) for v in self.successes.values())} successes"
            )

        except Exception as e:
            logger.error(f"[FAILURE_MEMORY] Failed to load from disk: {e}")

    # =========================================================================
    # STATISTICS & REPORTING
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        with self._lock:
            return {
                **self.stats,
                "product_types_tracked": len(set(self.failures.keys()) | set(self.successes.keys())),
                "patterns_active": len(self.patterns),
                "optimized_prompts_cached": len(self.optimized_prompts),
                "recovery_success_rate": (
                    self.stats["recoveries_successful"] / self.stats["recoveries_attempted"]
                    if self.stats["recoveries_attempted"] > 0 else 0.0
                )
            }

    def get_product_type_summary(self, product_type: str) -> Dict[str, Any]:
        """Get summary for a specific product type."""
        with self._lock:
            failures = self.failures.get(product_type, [])
            successes = self.successes.get(product_type, [])

            return {
                "product_type": product_type,
                "total_failures": len(failures),
                "total_successes": len(successes),
                "has_optimized_prompt": product_type in self.optimized_prompts,
                "failure_types": list(set(f.failure_type.value for f in failures)),
                "avg_success_confidence": (
                    sum(s.confidence_score for s in successes) / len(successes)
                    if successes else 0.0
                ),
                "avg_fields_populated": (
                    sum(s.fields_populated for s in successes) / len(successes)
                    if successes else 0.0
                )
            }

    def clear_history(self, product_type: Optional[str] = None):
        """Clear history for a product type or all."""
        with self._lock:
            if product_type:
                self.failures.pop(product_type, None)
                self.successes.pop(product_type, None)
                self.optimized_prompts.pop(product_type, None)
                logger.info(f"[FAILURE_MEMORY] Cleared history for {product_type}")
            else:
                self.failures.clear()
                self.successes.clear()
                self.patterns.clear()
                self.optimized_prompts.clear()
                self.stats = {k: 0 for k in self.stats}
                logger.info("[FAILURE_MEMORY] Cleared all history")

            if self.persistence_path:
                self._save_to_disk()


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_global_failure_memory: Optional[SchemaFailureMemory] = None
_memory_lock = threading.Lock()


def get_schema_failure_memory(persistence_path: Optional[str] = None) -> SchemaFailureMemory:
    """Get or create the global failure memory instance."""
    global _global_failure_memory

    with _memory_lock:
        if _global_failure_memory is None:
            _global_failure_memory = SchemaFailureMemory(
                persistence_path=persistence_path or "data/schema_failure_memory.json"
            )
        return _global_failure_memory


def reset_failure_memory():
    """Reset the global failure memory (for testing)."""
    global _global_failure_memory
    with _memory_lock:
        _global_failure_memory = None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "SchemaFailureMemory",
    "FailureEntry",
    "SuccessEntry",
    "FailureType",
    "RecoveryAction",
    "FailurePattern",
    "get_schema_failure_memory",
    "reset_failure_memory"
]
