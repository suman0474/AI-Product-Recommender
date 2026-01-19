# profiler/time_profiler.py
# =============================================================================
# TIME PROFILER FOR INSTRUMENT IDENTIFICATION WORKFLOW
# =============================================================================
#
# This script analyzes time consumption in the instrument identification process.
# Run this script to profile LLM calls, spec extraction, and workflow nodes.
#
# Usage: python -m profiler.time_profiler --log-file <log_file>
#        python -m profiler.time_profiler --live (to profile in real-time)
#
# =============================================================================

import re
import sys
import time
import logging
import argparse
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
from functools import wraps
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# PROFILING DECORATOR
# =============================================================================

_timing_results: Dict[str, List[float]] = defaultdict(list)
_current_operation: Optional[str] = None
_operation_start: Optional[float] = None


def profile_function(category: str = "general"):
    """
    Decorator to profile function execution time.
    
    Usage:
        @profile_function("llm_call")
        def my_function():
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{category}:{func.__name__}"
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed = time.time() - start_time
                _timing_results[func_name].append(elapsed)
                logger.info(f"[PROFILER] {func_name}: {elapsed:.2f}s")
        return wrapper
    return decorator


def start_operation(name: str):
    """Start timing an operation."""
    global _current_operation, _operation_start
    _current_operation = name
    _operation_start = time.time()
    logger.info(f"[PROFILER] START: {name}")


def end_operation(name: Optional[str] = None):
    """End timing an operation."""
    global _current_operation, _operation_start
    op_name = name or _current_operation
    if _operation_start:
        elapsed = time.time() - _operation_start
        _timing_results[op_name].append(elapsed)
        logger.info(f"[PROFILER] END: {op_name} ({elapsed:.2f}s)")
    _current_operation = None
    _operation_start = None


def get_profiling_summary() -> Dict[str, Any]:
    """Get summary of all profiling results."""
    summary = {}
    for operation, times in _timing_results.items():
        summary[operation] = {
            "count": len(times),
            "total_time_s": sum(times),
            "avg_time_s": sum(times) / len(times) if times else 0,
            "min_time_s": min(times) if times else 0,
            "max_time_s": max(times) if times else 0,
        }
    return summary


def print_profiling_report():
    """Print a formatted profiling report."""
    summary = get_profiling_summary()
    
    print("\n" + "=" * 80)
    print(" INSTRUMENT IDENTIFICATION TIME PROFILING REPORT")
    print("=" * 80)
    
    # Sort by total time (descending)
    sorted_ops = sorted(summary.items(), key=lambda x: x[1]["total_time_s"], reverse=True)
    
    total_time = sum(s["total_time_s"] for _, s in sorted_ops)
    
    print(f"\n{'Operation':<50} {'Count':>6} {'Total':>10} {'Avg':>8} {'%':>6}")
    print("-" * 80)
    
    for op_name, stats in sorted_ops:
        pct = (stats["total_time_s"] / total_time * 100) if total_time > 0 else 0
        print(f"{op_name:<50} {stats['count']:>6} {stats['total_time_s']:>9.2f}s {stats['avg_time_s']:>7.2f}s {pct:>5.1f}%")
    
    print("-" * 80)
    print(f"{'TOTAL':<50} {'':<6} {total_time:>9.2f}s")
    print("=" * 80)
    
    # Time consumers analysis
    print("\n" + "=" * 80)
    print(" TOP TIME CONSUMERS & REASONS")
    print("=" * 80 + "\n")
    
    for i, (op_name, stats) in enumerate(sorted_ops[:5], 1):
        pct = (stats["total_time_s"] / total_time * 100) if total_time > 0 else 0
        print(f"{i}. {op_name}")
        print(f"   Time: {stats['total_time_s']:.2f}s ({pct:.1f}% of total)")
        print(f"   Calls: {stats['count']} (avg {stats['avg_time_s']:.2f}s per call)")
        print(f"   Reason: {analyze_time_consumer_reason(op_name, stats)}")
        print()


def analyze_time_consumer_reason(op_name: str, stats: Dict[str, Any]) -> str:
    """Analyze and provide reason for time consumption."""
    reasons = []
    
    if "llm" in op_name.lower() or "specs" in op_name.lower():
        reasons.append("LLM API calls have network latency + processing time")
        if stats["count"] > 3:
            reasons.append(f"Multiple sequential calls ({stats['count']}x) - consider parallelization")
        if stats["avg_time_s"] > 5:
            reasons.append("High average time - check API quota/rate limits")
    
    if "standards" in op_name.lower() or "rag" in op_name.lower():
        reasons.append("Standards RAG requires document embedding search + retrieval")
        if stats["avg_time_s"] > 3:
            reasons.append("Consider caching frequently accessed standards")
    
    if "identify" in op_name.lower():
        reasons.append("Identification uses LLM for classification + context extraction")
        if stats["count"] > 2:
            reasons.append("Instruments & accessories identified separately - could batch")
    
    if "user_specs" in op_name.lower():
        reasons.append("Extracts explicit user specifications via LLM")
        if stats["avg_time_s"] > 2:
            reasons.append("Consider caching common patterns")
    
    if "batch" in op_name.lower():
        reasons.append("Batch processing - time scales with number of items")
    
    return " | ".join(reasons) if reasons else "General processing overhead"


# =============================================================================
# LOG FILE ANALYZER
# =============================================================================

class LogTimeAnalyzer:
    """Analyze time consumption from log files."""
    
    def __init__(self):
        self.events: List[Dict[str, Any]] = []
        self.llm_calls: List[Dict[str, Any]] = []
        self.specs_generation: List[Dict[str, Any]] = []
        self.workflow_nodes: List[Dict[str, Any]] = []
    
    def parse_log_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a single log line for timing information."""
        # Timestamp patterns
        timestamp_pattern = r'(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})'
        
        event = {"raw": line}
        
        # Extract timestamp
        ts_match = re.search(timestamp_pattern, line)
        if ts_match:
            event["timestamp"] = ts_match.group(1)
        
        # LLM Fallback patterns
        if "[LLM_FALLBACK]" in line:
            event["category"] = "llm_initialization"
            if "Attempting to use" in line:
                event["type"] = "start"
                model_match = re.search(r'Google Gemini: (\S+)', line)
                if model_match:
                    event["model"] = model_match.group(1)
            elif "Successfully initialized" in line:
                event["type"] = "end"
        
        # LLM Specs patterns
        if "[LLM_SPECS]" in line:
            event["category"] = "llm_specs_generation"
            if "Generating specs for:" in line:
                event["type"] = "start"
                product_match = re.search(r'Generating specs for: (.+)$', line)
                if product_match:
                    event["product"] = product_match.group(1).strip()
            elif "Generated" in line:
                event["type"] = "end"
                count_match = re.search(r'Generated (\d+) specs for (.+)$', line)
                if count_match:
                    event["spec_count"] = int(count_match.group(1))
                    event["product"] = count_match.group(2).strip()
        
        # User Specs patterns
        if "[USER_SPECS]" in line:
            event["category"] = "user_specs_extraction"
            if "Extracting specs for:" in line:
                event["type"] = "start"
                product_match = re.search(r'Extracting specs for: (.+)$', line)
                if product_match:
                    event["product"] = product_match.group(1).strip()
            elif "Extracted" in line:
                event["type"] = "end"
                count_match = re.search(r'Extracted (\d+) specs for (.+)$', line)
                if count_match:
                    event["spec_count"] = int(count_match.group(1))
                    event["product"] = count_match.group(2).strip()
        
        # HTTP Request patterns
        if "HTTP Request:" in line:
            event["category"] = "http_request"
            event["type"] = "complete"
            url_match = re.search(r'POST (https?://\S+)', line)
            if url_match:
                event["url"] = url_match.group(1)
            status_match = re.search(r'"(HTTP/\S+ \d+ \S+)"', line)
            if status_match:
                event["status"] = status_match.group(1)
        
        # Workflow node patterns
        if "[WORKFLOW]" in line or "[NODE]" in line:
            event["category"] = "workflow"
            if any(x in line for x in ["START", "BEGIN", "Entering"]):
                event["type"] = "start"
            elif any(x in line for x in ["END", "COMPLETE", "Finished"]):
                event["type"] = "end"
        
        return event if "category" in event else None
    
    def analyze_log_content(self, log_content: str) -> Dict[str, Any]:
        """Analyze log content and extract timing information."""
        lines = log_content.strip().split('\n')
        
        pending_operations: Dict[str, Dict[str, Any]] = {}
        completed_operations: List[Dict[str, Any]] = []
        
        for i, line in enumerate(lines):
            event = self.parse_log_line(line)
            if not event:
                continue
            
            self.events.append(event)
            
            category = event.get("category", "")
            op_type = event.get("type", "")
            product = event.get("product", "")
            
            key = f"{category}:{product}" if product else category
            
            if op_type == "start":
                pending_operations[key] = {
                    "category": category,
                    "product": product,
                    "start_line": i,
                    "start_event": event
                }
            elif op_type == "end" and key in pending_operations:
                op = pending_operations.pop(key)
                op["end_line"] = i
                op["end_event"] = event
                op["line_count"] = i - op["start_line"]
                completed_operations.append(op)
        
        # Categorize operations
        analysis = {
            "total_events": len(self.events),
            "llm_calls": [],
            "specs_generation": [],
            "user_specs": [],
            "http_requests": [],
            "summary": {}
        }
        
        for op in completed_operations:
            if op["category"] == "llm_specs_generation":
                analysis["specs_generation"].append(op)
            elif op["category"] == "user_specs_extraction":
                analysis["user_specs"].append(op)
            elif op["category"] == "llm_initialization":
                analysis["llm_calls"].append(op)
        
        # HTTP requests (already complete)
        for event in self.events:
            if event.get("category") == "http_request":
                analysis["http_requests"].append(event)
        
        # Summary
        analysis["summary"] = {
            "total_llm_initializations": len(analysis["llm_calls"]),
            "total_specs_generations": len(analysis["specs_generation"]),
            "total_user_specs_extractions": len(analysis["user_specs"]),
            "total_http_requests": len(analysis["http_requests"]),
            "products_processed": list(set(
                op.get("product", "") for op in completed_operations if op.get("product")
            ))
        }
        
        return analysis
    
    def print_analysis_report(self, analysis: Dict[str, Any]):
        """Print formatted analysis report."""
        print("\n" + "=" * 80)
        print(" LOG FILE TIME ANALYSIS REPORT")
        print("=" * 80)
        
        summary = analysis.get("summary", {})
        
        print(f"\n📊 SUMMARY")
        print(f"   Total Events Parsed: {analysis.get('total_events', 0)}")
        print(f"   LLM Initializations: {summary.get('total_llm_initializations', 0)}")
        print(f"   Specs Generations:   {summary.get('total_specs_generations', 0)}")
        print(f"   User Specs Extractions: {summary.get('total_user_specs_extractions', 0)}")
        print(f"   HTTP Requests:       {summary.get('total_http_requests', 0)}")
        
        products = summary.get("products_processed", [])
        if products:
            print(f"\n📦 PRODUCTS PROCESSED ({len(products)}):")
            for p in products:
                print(f"   - {p}")
        
        # Time estimation based on typical durations
        print(f"\n⏱️  ESTIMATED TIME BREAKDOWN:")
        print(f"   (Based on operation counts and typical durations)")
        print("-" * 60)
        
        # Estimated times
        estimated_times = {
            "LLM Initialization (test call)": summary.get("total_llm_initializations", 0) * 1.5,
            "Specs Generation (per product)": summary.get("total_specs_generations", 0) * 3.0,
            "User Specs Extraction (per product)": summary.get("total_user_specs_extractions", 0) * 2.0,
            "Network Overhead (HTTP requests)": summary.get("total_http_requests", 0) * 0.3,
        }
        
        total_estimated = sum(estimated_times.values())
        
        for op, time_s in sorted(estimated_times.items(), key=lambda x: x[1], reverse=True):
            pct = (time_s / total_estimated * 100) if total_estimated > 0 else 0
            print(f"   {op:<45} {time_s:>7.1f}s ({pct:>5.1f}%)")
        
        print("-" * 60)
        print(f"   {'TOTAL ESTIMATED':<45} {total_estimated:>7.1f}s")
        
        # Reasons & Recommendations
        print(f"\n🔍 TIME CONSUMERS & REASONS:")
        print("=" * 60)
        
        reasons = self._identify_time_consumers(analysis, estimated_times)
        for i, (reason, details) in enumerate(reasons, 1):
            print(f"\n{i}. {reason}")
            print(f"   {details}")
        
        # Optimization suggestions
        print(f"\n💡 OPTIMIZATION RECOMMENDATIONS:")
        print("=" * 60)
        self._print_recommendations(analysis, estimated_times)
    
    def _identify_time_consumers(self, analysis: Dict[str, Any], estimated_times: Dict[str, float]) -> List[Tuple[str, str]]:
        """Identify main time consumers."""
        reasons = []
        summary = analysis.get("summary", {})
        
        # LLM Initialization redundancy
        llm_inits = summary.get("total_llm_initializations", 0)
        if llm_inits > 3:
            reasons.append((
                f"EXCESSIVE LLM INITIALIZATIONS ({llm_inits}x)",
                f"Each init includes a test call. Consider reusing LLM instances."
            ))
        
        # Sequential processing
        specs_gens = summary.get("total_specs_generations", 0)
        user_specs = summary.get("total_user_specs_extractions", 0)
        if specs_gens > 2 or user_specs > 2:
            reasons.append((
                f"SEQUENTIAL PROCESSING ({specs_gens + user_specs} operations)",
                "LLM specs and user specs are processed one item at a time. " +
                "Each item requires separate LLM calls, adding latency."
            ))
        
        # HTTP requests overhead
        http_count = summary.get("total_http_requests", 0)
        if http_count > 10:
            reasons.append((
                f"HIGH HTTP REQUEST COUNT ({http_count})",
                "Each API call has network latency (~100-500ms). " +
                "Consider batch APIs if available."
            ))
        
        # No parallelization detected
        products = summary.get("products_processed", [])
        if len(products) > 2:
            reasons.append((
                f"MULTIPLE PRODUCTS ({len(products)}) - POSSIBLE SEQUENTIAL PROCESSING",
                "If products are processed sequentially, total time = N × single_item_time. " +
                "Parallel processing could reduce this significantly."
            ))
        
        return reasons
    
    def _print_recommendations(self, analysis: Dict[str, Any], estimated_times: Dict[str, float]):
        """Print optimization recommendations."""
        summary = analysis.get("summary", {})
        
        recommendations = []
        
        # LLM reuse
        if summary.get("total_llm_initializations", 0) > 3:
            recommendations.append(
                "1. REUSE LLM INSTANCES: Create one LLM client and pass it to functions " +
                "instead of creating new ones in each call."
            )
        
        # Batch processing
        if summary.get("total_specs_generations", 0) > 2:
            recommendations.append(
                "2. BATCH LLM PROMPTS: Combine multiple product specs into a single " +
                "LLM request when possible (reduces network round-trips)."
            )
        
        # Parallelization
        products = summary.get("products_processed", [])
        if len(products) > 2:
            recommendations.append(
                "3. PARALLEL PROCESSING: Use asyncio.gather() or ThreadPoolExecutor " +
                "to process multiple products simultaneously."
            )
        
        # Caching
        recommendations.append(
            "4. IMPLEMENT CACHING: Cache LLM results for common product types " +
            "(e.g., 'Pressure Transmitter' specs rarely change)."
        )
        
        # Timeout optimization (already done)
        recommendations.append(
            "5. OPTIMIZE TIMEOUT: Current timeout is 150s. Consider if this is " +
            "appropriate for expected response times (~2-5s per call)."
        )
        
        for rec in recommendations:
            print(f"\n   {rec}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Time Profiler for Instrument Identification")
    parser.add_argument("--log-file", type=str, help="Path to log file to analyze")
    parser.add_argument("--log-content", type=str, help="Raw log content to analyze")
    parser.add_argument("--live", action="store_true", help="Enable live profiling mode")
    
    args = parser.parse_args()
    
    if args.log_file:
        try:
            with open(args.log_file, 'r', encoding='utf-8') as f:
                content = f.read()
            analyzer = LogTimeAnalyzer()
            analysis = analyzer.analyze_log_content(content)
            analyzer.print_analysis_report(analysis)
        except FileNotFoundError:
            print(f"Error: Log file not found: {args.log_file}")
            sys.exit(1)
    
    elif args.log_content:
        analyzer = LogTimeAnalyzer()
        analysis = analyzer.analyze_log_content(args.log_content)
        analyzer.print_analysis_report(analysis)
    
    elif args.live:
        print("Live profiling mode enabled.")
        print("Import this module and use @profile_function decorator or start_operation/end_operation.")
        print("Call print_profiling_report() to see results.")
    
    else:
        # Demo with sample log content
        sample_log = """
INFO:llm_fallback:[LLM_FALLBACK] Attempting to use Google Gemini: gemini-2.5-flash
INFO:httpx:HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent "HTTP/1.1 200 OK"
INFO:llm_fallback:[LLM_FALLBACK] Successfully initialized Google Gemini: gemini-2.5-flash
INFO:agentic.deep_agent.llm_specs_generator:[LLM_SPECS] Generating specs for: Reactor Pressure Transmitter
INFO:httpx:HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent "HTTP/1.1 200 OK"
INFO:agentic.deep_agent.llm_specs_generator:[LLM_SPECS] Generated 21 specs for Reactor Pressure Transmitter
INFO:agentic.deep_agent.llm_specs_generator:[LLM_SPECS] Generating specs for: Pressure Transmitter 3-Valve Manifold
INFO:httpx:HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent "HTTP/1.1 200 OK"
INFO:agentic.deep_agent.llm_specs_generator:[LLM_SPECS] Generated 8 specs for Pressure Transmitter 3-Valve Manifold
INFO:agentic.deep_agent.user_specs_extractor:[USER_SPECS] Extracting specs for: Stainless Steel Impulse Line Kit
INFO:httpx:HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent "HTTP/1.1 200 OK"
INFO:agentic.deep_agent.user_specs_extractor:[USER_SPECS] Extracted 2 specs for Stainless Steel Impulse Line Kit
INFO:agentic.deep_agent.user_specs_extractor:[USER_SPECS] Extracting specs for: Hot Oil Pressure Transmitter
INFO:httpx:HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent "HTTP/1.1 200 OK"
INFO:agentic.deep_agent.user_specs_extractor:[USER_SPECS] Extracted 11 specs for Hot Oil Pressure Transmitter
"""
        
        print("No arguments provided. Running demo analysis with sample log content.")
        print("Use --help to see available options.\n")
        
        analyzer = LogTimeAnalyzer()
        analysis = analyzer.analyze_log_content(sample_log)
        analyzer.print_analysis_report(analysis)


if __name__ == "__main__":
    main()
