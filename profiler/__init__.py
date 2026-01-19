# profiler/__init__.py
"""
Time Profiling Module for Instrument Identification Workflow

Usage:
    1. Analyze log file:
       python -m profiler.time_profiler --log-file ./logs/app.log
    
    2. Analyze log content directly:
       python -m profiler.time_profiler --log-content "<log text>"
    
    3. Live profiling (use in code):
       from profiler.time_profiler import profile_function, print_profiling_report
       
       @profile_function("llm_call")
       def my_function():
           ...
       
       # After execution:
       print_profiling_report()
"""

from .time_profiler import (
    profile_function,
    start_operation,
    end_operation,
    get_profiling_summary,
    print_profiling_report,
    LogTimeAnalyzer
)

__all__ = [
    "profile_function",
    "start_operation",
    "end_operation",
    "get_profiling_summary",
    "print_profiling_report",
    "LogTimeAnalyzer"
]
