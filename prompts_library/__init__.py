# prompts_library/__init__.py
# =============================================================================
# PROMPTS LIBRARY PACKAGE
# =============================================================================
#
# This package contains all LLM prompts used throughout the AIPR application.
# Prompts are stored in individual .txt files and loaded at runtime.
#
# Usage:
#   from prompts_library import load_prompt
#   
#   PLANNER_PROMPT = load_prompt("deep_agent_planner_prompt")
#
# =============================================================================

from .prompt_loader import (
    load_prompt,
    get_prompt_metadata,
    list_available_prompts,
    reload_prompt,
    clear_prompt_cache,
    get_deep_agent_prompts,
    get_shared_agent_prompts,
    get_rag_prompts,
)

__all__ = [
    "load_prompt",
    "get_prompt_metadata",
    "list_available_prompts",
    "reload_prompt",
    "clear_prompt_cache",
    "get_deep_agent_prompts",
    "get_shared_agent_prompts",
    "get_rag_prompts",
]
