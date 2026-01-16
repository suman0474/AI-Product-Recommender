========================================
GEMINI MODEL VERSION UPDATE SCRIPTS
========================================

Created: 2026-01-16

QUICK START:
============

On Windows (PowerShell):
  cd "D:\AI PR\AIPR\backend"
  .\update_gemini_models.ps1

On Unix/Linux/macOS/WSL (Bash):
  cd "D:/AI PR/AIPR/backend"
  bash update_gemini_models.sh

WHAT WILL BE UPDATED:
=====================

The scripts will update 37 Python files with these replacements:

  1. gemini-2.0-flash-exp  →  gemini-2.5-flash
  2. gemini-2.0-flash      →  gemini-2.5-flash
  3. gemini-1.5-flash      →  gemini-2.5-flash

FILES INCLUDED:
===============

Core backend files:
  - advanced_parameters.py
  - main.py
  - llm_fallback.py
  - llm_standardization.py
  - loading.py
  - pdf_processor.py
  - pdf_utils.py
  - test.py
  - diagnose_llm.py

Agentic workflows:
  - agentic/api.py
  - agentic/grounded_chat_workflow.py
  - agentic/instrument_identifier_workflow.py
  - agentic/potential_product_index.py
  - agentic/product_info_orchestrator.py
  - agentic/rag_components.py
  - agentic/shared_agents.py
  - agentic/solution_workflow.py
  - agentic/workflow_orchestrator.py

Deep agent workflows:
  - agentic/deep_agent/llm_specs_generator.py
  - agentic/deep_agent/schema_populator.py
  - agentic/deep_agent/standards_deep_agent.py
  - agentic/deep_agent/user_specs_extractor.py
  - agentic/deep_agent/workflow.py

RAG workflows:
  - agentic/index_rag/index_rag_agent.py
  - agentic/index_rag/index_rag_memory.py
  - agentic/standards_rag/standards_chat_agent.py
  - agentic/standards_rag/standards_rag_workflow.py
  - agentic/strategy_rag/strategy_chat_agent.py

Tools:
  - tools/analysis_tools.py
  - tools/instrument_tools.py
  - tools/intent_tools.py
  - tools/metadata_filter.py
  - tools/parallel_indexer.py
  - tools/ranking_tools.py
  - tools/sales_agent_tools.py
  - tools/sales_workflow_tools.py
  - tools/schema_tools.py

Product search workflow:
  - product_search_workflow/ppi_tools.py

SAFETY FEATURES:
================

✓ Automatic backups (.bak files) created for each modified file
✓ Smart regex patterns to avoid false positives
✓ Detailed reporting of all changes
✓ Only modifies files where replacements are found
✓ Excludes venv/ directory (3rd party code)

VERIFICATION:
==============

After running the script, check changes with:

  # Show all replacements
  grep -r "gemini-2.5-flash" agentic/

  # Find all backup files
  find . -name "*.bak"

ROLLBACK:
=========

If you need to undo changes:

  # Restore a single file
  cp llm_fallback.py.bak llm_fallback.py

  # Restore all files
  for file in *.bak; do cp "$file" "${file%.bak}"; done

SCRIPTS INCLUDED:
=================

1. update_gemini_models.ps1
   - PowerShell script for Windows
   - Run with: .\update_gemini_models.ps1

2. update_gemini_models.sh
   - Bash script for Unix/Linux/macOS/WSL
   - Run with: bash update_gemini_models.sh

3. GEMINI_UPDATE_MANIFEST.md
   - Detailed documentation

4. GEMINI_UPDATE_README.txt
   - This file

SUPPORT:
========

For detailed information, see GEMINI_UPDATE_MANIFEST.md

========================================
