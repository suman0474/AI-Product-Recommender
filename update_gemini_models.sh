#!/bin/bash

# Bulk update script to replace Gemini model versions
# This script replaces:
# - gemini-2.0-flash-exp -> gemini-2.5-flash
# - gemini-2.0-flash -> gemini-2.5-flash
# - gemini-1.5-flash -> gemini-2.5-flash
#
# Usage: bash update_gemini_models.sh

set -e

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Array of files to update (excluding venv)
FILES_TO_UPDATE=(
    "advanced_parameters.py"
    "agentic/api.py"
    "agentic/deep_agent/llm_specs_generator.py"
    "agentic/deep_agent/schema_populator.py"
    "agentic/deep_agent/standards_deep_agent.py"
    "agentic/deep_agent/user_specs_extractor.py"
    "agentic/deep_agent/workflow.py"
    "agentic/grounded_chat_workflow.py"
    "agentic/index_rag/index_rag_agent.py"
    "agentic/index_rag/index_rag_memory.py"
    "agentic/instrument_identifier_workflow.py"
    "agentic/potential_product_index.py"
    "agentic/product_info_orchestrator.py"
    "agentic/rag_components.py"
    "agentic/shared_agents.py"
    "agentic/solution_workflow.py"
    "agentic/standards_rag/standards_chat_agent.py"
    "agentic/standards_rag/standards_rag_workflow.py"
    "agentic/strategy_rag/strategy_chat_agent.py"
    "agentic/workflow_orchestrator.py"
    "diagnose_llm.py"
    "llm_fallback.py"
    "llm_standardization.py"
    "loading.py"
    "main.py"
    "pdf_processor.py"
    "pdf_utils.py"
    "product_search_workflow/ppi_tools.py"
    "test.py"
    "tools/analysis_tools.py"
    "tools/instrument_tools.py"
    "tools/intent_tools.py"
    "tools/metadata_filter.py"
    "tools/parallel_indexer.py"
    "tools/ranking_tools.py"
    "tools/sales_agent_tools.py"
    "tools/sales_workflow_tools.py"
    "tools/schema_tools.py"
)

TOTAL_REPLACEMENTS=0
UPDATED_FILES=()

echo ""
echo "========================================="
echo "Starting Gemini model version updates..."
echo "========================================="
echo ""

for file in "${FILES_TO_UPDATE[@]}"; do
    if [ ! -f "$file" ]; then
        echo "SKIPPED: $file (file not found)"
        continue
    fi
    
    # Read file content
    original_content=$(cat "$file")
    content="$original_content"
    
    # Count and track replacements
    replacements_in_file=0
    
    # Replace gemini-2.0-flash-exp with gemini-2.5-flash
    if grep -q "gemini-2\.0-flash-exp" "$file"; then
        count=$(grep -o "gemini-2\.0-flash-exp" "$file" | wc -l)
        content=$(echo "$content" | sed 's/gemini-2\.0-flash-exp/gemini-2.5-flash/g')
        replacements_in_file=$((replacements_in_file + count))
    fi
    
    # Replace gemini-2.0-flash with gemini-2.5-flash (not gemini-2.0-flash-exp)
    if grep -q "gemini-2\.0-flash\"" "$file" || grep -q "gemini-2\.0-flash'" "$file"; then
        # Use a more precise pattern to avoid matching gemini-2.0-flash-exp
        count=$(echo "$content" | grep -o "gemini-2\.0-flash['\"]" | wc -l)
        if [ "$count" -gt 0 ]; then
            content=$(echo "$content" | sed "s/gemini-2\.0-flash\(['\"\`]\)/gemini-2.5-flash\1/g")
            replacements_in_file=$((replacements_in_file + count))
        fi
    fi
    
    # Replace gemini-1.5-flash with gemini-2.5-flash
    if grep -q "gemini-1\.5-flash" "$file"; then
        count=$(grep -o "gemini-1\.5-flash" "$file" | wc -l)
        content=$(echo "$content" | sed 's/gemini-1\.5-flash/gemini-2.5-flash/g')
        replacements_in_file=$((replacements_in_file + count))
    fi
    
    # Only update file if changes were made
    if [ "$content" != "$original_content" ]; then
        # Create backup
        cp "$file" "$file.bak"
        
        # Write updated content
        echo "$content" > "$file"
        
        UPDATED_FILES+=("$file")
        TOTAL_REPLACEMENTS=$((TOTAL_REPLACEMENTS + replacements_in_file))
        
        echo "UPDATED: $file"
        echo "  - Replacements made: $replacements_in_file"
        echo "  - Backup created: ${file}.bak"
    else
        echo "SKIPPED: $file (no matching patterns found)"
    fi
done

echo ""
echo "========================================="
echo "Update Summary:"
echo "========================================="
echo "Total files updated: ${#UPDATED_FILES[@]}"
echo "Total replacements made: $TOTAL_REPLACEMENTS"
echo ""

if [ ${#UPDATED_FILES[@]} -gt 0 ]; then
    echo "Files updated:"
    for file in "${UPDATED_FILES[@]}"; do
        echo "  - $file"
    done
    echo ""
    echo "Backup files created with .bak extension for each updated file."
fi

echo ""
echo "Replacements made:"
echo "  - gemini-2.0-flash-exp -> gemini-2.5-flash"
echo "  - gemini-2.0-flash -> gemini-2.5-flash"
echo "  - gemini-1.5-flash -> gemini-2.5-flash"
echo ""

if [ ${#UPDATED_FILES[@]} -eq 0 ]; then
    echo "No files needed updating."
else
    echo "Update completed successfully!"
fi

echo ""
